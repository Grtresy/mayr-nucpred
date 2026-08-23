"""Post-evaluation all-data refit for the publication Mayr N predictor."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from nucpred.core.files import atomic_write_json, sha256_file
from nucpred.experiments.mayr import nextgen_site_identification_v6 as v6
from nucpred.publication import mayr_site_publication as shared
from nucpred.publication.mayr_n_modeling import (
    _bound_inputs,
    _in_memory_c2,
    _in_memory_eb,
    _pretraining_entry,
    _training_configs,
    read_config as read_n_config,
)
from nucpred.publication.mayr_n_outer import (
    OUTER_CHECKPOINT_SCHEMA,
    _train_base,
    _train_eb,
    _train_ec,
    load_outer_checkpoint,
)
from nucpred.publication.mayr_site_training import (
    HIERARCHICAL_EXACT,
    RANKER_CHECKPOINT_SCHEMA,
    _checkpoint as _site_checkpoint,
    _mined_training_frame,
    _source_bindings as _site_source_bindings,
)
from nucpred.training.mayr_node_xtb_scratch import SolventVocabulary
from nucpred.training.mayr_site_confidence import tensor_mapping_sha256
from nucpred.training.mayr_site_n import (
    fit_site_n_preprocessor,
    load_site_n_examples,
)
from nucpred.training.mayr_site_ranker import (
    fit_type_aware_platt,
    site_type_indices,
)
from nucpred.training.mayr_site_region_residual import (
    REGION_FEATURE_SCHEMA,
    REGION_RESIDUAL_SCHEMA,
    REGION_SITE_TYPE,
    context_balanced_exact_weights,
    fit_region_residual_ensemble,
    origin_vocabulary,
    region_feature_matrix,
)
from nucpred.training.mayr_site_structured_ranker import (
    reduce_frozen_ensemble_features,
)


FINAL_SELECTION_SCHEMA = "nucpred.mayr-n-publication-final-selection.v1"
FINAL_N_SCHEMA = "nucpred.mayr-n-publication-final-n-refit.v1"
FINAL_SITE_SCHEMA = "nucpred.mayr-n-publication-final-site-refit.v1"


def _root(config: Mapping[str, Any]) -> Path:
    return shared.project_path(config["output_directory"], label="site output")


def _identifier_sha256(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(map(str, values)):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _upper_median(values: Sequence[int]) -> int:
    ordered = sorted(map(int, values))
    if not ordered:
        raise shared.PublicationSiteError("Cannot select from no epochs")
    return ordered[len(ordered) // 2]


def _optional(value: object, cast: Any) -> object:
    return None if value is None or pd.isna(value) else cast(value)


def _region_key(row: Mapping[str, Any]) -> tuple[object, ...]:
    return (
        str(row["arm"]),
        _optional(row.get("minimum_samples_leaf"), int),
        float(row.get("residual_weight", 0.0)),
        _optional(row.get("maximum_base_margin"), float),
        _optional(row.get("top_k"), int),
    )


def select_final(
    site_config_path: str | Path = shared.DEFAULT_CONFIG,
) -> dict[str, object]:
    site_config, resolved = shared.read_config(site_config_path)
    root = _root(site_config)
    destination = root / "final_selection"
    if destination.exists():
        raise shared.PublicationSiteError("Refusing to overwrite final selection")
    destination.parent.mkdir(parents=True, exist_ok=True)
    evaluation = root / "outer_evaluation" / "summary.json"
    evaluation_summary = json.loads(evaluation.read_text(encoding="utf-8"))
    if evaluation_summary.get("status") != "pass":
        raise shared.PublicationSiteError("Outer evaluation is not sealed")

    modeling_root = root.parent
    n_epochs: dict[str, list[int]] = defaultdict(list)
    n_bindings: list[dict[str, object]] = []
    for outer_fold in range(int(site_config["outer_fold_count"])):
        path = modeling_root / "outer_epoch_selection" / f"outer-{outer_fold}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("status") != "frozen"
            or int(payload.get("outer_test_metrics_computed", -1)) != 0
        ):
            raise shared.PublicationSiteError("N outer selection boundary changed")
        for stage, epoch in payload["selected_epochs"].items():
            n_epochs[str(stage)].append(int(epoch))
        n_bindings.append(
            {
                "outer_fold": outer_fold,
                "path": path.relative_to(shared.ROOT).as_posix(),
                "sha256": sha256_file(path),
            }
        )
    selected_n = {stage: _upper_median(values) for stage, values in n_epochs.items()}

    site_epochs: list[int] = []
    site_bindings: list[dict[str, object]] = []
    region_rows: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for outer_fold in range(int(site_config["outer_fold_count"])):
        selection_path = (
            root / "outer_selection" / f"outer-{outer_fold}" / "selection.json"
        )
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if (
            selection.get("status") != "pass"
            or int(selection.get("outer_test_target_rows_loaded", -1)) != 0
        ):
            raise shared.PublicationSiteError("Site outer selection boundary changed")
        site_epochs.append(int(selection["selected_base_epoch"]))
        site_bindings.append(
            {
                "outer_fold": outer_fold,
                "path": selection_path.relative_to(shared.ROOT).as_posix(),
                "sha256": sha256_file(selection_path),
            }
        )
        for inner_fold in range(int(site_config["inner_fold_count"])):
            search_path = (
                root
                / "nested_inner"
                / f"outer-{outer_fold}"
                / f"inner-{inner_fold}"
                / "region_search.csv"
            )
            for row in pd.read_csv(search_path).to_dict("records"):
                region_rows[_region_key(row)].append(row)
    complete_count = int(site_config["outer_fold_count"]) * int(
        site_config["inner_fold_count"]
    )
    macro_rows: list[dict[str, object]] = []
    for key, values in region_rows.items():
        if len(values) != complete_count:
            continue
        macro_rows.append(
            {
                "arm": key[0],
                "minimum_samples_leaf": key[1],
                "residual_weight": key[2],
                "maximum_base_margin": key[3],
                "top_k": key[4],
                "macro_exact_top1_recall": float(
                    np.mean([float(row["exact_top1_recall"]) for row in values])
                ),
                "macro_mrr": float(np.mean([float(row["mrr"]) for row in values])),
                "macro_exact_top3_recall": float(
                    np.mean([float(row["exact_top3_recall"]) for row in values])
                ),
                "inner_job_count": len(values),
            }
        )
    selected_region = max(
        macro_rows,
        key=lambda row: (
            float(row["macro_exact_top1_recall"]),
            float(row["macro_mrr"]),
            float(row["macro_exact_top3_recall"]),
        ),
    )
    if not macro_rows:
        raise shared.PublicationSiteError(
            "No complete inner-fold region setting exists"
        )
    payload: dict[str, object] = {
        "schema_version": FINAL_SELECTION_SCHEMA,
        "status": "frozen",
        "campaign_id": site_config["campaign_id"],
        "selected_N_epochs": selected_n,
        "outer_selected_N_epochs": dict(n_epochs),
        "selected_site_epoch": _upper_median(site_epochs),
        "outer_selected_site_epochs": site_epochs,
        "selected_region": selected_region,
        "selection_uses_outer_test_metrics": False,
        "selection_uses_external_labels": False,
        "outer_evaluation_phase_gate_path": evaluation.relative_to(
            shared.ROOT
        ).as_posix(),
        "outer_evaluation_phase_gate_sha256": sha256_file(evaluation),
        "N_selection_bindings": n_bindings,
        "site_selection_bindings": site_bindings,
        "site_config_sha256": sha256_file(resolved),
        "source_path": Path(__file__).resolve().relative_to(shared.ROOT).as_posix(),
        "source_sha256": sha256_file(Path(__file__).resolve()),
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    staging = Path(
        tempfile.mkdtemp(prefix=".final-selection.staging-", dir=destination.parent)
    )
    try:
        pd.DataFrame(macro_rows).to_csv(
            staging / "region_macro_search.csv", index=False, lineterminator="\n"
        )
        atomic_write_json(staging / "selection.json", payload)
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return payload


def _selection(site_config: Mapping[str, Any]) -> tuple[dict[str, Any], Path]:
    path = _root(site_config) / "final_selection" / "selection.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != FINAL_SELECTION_SCHEMA
        or payload.get("status") != "frozen"
    ):
        raise shared.PublicationSiteError("Final selection is not frozen")
    if payload.get("selection_uses_outer_test_metrics") is not False:
        raise shared.PublicationSiteError("Final selection used outer metrics")
    return payload, path


def run_final_n(
    *,
    initialization_seed: int,
    site_config_path: str | Path = shared.DEFAULT_CONFIG,
    device: str | None = None,
) -> dict[str, object]:
    started = time.perf_counter()
    site_config, _ = shared.read_config(site_config_path)
    selection, selection_path = _selection(site_config)
    n_config_path = shared.project_path(
        site_config["lineage"]["conditional_n_config_path"], label="N config"
    )
    n_config, n_resolved = read_n_config(n_config_path)
    allowed_seeds = tuple(map(int, site_config["initialization_seeds"]))
    if initialization_seed not in allowed_seeds:
        raise shared.PublicationSiteError("Final N seed is not registered")
    selected_device = torch.device(device or str(site_config["device"]))
    _bound_inputs(n_config, n_resolved)
    entry, pretraining_checkpoint, checkpoint_audit = _pretraining_entry(
        n_config, initialization_seed
    )
    dataset = shared.project_path(site_config["dataset"]["directory"], label="dataset")
    target_ids = (
        pd.read_parquet(dataset / "targets.parquet", columns=["target_id"])["target_id"]
        .astype(str)
        .tolist()
    )
    examples = load_site_n_examples(dataset, target_ids=set(target_ids))
    if sum(item.num_sites for item in examples) != len(target_ids):
        raise shared.PublicationSiteError("Final N refit target coverage changed")
    preprocessor = fit_site_n_preprocessor(examples)
    vocabulary = SolventVocabulary.from_values(
        [example.solvent_raw for example in examples]
    )
    base_config, stage_c_config, stage_eb_config, stage_ec_config = _training_configs(
        n_config
    )
    epochs = {key: int(value) for key, value in selection["selected_N_epochs"].items()}
    target = (
        _root(site_config)
        / "final_refit"
        / "conditional_n"
        / f"init-{initialization_seed}"
    )
    if target.exists():
        raise shared.PublicationSiteError("Refusing to overwrite final N refit")
    try:
        c2_model, c2_curves, c2_audit = _train_base(
            examples,
            config=n_config,
            base_config=base_config,
            stage_config=stage_c_config,
            preprocessor=preprocessor,
            vocabulary=vocabulary,
            initialization_seed=initialization_seed,
            checkpoint=pretraining_checkpoint,
            epochs=epochs["base_c2"],
            device=selected_device,
        )
        frozen_c2 = _in_memory_c2(c2_model, preprocessor, vocabulary)
        eb_model, eb_curves, eb_audit = _train_eb(
            examples,
            frozen=frozen_c2,
            config=stage_eb_config,
            initialization_seed=initialization_seed,
            epochs=epochs["stage_e_b_n1"],
            device=selected_device,
        )
        frozen_eb = _in_memory_eb(eb_model, preprocessor, vocabulary)
        model, ec_curves, ec_audit = _train_ec(
            examples,
            frozen=frozen_eb,
            config=stage_ec_config,
            initialization_seed=initialization_seed,
            epochs=epochs["stage_e_c_n3"],
            device=selected_device,
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".init-{initialization_seed}.staging-", dir=target.parent
            )
        )
        try:
            for name, curves in (
                ("base_c2", c2_curves),
                ("stage_e_b_n1", eb_curves),
                ("stage_e_c_n3", ec_curves),
            ):
                curves.to_csv(staging / f"{name}_training_curves.csv", index=False)
            state = {
                name: tensor.detach().cpu()
                for name, tensor in model.state_dict().items()
            }
            contract = {
                "schema_version": "nucpred.mayr-n-publication-final-n-contract.v1",
                "initialization_seed": initialization_seed,
                "pretraining_seed": int(entry["pretraining_seed"]),
                "target_count": len(target_ids),
                "target_id_sha256": _identifier_sha256(target_ids),
                "all_corrected_v2_targets_used": True,
                "post_outer_evaluation_refit": True,
                "reported_outer_metrics_modified": False,
                "external_sources_or_labels_used": False,
                "selection_path": selection_path.relative_to(shared.ROOT).as_posix(),
                "selection_sha256": sha256_file(selection_path),
            }
            checkpoint: dict[str, object] = {
                "schema_version": OUTER_CHECKPOINT_SCHEMA,
                "phase": "post_outer_evaluation_all_data_final_refit",
                "model_lineage": "pre-sN_C2_to_E-B-N1_to_E-C-N3",
                "model_architecture": model.architecture,
                "model_state_dict": state,
                "model_state_sha256": tensor_mapping_sha256(state),
                "preprocessor": preprocessor.to_json(),
                "solvent_vocabulary": list(vocabulary.tokens),
                "trained_epochs": epochs,
                "contract": contract,
                "stage_audits": {
                    "base_c2": c2_audit,
                    "stage_e_b_n1": eb_audit,
                    "stage_e_c_n3": ec_audit,
                },
            }
            torch.save(checkpoint, staging / "model.pt")
            summary: dict[str, object] = {
                "schema_version": FINAL_N_SCHEMA,
                "status": "pass",
                "campaign_id": site_config["campaign_id"],
                "initialization_seed": initialization_seed,
                "trained_epochs": epochs,
                "target_count": len(target_ids),
                "model_state_sha256": checkpoint["model_state_sha256"],
                "model_checkpoint_sha256": sha256_file(staging / "model.pt"),
                "pretraining_checkpoint_audit": checkpoint_audit,
                "contract": contract,
                "source_path": Path(__file__)
                .resolve()
                .relative_to(shared.ROOT)
                .as_posix(),
                "source_sha256": sha256_file(Path(__file__).resolve()),
                "wall_seconds": time.perf_counter() - started,
                "completed_at_utc": datetime.now(UTC).isoformat(),
            }
            atomic_write_json(staging / "summary.json", summary)
            os.replace(staging, target)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return summary
    finally:
        gc.collect()
        if selected_device.type == "cuda":
            torch.cuda.empty_cache()


def _load_final_n_models(
    site_config: Mapping[str, Any], *, device: torch.device
) -> list[tuple[int, torch.nn.Module, Any, Any, Mapping[str, Any], Path]]:
    models = []
    for seed in map(int, site_config["initialization_seeds"]):
        path = (
            _root(site_config)
            / "final_refit"
            / "conditional_n"
            / f"init-{seed}"
            / "model.pt"
        )
        summary = json.loads((path.parent / "summary.json").read_text(encoding="utf-8"))
        if summary.get("schema_version") != FINAL_N_SCHEMA or sha256_file(
            path
        ) != summary.get("model_checkpoint_sha256"):
            raise shared.PublicationSiteError("Final N checkpoint is not frozen")
        model, preprocessor, vocabulary, payload = load_outer_checkpoint(
            path,
            config_path=shared.project_path(
                site_config["lineage"]["conditional_n_config_path"], label="N config"
            ),
            device=device,
        )
        models.append((seed, model, preprocessor, vocabulary, payload, path))
    return models


def _select_final_margin_gate(
    context_oof: pd.DataFrame,
    settings: Mapping[str, Any],
) -> dict[str, object]:
    """Fit the deployment gate without inventing a singleton margin.

    A top1-minus-top2 margin is undefined when the deployment policy emits
    exactly one candidate.  Such contexts are excluded from threshold fitting
    and conservatively abstained at inference.  The gate is a post-evaluation
    deployment asset fitted from cross-fitted OOF predictions and labels; it
    must never be described as part of the sealed outer performance estimate.
    """

    required = {"candidate_count", "top1_margin", "site_top1_correct"}
    if not required.issubset(context_oof):
        raise shared.PublicationSiteError("OOF margin table is incomplete")
    margins = context_oof["top1_margin"].to_numpy(dtype=float)
    finite = np.isfinite(margins)
    undefined = ~finite
    if (
        np.isinf(margins).any()
        or not context_oof.loc[undefined, "candidate_count"].eq(1).all()
    ):
        raise shared.PublicationSiteError(
            "Only singleton contexts may have an undefined OOF margin"
        )
    if not finite.any():
        raise shared.PublicationSiteError("No multi-candidate OOF margin is available")
    selected = dict(
        v6.select_margin_threshold(
            margins=margins[finite],
            top1_correct=context_oof.loc[finite, "site_top1_correct"].to_numpy(bool),
            thresholds=settings["threshold_grid"],
            minimum_precision=float(settings["minimum_development_precision"]),
            minimum_accepted_count=int(settings["minimum_accepted_count"]),
        )
    )
    selected.update(
        {
            "selection_uses_test_labels": True,
            "selection_uses_outer_oof_labels": True,
            "selection_phase": "post_outer_evaluation_deployment_calibration",
            "reported_outer_metrics_modified": False,
            "selection_input_context_count": int(finite.sum()),
            "undefined_singleton_context_count": int(undefined.sum()),
            "undefined_singleton_policy": "abstain_margin_undefined",
            "deployment_acceptance_requires_at_least_two_candidates": True,
            "selected_coverage_all_oof_contexts": float(
                int(selected["selected_accepted_count"]) / len(context_oof)
            ),
        }
    )
    return selected


def run_final_site(
    site_config_path: str | Path = shared.DEFAULT_CONFIG,
    *,
    device: str | None = None,
) -> dict[str, object]:
    started = time.perf_counter()
    config, resolved = shared.read_config(site_config_path)
    selection, selection_path = _selection(config)
    selected_device = torch.device(device or str(config["device"]))
    destination = _root(config) / "final_refit" / "site_ranker"
    if destination.exists():
        raise shared.PublicationSiteError("Refusing to overwrite final site refit")
    destination.parent.mkdir(parents=True, exist_ok=True)
    dataset = shared.project_path(config["dataset"]["directory"], label="dataset")
    contexts = pd.read_parquet(dataset / "contexts.parquet")
    targets = pd.read_parquet(dataset / "targets.parquet")
    species = pd.read_parquet(dataset / "species.parquet")
    candidates, _ = shared.deployment_candidates(config, species)
    identities = (
        targets[["context_id", "species_id", "connectivity_id"]]
        .drop_duplicates()
        .sort_values("context_id", kind="stable")
    )
    queries = shared.candidate_universe(test_contexts=identities, candidates=candidates)
    models = _load_final_n_models(config, device=selected_device)
    query_ids, source_features, n_mean, n_std, conditional_bindings = (
        shared.encode_queries(
            models=models,
            queries=queries,
            contexts=contexts,
            config=config,
            device=selected_device,
        )
    )
    shared.release_models(models, device=selected_device)
    del models
    ordered = (
        queries.set_index("query_id", drop=False).loc[query_ids].reset_index(drop=True)
    )
    ordered["source_feature_index"] = np.arange(len(ordered), dtype=int)
    ordered["conditional_N_mean"] = n_mean
    ordered["conditional_N_std"] = n_std
    ordered["exact_label"] = v6._exact_labels(ordered, targets).astype(int)
    train_frame = _mined_training_frame(
        ordered, targets=targets, settings=config["ranker"]
    )
    views = reduce_frozen_ensemble_features(
        source_features,
        ensemble_size=int(config["outer_conditional_n_ensemble_size"]),
        block_dim=int(config["ranker"]["block_dim"]),
    )
    train_indices = torch.tensor(
        train_frame["source_feature_index"].to_numpy(dtype=int), dtype=torch.long
    )
    train_frame = train_frame.reset_index(drop=True)
    router_indices, router_targets, router_contexts = v6._context_router_indices(
        train_frame, targets
    )
    fixed_epoch = int(selection["selected_site_epoch"])
    settings = dict(config["ranker"])
    settings.update(
        {
            "maximum_epochs": fixed_epoch,
            "minimum_epochs": fixed_epoch,
            "evaluation_interval": fixed_epoch,
            "early_stopping_patience_evaluations": 1,
        }
    )
    fit = v6._fit_structured_arm(
        arm=HIERARCHICAL_EXACT,
        train_frame=train_frame,
        train_candidate_features=views.candidate[train_indices],
        train_context_features=views.context[train_indices],
        validation_frame=ordered,
        validation_candidate_features=views.candidate,
        validation_context_features=views.context,
        validation_targets=targets,
        router_indices=router_indices,
        router_targets=router_targets,
        settings=settings,
        ensemble_size=int(config["outer_conditional_n_ensemble_size"]),
        source_input_dim=int(source_features.shape[1]),
        block_dim=int(config["ranker"]["block_dim"]),
        seed=int(config["ranker"]["training_seed_offset"]) + 2000,
    )
    type_index = site_type_indices(ordered["site_type"].astype(str))
    with torch.no_grad():
        component_tensors = fit.model.forward_components(
            views.candidate, views.context, type_index
        )
    components = {
        key: value.detach().cpu().numpy() for key, value in component_tensors.items()
    }
    origins = origin_vocabulary(candidates)
    region_positions, region_features, feature_names = region_feature_matrix(
        ordered,
        membership_logits=components["membership_logit"],
        compatibility_logits=components["compatibility_logit"],
        conditional_n_mean=n_mean,
        conditional_n_std=n_std,
        origin_vocabulary_values=origins,
    )
    region_frame = ordered.iloc[region_positions].reset_index(drop=True)
    region_contexts = set(
        targets.loc[
            targets["site_type"].astype(str).eq(REGION_SITE_TYPE), "context_id"
        ].astype(str)
    )
    region_mask = region_frame["context_id"].astype(str).isin(region_contexts)
    region_training = region_frame.loc[region_mask].reset_index(drop=True)
    region_weights = context_balanced_exact_weights(region_training)
    region_selection = selection["selected_region"]
    if (
        region_selection.get("arm") != "region_structural_residual"
        or region_selection.get("minimum_samples_leaf") is None
    ):
        raise shared.PublicationSiteError(
            "Final region selection is not a trainable structural residual"
        )
    region_settings = config["region_residual"]
    region_seeds = [
        int(region_settings["training_seed_offset"]) + 2000 + int(offset)
        for offset in region_settings["ensemble_seed_offsets"]
    ]
    bundle = fit_region_residual_ensemble(
        region_features[region_mask.to_numpy()],
        region_training["exact_label"].to_numpy(dtype=int),
        sample_weights=region_weights,
        minimum_samples_leaf=int(region_selection["minimum_samples_leaf"]),
        estimator_count_per_seed=int(region_settings["estimator_count_per_seed"]),
        maximum_features=float(region_settings["maximum_features"]),
        seeds=region_seeds,
        feature_names=feature_names,
        origin_vocabulary_values=origins,
    )
    evaluation = _root(config) / "outer_evaluation"
    candidate_oof = pd.read_parquet(evaluation / "candidate_evaluation.parquet")
    context_oof = pd.read_parquet(evaluation / "context_evaluation.parquet")
    counts = candidate_oof.groupby("context_id")["query_id"].transform("count")
    calibration_weights = torch.tensor(
        (1.0 / counts.to_numpy(dtype=float)) / candidate_oof["context_id"].nunique(),
        dtype=torch.float32,
    )
    calibrator, calibrator_audit = fit_type_aware_platt(
        logits=torch.tensor(
            candidate_oof["canonical_logit"].to_numpy(float), dtype=torch.float32
        ),
        type_index=site_type_indices(candidate_oof["site_type"].astype(str)),
        labels=torch.tensor(
            candidate_oof["exact_label"].to_numpy(float), dtype=torch.float32
        ),
        weights=calibration_weights,
        l2_type_offset=float(config["calibration"]["l2_type_offset"]),
        l2_log_slope=float(config["calibration"]["l2_log_slope"]),
        maximum_iterations=int(config["calibration"]["maximum_iterations"]),
    )
    calibrator_audit = {
        **calibrator_audit,
        "fit_source": "cross_fitted_outer_oof_candidate_predictions_and_labels",
        "outer_oof_label_rows_used": len(candidate_oof),
        "outer_oof_labels_used": True,
        "fit_phase": "post_outer_evaluation_deployment_calibration",
        "reported_outer_metrics_modified": False,
    }
    margin_gate = _select_final_margin_gate(
        context_oof,
        config["abstention"],
    )
    staging = Path(
        tempfile.mkdtemp(prefix=".site-ranker.staging-", dir=destination.parent)
    )
    residual_path = staging / "region_residual.joblib"
    joblib.dump(bundle, residual_path, compress=3)
    residual_binding: dict[str, object] = {
        "enabled": True,
        "schema_version": REGION_RESIDUAL_SCHEMA,
        "feature_schema_version": REGION_FEATURE_SCHEMA,
        "target_site_type": REGION_SITE_TYPE,
        "path": (destination / residual_path.name).relative_to(shared.ROOT).as_posix(),
        "sha256": sha256_file(residual_path),
        "minimum_samples_leaf": int(region_selection["minimum_samples_leaf"]),
        "estimator_count_per_seed": int(region_settings["estimator_count_per_seed"]),
        "maximum_features": float(region_settings["maximum_features"]),
        "seeds": region_seeds,
        "feature_names": list(feature_names),
        "origin_vocabulary": list(origins),
        "residual_weight": float(region_selection["residual_weight"]),
        "maximum_base_margin": region_selection.get("maximum_base_margin"),
        "top_k": region_selection.get("top_k"),
        "candidate_set_conditioned": True,
        "type_level_maximum_preserved": True,
        "candidate_softmax_used": False,
    }
    checkpoint = _site_checkpoint(
        config=config,
        phase="post_outer_evaluation_all_data_final_refit",
        outer_fold=-1,
        inner_fold=None,
        model=fit.model,
        conditional_bindings=conditional_bindings,
        calibrator_payload=calibrator.to_payload(),
        calibrator_audit=calibrator_audit,
        margin_gate=margin_gate,
        selected_region=residual_binding,
        source_bindings=_site_source_bindings(resolved),
    )
    checkpoint.update(
        {
            "schema_version": RANKER_CHECKPOINT_SCHEMA,
            "final_refit_performed": True,
            "all_corrected_v2_targets_used": True,
            "reported_outer_metrics_modified": False,
            "post_evaluation_oof_calibration": True,
            "outer_oof_labels_used_for_deployment_calibration": True,
            "outer_test_partition_applicable": False,
            "outer_test_target_rows_loaded": len(targets),
            "all_data_target_rows_loaded": len(targets),
            "fixed_epoch_in_sample_monitoring_metrics_reported": False,
            "final_selection_binding": {
                "path": selection_path.relative_to(shared.ROOT).as_posix(),
                "sha256": sha256_file(selection_path),
            },
            "selected_base_epoch": fixed_epoch,
            "region_membership_residual": residual_binding,
        }
    )
    checkpoint_path = staging / "ranker_checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    summary: dict[str, object] = {
        "schema_version": FINAL_SITE_SCHEMA,
        "status": "pass",
        "campaign_id": config["campaign_id"],
        "target_count": len(targets),
        "context_count": int(targets["context_id"].nunique()),
        "candidate_count": len(ordered),
        "mined_candidate_count": len(train_frame),
        "selected_base_epoch": fixed_epoch,
        "selected_region": region_selection,
        "ranker_state_sha256": checkpoint["ranker_state_sha256"],
        "ranker_checkpoint_sha256": sha256_file(checkpoint_path),
        "region_residual_sha256": sha256_file(residual_path),
        "conditional_n_bindings": conditional_bindings,
        "calibrator_fit_audit": calibrator_audit,
        "margin_abstention": margin_gate,
        "router_training_context_count": len(router_contexts),
        "post_evaluation_oof_calibration": True,
        "outer_oof_labels_used_for_deployment_calibration": True,
        "outer_test_partition_applicable": False,
        "all_data_target_rows_loaded": len(targets),
        "fixed_epoch_in_sample_monitoring_metrics_reported": False,
        "reported_outer_metrics_modified": False,
        "external_sources_or_labels_used": False,
        "source_path": Path(__file__).resolve().relative_to(shared.ROOT).as_posix(),
        "source_sha256": sha256_file(Path(__file__).resolve()),
        "wall_seconds": time.perf_counter() - started,
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    atomic_write_json(staging / "summary.json", summary)
    try:
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    del source_features, views, component_tensors, components, fit
    gc.collect()
    if selected_device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=shared.DEFAULT_CONFIG)
    parser.add_argument("--device")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("select")
    n_parser = subparsers.add_parser("n")
    n_parser.add_argument("--initialization-seed", type=int, required=True)
    subparsers.add_parser("n-all")
    subparsers.add_parser("site")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config, _ = shared.read_config(args.config)
    if args.command == "select":
        payload = select_final(args.config)
    elif args.command == "n":
        payload = run_final_n(
            initialization_seed=args.initialization_seed,
            site_config_path=args.config,
            device=args.device,
        )
    elif args.command == "n-all":
        payload = {}
        for seed in map(int, config["initialization_seeds"]):
            payload[str(seed)] = run_final_n(
                initialization_seed=seed,
                site_config_path=args.config,
                device=args.device,
            )
    elif args.command == "site":
        payload = run_final_site(args.config, device=args.device)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
