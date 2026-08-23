"""Nested training and outer-development refit for automatic Mayr site prediction.

The module deliberately keeps outer-test targets outside every callable here.
It trains only an endpoint-relative site ranker and a region-internal residual
on top of frozen conditional-N representations.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
import gc
import json
import math
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from nucpred.core.files import atomic_write_json, sha256_file
from nucpred.experiments.mayr import nextgen_site_identification_v6 as v6
from nucpred.experiments.mayr import nextgen_site_identification_v7 as v7
from nucpred.publication import mayr_site_publication as shared
from nucpred.training.mayr_site_confidence import tensor_mapping_sha256
from nucpred.training.mayr_site_ranker import (
    fit_type_aware_platt,
)
from nucpred.training.mayr_site_region_residual import (
    REGION_FEATURE_SCHEMA,
    REGION_RESIDUAL_SCHEMA,
    REGION_SITE_TYPE,
    apply_region_residual,
    context_balanced_exact_weights,
    fit_region_residual_ensemble,
    origin_vocabulary,
    region_feature_matrix,
    score_region_residual,
)
from nucpred.training.mayr_site_structured_ranker import (
    HIERARCHICAL_EXACT,
    reduce_frozen_ensemble_features,
)


INNER_SCHEMA = "nucpred.mayr-n-publication-site-inner-fit.v1"
OUTER_SELECTION_SCHEMA = "nucpred.mayr-n-publication-site-outer-selection.v1"
OUTER_REFIT_SCHEMA = "nucpred.mayr-n-publication-site-outer-refit.v1"
RANKER_CHECKPOINT_SCHEMA = "nucpred.mayr-n-publication-site-ranker-checkpoint.v1"


def _output_root(config: Mapping[str, Any]) -> Path:
    return shared.project_path(config["output_directory"], label="site output")


def _source_bindings(config_path: Path) -> dict[str, dict[str, object]]:
    files = {
        "config": config_path,
        "training_source": Path(__file__).resolve(),
        "shared_source": Path(shared.__file__).resolve(),
        "structured_ranker_source": Path(v6.__file__).resolve(),
        "region_workflow_source": Path(v7.__file__).resolve(),
    }
    return {
        name: {
            "path": path.relative_to(shared.ROOT).as_posix(),
            "sha256": sha256_file(path),
            "bytes": int(path.stat().st_size),
        }
        for name, path in files.items()
    }


def _require_empty_destination(path: Path) -> None:
    if path.exists():
        raise shared.PublicationSiteError(
            f"Refusing to overwrite an existing publication artifact: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir()


def _role_data(
    config: Mapping[str, Any],
    *,
    outer_fold: int,
    inner_fold: int | None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    dataset_root = shared.project_path(config["dataset"]["directory"], label="dataset")
    contexts = pd.read_parquet(dataset_root / "contexts.parquet")
    species = pd.read_parquet(dataset_root / "species.parquet")
    outer = pd.read_csv(dataset_root / "outer_fold_membership.csv")
    nested = pd.read_csv(dataset_root / "nested_split_membership.csv")
    if inner_fold is None:
        membership = outer.loc[
            outer["outer_fold"].eq(outer_fold) & outer["role"].eq("development")
        ].copy()
        membership["role"] = "train"
    else:
        membership = nested.loc[
            nested["outer_fold"].eq(outer_fold) & nested["inner_fold"].eq(inner_fold)
        ].copy()
        if set(membership["role"].astype(str)) != {"train", "validation"}:
            raise shared.PublicationSiteError("Nested role coverage changed")

    outer_test_ids = set(
        outer.loc[
            outer["outer_fold"].eq(outer_fold) & outer["role"].eq("test"),
            "target_id",
        ].astype(str)
    )
    selected_ids = set(membership["target_id"].astype(str))
    if selected_ids & outer_test_ids:
        raise shared.PublicationSiteError("Site training selected an outer-test target")
    # Apply the membership predicate at the Parquet read boundary.  The target
    # table contains both site labels and N values, so loading the whole table
    # and filtering afterwards would make the outer-test audit untrue even if
    # no selected tensor happened to use those rows.
    targets = pd.read_parquet(
        dataset_root / "targets.parquet",
        filters=[("target_id", "in", sorted(selected_ids))],
    )
    if set(targets["target_id"].astype(str)) & outer_test_ids:
        raise shared.PublicationSiteError(
            "Target Parquet read exposed outer-test labels"
        )
    job_targets = targets.merge(
        membership[["target_id", "role"]],
        on="target_id",
        how="inner",
        validate="one_to_one",
    )
    if set(job_targets["target_id"].astype(str)) != selected_ids:
        raise shared.PublicationSiteError("Site training target coverage changed")
    context_roles = (
        membership[["context_id", "species_id", "connectivity_id", "role"]]
        .drop_duplicates()
        .sort_values("context_id", kind="stable")
        .reset_index(drop=True)
    )
    if context_roles.groupby("context_id")["role"].nunique().ne(1).any():
        raise shared.PublicationSiteError("One context crosses site-training roles")
    candidates, _ = shared.deployment_candidates(config, species)
    queries = shared.candidate_universe(
        test_contexts=context_roles.drop(columns="role"),
        candidates=candidates,
    ).merge(
        context_roles[["context_id", "role"]],
        on="context_id",
        how="left",
        validate="many_to_one",
    )
    if queries["role"].isna().any():
        raise shared.PublicationSiteError("Candidate query lost its role")
    return contexts, job_targets, candidates, queries, outer, membership


def _ordered_encoded_frame(
    *,
    queries: pd.DataFrame,
    query_ids: Sequence[str],
    source_features: torch.Tensor,
    n_mean: np.ndarray,
    n_std: np.ndarray,
    targets: pd.DataFrame,
) -> pd.DataFrame:
    ordered = (
        queries.set_index("query_id", drop=False)
        .loc[list(map(str, query_ids))]
        .reset_index(drop=True)
    )
    if len(ordered) != int(source_features.shape[0]):
        raise shared.PublicationSiteError("Encoded site features lost candidate rows")
    ordered["source_feature_index"] = np.arange(len(ordered), dtype=int)
    ordered["conditional_N_mean"] = np.asarray(n_mean, dtype=float)
    ordered["conditional_N_std"] = np.asarray(n_std, dtype=float)
    ordered["exact_label"] = v6._exact_labels(ordered, targets).astype(int)
    coverage = ordered.loc[ordered["exact_label"].eq(1)].groupby("context_id").size()
    if set(coverage.index.astype(str)) != set(targets["context_id"].astype(str)):
        raise shared.PublicationSiteError("Complete candidate universe misses a target")
    return ordered


def _mined_training_frame(
    frame: pd.DataFrame,
    *,
    targets: pd.DataFrame,
    settings: Mapping[str, Any],
) -> pd.DataFrame:
    work = frame.copy()
    # This is a target-independent frozen-model score used only to choose a
    # bounded endpoint-relative comparison set.  It is not a chemical label.
    work["baseline_validity_logit"] = work["conditional_N_mean"].to_numpy(float)
    selected = v6._mark_training_selection(
        work,
        train_targets=targets,
        reviewed_labels={},
        settings=settings,
    )
    if int(selected["compatible_proxy"].sum()) != 0:
        raise shared.PublicationSiteError("Publication mining imported proxy labels")
    nonexact = selected.loc[selected["exact_label"].eq(0), "auxiliary_label"]
    if nonexact.notna().any():
        raise shared.PublicationSiteError(
            "Unknown candidates became auxiliary negatives"
        )
    return selected.reset_index(drop=True)


def _router_delta(frame: pd.DataFrame, context_features: torch.Tensor) -> float:
    maximum = 0.0
    for _, group in frame.groupby("context_id", sort=True):
        positions = torch.tensor(group.index.to_numpy(dtype=int), dtype=torch.long)
        reference = context_features[positions[0]]
        delta = float(torch.max(torch.abs(context_features[positions] - reference)))
        maximum = max(maximum, delta)
    if maximum > 1e-5:
        raise shared.PublicationSiteError("Ranker router view is candidate-dependent")
    return maximum


def _region_search_settings(
    settings: Mapping[str, Any], *, residual_weight: float
) -> list[tuple[float | None, int | None]]:
    values: list[tuple[float | None, int | None]] = []
    if bool(settings["include_ungated"]):
        values.append((None, None))
    if residual_weight in set(map(float, settings["low_margin_top_k_weight_grid"])):
        values.append(
            (
                float(settings["low_margin_top_k_threshold"]),
                int(settings["low_margin_top_k"]),
            )
        )
    return list(dict.fromkeys(values))


def _clean_search_row(row: Mapping[str, Any]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in row.items():
        if value is None or (isinstance(value, float) and math.isnan(value)):
            result[key] = None
        elif isinstance(value, (np.integer,)):
            result[key] = int(value)
        elif isinstance(value, (np.floating,)):
            result[key] = float(value)
        else:
            result[key] = value
    return result


def _fit_and_search_region_residual(
    *,
    config: Mapping[str, Any],
    outer_fold: int,
    inner_fold: int,
    ordered: pd.DataFrame,
    targets: pd.DataFrame,
    candidates: pd.DataFrame,
    components: Mapping[str, np.ndarray],
) -> tuple[
    np.ndarray,
    dict[str, object],
    dict[str, object] | None,
    list[dict[str, object]],
    dict[int, np.ndarray],
    dict[str, object],
]:
    base_logits = np.asarray(components["canonical_logit"], dtype=float)
    if not shared.type_dependent_region_residual_enabled(config):
        base_metrics = v7._fast_retrieval_metrics(
            ordered.loc[ordered["role"].astype(str).eq("validation")].reset_index(
                drop=True
            ),
            base_logits[ordered["role"].astype(str).eq("validation").to_numpy()],
        )
        selected = {
            "arm": "frozen_hierarchical_exact",
            "minimum_samples_leaf": None,
            "residual_weight": 0.0,
            "maximum_base_margin": None,
            "top_k": None,
            **base_metrics,
        }
        return (
            base_logits.copy(),
            selected,
            None,
            [selected],
            {},
            {
                "schema_version": ("nucpred.mayr-n-publication-region-search-audit.v1"),
                "selected": selected,
                "base_metrics": base_metrics,
                "enabled": False,
                "reason": "true_site_type_is_not_available_to_the_predictor",
                "unknown_as_negative_count": 0,
                "outer_test_target_rows_loaded": 0,
            },
        )
    origins = origin_vocabulary(candidates)
    region_positions, region_features, feature_names = region_feature_matrix(
        ordered,
        membership_logits=components["membership_logit"],
        compatibility_logits=components["compatibility_logit"],
        conditional_n_mean=ordered["conditional_N_mean"],
        conditional_n_std=ordered["conditional_N_std"],
        origin_vocabulary_values=origins,
    )
    region_rows = ordered.iloc[region_positions].reset_index(drop=True)
    train_targets = targets.loc[targets["role"].eq("train")]
    train_region_contexts = set(
        train_targets.loc[
            train_targets["site_type"].astype(str).eq(REGION_SITE_TYPE),
            "context_id",
        ].astype(str)
    )
    train_mask = (
        region_rows["role"].astype(str).eq("train")
        & region_rows["context_id"].astype(str).isin(train_region_contexts)
    ).to_numpy()
    training_frame = region_rows.loc[train_mask].reset_index(drop=True)
    if not len(training_frame) or not train_region_contexts:
        raise shared.PublicationSiteError(
            "Nested fold has no region training population"
        )
    if (
        set(
            training_frame.loc[
                training_frame["exact_label"].eq(1), "context_id"
            ].astype(str)
        )
        != train_region_contexts
    ):
        raise shared.PublicationSiteError("Region training loses exact contexts")
    training_features = region_features[train_mask]
    training_weights = context_balanced_exact_weights(training_frame)
    settings = config["region_residual"]
    seeds = [
        int(settings["training_seed_offset"])
        + outer_fold * 100
        + inner_fold * 10
        + int(offset)
        for offset in settings["ensemble_seed_offsets"]
    ]
    validation_mask = ordered["role"].astype(str).eq("validation").to_numpy()
    validation_frame = ordered.loc[validation_mask].reset_index(drop=True)
    base_metrics = v7._fast_retrieval_metrics(
        validation_frame, base_logits[validation_mask]
    )
    search_rows: list[dict[str, object]] = [
        {
            "arm": "frozen_hierarchical_exact",
            "minimum_samples_leaf": None,
            "residual_weight": 0.0,
            "maximum_base_margin": None,
            "top_k": None,
            **base_metrics,
        }
    ]
    bundles: dict[int, dict[str, object]] = {}
    probability_by_leaf: dict[int, np.ndarray] = {}
    best_logits = base_logits.copy()
    best_row: dict[str, object] = search_rows[0]
    best_bundle: dict[str, object] | None = None
    best_application: dict[str, object] = {"enabled": False}
    for leaf_raw in settings["minimum_samples_leaf_grid"]:
        leaf = int(leaf_raw)
        bundle = fit_region_residual_ensemble(
            training_features,
            training_frame["exact_label"].to_numpy(dtype=int),
            sample_weights=training_weights,
            minimum_samples_leaf=leaf,
            estimator_count_per_seed=int(settings["estimator_count_per_seed"]),
            maximum_features=float(settings["maximum_features"]),
            seeds=seeds,
            feature_names=feature_names,
            origin_vocabulary_values=origins,
        )
        bundles[leaf] = bundle
        probabilities = score_region_residual(
            bundle,
            region_features,
            expected_feature_names=feature_names,
        )
        probability_by_leaf[leaf] = probabilities
        for weight_raw in settings["residual_weight_grid"]:
            weight = float(weight_raw)
            for maximum_margin, top_k in _region_search_settings(
                settings, residual_weight=weight
            ):
                candidate_logits, application = apply_region_residual(
                    ordered,
                    base_logits=base_logits,
                    region_positions=region_positions,
                    residual_probabilities=probabilities,
                    residual_weight=weight,
                    maximum_base_margin=maximum_margin,
                    top_k=top_k,
                )
                metrics = v7._fast_retrieval_metrics(
                    validation_frame, candidate_logits[validation_mask]
                )
                row: dict[str, object] = {
                    "arm": "region_structural_residual",
                    "minimum_samples_leaf": leaf,
                    "residual_weight": weight,
                    "maximum_base_margin": maximum_margin,
                    "top_k": top_k,
                    **metrics,
                }
                search_rows.append(row)
                if v7._selection_key(row) > v7._selection_key(best_row):
                    best_row = row
                    best_logits = candidate_logits
                    best_bundle = bundle
                    best_application = application
    selected = _clean_search_row(best_row)
    audit = {
        "schema_version": "nucpred.mayr-n-publication-region-search-audit.v1",
        "selected": selected,
        "base_metrics": base_metrics,
        "training_region_context_count": len(train_region_contexts),
        "training_region_candidate_count": len(training_frame),
        "training_exact_positive_count": int(training_frame["exact_label"].sum()),
        "origin_vocabulary": list(origins),
        "feature_names": list(feature_names),
        "ensemble_seeds": seeds,
        "selected_application": best_application,
        "unknown_as_negative_count": 0,
        "outer_test_target_rows_loaded": 0,
    }
    return (
        best_logits,
        selected,
        best_bundle,
        search_rows,
        probability_by_leaf,
        audit,
    )


def _checkpoint(
    *,
    config: Mapping[str, Any],
    phase: str,
    outer_fold: int,
    inner_fold: int | None,
    model: torch.nn.Module,
    conditional_bindings: Sequence[Mapping[str, object]],
    calibrator_payload: Mapping[str, object] | None,
    calibrator_audit: Mapping[str, object] | None,
    margin_gate: Mapping[str, object] | None,
    selected_region: Mapping[str, object],
    source_bindings: Mapping[str, object],
) -> dict[str, object]:
    state = deepcopy(model.state_dict())
    payload: dict[str, object] = {
        "schema_version": RANKER_CHECKPOINT_SCHEMA,
        "phase": phase,
        "campaign_id": config["campaign_id"],
        "experiment_id": config["experiment_id"],
        "outer_fold": int(outer_fold),
        "inner_fold": int(inner_fold) if inner_fold is not None else None,
        "selected_arm": HIERARCHICAL_EXACT,
        "ranker_architecture": model.architecture,
        "ranker_state_dict": state,
        "ranker_state_sha256": tensor_mapping_sha256(state),
        "conditional_n_bindings": list(conditional_bindings),
        "region_membership_residual": dict(selected_region),
        "calibrator": dict(calibrator_payload) if calibrator_payload else None,
        "calibrator_fit_audit": dict(calibrator_audit) if calibrator_audit else None,
        "margin_abstention": dict(margin_gate) if margin_gate else None,
        "source_bindings": dict(source_bindings),
        "conditional_n_backbone_frozen": True,
        "unknown_as_negative_count": 0,
        "endpoint_relative_noncanonical_not_universal_negative": True,
        "candidate_softmax_used": False,
        "outer_test_target_rows_loaded": 0,
        "outer_test_predictions_computed": False,
        "input_ablation": (
            dict(config["ablation"])
            if isinstance(config.get("ablation"), Mapping)
            else None
        ),
        "true_site_type_available_to_predictor": (
            shared.ablation_name(config) != "without_site_type"
        ),
    }
    return payload


def run_inner(
    config_path: str | Path,
    *,
    outer_fold: int,
    inner_fold: int,
    output_root: str | Path | None = None,
) -> dict[str, object]:
    config, resolved = shared.read_config(config_path)
    shared.verify_bindings(config, resolved)
    fold_count = int(config["outer_fold_count"])
    inner_count = int(config["inner_fold_count"])
    if not 0 <= outer_fold < fold_count or not 0 <= inner_fold < inner_count:
        raise shared.PublicationSiteError("Nested site fold is out of range")
    root = (
        Path(output_root).resolve() if output_root is not None else _output_root(config)
    )
    destination = root / "nested_inner" / f"outer-{outer_fold}" / f"inner-{inner_fold}"
    _require_empty_destination(destination)
    device = torch.device(str(config["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise shared.PublicationSiteError("Configured CUDA device is unavailable")
    contexts, targets, candidates, queries, outer, _ = _role_data(
        config, outer_fold=outer_fold, inner_fold=inner_fold
    )
    models = shared.inner_models_for_encoding(
        config,
        outer_fold=outer_fold,
        inner_fold=inner_fold,
        device=device,
    )
    query_ids, source_features, n_mean, n_std, conditional_bindings = (
        shared.encode_queries(
            models=models,
            queries=queries,
            contexts=contexts,
            config=config,
            device=device,
        )
    )
    shared.release_models(models, device=device)
    del models
    ordered = _ordered_encoded_frame(
        queries=queries,
        query_ids=query_ids,
        source_features=source_features,
        n_mean=n_mean,
        n_std=n_std,
        targets=targets,
    )
    train_targets = targets.loc[targets["role"].eq("train")].copy()
    validation_targets = targets.loc[targets["role"].eq("validation")].copy()
    train_all = ordered.loc[ordered["role"].eq("train")].copy()
    validation_frame = ordered.loc[ordered["role"].eq("validation")].copy()
    train_frame = _mined_training_frame(
        train_all,
        targets=train_targets,
        settings=config["ranker"],
    )
    views = reduce_frozen_ensemble_features(
        source_features,
        ensemble_size=int(config["inner_conditional_n_ensemble_size"]),
        block_dim=int(config["ranker"]["block_dim"]),
    )
    train_indices = torch.tensor(
        train_frame["source_feature_index"].to_numpy(dtype=int), dtype=torch.long
    )
    validation_indices = torch.tensor(
        validation_frame["source_feature_index"].to_numpy(dtype=int), dtype=torch.long
    )
    validation_frame = validation_frame.reset_index(drop=True)
    delta = _router_delta(validation_frame, views.context[validation_indices])
    train_frame = train_frame.reset_index(drop=True)
    train_model_frame = shared.model_facing_frame(train_frame, config)
    validation_model_frame = shared.model_facing_frame(validation_frame, config)
    train_model_targets = shared.model_facing_frame(train_targets, config)
    validation_model_targets = shared.model_facing_frame(validation_targets, config)
    router_indices, router_targets, router_contexts = v6._context_router_indices(
        train_model_frame, train_model_targets
    )
    result = v6._fit_structured_arm(
        arm=HIERARCHICAL_EXACT,
        train_frame=train_model_frame,
        train_candidate_features=views.candidate[train_indices],
        train_context_features=views.context[train_indices],
        validation_frame=validation_model_frame,
        validation_candidate_features=views.candidate[validation_indices],
        validation_context_features=views.context[validation_indices],
        validation_targets=validation_model_targets,
        router_indices=router_indices,
        router_targets=router_targets,
        settings=config["ranker"],
        ensemble_size=int(config["inner_conditional_n_ensemble_size"]),
        source_input_dim=int(source_features.shape[1]),
        block_dim=int(config["ranker"]["block_dim"]),
        seed=int(config["ranker"]["training_seed_offset"])
        + outer_fold * 10
        + inner_fold,
    )
    all_types = shared.model_facing_site_type_indices(ordered, config)
    with torch.no_grad():
        component_tensors = result.model.forward_components(
            views.candidate,
            views.context,
            all_types,
        )
    components = {
        key: value.detach().cpu().numpy() for key, value in component_tensors.items()
    }
    (
        selected_logits,
        selected_region,
        selected_bundle,
        search_rows,
        probability_by_leaf,
        region_audit,
    ) = _fit_and_search_region_residual(
        config=config,
        outer_fold=outer_fold,
        inner_fold=inner_fold,
        ordered=ordered,
        targets=targets,
        candidates=candidates,
        components=components,
    )
    validation_mask = ordered["role"].astype(str).eq("validation").to_numpy()
    final_validation_logits = selected_logits[validation_mask]
    final_metrics = v6._validation_metrics(
        frame=validation_frame,
        logits=final_validation_logits,
        membership_logits=components["membership_logit"][validation_mask],
        router_logits=components["router_logits"][validation_mask],
        validation_targets=validation_targets,
    )
    validation_weights = v6._context_uniform_candidate_weights(validation_frame)
    validation_type_index = shared.model_facing_site_type_indices(
        validation_frame, config
    )
    calibrator, calibrator_audit = fit_type_aware_platt(
        logits=torch.tensor(final_validation_logits, dtype=torch.float32),
        type_index=validation_type_index,
        labels=torch.tensor(
            validation_frame["exact_label"].to_numpy(dtype=float),
            dtype=torch.float32,
        ),
        weights=validation_weights,
        l2_type_offset=float(config["calibration"]["l2_type_offset"]),
        l2_log_slope=float(config["calibration"]["l2_log_slope"]),
        maximum_iterations=int(config["calibration"]["maximum_iterations"]),
    )
    margin_gate = v6.select_margin_threshold(
        margins=final_metrics["top1_margin"],
        top1_correct=final_metrics["top1_correct"],
        thresholds=config["abstention"]["threshold_grid"],
        minimum_precision=float(config["abstention"]["minimum_development_precision"]),
        minimum_accepted_count=int(config["abstention"]["minimum_accepted_count"]),
    )
    checkpoint = _checkpoint(
        config=config,
        phase="nested_inner_selection",
        outer_fold=outer_fold,
        inner_fold=inner_fold,
        model=result.model,
        conditional_bindings=conditional_bindings,
        calibrator_payload=calibrator.to_payload(),
        calibrator_audit=calibrator_audit,
        margin_gate=margin_gate,
        selected_region=selected_region,
        source_bindings=_source_bindings(resolved),
    )
    torch.save(checkpoint, destination / "ranker_checkpoint.pt")
    if selected_bundle is not None:
        joblib.dump(selected_bundle, destination / "region_residual.joblib", compress=3)
    predictions = validation_frame.copy()
    predictions["base_validity_logit"] = components["canonical_logit"][validation_mask]
    predictions["validity_logit"] = final_validation_logits
    predictions["membership_logit"] = components["membership_logit"][validation_mask]
    predictions["router_selected_logit"] = components["router_selected_logit"][
        validation_mask
    ]
    predictions["compatibility_logit"] = components["compatibility_logit"][
        validation_mask
    ]
    region_positions = np.flatnonzero(
        ordered["site_type"].astype(str).to_numpy() == REGION_SITE_TYPE
    )
    for leaf, probabilities in probability_by_leaf.items():
        full = np.full(len(ordered), np.nan, dtype=float)
        full[region_positions] = probabilities
        predictions[f"region_probability_leaf_{leaf}"] = full[validation_mask]
    shared.atomic_parquet(destination / "validation_predictions.parquet", predictions)
    shared.atomic_parquet(
        destination / "training_mined_candidates.parquet",
        train_frame[
            [
                "query_id",
                "context_id",
                "candidate_site_id",
                "site_type",
                "exact_label",
                "auxiliary_label",
                "mine_true_type",
                "mine_overlap_or_nested",
                "mine_hard_global",
                "mine_hard_wrong_type",
                "conditional_N_mean",
            ]
        ],
    )
    pd.DataFrame(search_rows).to_csv(destination / "region_search.csv", index=False)
    atomic_write_json(destination / "region_search_audit.json", region_audit)
    serial_metrics = {
        key: value
        for key, value in final_metrics.items()
        if key not in {"top1_correct", "top1_margin"}
    }
    outer_test_ids = set(
        outer.loc[
            outer["outer_fold"].eq(outer_fold) & outer["role"].eq("test"),
            "target_id",
        ].astype(str)
    )
    summary: dict[str, object] = {
        "schema_version": INNER_SCHEMA,
        "status": "pass",
        "campaign_id": config["campaign_id"],
        "outer_fold": outer_fold,
        "inner_fold": inner_fold,
        "best_base_epoch": int(result.audit["best_epoch"]),
        "base_fit_audit": result.audit,
        "selected_region": selected_region,
        "validation_metrics": serial_metrics,
        "train_target_count": len(train_targets),
        "validation_target_count": len(validation_targets),
        "train_context_count": int(train_frame["context_id"].nunique()),
        "validation_context_count": int(validation_frame["context_id"].nunique()),
        "train_full_candidate_count": len(train_all),
        "train_mined_candidate_count": len(train_frame),
        "validation_candidate_count": len(validation_frame),
        "router_training_context_count": len(router_contexts),
        "router_context_max_candidate_delta": delta,
        "margin_abstention": margin_gate,
        "ranker_checkpoint_sha256": sha256_file(destination / "ranker_checkpoint.pt"),
        "outer_test_target_count": len(outer_test_ids),
        "outer_test_target_rows_loaded": 0,
        "outer_test_predictions_computed": False,
        "unknown_as_negative_count": 0,
        "candidate_softmax_used": False,
        "source_bindings": _source_bindings(resolved),
    }
    atomic_write_json(destination / "summary.json", summary)
    del source_features, views, component_tensors, components, result
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(
        f"outer={outer_fold} inner={inner_fold} "
        f"epoch={summary['best_base_epoch']} "
        f"top1={serial_metrics['exact_top1_recall']:.6f} "
        f"mrr={serial_metrics['mrr']:.6f}",
        file=sys.stderr,
        flush=True,
    )
    return summary


def _search_key(row: Mapping[str, Any]) -> tuple[object, ...]:
    def optional(value: object, cast: Any) -> object:
        if value is None or pd.isna(value):
            return None
        return cast(value)

    return (
        str(row["arm"]),
        optional(row.get("minimum_samples_leaf"), int),
        float(row.get("residual_weight", 0.0)),
        optional(row.get("maximum_base_margin"), float),
        optional(row.get("top_k"), int),
    )


def select_outer(
    config_path: str | Path,
    *,
    outer_fold: int,
    output_root: str | Path | None = None,
) -> dict[str, object]:
    config, resolved = shared.read_config(config_path)
    root = Path(output_root).resolve() if output_root else _output_root(config)
    epochs: list[int] = []
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    inner_bindings: list[dict[str, object]] = []
    for inner_fold in range(int(config["inner_fold_count"])):
        directory = (
            root / "nested_inner" / f"outer-{outer_fold}" / f"inner-{inner_fold}"
        )
        summary_path = directory / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            summary.get("schema_version") != INNER_SCHEMA
            or summary.get("status") != "pass"
        ):
            raise shared.PublicationSiteError("Inner site summary is not frozen")
        if int(summary["outer_test_target_rows_loaded"]) != 0:
            raise shared.PublicationSiteError("Inner site fit read outer-test targets")
        epochs.append(int(summary["best_base_epoch"]))
        rows = pd.read_csv(directory / "region_search.csv")
        for row in rows.to_dict("records"):
            grouped[_search_key(row)].append(row)
        inner_bindings.append(
            {
                "inner_fold": inner_fold,
                "summary_path": summary_path.relative_to(shared.ROOT).as_posix(),
                "summary_sha256": sha256_file(summary_path),
                "ranker_checkpoint_sha256": summary["ranker_checkpoint_sha256"],
            }
        )
    common = {
        key: values
        for key, values in grouped.items()
        if len(values) == int(config["inner_fold_count"])
    }
    if not common:
        raise shared.PublicationSiteError("Region search has no common inner grid")
    macro_rows: list[dict[str, object]] = []
    for key, values in common.items():
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
                "inner_fold_count": len(values),
            }
        )
    selected = max(
        macro_rows,
        key=lambda row: (
            float(row["macro_exact_top1_recall"]),
            float(row["macro_mrr"]),
            float(row["macro_exact_top3_recall"]),
        ),
    )
    ordered_epochs = sorted(epochs)
    selected_epoch = ordered_epochs[len(ordered_epochs) // 2]
    destination = root / "outer_selection" / f"outer-{outer_fold}"
    _require_empty_destination(destination)
    pd.DataFrame(macro_rows).to_csv(
        destination / "region_macro_search.csv", index=False
    )
    payload: dict[str, object] = {
        "schema_version": OUTER_SELECTION_SCHEMA,
        "status": "pass",
        "campaign_id": config["campaign_id"],
        "outer_fold": outer_fold,
        "selected_base_epoch": selected_epoch,
        "base_epoch_rule": config["ranker"]["outer_epoch_rule"],
        "inner_best_epochs": epochs,
        "selected_region": _clean_search_row(selected),
        "region_selection_rule": config["region_residual"]["outer_selection_rule"],
        "inner_bindings": inner_bindings,
        "outer_test_target_rows_loaded": 0,
        "outer_test_predictions_computed": False,
        "source_bindings": _source_bindings(resolved),
    }
    atomic_write_json(destination / "selection.json", payload)
    return payload


def _apply_selected_region_to_validation(
    frame: pd.DataFrame,
    selected_region: Mapping[str, Any],
) -> np.ndarray:
    base_logits = frame["base_validity_logit"].to_numpy(dtype=float)
    if str(selected_region["arm"]) != "region_structural_residual":
        return base_logits
    leaf = int(selected_region["minimum_samples_leaf"])
    positions = np.flatnonzero(
        frame["site_type"].astype(str).to_numpy() == REGION_SITE_TYPE
    )
    probabilities = frame.iloc[positions][f"region_probability_leaf_{leaf}"].to_numpy(
        dtype=float
    )
    logits, _ = apply_region_residual(
        frame.reset_index(drop=True),
        base_logits=base_logits,
        region_positions=positions,
        residual_probabilities=probabilities,
        residual_weight=float(selected_region["residual_weight"]),
        maximum_base_margin=(
            float(selected_region["maximum_base_margin"])
            if selected_region.get("maximum_base_margin") is not None
            else None
        ),
        top_k=(
            int(selected_region["top_k"])
            if selected_region.get("top_k") is not None
            else None
        ),
    )
    return logits


def _outer_development_calibration(
    *,
    config: Mapping[str, Any],
    root: Path,
    outer_fold: int,
    selected_region: Mapping[str, Any],
    all_targets: pd.DataFrame,
) -> tuple[dict[str, object], dict[str, object], pd.DataFrame, dict[str, object]]:
    frames: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, object]] = []
    for inner_fold in range(int(config["inner_fold_count"])):
        path = (
            root
            / "nested_inner"
            / f"outer-{outer_fold}"
            / f"inner-{inner_fold}"
            / "validation_predictions.parquet"
        )
        frame = pd.read_parquet(path).reset_index(drop=True)
        frame["outer_selected_validity_logit"] = _apply_selected_region_to_validation(
            frame, selected_region
        )
        targets = all_targets.loc[
            all_targets["context_id"]
            .astype(str)
            .isin(set(frame["context_id"].astype(str)))
        ]
        metrics = v6._validation_metrics(
            frame=frame,
            logits=frame["outer_selected_validity_logit"].to_numpy(float),
            membership_logits=frame["membership_logit"].to_numpy(float),
            router_logits=None,
            validation_targets=targets,
        )
        fold_metrics.append(
            {
                "inner_fold": inner_fold,
                **{
                    key: value
                    for key, value in metrics.items()
                    if key not in {"top1_correct", "top1_margin"}
                },
            }
        )
        frames.append(frame)
    oof = pd.concat(frames, ignore_index=True)
    if oof["query_id"].astype(str).duplicated().any():
        raise shared.PublicationSiteError("Inner OOF predictions duplicate a query")
    development_metrics = v6._validation_metrics(
        frame=oof,
        logits=oof["outer_selected_validity_logit"].to_numpy(float),
        membership_logits=oof["membership_logit"].to_numpy(float),
        router_logits=None,
        validation_targets=all_targets.loc[
            all_targets["context_id"]
            .astype(str)
            .isin(set(oof["context_id"].astype(str)))
        ],
    )
    weights = v6._context_uniform_candidate_weights(oof)
    type_index = shared.model_facing_site_type_indices(oof, config)
    calibrator, calibrator_audit = fit_type_aware_platt(
        logits=torch.tensor(
            oof["outer_selected_validity_logit"].to_numpy(float),
            dtype=torch.float32,
        ),
        type_index=type_index,
        labels=torch.tensor(oof["exact_label"].to_numpy(float), dtype=torch.float32),
        weights=weights,
        l2_type_offset=float(config["calibration"]["l2_type_offset"]),
        l2_log_slope=float(config["calibration"]["l2_log_slope"]),
        maximum_iterations=int(config["calibration"]["maximum_iterations"]),
    )
    margin_gate = v6.select_margin_threshold(
        margins=development_metrics["top1_margin"],
        top1_correct=development_metrics["top1_correct"],
        thresholds=config["abstention"]["threshold_grid"],
        minimum_precision=float(config["abstention"]["minimum_development_precision"]),
        minimum_accepted_count=int(config["abstention"]["minimum_accepted_count"]),
    )
    audit = {
        "schema_version": "nucpred.mayr-n-publication-site-oof-calibration.v1",
        "outer_fold": outer_fold,
        "inner_fold_metrics": fold_metrics,
        "development_metrics": {
            key: value
            for key, value in development_metrics.items()
            if key not in {"top1_correct", "top1_margin"}
        },
        "calibrator_fit": calibrator_audit,
        "margin_abstention": margin_gate,
        "calibration_uses_outer_test": False,
        "outer_test_target_rows_loaded": 0,
    }
    return calibrator.to_payload(), margin_gate, oof, audit


def run_outer_refit(
    config_path: str | Path,
    *,
    outer_fold: int,
    output_root: str | Path | None = None,
) -> dict[str, object]:
    config, resolved = shared.read_config(config_path)
    shared.verify_bindings(config, resolved)
    root = Path(output_root).resolve() if output_root else _output_root(config)
    selection_path = root / "outer_selection" / f"outer-{outer_fold}" / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("schema_version") != OUTER_SELECTION_SCHEMA:
        raise shared.PublicationSiteError("Outer site selection is not frozen")
    destination = root / "outer_refit" / f"outer-{outer_fold}"
    _require_empty_destination(destination)
    device = torch.device(str(config["device"]))
    contexts, targets, candidates, queries, outer, _ = _role_data(
        config, outer_fold=outer_fold, inner_fold=None
    )
    models = shared.load_outer_conditional_ensemble(
        config, outer_fold=outer_fold, device=device
    )
    query_ids, source_features, n_mean, n_std, conditional_bindings = (
        shared.encode_queries(
            models=models,
            queries=queries,
            contexts=contexts,
            config=config,
            device=device,
        )
    )
    shared.release_models(models, device=device)
    del models
    ordered = _ordered_encoded_frame(
        queries=queries,
        query_ids=query_ids,
        source_features=source_features,
        n_mean=n_mean,
        n_std=n_std,
        targets=targets,
    )
    train_frame = _mined_training_frame(
        ordered,
        targets=targets,
        settings=config["ranker"],
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
    train_model_frame = shared.model_facing_frame(train_frame, config)
    model_targets = shared.model_facing_frame(targets, config)
    validation_model_frame = shared.model_facing_frame(
        ordered.reset_index(drop=True), config
    )
    router_indices, router_targets, router_contexts = v6._context_router_indices(
        train_model_frame, model_targets
    )
    fixed_settings = dict(config["ranker"])
    fixed_epoch = int(selection["selected_base_epoch"])
    fixed_settings.update(
        {
            "maximum_epochs": fixed_epoch,
            "minimum_epochs": fixed_epoch,
            "evaluation_interval": fixed_epoch,
            "early_stopping_patience_evaluations": 1,
        }
    )
    result = v6._fit_structured_arm(
        arm=HIERARCHICAL_EXACT,
        train_frame=train_model_frame,
        train_candidate_features=views.candidate[train_indices],
        train_context_features=views.context[train_indices],
        validation_frame=validation_model_frame,
        validation_candidate_features=views.candidate,
        validation_context_features=views.context,
        validation_targets=model_targets,
        router_indices=router_indices,
        router_targets=router_targets,
        settings=fixed_settings,
        ensemble_size=int(config["outer_conditional_n_ensemble_size"]),
        source_input_dim=int(source_features.shape[1]),
        block_dim=int(config["ranker"]["block_dim"]),
        seed=int(config["ranker"]["training_seed_offset"]) + 1000 + outer_fold,
    )
    all_types = shared.model_facing_site_type_indices(ordered, config)
    with torch.no_grad():
        component_tensors = result.model.forward_components(
            views.candidate, views.context, all_types
        )
    components = {
        key: value.detach().cpu().numpy() for key, value in component_tensors.items()
    }
    selected_region = dict(selection["selected_region"])
    selected_bundle: dict[str, object] | None = None
    region_fit_audit: dict[str, object] = {
        "enabled": str(selected_region["arm"]) == "region_structural_residual"
    }
    if str(selected_region["arm"]) == "region_structural_residual":
        origins = origin_vocabulary(candidates)
        region_positions, region_features, feature_names = region_feature_matrix(
            ordered.reset_index(drop=True),
            membership_logits=components["membership_logit"],
            compatibility_logits=components["compatibility_logit"],
            conditional_n_mean=ordered["conditional_N_mean"],
            conditional_n_std=ordered["conditional_N_std"],
            origin_vocabulary_values=origins,
        )
        region_frame = ordered.iloc[region_positions].reset_index(drop=True)
        region_contexts = set(
            targets.loc[
                targets["site_type"].astype(str).eq(REGION_SITE_TYPE), "context_id"
            ].astype(str)
        )
        selected_mask = region_frame["context_id"].astype(str).isin(region_contexts)
        training_frame = region_frame.loc[selected_mask].reset_index(drop=True)
        training_features = region_features[selected_mask.to_numpy()]
        weights = context_balanced_exact_weights(training_frame)
        region_settings = config["region_residual"]
        seeds = [
            int(region_settings["training_seed_offset"])
            + 1000
            + outer_fold * 10
            + int(offset)
            for offset in region_settings["ensemble_seed_offsets"]
        ]
        selected_bundle = fit_region_residual_ensemble(
            training_features,
            training_frame["exact_label"].to_numpy(dtype=int),
            sample_weights=weights,
            minimum_samples_leaf=int(selected_region["minimum_samples_leaf"]),
            estimator_count_per_seed=int(region_settings["estimator_count_per_seed"]),
            maximum_features=float(region_settings["maximum_features"]),
            seeds=seeds,
            feature_names=feature_names,
            origin_vocabulary_values=origins,
        )
        joblib.dump(selected_bundle, destination / "region_residual.joblib", compress=3)
        region_fit_audit = {
            "enabled": True,
            "training_region_context_count": len(region_contexts),
            "training_region_candidate_count": len(training_frame),
            "training_exact_positive_count": int(training_frame["exact_label"].sum()),
            "feature_schema_version": REGION_FEATURE_SCHEMA,
            "residual_schema_version": REGION_RESIDUAL_SCHEMA,
            "feature_names": list(feature_names),
            "origin_vocabulary": list(origins),
            "seeds": seeds,
            "unknown_as_negative_count": 0,
        }
    calibrator_payload, margin_gate, oof, calibration_audit = (
        _outer_development_calibration(
            config=config,
            root=root,
            outer_fold=outer_fold,
            selected_region=selected_region,
            all_targets=targets,
        )
    )
    checkpoint = _checkpoint(
        config=config,
        phase="outer_development_refit",
        outer_fold=outer_fold,
        inner_fold=None,
        model=result.model,
        conditional_bindings=conditional_bindings,
        calibrator_payload=calibrator_payload,
        calibrator_audit=calibration_audit["calibrator_fit"],
        margin_gate=margin_gate,
        selected_region=selected_region,
        source_bindings=_source_bindings(resolved),
    )
    checkpoint["outer_selection_binding"] = {
        "path": selection_path.relative_to(shared.ROOT).as_posix(),
        "sha256": sha256_file(selection_path),
    }
    checkpoint["selected_base_epoch"] = fixed_epoch
    torch.save(checkpoint, destination / "ranker_checkpoint.pt")
    shared.atomic_parquet(destination / "development_oof_predictions.parquet", oof)
    atomic_write_json(destination / "calibration_audit.json", calibration_audit)
    atomic_write_json(destination / "region_refit_audit.json", region_fit_audit)
    summary: dict[str, object] = {
        "schema_version": OUTER_REFIT_SCHEMA,
        "status": "pass",
        "campaign_id": config["campaign_id"],
        "outer_fold": outer_fold,
        "selected_base_epoch": fixed_epoch,
        "selected_region": selected_region,
        "development_target_count": len(targets),
        "development_context_count": int(targets["context_id"].nunique()),
        "development_full_candidate_count": len(ordered),
        "development_mined_candidate_count": len(train_frame),
        "router_training_context_count": len(router_contexts),
        "ranker_fit_audit": result.audit,
        "region_refit_audit": region_fit_audit,
        "calibration_audit": calibration_audit,
        "ranker_checkpoint_sha256": sha256_file(destination / "ranker_checkpoint.pt"),
        "outer_selection_sha256": sha256_file(selection_path),
        "outer_test_target_rows_loaded": 0,
        "outer_test_predictions_computed": False,
        "unknown_as_negative_count": 0,
        "candidate_softmax_used": False,
        "source_bindings": _source_bindings(resolved),
    }
    atomic_write_json(destination / "summary.json", summary)
    del source_features, views, component_tensors, components, result
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(
        f"outer={outer_fold} refit_epoch={fixed_epoch} "
        f"oof_top1={calibration_audit['development_metrics']['exact_top1_recall']:.6f}",
        file=sys.stderr,
        flush=True,
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=shared.DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path)
    subparsers = parser.add_subparsers(dest="command", required=True)
    inner = subparsers.add_parser("inner")
    inner.add_argument("--outer-fold", type=int, required=True)
    inner.add_argument("--inner-fold", type=int, required=True)
    inner_all = subparsers.add_parser("inner-all")
    inner_all.add_argument("--start-outer-fold", type=int, default=0)
    select = subparsers.add_parser("select")
    select.add_argument("--outer-fold", type=int, required=True)
    subparsers.add_parser("select-all")
    outer = subparsers.add_parser("outer")
    outer.add_argument("--outer-fold", type=int, required=True)
    subparsers.add_parser("outer-all")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config, _ = shared.read_config(args.config)
    if args.command == "inner":
        run_inner(
            args.config,
            outer_fold=args.outer_fold,
            inner_fold=args.inner_fold,
            output_root=args.output_root,
        )
    elif args.command == "inner-all":
        for outer_fold in range(args.start_outer_fold, int(config["outer_fold_count"])):
            for inner_fold in range(int(config["inner_fold_count"])):
                run_inner(
                    args.config,
                    outer_fold=outer_fold,
                    inner_fold=inner_fold,
                    output_root=args.output_root,
                )
    elif args.command == "select":
        select_outer(
            args.config,
            outer_fold=args.outer_fold,
            output_root=args.output_root,
        )
    elif args.command == "select-all":
        for outer_fold in range(int(config["outer_fold_count"])):
            select_outer(
                args.config, outer_fold=outer_fold, output_root=args.output_root
            )
    elif args.command == "outer":
        run_outer_refit(
            args.config,
            outer_fold=args.outer_fold,
            output_root=args.output_root,
        )
    elif args.command == "outer-all":
        for outer_fold in range(int(config["outer_fold_count"])):
            run_outer_refit(
                args.config, outer_fold=outer_fold, output_root=args.output_root
            )
    else:  # pragma: no cover
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
