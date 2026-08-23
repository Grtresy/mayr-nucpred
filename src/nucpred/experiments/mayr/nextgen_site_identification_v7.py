"""Mayr v7 site identification with a structural region residual.

The v6 neural rankers and all Stage E-C conditional-N checkpoints remain
frozen.  v7 learns only a candidate-set-aware residual that reorders nested
``delocalized_region`` candidates while preserving the v6 type-level maximum.
Development selection, exact calibration, and abstention use validation only;
test prediction is frozen before the sealed labels are opened.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
import gc
import json
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from nucpred.core.files import atomic_write_json, sha256_file
from nucpred.experiments.mayr import nextgen_site_identification as base
from nucpred.experiments.mayr import nextgen_site_identification_v6 as v6
from nucpred.experiments.mayr.nextgen_site_identification_v6_finalize import (
    _verify_stage_manifest,
)
from nucpred.project import get_project_layout
from nucpred.training.mayr_site_inference_assets import (
    ranker_from_checkpoint,
    score_ranker_from_source_features,
)
from nucpred.training.mayr_site_ranker import (
    RANKER_SITE_TYPES,
    TypeAwarePlattCalibrator,
    fit_type_aware_platt,
    site_type_indices,
)
from nucpred.training.mayr_site_region_residual import (
    REGION_FEATURE_SCHEMA,
    REGION_RESIDUAL_SCHEMA,
    REGION_SITE_TYPE,
    RegionResidualError,
    apply_region_residual,
    context_balanced_exact_weights,
    fit_region_residual_ensemble,
    origin_vocabulary,
    region_feature_matrix,
    score_region_residual,
)


ROOT = get_project_layout().root
DEFAULT_CONFIG = ROOT / "configs/mayr_nextgen_site_identification_v7.toml"
DEVELOPMENT_SCHEMA = "nucpred.mayr-site-identification-region-residual-development.v1"
TEST_PREDICTION_SCHEMA = (
    "nucpred.mayr-site-identification-region-residual-test-prediction.v1"
)
DEPLOYMENT_SCHEMA = "nucpred.mayr-site-identification-region-residual-deployment.v1"


def _base_development(config: Mapping[str, Any]) -> Path:
    return base._repo_path(
        config["ranker"]["base_development_directory"],
        label="v6 base development directory",
    )


def _verify_base_development(config: Mapping[str, Any]) -> dict[str, object]:
    directory = _base_development(config)
    base._verify_sha(
        directory / "run_manifest.json",
        config["ranker"]["base_development_manifest_sha256"],
        label="v6 base development manifest",
    )
    return _verify_stage_manifest(directory)


def _load_base_checkpoint(
    config: Mapping[str, Any],
    *,
    split_seed: int,
) -> tuple[dict[str, Any], Path]:
    path = _base_development(config) / f"split-{split_seed}" / "ranker_checkpoint.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise base.SiteIdentificationError("v6 base checkpoint is not a mapping")
    expected = {
        "schema_version": config["ranker"]["checkpoint_schema"],
        "phase": "development_frozen",
        "campaign_id": config["ranker"]["base_campaign_id"],
        "split_seed": split_seed,
        "test_labels_read": False,
        "test_predictions_computed": False,
        "conditional_n_backbone_frozen": True,
        "unknown_as_negative_count": 0,
        "candidate_softmax_used": False,
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise base.SiteIdentificationError(
                f"v6 base checkpoint boundary changed: {key}"
            )
    # Instantiation also verifies the frozen neural state hash.
    ranker_from_checkpoint(checkpoint)
    return checkpoint, path


def _fast_retrieval_metrics(
    frame: pd.DataFrame,
    scores: Sequence[float],
) -> dict[str, float | int]:
    values = np.asarray(scores, dtype=float)
    if values.shape != (len(frame),) or not np.isfinite(values).all():
        raise base.SiteIdentificationError("Residual validation scores are invalid")
    result = v6._complete_candidate_retrieval_metrics(
        labels=frame["exact_label"].to_numpy(dtype=int),
        scores=values,
        context_ids=frame["context_id"].astype(str).to_numpy(),
    )
    return {
        "eligible_context_count": int(result["eligible_target_count"]),
        "exact_top1_recall": float(result["top1_recall"]),
        "mrr": float(result["mrr"]),
        "exact_top3_recall": float(result["top3_recall"]),
    }


def _selection_key(metrics: Mapping[str, object]) -> tuple[float, float, float]:
    return (
        float(metrics["exact_top1_recall"]),
        float(metrics["mrr"]),
        float(metrics["exact_top3_recall"]),
    )


def _top_candidate_ids(
    frame: pd.DataFrame,
    scores: Sequence[float],
) -> dict[str, str]:
    work = frame.assign(_score=np.asarray(scores, dtype=float))
    result: dict[str, str] = {}
    for context_id, group in work.groupby("context_id", sort=True):
        ordered = group.sort_values(
            ["_score", "candidate_site_id"],
            ascending=[False, True],
            kind="stable",
        )
        result[str(context_id)] = str(ordered.iloc[0]["candidate_site_id"])
    return result


def _residual_search_settings(
    settings: Mapping[str, Any],
    *,
    residual_weight: float,
) -> list[tuple[float | None, int | None]]:
    values: list[tuple[float | None, int | None]] = []
    if bool(settings["include_ungated"]):
        values.append((None, None))
    # A deliberately small, predeclared low-margin Top-k ablation avoids
    # selecting across a combinatorial validation grid.
    if residual_weight in {
        float(value) for value in settings["low_margin_top_k_weight_grid"]
    }:
        values.append(
            (
                float(settings["low_margin_top_k_threshold"]),
                int(settings["low_margin_top_k"]),
            )
        )
    return list(dict.fromkeys(values))


def _region_true_contexts(targets: pd.DataFrame) -> set[str]:
    return set(
        targets.loc[
            targets["site_type"].astype(str).eq(REGION_SITE_TYPE),
            "context_id",
        ].astype(str)
    )


def _fit_feature_importance_audit(
    bundle: Mapping[str, object],
) -> list[dict[str, object]]:
    estimators = bundle.get("estimators")
    names = list(map(str, bundle.get("feature_names", ())))
    if not isinstance(estimators, list) or not estimators:
        raise base.SiteIdentificationError("Residual estimator ensemble is missing")
    importance = np.mean(
        np.stack([estimator.feature_importances_ for estimator in estimators]),
        axis=0,
    )
    order = np.argsort(-importance, kind="stable")
    return [
        {"feature": names[index], "mean_importance": float(importance[index])}
        for index in order[:20]
    ]


def _development_split(
    *,
    config: Mapping[str, Any],
    split_seed: int,
    output_directory: Path,
    final_output_directory: Path,
) -> dict[str, object]:
    contexts, targets, _, splits = base._dataset_tables(config)
    candidates, _ = base._deployment_candidates(config)
    membership = splits.loc[splits["split_seed"].eq(split_seed)].copy()
    development_membership = membership.loc[
        membership["role"].isin(["train", "validation"])
    ].copy()
    development_targets = targets.merge(
        development_membership[["target_id", "role"]],
        on="target_id",
        how="inner",
        validate="one_to_one",
    )
    development_contexts = (
        development_membership[["context_id", "species_id", "connectivity_id", "role"]]
        .drop_duplicates()
        .sort_values("context_id", kind="stable")
    )
    if development_contexts.groupby("context_id")["role"].nunique().ne(1).any():
        raise base.SiteIdentificationError("One context crosses development roles")
    universe = base._candidate_universe(
        test_contexts=development_contexts.drop(columns="role"),
        candidates=candidates,
    ).merge(
        development_contexts[["context_id", "role"]],
        on="context_id",
        how="left",
        validate="many_to_one",
    )
    query_ids, source_features, n_mean, n_std, backbone_bindings = (
        base._encode_split_ensemble(
            config=config,
            split_seed=split_seed,
            queries=universe,
            contexts=contexts,
            device=torch.device(str(config["device"])),
        )
    )
    ordered = (
        universe.set_index("query_id", drop=False).loc[query_ids].reset_index(drop=True)
    )
    ordered["conditional_N_mean"] = n_mean
    ordered["conditional_N_std"] = n_std
    ordered["exact_label"] = v6._exact_labels(
        ordered,
        development_targets,
    ).astype(int)

    base_checkpoint, base_checkpoint_path = _load_base_checkpoint(
        config,
        split_seed=split_seed,
    )
    base_ranker = ranker_from_checkpoint(base_checkpoint)
    type_index = site_type_indices(ordered["site_type"].astype(str))
    with torch.no_grad():
        component_tensors = score_ranker_from_source_features(
            ranker=base_ranker,
            checkpoint=base_checkpoint,
            source_features=source_features,
            type_index=type_index,
        )
    components = {key: value.cpu().numpy() for key, value in component_tensors.items()}
    base_logits = components["canonical_logit"]
    validation_mask = ordered["role"].astype(str).eq("validation").to_numpy()
    validation_frame_for_search = ordered.loc[validation_mask].reset_index(drop=True)
    origin_values = origin_vocabulary(candidates)
    region_positions, region_features, feature_names = region_feature_matrix(
        ordered,
        membership_logits=components["membership_logit"],
        compatibility_logits=components["compatibility_logit"],
        conditional_n_mean=n_mean,
        conditional_n_std=n_std,
        origin_vocabulary_values=origin_values,
    )

    train_targets = development_targets.loc[
        development_targets["role"].eq("train")
    ].copy()
    validation_targets = development_targets.loc[
        development_targets["role"].eq("validation")
    ].copy()
    train_region_contexts = _region_true_contexts(train_targets)
    region_rows = ordered.iloc[region_positions].reset_index(drop=True)
    training_feature_mask = (
        region_rows["role"].astype(str).eq("train")
        & region_rows["context_id"].astype(str).isin(train_region_contexts)
    ).to_numpy()
    training_frame = region_rows.loc[training_feature_mask].reset_index(drop=True)
    training_features = region_features[training_feature_mask]
    training_weights = context_balanced_exact_weights(training_frame)
    if int(training_frame["exact_label"].sum()) != len(train_region_contexts):
        raise base.SiteIdentificationError(
            "Region residual training target coverage changed"
        )

    residual_settings = config["ranker"]["region_residual"]
    ensemble_seeds = [
        int(residual_settings["training_seed_offset"]) + split_seed + int(offset)
        for offset in residual_settings["ensemble_seed_offsets"]
    ]
    base_metrics = _fast_retrieval_metrics(
        validation_frame_for_search,
        base_logits[validation_mask],
    )
    search_rows: list[dict[str, object]] = [
        {
            "arm": "frozen_v6",
            "minimum_samples_leaf": None,
            "residual_weight": 0.0,
            "maximum_base_margin": None,
            "top_k": None,
            **base_metrics,
        }
    ]
    bundles: dict[int, dict[str, object]] = {}
    residual_scores: dict[int, np.ndarray] = {}
    candidate_outputs: dict[
        tuple[int, float, float | None, int | None], np.ndarray
    ] = {}
    application_audits: dict[
        tuple[int, float, float | None, int | None], dict[str, object]
    ] = {}
    for minimum_samples_leaf_raw in residual_settings["minimum_samples_leaf_grid"]:
        minimum_samples_leaf = int(minimum_samples_leaf_raw)
        bundle = fit_region_residual_ensemble(
            training_features,
            training_frame["exact_label"].to_numpy(dtype=int),
            sample_weights=training_weights,
            minimum_samples_leaf=minimum_samples_leaf,
            estimator_count_per_seed=int(residual_settings["estimator_count_per_seed"]),
            maximum_features=float(residual_settings["maximum_features"]),
            seeds=ensemble_seeds,
            feature_names=feature_names,
            origin_vocabulary_values=origin_values,
        )
        bundles[minimum_samples_leaf] = bundle
        probabilities = score_region_residual(
            bundle,
            region_features,
            expected_feature_names=feature_names,
        )
        residual_scores[minimum_samples_leaf] = probabilities
        for residual_weight_raw in residual_settings["residual_weight_grid"]:
            residual_weight = float(residual_weight_raw)
            for maximum_margin, top_k in _residual_search_settings(
                residual_settings,
                residual_weight=residual_weight,
            ):
                logits, application_audit = apply_region_residual(
                    ordered,
                    base_logits=base_logits,
                    region_positions=region_positions,
                    residual_probabilities=probabilities,
                    residual_weight=residual_weight,
                    maximum_base_margin=maximum_margin,
                    top_k=top_k,
                )
                metrics = _fast_retrieval_metrics(
                    validation_frame_for_search,
                    logits[validation_mask],
                )
                key = (
                    minimum_samples_leaf,
                    residual_weight,
                    maximum_margin,
                    top_k,
                )
                candidate_outputs[key] = logits
                application_audits[key] = application_audit
                search_rows.append(
                    {
                        "arm": "region_structural_residual",
                        "minimum_samples_leaf": minimum_samples_leaf,
                        "residual_weight": residual_weight,
                        "maximum_base_margin": maximum_margin,
                        "top_k": top_k,
                        **metrics,
                    }
                )

    selected_row = max(
        search_rows,
        key=lambda row: _selection_key(row),
    )
    selected_enabled = selected_row["arm"] == "region_structural_residual"
    selected_bundle: dict[str, object] | None = None
    selected_probabilities = np.full(len(region_positions), np.nan, dtype=float)
    selected_application: dict[str, object] = {
        "schema_version": "nucpred.mayr-region-residual-application-audit.v1",
        "enabled": False,
    }
    if selected_enabled:
        selected_key = (
            int(selected_row["minimum_samples_leaf"]),
            float(selected_row["residual_weight"]),
            (
                float(selected_row["maximum_base_margin"])
                if pd.notna(selected_row["maximum_base_margin"])
                else None
            ),
            int(selected_row["top_k"]) if pd.notna(selected_row["top_k"]) else None,
        )
        selected_logits = candidate_outputs[selected_key]
        selected_application = application_audits[selected_key]
        selected_bundle = bundles[selected_key[0]]
        selected_probabilities = residual_scores[selected_key[0]]
    else:
        selected_logits = base_logits.copy()

    validation_frame = ordered.loc[validation_mask].reset_index(drop=True)
    validation_logits = selected_logits[validation_mask]
    validation_components = {
        key: value[validation_mask] for key, value in components.items()
    }
    validation_metrics = v6._validation_metrics(
        frame=validation_frame,
        logits=validation_logits,
        membership_logits=validation_components["membership_logit"],
        router_logits=validation_components["router_logits"],
        validation_targets=validation_targets,
    )
    validation_weights = v6._context_uniform_candidate_weights(validation_frame)
    validation_type_index = site_type_indices(validation_frame["site_type"].astype(str))
    calibrator, calibrator_audit = fit_type_aware_platt(
        logits=torch.tensor(validation_logits, dtype=torch.float32),
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
    with torch.no_grad():
        calibrated = calibrator(
            torch.tensor(validation_logits, dtype=torch.float32),
            validation_type_index,
        ).numpy()
    margin_gate = v6.select_margin_threshold(
        margins=validation_metrics["top1_margin"],
        top1_correct=validation_metrics["top1_correct"],
        thresholds=config["abstention"]["threshold_grid"],
        minimum_precision=float(config["abstention"]["minimum_development_precision"]),
        minimum_accepted_count=int(config["abstention"]["minimum_accepted_count"]),
    )

    split_directory = output_directory / f"split-{split_seed}"
    split_directory.mkdir(parents=True)
    residual_binding: dict[str, object] = {
        "enabled": selected_enabled,
        "schema_version": REGION_RESIDUAL_SCHEMA,
        "feature_schema_version": REGION_FEATURE_SCHEMA,
        "target_site_type": REGION_SITE_TYPE,
        "origin_vocabulary": list(origin_values),
        "feature_names": list(feature_names),
        "residual_weight": float(selected_row["residual_weight"]),
        "maximum_base_margin": (
            float(selected_row["maximum_base_margin"])
            if pd.notna(selected_row["maximum_base_margin"])
            else None
        ),
        "top_k": (
            int(selected_row["top_k"]) if pd.notna(selected_row["top_k"]) else None
        ),
        "candidate_set_conditioned": True,
        "type_level_maximum_preserved": True,
        "candidate_softmax_used": False,
    }
    if selected_bundle is not None:
        residual_path = split_directory / "region_residual.joblib"
        joblib.dump(selected_bundle, residual_path, compress=3)
        final_residual_path = (
            final_output_directory / f"split-{split_seed}" / "region_residual.joblib"
        )
        residual_binding.update(
            {
                "path": base._display_path(final_residual_path),
                "sha256": sha256_file(residual_path),
                "minimum_samples_leaf": int(selected_bundle["minimum_samples_leaf"]),
                "estimator_count_per_seed": int(
                    selected_bundle["estimator_count_per_seed"]
                ),
                "maximum_features": float(selected_bundle["maximum_features"]),
                "seeds": list(selected_bundle["seeds"]),
            }
        )
    checkpoint = deepcopy(base_checkpoint)
    checkpoint.update(
        {
            "campaign_id": config["campaign_id"],
            "selected_arm": str(selected_row["arm"]),
            "calibrator": calibrator.to_payload(),
            "calibrator_fit_audit": calibrator_audit,
            "calibration_population": config["calibration"]["population"],
            "calibration_weighting": config["calibration"]["weighting"],
            "margin_abstention": margin_gate,
            "backbone_bindings": backbone_bindings,
            "base_v6_binding": {
                "campaign_id": config["ranker"]["base_campaign_id"],
                "path": base._display_path(base_checkpoint_path),
                "sha256": sha256_file(base_checkpoint_path),
            },
            "region_membership_residual": residual_binding,
            "training_roles": ["train"],
            "selection_calibration_abstention_roles": ["validation"],
            "test_labels_read": False,
            "test_predictions_computed": False,
            "conditional_n_backbone_frozen": True,
            "unknown_as_negative_count": 0,
            "candidate_softmax_used": False,
            "candidate_set_conditioned_structural_residual": selected_enabled,
            "type_level_maximum_preserved": True,
            "final_refit_performed": False,
        }
    )
    torch.save(checkpoint, split_directory / "ranker_checkpoint.pt")

    validation_region_mask = (
        validation_frame["site_type"].astype(str).eq(REGION_SITE_TYPE)
    )
    validation_predictions = validation_frame.copy()
    validation_predictions["base_v6_validity_logit"] = validation_components[
        "canonical_logit"
    ]
    validation_predictions["validity_logit"] = validation_logits
    validation_predictions["membership_logit"] = validation_components[
        "membership_logit"
    ]
    validation_predictions["router_selected_logit"] = validation_components[
        "router_selected_logit"
    ]
    validation_predictions["compatibility_logit"] = validation_components[
        "compatibility_logit"
    ]
    validation_predictions["region_residual_probability"] = np.nan
    validation_region_source_positions = np.flatnonzero(
        np.isin(region_positions, np.flatnonzero(validation_mask))
    )
    validation_predictions.loc[
        validation_region_mask,
        "region_residual_probability",
    ] = selected_probabilities[validation_region_source_positions]
    validation_predictions["raw_sigmoid_score"] = base._sigmoid(validation_logits)
    validation_predictions["absolute_site_probability"] = calibrated
    validation_predictions["evaluation_weight"] = validation_weights.numpy()
    validation_predictions.to_parquet(
        split_directory / "validation_fullspace_predictions.parquet",
        index=False,
        compression="zstd",
    )
    training_export = training_frame[
        [
            "context_id",
            "candidate_site_id",
            "site_type",
            "member_atom_indices_json",
            "candidate_origins_json",
            "exact_label",
        ]
    ].copy()
    training_export["training_weight"] = training_weights
    training_export.to_parquet(
        split_directory / "region_residual_training_population.parquet",
        index=False,
        compression="zstd",
    )
    pd.DataFrame(search_rows).to_csv(
        split_directory / "residual_arm_search.csv",
        index=False,
    )
    base_top = _top_candidate_ids(
        validation_frame, validation_components["canonical_logit"]
    )
    selected_top = _top_candidate_ids(validation_frame, validation_logits)
    truth = (
        validation_targets.groupby("context_id")["site_object_id"]
        .agg(lambda values: set(map(str, values)))
        .to_dict()
    )
    changed = [
        context_id
        for context_id in base_top
        if base_top[context_id] != selected_top[context_id]
    ]
    transition_audit = {
        "changed_top1_context_count": len(changed),
        "v6_wrong_v7_correct_count": sum(
            base_top[context_id] not in truth[context_id]
            and selected_top[context_id] in truth[context_id]
            for context_id in changed
        ),
        "v6_correct_v7_wrong_count": sum(
            base_top[context_id] in truth[context_id]
            and selected_top[context_id] not in truth[context_id]
            for context_id in changed
        ),
    }
    fit_audit = {
        "schema_version": "nucpred.mayr-region-residual-fit-audit.v1",
        "split_seed": split_seed,
        "selected_arm": selected_row,
        "base_v6_metrics": base_metrics,
        "selected_metrics": {
            key: value
            for key, value in validation_metrics.items()
            if key not in {"top1_correct", "top1_margin"}
        },
        "training_region_context_count": len(train_region_contexts),
        "training_region_candidate_count": len(training_frame),
        "training_exact_positive_count": int(training_frame["exact_label"].sum()),
        "origin_vocabulary": list(origin_values),
        "feature_names": list(feature_names),
        "ensemble_seeds": ensemble_seeds,
        "selected_application": selected_application,
        "validation_transition_audit": transition_audit,
        "top_feature_importances": (
            _fit_feature_importance_audit(selected_bundle)
            if selected_bundle is not None
            else []
        ),
        "unknown_as_negative_count": 0,
        "test_labels_read": False,
    }
    atomic_write_json(split_directory / "region_residual_fit_audit.json", fit_audit)
    atomic_write_json(split_directory / "margin_abstention.json", margin_gate)

    result = {
        "split_seed": split_seed,
        "selected_arm": str(selected_row["arm"]),
        "selection_key": list(_selection_key(selected_row)),
        "base_v6_selection_key": list(_selection_key(base_metrics)),
        "validation_metrics": fit_audit["selected_metrics"],
        "base_v6_validation_metrics": base_metrics,
        "validation_top1_delta_vs_v6": float(validation_metrics["exact_top1_recall"])
        - float(base_metrics["exact_top1_recall"]),
        "training_region_context_count": len(train_region_contexts),
        "training_region_candidate_count": len(training_frame),
        "validation_full_candidate_count": len(validation_frame),
        "validation_context_count": int(
            validation_frame["context_id"].astype(str).nunique()
        ),
        "selected_residual": residual_binding,
        "validation_transition_audit": transition_audit,
        "margin_abstention": margin_gate,
        "ranker_checkpoint_sha256": sha256_file(
            split_directory / "ranker_checkpoint.pt"
        ),
        "test_labels_read": False,
        "test_predictions_computed": False,
    }
    del source_features, base_ranker, component_tensors, components, bundles
    gc.collect()
    print(
        f"split={split_seed} v6_top1={base_metrics['exact_top1_recall']:.6f} "
        f"v7_top1={validation_metrics['exact_top1_recall']:.6f} "
        f"arm={selected_row['arm']}",
        file=sys.stderr,
        flush=True,
    )
    return result


def run_development(
    config: Mapping[str, Any],
    *,
    config_path: Path,
) -> dict[str, Any]:
    output_root = base._repo_path(config["output_directory"], label="output directory")
    preflight = output_root / "preflight"
    if base._load_json(preflight / "summary.json").get("status") != "pass":
        raise base.SiteIdentificationError("Preflight is not complete")
    base_development_audit = _verify_base_development(config)
    target = output_root / "development"

    def writer(staged: Path) -> dict[str, Any]:
        split_summaries = [
            _development_split(
                config=config,
                split_seed=int(split_seed),
                output_directory=staged,
                final_output_directory=target,
            )
            for split_seed in config["backbone"]["split_seeds"]
        ]
        rows = pd.DataFrame(
            [
                {
                    "split_seed": item["split_seed"],
                    "selected_arm": item["selected_arm"],
                    "base_v6_exact_top1_recall": item["base_v6_validation_metrics"][
                        "exact_top1_recall"
                    ],
                    "v7_exact_top1_recall": item["validation_metrics"][
                        "exact_top1_recall"
                    ],
                    "exact_top1_delta_v7_minus_v6": item["validation_top1_delta_vs_v6"],
                    "v7_mrr": item["validation_metrics"]["mrr"],
                    "changed_top1_context_count": item["validation_transition_audit"][
                        "changed_top1_context_count"
                    ],
                    "v6_wrong_v7_correct_count": item["validation_transition_audit"][
                        "v6_wrong_v7_correct_count"
                    ],
                    "v6_correct_v7_wrong_count": item["validation_transition_audit"][
                        "v6_correct_v7_wrong_count"
                    ],
                    "margin_threshold": item["margin_abstention"]["selected_threshold"],
                }
                for item in split_summaries
            ]
        )
        rows.to_csv(staged / "split_summary.csv", index=False)
        v7 = rows["v7_exact_top1_recall"].to_numpy(dtype=float)
        v6_values = rows["base_v6_exact_top1_recall"].to_numpy(dtype=float)
        return {
            "schema_version": DEVELOPMENT_SCHEMA,
            "status": "pass",
            "campaign_id": config["campaign_id"],
            "config_sha256": sha256_file(config_path),
            "preflight_manifest_sha256": sha256_file(preflight / "run_manifest.json"),
            "base_v6_development_audit": base_development_audit,
            "split_summaries": split_summaries,
            "macro_validation_exact_top1_recall": float(v7.mean()),
            "macro_base_v6_validation_exact_top1_recall": float(v6_values.mean()),
            "macro_validation_top1_delta_vs_v6": float((v7 - v6_values).mean()),
            "non_regressing_split_count": int((v7 >= v6_values).sum()),
            "improving_split_count": int((v7 > v6_values).sum()),
            "development_frozen": True,
            "full_candidate_validation_used": True,
            "conditional_n_backbone_frozen": True,
            "base_v6_rankers_frozen": True,
            "region_type_level_maximum_preserved": True,
            "test_labels_read": False,
            "test_predictions_computed": False,
            "unknown_as_negative_count": 0,
            "candidate_softmax_used": False,
        }

    return base._publish_stage(
        target,
        schema_version=DEVELOPMENT_SCHEMA,
        writer=writer,
    )


def _load_residual_bundle(
    checkpoint: Mapping[str, Any],
) -> tuple[Mapping[str, object] | None, Mapping[str, object]]:
    binding = checkpoint.get("region_membership_residual")
    if not isinstance(binding, Mapping):
        raise base.SiteIdentificationError("Region residual binding is missing")
    if not bool(binding.get("enabled")):
        return None, binding
    path = base._repo_path(binding["path"], label="region residual checkpoint")
    base._verify_sha(path, binding["sha256"], label="region residual checkpoint")
    bundle = joblib.load(path)
    if not isinstance(bundle, Mapping):
        raise base.SiteIdentificationError("Region residual checkpoint is invalid")
    return bundle, binding


def score_checkpoint_with_region_residual(
    *,
    frame: pd.DataFrame,
    checkpoint: Mapping[str, Any],
    components: Mapping[str, torch.Tensor],
    conditional_n_mean: Sequence[float],
    conditional_n_std: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Apply the frozen optional residual to one complete candidate set."""

    base_logits = components["canonical_logit"].cpu().numpy()
    bundle, binding = _load_residual_bundle(checkpoint)
    if bundle is None:
        return (
            base_logits,
            np.full(len(frame), np.nan, dtype=float),
            {"enabled": False},
        )
    try:
        positions, features, feature_names = region_feature_matrix(
            frame,
            membership_logits=components["membership_logit"].cpu().numpy(),
            compatibility_logits=components["compatibility_logit"].cpu().numpy(),
            conditional_n_mean=conditional_n_mean,
            conditional_n_std=conditional_n_std,
            origin_vocabulary_values=binding["origin_vocabulary"],
        )
        probabilities = score_region_residual(
            bundle,
            features,
            expected_feature_names=feature_names,
        )
        logits, audit = apply_region_residual(
            frame,
            base_logits=base_logits,
            region_positions=positions,
            residual_probabilities=probabilities,
            residual_weight=float(binding["residual_weight"]),
            maximum_base_margin=(
                float(binding["maximum_base_margin"])
                if binding.get("maximum_base_margin") is not None
                else None
            ),
            top_k=(int(binding["top_k"]) if binding.get("top_k") is not None else None),
        )
    except RegionResidualError as exc:
        raise base.SiteIdentificationError(str(exc)) from exc
    full_probabilities = np.full(len(frame), np.nan, dtype=float)
    full_probabilities[positions] = probabilities
    audit["enabled"] = True
    return logits, full_probabilities, audit


def run_test_predictions(
    config: Mapping[str, Any],
    *,
    config_path: Path,
) -> dict[str, Any]:
    """Freeze v7 full-space test predictions without opening test labels."""

    output_root = base._repo_path(config["output_directory"], label="output directory")
    preflight = output_root / "preflight"
    development = output_root / "development"
    development_summary = base._load_json(development / "summary.json")
    if (
        development_summary.get("status") != "pass"
        or development_summary.get("development_frozen") is not True
        or development_summary.get("test_labels_read") is not False
    ):
        raise base.SiteIdentificationError("Development freeze is not complete")
    target = output_root / "test_predictions"

    def writer(staged: Path) -> dict[str, Any]:
        contexts, _, _, _ = base._dataset_tables(config)
        candidates, candidate_policy_audit = base._deployment_candidates(config)
        split_summaries: list[dict[str, object]] = []
        for split_seed_raw in config["backbone"]["split_seeds"]:
            split_seed = int(split_seed_raw)
            test_contexts = pd.read_parquet(
                preflight / f"split-{split_seed}" / "test_contexts.unlabeled.parquet"
            )
            universe = base._candidate_universe(
                test_contexts=test_contexts,
                candidates=candidates,
            )
            query_ids, features, n_mean, n_std, backbone_bindings = (
                base._encode_split_ensemble(
                    config=config,
                    split_seed=split_seed,
                    queries=universe,
                    contexts=contexts,
                    device=torch.device(str(config["device"])),
                )
            )
            ordered = (
                universe.set_index("query_id", drop=False)
                .loc[query_ids]
                .reset_index(drop=True)
            )
            checkpoint_path = (
                development / f"split-{split_seed}" / "ranker_checkpoint.pt"
            )
            checkpoint = base._load_ranker_checkpoint(
                checkpoint_path,
                split_seed=split_seed,
                config=config,
            )
            ranker = ranker_from_checkpoint(checkpoint)
            type_index = site_type_indices(ordered["site_type"].astype(str))
            with torch.no_grad():
                components = score_ranker_from_source_features(
                    ranker=ranker,
                    checkpoint=checkpoint,
                    source_features=features,
                    type_index=type_index,
                )
            logits, residual_probability, residual_audit = (
                score_checkpoint_with_region_residual(
                    frame=ordered,
                    checkpoint=checkpoint,
                    components=components,
                    conditional_n_mean=n_mean,
                    conditional_n_std=n_std,
                )
            )
            calibrator_payload = checkpoint.get("calibrator")
            if not isinstance(calibrator_payload, Mapping):
                raise base.SiteIdentificationError("Ranker calibrator is missing")
            calibrator = TypeAwarePlattCalibrator.from_payload(calibrator_payload)
            with torch.no_grad():
                probability = calibrator(
                    torch.tensor(logits, dtype=torch.float32),
                    type_index,
                ).numpy()
            predictions = ordered.copy()
            predictions["base_v6_validity_logit"] = components[
                "canonical_logit"
            ].numpy()
            predictions["validity_logit"] = logits
            predictions["membership_logit"] = components["membership_logit"].numpy()
            predictions["router_selected_logit"] = components[
                "router_selected_logit"
            ].numpy()
            predictions["compatibility_logit"] = components[
                "compatibility_logit"
            ].numpy()
            predictions["region_residual_probability"] = residual_probability
            predictions["raw_sigmoid_score"] = base._sigmoid(logits)
            predictions["absolute_site_probability"] = probability
            predictions["conditional_N_mean"] = n_mean
            predictions["conditional_N_std"] = n_std
            predictions["prediction_split_seed"] = split_seed
            predictions["selected_ranker_arm"] = str(checkpoint["selected_arm"])
            predictions["candidate_scores_independent"] = True
            predictions["candidate_set_conditioned_structural_residual"] = bool(
                checkpoint["region_membership_residual"]["enabled"]
            )
            predictions["candidate_softmax_used"] = False
            predictions["target_or_site_label_read"] = False
            split_directory = staged / f"split-{split_seed}"
            split_directory.mkdir()
            prediction_path = split_directory / "candidate_predictions.parquet"
            predictions.to_parquet(
                prediction_path,
                index=False,
                compression="zstd",
            )
            freeze = {
                "schema_version": TEST_PREDICTION_SCHEMA,
                "campaign_id": config["campaign_id"],
                "split_seed": split_seed,
                "candidate_prediction_path": base._display_path(
                    output_root
                    / "test_predictions"
                    / f"split-{split_seed}"
                    / "candidate_predictions.parquet"
                ),
                "candidate_prediction_sha256": sha256_file(prediction_path),
                "candidate_prediction_row_count": len(predictions),
                "test_context_count": int(
                    predictions["context_id"].astype(str).nunique()
                ),
                "ranker_checkpoint_path": base._display_path(checkpoint_path),
                "ranker_checkpoint_sha256": sha256_file(checkpoint_path),
                "region_residual_binding": checkpoint["region_membership_residual"],
                "region_residual_application_audit": residual_audit,
                "backbone_bindings": backbone_bindings,
                "test_labels_read": False,
                "target_values_read": False,
                "candidate_softmax_used": False,
                "unknown_as_negative_count": 0,
            }
            freeze["freeze_sha256"] = base._canonical_sha256(freeze)
            atomic_write_json(split_directory / "prediction_freeze.json", freeze)
            counts = Counter(predictions["site_type"].astype(str))
            split_summaries.append(
                {
                    "split_seed": split_seed,
                    "candidate_prediction_count": len(predictions),
                    "test_context_count": int(
                        predictions["context_id"].astype(str).nunique()
                    ),
                    "candidate_count_by_type": {
                        site_type: int(counts[site_type])
                        for site_type in RANKER_SITE_TYPES
                    },
                    "selected_ranker_arm": str(checkpoint["selected_arm"]),
                    "region_residual_application_audit": residual_audit,
                    "prediction_sha256": freeze["candidate_prediction_sha256"],
                    "test_labels_read": False,
                }
            )
            del features, ranker, predictions, components
            gc.collect()
        pd.DataFrame(
            [
                {
                    "split_seed": item["split_seed"],
                    "candidate_prediction_count": item["candidate_prediction_count"],
                    "test_context_count": item["test_context_count"],
                    "selected_ranker_arm": item["selected_ranker_arm"],
                    "prediction_sha256": item["prediction_sha256"],
                }
                for item in split_summaries
            ]
        ).to_csv(staged / "split_summary.csv", index=False)
        return {
            "schema_version": TEST_PREDICTION_SCHEMA,
            "status": "pass",
            "campaign_id": config["campaign_id"],
            "config_sha256": sha256_file(config_path),
            "development_manifest_sha256": sha256_file(
                development / "run_manifest.json"
            ),
            "split_summaries": split_summaries,
            "candidate_policy_audit": candidate_policy_audit,
            "test_predictions_frozen": True,
            "test_labels_read": False,
            "target_values_read": False,
            "unknown_as_negative_count": 0,
            "candidate_softmax_used": False,
        }

    return base._publish_stage(
        target,
        schema_version=TEST_PREDICTION_SCHEMA,
        writer=writer,
    )


def run_deployment_registry(
    config: Mapping[str, Any],
    *,
    config_path: Path,
) -> dict[str, Any]:
    """Register v7 only after its paired v6 comparison clears every gate."""

    output_root = base._repo_path(config["output_directory"], label="output directory")
    development = output_root / "development"
    evaluation = output_root / "test_evaluation"
    comparison = output_root / "comparison"
    evaluation_summary = base._load_json(evaluation / "summary.json")
    comparison_summary = base._load_json(comparison / "summary.json")
    if (
        evaluation_summary.get("five_split_test_complete") is not True
        or comparison_summary.get("deployment_approved") is not True
    ):
        raise base.SiteIdentificationError(
            "v7 formal evidence did not authorize deployment"
        )
    target = output_root / "deployment"

    def writer(staged: Path) -> dict[str, Any]:
        split_models: list[dict[str, object]] = []
        margin_thresholds: list[float] = []
        for split_seed_raw in config["backbone"]["split_seeds"]:
            split_seed = int(split_seed_raw)
            ranker_path = development / f"split-{split_seed}" / "ranker_checkpoint.pt"
            checkpoint = base._load_ranker_checkpoint(
                ranker_path,
                split_seed=split_seed,
                config=config,
            )
            backbone_bindings = checkpoint.get("backbone_bindings")
            if not isinstance(backbone_bindings, list):
                raise base.SiteIdentificationError("Backbone bindings are missing")
            for binding in backbone_bindings:
                if not isinstance(binding, Mapping):
                    raise base.SiteIdentificationError("Invalid backbone binding")
                path = base._repo_path(
                    binding["path"],
                    label="registered backbone checkpoint",
                )
                base._verify_sha(
                    path,
                    binding["sha256"],
                    label="registered backbone checkpoint",
                )
            residual_bundle, residual_binding = _load_residual_bundle(checkpoint)
            if residual_bundle is None or residual_binding.get("enabled") is not True:
                raise base.SiteIdentificationError(
                    "Selected v7 split lacks its region residual"
                )
            margin_payload = checkpoint.get("margin_abstention")
            if not isinstance(margin_payload, Mapping):
                raise base.SiteIdentificationError(
                    "Checkpoint margin abstention payload is invalid"
                )
            margin_thresholds.append(float(margin_payload["selected_threshold"]))
            split_models.append(
                {
                    "split_seed": split_seed,
                    "ranker_checkpoint_path": base._display_path(ranker_path),
                    "ranker_checkpoint_sha256": sha256_file(ranker_path),
                    "selected_arm": checkpoint["selected_arm"],
                    "base_v6_binding": checkpoint["base_v6_binding"],
                    "region_membership_residual": dict(residual_binding),
                    "backbone_bindings": backbone_bindings,
                    "margin_abstention": dict(margin_payload),
                }
            )
        margin_enabled = bool(config["runtime"]["margin_abstention_enabled"])
        if margin_enabled and len(margin_thresholds) != len(split_models):
            raise base.SiteIdentificationError(
                "Runtime margin gate lacks one or more split thresholds"
            )
        registry: dict[str, object] = {
            "schema_version": base.RUNTIME_REGISTRY_SCHEMA,
            "campaign_id": config["campaign_id"],
            "created_at_utc": base._utc_now(),
            "runtime_mode": config["runtime"]["mode"],
            "unseen_feature_policy": config["runtime"]["unseen_feature_policy"],
            "dataset_directory": config["dataset"]["directory"],
            "dataset_manifest_sha256": config["dataset"]["dataset_manifest_sha256"],
            "candidate_policy_path": config["candidate_policy"]["path"],
            "candidate_policy_sha256": config["candidate_policy"]["sha256"],
            "candidate_policy_filter": (
                "gate_a_deployment_and_response_membership_contract"
            ),
            "candidate_types": list(RANKER_SITE_TYPES),
            "calibrated_site_types": list(config["runtime"]["calibrated_site_types"]),
            "split_models": split_models,
            "validity_ensemble_semantics": (
                "mean_of_five_cross_fit_v6_logits_after_split_region_residual"
            ),
            "probability_ensemble_semantics": (
                "mean_of_five_split_specific_v7_calibrated_probabilities"
            ),
            "conditional_n_ensemble_semantics": (
                "mean_of_fifteen_frozen_stage_e_c_predictions"
            ),
            "region_residual_ensemble_semantics": (
                "three_seed_extra_trees_mean_then_type_max_preserving_rerank"
            ),
            "formal_test_summary_path": base._display_path(evaluation / "summary.json"),
            "formal_test_summary_sha256": sha256_file(evaluation / "summary.json"),
            "paired_v6_comparison_path": base._display_path(
                comparison / "summary.json"
            ),
            "paired_v6_comparison_sha256": sha256_file(comparison / "summary.json"),
            "response_schema_path": config["contract"]["response_schema_path"],
            "response_schema_sha256": config["contract"]["response_schema_sha256"],
            "candidate_generator_path": config["contract"]["candidate_generator_path"],
            "candidate_generator_sha256": config["contract"][
                "candidate_generator_sha256"
            ],
            "conditional_n_backbone_frozen": True,
            "base_v6_rankers_frozen": True,
            "final_refit_performed": False,
            "target_or_site_label_read_at_inference": False,
            "candidate_scores_independent": True,
            "candidate_set_conditioned_structural_residual": True,
            "region_type_level_maximum_preserved": True,
            "candidate_softmax_used": False,
            "no_site_claim_permitted": False,
            "margin_abstention_enabled": margin_enabled,
            "margin_threshold_aggregation": config["abstention"][
                "runtime_threshold_aggregation"
            ],
            "runtime_margin_threshold": float(np.median(margin_thresholds)),
            "low_margin_runtime_status": config["abstention"][
                "low_margin_runtime_status"
            ],
        }
        registry["registry_sha256"] = base._canonical_sha256(registry)
        atomic_write_json(staged / "runtime_registry.json", registry)
        return {
            "schema_version": DEPLOYMENT_SCHEMA,
            "status": "pass",
            "campaign_id": config["campaign_id"],
            "config_sha256": sha256_file(config_path),
            "runtime_registry_path": base._display_path(
                output_root / "deployment" / "runtime_registry.json"
            ),
            "runtime_registry_sha256": sha256_file(staged / "runtime_registry.json"),
            "registered_split_model_count": len(split_models),
            "registered_region_residual_count": len(split_models),
            "registered_backbone_checkpoint_count": sum(
                len(item["backbone_bindings"]) for item in split_models
            ),
            "formal_test_summary_sha256": registry["formal_test_summary_sha256"],
            "paired_v6_comparison_sha256": registry["paired_v6_comparison_sha256"],
            "deployment_criteria": comparison_summary["deployment_criteria"],
            "final_refit_performed": False,
            "target_or_site_label_read_at_inference": False,
            "candidate_softmax_used": False,
            "margin_abstention_enabled": margin_enabled,
            "runtime_margin_threshold": registry["runtime_margin_threshold"],
        }

    return base._publish_stage(
        target,
        schema_version=DEPLOYMENT_SCHEMA,
        writer=writer,
    )


def run_all(
    config: Mapping[str, Any],
    *,
    config_path: Path,
) -> dict[str, Any]:
    return {
        "status": "pass",
        "preflight": base.run_preflight(config, config_path=config_path),
        "development": run_development(config, config_path=config_path),
        "test_predictions": run_test_predictions(config, config_path=config_path),
        "test_evaluation": base.run_test_evaluation(
            config,
            config_path=config_path,
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["preflight", "develop", "predict-test", "test", "deploy", "all"],
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = args.config.resolve()
    config = base.read_config(config_path)
    runners = {
        "preflight": base.run_preflight,
        "develop": run_development,
        "predict-test": run_test_predictions,
        "test": base.run_test_evaluation,
        "deploy": run_deployment_registry,
        "all": run_all,
    }
    result = runners[args.command](config, config_path=config_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
