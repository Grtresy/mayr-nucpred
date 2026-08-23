"""Aggregate formal outer OOF predictions and apply the prototype hard gate."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np
import pandas as pd

from nucpred.core.files import atomic_write_json, sha256_file
from nucpred.experiments.mayr.joint_site_n import (
    DEFAULT_CONFIG,
    JointSiteNExperimentError,
    _display_path,
    _project_path,
    read_config,
    verify_input_bindings,
)
from nucpred.experiments.mayr.joint_site_n_outer import (
    EVALUATION_SUMMARY_SCHEMA,
    OUTER_SUMMARY_SCHEMA,
    SCORE_SUMMARY_SCHEMA,
    _assert_current_source_hashes,
    _context_metric_summary,
    _implementation_source_hashes,
    _load_json,
    _outer_refit_directory,
    _verify_manifest,
)


GATE_SUMMARY_SCHEMA = "nucpred.mayr-joint-site-n-prototype-gate.v1"
PRE_ROUTER_CANONICAL_LOGIT_SEMANTICS = (
    "frozen_v2_base_plus_mean_joint_residual"
)
ROUTED_CANONICAL_LOGIT_SEMANTICS = (
    "type_router_plus_weighted_pre_router_type_max_plus_"
    "within_type_relative.v1"
)


def _assert_outer_ensemble_score_contract(
    score_summary: Mapping[str, object],
    candidates: pd.DataFrame,
) -> None:
    """Verify the routed ensemble's frozen score composition and lineage."""

    router_binding = score_summary.get("type_router_binding")
    transport_audit = score_summary.get("type_router_feature_transport_audit")
    if (
        score_summary.get("pre_router_canonical_logit_semantics")
        != PRE_ROUTER_CANONICAL_LOGIT_SEMANTICS
        or score_summary.get("canonical_logit_semantics")
        != ROUTED_CANONICAL_LOGIT_SEMANTICS
        or not isinstance(router_binding, Mapping)
        or router_binding.get("status") != "pass"
        or not isinstance(transport_audit, Mapping)
        or transport_audit.get("status") != "pass"
        or transport_audit.get("feature_range_clipping_contract")
        != "per_feature_training_min_max_before_standardization.v1"
        or int(transport_audit.get("conditional_n_seed_std_feature_count", -1))
        != 0
    ):
        raise JointSiteNExperimentError("Outer routed score contract changed")

    required = {
        "context_id",
        "site_type",
        "canonical_logit",
        "pre_router_canonical_logit",
        "type_router_logit",
        "within_type_relative_logit",
        "base_canonical_logit",
        "residual_canonical_logit",
    }
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise JointSiteNExperimentError(
            f"Outer routed score columns changed: {missing}"
        )
    numeric_columns = sorted(required - {"context_id", "site_type"})
    numeric = candidates[numeric_columns].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise JointSiteNExperimentError("Outer routed scores are non-finite")
    if not np.allclose(
        candidates["pre_router_canonical_logit"].to_numpy(dtype=float),
        candidates["base_canonical_logit"].to_numpy(dtype=float)
        + candidates["residual_canonical_logit"].to_numpy(dtype=float),
        rtol=0.0,
        atol=2e-6,
    ):
        raise JointSiteNExperimentError(
            "Outer pre-router ensemble score decomposition changed"
        )
    if not np.allclose(
        candidates["canonical_logit"].to_numpy(dtype=float),
        candidates["type_router_logit"].to_numpy(dtype=float)
        + candidates["within_type_relative_logit"].to_numpy(dtype=float),
        rtol=0.0,
        atol=2e-6,
    ):
        raise JointSiteNExperimentError(
            "Outer routed ensemble score decomposition changed"
        )
    within_type_maxima = candidates.groupby(
        ["context_id", "site_type"], sort=False
    )["within_type_relative_logit"].max()
    if not np.allclose(
        within_type_maxima.to_numpy(dtype=float),
        0.0,
        rtol=0.0,
        atol=2e-6,
    ):
        raise JointSiteNExperimentError("Outer within-type score origin changed")


def paired_connectivity_bootstrap(
    comparison: pd.DataFrame,
    *,
    replicates: int,
    confidence_level: float,
    seed: int,
) -> dict[str, object]:
    """Paired cluster bootstrap with contexts retained inside sampled connectivities."""

    required = {
        "context_id",
        "connectivity_id",
        "top1_improvement",
        "automatic_n_mae_reduction",
    }
    if not required <= set(comparison):
        raise JointSiteNExperimentError("Paired comparison schema changed")
    if comparison.empty or comparison["context_id"].astype(str).duplicated().any():
        raise JointSiteNExperimentError("Paired comparison contexts are invalid")
    if replicates < 1000 or not 0.0 < confidence_level < 1.0:
        raise JointSiteNExperimentError("Bootstrap settings are invalid")
    grouped = (
        comparison.groupby("connectivity_id", sort=True)
        .agg(
            context_count=("context_id", "size"),
            top1_sum=("top1_improvement", "sum"),
            automatic_n_sum=("automatic_n_mae_reduction", "sum"),
        )
        .reset_index()
    )
    if grouped.empty:
        raise JointSiteNExperimentError("No connectivity clusters to bootstrap")
    counts = grouped["context_count"].to_numpy(dtype=float)
    top1_sums = grouped["top1_sum"].to_numpy(dtype=float)
    automatic_sums = grouped["automatic_n_sum"].to_numpy(dtype=float)
    rng = np.random.default_rng(int(seed))
    top1_draws = np.empty(replicates, dtype=float)
    automatic_draws = np.empty(replicates, dtype=float)
    cluster_count = len(grouped)
    for index in range(replicates):
        draw = rng.integers(0, cluster_count, size=cluster_count)
        denominator = float(counts[draw].sum())
        top1_draws[index] = float(top1_sums[draw].sum() / denominator)
        automatic_draws[index] = float(automatic_sums[draw].sum() / denominator)
    alpha = (1.0 - confidence_level) / 2.0

    def summarize(values: np.ndarray, column: str) -> dict[str, float]:
        return {
            "estimate": float(comparison[column].mean()),
            "ci_low": float(np.quantile(values, alpha)),
            "ci_high": float(np.quantile(values, 1.0 - alpha)),
        }

    return {
        "schema_version": "nucpred.mayr-joint-site-n-paired-bootstrap.v1",
        "status": "pass",
        "unit": "connectivity_id",
        "paired_context_count": int(len(comparison)),
        "connectivity_count": int(cluster_count),
        "replicates": int(replicates),
        "confidence_level": float(confidence_level),
        "seed": int(seed),
        "interval": "two_sided_percentile_cluster_bootstrap",
        "top1_improvement": summarize(top1_draws, "top1_improvement"),
        "automatic_n_mae_reduction": summarize(
            automatic_draws, "automatic_n_mae_reduction"
        ),
    }


def build_paired_comparison(
    joint: pd.DataFrame,
    baseline: pd.DataFrame,
) -> pd.DataFrame:
    """Join new and v2 OOF rows exactly, rejecting any population drift."""

    if joint["context_id"].astype(str).duplicated().any() or baseline[
        "context_id"
    ].astype(str).duplicated().any():
        raise JointSiteNExperimentError("OOF context identity is not unique")
    joint_ids = set(joint["context_id"].astype(str))
    baseline_ids = set(baseline["context_id"].astype(str))
    if joint_ids != baseline_ids:
        raise JointSiteNExperimentError("Joint and baseline OOF populations differ")
    baseline_columns = baseline[
        [
            "context_id",
            "outer_fold",
            "connectivity_id",
            "single_target_id",
            "single_true_site_id",
            "single_true_site_type",
            "single_N_true",
            "site_top1_correct",
            "automatic_N_error",
            "oracle_site_N_error",
        ]
    ].rename(
        columns={
            "outer_fold": "baseline_outer_fold",
            "connectivity_id": "baseline_connectivity_id",
            "site_top1_correct": "baseline_site_top1_correct",
            "automatic_N_error": "baseline_automatic_n_error",
            "oracle_site_N_error": "baseline_known_site_n_error",
        }
    )
    selected = joint[
        [
            "context_id",
            "outer_fold",
            "connectivity_id",
            "true_target_id",
            "true_candidate_site_id",
            "true_site_type",
            "N_true",
            "site_top1_correct",
            "automatic_n_error",
            "known_site_n_error",
        ]
    ].rename(
        columns={
            "site_top1_correct": "joint_site_top1_correct",
            "automatic_n_error": "joint_automatic_n_error",
            "known_site_n_error": "joint_known_site_n_error",
        }
    )
    paired = selected.merge(
        baseline_columns,
        on="context_id",
        how="inner",
        validate="one_to_one",
    )
    exact_pairs = (
        (paired["outer_fold"] == paired["baseline_outer_fold"])
        & (
            paired["connectivity_id"].astype(str)
            == paired["baseline_connectivity_id"].astype(str)
        )
        & (
            paired["true_target_id"].astype(str)
            == paired["single_target_id"].astype(str)
        )
        & (
            paired["true_candidate_site_id"].astype(str)
            == paired["single_true_site_id"].astype(str)
        )
        & (
            paired["true_site_type"].astype(str)
            == paired["single_true_site_type"].astype(str)
        )
        & np.isclose(
            paired["N_true"].to_numpy(dtype=float),
            paired["single_N_true"].to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-12,
        )
    )
    if not bool(exact_pairs.all()):
        raise JointSiteNExperimentError("Paired OOF truth or split identity drifted")
    paired["top1_improvement"] = (
        paired["joint_site_top1_correct"].astype(float)
        - paired["baseline_site_top1_correct"].astype(float)
    )
    paired["baseline_automatic_n_abs_error"] = paired[
        "baseline_automatic_n_error"
    ].abs()
    paired["joint_automatic_n_abs_error"] = paired["joint_automatic_n_error"].abs()
    paired["automatic_n_mae_reduction"] = (
        paired["baseline_automatic_n_abs_error"]
        - paired["joint_automatic_n_abs_error"]
    )
    paired["baseline_known_site_n_abs_error"] = paired[
        "baseline_known_site_n_error"
    ].abs()
    paired["joint_known_site_n_abs_error"] = paired[
        "joint_known_site_n_error"
    ].abs()
    paired["known_site_n_mae_degradation"] = (
        paired["joint_known_site_n_abs_error"]
        - paired["baseline_known_site_n_abs_error"]
    )
    return paired.sort_values("context_id", kind="stable").reset_index(drop=True)


def prototype_gate_checks(
    config: Mapping[str, Any],
    *,
    joint_overall: Mapping[str, float | int],
    baseline_overall: Mapping[str, float | int],
    joint_by_type: pd.DataFrame,
    bootstrap: Mapping[str, Any],
    candidate_recall: float,
    unknown_direct_loss: float,
) -> tuple[list[dict[str, object]], bool]:
    gate = config["prototype_gate"]
    type_metrics = joint_by_type.set_index("site_type")
    required_types = {"atom_group", "delocalized_region"}
    if not required_types <= set(type_metrics.index.astype(str)):
        raise JointSiteNExperimentError("Weak-type gate slices are missing")
    known_degradation = float(joint_overall["known_site_n_mae"]) - float(
        baseline_overall["known_site_n_mae"]
    )
    raw = [
        (
            "exact_top1_minimum",
            float(joint_overall["exact_top1"]),
            ">=",
            float(gate["exact_top1_minimum"]),
        ),
        (
            "automatic_n_mae_maximum",
            float(joint_overall["automatic_n_mae"]),
            "<=",
            float(gate["automatic_n_mae_maximum"]),
        ),
        (
            "atom_group_top1_minimum",
            float(type_metrics.loc["atom_group", "exact_top1"]),
            ">=",
            float(gate["atom_group_top1_minimum"]),
        ),
        (
            "delocalized_region_top1_minimum",
            float(type_metrics.loc["delocalized_region", "exact_top1"]),
            ">=",
            float(gate["delocalized_region_top1_minimum"]),
        ),
        (
            "top1_improvement_ci_low",
            float(bootstrap["top1_improvement"]["ci_low"]),
            ">",
            float(gate["top1_delta_ci_low_must_exceed"]),
        ),
        (
            "automatic_n_mae_reduction_ci_low",
            float(bootstrap["automatic_n_mae_reduction"]["ci_low"]),
            ">",
            float(gate["automatic_n_mae_reduction_ci_low_must_exceed"]),
        ),
        (
            "known_site_n_mae_degradation_maximum",
            known_degradation,
            "<=",
            float(gate["known_site_n_mae_maximum_degradation"]),
        ),
        (
            "candidate_recall_required",
            float(candidate_recall),
            "==",
            float(gate["candidate_recall_required"]),
        ),
        (
            "unknown_direct_loss_required",
            float(unknown_direct_loss),
            "==",
            float(gate["unknown_direct_loss_required"]),
        ),
    ]
    checks: list[dict[str, object]] = []
    for name, observed, operator, threshold in raw:
        if operator == ">=":
            passed = observed >= threshold
        elif operator == "<=":
            passed = observed <= threshold
        elif operator == ">":
            passed = observed > threshold
        elif operator == "==":
            passed = math.isclose(observed, threshold, rel_tol=0.0, abs_tol=1e-12)
        else:  # pragma: no cover
            raise AssertionError(operator)
        checks.append(
            {
                "name": name,
                "observed": observed,
                "operator": operator,
                "threshold": threshold,
                "passed": bool(passed),
            }
        )
    return checks, all(bool(check["passed"]) for check in checks)


def _formal_outer_training_audit(
    config: Mapping[str, Any],
    *,
    config_path: Path,
    variant: str,
) -> dict[str, object]:
    bindings: list[dict[str, object]] = []
    maximum_unknown_loss = 0.0
    maximum_out_of_scope_loss = 0.0
    seeds = tuple(map(int, config["initialization_seeds"]))
    for outer_fold in range(int(config["outer_fold_count"])):
        for seed in seeds:
            directory = _outer_refit_directory(
                config,
                variant=variant,
                outer_fold=outer_fold,
                initialization_seed=seed,
            )
            verification = _verify_manifest(directory)
            summary_path = directory / "summary.json"
            summary = _load_json(summary_path)
            if (
                summary.get("schema_version") != OUTER_SUMMARY_SCHEMA
                or summary.get("status") != "pass"
                or summary.get("variant") != variant
                or int(summary.get("outer_fold", -1)) != outer_fold
                or int(summary.get("initialization_seed", -1)) != seed
                or summary.get("diagnostic_epoch_override") is not False
                or summary.get("eligible_for_formal_outer_scoring") is not True
                or int(summary.get("outer_test_target_rows_loaded", -1)) != 0
                or int(summary.get("outer_test_predictions_computed", -1)) != 0
            ):
                raise JointSiteNExperimentError("Outer refit is ineligible for the gate")
            _assert_current_source_hashes(
                summary.get("source_hashes"),
                config_path=config_path,
                label=f"outer-{outer_fold} seed-{seed} refit",
            )
            base_audit = summary.get("frozen_base_audit")
            transfer_audit = summary.get("transfer_audit")
            if (
                not isinstance(base_audit, Mapping)
                or base_audit.get("source")
                != "four_inner_oof_outer_selected_validity_logit"
                or base_audit.get("development_scores_crossfitted") is not True
                or base_audit.get("candidate_identity_exact") is not True
                or int(base_audit.get("outer_test_target_rows_loaded", -1)) != 0
                or base_audit.get("outer_test_predictions_computed") is not False
                or not isinstance(transfer_audit, Mapping)
                or transfer_audit.get("site_ranker") is not None
                or transfer_audit.get("site_logit_base_mode")
                != "external_split_safe_frozen_logits"
            ):
                raise JointSiteNExperimentError(
                    "Outer refit did not use the frozen cross-fitted base"
                )
            harm = summary.get("teacher_harm_audit")
            if not isinstance(harm, Mapping) or int(
                harm.get("unknown_nonzero_count", -1)
            ) != 0:
                raise JointSiteNExperimentError("Teacher N-harm touched unknown rows")
            curves_path = directory / "training_curves.csv"
            curves = pd.read_csv(curves_path)
            required = {
                "unknown_direct_loss",
                "ontology_out_of_scope_direct_loss",
            }
            if curves.empty or not required <= set(curves):
                raise JointSiteNExperimentError("Outer training loss audit is missing")
            maximum_unknown_loss = max(
                maximum_unknown_loss,
                float(curves["unknown_direct_loss"].abs().max()),
            )
            maximum_out_of_scope_loss = max(
                maximum_out_of_scope_loss,
                float(curves["ontology_out_of_scope_direct_loss"].abs().max()),
            )
            bindings.append(
                {
                    "outer_fold": outer_fold,
                    "initialization_seed": seed,
                    "summary_path": _display_path(summary_path),
                    "summary_sha256": sha256_file(summary_path),
                    "training_curves_sha256": sha256_file(curves_path),
                    **verification,
                }
            )
    if maximum_out_of_scope_loss != 0.0:
        raise JointSiteNExperimentError("Out-of-scope candidates received direct loss")
    return {
        "status": "pass",
        "formal_outer_refit_count": len(bindings),
        "outer_test_target_rows_loaded_during_training": 0,
        "maximum_unknown_direct_loss": maximum_unknown_loss,
        "maximum_ontology_out_of_scope_direct_loss": maximum_out_of_scope_loss,
        "bindings": bindings,
    }


def _formal_outer_oof(
    config: Mapping[str, Any],
    *,
    config_path: Path,
    variant: str,
) -> tuple[pd.DataFrame, list[dict[str, object]], float]:
    frames: list[pd.DataFrame] = []
    bindings: list[dict[str, object]] = []
    candidate_recalls: list[float] = []
    expected_seeds = list(map(int, config["initialization_seeds"]))
    root = _project_path(config["output_directory"], label="output directory")
    for outer_fold in range(int(config["outer_fold_count"])):
        score_directory = (
            root
            / "prototype"
            / variant
            / "outer_score_freeze"
            / f"outer-{outer_fold}"
            / "ensemble"
        )
        score_verification = _verify_manifest(score_directory)
        score_summary_path = score_directory / "summary.json"
        score_summary = _load_json(score_summary_path)
        if (
            score_summary.get("schema_version") != SCORE_SUMMARY_SCHEMA
            or score_summary.get("status") != "frozen"
            or score_summary.get("variant") != variant
            or int(score_summary.get("outer_fold", -1)) != outer_fold
            or score_summary.get("initialization_seed") is not None
            or score_summary.get("initialization_seeds") != expected_seeds
            or score_summary.get("target_or_site_labels_read") is not False
            or score_summary.get("eligible_for_formal_evaluation") is not True
        ):
            raise JointSiteNExperimentError("Outer score ensemble is ineligible")
        _assert_current_source_hashes(
            score_summary.get("source_hashes"),
            config_path=config_path,
            label=f"outer-{outer_fold} ensemble score",
        )
        for seed_binding in score_summary.get("seed_score_bindings", []):
            if not isinstance(seed_binding, Mapping):
                raise JointSiteNExperimentError("Seed score binding is malformed")
            path = _project_path(seed_binding["path"], label="seed score binding")
            if sha256_file(path) != seed_binding.get("sha256"):
                raise JointSiteNExperimentError("Seed score binding drifted")
            seed_directory = path.parent
            _verify_manifest(seed_directory)
            seed_summary = _load_json(seed_directory / "summary.json")
            _assert_current_source_hashes(
                seed_summary.get("source_hashes"),
                config_path=config_path,
                label=f"outer-{outer_fold} seed score",
            )
            base_audit = seed_summary.get("frozen_base_audit")
            if (
                seed_summary.get("eligible_for_formal_ensemble") is not True
                or seed_summary.get("canonical_logit_semantics")
                != "frozen_v2_base_plus_joint_residual"
                or not isinstance(base_audit, Mapping)
                or base_audit.get("source")
                != "label_blind_frozen_v2_canonical_logit"
                or base_audit.get("target_or_site_labels_read") is not False
                or base_audit.get("metrics_computed") is not False
                or base_audit.get("candidate_identity_exact") is not True
            ):
                raise JointSiteNExperimentError("Seed score is ineligible")

        ensemble_candidates = pd.read_parquet(
            score_directory / "candidate_scores.parquet"
        )
        _assert_outer_ensemble_score_contract(score_summary, ensemble_candidates)

        directory = (
            root
            / "prototype"
            / variant
            / "outer_evaluation"
            / f"outer-{outer_fold}"
            / "ensemble"
        )
        verification = _verify_manifest(directory)
        summary_path = directory / "summary.json"
        summary = _load_json(summary_path)
        if (
            summary.get("schema_version") != EVALUATION_SUMMARY_SCHEMA
            or summary.get("status") != "pass"
            or summary.get("variant") != variant
            or int(summary.get("outer_fold", -1)) != outer_fold
            or summary.get("initialization_seed") is not None
            or summary.get("initialization_seeds") != expected_seeds
            or summary.get("eligible_for_formal_gate") is not True
            or summary.get("outer_test_used_for_selection") is not False
            or summary.get("score_summary_sha256") != sha256_file(score_summary_path)
            or summary.get("score_frozen_before_label_read_sha256")
            != sha256_file(score_directory / "candidate_scores.parquet")
        ):
            raise JointSiteNExperimentError("Outer evaluation is ineligible for the gate")
        _assert_current_source_hashes(
            summary.get("source_hashes"),
            config_path=config_path,
            label=f"outer-{outer_fold} evaluation",
        )
        evaluation = summary.get("evaluation")
        if not isinstance(evaluation, Mapping):
            raise JointSiteNExperimentError("Outer evaluation summary is malformed")
        candidate_recalls.append(float(evaluation["candidate_recall"]))
        frame_path = directory / "context_evaluation.parquet"
        frame = pd.read_parquet(frame_path)
        if len(frame) != int(evaluation["single_target_context_count"]):
            raise JointSiteNExperimentError("Outer evaluation row count drifted")
        frame.insert(0, "outer_fold", outer_fold)
        frames.append(frame)
        bindings.append(
            {
                "outer_fold": outer_fold,
                "evaluation_summary_path": _display_path(summary_path),
                "evaluation_summary_sha256": sha256_file(summary_path),
                "score_summary_path": _display_path(score_summary_path),
                "score_summary_sha256": sha256_file(score_summary_path),
                "score_manifest_verification": score_verification,
                **verification,
            }
        )
    oof = pd.concat(frames, ignore_index=True)
    if oof["context_id"].astype(str).duplicated().any():
        raise JointSiteNExperimentError("Outer OOF contexts overlap")
    return oof, bindings, min(candidate_recalls)


def _metric_slices(frame: pd.DataFrame, column: str, name: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {name: value, **_context_metric_summary(selected)}
            for value, selected in frame.groupby(column, sort=True, dropna=False)
        ]
    )


def _baseline_metric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rename(
        columns={
            "automatic_N_error": "automatic_n_error",
            "oracle_site_N_error": "known_site_n_error",
        }
    )


def run_prototype_gate(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    variant: str = "joint_full",
    output_directory: str | Path | None = None,
) -> dict[str, object]:
    """Freeze a five-fold paired gate result; scientific failure is a valid output."""

    if variant != "joint_full":
        raise JointSiteNExperimentError("Only joint_full may request the prototype gate")
    config, resolved = read_config(config_path)
    verify_input_bindings(config, resolved)
    training_audit = _formal_outer_training_audit(
        config,
        config_path=resolved,
        variant=variant,
    )
    joint, outer_bindings, candidate_recall = _formal_outer_oof(
        config,
        config_path=resolved,
        variant=variant,
    )
    baseline_path = _project_path(
        config["baseline"]["single_target_evaluation_path"],
        label="baseline evaluation",
    )
    baseline = pd.read_parquet(baseline_path)
    paired = build_paired_comparison(joint, baseline)
    contexts = pd.read_parquet(
        _project_path(config["dataset"]["contexts_path"], label="contexts"),
        columns=["context_id", "solvent_raw"],
    )
    joint = joint.merge(contexts, on="context_id", how="left", validate="one_to_one")
    if joint["solvent_raw"].isna().any():
        raise JointSiteNExperimentError("Joint OOF lost solvent identity")

    joint_overall = _context_metric_summary(joint)
    baseline_metrics_frame = _baseline_metric_frame(baseline)
    baseline_overall = _context_metric_summary(baseline_metrics_frame)
    joint_by_type = _metric_slices(joint, "true_site_type", "site_type")
    baseline_by_type = _metric_slices(
        baseline_metrics_frame, "single_true_site_type", "site_type"
    )
    joint_by_solvent = _metric_slices(joint, "solvent_raw", "solvent_raw")
    baseline_with_solvent = baseline_metrics_frame.merge(
        contexts, on="context_id", how="left", validate="one_to_one"
    )
    baseline_by_solvent = _metric_slices(
        baseline_with_solvent, "solvent_raw", "solvent_raw"
    )
    joint_by_outer = _metric_slices(joint, "outer_fold", "outer_fold")
    baseline_by_outer = _metric_slices(
        baseline_metrics_frame, "outer_fold", "outer_fold"
    )
    bootstrap = paired_connectivity_bootstrap(
        paired,
        replicates=int(config["bootstrap"]["replicates"]),
        confidence_level=float(
            config["prototype_gate"]["paired_bootstrap_confidence_level"]
        ),
        seed=int(config["bootstrap"]["seed"]),
    )
    checks, passed = prototype_gate_checks(
        config,
        joint_overall=joint_overall,
        baseline_overall=baseline_overall,
        joint_by_type=joint_by_type,
        bootstrap=bootstrap,
        candidate_recall=candidate_recall,
        unknown_direct_loss=float(training_audit["maximum_unknown_direct_loss"]),
    )
    summary: dict[str, object] = {
        "schema_version": GATE_SUMMARY_SCHEMA,
        "status": "pass" if passed else "fail",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "experiment_id": config["experiment_id"],
        "variant": variant,
        "formal_outer_oof_eligible": True,
        "retrospective_baseline_semantics": (
            "retrospective_connectivity_grouped_evaluation"
        ),
        "joint_overall": joint_overall,
        "baseline_overall": baseline_overall,
        "paired_bootstrap": bootstrap,
        "gate_checks": checks,
        "all_hard_gates_passed": passed,
        "v3_review_authorized": passed,
        "v3_review_started": False,
        "next_phase": (
            "v3_model_blind_source_feasibility_inventory"
            if passed
            else "prototype_diagnostics_and_ablations_only"
        ),
        "candidate_recall": candidate_recall,
        "unknown_direct_loss": training_audit["maximum_unknown_direct_loss"],
        "training_audit": training_audit,
        "outer_evaluation_bindings": outer_bindings,
        "source_hashes": {
            **_implementation_source_hashes(resolved),
            "gate_runner": sha256_file(Path(__file__).resolve()),
            "baseline_evaluation": sha256_file(baseline_path),
        },
    }
    target = (
        Path(output_directory).resolve()
        if output_directory is not None
        else _project_path(config["output_directory"], label="output directory")
        / "prototype"
        / variant
        / "gate"
    )
    if target.exists():
        raise JointSiteNExperimentError(f"Refusing to overwrite prototype gate: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        joint.to_parquet(staging / "outer_oof_context_evaluation.parquet", index=False)
        paired.to_parquet(staging / "paired_context_comparison.parquet", index=False)
        pd.concat(
            [
                joint_by_type.assign(model="joint_full"),
                baseline_by_type.assign(model="v2_baseline"),
            ],
            ignore_index=True,
        ).to_csv(staging / "site_type_metrics.csv", index=False)
        pd.concat(
            [
                joint_by_solvent.assign(model="joint_full"),
                baseline_by_solvent.assign(model="v2_baseline"),
            ],
            ignore_index=True,
        ).to_csv(staging / "solvent_metrics.csv", index=False)
        pd.concat(
            [
                joint_by_outer.assign(model="joint_full"),
                baseline_by_outer.assign(model="v2_baseline"),
            ],
            ignore_index=True,
        ).to_csv(staging / "outer_fold_metrics.csv", index=False)
        atomic_write_json(staging / "paired_bootstrap.json", bootstrap, ensure_ascii=False)
        atomic_write_json(staging / "summary.json", summary, ensure_ascii=False)
        files = {
            path.name: {
                "sha256": sha256_file(path),
                "bytes": int(path.stat().st_size),
            }
            for path in sorted(staging.iterdir())
        }
        atomic_write_json(
            staging / "manifest.json",
            {
                "schema_version": "nucpred.mayr-joint-site-n-gate-manifest.v1",
                "status": "frozen",
                "files": files,
            },
            ensure_ascii=False,
        )
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    summary["output_directory"] = _display_path(target)
    summary["manifest_sha256"] = sha256_file(target / "manifest.json")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-directory")
    args = parser.parse_args(argv)
    result = run_prototype_gate(
        config_path=args.config,
        output_directory=args.output_directory,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
