"""Stage-gated joint exact-site and Mayr N prototype workflow."""

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
import tomllib
from typing import Any

import pandas as pd

from nucpred.core.files import atomic_write_json, sha256_file
from nucpred.project import get_project_layout
from nucpred.training.mayr_site_n import SITE_TYPE_NAMES


ROOT = get_project_layout().root
DEFAULT_CONFIG = ROOT / "configs/mayr_joint_site_n_v1.toml"
CONFIG_SCHEMA = "nucpred.mayr-joint-site-n-prototype-config.v1"
EXPERIMENT_ID = "mayr-joint-site-n-prototype-20260810-v1"


class JointSiteNExperimentError(RuntimeError):
    """Raised when a frozen prototype contract is violated."""


def _project_path(value: object, *, label: str) -> Path:
    path = (ROOT / str(value)).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as exc:
        raise JointSiteNExperimentError(f"{label} escapes the project root") from exc
    return path


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise JointSiteNExperimentError(f"Expected JSON object: {path}")
    return payload


def read_config(
    path: str | Path = DEFAULT_CONFIG,
) -> tuple[dict[str, Any], Path]:
    resolved = Path(path).resolve()
    config = tomllib.loads(resolved.read_text(encoding="utf-8"))
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise JointSiteNExperimentError("Unsupported joint site-N config schema")
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise JointSiteNExperimentError("Joint site-N experiment identity changed")
    if int(config.get("outer_fold_count", -1)) != 5 or int(
        config.get("inner_fold_count", -1)
    ) != 4:
        raise JointSiteNExperimentError("The 5 outer x 4 inner split changed")
    if len(tuple(config.get("initialization_seeds", ()))) != 3:
        raise JointSiteNExperimentError("Exactly three initialization seeds are required")
    if len(tuple(config.get("conditional_teacher_seeds", ()))) != 3:
        raise JointSiteNExperimentError("Exactly three conditional teacher seeds are required")
    if config.get("unknown_is_negative") is not False:
        raise JointSiteNExperimentError("Unknown candidates cannot become negatives")
    if config.get("candidate_softmax_used") is not False:
        raise JointSiteNExperimentError("Candidate softmax is forbidden")
    if config.get("outer_test_used_for_selection") is not False:
        raise JointSiteNExperimentError("Outer test cannot select the prototype")
    if config.get("v3_review_permitted_before_prototype_gate") is not False:
        raise JointSiteNExperimentError("v3 review cannot start before the prototype gate")
    evidence = config["prototype_evidence"]
    if (
        evidence.get("role") != "prototype_train_and_diagnostic_only"
        or evidence.get("may_enter_v3_formal_calibration_or_test") is not False
        or float(evidence.get("unknown_loss_weight", -1)) != 0.0
        or float(evidence.get("ontology_out_of_scope_loss_weight", -1)) != 0.0
    ):
        raise JointSiteNExperimentError("Historical evidence boundary changed")
    phases = config["phase_separation"]
    required_false = (
        "inner_training_reads_outer_test_targets",
        "outer_training_reads_outer_test_targets",
        "outer_test_used_for_model_selection",
        "outer_test_used_for_calibration",
    )
    if any(phases.get(name) is not False for name in required_false):
        raise JointSiteNExperimentError("Outer-test isolation changed")
    if phases.get("outer_test_scored_only_after_checkpoint_freeze") is not True:
        raise JointSiteNExperimentError("Outer-test score-freeze ordering changed")
    if tuple(config["model"]["site_types"]) != SITE_TYPE_NAMES:
        raise JointSiteNExperimentError("Five-type ontology order changed")
    model = config["model"]
    if (
        model.get("site_logit_design")
        != "frozen_split_safe_v2_base_plus_trainable_joint_residual"
        or model.get("inner_base_source")
        != "materialized_split_safe_publication_inner_ranker_plus_region_residual"
        or model.get("outer_development_base_source")
        != "four_inner_oof_outer_selected_validity_logit"
        or model.get("outer_test_base_source")
        != "label_blind_frozen_v2_canonical_logit"
        or model.get("site_ranker_projection")
        != "forbidden_for_outer_three_member_ensemble"
        or model.get("candidate_set_residual_contract")
        != "context_type_internal_reordering_with_frozen_type_maximum"
        or model.get("router_base_adapter")
        != "positive_per_type_affine_plus_context_residual_identity_initialized"
        or model.get("router_candidate_set_context")
        != "typed_fused_mean_max_logmeanexp_count_base_max_topk.v1"
    ):
        raise JointSiteNExperimentError("Frozen-base residual contract changed")
    if config["loss"].get("type_router_loss") != (
        "exact_endpoint_type_listwise_plus_balanced_reviewed_bce_pairwise.v1"
    ):
        raise JointSiteNExperimentError("Type-router loss contract changed")
    type_router = config["type_router"]
    if (
        type_router.get("enabled") is not True
        or type_router.get("fit_role")
        != "outer_development_inner_oof_only"
        or type_router.get("feature_contract")
        != "morgan1024_context_physical_typed_candidate_set_structural.v3"
        or tuple(type_router.get("excluded_model_derived_features", ()))
        != ("conditional_n_seed_std",)
        or type_router.get("feature_range_clipping_contract")
        != "per_feature_training_min_max_before_standardization.v1"
        or int(type_router.get("morgan_radius", -1)) != 2
        or int(type_router.get("morgan_bits", -1)) != 1024
        or type_router.get("canonical_composition")
        != "type_router_plus_weighted_pre_router_type_max_plus_within_type_relative.v1"
        or type_router.get("candidate_unknown_used_as_binary_negative") is not False
    ):
        raise JointSiteNExperimentError("Hierarchical type-router contract changed")
    try:
        regularization_grid = tuple(
            map(float, type_router["regularization_c_grid"])
        )
        prior_weight_grid = tuple(
            map(float, type_router["pre_router_type_max_weight_grid"])
        )
        atom_group_bias_grid = tuple(
            map(float, type_router["atom_group_bias_grid"])
        )
        region_bias_grid = tuple(
            map(float, type_router["delocalized_region_bias_grid"])
        )
        weak_minimum = float(type_router["weak_type_minimum"])
    except (KeyError, TypeError, ValueError) as exc:
        raise JointSiteNExperimentError(
            "Type-router selection grid is invalid"
        ) from exc
    if (
        not regularization_grid
        or any(not math.isfinite(value) or value <= 0 for value in regularization_grid)
        or not prior_weight_grid
        or any(not math.isfinite(value) or value < 0 for value in prior_weight_grid)
        or not atom_group_bias_grid
        or any(not math.isfinite(value) for value in atom_group_bias_grid)
        or not region_bias_grid
        or any(not math.isfinite(value) for value in region_bias_grid)
        or not math.isfinite(weak_minimum)
        or not 0.0 <= weak_minimum <= 1.0
    ):
        raise JointSiteNExperimentError("Type-router selection grid is invalid")
    baseline = config["baseline"]
    frozen_fold_hash_keys = (
        "outer_development_oof_sha256",
        "outer_development_summary_sha256",
        "outer_score_candidate_sha256",
        "outer_score_summary_sha256",
    )
    if any(
        len(tuple(baseline.get(key, ()))) != int(config["outer_fold_count"])
        for key in frozen_fold_hash_keys
    ):
        raise JointSiteNExperimentError("Frozen outer base-score bindings changed")
    compatibility = config.get("inner_artifact_compatibility")
    if (
        not isinstance(compatibility, Mapping)
        or compatibility.get("reuse_scope")
        != "joint_inner_and_router_independent_outer_epoch_selection_only"
        or compatibility.get("reason")
        != "router_only_feature_lineage_and_range_clipping_fix"
        or len(str(compatibility.get("legacy_config_sha256", ""))) != 64
    ):
        raise JointSiteNExperimentError("Inner artifact compatibility changed")
    bootstrap = config["bootstrap"]
    if (
        bootstrap.get("unit") != "connectivity_id"
        or int(bootstrap.get("replicates", 0)) < 1000
    ):
        raise JointSiteNExperimentError("Connectivity bootstrap contract changed")
    return config, resolved


def _verified_binding(path: Path, expected: object, *, label: str) -> dict[str, object]:
    if not path.is_file():
        raise JointSiteNExperimentError(f"Missing {label}: {path}")
    observed = sha256_file(path)
    if observed != str(expected):
        raise JointSiteNExperimentError(
            f"Frozen {label} drifted: {observed} != {expected}"
        )
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": observed,
        "bytes": int(path.stat().st_size),
    }


def verify_input_bindings(
    config: Mapping[str, Any], config_path: Path
) -> dict[str, dict[str, object]]:
    bindings: dict[str, dict[str, object]] = {
        "config": {
            "path": config_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(config_path),
            "bytes": int(config_path.stat().st_size),
        }
    }
    compatibility = config["inner_artifact_compatibility"]
    bindings["inner_artifact_compatibility.legacy_config_path"] = _verified_binding(
        _project_path(
            compatibility["legacy_config_path"],
            label="inner artifact legacy config",
        ),
        compatibility["legacy_config_sha256"],
        label="inner artifact legacy config",
    )
    pairs = {
        "dataset": (
            ("manifest_path", "manifest_sha256"),
            ("contexts_path", "contexts_sha256"),
            ("targets_path", "targets_sha256"),
            ("candidates_path", "candidates_sha256"),
            ("species_path", "species_sha256"),
            ("outer_membership_path", "outer_membership_sha256"),
            ("nested_membership_path", "nested_membership_sha256"),
        ),
        "prototype_evidence": (("path", "sha256"),),
        "baseline": (
            ("publication_protocol_path", "publication_protocol_sha256"),
            ("automatic_site_config_path", "automatic_site_config_sha256"),
            ("runtime_registry_path", "runtime_registry_sha256"),
            ("evaluation_summary_path", "evaluation_summary_sha256"),
            (
                "single_target_evaluation_path",
                "single_target_evaluation_sha256",
            ),
            ("site_type_metrics_path", "site_type_metrics_sha256"),
            ("outer_fold_metrics_path", "outer_fold_metrics_sha256"),
            ("paired_comparisons_path", "paired_comparisons_sha256"),
        ),
    }
    for section_name, section_pairs in pairs.items():
        section = config[section_name]
        for path_key, hash_key in section_pairs:
            label = f"{section_name}.{path_key}"
            bindings[label] = _verified_binding(
                _project_path(section[path_key], label=label),
                section[hash_key],
                label=label,
            )
    return bindings


def audit_nested_split(
    config: Mapping[str, Any],
    outer: pd.DataFrame,
    nested: pd.DataFrame,
) -> dict[str, object]:
    required_outer = {
        "outer_fold",
        "role",
        "target_id",
        "context_id",
        "connectivity_id",
    }
    required_nested = required_outer | {"inner_fold"}
    if not required_outer <= set(outer) or not required_nested <= set(nested):
        raise JointSiteNExperimentError("Split membership schema changed")
    outer_test = outer.loc[outer["role"].eq("test")]
    test_counts = outer_test.groupby("target_id").size()
    if test_counts.empty or not test_counts.eq(1).all():
        raise JointSiteNExperimentError("Outer tests do not partition targets")
    folds: list[dict[str, object]] = []
    for outer_fold in range(int(config["outer_fold_count"])):
        selected_outer = outer.loc[outer["outer_fold"].eq(outer_fold)]
        development = selected_outer.loc[selected_outer["role"].eq("development")]
        test = selected_outer.loc[selected_outer["role"].eq("test")]
        development_ids = set(development["connectivity_id"].astype(str))
        test_ids = set(test["connectivity_id"].astype(str))
        if not development_ids or not test_ids or development_ids & test_ids:
            raise JointSiteNExperimentError("Outer connectivity split leaks")
        for inner_fold in range(int(config["inner_fold_count"])):
            selected_inner = nested.loc[
                nested["outer_fold"].eq(outer_fold)
                & nested["inner_fold"].eq(inner_fold)
            ]
            train = set(
                selected_inner.loc[
                    selected_inner["role"].eq("train"), "connectivity_id"
                ].astype(str)
            )
            validation = set(
                selected_inner.loc[
                    selected_inner["role"].eq("validation"), "connectivity_id"
                ].astype(str)
            )
            if (
                not train
                or not validation
                or train & validation
                or train & test_ids
                or validation & test_ids
                or train | validation != development_ids
            ):
                raise JointSiteNExperimentError("Nested connectivity split leaks")
            folds.append(
                {
                    "outer_fold": outer_fold,
                    "inner_fold": inner_fold,
                    "train_connectivity_count": len(train),
                    "validation_connectivity_count": len(validation),
                    "outer_test_connectivity_count": len(test_ids),
                    "all_pairwise_connectivity_overlap": 0,
                }
            )
    return {
        "status": "pass",
        "outer_fold_count": int(config["outer_fold_count"]),
        "inner_fold_count": int(config["inner_fold_count"]),
        "each_target_exactly_one_outer_test": True,
        "all_roles_connectivity_disjoint": True,
        "folds": folds,
    }


def _metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    if frame.empty:
        raise JointSiteNExperimentError("Cannot compute an empty baseline slice")
    correct = frame["site_top1_correct"].astype(bool)
    automatic_abs = frame["automatic_N_error"].abs()
    oracle_abs = frame["oracle_site_N_error"].abs()
    wrong = ~correct
    return {
        "context_count": int(len(frame)),
        "connectivity_count": int(frame["connectivity_id"].astype(str).nunique()),
        "exact_top1": float(frame["site_top1_correct"].mean()),
        "exact_top3": float(frame["site_top3_correct"].mean()),
        "exact_top5": float(frame["site_top5_correct"].mean()),
        "mrr": float(frame["site_reciprocal_rank"].mean()),
        "automatic_n_mae": float(automatic_abs.mean()),
        "known_site_n_mae": float(oracle_abs.mean()),
        "site_addressable_n_mae_gap": float(automatic_abs.mean() - oracle_abs.mean()),
        "correct_site_count": int(correct.sum()),
        "correct_site_n_mae": float(automatic_abs.loc[correct].mean()),
        "wrong_site_count": int(wrong.sum()),
        "wrong_site_n_mae": float(automatic_abs.loc[wrong].mean()),
    }


def _slice_metrics(
    frame: pd.DataFrame, *, column: str, output_name: str
) -> pd.DataFrame:
    rows = [
        {output_name: str(value), **_metrics(selected)}
        for value, selected in frame.groupby(column, sort=True, dropna=False)
    ]
    return pd.DataFrame(rows)


def _assert_baseline_matches_summary(
    metrics: Mapping[str, float | int], summary: Mapping[str, Any]
) -> None:
    expected = {
        "context_count": summary["single_target_context_count"],
        "exact_top1": summary["primary_site_metrics"]["exact_top1_recall"],
        "exact_top3": summary["primary_site_metrics"]["exact_top3_recall"],
        "exact_top5": summary["primary_site_metrics"]["exact_top5_recall"],
        "mrr": summary["primary_site_metrics"]["mrr"],
        "automatic_n_mae": summary["primary_automatic_site_N_metrics"]["mae"],
        "known_site_n_mae": summary[
            "oracle_site_N_metrics_same_single_target_population"
        ]["mae"],
        "correct_site_count": summary["correct_site_only_N_metrics"]["count"],
        "correct_site_n_mae": summary["correct_site_only_N_metrics"]["mae"],
        "wrong_site_count": summary["wrong_site_only_N_metrics"]["count"],
        "wrong_site_n_mae": summary["wrong_site_only_N_metrics"]["mae"],
    }
    for name, value in expected.items():
        observed = metrics[name]
        if isinstance(value, int):
            matches = int(observed) == value
        else:
            matches = math.isclose(float(observed), float(value), abs_tol=1e-12)
        if not matches:
            raise JointSiteNExperimentError(
                f"Frozen baseline metric drifted: {name}={observed} != {value}"
            )


def _checkpoint_bindings(config: Mapping[str, Any]) -> list[dict[str, object]]:
    baseline = config["baseline"]
    roots = (
        (
            "nested_inner",
            _project_path(baseline["inner_checkpoint_root"], label="inner checkpoints"),
            "outer-*/inner-*/selection_checkpoint.pt",
            20,
        ),
        (
            "outer_refit",
            _project_path(baseline["outer_checkpoint_root"], label="outer checkpoints"),
            "outer-*/init-*/model.pt",
            15,
        ),
        (
            "nested_inner_site_ranker",
            _project_path(
                baseline["inner_site_ranker_root"],
                label="inner site-ranker checkpoints",
            ),
            "outer-*/inner-*/ranker_checkpoint.pt",
            20,
        ),
        (
            "outer_refit_site_ranker",
            _project_path(
                baseline["outer_site_ranker_root"],
                label="outer site-ranker checkpoints",
            ),
            "outer-*/ranker_checkpoint.pt",
            5,
        ),
    )
    result: list[dict[str, object]] = []
    for family, root, pattern, expected_count in roots:
        paths = sorted(root.glob(pattern))
        if len(paths) != expected_count:
            raise JointSiteNExperimentError(
                f"Expected {expected_count} {family} checkpoints, found {len(paths)}"
            )
        result.extend(
            {
                "family": family,
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(path),
                "bytes": int(path.stat().st_size),
            }
            for path in paths
        )
    return result


def _verify_runtime_registry(registry: Mapping[str, Any]) -> dict[str, object]:
    if registry.get("candidate_softmax_used") is not False:
        raise JointSiteNExperimentError("Frozen runtime unexpectedly uses softmax")
    if registry.get("absolute_site_probability_enabled") is not False:
        raise JointSiteNExperimentError("Frozen runtime probability boundary changed")
    if registry.get("deployment_candidate_count") != 119402:
        raise JointSiteNExperimentError("Frozen deployment candidate count changed")
    checked: list[dict[str, object]] = []
    model = registry.get("publication_model")
    if not isinstance(model, Mapping):
        raise JointSiteNExperimentError("Runtime registry lacks publication model")
    raw_bindings: list[Mapping[str, Any]] = []
    conditional = model.get("conditional_n_bindings")
    if not isinstance(conditional, list):
        raise JointSiteNExperimentError("Runtime registry lacks N bindings")
    raw_bindings.extend(value for value in conditional if isinstance(value, Mapping))
    for key in ("ranker_checkpoint", "region_membership_residual"):
        value = model.get(key)
        if isinstance(value, Mapping):
            raw_bindings.append(value)
    for binding in raw_bindings:
        path = _project_path(binding["path"], label="runtime model asset")
        checked.append(
            _verified_binding(path, binding["sha256"], label="runtime model asset")
        )
    return {
        "status": "pass",
        "absolute_site_probability_enabled": False,
        "candidate_softmax_used": False,
        "verified_model_asset_count": len(checked),
        "verified_model_assets": checked,
    }


def baseline_preflight(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    output_directory: str | Path | None = None,
) -> dict[str, object]:
    config, resolved = read_config(config_path)
    bindings = verify_input_bindings(config, resolved)
    dataset = config["dataset"]
    outer = pd.read_csv(_project_path(dataset["outer_membership_path"], label="outer"))
    nested = pd.read_csv(
        _project_path(dataset["nested_membership_path"], label="nested")
    )
    split_audit = audit_nested_split(config, outer, nested)

    baseline = config["baseline"]
    summary = _read_json(
        _project_path(baseline["evaluation_summary_path"], label="baseline summary")
    )
    evaluation = pd.read_parquet(
        _project_path(
            baseline["single_target_evaluation_path"], label="baseline evaluation"
        )
    )
    contexts = pd.read_parquet(
        _project_path(dataset["contexts_path"], label="contexts"),
        columns=["context_id", "solvent_raw"],
    )
    evaluation = evaluation.merge(
        contexts,
        on="context_id",
        how="left",
        validate="one_to_one",
    )
    if evaluation["solvent_raw"].isna().any():
        raise JointSiteNExperimentError("Baseline evaluation lost solvent identity")
    overall = _metrics(evaluation)
    _assert_baseline_matches_summary(overall, summary)
    by_type = _slice_metrics(
        evaluation, column="single_true_site_type", output_name="site_type"
    )
    by_solvent = _slice_metrics(
        evaluation, column="solvent_raw", output_name="solvent_raw"
    )
    by_outer = _slice_metrics(
        evaluation, column="outer_fold", output_name="outer_fold"
    )
    evidence = pd.read_parquet(
        _project_path(config["prototype_evidence"]["path"], label="prototype evidence")
    )
    evidence_counts = evidence["validity_label"].astype(int).value_counts().to_dict()
    if evidence_counts.get(0, 0) != int(
        config["prototype_evidence"]["required_negative_count"]
    ) or evidence_counts.get(1, 0) != int(
        config["prototype_evidence"]["required_positive_count"]
    ):
        raise JointSiteNExperimentError("Historical evidence population changed")
    if evidence["unknown_is_negative"].astype(bool).any():
        raise JointSiteNExperimentError("Historical evidence marks unknown as negative")

    registry = _read_json(
        _project_path(baseline["runtime_registry_path"], label="runtime registry")
    )
    runtime_audit = _verify_runtime_registry(registry)
    checkpoints = _checkpoint_bindings(config)
    payload: dict[str, object] = {
        "schema_version": "nucpred.mayr-joint-site-n-baseline-preflight.v1",
        "status": "pass",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "experiment_id": config["experiment_id"],
        "baseline_semantics": "retrospective_connectivity_grouped_evaluation",
        "v2_frozen_artifacts_modified": False,
        "v3_review_started": False,
        "input_bindings": bindings,
        "checkpoint_bindings": checkpoints,
        "split_audit": split_audit,
        "runtime_registry_audit": runtime_audit,
        "prototype_evidence_audit": {
            "role": config["prototype_evidence"]["role"],
            "may_enter_v3_formal_calibration_or_test": False,
            "row_count": int(len(evidence)),
            "positive_count": int(evidence_counts[1]),
            "endpoint_excluded_count": int(evidence_counts[0]),
            "site_type_counts": {
                str(name): int(value)
                for name, value in evidence.groupby("site_type").size().items()
            },
        },
        "implementation_bindings": {
            "baseline_runner": {
                "path": Path(__file__).resolve().relative_to(ROOT).as_posix(),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "candidate_policy": {
                "path": "src/nucpred/datasets/mayr_site_candidate_policy.py",
                "sha256": sha256_file(
                    ROOT / "src/nucpred/datasets/mayr_site_candidate_policy.py"
                ),
            },
            "site_n_features_and_model": {
                "path": "src/nucpred/training/mayr_site_n.py",
                "sha256": sha256_file(
                    ROOT / "src/nucpred/training/mayr_site_n.py"
                ),
            },
            "structured_ranker": {
                "path": "src/nucpred/training/mayr_site_structured_ranker.py",
                "sha256": sha256_file(
                    ROOT / "src/nucpred/training/mayr_site_structured_ranker.py"
                ),
            },
        },
        "overall": overall,
        "baseline_probability_claim": "unavailable_absolute_probability",
    }
    target = (
        Path(output_directory).resolve()
        if output_directory is not None
        else _project_path(config["output_directory"], label="output directory")
        / "baseline_v2"
    )
    if target.exists():
        raise JointSiteNExperimentError(f"Refusing to overwrite baseline: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        by_type.to_csv(staging / "site_type_metrics.csv", index=False)
        by_solvent.to_csv(staging / "solvent_metrics.csv", index=False)
        by_outer.to_csv(staging / "outer_fold_metrics.csv", index=False)
        atomic_write_json(staging / "baseline_summary.json", payload, ensure_ascii=False)
        output_bindings = {
            path.name: {
                "sha256": sha256_file(path),
                "bytes": int(path.stat().st_size),
            }
            for path in sorted(staging.iterdir())
        }
        atomic_write_json(
            staging / "manifest.json",
            {
                "schema_version": "nucpred.mayr-joint-site-n-baseline-manifest.v1",
                "status": "frozen",
                "files": output_bindings,
            },
            ensure_ascii=False,
        )
        os.replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    payload["output_directory"] = _display_path(target)
    payload["output_manifest_sha256"] = sha256_file(target / "manifest.json")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    subparsers = parser.add_subparsers(dest="action", required=True)
    baseline = subparsers.add_parser("baseline")
    baseline.add_argument("--output-directory")
    gate = subparsers.add_parser("gate")
    gate.add_argument("--output-directory")
    inner = subparsers.add_parser("inner")
    inner.add_argument("--outer-fold", type=int, required=True)
    inner.add_argument("--inner-fold", type=int, required=True)
    inner.add_argument("--initialization-seed", type=int, required=True)
    inner.add_argument(
        "--variant",
        choices=(
            "joint_full",
            "frozen_backbone",
            "without_set_pooling",
            "without_evidence_bce",
            "without_n_harm",
            "without_historical_evidence",
        ),
        default="joint_full",
    )
    inner.add_argument("--device")
    inner.add_argument("--maximum-epochs", type=int)
    inner.add_argument("--head-learning-rate", type=float)
    inner.add_argument("--output-directory")
    outer = subparsers.add_parser("outer")
    outer.add_argument(
        "outer_action",
        choices=(
            "select-epochs",
            "fit-router",
            "refit",
            "score",
            "ensemble",
            "evaluate",
        ),
    )
    outer.add_argument("--outer-fold", type=int, required=True)
    outer.add_argument("--initialization-seed", type=int)
    outer.add_argument(
        "--variant",
        choices=(
            "joint_full",
            "frozen_backbone",
            "without_set_pooling",
            "without_evidence_bce",
            "without_n_harm",
            "without_historical_evidence",
        ),
        default="joint_full",
    )
    outer.add_argument("--device")
    outer.add_argument("--maximum-epochs", type=int)
    outer.add_argument("--output-directory")
    outer.add_argument("--output-path")
    outer.add_argument("--checkpoint-directory")
    outer.add_argument("--score-directory")
    outer.add_argument("--seed-score-directory", action="append")
    outer.add_argument("--type-router-directory")
    args = parser.parse_args(argv)
    if args.action == "baseline":
        result = baseline_preflight(
            args.config,
            output_directory=args.output_directory,
        )
    elif args.action == "gate":
        from nucpred.experiments.mayr.joint_site_n_gate import run_prototype_gate

        result = run_prototype_gate(
            config_path=args.config,
            output_directory=args.output_directory,
        )
    elif args.action == "inner":
        from nucpred.experiments.mayr.joint_site_n_training import run_inner

        result = run_inner(
            outer_fold=args.outer_fold,
            inner_fold=args.inner_fold,
            initialization_seed=args.initialization_seed,
            variant=args.variant,
            config_path=args.config,
            device=args.device,
            maximum_epochs=args.maximum_epochs,
            head_learning_rate=args.head_learning_rate,
            output_directory=args.output_directory,
        )
    elif args.action == "outer":
        from nucpred.experiments.mayr import joint_site_n_outer

        common = {
            "outer_fold": args.outer_fold,
            "variant": args.variant,
            "config_path": args.config,
        }
        if args.outer_action == "select-epochs":
            result = joint_site_n_outer.select_outer_epochs(
                **common,
                output_path=args.output_path,
            )
        elif args.outer_action == "fit-router":
            result = joint_site_n_outer.fit_outer_type_router(
                **common,
                output_directory=args.output_directory,
            )
        elif args.outer_action == "ensemble":
            result = joint_site_n_outer.freeze_outer_ensemble_scores(
                **common,
                score_directories=args.seed_score_directory,
                type_router_directory=args.type_router_directory,
                output_directory=args.output_directory,
            )
        else:
            if args.initialization_seed is None and args.outer_action != "evaluate":
                parser.error("outer refit/score require --initialization-seed")
            if args.outer_action == "refit":
                result = joint_site_n_outer.run_outer_refit(
                    **common,
                    initialization_seed=args.initialization_seed,
                    device=args.device,
                    maximum_epochs=args.maximum_epochs,
                    output_directory=args.output_directory,
                )
            elif args.outer_action == "score":
                result = joint_site_n_outer.freeze_outer_scores(
                    **common,
                    initialization_seed=args.initialization_seed,
                    device=args.device,
                    checkpoint_directory=args.checkpoint_directory,
                    output_directory=args.output_directory,
                )
            elif args.outer_action == "evaluate":
                result = joint_site_n_outer.evaluate_outer_scores(
                    **common,
                    initialization_seed=args.initialization_seed,
                    score_directory=args.score_directory,
                    output_directory=args.output_directory,
                )
            else:  # pragma: no cover
                raise AssertionError(args.outer_action)
    else:  # pragma: no cover - argparse enforces the action set.
        raise AssertionError(args.action)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
