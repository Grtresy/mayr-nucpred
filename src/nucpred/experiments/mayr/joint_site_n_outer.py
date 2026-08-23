"""Outer refit, label-blind score freeze, and delayed evaluation for joint site-N."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
import gc
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import tempfile
import time
from typing import Any

import numpy as np
import pandas as pd
import torch

from nucpred.core.files import atomic_write_json, sha256_file
from nucpred.experiments.mayr.joint_site_n import (
    DEFAULT_CONFIG,
    ROOT,
    JointSiteNExperimentError,
    _display_path,
    _project_path,
    read_config,
    verify_input_bindings,
)
from nucpred.experiments.mayr.joint_site_n_training import (
    TRAINABLE_VARIANTS,
    _batch_query_ids,
    _context_metric_summary,
    _device,
    _example_batches,
    _frozen_logit_sha256,
    _joint_model_from_teacher,
    _membership_tables,
    _save_checkpoint,
    _site_config,
    _train_epoch,
    attach_frozen_base_logits,
    compute_teacher_n_harm,
    evaluate_labeled,
    new_joint_model,
    variant_settings,
)
from nucpred.experiments.mayr.site_n_formal import _tensor_mapping_sha256
from nucpred.publication.mayr_n_outer import load_outer_checkpoint
from nucpred.publication.mayr_site_publication import conditional_config
from nucpred.training.mayr_joint_site_n import (
    MayrJointSiteNModel,
    joint_optimizer_parameter_groups,
    set_heads_only_warmup,
)
from nucpred.training.mayr_joint_site_n_data import (
    balanced_router_cell_weights,
    load_joint_candidate_universe,
    load_joint_site_n_corpus,
    site_type_balanced_context_weights,
)
from nucpred.training.mayr_joint_site_type_router import (
    TYPE_ROUTER_SCHEMA_VERSION,
    TYPE_ROUTER_SELECTION_SCHEMA_VERSION,
    apply_type_router,
    build_type_router_features,
    predict_type_router_logits,
    select_and_fit_type_router,
    type_router_feature_transport_audit,
)
from nucpred.training.mayr_node_xtb_scratch import SolventVocabulary
from nucpred.training.mayr_site_n import (
    SiteNFoldPreprocessor,
    pack_site_n_batch,
)


OUTER_CHECKPOINT_SCHEMA = "nucpred.mayr-joint-site-n-outer-checkpoint.v1"
OUTER_SUMMARY_SCHEMA = "nucpred.mayr-joint-site-n-outer-refit-summary.v1"
SCORE_SUMMARY_SCHEMA = "nucpred.mayr-joint-site-n-outer-score-freeze.v1"
EVALUATION_SUMMARY_SCHEMA = "nucpred.mayr-joint-site-n-outer-evaluation.v1"
TYPE_ROUTER_OUTER_SUMMARY_SCHEMA = (
    "nucpred.mayr-joint-site-type-router-outer-fit.v1"
)

# Exact implementation release frozen by joint_full and the first four
# ablations.  The successor release only makes the router-balance table obey
# the already-declared without_historical_evidence mask; the default evidence
# path is unchanged.  Register the complete old implementation tuple so its
# immutable artifacts remain verifiable without permitting mixed source sets.
LEGACY_PRE_HISTORY_MASK_SOURCE_HASHES = {
    "runner": "3c96b6a365530f772a7c975de698ddee5d26e22f4b37704797e1b41feecefc37",
    "training_runner": (
        "13e367c7b997c475687ca1e2e98fe910cc9674164f67b25284ceda35b1368a99"
    ),
    "data_adapter": (
        "77882207efa5b8aef471972313c3573772f4f3210260b3c23a0f20f8ec291b3f"
    ),
}


def _implementation_source_hashes(config_path: Path) -> dict[str, str]:
    return {
        "config": sha256_file(config_path),
        "runner": sha256_file(Path(__file__).resolve()),
        "training_runner": sha256_file(
            ROOT / "src/nucpred/experiments/mayr/joint_site_n_training.py"
        ),
        "model": sha256_file(ROOT / "src/nucpred/training/mayr_joint_site_n.py"),
        "data_adapter": sha256_file(
            ROOT / "src/nucpred/training/mayr_joint_site_n_data.py"
        ),
        "type_router": sha256_file(
            ROOT / "src/nucpred/training/mayr_joint_site_type_router.py"
        ),
        "publication_structured_ranker": sha256_file(
            ROOT / "src/nucpred/training/mayr_site_structured_ranker.py"
        ),
        "publication_ranker_type_contract": sha256_file(
            ROOT / "src/nucpred/training/mayr_site_ranker.py"
        ),
    }


def _inner_implementation_source_hashes(config_path: Path) -> dict[str, str]:
    """Return the source contract written by the inner-training runner."""

    expected = _implementation_source_hashes(config_path)
    expected["runner"] = expected.pop("training_runner")
    # Inner optimization never imports or applies the post-hoc structural
    # type router.  Keeping that unrelated digest in the inner contract made a
    # router-only scoring fix spuriously invalidate 60 expensive GPU fits.
    expected.pop("type_router")
    return expected


def _assert_source_hashes(
    observed: object,
    *,
    expected: Mapping[str, str],
    label: str,
) -> None:
    if not isinstance(observed, Mapping):
        raise JointSiteNExperimentError(f"{label} has no source-hash binding")
    for name, digest in expected.items():
        if observed.get(name) != digest:
            raise JointSiteNExperimentError(f"{label} source drifted: {name}")


def _source_hashes_match(
    observed: object,
    *,
    expected: Mapping[str, str],
) -> bool:
    return isinstance(observed, Mapping) and all(
        observed.get(name) == digest for name, digest in expected.items()
    )


def _assert_current_source_hashes(
    observed: object,
    *,
    config_path: Path,
    label: str,
) -> None:
    current = _implementation_source_hashes(config_path)
    legacy = {**current, **LEGACY_PRE_HISTORY_MASK_SOURCE_HASHES}
    if _source_hashes_match(observed, expected=current) or _source_hashes_match(
        observed, expected=legacy
    ):
        return
    _assert_source_hashes(observed, expected=current, label=label)


def _assert_current_inner_source_hashes(
    observed: object,
    *,
    config_path: Path,
    label: str,
) -> None:
    if not isinstance(observed, Mapping):
        raise JointSiteNExperimentError(f"{label} has no source-hash binding")
    expected = _inner_implementation_source_hashes(config_path)
    config, _ = read_config(config_path)
    allowed_config_hashes = {
        expected.pop("config"),
        str(config["inner_artifact_compatibility"]["legacy_config_sha256"]),
    }
    if observed.get("config") not in allowed_config_hashes:
        raise JointSiteNExperimentError(f"{label} source drifted: config")
    legacy = {
        **expected,
        "runner": LEGACY_PRE_HISTORY_MASK_SOURCE_HASHES["training_runner"],
        "data_adapter": LEGACY_PRE_HISTORY_MASK_SOURCE_HASHES["data_adapter"],
    }
    if _source_hashes_match(observed, expected=expected) or _source_hashes_match(
        observed, expected=legacy
    ):
        return
    _assert_source_hashes(observed, expected=expected, label=label)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise JointSiteNExperimentError(f"Expected JSON object: {path}")
    return payload


def _verify_manifest(directory: Path) -> dict[str, object]:
    manifest_path = directory / "manifest.json"
    manifest = _load_json(manifest_path)
    files = manifest.get("files")
    if manifest.get("status") != "frozen" or not isinstance(files, Mapping):
        raise JointSiteNExperimentError(f"Artifact is not frozen: {directory}")
    for name, raw in files.items():
        if not isinstance(raw, Mapping):
            raise JointSiteNExperimentError("Malformed frozen file binding")
        path = directory / str(name)
        if (
            not path.is_file()
            or int(path.stat().st_size) != int(raw["bytes"])
            or sha256_file(path) != str(raw["sha256"])
        ):
            raise JointSiteNExperimentError(f"Frozen artifact drifted: {path}")
    return {
        "status": "pass",
        "manifest_sha256": sha256_file(manifest_path),
        "verified_file_count": len(files),
    }


def _inner_directory(
    config: Mapping[str, Any],
    *,
    variant: str,
    outer_fold: int,
    inner_fold: int,
    initialization_seed: int,
) -> Path:
    return (
        _project_path(config["output_directory"], label="output directory")
        / "prototype"
        / variant
        / "inner"
        / f"outer-{outer_fold}"
        / f"inner-{inner_fold}"
        / f"seed-{initialization_seed}"
    )


def select_outer_epochs(
    *,
    outer_fold: int,
    variant: str = "joint_full",
    config_path: str | Path = DEFAULT_CONFIG,
    output_path: str | Path | None = None,
) -> dict[str, object]:
    """Freeze one epoch count per seed from four inner validation folds."""

    config, resolved = read_config(config_path)
    verify_input_bindings(config, resolved)
    variant_settings(variant)
    if outer_fold not in range(int(config["outer_fold_count"])):
        raise JointSiteNExperimentError("Outer fold is outside the frozen split")
    bindings: list[dict[str, object]] = []
    selected: dict[str, int] = {}
    validation_rows: list[dict[str, object]] = []
    for seed in map(int, config["initialization_seeds"]):
        epochs: list[int] = []
        for inner_fold in range(int(config["inner_fold_count"])):
            directory = _inner_directory(
                config,
                variant=variant,
                outer_fold=outer_fold,
                inner_fold=inner_fold,
                initialization_seed=seed,
            )
            verification = _verify_manifest(directory)
            summary_path = directory / "summary.json"
            summary = _load_json(summary_path)
            frozen_base = summary.get("frozen_base_audit")
            if (
                summary.get("schema_version")
                != "nucpred.mayr-joint-site-n-inner-summary.v1"
                or summary.get("status") != "pass"
                or summary.get("variant") != variant
                or int(summary.get("outer_fold", -1)) != outer_fold
                or int(summary.get("inner_fold", -1)) != inner_fold
                or int(summary.get("initialization_seed", -1)) != seed
                or summary.get("eligible_for_formal_inner_selection") is not True
                or summary.get("diagnostic_epoch_override") is not False
                or int(summary.get("outer_test_target_rows_loaded", -1)) != 0
                or not isinstance(frozen_base, Mapping)
                or frozen_base.get("source")
                != "split_safe_publication_inner_ranker_plus_region_residual"
                or frozen_base.get("candidate_identity_exact") is not True
                or int(frozen_base.get("outer_test_target_rows_loaded", -1)) != 0
            ):
                raise JointSiteNExperimentError(
                    f"Inner result is ineligible for epoch selection: {directory}"
                )
            _assert_current_inner_source_hashes(
                summary.get("source_hashes"),
                config_path=resolved,
                label=f"outer-{outer_fold} inner-{inner_fold} seed-{seed}",
            )
            epoch = int(summary["best_epoch"])
            epochs.append(epoch)
            metrics = summary["best_validation"]["overall"]
            validation_rows.append(
                {
                    "outer_fold": outer_fold,
                    "inner_fold": inner_fold,
                    "initialization_seed": seed,
                    "best_epoch": epoch,
                    "exact_top1": float(metrics["exact_top1"]),
                    "automatic_n_mae": float(metrics["automatic_n_mae"]),
                }
            )
            bindings.append(
                {
                    "inner_fold": inner_fold,
                    "initialization_seed": seed,
                    "summary_path": summary_path.relative_to(ROOT).as_posix(),
                    "summary_sha256": sha256_file(summary_path),
                    **verification,
                }
            )
        selected[str(seed)] = int(statistics.median_high(epochs))
    payload: dict[str, object] = {
        "schema_version": "nucpred.mayr-joint-site-n-outer-epoch-selection.v1",
        "status": "frozen",
        "experiment_id": config["experiment_id"],
        "variant": variant,
        "outer_fold": outer_fold,
        "rule": "per_seed_upper_median_of_four_inner_best_epochs",
        "selected_epochs_by_initialization_seed": selected,
        "inner_run_bindings": bindings,
        "outer_test_targets_or_metrics_read": False,
        "config_sha256": sha256_file(resolved),
    }
    target = (
        Path(output_path).resolve()
        if output_path is not None
        else _project_path(config["output_directory"], label="output directory")
        / "prototype"
        / variant
        / "outer_epoch_selection"
        / f"outer-{outer_fold}.json"
    )
    if target.exists():
        raise JointSiteNExperimentError(f"Refusing to overwrite epoch selection: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(validation_rows).to_csv(
        target.with_name(f"{target.stem}_inner_metrics.csv"), index=False
    )
    atomic_write_json(target, payload, ensure_ascii=False)
    payload["output_path"] = _display_path(target)
    payload["output_sha256"] = sha256_file(target)
    return payload


def _ensemble_inner_oof_scores(
    frames: Sequence[pd.DataFrame],
    *,
    initialization_seeds: Sequence[int],
) -> pd.DataFrame:
    if len(frames) != len(initialization_seeds) or len(frames) < 2:
        raise JointSiteNExperimentError(
            "Inner OOF ensemble requires one frame per initialization seed"
        )
    identity_columns = [
        "query_id",
        "context_id",
        "species_id",
        "connectivity_id",
        "candidate_site_id",
        "site_type",
    ]
    value_columns = [
        "canonical_logit",
        "base_canonical_logit",
        "residual_canonical_logit",
        "conditional_n_prediction",
    ]
    ordered = [
        frame.sort_values("query_id", kind="stable").reset_index(drop=True)
        for frame in frames
    ]
    _ = [
        _require_score_columns(frame, identity_columns + value_columns)
        for frame in ordered
    ]
    reference = ordered[0][identity_columns]
    for frame in ordered[1:]:
        if not reference.equals(frame[identity_columns]):
            raise JointSiteNExperimentError(
                "Inner OOF candidate identity changed across seeds"
            )
    values = {
        column: np.stack(
            [frame[column].to_numpy(dtype=float) for frame in ordered], axis=1
        )
        for column in value_columns
    }
    if not np.allclose(
        values["base_canonical_logit"],
        values["base_canonical_logit"][:, :1],
        rtol=0.0,
        atol=1e-7,
    ):
        raise JointSiteNExperimentError(
            "Frozen inner OOF base logits differ across seeds"
        )
    if not all(np.isfinite(value).all() for value in values.values()):
        raise JointSiteNExperimentError("Inner OOF scores are non-finite")
    result = reference.copy()
    result["base_canonical_logit"] = values["base_canonical_logit"][:, 0]
    result["residual_canonical_logit"] = values[
        "residual_canonical_logit"
    ].mean(axis=1)
    result["residual_canonical_logit_seed_std"] = values[
        "residual_canonical_logit"
    ].std(axis=1, ddof=0)
    result["canonical_logit"] = values["canonical_logit"].mean(axis=1)
    result["canonical_logit_seed_std"] = values["canonical_logit"].std(
        axis=1, ddof=0
    )
    result["conditional_n_prediction"] = values[
        "conditional_n_prediction"
    ].mean(axis=1)
    result["conditional_n_seed_std"] = values["conditional_n_prediction"].std(
        axis=1, ddof=0
    )
    result["ensemble_member_count"] = len(frames)
    return result


def _require_score_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise JointSiteNExperimentError(
            f"Candidate score frame lacks columns: {missing}"
        )


def _type_router_directory(
    config: Mapping[str, Any],
    *,
    variant: str,
    outer_fold: int,
) -> Path:
    return (
        _project_path(config["output_directory"], label="output directory")
        / "prototype"
        / variant
        / "type_router"
        / f"outer-{outer_fold}"
    )


def fit_outer_type_router(
    *,
    outer_fold: int,
    variant: str = "joint_full",
    config_path: str | Path = DEFAULT_CONFIG,
    output_directory: str | Path | None = None,
) -> dict[str, object]:
    """Fit one outer-fold router from four inner OOF folds and no outer test."""

    started = time.perf_counter()
    config, resolved = read_config(config_path)
    verify_input_bindings(config, resolved)
    variant_settings(variant)
    if outer_fold not in range(int(config["outer_fold_count"])):
        raise JointSiteNExperimentError("Outer fold is outside the frozen split")
    development_ids, split_audit = outer_development_target_ids(
        config, outer_fold=outer_fold
    )
    corpus = load_joint_site_n_corpus(
        _project_path(config["dataset"]["directory"], label="dataset"),
        evidence_path=_project_path(
            config["prototype_evidence"]["path"], label="prototype evidence"
        ),
        target_ids=sorted(development_ids),
    )
    development_contexts = set(corpus.targets["context_id"].astype(str))
    seeds = tuple(map(int, config["initialization_seeds"]))
    inner_frames: list[pd.DataFrame] = []
    inner_bindings: list[dict[str, object]] = []
    fold_by_context: dict[str, int] = {}
    for inner_fold in range(int(config["inner_fold_count"])):
        seed_frames: list[pd.DataFrame] = []
        fold_contexts: set[str] | None = None
        for seed in seeds:
            directory = _inner_directory(
                config,
                variant=variant,
                outer_fold=outer_fold,
                inner_fold=inner_fold,
                initialization_seed=seed,
            )
            verification = _verify_manifest(directory)
            summary_path = directory / "summary.json"
            summary = _load_json(summary_path)
            if (
                summary.get("schema_version")
                != "nucpred.mayr-joint-site-n-inner-summary.v1"
                or summary.get("status") != "pass"
                or summary.get("variant") != variant
                or int(summary.get("outer_fold", -1)) != outer_fold
                or int(summary.get("inner_fold", -1)) != inner_fold
                or int(summary.get("initialization_seed", -1)) != seed
                or summary.get("eligible_for_formal_inner_selection") is not True
                or summary.get("diagnostic_epoch_override") is not False
                or int(summary.get("outer_test_target_rows_loaded", -1)) != 0
            ):
                raise JointSiteNExperimentError(
                    f"Inner result is ineligible for type routing: {directory}"
                )
            _assert_current_inner_source_hashes(
                summary.get("source_hashes"),
                config_path=resolved,
                label=f"outer-{outer_fold} inner-{inner_fold} seed-{seed}",
            )
            score_path = directory / "validation_candidate_scores.parquet"
            frame = pd.read_parquet(score_path)
            observed_contexts = set(frame["context_id"].astype(str))
            if fold_contexts is None:
                fold_contexts = observed_contexts
            elif fold_contexts != observed_contexts:
                raise JointSiteNExperimentError(
                    "Inner validation contexts changed across seeds"
                )
            seed_frames.append(frame)
            inner_bindings.append(
                {
                    "inner_fold": inner_fold,
                    "initialization_seed": seed,
                    "score_path": _display_path(score_path),
                    "score_sha256": sha256_file(score_path),
                    "summary_path": _display_path(summary_path),
                    "summary_sha256": sha256_file(summary_path),
                    **verification,
                }
            )
        if not fold_contexts:
            raise JointSiteNExperimentError("Inner OOF fold has no contexts")
        overlap = set(fold_by_context) & fold_contexts
        if overlap:
            raise JointSiteNExperimentError(
                "Inner OOF contexts occur in more than one validation fold"
            )
        fold_by_context.update(
            {context_id: inner_fold for context_id in fold_contexts}
        )
        ensemble = _ensemble_inner_oof_scores(
            seed_frames, initialization_seeds=seeds
        )
        ensemble["inner_fold"] = inner_fold
        inner_frames.append(ensemble)
    if set(fold_by_context) != development_contexts:
        raise JointSiteNExperimentError(
            "Inner OOF router coverage differs from outer development"
        )
    development_scores = pd.concat(inner_frames, ignore_index=True)
    if development_scores["query_id"].duplicated().any():
        raise JointSiteNExperimentError("Inner OOF candidate identity is duplicated")
    features = build_type_router_features(
        development_scores,
        queries=corpus.queries,
        contexts=corpus.contexts,
    )
    router_config = config["type_router"]
    bundle, selection, crossfit_contexts = select_and_fit_type_router(
        features,
        development_scores,
        corpus.targets,
        inner_fold_by_context=fold_by_context,
        regularization_grid=tuple(
            map(float, router_config["regularization_c_grid"])
        ),
        pre_router_type_max_weight_grid=tuple(
            map(float, router_config["pre_router_type_max_weight_grid"])
        ),
        atom_group_bias_grid=tuple(
            map(float, router_config["atom_group_bias_grid"])
        ),
        region_bias_grid=tuple(
            map(float, router_config["delocalized_region_bias_grid"])
        ),
        weak_type_minimum=float(router_config["weak_type_minimum"]),
    )
    if selection.get("schema_version") != TYPE_ROUTER_SELECTION_SCHEMA_VERSION:
        raise JointSiteNExperimentError("Type-router selection schema changed")
    target = (
        Path(output_directory).resolve()
        if output_directory is not None
        else _type_router_directory(
            config, variant=variant, outer_fold=outer_fold
        )
    )
    if target.exists():
        raise JointSiteNExperimentError(
            f"Refusing to overwrite type-router fit: {target}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        atomic_write_json(staging / "router_bundle.json", bundle, ensure_ascii=False)
        features.to_parquet(staging / "development_features.parquet", index=False)
        development_scores.to_parquet(
            staging / "development_oof_candidate_scores.parquet", index=False
        )
        crossfit_contexts.to_parquet(
            staging / "crossfit_context_predictions.parquet", index=False
        )
        summary: dict[str, object] = {
            "schema_version": TYPE_ROUTER_OUTER_SUMMARY_SCHEMA,
            "status": "frozen",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "experiment_id": config["experiment_id"],
            "variant": variant,
            "outer_fold": outer_fold,
            "fit_role": "outer_development_inner_oof_only",
            "router_bundle_schema_version": TYPE_ROUTER_SCHEMA_VERSION,
            "router_bundle_sha256": bundle["bundle_sha256"],
            "selection": selection,
            "inner_run_bindings": inner_bindings,
            "split_audit": split_audit,
            "corpus_audit": dict(corpus.audit),
            "feature_context_count": int(len(features)),
            "development_candidate_count": int(len(development_scores)),
            "outer_test_target_rows_loaded": 0,
            "outer_test_predictions_computed": 0,
            "candidate_unknown_used_as_binary_negative": False,
            "config_sha256": sha256_file(resolved),
            "source_hashes": _implementation_source_hashes(resolved),
            "eligible_for_formal_outer_scoring": True,
            "elapsed_seconds": time.perf_counter() - started,
        }
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
                "schema_version": "nucpred.mayr-joint-site-type-router-manifest.v1",
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


def outer_development_target_ids(
    config: Mapping[str, Any], *, outer_fold: int
) -> tuple[set[str], dict[str, object]]:
    outer, _ = _membership_tables(config)
    selected = outer.loc[outer["outer_fold"].eq(outer_fold)]
    development = selected.loc[selected["role"].eq("development")]
    test = selected.loc[selected["role"].eq("test")]
    development_ids = set(development["target_id"].astype(str))
    test_ids = set(test["target_id"].astype(str))
    development_connectivity = set(development["connectivity_id"].astype(str))
    test_connectivity = set(test["connectivity_id"].astype(str))
    if (
        not development_ids
        or not test_ids
        or development_ids & test_ids
        or development_connectivity & test_connectivity
    ):
        raise JointSiteNExperimentError("Outer development/test split leaks")
    return development_ids, {
        "schema_version": "nucpred.mayr-joint-site-n-outer-split-audit.v1",
        "outer_fold": outer_fold,
        "development_target_count": len(development_ids),
        "development_connectivity_count": len(development_connectivity),
        "outer_test_membership_target_count": len(test_ids),
        "outer_test_membership_connectivity_count": len(test_connectivity),
        "connectivity_overlap": 0,
        "outer_test_target_rows_loaded": 0,
    }


def _frozen_fold_hash(
    config: Mapping[str, Any], *, key: str, outer_fold: int
) -> str:
    values = tuple(map(str, config["baseline"].get(key, ())))
    if len(values) != int(config["outer_fold_count"]):
        raise JointSiteNExperimentError(f"Frozen fold hash list changed: {key}")
    return values[outer_fold]


def _read_exact_query_logits(
    path: Path,
    *,
    value_column: str,
    expected_query_ids: Sequence[str],
) -> dict[str, float]:
    frame = pd.read_parquet(path, columns=["query_id", value_column])
    query_ids = frame["query_id"].astype(str)
    expected = set(map(str, expected_query_ids))
    observed = set(query_ids)
    values = frame[value_column].to_numpy(dtype=float)
    if (
        query_ids.duplicated().any()
        or observed != expected
        or len(frame) != len(expected)
        or not np.isfinite(values).all()
    ):
        raise JointSiteNExperimentError(
            "Frozen base-score candidate identity or values changed"
        )
    return dict(zip(query_ids, map(float, values), strict=True))


def _router_context_type_maximum_delta(path: Path) -> float:
    frame = pd.read_parquet(
        path,
        columns=["context_id", "site_type", "router_selected_logit"],
    )
    spread = frame.groupby(["context_id", "site_type"], sort=False)[
        "router_selected_logit"
    ].agg(lambda values: float(values.max() - values.min()))
    maximum = float(spread.max()) if len(spread) else 0.0
    if not math.isfinite(maximum) or maximum > 5e-5:
        raise JointSiteNExperimentError(
            "Frozen router logits vary within a context/type group"
        )
    return maximum


def load_outer_development_base_logits(
    config: Mapping[str, Any],
    *,
    outer_fold: int,
    expected_query_ids: Sequence[str],
) -> tuple[dict[str, float], dict[str, float], dict[str, object]]:
    """Read only four-inner-OOF development logits for outer residual training."""

    directory = (
        _project_path(
            config["baseline"]["outer_development_oof_root"],
            label="outer development OOF root",
        )
        / f"outer-{outer_fold}"
    )
    summary_path = directory / "summary.json"
    score_path = directory / "development_oof_predictions.parquet"
    expected_summary_hash = _frozen_fold_hash(
        config,
        key="outer_development_summary_sha256",
        outer_fold=outer_fold,
    )
    expected_score_hash = _frozen_fold_hash(
        config,
        key="outer_development_oof_sha256",
        outer_fold=outer_fold,
    )
    if (
        sha256_file(summary_path) != expected_summary_hash
        or sha256_file(score_path) != expected_score_hash
    ):
        raise JointSiteNExperimentError("Frozen outer-development OOF artifact drifted")
    summary = _load_json(summary_path)
    calibration = summary.get("calibration_audit")
    if (
        summary.get("schema_version")
        != "nucpred.mayr-n-publication-site-outer-refit.v1"
        or summary.get("status") != "pass"
        or int(summary.get("outer_fold", -1)) != outer_fold
        or summary.get("candidate_softmax_used") is not False
        or int(summary.get("unknown_as_negative_count", -1)) != 0
        or int(summary.get("outer_test_target_rows_loaded", -1)) != 0
        or summary.get("outer_test_predictions_computed") is not False
        or not isinstance(calibration, Mapping)
        or calibration.get("calibration_uses_outer_test") is not False
        or int(calibration.get("outer_test_target_rows_loaded", -1)) != 0
    ):
        raise JointSiteNExperimentError(
            "Outer-development OOF split-safety contract changed"
        )
    values = _read_exact_query_logits(
        score_path,
        value_column="outer_selected_validity_logit",
        expected_query_ids=expected_query_ids,
    )
    router_values = _read_exact_query_logits(
        score_path,
        value_column="router_selected_logit",
        expected_query_ids=expected_query_ids,
    )
    router_delta = _router_context_type_maximum_delta(score_path)
    if int(summary.get("development_full_candidate_count", -1)) != len(values):
        raise JointSiteNExperimentError("Outer-development OOF row count changed")
    return values, router_values, {
        "schema_version": "nucpred.mayr-joint-site-n-outer-development-base.v1",
        "status": "pass",
        "source": "four_inner_oof_outer_selected_validity_logit",
        "outer_fold": outer_fold,
        "candidate_count": len(values),
        "candidate_identity_exact": True,
        "development_scores_crossfitted": True,
        "candidate_softmax_used": False,
        "unknown_as_negative_count": 0,
        "columns_read": [
            "query_id",
            "outer_selected_validity_logit",
            "router_selected_logit",
        ],
        "score_path": _display_path(score_path),
        "score_sha256": expected_score_hash,
        "summary_path": _display_path(summary_path),
        "summary_sha256": expected_summary_hash,
        "mapping_sha256": _frozen_logit_sha256(values),
        "router_mapping_sha256": _frozen_logit_sha256(router_values),
        "router_context_type_maximum_delta": router_delta,
        "outer_test_target_rows_loaded": 0,
        "outer_test_predictions_computed": False,
    }


def load_outer_test_base_logits(
    config: Mapping[str, Any],
    *,
    outer_fold: int,
    expected_query_ids: Sequence[str],
) -> tuple[dict[str, float], dict[str, float], dict[str, object]]:
    """Read the frozen v2 outer-test logits without opening labels or metrics."""

    directory = (
        _project_path(
            config["baseline"]["outer_score_freeze_root"],
            label="outer score-freeze root",
        )
        / f"outer-{outer_fold}"
    )
    summary_path = directory / "summary.json"
    score_path = directory / "candidate_scores.parquet"
    expected_summary_hash = _frozen_fold_hash(
        config,
        key="outer_score_summary_sha256",
        outer_fold=outer_fold,
    )
    expected_score_hash = _frozen_fold_hash(
        config,
        key="outer_score_candidate_sha256",
        outer_fold=outer_fold,
    )
    if (
        sha256_file(summary_path) != expected_summary_hash
        or sha256_file(score_path) != expected_score_hash
    ):
        raise JointSiteNExperimentError("Frozen v2 outer score package drifted")
    summary = _load_json(summary_path)
    if (
        summary.get("schema_version")
        != "nucpred.mayr-n-publication-automatic-site-score-freeze.v1"
        or summary.get("status") != "frozen"
        or int(summary.get("outer_fold", -1)) != outer_fold
        or summary.get("candidate_softmax_used") is not False
        or summary.get("target_table_opened") is not False
        or summary.get("target_id_column_requested") is not False
        or summary.get("site_labels_read_before_score_freeze") is not False
        or summary.get("N_labels_read_before_score_freeze") is not False
        or summary.get("metrics_computed_before_score_freeze") is not False
        or int(summary.get("outer_refit_outer_test_rows_loaded", -1)) != 0
        or summary.get("candidate_score_sha256") != expected_score_hash
    ):
        raise JointSiteNExperimentError("Frozen v2 label-blind score contract changed")
    values = _read_exact_query_logits(
        score_path,
        value_column="canonical_logit",
        expected_query_ids=expected_query_ids,
    )
    router_values = _read_exact_query_logits(
        score_path,
        value_column="router_selected_logit",
        expected_query_ids=expected_query_ids,
    )
    router_delta = _router_context_type_maximum_delta(score_path)
    if int(summary.get("candidate_score_count", -1)) != len(values):
        raise JointSiteNExperimentError("Frozen v2 outer score row count changed")
    return values, router_values, {
        "schema_version": "nucpred.mayr-joint-site-n-outer-test-base.v1",
        "status": "pass",
        "source": "label_blind_frozen_v2_canonical_logit",
        "outer_fold": outer_fold,
        "candidate_count": len(values),
        "candidate_identity_exact": True,
        "candidate_softmax_used": False,
        "columns_read": ["query_id", "canonical_logit", "router_selected_logit"],
        "score_path": _display_path(score_path),
        "score_sha256": expected_score_hash,
        "summary_path": _display_path(summary_path),
        "summary_sha256": expected_summary_hash,
        "mapping_sha256": _frozen_logit_sha256(values),
        "router_mapping_sha256": _frozen_logit_sha256(router_values),
        "router_context_type_maximum_delta": router_delta,
        "target_table_opened": False,
        "target_or_site_labels_read": False,
        "metrics_computed": False,
    }


def _epoch_selection(
    config: Mapping[str, Any], *, variant: str, outer_fold: int
) -> tuple[dict[str, Any], Path]:
    path = (
        _project_path(config["output_directory"], label="output directory")
        / "prototype"
        / variant
        / "outer_epoch_selection"
        / f"outer-{outer_fold}.json"
    )
    payload = _load_json(path)
    if (
        payload.get("status") != "frozen"
        or payload.get("variant") != variant
        or int(payload.get("outer_fold", -1)) != outer_fold
        or payload.get("outer_test_targets_or_metrics_read") is not False
    ):
        raise JointSiteNExperimentError("Outer epoch selection is invalid")
    return payload, path


def _outer_teacher(
    config: Mapping[str, Any],
    *,
    outer_fold: int,
    initialization_seed: int,
    device: torch.device,
) -> tuple[torch.nn.Module, SiteNFoldPreprocessor, SolventVocabulary, dict[str, Any], Path]:
    seeds = tuple(map(int, config["initialization_seeds"]))
    teacher_seeds = tuple(map(int, config["conditional_teacher_seeds"]))
    teacher_seed = teacher_seeds[seeds.index(initialization_seed)]
    site_config, _ = _site_config(config)
    _, conditional_path = conditional_config(site_config)
    path = (
        _project_path(config["baseline"]["outer_checkpoint_root"], label="outer root")
        / f"outer-{outer_fold}"
        / f"init-{teacher_seed}"
        / "model.pt"
    )
    model, preprocessor, vocabulary, payload = load_outer_checkpoint(
        path,
        config_path=conditional_path,
        device=device,
    )
    return model, preprocessor, vocabulary, payload, path


def _outer_epoch_zero_curve(
    evaluation: Mapping[str, object],
) -> dict[str, object]:
    """Record the transferred checkpoint as a real, no-update training state."""

    overall = evaluation.get("overall")
    diagnostics = evaluation.get("score_diagnostics")
    if not isinstance(overall, Mapping) or not isinstance(diagnostics, Mapping):
        raise JointSiteNExperimentError("Initial outer-development evaluation is invalid")
    return {
        "epoch": 0,
        "phase": "transferred_initialization",
        "optimizer_step_count": 0,
        "unknown_direct_loss": 0.0,
        "ontology_out_of_scope_direct_loss": 0.0,
        **{f"development_{key}": value for key, value in overall.items()},
        **{f"development_{key}": value for key, value in diagnostics.items()},
    }


def run_outer_refit(
    *,
    outer_fold: int,
    initialization_seed: int,
    variant: str = "joint_full",
    config_path: str | Path = DEFAULT_CONFIG,
    device: str | None = None,
    maximum_epochs: int | None = None,
    output_directory: str | Path | None = None,
) -> dict[str, object]:
    """Refit on outer development and freeze before any outer-test label read."""

    started = time.perf_counter()
    config, resolved = read_config(config_path)
    verify_input_bindings(config, resolved)
    if outer_fold not in range(int(config["outer_fold_count"])):
        raise JointSiteNExperimentError("Outer fold is outside the frozen split")
    seeds = tuple(map(int, config["initialization_seeds"]))
    if initialization_seed not in seeds:
        raise JointSiteNExperimentError("Initialization seed is not registered")
    settings = variant_settings(variant)
    selected_device = _device(device or str(config["device"]))
    diagnostic_override = maximum_epochs is not None
    if diagnostic_override:
        epochs = int(maximum_epochs)
        if epochs < 0 or epochs > int(config["training"]["maximum_epochs"]):
            raise JointSiteNExperimentError("Diagnostic outer epoch override is invalid")
        selection_path = None
    else:
        selection, selection_path = _epoch_selection(
            config, variant=variant, outer_fold=outer_fold
        )
        epochs = int(
            selection["selected_epochs_by_initialization_seed"][str(initialization_seed)]
        )
    development_ids, split_audit = outer_development_target_ids(
        config, outer_fold=outer_fold
    )
    corpus = load_joint_site_n_corpus(
        _project_path(config["dataset"]["directory"], label="dataset"),
        evidence_path=_project_path(
            config["prototype_evidence"]["path"], label="prototype evidence"
        ),
        target_ids=sorted(development_ids),
    )
    (
        development_base_logits,
        development_base_router_logits,
        frozen_base_audit,
    ) = (
        load_outer_development_base_logits(
            config,
            outer_fold=outer_fold,
            expected_query_ids=tuple(corpus.queries["query_id"].astype(str)),
        )
    )
    corpus = attach_frozen_base_logits(
        corpus,
        development_base_logits,
        development_base_router_logits,
    )
    development_contexts = corpus.context_ids_for_target_ids(sorted(development_ids))
    examples = corpus.examples(development_contexts)
    site_context_weights, site_type_balance_audit = (
        site_type_balanced_context_weights(
            corpus.queries,
            context_ids=development_contexts,
            exponent=float(
                config["training"]["site_type_context_balance_exponent"]
            ),
        )
    )
    router_cell_weights, router_balance_audit = balanced_router_cell_weights(
        corpus.queries,
        context_ids=development_contexts,
        use_historical_evidence=settings.use_historical_evidence,
    )
    teacher, preprocessor, vocabulary, teacher_payload, teacher_path = _outer_teacher(
        config,
        outer_fold=outer_fold,
        initialization_seed=initialization_seed,
        device=selected_device,
    )
    model, transfer_audit = _joint_model_from_teacher(
        teacher,
        config=config,
        vocabulary=vocabulary,
        preprocessor=preprocessor,
        ranker_checkpoint=None,
        initialization_seed=initialization_seed,
        settings=settings,
        device=selected_device,
    )
    del teacher
    initial_development_evaluation, _, _, _ = evaluate_labeled(
        model,
        examples,
        corpus=corpus,
        preprocessor=preprocessor,
        vocabulary=vocabulary,
        batch_size=int(config["training"]["batch_size_contexts"]),
        device=selected_device,
    )
    harm, harm_audit = compute_teacher_n_harm(
        model,
        examples,
        corpus=corpus,
        preprocessor=preprocessor,
        vocabulary=vocabulary,
        batch_size=int(config["training"]["batch_size_contexts"]),
        device=selected_device,
        cap_quantile=float(config["loss"]["n_harm_cap_quantile"]),
        use_historical_evidence=(
            settings.use_historical_evidence and settings.use_n_harm
        ),
    )
    optimizer = torch.optim.AdamW(
        joint_optimizer_parameter_groups(
            model,
            head_learning_rate=float(config["training"]["head_learning_rate"]),
            backbone_multiplier=float(
                config["training"]["backbone_learning_rate_multiplier"]
            ),
        ),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    curves: list[dict[str, object]] = [
        _outer_epoch_zero_curve(initial_development_evaluation)
    ]
    warmup = int(config["training"]["heads_only_warmup_epochs"])
    for epoch in range(1, epochs + 1):
        heads_only = epoch <= warmup or not settings.train_backbone_after_warmup
        set_heads_only_warmup(model, enabled=heads_only)
        metrics = _train_epoch(
            model,
            examples,
            corpus=corpus,
            preprocessor=preprocessor,
            vocabulary=vocabulary,
            candidate_n_harm=harm,
            router_cell_weights=router_cell_weights,
            site_context_weights=site_context_weights,
            settings=settings,
            config=config,
            optimizer=optimizer,
            device=selected_device,
            shuffle_seed=initialization_seed + epoch,
        )
        curves.append(
            {
                "epoch": epoch,
                "phase": "heads_only" if heads_only else "joint_finetune",
                **metrics,
            }
        )
    set_heads_only_warmup(model, enabled=False)
    target = (
        Path(output_directory).resolve()
        if output_directory is not None
        else _project_path(config["output_directory"], label="output directory")
        / "prototype"
        / variant
        / "outer_refit"
        / f"outer-{outer_fold}"
        / f"seed-{initialization_seed}"
    )
    if diagnostic_override and output_directory is None:
        target = target.with_name(f"seed-{initialization_seed}-diagnostic-{epochs}ep")
    if target.exists():
        raise JointSiteNExperimentError(f"Refusing to overwrite outer refit: {target}")
    source_hashes = {
        **_implementation_source_hashes(resolved),
        "dataset_manifest": str(config["dataset"]["manifest_sha256"]),
        "outer_membership": str(config["dataset"]["outer_membership_sha256"]),
        "prototype_evidence": str(config["prototype_evidence"]["sha256"]),
        "conditional_teacher_checkpoint": sha256_file(teacher_path),
        "frozen_development_base_scores": str(frozen_base_audit["score_sha256"]),
        "frozen_development_base_summary": str(frozen_base_audit["summary_sha256"]),
    }
    if selection_path is not None:
        source_hashes["epoch_selection"] = sha256_file(selection_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        saved = _save_checkpoint(
            staging / "model.pt",
            model=model,
            preprocessor=preprocessor,
            vocabulary=vocabulary,
            schema_version=OUTER_CHECKPOINT_SCHEMA,
            payload={
                "phase": "outer_development_refit_frozen_before_test_scoring",
                "experiment_id": config["experiment_id"],
                "variant": variant,
                "variant_settings": asdict(settings),
                "outer_fold": outer_fold,
                "initialization_seed": initialization_seed,
                "trained_epochs": epochs,
                "diagnostic_epoch_override": diagnostic_override,
                "eligible_for_formal_outer_scoring": not diagnostic_override,
                "split_audit": split_audit,
                "source_hashes": source_hashes,
                "teacher_model_state_sha256": teacher_payload["model_state_sha256"],
                "transfer_audit": transfer_audit,
                "frozen_base_audit": frozen_base_audit,
                "initial_development_diagnostic": initial_development_evaluation,
                "initial_development_used_for_epoch_selection": False,
                "teacher_harm_audit": harm_audit,
                "site_type_balance_audit": site_type_balance_audit,
                "router_balance_audit": router_balance_audit,
                "outer_test_target_rows_loaded": 0,
                "outer_test_predictions_computed": 0,
            },
        )
        pd.DataFrame(curves).to_csv(staging / "training_curves.csv", index=False)
        summary: dict[str, object] = {
            "schema_version": OUTER_SUMMARY_SCHEMA,
            "status": "pass",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "experiment_id": config["experiment_id"],
            "variant": variant,
            "outer_fold": outer_fold,
            "initialization_seed": initialization_seed,
            "trained_epochs": epochs,
            "diagnostic_epoch_override": diagnostic_override,
            "eligible_for_formal_outer_scoring": not diagnostic_override,
            "device": str(selected_device),
            "split_audit": split_audit,
            "corpus_audit": dict(corpus.audit),
            "transfer_audit": transfer_audit,
            "frozen_base_audit": frozen_base_audit,
            "initial_development_diagnostic": initial_development_evaluation,
            "initial_development_used_for_epoch_selection": False,
            "teacher_harm_audit": harm_audit,
            "site_type_balance_audit": site_type_balance_audit,
            "router_balance_audit": router_balance_audit,
            "source_hashes": source_hashes,
            "model_state_sha256": saved["model_state_sha256"],
            "phase": "frozen_before_outer_test_scoring",
            "outer_test_target_rows_loaded": 0,
            "outer_test_predictions_computed": 0,
            "elapsed_seconds": time.perf_counter() - started,
        }
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
                "schema_version": "nucpred.mayr-joint-site-n-outer-refit-manifest.v1",
                "status": "frozen",
                "files": files,
            },
            ensure_ascii=False,
        )
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        gc.collect()
        if selected_device.type == "cuda":
            torch.cuda.empty_cache()
    summary["output_directory"] = _display_path(target)
    summary["manifest_sha256"] = sha256_file(target / "manifest.json")
    return summary


def _outer_refit_directory(
    config: Mapping[str, Any],
    *,
    variant: str,
    outer_fold: int,
    initialization_seed: int,
) -> Path:
    return (
        _project_path(config["output_directory"], label="output directory")
        / "prototype"
        / variant
        / "outer_refit"
        / f"outer-{outer_fold}"
        / f"seed-{initialization_seed}"
    )


def load_joint_outer_checkpoint(
    path: str | Path,
    *,
    config: Mapping[str, Any],
    device: torch.device,
) -> tuple[MayrJointSiteNModel, SiteNFoldPreprocessor, SolventVocabulary, dict[str, Any]]:
    resolved = Path(path).resolve()
    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != OUTER_CHECKPOINT_SCHEMA:
        raise JointSiteNExperimentError("Outer joint checkpoint schema changed")
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping) or _tensor_mapping_sha256(state) != payload.get(
        "model_state_sha256"
    ):
        raise JointSiteNExperimentError("Outer joint checkpoint state drifted")
    if (
        payload.get("phase")
        != "outer_development_refit_frozen_before_test_scoring"
        or int(payload.get("outer_test_target_rows_loaded", -1)) != 0
        or int(payload.get("outer_test_predictions_computed", -1)) != 0
    ):
        raise JointSiteNExperimentError("Outer joint checkpoint phase changed")
    vocabulary = SolventVocabulary(tuple(map(str, payload["solvent_vocabulary"])))
    preprocessor = SiteNFoldPreprocessor.from_json(payload["preprocessor"])
    settings = variant_settings(str(payload["variant"]))
    model_architecture = payload.get("model_architecture")
    if not isinstance(model_architecture, Mapping):
        raise JointSiteNExperimentError("Outer joint model architecture is missing")
    ranker_architecture = model_architecture.get(
        "publication_site_ranker_architecture"
    )
    if ranker_architecture is not None and not isinstance(
        ranker_architecture, Mapping
    ):
        raise JointSiteNExperimentError("Outer joint ranker architecture is invalid")
    model = new_joint_model(
        config=config,
        vocabulary=vocabulary,
        preprocessor=preprocessor,
        publication_ranker_architecture=ranker_architecture,
        initialization_seed=int(payload["initialization_seed"]),
        settings=settings,
        device=device,
    )
    if payload.get("model_architecture") != model.architecture:
        raise JointSiteNExperimentError("Outer joint architecture drifted")
    model.load_state_dict(state, strict=True)
    if _tensor_mapping_sha256(model.state_dict()) != payload["model_state_sha256"]:
        raise JointSiteNExperimentError("Outer joint checkpoint exact load failed")
    model.eval()
    return model, preprocessor, vocabulary, payload


def _score_unlabeled(
    model: MayrJointSiteNModel,
    *,
    examples,
    queries: pd.DataFrame,
    preprocessor: SiteNFoldPreprocessor,
    vocabulary: SolventVocabulary,
    base_canonical_logits: Mapping[str, float],
    base_router_selected_logits: Mapping[str, float],
    batch_size: int,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = queries.set_index("query_id", drop=False)
    rows: list[dict[str, object]] = []
    model.eval()
    with torch.no_grad():
        for selected in _example_batches(examples, batch_size=batch_size, shuffle_seed=None):
            packed = pack_site_n_batch(
                selected,
                preprocessor=preprocessor,
                solvent_vocabulary=vocabulary,
            )
            query_ids = _batch_query_ids(selected)
            try:
                base = torch.tensor(
                    [float(base_canonical_logits[query_id]) for query_id in query_ids],
                    dtype=torch.float32,
                    device=device,
                )
                base_router = torch.tensor(
                    [
                        float(base_router_selected_logits[query_id])
                        for query_id in query_ids
                    ],
                    dtype=torch.float32,
                    device=device,
                )
            except KeyError as exc:
                raise JointSiteNExperimentError(
                    "Outer-test candidate lacks a frozen v2 base logit"
                ) from exc
            output = model(
                packed.inputs.to(device),
                base_canonical_logits=base,
                base_router_selected_logits=base_router,
            )
            logits = output.canonical_logits.detach().cpu().numpy()
            base_logits = output.base_canonical_logits.detach().cpu().numpy()
            residual_logits = output.residual_canonical_logits.detach().cpu().numpy()
            predictions = (
                output.n_prediction_standardized.detach().cpu().numpy()
                * float(preprocessor.target_scale)
                + float(preprocessor.target_mean)
            )
            for query_id, logit, base_logit, residual_logit, prediction in zip(
                query_ids,
                logits,
                base_logits,
                residual_logits,
                predictions,
                strict=True,
            ):
                row = metadata.loc[str(query_id)]
                rows.append(
                    {
                        "query_id": str(query_id),
                        "context_id": str(row["context_id"]),
                        "species_id": str(row["species_id"]),
                        "connectivity_id": str(row["connectivity_id"]),
                        "candidate_site_id": str(row["candidate_site_id"]),
                        "site_type": str(row["site_type"]),
                        "member_atom_indices_json": str(
                            row["member_atom_indices_json"]
                        ),
                        "canonical_logit": float(logit),
                        "base_canonical_logit": float(base_logit),
                        "residual_canonical_logit": float(residual_logit),
                        "conditional_n_prediction": float(prediction),
                    }
                )
    candidates = pd.DataFrame(rows).sort_values(
        ["context_id", "canonical_logit", "query_id"],
        ascending=[True, False, True],
        kind="stable",
    ).reset_index(drop=True)
    if not np.allclose(
        candidates["canonical_logit"].to_numpy(dtype=float),
        candidates["base_canonical_logit"].to_numpy(dtype=float)
        + candidates["residual_canonical_logit"].to_numpy(dtype=float),
        rtol=0.0,
        atol=2e-6,
    ):
        raise JointSiteNExperimentError("Canonical score decomposition changed")
    candidates["candidate_rank"] = candidates.groupby("context_id").cumcount() + 1
    return candidates, _ranked_context_frame(candidates)


def _ranked_context_frame(candidates: pd.DataFrame) -> pd.DataFrame:
    contexts: list[dict[str, object]] = []
    for context_id, group in candidates.groupby("context_id", sort=True):
        top = group.iloc[0]
        contexts.append(
            {
                "context_id": str(context_id),
                "species_id": str(top["species_id"]),
                "connectivity_id": str(top["connectivity_id"]),
                "predicted_candidate_site_id": str(top["candidate_site_id"]),
                "predicted_site_type": str(top["site_type"]),
                "predicted_n": float(top["conditional_n_prediction"]),
                "top1_canonical_logit": float(top["canonical_logit"]),
                "top1_margin": (
                    float(top["canonical_logit"] - group.iloc[1]["canonical_logit"])
                    if len(group) > 1
                    else np.nan
                ),
                "candidate_count": int(len(group)),
            }
        )
    context_frame = pd.DataFrame(contexts)
    forbidden = {
        "target_id",
        "true_site_id",
        "N_true",
        "evidence_state",
        "validity_label",
        "split_role",
    }
    if forbidden & (set(candidates) | set(context_frame)):
        raise JointSiteNExperimentError("Frozen scores expose target labels")
    return context_frame


def freeze_outer_scores(
    *,
    outer_fold: int,
    initialization_seed: int,
    variant: str = "joint_full",
    config_path: str | Path = DEFAULT_CONFIG,
    device: str | None = None,
    checkpoint_directory: str | Path | None = None,
    output_directory: str | Path | None = None,
) -> dict[str, object]:
    """Score outer-test candidates without reading targets or evidence."""

    started = time.perf_counter()
    config, resolved = read_config(config_path)
    verify_input_bindings(config, resolved)
    selected_device = _device(device or str(config["device"]))
    directory = (
        Path(checkpoint_directory).resolve()
        if checkpoint_directory is not None
        else _outer_refit_directory(
            config,
            variant=variant,
            outer_fold=outer_fold,
            initialization_seed=initialization_seed,
        )
    )
    checkpoint_verification = _verify_manifest(directory)
    checkpoint_path = directory / "model.pt"
    model, preprocessor, vocabulary, checkpoint = load_joint_outer_checkpoint(
        checkpoint_path,
        config=config,
        device=selected_device,
    )
    if (
        checkpoint.get("variant") != variant
        or int(checkpoint.get("outer_fold", -1)) != outer_fold
        or int(checkpoint.get("initialization_seed", -1)) != initialization_seed
    ):
        raise JointSiteNExperimentError("Outer checkpoint identity changed")
    _assert_current_source_hashes(
        checkpoint.get("source_hashes"),
        config_path=resolved,
        label="outer checkpoint",
    )
    membership = pd.read_csv(
        _project_path(
            config["dataset"]["outer_membership_path"], label="outer membership"
        ),
        usecols=["outer_fold", "role", "context_id", "species_id", "connectivity_id"],
    )
    selected = membership.loc[
        membership["outer_fold"].eq(outer_fold) & membership["role"].eq("test")
    ].drop_duplicates("context_id")
    context_ids = tuple(sorted(selected["context_id"].astype(str)))
    universe = load_joint_candidate_universe(
        _project_path(config["dataset"]["directory"], label="dataset"),
        context_ids=context_ids,
    )
    (
        frozen_base_logits,
        frozen_base_router_logits,
        frozen_base_audit,
    ) = load_outer_test_base_logits(
        config,
        outer_fold=outer_fold,
        expected_query_ids=tuple(universe.queries["query_id"].astype(str)),
    )
    candidates, contexts = _score_unlabeled(
        model,
        examples=universe.examples(),
        queries=universe.queries,
        preprocessor=preprocessor,
        vocabulary=vocabulary,
        base_canonical_logits=frozen_base_logits,
        base_router_selected_logits=frozen_base_router_logits,
        batch_size=int(config["training"]["batch_size_contexts"]),
        device=selected_device,
    )
    target = (
        Path(output_directory).resolve()
        if output_directory is not None
        else _project_path(config["output_directory"], label="output directory")
        / "prototype"
        / variant
        / "outer_score_freeze"
        / f"outer-{outer_fold}"
        / f"seed-{initialization_seed}"
    )
    if target.exists():
        raise JointSiteNExperimentError(f"Refusing to overwrite score freeze: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        candidates.to_parquet(staging / "candidate_scores.parquet", index=False)
        contexts.to_parquet(staging / "context_scores.parquet", index=False)
        summary: dict[str, object] = {
            "schema_version": SCORE_SUMMARY_SCHEMA,
            "status": "frozen",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "experiment_id": config["experiment_id"],
            "variant": variant,
            "outer_fold": outer_fold,
            "initialization_seed": initialization_seed,
            "candidate_count": int(len(candidates)),
            "context_count": int(len(contexts)),
            "candidate_universe_audit": dict(universe.audit),
            "checkpoint_path": _display_path(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "checkpoint_model_state_sha256": checkpoint["model_state_sha256"],
            "checkpoint_manifest_verification": checkpoint_verification,
            "frozen_base_audit": frozen_base_audit,
            "config_sha256": sha256_file(resolved),
            "source_hashes": _implementation_source_hashes(resolved),
            "direct_outputs": [
                "canonical_logit",
                "base_canonical_logit",
                "residual_canonical_logit",
                "conditional_N",
            ],
            "canonical_logit_semantics": "frozen_v2_base_plus_joint_residual",
            "candidate_softmax_used": False,
            "target_rows_loaded": 0,
            "evidence_rows_loaded": 0,
            "target_or_site_labels_read": False,
            "outer_test_metrics_computed": 0,
            "diagnostic_epoch_override": bool(
                checkpoint.get("diagnostic_epoch_override", False)
            ),
            "eligible_for_formal_ensemble": bool(
                checkpoint.get("eligible_for_formal_outer_scoring", False)
            ),
            "elapsed_seconds": time.perf_counter() - started,
        }
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
                "schema_version": "nucpred.mayr-joint-site-n-score-manifest.v1",
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


def _seed_score_directory(
    config: Mapping[str, Any],
    *,
    variant: str,
    outer_fold: int,
    initialization_seed: int,
) -> Path:
    return (
        _project_path(config["output_directory"], label="output directory")
        / "prototype"
        / variant
        / "outer_score_freeze"
        / f"outer-{outer_fold}"
        / f"seed-{initialization_seed}"
    )


def ensemble_candidate_scores(
    frames: Sequence[pd.DataFrame],
    *,
    initialization_seeds: Sequence[int],
) -> pd.DataFrame:
    """Average raw fold-matched seed outputs before any label is read."""

    if len(frames) != len(initialization_seeds) or len(frames) < 2:
        raise JointSiteNExperimentError("Ensemble requires matching score frames and seeds")
    identity_columns = [
        "query_id",
        "context_id",
        "species_id",
        "connectivity_id",
        "candidate_site_id",
        "site_type",
        "member_atom_indices_json",
    ]
    ordered = [
        frame.sort_values("query_id", kind="stable").reset_index(drop=True)
        for frame in frames
    ]
    reference = ordered[0][identity_columns]
    for frame in ordered[1:]:
        if not reference.equals(frame[identity_columns]):
            raise JointSiteNExperimentError("Seed score candidate identity changed")
    logits = np.stack(
        [frame["canonical_logit"].to_numpy(dtype=float) for frame in ordered], axis=1
    )
    n_values = np.stack(
        [
            frame["conditional_n_prediction"].to_numpy(dtype=float)
            for frame in ordered
        ],
        axis=1,
    )
    base_logits = np.stack(
        [frame["base_canonical_logit"].to_numpy(dtype=float) for frame in ordered],
        axis=1,
    )
    residual_logits = np.stack(
        [
            frame["residual_canonical_logit"].to_numpy(dtype=float)
            for frame in ordered
        ],
        axis=1,
    )
    if not np.allclose(base_logits, base_logits[:, :1], rtol=0.0, atol=1e-7):
        raise JointSiteNExperimentError("Frozen base logits differ across seeds")
    if not np.allclose(logits, base_logits + residual_logits, rtol=0.0, atol=2e-6):
        raise JointSiteNExperimentError("Seed score decomposition changed")
    result = reference.copy()
    result["base_canonical_logit"] = base_logits[:, 0]
    result["residual_canonical_logit"] = residual_logits.mean(axis=1)
    result["residual_canonical_logit_seed_std"] = residual_logits.std(
        axis=1, ddof=0
    )
    result["canonical_logit"] = logits.mean(axis=1)
    result["canonical_logit_seed_std"] = logits.std(axis=1, ddof=0)
    result["conditional_n_prediction"] = n_values.mean(axis=1)
    result["conditional_n_seed_std"] = n_values.std(axis=1, ddof=0)
    result["ensemble_member_count"] = len(frames)
    result = result.sort_values(
        ["context_id", "canonical_logit", "query_id"],
        ascending=[True, False, True],
        kind="stable",
    ).reset_index(drop=True)
    result["candidate_rank"] = result.groupby("context_id").cumcount() + 1
    return result


def freeze_outer_ensemble_scores(
    *,
    outer_fold: int,
    variant: str = "joint_full",
    config_path: str | Path = DEFAULT_CONFIG,
    score_directories: Sequence[str | Path] | None = None,
    type_router_directory: str | Path | None = None,
    output_directory: str | Path | None = None,
) -> dict[str, object]:
    """Freeze the three-seed mean/std package while scores remain label-blind."""

    config, resolved = read_config(config_path)
    verify_input_bindings(config, resolved)
    frames: list[pd.DataFrame] = []
    bindings: list[dict[str, object]] = []
    eligible = True
    seeds = tuple(map(int, config["initialization_seeds"]))
    if score_directories is not None and len(score_directories) != len(seeds):
        raise JointSiteNExperimentError(
            "Diagnostic ensemble requires one score directory per seed"
        )
    for position, seed in enumerate(seeds):
        directory = (
            Path(score_directories[position]).resolve()
            if score_directories is not None
            else _seed_score_directory(
                config,
                variant=variant,
                outer_fold=outer_fold,
                initialization_seed=seed,
            )
        )
        verification = _verify_manifest(directory)
        summary = _load_json(directory / "summary.json")
        base_audit = summary.get("frozen_base_audit")
        if (
            summary.get("schema_version") != SCORE_SUMMARY_SCHEMA
            or summary.get("status") != "frozen"
            or summary.get("variant") != variant
            or int(summary.get("outer_fold", -1)) != outer_fold
            or int(summary.get("initialization_seed", -1)) != seed
            or summary.get("target_or_site_labels_read") is not False
            or not isinstance(base_audit, Mapping)
            or base_audit.get("source")
            != "label_blind_frozen_v2_canonical_logit"
            or base_audit.get("target_or_site_labels_read") is not False
            or base_audit.get("metrics_computed") is not False
        ):
            raise JointSiteNExperimentError("Seed score package contract changed")
        _assert_current_source_hashes(
            summary.get("source_hashes"),
            config_path=resolved,
            label=f"outer-{outer_fold} seed-{seed} score package",
        )
        eligible = eligible and bool(summary.get("eligible_for_formal_ensemble"))
        candidate_path = directory / "candidate_scores.parquet"
        frames.append(pd.read_parquet(candidate_path))
        bindings.append(
            {
                "initialization_seed": seed,
                "path": _display_path(candidate_path),
                "sha256": sha256_file(candidate_path),
                **verification,
            }
        )
    candidates = ensemble_candidate_scores(frames, initialization_seeds=seeds)
    router_directory = (
        Path(type_router_directory).resolve()
        if type_router_directory is not None
        else _type_router_directory(
            config, variant=variant, outer_fold=outer_fold
        )
    )
    router_verification = _verify_manifest(router_directory)
    router_summary_path = router_directory / "summary.json"
    router_bundle_path = router_directory / "router_bundle.json"
    router_summary = _load_json(router_summary_path)
    router_bundle = _load_json(router_bundle_path)
    if (
        router_summary.get("schema_version") != TYPE_ROUTER_OUTER_SUMMARY_SCHEMA
        or router_summary.get("status") != "frozen"
        or router_summary.get("variant") != variant
        or int(router_summary.get("outer_fold", -1)) != outer_fold
        or router_summary.get("fit_role")
        != "outer_development_inner_oof_only"
        or int(router_summary.get("outer_test_target_rows_loaded", -1)) != 0
        or int(router_summary.get("outer_test_predictions_computed", -1)) != 0
        or router_summary.get("eligible_for_formal_outer_scoring") is not True
        or router_bundle.get("schema_version") != TYPE_ROUTER_SCHEMA_VERSION
        or router_summary.get("router_bundle_sha256")
        != router_bundle.get("bundle_sha256")
    ):
        raise JointSiteNExperimentError("Frozen type-router contract changed")
    _assert_current_source_hashes(
        router_summary.get("source_hashes"),
        config_path=resolved,
        label=f"outer-{outer_fold} type-router",
    )
    context_ids = tuple(sorted(set(candidates["context_id"].astype(str))))
    universe = load_joint_candidate_universe(
        _project_path(config["dataset"]["directory"], label="dataset"),
        context_ids=context_ids,
    )
    features = build_type_router_features(
        candidates,
        queries=universe.queries,
        contexts=universe.contexts,
    )
    router_transport_audit = type_router_feature_transport_audit(
        router_bundle,
        features,
    )
    type_logits = predict_type_router_logits(router_bundle, features)
    candidates = apply_type_router(candidates, type_logits)
    contexts = _ranked_context_frame(candidates)
    target = (
        Path(output_directory).resolve()
        if output_directory is not None
        else _project_path(config["output_directory"], label="output directory")
        / "prototype"
        / variant
        / "outer_score_freeze"
        / f"outer-{outer_fold}"
        / "ensemble"
    )
    if target.exists():
        raise JointSiteNExperimentError(f"Refusing to overwrite score ensemble: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        candidates.to_parquet(staging / "candidate_scores.parquet", index=False)
        contexts.to_parquet(staging / "context_scores.parquet", index=False)
        type_logits.to_parquet(staging / "type_router_logits.parquet", index=False)
        summary: dict[str, object] = {
            "schema_version": SCORE_SUMMARY_SCHEMA,
            "status": "frozen",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "experiment_id": config["experiment_id"],
            "variant": variant,
            "outer_fold": outer_fold,
            "initialization_seed": None,
            "initialization_seeds": list(seeds),
            "ensemble_member_count": len(seeds),
            "ensemble_method": "arithmetic_mean_with_population_std",
            "seed_score_bindings": bindings,
            "type_router_binding": {
                "directory": _display_path(router_directory),
                "summary_path": _display_path(router_summary_path),
                "summary_sha256": sha256_file(router_summary_path),
                "bundle_path": _display_path(router_bundle_path),
                "bundle_file_sha256": sha256_file(router_bundle_path),
                "bundle_payload_sha256": router_bundle["bundle_sha256"],
                **router_verification,
            },
            "type_router_feature_transport_audit": router_transport_audit,
            "candidate_universe_audit": dict(universe.audit),
            "candidate_count": int(len(candidates)),
            "context_count": int(len(contexts)),
            "config_sha256": sha256_file(resolved),
            "source_hashes": _implementation_source_hashes(resolved),
            "direct_outputs": [
                "canonical_logit",
                "pre_router_canonical_logit",
                "type_router_logit",
                "within_type_relative_logit",
                "base_canonical_logit",
                "residual_canonical_logit",
                "conditional_N",
            ],
            "pre_router_canonical_logit_semantics": (
                "frozen_v2_base_plus_mean_joint_residual"
            ),
            "canonical_logit_semantics": (
                "type_router_plus_weighted_pre_router_type_max_plus_"
                "within_type_relative.v1"
            ),
            "candidate_softmax_used": False,
            "target_rows_loaded": 0,
            "evidence_rows_loaded": 0,
            "target_or_site_labels_read": False,
            "outer_test_metrics_computed": 0,
            "eligible_for_formal_evaluation": eligible,
        }
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
                "schema_version": "nucpred.mayr-joint-site-n-score-manifest.v1",
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


def _evaluate_frozen_candidates(
    candidates: pd.DataFrame,
    targets: pd.DataFrame,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    candidate_identity = set(
        zip(
            candidates["context_id"].astype(str),
            candidates["candidate_site_id"].astype(str),
            strict=True,
        )
    )
    target_identity = set(
        zip(
            targets["context_id"].astype(str),
            targets["site_object_id"].astype(str),
            strict=True,
        )
    )
    if not target_identity <= candidate_identity:
        raise JointSiteNExperimentError("Frozen score candidates miss an outer target")
    target_groups = {
        str(context_id): group
        for context_id, group in targets.groupby("context_id", sort=True)
    }
    context_rows: list[dict[str, object]] = []
    for context_id, group in candidates.groupby("context_id", sort=True):
        truth_group = target_groups.get(str(context_id))
        if truth_group is None:
            raise JointSiteNExperimentError("Frozen score context has no target")
        if len(truth_group) != 1:
            continue
        truth = truth_group.iloc[0]
        selected = group.loc[
            group["candidate_site_id"].astype(str).eq(str(truth["site_object_id"]))
        ]
        if len(selected) != 1:
            raise JointSiteNExperimentError("True candidate score is not unique")
        oracle = selected.iloc[0]
        top = group.iloc[0]
        rank = int(oracle["candidate_rank"])
        n_true = float(truth["N_mean"])
        context_rows.append(
            {
                "context_id": str(context_id),
                "species_id": str(top["species_id"]),
                "connectivity_id": str(top["connectivity_id"]),
                "true_target_id": str(truth["target_id"]),
                "true_candidate_site_id": str(truth["site_object_id"]),
                "true_site_type": str(truth["site_type"]),
                "predicted_candidate_site_id": str(top["candidate_site_id"]),
                "predicted_site_type": str(top["site_type"]),
                "candidate_count": int(len(group)),
                "exact_rank": rank,
                "site_top1_correct": rank <= 1,
                "site_top3_correct": rank <= 3,
                "site_top5_correct": rank <= 5,
                "site_reciprocal_rank": 1.0 / rank,
                "N_true": n_true,
                "automatic_n_prediction": float(top["conditional_n_prediction"]),
                "known_site_n_prediction": float(
                    oracle["conditional_n_prediction"]
                ),
                "automatic_n_error": float(top["conditional_n_prediction"]) - n_true,
                "known_site_n_error": (
                    float(oracle["conditional_n_prediction"]) - n_true
                ),
                "top1_canonical_logit": float(top["canonical_logit"]),
                "top1_margin": (
                    float(top["canonical_logit"] - group.iloc[1]["canonical_logit"])
                    if len(group) > 1
                    else np.nan
                ),
            }
        )
    contexts = pd.DataFrame(context_rows)
    overall = _context_metric_summary(contexts)
    by_type = pd.DataFrame(
        [
            {"site_type": str(site_type), **_context_metric_summary(selected)}
            for site_type, selected in contexts.groupby("true_site_type", sort=True)
        ]
    )
    return (
        {
            "schema_version": "nucpred.mayr-joint-site-n-frozen-score-evaluation.v1",
            "primary_population": "single_target_outer_test_contexts",
            "overall": overall,
            "single_target_context_count": int(len(contexts)),
            "multi_target_context_count": int(len(target_groups) - len(contexts)),
            "candidate_recall": 1.0,
        },
        contexts,
        by_type,
    )


def evaluate_outer_scores(
    *,
    outer_fold: int,
    initialization_seed: int | None = None,
    variant: str = "joint_full",
    config_path: str | Path = DEFAULT_CONFIG,
    score_directory: str | Path | None = None,
    output_directory: str | Path | None = None,
) -> dict[str, object]:
    """Read outer labels only after verifying the immutable score package."""

    started = time.perf_counter()
    config, resolved = read_config(config_path)
    verify_input_bindings(config, resolved)
    score_root = (
        Path(score_directory).resolve()
        if score_directory is not None
        else _project_path(config["output_directory"], label="output directory")
        / "prototype"
        / variant
        / "outer_score_freeze"
        / f"outer-{outer_fold}"
        / (f"seed-{initialization_seed}" if initialization_seed is not None else "ensemble")
    )
    score_verification = _verify_manifest(score_root)
    score_summary = _load_json(score_root / "summary.json")
    expected_seed = initialization_seed
    if (
        score_summary.get("schema_version") != SCORE_SUMMARY_SCHEMA
        or score_summary.get("status") != "frozen"
        or score_summary.get("variant") != variant
        or int(score_summary.get("outer_fold", -1)) != outer_fold
        or score_summary.get("initialization_seed") != expected_seed
        or score_summary.get("target_or_site_labels_read") is not False
        or int(score_summary.get("outer_test_metrics_computed", -1)) != 0
    ):
        raise JointSiteNExperimentError("Score package is not label-blind and frozen")
    _assert_current_source_hashes(
        score_summary.get("source_hashes"),
        config_path=resolved,
        label=f"outer-{outer_fold} frozen score package",
    )
    score_frozen_before_label_read_sha256 = sha256_file(
        score_root / "candidate_scores.parquet"
    )
    candidates = pd.read_parquet(score_root / "candidate_scores.parquet")
    membership = pd.read_csv(
        _project_path(
            config["dataset"]["outer_membership_path"], label="outer membership"
        ),
        usecols=["outer_fold", "role", "target_id"],
    )
    target_ids = set(
        membership.loc[
            membership["outer_fold"].eq(outer_fold)
            & membership["role"].eq("test"),
            "target_id",
        ].astype(str)
    )
    target_path = _project_path(config["dataset"]["targets_path"], label="targets")
    targets = pd.read_parquet(
        target_path,
        filters=[("target_id", "in", sorted(target_ids))],
    )
    targets = targets.loc[targets["target_id"].astype(str).isin(target_ids)]
    if len(targets) != len(target_ids):
        raise JointSiteNExperimentError("Outer evaluation target membership changed")
    evaluation, contexts, by_type = _evaluate_frozen_candidates(candidates, targets)
    target = (
        Path(output_directory).resolve()
        if output_directory is not None
        else _project_path(config["output_directory"], label="output directory")
        / "prototype"
        / variant
        / "outer_evaluation"
        / f"outer-{outer_fold}"
        / (f"seed-{initialization_seed}" if initialization_seed is not None else "ensemble")
    )
    if target.exists():
        raise JointSiteNExperimentError(f"Refusing to overwrite outer evaluation: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        contexts.to_parquet(staging / "context_evaluation.parquet", index=False)
        by_type.to_csv(staging / "site_type_metrics.csv", index=False)
        summary: dict[str, object] = {
            "schema_version": EVALUATION_SUMMARY_SCHEMA,
            "status": "pass",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "experiment_id": config["experiment_id"],
            "variant": variant,
            "outer_fold": outer_fold,
            "initialization_seed": initialization_seed,
            "initialization_seeds": score_summary.get("initialization_seeds"),
            "ensemble_member_count": int(
                score_summary.get("ensemble_member_count", 1)
            ),
            "eligible_for_formal_gate": bool(
                score_summary.get(
                    "eligible_for_formal_evaluation",
                    score_summary.get("eligible_for_formal_ensemble", False),
                )
            ),
            "evaluation": evaluation,
            "score_manifest_verification": score_verification,
            "score_summary_sha256": sha256_file(score_root / "summary.json"),
            "score_frozen_before_label_read_sha256": (
                score_frozen_before_label_read_sha256
            ),
            "target_path": target_path.relative_to(ROOT).as_posix(),
            "target_sha256": sha256_file(target_path),
            "target_rows_loaded_after_score_freeze": int(len(targets)),
            "outer_test_used_for_selection": False,
            "config_sha256": sha256_file(resolved),
            "source_hashes": _implementation_source_hashes(resolved),
            "elapsed_seconds": time.perf_counter() - started,
        }
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
                "schema_version": "nucpred.mayr-joint-site-n-evaluation-manifest.v1",
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
    subparsers = parser.add_subparsers(dest="action", required=True)
    selection = subparsers.add_parser("select-epochs")
    selection.add_argument("--outer-fold", type=int, required=True)
    selection.add_argument("--variant", choices=TRAINABLE_VARIANTS, default="joint_full")
    selection.add_argument("--output-path")
    router = subparsers.add_parser("fit-router")
    router.add_argument("--outer-fold", type=int, required=True)
    router.add_argument("--variant", choices=TRAINABLE_VARIANTS, default="joint_full")
    router.add_argument("--output-directory")
    refit = subparsers.add_parser("refit")
    refit.add_argument("--outer-fold", type=int, required=True)
    refit.add_argument("--initialization-seed", type=int, required=True)
    refit.add_argument("--variant", choices=TRAINABLE_VARIANTS, default="joint_full")
    refit.add_argument("--device")
    refit.add_argument("--maximum-epochs", type=int)
    refit.add_argument("--output-directory")
    score = subparsers.add_parser("score")
    score.add_argument("--outer-fold", type=int, required=True)
    score.add_argument("--initialization-seed", type=int, required=True)
    score.add_argument("--variant", choices=TRAINABLE_VARIANTS, default="joint_full")
    score.add_argument("--device")
    score.add_argument("--checkpoint-directory")
    score.add_argument("--output-directory")
    ensemble = subparsers.add_parser("ensemble")
    ensemble.add_argument("--outer-fold", type=int, required=True)
    ensemble.add_argument("--variant", choices=TRAINABLE_VARIANTS, default="joint_full")
    ensemble.add_argument("--output-directory")
    ensemble.add_argument("--seed-score-directory", action="append")
    ensemble.add_argument("--type-router-directory")
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--outer-fold", type=int, required=True)
    evaluate.add_argument("--initialization-seed", type=int)
    evaluate.add_argument("--variant", choices=TRAINABLE_VARIANTS, default="joint_full")
    evaluate.add_argument("--score-directory")
    evaluate.add_argument("--output-directory")
    args = parser.parse_args(argv)
    common = {
        "outer_fold": args.outer_fold,
        "variant": args.variant,
        "config_path": args.config,
    }
    if args.action == "select-epochs":
        result = select_outer_epochs(**common, output_path=args.output_path)
    elif args.action == "fit-router":
        result = fit_outer_type_router(
            **common,
            output_directory=args.output_directory,
        )
    elif args.action == "refit":
        result = run_outer_refit(
            **common,
            initialization_seed=args.initialization_seed,
            device=args.device,
            maximum_epochs=args.maximum_epochs,
            output_directory=args.output_directory,
        )
    elif args.action == "score":
        result = freeze_outer_scores(
            **common,
            initialization_seed=args.initialization_seed,
            device=args.device,
            checkpoint_directory=args.checkpoint_directory,
            output_directory=args.output_directory,
        )
    elif args.action == "ensemble":
        result = freeze_outer_ensemble_scores(
            **common,
            score_directories=args.seed_score_directory,
            type_router_directory=args.type_router_directory,
            output_directory=args.output_directory,
        )
    elif args.action == "evaluate":
        result = evaluate_outer_scores(
            **common,
            initialization_seed=args.initialization_seed,
            score_directory=args.score_directory,
            output_directory=args.output_directory,
        )
    else:  # pragma: no cover
        raise AssertionError(args.action)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
