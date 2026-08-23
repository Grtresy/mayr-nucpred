"""Formal multitype site identification and oracle-gap evaluation.

The workflow is deliberately phase-separated:

``preflight`` seals labels, ``develop`` trains only on train/validation,
``predict-test`` freezes unlabeled full-candidate predictions, and ``test``
is the only phase allowed to open sealed labels.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import gc
import hashlib
import json
import math
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
import torch

from nucpred.core.files import atomic_write_json, sha256_file
from nucpred.project import get_project_layout
from nucpred.training.mayr_site_confidence import tensor_mapping_sha256
from nucpred.training.mayr_site_inference_assets import (
    MayrSiteInferenceAssetError,
    candidate_universe as build_candidate_universe,
    deployment_candidates as load_deployment_candidates,
    encode_split_ensemble as encode_frozen_split_ensemble,
    load_ranker_checkpoint as read_ranker_checkpoint,
    ranker_from_checkpoint as build_ranker_from_checkpoint,
    read_site_identification_config,
    score_ranker_from_source_features,
)
from nucpred.training.mayr_site_ranker import (
    RANKER_ARMS,
    RANKER_SITE_TYPES,
    IndependentSiteRanker,
    TypeAwarePlattCalibrator,
    balanced_site_label_connectivity_weights,
    fit_ranker_arm,
    fit_type_aware_platt,
    retrieval_metrics,
    site_type_indices,
)


ROOT = get_project_layout().root
DEFAULT_CONFIG = ROOT / "configs/mayr_nextgen_site_identification.toml"
CONFIG_SCHEMA = "nucpred.mayr-nextgen-site-identification-config.v1"
PREFLIGHT_SCHEMA = "nucpred.mayr-site-identification-preflight.v1"
DEVELOPMENT_SCHEMA = "nucpred.mayr-site-identification-development.v1"
TEST_PREDICTION_SCHEMA = "nucpred.mayr-site-identification-test-prediction.v1"
TEST_EVALUATION_SCHEMA = "nucpred.mayr-site-identification-test-evaluation.v1"
RUNTIME_REGISTRY_SCHEMA = "nucpred.mayr-site-identification-runtime-registry.v1"


class SiteIdentificationError(RuntimeError):
    """Raised when a formal site-identification boundary is violated."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SiteIdentificationError(f"Cannot read JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise SiteIdentificationError(f"Expected JSON object: {path}")
    return payload


def read_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    try:
        return read_site_identification_config(path)
    except MayrSiteInferenceAssetError as exc:
        raise SiteIdentificationError(str(exc)) from exc


def _repo_path(raw: object, *, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise SiteIdentificationError(f"{label} must be a repo-relative path")
    relative = Path(raw)
    if relative.is_absolute():
        raise SiteIdentificationError(f"{label} must be repo-relative")
    resolved = (ROOT / relative).resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise SiteIdentificationError(f"{label} escapes repository") from exc
    return resolved


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _verify_sha(path: Path, expected: object, *, label: str) -> None:
    if not path.is_file():
        raise SiteIdentificationError(f"Missing {label}: {path}")
    observed = sha256_file(path)
    if observed != str(expected):
        raise SiteIdentificationError(
            f"{label} hash changed: expected {expected}, observed {observed}"
        )


def _manifest(directory: Path, *, schema_version: str) -> dict[str, object]:
    files: list[dict[str, object]] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.name == "run_manifest.json":
            continue
        files.append(
            {
                "path": path.relative_to(directory).as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    payload: dict[str, object] = {
        "schema_version": schema_version,
        "created_at_utc": _utc_now(),
        "files": files,
    }
    payload["content_sha256"] = _canonical_sha256(files)
    return payload


def _publish_stage(
    target: Path,
    *,
    schema_version: str,
    writer: Any,
) -> dict[str, Any]:
    if target.exists():
        summary_path = target / "summary.json"
        manifest_path = target / "run_manifest.json"
        if summary_path.is_file() and manifest_path.is_file():
            summary = _load_json(summary_path)
            if summary.get("status") == "pass":
                return summary
        raise SiteIdentificationError(f"Partial or stale stage exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=target.parent,
        prefix=f".{target.name}.staging-",
    ) as temporary:
        staged = Path(temporary) / target.name
        staged.mkdir()
        summary = writer(staged)
        if not isinstance(summary, dict) or summary.get("status") != "pass":
            raise SiteIdentificationError("Stage writer did not return pass summary")
        atomic_write_json(staged / "summary.json", summary)
        atomic_write_json(
            staged / "run_manifest.json",
            _manifest(staged, schema_version=schema_version),
        )
        staged.replace(target)
    return _load_json(target / "summary.json")


def _source_paths(config: Mapping[str, Any]) -> dict[str, tuple[Path, str]]:
    dataset = config["dataset"]
    evidence = config["evidence"]
    candidate_policy = config["candidate_policy"]
    contract = config["contract"]
    return {
        "dataset_manifest": (
            _repo_path(dataset["dataset_manifest_path"], label="dataset manifest"),
            str(dataset["dataset_manifest_sha256"]),
        ),
        "split_manifest": (
            _repo_path(dataset["split_manifest_path"], label="split manifest"),
            str(dataset["split_manifest_sha256"]),
        ),
        "typed_e3": (
            _repo_path(evidence["typed_e3_path"], label="E3 evidence"),
            str(evidence["typed_e3_sha256"]),
        ),
        "atom_review": (
            _repo_path(evidence["atom_review_path"], label="atom review"),
            str(evidence["atom_review_sha256"]),
        ),
        "atom_group_positive": (
            _repo_path(
                evidence["atom_group_positive_path"],
                label="atom-group positive evidence",
            ),
            str(evidence["atom_group_positive_sha256"]),
        ),
        "candidate_policy": (
            _repo_path(candidate_policy["path"], label="candidate policy"),
            str(candidate_policy["sha256"]),
        ),
        "response_schema": (
            _repo_path(
                contract["response_schema_path"],
                label="response schema",
            ),
            str(contract["response_schema_sha256"]),
        ),
        "candidate_generator": (
            _repo_path(
                contract["candidate_generator_path"],
                label="candidate generator",
            ),
            str(contract["candidate_generator_sha256"]),
        ),
    }


def _verify_sources(config: Mapping[str, Any]) -> dict[str, object]:
    inventory: dict[str, object] = {}
    for label, (path, expected) in _source_paths(config).items():
        _verify_sha(path, expected, label=label)
        inventory[label] = {
            "path": _display_path(path),
            "sha256": expected,
        }
    return inventory


def _deployment_candidates(
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, object]]:
    try:
        return load_deployment_candidates(config)
    except MayrSiteInferenceAssetError as exc:
        raise SiteIdentificationError(str(exc)) from exc


def _dataset_tables(
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = _repo_path(config["dataset"]["directory"], label="dataset directory")
    contexts = pd.read_parquet(root / "contexts.parquet")
    targets = pd.read_parquet(root / "targets.parquet")
    candidates = pd.read_parquet(root / "candidate_sites.parquet")
    splits = pd.read_csv(root / "split_membership.csv")
    return contexts, targets, candidates, splits


def _standardize_evidence(
    config: Mapping[str, Any],
    *,
    targets: pd.DataFrame,
    candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    evidence_config = config["evidence"]
    e3 = pd.read_parquet(
        _repo_path(evidence_config["typed_e3_path"], label="E3 evidence")
    )
    atom_review = pd.read_parquet(
        _repo_path(evidence_config["atom_review_path"], label="atom review")
    )
    atom_group = pd.read_parquet(
        _repo_path(
            evidence_config["atom_group_positive_path"],
            label="atom-group positive evidence",
        )
    )
    if (
        bool(e3["unknown_is_negative"].any())
        or bool(atom_group["unknown_is_negative"].any())
        or not atom_group["formal_positive_eligible"].astype(bool).all()
    ):
        raise SiteIdentificationError("Evidence violates unknown/formal boundary")

    common = [
        "evidence_query_id",
        "target_id",
        "context_id",
        "species_id",
        "connectivity_id",
        "candidate_site_id",
        "site_type",
        "member_atom_indices_json",
        "member_bond_pairs_json",
        "validity_label",
        "evidence_strength_weight",
        "candidate_sampling_probability",
        "candidate_inverse_probability_weight",
        "label_source",
        "endpoint_relative",
        "unknown_is_negative",
    ]

    e3_frame = e3.rename(columns={"binary_site_label": "validity_label"}).copy()
    e3_frame["label_source"] = "stage_d_e3:" + e3_frame["evidence_tier"].astype(str)
    e3_frame["endpoint_relative"] = True
    e3_frame = e3_frame[common]

    candidate_columns = candidates[
        [
            "candidate_site_id",
            "species_id",
            "site_type",
            "member_atom_indices_json",
            "member_bond_pairs_json",
        ]
    ]
    negative = atom_review.merge(
        candidate_columns,
        on=["candidate_site_id", "species_id"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_candidate"),
    )
    if negative["site_type"].isna().any():
        raise SiteIdentificationError("Atom review candidate is missing")
    negative_frame = pd.DataFrame(
        {
            "evidence_query_id": negative["review_id"].astype(str),
            "target_id": negative["target_id"].astype(str),
            "context_id": negative["context_id"].astype(str),
            "species_id": negative["species_id"].astype(str),
            "connectivity_id": negative["connectivity_id"].astype(str),
            "candidate_site_id": negative["candidate_site_id"].astype(str),
            "site_type": negative["site_type"].astype(str),
            "member_atom_indices_json": negative["member_atom_indices_json"].astype(
                str
            ),
            "member_bond_pairs_json": negative["member_bond_pairs_json"].astype(str),
            "validity_label": 0,
            "evidence_strength_weight": 1.0,
            "candidate_sampling_probability": negative[
                "candidate_sampling_probability"
            ].astype(float),
            "candidate_inverse_probability_weight": negative[
                "candidate_inverse_probability_weight"
            ].astype(float),
            "label_source": "atom_formal_review_negative",
            "endpoint_relative": True,
            "unknown_is_negative": False,
        }
    )

    atom_target_ids = sorted(set(atom_review["target_id"].astype(str)))
    positive_targets = targets.loc[
        targets["target_id"].astype(str).isin(atom_target_ids)
    ].copy()
    if (
        len(positive_targets) != len(atom_target_ids)
        or not positive_targets["site_type"].eq("atom").all()
    ):
        raise SiteIdentificationError("Atom review positive targets changed")
    positive_frame = pd.DataFrame(
        {
            "evidence_query_id": "atom-positive:"
            + positive_targets["target_id"].astype(str),
            "target_id": positive_targets["target_id"].astype(str),
            "context_id": positive_targets["context_id"].astype(str),
            "species_id": positive_targets["species_id"].astype(str),
            "connectivity_id": positive_targets["connectivity_id"].astype(str),
            "candidate_site_id": positive_targets["site_object_id"].astype(str),
            "site_type": positive_targets["site_type"].astype(str),
            "member_atom_indices_json": positive_targets[
                "member_atom_indices_json"
            ].astype(str),
            "member_bond_pairs_json": positive_targets["member_bond_pairs_json"].astype(
                str
            ),
            "validity_label": 1,
            "evidence_strength_weight": 1.0,
            "candidate_sampling_probability": 1.0,
            "candidate_inverse_probability_weight": 1.0,
            "label_source": "atom_formal_exact_target_positive",
            "endpoint_relative": True,
            "unknown_is_negative": False,
        }
    )

    atom_group_frame = pd.DataFrame(
        {
            "evidence_query_id": "atom-group-positive:"
            + atom_group["target_id"].astype(str),
            "target_id": atom_group["target_id"].astype(str),
            "context_id": atom_group["context_id"].astype(str),
            "species_id": atom_group["species_id"].astype(str),
            "connectivity_id": atom_group["connectivity_id"].astype(str),
            "candidate_site_id": atom_group["candidate_site_id"].astype(str),
            "site_type": atom_group["site_type"].astype(str),
            "member_atom_indices_json": atom_group["member_atom_indices_json"].astype(
                str
            ),
            "member_bond_pairs_json": atom_group["member_bond_pairs_json"].astype(str),
            "validity_label": 1,
            "evidence_strength_weight": 1.0,
            "candidate_sampling_probability": 1.0,
            "candidate_inverse_probability_weight": 1.0,
            "label_source": "stage_e_c_formal_exact_endpoint_positive",
            "endpoint_relative": True,
            "unknown_is_negative": False,
        }
    )

    evidence = pd.concat(
        [e3_frame, negative_frame, positive_frame, atom_group_frame],
        ignore_index=True,
    )
    if evidence["evidence_query_id"].astype(str).duplicated().any():
        raise SiteIdentificationError("Evidence query IDs are not unique")
    label_counts = evidence.groupby(
        ["target_id", "candidate_site_id"],
        sort=True,
    )["validity_label"].nunique()
    if int((label_counts > 1).sum()):
        raise SiteIdentificationError("Conflicting reviewed candidate labels")
    if evidence.duplicated(["target_id", "candidate_site_id"]).any():
        raise SiteIdentificationError("Duplicate reviewed candidate evidence")
    if set(evidence["site_type"].astype(str)) != set(RANKER_SITE_TYPES):
        raise SiteIdentificationError("Evidence does not cover all five site types")
    if set(evidence["validity_label"].astype(int)) != {0, 1}:
        raise SiteIdentificationError("Evidence is not binary")

    candidate_identity = candidates[
        [
            "candidate_site_id",
            "species_id",
            "site_type",
            "member_atom_indices_json",
            "member_bond_pairs_json",
        ]
    ].rename(
        columns={
            "species_id": "candidate_species_id",
            "site_type": "candidate_site_type",
            "member_atom_indices_json": "candidate_members_json",
            "member_bond_pairs_json": "candidate_bonds_json",
        }
    )
    audit = evidence.merge(
        candidate_identity,
        on="candidate_site_id",
        how="left",
        validate="many_to_one",
    )
    mismatched = (
        audit["candidate_species_id"].isna()
        | audit["species_id"].astype(str).ne(audit["candidate_species_id"].astype(str))
        | audit["site_type"].astype(str).ne(audit["candidate_site_type"].astype(str))
        | audit["member_atom_indices_json"]
        .astype(str)
        .ne(audit["candidate_members_json"].astype(str))
        | audit["member_bond_pairs_json"]
        .astype(str)
        .ne(audit["candidate_bonds_json"].astype(str))
    )
    if bool(mismatched.any()):
        raise SiteIdentificationError("Evidence/candidate identity mismatch")
    deployment_candidates, _ = _deployment_candidates(config)
    deployment_ids = set(deployment_candidates["candidate_site_id"].astype(str))
    deployment_eligible = evidence["candidate_site_id"].astype(str).isin(deployment_ids)
    excluded = evidence.loc[~deployment_eligible]
    filter_audit = {
        "reviewed_evidence_before_candidate_policy_count": len(evidence),
        "reviewed_evidence_excluded_by_candidate_policy_count": len(excluded),
        "reviewed_evidence_after_candidate_policy_count": int(
            deployment_eligible.sum()
        ),
        "excluded_counts_by_site_type_and_label": (
            excluded.groupby(["site_type", "validity_label"])
            .size()
            .rename("row_count")
            .reset_index()
            .to_dict("records")
        ),
        "excluded_positive_count": int(excluded["validity_label"].astype(int).sum()),
        "filter_target_independent": True,
    }
    if filter_audit["excluded_positive_count"] != 0:
        raise SiteIdentificationError(
            "Deployment candidate filter removed a reviewed positive"
        )
    evidence = evidence.loc[deployment_eligible].copy()
    evidence["query_id"] = evidence["evidence_query_id"].astype(str)
    evidence["N_value"] = np.nan
    return (
        evidence.sort_values("query_id", kind="stable").reset_index(drop=True),
        filter_audit,
    )


def run_preflight(
    config: Mapping[str, Any],
    *,
    config_path: Path,
) -> dict[str, Any]:
    output_root = _repo_path(
        config["output_directory"],
        label="output directory",
    )
    target = output_root / "preflight"

    def writer(staged: Path) -> dict[str, Any]:
        source_inventory = _verify_sources(config)
        contexts, targets, candidates, splits = _dataset_tables(config)
        deployment_candidates, candidate_policy_audit = _deployment_candidates(config)
        target_covered = (
            targets["site_object_id"]
            .astype(str)
            .isin(set(deployment_candidates["candidate_site_id"].astype(str)))
        )
        if not bool(target_covered.all()):
            raise SiteIdentificationError(
                "Contract-compatible deployment candidate target coverage changed"
            )
        coverage_by_type = {
            str(site_type): float(
                group["site_object_id"]
                .astype(str)
                .isin(set(deployment_candidates["candidate_site_id"].astype(str)))
                .mean()
            )
            for site_type, group in targets.groupby("site_type", sort=True)
        }
        candidate_policy_audit["known_target_coverage_by_type"] = coverage_by_type
        evidence, evidence_filter_audit = _standardize_evidence(
            config,
            targets=targets,
            candidates=candidates,
        )
        configured_splits = tuple(
            int(value) for value in config["backbone"]["split_seeds"]
        )
        observed_splits = tuple(sorted(splits["split_seed"].unique()))
        if configured_splits != observed_splits:
            raise SiteIdentificationError("Configured split seeds changed")
        if not targets["formal_training_eligible"].astype(bool).all():
            raise SiteIdentificationError("Target table contains ineligible rows")

        split_summaries: list[dict[str, object]] = []
        for split_seed in configured_splits:
            membership = splits.loc[splits["split_seed"].eq(split_seed)].copy()
            if membership["target_id"].astype(str).duplicated().any():
                raise SiteIdentificationError("Split target membership is duplicated")
            role_connectivities = {
                role: set(
                    membership.loc[
                        membership["role"].eq(role),
                        "connectivity_id",
                    ].astype(str)
                )
                for role in ("train", "validation", "test")
            }
            if (
                role_connectivities["train"] & role_connectivities["validation"]
                or role_connectivities["train"] & role_connectivities["test"]
                or role_connectivities["validation"] & role_connectivities["test"]
            ):
                raise SiteIdentificationError("Split connectivity leakage detected")

            labeled = evidence.merge(
                membership[["target_id", "role"]],
                on="target_id",
                how="left",
                validate="many_to_one",
            )
            if labeled["role"].isna().any():
                raise SiteIdentificationError("Evidence lacks split membership")
            development = labeled.loc[
                labeled["role"].isin(["train", "validation"])
            ].copy()
            test_reviewed = labeled.loc[labeled["role"].eq("test")].copy()
            test_membership = membership.loc[membership["role"].eq("test")].copy()
            test_targets = targets.merge(
                test_membership[["target_id"]],
                on="target_id",
                how="inner",
                validate="one_to_one",
            )
            if len(test_targets) != len(test_membership):
                raise SiteIdentificationError("Sealed test target count changed")
            unlabeled_contexts = (
                test_membership[["context_id", "species_id", "connectivity_id"]]
                .drop_duplicates()
                .sort_values("context_id", kind="stable")
                .reset_index(drop=True)
            )
            split_dir = staged / f"split-{split_seed}"
            sealed_dir = staged / "sealed" / f"split-{split_seed}"
            split_dir.mkdir(parents=True)
            sealed_dir.mkdir(parents=True)
            development.to_parquet(
                split_dir / "development_labeled_queries.parquet",
                index=False,
                compression="zstd",
            )
            unlabeled_contexts.to_parquet(
                split_dir / "test_contexts.unlabeled.parquet",
                index=False,
                compression="zstd",
            )
            test_reviewed.to_parquet(
                sealed_dir / "reviewed_test_labels.parquet",
                index=False,
                compression="zstd",
            )
            test_targets.to_parquet(
                sealed_dir / "target_test_labels.parquet",
                index=False,
                compression="zstd",
            )
            counts = (
                labeled.groupby(["role", "site_type", "validity_label"])
                .size()
                .rename("row_count")
                .reset_index()
            )
            counts.to_csv(split_dir / "evidence_counts.csv", index=False)
            split_summaries.append(
                {
                    "split_seed": split_seed,
                    "development_query_count": len(development),
                    "reviewed_test_query_count": len(test_reviewed),
                    "sealed_test_target_count": len(test_targets),
                    "unlabeled_test_context_count": len(unlabeled_contexts),
                    "train_connectivity_count": len(role_connectivities["train"]),
                    "validation_connectivity_count": len(
                        role_connectivities["validation"]
                    ),
                    "test_connectivity_count": len(role_connectivities["test"]),
                    "connectivity_disjoint": True,
                }
            )

        evidence.to_parquet(
            staged / "combined_reviewed_evidence.parquet",
            index=False,
            compression="zstd",
        )
        atomic_write_json(staged / "source_inventory.json", source_inventory)
        pd.DataFrame(split_summaries).to_csv(
            staged / "split_summary.csv",
            index=False,
        )
        type_label_counts = (
            evidence.groupby(["site_type", "validity_label"])
            .size()
            .rename("row_count")
            .reset_index()
            .to_dict("records")
        )
        return {
            "schema_version": PREFLIGHT_SCHEMA,
            "status": "pass",
            "campaign_id": config["campaign_id"],
            "goal_thread_id": config["goal_thread_id"],
            "config_path": _display_path(config_path),
            "config_sha256": sha256_file(config_path),
            "reviewed_evidence_query_count": len(evidence),
            "reviewed_target_count": int(evidence["target_id"].nunique()),
            "site_type_label_counts": type_label_counts,
            "candidate_policy_audit": candidate_policy_audit,
            "reviewed_evidence_filter_audit": evidence_filter_audit,
            "split_summaries": split_summaries,
            "unknown_as_negative_count": 0,
            "stage_e_a_diagnostic_positive_used": False,
            "test_labels_sealed": True,
            "test_labels_used_for_selection": False,
            "candidate_softmax_used": False,
        }

    return _publish_stage(
        target,
        schema_version=PREFLIGHT_SCHEMA,
        writer=writer,
    )


def _encode_split_ensemble(
    *,
    config: Mapping[str, Any],
    split_seed: int,
    queries: pd.DataFrame,
    contexts: pd.DataFrame,
    device: torch.device,
) -> tuple[list[str], torch.Tensor, np.ndarray, np.ndarray, list[dict[str, object]]]:
    try:
        return encode_frozen_split_ensemble(
            config=config,
            split_seed=split_seed,
            queries=queries,
            contexts=contexts,
            device=device,
        )
    except MayrSiteInferenceAssetError as exc:
        raise SiteIdentificationError(str(exc)) from exc


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))


def _ece(
    labels: np.ndarray,
    probability: np.ndarray,
    weights: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = float(weights.sum())
    value = 0.0
    for index in range(bins):
        if index == bins - 1:
            selected = (probability >= edges[index]) & (probability <= edges[index + 1])
        else:
            selected = (probability >= edges[index]) & (probability < edges[index + 1])
        if not selected.any():
            continue
        selected_weights = weights[selected]
        mass = float(selected_weights.sum())
        confidence = float(np.average(probability[selected], weights=selected_weights))
        accuracy = float(np.average(labels[selected], weights=selected_weights))
        value += mass / total * abs(confidence - accuracy)
    return float(value)


def _binary_metrics(
    frame: pd.DataFrame,
    *,
    probability_column: str,
    slice_name: str,
) -> dict[str, object]:
    labels = frame["validity_label"].to_numpy(dtype=int)
    probability = frame[probability_column].to_numpy(dtype=float)
    weights = frame["evaluation_weight"].to_numpy(dtype=float)
    prevalence = float(np.average(labels, weights=weights))
    return {
        "slice": slice_name,
        "query_count": len(frame),
        "positive_query_count": int(labels.sum()),
        "negative_query_count": int((labels == 0).sum()),
        "positive_connectivity_count": int(
            frame.loc[frame["validity_label"].eq(1), "connectivity_id"].nunique()
        ),
        "negative_connectivity_count": int(
            frame.loc[frame["validity_label"].eq(0), "connectivity_id"].nunique()
        ),
        "roc_auc": (
            float(roc_auc_score(labels, probability, sample_weight=weights))
            if len(set(labels)) == 2
            else float("nan")
        ),
        "average_precision": (
            float(
                average_precision_score(
                    labels,
                    probability,
                    sample_weight=weights,
                )
            )
            if int(labels.sum()) > 0
            else float("nan")
        ),
        "weighted_prevalence": prevalence,
        "weighted_brier": float(
            np.average((probability - labels) ** 2, weights=weights)
        ),
        "weighted_ece_10_bin": _ece(
            labels,
            probability,
            weights,
            bins=10,
        ),
    }


def _review_population_weights(frame: pd.DataFrame) -> torch.Tensor:
    """Recover reviewed candidate-population mass from sampling probabilities."""

    raw = frame["candidate_inverse_probability_weight"].to_numpy(dtype=float) * frame[
        "evidence_strength_weight"
    ].to_numpy(dtype=float)
    if not len(raw) or not np.isfinite(raw).all() or (raw <= 0).any():
        raise SiteIdentificationError(
            "Review population weights must be finite and positive"
        )
    raw = raw / raw.sum()
    return torch.tensor(raw, dtype=torch.float32)


def _ranker_from_checkpoint(
    checkpoint: Mapping[str, Any],
) -> IndependentSiteRanker:
    try:
        return build_ranker_from_checkpoint(checkpoint)
    except MayrSiteInferenceAssetError as exc:
        raise SiteIdentificationError(str(exc)) from exc


def run_development(
    config: Mapping[str, Any],
    *,
    config_path: Path,
) -> dict[str, Any]:
    output_root = _repo_path(
        config["output_directory"],
        label="output directory",
    )
    preflight = output_root / "preflight"
    if _load_json(preflight / "summary.json").get("status") != "pass":
        raise SiteIdentificationError("Preflight is not complete")
    target = output_root / "development"
    device = torch.device(str(config["device"]))
    settings = config["ranker"]
    calibration_settings = config["calibration"]
    _, _, _, _ = _dataset_tables(config)

    def writer(staged: Path) -> dict[str, Any]:
        contexts, _, _, _ = _dataset_tables(config)
        split_summaries: list[dict[str, object]] = []
        for split_seed_raw in config["backbone"]["split_seeds"]:
            split_seed = int(split_seed_raw)
            development = pd.read_parquet(
                preflight
                / f"split-{split_seed}"
                / "development_labeled_queries.parquet"
            )
            if not set(development["role"].astype(str)) <= {
                "train",
                "validation",
            }:
                raise SiteIdentificationError("Development opened a test label")
            query_ids, features, n_mean, n_std, backbone_bindings = (
                _encode_split_ensemble(
                    config=config,
                    split_seed=split_seed,
                    queries=development,
                    contexts=contexts,
                    device=device,
                )
            )
            ordered = (
                development.set_index("query_id", drop=False)
                .loc[query_ids]
                .reset_index(drop=True)
            )
            ordered["conditional_N_mean"] = n_mean
            ordered["conditional_N_std"] = n_std
            train_mask = ordered["role"].eq("train").to_numpy()
            validation_mask = ordered["role"].eq("validation").to_numpy()
            train = ordered.loc[train_mask].reset_index(drop=True)
            validation = ordered.loc[validation_mask].reset_index(drop=True)
            train_features = features[torch.from_numpy(train_mask)]
            validation_features = features[torch.from_numpy(validation_mask)]
            train_type_index = site_type_indices(train["site_type"].astype(str))
            validation_type_index = site_type_indices(
                validation["site_type"].astype(str)
            )
            train_labels = torch.tensor(
                train["validity_label"].to_numpy(dtype=float),
                dtype=torch.float32,
            )
            validation_labels = validation["validity_label"].to_numpy(dtype=int)
            train_weights = balanced_site_label_connectivity_weights(
                labels=train["validity_label"].astype(int),
                site_types=train["site_type"].astype(str),
                connectivity_ids=train["connectivity_id"].astype(str),
                sampling_weights=(
                    train["candidate_inverse_probability_weight"].astype(float)
                    * train["evidence_strength_weight"].astype(float)
                ),
            )
            validation_weights = _review_population_weights(validation)
            target_codes, _ = pd.factorize(
                train["target_id"].astype(str),
                sort=True,
            )
            train_group_index = torch.tensor(target_codes, dtype=torch.long)

            def validation_ap(
                labels: np.ndarray,
                logits: np.ndarray,
            ) -> float:
                return float(
                    average_precision_score(
                        labels,
                        _sigmoid(logits),
                        sample_weight=validation_weights.numpy(),
                    )
                )

            arm_results = {}
            for arm_index, arm in enumerate(RANKER_ARMS):
                arm_results[arm] = fit_ranker_arm(
                    arm=arm,
                    train_features=train_features,
                    train_type_index=train_type_index,
                    train_labels=train_labels,
                    train_group_index=train_group_index,
                    train_weights=train_weights,
                    validation_features=validation_features,
                    validation_type_index=validation_type_index,
                    validation_labels=validation_labels,
                    validation_target_ids=validation["target_id"].astype(str),
                    validation_average_precision=validation_ap,
                    hidden_dim=int(settings["hidden_dim"]),
                    type_adapter_dim=int(settings["type_adapter_dim"]),
                    learning_rate=float(settings["learning_rate"]),
                    weight_decay=float(settings["weight_decay"]),
                    maximum_epochs=int(settings["maximum_epochs"]),
                    minimum_epochs=int(settings["minimum_epochs"]),
                    evaluation_interval=int(settings["evaluation_interval"]),
                    patience_evaluations=int(
                        settings["early_stopping_patience_evaluations"]
                    ),
                    pairwise_loss_weight=float(settings["pairwise_loss_weight"]),
                    gradient_clip_norm=float(settings["gradient_clip_norm"]),
                    seed=int(settings["training_seed_offset"]) + split_seed + arm_index,
                )
            selection_keys = {
                arm: tuple(result.audit["best_selection_key"])
                for arm, result in arm_results.items()
            }
            selected_arm = max(RANKER_ARMS, key=selection_keys.__getitem__)
            selected = arm_results[selected_arm]
            validation_logits = selected.validation_logits
            calibrator, calibrator_audit = fit_type_aware_platt(
                logits=torch.tensor(validation_logits, dtype=torch.float32),
                type_index=validation_type_index,
                labels=torch.tensor(validation_labels, dtype=torch.float32),
                weights=validation_weights,
                l2_type_offset=float(calibration_settings["l2_type_offset"]),
                l2_log_slope=float(calibration_settings["l2_log_slope"]),
                maximum_iterations=int(calibration_settings["maximum_iterations"]),
            )
            with torch.no_grad():
                calibrated = calibrator(
                    torch.tensor(validation_logits, dtype=torch.float32),
                    validation_type_index,
                ).numpy()
            validation_predictions = validation.copy()
            validation_predictions["validity_logit"] = validation_logits
            validation_predictions["raw_sigmoid_score"] = _sigmoid(validation_logits)
            validation_predictions["absolute_site_probability"] = calibrated
            validation_predictions["evaluation_weight"] = validation_weights.numpy()
            calibration_metrics = [
                _binary_metrics(
                    validation_predictions,
                    probability_column="absolute_site_probability",
                    slice_name="overall",
                )
            ]
            calibration_metrics.extend(
                _binary_metrics(
                    group,
                    probability_column="absolute_site_probability",
                    slice_name=str(site_type),
                )
                for site_type, group in validation_predictions.groupby(
                    "site_type",
                    sort=True,
                )
            )
            retrieval = retrieval_metrics(
                labels=validation_labels,
                scores=validation_logits,
                target_ids=validation["target_id"].astype(str),
            )
            state = deepcopy(selected.model.state_dict())
            checkpoint: dict[str, object] = {
                "schema_version": "nucpred.mayr-site-ranker-checkpoint.v1",
                "phase": "development_frozen",
                "campaign_id": config["campaign_id"],
                "split_seed": split_seed,
                "selected_arm": selected_arm,
                "ranker_architecture": selected.model.architecture,
                "ranker_state_dict": state,
                "ranker_state_sha256": tensor_mapping_sha256(state),
                "calibrator": calibrator.to_payload(),
                "calibrator_fit_audit": calibrator_audit,
                "calibration_weighting": calibration_settings["weighting"],
                "backbone_bindings": backbone_bindings,
                "training_roles": ["train"],
                "selection_and_calibration_roles": ["validation"],
                "test_labels_read": False,
                "test_predictions_computed": False,
                "conditional_n_backbone_frozen": True,
                "unknown_as_negative_count": 0,
                "candidate_softmax_used": False,
            }
            split_dir = staged / f"split-{split_seed}"
            split_dir.mkdir()
            torch.save(checkpoint, split_dir / "ranker_checkpoint.pt")
            validation_predictions.to_parquet(
                split_dir / "validation_predictions.parquet",
                index=False,
                compression="zstd",
            )
            pd.DataFrame(calibration_metrics).to_csv(
                split_dir / "validation_calibration_metrics.csv",
                index=False,
            )
            atomic_write_json(
                split_dir / "arm_fit_audits.json",
                {arm: arm_results[arm].audit for arm in RANKER_ARMS},
            )
            split_summaries.append(
                {
                    "split_seed": split_seed,
                    "selected_arm": selected_arm,
                    "train_query_count": len(train),
                    "validation_query_count": len(validation),
                    "validation_eligible_target_count": retrieval[
                        "eligible_target_count"
                    ],
                    "validation_mrr": retrieval["mrr"],
                    "validation_top1_recall": retrieval["top1_recall"],
                    "validation_top3_recall": retrieval["top3_recall"],
                    "calibrator_positive_slope": calibrator.to_payload()[
                        "positive_slope"
                    ],
                    "test_labels_read": False,
                    "test_predictions_computed": False,
                }
            )
            del (
                features,
                train_features,
                validation_features,
                arm_results,
                selected,
            )
            gc.collect()
        pd.DataFrame(split_summaries).to_csv(
            staged / "split_summary.csv",
            index=False,
        )
        return {
            "schema_version": DEVELOPMENT_SCHEMA,
            "status": "pass",
            "campaign_id": config["campaign_id"],
            "config_sha256": sha256_file(config_path),
            "preflight_manifest_sha256": sha256_file(preflight / "run_manifest.json"),
            "split_summaries": split_summaries,
            "development_frozen": True,
            "conditional_n_backbone_frozen": True,
            "test_labels_read": False,
            "test_predictions_computed": False,
            "unknown_as_negative_count": 0,
            "candidate_softmax_used": False,
            "calibration_weighting": calibration_settings["weighting"],
        }

    return _publish_stage(
        target,
        schema_version=DEVELOPMENT_SCHEMA,
        writer=writer,
    )


def _candidate_universe(
    *,
    test_contexts: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    try:
        return build_candidate_universe(
            test_contexts=test_contexts,
            candidates=candidates,
        )
    except MayrSiteInferenceAssetError as exc:
        raise SiteIdentificationError(str(exc)) from exc


def _load_ranker_checkpoint(
    path: Path,
    *,
    split_seed: int,
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    try:
        return read_ranker_checkpoint(
            path,
            split_seed=split_seed,
            config=config,
        )
    except MayrSiteInferenceAssetError as exc:
        raise SiteIdentificationError(str(exc)) from exc


def run_test_predictions(
    config: Mapping[str, Any],
    *,
    config_path: Path,
) -> dict[str, Any]:
    """Freeze full candidate-space test predictions without opening labels."""

    output_root = _repo_path(
        config["output_directory"],
        label="output directory",
    )
    preflight = output_root / "preflight"
    development = output_root / "development"
    if _load_json(development / "summary.json").get("status") != "pass":
        raise SiteIdentificationError("Development freeze is not complete")
    target = output_root / "test_predictions"
    device = torch.device(str(config["device"]))

    def writer(staged: Path) -> dict[str, Any]:
        contexts, _, _, _ = _dataset_tables(config)
        candidates, candidate_policy_audit = _deployment_candidates(config)
        split_summaries: list[dict[str, object]] = []
        for split_seed_raw in config["backbone"]["split_seeds"]:
            split_seed = int(split_seed_raw)
            # This is the only preflight test asset that prediction may read.
            # It contains context identity, never target site, N, or labels.
            test_contexts = pd.read_parquet(
                preflight / f"split-{split_seed}" / "test_contexts.unlabeled.parquet"
            )
            universe = _candidate_universe(
                test_contexts=test_contexts,
                candidates=candidates,
            )
            query_ids, features, n_mean, n_std, backbone_bindings = (
                _encode_split_ensemble(
                    config=config,
                    split_seed=split_seed,
                    queries=universe,
                    contexts=contexts,
                    device=device,
                )
            )
            ordered = (
                universe.set_index("query_id", drop=False)
                .loc[query_ids]
                .reset_index(drop=True)
            )
            ranker_path = development / f"split-{split_seed}" / "ranker_checkpoint.pt"
            ranker_checkpoint = _load_ranker_checkpoint(
                ranker_path,
                split_seed=split_seed,
                config=config,
            )
            ranker = _ranker_from_checkpoint(ranker_checkpoint)
            type_index = site_type_indices(ordered["site_type"].astype(str))
            with torch.no_grad():
                score_components = score_ranker_from_source_features(
                    ranker=ranker,
                    checkpoint=ranker_checkpoint,
                    source_features=features,
                    type_index=type_index,
                )
                logits = score_components["canonical_logit"]
            calibrator_payload = ranker_checkpoint["calibrator"]
            if not isinstance(calibrator_payload, Mapping):
                raise SiteIdentificationError("Ranker calibrator is missing")
            calibrator = TypeAwarePlattCalibrator.from_payload(calibrator_payload)
            with torch.no_grad():
                probability = calibrator(logits, type_index)
            predictions = ordered.copy()
            predictions["validity_logit"] = logits.numpy()
            predictions["membership_logit"] = score_components[
                "membership_logit"
            ].numpy()
            predictions["router_selected_logit"] = score_components[
                "router_selected_logit"
            ].numpy()
            predictions["compatibility_logit"] = score_components[
                "compatibility_logit"
            ].numpy()
            predictions["raw_sigmoid_score"] = _sigmoid(logits.numpy())
            predictions["absolute_site_probability"] = probability.numpy()
            predictions["conditional_N_mean"] = n_mean
            predictions["conditional_N_std"] = n_std
            predictions["prediction_split_seed"] = split_seed
            predictions["selected_ranker_arm"] = str(ranker_checkpoint["selected_arm"])
            predictions["candidate_scores_independent"] = True
            predictions["candidate_softmax_used"] = False
            predictions["target_or_site_label_read"] = False
            split_dir = staged / f"split-{split_seed}"
            split_dir.mkdir()
            prediction_path = split_dir / "candidate_predictions.parquet"
            predictions.to_parquet(
                prediction_path,
                index=False,
                compression="zstd",
            )
            freeze = {
                "schema_version": "nucpred.mayr-test-prediction-freeze.v1",
                "campaign_id": config["campaign_id"],
                "split_seed": split_seed,
                "candidate_prediction_path": _display_path(
                    output_root
                    / "test_predictions"
                    / f"split-{split_seed}"
                    / "candidate_predictions.parquet"
                ),
                "candidate_prediction_sha256": sha256_file(prediction_path),
                "candidate_prediction_row_count": len(predictions),
                "test_context_count": int(predictions["context_id"].nunique()),
                "ranker_checkpoint_path": _display_path(ranker_path),
                "ranker_checkpoint_sha256": sha256_file(ranker_path),
                "backbone_bindings": backbone_bindings,
                "test_labels_read": False,
                "target_values_read": False,
                "candidate_softmax_used": False,
                "unknown_as_negative_count": 0,
            }
            freeze["freeze_sha256"] = _canonical_sha256(freeze)
            atomic_write_json(split_dir / "prediction_freeze.json", freeze)
            counts = Counter(predictions["site_type"].astype(str))
            split_summaries.append(
                {
                    "split_seed": split_seed,
                    "candidate_prediction_count": len(predictions),
                    "test_context_count": int(predictions["context_id"].nunique()),
                    "candidate_count_by_type": {
                        site_type: int(counts[site_type])
                        for site_type in RANKER_SITE_TYPES
                    },
                    "selected_ranker_arm": str(ranker_checkpoint["selected_arm"]),
                    "prediction_sha256": freeze["candidate_prediction_sha256"],
                    "test_labels_read": False,
                }
            )
            del features, ranker, predictions
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

    return _publish_stage(
        target,
        schema_version=TEST_PREDICTION_SCHEMA,
        writer=writer,
    )


def _regression_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float | int]:
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    finite = np.isfinite(truth) & np.isfinite(prediction)
    truth = truth[finite]
    prediction = prediction[finite]
    if not len(truth):
        return {
            "count": 0,
            "mae": float("nan"),
            "rmse": float("nan"),
            "r2": float("nan"),
        }
    residual = prediction - truth
    denominator = float(np.sum((truth - truth.mean()) ** 2))
    return {
        "count": len(truth),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "r2": (
            float(1.0 - np.sum(residual**2) / denominator)
            if denominator > 0
            else float("nan")
        ),
    }


def _parse_member_set(value: object) -> frozenset[int]:
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, list):
        raise SiteIdentificationError("Candidate membership is not a list")
    return frozenset(int(item) for item in parsed)


def _membership_jaccard(left: object, right: object) -> float:
    left_set = _parse_member_set(left)
    right_set = _parse_member_set(right)
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 0.0


def _full_space_target_results(
    *,
    predictions: pd.DataFrame,
    targets: pd.DataFrame,
    compatible_jaccard_threshold: float,
) -> pd.DataFrame:
    identity_columns = [
        "context_id",
        "candidate_site_id",
        "site_type",
        "member_atom_indices_json",
        "validity_logit",
        "absolute_site_probability",
        "conditional_N_mean",
        "conditional_N_std",
    ]
    ranked = predictions[identity_columns].sort_values(
        ["context_id", "validity_logit", "candidate_site_id"],
        ascending=[True, False, True],
        kind="stable",
    )
    ranked = ranked.copy()
    ranked["candidate_rank"] = ranked.groupby("context_id", sort=False).cumcount() + 1
    oracle = targets.merge(
        ranked,
        left_on=["context_id", "site_object_id"],
        right_on=["context_id", "candidate_site_id"],
        how="left",
        validate="one_to_one",
        suffixes=("_true", "_oracle"),
    )
    if oracle["candidate_rank"].isna().any():
        raise SiteIdentificationError("True test site is missing from candidates")
    top1 = (
        ranked.loc[ranked["candidate_rank"].eq(1)]
        .drop(columns="candidate_rank")
        .rename(
            columns={
                "candidate_site_id": "automatic_candidate_site_id",
                "site_type": "automatic_site_type",
                "member_atom_indices_json": "automatic_members_json",
                "validity_logit": "automatic_validity_logit",
                "absolute_site_probability": ("automatic_absolute_site_probability"),
                "conditional_N_mean": "automatic_N_prediction",
                "conditional_N_std": "automatic_N_std",
            }
        )
    )
    results = oracle.merge(
        top1,
        on="context_id",
        how="left",
        validate="many_to_one",
    )
    results = results.rename(
        columns={
            "candidate_rank": "true_candidate_rank",
            "conditional_N_mean": "oracle_N_prediction",
            "conditional_N_std": "oracle_N_std",
            "validity_logit": "oracle_validity_logit",
            "absolute_site_probability": "oracle_absolute_site_probability",
            "member_atom_indices_json_true": "true_members_json",
        }
    )
    # Depending on pandas suffix resolution, the target membership can retain
    # its original name when the right side has a different suffix.
    if "true_members_json" not in results and "member_atom_indices_json" in results:
        results = results.rename(
            columns={"member_atom_indices_json": "true_members_json"}
        )
    context_target_count = results.groupby("context_id")["target_id"].transform(
        "nunique"
    )
    results["context_target_count"] = context_target_count
    results["single_target_context"] = context_target_count.eq(1)
    results["exact_top1"] = results["true_candidate_rank"].le(1)
    results["exact_top3"] = results["true_candidate_rank"].le(3)
    results["exact_top5"] = results["true_candidate_rank"].le(5)
    results["reciprocal_rank"] = 1.0 / results["true_candidate_rank"].astype(float)
    results["automatic_exact_match"] = (
        results["automatic_candidate_site_id"]
        .astype(str)
        .eq(results["site_object_id"].astype(str))
    )
    results["top1_membership_jaccard"] = [
        _membership_jaccard(automatic, truth)
        for automatic, truth in zip(
            results["automatic_members_json"],
            results["true_members_json"],
            strict=True,
        )
    ]
    group_or_region = results["site_type_true"].isin(
        ["atom_group", "delocalized_region"]
    )
    same_type = (
        results["automatic_site_type"]
        .astype(str)
        .eq(results["site_type_true"].astype(str))
    )
    results["compatible_top1"] = results["automatic_exact_match"] | (
        group_or_region
        & same_type
        & results["top1_membership_jaccard"].ge(compatible_jaccard_threshold)
    )
    return results


def _target_retrieval_summary(
    frame: pd.DataFrame,
    *,
    population: str,
) -> dict[str, object]:
    return {
        "population": population,
        "target_count": len(frame),
        "context_count": int(frame["context_id"].nunique()),
        "exact_top1_recall": float(frame["exact_top1"].mean()),
        "exact_top3_recall": float(frame["exact_top3"].mean()),
        "exact_top5_recall": float(frame["exact_top5"].mean()),
        "mrr": float(frame["reciprocal_rank"].mean()),
        "compatible_top1_recall": float(frame["compatible_top1"].mean()),
        "mean_top1_membership_jaccard": float(frame["top1_membership_jaccard"].mean()),
    }


def _multi_target_set_metrics(
    *,
    predictions: pd.DataFrame,
    targets: pd.DataFrame,
) -> dict[str, object]:
    multi_contexts = (
        targets.groupby("context_id")["site_object_id"]
        .nunique()
        .loc[lambda values: values > 1]
        .index
    )
    if not len(multi_contexts):
        return {
            "context_count": 0,
            "set_hit_at_1": float("nan"),
            "set_hit_at_3": float("nan"),
            "set_coverage_at_5": float("nan"),
        }
    ranked = predictions.sort_values(
        ["context_id", "validity_logit", "candidate_site_id"],
        ascending=[True, False, True],
        kind="stable",
    ).copy()
    ranked["rank"] = ranked.groupby("context_id", sort=False).cumcount() + 1
    hits1: list[float] = []
    hits3: list[float] = []
    coverage5: list[float] = []
    target_sets = (
        targets.loc[targets["context_id"].isin(multi_contexts)]
        .groupby("context_id")["site_object_id"]
        .agg(lambda values: set(map(str, values)))
    )
    for context_id, truth in target_sets.items():
        group = ranked.loc[ranked["context_id"].eq(context_id)]
        top1 = set(group.loc[group["rank"].le(1), "candidate_site_id"].astype(str))
        top3 = set(group.loc[group["rank"].le(3), "candidate_site_id"].astype(str))
        top5 = set(group.loc[group["rank"].le(5), "candidate_site_id"].astype(str))
        hits1.append(float(bool(truth & top1)))
        hits3.append(float(bool(truth & top3)))
        coverage5.append(len(truth & top5) / len(truth))
    return {
        "context_count": len(target_sets),
        "set_hit_at_1": float(np.mean(hits1)),
        "set_hit_at_3": float(np.mean(hits3)),
        "set_coverage_at_5": float(np.mean(coverage5)),
    }


def _bootstrap_oracle_gap(
    frame: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
    connectivity_values = frame["connectivity_id"].astype(str).to_numpy()
    connectivities = np.asarray(sorted(set(connectivity_values)), dtype=object)
    group_indices = [
        np.flatnonzero(connectivity_values == value) for value in connectivities
    ]
    truth = frame["N_mean"].to_numpy(dtype=float)
    oracle_prediction = frame["oracle_N_prediction"].to_numpy(dtype=float)
    automatic_prediction = frame["automatic_N_prediction"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    delta_r2: list[float] = []
    delta_mae: list[float] = []
    for _ in range(replicates):
        sampled_group_indices = rng.integers(
            0,
            len(group_indices),
            size=len(connectivities),
        )
        positions = np.concatenate(
            [group_indices[index] for index in sampled_group_indices]
        )
        oracle = _regression_metrics(
            truth[positions],
            oracle_prediction[positions],
        )
        automatic = _regression_metrics(
            truth[positions],
            automatic_prediction[positions],
        )
        if math.isfinite(float(oracle["r2"])) and math.isfinite(float(automatic["r2"])):
            delta_r2.append(float(automatic["r2"]) - float(oracle["r2"]))
        delta_mae.append(float(automatic["mae"]) - float(oracle["mae"]))
    return {
        "replicates_requested": replicates,
        "replicates_with_finite_r2": len(delta_r2),
        "automatic_minus_oracle_r2_ci_low": float(np.quantile(delta_r2, 0.025)),
        "automatic_minus_oracle_r2_ci_high": float(np.quantile(delta_r2, 0.975)),
        "automatic_minus_oracle_mae_ci_low": float(np.quantile(delta_mae, 0.025)),
        "automatic_minus_oracle_mae_ci_high": float(np.quantile(delta_mae, 0.975)),
    }


def run_test_evaluation(
    config: Mapping[str, Any],
    *,
    config_path: Path,
) -> dict[str, Any]:
    """Open sealed labels only after prediction files and hashes are frozen."""

    output_root = _repo_path(
        config["output_directory"],
        label="output directory",
    )
    preflight = output_root / "preflight"
    test_predictions = output_root / "test_predictions"
    prediction_summary = _load_json(test_predictions / "summary.json")
    if (
        prediction_summary.get("status") != "pass"
        or prediction_summary.get("test_predictions_frozen") is not True
        or prediction_summary.get("test_labels_read") is not False
    ):
        raise SiteIdentificationError("Test predictions are not safely frozen")
    prediction_manifest_sha = sha256_file(test_predictions / "run_manifest.json")
    target = output_root / "test_evaluation"
    evaluation_settings = config["evaluation"]

    def writer(staged: Path) -> dict[str, Any]:
        split_summaries: list[dict[str, object]] = []
        target_result_parts: list[pd.DataFrame] = []
        reviewed_parts: list[pd.DataFrame] = []
        exact_candidate_parts: list[pd.DataFrame] = []
        split_metric_rows: list[dict[str, object]] = []
        structured_exact_calibration = (
            config["ranker"].get("training_population")
            == "full_candidate_endpoint_retrieval"
        )
        for split_seed_raw in config["backbone"]["split_seeds"]:
            split_seed = int(split_seed_raw)
            prediction_path = (
                test_predictions
                / f"split-{split_seed}"
                / "candidate_predictions.parquet"
            )
            freeze_path = (
                test_predictions / f"split-{split_seed}" / "prediction_freeze.json"
            )
            freeze = _load_json(freeze_path)
            claimed_freeze_sha = str(freeze.pop("freeze_sha256"))
            if _canonical_sha256(freeze) != claimed_freeze_sha:
                raise SiteIdentificationError("Prediction freeze hash changed")
            if (
                freeze.get("test_labels_read") is not False
                or freeze.get("target_values_read") is not False
                or sha256_file(prediction_path) != freeze["candidate_prediction_sha256"]
            ):
                raise SiteIdentificationError("Frozen test prediction changed")
            predictions = pd.read_parquet(prediction_path)
            if (
                predictions["target_or_site_label_read"].astype(bool).any()
                or predictions["candidate_softmax_used"].astype(bool).any()
            ):
                raise SiteIdentificationError("Test prediction boundary changed")

            # The following two reads are intentionally below all prediction
            # integrity checks and occur only in this coordinator phase.
            sealed = preflight / "sealed" / f"split-{split_seed}"
            reviewed = pd.read_parquet(sealed / "reviewed_test_labels.parquet")
            target_labels = pd.read_parquet(sealed / "target_test_labels.parquet")
            reviewed_predictions = reviewed.merge(
                predictions[
                    [
                        "context_id",
                        "candidate_site_id",
                        "validity_logit",
                        "raw_sigmoid_score",
                        "absolute_site_probability",
                        "conditional_N_mean",
                        "conditional_N_std",
                    ]
                ],
                on=["context_id", "candidate_site_id"],
                how="left",
                validate="many_to_one",
            )
            if reviewed_predictions["validity_logit"].isna().any():
                raise SiteIdentificationError(
                    "Reviewed test candidate lacks frozen prediction"
                )
            reviewed_weights = _review_population_weights(reviewed_predictions)
            reviewed_predictions["evaluation_weight"] = reviewed_weights.numpy()
            calibration_metrics = [
                _binary_metrics(
                    reviewed_predictions,
                    probability_column="absolute_site_probability",
                    slice_name="overall",
                )
            ]
            calibration_metrics.extend(
                _binary_metrics(
                    group,
                    probability_column="absolute_site_probability",
                    slice_name=str(site_type),
                )
                for site_type, group in reviewed_predictions.groupby(
                    "site_type",
                    sort=True,
                )
            )
            reviewed_retrieval = retrieval_metrics(
                labels=reviewed_predictions["validity_label"].astype(int),
                scores=reviewed_predictions["validity_logit"].astype(float),
                target_ids=reviewed_predictions["target_id"].astype(str),
            )

            exact_calibration_metrics: list[dict[str, object]] = []
            exact_candidate_predictions: pd.DataFrame | None = None
            if structured_exact_calibration:
                exact_ids_by_context = (
                    target_labels.groupby("context_id")["site_object_id"]
                    .agg(lambda values: set(map(str, values)))
                    .to_dict()
                )
                exact_candidate_predictions = predictions.copy()
                exact_candidate_predictions["validity_label"] = [
                    int(str(candidate_id) in exact_ids_by_context[str(context_id)])
                    for context_id, candidate_id in zip(
                        exact_candidate_predictions["context_id"],
                        exact_candidate_predictions["candidate_site_id"],
                        strict=True,
                    )
                ]
                candidate_counts = exact_candidate_predictions.groupby("context_id")[
                    "candidate_site_id"
                ].transform("count")
                exact_candidate_predictions["evaluation_weight"] = (
                    1.0
                    / candidate_counts.to_numpy(dtype=float)
                    / exact_candidate_predictions["context_id"].nunique()
                )
                exact_calibration_metrics = [
                    _binary_metrics(
                        exact_candidate_predictions,
                        probability_column="absolute_site_probability",
                        slice_name="exact_fullspace:overall",
                    )
                ]
                exact_calibration_metrics.extend(
                    _binary_metrics(
                        group,
                        probability_column="absolute_site_probability",
                        slice_name=f"exact_fullspace:{site_type}",
                    )
                    for site_type, group in exact_candidate_predictions.groupby(
                        "site_type",
                        sort=True,
                    )
                )

            target_results = _full_space_target_results(
                predictions=predictions,
                targets=target_labels,
                compatible_jaccard_threshold=float(
                    evaluation_settings["membership_compatible_jaccard_threshold"]
                ),
            )
            target_results["split_seed"] = split_seed
            reviewed_predictions["split_seed"] = split_seed
            margin_threshold: float | None = None
            if structured_exact_calibration:
                development_checkpoint_path = (
                    output_root
                    / "development"
                    / f"split-{split_seed}"
                    / "ranker_checkpoint.pt"
                )
                checkpoint = _load_ranker_checkpoint(
                    development_checkpoint_path,
                    split_seed=split_seed,
                    config=config,
                )
                margin_payload = checkpoint.get("margin_abstention")
                if not isinstance(margin_payload, Mapping):
                    raise SiteIdentificationError(
                        "Structured checkpoint lacks frozen margin gate"
                    )
                margin_threshold = float(margin_payload["selected_threshold"])
                ranked_for_margin = predictions.sort_values(
                    ["context_id", "validity_logit", "candidate_site_id"],
                    ascending=[True, False, True],
                    kind="stable",
                ).copy()
                ranked_for_margin["_margin_rank"] = ranked_for_margin.groupby(
                    "context_id",
                    sort=False,
                ).cumcount()
                top_two = ranked_for_margin.loc[
                    ranked_for_margin["_margin_rank"].le(1),
                    ["context_id", "_margin_rank", "validity_logit"],
                ]
                margin_wide = top_two.pivot(
                    index="context_id",
                    columns="_margin_rank",
                    values="validity_logit",
                )
                margin_values = (
                    margin_wide[0] - margin_wide.get(1, float("-inf"))
                ).rename("automatic_top1_top2_margin")
                target_results = target_results.merge(
                    margin_values,
                    on="context_id",
                    how="left",
                    validate="many_to_one",
                )
                target_results["automatic_prediction_accepted"] = target_results[
                    "automatic_top1_top2_margin"
                ].ge(margin_threshold)
            single = target_results.loc[target_results["single_target_context"]].copy()
            multi = target_results.loc[~target_results["single_target_context"]].copy()
            if single.empty:
                raise SiteIdentificationError(
                    "Primary single-target test population is empty"
                )
            oracle_metrics = _regression_metrics(
                single["N_mean"].to_numpy(dtype=float),
                single["oracle_N_prediction"].to_numpy(dtype=float),
            )
            automatic_metrics = _regression_metrics(
                single["N_mean"].to_numpy(dtype=float),
                single["automatic_N_prediction"].to_numpy(dtype=float),
            )
            selective_metrics: dict[str, object] | None = None
            selective_n_metrics: dict[str, float | int] | None = None
            if structured_exact_calibration:
                accepted = single.loc[
                    single["automatic_prediction_accepted"].astype(bool)
                ]
                selective_metrics = {
                    "margin_threshold": margin_threshold,
                    "accepted_count": len(accepted),
                    "coverage": float(len(accepted) / len(single)),
                    "exact_top1_precision": (
                        float(accepted["exact_top1"].mean())
                        if len(accepted)
                        else float("nan")
                    ),
                    "compatible_top1_precision": (
                        float(accepted["compatible_top1"].mean())
                        if len(accepted)
                        else float("nan")
                    ),
                }
                selective_n_metrics = _regression_metrics(
                    accepted["N_mean"].to_numpy(dtype=float),
                    accepted["automatic_N_prediction"].to_numpy(dtype=float),
                )
            retrieval_overall = _target_retrieval_summary(
                target_results,
                population="all_test_targets",
            )
            retrieval_single = _target_retrieval_summary(
                single,
                population="single_target_test_contexts",
            )
            retrieval_by_type = [
                _target_retrieval_summary(
                    group,
                    population=f"single_target:{site_type}",
                )
                for site_type, group in single.groupby(
                    "site_type_true",
                    sort=True,
                )
            ]
            multi_set = _multi_target_set_metrics(
                predictions=predictions,
                targets=target_labels,
            )
            bootstrap = _bootstrap_oracle_gap(
                single,
                replicates=int(evaluation_settings["bootstrap_replicates"]),
                seed=int(evaluation_settings["bootstrap_seed"]) + split_seed,
            )

            split_dir = staged / f"split-{split_seed}"
            split_dir.mkdir()
            reviewed_predictions.to_parquet(
                split_dir / "reviewed_test_predictions.parquet",
                index=False,
                compression="zstd",
            )
            target_results.to_parquet(
                split_dir / "target_level_results.parquet",
                index=False,
                compression="zstd",
            )
            pd.DataFrame(calibration_metrics).to_csv(
                split_dir / "calibration_metrics_by_type.csv",
                index=False,
            )
            if structured_exact_calibration:
                pd.DataFrame(exact_calibration_metrics).to_csv(
                    split_dir / "exact_fullspace_calibration_metrics_by_type.csv",
                    index=False,
                )
                assert exact_candidate_predictions is not None
                exact_candidate_predictions["split_seed"] = split_seed
                exact_candidate_predictions.to_parquet(
                    split_dir / "exact_fullspace_candidate_predictions.parquet",
                    index=False,
                    compression="zstd",
                )
            pd.DataFrame(
                [retrieval_overall, retrieval_single, *retrieval_by_type]
            ).to_csv(
                split_dir / "full_space_retrieval_metrics.csv",
                index=False,
            )
            split_summary = {
                "schema_version": "nucpred.mayr-site-id-split-test.v1",
                "split_seed": split_seed,
                "prediction_freeze_sha256": claimed_freeze_sha,
                "reviewed_calibration_metrics": calibration_metrics,
                "reviewed_calibration_semantics": (
                    "compatible_proxy_diagnostic_not_exact_calibrator_population"
                    if structured_exact_calibration
                    else "reviewed_endpoint_relative_calibrator_population"
                ),
                "exact_fullspace_calibration_metrics": exact_calibration_metrics,
                "reviewed_retrieval": reviewed_retrieval,
                "full_space_retrieval_overall": retrieval_overall,
                "full_space_retrieval_single_target": retrieval_single,
                "full_space_retrieval_by_true_type": retrieval_by_type,
                "multi_target_set_metrics": multi_set,
                "single_target_count": len(single),
                "multi_target_count": len(multi),
                "oracle_conditional_n": oracle_metrics,
                "automatic_top1_conditional_n": automatic_metrics,
                "margin_selective_retrieval": selective_metrics,
                "margin_selective_conditional_n": selective_n_metrics,
                "automatic_minus_oracle": {
                    "mae": float(automatic_metrics["mae"])
                    - float(oracle_metrics["mae"]),
                    "rmse": float(automatic_metrics["rmse"])
                    - float(oracle_metrics["rmse"]),
                    "r2": float(automatic_metrics["r2"]) - float(oracle_metrics["r2"]),
                },
                "oracle_gap_connectivity_bootstrap": bootstrap,
                "test_labels_read_after_prediction_freeze": True,
            }
            atomic_write_json(split_dir / "summary.json", split_summary)
            split_summaries.append(split_summary)
            target_result_parts.append(target_results)
            reviewed_parts.append(reviewed_predictions)
            if exact_candidate_predictions is not None:
                exact_candidate_parts.append(exact_candidate_predictions)
            split_metric_rows.append(
                {
                    "split_seed": split_seed,
                    "single_target_count": len(single),
                    "exact_top1_recall": retrieval_single["exact_top1_recall"],
                    "exact_top3_recall": retrieval_single["exact_top3_recall"],
                    "exact_top5_recall": retrieval_single["exact_top5_recall"],
                    "mrr": retrieval_single["mrr"],
                    "compatible_top1_recall": retrieval_single[
                        "compatible_top1_recall"
                    ],
                    "oracle_mae": oracle_metrics["mae"],
                    "oracle_rmse": oracle_metrics["rmse"],
                    "oracle_r2": oracle_metrics["r2"],
                    "automatic_mae": automatic_metrics["mae"],
                    "automatic_rmse": automatic_metrics["rmse"],
                    "automatic_r2": automatic_metrics["r2"],
                    "automatic_minus_oracle_r2": (
                        float(automatic_metrics["r2"]) - float(oracle_metrics["r2"])
                    ),
                    "selective_coverage": (
                        selective_metrics["coverage"]
                        if selective_metrics is not None
                        else float("nan")
                    ),
                    "selective_exact_top1_precision": (
                        selective_metrics["exact_top1_precision"]
                        if selective_metrics is not None
                        else float("nan")
                    ),
                    "selective_automatic_r2": (
                        selective_n_metrics["r2"]
                        if selective_n_metrics is not None
                        else float("nan")
                    ),
                }
            )

        all_targets = pd.concat(target_result_parts, ignore_index=True)
        all_reviewed = pd.concat(reviewed_parts, ignore_index=True)
        all_exact_candidates = (
            pd.concat(exact_candidate_parts, ignore_index=True)
            if exact_candidate_parts
            else None
        )
        metric_frame = pd.DataFrame(split_metric_rows)
        metric_frame.to_csv(staged / "split_metrics.csv", index=False)
        all_targets.to_parquet(
            staged / "all_split_target_level_results.parquet",
            index=False,
            compression="zstd",
        )
        all_reviewed.to_parquet(
            staged / "all_split_reviewed_test_predictions.parquet",
            index=False,
            compression="zstd",
        )
        macro: dict[str, dict[str, float]] = {}
        metric_columns = [
            column
            for column in metric_frame.columns
            if column not in {"split_seed", "single_target_count"}
        ]
        for column in metric_columns:
            values = metric_frame[column].to_numpy(dtype=float)
            macro[column] = {
                "mean": float(np.mean(values)),
                "std_population": float(np.std(values)),
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
            }
        pooled_single = all_targets.loc[all_targets["single_target_context"]].copy()
        pooled_retrieval_by_type = [
            _target_retrieval_summary(
                group,
                population=f"pooled_split_records:{site_type}",
            )
            for site_type, group in pooled_single.groupby(
                "site_type_true",
                sort=True,
            )
        ]
        pooled_calibration = [
            _binary_metrics(
                all_reviewed,
                probability_column="absolute_site_probability",
                slice_name="pooled_split_records:overall",
            )
        ]
        pooled_calibration.extend(
            _binary_metrics(
                group,
                probability_column="absolute_site_probability",
                slice_name=f"pooled_split_records:{site_type}",
            )
            for site_type, group in all_reviewed.groupby(
                "site_type",
                sort=True,
            )
        )
        pooled_exact_calibration: list[dict[str, object]] = []
        if all_exact_candidates is not None:
            all_exact_candidates.to_parquet(
                staged / "all_split_exact_fullspace_candidate_predictions.parquet",
                index=False,
                compression="zstd",
            )
            pooled_exact_calibration = [
                _binary_metrics(
                    all_exact_candidates,
                    probability_column="absolute_site_probability",
                    slice_name="pooled_exact_fullspace:overall",
                )
            ]
            pooled_exact_calibration.extend(
                _binary_metrics(
                    group,
                    probability_column="absolute_site_probability",
                    slice_name=f"pooled_exact_fullspace:{site_type}",
                )
                for site_type, group in all_exact_candidates.groupby(
                    "site_type",
                    sort=True,
                )
            )
            pd.DataFrame(pooled_exact_calibration).to_csv(
                staged / "pooled_exact_fullspace_calibration_by_type.csv",
                index=False,
            )
        pd.DataFrame(pooled_retrieval_by_type).to_csv(
            staged / "pooled_retrieval_by_type.csv",
            index=False,
        )
        pd.DataFrame(pooled_calibration).to_csv(
            staged / "pooled_calibration_by_type.csv",
            index=False,
        )
        return {
            "schema_version": TEST_EVALUATION_SCHEMA,
            "status": "pass",
            "campaign_id": config["campaign_id"],
            "config_sha256": sha256_file(config_path),
            "prediction_manifest_sha256_before_label_read": (prediction_manifest_sha),
            "split_summaries": split_summaries,
            "macro_split_metrics": macro,
            "pooled_split_record_retrieval_by_true_type": (pooled_retrieval_by_type),
            "pooled_split_record_calibration": pooled_calibration,
            "pooled_exact_fullspace_calibration": pooled_exact_calibration,
            "five_split_test_complete": True,
            "test_labels_read_after_prediction_freeze": True,
            "conditional_n_backbone_frozen": True,
            "unknown_as_negative_count": 0,
            "candidate_softmax_used": False,
            "final_refit_performed": False,
            "calibration_weighting": evaluation_settings["calibration_weighting"],
            "evaluation_status": evaluation_settings.get(
                "evaluation_status",
                "confirmatory",
            ),
            "prior_test_results_informed_architecture": evaluation_settings.get(
                "prior_v5_test_results_informed_architecture",
                False,
            ),
        }

    return _publish_stage(
        target,
        schema_version=TEST_EVALUATION_SCHEMA,
        writer=writer,
    )


def run_deployment_registry(
    config: Mapping[str, Any],
    *,
    config_path: Path,
) -> dict[str, Any]:
    """Register the five cross-fit heads without training a final refit."""

    output_root = _repo_path(
        config["output_directory"],
        label="output directory",
    )
    development = output_root / "development"
    evaluation = output_root / "test_evaluation"
    evaluation_summary = _load_json(evaluation / "summary.json")
    if (
        evaluation_summary.get("status") != "pass"
        or evaluation_summary.get("five_split_test_complete") is not True
    ):
        raise SiteIdentificationError("Formal test evaluation is incomplete")
    target = output_root / "deployment"

    def writer(staged: Path) -> dict[str, Any]:
        split_models: list[dict[str, object]] = []
        margin_thresholds: list[float] = []
        for split_seed_raw in config["backbone"]["split_seeds"]:
            split_seed = int(split_seed_raw)
            ranker_path = development / f"split-{split_seed}" / "ranker_checkpoint.pt"
            checkpoint = _load_ranker_checkpoint(
                ranker_path,
                split_seed=split_seed,
                config=config,
            )
            backbone_bindings = checkpoint["backbone_bindings"]
            if not isinstance(backbone_bindings, list):
                raise SiteIdentificationError("Backbone bindings are missing")
            for binding in backbone_bindings:
                if not isinstance(binding, Mapping):
                    raise SiteIdentificationError("Invalid backbone binding")
                path = _repo_path(
                    binding["path"],
                    label="registered backbone checkpoint",
                )
                _verify_sha(
                    path,
                    binding["sha256"],
                    label="registered backbone checkpoint",
                )
            split_model: dict[str, object] = {
                "split_seed": split_seed,
                "ranker_checkpoint_path": _display_path(ranker_path),
                "ranker_checkpoint_sha256": sha256_file(ranker_path),
                "selected_arm": checkpoint["selected_arm"],
                "backbone_bindings": backbone_bindings,
            }
            margin_payload = checkpoint.get("margin_abstention")
            if margin_payload is not None:
                if not isinstance(margin_payload, Mapping):
                    raise SiteIdentificationError(
                        "Checkpoint margin abstention payload is invalid"
                    )
                threshold = float(margin_payload["selected_threshold"])
                margin_thresholds.append(threshold)
                split_model["margin_abstention"] = margin_payload
            split_models.append(split_model)
        margin_enabled = bool(config["runtime"].get("margin_abstention_enabled"))
        if margin_enabled and len(margin_thresholds) != len(split_models):
            raise SiteIdentificationError(
                "Runtime margin gate lacks one or more split thresholds"
            )
        registry: dict[str, object] = {
            "schema_version": RUNTIME_REGISTRY_SCHEMA,
            "campaign_id": config["campaign_id"],
            "created_at_utc": _utc_now(),
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
            "validity_ensemble_semantics": ("mean_of_five_cross_fit_split_logits"),
            "probability_ensemble_semantics": (
                "mean_of_five_split_specific_calibrated_probabilities"
            ),
            "conditional_n_ensemble_semantics": (
                "mean_of_fifteen_frozen_stage_e_c_predictions"
            ),
            "formal_test_summary_path": _display_path(evaluation / "summary.json"),
            "formal_test_summary_sha256": sha256_file(evaluation / "summary.json"),
            "response_schema_path": config["contract"]["response_schema_path"],
            "response_schema_sha256": config["contract"]["response_schema_sha256"],
            "candidate_generator_path": config["contract"]["candidate_generator_path"],
            "candidate_generator_sha256": config["contract"][
                "candidate_generator_sha256"
            ],
            "conditional_n_backbone_frozen": True,
            "final_refit_performed": False,
            "target_or_site_label_read_at_inference": False,
            "candidate_scores_independent": True,
            "candidate_softmax_used": False,
            "no_site_claim_permitted": False,
            "margin_abstention_enabled": margin_enabled,
            "margin_threshold_aggregation": (
                config.get("abstention", {}).get("runtime_threshold_aggregation")
                if margin_enabled
                else None
            ),
            "runtime_margin_threshold": (
                float(np.median(margin_thresholds)) if margin_enabled else None
            ),
            "low_margin_runtime_status": (
                config.get("abstention", {}).get("low_margin_runtime_status")
                if margin_enabled
                else None
            ),
        }
        registry["registry_sha256"] = _canonical_sha256(registry)
        atomic_write_json(staged / "runtime_registry.json", registry)
        return {
            "schema_version": "nucpred.mayr-site-id-deployment.v1",
            "status": "pass",
            "campaign_id": config["campaign_id"],
            "config_sha256": sha256_file(config_path),
            "runtime_registry_path": _display_path(
                output_root / "deployment" / "runtime_registry.json"
            ),
            "runtime_registry_sha256": sha256_file(staged / "runtime_registry.json"),
            "registered_split_model_count": len(split_models),
            "registered_backbone_checkpoint_count": sum(
                len(item["backbone_bindings"]) for item in split_models
            ),
            "formal_test_summary_sha256": registry["formal_test_summary_sha256"],
            "final_refit_performed": False,
            "target_or_site_label_read_at_inference": False,
            "candidate_softmax_used": False,
            "margin_abstention_enabled": margin_enabled,
            "runtime_margin_threshold": registry["runtime_margin_threshold"],
        }

    return _publish_stage(
        target,
        schema_version="nucpred.mayr-site-id-deployment.v1",
        writer=writer,
    )


def run_all(
    config: Mapping[str, Any],
    *,
    config_path: Path,
) -> dict[str, Any]:
    preflight = run_preflight(config, config_path=config_path)
    development = run_development(config, config_path=config_path)
    predictions = run_test_predictions(config, config_path=config_path)
    evaluation = run_test_evaluation(config, config_path=config_path)
    deployment = run_deployment_registry(config, config_path=config_path)
    return {
        "status": "pass",
        "preflight": preflight,
        "development": development,
        "test_predictions": predictions,
        "test_evaluation": evaluation,
        "deployment": deployment,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run formal Mayr multitype site identification",
    )
    parser.add_argument(
        "command",
        choices=[
            "preflight",
            "develop",
            "predict-test",
            "test",
            "deploy",
            "all",
        ],
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = args.config.resolve()
    config = read_config(config_path)
    runners = {
        "preflight": run_preflight,
        "develop": run_development,
        "predict-test": run_test_predictions,
        "test": run_test_evaluation,
        "deploy": run_deployment_registry,
        "all": run_all,
    }
    result = runners[args.command](config, config_path=config_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
