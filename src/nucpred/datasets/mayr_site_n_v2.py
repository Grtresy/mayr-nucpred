"""Build the corrected, publication-scoped Mayr site-N v2 dataset.

The v1 dataset remains immutable.  This module applies only frozen Stage-B
endpoint decisions: exact re-projections retain their measured N values under
the corrected physical site, while unresolved targets lose all N/site
supervision.  It also replaces repeated random holdouts with a mutually
exclusive outer connectivity partition and nested inner folds.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import shutil
import tempfile
import tomllib
from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from nucpred.core.files import atomic_write_json, sha256_file
from nucpred.datasets import mayr_site_n as v1
from nucpred.project import get_project_layout


ROOT = get_project_layout().root
DEFAULT_CONFIG = ROOT / "configs/mayr_site_n_v2.toml"
CONFIG_SCHEMA = "nucpred.mayr-site-n-v2-config.v1"
DATASET_SCHEMA = "nucpred.mayr-site-n-dataset.v2"


class SiteNV2DatasetError(RuntimeError):
    """Raised when frozen evidence cannot produce the v2 dataset safely."""


def _project_path(value: object) -> Path:
    path = (ROOT / str(value)).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as exc:
        raise SiteNV2DatasetError(f"Path escapes project: {value}") from exc
    return path


def _read_config(path: str | Path = DEFAULT_CONFIG) -> tuple[dict[str, Any], Path]:
    resolved = Path(path).resolve()
    config = tomllib.loads(resolved.read_text(encoding="utf-8"))
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise SiteNV2DatasetError("Unsupported Mayr site-N v2 config schema")
    policy = config["policy"]
    if policy.get("parent_dataset_is_immutable") is not True:
        raise SiteNV2DatasetError("The parent dataset must remain immutable")
    if policy.get("unknown_is_negative") is not False:
        raise SiteNV2DatasetError("Unknown targets cannot become negatives")
    if policy.get("unresolved_enters_n_supervision") is not False:
        raise SiteNV2DatasetError("Unresolved targets cannot enter N supervision")
    if policy.get("unresolved_enters_site_supervision") is not False:
        raise SiteNV2DatasetError("Unresolved targets cannot enter site supervision")
    if config["evidence"].get("stage_e_a_site_only_changes_n_target") is not False:
        raise SiteNV2DatasetError("Site-only evidence cannot rewrite an N target")
    return config, resolved


def _verify_hash(path: Path, expected: object, *, label: str) -> str:
    if not path.is_file():
        raise SiteNV2DatasetError(f"Missing {label}: {path}")
    observed = sha256_file(path)
    if observed != str(expected):
        raise SiteNV2DatasetError(f"Frozen {label} drifted: {observed} != {expected}")
    return observed


def _load_inputs(config: Mapping[str, Any]) -> dict[str, object]:
    publication_path = _project_path(config["publication_config_path"])
    _verify_hash(
        publication_path,
        config["publication_config_sha256"],
        label="publication config",
    )
    parent = config["parent"]
    root = _project_path(parent["directory"])
    files = {
        "manifest": root / "dataset_manifest.json",
        "species": root / "species.parquet",
        "contexts": root / "contexts.parquet",
        "sites": root / "sites.parquet",
        "measurements": root / "measurements.parquet",
        "targets": root / "targets.parquet",
        "candidate_sites": root / "candidate_sites.parquet",
        "context_feature_audit": root / "context_feature_audit.csv",
        "pretraining_overlap_audit": root / "pretraining_overlap_audit.csv",
    }
    hashes = {
        name: _verify_hash(path, parent[f"{name}_sha256"], label=f"parent {name}")
        for name, path in files.items()
    }
    evidence = config["evidence"]
    projection_path = _project_path(evidence["stage_b_target_projections_path"])
    hashes["stage_b_target_projections"] = _verify_hash(
        projection_path,
        evidence["stage_b_target_projections_sha256"],
        label="Stage-B target projections",
    )
    return {
        "paths": files,
        "hashes": hashes,
        "species": pd.read_parquet(files["species"]),
        "contexts": pd.read_parquet(files["contexts"]),
        "sites": pd.read_parquet(files["sites"]),
        "measurements": pd.read_parquet(files["measurements"]),
        "targets": pd.read_parquet(files["targets"]),
        "candidates": pd.read_parquet(files["candidate_sites"]),
        "context_feature_audit": pd.read_csv(files["context_feature_audit"]),
        "pretraining_overlap_audit": pd.read_csv(files["pretraining_overlap_audit"]),
        "projections": pd.read_parquet(projection_path),
    }


def _source_ids(value: object) -> list[str]:
    parsed = json.loads(str(value))
    if not isinstance(parsed, list):
        raise SiteNV2DatasetError("source_ids_json must contain a list")
    return sorted(str(item) for item in parsed)


def _resolution(site_type: str) -> str:
    return {
        "atom": "exact",
        "bond": "exact",
        "delocalized_region": "collective",
        "atom_group": "equivalent_or_collective",
        "transferable_h_group": "equivalent_or_indistinguishable",
    }[site_type]


def _projection_counts(
    projections: pd.DataFrame, config: Mapping[str, Any]
) -> dict[str, int]:
    evidence = config["evidence"]
    counts = {
        "exact_current": int(
            projections["projection_status"].eq("exact_current_candidate").sum()
        ),
        "exact_reprojected": int(
            projections["projection_status"].eq("exact_reprojected_candidate").sum()
        ),
        "unresolved": int(
            projections["projection_status"]
            .eq("unknown_nonprimary_or_mixed_evidence")
            .sum()
        ),
        "formal_positive": int(projections["formal_positive_eligible"].sum()),
    }
    for name, value in counts.items():
        expected = int(evidence[f"{name}_expected"])
        if value != expected:
            raise SiteNV2DatasetError(
                f"Stage-B {name} count changed: {value} != {expected}"
            )
    return counts


def _adjudicate_measurements(
    *,
    measurements: pd.DataFrame,
    targets: pd.DataFrame,
    candidates: pd.DataFrame,
    projections: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = measurements.copy()
    result["publication_parent_target_id"] = ""
    result["publication_adjudication_status"] = "parent_unresolved_measurement"
    target_index = targets.set_index("target_id", drop=False)
    candidate_index = candidates.set_index("candidate_site_id", drop=False)
    source_owner: dict[str, str] = {}
    for target in targets.itertuples(index=False):
        for source_id in _source_ids(target.source_ids_json):
            previous = source_owner.setdefault(source_id, str(target.target_id))
            if previous != str(target.target_id):
                raise SiteNV2DatasetError(
                    f"Measurement {source_id} belongs to multiple parent targets"
                )
    owner = result["source_id"].astype(str).map(source_owner).fillna("")
    result["publication_parent_target_id"] = owner
    result.loc[owner.ne(""), "publication_adjudication_status"] = "parent_unchanged"

    audit_rows: list[dict[str, object]] = []
    seen_targets: set[str] = set()
    for projection in projections.to_dict("records"):
        target_id = str(projection["target_id"])
        if target_id in seen_targets or target_id not in target_index.index:
            raise SiteNV2DatasetError(f"Invalid Stage-B target identity: {target_id}")
        seen_targets.add(target_id)
        target = target_index.loc[target_id]
        identifiers = ("context_id", "species_id", "connectivity_id")
        if any(str(target[key]) != str(projection[key]) for key in identifiers):
            raise SiteNV2DatasetError(f"Stage-B identity drifted: {target_id}")
        if abs(float(target["N_mean"]) - float(projection["N_mean"])) > 1e-12:
            raise SiteNV2DatasetError(f"Stage-B N value drifted: {target_id}")
        source_ids = _source_ids(target["source_ids_json"])
        selected = (
            result["source_id"].astype(str).isin(source_ids)
            & result["context_id"].astype(str).eq(str(target["context_id"]))
            & result["site_object_id"].astype(str).eq(str(target["site_object_id"]))
        )
        if sorted(result.loc[selected, "source_id"].astype(str)) != source_ids:
            raise SiteNV2DatasetError(
                f"Stage-B target does not resolve exact measurements: {target_id}"
            )
        status = str(projection["projection_status"])
        if status == "unknown_nonprimary_or_mixed_evidence":
            if bool(projection["formal_positive_eligible"]):
                raise SiteNV2DatasetError("Unresolved target became formally positive")
            result.loc[selected, "site_object_id"] = ""
            result.loc[selected, "measurement_training_eligible"] = False
            result.loc[selected, "measurement_site_type"] = v1.UNRESOLVED_SITE_TYPE
            adjudication = "excluded_unresolved_primary_evidence"
            new_site_id = ""
        elif status in {"exact_current_candidate", "exact_reprojected_candidate"}:
            if not bool(projection["formal_positive_eligible"]):
                raise SiteNV2DatasetError(
                    "Exact Stage-B target is not formally positive"
                )
            new_site_id = str(projection["projected_candidate_id"])
            if new_site_id not in candidate_index.index:
                raise SiteNV2DatasetError(
                    f"Projected candidate is absent: {target_id}: {new_site_id}"
                )
            candidate = candidate_index.loc[new_site_id]
            comparable = {
                "species_id": str(projection["species_id"]),
                "site_type": str(projection["projected_endpoint_family"]),
                "member_atom_indices_json": str(
                    projection["projected_member_atom_indices_json"]
                ),
                "member_bond_pairs_json": str(
                    projection["projected_member_bond_pairs_json"]
                ),
            }
            if any(str(candidate[key]) != value for key, value in comparable.items()):
                raise SiteNV2DatasetError(f"Projected candidate drifted: {target_id}")
            result.loc[selected, "site_object_id"] = new_site_id
            result.loc[selected, "measurement_training_eligible"] = True
            result.loc[selected, "measurement_site_type"] = comparable["site_type"]
            adjudication = (
                "retained_exact_current_primary_endpoint"
                if status == "exact_current_candidate"
                else "corrected_exact_reprojected_primary_endpoint"
            )
        else:
            raise SiteNV2DatasetError(f"Unsupported Stage-B status: {status}")
        result.loc[selected, "publication_adjudication_status"] = adjudication
        audit_rows.append(
            {
                "parent_target_id": target_id,
                "context_id": str(target["context_id"]),
                "species_id": str(target["species_id"]),
                "connectivity_id": str(target["connectivity_id"]),
                "parent_site_object_id": str(target["site_object_id"]),
                "parent_site_type": str(target["site_type"]),
                "v2_site_object_id": new_site_id,
                "v2_site_type": (
                    ""
                    if not new_site_id
                    else str(projection["projected_endpoint_family"])
                ),
                "stage_b_projection_status": status,
                "publication_adjudication_status": adjudication,
                "source_ids_json": str(target["source_ids_json"]),
                "N_mean_preserved": (
                    None if not new_site_id else float(target["N_mean"])
                ),
                "formal_n_supervision_eligible": bool(new_site_id),
                "unknown_is_negative": False,
            }
        )
    if len(seen_targets) != len(projections):
        raise SiteNV2DatasetError("Stage-B projection coverage changed")
    return result, pd.DataFrame(audit_rows).sort_values("parent_target_id")


def _rebuild_sites(
    *,
    measurements: pd.DataFrame,
    parent_sites: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    eligible = measurements.loc[
        measurements["measurement_training_eligible"].astype(bool)
        & measurements["site_object_id"].astype(str).ne("")
    ].copy()
    required_ids = sorted(eligible["site_object_id"].astype(str).unique())
    parent_index = parent_sites.set_index("site_object_id", drop=False)
    candidate_index = candidates.set_index("candidate_site_id", drop=False)
    rows: list[dict[str, object]] = []
    for site_id in required_ids:
        source_ids = sorted(
            eligible.loc[
                eligible["site_object_id"].astype(str).eq(site_id), "source_id"
            ].astype(str)
        )
        if site_id in parent_index.index:
            row = parent_index.loc[site_id].to_dict()
        elif site_id in candidate_index.index:
            candidate = candidate_index.loc[site_id]
            site_type = str(candidate["site_type"])
            row = {
                "schema_version": "nucpred.mayr-site-object.v2",
                "site_object_id": site_id,
                "species_id": str(candidate["species_id"]),
                "site_type": site_type,
                "physical_scope": site_type,
                "assignment_resolution": _resolution(site_type),
                "member_atom_indices_json": str(candidate["member_atom_indices_json"]),
                "member_bond_pairs_json": str(candidate["member_bond_pairs_json"]),
                "member_atom_count": int(candidate["member_atom_count"]),
                "source_supervision_level": (
                    "primary_source_exact_endpoint_reprojection"
                ),
                "formal_supervision_eligible": True,
            }
        else:
            raise SiteNV2DatasetError(f"Eligible site is not a candidate: {site_id}")
        row["measurement_source_ids_json"] = json.dumps(
            source_ids, separators=(",", ":")
        )
        rows.append(row)
    sites = pd.DataFrame(rows).sort_values(["species_id", "site_object_id"])
    if set(sites["site_object_id"].astype(str)) != set(required_ids):
        raise SiteNV2DatasetError("Rebuilt sites do not cover eligible measurements")
    return sites.reset_index(drop=True)


def _target_lineage(
    *,
    parent_targets: pd.DataFrame,
    measurements: pd.DataFrame,
    v2_targets: pd.DataFrame,
    adjudication: pd.DataFrame,
) -> pd.DataFrame:
    adjudication_index = adjudication.set_index("parent_target_id", drop=False)
    v2_lookup = v2_targets.set_index(["context_id", "site_object_id"], drop=False)
    rows: list[dict[str, object]] = []
    for target in parent_targets.to_dict("records"):
        parent_id = str(target["target_id"])
        if parent_id in adjudication_index.index:
            decision = adjudication_index.loc[parent_id]
            site_id = str(decision["v2_site_object_id"])
            status = str(decision["publication_adjudication_status"])
        else:
            site_id = str(target["site_object_id"])
            status = "parent_unchanged_outside_stage_b_review"
        if not site_id:
            v2_target_id = ""
            v2_site_type = ""
            eligible = False
        else:
            key = (str(target["context_id"]), site_id)
            if key not in v2_lookup.index:
                raise SiteNV2DatasetError(f"No v2 target for parent target {parent_id}")
            v2_target = v2_lookup.loc[key]
            v2_target_id = str(v2_target["target_id"])
            v2_site_type = str(v2_target["site_type"])
            eligible = True
            if abs(float(v2_target["N_mean"]) - float(target["N_mean"])) > 1e-12:
                raise SiteNV2DatasetError(
                    f"N changed during target rewrite: {parent_id}"
                )
        rows.append(
            {
                "parent_target_id": parent_id,
                "v2_target_id": v2_target_id,
                "context_id": str(target["context_id"]),
                "connectivity_id": str(target["connectivity_id"]),
                "parent_site_object_id": str(target["site_object_id"]),
                "v2_site_object_id": site_id,
                "parent_site_type": str(target["site_type"]),
                "v2_site_type": v2_site_type,
                "N_mean": float(target["N_mean"]),
                "publication_adjudication_status": status,
                "formal_training_eligible": eligible,
            }
        )
    lineage = pd.DataFrame(rows).sort_values("parent_target_id").reset_index(drop=True)
    if lineage["parent_target_id"].duplicated().any():
        raise SiteNV2DatasetError("Parent target lineage is not one-to-one")
    return lineage


def _n_bin(values: pd.Series) -> pd.Series:
    return pd.cut(
        values.astype(float),
        bins=[-float("inf"), 0.0, 5.0, 10.0, 15.0, float("inf")],
        labels=["N<0", "0<=N<5", "5<=N<10", "10<=N<15", "N>=15"],
        right=False,
    ).astype(str)


def _nested_splits(
    targets: pd.DataFrame,
    *,
    outer_folds: int,
    inner_folds: int,
    random_seed: int,
    compatibility_validation_inner_fold: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    ordered = targets.sort_values("target_id").reset_index(drop=True)
    groups = ordered["connectivity_id"].astype(str).to_numpy()
    labels = ordered["site_type"].astype(str).to_numpy()
    outer = StratifiedGroupKFold(
        n_splits=outer_folds,
        shuffle=True,
        random_state=random_seed,
    )
    outer_rows: list[dict[str, object]] = []
    nested_rows: list[dict[str, object]] = []
    compatibility_rows: list[dict[str, object]] = []
    balance_rows: list[dict[str, object]] = []
    outer_runs: list[dict[str, object]] = []
    test_counts: dict[str, int] = {
        value: 0 for value in ordered["target_id"].astype(str)
    }
    for outer_fold, (development_index, test_index) in enumerate(
        outer.split(ordered, labels, groups)
    ):
        development = ordered.iloc[development_index].reset_index(drop=True)
        test = ordered.iloc[test_index].reset_index(drop=True)
        development_groups = set(development["connectivity_id"].astype(str))
        test_groups = set(test["connectivity_id"].astype(str))
        if development_groups & test_groups:
            raise SiteNV2DatasetError(f"Outer fold {outer_fold} leaks connectivity")
        for role, frame in (("development", development), ("test", test)):
            for row in frame.itertuples(index=False):
                outer_rows.append(
                    {
                        "outer_fold": outer_fold,
                        "role": role,
                        "target_id": str(row.target_id),
                        "context_id": str(row.context_id),
                        "species_id": str(row.species_id),
                        "connectivity_id": str(row.connectivity_id),
                    }
                )
                if role == "test":
                    test_counts[str(row.target_id)] += 1
        inner = StratifiedGroupKFold(
            n_splits=inner_folds,
            shuffle=True,
            random_state=random_seed + 10_000 + outer_fold,
        )
        inner_assignments: dict[int, tuple[set[str], set[str]]] = {}
        for inner_fold, (train_index, validation_index) in enumerate(
            inner.split(
                development,
                development["site_type"].astype(str),
                development["connectivity_id"].astype(str),
            )
        ):
            train = development.iloc[train_index]
            validation = development.iloc[validation_index]
            train_groups = set(train["connectivity_id"].astype(str))
            validation_groups = set(validation["connectivity_id"].astype(str))
            if train_groups & validation_groups:
                raise SiteNV2DatasetError(
                    f"Outer {outer_fold} inner {inner_fold} leaks connectivity"
                )
            inner_assignments[inner_fold] = (train_groups, validation_groups)
            for role, frame in (("train", train), ("validation", validation)):
                for row in frame.itertuples(index=False):
                    nested_rows.append(
                        {
                            "outer_fold": outer_fold,
                            "inner_fold": inner_fold,
                            "role": role,
                            "target_id": str(row.target_id),
                            "context_id": str(row.context_id),
                            "species_id": str(row.species_id),
                            "connectivity_id": str(row.connectivity_id),
                        }
                    )
        chosen_train, chosen_validation = inner_assignments[
            compatibility_validation_inner_fold
        ]
        for row in ordered.itertuples(index=False):
            connectivity = str(row.connectivity_id)
            if connectivity in test_groups:
                role = "test"
            elif connectivity in chosen_validation:
                role = "validation"
            elif connectivity in chosen_train:
                role = "train"
            else:
                raise SiteNV2DatasetError("Compatibility role assignment is incomplete")
            compatibility_rows.append(
                {
                    "split_seed": random_seed + outer_fold,
                    "outer_fold": outer_fold,
                    "role": role,
                    "target_id": str(row.target_id),
                    "context_id": str(row.context_id),
                    "species_id": str(row.species_id),
                    "connectivity_id": connectivity,
                }
            )
        annotated = test.assign(N_bin=_n_bin(test["N_mean"]))
        for dimension in ("site_type", "N_bin", "solvent_raw"):
            for value, count in annotated[dimension].value_counts().items():
                balance_rows.append(
                    {
                        "outer_fold": outer_fold,
                        "role": "test",
                        "dimension": dimension,
                        "value": str(value),
                        "target_count": int(count),
                        "connectivity_count": int(
                            annotated.loc[
                                annotated[dimension].astype(str).eq(str(value)),
                                "connectivity_id",
                            ].nunique()
                        ),
                    }
                )
        outer_runs.append(
            {
                "outer_fold": outer_fold,
                "development_target_count": int(len(development)),
                "test_target_count": int(len(test)),
                "development_connectivity_count": int(len(development_groups)),
                "test_connectivity_count": int(len(test_groups)),
                "connectivity_overlap": 0,
                "inner_fold_count": inner_folds,
            }
        )
    if set(test_counts.values()) != {1}:
        raise SiteNV2DatasetError("Each target must appear in exactly one outer test")
    manifest = {
        "schema_version": "nucpred.mayr-site-n-nested-splits.v1",
        "group_identity": "standard_inchi_key_connectivity_block",
        "stratification_target": "site_type",
        "outer_folds": outer_folds,
        "inner_folds": inner_folds,
        "random_seed": random_seed,
        "target_count": int(len(ordered)),
        "connectivity_count": int(ordered["connectivity_id"].nunique()),
        "each_target_exactly_one_outer_test": True,
        "outer_test_used_for_selection": False,
        "compatibility_validation_inner_fold": compatibility_validation_inner_fold,
        "runs": outer_runs,
    }
    return (
        pd.DataFrame(outer_rows).sort_values(["outer_fold", "role", "target_id"]),
        pd.DataFrame(nested_rows).sort_values(
            ["outer_fold", "inner_fold", "role", "target_id"]
        ),
        pd.DataFrame(compatibility_rows).sort_values(
            ["outer_fold", "role", "target_id"]
        ),
        pd.DataFrame(balance_rows).sort_values(["outer_fold", "dimension", "value"]),
        manifest,
    )


def build_tables(config: Mapping[str, Any]) -> dict[str, object]:
    loaded = _load_inputs(config)
    counts = _projection_counts(loaded["projections"], config)
    measurements, adjudication = _adjudicate_measurements(
        measurements=loaded["measurements"],
        targets=loaded["targets"],
        candidates=loaded["candidates"],
        projections=loaded["projections"],
    )
    sites = _rebuild_sites(
        measurements=measurements,
        parent_sites=loaded["sites"],
        candidates=loaded["candidates"],
    )
    targets, aggregation_audit = v1._aggregate_targets(
        measurements,
        sites=sites,
        contexts=loaded["contexts"],
    )
    expected_targets = int(config["policy"]["expected_v2_targets"])
    if len(targets) != expected_targets:
        raise SiteNV2DatasetError(
            f"v2 target count changed: {len(targets)} != {expected_targets}"
        )
    candidate_coverage = v1._candidate_coverage(sites, loaded["candidates"])
    if not candidate_coverage["covered"].all():
        missing = candidate_coverage.loc[~candidate_coverage["covered"]]
        raise SiteNV2DatasetError(
            f"v2 contains {len(missing)} sites outside the candidate universe"
        )
    multi_site_pairs = v1._build_multi_site_pairs(targets)
    lineage = _target_lineage(
        parent_targets=loaded["targets"],
        measurements=measurements,
        v2_targets=targets,
        adjudication=adjudication,
    )
    split = config["splits"]
    (
        outer_membership,
        nested_membership,
        compatibility_membership,
        split_balance,
        split_manifest,
    ) = _nested_splits(
        targets,
        outer_folds=int(split["outer_folds"]),
        inner_folds=int(split["inner_folds"]),
        random_seed=int(split["random_seed"]),
        compatibility_validation_inner_fold=int(
            split["compatibility_validation_inner_fold"]
        ),
    )
    return {
        **loaded,
        "measurements": measurements,
        "sites": sites,
        "targets": targets,
        "aggregation_audit": aggregation_audit,
        "candidate_coverage": candidate_coverage,
        "multi_site_pairs": multi_site_pairs,
        "adjudication": adjudication,
        "target_lineage": lineage,
        "outer_membership": outer_membership,
        "nested_membership": nested_membership,
        "compatibility_membership": compatibility_membership,
        "split_balance": split_balance,
        "split_manifest": split_manifest,
        "projection_counts": counts,
    }


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    frame.to_parquet(path, index=False, engine="pyarrow", compression="zstd")


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False)


def _entry(path: Path, *, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def build_dataset(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    output_directory: str | Path | None = None,
) -> dict[str, object]:
    config, resolved_config = _read_config(config_path)
    output = (
        Path(output_directory).resolve()
        if output_directory is not None
        else _project_path(config["output_directory"])
    )
    if output.exists():
        raise SiteNV2DatasetError(
            f"Refusing to overwrite immutable v2 dataset: {output}"
        )
    tables = build_tables(config)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        outputs = {
            "species": staging / "species.parquet",
            "contexts": staging / "contexts.parquet",
            "sites": staging / "sites.parquet",
            "measurements": staging / "measurements.parquet",
            "targets": staging / "targets.parquet",
            "candidate_sites": staging / "candidate_sites.parquet",
            "multi_site_pairs": staging / "multi_site_pairs.csv",
            "aggregation_audit": staging / "aggregation_audit.csv",
            "context_feature_audit": staging / "context_feature_audit.csv",
            "candidate_coverage": staging / "candidate_coverage.csv",
            "pretraining_overlap_audit": staging / "pretraining_overlap_audit.csv",
            "stage_b_adjudication": staging / "stage_b_adjudication.csv",
            "target_lineage": staging / "target_lineage.csv",
            "outer_fold_membership": staging / "outer_fold_membership.csv",
            "nested_split_membership": staging / "nested_split_membership.csv",
            "split_membership": staging / "split_membership.csv",
            "split_balance": staging / "split_balance.csv",
            "split_manifest": staging / "split_manifest.json",
            "summary": staging / "summary.json",
        }
        for name in ("species", "contexts", "sites", "measurements", "targets"):
            _write_parquet(outputs[name], tables[name])
        _write_parquet(outputs["candidate_sites"], tables["candidates"])
        _write_csv(outputs["multi_site_pairs"], tables["multi_site_pairs"])
        _write_csv(outputs["aggregation_audit"], tables["aggregation_audit"])
        _write_csv(outputs["context_feature_audit"], tables["context_feature_audit"])
        _write_csv(outputs["candidate_coverage"], tables["candidate_coverage"])
        _write_csv(
            outputs["pretraining_overlap_audit"],
            tables["pretraining_overlap_audit"],
        )
        _write_csv(outputs["stage_b_adjudication"], tables["adjudication"])
        _write_csv(outputs["target_lineage"], tables["target_lineage"])
        _write_csv(outputs["outer_fold_membership"], tables["outer_membership"])
        _write_csv(outputs["nested_split_membership"], tables["nested_membership"])
        _write_csv(outputs["split_membership"], tables["compatibility_membership"])
        _write_csv(outputs["split_balance"], tables["split_balance"])
        atomic_write_json(outputs["split_manifest"], tables["split_manifest"])
        type_counts = {
            str(key): int(value)
            for key, value in tables["targets"]["site_type"].value_counts().items()
        }
        summary = {
            "schema_version": "nucpred.mayr-site-n-v2-summary.v1",
            "dataset_id": config["dataset_id"],
            "parent_dataset_id": config["parent"]["dataset_id"],
            "raw_measurement_count": int(len(tables["measurements"])),
            "formal_target_count": int(len(tables["targets"])),
            "formal_site_count": int(len(tables["sites"])),
            "formal_connectivity_count": int(
                tables["targets"]["connectivity_id"].nunique()
            ),
            "target_type_counts": type_counts,
            "stage_b_projection_counts": tables["projection_counts"],
            "excluded_unresolved_target_count": int(
                (~tables["target_lineage"]["formal_training_eligible"]).sum()
            ),
            "candidate_coverage_fraction": float(
                tables["candidate_coverage"]["covered"].mean()
            ),
            "outer_fold_count": int(config["splits"]["outer_folds"]),
            "inner_fold_count": int(config["splits"]["inner_folds"]),
            "each_target_exactly_one_outer_test": True,
            "internal_evaluation_status": ("retrospective_nested_grouped_not_pristine"),
            "unknown_is_negative": False,
            "stage_e_a_site_only_changes_n_target": False,
        }
        atomic_write_json(outputs["summary"], summary)
        files = {name: _entry(path, root=staging) for name, path in outputs.items()}
        manifest = {
            "schema_version": DATASET_SCHEMA,
            "dataset_id": config["dataset_id"],
            "generated_by": "nucpred.datasets.mayr_site_n_v2",
            "config_path": resolved_config.relative_to(ROOT).as_posix(),
            "config_sha256": sha256_file(resolved_config),
            "parent_dataset_id": config["parent"]["dataset_id"],
            "source_hashes": tables["hashes"],
            "contracts": {
                "stage_b_exact_reprojection_authoritative": True,
                "stage_b_unresolved_excluded": True,
                "raw_measurements_retained": True,
                "unknown_is_negative": False,
                "candidate_enumeration_reads_labels": False,
                "outer_test_partition": True,
                "internal_evaluation_is_pristine": False,
            },
            "files": files,
            "summary": summary,
        }
        atomic_write_json(staging / "dataset_manifest.json", manifest)
        staging.replace(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return verify_dataset(output)


def verify_dataset(directory: str | Path) -> dict[str, object]:
    root = Path(directory).resolve()
    manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != DATASET_SCHEMA:
        raise SiteNV2DatasetError("Unsupported v2 dataset manifest")
    for entry in manifest["files"].values():
        path = root / str(entry["path"])
        if not path.is_file():
            raise SiteNV2DatasetError(f"Missing v2 file: {path}")
        if path.stat().st_size != int(entry["bytes"]):
            raise SiteNV2DatasetError(f"v2 file size drifted: {path}")
        if sha256_file(path) != str(entry["sha256"]):
            raise SiteNV2DatasetError(f"v2 file hash drifted: {path}")
    summary = manifest["summary"]
    if int(summary["formal_target_count"]) != 1038:
        raise SiteNV2DatasetError("Formal v2 target count changed")
    if summary.get("each_target_exactly_one_outer_test") is not True:
        raise SiteNV2DatasetError("Outer test partition contract changed")
    return {
        "schema_version": "nucpred.mayr-site-n-v2-verification.v1",
        "dataset_id": manifest["dataset_id"],
        "status": "pass",
        "verified_file_count": int(len(manifest["files"])),
        "formal_target_count": int(summary["formal_target_count"]),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--verify-directory", type=Path)
    arguments = parser.parse_args(argv)
    if arguments.verify_directory is not None:
        result = verify_dataset(arguments.verify_directory)
    else:
        result = build_dataset(
            arguments.config,
            output_directory=arguments.output_directory,
        )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
