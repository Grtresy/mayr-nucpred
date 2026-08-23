"""Materialize an ESNUEL pilot subset from a verified frozen full dataset.

This path is intentionally data-only: it never reads mutable G1/xTB caches and
never recomputes scientific features.  It is used when cache code-parity gates
correctly reject reuse but the desired records already exist inside a verified,
immutable full dataset.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import pandas as pd

from nucpred.core.files import atomic_write_json, sha256_file
from nucpred.datasets.esnuel_d_node_xtb_pretraining import (
    DATASET_SCHEMA,
    EDGE_CATEGORICAL_FEATURES,
    EDGE_CATEGORY_SIZES,
    ELEMENT_VOCABULARY,
    EXPECTED_ELIGIBLE_RECORDS,
    EXPECTED_ESNUEL_RECORDS,
    EXPECTED_OVERLAP_RECORDS,
    GLOBAL_FEATURES,
    LOCAL_FEATURES,
    NODE_CATEGORICAL_FEATURES,
    NODE_CATEGORY_SIZES,
    verify_dataset,
)
from nucpred.project import get_project_layout


_LAYOUT = get_project_layout()
ROOT = _LAYOUT.root
DEFAULT_FULL = (
    ROOT
    / "data/processed/esnuel_d_node_xtb_pretraining"
    / "esnuel-d-node-xtb-pretraining-20260726-v1-full"
)
DEFAULT_SCOPE_DIRECTORY = (
    ROOT
    / "data/interim/esnuel_d_node_xtb_pretraining"
    / "esnuel-d-node-xtb-pretraining-20260726-v1"
    / "scopes/pilot4096"
)
DEFAULT_OUTPUT = (
    ROOT
    / "data/processed/esnuel_d_node_xtb_pretraining"
    / "esnuel-d-node-xtb-pretraining-20260726-v1-pilot4096"
)
EXPECTED_SCOPE = "pilot4096"
EXPECTED_RECORDS = 4096
EXPECTED_SOURCE_ID_SHA256 = (
    "84449162ba65bf47b88e6d0b41a664a7b0edb8a97b6867627d64079a45d34503"
)
SELECTION_COLUMNS = (
    "native_pretraining_split",
    "pretraining_role",
    "selection_hash_rank",
    "selection_mandatory",
    "selection_mandatory_strata_json",
    "within_role_selection_rank",
    "selection_index",
)


class FrozenSubsetError(RuntimeError):
    """Raised if a frozen subset cannot be proven to be a pure projection."""


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _source_id_sha256(values: Sequence[str]) -> str:
    return hashlib.sha256(
        ("\n".join(sorted(values)) + "\n").encode("utf-8")
    ).hexdigest()


def _ordered_subset(
    frame: pd.DataFrame,
    *,
    source_ids: Sequence[str],
) -> pd.DataFrame:
    rank = {source_id: index for index, source_id in enumerate(source_ids)}
    selected = frame.loc[
        frame["source_id"].astype(str).isin(set(source_ids))
    ].copy()
    selected["_source_rank"] = selected["source_id"].astype(str).map(rank)
    selected["_source_row_order"] = range(len(selected))
    selected = selected.sort_values(
        ["_source_rank", "_source_row_order"],
        kind="stable",
    )
    return selected.drop(
        columns=["_source_rank", "_source_row_order"]
    ).reset_index(drop=True)


def _coverage_payload(
    records: pd.DataFrame,
    atoms: pd.DataFrame,
    molecules: pd.DataFrame,
    *,
    threshold: float,
) -> dict[str, object]:
    fraction = float(records["complete_xtb10"].mean())
    return {
        "schema_version": "nucpred.esnuel-d-node-xtb-coverage.v1",
        "record_count": len(records),
        "atom_count": len(atoms),
        "complete_xtb10_record_count": int(
            records["complete_xtb10"].sum()
        ),
        "complete_xtb10_fraction": fraction,
        "minimum_complete_xtb10_fraction": threshold,
        "coverage_gate_pass": fraction >= threshold,
        "local_feature_coverage": {
            name: {
                "available_atoms": int(
                    atoms[f"{name}__available"].sum()
                ),
                "total_atoms": len(atoms),
                "fraction": float(
                    atoms[f"{name}__available"].mean()
                ),
            }
            for name in LOCAL_FEATURES
        },
        "global_feature_coverage": {
            name: {
                "available_records": int(
                    molecules[f"{name}__available"].sum()
                ),
                "total_records": len(molecules),
                "fraction": float(
                    molecules[f"{name}__available"].mean()
                ),
            }
            for name in GLOBAL_FEATURES
        },
        "missing_value_policy": (
            "training_fold_median_imputation_plus_availability_masks"
        ),
    }


def _asset(path: Path, role: str, format_name: str) -> dict[str, object]:
    return {
        "path": path.name,
        "role": role,
        "format": format_name,
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def materialize_pilot4096(
    *,
    full_directory: str | Path = DEFAULT_FULL,
    scope_directory: str | Path = DEFAULT_SCOPE_DIRECTORY,
    output_directory: str | Path = DEFAULT_OUTPUT,
) -> dict[str, object]:
    full = Path(full_directory).resolve()
    scope = Path(scope_directory).resolve()
    output = Path(output_directory).resolve()
    if output.is_dir():
        return verify_dataset(output)
    if output.exists():
        raise FrozenSubsetError("Output exists but is not a directory")
    full_verification = verify_dataset(full)
    selection_path = scope / "selection_manifest.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if (
        selection.get("scope") != EXPECTED_SCOPE
        or int(selection.get("selected_record_count", -1))
        != EXPECTED_RECORDS
        or selection.get("selected_source_id_sha256")
        != EXPECTED_SOURCE_ID_SHA256
    ):
        raise FrozenSubsetError("Pilot4096 selection contract changed")
    membership = pd.read_csv(scope / "split_membership.csv").sort_values(
        "selection_index"
    )
    source_ids = membership["source_id"].astype(str).tolist()
    if (
        len(source_ids) != EXPECTED_RECORDS
        or len(set(source_ids)) != EXPECTED_RECORDS
        or _source_id_sha256(source_ids) != EXPECTED_SOURCE_ID_SHA256
    ):
        raise FrozenSubsetError("Pilot4096 membership identity changed")

    inventory = pd.read_parquet(scope / "inventory.parquet")
    inventory = inventory.set_index("source_id", drop=False)
    full_records = pd.read_parquet(full / "records.parquet")
    if not set(source_ids).issubset(set(full_records["source_id"].astype(str))):
        raise FrozenSubsetError("Pilot4096 is not a subset of frozen full")
    records = _ordered_subset(full_records, source_ids=source_ids)
    inventory_ordered = inventory.loc[source_ids]
    graph_match = (
        records["model_graph_sha256"].astype(str).to_numpy()
        == inventory_ordered["model_graph_sha256"].astype(str).to_numpy()
    )
    if not bool(graph_match.all()):
        raise FrozenSubsetError("Frozen/full graph identity drifted")
    for column in SELECTION_COLUMNS:
        records[column] = inventory_ordered[column].tolist()

    atoms = _ordered_subset(
        pd.read_parquet(full / "atom_features.parquet"),
        source_ids=source_ids,
    )
    molecules = _ordered_subset(
        pd.read_parquet(full / "molecule_features.parquet"),
        source_ids=source_ids,
    )
    primitives = _ordered_subset(
        pd.read_parquet(full / "raw_xtb_primitives.parquet"),
        source_ids=source_ids,
    )
    geometry = _ordered_subset(
        pd.read_csv(full / "geometry_ledger.csv"),
        source_ids=source_ids,
    )
    calculations = _ordered_subset(
        pd.read_csv(full / "calculation_ledger.csv"),
        source_ids=source_ids,
    )
    failures = _ordered_subset(
        pd.read_csv(full / "failure_ledger.csv"),
        source_ids=source_ids,
    )
    mapping = pd.read_csv(scope / "mapping_audit.csv")
    overlap = pd.read_csv(scope / "overlap_audit.csv")
    target_audit = pd.read_csv(scope / "target_connectivity_audit.csv")
    if (
        len(records) != EXPECTED_RECORDS
        or len(molecules) != EXPECTED_RECORDS
        or len(mapping) != EXPECTED_RECORDS
        or len(membership) != EXPECTED_RECORDS
        or len(overlap) != EXPECTED_ESNUEL_RECORDS
        or int(
            overlap["excluded_for_mayr_connectivity_overlap"].sum()
        )
        != EXPECTED_OVERLAP_RECORDS
    ):
        raise FrozenSubsetError("Frozen subset table counts changed")
    coverage = _coverage_payload(
        records,
        atoms,
        molecules,
        threshold=0.95,
    )
    if coverage["coverage_gate_pass"] is not True:
        raise FrozenSubsetError("Frozen subset xTB coverage failed")

    dataset_id = "esnuel-d-node-xtb-pretraining-20260726-v1-pilot4096"
    native_counts = {
        role: int(records["pretraining_role"].eq(role).sum())
        for role in ("train", "validation", "audit_test")
    }
    if native_counts != {
        "train": 2861,
        "validation": 705,
        "audit_test": 530,
    }:
        raise FrozenSubsetError("Pilot4096 native role counts changed")
    summary = {
        "schema_version": "nucpred.esnuel-d-node-xtb-summary.v1",
        "dataset_id": dataset_id,
        "scope": EXPECTED_SCOPE,
        "parent_esnuel_record_count": EXPECTED_ESNUEL_RECORDS,
        "excluded_mayr_connectivity_overlap_record_count": (
            EXPECTED_OVERLAP_RECORDS
        ),
        "eligible_esnuel_record_count": EXPECTED_ELIGIBLE_RECORDS,
        "selected_record_count": len(records),
        "selected_native_role_counts": native_counts,
        "atom_count": len(atoms),
        "hydrogen_atom_count": int(atoms["is_hydrogen"].sum()),
        "complete_xtb10_record_count": int(
            records["complete_xtb10"].sum()
        ),
        "complete_xtb10_fraction": float(
            records["complete_xtb10"].mean()
        ),
        "coverage_gate_pass": True,
        "failure_ledger_row_count": len(failures),
        "geometry": "G1 ETKDGv3 20-conformer MMFF94s/UFF",
        "electronic_method": "fixed-G1 GFN1-xTB gas + ALPB-DMSO",
        "pretraining": True,
        "mayr_branch_loaded": False,
        "mayr_structure_identity_loaded_for_exclusion": True,
        "mayr_labels_used": False,
        "materialization": "pure_subset_of_verified_frozen_full",
    }
    feature_schema = {
        "schema_version": (
            "nucpred.esnuel-d-node-xtb-feature-schema.v1"
        ),
        "element_vocabulary": ELEMENT_VOCABULARY,
        "hydrogen_policy": "ordinary_element_in_shared_vocabulary",
        "node_categorical_features": NODE_CATEGORICAL_FEATURES,
        "node_category_sizes": NODE_CATEGORY_SIZES,
        "edge_categorical_features": EDGE_CATEGORICAL_FEATURES,
        "edge_category_sizes": EDGE_CATEGORY_SIZES,
        "node_local_features": LOCAL_FEATURES,
        "node_local_entry_point": "before_message_passing",
        "global_xtb_features": GLOBAL_FEATURES,
        "global_xtb_entry_point": "after_site_pooling",
        "availability_mask_for_every_xtb_feature": True,
        "esnuel_targets": (
            "mca_targets_all_atom",
            "gcs_targets_all_atom",
            "site_mask_all_atom",
        ),
        "gcs_dimension": 53,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{dataset_id}.staging-", dir=output.parent)
    )
    try:
        files: list[tuple[Path, str, str]] = []

        def parquet(name: str, frame: pd.DataFrame, role: str) -> None:
            path = staging / name
            frame.to_parquet(
                path,
                index=False,
                engine="pyarrow",
                compression="zstd",
            )
            files.append((path, role, "parquet"))

        def csv(name: str, frame: pd.DataFrame, role: str) -> None:
            path = staging / name
            frame.to_csv(path, index=False, lineterminator="\n")
            files.append((path, role, "csv"))

        def json_file(name: str, payload: object, role: str) -> None:
            path = staging / name
            atomic_write_json(path, payload, ensure_ascii=False)
            files.append((path, role, "json"))

        parquet("records.parquet", records, "records")
        parquet("atom_features.parquet", atoms, "atom_features")
        parquet(
            "molecule_features.parquet",
            molecules,
            "molecule_features",
        )
        parquet(
            "raw_xtb_primitives.parquet",
            primitives,
            "raw_xtb_primitives",
        )
        csv("geometry_ledger.csv", geometry, "geometry_ledger")
        csv(
            "calculation_ledger.csv",
            calculations,
            "calculation_ledger",
        )
        csv("failure_ledger.csv", failures, "failure_ledger")
        csv("overlap_audit.csv", overlap, "overlap_audit")
        csv(
            "target_connectivity_audit.csv",
            target_audit,
            "target_connectivity_audit",
        )
        csv("mapping_audit.csv", mapping, "mapping_audit")
        csv(
            "split_membership.csv",
            membership,
            "split_membership",
        )
        json_file(
            "selection_manifest.json",
            selection,
            "selection_manifest",
        )
        json_file("coverage.json", coverage, "coverage")
        json_file(
            "feature_schema.json",
            feature_schema,
            "feature_schema",
        )
        json_file("summary.json", summary, "summary")
        assets = [
            _asset(path, role, format_name)
            for path, role, format_name in files
        ]
        source_file = Path(__file__).resolve()
        full_manifest = full / "dataset_manifest.json"
        manifest = {
            "schema_version": DATASET_SCHEMA,
            "dataset_id": dataset_id,
            "scope": EXPECTED_SCOPE,
            "generated_by": "nucpred.datasets.esnuel_frozen_subset",
            "builder_source": _display_path(source_file),
            "builder_source_sha256": sha256_file(source_file),
            "source_hashes": {
                "builder": sha256_file(source_file),
                "frozen_full_manifest": sha256_file(full_manifest),
                "selection_manifest": sha256_file(selection_path),
            },
            "inputs": [
                {
                    "path": _display_path(full),
                    "manifest_sha256": sha256_file(full_manifest),
                    "verification": full_verification,
                    "role": "verified_frozen_full_feature_parent",
                },
                {
                    "path": _display_path(scope),
                    "selection_manifest_sha256": sha256_file(
                        selection_path
                    ),
                    "role": "target_independent_pilot4096_selection",
                },
            ],
            "assets": assets,
            "contracts": {
                "parent_esnuel_records": EXPECTED_ESNUEL_RECORDS,
                "excluded_connectivity_overlap_records": (
                    EXPECTED_OVERLAP_RECORDS
                ),
                "eligible_esnuel_records": EXPECTED_ELIGIBLE_RECORDS,
                "mayr_branch_loaded": False,
                "mayr_structure_identity_loaded_for_exclusion": True,
                "mayr_labels_loaded": False,
                "mayr_labels_used": False,
                "native_esnuel_split_preserved": True,
                "audit_test_fit_allowed": False,
                "hydrogen_is_ordinary_element": True,
                "added_h_mca_gcs_site_labels_allowed": False,
                "fixed_geometry": "G1",
                "no_dft": True,
                "no_xtb_geometry_optimization": True,
                "node_local_feature_count": len(LOCAL_FEATURES),
                "global_xtb_feature_count": len(GLOBAL_FEATURES),
                "missing_values_preserved_with_masks": True,
                "minimum_complete_xtb10_fraction": 0.95,
                "scientific_values_recomputed": False,
                "pure_subset_of_verified_frozen_full": True,
                "mutable_cache_read": False,
            },
        }
        atomic_write_json(
            staging / "dataset_manifest.json",
            manifest,
            ensure_ascii=False,
        )
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return verify_dataset(output)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-directory", type=Path, default=DEFAULT_FULL)
    parser.add_argument(
        "--scope-directory",
        type=Path,
        default=DEFAULT_SCOPE_DIRECTORY,
    )
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    payload = materialize_pilot4096(
        full_directory=arguments.full_directory,
        scope_directory=arguments.scope_directory,
        output_directory=arguments.output_directory,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
