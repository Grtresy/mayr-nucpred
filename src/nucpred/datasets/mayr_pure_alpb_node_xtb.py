"""Build the fixed 1,136-row all-atom Mayr G1 + GFN1-xTB dataset.

The historical cheap-feature table is used only to freeze population identity.
Every geometry and electronic primitive is recomputed on a target-independent
nucleophile fragment, and all resumable cache entries are bound to config and
source hashes so a long-lived process cannot silently mix source revisions.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
import tomllib
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import rdBase

from nucpred.core.files import atomic_write_json, sha256_file
from nucpred.features.all_atom_graph import (
    EDGE_CATEGORICAL_FEATURES,
    EDGE_CATEGORY_SIZES,
    ELEMENT_VOCABULARY,
    NODE_CATEGORICAL_FEATURES,
    NODE_CATEGORY_SIZES,
    assert_category_ranges,
    featurize_explicit_molecule,
)
from nucpred.project import get_project_layout
from nucpred.protocols import xtb_runtime as xtb


_LAYOUT = get_project_layout()
ROOT = _LAYOUT.root
DEFAULT_CONFIG = ROOT / "configs/mayr_pure_alpb_node_xtb.toml"
DEFAULT_DATASET_ID = "mayr-pure-alpb-explicit-h-node-xtb-20260725-v1"
CONFIG_SCHEMA = "nucpred.mayr-pure-alpb-node-xtb-config.v1"
DATASET_SCHEMA = "nucpred.mayr-pure-alpb-node-xtb-dataset.v1"
EXPECTED_RECORDS = 1136
EXPECTED_SITE_SUPERVISED = 1116
EXPECTED_H_GROUPS = 119
EXPECTED_MULTIFRAGMENT = 40
LOCAL_FEATURES = (
    "alpb_fukui_minus_density",
    "alpb_cm5",
    "alpb_condensed_softness",
    "alpb_condensed_nucleophilicity_index_ev",
)
GLOBAL_FEATURES = (
    "alpb_homo_n_hartree",
    "alpb_softness_hartree_inverse",
    "alpb_vip_hartree",
    "alpb_nucleophilicity_index_hartree",
    "alpb_homo_n_minus_1_hartree",
    "delta_e_solv_hartree",
)
SOLVENT_DESCRIPTOR_COLUMNS = (
    "solvent_nD",
    "solvent_f(n^2)",
    "solvent_epsilon_r",
    "solvent_ET(30)",
    "solvent_DI",
    "solvent_ES",
    "solvent_alpha_1",
    "solvent_beta_1",
    "solvent_alpha",
    "solvent_beta",
    "solvent_pi_*",
    "solvent_SPP",
    "solvent_SB",
    "solvent_SA",
    "solvent_delta_d",
    "solvent_delta_p",
    "solvent_delta_h",
    "solvent_delta",
)
VALIDATION_COLUMNS = ("severity", "code", "source_id", "message")


@dataclass(frozen=True, slots=True)
class FragmentSelection:
    molecule: Chem.Mol
    new_to_old: tuple[int, ...]
    removed_smiles: tuple[str, ...]
    original_fragment_count: int
    rule: str


class DatasetBuildError(RuntimeError):
    """Raised when a frozen data or execution contract cannot be satisfied."""


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _stable_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _json_compact(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=isinstance(value, dict),
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _parse_indices(value: object) -> list[int]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    text = str(value).strip()
    if not text:
        return []
    parsed = json.loads(text)
    if not isinstance(parsed, list):
        raise ValueError("Expected a JSON index list")
    return [int(item) for item in parsed]


def _json_safe(value: object) -> object:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.floating):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _float_array(value: object, expected: int) -> np.ndarray:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list):
        raise ValueError(f"Expected a {expected}-value array")
    object_array = np.asarray(value, dtype=object)
    if object_array.size != expected:
        raise ValueError(f"Expected a {expected}-value array")
    return np.asarray(
        [
            math.nan if item is None else float(item)
            for item in object_array.reshape(-1).tolist()
        ],
        dtype=float,
    )


def _read_config(path: Path) -> dict[str, Any]:
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != CONFIG_SCHEMA:
        raise DatasetBuildError("Unsupported Mayr node-xTB config schema")
    expected = {
        "dataset_id": DEFAULT_DATASET_ID,
        "expected_records": EXPECTED_RECORDS,
        "expected_site_supervised_records": EXPECTED_SITE_SUPERVISED,
        "expected_h_group_records": EXPECTED_H_GROUPS,
        "expected_multifragment_records": EXPECTED_MULTIFRAGMENT,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise DatasetBuildError(f"Config changed frozen field {key}")
    if tuple(payload["features"]["node_local"]) != LOCAL_FEATURES:
        raise DatasetBuildError("Local xTB feature identity or order changed")
    if tuple(payload["features"]["global"]) != GLOBAL_FEATURES:
        raise DatasetBuildError("Global xTB feature identity or order changed")
    geometry = payload["geometry"]
    if (
        geometry.get("name") != "G1"
        or geometry.get("method") != "ETKDGv3"
        or int(geometry.get("conformer_count", 0)) != 20
        or geometry.get("primary_force_field") != "MMFF94s"
        or geometry.get("fallback_force_field") != "UFF"
        or bool(geometry.get("geometry_optimization_with_xtb", True))
    ):
        raise DatasetBuildError("G1 geometry contract changed")
    xtb_section = payload["xtb"]
    if (
        xtb_section.get("method") != "GFN1-xTB"
        or int(xtb_section.get("gfn", 0)) != 1
        or xtb_section.get("solvation_model") != "ALPB"
        or bool(xtb_section.get("geometry_optimization", True))
    ):
        raise DatasetBuildError("GFN1-xTB/ALPB contract changed")
    if set(payload["quality_control"]) != set(xtb.QC_KEYS):
        raise DatasetBuildError("xTB QC surface changed")
    return payload


def _resolve_inputs(
    config: Mapping[str, Any],
) -> tuple[Path, Path, Path, Path]:
    parent = (ROOT / str(config["parent_records_path"])).resolve()
    split = (ROOT / str(config["parent_split_manifest_path"])).resolve()
    cohort = (ROOT / str(config["cohort_evidence_path"])).resolve()
    archive = (ROOT / str(config["xtb"]["archive_path"])).resolve()
    expected = (
        (parent, str(config["parent_records_sha256"])),
        (split, str(config["parent_split_manifest_sha256"])),
        (cohort, str(config["cohort_evidence_sha256"])),
        (archive, str(config["xtb"]["archive_sha256"])),
    )
    for path, digest in expected:
        if sha256_file(path) != digest:
            raise DatasetBuildError(f"Frozen input digest changed: {_display_path(path)}")
    return parent, split, cohort, archive


def _cache_name(source_id: str) -> str:
    return hashlib.sha256(str(source_id).encode("utf-8")).hexdigest() + ".json"


def _mapped_smiles(molecule: Chem.Mol) -> str:
    mapped = Chem.Mol(molecule)
    for index, atom in enumerate(mapped.GetAtoms()):
        atom.SetAtomMapNum(index + 1)
    return Chem.MolToSmiles(
        mapped,
        canonical=False,
        isomericSmiles=True,
        allHsExplicit=False,
    )


def _molecule_from_mapped_smiles(smiles: str, source_atom_count: int) -> Chem.Mol:
    parsed = Chem.MolFromSmiles(str(smiles))
    if parsed is None or parsed.GetNumAtoms() != int(source_atom_count):
        raise DatasetBuildError("Mapped model SMILES failed to round-trip")
    numbered = {
        int(atom.GetAtomMapNum()): int(atom.GetIdx()) for atom in parsed.GetAtoms()
    }
    if set(numbered) != set(range(1, source_atom_count + 1)):
        raise DatasetBuildError("Mapped model SMILES lost atom-map identity")
    order = [numbered[index] for index in range(1, source_atom_count + 1)]
    reordered = Chem.RenumberAtoms(parsed, order)
    for atom in reordered.GetAtoms():
        atom.SetAtomMapNum(0)
    return reordered


def _is_monatomic_spectator(
    molecule: Chem.Mol, *, elements: set[str], charge: int
) -> bool:
    if molecule.GetNumAtoms() != 1:
        return False
    atom = molecule.GetAtomWithIdx(0)
    return atom.GetSymbol() in elements and atom.GetFormalCharge() == charge


def _select_fragment(
    smiles: str,
    *,
    fragment_policy: Mapping[str, Any],
) -> FragmentSelection:
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None or molecule.GetNumAtoms() == 0:
        raise DatasetBuildError(f"Cannot parse curated structure: {smiles!r}")
    mappings: list[tuple[int, ...]] = []
    fragments = Chem.GetMolFrags(
        molecule,
        asMols=True,
        sanitizeFrags=True,
        fragsMolAtomMapping=mappings,
    )
    if len(fragments) == 1:
        return FragmentSelection(
            molecule=fragments[0],
            new_to_old=tuple(int(value) for value in mappings[0]),
            removed_smiles=(),
            original_fragment_count=1,
            rule="single_fragment_identity",
        )
    spectator_elements = set(
        map(str, fragment_policy["monatomic_spectator_elements"])
    )
    spectator_charge = int(fragment_policy["monatomic_spectator_charge"])
    triflate = Chem.MolFromSmiles(str(fragment_policy["triflate_smiles"]))
    if triflate is None:
        raise DatasetBuildError("Configured triflate pattern cannot be parsed")
    triflate_key = Chem.MolToSmiles(triflate, isomericSmiles=True)
    retained: list[int] = []
    removed: list[str] = []
    rules: list[str] = []
    for index, fragment in enumerate(fragments):
        canonical = Chem.MolToSmiles(fragment, isomericSmiles=True)
        if _is_monatomic_spectator(
            fragment,
            elements=spectator_elements,
            charge=spectator_charge,
        ):
            removed.append(canonical)
            rules.append("monatomic_alkali_spectator")
        elif canonical == triflate_key:
            removed.append(canonical)
            rules.append("triflate_spectator")
        else:
            retained.append(index)
    if len(retained) != 1:
        raise DatasetBuildError(
            "Known-spectator policy did not leave exactly one fragment: "
            f"retained={len(retained)} smiles={smiles}"
        )
    selected = retained[0]
    return FragmentSelection(
        molecule=fragments[selected],
        new_to_old=tuple(int(value) for value in mappings[selected]),
        removed_smiles=tuple(removed),
        original_fragment_count=len(fragments),
        rule="+".join(sorted(set(rules))),
    )


def _label_projection(
    row: Mapping[str, object],
    *,
    old_to_new: Mapping[int, int],
    graph_atomic_numbers: Sequence[int],
    hydrogen_parents: Sequence[int],
) -> dict[str, object]:
    source_id = str(row["source_id"])
    parent_kind = str(row["site_target_kind_all_atom"])
    parent_mask = bool(row["site_target_mask_all_atom"])
    parent_targets = _parse_indices(row["site_target_atoms_all_atom_json"])
    parent_donors = _parse_indices(row["donor_heavy_atom_indices_json"])
    if str(row["mechanism_type"]) == "hydride_transfer":
        donors = sorted(
            {old_to_new[index] for index in parent_donors if index in old_to_new}
        )
        targets = sorted(
            index
            for index, parent in enumerate(hydrogen_parents)
            if parent in set(donors)
        )
        kind = "candidate_set" if parent_mask else "none"
        if parent_mask and (not donors or not targets):
            raise DatasetBuildError(f"{source_id}: H-group label was lost")
    else:
        unmapped = [index for index in parent_targets if index not in old_to_new]
        if parent_mask and unmapped:
            raise DatasetBuildError(
                f"{source_id}: supervised target entered a removed spectator"
            )
        targets = sorted({old_to_new[index] for index in parent_targets if index in old_to_new})
        donors = []
        kind = parent_kind if parent_mask else "none"
        if parent_mask and not targets:
            raise DatasetBuildError(f"{source_id}: mapped target set is empty")
    if any(
        index < 0 or index >= len(graph_atomic_numbers) for index in targets
    ):
        raise DatasetBuildError(f"{source_id}: mapped site index is out of range")
    if str(row["mechanism_type"]) == "hydride_transfer" and any(
        graph_atomic_numbers[index] != 1 for index in targets
    ):
        raise DatasetBuildError(f"{source_id}: mapped H-group contains a non-H atom")
    return {
        "site_target_mask_model": bool(parent_mask),
        "site_target_kind_model": kind,
        "site_target_atoms_model_json": _json_compact(targets),
        "donor_heavy_atom_indices_model_json": _json_compact(donors),
        "site_target_count_model": len(targets),
        "label_mapping_target_independent": True,
    }


def _inventory_rows(
    parent: pd.DataFrame,
    cohort_ids: set[str],
    *,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected_parent = parent.loc[
        parent["source_id"].astype(str).isin(cohort_ids)
    ].copy()
    if len(selected_parent) != EXPECTED_RECORDS:
        raise DatasetBuildError("Frozen cohort did not join to exactly 1,136 parents")
    rows: list[dict[str, object]] = []
    fragment_audit: list[dict[str, object]] = []
    label_audit: list[dict[str, object]] = []
    solvent_aliases = {str(key): str(value) for key, value in config["solvents"].items()}
    for cohort_index, parent_row in enumerate(
        selected_parent.to_dict(orient="records")
    ):
        source_id = str(parent_row["source_id"])
        selection = _select_fragment(
            str(parent_row["curated_canonical_smiles"]),
            fragment_policy=config["fragment_policy"],
        )
        old_to_new = {
            int(old): int(new) for new, old in enumerate(selection.new_to_old)
        }
        source_molecule = Chem.Mol(selection.molecule)
        source_count = source_molecule.GetNumAtoms()
        explicit = Chem.AddHs(Chem.Mol(source_molecule), addCoords=False)
        if explicit.GetNumAtoms() < source_count:
            raise DatasetBuildError(f"{source_id}: Chem.AddHs removed source atoms")
        graph = featurize_explicit_molecule(
            explicit,
            source_atom_count=source_count,
        )
        assert_category_ranges(graph.node_categorical, NODE_CATEGORY_SIZES)
        assert_category_ranges(graph.edge_categorical, EDGE_CATEGORY_SIZES)
        label = _label_projection(
            parent_row,
            old_to_new=old_to_new,
            graph_atomic_numbers=graph.atomic_numbers,
            hydrogen_parents=graph.hydrogen_parent_index,
        )
        solvent_raw = str(parent_row["solvent_raw"])
        if solvent_raw not in solvent_aliases:
            raise DatasetBuildError(
                f"{source_id}: unsupported frozen pure solvent {solvent_raw!r}"
            )
        model_canonical = Chem.MolToSmiles(
            source_molecule, isomericSmiles=True
        )
        model_mapped = _mapped_smiles(source_molecule)
        model_charge = int(Chem.GetFormalCharge(source_molecule))
        radical_electrons = int(
            sum(
                atom.GetNumRadicalElectrons()
                for atom in source_molecule.GetAtoms()
            )
        )
        graph_values = {
            "model_canonical_smiles": model_canonical,
            "model_mapped_smiles": model_mapped,
            "model_source_atom_count": source_count,
            "model_all_atom_count": graph.atom_count,
            "model_hydrogen_atom_count": int(
                sum(value == 1 for value in graph.atomic_numbers)
            ),
            "model_formal_charge": model_charge,
            "model_radical_electrons": radical_electrons,
            "model_atomic_numbers_json": _json_compact(graph.atomic_numbers),
            "model_node_categorical_json": _json_compact(
                graph.node_categorical
            ),
            "model_directed_edges_json": _json_compact(graph.directed_edges),
            "model_edge_categorical_json": _json_compact(
                graph.edge_categorical
            ),
            "model_hydrogen_parent_index_json": _json_compact(
                graph.hydrogen_parent_index
            ),
            "model_graph_sha256": graph.mapping_sha256,
        }
        rows.append(
            {
                **parent_row,
                "cohort_index": cohort_index,
                "cohort_selection_evidence": (
                    "historical_xtb_solvent_available_equals_1_identity_only"
                ),
                "original_full_structure_smiles": str(
                    parent_row["curated_canonical_smiles"]
                ),
                "original_formal_charge": int(parent_row["formal_charge"]),
                "original_fragment_count": selection.original_fragment_count,
                "spectator_stripped": selection.original_fragment_count > 1,
                "fragment_selection_rule": selection.rule,
                "removed_fragment_smiles_json": _json_compact(
                    selection.removed_smiles
                ),
                "model_to_original_source_atom_json": _json_compact(
                    selection.new_to_old
                ),
                "original_to_model_source_atom_json": _json_compact(
                    {str(key): value for key, value in sorted(old_to_new.items())}
                ),
                "fragment_selection_target_independent": True,
                "xtb_alpb_solvent": solvent_aliases[solvent_raw],
                "geometry_seed": int(config["geometry"]["random_seed"])
                + cohort_index,
                **graph_values,
                **label,
            }
        )
        fragment_audit.append(
            {
                "source_id": source_id,
                "original_fragment_count": selection.original_fragment_count,
                "selected_source_atom_count": source_count,
                "original_formal_charge": int(parent_row["formal_charge"]),
                "model_formal_charge": model_charge,
                "selection_rule": selection.rule,
                "removed_fragment_smiles_json": _json_compact(
                    selection.removed_smiles
                ),
                "model_to_original_source_atom_json": _json_compact(
                    selection.new_to_old
                ),
                "used_target_fields": False,
                "status": "pass",
            }
        )
        label_audit.append(
            {
                "source_id": source_id,
                "supervision_level": str(parent_row["supervision_level"]),
                "parent_target_mask": bool(
                    parent_row["site_target_mask_all_atom"]
                ),
                "parent_target_atoms_json": str(
                    parent_row["site_target_atoms_all_atom_json"]
                ),
                "model_target_mask": label["site_target_mask_model"],
                "model_target_atoms_json": label[
                    "site_target_atoms_model_json"
                ],
                "model_target_count": label["site_target_count_model"],
                "target_in_removed_fragment": False,
                "status": "pass",
            }
        )
    return (
        pd.DataFrame(rows),
        pd.DataFrame(fragment_audit),
        pd.DataFrame(label_audit),
    )


def _source_hashes(config_path: Path) -> dict[str, str]:
    paths = {
        "builder": Path(__file__).resolve(),
        "all_atom_graph": (
            ROOT / "src/nucpred/features/all_atom_graph.py"
        ).resolve(),
        "xtb_helper": (
            ROOT / "src/nucpred/protocols/xtb_runtime.py"
        ).resolve(),
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def build_inventory(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
) -> dict[str, object]:
    config_file = Path(config_path).resolve()
    config = _read_config(config_file)
    parent_path, _, cohort_path, _ = _resolve_inputs(config)
    working = (ROOT / str(config["working_directory"])).resolve()
    working.mkdir(parents=True, exist_ok=True)
    inventory_path = working / "inventory.parquet"
    manifest_path = working / "inventory_manifest.json"
    current_hashes = _source_hashes(config_file)
    config_hash = sha256_file(config_file)
    if inventory_path.exists() or manifest_path.exists():
        if not inventory_path.is_file() or not manifest_path.is_file():
            raise DatasetBuildError("Partial inventory cache is present")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("config_sha256") != config_hash
            or manifest.get("source_hashes") != current_hashes
            or manifest.get("inventory_sha256") != sha256_file(inventory_path)
        ):
            raise DatasetBuildError(
                "Existing inventory was built from different config/source bytes"
            )
        return {
            "status": "reused",
            "record_count": int(manifest["record_count"]),
            "working_directory": _display_path(working),
            "inventory_sha256": manifest["inventory_sha256"],
        }
    parent = pd.read_parquet(parent_path)
    cohort = pd.read_parquet(cohort_path)
    column = str(config["cohort_selection_column"])
    selected = cohort.loc[
        cohort[column].eq(config["cohort_selection_value"]), "source_id"
    ].astype(str)
    cohort_ids = set(selected)
    if len(selected) != EXPECTED_RECORDS or len(cohort_ids) != EXPECTED_RECORDS:
        raise DatasetBuildError("Cohort evidence no longer selects exactly 1,136 IDs")
    ordered_ids = sorted(cohort_ids)
    cohort_digest = hashlib.sha256(
        ("\n".join(ordered_ids) + "\n").encode("utf-8")
    ).hexdigest()
    if cohort_digest != str(config["cohort_source_id_sha256"]):
        raise DatasetBuildError("Frozen cohort source-id digest changed")
    records, fragment_audit, label_audit = _inventory_rows(
        parent,
        cohort_ids,
        config=config,
    )
    if records["source_id"].nunique() != EXPECTED_RECORDS:
        raise DatasetBuildError("Inventory source IDs are not unique")
    observed = {
        "site_supervised": int(records["site_target_mask_model"].sum()),
        "h_groups": int(
            records["supervision_level"]
            .eq("equivalent_or_indistinguishable_h_group")
            .sum()
        ),
        "multifragment": int(records["spectator_stripped"].sum()),
    }
    expected = {
        "site_supervised": EXPECTED_SITE_SUPERVISED,
        "h_groups": EXPECTED_H_GROUPS,
        "multifragment": EXPECTED_MULTIFRAGMENT,
    }
    if observed != expected:
        raise DatasetBuildError(
            f"Inventory population counts changed: {observed} != {expected}"
        )
    cohort_manifest = records[
        [
            "cohort_index",
            "source_id",
            "model_canonical_smiles",
            "solvent_raw",
            "xtb_alpb_solvent",
            "spectator_stripped",
            "model_formal_charge",
            "supervision_level",
            "site_target_mask_model",
        ]
    ].copy()
    records.to_parquet(
        inventory_path, index=False, engine="pyarrow", compression="zstd"
    )
    cohort_manifest.to_csv(
        working / "cohort_manifest.csv", index=False, lineterminator="\n"
    )
    fragment_audit.to_csv(
        working / "fragment_selection_audit.csv",
        index=False,
        lineterminator="\n",
    )
    label_audit.to_csv(
        working / "label_mapping_audit.csv",
        index=False,
        lineterminator="\n",
    )
    summary = {
        "schema_version": "nucpred.mayr-pure-alpb-node-xtb-inventory-summary.v1",
        "dataset_id": config["dataset_id"],
        "record_count": len(records),
        "site_supervised_record_count": observed["site_supervised"],
        "n_only_record_count": len(records) - observed["site_supervised"],
        "h_group_record_count": observed["h_groups"],
        "spectator_stripped_record_count": observed["multifragment"],
        "single_fragment_record_count": int(
            records["spectator_stripped"].eq(False).sum()
        ),
        "model_formal_charge_counts": {
            str(key): int(value)
            for key, value in records["model_formal_charge"]
            .value_counts()
            .sort_index()
            .items()
        },
        "solvent_counts": {
            str(key): int(value)
            for key, value in records["solvent_raw"]
            .value_counts()
            .sort_index()
            .items()
        },
        "cohort_source_id_sha256": cohort_digest,
        "fragment_selection_target_independent": True,
        "label_mapping_target_independent": True,
    }
    atomic_write_json(working / "inventory_summary.json", summary, ensure_ascii=False)
    manifest = {
        "schema_version": "nucpred.mayr-pure-alpb-node-xtb-inventory.v1",
        "dataset_id": config["dataset_id"],
        "config_path": _display_path(config_file),
        "config_sha256": config_hash,
        "source_hashes": current_hashes,
        "record_count": len(records),
        "inventory_sha256": sha256_file(inventory_path),
        "cohort_source_id_sha256": cohort_digest,
        "rdkit_version": rdBase.rdkitVersion,
        "assets": {
            name: sha256_file(working / name)
            for name in (
                "cohort_manifest.csv",
                "fragment_selection_audit.csv",
                "label_mapping_audit.csv",
                "inventory_summary.json",
            )
        },
    }
    atomic_write_json(manifest_path, manifest, ensure_ascii=False)
    return {
        "status": "built",
        "record_count": len(records),
        "working_directory": _display_path(working),
        "inventory_sha256": manifest["inventory_sha256"],
    }


def _load_inventory(
    config_file: Path, config: Mapping[str, Any]
) -> tuple[pd.DataFrame, Path, dict[str, object]]:
    working = (ROOT / str(config["working_directory"])).resolve()
    inventory_path = working / "inventory.parquet"
    manifest_path = working / "inventory_manifest.json"
    if not inventory_path.is_file() or not manifest_path.is_file():
        build_inventory(config_path=config_file)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("config_sha256") != sha256_file(config_file)
        or manifest.get("source_hashes") != _source_hashes(config_file)
        or manifest.get("inventory_sha256") != sha256_file(inventory_path)
    ):
        raise DatasetBuildError(
            "Inventory source/config parity failed; do not mix long-run caches"
        )
    inventory = pd.read_parquet(inventory_path)
    if (
        len(inventory) != EXPECTED_RECORDS
        or inventory["source_id"].nunique() != EXPECTED_RECORDS
    ):
        raise DatasetBuildError("Inventory identity changed")
    return inventory, working, manifest


def _canonical_xyz(
    atomic_numbers: Sequence[int],
    positions: Sequence[Sequence[float]],
    *,
    comment: str,
    decimals: int,
) -> str:
    if len(atomic_numbers) != len(positions):
        raise ValueError("Atomic-number and coordinate lengths differ")
    periodic_table = Chem.GetPeriodicTable()
    rows = [str(len(atomic_numbers)), str(comment)]
    for atomic_number, position in zip(atomic_numbers, positions, strict=True):
        if len(position) != 3 or not all(
            math.isfinite(float(value)) for value in position
        ):
            raise ValueError("Geometry contains invalid coordinates")
        rows.append(
            f"{periodic_table.GetElementSymbol(int(atomic_number)):<2s} "
            f"{float(position[0]): .{decimals}f} "
            f"{float(position[1]): .{decimals}f} "
            f"{float(position[2]): .{decimals}f}"
        )
    return "\n".join(rows) + "\n"


def _force_field_results(
    molecule: Chem.Mol,
    *,
    force_field: str,
    max_iterations: int,
) -> list[tuple[int, float]]:
    if force_field == "MMFF94s":
        return [
            (int(status), float(energy))
            for status, energy in AllChem.MMFFOptimizeMoleculeConfs(
                molecule,
                numThreads=1,
                maxIters=max_iterations,
                mmffVariant="MMFF94s",
            )
        ]
    if force_field == "UFF":
        return [
            (int(status), float(energy))
            for status, energy in AllChem.UFFOptimizeMoleculeConfs(
                molecule,
                numThreads=1,
                maxIters=max_iterations,
            )
        ]
    raise ValueError(f"Unsupported force field {force_field!r}")


def _geometry_record(
    row: Mapping[str, object],
    *,
    config: Mapping[str, Any],
    config_sha256: str,
    source_hashes: Mapping[str, str],
) -> dict[str, object]:
    started = time.perf_counter()
    section = config["geometry"]
    source_count = int(row["model_source_atom_count"])
    source = _molecule_from_mapped_smiles(
        str(row["model_mapped_smiles"]), source_count
    )
    explicit = Chem.AddHs(Chem.Mol(source), addCoords=False)
    atomic_numbers = tuple(atom.GetAtomicNum() for atom in explicit.GetAtoms())
    expected_numbers = tuple(
        _parse_indices(row["model_atomic_numbers_json"])
    )
    if atomic_numbers != expected_numbers:
        raise DatasetBuildError(
            f"{row['source_id']}: G1 atom order differs from inventory"
        )
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = int(row["geometry_seed"])
    parameters.pruneRmsThresh = float(
        section["prune_rms_threshold_angstrom"]
    )
    parameters.enforceChirality = bool(section["enforce_chirality"])
    parameters.useSmallRingTorsions = bool(
        section["use_small_ring_torsions"]
    )
    parameters.useMacrocycleTorsions = bool(
        section["use_macrocycle_torsions"]
    )
    parameters.numThreads = 1
    conformer_ids = tuple(
        int(value)
        for value in AllChem.EmbedMultipleConfs(
            explicit,
            numConfs=int(section["conformer_count"]),
            params=parameters,
        )
    )
    if not conformer_ids:
        raise DatasetBuildError(
            f"{row['source_id']}: ETKDGv3 generated no conformers"
        )
    if AllChem.MMFFHasAllMoleculeParams(explicit):
        force_field = "MMFF94s"
        fallback_reason = None
    elif bool(section["allow_force_field_fallback"]) and (
        AllChem.UFFHasAllMoleculeParams(explicit)
    ):
        force_field = "UFF"
        fallback_reason = "MMFF94s_parameters_unavailable"
    else:
        raise DatasetBuildError(
            f"{row['source_id']}: no permitted force field has all parameters"
        )
    results = _force_field_results(
        explicit,
        force_field=force_field,
        max_iterations=int(section["force_field_max_iterations"]),
    )
    if len(results) != len(conformer_ids):
        raise DatasetBuildError("Force-field result count changed")
    eligible = [
        (conf_id, status, energy)
        for conf_id, (status, energy) in zip(
            conformer_ids, results, strict=True
        )
        if math.isfinite(energy)
        and (
            status == 0
            or not bool(section["require_force_field_convergence"])
        )
    ]
    if not eligible:
        statuses = [status for status, _ in results]
        raise DatasetBuildError(
            f"{row['source_id']}: no converged {force_field} conformer "
            f"(statuses={statuses})"
        )
    selected_conf, selected_status, selected_energy = min(
        eligible, key=lambda item: (item[2], item[0])
    )
    conformer = explicit.GetConformer(selected_conf)
    positions = tuple(
        (
            float(conformer.GetAtomPosition(index).x),
            float(conformer.GetAtomPosition(index).y),
            float(conformer.GetAtomPosition(index).z),
        )
        for index in range(explicit.GetNumAtoms())
    )
    xyz = _canonical_xyz(
        atomic_numbers,
        positions,
        comment=(
            f"{row['source_id']} G1 ETKDGv3 {force_field} "
            "fixed-geometry xTB input"
        ),
        decimals=int(section["coordinate_precision_decimals"]),
    )
    return {
        "schema_version": "nucpred.mayr-g1-geometry-cache.v1",
        "source_id": str(row["source_id"]),
        "cohort_index": int(row["cohort_index"]),
        "status": "success",
        "config_sha256": config_sha256,
        "source_hashes": dict(source_hashes),
        "model_graph_sha256": str(row["model_graph_sha256"]),
        "model_canonical_smiles": str(row["model_canonical_smiles"]),
        "model_formal_charge": int(row["model_formal_charge"]),
        "derived_random_seed": int(row["geometry_seed"]),
        "method": "ETKDGv3",
        "force_field": force_field,
        "fallback_reason": fallback_reason,
        "requested_conformer_count": int(section["conformer_count"]),
        "embedded_conformer_count": len(conformer_ids),
        "converged_conformer_count": len(eligible),
        "selected_conformer_id": selected_conf,
        "convergence_code": selected_status,
        "selected_energy_kcal_mol": selected_energy,
        "atomic_numbers": list(atomic_numbers),
        "positions_angstrom": positions,
        "xyz_sha256": hashlib.sha256(xyz.encode("utf-8")).hexdigest(),
        "xyz_text": xyz,
        "wall_seconds": time.perf_counter() - started,
        "rdkit_version": rdBase.rdkitVersion,
    }


def _failed_geometry_record(
    row: Mapping[str, object],
    error: BaseException,
    *,
    config_sha256: str,
    source_hashes: Mapping[str, str],
) -> dict[str, object]:
    return {
        "schema_version": "nucpred.mayr-g1-geometry-cache.v1",
        "source_id": str(row["source_id"]),
        "cohort_index": int(row["cohort_index"]),
        "status": "failed",
        "config_sha256": config_sha256,
        "source_hashes": dict(source_hashes),
        "model_graph_sha256": str(row["model_graph_sha256"]),
        "error_type": type(error).__name__,
        "error": str(error),
    }


def _preflight_source_ids(inventory: pd.DataFrame, maximum: int) -> list[str]:
    ordered = inventory.sort_values("cohort_index").copy()
    mandatory: list[str] = ordered.loc[
        ordered["spectator_stripped"].eq(True), "source_id"
    ].astype(str).tolist()
    mandatory.extend(
        ordered.loc[
            ordered["model_radical_electrons"].gt(0), "source_id"
        ].astype(str)
    )
    for column in (
        "solvent_raw",
        "model_formal_charge",
        "supervision_level",
        "legacy_site_target_kind",
    ):
        for _, group in ordered.groupby(column, sort=True, dropna=False):
            mandatory.append(str(group.iloc[0]["source_id"]))
    element_rows: dict[int, tuple[int, str]] = {}
    for row in ordered.itertuples(index=False):
        for atomic_number in set(
            _parse_indices(row.model_atomic_numbers_json)
        ):
            candidate = (int(row.model_all_atom_count), str(row.source_id))
            if atomic_number not in element_rows or (
                candidate < element_rows[atomic_number]
            ):
                element_rows[atomic_number] = candidate
    mandatory.extend(value[1] for value in element_rows.values())
    mandatory.extend(
        ordered.nlargest(6, "model_all_atom_count")["source_id"]
        .astype(str)
        .tolist()
    )
    chosen = list(dict.fromkeys(mandatory))
    if len(chosen) > maximum:
        raise DatasetBuildError(
            f"Preflight mandatory strata ({len(chosen)}) exceed max {maximum}"
        )
    if len(chosen) < maximum:
        remaining = ordered.loc[
            ~ordered["source_id"].astype(str).isin(set(chosen))
        ]
        needed = maximum - len(chosen)
        positions = np.linspace(
            0, max(0, len(remaining) - 1), needed, dtype=int
        )
        chosen.extend(
            str(remaining.iloc[index]["source_id"]) for index in positions
        )
    return list(dict.fromkeys(chosen))[:maximum]


def _cache_entry_matches(
    payload: Mapping[str, object],
    row: Mapping[str, object],
    *,
    schema_version: str,
    config_sha256: str,
    source_hashes: Mapping[str, str],
) -> bool:
    return (
        payload.get("schema_version") == schema_version
        and payload.get("source_id") == str(row["source_id"])
        and payload.get("config_sha256") == config_sha256
        and payload.get("source_hashes") == dict(source_hashes)
        and payload.get("model_graph_sha256")
        == str(row["model_graph_sha256"])
    )


def generate_geometries(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    scope: str = "preflight",
) -> dict[str, object]:
    if scope not in {"preflight", "full"}:
        raise ValueError("Geometry scope must be preflight or full")
    config_file = Path(config_path).resolve()
    config = _read_config(config_file)
    _resolve_inputs(config)
    inventory, working, _ = _load_inventory(config_file, config)
    config_hash = sha256_file(config_file)
    source_hashes = _source_hashes(config_file)
    if scope == "preflight":
        selected_ids = set(
            _preflight_source_ids(
                inventory, int(config["execution"]["preflight_max_records"])
            )
        )
        selected = inventory.loc[
            inventory["source_id"].astype(str).isin(selected_ids)
        ].copy()
    else:
        selected = inventory.copy()
    cache_directory = working / "geometry"
    cache_directory.mkdir(parents=True, exist_ok=True)
    pending: list[dict[str, object]] = []
    reused = 0
    for row in selected.to_dict(orient="records"):
        path = cache_directory / _cache_name(str(row["source_id"]))
        if path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                payload = {}
            if _cache_entry_matches(
                payload,
                row,
                schema_version="nucpred.mayr-g1-geometry-cache.v1",
                config_sha256=config_hash,
                source_hashes=source_hashes,
            ):
                reused += 1
                continue
            raise DatasetBuildError(
                f"Geometry cache parity mismatch: {row['source_id']}"
            )
        pending.append(row)

    def calculate(row: dict[str, object]) -> tuple[Path, dict[str, object]]:
        path = cache_directory / _cache_name(str(row["source_id"]))
        try:
            payload = _geometry_record(
                row,
                config=config,
                config_sha256=config_hash,
                source_hashes=source_hashes,
            )
        except BaseException as exc:
            payload = _failed_geometry_record(
                row,
                exc,
                config_sha256=config_hash,
                source_hashes=source_hashes,
            )
        return path, payload

    workers = min(int(config["execution"]["workers"]), max(1, len(pending)))
    completed = 0
    if pending:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(calculate, row) for row in pending]
            for future in as_completed(futures):
                path, payload = future.result()
                atomic_write_json(path, _json_safe(payload), ensure_ascii=False)
                completed += 1
                if completed % 25 == 0 or completed == len(pending):
                    print(
                        f"G1 {scope}: {completed}/{len(pending)} newly completed",
                        flush=True,
                    )
    payloads = [
        json.loads(
            (cache_directory / _cache_name(str(source_id))).read_text(
                encoding="utf-8"
            )
        )
        for source_id in selected["source_id"].astype(str)
    ]
    failures = [row for row in payloads if row.get("status") != "success"]
    summary = {
        "schema_version": "nucpred.mayr-g1-geometry-scope-summary.v1",
        "scope": scope,
        "requested_record_count": len(selected),
        "success_record_count": len(selected) - len(failures),
        "failure_record_count": len(failures),
        "new_record_count": completed,
        "reused_record_count": reused,
        "config_sha256": config_hash,
        "source_hashes": source_hashes,
        "failed_source_ids": sorted(str(row["source_id"]) for row in failures),
        "unsupported_force_field_policy": str(
            config["geometry"]["unsupported_force_field_policy"]
        ),
        "downstream_missingness_allowed": True,
    }
    atomic_write_json(
        working / f"geometry_{scope}_summary.json",
        summary,
        ensure_ascii=False,
    )
    return summary


def _gas_single_point(
    xyz_text: str,
    *,
    source_id: str,
    binary: Path,
    charge: int,
    uhf: int,
    timeout_seconds: int,
) -> tuple[xtb.PropertyResult, dict[str, object]]:
    execution = xtb._run_xtb(
        xyz_text,
        binary=binary,
        charge=charge,
        uhf=uhf,
        solvent=None,
        mode="",
        timeout_seconds=timeout_seconds,
    )
    atom_count = int(xyz_text.splitlines()[0].strip())
    result = xtb._parse_property(
        execution.stdout, atom_count, with_fukui=False
    )
    return (
        result,
        xtb._ledger_row(
            source_id=source_id,
            environment="gas",
            calculation="neutral_single_point",
            charge=charge,
            uhf=uhf,
            execution=execution,
        ),
    )


def _derived_xtb_features(
    *,
    gas_property: xtb.PropertyResult,
    alpb: xtb.EnvironmentResult,
    formal_charge: int,
    tce_homo_hartree: float,
    quality_control: Mapping[str, object],
) -> dict[str, object]:
    gas_qc = xtb._property_qc(
        gas_property,
        formal_charge=formal_charge,
        thresholds=quality_control,
    )
    property_qc = xtb._property_qc(
        alpb.property,
        formal_charge=formal_charge,
        thresholds=quality_control,
    )
    fukui_qc = xtb._fukui_qc(
        alpb.property,
        thresholds=quality_control,
    )
    vipea_qc = xtb._vipea_qc(alpb, thresholds=quality_control)
    cation_qc = xtb._cation_homo_qc(
        alpb, thresholds=quality_control
    )
    gas_environment = xtb.EnvironmentResult(
        property=gas_property,
        vip_ev=math.nan,
        vea_ev=math.nan,
        cation_homo_ev=math.nan,
    )
    response_qc = xtb._response_qc(
        alpb,
        gas=gas_environment,
        environment_property_qc=property_qc,
        gas_property_qc=gas_qc,
        thresholds=quality_control,
    )
    property_valid = property_qc.passed
    fukui_valid = property_valid and fukui_qc.passed
    vipea_valid = property_valid and vipea_qc.passed
    cation_valid = cation_qc.passed
    response_valid = response_qc.passed
    homo = (
        alpb.property.homo_ev / xtb.HARTREE_TO_EV
        if property_valid
        else math.nan
    )
    vip = (
        alpb.vip_ev / xtb.HARTREE_TO_EV
        if vipea_valid
        else math.nan
    )
    vea = (
        alpb.vea_ev / xtb.HARTREE_TO_EV
        if vipea_valid
        else math.nan
    )
    hardness = vip - vea
    softness = (
        1.0 / hardness
        if vipea_valid and math.isfinite(hardness) and hardness > 0.0
        else math.nan
    )
    nucleophilicity = (
        homo - float(tce_homo_hartree)
        if property_valid
        else math.nan
    )
    cation_homo = (
        alpb.cation_homo_ev / xtb.HARTREE_TO_EV
        if cation_valid
        else math.nan
    )
    delta_e = (
        alpb.property.energy_hartree - gas_property.energy_hartree
        if response_valid
        else math.nan
    )
    atom_count = len(alpb.property.cm5)
    fukui = np.full(atom_count, np.nan, dtype=float)
    cm5 = np.full(atom_count, np.nan, dtype=float)
    if fukui_valid:
        fukui = -np.asarray(alpb.property.fukui_minus_raw, dtype=float)
    if property_valid:
        cm5 = np.asarray(alpb.property.cm5, dtype=float)
    condensed_softness = (
        fukui * softness
        if math.isfinite(softness)
        else np.full(atom_count, np.nan, dtype=float)
    )
    condensed_nucleophilicity = (
        fukui * nucleophilicity * xtb.HARTREE_TO_EV
        if math.isfinite(nucleophilicity)
        else np.full(atom_count, np.nan, dtype=float)
    )
    local_values = np.stack(
        (fukui, cm5, condensed_softness, condensed_nucleophilicity),
        axis=1,
    )
    local_mask = np.isfinite(local_values)
    global_values = np.asarray(
        (homo, softness, vip, nucleophilicity, cation_homo, delta_e),
        dtype=float,
    )
    global_mask = np.isfinite(global_values)
    qc = {
        "gas_property": gas_qc,
        "alpb_property": property_qc,
        "alpb_fukui": fukui_qc,
        "alpb_vipea": vipea_qc,
        "alpb_cation_homo": cation_qc,
        "alpb_response": response_qc,
    }
    return {
        "local_values": local_values.tolist(),
        "local_mask": local_mask.tolist(),
        "global_values": global_values.tolist(),
        "global_mask": global_mask.tolist(),
        "complete_xtb10": bool(local_mask.all() and global_mask.all()),
        "qc": {
            name: {"passed": result.passed, "reason": result.reason}
            for name, result in qc.items()
        },
        "raw": {
            "gas_total_energy_hartree": gas_property.energy_hartree,
            "gas_homo_hartree": gas_property.homo_ev / xtb.HARTREE_TO_EV,
            "gas_lumo_hartree": gas_property.lumo_ev / xtb.HARTREE_TO_EV,
            "gas_cm5": gas_property.cm5,
            "alpb_total_energy_hartree": alpb.property.energy_hartree,
            "alpb_homo_hartree": (
                alpb.property.homo_ev / xtb.HARTREE_TO_EV
            ),
            "alpb_lumo_hartree": (
                alpb.property.lumo_ev / xtb.HARTREE_TO_EV
            ),
            "alpb_dipole_debye": alpb.property.dipole_debye,
            "alpb_cm5": alpb.property.cm5,
            "alpb_fukui_minus_raw": alpb.property.fukui_minus_raw,
            "alpb_fukui_minus_signed": (
                -np.asarray(alpb.property.fukui_minus_raw, dtype=float)
            ).tolist(),
            "alpb_vip_hartree": alpb.vip_ev / xtb.HARTREE_TO_EV,
            "alpb_vea_hartree": alpb.vea_ev / xtb.HARTREE_TO_EV,
            "alpb_cation_homo_hartree": (
                alpb.cation_homo_ev / xtb.HARTREE_TO_EV
            ),
        },
    }


def _xtb_record(
    row: Mapping[str, object],
    geometry: Mapping[str, object],
    *,
    config: Mapping[str, Any],
    config_sha256: str,
    source_hashes: Mapping[str, str],
    binary: Path,
    tce_homo_hartree: float,
) -> dict[str, object]:
    started = time.perf_counter()
    source_id = str(row["source_id"])
    atomic_numbers = tuple(int(value) for value in geometry["atomic_numbers"])
    if tuple(_parse_indices(row["model_atomic_numbers_json"])) != atomic_numbers:
        raise DatasetBuildError(f"{source_id}: xTB/G1 atom order mismatch")
    charge = int(row["model_formal_charge"])
    uhf, spin_source = xtb._validated_neutral_uhf(
        atomic_numbers,
        charge,
        int(row["model_radical_electrons"]),
    )
    cation_uhf = xtb._minimal_uhf(atomic_numbers, charge + 1)
    timeout_seconds = int(config["execution"]["timeout_seconds"])
    xyz_text = str(geometry["xyz_text"])
    gas_property, gas_ledger = _gas_single_point(
        xyz_text,
        source_id=source_id,
        binary=binary,
        charge=charge,
        uhf=uhf,
        timeout_seconds=timeout_seconds,
    )
    alpb, alpb_ledger = xtb._environment_calculations(
        xyz_text,
        source_id=source_id,
        binary=binary,
        charge=charge,
        uhf=uhf,
        cation_uhf=cation_uhf,
        solvent=str(row["xtb_alpb_solvent"]),
        timeout_seconds=timeout_seconds,
        atom_count=len(atomic_numbers),
    )
    derived = _derived_xtb_features(
        gas_property=gas_property,
        alpb=alpb,
        formal_charge=charge,
        tce_homo_hartree=tce_homo_hartree,
        quality_control=config["quality_control"],
    )
    return {
        "schema_version": "nucpred.mayr-node-xtb-cache.v1",
        "source_id": source_id,
        "cohort_index": int(row["cohort_index"]),
        "status": "success",
        "config_sha256": config_sha256,
        "source_hashes": dict(source_hashes),
        "model_graph_sha256": str(row["model_graph_sha256"]),
        "geometry_xyz_sha256": str(geometry["xyz_sha256"]),
        "formal_charge": charge,
        "neutral_uhf": uhf,
        "neutral_spin_source": spin_source,
        "cation_uhf": cation_uhf,
        "solvent": str(row["xtb_alpb_solvent"]),
        "tce_homo_hartree": tce_homo_hartree,
        "local_feature_names": LOCAL_FEATURES,
        "global_feature_names": GLOBAL_FEATURES,
        **derived,
        "ledger": [gas_ledger, *alpb_ledger],
        "wall_seconds": time.perf_counter() - started,
    }


def _failed_xtb_record(
    row: Mapping[str, object],
    geometry: Mapping[str, object],
    error: BaseException,
    *,
    config_sha256: str,
    source_hashes: Mapping[str, str],
) -> dict[str, object]:
    atom_count = int(row["model_all_atom_count"])
    return {
        "schema_version": "nucpred.mayr-node-xtb-cache.v1",
        "source_id": str(row["source_id"]),
        "cohort_index": int(row["cohort_index"]),
        "status": "failed",
        "config_sha256": config_sha256,
        "source_hashes": dict(source_hashes),
        "model_graph_sha256": str(row["model_graph_sha256"]),
        "geometry_xyz_sha256": str(geometry.get("xyz_sha256", "")),
        "local_feature_names": LOCAL_FEATURES,
        "global_feature_names": GLOBAL_FEATURES,
        "local_values": [[None] * len(LOCAL_FEATURES) for _ in range(atom_count)],
        "local_mask": [[False] * len(LOCAL_FEATURES) for _ in range(atom_count)],
        "global_values": [None] * len(GLOBAL_FEATURES),
        "global_mask": [False] * len(GLOBAL_FEATURES),
        "complete_xtb10": False,
        "error_type": type(error).__name__,
        "error": str(error),
        "ledger": [],
    }


def _load_or_build_tce(
    *,
    working: Path,
    config: Mapping[str, Any],
    config_sha256: str,
    source_hashes: Mapping[str, str],
    binary: Path,
) -> dict[str, object]:
    path = working / "tce_reference.json"
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("schema_version")
            != "nucpred.mayr-node-xtb-tce-reference.v1"
            or payload.get("config_sha256") != config_sha256
            or payload.get("source_hashes") != dict(source_hashes)
        ):
            raise DatasetBuildError("TCE reference cache source parity failed")
        return payload
    value, ledger, xyz = xtb._tce_reference(
        binary,
        smiles=str(config["xtb"]["tce_reference_smiles"]),
        timeout_seconds=int(config["execution"]["timeout_seconds"]),
    )
    payload = {
        "schema_version": "nucpred.mayr-node-xtb-tce-reference.v1",
        "config_sha256": config_sha256,
        "source_hashes": dict(source_hashes),
        "smiles": str(config["xtb"]["tce_reference_smiles"]),
        "environment": "gas",
        "homo_hartree": value,
        "geometry_xyz_sha256": hashlib.sha256(
            xyz.encode("utf-8")
        ).hexdigest(),
        "ledger": ledger,
    }
    atomic_write_json(path, _json_safe(payload), ensure_ascii=False)
    return payload


def _xtb_scope_summary(
    payloads: Sequence[Mapping[str, object]],
    *,
    scope: str,
    config_hash: str,
    source_hashes: Mapping[str, str],
    selected_ids: Sequence[str],
) -> dict[str, object]:
    successes = [row for row in payloads if row.get("status") == "success"]
    complete = [row for row in successes if row.get("complete_xtb10") is True]
    denominator = len(payloads)
    complete_fraction = len(complete) / denominator if denominator else 0.0
    qc_counts: dict[str, int] = {}
    for row in successes:
        for name, audit in dict(row.get("qc", {})).items():
            if not dict(audit).get("passed"):
                qc_counts[name] = qc_counts.get(name, 0) + 1
    return {
        "schema_version": "nucpred.mayr-node-xtb-scope-summary.v1",
        "scope": scope,
        "requested_record_count": denominator,
        "execution_success_record_count": len(successes),
        "execution_failure_record_count": denominator - len(successes),
        "complete_xtb10_record_count": len(complete),
        "complete_xtb10_fraction": complete_fraction,
        "minimum_complete_xtb10_fraction": None,
        "coverage_gate_pass": None,
        "config_sha256": config_hash,
        "source_hashes": dict(source_hashes),
        "selected_source_id_sha256": hashlib.sha256(
            ("\n".join(sorted(map(str, selected_ids))) + "\n").encode("utf-8")
        ).hexdigest(),
        "failed_source_ids": sorted(
            str(row["source_id"])
            for row in payloads
            if row.get("status") != "success"
        ),
        "incomplete_source_ids": sorted(
            str(row["source_id"])
            for row in payloads
            if row.get("complete_xtb10") is not True
        ),
        "failed_qc_counts": dict(sorted(qc_counts.items())),
    }


def generate_xtb_features(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    scope: str = "preflight",
) -> dict[str, object]:
    if scope not in {"preflight", "full"}:
        raise ValueError("xTB scope must be preflight or full")
    config_file = Path(config_path).resolve()
    config = _read_config(config_file)
    _, _, _, archive = _resolve_inputs(config)
    inventory, working, _ = _load_inventory(config_file, config)
    config_hash = sha256_file(config_file)
    source_hashes = _source_hashes(config_file)
    threshold = float(config["minimum_complete_xtb10_fraction"])
    if scope == "full":
        preflight_path = working / "xtb_preflight_summary.json"
        if not preflight_path.is_file():
            raise DatasetBuildError("Full xTB run requires a passing preflight")
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        if (
            preflight.get("config_sha256") != config_hash
            or preflight.get("source_hashes") != source_hashes
            or preflight.get("coverage_gate_pass") is not True
        ):
            raise DatasetBuildError("xTB preflight is absent, stale, or failed")
    if scope == "preflight":
        selected_ids = _preflight_source_ids(
            inventory, int(config["execution"]["preflight_max_records"])
        )
        selected_set = set(selected_ids)
        selected = inventory.loc[
            inventory["source_id"].astype(str).isin(selected_set)
        ].copy()
    else:
        selected = inventory.copy()
        selected_ids = selected["source_id"].astype(str).tolist()
    generate_geometries(config_path=config_file, scope=scope)
    geometry_directory = working / "geometry"
    cache_directory = working / "xtb"
    cache_directory.mkdir(parents=True, exist_ok=True)
    geometries = {
        str(row["source_id"]): json.loads(
            (
                geometry_directory
                / _cache_name(str(row["source_id"]))
            ).read_text(encoding="utf-8")
        )
        for row in selected.to_dict(orient="records")
    }
    with tempfile.TemporaryDirectory(
        prefix="nucpred_mayr_node_xtb_distribution_"
    ) as raw_distribution:
        binary = xtb._safe_extract_archive(archive, Path(raw_distribution))
        tce = _load_or_build_tce(
            working=working,
            config=config,
            config_sha256=config_hash,
            source_hashes=source_hashes,
            binary=binary,
        )
        pending: list[dict[str, object]] = []
        reused = 0
        for row in selected.to_dict(orient="records"):
            path = cache_directory / _cache_name(str(row["source_id"]))
            if path.is_file():
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError):
                    payload = {}
                if _cache_entry_matches(
                    payload,
                    row,
                    schema_version="nucpred.mayr-node-xtb-cache.v1",
                    config_sha256=config_hash,
                    source_hashes=source_hashes,
                ) and payload.get("geometry_xyz_sha256") == geometries[
                    str(row["source_id"])
                ].get("xyz_sha256"):
                    reused += 1
                    continue
                raise DatasetBuildError(
                    f"xTB cache parity mismatch: {row['source_id']}"
                )
            pending.append(row)

        def calculate(
            row: dict[str, object],
        ) -> tuple[Path, dict[str, object]]:
            source_id = str(row["source_id"])
            geometry = geometries[source_id]
            path = cache_directory / _cache_name(source_id)
            try:
                if geometry.get("status") != "success":
                    raise DatasetBuildError(
                        f"{source_id}: G1 unavailable; xTB features are masked"
                    )
                payload = _xtb_record(
                    row,
                    geometry,
                    config=config,
                    config_sha256=config_hash,
                    source_hashes=source_hashes,
                    binary=binary,
                    tce_homo_hartree=float(tce["homo_hartree"]),
                )
            except BaseException as exc:
                payload = _failed_xtb_record(
                    row,
                    geometry,
                    exc,
                    config_sha256=config_hash,
                    source_hashes=source_hashes,
                )
            return path, payload

        workers = min(
            int(config["execution"]["workers"]), max(1, len(pending))
        )
        completed = 0
        if pending:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(calculate, row) for row in pending]
                for future in as_completed(futures):
                    path, payload = future.result()
                    atomic_write_json(
                        path, _json_safe(payload), ensure_ascii=False
                    )
                    completed += 1
                    if completed % 10 == 0 or completed == len(pending):
                        print(
                            f"xTB {scope}: {completed}/{len(pending)} "
                            "new records completed",
                            flush=True,
                        )
    payloads = [
        json.loads(
            (cache_directory / _cache_name(source_id)).read_text(
                encoding="utf-8"
            )
        )
        for source_id in selected["source_id"].astype(str)
    ]
    summary = _xtb_scope_summary(
        payloads,
        scope=scope,
        config_hash=config_hash,
        source_hashes=source_hashes,
        selected_ids=selected_ids,
    )
    summary["new_record_count"] = completed
    summary["reused_record_count"] = reused
    summary["minimum_complete_xtb10_fraction"] = threshold
    summary["coverage_gate_pass"] = (
        float(summary["complete_xtb10_fraction"]) >= threshold
    )
    atomic_write_json(
        working / f"xtb_{scope}_summary.json",
        summary,
        ensure_ascii=False,
    )
    if not summary["coverage_gate_pass"]:
        raise DatasetBuildError(
            f"xTB {scope} complete coverage "
            f"{summary['complete_xtb10_fraction']:.3%} is below "
            f"{threshold:.1%}; stop for user decision"
        )
    return summary


def _repair_canonical_splits(
    *,
    records: pd.DataFrame,
    parent_split_path: Path,
    config: Mapping[str, Any],
) -> dict[str, object]:
    parent = json.loads(parent_split_path.read_text(encoding="utf-8"))
    source_ids = set(records["source_id"].astype(str))
    by_source = records.set_index("source_id", drop=False)
    runs: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    for parent_run in parent["runs"]:
        if str(parent_run["split"]) != "canonical_smiles_group":
            continue
        train = set(map(str, parent_run["train_source_ids"])) & source_ids
        test = set(map(str, parent_run["test_source_ids"])) & source_ids
        if train & test or train | test != source_ids:
            raise DatasetBuildError("Filtered parent canonical split lost identity")
        repairs: list[dict[str, object]] = []
        train_groups = set(
            by_source.loc[
                sorted(train), "model_canonical_smiles"
            ].astype(str)
        )
        test_groups = set(
            by_source.loc[
                sorted(test), "model_canonical_smiles"
            ].astype(str)
        )
        for group in sorted(train_groups & test_groups):
            members = set(
                by_source.loc[
                    by_source["model_canonical_smiles"].astype(str).eq(group),
                    "source_id",
                ].astype(str)
            )
            train_count = len(members & train)
            test_count = len(members & test)
            destination = "test" if test_count > train_count else "train"
            train -= members
            test -= members
            if destination == "train":
                train |= members
            else:
                test |= members
            repairs.append(
                {
                    "model_canonical_smiles": group,
                    "source_ids": sorted(members),
                    "parent_train_count": train_count,
                    "parent_test_count": test_count,
                    "destination": destination,
                    "used_target_fields": False,
                }
            )
        train_groups = set(
            by_source.loc[
                sorted(train), "model_canonical_smiles"
            ].astype(str)
        )
        test_groups = set(
            by_source.loc[
                sorted(test), "model_canonical_smiles"
            ].astype(str)
        )
        if train & test or train | test != source_ids or train_groups & test_groups:
            raise DatasetBuildError("Post-stripping canonical split repair failed")
        seed = int(parent_run["seed"])
        run = {
            "split": "canonical_smiles_group",
            "seed": seed,
            "status": "ok",
            "train_source_ids": sorted(train),
            "test_source_ids": sorted(test),
            "dropped_source_ids": [],
            "summary": {
                "group_column": "model_canonical_smiles",
                "train_records": len(train),
                "test_records": len(test),
                "train_groups": len(train_groups),
                "test_groups": len(test_groups),
                "model_canonical_smiles_overlap": 0,
                "repair_count": len(repairs),
            },
            "repairs": repairs,
        }
        runs.append(run)
        audits.append(
            {
                "split": "canonical_smiles_group",
                "seed": seed,
                "train_records": len(train),
                "test_records": len(test),
                "source_id_overlap": 0,
                "model_canonical_smiles_overlap": 0,
                "repaired_group_count": len(repairs),
                "repaired_source_id_count": sum(
                    len(item["source_ids"]) for item in repairs
                ),
                "repair_target_independent": True,
                "status": "pass",
            }
        )
    if len(runs) != 5:
        raise DatasetBuildError("Expected five canonical parent split seeds")
    return {
        "schema_version": "nucpred.mayr-pure-alpb-node-xtb-splits.v1",
        "dataset_id": config["dataset_id"],
        "parent_manifest": _display_path(parent_split_path),
        "parent_manifest_sha256": sha256_file(parent_split_path),
        "population_policy": "filter_parent_membership_to_fixed_1136_source_ids",
        "group_identity": "post_spectator_stripping_model_canonical_smiles",
        "overlap_repair": (
            "complete_group_to_parent_majority_side_tie_to_train"
        ),
        "repair_may_use_target_fields": False,
        "split_names": ["canonical_smiles_group"],
        "seeds": [int(run["seed"]) for run in runs],
        "runs": runs,
        "audits": audits,
        "invariants": {
            "record_count": len(records),
            "source_id_parity": True,
            "source_id_overlap_zero": True,
            "post_stripping_group_overlap_zero": True,
            "repair_target_independent": True,
        },
    }


def _dataset_tables(
    inventory: pd.DataFrame,
    *,
    working: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    record_rows: list[dict[str, object]] = []
    atom_rows: list[dict[str, object]] = []
    molecule_rows: list[dict[str, object]] = []
    primitive_rows: list[dict[str, object]] = []
    geometry_ledger: list[dict[str, object]] = []
    calculation_ledger: list[dict[str, object]] = []
    geometry_directory = working / "geometry"
    xtb_directory = working / "xtb"
    for parent_row in inventory.to_dict(orient="records"):
        source_id = str(parent_row["source_id"])
        geometry = json.loads(
            (geometry_directory / _cache_name(source_id)).read_text(
                encoding="utf-8"
            )
        )
        electronic = json.loads(
            (xtb_directory / _cache_name(source_id)).read_text(
                encoding="utf-8"
            )
        )
        atom_count = int(parent_row["model_all_atom_count"])
        atomic_numbers = _parse_indices(
            parent_row["model_atomic_numbers_json"]
        )
        node_categorical = json.loads(
            str(parent_row["model_node_categorical_json"])
        )
        geometry_success = geometry.get("status") == "success"
        positions = (
            geometry["positions_angstrom"]
            if geometry_success
            else [[math.nan, math.nan, math.nan] for _ in range(atom_count)]
        )
        local_values = _float_array(
            electronic["local_values"], atom_count * len(LOCAL_FEATURES)
        ).reshape(atom_count, len(LOCAL_FEATURES))
        local_mask_raw = electronic["local_mask"]
        if (
            not isinstance(local_mask_raw, list)
            or len(local_mask_raw) != atom_count
        ):
            raise DatasetBuildError(f"{source_id}: invalid local mask")
        local_mask = np.asarray(local_mask_raw, dtype=bool)
        global_values = _float_array(
            electronic["global_values"], len(GLOBAL_FEATURES)
        )
        global_mask = np.asarray(
            electronic["global_mask"], dtype=bool
        )
        if (
            len(atomic_numbers) != atom_count
            or len(node_categorical) != atom_count
            or len(positions) != atom_count
            or local_mask.shape
            != (atom_count, len(LOCAL_FEATURES))
            or global_mask.shape != (len(GLOBAL_FEATURES),)
        ):
            raise DatasetBuildError(f"{source_id}: finalized tensor shape changed")
        for atom_index in range(atom_count):
            row = {
                "source_id": source_id,
                "atom_index": atom_index,
                "atomic_number": int(atomic_numbers[atom_index]),
                "element": Chem.GetPeriodicTable().GetElementSymbol(
                    int(atomic_numbers[atom_index])
                ),
                "is_hydrogen": int(atomic_numbers[atom_index]) == 1,
                "position_x_angstrom": float(positions[atom_index][0]),
                "position_y_angstrom": float(positions[atom_index][1]),
                "position_z_angstrom": float(positions[atom_index][2]),
            }
            for feature_index, name in enumerate(
                NODE_CATEGORICAL_FEATURES
            ):
                row[f"rdkit_{name}"] = int(
                    node_categorical[atom_index][feature_index]
                )
            for feature_index, name in enumerate(LOCAL_FEATURES):
                row[name] = float(local_values[atom_index, feature_index])
                row[f"{name}__available"] = bool(
                    local_mask[atom_index, feature_index]
                )
            atom_rows.append(row)
        molecule_row: dict[str, object] = {
            "source_id": source_id,
            "model_canonical_smiles": str(
                parent_row["model_canonical_smiles"]
            ),
            "model_formal_charge": int(parent_row["model_formal_charge"]),
            "solvent_raw": str(parent_row["solvent_raw"]),
            "xtb_alpb_solvent": str(parent_row["xtb_alpb_solvent"]),
            "complete_xtb10": bool(electronic["complete_xtb10"]),
        }
        for name in SOLVENT_DESCRIPTOR_COLUMNS:
            molecule_row[name] = float(parent_row[name])
        for feature_index, name in enumerate(GLOBAL_FEATURES):
            molecule_row[name] = float(global_values[feature_index])
            molecule_row[f"{name}__available"] = bool(
                global_mask[feature_index]
            )
        molecule_rows.append(molecule_row)
        primitive_rows.append(
            {
                "source_id": source_id,
                "status": str(electronic["status"]),
                "raw_primitives_json": _json_compact(
                    electronic.get("raw", {})
                ),
                "qc_json": _json_compact(electronic.get("qc", {})),
                "geometry_xyz_sha256": str(
                    geometry.get("xyz_sha256", "")
                ),
                "xtb_complete": bool(electronic["complete_xtb10"]),
                "error_type": str(electronic.get("error_type", "")),
                "error": str(electronic.get("error", "")),
            }
        )
        if geometry_success:
            geometry_ledger.append(
                {
                    "source_id": source_id,
                    "method": str(geometry["method"]),
                    "force_field": str(geometry["force_field"]),
                    "fallback_reason": geometry.get("fallback_reason"),
                    "derived_random_seed": int(
                        geometry["derived_random_seed"]
                    ),
                    "embedded_conformer_count": int(
                        geometry["embedded_conformer_count"]
                    ),
                    "converged_conformer_count": int(
                        geometry["converged_conformer_count"]
                    ),
                    "selected_conformer_id": int(
                        geometry["selected_conformer_id"]
                    ),
                    "selected_energy_kcal_mol": float(
                        geometry["selected_energy_kcal_mol"]
                    ),
                    "xyz_sha256": str(geometry["xyz_sha256"]),
                    "wall_seconds": float(geometry["wall_seconds"]),
                    "error_type": "",
                    "error": "",
                    "status": "success",
                }
            )
        else:
            geometry_ledger.append(
                {
                    "source_id": source_id,
                    "method": "ETKDGv3",
                    "force_field": "",
                    "fallback_reason": "",
                    "derived_random_seed": int(
                        parent_row["geometry_seed"]
                    ),
                    "embedded_conformer_count": 0,
                    "converged_conformer_count": 0,
                    "selected_conformer_id": -1,
                    "selected_energy_kcal_mol": math.nan,
                    "xyz_sha256": "",
                    "wall_seconds": math.nan,
                    "error_type": str(geometry.get("error_type", "")),
                    "error": str(geometry.get("error", "")),
                    "status": "failed",
                }
            )
        for ledger_row in electronic.get("ledger", []):
            calculation_ledger.append(
                {"source_id": source_id, **dict(ledger_row)}
            )
        record_rows.append(
            {
                **parent_row,
                "g1_status": str(geometry["status"]),
                "g1_force_field": str(geometry.get("force_field", "")),
                "g1_fallback_reason": geometry.get("fallback_reason"),
                "g1_failure_reason": str(geometry.get("error", "")),
                "g1_selected_energy_kcal_mol": (
                    float(geometry["selected_energy_kcal_mol"])
                    if geometry_success
                    else math.nan
                ),
                "g1_positions_angstrom_json": _json_compact(
                    _json_safe(positions)
                ),
                "g1_xyz_sha256": str(geometry.get("xyz_sha256", "")),
                "node_local4_json": _json_compact(
                    _json_safe(local_values.tolist())
                ),
                "node_local4_available_json": _json_compact(
                    local_mask.tolist()
                ),
                "molecule_global6_json": _json_compact(
                    _json_safe(global_values.tolist())
                ),
                "molecule_global6_available_json": _json_compact(
                    global_mask.tolist()
                ),
                "complete_xtb10": bool(electronic["complete_xtb10"]),
                "electronic_cache_status": str(electronic["status"]),
            }
        )
    tce_path = working / "tce_reference.json"
    if tce_path.is_file():
        tce = json.loads(tce_path.read_text(encoding="utf-8"))
        calculation_ledger.append(
            {
                "source_id": "reference:tetracyanoethylene",
                **dict(tce["ledger"]),
            }
        )
    return (
        pd.DataFrame(record_rows),
        pd.DataFrame(atom_rows),
        pd.DataFrame(molecule_rows),
        pd.DataFrame(primitive_rows),
        pd.DataFrame(geometry_ledger),
        pd.DataFrame(calculation_ledger),
    )


def _coverage_payload(
    records: pd.DataFrame,
    atom_features: pd.DataFrame,
    molecule_features: pd.DataFrame,
    *,
    threshold: float,
) -> dict[str, object]:
    record_fraction = float(records["complete_xtb10"].mean())
    local = {
        name: {
            "available_atoms": int(
                atom_features[f"{name}__available"].sum()
            ),
            "total_atoms": len(atom_features),
            "fraction": float(
                atom_features[f"{name}__available"].mean()
            ),
        }
        for name in LOCAL_FEATURES
    }
    global_values = {
        name: {
            "available_records": int(
                molecule_features[f"{name}__available"].sum()
            ),
            "total_records": len(molecule_features),
            "fraction": float(
                molecule_features[f"{name}__available"].mean()
            ),
        }
        for name in GLOBAL_FEATURES
    }
    return {
        "schema_version": "nucpred.mayr-node-xtb-coverage.v1",
        "record_count": len(records),
        "atom_count": len(atom_features),
        "complete_xtb10_record_count": int(
            records["complete_xtb10"].sum()
        ),
        "complete_xtb10_fraction": record_fraction,
        "minimum_complete_xtb10_fraction": threshold,
        "coverage_gate_pass": record_fraction >= threshold,
        "local_feature_coverage": local,
        "global_feature_coverage": global_values,
        "missing_value_policy": (
            "fold_train_median_imputation_plus_availability_masks"
        ),
    }


def _file_entry(path: Path, role: str, format_name: str) -> dict[str, object]:
    return {
        "path": path.name,
        "role": role,
        "format": format_name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def finalize_dataset(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    output_directory: str | Path | None = None,
) -> dict[str, object]:
    config_file = Path(config_path).resolve()
    config = _read_config(config_file)
    _, parent_split_path, _, _ = _resolve_inputs(config)
    inventory, working, inventory_manifest = _load_inventory(
        config_file, config
    )
    full_summary_path = working / "xtb_full_summary.json"
    if not full_summary_path.is_file():
        raise DatasetBuildError("Finalize requires a passing full xTB scope")
    full_summary = json.loads(full_summary_path.read_text(encoding="utf-8"))
    current_sources = _source_hashes(config_file)
    if (
        full_summary.get("config_sha256") != sha256_file(config_file)
        or full_summary.get("source_hashes") != current_sources
        or full_summary.get("coverage_gate_pass") is not True
        or int(full_summary.get("requested_record_count", -1))
        != EXPECTED_RECORDS
    ):
        raise DatasetBuildError("Full xTB summary is stale, partial, or failed")
    output = (
        Path(output_directory).resolve()
        if output_directory is not None
        else (ROOT / str(config["output_directory"])).resolve()
    )
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite versioned node-xTB dataset: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    (
        records,
        atom_features,
        molecule_features,
        primitives,
        geometry_ledger,
        calculation_ledger,
    ) = _dataset_tables(inventory, working=working)
    threshold = float(config["minimum_complete_xtb10_fraction"])
    coverage = _coverage_payload(
        records,
        atom_features,
        molecule_features,
        threshold=threshold,
    )
    if coverage["coverage_gate_pass"] is not True:
        raise DatasetBuildError("Final xTB coverage fell below the frozen gate")
    split_manifest = _repair_canonical_splits(
        records=records,
        parent_split_path=parent_split_path,
        config=config,
    )
    source_ids = sorted(records["source_id"].astype(str))
    source_parity = {
        "schema_version": "nucpred.mayr-node-xtb-source-parity.v1",
        "record_count": len(source_ids),
        "unique_source_id_count": len(set(source_ids)),
        "source_id_sha256": hashlib.sha256(
            ("\n".join(source_ids) + "\n").encode("utf-8")
        ).hexdigest(),
        "expected_source_id_sha256": str(
            config["cohort_source_id_sha256"]
        ),
        "parity": (
            hashlib.sha256(
                ("\n".join(source_ids) + "\n").encode("utf-8")
            ).hexdigest()
            == str(config["cohort_source_id_sha256"])
        ),
    }
    if not source_parity["parity"]:
        raise DatasetBuildError("Final source-id parity failed")
    failure_values: list[dict[str, object]] = []
    for row in geometry_ledger.loc[
        geometry_ledger["status"].eq("failed")
    ].to_dict(orient="records"):
        failure_values.append(
            {
                "source_id": str(row["source_id"]),
                "stage": "geometry",
                "calculation": "G1_ETKDGv3_MMFF94s_or_UFF",
                "environment": "force_field",
                "error_type": str(row["error_type"]),
                "error": str(row["error"]),
            }
        )
    for row in primitives.loc[
        primitives["status"].eq("failed")
    ].to_dict(orient="records"):
        failure_values.append(
            {
                "source_id": str(row["source_id"]),
                "stage": "xtb_record",
                "calculation": "fixed_G1_descriptor_panel",
                "environment": "gas_and_alpb",
                "error_type": str(row["error_type"]),
                "error": str(row["error"]),
            }
        )
    for row in calculation_ledger.loc[
        calculation_ledger["normal_termination"].eq(False)
    ].to_dict(orient="records"):
        failure_values.append(
            {
                "source_id": str(row["source_id"]),
                "stage": "xtb_subcalculation",
                "calculation": str(row["calculation"]),
                "environment": str(row["environment"]),
                "error_type": "subcalculation_failure",
                "error": str(row["error"]),
            }
        )
    failure_rows = pd.DataFrame(
        failure_values,
        columns=(
            "source_id",
            "stage",
            "calculation",
            "environment",
            "error_type",
            "error",
        ),
    )
    validation = pd.DataFrame(columns=VALIDATION_COLUMNS)
    feature_schema = {
        "schema_version": "nucpred.mayr-node-xtb-feature-schema.v1",
        "element_vocabulary": ELEMENT_VOCABULARY,
        "hydrogen_policy": "ordinary_element_in_shared_vocabulary",
        "node_categorical_features": NODE_CATEGORICAL_FEATURES,
        "node_category_sizes": NODE_CATEGORY_SIZES,
        "edge_categorical_features": EDGE_CATEGORICAL_FEATURES,
        "edge_category_sizes": EDGE_CATEGORY_SIZES,
        "node_local_features": LOCAL_FEATURES,
        "node_local_entry_point": "before_message_passing",
        "global_xtb_features": GLOBAL_FEATURES,
        "solvent_descriptor_features": SOLVENT_DESCRIPTOR_COLUMNS,
        "model_charge_field": "model_formal_charge",
        "availability_mask_for_every_xtb_feature": True,
        "forbidden_oracle_inputs": (
            "nuc_index",
            "site_target_atoms",
            "site_target_distribution",
            "donor_heavy_atom_indices",
            "hydrogen_candidate_indices",
        ),
    }
    summary = {
        "schema_version": "nucpred.mayr-node-xtb-summary.v1",
        "dataset_id": config["dataset_id"],
        "record_count": len(records),
        "atom_count": len(atom_features),
        "hydrogen_atom_count": int(
            atom_features["is_hydrogen"].sum()
        ),
        "site_supervised_record_count": int(
            records["site_target_mask_model"].sum()
        ),
        "n_only_record_count": int(
            records["site_target_mask_model"].eq(False).sum()
        ),
        "h_group_record_count": int(
            records["supervision_level"]
            .eq("equivalent_or_indistinguishable_h_group")
            .sum()
        ),
        "spectator_stripped_record_count": int(
            records["spectator_stripped"].sum()
        ),
        "single_fragment_sensitivity_record_count": int(
            records["spectator_stripped"].eq(False).sum()
        ),
        "complete_xtb10_record_count": int(
            records["complete_xtb10"].sum()
        ),
        "complete_xtb10_fraction": float(
            records["complete_xtb10"].mean()
        ),
        "failed_subcalculation_count": len(failure_rows),
        "hard_validation_error_count": 0,
        "source_id_parity": True,
        "post_stripping_split_group_overlap_zero": True,
        "geometry": "G1 ETKDGv3 20-conformer MMFF94s/UFF",
        "electronic_method": "fixed-G1 GFN1-xTB gas + pure-solvent ALPB",
        "pretraining": False,
    }
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{config['dataset_id']}.staging-",
            dir=output.parent,
        )
    )
    try:
        files: list[tuple[Path, str, str]] = []

        def parquet(
            name: str, frame: pd.DataFrame, role: str
        ) -> None:
            path = staging / name
            frame.to_parquet(
                path, index=False, engine="pyarrow", compression="zstd"
            )
            files.append((path, role, "parquet"))

        def csv(name: str, frame: pd.DataFrame, role: str) -> None:
            path = staging / name
            frame.to_csv(path, index=False, lineterminator="\n")
            files.append((path, role, "csv"))

        def json_file(name: str, value: object, role: str) -> None:
            path = staging / name
            atomic_write_json(path, _json_safe(value), ensure_ascii=False)
            files.append((path, role, "json"))

        parquet("records.parquet", records, "records")
        parquet("atom_features.parquet", atom_features, "atom_features")
        parquet(
            "molecule_features.parquet",
            molecule_features,
            "molecule_features",
        )
        parquet(
            "raw_xtb_primitives.parquet",
            primitives,
            "raw_xtb_primitives",
        )
        csv(
            "cohort_manifest.csv",
            pd.read_csv(working / "cohort_manifest.csv"),
            "cohort_manifest",
        )
        csv(
            "fragment_selection_audit.csv",
            pd.read_csv(working / "fragment_selection_audit.csv"),
            "fragment_selection_audit",
        )
        csv(
            "label_mapping_audit.csv",
            pd.read_csv(working / "label_mapping_audit.csv"),
            "label_mapping_audit",
        )
        csv("geometry_ledger.csv", geometry_ledger, "geometry_ledger")
        csv(
            "calculation_ledger.csv",
            calculation_ledger,
            "calculation_ledger",
        )
        csv("failure_ledger.csv", failure_rows, "failure_ledger")
        csv("validation_errors.csv", validation, "validation_errors")
        json_file("split_manifest.json", split_manifest, "split_manifest")
        json_file("coverage.json", coverage, "coverage")
        json_file(
            "source_id_parity.json", source_parity, "source_id_parity"
        )
        json_file(
            "feature_schema.json", feature_schema, "feature_schema"
        )
        json_file("summary.json", summary, "summary")
        assets = [
            _file_entry(path, role, format_name)
            for path, role, format_name in files
        ]
        manifest = {
            "schema_version": DATASET_SCHEMA,
            "dataset_id": config["dataset_id"],
            "parent_dataset_ids": [config["parent_dataset_id"]],
            "generated_by": (
                "nucpred.datasets.mayr_pure_alpb_node_xtb"
            ),
            "builder_source": _display_path(Path(__file__)),
            "builder_source_sha256": sha256_file(Path(__file__)),
            "config": {
                "path": _display_path(config_file),
                "sha256": sha256_file(config_file),
            },
            "source_hashes": current_sources,
            "inventory_manifest_sha256": sha256_file(
                working / "inventory_manifest.json"
            ),
            "inventory_source_hashes": inventory_manifest[
                "source_hashes"
            ],
            "full_execution_summary_sha256": sha256_file(
                full_summary_path
            ),
            "inputs": [
                {
                    "path": str(config["parent_records_path"]),
                    "sha256": str(config["parent_records_sha256"]),
                    "role": "parent_records_labels_and_solvent_descriptors",
                },
                {
                    "path": str(config["parent_split_manifest_path"]),
                    "sha256": str(
                        config["parent_split_manifest_sha256"]
                    ),
                    "role": "parent_canonical_split_membership",
                },
                {
                    "path": str(config["cohort_evidence_path"]),
                    "sha256": str(config["cohort_evidence_sha256"]),
                    "role": "fixed_1136_population_identity_only",
                },
                {
                    "path": str(config["xtb"]["archive_path"]),
                    "sha256": str(config["xtb"]["archive_sha256"]),
                    "role": "xtb_distribution",
                },
            ],
            "assets": assets,
            "contracts": {
                "record_count": EXPECTED_RECORDS,
                "cohort_source_id_sha256": str(
                    config["cohort_source_id_sha256"]
                ),
                "hydrogen_is_ordinary_element": True,
                "fragment_selection_target_independent": True,
                "split_repair_target_independent": True,
                "no_dft": True,
                "no_xtb_geometry_optimization": True,
                "fixed_geometry": "G1",
                "node_local_feature_count": len(LOCAL_FEATURES),
                "global_xtb_feature_count": len(GLOBAL_FEATURES),
                "missing_values_preserved_with_masks": True,
                "model_imputation_scope": "outer_fold_train_only",
                "oracle_site_fields_forbidden": True,
            },
        }
        manifest_path = staging / "dataset_manifest.json"
        atomic_write_json(
            manifest_path, _json_safe(manifest), ensure_ascii=False
        )
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "schema_version": DATASET_SCHEMA,
        "dataset_id": config["dataset_id"],
        "status": "pass",
        "output_directory": _display_path(output),
        "record_count": len(records),
        "complete_xtb10_fraction": coverage[
            "complete_xtb10_fraction"
        ],
        "dataset_manifest_sha256": sha256_file(
            output / "dataset_manifest.json"
        ),
    }


def verify_dataset(directory: str | Path) -> dict[str, object]:
    selected = Path(directory).resolve()
    manifest = json.loads(
        (selected / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    if (
        manifest.get("schema_version") != DATASET_SCHEMA
        or manifest.get("dataset_id") != DEFAULT_DATASET_ID
    ):
        raise DatasetBuildError("Unsupported node-xTB dataset manifest")
    for asset in manifest["assets"]:
        path = selected / str(asset["path"])
        if (
            not path.is_file()
            or path.stat().st_size != int(asset["bytes"])
            or sha256_file(path) != str(asset["sha256"])
        ):
            raise DatasetBuildError(f"Dataset asset changed: {path}")
    records = pd.read_parquet(selected / "records.parquet")
    atoms = pd.read_parquet(selected / "atom_features.parquet")
    molecules = pd.read_parquet(
        selected / "molecule_features.parquet"
    )
    if (
        len(records) != EXPECTED_RECORDS
        or records["source_id"].nunique() != EXPECTED_RECORDS
        or len(molecules) != EXPECTED_RECORDS
    ):
        raise DatasetBuildError("Dataset record identity changed")
    if int(records["spectator_stripped"].sum()) != EXPECTED_MULTIFRAGMENT:
        raise DatasetBuildError("Spectator-stripped population changed")
    if int(records["site_target_mask_model"].sum()) != EXPECTED_SITE_SUPERVISED:
        raise DatasetBuildError("Site-supervision population changed")
    if (
        int(
            records["supervision_level"]
            .eq("equivalent_or_indistinguishable_h_group")
            .sum()
        )
        != EXPECTED_H_GROUPS
    ):
        raise DatasetBuildError("H-group population changed")
    if "H" not in ELEMENT_VOCABULARY:
        raise DatasetBuildError("H left the ordinary element vocabulary")
    h_element_index = ELEMENT_VOCABULARY.index("H")
    h_rows = atoms.loc[atoms["is_hydrogen"].eq(True)]
    if h_rows.empty or not h_rows["rdkit_element"].eq(h_element_index).all():
        raise DatasetBuildError("H nodes are not encoded as ordinary H elements")
    forbidden_tokens = (
        "nuc_index",
        "site_target",
        "donor",
        "candidate",
        "target_distribution",
    )
    for name, frame in (
        ("atom_features", atoms),
        ("molecule_features", molecules),
    ):
        forbidden = [
            column
            for column in frame.columns
            if any(token in column.lower() for token in forbidden_tokens)
        ]
        if forbidden:
            raise DatasetBuildError(
                f"Oracle fields entered {name}: {forbidden}"
            )
    coverage = json.loads(
        (selected / "coverage.json").read_text(encoding="utf-8")
    )
    if coverage.get("coverage_gate_pass") is not True:
        raise DatasetBuildError("Dataset xTB coverage gate failed")
    split = json.loads(
        (selected / "split_manifest.json").read_text(encoding="utf-8")
    )
    if (
        len(split.get("runs", [])) != 5
        or not all(
            audit.get("status") == "pass"
            and audit.get("source_id_overlap") == 0
            and audit.get("model_canonical_smiles_overlap") == 0
            and audit.get("repair_target_independent")
            for audit in split.get("audits", [])
        )
    ):
        raise DatasetBuildError("Canonical split leakage audit failed")
    for row in records.itertuples(index=False):
        targets = _parse_indices(row.site_target_atoms_model_json)
        if bool(row.site_target_mask_model) and not targets:
            raise DatasetBuildError(f"{row.source_id}: empty model target")
        if any(
            index < 0 or index >= int(row.model_all_atom_count)
            for index in targets
        ):
            raise DatasetBuildError(f"{row.source_id}: target out of range")
        if (
            row.supervision_level
            == "equivalent_or_indistinguishable_h_group"
        ):
            numbers = _parse_indices(row.model_atomic_numbers_json)
            if any(numbers[index] != 1 for index in targets):
                raise DatasetBuildError(
                    f"{row.source_id}: H group contains non-H"
                )
    return {
        "schema_version": (
            "nucpred.mayr-pure-alpb-node-xtb-verification.v1"
        ),
        "dataset_id": DEFAULT_DATASET_ID,
        "status": "pass",
        "record_count": len(records),
        "atom_count": len(atoms),
        "complete_xtb10_fraction": float(
            coverage["complete_xtb10_fraction"]
        ),
        "verified_file_count": len(manifest["assets"]),
        "manifest_sha256": sha256_file(
            selected / "dataset_manifest.json"
        ),
    }


def run_pipeline(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    scope: str,
) -> dict[str, object]:
    inventory = build_inventory(config_path=config_path)
    geometry = generate_geometries(
        config_path=config_path, scope=scope
    )
    electronic = generate_xtb_features(
        config_path=config_path, scope=scope
    )
    payload: dict[str, object] = {
        "scope": scope,
        "inventory": inventory,
        "geometry": geometry,
        "electronic": electronic,
    }
    if scope == "full":
        payload["dataset"] = finalize_dataset(config_path=config_path)
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build or verify the fixed 1,136-row Mayr node-xTB dataset."
    )
    commands = parser.add_subparsers(dest="action", required=True)
    inventory = commands.add_parser("inventory")
    inventory.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    geometry = commands.add_parser("geometry")
    geometry.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    geometry.add_argument(
        "--scope", choices=("preflight", "full"), default="preflight"
    )
    electronic = commands.add_parser("xtb")
    electronic.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    electronic.add_argument(
        "--scope", choices=("preflight", "full"), default="preflight"
    )
    build = commands.add_parser("build")
    build.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    build.add_argument(
        "--scope", choices=("preflight", "full"), required=True
    )
    finalize = commands.add_parser("finalize")
    finalize.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    finalize.add_argument("--output-directory", type=Path)
    verify = commands.add_parser("verify")
    verify.add_argument(
        "--directory",
        type=Path,
        default=(
            ROOT
            / "data/processed/mayr_pure_alpb_node_xtb"
            / DEFAULT_DATASET_ID
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.action == "inventory":
        payload = build_inventory(config_path=args.config)
    elif args.action == "geometry":
        payload = generate_geometries(
            config_path=args.config, scope=args.scope
        )
    elif args.action == "xtb":
        payload = generate_xtb_features(
            config_path=args.config, scope=args.scope
        )
    elif args.action == "build":
        payload = run_pipeline(
            config_path=args.config, scope=args.scope
        )
    elif args.action == "finalize":
        payload = finalize_dataset(
            config_path=args.config,
            output_directory=args.output_directory,
        )
    else:
        payload = verify_dataset(args.directory)
    print(json.dumps(_json_safe(payload), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
