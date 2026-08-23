"""Build ESNUEL-only, D-isomorphic all-atom G1 + xTB pretraining data.

The builder deliberately has no dependency on the Mayr label branch.  The
frozen 1,136-row Mayr dataset is projected to structure identifiers only and
is used solely to exclude connectivity overlap before deterministic sampling.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
import tomllib
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, inchi
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
DEFAULT_CONFIG = (
    ROOT / "configs/mayr_explicit_h_node_xtb_pretraining.toml"
)
CONFIG_SCHEMA = "nucpred.esnuel-d-node-xtb-pretraining-config.v1"
DATASET_SCHEMA = "nucpred.esnuel-d-node-xtb-pretraining-dataset.v1"
SELECTION_SCHEMA = "nucpred.esnuel-d-node-xtb-selection.v1"
G1_CACHE_SCHEMA = "nucpred.esnuel-d-node-xtb-g1-cache.v1"
XTB_CACHE_SCHEMA = "nucpred.esnuel-d-node-xtb-cache.v1"
TCE_CACHE_SCHEMA = "nucpred.esnuel-d-node-xtb-tce-reference.v1"
EXPECTED_ESNUEL_RECORDS = 47_921
EXPECTED_TARGET_RECORDS = 1_136
EXPECTED_OVERLAP_RECORDS = 6
EXPECTED_ELIGIBLE_RECORDS = 47_915
NATIVE_ROLE_ORDER = ("train", "validation", "audit_test")
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
MAPPING_COLUMNS = (
    "source_id",
    "model_graph_sha256",
    "parent_all_atom_count",
    "model_all_atom_count",
    "atomic_number_order_identity",
    "directed_edge_order_identity",
    "hydrogen_parent_order_identity",
    "source_to_all_atom_identity",
    "mca_target_length_valid",
    "mca_mask_length_valid",
    "gcs_target_length_valid",
    "gcs_mask_length_valid",
    "site_mask_length_valid",
    "added_h_numeric_supervision_count",
    "added_h_site_supervision_count",
    "status",
)
INCHI_KEY_PATTERN = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")
_IMPORT_CODE_HASHES = {
    "builder": sha256_file(Path(__file__).resolve()),
    "all_atom_graph": sha256_file(
        (ROOT / "src/nucpred/features/all_atom_graph.py").resolve()
    ),
    "xtb_runtime": sha256_file(
        (ROOT / "src/nucpred/protocols/xtb_runtime.py").resolve()
    ),
}


class DatasetBuildError(RuntimeError):
    """Raised when a frozen pretraining-data contract is not satisfied."""


@dataclass(frozen=True, slots=True)
class ConnectivityIdentity:
    block: str
    inchi_key: str
    method: str
    dative_bond_count: int


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _json_compact(value: object) -> str:
    return json.dumps(
        _json_safe(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=isinstance(value, Mapping),
    )


def _json_safe(value: object) -> object:
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _as_sequence(value: object) -> list[object]:
    if isinstance(value, str):
        parsed = json.loads(value)
        if not isinstance(parsed, list):
            raise ValueError("Expected a JSON list")
        return parsed
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        parsed = tolist()
        return parsed if isinstance(parsed, list) else [parsed]
    raise TypeError(f"Expected an array-like value, got {type(value).__name__}")


def _float_array(value: object, expected: int) -> np.ndarray:
    values = _as_sequence(value)
    if len(values) != expected:
        raise ValueError(f"Expected {expected} values, got {len(values)}")
    return np.asarray(
        [math.nan if item is None else float(item) for item in values],
        dtype=float,
    )


def _float_matrix(
    value: object, rows: int, columns: int
) -> np.ndarray:
    values = _as_sequence(value)
    if len(values) != rows:
        raise ValueError(f"Expected {rows} rows, got {len(values)}")
    parsed: list[list[float]] = []
    for row in values:
        items = _as_sequence(row)
        if len(items) != columns:
            raise ValueError(
                f"Expected matrix width {columns}, got {len(items)}"
            )
        parsed.append(
            [math.nan if item is None else float(item) for item in items]
        )
    return np.asarray(parsed, dtype=float)


def _source_id_digest(values: Sequence[object]) -> str:
    ordered = sorted(str(value) for value in values)
    return hashlib.sha256(
        ("\n".join(ordered) + "\n").encode("utf-8")
    ).hexdigest()


def _hash_rank(namespace: str, *values: object) -> str:
    text = "\0".join((str(namespace), *(str(value) for value in values)))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cache_name(source_id: str) -> str:
    return hashlib.sha256(str(source_id).encode("utf-8")).hexdigest() + ".json"


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_parquet(
            temporary,
            index=False,
            engine="pyarrow",
            compression="zstd",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(temporary, index=False, lineterminator="\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = Path(path).resolve()
    payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != CONFIG_SCHEMA:
        raise DatasetBuildError("Unsupported ESNUEL D-pretraining config")
    if tuple(payload["features"]["node_local"]) != LOCAL_FEATURES:
        raise DatasetBuildError("D local4 identity or order changed")
    if tuple(payload["features"]["global"]) != GLOBAL_FEATURES:
        raise DatasetBuildError("D global6 identity or order changed")
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
    electronic = payload["xtb"]
    if (
        electronic.get("method") != "GFN1-xTB"
        or int(electronic.get("gfn", 0)) != 1
        or electronic.get("solvation_model") != "ALPB"
        or str(electronic.get("solvent")).lower() != "dmso"
        or bool(electronic.get("geometry_optimization", True))
    ):
        raise DatasetBuildError("GFN1 gas + ALPB-DMSO contract changed")
    if set(payload["quality_control"]) != set(xtb.QC_KEYS):
        raise DatasetBuildError("Shared xTB QC surface changed")
    overlap = payload["overlap"]
    expected_overlap = {
        "identity": "inchi_key_first_block",
        "expected_excluded_esnuel_records": EXPECTED_OVERLAP_RECORDS,
        "expected_eligible_esnuel_records": EXPECTED_ELIGIBLE_RECORDS,
    }
    for key, value in expected_overlap.items():
        if overlap.get(key) != value:
            raise DatasetBuildError(f"Overlap contract changed {key}")
    parent_esnuel = payload["parents"]["m7_esnuel"]
    parent_target = payload["parents"]["mayr_target"]
    if (
        int(parent_esnuel["expected_records"]) != EXPECTED_ESNUEL_RECORDS
        or not bool(parent_esnuel["forbid_mayr_branch"])
        or int(parent_target["expected_records"]) != EXPECTED_TARGET_RECORDS
    ):
        raise DatasetBuildError("Frozen parent population contract changed")
    selection = payload["selection"]
    allowed = tuple(str(value) for value in selection["allowed_scopes"])
    if allowed != ("pilot1024", "pilot4096", "full"):
        raise DatasetBuildError("Selection scopes changed")
    for scope in allowed:
        section = selection[scope]
        role_total = sum(
            int(section[f"{role}_records"]) for role in NATIVE_ROLE_ORDER
        )
        if role_total != int(section["total_records"]):
            raise DatasetBuildError(f"{scope} role quotas do not sum")
    if not 0.0 < float(payload["minimum_complete_xtb10_fraction"]) <= 1.0:
        raise DatasetBuildError("Invalid xTB10 coverage threshold")
    return payload


def _resolve_inputs(
    config: Mapping[str, Any],
) -> dict[str, Path]:
    esnuel = config["parents"]["m7_esnuel"]
    target = config["parents"]["mayr_target"]
    electronic = config["xtb"]
    paths = {
        "esnuel_records": (
            ROOT / str(esnuel["records_path"])
        ).resolve(),
        "esnuel_manifest": (
            ROOT / str(esnuel["manifest_path"])
        ).resolve(),
        "target_records": (
            ROOT / str(target["records_path"])
        ).resolve(),
        "target_manifest": (
            ROOT / str(target["manifest_path"])
        ).resolve(),
        "xtb_archive": (
            ROOT / str(electronic["archive_path"])
        ).resolve(),
    }
    expected = {
        "esnuel_records": str(esnuel["records_sha256"]),
        "esnuel_manifest": str(esnuel["manifest_sha256"]),
        "target_records": str(target["records_sha256"]),
        "target_manifest": str(target["manifest_sha256"]),
        "xtb_archive": str(electronic["archive_sha256"]),
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256_file(path)
        if observed != expected[name]:
            raise DatasetBuildError(
                f"Frozen input digest changed: {_display_path(path)}"
            )
    return paths


def _source_hashes(config_path: Path) -> dict[str, str]:
    paths = {
        "builder": Path(__file__).resolve(),
        "all_atom_graph": (
            ROOT / "src/nucpred/features/all_atom_graph.py"
        ).resolve(),
        "xtb_runtime": (
            ROOT / "src/nucpred/protocols/xtb_runtime.py"
        ).resolve(),
    }
    values = {name: sha256_file(path) for name, path in paths.items()}
    if values != _IMPORT_CODE_HASHES:
        raise DatasetBuildError(
            "Scientific source bytes changed after Python import; restart "
            "the process before building or resuming"
        )
    values["config"] = sha256_file(config_path)
    return values


def _valid_inchi_key(value: str) -> bool:
    return bool(INCHI_KEY_PATTERN.fullmatch(str(value)))


def _connectivity_identity(smiles: str) -> ConnectivityIdentity:
    """Return a first-block InChI identity with an audited dative fallback."""

    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None or molecule.GetNumAtoms() == 0:
        raise DatasetBuildError(
            f"Cannot parse structure for connectivity audit: {smiles!r}"
        )
    dative_count = sum(
        str(bond.GetBondType()) == "DATIVE"
        for bond in molecule.GetBonds()
    )
    key = str(inchi.MolToInchiKey(molecule))
    method = "standard_inchi"
    if not _valid_inchi_key(key):
        if not dative_count:
            raise DatasetBuildError(
                "Standard InChIKey generation failed without an auditable "
                f"dative-bond fallback: {smiles!r}"
            )
        identity_copy = Chem.RWMol(molecule)
        changed = 0
        for bond in identity_copy.GetBonds():
            if str(bond.GetBondType()) == "DATIVE":
                bond.SetBondType(Chem.BondType.SINGLE)
                changed += 1
        if changed != dative_count:
            raise DatasetBuildError("Dative fallback did not cover every bond")
        key = str(inchi.MolToInchiKey(identity_copy.GetMol()))
        method = "identity_only_dative_to_single_inchi"
    if not _valid_inchi_key(key):
        raise DatasetBuildError(
            f"Connectivity InChIKey generation failed: {smiles!r}"
        )
    return ConnectivityIdentity(
        block=key.split("-", maxsplit=1)[0],
        inchi_key=key,
        method=method,
        dative_bond_count=int(dative_count),
    )


def _connectivity_overlap_audits(
    esnuel: pd.DataFrame,
    target: pd.DataFrame,
    *,
    target_smiles_columns: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create complete ESNUEL and Mayr structure-only connectivity audits."""

    required_esnuel = {"source_id", "canonical_smiles"}
    required_target = {"source_id", *map(str, target_smiles_columns)}
    if not required_esnuel <= set(esnuel):
        raise DatasetBuildError("ESNUEL connectivity columns are missing")
    if not required_target <= set(target):
        raise DatasetBuildError("Mayr connectivity columns are missing")
    target_rows: list[dict[str, object]] = []
    block_to_matches: dict[str, list[dict[str, str]]] = {}
    for row in target.to_dict(orient="records"):
        source_id = str(row["source_id"])
        for column in target_smiles_columns:
            identity = _connectivity_identity(str(row[str(column)]))
            audit = {
                "target_source_id": source_id,
                "smiles_column": str(column),
                "smiles": str(row[str(column)]),
                "connectivity_inchi_block": identity.block,
                "inchi_key": identity.inchi_key,
                "identity_method": identity.method,
                "dative_bond_count": identity.dative_bond_count,
                "status": "pass",
            }
            target_rows.append(audit)
            block_to_matches.setdefault(identity.block, []).append(
                {
                    "target_source_id": source_id,
                    "smiles_column": str(column),
                }
            )
    esnuel_rows: list[dict[str, object]] = []
    for row in esnuel.to_dict(orient="records"):
        identity = _connectivity_identity(str(row["canonical_smiles"]))
        matches = block_to_matches.get(identity.block, [])
        target_ids = sorted(
            {str(item["target_source_id"]) for item in matches}
        )
        match_columns = sorted(
            {
                f"{item['target_source_id']}:{item['smiles_column']}"
                for item in matches
            }
        )
        esnuel_rows.append(
            {
                "source_id": str(row["source_id"]),
                "canonical_smiles": str(row["canonical_smiles"]),
                "connectivity_inchi_block": identity.block,
                "inchi_key": identity.inchi_key,
                "identity_method": identity.method,
                "dative_bond_count": identity.dative_bond_count,
                "excluded_for_mayr_connectivity_overlap": bool(target_ids),
                "matched_target_source_ids_json": _json_compact(target_ids),
                "matched_target_identity_columns_json": _json_compact(
                    match_columns
                ),
                "status": "excluded_overlap" if target_ids else "eligible",
            }
        )
    return pd.DataFrame(esnuel_rows), pd.DataFrame(target_rows)


def _native_pretraining_role(value: object) -> str:
    token = str(value)
    mapping = {
        "train": "train",
        "validation": "validation",
        "test": "audit_test",
        "audit_test": "audit_test",
    }
    if token not in mapping:
        raise DatasetBuildError(f"Unknown native ESNUEL split: {token!r}")
    return mapping[token]


def _scope_quotas(
    config: Mapping[str, Any], scope: str
) -> dict[str, int]:
    selection = config["selection"]
    if scope not in set(map(str, selection["allowed_scopes"])):
        raise ValueError(f"Unsupported selection scope {scope!r}")
    section = selection[scope]
    return {
        role: int(section[f"{role}_records"])
        for role in NATIVE_ROLE_ORDER
    }


def _select_stratified_records(
    eligible: pd.DataFrame,
    *,
    quotas: Mapping[str, int],
    namespace: str,
    mandatory_strata: Mapping[str, Sequence[str]] | None = None,
) -> pd.DataFrame:
    """Select deterministic, nested hash prefixes inside native splits."""

    required = {"source_id", "pretraining_role"}
    if not required <= set(eligible):
        raise DatasetBuildError("Pilot selection columns are missing")
    if eligible["source_id"].astype(str).duplicated().any():
        raise DatasetBuildError("Eligible ESNUEL source IDs are not unique")
    strata_by_source: dict[str, list[str]] = {}
    for stratum, source_ids in dict(mandatory_strata or {}).items():
        for source_id in source_ids:
            strata_by_source.setdefault(str(source_id), []).append(
                str(stratum)
            )
    unknown_mandatory = set(strata_by_source) - set(
        eligible["source_id"].astype(str)
    )
    if unknown_mandatory:
        raise DatasetBuildError(
            f"Mandatory pilot IDs are ineligible: {sorted(unknown_mandatory)}"
        )
    chosen: list[pd.DataFrame] = []
    for role in NATIVE_ROLE_ORDER:
        candidate = eligible.loc[
            eligible["pretraining_role"].astype(str).eq(role)
        ].copy()
        candidate["selection_hash_rank"] = candidate["source_id"].map(
            lambda value: _hash_rank(namespace, role, str(value))
        )
        candidate["selection_mandatory"] = candidate["source_id"].map(
            lambda value: str(value) in strata_by_source
        )
        candidate["selection_mandatory_strata_json"] = candidate[
            "source_id"
        ].map(
            lambda value: _json_compact(
                sorted(strata_by_source.get(str(value), []))
            )
        )
        candidate = candidate.sort_values(
            ["selection_mandatory", "selection_hash_rank", "source_id"],
            ascending=[False, True, True],
            kind="stable",
        )
        quota = int(quotas[role])
        if len(candidate) < quota:
            raise DatasetBuildError(
                f"{role} has {len(candidate)} eligible rows, needs {quota}"
            )
        selected = candidate.iloc[:quota].copy()
        selected["within_role_selection_rank"] = np.arange(
            len(selected), dtype=int
        )
        chosen.append(selected)
    result = pd.concat(chosen, ignore_index=True)
    result["selection_index"] = np.arange(len(result), dtype=int)
    return result


def _mandatory_pilot_strata(
    eligible: pd.DataFrame,
    *,
    required_elements: Sequence[str],
    high_atom_count_quantile: float,
    namespace: str,
) -> tuple[dict[str, list[str]], dict[str, object]]:
    """Choose deterministic element and molecular-size gate representatives."""

    required = {"source_id", "atomic_numbers_json", "all_atom_count"}
    if not required <= set(eligible):
        raise DatasetBuildError("Mandatory pilot strata columns are missing")
    periodic_table = Chem.GetPeriodicTable()
    required_numbers = {
        str(symbol): int(periodic_table.GetAtomicNumber(str(symbol)))
        for symbol in required_elements
    }
    atom_sets = {
        str(row.source_id): {
            int(value) for value in _as_sequence(row.atomic_numbers_json)
        }
        for row in eligible.itertuples(index=False)
    }
    strata: dict[str, list[str]] = {}
    chosen: set[str] = set()
    for symbol, atomic_number in required_numbers.items():
        candidates = sorted(
            (
                str(source_id)
                for source_id, numbers in atom_sets.items()
                if atomic_number in numbers
            ),
            key=lambda source_id: (
                source_id in chosen,
                _hash_rank(
                    namespace, "mandatory_element", symbol, source_id
                ),
                source_id,
            ),
        )
        if not candidates:
            raise DatasetBuildError(
                f"No eligible pilot record contains required element {symbol}"
            )
        representative = candidates[0]
        chosen.add(representative)
        strata[f"element:{symbol}"] = [representative]
    quantile = float(high_atom_count_quantile)
    if not 0.0 < quantile < 1.0:
        raise DatasetBuildError("Invalid high atom-count quantile")
    threshold = float(
        eligible["all_atom_count"].astype(float).quantile(
            quantile, interpolation="lower"
        )
    )
    high_tail = eligible.loc[
        eligible["all_atom_count"].astype(float).ge(threshold)
    ].copy()
    high_tail["mandatory_hash_rank"] = high_tail["source_id"].map(
        lambda value: _hash_rank(
            namespace, "mandatory_high_atom_count", str(value)
        )
    )
    high_tail = high_tail.sort_values(
        ["mandatory_hash_rank", "source_id"], kind="stable"
    )
    if high_tail.empty:
        raise DatasetBuildError("High atom-count mandatory tail is empty")
    high_source_id = str(high_tail.iloc[0]["source_id"])
    strata["high_atom_count_tail"] = [high_source_id]
    audit = {
        "required_elements": list(map(str, required_elements)),
        "required_atomic_numbers": required_numbers,
        "high_atom_count_quantile": quantile,
        "high_atom_count_threshold": threshold,
        "high_atom_count_eligible_records": len(high_tail),
        "representatives": {
            name: list(values) for name, values in strata.items()
        },
        "unique_mandatory_source_id_count": len(
            {
                source_id
                for source_ids in strata.values()
                for source_id in source_ids
            }
        ),
    }
    return strata, audit


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


def _molecule_from_mapped_smiles(
    smiles: str, source_atom_count: int
) -> Chem.Mol:
    parsed = Chem.MolFromSmiles(str(smiles))
    if parsed is None or parsed.GetNumAtoms() != int(source_atom_count):
        raise DatasetBuildError("Mapped ESNUEL SMILES failed to round-trip")
    numbered = {
        int(atom.GetAtomMapNum()): int(atom.GetIdx())
        for atom in parsed.GetAtoms()
    }
    if set(numbered) != set(range(1, int(source_atom_count) + 1)):
        raise DatasetBuildError("Mapped ESNUEL SMILES lost atom identity")
    order = [
        numbered[index] for index in range(1, int(source_atom_count) + 1)
    ]
    molecule = Chem.RenumberAtoms(parsed, order)
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    return molecule


def _geometry_seed(namespace: str, source_id: str) -> int:
    digest = _hash_rank(namespace, source_id)
    return 1 + (int(digest[:15], 16) % 2_147_483_646)


def _prepare_graph_inventory(
    selected: pd.DataFrame,
    *,
    geometry_seed_namespace: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map M7 ESNUEL rows to exactly the categorical graph contract used by D."""

    rows: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    for parent in selected.to_dict(orient="records"):
        source_id = str(parent["source_id"])
        source = Chem.MolFromSmiles(str(parent["canonical_smiles"]))
        if source is None or source.GetNumAtoms() == 0:
            raise DatasetBuildError(f"{source_id}: canonical SMILES failed")
        source_count = source.GetNumAtoms()
        explicit = Chem.AddHs(Chem.Mol(source), addCoords=False)
        graph = featurize_explicit_molecule(
            explicit, source_atom_count=source_count
        )
        assert_category_ranges(
            graph.node_categorical, NODE_CATEGORY_SIZES
        )
        assert_category_ranges(
            graph.edge_categorical, EDGE_CATEGORY_SIZES
        )
        parent_numbers = tuple(
            int(value)
            for value in _as_sequence(parent["atomic_numbers_json"])
        )
        parent_edges = tuple(
            tuple(int(item) for item in edge)
            for edge in _as_sequence(parent["directed_bonds_json"])
        )
        parent_hydrogen_parents = tuple(
            int(value)
            for value in _as_sequence(
                parent["hydrogen_parent_index_json"]
            )
        )
        parent_source_mapping = tuple(
            int(value)
            for value in _as_sequence(parent["source_to_all_atom_json"])
        )
        count = graph.atom_count
        if (
            int(parent["source_atom_count"]) != source_count
            or int(parent["all_atom_count"]) != count
        ):
            raise DatasetBuildError(
                f"{source_id}: parent/model atom counts differ"
            )
        atomic_identity = parent_numbers == graph.atomic_numbers
        edge_identity = parent_edges == graph.directed_edges
        hydrogen_parent_identity = (
            parent_hydrogen_parents == graph.hydrogen_parent_index
        )
        source_mapping_identity = parent_source_mapping == tuple(
            range(source_count)
        )
        if not (
            atomic_identity
            and edge_identity
            and hydrogen_parent_identity
            and source_mapping_identity
        ):
            raise DatasetBuildError(
                f"{source_id}: D graph/source mapping differs from M7"
            )
        mca_targets = _as_sequence(parent["mca_targets_all_atom"])
        mca_mask = [
            bool(value)
            for value in _as_sequence(parent["mca_target_mask_all_atom"])
        ]
        gcs_targets = _as_sequence(parent["gcs_targets_all_atom"])
        gcs_mask = [
            bool(value)
            for value in _as_sequence(parent["gcs_target_mask_all_atom"])
        ]
        site_mask = [
            bool(value)
            for value in _as_sequence(parent["site_mask_all_atom"])
        ]
        added_h = [
            bool(value)
            for value in _as_sequence(parent["added_hydrogen_mask_json"])
        ]
        if any(
            len(values) != count
            for values in (
                mca_targets,
                mca_mask,
                gcs_targets,
                gcs_mask,
                site_mask,
                added_h,
            )
        ):
            raise DatasetBuildError(
                f"{source_id}: ESNUEL target/mask length changed"
            )
        for atom_index, enabled in enumerate(gcs_mask):
            if not enabled:
                continue
            vector = _as_sequence(gcs_targets[atom_index])
            if len(vector) != 53 or not all(
                math.isfinite(float(value)) for value in vector
            ):
                raise DatasetBuildError(
                    f"{source_id}: enabled GCS vector is not finite 53D"
                )
        for atom_index, enabled in enumerate(mca_mask):
            if enabled and not math.isfinite(
                float(mca_targets[atom_index])
            ):
                raise DatasetBuildError(
                    f"{source_id}: enabled MCA target is non-finite"
                )
        added_numeric = sum(
            is_added and (mca_mask[index] or gcs_mask[index])
            for index, is_added in enumerate(added_h)
        )
        added_site = sum(
            is_added and site_mask[index]
            for index, is_added in enumerate(added_h)
        )
        if added_numeric or added_site:
            raise DatasetBuildError(
                f"{source_id}: added H received forged ESNUEL supervision"
            )
        model_values = {
            "native_pretraining_split": str(
                parent["native_continuation_split"]
            ),
            "pretraining_role": str(parent["pretraining_role"]),
            "model_canonical_smiles": Chem.MolToSmiles(
                source, isomericSmiles=True
            ),
            "model_mapped_smiles": _mapped_smiles(source),
            "model_source_atom_count": source_count,
            "model_all_atom_count": count,
            "model_hydrogen_atom_count": int(
                sum(number == 1 for number in graph.atomic_numbers)
            ),
            "model_formal_charge": int(Chem.GetFormalCharge(source)),
            "model_radical_electrons": int(
                sum(
                    atom.GetNumRadicalElectrons()
                    for atom in source.GetAtoms()
                )
            ),
            "model_atomic_numbers_json": _json_compact(
                graph.atomic_numbers
            ),
            "model_node_categorical_json": _json_compact(
                graph.node_categorical
            ),
            "model_directed_edges_json": _json_compact(
                graph.directed_edges
            ),
            "model_edge_categorical_json": _json_compact(
                graph.edge_categorical
            ),
            "model_hydrogen_parent_index_json": _json_compact(
                graph.hydrogen_parent_index
            ),
            "model_graph_sha256": graph.mapping_sha256,
            "xtb_alpb_solvent": "dmso",
            "geometry_seed": _geometry_seed(
                geometry_seed_namespace, source_id
            ),
        }
        rows.append({**parent, **model_values})
        audits.append(
            {
                "source_id": source_id,
                "model_graph_sha256": graph.mapping_sha256,
                "parent_all_atom_count": int(parent["all_atom_count"]),
                "model_all_atom_count": count,
                "atomic_number_order_identity": atomic_identity,
                "directed_edge_order_identity": edge_identity,
                "hydrogen_parent_order_identity": (
                    hydrogen_parent_identity
                ),
                "source_to_all_atom_identity": source_mapping_identity,
                "mca_target_length_valid": len(mca_targets) == count,
                "mca_mask_length_valid": len(mca_mask) == count,
                "gcs_target_length_valid": len(gcs_targets) == count,
                "gcs_mask_length_valid": len(gcs_mask) == count,
                "site_mask_length_valid": len(site_mask) == count,
                "added_h_numeric_supervision_count": int(added_numeric),
                "added_h_site_supervision_count": int(added_site),
                "status": "pass",
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(audits, columns=MAPPING_COLUMNS)


def _load_structure_only_inputs(
    config: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    esnuel = pd.read_parquet(paths["esnuel_records"])
    target_columns = [
        "source_id",
        *map(
            str,
            config["parents"]["mayr_target"][
                "connectivity_smiles_columns"
            ],
        ),
    ]
    target = pd.read_parquet(
        paths["target_records"], columns=target_columns
    )
    if len(esnuel) != EXPECTED_ESNUEL_RECORDS:
        raise DatasetBuildError("M7 ESNUEL parent count changed")
    if len(target) != EXPECTED_TARGET_RECORDS:
        raise DatasetBuildError("Frozen Mayr target count changed")
    if esnuel["source_id"].astype(str).duplicated().any():
        raise DatasetBuildError("M7 ESNUEL source IDs are not unique")
    required_corpus = str(
        config["parents"]["m7_esnuel"]["required_corpus"]
    )
    if (
        "corpus" not in esnuel
        or set(esnuel["corpus"].astype(str)) != {required_corpus}
        or esnuel["source_id"].astype(str).str.startswith("mayr:").any()
    ):
        raise DatasetBuildError(
            "The ESNUEL-only input contains a non-ESNUEL/Mayr branch"
        )
    return esnuel, target


def _selection_directory(
    config: Mapping[str, Any], scope: str
) -> Path:
    working = (ROOT / str(config["working_directory"])).resolve()
    return working / "scopes" / str(scope)


def build_selection(
    *,
    scope: str = "pilot1024",
    config_path: str | Path = DEFAULT_CONFIG,
) -> dict[str, object]:
    """Build deterministic ESNUEL selection, graph mapping, and audits."""

    config_file = Path(config_path).resolve()
    config = _read_config(config_file)
    paths = _resolve_inputs(config)
    quotas = _scope_quotas(config, scope)
    scope_directory = _selection_directory(config, scope)
    inventory_path = scope_directory / "inventory.parquet"
    manifest_path = scope_directory / "selection_manifest.json"
    source_hashes = _source_hashes(config_file)
    if inventory_path.exists() or manifest_path.exists():
        if not inventory_path.is_file() or not manifest_path.is_file():
            raise DatasetBuildError("Partial selection cache is present")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("schema_version") != SELECTION_SCHEMA
            or manifest.get("scope") != scope
            or manifest.get("source_hashes") != source_hashes
            or manifest.get("inventory_sha256")
            != sha256_file(inventory_path)
        ):
            raise DatasetBuildError(
                "Existing selection was built from different source bytes"
            )
        return {
            "status": "reused",
            "scope": scope,
            "selected_record_count": int(
                manifest["selected_record_count"]
            ),
            "eligible_record_count": int(
                manifest["eligible_record_count"]
            ),
            "inventory_sha256": str(manifest["inventory_sha256"]),
        }
    esnuel, target = _load_structure_only_inputs(config, paths)
    native_column = str(config["selection"]["native_split_column"])
    esnuel = esnuel.copy()
    esnuel["native_pretraining_split"] = esnuel[native_column].astype(str)
    esnuel["pretraining_role"] = esnuel[
        "native_pretraining_split"
    ].map(_native_pretraining_role)
    parent_counts = {
        role: int(esnuel["pretraining_role"].eq(role).sum())
        for role in NATIVE_ROLE_ORDER
    }
    expected_parent_counts = {
        "train": int(
            config["parents"]["m7_esnuel"]["expected_train"]
        ),
        "validation": int(
            config["parents"]["m7_esnuel"]["expected_validation"]
        ),
        "audit_test": int(
            config["parents"]["m7_esnuel"]["expected_audit_test"]
        ),
    }
    if parent_counts != expected_parent_counts:
        raise DatasetBuildError(
            f"Native ESNUEL split counts changed: {parent_counts}"
        )
    overlap, target_audit = _connectivity_overlap_audits(
        esnuel,
        target,
        target_smiles_columns=config["parents"]["mayr_target"][
            "connectivity_smiles_columns"
        ],
    )
    excluded_ids = set(
        overlap.loc[
            overlap["excluded_for_mayr_connectivity_overlap"],
            "source_id",
        ].astype(str)
    )
    if len(excluded_ids) != EXPECTED_OVERLAP_RECORDS:
        raise DatasetBuildError(
            f"Connectivity overlap changed: {len(excluded_ids)} != 6"
        )
    excluded_digest = _source_id_digest(sorted(excluded_ids))
    if excluded_digest != str(
        config["overlap"]["excluded_source_id_sha256"]
    ):
        raise DatasetBuildError(
            "Excluded connectivity-overlap source IDs changed"
        )
    eligible = esnuel.loc[
        ~esnuel["source_id"].astype(str).isin(excluded_ids)
    ].copy()
    if len(eligible) != EXPECTED_ELIGIBLE_RECORDS:
        raise DatasetBuildError("Eligible ESNUEL hard count changed")
    eligible_digest = _source_id_digest(
        eligible["source_id"].astype(str).tolist()
    )
    if eligible_digest != str(
        config["overlap"]["eligible_source_id_sha256"]
    ):
        raise DatasetBuildError("Eligible ESNUEL source-id digest changed")
    mandatory_config = config["selection"]["mandatory_strata"]
    mandatory_strata, mandatory_audit = _mandatory_pilot_strata(
        eligible,
        required_elements=mandatory_config["required_elements"],
        high_atom_count_quantile=float(
            mandatory_config["high_atom_count_quantile"]
        ),
        namespace=str(config["selection"]["hash_namespace"]),
    )
    selected = _select_stratified_records(
        eligible,
        quotas=quotas,
        namespace=str(config["selection"]["hash_namespace"]),
        mandatory_strata=mandatory_strata,
    )
    selected_counts = {
        role: int(selected["pretraining_role"].eq(role).sum())
        for role in NATIVE_ROLE_ORDER
    }
    if selected_counts != quotas:
        raise DatasetBuildError("Selected native split quotas changed")
    selected_digest = _source_id_digest(
        selected["source_id"].astype(str).tolist()
    )
    if selected_digest != str(
        config["selection"][scope]["source_id_sha256"]
    ):
        raise DatasetBuildError(
            f"{scope} deterministic source-id selection changed"
        )
    selected_atom_numbers: set[int] = set()
    for value in selected["atomic_numbers_json"]:
        selected_atom_numbers.update(
            int(number) for number in _as_sequence(value)
        )
    missing_elements = [
        symbol
        for symbol, atomic_number in mandatory_audit[
            "required_atomic_numbers"
        ].items()
        if int(atomic_number) not in selected_atom_numbers
    ]
    maximum_selected_atoms = int(
        selected["all_atom_count"].astype(int).max()
    )
    if (
        missing_elements
        or maximum_selected_atoms
        < float(mandatory_audit["high_atom_count_threshold"])
    ):
        raise DatasetBuildError(
            "Mandatory pilot element/high-size coverage failed"
        )
    inventory, mapping = _prepare_graph_inventory(
        selected,
        geometry_seed_namespace=str(
            config["geometry"]["random_seed_namespace"]
        ),
    )
    scope_directory.mkdir(parents=True, exist_ok=True)
    overlap_path = scope_directory / "overlap_audit.csv"
    target_path = scope_directory / "target_connectivity_audit.csv"
    mapping_path = scope_directory / "mapping_audit.csv"
    membership_path = scope_directory / "split_membership.csv"
    membership = inventory[
        [
            "selection_index",
            "within_role_selection_rank",
            "selection_hash_rank",
            "source_id",
            "native_pretraining_split",
            "pretraining_role",
        ]
    ].copy()
    _atomic_parquet(inventory_path, inventory)
    _atomic_csv(overlap_path, overlap)
    _atomic_csv(target_path, target_audit)
    _atomic_csv(mapping_path, mapping)
    _atomic_csv(membership_path, membership)
    overlap_matches = overlap.loc[
        overlap["excluded_for_mayr_connectivity_overlap"],
        [
            "source_id",
            "connectivity_inchi_block",
            "matched_target_source_ids_json",
        ],
    ].sort_values("source_id")
    manifest = {
        "schema_version": SELECTION_SCHEMA,
        "dataset_id_prefix": str(config["dataset_id_prefix"]),
        "scope": scope,
        "generated_by": (
            "nucpred.datasets.esnuel_d_node_xtb_pretraining"
        ),
        "source_hashes": source_hashes,
        "parent_esnuel_record_count": len(esnuel),
        "mayr_structure_only_record_count": len(target),
        "excluded_overlap_record_count": len(excluded_ids),
        "eligible_record_count": len(eligible),
        "selected_record_count": len(inventory),
        "native_parent_counts": parent_counts,
        "eligible_native_counts": {
            role: int(eligible["pretraining_role"].eq(role).sum())
            for role in NATIVE_ROLE_ORDER
        },
        "selected_native_counts": selected_counts,
        "mandatory_strata_audit": mandatory_audit,
        "selected_element_symbols": [
            Chem.GetPeriodicTable().GetElementSymbol(number)
            for number in sorted(selected_atom_numbers)
        ],
        "selected_maximum_all_atom_count": maximum_selected_atoms,
        "eligible_source_id_sha256": eligible_digest,
        "selected_source_id_sha256": selected_digest,
        "excluded_overlap_source_ids": sorted(excluded_ids),
        "excluded_overlap_matches": overlap_matches.to_dict(
            orient="records"
        ),
        "inventory_sha256": sha256_file(inventory_path),
        "assets": {
            path.name: sha256_file(path)
            for path in (
                overlap_path,
                target_path,
                mapping_path,
                membership_path,
            )
        },
        "contracts": {
            "mayr_branch_loaded": False,
            "mayr_structure_identity_loaded_for_exclusion": True,
            "mayr_labels_loaded": False,
            "mayr_labels_used": False,
            "overlap_identity": "inchi_key_first_block",
            "native_esnuel_split_preserved": True,
            "selection_target_independent": True,
            "hydrogen_is_ordinary_element": True,
            "added_h_mca_gcs_site_labels_allowed": False,
        },
    }
    atomic_write_json(manifest_path, _json_safe(manifest), ensure_ascii=False)
    return {
        "status": "built",
        "scope": scope,
        "selected_record_count": len(inventory),
        "eligible_record_count": len(eligible),
        "excluded_overlap_record_count": len(excluded_ids),
        "inventory_sha256": sha256_file(inventory_path),
        "selection_manifest_sha256": sha256_file(manifest_path),
    }


def _load_inventory(
    config_file: Path,
    config: Mapping[str, Any],
    scope: str,
) -> tuple[pd.DataFrame, Path, dict[str, object]]:
    scope_directory = _selection_directory(config, scope)
    inventory_path = scope_directory / "inventory.parquet"
    manifest_path = scope_directory / "selection_manifest.json"
    if not inventory_path.is_file() or not manifest_path.is_file():
        build_selection(scope=scope, config_path=config_file)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != SELECTION_SCHEMA
        or manifest.get("scope") != scope
        or manifest.get("source_hashes") != _source_hashes(config_file)
        or manifest.get("inventory_sha256") != sha256_file(inventory_path)
    ):
        raise DatasetBuildError("Selection source/config parity failed")
    inventory = pd.read_parquet(inventory_path)
    expected = sum(_scope_quotas(config, scope).values())
    if (
        len(inventory) != expected
        or inventory["source_id"].astype(str).nunique() != expected
    ):
        raise DatasetBuildError("Selection inventory identity changed")
    return inventory, scope_directory, manifest


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
    for atomic_number, position in zip(
        atomic_numbers, positions, strict=True
    ):
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
    source = _molecule_from_mapped_smiles(
        str(row["model_mapped_smiles"]),
        int(row["model_source_atom_count"]),
    )
    explicit = Chem.AddHs(Chem.Mol(source), addCoords=False)
    atomic_numbers = tuple(
        atom.GetAtomicNum() for atom in explicit.GetAtoms()
    )
    expected_numbers = tuple(
        int(value)
        for value in _as_sequence(row["model_atomic_numbers_json"])
    )
    if atomic_numbers != expected_numbers:
        raise DatasetBuildError(
            f"{row['source_id']}: G1 atom order differs from D graph"
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
            f"{row['source_id']}: no permitted force field has parameters"
        )
    results = _force_field_results(
        explicit,
        force_field=force_field,
        max_iterations=int(section["force_field_max_iterations"]),
    )
    eligible = [
        (conformer_id, status, energy)
        for conformer_id, (status, energy) in zip(
            conformer_ids, results, strict=True
        )
        if math.isfinite(energy)
        and (
            status == 0
            or not bool(section["require_force_field_convergence"])
        )
    ]
    if not eligible:
        raise DatasetBuildError(
            f"{row['source_id']}: no converged {force_field} conformer"
        )
    selected_id, convergence_code, selected_energy = min(
        eligible, key=lambda item: (item[2], item[0])
    )
    conformer = explicit.GetConformer(selected_id)
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
        "schema_version": G1_CACHE_SCHEMA,
        "source_id": str(row["source_id"]),
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
        "selected_conformer_id": int(selected_id),
        "convergence_code": int(convergence_code),
        "selected_energy_kcal_mol": float(selected_energy),
        "atomic_numbers": atomic_numbers,
        "positions_angstrom": positions,
        "xyz_sha256": hashlib.sha256(
            xyz.encode("utf-8")
        ).hexdigest(),
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
        "schema_version": G1_CACHE_SCHEMA,
        "source_id": str(row["source_id"]),
        "status": "failed",
        "config_sha256": config_sha256,
        "source_hashes": dict(source_hashes),
        "model_graph_sha256": str(row["model_graph_sha256"]),
        "error_type": type(error).__name__,
        "error": str(error),
    }


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
    scope: str = "pilot1024",
    config_path: str | Path = DEFAULT_CONFIG,
) -> dict[str, object]:
    """Generate or resume the 20-conformer all-atom G1 geometry scope."""

    config_file = Path(config_path).resolve()
    config = _read_config(config_file)
    _resolve_inputs(config)
    inventory, scope_directory, _ = _load_inventory(
        config_file, config, scope
    )
    config_hash = sha256_file(config_file)
    source_hashes = _source_hashes(config_file)
    working = (ROOT / str(config["working_directory"])).resolve()
    cache_directory = working / "cache" / "geometry"
    cache_directory.mkdir(parents=True, exist_ok=True)
    pending: list[dict[str, object]] = []
    reused = 0
    for row in inventory.to_dict(orient="records"):
        path = cache_directory / _cache_name(str(row["source_id"]))
        if path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                payload = {}
            if _cache_entry_matches(
                payload,
                row,
                schema_version=G1_CACHE_SCHEMA,
                config_sha256=config_hash,
                source_hashes=source_hashes,
            ):
                reused += 1
                continue
            raise DatasetBuildError(
                f"G1 cache parity mismatch: {row['source_id']}"
            )
        pending.append(row)

    def calculate(
        row: dict[str, object],
    ) -> tuple[Path, dict[str, object]]:
        path = cache_directory / _cache_name(str(row["source_id"]))
        try:
            payload = _geometry_record(
                row,
                config=config,
                config_sha256=config_hash,
                source_hashes=source_hashes,
            )
        except BaseException as error:
            payload = _failed_geometry_record(
                row,
                error,
                config_sha256=config_hash,
                source_hashes=source_hashes,
            )
        return path, payload

    completed = 0
    workers = min(
        int(config["execution"]["workers"]), max(1, len(pending))
    )
    if pending:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(calculate, row) for row in pending]
            for future in as_completed(futures):
                path, payload = future.result()
                atomic_write_json(
                    path, _json_safe(payload), ensure_ascii=False
                )
                completed += 1
                if completed % 25 == 0 or completed == len(pending):
                    print(
                        f"ESNUEL G1 {scope}: {completed}/{len(pending)} "
                        "new records",
                        flush=True,
                    )
    payloads = [
        json.loads(
            (
                cache_directory / _cache_name(str(source_id))
            ).read_text(encoding="utf-8")
        )
        for source_id in inventory["source_id"].astype(str)
    ]
    failures = [
        row for row in payloads if row.get("status") != "success"
    ]
    summary = {
        "schema_version": (
            "nucpred.esnuel-d-node-xtb-g1-scope-summary.v1"
        ),
        "scope": scope,
        "requested_record_count": len(inventory),
        "success_record_count": len(inventory) - len(failures),
        "failure_record_count": len(failures),
        "new_record_count": completed,
        "reused_record_count": reused,
        "config_sha256": config_hash,
        "source_hashes": source_hashes,
        "failed_source_ids": sorted(
            str(row["source_id"]) for row in failures
        ),
        "downstream_missingness_allowed": True,
    }
    atomic_write_json(
        scope_directory / "geometry_summary.json",
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
    ledger = xtb._ledger_row(
        source_id=source_id,
        environment="gas",
        calculation="neutral_single_point",
        charge=charge,
        uhf=uhf,
        execution=execution,
    )
    return result, ledger


def _derived_xtb_features(
    *,
    gas_property: xtb.PropertyResult,
    alpb: xtb.EnvironmentResult,
    formal_charge: int,
    tce_homo_hartree: float,
    quality_control: Mapping[str, object],
) -> dict[str, object]:
    """Derive local4/global6 in the same order and units as Mayr arm D."""

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
        alpb.property, thresholds=quality_control
    )
    vipea_qc = xtb._vipea_qc(
        alpb, thresholds=quality_control
    )
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
        fukui = -np.asarray(
            alpb.property.fukui_minus_raw, dtype=float
        )
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
        "complete_xtb10": bool(
            local_mask.all() and global_mask.all()
        ),
        "qc": {
            name: {
                "passed": result.passed,
                "reason": result.reason,
            }
            for name, result in qc.items()
        },
        "raw": {
            "gas_total_energy_hartree": gas_property.energy_hartree,
            "gas_homo_hartree": (
                gas_property.homo_ev / xtb.HARTREE_TO_EV
            ),
            "gas_lumo_hartree": (
                gas_property.lumo_ev / xtb.HARTREE_TO_EV
            ),
            "gas_cm5": gas_property.cm5,
            "alpb_total_energy_hartree": (
                alpb.property.energy_hartree
            ),
            "alpb_homo_hartree": (
                alpb.property.homo_ev / xtb.HARTREE_TO_EV
            ),
            "alpb_lumo_hartree": (
                alpb.property.lumo_ev / xtb.HARTREE_TO_EV
            ),
            "alpb_dipole_debye": alpb.property.dipole_debye,
            "alpb_cm5": alpb.property.cm5,
            "alpb_fukui_minus_raw": (
                alpb.property.fukui_minus_raw
            ),
            "alpb_fukui_minus_signed": (
                -np.asarray(
                    alpb.property.fukui_minus_raw, dtype=float
                )
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
    atomic_numbers = tuple(
        int(value) for value in geometry["atomic_numbers"]
    )
    expected = tuple(
        int(value)
        for value in _as_sequence(row["model_atomic_numbers_json"])
    )
    if atomic_numbers != expected:
        raise DatasetBuildError(
            f"{source_id}: xTB/G1 atom order mismatch"
        )
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
        solvent=str(config["xtb"]["solvent"]),
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
        "schema_version": XTB_CACHE_SCHEMA,
        "source_id": source_id,
        "status": "success",
        "config_sha256": config_sha256,
        "source_hashes": dict(source_hashes),
        "model_graph_sha256": str(row["model_graph_sha256"]),
        "geometry_xyz_sha256": str(geometry["xyz_sha256"]),
        "formal_charge": charge,
        "neutral_uhf": uhf,
        "neutral_spin_source": spin_source,
        "cation_uhf": cation_uhf,
        "solvent": str(config["xtb"]["solvent"]),
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
        "schema_version": XTB_CACHE_SCHEMA,
        "source_id": str(row["source_id"]),
        "status": "failed",
        "config_sha256": config_sha256,
        "source_hashes": dict(source_hashes),
        "model_graph_sha256": str(row["model_graph_sha256"]),
        "geometry_xyz_sha256": str(
            geometry.get("xyz_sha256", "")
        ),
        "local_feature_names": LOCAL_FEATURES,
        "global_feature_names": GLOBAL_FEATURES,
        "local_values": [
            [None] * len(LOCAL_FEATURES) for _ in range(atom_count)
        ],
        "local_mask": [
            [False] * len(LOCAL_FEATURES) for _ in range(atom_count)
        ],
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
    path = working / "cache" / "tce_reference.json"
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("schema_version") != TCE_CACHE_SCHEMA
            or payload.get("config_sha256") != config_sha256
            or payload.get("source_hashes") != dict(source_hashes)
        ):
            raise DatasetBuildError("TCE cache source parity failed")
        return payload
    value, ledger, xyz = xtb._tce_reference(
        binary,
        smiles=str(config["xtb"]["tce_reference_smiles"]),
        timeout_seconds=int(config["execution"]["timeout_seconds"]),
    )
    payload = {
        "schema_version": TCE_CACHE_SCHEMA,
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


def _require_expansion_gate(
    *,
    scope: str,
    config: Mapping[str, Any],
    config_hash: str,
    source_hashes: Mapping[str, str],
) -> None:
    prerequisite = str(
        config["selection"]["full_expansion_requires_passing_scope"]
    )
    if scope == prerequisite:
        return
    summary_path = (
        _selection_directory(config, prerequisite) / "xtb_summary.json"
    )
    if not summary_path.is_file():
        raise DatasetBuildError(
            f"{scope} requires a passing {prerequisite} xTB pilot"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("scope") != prerequisite
        or summary.get("config_sha256") != config_hash
        or summary.get("source_hashes") != dict(source_hashes)
        or summary.get("coverage_gate_pass") is not True
    ):
        raise DatasetBuildError(
            f"{scope} prerequisite xTB pilot is stale or failed"
        )


def generate_xtb_features(
    *,
    scope: str = "pilot1024",
    config_path: str | Path = DEFAULT_CONFIG,
) -> dict[str, object]:
    """Generate/resume fixed-G1 gas + ALPB-DMSO local4/global6."""

    config_file = Path(config_path).resolve()
    config = _read_config(config_file)
    paths = _resolve_inputs(config)
    inventory, scope_directory, _ = _load_inventory(
        config_file, config, scope
    )
    config_hash = sha256_file(config_file)
    source_hashes = _source_hashes(config_file)
    _require_expansion_gate(
        scope=scope,
        config=config,
        config_hash=config_hash,
        source_hashes=source_hashes,
    )
    generate_geometries(scope=scope, config_path=config_file)
    working = (ROOT / str(config["working_directory"])).resolve()
    geometry_directory = working / "cache" / "geometry"
    cache_directory = working / "cache" / "xtb"
    cache_directory.mkdir(parents=True, exist_ok=True)
    geometries = {
        str(row["source_id"]): json.loads(
            (
                geometry_directory
                / _cache_name(str(row["source_id"]))
            ).read_text(encoding="utf-8")
        )
        for row in inventory.to_dict(orient="records")
    }
    completed = 0
    reused = 0
    with tempfile.TemporaryDirectory(
        prefix="nucpred_esnuel_d_node_xtb_distribution_"
    ) as raw_distribution:
        binary = xtb._safe_extract_archive(
            paths["xtb_archive"], Path(raw_distribution)
        )
        tce = _load_or_build_tce(
            working=working,
            config=config,
            config_sha256=config_hash,
            source_hashes=source_hashes,
            binary=binary,
        )
        pending: list[dict[str, object]] = []
        for row in inventory.to_dict(orient="records"):
            source_id = str(row["source_id"])
            path = cache_directory / _cache_name(source_id)
            if path.is_file():
                try:
                    payload = json.loads(
                        path.read_text(encoding="utf-8")
                    )
                except (OSError, UnicodeError, json.JSONDecodeError):
                    payload = {}
                if (
                    _cache_entry_matches(
                        payload,
                        row,
                        schema_version=XTB_CACHE_SCHEMA,
                        config_sha256=config_hash,
                        source_hashes=source_hashes,
                    )
                    and payload.get("geometry_xyz_sha256")
                    == geometries[source_id].get("xyz_sha256")
                ):
                    reused += 1
                    continue
                raise DatasetBuildError(
                    f"xTB cache parity mismatch: {source_id}"
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
                        f"{source_id}: G1 unavailable; xTB is masked"
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
            except BaseException as error:
                payload = _failed_xtb_record(
                    row,
                    geometry,
                    error,
                    config_sha256=config_hash,
                    source_hashes=source_hashes,
                )
            return path, payload

        workers = min(
            int(config["execution"]["workers"]),
            max(1, len(pending)),
        )
        if pending:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(calculate, row) for row in pending
                ]
                for future in as_completed(futures):
                    path, payload = future.result()
                    atomic_write_json(
                        path, _json_safe(payload), ensure_ascii=False
                    )
                    completed += 1
                    if completed % 10 == 0 or completed == len(pending):
                        print(
                            f"ESNUEL xTB {scope}: "
                            f"{completed}/{len(pending)} new records",
                            flush=True,
                        )
    payloads = [
        json.loads(
            (
                cache_directory / _cache_name(str(source_id))
            ).read_text(encoding="utf-8")
        )
        for source_id in inventory["source_id"].astype(str)
    ]
    successes = [
        row for row in payloads if row.get("status") == "success"
    ]
    complete = [
        row
        for row in payloads
        if row.get("complete_xtb10") is True
    ]
    fraction = len(complete) / len(payloads) if payloads else 0.0
    threshold = float(config["minimum_complete_xtb10_fraction"])
    qc_counts: dict[str, int] = {}
    for row in successes:
        for name, audit in dict(row.get("qc", {})).items():
            if not dict(audit).get("passed"):
                qc_counts[name] = qc_counts.get(name, 0) + 1
    summary = {
        "schema_version": (
            "nucpred.esnuel-d-node-xtb-scope-summary.v1"
        ),
        "scope": scope,
        "requested_record_count": len(payloads),
        "execution_success_record_count": len(successes),
        "execution_failure_record_count": len(payloads) - len(successes),
        "complete_xtb10_record_count": len(complete),
        "complete_xtb10_fraction": fraction,
        "minimum_complete_xtb10_fraction": threshold,
        "coverage_gate_pass": fraction >= threshold,
        "new_record_count": completed,
        "reused_record_count": reused,
        "config_sha256": config_hash,
        "source_hashes": source_hashes,
        "selected_source_id_sha256": _source_id_digest(
            inventory["source_id"].astype(str).tolist()
        ),
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
    atomic_write_json(
        scope_directory / "xtb_summary.json",
        summary,
        ensure_ascii=False,
    )
    if not summary["coverage_gate_pass"]:
        raise DatasetBuildError(
            f"xTB {scope} complete coverage {fraction:.3%} is below "
            f"{threshold:.1%}; stop for user decision"
        )
    return summary


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
    pd.DataFrame,
]:
    records: list[dict[str, object]] = []
    atoms: list[dict[str, object]] = []
    molecules: list[dict[str, object]] = []
    primitives: list[dict[str, object]] = []
    geometry_ledger: list[dict[str, object]] = []
    calculation_ledger: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    geometry_directory = working / "cache" / "geometry"
    xtb_directory = working / "cache" / "xtb"
    for row in inventory.to_dict(orient="records"):
        source_id = str(row["source_id"])
        cache_name = _cache_name(source_id)
        geometry = json.loads(
            (geometry_directory / cache_name).read_text(encoding="utf-8")
        )
        electronic = json.loads(
            (xtb_directory / cache_name).read_text(encoding="utf-8")
        )
        atom_count = int(row["model_all_atom_count"])
        local = _float_matrix(
            electronic["local_values"],
            atom_count,
            len(LOCAL_FEATURES),
        )
        local_mask = np.asarray(
            electronic["local_mask"], dtype=bool
        )
        global_values = _float_array(
            electronic["global_values"], len(GLOBAL_FEATURES)
        )
        global_mask = np.asarray(
            electronic["global_mask"], dtype=bool
        )
        if (
            local_mask.shape != local.shape
            or global_mask.shape != global_values.shape
        ):
            raise DatasetBuildError(
                f"{source_id}: xTB feature mask shape changed"
            )
        numbers = [
            int(value)
            for value in _as_sequence(row["model_atomic_numbers_json"])
        ]
        records.append(
            {
                **row,
                "g1_status": str(geometry["status"]),
                "g1_force_field": geometry.get("force_field"),
                "g1_fallback_reason": geometry.get("fallback_reason"),
                "g1_failure_reason": geometry.get("error", ""),
                "g1_selected_energy_kcal_mol": geometry.get(
                    "selected_energy_kcal_mol"
                ),
                "g1_positions_angstrom_json": _json_compact(
                    geometry.get("positions_angstrom", [])
                ),
                "g1_xyz_sha256": str(
                    geometry.get("xyz_sha256", "")
                ),
                "node_local4_json": _json_compact(local.tolist()),
                "node_local4_available_json": _json_compact(
                    local_mask.tolist()
                ),
                "molecule_global6_json": _json_compact(
                    global_values.tolist()
                ),
                "molecule_global6_available_json": _json_compact(
                    global_mask.tolist()
                ),
                "complete_xtb10": bool(
                    electronic["complete_xtb10"]
                ),
                "electronic_cache_status": str(electronic["status"]),
            }
        )
        for atom_index, atomic_number in enumerate(numbers):
            atom_row: dict[str, object] = {
                "source_id": source_id,
                "atom_index": atom_index,
                "atomic_number": atomic_number,
                "is_hydrogen": atomic_number == 1,
                "pretraining_role": str(row["pretraining_role"]),
            }
            for feature_index, name in enumerate(LOCAL_FEATURES):
                atom_row[name] = float(local[atom_index, feature_index])
                atom_row[f"{name}__available"] = bool(
                    local_mask[atom_index, feature_index]
                )
            atoms.append(atom_row)
        molecule_row: dict[str, object] = {
            "source_id": source_id,
            "pretraining_role": str(row["pretraining_role"]),
        }
        for feature_index, name in enumerate(GLOBAL_FEATURES):
            molecule_row[name] = float(global_values[feature_index])
            molecule_row[f"{name}__available"] = bool(
                global_mask[feature_index]
            )
        molecules.append(molecule_row)
        primitives.append(
            {
                "source_id": source_id,
                "status": str(electronic["status"]),
                "formal_charge": electronic.get("formal_charge"),
                "neutral_uhf": electronic.get("neutral_uhf"),
                "cation_uhf": electronic.get("cation_uhf"),
                "solvent": electronic.get("solvent", "dmso"),
                "tce_homo_hartree": electronic.get(
                    "tce_homo_hartree"
                ),
                "qc_json": _json_compact(electronic.get("qc", {})),
                "raw_xtb_primitives_json": _json_compact(
                    electronic.get("raw", {})
                ),
                "error_type": electronic.get("error_type", ""),
                "error": electronic.get("error", ""),
            }
        )
        geometry_ledger.append(
            {
                "source_id": source_id,
                "status": str(geometry["status"]),
                "method": geometry.get("method", "ETKDGv3"),
                "force_field": geometry.get("force_field", ""),
                "fallback_reason": geometry.get("fallback_reason", ""),
                "requested_conformer_count": geometry.get(
                    "requested_conformer_count", 20
                ),
                "embedded_conformer_count": geometry.get(
                    "embedded_conformer_count", 0
                ),
                "converged_conformer_count": geometry.get(
                    "converged_conformer_count", 0
                ),
                "selected_energy_kcal_mol": geometry.get(
                    "selected_energy_kcal_mol"
                ),
                "xyz_sha256": geometry.get("xyz_sha256", ""),
                "wall_seconds": geometry.get("wall_seconds"),
                "error_type": geometry.get("error_type", ""),
                "error": geometry.get("error", ""),
            }
        )
        calculation_ledger.extend(
            dict(entry) for entry in electronic.get("ledger", [])
        )
        if geometry.get("status") != "success":
            failures.append(
                {
                    "source_id": source_id,
                    "stage": "geometry",
                    "calculation": "G1_ETKDGv3_MMFF94s_or_UFF",
                    "environment": "force_field",
                    "error_type": geometry.get("error_type", ""),
                    "error": geometry.get("error", ""),
                }
            )
        if electronic.get("status") != "success":
            failures.append(
                {
                    "source_id": source_id,
                    "stage": "xtb_record",
                    "calculation": "fixed_G1_descriptor_panel",
                    "environment": "gas_and_alpb_dmso",
                    "error_type": electronic.get("error_type", ""),
                    "error": electronic.get("error", ""),
                }
            )
        for entry in electronic.get("ledger", []):
            if not bool(entry.get("normal_termination")):
                failures.append(
                    {
                        "source_id": source_id,
                        "stage": "xtb_subcalculation",
                        "calculation": entry.get("calculation", ""),
                        "environment": entry.get("environment", ""),
                        "error_type": "subcalculation_failure",
                        "error": entry.get("error", ""),
                    }
                )
    failure_columns = (
        "source_id",
        "stage",
        "calculation",
        "environment",
        "error_type",
        "error",
    )
    return (
        pd.DataFrame(records),
        pd.DataFrame(atoms),
        pd.DataFrame(molecules),
        pd.DataFrame(primitives),
        pd.DataFrame(geometry_ledger),
        pd.DataFrame(calculation_ledger),
        pd.DataFrame(failures, columns=failure_columns),
    )


def _coverage_payload(
    records: pd.DataFrame,
    atom_features: pd.DataFrame,
    molecule_features: pd.DataFrame,
    *,
    threshold: float,
) -> dict[str, object]:
    fraction = float(records["complete_xtb10"].mean())
    return {
        "schema_version": (
            "nucpred.esnuel-d-node-xtb-coverage.v1"
        ),
        "record_count": len(records),
        "atom_count": len(atom_features),
        "complete_xtb10_record_count": int(
            records["complete_xtb10"].sum()
        ),
        "complete_xtb10_fraction": fraction,
        "minimum_complete_xtb10_fraction": threshold,
        "coverage_gate_pass": fraction >= threshold,
        "local_feature_coverage": {
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
        },
        "global_feature_coverage": {
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
        },
        "missing_value_policy": (
            "training_fold_median_imputation_plus_availability_masks"
        ),
    }


def _file_entry(
    path: Path, role: str, format_name: str
) -> dict[str, object]:
    return {
        "path": path.name,
        "role": role,
        "format": format_name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def finalize_dataset(
    *,
    scope: str = "pilot1024",
    config_path: str | Path = DEFAULT_CONFIG,
    output_directory: str | Path | None = None,
) -> dict[str, object]:
    """Finalize a passing scope into immutable, hash-manifested assets."""

    config_file = Path(config_path).resolve()
    config = _read_config(config_file)
    paths = _resolve_inputs(config)
    inventory, scope_directory, selection_manifest = _load_inventory(
        config_file, config, scope
    )
    summary_path = scope_directory / "xtb_summary.json"
    if not summary_path.is_file():
        raise DatasetBuildError("Finalize requires a passing xTB scope")
    execution_summary = json.loads(
        summary_path.read_text(encoding="utf-8")
    )
    current_hashes = _source_hashes(config_file)
    if (
        execution_summary.get("scope") != scope
        or execution_summary.get("config_sha256")
        != sha256_file(config_file)
        or execution_summary.get("source_hashes") != current_hashes
        or execution_summary.get("coverage_gate_pass") is not True
        or int(execution_summary.get("requested_record_count", -1))
        != len(inventory)
    ):
        raise DatasetBuildError("xTB scope summary is stale or failed")
    working = (ROOT / str(config["working_directory"])).resolve()
    (
        records,
        atom_features,
        molecule_features,
        primitives,
        geometry_ledger,
        calculation_ledger,
        failures,
    ) = _dataset_tables(inventory, working=working)
    coverage = _coverage_payload(
        records,
        atom_features,
        molecule_features,
        threshold=float(config["minimum_complete_xtb10_fraction"]),
    )
    if coverage["coverage_gate_pass"] is not True:
        raise DatasetBuildError("Final xTB10 coverage gate failed")
    dataset_id = f"{config['dataset_id_prefix']}-{scope}"
    output = (
        Path(output_directory).resolve()
        if output_directory is not None
        else (
            ROOT
            / str(config["output_root"])
            / dataset_id
        ).resolve()
    )
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite versioned pretraining data: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    overlap = pd.read_csv(scope_directory / "overlap_audit.csv")
    target_audit = pd.read_csv(
        scope_directory / "target_connectivity_audit.csv"
    )
    mapping = pd.read_csv(scope_directory / "mapping_audit.csv")
    membership = pd.read_csv(
        scope_directory / "split_membership.csv"
    )
    if (
        len(overlap) != EXPECTED_ESNUEL_RECORDS
        or int(
            overlap[
                "excluded_for_mayr_connectivity_overlap"
            ].sum()
        )
        != EXPECTED_OVERLAP_RECORDS
        or int(selection_manifest["eligible_record_count"])
        != EXPECTED_ELIGIBLE_RECORDS
    ):
        raise DatasetBuildError("47,921 -> 47,915 overlap gate failed")
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
    native_counts = {
        role: int(records["pretraining_role"].eq(role).sum())
        for role in NATIVE_ROLE_ORDER
    }
    summary = {
        "schema_version": (
            "nucpred.esnuel-d-node-xtb-summary.v1"
        ),
        "dataset_id": dataset_id,
        "scope": scope,
        "parent_esnuel_record_count": EXPECTED_ESNUEL_RECORDS,
        "excluded_mayr_connectivity_overlap_record_count": (
            EXPECTED_OVERLAP_RECORDS
        ),
        "eligible_esnuel_record_count": EXPECTED_ELIGIBLE_RECORDS,
        "selected_record_count": len(records),
        "selected_native_role_counts": native_counts,
        "atom_count": len(atom_features),
        "hydrogen_atom_count": int(
            atom_features["is_hydrogen"].sum()
        ),
        "complete_xtb10_record_count": int(
            records["complete_xtb10"].sum()
        ),
        "complete_xtb10_fraction": float(
            records["complete_xtb10"].mean()
        ),
        "coverage_gate_pass": True,
        "failure_ledger_row_count": len(failures),
        "geometry": "G1 ETKDGv3 20-conformer MMFF94s/UFF",
        "electronic_method": (
            "fixed-G1 GFN1-xTB gas + ALPB-DMSO"
        ),
        "pretraining": True,
        "mayr_branch_loaded": False,
        "mayr_structure_identity_loaded_for_exclusion": True,
        "mayr_labels_used": False,
    }
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{dataset_id}.staging-",
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
                path,
                index=False,
                engine="pyarrow",
                compression="zstd",
            )
            files.append((path, role, "parquet"))

        def csv(
            name: str, frame: pd.DataFrame, role: str
        ) -> None:
            path = staging / name
            frame.to_csv(path, index=False, lineterminator="\n")
            files.append((path, role, "csv"))

        def json_file(
            name: str, value: object, role: str
        ) -> None:
            path = staging / name
            atomic_write_json(path, _json_safe(value), ensure_ascii=False)
            files.append((path, role, "json"))

        parquet("records.parquet", records, "records")
        parquet(
            "atom_features.parquet",
            atom_features,
            "atom_features",
        )
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
            "geometry_ledger.csv",
            geometry_ledger,
            "geometry_ledger",
        )
        csv(
            "calculation_ledger.csv",
            calculation_ledger,
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
            selection_manifest,
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
            _file_entry(path, role, format_name)
            for path, role, format_name in files
        ]
        manifest = {
            "schema_version": DATASET_SCHEMA,
            "dataset_id": dataset_id,
            "scope": scope,
            "generated_by": (
                "nucpred.datasets.esnuel_d_node_xtb_pretraining"
            ),
            "builder_source": _display_path(Path(__file__)),
            "builder_source_sha256": sha256_file(Path(__file__)),
            "config": {
                "path": _display_path(config_file),
                "sha256": sha256_file(config_file),
            },
            "source_hashes": current_hashes,
            "inputs": [
                {
                    "path": _display_path(paths["esnuel_records"]),
                    "sha256": sha256_file(paths["esnuel_records"]),
                    "role": (
                        "esnuel_only_mca_gcs_graph_parent"
                    ),
                },
                {
                    "path": _display_path(paths["target_records"]),
                    "sha256": sha256_file(paths["target_records"]),
                    "role": (
                        "mayr_structure_identity_for_overlap_only"
                    ),
                    "columns_loaded": [
                        "source_id",
                        *config["parents"]["mayr_target"][
                            "connectivity_smiles_columns"
                        ],
                    ],
                },
                {
                    "path": _display_path(paths["xtb_archive"]),
                    "sha256": sha256_file(paths["xtb_archive"]),
                    "role": "xtb_distribution",
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
                "minimum_complete_xtb10_fraction": float(
                    config["minimum_complete_xtb10_fraction"]
                ),
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
        "dataset_id": dataset_id,
        "scope": scope,
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
    """Verify immutable assets, overlap hard counts, mapping, and masks."""

    selected = Path(directory).resolve()
    manifest = json.loads(
        (selected / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("schema_version") != DATASET_SCHEMA:
        raise DatasetBuildError("Unsupported pretraining dataset manifest")
    contracts = manifest.get("contracts", {})
    if (
        int(contracts.get("parent_esnuel_records", -1))
        != EXPECTED_ESNUEL_RECORDS
        or int(
            contracts.get(
                "excluded_connectivity_overlap_records", -1
            )
        )
        != EXPECTED_OVERLAP_RECORDS
        or int(contracts.get("eligible_esnuel_records", -1))
        != EXPECTED_ELIGIBLE_RECORDS
        or contracts.get("mayr_branch_loaded") is not False
        or contracts.get("mayr_labels_loaded") is not False
        or contracts.get("mayr_labels_used") is not False
        or contracts.get("hydrogen_is_ordinary_element") is not True
        or contracts.get("no_xtb_geometry_optimization") is not True
    ):
        raise DatasetBuildError("Pretraining data contracts changed")
    for asset in manifest["assets"]:
        path = selected / str(asset["path"])
        if path.stat().st_size != int(asset["bytes"]):
            raise DatasetBuildError(f"Asset size changed: {path}")
        if sha256_file(path) != str(asset["sha256"]):
            raise DatasetBuildError(f"Asset hash changed: {path}")
    records = pd.read_parquet(selected / "records.parquet")
    atoms = pd.read_parquet(selected / "atom_features.parquet")
    molecules = pd.read_parquet(
        selected / "molecule_features.parquet"
    )
    overlap = pd.read_csv(selected / "overlap_audit.csv")
    mapping = pd.read_csv(selected / "mapping_audit.csv")
    membership = pd.read_csv(selected / "split_membership.csv")
    coverage = json.loads(
        (selected / "coverage.json").read_text(encoding="utf-8")
    )
    summary = json.loads(
        (selected / "summary.json").read_text(encoding="utf-8")
    )
    if (
        len(overlap) != EXPECTED_ESNUEL_RECORDS
        or int(
            overlap[
                "excluded_for_mayr_connectivity_overlap"
            ].sum()
        )
        != EXPECTED_OVERLAP_RECORDS
        or int(summary["eligible_esnuel_record_count"])
        != EXPECTED_ELIGIBLE_RECORDS
    ):
        raise DatasetBuildError("Overlap hard-count audit changed")
    if (
        len(records) != int(summary["selected_record_count"])
        or records["source_id"].astype(str).duplicated().any()
        or len(molecules) != len(records)
        or len(mapping) != len(records)
        or len(membership) != len(records)
        or coverage.get("coverage_gate_pass") is not True
    ):
        raise DatasetBuildError("Final table count/coverage invariant failed")
    if set(records["pretraining_role"].astype(str)) - set(
        NATIVE_ROLE_ORDER
    ):
        raise DatasetBuildError("Native pretraining role changed")
    if not mapping["status"].astype(str).eq("pass").all():
        raise DatasetBuildError("Graph mapping audit failed")
    if (
        int(mapping["added_h_numeric_supervision_count"].sum()) != 0
        or int(mapping["added_h_site_supervision_count"].sum()) != 0
    ):
        raise DatasetBuildError("Added H received forged ESNUEL labels")
    atom_counts = atoms.groupby("source_id").size().to_dict()
    for row in records.itertuples(index=False):
        count = int(row.model_all_atom_count)
        if int(atom_counts.get(str(row.source_id), -1)) != count:
            raise DatasetBuildError(
                f"{row.source_id}: atom-feature count changed"
            )
        local = _float_matrix(
            row.node_local4_json, count, len(LOCAL_FEATURES)
        )
        local_mask = np.asarray(
            _as_sequence(row.node_local4_available_json),
            dtype=bool,
        )
        global_values = _float_array(
            row.molecule_global6_json, len(GLOBAL_FEATURES)
        )
        global_mask = np.asarray(
            _as_sequence(row.molecule_global6_available_json),
            dtype=bool,
        )
        if (
            local.shape != local_mask.shape
            or global_values.shape != global_mask.shape
        ):
            raise DatasetBuildError(
                f"{row.source_id}: feature/mask shape changed"
            )
    return {
        "schema_version": (
            "nucpred.esnuel-d-node-xtb-verification.v1"
        ),
        "dataset_id": str(manifest["dataset_id"]),
        "scope": str(manifest["scope"]),
        "status": "pass",
        "record_count": len(records),
        "atom_count": len(atoms),
        "eligible_esnuel_record_count": EXPECTED_ELIGIBLE_RECORDS,
        "excluded_overlap_record_count": EXPECTED_OVERLAP_RECORDS,
        "verified_file_count": len(manifest["assets"]),
        "manifest_sha256": sha256_file(
            selected / "dataset_manifest.json"
        ),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build ESNUEL-only D-isomorphic G1 + xTB pretraining data."
        )
    )
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("select", "geometry", "xtb", "finalize"):
        command = commands.add_parser(action)
        command.add_argument(
            "--scope",
            choices=("pilot1024", "pilot4096", "full"),
            default="pilot1024",
        )
        command.add_argument(
            "--config", type=Path, default=DEFAULT_CONFIG
        )
        if action == "finalize":
            command.add_argument("--output-directory", type=Path)
    verify = commands.add_parser("verify")
    verify.add_argument("--directory", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.action == "select":
        payload = build_selection(
            scope=args.scope, config_path=args.config
        )
    elif args.action == "geometry":
        payload = generate_geometries(
            scope=args.scope, config_path=args.config
        )
    elif args.action == "xtb":
        payload = generate_xtb_features(
            scope=args.scope, config_path=args.config
        )
    elif args.action == "finalize":
        payload = finalize_dataset(
            scope=args.scope,
            config_path=args.config,
            output_directory=args.output_directory,
        )
    else:
        payload = verify_dataset(args.directory)
    print(json.dumps(_json_safe(payload), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
