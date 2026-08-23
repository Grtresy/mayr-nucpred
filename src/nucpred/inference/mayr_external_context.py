"""Target-blind pure-solvent feature construction for external Mayr queries."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Mapping

import pandas as pd
from rdkit import Chem
from rdkit import rdBase

from nucpred.core.files import atomic_write_json, sha256_file
from nucpred.datasets import mayr_pure_alpb_node_xtb as node_xtb
from nucpred.datasets.mayr_site_n import (
    _connectivity_identity,
    _species_context_ids,
)
from nucpred.features.all_atom_graph import (
    EDGE_CATEGORY_SIZES,
    NODE_CATEGORY_SIZES,
    assert_category_ranges,
    featurize_explicit_molecule,
)
from nucpred.project import get_project_layout
from nucpred.protocols import xtb_runtime as xtb


ROOT = get_project_layout().root
DEFAULT_XTB_CONFIG = ROOT / "configs/mayr_pure_alpb_node_xtb.toml"
DEFAULT_REFERENCE_CONTEXTS = (
    ROOT / "data/processed/mayr_site_n/mayr-site-n-20260805-v2" / "contexts.parquet"
)


class ExternalMayrContextError(RuntimeError):
    """Raised when an external context cannot satisfy the frozen feature contract."""


@dataclass(frozen=True, slots=True)
class ExternalContextBuild:
    contexts: pd.DataFrame
    feature_audit: pd.DataFrame


def _json_compact(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _canonical_solvent(value: object) -> str:
    return "".join(str(value).casefold().split())


def _stable_geometry_seed(species_id: str) -> int:
    """Return a positive RDKit-compatible seed derived only from species identity."""

    digest = hashlib.sha256(
        f"mayr-external-g1-seed-v1\0{species_id}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big") % (2**31 - 2) + 1


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


def _solvent_descriptor_row(
    reference_contexts: pd.DataFrame,
    *,
    solvent_raw: str,
) -> dict[str, float]:
    selected = reference_contexts.loc[
        reference_contexts["solvent_raw"]
        .map(_canonical_solvent)
        .eq(_canonical_solvent(solvent_raw)),
        list(node_xtb.SOLVENT_DESCRIPTOR_COLUMNS),
    ].drop_duplicates()
    if len(selected) != 1:
        raise ExternalMayrContextError(
            f"Expected one frozen descriptor pattern for {solvent_raw!r}, "
            f"found {len(selected)}"
        )
    values = {
        column: float(selected.iloc[0][column])
        for column in node_xtb.SOLVENT_DESCRIPTOR_COLUMNS
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise ExternalMayrContextError(
            f"Frozen solvent descriptors are incomplete for {solvent_raw!r}"
        )
    return values


def _inventory_row(
    query: Mapping[str, object],
    *,
    cohort_index: int,
    config: Mapping[str, Any],
    reference_contexts: pd.DataFrame,
) -> dict[str, object]:
    required = {"fit_id", "canonical_smiles", "formal_charge", "solvent_raw"}
    missing = sorted(required - set(query))
    if missing:
        raise ExternalMayrContextError(f"External query lacks fields: {missing}")
    source_id = str(query["fit_id"])
    molecule = Chem.MolFromSmiles(str(query["canonical_smiles"]))
    if molecule is None:
        raise ExternalMayrContextError(f"Invalid external SMILES: {source_id}")
    if len(Chem.GetMolFrags(molecule)) != 1:
        raise ExternalMayrContextError(
            f"External pure-context adapter requires one fragment: {source_id}"
        )
    canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    if canonical != str(query["canonical_smiles"]):
        raise ExternalMayrContextError(
            f"External SMILES must already be canonical: {source_id}: {canonical}"
        )
    observed_charge = int(Chem.GetFormalCharge(molecule))
    if observed_charge != int(query["formal_charge"]):
        raise ExternalMayrContextError(
            f"External formal charge changed for {source_id}"
        )
    solvent_raw = str(query["solvent_raw"])
    solvent_aliases: dict[str, tuple[str, str]] = {}
    for key, value in config["solvents"].items():
        pair = (str(key), str(value))
        for alias in pair:
            normalized = _canonical_solvent(alias)
            existing = solvent_aliases.get(normalized)
            if existing is not None and existing != pair:
                raise ExternalMayrContextError(
                    f"Ambiguous frozen solvent alias: {alias!r}"
                )
            solvent_aliases[normalized] = pair
    solvent_key = _canonical_solvent(solvent_raw)
    if solvent_key not in solvent_aliases:
        raise ExternalMayrContextError(
            f"Unsupported frozen pure solvent: {solvent_raw!r}"
        )
    canonical_solvent_raw, xtb_solvent = solvent_aliases[solvent_key]

    explicit = Chem.AddHs(Chem.Mol(molecule), addCoords=False)
    graph = featurize_explicit_molecule(
        explicit,
        source_atom_count=molecule.GetNumAtoms(),
    )
    assert_category_ranges(graph.node_categorical, NODE_CATEGORY_SIZES)
    assert_category_ranges(graph.edge_categorical, EDGE_CATEGORY_SIZES)
    row: dict[str, object] = {
        "source_id": source_id,
        "cohort_index": int(cohort_index),
        "solvent_raw": canonical_solvent_raw,
        "formal_charge": observed_charge,
        "species_state": "external_query",
        "xtb_alpb_solvent": xtb_solvent,
        "model_canonical_smiles": canonical,
        "model_mapped_smiles": _mapped_smiles(molecule),
        "model_source_atom_count": molecule.GetNumAtoms(),
        "model_all_atom_count": graph.atom_count,
        "model_hydrogen_atom_count": sum(
            atomic_number == 1 for atomic_number in graph.atomic_numbers
        ),
        "model_formal_charge": observed_charge,
        "model_radical_electrons": sum(
            atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms()
        ),
        "model_atomic_numbers_json": _json_compact(graph.atomic_numbers),
        "model_node_categorical_json": _json_compact(graph.node_categorical),
        "model_directed_edges_json": _json_compact(graph.directed_edges),
        "model_edge_categorical_json": _json_compact(graph.edge_categorical),
        "model_hydrogen_parent_index_json": _json_compact(graph.hydrogen_parent_index),
        "model_graph_sha256": graph.mapping_sha256,
        "model_to_original_source_atom_json": _json_compact(
            list(range(molecule.GetNumAtoms()))
        ),
        "equivalent_h_groups_json": "[]",
        **_solvent_descriptor_row(
            reference_contexts,
            solvent_raw=canonical_solvent_raw,
        ),
    }
    species_id, context_id = _species_context_ids(row)
    connectivity_id, connectivity_key, identity_method = _connectivity_identity(
        canonical
    )
    row.update(
        {
            "species_id": species_id,
            "context_id": context_id,
            "connectivity_id": connectivity_id,
            "connectivity_inchi_key": connectivity_key,
            "connectivity_identity_method": identity_method,
            "geometry_seed": _stable_geometry_seed(species_id),
        }
    )
    return row


def _cache_path(directory: Path, source_id: str) -> Path:
    digest = hashlib.sha256(source_id.encode("utf-8")).hexdigest()
    return directory / f"{digest}.json"


def _frozen_species_geometry(
    row: Mapping[str, object],
    *,
    reference_contexts: pd.DataFrame,
    config: Mapping[str, Any],
    config_sha256: str,
    source_hashes: Mapping[str, str],
) -> dict[str, object]:
    """Reuse one target-independent G1 geometry for a seen connectivity."""

    matches = reference_contexts.loc[
        reference_contexts["species_id"].astype(str).eq(str(row["species_id"]))
        & reference_contexts["model_graph_sha256"]
        .astype(str)
        .eq(str(row["model_graph_sha256"]))
        & reference_contexts["g1_status"].astype(str).eq("success")
    ].sort_values(["g1_xyz_sha256", "context_id"], kind="stable")
    if matches.empty:
        raise ExternalMayrContextError(
            f"No frozen G1 geometry exists for seen species {row['species_id']}"
        )
    source = matches.iloc[0]
    atomic_numbers = tuple(
        int(value) for value in json.loads(str(row["model_atomic_numbers_json"]))
    )
    source_numbers = tuple(
        int(value) for value in json.loads(str(source["model_atomic_numbers_json"]))
    )
    if source_numbers != atomic_numbers:
        raise ExternalMayrContextError("Frozen G1 atom ordering changed")
    positions = json.loads(str(source["g1_positions_angstrom_json"]))
    xyz = node_xtb._canonical_xyz(
        atomic_numbers,
        positions,
        comment=(
            f"{row['source_id']} reused frozen species G1 fixed-geometry xTB input"
        ),
        decimals=int(config["geometry"]["coordinate_precision_decimals"]),
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
        "derived_random_seed": None,
        "method": "frozen_seen_species_G1_reuse",
        "force_field": str(source["g1_force_field"]),
        "fallback_reason": (
            None
            if pd.isna(source["g1_fallback_reason"])
            else str(source["g1_fallback_reason"])
        ),
        "requested_conformer_count": None,
        "embedded_conformer_count": None,
        "converged_conformer_count": None,
        "selected_conformer_id": None,
        "convergence_code": None,
        "selected_energy_kcal_mol": float(source["g1_selected_energy_kcal_mol"]),
        "atomic_numbers": list(atomic_numbers),
        "positions_angstrom": positions,
        "xyz_sha256": hashlib.sha256(xyz.encode("utf-8")).hexdigest(),
        "xyz_text": xyz,
        "wall_seconds": 0.0,
        "rdkit_version": rdBase.rdkitVersion,
        "geometry_source": "frozen_seen_species_context",
        "source_context_id": str(source["context_id"]),
        "source_g1_xyz_sha256": str(source["g1_xyz_sha256"]),
        "selection_rule": "lexicographically_smallest_g1_xyz_sha256_then_context_id",
        "selection_reads_target_or_site": False,
    }


def _target_blind_geometry(
    row: Mapping[str, object],
    *,
    reference_contexts: pd.DataFrame,
    config: Mapping[str, Any],
    config_sha256: str,
    source_hashes: Mapping[str, str],
) -> dict[str, object]:
    """Reuse a frozen G1 when possible, otherwise generate deterministic G1."""

    seen_species = reference_contexts["species_id"].astype(str).eq(
        str(row["species_id"])
    ) & reference_contexts["model_graph_sha256"].astype(str).eq(
        str(row["model_graph_sha256"])
    )
    if bool(seen_species.any()):
        return _frozen_species_geometry(
            row,
            reference_contexts=reference_contexts,
            config=config,
            config_sha256=config_sha256,
            source_hashes=source_hashes,
        )
    geometry = node_xtb._geometry_record(
        row,
        config=config,
        config_sha256=config_sha256,
        source_hashes=source_hashes,
    )
    geometry.update(
        {
            "geometry_source": "generated_unseen_species_target_blind",
            "source_context_id": None,
            "source_g1_xyz_sha256": None,
            "selection_rule": (
                "minimum_converged_force_field_energy_then_conformer_id"
            ),
            "selection_reads_target_or_site": False,
        }
    )
    return geometry


def _load_cache(
    path: Path,
    *,
    schema_version: str,
    row: Mapping[str, object],
    config_sha256: str,
    source_hashes: Mapping[str, str],
) -> dict[str, object] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": schema_version,
        "source_id": str(row["source_id"]),
        "config_sha256": config_sha256,
        "source_hashes": dict(source_hashes),
        "model_graph_sha256": str(row["model_graph_sha256"]),
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ExternalMayrContextError(f"Stale external feature cache: {path}")
    if schema_version == "nucpred.mayr-node-xtb-cache.v1" and payload.get(
        "solvent"
    ) != str(row["xtb_alpb_solvent"]):
        raise ExternalMayrContextError(f"External xTB solvent cache changed: {path}")
    return payload


def _context_row(
    row: Mapping[str, object],
    *,
    geometry: Mapping[str, object],
    electronic: Mapping[str, object],
) -> dict[str, object]:
    if geometry.get("status") != "success":
        raise ExternalMayrContextError(
            f"G1 geometry failed for {row['source_id']}: {geometry.get('error', '')}"
        )
    if (
        electronic.get("status") != "success"
        or electronic.get("complete_xtb10") is not True
    ):
        raise ExternalMayrContextError(
            f"Complete xTB10 features unavailable for {row['source_id']}: "
            f"{electronic.get('error', '')}"
        )
    context = dict(row)
    context.update(
        {
            "g1_status": "success",
            "g1_force_field": str(geometry["force_field"]),
            "g1_fallback_reason": geometry.get("fallback_reason"),
            "g1_failure_reason": "",
            "g1_selected_energy_kcal_mol": float(geometry["selected_energy_kcal_mol"]),
            "g1_positions_angstrom_json": _json_compact(geometry["positions_angstrom"]),
            "g1_xyz_sha256": str(geometry["xyz_sha256"]),
            "g1_geometry_source": str(geometry["geometry_source"]),
            "g1_geometry_seed": geometry.get("derived_random_seed"),
            "node_local4_json": _json_compact(electronic["local_values"]),
            "node_local4_available_json": _json_compact(electronic["local_mask"]),
            "molecule_global6_json": _json_compact(electronic["global_values"]),
            "molecule_global6_available_json": _json_compact(electronic["global_mask"]),
            "complete_xtb10": True,
            "electronic_cache_status": "success",
            "representative_source_id": str(row["source_id"]),
            "context_source_ids_json": _json_compact([str(row["source_id"])]),
            "context_measurement_count": 0,
            "representative_selection_target_independent": True,
            "representative_feature_policy": "external_target_blind_query_v1",
        }
    )
    return context


def build_external_pure_contexts(
    queries: pd.DataFrame,
    *,
    cache_directory: str | Path,
    config_path: str | Path = DEFAULT_XTB_CONFIG,
    reference_contexts_path: str | Path = DEFAULT_REFERENCE_CONTEXTS,
) -> ExternalContextBuild:
    """Build complete model-ready contexts without reading N or site labels."""

    safe_columns = ("fit_id", "canonical_smiles", "formal_charge", "solvent_raw")
    if sorted(queries.columns) != sorted(safe_columns):
        raise ExternalMayrContextError(
            "External feature queries must contain only fit_id, canonical_smiles, "
            "formal_charge, and solvent_raw"
        )
    if queries.empty or queries["fit_id"].astype(str).duplicated().any():
        raise ExternalMayrContextError("External feature query IDs are invalid")

    config_file = Path(config_path).resolve()
    config = node_xtb._read_config(config_file)
    _, _, _, archive = node_xtb._resolve_inputs(config)
    reference_contexts_file = Path(reference_contexts_path).resolve()
    reference_contexts = pd.read_parquet(reference_contexts_file)
    config_hash = sha256_file(config_file)
    frozen_source_hashes = node_xtb._source_hashes(config_file)
    source_hashes = {
        **frozen_source_hashes,
        "external_context_adapter": sha256_file(Path(__file__).resolve()),
        "reference_contexts": sha256_file(reference_contexts_file),
    }
    cache_root = Path(cache_directory).resolve()
    geometry_directory = cache_root / "geometry"
    electronic_directory = cache_root / "xtb"
    geometry_directory.mkdir(parents=True, exist_ok=True)
    electronic_directory.mkdir(parents=True, exist_ok=True)
    inventory = [
        _inventory_row(
            query,
            cohort_index=index,
            config=config,
            reference_contexts=reference_contexts,
        )
        for index, query in enumerate(queries.to_dict(orient="records"))
    ]
    if len({row["context_id"] for row in inventory}) != len(inventory):
        raise ExternalMayrContextError("External context identities are duplicated")

    geometries: dict[str, dict[str, object]] = {}
    for row in inventory:
        source_id = str(row["source_id"])
        path = _cache_path(geometry_directory, source_id)
        payload = _load_cache(
            path,
            schema_version="nucpred.mayr-g1-geometry-cache.v1",
            row=row,
            config_sha256=config_hash,
            source_hashes=source_hashes,
        )
        if payload is None:
            try:
                payload = _target_blind_geometry(
                    row,
                    reference_contexts=reference_contexts,
                    config=config,
                    config_sha256=config_hash,
                    source_hashes=source_hashes,
                )
            except Exception as exc:
                payload = node_xtb._failed_geometry_record(
                    row,
                    exc,
                    config_sha256=config_hash,
                    source_hashes=source_hashes,
                )
            atomic_write_json(path, node_xtb._json_safe(payload), ensure_ascii=False)
        geometries[source_id] = payload

    electronics: dict[str, dict[str, object]] = {}
    pending = [
        row
        for row in inventory
        if _load_cache(
            _cache_path(electronic_directory, str(row["source_id"])),
            schema_version="nucpred.mayr-node-xtb-cache.v1",
            row=row,
            config_sha256=config_hash,
            source_hashes=source_hashes,
        )
        is None
    ]
    tce_cache_path = cache_root / "tce_reference.json"
    binary_required = bool(pending) or not tce_cache_path.is_file()
    with tempfile.TemporaryDirectory(
        prefix="nucpred_external_mayr_xtb_distribution_"
    ) as raw_distribution:
        binary = (
            xtb._safe_extract_archive(archive, Path(raw_distribution))
            if binary_required
            else archive
        )
        tce = node_xtb._load_or_build_tce(
            working=cache_root,
            config=config,
            config_sha256=config_hash,
            source_hashes=source_hashes,
            binary=binary,
        )
        if not math.isfinite(float(tce.get("homo_hartree", math.nan))):
            raise ExternalMayrContextError("External TCE reference is invalid")
        for row in inventory:
            source_id = str(row["source_id"])
            path = _cache_path(electronic_directory, source_id)
            payload = _load_cache(
                path,
                schema_version="nucpred.mayr-node-xtb-cache.v1",
                row=row,
                config_sha256=config_hash,
                source_hashes=source_hashes,
            )
            if payload is None:
                try:
                    if geometries[source_id].get("status") != "success":
                        raise ExternalMayrContextError("G1 geometry unavailable")
                    payload = node_xtb._xtb_record(
                        row,
                        geometries[source_id],
                        config=config,
                        config_sha256=config_hash,
                        source_hashes=source_hashes,
                        binary=binary,
                        tce_homo_hartree=float(tce["homo_hartree"]),
                    )
                except Exception as exc:
                    payload = node_xtb._failed_xtb_record(
                        row,
                        geometries[source_id],
                        exc,
                        config_sha256=config_hash,
                        source_hashes=source_hashes,
                    )
                atomic_write_json(
                    path,
                    node_xtb._json_safe(payload),
                    ensure_ascii=False,
                )
            electronics[source_id] = payload

    context_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    for row in inventory:
        source_id = str(row["source_id"])
        geometry = geometries[source_id]
        electronic = electronics[source_id]
        context_rows.append(_context_row(row, geometry=geometry, electronic=electronic))
        audit_rows.append(
            {
                "fit_id": source_id,
                "context_id": str(row["context_id"]),
                "species_id": str(row["species_id"]),
                "connectivity_id": str(row["connectivity_id"]),
                "model_graph_sha256": str(row["model_graph_sha256"]),
                "geometry_seed": int(row["geometry_seed"]),
                "g1_status": str(geometry.get("status")),
                "g1_force_field": str(geometry.get("force_field", "")),
                "g1_xyz_sha256": str(geometry.get("xyz_sha256", "")),
                "g1_structure_asset_reused": (
                    geometry.get("geometry_source") == "frozen_seen_species_context"
                ),
                "g1_geometry_source": str(geometry.get("geometry_source", "")),
                "g1_source_context_id": str(geometry.get("source_context_id") or ""),
                "g1_source_xyz_sha256": str(geometry.get("source_g1_xyz_sha256") or ""),
                "xtb_status": str(electronic.get("status")),
                "xtb_solvent": str(row["xtb_alpb_solvent"]),
                "xtb_complete": bool(electronic.get("complete_xtb10", False)),
                "feature_cache_policy": "hash_bound_reusable",
                "local_feature_count": len(node_xtb.LOCAL_FEATURES),
                "global_feature_count": len(node_xtb.GLOBAL_FEATURES),
                "target_N_read": False,
                "site_label_read": False,
            }
        )
    return ExternalContextBuild(
        contexts=pd.DataFrame(context_rows)
        .sort_values("context_id")
        .reset_index(drop=True),
        feature_audit=pd.DataFrame(audit_rows)
        .sort_values("fit_id")
        .reset_index(drop=True),
    )
