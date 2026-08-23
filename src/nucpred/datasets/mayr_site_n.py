"""Build the site-object Mayr N database from the frozen 1,136-row cohort.

The parent dataset remains immutable evidence.  This builder projects it into
separate species, solvent-context, site-object, raw-measurement, and aggregated
target tables.  A target is one ``context + site`` object; repeated observations
are averaged only after their raw values and provenance have been preserved.

Candidate enumeration is deliberately target-independent.  It receives only a
model graph and structural metadata, and unmeasured candidates never become
negative labels.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from pathlib import Path
import tempfile
import tomllib
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import inchi
from sklearn.model_selection import GroupShuffleSplit

from nucpred.core.files import atomic_write_json, sha256_file
from nucpred.project import get_project_layout


_LAYOUT = get_project_layout()
ROOT = _LAYOUT.root
DEFAULT_CONFIG = ROOT / "configs/mayr_site_n.toml"
CONFIG_SCHEMA = "nucpred.mayr-site-n-config.v1"
DATASET_SCHEMA = "nucpred.mayr-site-n-dataset.v1"
SITE_SCHEMA = "nucpred.mayr-site-object.v1"
CANDIDATE_SCHEMA = "nucpred.mayr-site-candidate.v1"
EXPECTED_PARENT_ROWS = 1136

SITE_TYPES = (
    "atom",
    "bond",
    "delocalized_region",
    "atom_group",
    "transferable_h_group",
)
UNRESOLVED_SITE_TYPE = "unresolved_candidate_set"

CONTEXT_ID_COLUMNS = (
    "model_canonical_smiles",
    "xtb_alpb_solvent",
    "model_formal_charge",
)
REQUIRED_PARENT_COLUMNS = (
    "source_id",
    "model_canonical_smiles",
    "model_all_atom_count",
    "model_formal_charge",
    "model_graph_sha256",
    "model_atomic_numbers_json",
    "model_node_categorical_json",
    "model_directed_edges_json",
    "model_edge_categorical_json",
    "model_hydrogen_parent_index_json",
    "xtb_alpb_solvent",
    "solvent_raw",
    "N",
    "supervision_level",
    "site_target_mask_model",
    "site_target_atoms_model_json",
    "equivalent_h_groups_json",
    "complete_xtb10",
    "node_local4_json",
    "node_local4_available_json",
    "molecule_global6_json",
    "molecule_global6_available_json",
)


class SiteNDatasetError(RuntimeError):
    """Raised when the site-object dataset contract cannot be satisfied."""


@dataclass(frozen=True, slots=True)
class SiteNTables:
    species: pd.DataFrame
    contexts: pd.DataFrame
    sites: pd.DataFrame
    measurements: pd.DataFrame
    targets: pd.DataFrame
    candidates: pd.DataFrame
    multi_site_pairs: pd.DataFrame
    aggregation_audit: pd.DataFrame
    context_feature_audit: pd.DataFrame
    candidate_coverage: pd.DataFrame
    split_membership: pd.DataFrame
    split_manifest: dict[str, object]


def _stable_digest(namespace: str, *parts: object) -> str:
    payload = "\0".join((namespace, *(str(part) for part in parts)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _stable_id(prefix: str, namespace: str, *parts: object) -> str:
    return f"{prefix}:{_stable_digest(namespace, *parts)[:24]}"


def _json_compact(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=isinstance(value, dict),
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _as_list(value: object) -> list[object]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    text = str(value).strip()
    if not text:
        return []
    parsed = json.loads(text)
    if not isinstance(parsed, list):
        raise SiteNDatasetError("Expected a JSON list")
    return parsed


def _int_list(value: object) -> list[int]:
    return [int(item) for item in _as_list(value)]


def _normalise_solvent(value: object) -> str:
    solvent = str(value).strip().lower()
    if not solvent:
        raise SiteNDatasetError("A solvent context cannot be blank")
    return solvent


def _read_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = Path(path)
    payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != CONFIG_SCHEMA:
        raise SiteNDatasetError("Unsupported Mayr site-N config schema")
    if int(payload.get("expected_source_measurements", 0)) != EXPECTED_PARENT_ROWS:
        raise SiteNDatasetError("The frozen source-measurement count changed")
    if payload["identity"].get("same_context_same_site_target") != "arithmetic_mean":
        raise SiteNDatasetError("Repeated context-site targets must use arithmetic mean")
    if bool(payload["sites"].get("unmeasured_candidates_are_negative", True)):
        raise SiteNDatasetError("Unmeasured candidates cannot be negative labels")
    if bool(payload["sites"].get("site_probability_normalization", True)):
        raise SiteNDatasetError("The site-N contract forbids site softmax normalization")
    return payload


def _connectivity_identity(smiles: str) -> tuple[str, str, str]:
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise SiteNDatasetError(f"Cannot parse structure identity: {smiles!r}")
    key = str(inchi.MolToInchiKey(molecule))
    method = "standard_inchi_key"
    if not key:
        editable = Chem.RWMol(molecule)
        changed = False
        for bond in editable.GetBonds():
            if bond.GetBondType() == Chem.BondType.DATIVE:
                bond.SetBondType(Chem.BondType.SINGLE)
                changed = True
        if not changed:
            raise SiteNDatasetError(
                f"Standard InChIKey generation failed: {smiles!r}"
            )
        key = str(inchi.MolToInchiKey(editable.GetMol()))
        method = "dative_to_single_identity_copy"
    if not key or len(key.split("-")[0]) != 14:
        raise SiteNDatasetError(f"Invalid connectivity InChIKey: {smiles!r}")
    return key.split("-")[0], key, method


def _model_molecule(row: Mapping[str, object]) -> Chem.Mol:
    base = Chem.MolFromSmiles(str(row["model_canonical_smiles"]))
    if base is None:
        raise SiteNDatasetError(
            f"Cannot parse model graph for {row.get('source_id', 'UNKNOWN')}"
        )
    molecule = Chem.AddHs(base)
    expected_count = int(row["model_all_atom_count"])
    if molecule.GetNumAtoms() != expected_count:
        raise SiteNDatasetError(
            f"{row.get('source_id')}: rebuilt all-atom count changed"
        )
    observed_numbers = [atom.GetAtomicNum() for atom in molecule.GetAtoms()]
    declared_numbers = _int_list(row["model_atomic_numbers_json"])
    if observed_numbers != declared_numbers:
        raise SiteNDatasetError(
            f"{row.get('source_id')}: rebuilt all-atom order changed"
        )
    return molecule


def _species_context_ids(row: Mapping[str, object]) -> tuple[str, str]:
    canonical = str(row["model_canonical_smiles"]).strip()
    charge = int(row["model_formal_charge"])
    solvent = _normalise_solvent(row["xtb_alpb_solvent"])
    species_id = _stable_id(
        "species",
        "mayr-site-n-species-v1",
        canonical,
        charge,
    )
    context_id = _stable_id(
        "context",
        "mayr-site-n-context-v1",
        species_id,
        solvent,
    )
    return species_id, context_id


def _site_type_for_row(
    row: Mapping[str, object],
    molecule: Chem.Mol,
    members: Sequence[int],
) -> tuple[str, str, bool]:
    if not bool(row["site_target_mask_model"]):
        return UNRESOLVED_SITE_TYPE, "unresolved", False
    supervision = str(row["supervision_level"])
    if supervision == "exact_heavy_atom":
        if len(members) != 1:
            raise SiteNDatasetError("Exact-atom supervision changed cardinality")
        return "atom", "exact", True
    if supervision == "equivalent_or_indistinguishable_h_group":
        if not members or any(
            molecule.GetAtomWithIdx(index).GetAtomicNum() != 1 for index in members
        ):
            raise SiteNDatasetError("Transferable-H supervision contains non-H atoms")
        return "transferable_h_group", "equivalent_or_indistinguishable", True
    if supervision == "heavy_atom_group":
        if len(members) < 2:
            raise SiteNDatasetError("Heavy-atom group supervision collapsed")
        return "atom_group", "equivalent_or_collective", True
    if supervision == "heavy_bond_or_region":
        if len(members) < 2:
            raise SiteNDatasetError("Bond/region supervision collapsed")
        if (
            len(members) == 2
            and molecule.GetBondBetweenAtoms(int(members[0]), int(members[1]))
            is not None
        ):
            return "bond", "exact", True
        return "delocalized_region", "collective", True
    if supervision == "masked":
        return UNRESOLVED_SITE_TYPE, "unresolved", False
    raise SiteNDatasetError(f"Unsupported supervision level: {supervision!r}")


def _member_bonds(molecule: Chem.Mol, members: Sequence[int]) -> list[list[int]]:
    selected = set(int(index) for index in members)
    bonds: list[list[int]] = []
    for bond in molecule.GetBonds():
        left = bond.GetBeginAtomIdx()
        right = bond.GetEndAtomIdx()
        if left in selected and right in selected:
            bonds.append(sorted((left, right)))
    return sorted(bonds)


def _site_record(
    row: Mapping[str, object],
    *,
    species_id: str,
    molecule: Chem.Mol,
) -> dict[str, object] | None:
    members = sorted(set(_int_list(row["site_target_atoms_model_json"])))
    site_type, resolution, eligible = _site_type_for_row(row, molecule, members)
    if not eligible:
        return None
    if any(index < 0 or index >= molecule.GetNumAtoms() for index in members):
        raise SiteNDatasetError(f"{row['source_id']}: site atom index is invalid")
    site_id = _stable_id(
        "site",
        "mayr-site-n-object-v1",
        species_id,
        site_type,
        _json_compact(members),
    )
    return {
        "schema_version": SITE_SCHEMA,
        "site_object_id": site_id,
        "species_id": species_id,
        "site_type": site_type,
        "physical_scope": site_type,
        "assignment_resolution": resolution,
        "member_atom_indices_json": _json_compact(members),
        "member_bond_pairs_json": _json_compact(
            _member_bonds(molecule, members)
        ),
        "member_atom_count": len(members),
        "source_supervision_level": str(row["supervision_level"]),
        "formal_supervision_eligible": True,
    }


def _candidate_record(
    *,
    species_id: str,
    site_type: str,
    members: Sequence[int],
    molecule: Chem.Mol,
    origins: Iterable[str],
) -> dict[str, object]:
    ordered = sorted(set(int(index) for index in members))
    return {
        "schema_version": CANDIDATE_SCHEMA,
        "candidate_site_id": _stable_id(
            "site",
            "mayr-site-n-object-v1",
            species_id,
            site_type,
            _json_compact(ordered),
        ),
        "species_id": species_id,
        "site_type": site_type,
        "member_atom_indices_json": _json_compact(ordered),
        "member_bond_pairs_json": _json_compact(
            _member_bonds(molecule, ordered)
        ),
        "member_atom_count": len(ordered),
        "member_atomic_numbers_json": _json_compact(
            [molecule.GetAtomWithIdx(index).GetAtomicNum() for index in ordered]
        ),
        "candidate_origins_json": _json_compact(sorted(set(origins))),
        "label_independent": True,
    }


def enumerate_candidate_sites(
    row: Mapping[str, object],
    *,
    species_id: str,
    maximum_shortest_path_atoms: int = 8,
    maximum_group_combinations: int = 6,
) -> list[dict[str, object]]:
    """Enumerate structural candidates without reading target/site columns."""

    molecule = _model_molecule(row)
    candidates: dict[tuple[str, tuple[int, ...]], set[str]] = defaultdict(set)

    def add(site_type: str, members: Iterable[int], origin: str) -> None:
        ordered = tuple(sorted(set(int(index) for index in members)))
        if not ordered:
            return
        if site_type in {"bond", "atom_group", "delocalized_region"} and len(
            ordered
        ) < 2:
            return
        candidates[(site_type, ordered)].add(origin)

    all_indices = list(range(molecule.GetNumAtoms()))
    heavy = [
        index
        for index in all_indices
        if molecule.GetAtomWithIdx(index).GetAtomicNum() != 1
    ]
    for index in all_indices:
        add("atom", [index], "all_atoms")
    for bond in molecule.GetBonds():
        add(
            "bond",
            [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()],
            "all_bonds",
        )

    hydrogen_parent = _int_list(row["model_hydrogen_parent_index_json"])
    if len(hydrogen_parent) != molecule.GetNumAtoms():
        raise SiteNDatasetError("Hydrogen-parent vector length changed")
    hydrogens_by_parent: dict[int, list[int]] = defaultdict(list)
    for index, parent in enumerate(hydrogen_parent):
        if parent >= 0:
            if molecule.GetAtomWithIdx(index).GetAtomicNum() != 1:
                raise SiteNDatasetError("Hydrogen-parent vector marks a non-H atom")
            hydrogens_by_parent[int(parent)].append(index)
    for group in hydrogens_by_parent.values():
        add("transferable_h_group", group, "same_parent_explicit_h")

    symmetry = list(
        Chem.CanonicalRankAtoms(
            molecule,
            breakTies=False,
            includeChirality=True,
        )
    )
    by_rank: dict[int, list[int]] = defaultdict(list)
    for index in heavy:
        by_rank[int(symmetry[index])].append(index)
    for group in by_rank.values():
        if len(group) >= 2:
            add("atom_group", group, "rdkit_symmetry")
            equivalent_hydrogens = [
                hydrogen
                for parent in group
                for hydrogen in hydrogens_by_parent.get(parent, [])
            ]
            if equivalent_hydrogens:
                add(
                    "transferable_h_group",
                    equivalent_hydrogens,
                    "symmetry_equivalent_h_parents",
                )

    by_element_all: dict[int, list[int]] = defaultdict(list)
    for index in heavy:
        by_element_all[molecule.GetAtomWithIdx(index).GetAtomicNum()].append(index)
    for group in by_element_all.values():
        if len(group) < 2:
            continue
        add("atom_group", group, "whole_graph_same_element")
        if len(group) <= maximum_group_combinations:
            for size in range(2, min(3, len(group)) + 1):
                for subset in itertools.combinations(group, size):
                    add(
                        "atom_group",
                        subset,
                        "whole_graph_same_element_subset",
                    )

    ring_sets = [tuple(sorted(set(ring))) for ring in molecule.GetRingInfo().AtomRings()]
    ordered_rings = [
        tuple(int(index) for index in ring)
        for ring in molecule.GetRingInfo().AtomRings()
    ]
    for ring, ordered_ring in zip(ring_sets, ordered_rings, strict=True):
        add("delocalized_region", ring, "ring")
        _add_element_filtered_regions(
            add,
            molecule,
            ring,
            origin="ring_element_filtered",
        )
        if len(ordered_ring) >= 4:
            doubled = ordered_ring + ordered_ring
            for size in range(3, len(ordered_ring)):
                for start in range(len(ordered_ring)):
                    add(
                        "delocalized_region",
                        doubled[start : start + size],
                        "cyclic_ring_path",
                    )
        _add_same_element_groups(
            add,
            molecule,
            ring,
            origin="ring_same_element",
            combination_limit=maximum_group_combinations,
        )
    fused_ring_components = _overlap_components(ring_sets)
    for component in fused_ring_components:
        if len(component) >= 2:
            add("delocalized_region", component, "fused_ring_system")
            _add_element_filtered_regions(
                add,
                molecule,
                component,
                origin="fused_ring_element_filtered",
            )

    conjugated_edges: list[tuple[int, int]] = []
    for bond in molecule.GetBonds():
        if bond.GetIsAromatic() or bond.GetIsConjugated() or bond.GetBondTypeAsDouble() > 1:
            conjugated_edges.append(
                (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            )
    for component in _edge_components(conjugated_edges):
        if len(component) >= 2:
            add("delocalized_region", component, "conjugated_component")
            _add_element_filtered_regions(
                add,
                molecule,
                component,
                origin="conjugated_element_filtered",
            )
            _add_same_element_groups(
                add,
                molecule,
                component,
                origin="conjugated_same_element",
                combination_limit=maximum_group_combinations,
            )

    for center in heavy:
        neighbours = [
            atom.GetIdx()
            for atom in molecule.GetAtomWithIdx(center).GetNeighbors()
            if atom.GetAtomicNum() != 1
        ]
        if neighbours:
            add(
                "delocalized_region",
                [center, *neighbours],
                "heavy_radius_one",
            )
            _add_element_filtered_regions(
                add,
                molecule,
                [center, *neighbours],
                origin="heavy_radius_one_element_filtered",
            )
        by_element: dict[int, list[int]] = defaultdict(list)
        for index in neighbours:
            by_element[molecule.GetAtomWithIdx(index).GetAtomicNum()].append(index)
        for group in by_element.values():
            if len(group) >= 2:
                add("atom_group", group, "same_element_neighbours")
                add(
                    "delocalized_region",
                    group,
                    "same_element_neighbour_region",
                )

    reactive_carbons: list[int] = []
    non_aromatic_reactive_carbons: list[int] = []
    non_ring_reactive_carbons: list[int] = []
    aromatic_carbons: list[int] = []
    for index in heavy:
        atom = molecule.GetAtomWithIdx(index)
        if atom.GetAtomicNum() != 6:
            continue
        reactive = atom.GetFormalCharge() != 0 or any(
            bond.GetIsAromatic() or bond.GetBondTypeAsDouble() > 1
            for bond in atom.GetBonds()
        )
        if not reactive:
            continue
        reactive_carbons.append(index)
        if not atom.GetIsAromatic():
            non_aromatic_reactive_carbons.append(index)
        if not atom.IsInRing():
            non_ring_reactive_carbons.append(index)
        if atom.GetIsAromatic():
            aromatic_carbons.append(index)
    for origin, group in (
        ("all_reactive_carbons", reactive_carbons),
        ("non_aromatic_reactive_carbons", non_aromatic_reactive_carbons),
        ("non_ring_reactive_carbons", non_ring_reactive_carbons),
        ("aromatic_carbons", aromatic_carbons),
    ):
        add("delocalized_region", group, origin)

    carbon_edges = [
        (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        for bond in molecule.GetBonds()
        if molecule.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetAtomicNum() == 6
        and molecule.GetAtomWithIdx(bond.GetEndAtomIdx()).GetAtomicNum() == 6
    ]
    for component in _edge_components(carbon_edges):
        add(
            "delocalized_region",
            component,
            "carbon_component_after_heteroatom_cut",
        )

    for left_position, left in enumerate(heavy):
        for right in heavy[left_position + 1 :]:
            path = tuple(Chem.GetShortestPath(molecule, int(left), int(right)))
            if 3 <= len(path) <= maximum_shortest_path_atoms:
                add("delocalized_region", path, "heavy_shortest_path")

    return [
        _candidate_record(
            species_id=species_id,
            site_type=site_type,
            members=members,
            molecule=molecule,
            origins=origins,
        )
        for (site_type, members), origins in sorted(candidates.items())
    ]


def _add_same_element_groups(
    add: Any,
    molecule: Chem.Mol,
    members: Iterable[int],
    *,
    origin: str,
    combination_limit: int,
) -> None:
    by_element: dict[int, list[int]] = defaultdict(list)
    for index in sorted(set(int(value) for value in members)):
        if molecule.GetAtomWithIdx(index).GetAtomicNum() == 1:
            continue
        by_element[molecule.GetAtomWithIdx(index).GetAtomicNum()].append(index)
    for group in by_element.values():
        if len(group) < 2:
            continue
        add("atom_group", group, origin)
        if len(group) <= combination_limit:
            for size in range(2, min(3, len(group)) + 1):
                for subset in itertools.combinations(group, size):
                    add("atom_group", subset, f"{origin}_subset")


def _add_element_filtered_regions(
    add: Any,
    molecule: Chem.Mol,
    members: Iterable[int],
    *,
    origin: str,
) -> None:
    by_element: dict[int, list[int]] = defaultdict(list)
    for index in sorted(set(int(value) for value in members)):
        if molecule.GetAtomWithIdx(index).GetAtomicNum() == 1:
            continue
        by_element[molecule.GetAtomWithIdx(index).GetAtomicNum()].append(index)
    for group in by_element.values():
        if len(group) >= 2:
            add("delocalized_region", group, origin)


def _edge_components(edges: Iterable[tuple[int, int]]) -> list[list[int]]:
    adjacency: dict[int, set[int]] = defaultdict(set)
    for left, right in edges:
        adjacency[int(left)].add(int(right))
        adjacency[int(right)].add(int(left))
    components: list[list[int]] = []
    remaining = set(adjacency)
    while remaining:
        root = min(remaining)
        stack = [root]
        component: set[int] = set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            stack.extend(adjacency[current] - component)
        remaining -= component
        components.append(sorted(component))
    return components


def _overlap_components(groups: Iterable[Sequence[int]]) -> list[list[int]]:
    pending = [set(int(index) for index in group) for group in groups]
    components: list[list[int]] = []
    while pending:
        merged = pending.pop(0)
        changed = True
        while changed:
            changed = False
            keep: list[set[int]] = []
            for other in pending:
                if merged & other:
                    merged |= other
                    changed = True
                else:
                    keep.append(other)
            pending = keep
        components.append(sorted(merged))
    return components


def build_site_n_tables(
    parent: pd.DataFrame,
    *,
    split_seeds: Sequence[int],
    test_fraction: float,
    validation_fraction_of_development: float,
    maximum_shortest_path_atoms: int = 8,
    maximum_group_combinations: int = 6,
) -> SiteNTables:
    """Project one frozen measurement table into normalized site-N tables."""

    missing = sorted(set(REQUIRED_PARENT_COLUMNS) - set(parent.columns))
    if missing:
        raise SiteNDatasetError(f"Parent records are missing columns: {missing}")
    if parent["source_id"].astype(str).duplicated().any():
        raise SiteNDatasetError("Parent source_id values must be unique")
    if not 0 < float(test_fraction) < 1:
        raise SiteNDatasetError("test_fraction must be between zero and one")
    if not 0 < float(validation_fraction_of_development) < 1:
        raise SiteNDatasetError(
            "validation_fraction_of_development must be between zero and one"
        )

    records = parent.copy().reset_index(drop=True)
    species_ids: list[str] = []
    context_ids: list[str] = []
    connectivity_ids: list[str] = []
    connectivity_keys: list[str] = []
    identity_methods: list[str] = []
    for row in records.to_dict("records"):
        species_id, context_id = _species_context_ids(row)
        connectivity_id, full_key, method = _connectivity_identity(
            str(row["model_canonical_smiles"])
        )
        species_ids.append(species_id)
        context_ids.append(context_id)
        connectivity_ids.append(connectivity_id)
        connectivity_keys.append(full_key)
        identity_methods.append(method)
    records["_species_id"] = species_ids
    records["_context_id"] = context_ids
    records["_connectivity_id"] = connectivity_ids
    records["_connectivity_inchi_key"] = connectivity_keys
    records["_connectivity_method"] = identity_methods

    contexts, context_audit = _build_contexts(records)
    species = _build_species(contexts)
    molecule_by_species = {
        str(row["species_id"]): _model_molecule(row)
        for row in species.to_dict("records")
    }

    measurement_rows: list[dict[str, object]] = []
    site_rows: dict[str, dict[str, object]] = {}
    for row in records.to_dict("records"):
        species_id = str(row["_species_id"])
        context_id = str(row["_context_id"])
        site = _site_record(
            row,
            species_id=species_id,
            molecule=molecule_by_species[species_id],
        )
        site_id = "" if site is None else str(site["site_object_id"])
        if site is not None:
            previous = site_rows.get(site_id)
            if previous is not None and {
                key: previous[key]
                for key in (
                    "species_id",
                    "site_type",
                    "member_atom_indices_json",
                    "member_bond_pairs_json",
                )
            } != {
                key: site[key]
                for key in (
                    "species_id",
                    "site_type",
                    "member_atom_indices_json",
                    "member_bond_pairs_json",
                )
            }:
                raise SiteNDatasetError(f"Conflicting site object identity: {site_id}")
            site_rows[site_id] = site
        copied = dict(row)
        copied.pop("_species_id", None)
        copied.pop("_context_id", None)
        copied.pop("_connectivity_id", None)
        copied.pop("_connectivity_inchi_key", None)
        copied.pop("_connectivity_method", None)
        copied.update(
            {
                "measurement_id": str(row["source_id"]),
                "species_id": species_id,
                "context_id": context_id,
                "connectivity_id": str(row["_connectivity_id"]),
                "site_object_id": site_id,
                "measurement_training_eligible": site is not None,
                "measurement_site_type": (
                    UNRESOLVED_SITE_TYPE if site is None else site["site_type"]
                ),
                "aggregation_policy": (
                    "arithmetic_mean_within_context_and_site"
                ),
            }
        )
        measurement_rows.append(copied)
    measurements = pd.DataFrame(measurement_rows)

    sites = pd.DataFrame(site_rows.values())
    if sites.empty:
        raise SiteNDatasetError("No formal site objects were produced")
    site_sources = (
        measurements.loc[measurements["site_object_id"].ne("")]
        .groupby("site_object_id", sort=False)["source_id"]
        .agg(lambda values: _json_compact(sorted(map(str, values))))
    )
    sites["measurement_source_ids_json"] = sites["site_object_id"].map(site_sources)
    sites = sites.sort_values(["species_id", "site_object_id"]).reset_index(drop=True)

    targets, aggregation_audit = _aggregate_targets(
        measurements,
        sites=sites,
        contexts=contexts,
    )
    candidates = _build_candidates(
        species,
        maximum_shortest_path_atoms=maximum_shortest_path_atoms,
        maximum_group_combinations=maximum_group_combinations,
    )
    candidate_coverage = _candidate_coverage(sites, candidates)
    multi_site_pairs = _build_multi_site_pairs(targets)
    split_membership, split_manifest = _build_split_membership(
        targets,
        seeds=tuple(int(seed) for seed in split_seeds),
        test_fraction=float(test_fraction),
        validation_fraction=float(validation_fraction_of_development),
    )
    return SiteNTables(
        species=species,
        contexts=contexts,
        sites=sites,
        measurements=measurements,
        targets=targets,
        candidates=candidates,
        multi_site_pairs=multi_site_pairs,
        aggregation_audit=aggregation_audit,
        context_feature_audit=context_audit,
        candidate_coverage=candidate_coverage,
        split_membership=split_membership,
        split_manifest=split_manifest,
    )


def _build_contexts(records: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    context_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    feature_columns = [
        column
        for column in records.columns
        if (
            column.startswith("model_")
            or column.startswith("g1_")
            or column.startswith("node_local4")
            or column.startswith("molecule_global6")
            or column.startswith("solvent_")
            or column
            in {
                "xtb_alpb_solvent",
                "complete_xtb10",
                "electronic_cache_status",
                "species_state",
                "formal_charge",
                "equivalent_h_groups_json",
            }
        )
        and not column.startswith("_")
    ]
    for context_id, group in records.groupby("_context_id", sort=True):
        graph_hash_count = int(group["model_graph_sha256"].nunique(dropna=False))
        if graph_hash_count != 1:
            raise SiteNDatasetError(
                f"Context {context_id} contains multiple model graphs"
            )
        ranked = group.assign(
            _complete_rank=~group["complete_xtb10"].astype(bool),
            _source_rank=group["source_id"].map(
                lambda value: _stable_digest(
                    "mayr-site-n-context-representative-v1",
                    value,
                )
            ),
        ).sort_values(["_complete_rank", "_source_rank", "source_id"])
        representative = ranked.iloc[0]
        context_row = {
            column: representative[column] for column in feature_columns
        }
        context_row.update(
            {
                "context_id": str(context_id),
                "species_id": str(representative["_species_id"]),
                "connectivity_id": str(representative["_connectivity_id"]),
                "connectivity_inchi_key": str(
                    representative["_connectivity_inchi_key"]
                ),
                "connectivity_identity_method": str(
                    representative["_connectivity_method"]
                ),
                "representative_source_id": str(representative["source_id"]),
                "context_source_ids_json": _json_compact(
                    sorted(group["source_id"].astype(str).tolist())
                ),
                "context_measurement_count": int(len(group)),
                "representative_selection_target_independent": True,
                "representative_feature_policy": (
                    "prefer_complete_xtb10_then_sha256_source_id"
                ),
            }
        )
        context_rows.append(context_row)

        feature_hash_columns = {
            "graph": "model_graph_sha256",
            "geometry": "g1_xyz_sha256",
            "local4": "node_local4_json",
            "global6": "molecule_global6_json",
        }
        audit: dict[str, object] = {
            "context_id": str(context_id),
            "species_id": str(representative["_species_id"]),
            "source_ids_json": _json_compact(
                sorted(group["source_id"].astype(str).tolist())
            ),
            "measurement_count": int(len(group)),
            "representative_source_id": str(representative["source_id"]),
            "complete_xtb10_any": bool(group["complete_xtb10"].astype(bool).any()),
            "complete_xtb10_all": bool(group["complete_xtb10"].astype(bool).all()),
            "status": "pass",
        }
        for label, column in feature_hash_columns.items():
            if column not in group.columns:
                audit[f"{label}_variant_count"] = 0
                continue
            if column.endswith("_sha256"):
                variants = group[column].fillna("").astype(str)
            else:
                variants = group[column].map(
                    lambda value: _stable_digest(
                        f"mayr-site-n-{label}-value-v1",
                        value,
                    )
                )
            audit[f"{label}_variant_count"] = int(variants.nunique(dropna=False))
        audit_rows.append(audit)
    contexts = pd.DataFrame(context_rows).sort_values("context_id").reset_index(
        drop=True
    )
    audit_frame = pd.DataFrame(audit_rows).sort_values("context_id").reset_index(
        drop=True
    )
    return contexts, audit_frame


def _build_species(contexts: pd.DataFrame) -> pd.DataFrame:
    structural_columns = [
        column
        for column in contexts.columns
        if column.startswith("model_")
        and column
        not in {
            "model_node_local4_json",
            "model_molecule_global6_json",
        }
    ]
    rows: list[dict[str, object]] = []
    for species_id, group in contexts.groupby("species_id", sort=True):
        if int(group["model_graph_sha256"].nunique(dropna=False)) != 1:
            raise SiteNDatasetError(
                f"Species {species_id} contains inconsistent model graphs"
            )
        representative = group.sort_values("representative_source_id").iloc[0]
        row = {column: representative[column] for column in structural_columns}
        row.update(
            {
                "species_id": str(species_id),
                "connectivity_id": str(representative["connectivity_id"]),
                "connectivity_inchi_key": str(
                    representative["connectivity_inchi_key"]
                ),
                "connectivity_identity_method": str(
                    representative["connectivity_identity_method"]
                ),
                "representative_source_id": str(
                    representative["representative_source_id"]
                ),
                "context_ids_json": _json_compact(
                    sorted(group["context_id"].astype(str).tolist())
                ),
                "context_count": int(len(group)),
            }
        )
        if "equivalent_h_groups_json" in representative:
            row["equivalent_h_groups_json"] = representative[
                "equivalent_h_groups_json"
            ]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("species_id").reset_index(drop=True)


def _aggregate_targets(
    measurements: pd.DataFrame,
    *,
    sites: pd.DataFrame,
    contexts: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible = measurements.loc[
        measurements["measurement_training_eligible"].astype(bool)
        & pd.to_numeric(measurements["N"], errors="coerce").notna()
    ].copy()
    eligible["N"] = pd.to_numeric(eligible["N"], errors="raise").astype(float)
    site_index = sites.set_index("site_object_id", drop=False)
    context_index = contexts.set_index("context_id", drop=False)
    rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    for (context_id, site_id), group in eligible.groupby(
        ["context_id", "site_object_id"],
        sort=True,
    ):
        values = group["N"].to_numpy(dtype=float)
        site = site_index.loc[str(site_id)]
        context = context_index.loc[str(context_id)]
        source_ids = sorted(group["source_id"].astype(str).tolist())
        target_id = _stable_id(
            "target",
            "mayr-site-n-target-v1",
            context_id,
            site_id,
        )
        names = sorted(set(group.get("name", pd.Series("", index=group.index)).astype(str)))
        paper_keys = sorted(
            set(group.get("paper_key", pd.Series("", index=group.index)).astype(str))
        )
        removed = sorted(
            set(
                group.get(
                    "removed_fragment_smiles_json",
                    pd.Series("[]", index=group.index),
                ).astype(str)
            )
        )
        row = {
            "target_id": target_id,
            "context_id": str(context_id),
            "species_id": str(context["species_id"]),
            "connectivity_id": str(context["connectivity_id"]),
            "site_object_id": str(site_id),
            "site_type": str(site["site_type"]),
            "member_atom_indices_json": str(site["member_atom_indices_json"]),
            "member_bond_pairs_json": str(site["member_bond_pairs_json"]),
            "assignment_resolution": str(site["assignment_resolution"]),
            "solvent_raw": str(context["solvent_raw"]),
            "xtb_alpb_solvent": str(context["xtb_alpb_solvent"]),
            "model_canonical_smiles": str(context["model_canonical_smiles"]),
            "model_formal_charge": int(context["model_formal_charge"]),
            "N_mean": float(np.mean(values)),
            "N_std_population": float(np.std(values, ddof=0)),
            "N_min": float(np.min(values)),
            "N_max": float(np.max(values)),
            "N_range": float(np.max(values) - np.min(values)),
            "measurement_count": int(len(values)),
            "source_ids_json": _json_compact(source_ids),
            "N_values_json": _json_compact([float(value) for value in values]),
            "measurement_names_json": _json_compact(names),
            "paper_keys_json": _json_compact(paper_keys),
            "counterion_or_fragment_evidence_json": _json_compact(removed),
            "aggregation_policy": "arithmetic_mean",
            "formal_training_eligible": True,
        }
        rows.append(row)
        audit_rows.append(
            {
                "target_id": target_id,
                "context_id": str(context_id),
                "site_object_id": str(site_id),
                "measurement_count": int(len(values)),
                "source_ids_json": _json_compact(source_ids),
                "N_values_json": _json_compact(
                    [float(value) for value in values]
                ),
                "N_mean": float(np.mean(values)),
                "N_std_population": float(np.std(values, ddof=0)),
                "N_range": float(np.max(values) - np.min(values)),
                "is_aggregated_collision": bool(len(values) > 1),
                "status": "pass",
            }
        )
    targets = pd.DataFrame(rows).sort_values("target_id").reset_index(drop=True)
    audit = pd.DataFrame(audit_rows).sort_values("target_id").reset_index(drop=True)
    if targets["target_id"].duplicated().any():
        raise SiteNDatasetError("Aggregated target IDs must be unique")
    return targets, audit


def _build_candidates(
    species: pd.DataFrame,
    *,
    maximum_shortest_path_atoms: int,
    maximum_group_combinations: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in species.to_dict("records"):
        rows.extend(
            enumerate_candidate_sites(
                row,
                species_id=str(row["species_id"]),
                maximum_shortest_path_atoms=maximum_shortest_path_atoms,
                maximum_group_combinations=maximum_group_combinations,
            )
        )
    candidates = pd.DataFrame(rows)
    if candidates.empty or candidates["candidate_site_id"].duplicated().any():
        raise SiteNDatasetError("Candidate-site identities must be non-empty and unique")
    return candidates.sort_values(
        ["species_id", "site_type", "candidate_site_id"]
    ).reset_index(drop=True)


def _candidate_coverage(
    sites: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    candidate_ids = set(candidates["candidate_site_id"].astype(str))
    rows = []
    for row in sites.to_dict("records"):
        site_id = str(row["site_object_id"])
        rows.append(
            {
                "site_object_id": site_id,
                "species_id": str(row["species_id"]),
                "site_type": str(row["site_type"]),
                "member_atom_indices_json": str(row["member_atom_indices_json"]),
                "covered": site_id in candidate_ids,
                "coverage_status": "covered" if site_id in candidate_ids else "missing",
            }
        )
    return pd.DataFrame(rows).sort_values("site_object_id").reset_index(drop=True)


def _build_multi_site_pairs(targets: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for context_id, group in targets.groupby("context_id", sort=True):
        ordered = group.sort_values("target_id").reset_index(drop=True)
        if len(ordered) < 2:
            continue
        for left_index, right_index in itertools.combinations(range(len(ordered)), 2):
            left = ordered.iloc[left_index]
            right = ordered.iloc[right_index]
            rows.append(
                {
                    "pair_id": _stable_id(
                        "pair",
                        "mayr-site-n-pair-v1",
                        left["target_id"],
                        right["target_id"],
                    ),
                    "context_id": str(context_id),
                    "species_id": str(left["species_id"]),
                    "connectivity_id": str(left["connectivity_id"]),
                    "left_target_id": str(left["target_id"]),
                    "right_target_id": str(right["target_id"]),
                    "left_site_object_id": str(left["site_object_id"]),
                    "right_site_object_id": str(right["site_object_id"]),
                    "left_site_type": str(left["site_type"]),
                    "right_site_type": str(right["site_type"]),
                    "left_N_mean": float(left["N_mean"]),
                    "right_N_mean": float(right["N_mean"]),
                    "delta_N_left_minus_right": float(
                        left["N_mean"] - right["N_mean"]
                    ),
                    "ordering_sign": int(
                        np.sign(float(left["N_mean"] - right["N_mean"]))
                    ),
                }
            )
    columns = (
        "pair_id",
        "context_id",
        "species_id",
        "connectivity_id",
        "left_target_id",
        "right_target_id",
        "left_site_object_id",
        "right_site_object_id",
        "left_site_type",
        "right_site_type",
        "left_N_mean",
        "right_N_mean",
        "delta_N_left_minus_right",
        "ordering_sign",
    )
    return pd.DataFrame(rows, columns=columns)


def _build_split_membership(
    targets: pd.DataFrame,
    *,
    seeds: Sequence[int],
    test_fraction: float,
    validation_fraction: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    groups = targets["connectivity_id"].astype(str).to_numpy()
    indices = np.arange(len(targets))
    membership_rows: list[dict[str, object]] = []
    runs: list[dict[str, object]] = []
    for seed in seeds:
        outer = GroupShuffleSplit(
            n_splits=1,
            test_size=test_fraction,
            random_state=int(seed),
        )
        development_index, test_index = next(
            outer.split(indices, groups=groups)
        )
        inner_groups = groups[development_index]
        inner = GroupShuffleSplit(
            n_splits=1,
            test_size=validation_fraction,
            random_state=int(seed) + 1_000_003,
        )
        fit_relative, validation_relative = next(
            inner.split(development_index, groups=inner_groups)
        )
        train_index = development_index[fit_relative]
        validation_index = development_index[validation_relative]
        role_indices = {
            "train": train_index,
            "validation": validation_index,
            "test": test_index,
        }
        role_groups: dict[str, set[str]] = {}
        for role, selected in role_indices.items():
            role_groups[role] = set(groups[selected])
            for index in selected:
                row = targets.iloc[int(index)]
                membership_rows.append(
                    {
                        "split_seed": int(seed),
                        "role": role,
                        "target_id": str(row["target_id"]),
                        "context_id": str(row["context_id"]),
                        "species_id": str(row["species_id"]),
                        "connectivity_id": str(row["connectivity_id"]),
                    }
                )
        overlaps = {
            "train_validation": len(
                role_groups["train"] & role_groups["validation"]
            ),
            "train_test": len(role_groups["train"] & role_groups["test"]),
            "validation_test": len(
                role_groups["validation"] & role_groups["test"]
            ),
        }
        if any(overlaps.values()):
            raise SiteNDatasetError(f"Connectivity split leaked for seed {seed}")
        runs.append(
            {
                "seed": int(seed),
                "roles": {
                    role: {
                        "target_count": int(len(selected)),
                        "context_count": int(
                            targets.iloc[selected]["context_id"].nunique()
                        ),
                        "connectivity_count": int(len(role_groups[role])),
                        "target_id_sha256": _stable_digest(
                            "mayr-site-n-split-targets-v1",
                            *sorted(
                                targets.iloc[selected]["target_id"].astype(str)
                            ),
                        ),
                    }
                    for role, selected in role_indices.items()
                },
                "connectivity_overlap": overlaps,
                "status": "pass",
            }
        )
    membership = pd.DataFrame(membership_rows).sort_values(
        ["split_seed", "role", "target_id"]
    ).reset_index(drop=True)
    manifest: dict[str, object] = {
        "schema_version": "nucpred.mayr-site-n-splits.v1",
        "group_identity": "standard_inchi_key_connectivity_block",
        "test_fraction": float(test_fraction),
        "validation_fraction_of_development": float(validation_fraction),
        "seeds": [int(seed) for seed in seeds],
        "runs": runs,
        "test_is_never_used_for_selection": True,
    }
    return membership, manifest


def _pretraining_overlap_audit(
    *,
    config: Mapping[str, object],
    species: pd.DataFrame,
) -> pd.DataFrame:
    target_by_connectivity: dict[str, list[str]] = (
        species.groupby("connectivity_id", sort=True)["species_id"]
        .agg(lambda values: sorted(map(str, values)))
        .to_dict()
    )
    rows: list[dict[str, object]] = []
    section = dict(config["pretraining_overlap"])
    for scope in ("pilot", "full"):
        dataset_id = str(section[f"{scope}_dataset_id"])
        path = (ROOT / str(section[f"{scope}_records_path"])).resolve()
        expected_hash = str(section[f"{scope}_records_sha256"])
        observed_hash = sha256_file(path)
        if observed_hash != expected_hash:
            raise SiteNDatasetError(
                f"Pretraining {scope} records hash changed: {path}"
            )
        records = pd.read_parquet(path, columns=["source_id", "canonical_smiles"])
        identities: dict[str, tuple[str, str]] = {}
        overlap_source_ids: list[str] = []
        overlap_connectivity: set[str] = set()
        overlap_details: list[dict[str, object]] = []
        for record in records.itertuples(index=False):
            canonical = str(record.canonical_smiles)
            identity = identities.get(canonical)
            if identity is None:
                connectivity_id, full_key, _ = _connectivity_identity(canonical)
                identity = (connectivity_id, full_key)
                identities[canonical] = identity
            connectivity_id, full_key = identity
            if connectivity_id not in target_by_connectivity:
                continue
            source_id = str(record.source_id)
            overlap_source_ids.append(source_id)
            overlap_connectivity.add(connectivity_id)
            overlap_details.append(
                {
                    "pretraining_source_id": source_id,
                    "connectivity_id": connectivity_id,
                    "pretraining_inchi_key": full_key,
                    "target_species_ids": target_by_connectivity[connectivity_id],
                }
            )
        required = int(section.get("required_connectivity_overlap", 0))
        observed = len(overlap_source_ids)
        rows.append(
            {
                "dataset_scope": scope,
                "dataset_id": dataset_id,
                "records_path": str(path.relative_to(ROOT)),
                "records_sha256": observed_hash,
                "record_count": int(len(records)),
                "unique_canonical_smiles": int(len(identities)),
                "overlap_record_count": int(observed),
                "overlap_connectivity_count": int(len(overlap_connectivity)),
                "overlap_source_ids_json": _json_compact(
                    sorted(overlap_source_ids)
                ),
                "overlap_details_json": _json_compact(overlap_details),
                "required_overlap_record_count": required,
                "status": "pass" if observed == required else "fail",
            }
        )
        if observed != required:
            raise SiteNDatasetError(
                f"{dataset_id} has {observed} target connectivity overlaps; "
                f"expected {required}"
            )
    return pd.DataFrame(rows)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        mode="w",
        encoding="utf-8",
        newline="",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _file_entry(path: Path, *, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def _source_hashes(config_path: Path) -> dict[str, str]:
    source_path = Path(__file__)
    return {
        "config_sha256": sha256_file(config_path),
        "builder_sha256": sha256_file(source_path),
    }


def build_dataset(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    output_directory: str | Path | None = None,
) -> dict[str, Path]:
    config_file = Path(config_path).resolve()
    config = _read_config(config_file)
    parent_section = dict(config["parent"])
    parent_path = (ROOT / str(parent_section["records_path"])).resolve()
    parent_manifest = (ROOT / str(parent_section["manifest_path"])).resolve()
    if sha256_file(parent_path) != str(parent_section["records_sha256"]):
        raise SiteNDatasetError("Frozen parent records hash changed")
    if sha256_file(parent_manifest) != str(parent_section["manifest_sha256"]):
        raise SiteNDatasetError("Frozen parent manifest hash changed")
    parent = pd.read_parquet(parent_path)
    expected_rows = int(config["expected_source_measurements"])
    if len(parent) != expected_rows:
        raise SiteNDatasetError(
            f"Expected {expected_rows} parent measurements, found {len(parent)}"
        )

    candidate_config = dict(config["candidates"])
    tables = build_site_n_tables(
        parent,
        split_seeds=tuple(int(value) for value in config["split_seeds"]),
        test_fraction=float(config["test_fraction"]),
        validation_fraction_of_development=float(
            config["validation_fraction_of_development"]
        ),
        maximum_shortest_path_atoms=int(
            candidate_config["maximum_shortest_path_atoms"]
        ),
        maximum_group_combinations=int(
            candidate_config["maximum_group_combinations"]
        ),
    )
    overlap_audit = _pretraining_overlap_audit(
        config=config,
        species=tables.species,
    )

    output = (
        Path(output_directory).resolve()
        if output_directory is not None
        else (ROOT / str(config["output_directory"])).resolve()
    )
    try:
        output.relative_to(ROOT)
    except ValueError as exc:
        raise SiteNDatasetError("Output directory must stay inside the project") from exc
    output.mkdir(parents=True, exist_ok=True)

    outputs = {
        "species": output / "species.parquet",
        "contexts": output / "contexts.parquet",
        "sites": output / "sites.parquet",
        "measurements": output / "measurements.parquet",
        "targets": output / "targets.parquet",
        "candidates": output / "candidate_sites.parquet",
        "multi_site_pairs": output / "multi_site_pairs.csv",
        "aggregation_audit": output / "aggregation_audit.csv",
        "context_feature_audit": output / "context_feature_audit.csv",
        "candidate_coverage": output / "candidate_coverage.csv",
        "pretraining_overlap_audit": output / "pretraining_overlap_audit.csv",
        "split_membership": output / "split_membership.csv",
        "split_manifest": output / "split_manifest.json",
        "summary": output / "summary.json",
        "dataset_manifest": output / "dataset_manifest.json",
    }
    _atomic_parquet(outputs["species"], tables.species)
    _atomic_parquet(outputs["contexts"], tables.contexts)
    _atomic_parquet(outputs["sites"], tables.sites)
    _atomic_parquet(outputs["measurements"], tables.measurements)
    _atomic_parquet(outputs["targets"], tables.targets)
    _atomic_parquet(outputs["candidates"], tables.candidates)
    _atomic_csv(outputs["multi_site_pairs"], tables.multi_site_pairs)
    _atomic_csv(outputs["aggregation_audit"], tables.aggregation_audit)
    _atomic_csv(outputs["context_feature_audit"], tables.context_feature_audit)
    _atomic_csv(outputs["candidate_coverage"], tables.candidate_coverage)
    _atomic_csv(outputs["pretraining_overlap_audit"], overlap_audit)
    _atomic_csv(outputs["split_membership"], tables.split_membership)
    atomic_write_json(outputs["split_manifest"], tables.split_manifest)

    coverage_by_type = (
        tables.candidate_coverage.groupby("site_type", sort=True)["covered"]
        .agg(["sum", "count"])
        .reset_index()
    )
    coverage_by_type["fraction"] = (
        coverage_by_type["sum"] / coverage_by_type["count"]
    )
    summary: dict[str, object] = {
        "schema_version": "nucpred.mayr-site-n-summary.v1",
        "dataset_id": str(config["dataset_id"]),
        "parent_dataset_id": str(parent_section["dataset_id"]),
        "source_measurement_count": int(len(tables.measurements)),
        "species_count": int(len(tables.species)),
        "context_count": int(len(tables.contexts)),
        "site_object_count": int(len(tables.sites)),
        "formal_target_count": int(len(tables.targets)),
        "formal_measurement_count": int(
            tables.measurements["measurement_training_eligible"].sum()
        ),
        "unresolved_measurement_count": int(
            (~tables.measurements["measurement_training_eligible"]).sum()
        ),
        "aggregated_collision_group_count": int(
            tables.aggregation_audit["is_aggregated_collision"].sum()
        ),
        "aggregated_collision_measurement_count": int(
            tables.aggregation_audit.loc[
                tables.aggregation_audit["is_aggregated_collision"],
                "measurement_count",
            ].sum()
        ),
        "duplicate_context_count": int(
            (tables.context_feature_audit["measurement_count"] > 1).sum()
        ),
        "context_geometry_variant_count": int(
            (tables.context_feature_audit["geometry_variant_count"] > 1).sum()
        ),
        "candidate_site_count": int(len(tables.candidates)),
        "candidate_covered_site_count": int(
            tables.candidate_coverage["covered"].sum()
        ),
        "candidate_site_coverage_fraction": float(
            tables.candidate_coverage["covered"].mean()
        ),
        "candidate_coverage_by_type": {
            str(row.site_type): {
                "covered": int(row.sum),
                "total": int(row.count),
                "fraction": float(row.fraction),
            }
            for row in coverage_by_type.itertuples(index=False)
        },
        "multi_site_context_count": int(
            tables.multi_site_pairs["context_id"].nunique()
            if not tables.multi_site_pairs.empty
            else 0
        ),
        "multi_site_pair_count": int(len(tables.multi_site_pairs)),
        "site_probability_normalization": False,
        "unmeasured_candidates_are_negative": False,
        "same_context_same_site_target": "arithmetic_mean",
        "pretraining_connectivity_overlap": {
            str(row.dataset_scope): int(row.overlap_record_count)
            for row in overlap_audit.itertuples(index=False)
        },
        "stage_gate_status": "database_built_pending_model_and_pretraining_pilots",
    }
    atomic_write_json(outputs["summary"], summary)

    hashed_outputs = {
        name: _file_entry(path, root=output)
        for name, path in outputs.items()
        if name != "dataset_manifest"
    }
    source_hashes = _source_hashes(config_file)
    manifest: dict[str, object] = {
        "schema_version": DATASET_SCHEMA,
        "dataset_id": str(config["dataset_id"]),
        "generated_by": "nucpred.datasets.mayr_site_n",
        "parent_dataset_id": str(parent_section["dataset_id"]),
        "parent_records": {
            "path": str(parent_path.relative_to(ROOT)),
            "sha256": sha256_file(parent_path),
            "record_count": int(len(parent)),
        },
        "contracts": {
            "target_unit": "species+solvent_context+site_object",
            "aggregation": "arithmetic_mean",
            "counterion_policy": "spectator_stripped_parent_identity",
            "split_identity": "standard_inchi_key_connectivity_block",
            "site_probability_normalization": False,
            "unmeasured_candidates_are_negative": False,
            "representative_feature_policy": (
                "prefer_complete_xtb10_then_sha256_source_id"
            ),
        },
        "source_hashes": source_hashes,
        "files": hashed_outputs,
        "summary": summary,
    }
    atomic_write_json(outputs["dataset_manifest"], manifest)
    verify_dataset(output)
    return outputs


def verify_dataset(directory: str | Path) -> dict[str, object]:
    root = Path(directory).resolve()
    manifest_path = root / "dataset_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != DATASET_SCHEMA:
        raise SiteNDatasetError("Unsupported site-N dataset manifest")
    checked = 0
    for entry in dict(payload["files"]).values():
        relative = Path(str(entry["path"]))
        path = root / relative
        if not path.is_file():
            raise SiteNDatasetError(f"Dataset file is missing: {relative}")
        if int(path.stat().st_size) != int(entry["bytes"]):
            raise SiteNDatasetError(f"Dataset file size changed: {relative}")
        if sha256_file(path) != str(entry["sha256"]):
            raise SiteNDatasetError(f"Dataset file hash changed: {relative}")
        checked += 1
    return {
        "schema_version": "nucpred.mayr-site-n-verification.v1",
        "dataset_id": str(payload["dataset_id"]),
        "status": "pass",
        "verified_file_count": checked,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-directory")
    parser.add_argument("--verify-directory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    if arguments.verify_directory:
        result = verify_dataset(arguments.verify_directory)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    outputs = build_dataset(
        config_path=arguments.config,
        output_directory=arguments.output_directory,
    )
    print(
        json.dumps(
            {name: str(path) for name, path in outputs.items()},
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
