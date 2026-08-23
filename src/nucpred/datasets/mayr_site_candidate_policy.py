"""Target-blind deployment policy for dynamically enumerated Mayr site candidates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
from typing import Any

import pandas as pd
from rdkit import Chem


POLICY_ID = "mayr-multitype-deployment-candidates-v1"
SITE_TYPES = (
    "atom",
    "bond",
    "delocalized_region",
    "atom_group",
    "transferable_h_group",
)
DEFAULT_POLICY: dict[str, object] = {
    "policy_id": POLICY_ID,
    "region_audit_only_when_origins_exactly": ["heavy_shortest_path"],
    "atom_group_strong_origins": [
        "rdkit_symmetry",
        "same_element_neighbours",
        "ring_same_element",
        "conjugated_same_element",
    ],
    "atom_group_maximum_pairwise_graph_distance": 2,
    "selection_reads_target_columns": False,
}


class CandidatePolicyError(RuntimeError):
    """Raised when a candidate frame cannot satisfy the deployment policy."""


def _json_list(value: object) -> list[Any]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, list):
        return value
    parsed = json.loads(str(value))
    if not isinstance(parsed, list):
        raise CandidatePolicyError("Expected a JSON list")
    return parsed


def _explicit_h_molecule(smiles: str) -> Chem.Mol:
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise CandidatePolicyError(f"Cannot parse candidate species: {smiles!r}")
    return Chem.AddHs(molecule)


def _maximum_pairwise_graph_distance(
    molecule: Chem.Mol,
    members: Sequence[int],
) -> int | None:
    maximum = 0
    for left_position, left in enumerate(members):
        for right in members[left_position + 1 :]:
            path = tuple(Chem.GetShortestPath(molecule, int(left), int(right)))
            if not path:
                return None
            maximum = max(maximum, len(path) - 1)
    return maximum


def classify_deployment_candidates(
    candidates: pd.DataFrame,
    species: pd.DataFrame,
    *,
    policy: Mapping[str, object] = DEFAULT_POLICY,
    require_all_site_types: bool = False,
) -> pd.DataFrame:
    """Apply the frozen Gate-A rules without reading target or site labels.

    The same function applies to a complete frozen dataset or to candidates
    enumerated for a single previously unseen molecular species.
    """

    required = {
        "candidate_site_id",
        "species_id",
        "site_type",
        "member_atom_indices_json",
        "member_bond_pairs_json",
        "candidate_origins_json",
        "label_independent",
    }
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise CandidatePolicyError(f"Candidate table lacks columns: {missing}")
    if candidates.empty:
        raise CandidatePolicyError("Candidate table is empty")
    if not candidates["label_independent"].astype(bool).all():
        raise CandidatePolicyError("Candidate enumeration exposed a label")
    observed_types = set(candidates["site_type"].astype(str))
    if observed_types - set(SITE_TYPES):
        raise CandidatePolicyError("Candidate site-type vocabulary changed")
    if require_all_site_types and observed_types != set(SITE_TYPES):
        raise CandidatePolicyError("Complete policy audit lost a site type")
    if policy.get("selection_reads_target_columns") is not False:
        raise CandidatePolicyError("Deployment policy may not read target columns")

    required_species = {"species_id", "model_canonical_smiles"}
    missing_species = sorted(required_species - set(species.columns))
    if missing_species:
        raise CandidatePolicyError(f"Species table lacks columns: {missing_species}")
    if species["species_id"].astype(str).duplicated().any():
        raise CandidatePolicyError("Species identities are duplicated")
    smiles_by_species = (
        species.set_index("species_id")["model_canonical_smiles"].astype(str).to_dict()
    )
    missing_ids = sorted(
        set(candidates["species_id"].astype(str)) - set(smiles_by_species)
    )
    if missing_ids:
        raise CandidatePolicyError(
            f"Candidate species are unavailable: {missing_ids[:5]}"
        )

    strong_group_origins = set(map(str, policy["atom_group_strong_origins"]))
    region_backup_origins = set(
        map(str, policy["region_audit_only_when_origins_exactly"])
    )
    maximum_group_distance = int(policy["atom_group_maximum_pairwise_graph_distance"])
    molecule_cache: dict[str, Chem.Mol] = {}
    deployment: list[bool] = []
    reasons: list[str] = []
    maximum_distances: list[int | None] = []
    for row in candidates.to_dict(orient="records"):
        species_id = str(row["species_id"])
        site_type = str(row["site_type"])
        origins = set(map(str, _json_list(row["candidate_origins_json"])))
        members = tuple(
            sorted(
                set(int(index) for index in _json_list(row["member_atom_indices_json"]))
            )
        )
        if not origins or not members:
            raise CandidatePolicyError("Candidate identity is incomplete")
        maximum_distance: int | None = None
        if site_type == "delocalized_region":
            eligible = origins != region_backup_origins
            reason = (
                "chemically_structured_region"
                if eligible
                else "shortest_path_only_audit_backup"
            )
        elif site_type == "atom_group":
            molecule = molecule_cache.get(species_id)
            if molecule is None:
                molecule = _explicit_h_molecule(smiles_by_species[species_id])
                molecule_cache[species_id] = molecule
            maximum_distance = _maximum_pairwise_graph_distance(molecule, members)
            strong_origin = bool(origins & strong_group_origins)
            locally_bounded = (
                maximum_distance is not None
                and maximum_distance <= maximum_group_distance
            )
            eligible = strong_origin or locally_bounded
            if strong_origin:
                reason = "chemically_structured_atom_group"
            elif locally_bounded:
                reason = "local_same_element_atom_group"
            else:
                reason = "global_or_combinatorial_atom_group_audit_only"
        else:
            eligible = True
            try:
                reason = {
                    "atom": "all_explicit_atoms",
                    "bond": "all_graph_bonds",
                    "transferable_h_group": ("explicit_h_parent_or_symmetry_group"),
                }[site_type]
            except KeyError as exc:
                raise CandidatePolicyError(
                    f"Unsupported candidate type: {site_type}"
                ) from exc
        deployment.append(bool(eligible))
        reasons.append(reason)
        maximum_distances.append(maximum_distance)

    output = candidates.copy()
    output["candidate_tier"] = [
        "deployment" if value else "audit_only" for value in deployment
    ]
    output["deployment_eligible"] = deployment
    output["deployment_reason"] = reasons
    output["maximum_pairwise_graph_distance"] = pd.array(
        maximum_distances,
        dtype="Int64",
    )
    output["candidate_policy_id"] = str(policy["policy_id"])
    if output["candidate_site_id"].astype(str).duplicated().any():
        raise CandidatePolicyError("Candidate policy duplicated identities")
    return output


def select_deployment_candidates(
    candidates: pd.DataFrame,
    species: pd.DataFrame,
    *,
    policy: Mapping[str, object] = DEFAULT_POLICY,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Return runtime-compatible candidates and a target-blind policy audit."""

    classified = classify_deployment_candidates(
        candidates,
        species,
        policy=policy,
    )
    gate_a = classified["deployment_eligible"].astype(bool)
    no_internal_bond_region = classified["site_type"].astype(str).eq(
        "delocalized_region"
    ) & classified["member_bond_pairs_json"].astype(str).eq("[]")
    selected = classified.loc[gate_a & ~no_internal_bond_region].copy()
    if selected.empty:
        raise CandidatePolicyError("Deployment policy selected no candidates")
    audit = {
        "candidate_policy_id": str(policy["policy_id"]),
        "audit_population_count": len(classified),
        "gate_a_deployment_count": int(gate_a.sum()),
        "contract_incompatible_no_bond_region_count": int(
            (gate_a & no_internal_bond_region).sum()
        ),
        "final_deployment_candidate_count": len(selected),
        "filter_target_independent": True,
    }
    return selected.reset_index(drop=True), audit
