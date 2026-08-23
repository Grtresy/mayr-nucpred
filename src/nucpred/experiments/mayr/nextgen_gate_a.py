"""Build the target-blind candidate ontology and evidence census for Gate A."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import tomllib
from typing import Any

import pandas as pd
from rdkit import Chem

from nucpred.core.files import atomic_write_json, sha256_file
from nucpred.project import get_project_layout

from .site_n import SiteNCampaignError, _display_path, _write_manifest


_LAYOUT = get_project_layout()
ROOT = _LAYOUT.root
DEFAULT_CONFIG = ROOT / "configs/mayr_nextgen_gate_a.toml"
CONFIG_SCHEMA = "nucpred.mayr-nextgen-gate-a-config.v1"
CAMPAIGN_ID = "mayr-nextgen-gate-a-20260727-v1"
SITE_TYPES = (
    "atom",
    "bond",
    "delocalized_region",
    "atom_group",
    "transferable_h_group",
)
NON_ATOM_TYPES = (
    "bond",
    "delocalized_region",
    "atom_group",
    "transferable_h_group",
)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SiteNCampaignError(f"Expected a JSON object: {path}")
    return payload


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _read_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = Path(path).resolve()
    payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != CONFIG_SCHEMA:
        raise SiteNCampaignError("Unsupported next-generation Gate A config")
    if payload.get("campaign_id") != CAMPAIGN_ID:
        raise SiteNCampaignError("Gate A campaign identity changed")
    if payload.get("formal_feature_scope") != "strict_no_dft_rdkit_xtb":
        raise SiteNCampaignError("Gate A must remain strictly no-DFT")
    if (
        payload.get("deployment_population")
        != "mayr_like_molecules_not_arbitrary_molecules"
    ):
        raise SiteNCampaignError("Gate A deployment population changed")
    if payload.get("test_used_for_selection") is not False:
        raise SiteNCampaignError("Test-based model selection is forbidden")
    if int(payload.get("maximum_parallel_gpu_processes", 0)) != 3:
        raise SiteNCampaignError("Gate A GPU concurrency must remain three")
    if payload["review"].get("independent_second_reviewer_available") is not False:
        raise SiteNCampaignError("Reviewer availability declaration changed")
    if payload["review"].get("claim_two_independent_reviewers") is not False:
        raise SiteNCampaignError("Gate A cannot claim two reviewers")
    if payload["evidence"].get("unknown_is_negative") is not False:
        raise SiteNCampaignError("Unknown candidates cannot become negatives")
    if payload["candidate_policy"].get("selection_reads_target_columns") is not False:
        raise SiteNCampaignError("Candidate deployment policy must be target-blind")
    return payload


def _verify_bound_file(path: Path, expected_sha256: str) -> None:
    if not path.is_file():
        raise SiteNCampaignError(f"Bound Gate A input is missing: {path}")
    observed = sha256_file(path)
    if observed != str(expected_sha256):
        raise SiteNCampaignError(
            f"Bound Gate A input drifted: {_display_path(path)} "
            f"{observed} != {expected_sha256}"
        )


def _json_list(value: object) -> list[Any]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, list):
        return value
    parsed = json.loads(str(value))
    if not isinstance(parsed, list):
        raise SiteNCampaignError("Expected a JSON list")
    return parsed


def _explicit_h_molecule(smiles: str) -> Chem.Mol:
    base = Chem.MolFromSmiles(str(smiles))
    if base is None:
        raise SiteNCampaignError(f"Cannot parse Gate A species: {smiles!r}")
    return Chem.AddHs(base)


def _maximum_pairwise_graph_distance(
    molecule: Chem.Mol,
    members: Sequence[int],
) -> int | None:
    maximum = 0
    for left_position, left in enumerate(members):
        for right in members[left_position + 1 :]:
            path = tuple(
                Chem.GetShortestPath(molecule, int(left), int(right))
            )
            if not path:
                return None
            maximum = max(maximum, len(path) - 1)
    return maximum


def _classify_candidates(
    candidates: pd.DataFrame,
    species: pd.DataFrame,
    *,
    policy: Mapping[str, Any],
) -> pd.DataFrame:
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
        raise SiteNCampaignError(
            f"Candidate table is missing columns: {missing}"
        )
    if not candidates["label_independent"].astype(bool).all():
        raise SiteNCampaignError("Candidate enumeration is not target-independent")
    if set(candidates["site_type"].astype(str)) != set(SITE_TYPES):
        raise SiteNCampaignError("Candidate site-type vocabulary changed")

    smiles_by_species = (
        species.set_index("species_id")["model_canonical_smiles"]
        .astype(str)
        .to_dict()
    )
    molecule_cache: dict[str, Chem.Mol] = {}
    strong_group_origins = set(
        map(str, policy["atom_group_strong_origins"])
    )
    region_backup_origins = set(
        map(str, policy["region_audit_only_when_origins_exactly"])
    )
    maximum_group_distance = int(
        policy["atom_group_maximum_pairwise_graph_distance"]
    )

    output = candidates.copy()
    deployment: list[bool] = []
    reasons: list[str] = []
    maximum_distances: list[int | None] = []
    for row in output.to_dict("records"):
        site_type = str(row["site_type"])
        origins = set(map(str, _json_list(row["candidate_origins_json"])))
        members = tuple(
            sorted(
                set(
                    int(index)
                    for index in _json_list(
                        row["member_atom_indices_json"]
                    )
                )
            )
        )
        if not origins or not members:
            raise SiteNCampaignError("Candidate identity is incomplete")
        maximum_distance: int | None = None
        if site_type == "delocalized_region":
            eligible = origins != region_backup_origins
            reason = (
                "chemically_structured_region"
                if eligible
                else "shortest_path_only_audit_backup"
            )
        elif site_type == "atom_group":
            species_id = str(row["species_id"])
            molecule = molecule_cache.get(species_id)
            if molecule is None:
                molecule = _explicit_h_molecule(
                    smiles_by_species[species_id]
                )
                molecule_cache[species_id] = molecule
            maximum_distance = _maximum_pairwise_graph_distance(
                molecule, members
            )
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
            reason = {
                "atom": "all_explicit_atoms",
                "bond": "all_graph_bonds",
                "transferable_h_group": "explicit_h_parent_or_symmetry_group",
            }[site_type]
        deployment.append(bool(eligible))
        reasons.append(reason)
        maximum_distances.append(maximum_distance)

    output["candidate_tier"] = [
        "deployment" if value else "audit_only" for value in deployment
    ]
    output["deployment_eligible"] = deployment
    output["deployment_reason"] = reasons
    output["maximum_pairwise_graph_distance"] = pd.array(
        maximum_distances, dtype="Int64"
    )
    output["candidate_policy_id"] = str(policy["policy_id"])
    if output["candidate_site_id"].astype(str).duplicated().any():
        raise SiteNCampaignError("Candidate policy duplicated identities")
    return output


def _candidate_count_summary(policy_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for site_type in SITE_TYPES:
        group = policy_frame.loc[
            policy_frame["site_type"].astype(str).eq(site_type)
        ]
        audit_count = len(group)
        deployment_count = int(group["deployment_eligible"].sum())
        rows.append(
            {
                "site_type": site_type,
                "audit_candidate_count": audit_count,
                "deployment_candidate_count": deployment_count,
                "audit_only_candidate_count": (
                    audit_count - deployment_count
                ),
                "retained_fraction": (
                    deployment_count / audit_count if audit_count else math.nan
                ),
                "species_count": int(group["species_id"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def _deployment_coverage(
    sites: pd.DataFrame,
    policy_frame: pd.DataFrame,
) -> pd.DataFrame:
    deployment_ids = set(
        policy_frame.loc[
            policy_frame["deployment_eligible"].astype(bool),
            "candidate_site_id",
        ].astype(str)
    )
    evaluated = sites.loc[
        :, ("site_object_id", "species_id", "site_type")
    ].copy()
    evaluated["deployment_covered"] = evaluated[
        "site_object_id"
    ].astype(str).isin(deployment_ids)
    rows: list[dict[str, object]] = []
    for site_type in SITE_TYPES:
        group = evaluated.loc[
            evaluated["site_type"].astype(str).eq(site_type)
        ]
        covered = int(group["deployment_covered"].sum())
        rows.append(
            {
                "site_type": site_type,
                "known_site_object_count": len(group),
                "deployment_covered_count": covered,
                "deployment_coverage_fraction": (
                    covered / len(group) if len(group) else math.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _candidate_relation_census(
    policy_frame: pd.DataFrame,
    *,
    minimum_partial_jaccard: float,
) -> pd.DataFrame:
    counters: Counter[tuple[str, str, str, str, str]] = Counter()
    jaccard_sums: defaultdict[
        tuple[str, str, str, str, str], float
    ] = defaultdict(float)
    jaccard_maxima: defaultdict[
        tuple[str, str, str, str, str], float
    ] = defaultdict(float)
    deployment = policy_frame.loc[
        policy_frame["deployment_eligible"].astype(bool),
        (
            "species_id",
            "candidate_site_id",
            "site_type",
            "member_atom_indices_json",
        ),
    ]
    for _, group in deployment.groupby("species_id", sort=False):
        records = [
            {
                "candidate_site_id": str(row["candidate_site_id"]),
                "site_type": str(row["site_type"]),
                "members": frozenset(
                    int(index)
                    for index in _json_list(
                        row["member_atom_indices_json"]
                    )
                ),
            }
            for row in group.to_dict("records")
        ]
        by_atom: defaultdict[int, list[int]] = defaultdict(list)
        for position, record in enumerate(records):
            for atom_index in record["members"]:
                by_atom[int(atom_index)].append(position)
        overlapping_pairs: set[tuple[int, int]] = set()
        for positions in by_atom.values():
            for left_position, left in enumerate(positions):
                for right in positions[left_position + 1 :]:
                    overlapping_pairs.add(
                        (min(left, right), max(left, right))
                    )
        for left_position, right_position in overlapping_pairs:
            left = records[left_position]
            right = records[right_position]
            left_members = left["members"]
            right_members = right["members"]
            intersection = left_members & right_members
            union = left_members | right_members
            jaccard = len(intersection) / len(union)
            type_a, type_b = sorted(
                (str(left["site_type"]), str(right["site_type"]))
            )
            contained_type = ""
            container_type = ""
            if left_members == right_members:
                relation = "exact_same_members"
            elif left_members < right_members:
                relation = "strict_containment"
                contained_type = str(left["site_type"])
                container_type = str(right["site_type"])
            elif right_members < left_members:
                relation = "strict_containment"
                contained_type = str(right["site_type"])
                container_type = str(left["site_type"])
            else:
                if jaccard < minimum_partial_jaccard:
                    continue
                relation = "partial_overlap_at_or_above_threshold"
            key = (
                type_a,
                type_b,
                relation,
                contained_type,
                container_type,
            )
            counters[key] += 1
            jaccard_sums[key] += float(jaccard)
            jaccard_maxima[key] = max(jaccard_maxima[key], float(jaccard))
    rows = []
    for key, count in sorted(counters.items()):
        type_a, type_b, relation, contained_type, container_type = key
        rows.append(
            {
                "site_type_a": type_a,
                "site_type_b": type_b,
                "relation": relation,
                "contained_type": contained_type,
                "container_type": container_type,
                "pair_count": count,
                "mean_jaccard": jaccard_sums[key] / count,
                "maximum_jaccard": jaccard_maxima[key],
            }
        )
    return pd.DataFrame(rows)


def _paper_keys(value: object) -> list[str]:
    return sorted(
        {
            str(item).strip()
            for item in _json_list(value)
            if str(item).strip()
        }
    )


def _source_ids(value: object) -> list[str]:
    return sorted(
        {
            str(item).strip()
            for item in _json_list(value)
            if str(item).strip()
        }
    )


def _claim_local_paths(claim: Mapping[str, object]) -> list[str]:
    candidates: list[str] = []
    direct = str(claim.get("evidence_file", "")).strip()
    if direct:
        candidates.append(direct)
    reference = str(claim.get("evidence_reference", ""))
    candidates.extend(
        token.strip()
        for token in reference.split("|")
        if token.strip().startswith(("reference/", "data/"))
    )
    existing = []
    for value in sorted(set(candidates)):
        path = Path(value)
        resolved = path if path.is_absolute() else ROOT / path
        if resolved.is_file():
            existing.append(_display_path(resolved))
    return existing


def _evidence_outputs(
    positive_targets: pd.DataFrame,
    evidence_claims: pd.DataFrame,
    *,
    review_scope: Iterable[str],
    formal_tier: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    claims_by_target: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in evidence_claims.to_dict("records"):
        claims_by_target[str(row["target_id"])].append(row)

    target_rows: list[dict[str, object]] = []
    for row in positive_targets.to_dict("records"):
        target_id = str(row["target_id"])
        claims = claims_by_target.get(target_id, [])
        local_paths = sorted(
            {
                path
                for claim in claims
                for path in _claim_local_paths(claim)
            }
        )
        dois = sorted(
            {
                str(claim.get("doi", "")).strip()
                for claim in claims
                if str(claim.get("doi", "")).strip()
            }
        )
        paper_keys = _paper_keys(row["paper_keys_json"])
        target_rows.append(
            {
                "target_id": target_id,
                "context_id": str(row["context_id"]),
                "species_id": str(row["species_id"]),
                "connectivity_id": str(row["connectivity_id"]),
                "site_object_id": str(row["site_object_id"]),
                "site_type": str(row["site_type"]),
                "member_atom_indices_json": str(
                    row["member_atom_indices_json"]
                ),
                "N_mean": float(row["N_mean"]),
                "positive_evidence_tier": str(
                    row["positive_evidence_tier"]
                ),
                "source_ids_json": str(row["source_ids_json"]),
                "source_count": len(_source_ids(row["source_ids_json"])),
                "paper_keys_json": json.dumps(
                    paper_keys, ensure_ascii=False, separators=(",", ":")
                ),
                "primary_paper_key": paper_keys[0] if paper_keys else "",
                "dois_json": json.dumps(
                    dois, ensure_ascii=False, separators=(",", ":")
                ),
                "claim_count": len(claims),
                "local_evidence_paths_json": json.dumps(
                    local_paths,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                "local_primary_file_available": bool(local_paths),
                "requires_positive_evidence_review": (
                    str(row["site_type"]) in set(review_scope)
                    and str(row["positive_evidence_tier"]) != formal_tier
                ),
            }
        )
    target_census = pd.DataFrame(target_rows)

    evidence_rows: list[dict[str, object]] = []
    for (site_type, tier), group in target_census.groupby(
        ["site_type", "positive_evidence_tier"], sort=True
    ):
        evidence_rows.append(
            {
                "site_type": str(site_type),
                "positive_evidence_tier": str(tier),
                "target_count": len(group),
                "context_count": int(group["context_id"].nunique()),
                "connectivity_count": int(
                    group["connectivity_id"].nunique()
                ),
                "paper_count": int(
                    len(
                        {
                            paper
                            for value in group["paper_keys_json"]
                            for paper in _json_list(value)
                        }
                    )
                ),
                "local_primary_file_target_count": int(
                    group["local_primary_file_available"].sum()
                ),
            }
        )
    evidence_census = pd.DataFrame(evidence_rows)

    queue = target_census.loc[
        target_census["requires_positive_evidence_review"].astype(bool)
    ].copy()
    paper_frequency = (
        queue.groupby("primary_paper_key", dropna=False)
        .size()
        .to_dict()
    )
    queue["paper_review_target_count"] = queue[
        "primary_paper_key"
    ].map(paper_frequency)
    queue["review_priority_group"] = [
        (
            0
            if bool(local)
            else 1,
            -int(paper_frequency.get(str(paper), 0)),
            str(site_type),
            str(target_id),
        )
        for local, paper, site_type, target_id in zip(
            queue["local_primary_file_available"],
            queue["primary_paper_key"],
            queue["site_type"],
            queue["target_id"],
            strict=True,
        )
    ]
    queue = queue.sort_values("review_priority_group").reset_index(drop=True)
    queue["review_order"] = range(1, len(queue) + 1)
    queue["review_status"] = "pending_primary_source_positive_adjudication"
    queue["review_pass_a_status"] = "not_started"
    queue["review_pass_b_status"] = "not_started"
    queue["independent_second_reviewer_available"] = False
    queue = queue.drop(columns=["review_priority_group"])

    workload_rows: list[dict[str, object]] = []
    for paper_key, group in queue.groupby("primary_paper_key", sort=True):
        workload_rows.append(
            {
                "primary_paper_key": str(paper_key),
                "target_count": len(group),
                "connectivity_count": int(
                    group["connectivity_id"].nunique()
                ),
                "site_types_json": json.dumps(
                    sorted(set(group["site_type"].astype(str))),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                "local_primary_file_available": bool(
                    group["local_primary_file_available"].any()
                ),
                "first_review_order": int(group["review_order"].min()),
            }
        )
    workload = pd.DataFrame(workload_rows)
    return evidence_census, queue, workload


def _negative_pool_feasibility(
    positive_targets: pd.DataFrame,
    policy_frame: pd.DataFrame,
    *,
    review_scope: Iterable[str],
    minimum_disjoint_candidates: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    deployment = policy_frame.loc[
        policy_frame["deployment_eligible"].astype(bool),
        (
            "candidate_site_id",
            "species_id",
            "site_type",
            "member_atom_indices_json",
        ),
    ].copy()
    candidates_by_species: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in deployment.to_dict("records"):
        candidates_by_species[str(row["species_id"])].append(
            {
                "candidate_site_id": str(row["candidate_site_id"]),
                "site_type": str(row["site_type"]),
                "members": frozenset(
                    int(index)
                    for index in _json_list(
                        row["member_atom_indices_json"]
                    )
                ),
            }
        )

    scope = set(map(str, review_scope))
    rows: list[dict[str, object]] = []
    scoped_targets = positive_targets.loc[
        positive_targets["site_type"].astype(str).isin(scope)
    ]
    for row in scoped_targets.to_dict("records"):
        target_members = frozenset(
            int(index)
            for index in _json_list(row["member_atom_indices_json"])
        )
        counts = Counter()
        overlap_counts = Counter()
        for candidate in candidates_by_species[str(row["species_id"])]:
            if candidate["candidate_site_id"] == str(row["site_object_id"]):
                continue
            site_type = str(candidate["site_type"])
            if target_members.isdisjoint(candidate["members"]):
                counts[site_type] += 1
            else:
                overlap_counts[site_type] += 1
        same_type = str(row["site_type"])
        rows.append(
            {
                "target_id": str(row["target_id"]),
                "context_id": str(row["context_id"]),
                "species_id": str(row["species_id"]),
                "connectivity_id": str(row["connectivity_id"]),
                "site_object_id": str(row["site_object_id"]),
                "site_type": same_type,
                "positive_evidence_tier": str(
                    row["positive_evidence_tier"]
                ),
                "disjoint_deployment_candidate_counts_json": json.dumps(
                    {key: int(counts[key]) for key in SITE_TYPES},
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "overlapping_ambiguous_candidate_counts_json": json.dumps(
                    {
                        key: int(overlap_counts[key])
                        for key in SITE_TYPES
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "same_type_disjoint_candidate_count": int(
                    counts[same_type]
                ),
                "all_type_disjoint_candidate_count": int(sum(counts.values())),
                "same_type_review_pool_feasible": bool(
                    counts[same_type] >= minimum_disjoint_candidates
                ),
                "all_type_review_pool_feasible": bool(
                    sum(counts.values()) >= minimum_disjoint_candidates
                ),
                "candidate_pool_is_evidence_not_negative_labels": True,
            }
        )
    feasibility = pd.DataFrame(rows)
    summary_rows: list[dict[str, object]] = []
    for site_type, group in feasibility.groupby("site_type", sort=True):
        summary_rows.append(
            {
                "site_type": str(site_type),
                "target_count": len(group),
                "connectivity_count": int(
                    group["connectivity_id"].nunique()
                ),
                "same_type_feasible_target_count": int(
                    group["same_type_review_pool_feasible"].sum()
                ),
                "same_type_feasible_fraction": float(
                    group["same_type_review_pool_feasible"].mean()
                ),
                "median_same_type_disjoint_candidate_count": float(
                    group["same_type_disjoint_candidate_count"].median()
                ),
                "minimum_same_type_disjoint_candidate_count": int(
                    group["same_type_disjoint_candidate_count"].min()
                ),
                "median_all_type_disjoint_candidate_count": float(
                    group["all_type_disjoint_candidate_count"].median()
                ),
            }
        )
    return feasibility, pd.DataFrame(summary_rows)


def _multi_site_summary(path: Path) -> dict[str, object]:
    frame = pd.read_csv(path)
    return {
        "schema_version": "nucpred.mayr-nextgen-multi-site-census.v1",
        "pair_row_count": len(frame),
        "context_count": (
            int(frame["context_id"].nunique())
            if "context_id" in frame
            else None
        ),
        "connectivity_count": (
            int(frame["connectivity_id"].nunique())
            if "connectivity_id" in frame
            else None
        ),
        "columns": list(frame.columns),
        "status": "data_gap_requires_gate_a_review",
    }


def _feature_scope_audit(
    config: Mapping[str, Any],
    formal_config_path: Path,
) -> dict[str, object]:
    formal = tomllib.loads(formal_config_path.read_text(encoding="utf-8"))
    base_config_path = (
        ROOT / str(formal["base_config"])
    ).resolve()
    base = tomllib.loads(base_config_path.read_text(encoding="utf-8"))
    model = base["model"]
    if (
        int(model["local_feature_dim"]) != 4
        or int(model["global_feature_dim"]) != 6
        or model["spatial_edges"] is not False
    ):
        raise SiteNCampaignError("Frozen no-DFT model feature contract changed")
    return {
        "schema_version": "nucpred.mayr-nextgen-feature-scope-audit.v1",
        "status": "pass",
        "formal_feature_scope": config["formal_feature_scope"],
        "local_feature_block": "GFN1-xTB local4",
        "global_feature_block": "GFN1-xTB global6",
        "rdkit_features": True,
        "solvent_features": True,
        "formal_charge": True,
        "spatial_edges": False,
        "dft_features": False,
        "cdft_features": False,
        "dft_or_cdft_computation_authorized": False,
        "base_config_path": _display_path(base_config_path),
        "base_config_sha256": sha256_file(base_config_path),
    }


def _contract(
    *,
    config_path: Path,
    config: Mapping[str, Any],
    site_n_manifest: Path,
    confidence_manifest: Path,
    authorization: Path,
) -> dict[str, object]:
    paths = {
        "config": config_path,
        "runner": Path(__file__).resolve(),
        "site_n_manifest": site_n_manifest,
        "confidence_manifest": confidence_manifest,
        "authorization": authorization,
    }
    contract: dict[str, object] = {
        "schema_version": "nucpred.mayr-nextgen-gate-a-preflight-contract.v1",
        "campaign_id": CAMPAIGN_ID,
        "anchor_commit": config["anchor_commit"],
        "source_hashes": {
            key: sha256_file(path) for key, path in paths.items()
        },
        "candidate_policy": dict(config["candidate_policy"]),
        "candidate_relations": dict(config["candidate_relations"]),
        "evidence_policy": dict(config["evidence"]),
        "review_policy": dict(config["review"]),
        "formal_feature_scope": config["formal_feature_scope"],
        "deployment_population": config["deployment_population"],
        "test_used_for_selection": False,
    }
    contract["contract_sha256"] = _canonical_sha256(contract)
    return contract


def build_gate_a_preflight(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
) -> dict[str, object]:
    config_file = Path(config_path).resolve()
    config = _read_config(config_file)
    site_n = config["parent_site_n"]
    confidence = config["parent_confidence"]
    site_n_root = (ROOT / str(site_n["directory"])).resolve()
    confidence_root = (ROOT / str(confidence["directory"])).resolve()
    site_n_manifest = (ROOT / str(site_n["manifest_path"])).resolve()
    confidence_manifest = (
        ROOT / str(confidence["manifest_path"])
    ).resolve()
    formal_config_path = (
        ROOT / str(site_n["formal_config_path"])
    ).resolve()
    authorization = (
        ROOT / str(config["authorization"]["path"])
    ).resolve()
    for path, expected in (
        (site_n_manifest, site_n["manifest_sha256"]),
        (confidence_manifest, confidence["manifest_sha256"]),
        (formal_config_path, site_n["formal_config_sha256"]),
        (authorization, config["authorization"]["sha256"]),
    ):
        _verify_bound_file(path, str(expected))

    output_root = (ROOT / str(config["output_root"])).resolve()
    target = output_root / "preflight"
    contract = _contract(
        config_path=config_file,
        config=config,
        site_n_manifest=site_n_manifest,
        confidence_manifest=confidence_manifest,
        authorization=authorization,
    )
    summary_path = target / "summary.json"
    if summary_path.is_file():
        existing = _load_json(summary_path)
        if (
            existing.get("status") == "pass"
            and existing.get("contract") == contract
        ):
            return existing
        raise SiteNCampaignError("Existing Gate A preflight is stale")
    if target.exists():
        raise SiteNCampaignError("Partial Gate A preflight exists")

    species = pd.read_parquet(site_n_root / "species.parquet")
    measurements = pd.read_parquet(site_n_root / "measurements.parquet")
    targets = pd.read_parquet(site_n_root / "targets.parquet")
    sites = pd.read_parquet(site_n_root / "sites.parquet")
    candidates = pd.read_parquet(site_n_root / "candidate_sites.parquet")
    positive_targets = pd.read_parquet(
        confidence_root / "positive_targets.parquet"
    )
    evidence_claims = pd.read_parquet(
        confidence_root / "evidence_claims.parquet"
    )
    expected_counts = {
        "species": int(site_n["expected_species_count"]),
        "measurements": int(site_n["expected_measurement_count"]),
        "targets": int(site_n["expected_target_count"]),
        "sites": int(site_n["expected_site_object_count"]),
        "candidates": int(site_n["expected_candidate_count"]),
    }
    observed_counts = {
        "species": len(species),
        "measurements": len(measurements),
        "targets": len(targets),
        "sites": len(sites),
        "candidates": len(candidates),
    }
    if observed_counts != expected_counts:
        raise SiteNCampaignError(
            f"Frozen Gate A population changed: {observed_counts}"
        )

    policy_frame = _classify_candidates(
        candidates,
        species,
        policy=config["candidate_policy"],
    )
    candidate_counts = _candidate_count_summary(policy_frame)
    coverage = _deployment_coverage(sites, policy_frame)
    required_coverage = float(
        config["candidate_policy"]["known_site_coverage_required"]
    )
    if not coverage["deployment_coverage_fraction"].eq(
        required_coverage
    ).all():
        raise SiteNCampaignError(
            "Deployment candidate policy lost a known formal site"
        )
    relation_census = _candidate_relation_census(
        policy_frame,
        minimum_partial_jaccard=float(
            config["candidate_relations"][
                "minimum_reported_jaccard_overlap"
            ]
        ),
    )
    evidence_census, review_queue, review_workload = _evidence_outputs(
        positive_targets,
        evidence_claims,
        review_scope=config["evidence"]["gate_a_review_scope"],
        formal_tier=str(config["evidence"]["formal_positive_tier"]),
    )
    negative_pool, negative_summary = _negative_pool_feasibility(
        positive_targets,
        policy_frame,
        review_scope=config["evidence"]["gate_a_review_scope"],
        minimum_disjoint_candidates=int(
            config["review"][
                "minimum_disjoint_candidates_for_feasible_context"
            ]
        ),
    )
    multi_site = _multi_site_summary(site_n_root / "multi_site_pairs.csv")
    feature_audit = _feature_scope_audit(config, formal_config_path)

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".preflight.staging-", dir=target.parent)
    )
    try:
        policy_frame.to_parquet(
            staging / "candidate_policy.parquet",
            index=False,
            engine="pyarrow",
            compression="zstd",
        )
        candidate_counts.to_csv(
            staging / "candidate_counts.csv",
            index=False,
            lineterminator="\n",
        )
        coverage.to_csv(
            staging / "deployment_coverage.csv",
            index=False,
            lineterminator="\n",
        )
        relation_census.to_csv(
            staging / "candidate_relation_census.csv",
            index=False,
            lineterminator="\n",
        )
        evidence_census.to_csv(
            staging / "positive_evidence_census.csv",
            index=False,
            lineterminator="\n",
        )
        review_queue.to_csv(
            staging / "positive_evidence_review_queue.csv",
            index=False,
            lineterminator="\n",
        )
        review_workload.to_csv(
            staging / "positive_review_workload_by_paper.csv",
            index=False,
            lineterminator="\n",
        )
        negative_pool.to_parquet(
            staging / "negative_pool_feasibility.parquet",
            index=False,
            engine="pyarrow",
            compression="zstd",
        )
        negative_summary.to_csv(
            staging / "negative_pool_summary.csv",
            index=False,
            lineterminator="\n",
        )
        atomic_write_json(
            staging / "multi_site_census.json",
            multi_site,
            ensure_ascii=False,
        )
        atomic_write_json(
            staging / "feature_scope_audit.json",
            feature_audit,
            ensure_ascii=False,
        )
        atomic_write_json(
            staging / "run_config.json",
            {
                "schema_version": (
                    "nucpred.mayr-nextgen-gate-a-preflight-config.v1"
                ),
                "campaign_id": CAMPAIGN_ID,
                "config_path": _display_path(config_file),
                "config_sha256": sha256_file(config_file),
                "authorization_path": _display_path(authorization),
                "authorization_sha256": sha256_file(authorization),
                "site_n_manifest_sha256": sha256_file(site_n_manifest),
                "confidence_manifest_sha256": sha256_file(
                    confidence_manifest
                ),
                "observed_counts": observed_counts,
            },
            ensure_ascii=False,
        )
        summary: dict[str, object] = {
            "schema_version": (
                "nucpred.mayr-nextgen-gate-a-preflight-summary.v1"
            ),
            "status": "pass",
            "campaign_id": CAMPAIGN_ID,
            "contract": contract,
            "formal_feature_scope": config["formal_feature_scope"],
            "deployment_population": config["deployment_population"],
            "observed_counts": observed_counts,
            "deployment_candidate_count": int(
                policy_frame["deployment_eligible"].sum()
            ),
            "audit_only_candidate_count": int(
                (~policy_frame["deployment_eligible"].astype(bool)).sum()
            ),
            "known_site_coverage_minimum": float(
                coverage["deployment_coverage_fraction"].min()
            ),
            "positive_evidence_review_target_count": len(review_queue),
            "positive_evidence_review_paper_count": int(
                review_workload["primary_paper_key"].nunique()
            ),
            "negative_pool_target_count": len(negative_pool),
            "all_negative_pools_are_unknown_until_review": True,
            "multi_site_pair_count": int(multi_site["pair_row_count"]),
            "independent_second_reviewer_available": False,
            "claim_two_independent_reviewers": False,
            "test_used_for_selection": False,
            "dft_or_cdft_computation_authorized": False,
        }
        atomic_write_json(
            staging / "summary.json", summary, ensure_ascii=False
        )
        _write_manifest(staging)
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    result = build_gate_a_preflight(config_path=arguments.config)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
