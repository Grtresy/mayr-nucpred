"""Split-safe hierarchical endpoint-type routing for joint Mayr site scores."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
from itertools import combinations
import json
import math
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.linear_model import LogisticRegression

from nucpred.training.mayr_site_n import SITE_TYPE_NAMES


TYPE_ROUTER_SCHEMA_VERSION = "nucpred.mayr-joint-site-type-router.v3"
TYPE_ROUTER_SELECTION_SCHEMA_VERSION = (
    "nucpred.mayr-joint-site-type-router-selection.v2"
)
MORGAN_RADIUS = 2
MORGAN_BITS = 1024
TYPE_ROUTER_CLASSES = tuple(SITE_TYPE_NAMES)
CANDIDATE_ORIGIN_NAMES = (
    "all_atoms",
    "all_bonds",
    "all_reactive_carbons",
    "aromatic_carbons",
    "carbon_component_after_heteroatom_cut",
    "conjugated_component",
    "conjugated_element_filtered",
    "conjugated_same_element",
    "conjugated_same_element_subset",
    "cyclic_ring_path",
    "fused_ring_element_filtered",
    "fused_ring_system",
    "heavy_radius_one",
    "heavy_radius_one_element_filtered",
    "heavy_shortest_path",
    "non_aromatic_reactive_carbons",
    "non_ring_reactive_carbons",
    "rdkit_symmetry",
    "ring",
    "ring_element_filtered",
    "ring_same_element",
    "ring_same_element_subset",
    "same_element_neighbour_region",
    "same_element_neighbours",
    "same_parent_explicit_h",
    "symmetry_equivalent_h_parents",
    "whole_graph_same_element",
    "whole_graph_same_element_subset",
)
CONTEXT_NUMERIC_COLUMNS = (
    "formal_charge",
    "model_source_atom_count",
    "model_all_atom_count",
    "model_hydrogen_atom_count",
    "model_formal_charge",
    "model_radical_electrons",
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
ELEMENT_CHANNELS = (1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53)
SCORE_COLUMNS = (
    "pre_router_canonical_logit",
    "base_canonical_logit",
    "residual_canonical_logit",
    "conditional_n_prediction",
)
FEATURE_RANGE_CLIPPING_CONTRACT = (
    "per_feature_training_min_max_before_standardization.v1"
)


class JointSiteTypeRouterError(RuntimeError):
    """Raised when hierarchical type-router evidence or identity drifts."""


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    *,
    label: str,
) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise JointSiteTypeRouterError(f"{label} lacks columns: {missing}")


def _json_list(value: object, *, label: str) -> list[Any]:
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, list):
        raise JointSiteTypeRouterError(f"{label} must be a JSON list")
    return parsed


def _canonical_payload_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _candidate_metadata(
    scores: pd.DataFrame,
    queries: pd.DataFrame,
) -> pd.DataFrame:
    _require_columns(
        scores,
        {
            "query_id",
            "context_id",
            "species_id",
            "connectivity_id",
            "candidate_site_id",
            "site_type",
            "canonical_logit",
            "base_canonical_logit",
            "residual_canonical_logit",
            "conditional_n_prediction",
        },
        label="Candidate score frame",
    )
    _require_columns(
        queries,
        {
            "query_id",
            "context_id",
            "candidate_site_id",
            "site_type",
            "member_atom_indices_json",
            "member_bond_pairs_json",
            "member_atomic_numbers_json",
            "candidate_origins_json",
        },
        label="Candidate query frame",
    )
    if scores["query_id"].duplicated().any() or queries["query_id"].duplicated().any():
        raise JointSiteTypeRouterError("Candidate query identity is duplicated")
    metadata = queries[
        [
            "query_id",
            "context_id",
            "candidate_site_id",
            "site_type",
            "member_atom_indices_json",
            "member_bond_pairs_json",
            "member_atomic_numbers_json",
            "candidate_origins_json",
        ]
    ].copy()
    payload_columns = (
        "member_atom_indices_json",
        "member_bond_pairs_json",
        "member_atomic_numbers_json",
        "candidate_origins_json",
    )
    merged = scores.drop(
        columns=[column for column in payload_columns if column in scores]
    ).copy()
    merged["pre_router_canonical_logit"] = merged["canonical_logit"].astype(float)
    merged = merged.merge(
        metadata,
        on="query_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_metadata"),
    )
    for column in ("context_id", "candidate_site_id", "site_type"):
        metadata_column = f"{column}_metadata"
        if merged[metadata_column].isna().any() or not merged[column].astype(
            str
        ).equals(merged[metadata_column].astype(str)):
            raise JointSiteTypeRouterError(
                f"Candidate {column} changed while joining router metadata"
            )
        merged = merged.drop(columns=metadata_column)
    unknown_types = sorted(set(merged["site_type"].astype(str)) - set(SITE_TYPE_NAMES))
    if unknown_types:
        raise JointSiteTypeRouterError(
            f"Unknown candidate site types: {unknown_types}"
        )
    for column in SCORE_COLUMNS:
        values = merged[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise JointSiteTypeRouterError(
                f"Router score feature is non-finite: {column}"
            )
    observed_origins = {
        str(origin)
        for raw in merged["candidate_origins_json"]
        for origin in _json_list(raw, label="candidate origins")
    }
    unknown_origins = sorted(observed_origins - set(CANDIDATE_ORIGIN_NAMES))
    if unknown_origins:
        raise JointSiteTypeRouterError(
            f"Candidate origin vocabulary changed: {unknown_origins}"
        )
    return merged


def build_type_router_features(
    scores: pd.DataFrame,
    *,
    queries: pd.DataFrame,
    contexts: pd.DataFrame,
) -> pd.DataFrame:
    """Build fixed-width label-blind context/type routing features."""

    candidates = _candidate_metadata(scores, queries)
    _require_columns(
        contexts,
        {
            "context_id",
            "model_canonical_smiles",
            "molecule_global6_json",
            "molecule_global6_available_json",
            "model_atomic_numbers_json",
            *CONTEXT_NUMERIC_COLUMNS,
        },
        label="Router context frame",
    )
    context_table = contexts.set_index("context_id", drop=False)
    if context_table.index.duplicated().any():
        raise JointSiteTypeRouterError("Router context identity is duplicated")
    requested_contexts = set(candidates["context_id"].astype(str))
    if not requested_contexts <= set(context_table.index.astype(str)):
        raise JointSiteTypeRouterError("Router score context lacks molecular metadata")

    fingerprint_generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=MORGAN_RADIUS,
        fpSize=MORGAN_BITS,
    )
    fingerprint_cache: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    for context_id, group in candidates.groupby("context_id", sort=True):
        context = context_table.loc[str(context_id)]
        if isinstance(context, pd.DataFrame):
            raise JointSiteTypeRouterError("Router context identity is ambiguous")
        row: dict[str, object] = {
            "context_id": str(context_id),
            "species_id": str(group["species_id"].iloc[0]),
            "connectivity_id": str(group["connectivity_id"].iloc[0]),
        }
        for column in CONTEXT_NUMERIC_COLUMNS:
            row[f"context::{column}"] = float(context[column])
        global_values = _json_list(
            context["molecule_global6_json"], label="molecule global values"
        )
        global_available = _json_list(
            context["molecule_global6_available_json"],
            label="molecule global availability",
        )
        if len(global_values) != 6 or len(global_available) != 6:
            raise JointSiteTypeRouterError("Molecule-global feature width changed")
        for index, (value, available) in enumerate(
            zip(global_values, global_available, strict=True)
        ):
            row[f"context::global::{index}"] = (
                float(value) if bool(available) else math.nan
            )
            row[f"context::global_available::{index}"] = float(bool(available))
        atomic_numbers = tuple(
            map(
                int,
                _json_list(
                    context["model_atomic_numbers_json"],
                    label="model atomic numbers",
                ),
            )
        )
        if not atomic_numbers:
            raise JointSiteTypeRouterError("Router molecule has no atoms")
        for atomic_number in ELEMENT_CHANNELS:
            row[f"context::element_fraction::{atomic_number}"] = (
                atomic_numbers.count(atomic_number) / len(atomic_numbers)
            )
        smiles = str(context["model_canonical_smiles"])
        if smiles not in fingerprint_cache:
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                raise JointSiteTypeRouterError(
                    f"Cannot parse router molecule: {smiles!r}"
                )
            fingerprint = np.zeros(MORGAN_BITS, dtype=np.float64)
            DataStructs.ConvertToNumpyArray(
                fingerprint_generator.GetFingerprint(molecule),
                fingerprint,
            )
            fingerprint_cache[smiles] = fingerprint
        for index, value in enumerate(fingerprint_cache[smiles]):
            row[f"morgan::{index}"] = float(value)

        global_maximum = float(group["pre_router_canonical_logit"].max())
        top_members_by_type: dict[str, set[int]] = {}
        for site_type in SITE_TYPE_NAMES:
            selected = group.loc[
                group["site_type"].astype(str).eq(site_type)
            ].sort_values(
                ["pre_router_canonical_logit", "query_id"],
                ascending=[False, True],
                kind="stable",
            )
            prefix = f"type::{site_type}"
            row[f"{prefix}::present"] = float(not selected.empty)
            row[f"{prefix}::log1p_count"] = math.log1p(len(selected))
            if selected.empty:
                top_members_by_type[site_type] = set()
                for score_column in SCORE_COLUMNS:
                    for statistic in ("max", "mean", "std", "top2_gap"):
                        row[f"{prefix}::{score_column}::{statistic}"] = 0.0
                row[f"{prefix}::relative_global_max"] = 0.0
                row[f"{prefix}::top_member_log1p_count"] = 0.0
                row[f"{prefix}::top_internal_bond_log1p_count"] = 0.0
                row[f"{prefix}::top_internal_bond_per_member"] = 0.0
                row[f"{prefix}::top_member_fraction_of_model_atoms"] = 0.0
                row[f"{prefix}::top_member_fraction_of_source_atoms"] = 0.0
                row[f"{prefix}::top_unique_element_fraction"] = 0.0
                row[f"{prefix}::top_single_element_membership"] = 0.0
                row[f"{prefix}::top_same_element_coverage"] = 0.0
                row[f"{prefix}::top_rdkit_symmetry_same_element_coverage"] = 0.0
                for atomic_number in ELEMENT_CHANNELS:
                    row[f"{prefix}::top_element_fraction::{atomic_number}"] = 0.0
                for origin in CANDIDATE_ORIGIN_NAMES:
                    row[f"{prefix}::top_origin::{origin}"] = 0.0
                    row[f"{prefix}::origin_fraction::{origin}"] = 0.0
                continue
            top = selected.iloc[0]
            for score_column in SCORE_COLUMNS:
                values = selected[score_column].to_numpy(dtype=float)
                row[f"{prefix}::{score_column}::max"] = float(values.max())
                row[f"{prefix}::{score_column}::mean"] = float(values.mean())
                row[f"{prefix}::{score_column}::std"] = float(values.std(ddof=0))
                ordered_values = np.sort(values)[::-1]
                row[f"{prefix}::{score_column}::top2_gap"] = (
                    float(ordered_values[0] - ordered_values[1])
                    if len(ordered_values) > 1
                    else 0.0
                )
            row[f"{prefix}::relative_global_max"] = float(
                top["pre_router_canonical_logit"] - global_maximum
            )
            members = list(
                map(
                    int,
                    _json_list(
                        top["member_atom_indices_json"], label="candidate members"
                    ),
                )
            )
            member_atomic_numbers = list(
                map(
                    int,
                    _json_list(
                        top["member_atomic_numbers_json"],
                        label="candidate member atomic numbers",
                    ),
                )
            )
            internal_bonds = _json_list(
                top["member_bond_pairs_json"], label="candidate internal bonds"
            )
            if (
                not members
                or len(members) != len(set(members))
                or len(member_atomic_numbers) != len(members)
            ):
                raise JointSiteTypeRouterError(
                    "Top candidate member metadata is inconsistent"
                )
            top_members_by_type[site_type] = set(members)
            row[f"{prefix}::top_member_log1p_count"] = math.log1p(len(members))
            row[f"{prefix}::top_internal_bond_log1p_count"] = math.log1p(
                len(internal_bonds)
            )
            row[f"{prefix}::top_internal_bond_per_member"] = len(
                internal_bonds
            ) / len(members)
            row[f"{prefix}::top_member_fraction_of_model_atoms"] = len(
                members
            ) / len(atomic_numbers)
            row[f"{prefix}::top_member_fraction_of_source_atoms"] = len(
                members
            ) / max(float(context["model_source_atom_count"]), 1.0)
            unique_elements = set(member_atomic_numbers)
            row[f"{prefix}::top_unique_element_fraction"] = len(
                unique_elements
            ) / len(member_atomic_numbers)
            single_element = len(unique_elements) == 1
            row[f"{prefix}::top_single_element_membership"] = float(
                single_element
            )
            same_element_coverage = 0.0
            if single_element:
                element = member_atomic_numbers[0]
                same_element_coverage = len(member_atomic_numbers) / max(
                    atomic_numbers.count(element), 1
                )
            row[f"{prefix}::top_same_element_coverage"] = float(
                same_element_coverage
            )
            for atomic_number in ELEMENT_CHANNELS:
                row[f"{prefix}::top_element_fraction::{atomic_number}"] = (
                    member_atomic_numbers.count(atomic_number)
                    / len(member_atomic_numbers)
                )
            top_origins = set(
                map(
                    str,
                    _json_list(
                        top["candidate_origins_json"],
                        label="top candidate origins",
                    ),
                )
            )
            row[f"{prefix}::top_rdkit_symmetry_same_element_coverage"] = (
                float(same_element_coverage)
                if "rdkit_symmetry" in top_origins
                else 0.0
            )
            origin_sets = [
                set(
                    map(
                        str,
                        _json_list(raw, label="candidate origins"),
                    )
                )
                for raw in selected["candidate_origins_json"]
            ]
            for origin in CANDIDATE_ORIGIN_NAMES:
                row[f"{prefix}::top_origin::{origin}"] = float(
                    origin in top_origins
                )
                row[f"{prefix}::origin_fraction::{origin}"] = float(
                    np.mean([origin in values for values in origin_sets])
                )
        for left_type, right_type in combinations(SITE_TYPE_NAMES, 2):
            left = top_members_by_type[left_type]
            right = top_members_by_type[right_type]
            pair_prefix = f"pair::{left_type}::{right_type}"
            if not left or not right:
                row[f"{pair_prefix}::jaccard"] = 0.0
                row[f"{pair_prefix}::left_contained"] = 0.0
                row[f"{pair_prefix}::right_contained"] = 0.0
                row[f"{pair_prefix}::same_members"] = 0.0
                continue
            intersection = left & right
            row[f"{pair_prefix}::jaccard"] = len(intersection) / len(
                left | right
            )
            row[f"{pair_prefix}::left_contained"] = float(left <= right)
            row[f"{pair_prefix}::right_contained"] = float(right <= left)
            row[f"{pair_prefix}::same_members"] = float(left == right)
        rows.append(row)
    features = pd.DataFrame(rows).sort_values("context_id", kind="stable")
    if len(features) != len(requested_contexts):
        raise JointSiteTypeRouterError("Router feature context coverage changed")
    feature_names = sorted(
        set(features.columns) - {"context_id", "species_id", "connectivity_id"}
    )
    numeric = features[feature_names].to_numpy(dtype=float)
    if np.isinf(numeric).any():
        raise JointSiteTypeRouterError("Router features contain infinity")
    return features[["context_id", "species_id", "connectivity_id", *feature_names]]


def _feature_names(features: pd.DataFrame) -> tuple[str, ...]:
    names = tuple(
        column
        for column in features.columns
        if column not in {"context_id", "species_id", "connectivity_id"}
    )
    if not names:
        raise JointSiteTypeRouterError("Router feature matrix is empty")
    return names


def fit_type_router(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    regularization_c: float,
    class_bias: Mapping[str, float] | None = None,
    pre_router_type_max_weight: float = 0.0,
) -> dict[str, object]:
    """Fit a serializable balanced multinomial router on development contexts."""

    if not math.isfinite(regularization_c) or regularization_c <= 0:
        raise JointSiteTypeRouterError("Router regularization C is invalid")
    if (
        not math.isfinite(pre_router_type_max_weight)
        or pre_router_type_max_weight < 0
    ):
        raise JointSiteTypeRouterError("Pre-router type-maximum weight is invalid")
    _require_columns(
        labels,
        {"context_id", "connectivity_id", "true_site_type"},
        label="Router labels",
    )
    if labels["context_id"].duplicated().any():
        raise JointSiteTypeRouterError("Router labels must be one row per context")
    selected = features.merge(
        labels[["context_id", "connectivity_id", "true_site_type"]],
        on=["context_id", "connectivity_id"],
        how="inner",
        validate="one_to_one",
    )
    if len(selected) != len(features) or len(selected) != len(labels):
        raise JointSiteTypeRouterError("Router feature/label context coverage changed")
    observed_classes = set(selected["true_site_type"].astype(str))
    if observed_classes != set(TYPE_ROUTER_CLASSES):
        raise JointSiteTypeRouterError(
            f"Router training needs all five endpoint types: {sorted(observed_classes)}"
        )
    names = _feature_names(features)
    if any("conditional_n_seed_std" in name for name in names):
        raise JointSiteTypeRouterError(
            "Conditional-N seed disagreement has no common router lineage"
        )
    raw = selected[list(names)].to_numpy(dtype=np.float64)
    median = np.nanmedian(raw, axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    imputed = np.where(np.isfinite(raw), raw, median[None, :])
    mean = imputed.mean(axis=0)
    scale = imputed.std(axis=0, ddof=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    standardized = (imputed - mean[None, :]) / scale[None, :]
    classifier = LogisticRegression(
        C=float(regularization_c),
        class_weight="balanced",
        max_iter=3000,
        random_state=20260810,
    )
    classifier.fit(standardized, selected["true_site_type"].astype(str))
    if tuple(map(str, classifier.classes_)) != tuple(sorted(TYPE_ROUTER_CLASSES)):
        raise JointSiteTypeRouterError("Router class ordering changed")
    requested_bias = class_bias or {}
    unknown_bias = sorted(set(requested_bias) - set(TYPE_ROUTER_CLASSES))
    if unknown_bias:
        raise JointSiteTypeRouterError(f"Unknown router class biases: {unknown_bias}")
    bias = {
        site_type: float(requested_bias.get(site_type, 0.0))
        for site_type in classifier.classes_
    }
    if not all(math.isfinite(value) for value in bias.values()):
        raise JointSiteTypeRouterError("Router class bias is non-finite")
    fit_context_ids = tuple(sorted(selected["context_id"].astype(str)))
    fit_connectivities = tuple(sorted(set(selected["connectivity_id"].astype(str))))
    bundle: dict[str, object] = {
        "schema_version": TYPE_ROUTER_SCHEMA_VERSION,
        "status": "fitted",
        "feature_contract": (
            "morgan1024_context_physical_typed_candidate_set_structural.v3"
        ),
        "morgan_radius": MORGAN_RADIUS,
        "morgan_bits": MORGAN_BITS,
        "candidate_origin_names": list(CANDIDATE_ORIGIN_NAMES),
        "site_type_classes": list(map(str, classifier.classes_)),
        "feature_names": list(names),
        "feature_median": median.tolist(),
        "feature_minimum": imputed.min(axis=0).tolist(),
        "feature_maximum": imputed.max(axis=0).tolist(),
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "feature_range_clipping_contract": FEATURE_RANGE_CLIPPING_CONTRACT,
        "conditional_n_seed_std_used": False,
        "coefficient": classifier.coef_.astype(float).tolist(),
        "intercept": classifier.intercept_.astype(float).tolist(),
        "class_bias": bias,
        "pre_router_type_max_weight": float(pre_router_type_max_weight),
        "regularization_c": float(regularization_c),
        "class_weight": "balanced",
        "fit_context_count": len(fit_context_ids),
        "fit_connectivity_count": len(fit_connectivities),
        "fit_context_ids_sha256": hashlib.sha256(
            "\0".join(fit_context_ids).encode("utf-8")
        ).hexdigest(),
        "fit_connectivity_ids_sha256": hashlib.sha256(
            "\0".join(fit_connectivities).encode("utf-8")
        ).hexdigest(),
        "target_semantics": "exact_endpoint_type_context_label",
        "candidate_unknown_used_as_binary_negative": False,
        "candidate_softmax_used": False,
        "canonical_composition": (
            "type_router_plus_weighted_pre_router_type_max_plus_"
            "within_type_relative.v1"
        ),
    }
    bundle["bundle_sha256"] = _canonical_payload_sha256(bundle)
    return bundle


def _standardized_type_router_features(
    bundle: Mapping[str, object],
    features: pd.DataFrame,
) -> tuple[
    tuple[str, ...],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, object],
]:
    """Validate and range-bound router features before linear scoring."""

    if (
        bundle.get("schema_version") != TYPE_ROUTER_SCHEMA_VERSION
        or bundle.get("status") != "fitted"
    ):
        raise JointSiteTypeRouterError("Unsupported type-router bundle")
    payload = dict(bundle)
    observed_sha = str(payload.pop("bundle_sha256", ""))
    if observed_sha != _canonical_payload_sha256(payload):
        raise JointSiteTypeRouterError("Type-router bundle digest changed")
    names = tuple(map(str, bundle["feature_names"]))
    if _feature_names(features) != names:
        raise JointSiteTypeRouterError("Type-router feature contract changed")
    if any("conditional_n_seed_std" in name for name in names) or bundle.get(
        "conditional_n_seed_std_used"
    ) is not False:
        raise JointSiteTypeRouterError(
            "Conditional-N seed disagreement has no common router lineage"
        )
    if (
        bundle.get("feature_range_clipping_contract")
        != FEATURE_RANGE_CLIPPING_CONTRACT
    ):
        raise JointSiteTypeRouterError("Type-router range contract changed")
    classes = tuple(map(str, bundle["site_type_classes"]))
    if set(classes) != set(TYPE_ROUTER_CLASSES):
        raise JointSiteTypeRouterError("Type-router class vocabulary changed")
    raw = features[list(names)].to_numpy(dtype=np.float64)
    median = np.asarray(bundle["feature_median"], dtype=np.float64)
    minimum = np.asarray(bundle["feature_minimum"], dtype=np.float64)
    maximum = np.asarray(bundle["feature_maximum"], dtype=np.float64)
    mean = np.asarray(bundle["feature_mean"], dtype=np.float64)
    scale = np.asarray(bundle["feature_scale"], dtype=np.float64)
    coefficient = np.asarray(bundle["coefficient"], dtype=np.float64)
    intercept = np.asarray(bundle["intercept"], dtype=np.float64)
    width = len(names)
    if (
        median.shape != (width,)
        or minimum.shape != (width,)
        or maximum.shape != (width,)
        or mean.shape != (width,)
        or scale.shape != (width,)
        or coefficient.shape != (len(classes), width)
        or intercept.shape != (len(classes),)
        or not np.isfinite(median).all()
        or not np.isfinite(minimum).all()
        or not np.isfinite(maximum).all()
        or (minimum > maximum).any()
        or not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
        or (scale <= 0).any()
        or not np.isfinite(coefficient).all()
        or not np.isfinite(intercept).all()
    ):
        raise JointSiteTypeRouterError("Type-router parameter shape changed")
    nonfinite = ~np.isfinite(raw)
    imputed = np.where(np.isfinite(raw), raw, median[None, :])
    below = imputed < minimum[None, :]
    above = imputed > maximum[None, :]
    clipped_mask = below | above
    clipped = np.clip(imputed, minimum[None, :], maximum[None, :])
    standardized_before_clipping = (imputed - mean[None, :]) / scale[None, :]
    standardized = (clipped - mean[None, :]) / scale[None, :]
    if not np.isfinite(standardized).all():
        raise JointSiteTypeRouterError("Type-router standardized features are invalid")
    audit = {
        "schema_version": "nucpred.mayr-joint-site-type-router-transport-audit.v1",
        "status": "pass",
        "context_count": int(len(features)),
        "feature_count": int(width),
        "feature_range_clipping_contract": FEATURE_RANGE_CLIPPING_CONTRACT,
        "conditional_n_seed_std_feature_count": 0,
        "nonfinite_imputation_count": int(nonfinite.sum()),
        "below_training_minimum_count": int(below.sum()),
        "above_training_maximum_count": int(above.sum()),
        "clipped_cell_count": int(clipped_mask.sum()),
        "clipped_context_count": int(clipped_mask.any(axis=1).sum()),
        "maximum_absolute_standardized_before_clipping": float(
            np.abs(standardized_before_clipping).max(initial=0.0)
        ),
        "maximum_absolute_standardized_after_clipping": float(
            np.abs(standardized).max(initial=0.0)
        ),
    }
    return classes, standardized, coefficient, intercept, audit


def type_router_feature_transport_audit(
    bundle: Mapping[str, object],
    features: pd.DataFrame,
) -> dict[str, object]:
    """Report label-blind inner-to-score feature transport diagnostics."""

    return _standardized_type_router_features(bundle, features)[-1]


def predict_type_router_logits(
    bundle: Mapping[str, object],
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Score all five endpoint types from a frozen JSON-compatible bundle."""

    classes, standardized, coefficient, intercept, _ = (
        _standardized_type_router_features(bundle, features)
    )
    logits = standardized @ coefficient.T + intercept[None, :]
    bias = bundle["class_bias"]
    logits += np.asarray([float(bias[name]) for name in classes])[None, :]
    pre_router_weight = float(bundle.get("pre_router_type_max_weight", math.nan))
    if not math.isfinite(pre_router_weight) or pre_router_weight < 0:
        raise JointSiteTypeRouterError("Type-router prior weight changed")
    if pre_router_weight:
        relative_columns = [
            f"type::{site_type}::relative_global_max" for site_type in classes
        ]
        missing_relative = sorted(set(relative_columns) - set(features.columns))
        if missing_relative:
            raise JointSiteTypeRouterError(
                f"Type-router prior features are missing: {missing_relative}"
            )
        logits += pre_router_weight * features[relative_columns].to_numpy(
            dtype=np.float64
        )
    rows: list[dict[str, object]] = []
    for identity, values in zip(
        features[["context_id", "species_id", "connectivity_id"]].itertuples(
            index=False
        ),
        logits,
        strict=True,
    ):
        for site_type, value in zip(classes, values, strict=True):
            rows.append(
                {
                    "context_id": str(identity.context_id),
                    "species_id": str(identity.species_id),
                    "connectivity_id": str(identity.connectivity_id),
                    "site_type": site_type,
                    "type_router_logit": float(value),
                }
            )
    return pd.DataFrame(rows)


def apply_type_router(
    candidates: pd.DataFrame,
    type_logits: pd.DataFrame,
) -> pd.DataFrame:
    """Compose one direct canonical logit while preserving within-type order."""

    _require_columns(
        candidates,
        {"query_id", "context_id", "candidate_site_id", "site_type", "canonical_logit"},
        label="Pre-router candidate scores",
    )
    _require_columns(
        type_logits,
        {"context_id", "site_type", "type_router_logit"},
        label="Type-router logits",
    )
    if type_logits.duplicated(["context_id", "site_type"]).any():
        raise JointSiteTypeRouterError("Type-router context/type identity is duplicated")
    result = candidates.copy()
    result["pre_router_canonical_logit"] = result["canonical_logit"].astype(float)
    type_maximum = result.groupby(
        ["context_id", "site_type"]
    )["pre_router_canonical_logit"].transform("max")
    result["within_type_relative_logit"] = (
        result["pre_router_canonical_logit"] - type_maximum
    )
    result = result.merge(
        type_logits[["context_id", "site_type", "type_router_logit"]],
        on=["context_id", "site_type"],
        how="left",
        validate="many_to_one",
    )
    if result["type_router_logit"].isna().any():
        raise JointSiteTypeRouterError("Candidate lacks a frozen type-router logit")
    result["canonical_logit"] = (
        result["type_router_logit"] + result["within_type_relative_logit"]
    )
    if not np.isfinite(result["canonical_logit"].to_numpy(dtype=float)).all():
        raise JointSiteTypeRouterError("Composite canonical logits are non-finite")
    result = result.sort_values(
        ["context_id", "canonical_logit", "query_id"],
        ascending=[True, False, True],
        kind="stable",
    ).reset_index(drop=True)
    result["candidate_rank"] = result.groupby("context_id").cumcount() + 1
    return result


def _single_target_labels(targets: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        targets,
        {
            "context_id",
            "connectivity_id",
            "target_id",
            "site_object_id",
            "site_type",
            "N_mean",
        },
        label="Type-router targets",
    )
    counts = targets.groupby("context_id")["target_id"].transform("size")
    selected = targets.loc[
        counts.eq(1),
        [
            "context_id",
            "connectivity_id",
            "site_object_id",
            "site_type",
            "N_mean",
        ],
    ].copy()
    selected = selected.rename(
        columns={
            "site_object_id": "true_candidate_site_id",
            "site_type": "true_site_type",
            "N_mean": "N_true",
        }
    )
    if selected["context_id"].duplicated().any():
        raise JointSiteTypeRouterError("Single-target router labels are duplicated")
    return selected


def _router_metrics(
    candidates: pd.DataFrame,
    labels: pd.DataFrame,
    predicted_types: Mapping[str, str],
) -> tuple[dict[str, object], pd.DataFrame]:
    pre_top = candidates.sort_values(
        ["context_id", "pre_router_canonical_logit", "query_id"],
        ascending=[True, False, True],
        kind="stable",
    )
    rows: list[dict[str, object]] = []
    label_table = labels.set_index("context_id", drop=False)
    for context_id, group in pre_top.groupby("context_id", sort=True):
        if str(context_id) not in label_table.index:
            continue
        label = label_table.loc[str(context_id)]
        predicted_type = str(predicted_types[str(context_id)])
        selected = group.loc[group["site_type"].astype(str).eq(predicted_type)]
        if selected.empty:
            raise JointSiteTypeRouterError(
                "Router selected a type absent from the candidate set"
            )
        top = selected.iloc[0]
        rows.append(
            {
                "context_id": str(context_id),
                "connectivity_id": str(label["connectivity_id"]),
                "true_site_type": str(label["true_site_type"]),
                "predicted_site_type": predicted_type,
                "true_candidate_site_id": str(label["true_candidate_site_id"]),
                "predicted_candidate_site_id": str(top["candidate_site_id"]),
                "site_top1_correct": str(top["candidate_site_id"])
                == str(label["true_candidate_site_id"]),
                "type_correct": predicted_type == str(label["true_site_type"]),
                "N_true": float(label["N_true"]),
                "automatic_n_prediction": float(top["conditional_n_prediction"]),
            }
        )
    contexts = pd.DataFrame(rows)
    if len(contexts) != len(labels):
        raise JointSiteTypeRouterError("Router metric context coverage changed")
    exact = contexts["site_top1_correct"].to_numpy(dtype=bool)
    result: dict[str, object] = {
        "context_count": len(contexts),
        "connectivity_count": int(contexts["connectivity_id"].nunique()),
        "exact_top1": float(exact.mean()),
        "type_accuracy": float(contexts["type_correct"].mean()),
        "automatic_n_mae": float(
            np.mean(
                np.abs(
                    contexts["automatic_n_prediction"].to_numpy(dtype=float)
                    - contexts["N_true"].to_numpy(dtype=float)
                )
            )
        ),
        "by_type": {},
    }
    by_type: dict[str, object] = {}
    for site_type in SITE_TYPE_NAMES:
        selected = contexts.loc[
            contexts["true_site_type"].astype(str).eq(site_type)
        ]
        by_type[site_type] = {
            "context_count": len(selected),
            "exact_top1": (
                float(selected["site_top1_correct"].mean())
                if len(selected)
                else math.nan
            ),
            "type_accuracy": (
                float(selected["type_correct"].mean())
                if len(selected)
                else math.nan
            ),
        }
    result["by_type"] = by_type
    return result, contexts


def select_and_fit_type_router(
    features: pd.DataFrame,
    candidates: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    inner_fold_by_context: Mapping[str, int],
    regularization_grid: Sequence[float] = (0.005, 0.01, 0.03, 0.1),
    pre_router_type_max_weight_grid: Sequence[float] = (0.0,),
    atom_group_bias_grid: Sequence[float] = (0.0, 0.25, 0.5, 0.75),
    region_bias_grid: Sequence[float] = (0.0, 0.25, 0.5),
    weak_type_minimum: float = 0.75,
) -> tuple[dict[str, object], dict[str, object], pd.DataFrame]:
    """Cross-fit router choices on inner OOF, then refit all development rows."""

    regularization_grid = tuple(map(float, regularization_grid))
    pre_router_type_max_weight_grid = tuple(
        map(float, pre_router_type_max_weight_grid)
    )
    atom_group_bias_grid = tuple(map(float, atom_group_bias_grid))
    region_bias_grid = tuple(map(float, region_bias_grid))
    weak_type_minimum = float(weak_type_minimum)
    if (
        not regularization_grid
        or any(
            not math.isfinite(value) or value <= 0 for value in regularization_grid
        )
        or not pre_router_type_max_weight_grid
        or any(
            not math.isfinite(value) or value < 0
            for value in pre_router_type_max_weight_grid
        )
        or not atom_group_bias_grid
        or any(not math.isfinite(value) for value in atom_group_bias_grid)
        or not region_bias_grid
        or any(not math.isfinite(value) for value in region_bias_grid)
        or not math.isfinite(weak_type_minimum)
        or not 0.0 <= weak_type_minimum <= 1.0
    ):
        raise JointSiteTypeRouterError("Router selection grid is invalid")

    labels = _single_target_labels(targets)
    selected_contexts = set(labels["context_id"].astype(str))
    features = features.loc[
        features["context_id"].astype(str).isin(selected_contexts)
    ].reset_index(drop=True)
    candidates = candidates.loc[
        candidates["context_id"].astype(str).isin(selected_contexts)
    ].copy()
    if set(features["context_id"].astype(str)) != selected_contexts:
        raise JointSiteTypeRouterError("Router features omit a single-target context")
    fold_values = np.asarray(
        [int(inner_fold_by_context[str(value)]) for value in features["context_id"]],
        dtype=int,
    )
    folds = tuple(sorted(set(map(int, fold_values))))
    if folds != (0, 1, 2, 3):
        raise JointSiteTypeRouterError("Router selection requires four inner folds")
    candidates["pre_router_canonical_logit"] = candidates[
        "canonical_logit"
    ].astype(float)
    present_types = {
        str(context_id): set(group["site_type"].astype(str))
        for context_id, group in candidates.groupby("context_id", sort=True)
    }
    relative_grid = features.set_index("context_id")[
        [
            f"type::{site_type}::relative_global_max"
            for site_type in TYPE_ROUTER_CLASSES
        ]
    ].rename(
        columns={
            f"type::{site_type}::relative_global_max": site_type
            for site_type in TYPE_ROUTER_CLASSES
        }
    )
    trials: list[dict[str, object]] = []
    best_key: tuple[float, ...] | None = None
    best_trial: dict[str, object] | None = None
    best_contexts: pd.DataFrame | None = None
    for regularization_c in regularization_grid:
        crossfit_rows: list[pd.DataFrame] = []
        for fold in folds:
            train_features = features.loc[fold_values != fold].reset_index(drop=True)
            validation_features = features.loc[fold_values == fold].reset_index(
                drop=True
            )
            train_ids = set(train_features["context_id"].astype(str))
            train_labels = labels.loc[
                labels["context_id"].astype(str).isin(train_ids),
                ["context_id", "connectivity_id", "true_site_type"],
            ]
            bundle = fit_type_router(
                train_features,
                train_labels,
                regularization_c=float(regularization_c),
            )
            scored = predict_type_router_logits(bundle, validation_features)
            scored["inner_fold"] = fold
            crossfit_rows.append(scored)
        crossfit_logits = pd.concat(crossfit_rows, ignore_index=True)
        logit_grid = crossfit_logits.pivot(
            index="context_id", columns="site_type", values="type_router_logit"
        )
        logit_grid = logit_grid.reindex(columns=TYPE_ROUTER_CLASSES)
        if logit_grid.isna().any().any() or set(logit_grid.index) != selected_contexts:
            raise JointSiteTypeRouterError("Cross-fit router logit coverage changed")
        aligned_relative = relative_grid.reindex(logit_grid.index)
        if aligned_relative.isna().any().any():
            raise JointSiteTypeRouterError(
                "Pre-router type-maximum feature coverage changed"
            )
        for pre_router_type_max_weight in pre_router_type_max_weight_grid:
            for atom_group_bias in atom_group_bias_grid:
                for region_bias in region_bias_grid:
                    adjusted = logit_grid + float(
                        pre_router_type_max_weight
                    ) * aligned_relative
                    adjusted = adjusted.copy()
                    adjusted["atom_group"] += float(atom_group_bias)
                    adjusted["delocalized_region"] += float(region_bias)
                    predicted_types: dict[str, str] = {}
                    for context_id, row in adjusted.iterrows():
                        available = present_types[str(context_id)]
                        predicted_types[str(context_id)] = str(
                            row.loc[list(sorted(available))].idxmax()
                        )
                    metrics, context_predictions = _router_metrics(
                        candidates,
                        labels,
                        predicted_types,
                    )
                    by_type = metrics["by_type"]
                    atom_group_top1 = float(
                        by_type["atom_group"]["exact_top1"]
                    )
                    region_top1 = float(
                        by_type["delocalized_region"]["exact_top1"]
                    )
                    weak_pass_count = int(
                        atom_group_top1 >= weak_type_minimum
                    ) + int(region_top1 >= weak_type_minimum)
                    weak_total_shortfall = max(
                        0.0, weak_type_minimum - atom_group_top1
                    ) + max(0.0, weak_type_minimum - region_top1)
                    key = (
                        float(weak_pass_count),
                        -float(weak_total_shortfall),
                        float(metrics["exact_top1"]),
                        -float(metrics["automatic_n_mae"]),
                        min(atom_group_top1, region_top1),
                        -abs(float(atom_group_bias)) - abs(float(region_bias)),
                        -float(pre_router_type_max_weight),
                        -float(regularization_c),
                    )
                    trial = {
                        "regularization_c": float(regularization_c),
                        "pre_router_type_max_weight": float(
                            pre_router_type_max_weight
                        ),
                        "class_bias": {
                            "atom_group": float(atom_group_bias),
                            "delocalized_region": float(region_bias),
                        },
                        "weak_type_gate_pass_count": weak_pass_count,
                        "weak_type_total_shortfall": float(
                            weak_total_shortfall
                        ),
                        "metrics": metrics,
                        "selection_key": list(key),
                    }
                    trials.append(trial)
                    if best_key is None or key > best_key:
                        best_key = key
                        best_trial = trial
                        best_contexts = context_predictions
    if best_trial is None or best_contexts is None:
        raise JointSiteTypeRouterError("No type-router trial was evaluated")
    final_bundle = fit_type_router(
        features,
        labels[["context_id", "connectivity_id", "true_site_type"]],
        regularization_c=float(best_trial["regularization_c"]),
        class_bias=best_trial["class_bias"],
        pre_router_type_max_weight=float(
            best_trial["pre_router_type_max_weight"]
        ),
    )
    baseline_types = (
        candidates.sort_values(
            ["context_id", "pre_router_canonical_logit", "query_id"],
            ascending=[True, False, True],
            kind="stable",
        )
        .groupby("context_id", sort=False)
        .head(1)
        .set_index("context_id")["site_type"]
        .astype(str)
        .to_dict()
    )
    baseline_metrics, _ = _router_metrics(candidates, labels, baseline_types)
    summary: dict[str, object] = {
        "schema_version": TYPE_ROUTER_SELECTION_SCHEMA_VERSION,
        "status": "pass",
        "selection_role": "outer_development_inner_oof_only",
        "outer_test_target_rows_loaded": 0,
        "outer_test_predictions_computed": 0,
        "single_target_context_count": len(labels),
        "excluded_multi_target_context_count": int(
            targets["context_id"].nunique() - len(labels)
        ),
        "inner_folds": list(folds),
        "candidate_unknown_used_as_binary_negative": False,
        "candidate_softmax_used": False,
        "weak_type_minimum": float(weak_type_minimum),
        "baseline_metrics": baseline_metrics,
        "selected_trial": best_trial,
        "trial_count": len(trials),
        "trials": trials,
        "final_bundle_sha256": final_bundle["bundle_sha256"],
    }
    return final_bundle, summary, best_contexts
