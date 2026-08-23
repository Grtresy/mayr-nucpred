"""Candidate-set-aware structural residual for Mayr region membership.

The frozen v6 ranker is already strong at deciding whether a
``delocalized_region`` type is plausible, but nested region candidates remain
hard to order.  This module deliberately leaves the type-level maximum logit
unchanged and only reassigns the ordering *inside* the region type.  The
residual therefore cannot manufacture extra evidence for a region type.

All features are label independent: candidate membership, candidate-generator
origins, relations to the other enumerated region candidates, frozen v6
components, and frozen conditional-N outputs.  No candidate softmax is used.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier


REGION_SITE_TYPE = "delocalized_region"
REGION_FEATURE_SCHEMA = "nucpred.mayr-region-structural-features.v1"
REGION_RESIDUAL_SCHEMA = "nucpred.mayr-region-membership-residual.v1"
ELEMENT_BUCKETS = (1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53)


class RegionResidualError(RuntimeError):
    """Raised when a region-residual artifact or feature contract drifts."""


def _json_list(value: object, *, label: str) -> list[Any]:
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError as exc:
        raise RegionResidualError(f"Invalid {label} JSON") from exc
    if isinstance(parsed, np.ndarray):
        parsed = parsed.tolist()
    if not isinstance(parsed, (list, tuple)):
        raise RegionResidualError(f"{label} must be a list")
    return list(parsed)


def origin_vocabulary(frame: pd.DataFrame) -> tuple[str, ...]:
    """Freeze the sorted candidate-generator origin vocabulary."""

    if "candidate_origins_json" not in frame:
        raise RegionResidualError("Candidate origins are missing")
    values = {
        str(origin)
        for raw in frame["candidate_origins_json"]
        for origin in _json_list(raw, label="candidate origins")
    }
    if not values:
        raise RegionResidualError("Candidate origin vocabulary is empty")
    return tuple(sorted(values))


def region_feature_names(
    origin_vocabulary_values: Sequence[str],
) -> tuple[str, ...]:
    """Return the exact ordered feature contract for one residual."""

    origins = tuple(map(str, origin_vocabulary_values))
    return (
        "log1p_member_atom_count",
        "log1p_internal_bond_count",
        "internal_bond_per_member",
        "cycle_like_internal_bonds",
        "tree_like_internal_bonds",
        "unique_element_fraction",
        "single_element_membership",
        "log1p_origin_count",
        "other_element_fraction",
        *(f"element_fraction_Z{atomic_number}" for atomic_number in ELEMENT_BUCKETS),
        *(f"origin::{origin}" for origin in origins),
        "log1p_region_candidate_count",
        "member_fraction_of_largest_region_candidate",
        "strict_region_subset_count",
        "strict_region_superset_count",
        "overlapping_region_candidate_count",
        "maximum_region_jaccard",
        "duplicate_region_membership_count",
        "same_size_region_candidate_count",
        "base_membership_logit",
        "base_compatibility_logit",
        "conditional_N_mean",
        "conditional_N_std",
    )


def region_feature_matrix(
    frame: pd.DataFrame,
    *,
    membership_logits: Sequence[float],
    compatibility_logits: Sequence[float],
    conditional_n_mean: Sequence[float],
    conditional_n_std: Sequence[float],
    origin_vocabulary_values: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Build features for every enumerated region candidate.

    Returned positions refer to row positions in ``frame``.  Candidate-set
    relations compare only region candidates in the same context, making the
    train, validation, test, and runtime contracts identical.
    """

    required = {
        "context_id",
        "candidate_site_id",
        "site_type",
        "member_atom_indices_json",
        "member_bond_pairs_json",
        "member_atomic_numbers_json",
        "candidate_origins_json",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RegionResidualError(f"Region feature columns are missing: {missing}")
    if not frame.index.equals(pd.RangeIndex(len(frame))):
        raise RegionResidualError("Region feature frame must use a dense index")
    arrays = [
        np.asarray(values, dtype=float)
        for values in (
            membership_logits,
            compatibility_logits,
            conditional_n_mean,
            conditional_n_std,
        )
    ]
    if any(values.shape != (len(frame),) for values in arrays):
        raise RegionResidualError("Region model components do not align")
    if any(not np.isfinite(values).all() for values in arrays):
        raise RegionResidualError("Region model components are non-finite")

    origins = tuple(map(str, origin_vocabulary_values))
    if not origins or tuple(sorted(set(origins))) != origins:
        raise RegionResidualError("Region origin vocabulary is not canonical")
    origin_set = set(origins)
    positions = np.flatnonzero(
        frame["site_type"].astype(str).to_numpy() == REGION_SITE_TYPE
    )
    if not len(positions):
        raise RegionResidualError("Candidate inventory has no region candidates")
    region = frame.iloc[positions].copy().reset_index(drop=True)
    memberships: list[frozenset[int]] = []
    bonds: list[tuple[tuple[int, int], ...]] = []
    atomic_numbers: list[tuple[int, ...]] = []
    origin_sets: list[frozenset[str]] = []
    for row in region.itertuples(index=False):
        members = frozenset(
            int(value)
            for value in _json_list(
                row.member_atom_indices_json,
                label="member atom indices",
            )
        )
        if not members:
            raise RegionResidualError("Region candidate membership is empty")
        member_bonds = tuple(
            tuple(map(int, pair))
            for pair in _json_list(
                row.member_bond_pairs_json,
                label="member bond pairs",
            )
        )
        if any(len(pair) != 2 for pair in member_bonds):
            raise RegionResidualError("Region member bond pair is invalid")
        numbers = tuple(
            int(value)
            for value in _json_list(
                row.member_atomic_numbers_json,
                label="member atomic numbers",
            )
        )
        if len(numbers) != len(members):
            raise RegionResidualError("Region atomic numbers do not match membership")
        candidate_origins = frozenset(
            str(value)
            for value in _json_list(
                row.candidate_origins_json,
                label="candidate origins",
            )
        )
        unknown_origins = sorted(candidate_origins - origin_set)
        if unknown_origins:
            raise RegionResidualError(
                f"Unseen region candidate origins: {unknown_origins}"
            )
        memberships.append(members)
        bonds.append(member_bonds)
        atomic_numbers.append(numbers)
        origin_sets.append(candidate_origins)

    relation = np.zeros((len(region), 8), dtype=np.float64)
    for _, group in region.groupby("context_id", sort=True):
        group_positions = group.index.to_numpy(dtype=int)
        largest = max(len(memberships[index]) for index in group_positions)
        for index in group_positions:
            others = [value for value in group_positions if value != index]
            overlaps = [
                len(memberships[index] & memberships[other])
                / len(memberships[index] | memberships[other])
                for other in others
                if memberships[index] & memberships[other]
            ]
            relation[index] = (
                math.log1p(len(group_positions)),
                len(memberships[index]) / largest,
                sum(memberships[other] < memberships[index] for other in others),
                sum(memberships[index] < memberships[other] for other in others),
                sum(bool(memberships[index] & memberships[other]) for other in others),
                max(overlaps, default=0.0),
                sum(memberships[other] == memberships[index] for other in others),
                sum(
                    len(memberships[other]) == len(memberships[index])
                    for other in others
                ),
            )

    rows: list[list[float]] = []
    for index, source_position in enumerate(positions):
        numbers = atomic_numbers[index]
        member_count = len(memberships[index])
        bond_count = len(bonds[index])
        denominator = max(len(numbers), 1)
        element_fractions = [
            numbers.count(atomic_number) / denominator
            for atomic_number in ELEMENT_BUCKETS
        ]
        other_fraction = (
            sum(atomic_number not in ELEMENT_BUCKETS for atomic_number in numbers)
            / denominator
        )
        rows.append(
            [
                math.log1p(member_count),
                math.log1p(bond_count),
                bond_count / member_count,
                float(bond_count >= member_count),
                float(bond_count == max(member_count - 1, 0)),
                len(set(numbers)) / denominator,
                float(len(set(numbers)) == 1),
                math.log1p(len(origin_sets[index])),
                other_fraction,
                *element_fractions,
                *(float(origin in origin_sets[index]) for origin in origins),
                *relation[index].tolist(),
                float(arrays[0][source_position]),
                float(arrays[1][source_position]),
                float(arrays[2][source_position]),
                float(arrays[3][source_position]),
            ]
        )
    features = np.asarray(rows, dtype=np.float32)
    names = region_feature_names(origins)
    if features.shape != (len(positions), len(names)):
        raise RegionResidualError("Region feature width changed")
    if not np.isfinite(features).all():
        raise RegionResidualError("Region features are non-finite")
    return positions.astype(np.int64), features, names


def context_balanced_exact_weights(frame: pd.DataFrame) -> np.ndarray:
    """Assign equal context mass and balanced exact/non-target mass."""

    if not {"context_id", "exact_label"} <= set(frame.columns) or frame.empty:
        raise RegionResidualError("Region training frame is invalid")
    if not frame.index.equals(pd.RangeIndex(len(frame))):
        raise RegionResidualError("Region training frame must use a dense index")
    weights = np.zeros(len(frame), dtype=np.float64)
    context_count = frame["context_id"].astype(str).nunique()
    for context_id, group in frame.groupby("context_id", sort=True):
        positions = group.index.to_numpy(dtype=int)
        labels = group["exact_label"].to_numpy(dtype=int)
        positives = positions[labels == 1]
        negatives = positions[labels == 0]
        if not len(positives):
            raise RegionResidualError(
                f"Region training context lacks an exact target: {context_id}"
            )
        weights[positives] = 0.5 / context_count / len(positives)
        if len(negatives):
            weights[negatives] = 0.5 / context_count / len(negatives)
        else:
            weights[positives] = 1.0 / context_count / len(positives)
    if not math.isclose(float(weights.sum()), 1.0, abs_tol=1e-8):
        raise RegionResidualError("Region training weights do not sum to one")
    return weights


def fit_region_residual_ensemble(
    features: np.ndarray,
    labels: Sequence[int],
    *,
    sample_weights: Sequence[float],
    minimum_samples_leaf: int,
    estimator_count_per_seed: int,
    maximum_features: float,
    seeds: Sequence[int],
    feature_names: Sequence[str],
    origin_vocabulary_values: Sequence[str],
) -> dict[str, object]:
    """Fit a deterministic multi-seed ExtraTrees probability ensemble."""

    matrix = np.asarray(features, dtype=np.float32)
    target = np.asarray(labels, dtype=int)
    weights = np.asarray(sample_weights, dtype=float)
    seed_values = tuple(map(int, seeds))
    if (
        matrix.ndim != 2
        or not len(matrix)
        or target.shape != (len(matrix),)
        or weights.shape != (len(matrix),)
        or set(target) != {0, 1}
        or not np.isfinite(matrix).all()
        or not np.isfinite(weights).all()
        or bool((weights <= 0).any())
    ):
        raise RegionResidualError("Region residual training inputs are invalid")
    if minimum_samples_leaf <= 0 or estimator_count_per_seed <= 0:
        raise RegionResidualError("Region residual tree settings are invalid")
    if not 0.0 < maximum_features <= 1.0 or not seed_values:
        raise RegionResidualError("Region residual ensemble settings are invalid")
    names = tuple(map(str, feature_names))
    origins = tuple(map(str, origin_vocabulary_values))
    if names != region_feature_names(origins) or len(names) != matrix.shape[1]:
        raise RegionResidualError("Region residual feature contract changed")

    estimators: list[ExtraTreesClassifier] = []
    for seed in seed_values:
        estimator = ExtraTreesClassifier(
            n_estimators=int(estimator_count_per_seed),
            min_samples_leaf=int(minimum_samples_leaf),
            max_features=float(maximum_features),
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )
        estimator.fit(matrix, target, sample_weight=weights)
        estimators.append(estimator)
    return {
        "schema_version": REGION_RESIDUAL_SCHEMA,
        "feature_schema_version": REGION_FEATURE_SCHEMA,
        "target_site_type": REGION_SITE_TYPE,
        "feature_names": list(names),
        "origin_vocabulary": list(origins),
        "minimum_samples_leaf": int(minimum_samples_leaf),
        "estimator_count_per_seed": int(estimator_count_per_seed),
        "maximum_features": float(maximum_features),
        "seeds": list(seed_values),
        "estimators": estimators,
        "candidate_softmax_used": False,
        "type_level_maximum_preserved": True,
    }


def score_region_residual(
    bundle: Mapping[str, object],
    features: np.ndarray,
    *,
    expected_feature_names: Sequence[str],
) -> np.ndarray:
    """Average positive probabilities from a verified residual ensemble."""

    if (
        bundle.get("schema_version") != REGION_RESIDUAL_SCHEMA
        or bundle.get("feature_schema_version") != REGION_FEATURE_SCHEMA
        or bundle.get("target_site_type") != REGION_SITE_TYPE
        or bundle.get("candidate_softmax_used") is not False
        or bundle.get("type_level_maximum_preserved") is not True
    ):
        raise RegionResidualError("Region residual semantic contract changed")
    names = tuple(map(str, bundle.get("feature_names", ())))
    if names != tuple(map(str, expected_feature_names)):
        raise RegionResidualError("Region residual feature names changed")
    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != len(names):
        raise RegionResidualError("Region residual feature matrix changed")
    estimators = bundle.get("estimators")
    seeds = tuple(map(int, bundle.get("seeds", ())))
    if not isinstance(estimators, list) or len(estimators) != len(seeds) or not seeds:
        raise RegionResidualError("Region residual estimator ensemble changed")
    probabilities = []
    for estimator in estimators:
        if not isinstance(estimator, ExtraTreesClassifier):
            raise RegionResidualError("Region residual estimator type changed")
        values = estimator.predict_proba(matrix)
        if tuple(map(int, estimator.classes_)) != (0, 1) or values.shape != (
            len(matrix),
            2,
        ):
            raise RegionResidualError("Region residual class contract changed")
        probabilities.append(values[:, 1])
    result = np.mean(np.stack(probabilities, axis=1), axis=1)
    if (
        not np.isfinite(result).all()
        or bool((result < 0).any())
        or bool((result > 1).any())
    ):
        raise RegionResidualError("Region residual probabilities are invalid")
    return result


def apply_region_residual(
    frame: pd.DataFrame,
    *,
    base_logits: Sequence[float],
    region_positions: Sequence[int],
    residual_probabilities: Sequence[float],
    residual_weight: float,
    maximum_base_margin: float | None = None,
    top_k: int | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Rerank region candidates while preserving each context's type maximum."""

    logits = np.asarray(base_logits, dtype=float)
    positions = np.asarray(region_positions, dtype=int)
    probabilities = np.asarray(residual_probabilities, dtype=float)
    if (
        logits.shape != (len(frame),)
        or positions.ndim != 1
        or probabilities.shape != positions.shape
        or not np.isfinite(logits).all()
        or not np.isfinite(probabilities).all()
        or not math.isfinite(float(residual_weight))
        or residual_weight < 0
    ):
        raise RegionResidualError("Region reranking inputs are invalid")
    if maximum_base_margin is not None and (
        not math.isfinite(float(maximum_base_margin)) or maximum_base_margin < 0
    ):
        raise RegionResidualError("Region margin gate is invalid")
    if top_k is not None and top_k <= 0:
        raise RegionResidualError("Region top-k gate is invalid")
    if (
        not len(positions)
        or bool((positions < 0).any())
        or bool((positions >= len(frame)).any())
    ):
        raise RegionResidualError("Region positions are invalid")
    if not frame.iloc[positions]["site_type"].astype(str).eq(REGION_SITE_TYPE).all():
        raise RegionResidualError("Region positions include a non-region candidate")

    output = logits.copy()
    probability_by_position = {
        int(position): float(probability)
        for position, probability in zip(positions, probabilities, strict=True)
    }
    eligible_context_count = 0
    changed_context_count = 0
    unchanged_constant_probability_count = 0
    for _, group in frame.iloc[positions].groupby("context_id", sort=True):
        group_positions = group.index.to_numpy(dtype=int)
        ordered = sorted(
            group_positions,
            key=lambda position: (
                -float(logits[position]),
                str(frame.iloc[position]["candidate_site_id"]),
            ),
        )
        base_margin = (
            float(logits[ordered[0]] - logits[ordered[1]])
            if len(ordered) > 1
            else float("inf")
        )
        if maximum_base_margin is not None and base_margin > maximum_base_margin:
            continue
        pool = ordered[: int(top_k)] if top_k is not None else ordered
        values = np.asarray(
            [probability_by_position[position] for position in pool],
            dtype=float,
        )
        scale = float(values.std())
        eligible_context_count += 1
        if scale <= 1e-8 or len(pool) == 1:
            unchanged_constant_probability_count += 1
            continue
        selected = pool[int(np.argmax(values))]
        old_top = ordered[0]
        anchor = float(logits[old_top])
        normalized = (values - float(values.max())) / scale
        output[np.asarray(pool, dtype=int)] = (
            anchor + float(residual_weight) * normalized
        )
        changed_context_count += int(selected != old_top)
    audit = {
        "schema_version": "nucpred.mayr-region-residual-application-audit.v1",
        "region_candidate_count": int(len(positions)),
        "region_context_count": int(
            frame.iloc[positions]["context_id"].astype(str).nunique()
        ),
        "eligible_context_count": eligible_context_count,
        "changed_top_region_candidate_count": changed_context_count,
        "constant_probability_context_count": unchanged_constant_probability_count,
        "residual_weight": float(residual_weight),
        "maximum_base_margin": (
            float(maximum_base_margin) if maximum_base_margin is not None else None
        ),
        "top_k": int(top_k) if top_k is not None else None,
        "type_level_maximum_preserved": True,
        "candidate_softmax_used": False,
    }
    # Mathematical invariant: the maximum region logit per context is exactly
    # the frozen base maximum, including gated and constant-score contexts.
    for _, group in frame.iloc[positions].groupby("context_id", sort=True):
        group_positions = group.index.to_numpy(dtype=int)
        if not math.isclose(
            float(output[group_positions].max()),
            float(logits[group_positions].max()),
            abs_tol=1e-6,
        ):
            raise RegionResidualError("Region residual changed type-level evidence")
    return output, audit
