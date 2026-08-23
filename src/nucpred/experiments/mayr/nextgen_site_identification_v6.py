"""Structured v6 development for automatic Mayr site identification.

The v5 workflow remains the phase-separation and evaluation authority.  This
module replaces only development with complete-candidate validation, canonical
exact-site supervision, type routing, and type-conditioned membership ranking.
It then delegates label-blind test prediction, sealed evaluation, and registry
publication to the shared workflow.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
import gc
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
import torch
from torch.nn import functional as F

from nucpred.core.files import atomic_write_json, sha256_file
from nucpred.experiments.mayr import nextgen_site_identification as base
from nucpred.project import get_project_layout
from nucpred.training.mayr_site_confidence import tensor_mapping_sha256
from nucpred.training.mayr_site_inference_assets import (
    ranker_from_checkpoint,
    score_ranker_from_source_features,
)
from nucpred.training.mayr_site_ranker import (
    RANKER_SITE_TYPES,
    fit_type_aware_platt,
    site_type_indices,
)
from nucpred.training.mayr_site_structured_ranker import (
    FROZEN_V5_BASELINE,
    FULLSPACE_FLAT_EXACT,
    HIERARCHICAL_ONTOLOGY,
    STRUCTURED_RANKER_ARMS,
    StructuredRankerFitResult,
    StructuredSiteRanker,
    context_type_targets,
    endpoint_pairwise_logistic_loss,
    fit_feature_normalizer,
    reduce_frozen_ensemble_features,
    select_margin_threshold,
)


ROOT = get_project_layout().root
DEFAULT_CONFIG = ROOT / "configs/mayr_nextgen_site_identification_v6.toml"
DEVELOPMENT_SCHEMA = "nucpred.mayr-site-identification-structured-development.v2"


def _member_set(value: object) -> frozenset[int]:
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, list) or not parsed:
        raise base.SiteIdentificationError("Candidate membership is invalid")
    return frozenset(int(item) for item in parsed)


def _context_uniform_candidate_weights(frame: pd.DataFrame) -> torch.Tensor:
    """Give each context equal mass and each enumerated candidate equal mass."""

    if frame.empty or frame["context_id"].astype(str).duplicated().all():
        if frame.empty:
            raise base.SiteIdentificationError("Cannot weight an empty candidate frame")
    counts = frame.groupby("context_id")["candidate_site_id"].transform("count")
    context_count = int(frame["context_id"].nunique())
    values = 1.0 / counts.to_numpy(dtype=float) / context_count
    if not math.isclose(float(values.sum()), 1.0, abs_tol=1e-8):
        raise base.SiteIdentificationError("Context-uniform weights do not sum to one")
    return torch.tensor(values, dtype=torch.float32)


def _context_balanced_exact_weights(frame: pd.DataFrame) -> torch.Tensor:
    """Balance exact/non-target mass inside each training context."""

    values = np.zeros(len(frame), dtype=np.float64)
    contexts = sorted(set(frame["context_id"].astype(str)))
    eligible_contexts: list[str] = []
    context_positions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for context_id in contexts:
        positions = np.flatnonzero(
            frame["context_id"].astype(str).to_numpy() == context_id
        )
        labels = frame.iloc[positions]["exact_label"].to_numpy(dtype=int)
        positives = positions[labels == 1]
        negatives = positions[labels == 0]
        if not len(positives):
            raise base.SiteIdentificationError(
                f"Training context lacks an exact target: {context_id}"
            )
        if not len(negatives):
            # A one-candidate context (for example [F-]) is already solved by
            # candidate generation.  It remains router/auxiliary supervision
            # but contributes no artificial exact-vs-nontarget loss.
            continue
        eligible_contexts.append(context_id)
        context_positions[context_id] = (positives, negatives)
    if not eligible_contexts:
        raise base.SiteIdentificationError("No context supports exact ranking loss")
    for context_id in eligible_contexts:
        positives, negatives = context_positions[context_id]
        values[positives] = 0.5 / len(positives) / len(eligible_contexts)
        values[negatives] = 0.5 / len(negatives) / len(eligible_contexts)
    if not math.isclose(float(values.sum()), 1.0, abs_tol=1e-8):
        raise base.SiteIdentificationError("Exact-site weights do not sum to one")
    return torch.tensor(values, dtype=torch.float32)


def _balanced_binary_cell_weights(
    labels: np.ndarray,
    site_types: Sequence[str],
) -> torch.Tensor:
    """Equalize auxiliary mass across observed site-type and label cells."""

    labels = np.asarray(labels, dtype=int)
    types = np.asarray(site_types, dtype=str)
    if not len(labels) or len(labels) != len(types) or not set(labels) <= {0, 1}:
        raise base.SiteIdentificationError("Auxiliary labels are invalid")
    cells = sorted(set(zip(types, labels, strict=True)))
    values = np.zeros(len(labels), dtype=np.float64)
    for cell in cells:
        selected = (types == cell[0]) & (labels == cell[1])
        values[selected] = 1.0 / len(cells) / int(selected.sum())
    return torch.tensor(values, dtype=torch.float32)


def _balanced_router_weights(targets: torch.Tensor) -> torch.Tensor:
    """Equal mass for each site-type by present/absent router label."""

    if targets.ndim != 2 or targets.shape[1] != len(RANKER_SITE_TYPES):
        raise base.SiteIdentificationError("Router target shape changed")
    weights = torch.zeros_like(targets)
    for type_index in range(targets.shape[1]):
        for label in (0.0, 1.0):
            selected = targets[:, type_index].eq(label)
            if bool(selected.any()):
                weights[selected, type_index] = (
                    1.0 / targets.shape[1] / 2.0 / float(selected.sum())
                )
    weights = weights / weights.sum()
    return weights


def _exact_labels(
    frame: pd.DataFrame,
    targets: pd.DataFrame,
) -> np.ndarray:
    exact_by_context = (
        targets.groupby("context_id")["site_object_id"]
        .agg(lambda values: set(map(str, values)))
        .to_dict()
    )
    return np.asarray(
        [
            str(candidate_id) in exact_by_context[str(context_id)]
            for context_id, candidate_id in zip(
                frame["context_id"],
                frame["candidate_site_id"],
                strict=True,
            )
        ],
        dtype=bool,
    )


def _reviewed_context_labels(
    reviewed: pd.DataFrame,
) -> tuple[dict[tuple[str, str], int], int]:
    grouped = reviewed.groupby(
        ["context_id", "candidate_site_id"],
        sort=True,
    )["validity_label"]
    conflicts = grouped.nunique().gt(1)
    values = {
        (str(context_id), str(candidate_id)): int(labels.iloc[0])
        for (context_id, candidate_id), labels in grouped
        if int(labels.nunique()) == 1
    }
    return values, int(conflicts.sum())


def _load_baseline_checkpoint(
    config: Mapping[str, Any],
    *,
    split_seed: int,
) -> tuple[Mapping[str, Any], Path]:
    settings = config["ranker"]
    root = base._repo_path(
        settings["baseline_development_directory"],
        label="v5 baseline development directory",
    )
    manifest = root / "run_manifest.json"
    base._verify_sha(
        manifest,
        settings["baseline_development_manifest_sha256"],
        label="v5 baseline development manifest",
    )
    path = root / f"split-{split_seed}" / "ranker_checkpoint.pt"
    if not path.is_file():
        raise base.SiteIdentificationError(f"Missing v5 baseline checkpoint: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise base.SiteIdentificationError("v5 baseline checkpoint is invalid")
    expected = {
        "schema_version": "nucpred.mayr-site-ranker-checkpoint.v1",
        "campaign_id": settings["baseline_campaign_id"],
        "split_seed": split_seed,
        "phase": "development_frozen",
        "test_labels_read": False,
        "test_predictions_computed": False,
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise base.SiteIdentificationError(
                f"v5 baseline checkpoint boundary changed: {key}"
            )
    # Construction also verifies the state hash.
    ranker_from_checkpoint(checkpoint)
    return checkpoint, path


def _mark_training_selection(
    train: pd.DataFrame,
    *,
    train_targets: pd.DataFrame,
    reviewed_labels: Mapping[tuple[str, str], int],
    settings: Mapping[str, Any],
) -> pd.DataFrame:
    """Mine exact, true-type, cross-type hard, reviewed, and nested rows."""

    selected_frames: list[pd.DataFrame] = []
    target_rows = {
        str(context_id): group.copy()
        for context_id, group in train_targets.groupby("context_id", sort=True)
    }
    for context_id, group in train.groupby("context_id", sort=True):
        context_id = str(context_id)
        group = group.copy()
        truth = target_rows[context_id]
        exact_ids = set(truth["site_object_id"].astype(str))
        true_types = set(truth["site_type"].astype(str))
        exact_member_sets = [
            _member_set(value) for value in truth["member_atom_indices_json"]
        ]
        group["mine_exact"] = group["candidate_site_id"].astype(str).isin(exact_ids)
        group["mine_true_type"] = group["site_type"].astype(str).isin(true_types)
        group["mine_reviewed"] = [
            (context_id, str(candidate_id)) in reviewed_labels
            for candidate_id in group["candidate_site_id"]
        ]
        candidate_members = [
            _member_set(value) for value in group["member_atom_indices_json"]
        ]
        group["mine_overlap_or_nested"] = [
            any(
                bool(members & truth_members)
                and (
                    members <= truth_members
                    or truth_members <= members
                    or bool(members & truth_members)
                )
                for truth_members in exact_member_sets
            )
            for members in candidate_members
        ]
        group["mine_hard_global"] = False
        nonexact = group.loc[~group["mine_exact"]].sort_values(
            ["baseline_validity_logit", "candidate_site_id"],
            ascending=[False, True],
            kind="stable",
        )
        global_indices = nonexact.head(
            int(settings["hard_global_negative_count"])
        ).index
        group.loc[global_indices, "mine_hard_global"] = True
        group["mine_hard_wrong_type"] = False
        for site_type in RANKER_SITE_TYPES:
            if site_type in true_types:
                continue
            wrong_type = nonexact.loc[
                nonexact["site_type"].astype(str).eq(site_type)
            ].head(int(settings["hard_wrong_type_negative_count_per_type"]))
            group.loc[wrong_type.index, "mine_hard_wrong_type"] = True
        selection_columns = [
            "mine_exact",
            "mine_true_type",
            "mine_reviewed",
            "mine_overlap_or_nested",
            "mine_hard_global",
            "mine_hard_wrong_type",
        ]
        group["mined_for_training"] = group[selection_columns].any(axis=1)
        selected_frames.append(group.loc[group["mined_for_training"]])
    selected = pd.concat(selected_frames, ignore_index=False).sort_index()
    if not selected["mine_exact"].groupby(selected["context_id"]).any().all():
        raise base.SiteIdentificationError("Hard-negative mining lost an exact target")
    selected = selected.reset_index(drop=True)
    selected["exact_label"] = selected["mine_exact"].astype(int)
    selected["reviewed_label"] = [
        reviewed_labels.get((str(context_id), str(candidate_id)), np.nan)
        for context_id, candidate_id in zip(
            selected["context_id"],
            selected["candidate_site_id"],
            strict=True,
        )
    ]
    # Canonical exact targets are always compatible positives.  Reviewed proxy
    # positives remain auxiliary only and can never replace the exact label.
    selected["auxiliary_label"] = selected["reviewed_label"]
    selected.loc[selected["exact_label"].eq(1), "auxiliary_label"] = 1.0
    selected["compatible_proxy"] = selected["reviewed_label"].eq(1) & selected[
        "exact_label"
    ].eq(0)
    return selected


def _pair_indices(
    frame: pd.DataFrame,
    *,
    settings: Mapping[str, Any],
    ontology_weighting: bool,
    same_type_only: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, int]]:
    positive_indices: list[int] = []
    negative_indices: list[int] = []
    raw_weights: list[float] = []
    context_pair_slices: list[tuple[int, int]] = []
    cross_type_count = 0
    overlap_count = 0
    hard_count = 0
    for _, group in frame.groupby("context_id", sort=True):
        start = len(raw_weights)
        positives = group.loc[group["exact_label"].eq(1)]
        negatives = group.loc[group["exact_label"].eq(0)]
        for positive_index, positive in positives.iterrows():
            eligible = negatives
            if same_type_only:
                eligible = negatives.loc[
                    negatives["site_type"].astype(str).eq(str(positive["site_type"]))
                ]
            for negative_index, negative in eligible.iterrows():
                weight = 1.0
                cross_type = str(negative["site_type"]) != str(positive["site_type"])
                overlap = bool(negative["mine_overlap_or_nested"])
                hard = bool(
                    negative["mine_hard_global"] or negative["mine_hard_wrong_type"]
                )
                if ontology_weighting:
                    if cross_type:
                        weight *= float(settings["cross_type_pair_multiplier"])
                    if overlap:
                        weight *= float(settings["overlap_or_nested_pair_multiplier"])
                    if hard:
                        weight *= float(settings["hard_negative_pair_multiplier"])
                positive_indices.append(int(positive_index))
                negative_indices.append(int(negative_index))
                raw_weights.append(weight)
                cross_type_count += int(cross_type)
                overlap_count += int(overlap)
                hard_count += int(hard)
        stop = len(raw_weights)
        if stop > start:
            context_pair_slices.append((start, stop))
    if not raw_weights:
        raise base.SiteIdentificationError("No endpoint-relative ranking pairs exist")
    weights = np.asarray(raw_weights, dtype=np.float64)
    for start, stop in context_pair_slices:
        weights[start:stop] /= weights[start:stop].sum()
    weights /= len(context_pair_slices)
    return (
        torch.tensor(positive_indices, dtype=torch.long),
        torch.tensor(negative_indices, dtype=torch.long),
        torch.tensor(weights, dtype=torch.float32),
        {
            "pair_count": len(raw_weights),
            "context_count": len(context_pair_slices),
            "cross_type_pair_count": cross_type_count,
            "overlap_or_nested_pair_count": overlap_count,
            "hard_negative_pair_count": hard_count,
        },
    )


def _validation_metrics(
    *,
    frame: pd.DataFrame,
    logits: np.ndarray,
    membership_logits: np.ndarray,
    router_logits: np.ndarray | None,
    validation_targets: pd.DataFrame,
) -> dict[str, object]:
    labels = frame["exact_label"].to_numpy(dtype=int)
    context_ids = frame["context_id"].astype(str).to_numpy()
    retrieval = _complete_candidate_retrieval_metrics(
        labels=labels,
        scores=logits,
        context_ids=context_ids,
    )
    membership = _complete_candidate_retrieval_metrics(
        labels=labels,
        scores=membership_logits,
        context_ids=context_ids,
    )
    exact_by_context = (
        validation_targets.groupby("context_id")["site_object_id"]
        .agg(lambda values: set(map(str, values)))
        .to_dict()
    )
    true_types = (
        validation_targets.groupby("context_id")["site_type"]
        .agg(lambda values: set(map(str, values)))
        .to_dict()
    )
    top1_correct: list[bool] = []
    margins: list[float] = []
    router_correct: list[bool] = []
    for context_id, group in frame.assign(_score=logits).groupby(
        "context_id", sort=True
    ):
        ordered = group.sort_values(
            ["_score", "candidate_site_id"],
            ascending=[False, True],
            kind="stable",
        )
        top1_correct.append(
            str(ordered.iloc[0]["candidate_site_id"])
            in exact_by_context[str(context_id)]
        )
        margins.append(
            float(ordered.iloc[0]["_score"] - ordered.iloc[1]["_score"])
            if len(ordered) > 1
            else float("inf")
        )
        if router_logits is not None:
            first_index = int(group.index[0])
            predicted_type = RANKER_SITE_TYPES[
                int(np.argmax(router_logits[first_index]))
            ]
            router_correct.append(predicted_type in true_types[str(context_id)])
    weights = _context_uniform_candidate_weights(frame).numpy()
    return {
        "eligible_context_count": int(retrieval["eligible_target_count"]),
        "exact_top1_recall": float(retrieval["top1_recall"]),
        "exact_top3_recall": float(retrieval["top3_recall"]),
        "mrr": float(retrieval["mrr"]),
        "membership_only_top1_recall": float(membership["top1_recall"]),
        "exact_average_precision": float(
            average_precision_score(labels, logits, sample_weight=weights)
        ),
        "router_top1_type_recall": (
            float(np.mean(router_correct)) if router_correct else None
        ),
        "top1_correct": top1_correct,
        "top1_margin": margins,
    }


def _complete_candidate_retrieval_metrics(
    *,
    labels: np.ndarray,
    scores: np.ndarray,
    context_ids: np.ndarray,
) -> dict[str, float | int]:
    """Exact retrieval over every context, including trivial one-candidate cases."""

    reciprocal: list[float] = []
    top1 = 0
    top3 = 0
    for context_id in sorted(set(map(str, context_ids))):
        selected = context_ids.astype(str) == context_id
        group_labels = labels[selected]
        if not bool((group_labels == 1).any()):
            raise base.SiteIdentificationError(
                f"Validation context lacks an exact target: {context_id}"
            )
        group_scores = scores[selected]
        order = np.argsort(-group_scores, kind="stable")
        best_rank = int(np.flatnonzero(group_labels[order] == 1).min() + 1)
        reciprocal.append(1.0 / best_rank)
        top1 += int(best_rank <= 1)
        top3 += int(best_rank <= 3)
    count = len(reciprocal)
    return {
        "eligible_target_count": count,
        "mrr": float(np.mean(reciprocal)),
        "top1_recall": float(top1 / count),
        "top3_recall": float(top3 / count),
    }


def _selection_key(metrics: Mapping[str, object]) -> tuple[float, float, float, float]:
    return (
        float(metrics["exact_top1_recall"]),
        float(metrics["mrr"]),
        float(metrics["exact_top3_recall"]),
        float(metrics["exact_average_precision"]),
    )


def _fit_structured_arm(
    *,
    arm: str,
    train_frame: pd.DataFrame,
    train_candidate_features: torch.Tensor,
    train_context_features: torch.Tensor,
    validation_frame: pd.DataFrame,
    validation_candidate_features: torch.Tensor,
    validation_context_features: torch.Tensor,
    validation_targets: pd.DataFrame,
    router_indices: torch.Tensor,
    router_targets: torch.Tensor,
    settings: Mapping[str, Any],
    ensemble_size: int,
    source_input_dim: int,
    block_dim: int,
    seed: int,
) -> StructuredRankerFitResult:
    if arm not in STRUCTURED_RANKER_ARMS:
        raise base.SiteIdentificationError(f"Unknown structured arm: {arm}")
    torch.manual_seed(int(seed))
    candidate_mean, candidate_scale = fit_feature_normalizer(train_candidate_features)
    context_mean, context_scale = fit_feature_normalizer(train_context_features)
    model = StructuredSiteRanker(
        candidate_mean=candidate_mean,
        candidate_scale=candidate_scale,
        context_mean=context_mean,
        context_scale=context_scale,
        arm=arm,
        hidden_dim=int(settings["hidden_dim"]),
        router_hidden_dim=int(settings["router_hidden_dim"]),
        type_embedding_dim=int(settings["type_embedding_dim"]),
        router_logit_weight=float(settings["router_logit_weight"]),
        source_input_dim=source_input_dim,
        ensemble_size=ensemble_size,
        block_dim=block_dim,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    train_type_index = site_type_indices(train_frame["site_type"].astype(str))
    validation_type_index = site_type_indices(validation_frame["site_type"].astype(str))
    exact_labels = torch.tensor(
        train_frame["exact_label"].to_numpy(dtype=float),
        dtype=torch.float32,
    )
    exact_weights = _context_balanced_exact_weights(train_frame)
    normal_pair = _pair_indices(
        train_frame,
        settings=settings,
        ontology_weighting=False,
        same_type_only=False,
    )
    ontology_pair = _pair_indices(
        train_frame,
        settings=settings,
        ontology_weighting=True,
        same_type_only=False,
    )
    membership_pair = _pair_indices(
        train_frame,
        settings=settings,
        ontology_weighting=arm == HIERARCHICAL_ONTOLOGY,
        same_type_only=True,
    )
    endpoint_pair = ontology_pair if arm == HIERARCHICAL_ONTOLOGY else normal_pair
    router_weights = _balanced_router_weights(router_targets)
    auxiliary_mask = train_frame["auxiliary_label"].notna().to_numpy()
    auxiliary_indices = torch.tensor(
        np.flatnonzero(auxiliary_mask),
        dtype=torch.long,
    )
    auxiliary_labels = torch.tensor(
        train_frame.loc[auxiliary_mask, "auxiliary_label"].to_numpy(dtype=float),
        dtype=torch.float32,
    )
    auxiliary_weights = _balanced_binary_cell_weights(
        auxiliary_labels.numpy().astype(int),
        train_frame.loc[auxiliary_mask, "site_type"].astype(str),
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_key = (-math.inf, -math.inf, -math.inf, -math.inf)
    best_epoch = 0
    best_validation: dict[str, object] | None = None
    stale = 0
    final_losses: dict[str, float] = {}
    epochs_completed = 0
    for epoch in range(1, int(settings["maximum_epochs"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        components = model.forward_components(
            train_candidate_features,
            train_context_features,
            train_type_index,
        )
        canonical_bce_values = F.binary_cross_entropy_with_logits(
            components["canonical_logit"],
            exact_labels,
            reduction="none",
        )
        canonical_bce = torch.sum(canonical_bce_values * exact_weights)
        endpoint_pairwise = endpoint_pairwise_logistic_loss(
            components["canonical_logit"],
            endpoint_pair[0],
            endpoint_pair[1],
            pair_weights=endpoint_pair[2],
            margin=float(settings["pairwise_margin"]),
        )
        membership_pairwise = endpoint_pairwise_logistic_loss(
            components["membership_logit"],
            membership_pair[0],
            membership_pair[1],
            pair_weights=membership_pair[2],
            margin=float(settings["pairwise_margin"]),
        )
        router_bce_values = F.binary_cross_entropy_with_logits(
            components["router_logits"][router_indices],
            router_targets,
            reduction="none",
        )
        router_bce = torch.sum(router_bce_values * router_weights)
        auxiliary_bce_values = F.binary_cross_entropy_with_logits(
            components["compatibility_logit"][auxiliary_indices],
            auxiliary_labels,
            reduction="none",
        )
        auxiliary_bce = torch.sum(auxiliary_bce_values * auxiliary_weights)
        loss = (
            float(settings["canonical_bce_weight"]) * canonical_bce
            + float(settings["endpoint_pairwise_weight"]) * endpoint_pairwise
            + float(settings["membership_pairwise_weight"]) * membership_pairwise
        )
        if arm != FULLSPACE_FLAT_EXACT:
            loss = loss + float(settings["router_bce_weight"]) * router_bce
        if arm == HIERARCHICAL_ONTOLOGY:
            loss = loss + float(settings["compatible_auxiliary_weight"]) * auxiliary_bce
        if not bool(torch.isfinite(loss)):
            raise base.SiteIdentificationError("Structured ranker loss is non-finite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            float(settings["gradient_clip_norm"]),
        )
        optimizer.step()
        epochs_completed = epoch
        final_losses = {
            "total": float(loss.detach()),
            "canonical_bce": float(canonical_bce.detach()),
            "endpoint_pairwise": float(endpoint_pairwise.detach()),
            "membership_pairwise": float(membership_pairwise.detach()),
            "router_bce": float(router_bce.detach()),
            "compatible_auxiliary_bce": float(auxiliary_bce.detach()),
        }

        if epoch % int(settings["evaluation_interval"]) != 0:
            continue
        model.eval()
        with torch.no_grad():
            validation_components_tensor = model.forward_components(
                validation_candidate_features,
                validation_context_features,
                validation_type_index,
            )
        validation_components = {
            key: value.cpu().numpy()
            for key, value in validation_components_tensor.items()
        }
        validation_metrics = _validation_metrics(
            frame=validation_frame,
            logits=validation_components["canonical_logit"],
            membership_logits=validation_components["membership_logit"],
            router_logits=(
                validation_components["router_logits"]
                if arm != FULLSPACE_FLAT_EXACT
                else None
            ),
            validation_targets=validation_targets,
        )
        key = _selection_key(validation_metrics)
        if key > best_key:
            best_key = key
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            best_validation = validation_metrics
            stale = 0
        else:
            stale += 1
        if epoch >= int(settings["minimum_epochs"]) and stale >= int(
            settings["early_stopping_patience_evaluations"]
        ):
            break

    if best_state is None or best_validation is None:
        raise base.SiteIdentificationError("Structured ranker produced no checkpoint")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    with torch.no_grad():
        validation_components_tensor = model.forward_components(
            validation_candidate_features,
            validation_context_features,
            validation_type_index,
        )
    validation_components = {
        key: value.cpu().numpy() for key, value in validation_components_tensor.items()
    }
    validation_metrics = _validation_metrics(
        frame=validation_frame,
        logits=validation_components["canonical_logit"],
        membership_logits=validation_components["membership_logit"],
        router_logits=(
            validation_components["router_logits"]
            if arm != FULLSPACE_FLAT_EXACT
            else None
        ),
        validation_targets=validation_targets,
    )
    serializable_validation = {
        key: value
        for key, value in validation_metrics.items()
        if key not in {"top1_correct", "top1_margin"}
    }
    return StructuredRankerFitResult(
        model=model,
        validation_logits=validation_components["canonical_logit"],
        validation_components=validation_components,
        audit={
            "schema_version": "nucpred.mayr-structured-ranker-fit-audit.v1",
            "arm": arm,
            "seed": int(seed),
            "epochs_completed": epochs_completed,
            "best_epoch": best_epoch,
            "best_selection_key": list(best_key),
            "validation_full_candidate_retrieval": serializable_validation,
            "final_train_losses": final_losses,
            "endpoint_pair_audit": endpoint_pair[3],
            "membership_pair_audit": membership_pair[3],
            "auxiliary_labeled_candidate_count": int(auxiliary_mask.sum()),
            "compatible_proxy_positive_count": int(
                train_frame["compatible_proxy"].sum()
            ),
            "canonical_exact_primary": True,
            "compatible_proxy_auxiliary_only": arm == HIERARCHICAL_ONTOLOGY,
        },
    )


def _context_router_indices(
    train_frame: pd.DataFrame,
    train_targets: pd.DataFrame,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    contexts, targets = context_type_targets(
        context_ids=train_targets["context_id"].astype(str).tolist(),
        site_types=train_targets["site_type"].astype(str).tolist(),
    )
    indices: list[int] = []
    for context_id in contexts:
        group = train_frame.loc[
            train_frame["context_id"].astype(str).eq(context_id)
            & train_frame["exact_label"].eq(1)
        ]
        if group.empty:
            raise base.SiteIdentificationError(
                "Router context lost exact representative"
            )
        indices.append(int(group.index[0]))
    return torch.tensor(indices, dtype=torch.long), targets, contexts


def _baseline_validation_result(
    *,
    checkpoint: Mapping[str, Any],
    source_features: torch.Tensor,
    validation_frame: pd.DataFrame,
    validation_targets: pd.DataFrame,
) -> tuple[Any, dict[str, np.ndarray], dict[str, object]]:
    model = ranker_from_checkpoint(checkpoint)
    type_index = site_type_indices(validation_frame["site_type"].astype(str))
    with torch.no_grad():
        tensors = score_ranker_from_source_features(
            ranker=model,
            checkpoint=checkpoint,
            source_features=source_features,
            type_index=type_index,
        )
    components = {key: value.numpy() for key, value in tensors.items()}
    metrics = _validation_metrics(
        frame=validation_frame,
        logits=components["canonical_logit"],
        membership_logits=components["membership_logit"],
        router_logits=None,
        validation_targets=validation_targets,
    )
    return model, components, metrics


def _development_split(
    *,
    config: Mapping[str, Any],
    split_seed: int,
    preflight: Path,
    output_directory: Path,
) -> dict[str, object]:
    contexts, targets, _, splits = base._dataset_tables(config)
    candidates, _ = base._deployment_candidates(config)
    membership = splits.loc[splits["split_seed"].eq(split_seed)].copy()
    development_membership = membership.loc[
        membership["role"].isin(["train", "validation"])
    ].copy()
    development_targets = targets.merge(
        development_membership[["target_id", "role"]],
        on="target_id",
        how="inner",
        validate="one_to_one",
    )
    if set(development_targets["role"].astype(str)) != {"train", "validation"}:
        raise base.SiteIdentificationError(
            "Structured development role coverage changed"
        )
    development_contexts = (
        development_membership[["context_id", "species_id", "connectivity_id", "role"]]
        .drop_duplicates()
        .sort_values("context_id", kind="stable")
    )
    role_count = development_contexts.groupby("context_id")["role"].nunique()
    if int((role_count != 1).sum()):
        raise base.SiteIdentificationError("One context crosses development roles")
    universe = base._candidate_universe(
        test_contexts=development_contexts.drop(columns="role"),
        candidates=candidates,
    )
    universe = universe.merge(
        development_contexts[["context_id", "role"]],
        on="context_id",
        how="left",
        validate="many_to_one",
    )
    query_ids, source_features, n_mean, n_std, backbone_bindings = (
        base._encode_split_ensemble(
            config=config,
            split_seed=split_seed,
            queries=universe,
            contexts=contexts,
            device=torch.device(str(config["device"])),
        )
    )
    print(
        f"split={split_seed} encoded_full_development_candidates={len(query_ids)}",
        file=sys.stderr,
        flush=True,
    )
    ordered = (
        universe.set_index("query_id", drop=False).loc[query_ids].reset_index(drop=True)
    )
    ordered["source_feature_index"] = np.arange(len(ordered), dtype=int)
    ordered["conditional_N_mean"] = n_mean
    ordered["conditional_N_std"] = n_std
    ordered["exact_label"] = _exact_labels(ordered, development_targets).astype(int)

    baseline_checkpoint, baseline_path = _load_baseline_checkpoint(
        config,
        split_seed=split_seed,
    )
    baseline_model = ranker_from_checkpoint(baseline_checkpoint)
    type_index = site_type_indices(ordered["site_type"].astype(str))
    with torch.no_grad():
        baseline_components_all = score_ranker_from_source_features(
            ranker=baseline_model,
            checkpoint=baseline_checkpoint,
            source_features=source_features,
            type_index=type_index,
        )
    ordered["baseline_validity_logit"] = baseline_components_all[
        "canonical_logit"
    ].numpy()

    reviewed = pd.read_parquet(
        preflight / f"split-{split_seed}" / "development_labeled_queries.parquet"
    )
    reviewed_labels, reviewed_conflict_count = _reviewed_context_labels(reviewed)
    train_targets = development_targets.loc[
        development_targets["role"].eq("train")
    ].copy()
    validation_targets = development_targets.loc[
        development_targets["role"].eq("validation")
    ].copy()
    train_all = ordered.loc[ordered["role"].eq("train")].copy()
    validation_frame = ordered.loc[ordered["role"].eq("validation")].copy()
    train_frame = _mark_training_selection(
        train_all,
        train_targets=train_targets,
        reviewed_labels=reviewed_labels,
        settings=config["ranker"],
    )
    validation_frame["exact_label"] = _exact_labels(
        validation_frame,
        validation_targets,
    ).astype(int)
    train_source_indices = torch.tensor(
        train_frame["source_feature_index"].to_numpy(dtype=int),
        dtype=torch.long,
    )
    validation_source_indices = torch.tensor(
        validation_frame["source_feature_index"].to_numpy(dtype=int),
        dtype=torch.long,
    )
    validation_frame = validation_frame.reset_index(drop=True)
    ensemble_size = len(config["backbone"]["initialization_seeds"])
    block_dim = int(config["ranker"]["frozen_feature_block_dim"])
    train_views = reduce_frozen_ensemble_features(
        source_features[train_source_indices],
        ensemble_size=ensemble_size,
        block_dim=block_dim,
    )
    validation_views = reduce_frozen_ensemble_features(
        source_features[validation_source_indices],
        ensemble_size=ensemble_size,
        block_dim=block_dim,
    )
    # Prove the router view is candidate invariant inside every context.
    max_context_view_delta = 0.0
    for _, group in validation_frame.groupby("context_id", sort=True):
        positions = torch.tensor(group.index.to_numpy(dtype=int), dtype=torch.long)
        reference = validation_views.context[positions[0]]
        delta = float(
            torch.max(torch.abs(validation_views.context[positions] - reference))
        )
        max_context_view_delta = max(max_context_view_delta, delta)
    if max_context_view_delta > 1e-5:
        raise base.SiteIdentificationError(
            "Router context view is not candidate invariant"
        )
    del baseline_components_all, baseline_model

    # Reindex after source extraction so pair and router indices are dense.
    train_frame = train_frame.reset_index(drop=True)
    router_indices, router_targets, router_contexts = _context_router_indices(
        train_frame,
        train_targets,
    )
    baseline_validation_source = source_features[validation_source_indices]
    baseline_model, baseline_components, baseline_metrics = _baseline_validation_result(
        checkpoint=baseline_checkpoint,
        source_features=baseline_validation_source,
        validation_frame=validation_frame,
        validation_targets=validation_targets,
    )
    baseline_audit = {
        "schema_version": "nucpred.mayr-frozen-v5-fullspace-validation-audit.v1",
        "arm": FROZEN_V5_BASELINE,
        "checkpoint_path": base._display_path(baseline_path),
        "checkpoint_sha256": sha256_file(baseline_path),
        "validation_full_candidate_retrieval": {
            key: value
            for key, value in baseline_metrics.items()
            if key not in {"top1_correct", "top1_margin"}
        },
        "best_selection_key": list(_selection_key(baseline_metrics)),
    }

    arm_results: dict[str, StructuredRankerFitResult] = {}
    for arm_index, arm in enumerate(STRUCTURED_RANKER_ARMS):
        arm_results[arm] = _fit_structured_arm(
            arm=arm,
            train_frame=train_frame,
            train_candidate_features=train_views.candidate,
            train_context_features=train_views.context,
            validation_frame=validation_frame,
            validation_candidate_features=validation_views.candidate,
            validation_context_features=validation_views.context,
            validation_targets=validation_targets,
            router_indices=router_indices,
            router_targets=router_targets,
            settings=config["ranker"],
            ensemble_size=ensemble_size,
            source_input_dim=int(source_features.shape[1]),
            block_dim=block_dim,
            seed=int(config["ranker"]["training_seed_offset"]) + split_seed + arm_index,
        )
        arm_metrics = arm_results[arm].audit["validation_full_candidate_retrieval"]
        print(
            f"split={split_seed} arm={arm} "
            f"top1={arm_metrics['exact_top1_recall']:.6f} "
            f"mrr={arm_metrics['mrr']:.6f}",
            file=sys.stderr,
            flush=True,
        )

    selection_keys: dict[str, tuple[float, float, float, float]] = {
        FROZEN_V5_BASELINE: _selection_key(baseline_metrics),
        **{
            arm: tuple(map(float, result.audit["best_selection_key"]))
            for arm, result in arm_results.items()
        },
    }
    configured_arm_order = tuple(config["ranker"]["arms"])
    selected_arm = max(configured_arm_order, key=selection_keys.__getitem__)
    if selected_arm == FROZEN_V5_BASELINE:
        selected_model = baseline_model
        selected_components = baseline_components
        selected_metrics = baseline_metrics
        selected_architecture = baseline_checkpoint["ranker_architecture"]
    else:
        selected_result = arm_results[selected_arm]
        selected_model = selected_result.model
        selected_components = selected_result.validation_components
        selected_metrics = _validation_metrics(
            frame=validation_frame,
            logits=selected_components["canonical_logit"],
            membership_logits=selected_components["membership_logit"],
            router_logits=(
                selected_components["router_logits"]
                if selected_arm != FULLSPACE_FLAT_EXACT
                else None
            ),
            validation_targets=validation_targets,
        )
        selected_architecture = selected_model.architecture

    validation_weights = _context_uniform_candidate_weights(validation_frame)
    validation_type_index = site_type_indices(validation_frame["site_type"].astype(str))
    validation_labels = validation_frame["exact_label"].to_numpy(dtype=int)
    calibrator, calibrator_audit = fit_type_aware_platt(
        logits=torch.tensor(
            selected_components["canonical_logit"],
            dtype=torch.float32,
        ),
        type_index=validation_type_index,
        labels=torch.tensor(validation_labels, dtype=torch.float32),
        weights=validation_weights,
        l2_type_offset=float(config["calibration"]["l2_type_offset"]),
        l2_log_slope=float(config["calibration"]["l2_log_slope"]),
        maximum_iterations=int(config["calibration"]["maximum_iterations"]),
    )
    with torch.no_grad():
        calibrated = calibrator(
            torch.tensor(
                selected_components["canonical_logit"],
                dtype=torch.float32,
            ),
            validation_type_index,
        ).numpy()
    margin_gate = select_margin_threshold(
        margins=selected_metrics["top1_margin"],
        top1_correct=selected_metrics["top1_correct"],
        thresholds=config["abstention"]["threshold_grid"],
        minimum_precision=float(config["abstention"]["minimum_development_precision"]),
        minimum_accepted_count=int(config["abstention"]["minimum_accepted_count"]),
    )
    state = deepcopy(selected_model.state_dict())
    checkpoint: dict[str, object] = {
        "schema_version": config["ranker"]["checkpoint_schema"],
        "phase": "development_frozen",
        "campaign_id": config["campaign_id"],
        "split_seed": split_seed,
        "selected_arm": selected_arm,
        "ranker_architecture": selected_architecture,
        "ranker_state_dict": state,
        "ranker_state_sha256": tensor_mapping_sha256(state),
        "calibrator": calibrator.to_payload(),
        "calibrator_fit_audit": calibrator_audit,
        "calibration_population": config["calibration"]["population"],
        "calibration_weighting": config["calibration"]["weighting"],
        "margin_abstention": margin_gate,
        "backbone_bindings": backbone_bindings,
        "v5_baseline_binding": {
            "path": base._display_path(baseline_path),
            "sha256": sha256_file(baseline_path),
        },
        "training_roles": ["train"],
        "selection_calibration_abstention_roles": ["validation"],
        "test_labels_read": False,
        "test_predictions_computed": False,
        "conditional_n_backbone_frozen": True,
        "unknown_as_negative_count": 0,
        "fullspace_non_target_semantics": config["evidence"][
            "fullspace_non_target_semantics"
        ],
        "candidate_softmax_used": False,
        "canonical_exact_score_primary": True,
        "compatible_proxy_auxiliary_only": True,
    }
    split_dir = output_directory / f"split-{split_seed}"
    split_dir.mkdir(parents=True)
    torch.save(checkpoint, split_dir / "ranker_checkpoint.pt")
    validation_predictions = validation_frame.copy()
    validation_predictions["validity_logit"] = selected_components["canonical_logit"]
    validation_predictions["membership_logit"] = selected_components["membership_logit"]
    validation_predictions["router_selected_logit"] = selected_components[
        "router_selected_logit"
    ]
    validation_predictions["compatibility_logit"] = selected_components[
        "compatibility_logit"
    ]
    validation_predictions["raw_sigmoid_score"] = base._sigmoid(
        selected_components["canonical_logit"]
    )
    validation_predictions["absolute_site_probability"] = calibrated
    validation_predictions["evaluation_weight"] = validation_weights.numpy()
    validation_predictions.to_parquet(
        split_dir / "validation_fullspace_predictions.parquet",
        index=False,
        compression="zstd",
    )
    mining_columns = [
        "query_id",
        "context_id",
        "candidate_site_id",
        "site_type",
        "exact_label",
        "compatible_proxy",
        "mine_true_type",
        "mine_reviewed",
        "mine_overlap_or_nested",
        "mine_hard_global",
        "mine_hard_wrong_type",
        "baseline_validity_logit",
    ]
    train_frame[mining_columns].to_parquet(
        split_dir / "training_mined_candidates.parquet",
        index=False,
        compression="zstd",
    )
    arm_audits = {
        FROZEN_V5_BASELINE: baseline_audit,
        **{arm: result.audit for arm, result in arm_results.items()},
    }
    atomic_write_json(split_dir / "arm_fit_audits.json", arm_audits)
    atomic_write_json(split_dir / "margin_abstention.json", margin_gate)
    type_counts = Counter(train_frame["site_type"].astype(str))
    selected_metrics_serializable = {
        key: value
        for key, value in selected_metrics.items()
        if key not in {"top1_correct", "top1_margin"}
    }
    result = {
        "split_seed": split_seed,
        "selected_arm": selected_arm,
        "selection_key": list(selection_keys[selected_arm]),
        "v5_baseline_selection_key": list(selection_keys[FROZEN_V5_BASELINE]),
        "validation_metrics": selected_metrics_serializable,
        "v5_baseline_validation_metrics": baseline_audit[
            "validation_full_candidate_retrieval"
        ],
        "train_full_candidate_count": len(train_all),
        "train_mined_candidate_count": len(train_frame),
        "train_context_count": int(train_frame["context_id"].nunique()),
        "train_exact_positive_count": int(train_frame["exact_label"].sum()),
        "train_compatible_proxy_count": int(train_frame["compatible_proxy"].sum()),
        "train_cross_type_hard_candidate_count": int(
            train_frame["mine_hard_wrong_type"].sum()
        ),
        "train_candidate_count_by_type": {
            site_type: int(type_counts[site_type]) for site_type in RANKER_SITE_TYPES
        },
        "validation_full_candidate_count": len(validation_frame),
        "validation_context_count": int(validation_frame["context_id"].nunique()),
        "reviewed_context_candidate_label_conflict_count": reviewed_conflict_count,
        "router_training_context_count": len(router_contexts),
        "router_context_max_candidate_delta": max_context_view_delta,
        "margin_abstention": margin_gate,
        "ranker_checkpoint_sha256": sha256_file(split_dir / "ranker_checkpoint.pt"),
        "test_labels_read": False,
        "test_predictions_computed": False,
    }
    del (
        source_features,
        train_views,
        validation_views,
        arm_results,
        selected_model,
        baseline_model,
        baseline_validation_source,
    )
    gc.collect()
    return result


def run_development(
    config: Mapping[str, Any],
    *,
    config_path: Path,
) -> dict[str, Any]:
    output_root = base._repo_path(
        config["output_directory"],
        label="output directory",
    )
    preflight = output_root / "preflight"
    if base._load_json(preflight / "summary.json").get("status") != "pass":
        raise base.SiteIdentificationError("Preflight is not complete")
    target = output_root / "development"

    def writer(staged: Path) -> dict[str, Any]:
        split_summaries = [
            _development_split(
                config=config,
                split_seed=int(split_seed),
                preflight=preflight,
                output_directory=staged,
            )
            for split_seed in config["backbone"]["split_seeds"]
        ]
        pd.DataFrame(
            [
                {
                    "split_seed": item["split_seed"],
                    "selected_arm": item["selected_arm"],
                    "validation_exact_top1_recall": item["validation_metrics"][
                        "exact_top1_recall"
                    ],
                    "validation_mrr": item["validation_metrics"]["mrr"],
                    "v5_baseline_exact_top1_recall": item[
                        "v5_baseline_validation_metrics"
                    ]["exact_top1_recall"],
                    "margin_threshold": item["margin_abstention"]["selected_threshold"],
                    "margin_accepted_coverage": item["margin_abstention"][
                        "selected_coverage"
                    ],
                    "margin_accepted_precision": item["margin_abstention"][
                        "selected_precision"
                    ],
                }
                for item in split_summaries
            ]
        ).to_csv(staged / "split_summary.csv", index=False)
        selected_top1 = np.asarray(
            [
                item["validation_metrics"]["exact_top1_recall"]
                for item in split_summaries
            ],
            dtype=float,
        )
        baseline_top1 = np.asarray(
            [
                item["v5_baseline_validation_metrics"]["exact_top1_recall"]
                for item in split_summaries
            ],
            dtype=float,
        )
        return {
            "schema_version": DEVELOPMENT_SCHEMA,
            "status": "pass",
            "campaign_id": config["campaign_id"],
            "config_sha256": sha256_file(config_path),
            "preflight_manifest_sha256": sha256_file(preflight / "run_manifest.json"),
            "split_summaries": split_summaries,
            "macro_validation_exact_top1_recall": float(selected_top1.mean()),
            "macro_v5_baseline_validation_exact_top1_recall": float(
                baseline_top1.mean()
            ),
            "macro_validation_top1_delta_vs_v5": float(
                (selected_top1 - baseline_top1).mean()
            ),
            "development_frozen": True,
            "full_candidate_validation_used": True,
            "canonical_exact_score_primary": True,
            "compatible_proxy_auxiliary_only": True,
            "conditional_n_backbone_frozen": True,
            "test_labels_read": False,
            "test_predictions_computed": False,
            "unknown_as_negative_count": 0,
            "candidate_softmax_used": False,
        }

    return base._publish_stage(
        target,
        schema_version=DEVELOPMENT_SCHEMA,
        writer=writer,
    )


def run_all(
    config: Mapping[str, Any],
    *,
    config_path: Path,
) -> dict[str, Any]:
    return {
        "status": "pass",
        "preflight": base.run_preflight(config, config_path=config_path),
        "development": run_development(config, config_path=config_path),
        "test_predictions": base.run_test_predictions(
            config,
            config_path=config_path,
        ),
        "test_evaluation": base.run_test_evaluation(
            config,
            config_path=config_path,
        ),
        "deployment": base.run_deployment_registry(
            config,
            config_path=config_path,
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run structured v6 Mayr site identification",
    )
    parser.add_argument(
        "command",
        choices=["preflight", "develop", "predict-test", "test", "deploy", "all"],
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = args.config.resolve()
    config = base.read_config(config_path)
    runners = {
        "preflight": base.run_preflight,
        "develop": run_development,
        "predict-test": base.run_test_predictions,
        "test": base.run_test_evaluation,
        "deploy": base.run_deployment_registry,
        "all": run_all,
    }
    result = runners[args.command](config, config_path=config_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
