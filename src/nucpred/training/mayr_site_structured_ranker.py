"""Structured exact-site retrieval on frozen Stage E-C representations.

The structured ranker separates two questions that the original independent
validity head had to solve with one logit:

* which of the five site-object types are plausible for this molecular context;
* which candidate is the canonical endpoint site within a type.

The type router only receives candidate-invariant frozen context blocks.  The
membership head receives the full ensemble-mean candidate representation.  No
context softmax is used: the five router outputs and all candidate outputs are
independent logits, while endpoint-relative pairwise losses supply retrieval
supervision.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from nucpred.training.mayr_site_ranker import (
    RANKER_SITE_TYPES,
    RANKER_TYPE_TO_INDEX,
)


FULLSPACE_FLAT_EXACT = "fullspace_flat_exact"
HIERARCHICAL_EXACT = "hierarchical_exact"
HIERARCHICAL_ONTOLOGY = "hierarchical_ontology"
FROZEN_V5_BASELINE = "frozen_v5_baseline"
STRUCTURED_RANKER_ARMS = (
    FULLSPACE_FLAT_EXACT,
    HIERARCHICAL_EXACT,
    HIERARCHICAL_ONTOLOGY,
)
STRUCTURED_CAMPAIGN_ARMS = (FROZEN_V5_BASELINE, *STRUCTURED_RANKER_ARMS)
STRUCTURED_RANKER_SCHEMA_VERSION = "nucpred.mayr-structured-site-ranker.v2"
FROZEN_FEATURE_REDUCTION = "ensemble_mean_plus_block_rms_disagreement.v1"


@dataclass(frozen=True, slots=True)
class ReducedFeatureViews:
    """Candidate and candidate-invariant context views."""

    candidate: torch.Tensor
    context: torch.Tensor


@dataclass(frozen=True, slots=True)
class StructuredRankerFitResult:
    """Development-selected structured ranker state and validation outputs."""

    model: "StructuredSiteRanker"
    audit: dict[str, object]
    validation_logits: np.ndarray
    validation_components: dict[str, np.ndarray]


def reduce_frozen_ensemble_features(
    features: torch.Tensor,
    *,
    ensemble_size: int,
    block_dim: int = 128,
    fused_block_count: int = 6,
) -> ReducedFeatureViews:
    """Reduce concatenated frozen checkpoint representations.

    Each Stage E-C checkpoint contributes six ``block_dim``-wide blocks in
    this locked order: graph, site, continuous solvent, solvent embedding,
    charge, global xTB, followed by one conditional-N scalar.  The candidate
    view uses the ensemble mean of all blocks plus one RMS disagreement value
    per block and the N standard deviation.  The context view excludes the
    site block and N, guaranteeing invariance across candidates in one context.
    """

    if features.ndim != 2 or not features.shape[0]:
        raise ValueError("Frozen ensemble features must be a non-empty matrix")
    if ensemble_size <= 0 or block_dim <= 0 or fused_block_count != 6:
        raise ValueError("Frozen feature reduction dimensions are invalid")
    member_dim = fused_block_count * block_dim + 1
    expected_dim = ensemble_size * member_dim
    if int(features.shape[1]) != expected_dim:
        raise ValueError(
            "Frozen ensemble feature width changed: "
            f"expected {expected_dim}, observed {features.shape[1]}"
        )
    if not bool(torch.isfinite(features).all()):
        raise ValueError("Frozen ensemble features contain non-finite values")

    members = features.reshape(features.shape[0], ensemble_size, member_dim)
    ensemble_mean = members.mean(dim=1)
    ensemble_std = members.std(dim=1, correction=0)
    block_disagreement = torch.stack(
        [
            torch.sqrt(
                torch.mean(
                    ensemble_std[
                        :,
                        index * block_dim : (index + 1) * block_dim,
                    ]
                    ** 2,
                    dim=1,
                )
            )
            for index in range(fused_block_count)
        ],
        dim=1,
    )
    n_disagreement = ensemble_std[:, -1:].clone()
    candidate = torch.cat(
        (ensemble_mean, block_disagreement, n_disagreement),
        dim=1,
    )

    # The site block is index 1.  All remaining fused blocks are context-level
    # quantities and must be identical for candidates from the same context.
    context_block_indices = (0, 2, 3, 4, 5)
    context = torch.cat(
        tuple(
            ensemble_mean[
                :,
                index * block_dim : (index + 1) * block_dim,
            ]
            for index in context_block_indices
        )
        + (block_disagreement[:, list(context_block_indices)],),
        dim=1,
    )
    return ReducedFeatureViews(candidate=candidate, context=context)


def fit_feature_normalizer(
    features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit a finite train-only mean and population-scale transform."""

    if features.ndim != 2 or not features.shape[0]:
        raise ValueError("Structured ranker features must be a non-empty matrix")
    if not bool(torch.isfinite(features).all()):
        raise ValueError("Structured ranker features contain non-finite values")
    mean = features.mean(dim=0)
    scale = features.std(dim=0, correction=0)
    scale = torch.where(scale > 1e-6, scale, torch.ones_like(scale))
    return mean.detach(), scale.detach()


class StructuredSiteRanker(nn.Module):
    """Context type router plus type-conditioned canonical membership ranker."""

    def __init__(
        self,
        *,
        candidate_mean: torch.Tensor,
        candidate_scale: torch.Tensor,
        context_mean: torch.Tensor,
        context_scale: torch.Tensor,
        arm: str,
        hidden_dim: int = 96,
        router_hidden_dim: int = 64,
        type_embedding_dim: int = 16,
        router_logit_weight: float = 1.0,
        source_input_dim: int = 2307,
        ensemble_size: int = 3,
        block_dim: int = 128,
    ) -> None:
        super().__init__()
        if arm not in STRUCTURED_RANKER_ARMS:
            raise ValueError(f"Unsupported structured ranker arm: {arm}")
        if candidate_mean.ndim != 1 or candidate_scale.shape != candidate_mean.shape:
            raise ValueError("Candidate normalizer shape changed")
        if context_mean.ndim != 1 or context_scale.shape != context_mean.shape:
            raise ValueError("Context normalizer shape changed")
        if router_logit_weight < 0 or not math.isfinite(router_logit_weight):
            raise ValueError("Router logit weight must be finite and non-negative")
        self.arm = arm
        self.candidate_dim = int(candidate_mean.numel())
        self.context_dim = int(context_mean.numel())
        self.hidden_dim = int(hidden_dim)
        self.router_hidden_dim = int(router_hidden_dim)
        self.type_embedding_dim = int(type_embedding_dim)
        self.router_logit_weight = float(router_logit_weight)
        self.source_input_dim = int(source_input_dim)
        self.ensemble_size = int(ensemble_size)
        self.block_dim = int(block_dim)
        self.register_buffer("candidate_mean", candidate_mean.detach().clone())
        self.register_buffer("candidate_scale", candidate_scale.detach().clone())
        self.register_buffer("context_mean", context_mean.detach().clone())
        self.register_buffer("context_scale", context_scale.detach().clone())

        self.type_embedding = nn.Embedding(
            len(RANKER_SITE_TYPES),
            type_embedding_dim,
        )
        self.membership_encoder = nn.Sequential(
            nn.Linear(self.candidate_dim + type_embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.membership_shared = nn.Linear(hidden_dim, 1)
        self.membership_adapters = nn.ModuleDict(
            {site_type: nn.Linear(hidden_dim, 1) for site_type in RANKER_SITE_TYPES}
        )
        self.compatibility_head = nn.Sequential(
            nn.Linear(hidden_dim, max(hidden_dim // 2, 8)),
            nn.SiLU(),
            nn.Linear(max(hidden_dim // 2, 8), 1),
        )
        self.router = nn.Sequential(
            nn.Linear(self.context_dim, router_hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(router_hidden_dim),
            nn.Linear(router_hidden_dim, len(RANKER_SITE_TYPES)),
        )
        self.architecture = {
            "schema_version": STRUCTURED_RANKER_SCHEMA_VERSION,
            "arm": arm,
            "source_input_dim": self.source_input_dim,
            "candidate_dim": self.candidate_dim,
            "context_dim": self.context_dim,
            "hidden_dim": self.hidden_dim,
            "router_hidden_dim": self.router_hidden_dim,
            "type_embedding_dim": self.type_embedding_dim,
            "router_logit_weight": self.router_logit_weight,
            "ensemble_size": self.ensemble_size,
            "block_dim": self.block_dim,
            "feature_reduction": FROZEN_FEATURE_REDUCTION,
            "site_types": list(RANKER_SITE_TYPES),
            "candidate_scores_independent": True,
            "candidate_softmax_used": False,
            "canonical_exact_score_primary": True,
            "compatible_proxy_auxiliary_head": True,
            "context_router_excludes_site_block_and_conditional_n": True,
        }

    def forward_components(
        self,
        candidate_features: torch.Tensor,
        context_features: torch.Tensor,
        type_index: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return canonical, membership, router, and auxiliary logits."""

        if candidate_features.shape != (
            context_features.shape[0],
            self.candidate_dim,
        ):
            raise ValueError("Candidate feature shape changed")
        if context_features.shape != (
            candidate_features.shape[0],
            self.context_dim,
        ):
            raise ValueError("Context feature shape changed")
        if type_index.shape != (candidate_features.shape[0],):
            raise ValueError("Structured ranker type-index shape changed")
        if bool((type_index < 0).any()) or bool(
            (type_index >= len(RANKER_SITE_TYPES)).any()
        ):
            raise ValueError("Structured ranker type index is out of range")

        normalized_candidate = (
            candidate_features - self.candidate_mean
        ) / self.candidate_scale
        normalized_context = (context_features - self.context_mean) / self.context_scale
        type_embedding = self.type_embedding(type_index)
        hidden = self.membership_encoder(
            torch.cat((normalized_candidate, type_embedding), dim=1)
        )
        membership = self.membership_shared(hidden).squeeze(-1)
        membership_residual = torch.zeros_like(membership)
        for index, site_type in enumerate(RANKER_SITE_TYPES):
            selected = type_index.eq(index)
            if bool(selected.any()):
                membership_residual[selected] = self.membership_adapters[site_type](
                    hidden[selected]
                ).squeeze(-1)
        membership = membership + membership_residual
        compatibility = self.compatibility_head(hidden).squeeze(-1)
        router_all = self.router(normalized_context)
        router_selected = router_all.gather(1, type_index[:, None]).squeeze(1)
        if self.arm == FULLSPACE_FLAT_EXACT:
            canonical = membership
        else:
            canonical = membership + self.router_logit_weight * router_selected
        return {
            "canonical_logit": canonical,
            "membership_logit": membership,
            "router_logits": router_all,
            "router_selected_logit": router_selected,
            "compatibility_logit": compatibility,
        }

    def forward(
        self,
        candidate_features: torch.Tensor,
        context_features: torch.Tensor,
        type_index: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_components(
            candidate_features,
            context_features,
            type_index,
        )["canonical_logit"]


def structured_ranker_from_architecture(
    architecture: Mapping[str, object],
) -> StructuredSiteRanker:
    """Instantiate a structured ranker with placeholder normalizer buffers."""

    if architecture.get("schema_version") != STRUCTURED_RANKER_SCHEMA_VERSION:
        raise ValueError("Structured ranker architecture schema changed")
    if tuple(architecture.get("site_types", ())) != RANKER_SITE_TYPES:
        raise ValueError("Structured ranker site type order changed")
    candidate_dim = int(architecture["candidate_dim"])
    context_dim = int(architecture["context_dim"])
    return StructuredSiteRanker(
        candidate_mean=torch.zeros(candidate_dim),
        candidate_scale=torch.ones(candidate_dim),
        context_mean=torch.zeros(context_dim),
        context_scale=torch.ones(context_dim),
        arm=str(architecture["arm"]),
        hidden_dim=int(architecture["hidden_dim"]),
        router_hidden_dim=int(architecture["router_hidden_dim"]),
        type_embedding_dim=int(architecture["type_embedding_dim"]),
        router_logit_weight=float(architecture["router_logit_weight"]),
        source_input_dim=int(architecture["source_input_dim"]),
        ensemble_size=int(architecture["ensemble_size"]),
        block_dim=int(architecture["block_dim"]),
    )


def endpoint_pairwise_logistic_loss(
    logits: torch.Tensor,
    positive_indices: torch.Tensor,
    negative_indices: torch.Tensor,
    *,
    pair_weights: torch.Tensor | None = None,
    margin: float = 0.0,
) -> torch.Tensor:
    """Endpoint-relative canonical-positive versus non-target pairwise loss."""

    if logits.ndim != 1:
        raise ValueError("Pairwise logits must be a vector")
    if positive_indices.shape != negative_indices.shape:
        raise ValueError("Pairwise index arrays must align")
    if positive_indices.ndim != 1 or not positive_indices.numel():
        raise ValueError("Pairwise indices must be a non-empty vector")
    if bool((positive_indices < 0).any()) or bool(
        (positive_indices >= logits.numel()).any()
    ):
        raise ValueError("Positive pair index is out of range")
    if bool((negative_indices < 0).any()) or bool(
        (negative_indices >= logits.numel()).any()
    ):
        raise ValueError("Negative pair index is out of range")
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("Pairwise margin must be finite and non-negative")
    losses = F.softplus(
        float(margin) - (logits[positive_indices] - logits[negative_indices])
    )
    if pair_weights is None:
        return losses.mean()
    if pair_weights.shape != losses.shape:
        raise ValueError("Pairwise weights must align with pairs")
    if not bool(torch.isfinite(pair_weights).all()) or bool((pair_weights <= 0).any()):
        raise ValueError("Pairwise weights must be finite and positive")
    return torch.sum(losses * pair_weights) / pair_weights.sum().clamp_min(1e-12)


def context_type_targets(
    *,
    context_ids: Sequence[str],
    site_types: Sequence[str],
) -> tuple[list[str], torch.Tensor]:
    """Create one multi-hot type target per context, supporting multi-target rows."""

    if len(context_ids) != len(site_types) or not context_ids:
        raise ValueError("Context/type target inputs must be aligned and non-empty")
    unknown = sorted(set(map(str, site_types)) - set(RANKER_SITE_TYPES))
    if unknown:
        raise ValueError(f"Unknown site types in router targets: {unknown}")
    ordered_contexts = sorted(set(map(str, context_ids)))
    context_to_index = {
        context_id: index for index, context_id in enumerate(ordered_contexts)
    }
    targets = torch.zeros(
        (len(ordered_contexts), len(RANKER_SITE_TYPES)),
        dtype=torch.float32,
    )
    for context_id, site_type in zip(context_ids, site_types, strict=True):
        targets[
            context_to_index[str(context_id)],
            RANKER_TYPE_TO_INDEX[str(site_type)],
        ] = 1.0
    return ordered_contexts, targets


def select_margin_threshold(
    *,
    margins: Sequence[float],
    top1_correct: Sequence[bool],
    thresholds: Sequence[float],
    minimum_precision: float,
    minimum_accepted_count: int,
) -> dict[str, object]:
    """Freeze the highest-coverage margin gate meeting a development constraint."""

    margin_array = np.asarray(margins, dtype=float)
    correct_array = np.asarray(top1_correct, dtype=bool)
    threshold_array = np.asarray(sorted(set(map(float, thresholds))), dtype=float)
    if (
        margin_array.ndim != 1
        or correct_array.shape != margin_array.shape
        or not len(margin_array)
        or np.isnan(margin_array).any()
        or (margin_array < 0).any()
    ):
        raise ValueError("Margin-gate inputs are invalid")
    if (
        not len(threshold_array)
        or not np.isfinite(threshold_array).all()
        or (threshold_array < 0).any()
    ):
        raise ValueError("Margin thresholds must be finite and non-negative")
    if not 0.0 < minimum_precision <= 1.0 or minimum_accepted_count <= 0:
        raise ValueError("Margin-gate acceptance constraint is invalid")

    rows: list[dict[str, object]] = []
    for threshold in threshold_array:
        accepted = margin_array >= threshold
        count = int(accepted.sum())
        rows.append(
            {
                "threshold": float(threshold),
                "accepted_count": count,
                "coverage": float(count / len(margin_array)),
                "precision": (
                    float(correct_array[accepted].mean()) if count else float("nan")
                ),
            }
        )
    eligible = [
        row
        for row in rows
        if int(row["accepted_count"]) >= minimum_accepted_count
        and float(row["precision"]) >= minimum_precision
    ]
    if eligible:
        selected = max(
            eligible,
            key=lambda row: (
                float(row["coverage"]),
                float(row["precision"]),
                -float(row["threshold"]),
            ),
        )
        constraint_met = True
    else:
        nonempty = [row for row in rows if int(row["accepted_count"]) > 0]
        selected = max(
            nonempty,
            key=lambda row: (
                float(row["precision"]),
                float(row["coverage"]),
                float(row["threshold"]),
            ),
        )
        constraint_met = False
    return {
        "schema_version": "nucpred.mayr-margin-abstention-selection.v1",
        "selected_threshold": float(selected["threshold"]),
        "selected_accepted_count": int(selected["accepted_count"]),
        "selected_coverage": float(selected["coverage"]),
        "selected_precision": float(selected["precision"]),
        "minimum_precision": float(minimum_precision),
        "minimum_accepted_count": int(minimum_accepted_count),
        "constraint_met": constraint_met,
        "grid": rows,
        "selection_uses_test_labels": False,
    }
