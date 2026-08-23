"""Joint exact-site retrieval and conditional Mayr N prediction.

The existing :mod:`nucpred.training.mayr_site_n` encoder already represents a
molecule/solvent context once and can attach many typed site queries to that
graph.  This module keeps that path intact, adds linear-complexity candidate-set
context, and trains one canonical site logit together with conditional N.

Unreviewed candidates remain part of the inference-time candidate set and the
label-blind set encoder, but they never enter candidate-level retrieval,
pairwise, router-BCE, or binary evidence losses.  The exact endpoint does
provide one context-level endpoint-type label across the types present in the
candidate set; that supervision does not relabel any unknown candidate as a
chemical negative.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import hashlib
import math
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from nucpred.training.mayr_site_n import (
    MODEL_SCHEMA_VERSION,
    SITE_TYPE_NAMES,
    SITE_TYPE_TO_INDEX,
    MayrSiteNModel,
    SiteNModelInputs,
    _segment_mean,
)
from nucpred.training.mayr_site_n_stage_e_b import (
    E_B_N1,
)
from nucpred.training.mayr_site_n_stage_e_c import (
    COORDINATION_BOND_TYPE_CHANNELS,
    COORDINATION_ELEMENT_CHANNELS,
    E_C_N3,
    MayrSiteNStageECExpertModel,
    coordination_context_indicators,
)
from nucpred.training.mayr_site_ranker import RANKER_TYPE_TO_INDEX
from nucpred.training.mayr_site_structured_ranker import (
    STRUCTURED_RANKER_SCHEMA_VERSION,
    StructuredSiteRanker,
    structured_ranker_from_architecture,
)


JOINT_MODEL_SCHEMA_VERSION = "nucpred.mayr-joint-site-n-model.v1"
JOINT_TRANSFER_SCHEMA_VERSION = "nucpred.mayr-joint-site-n-transfer.v1"
PUBLICATION_TRANSFER_SCHEMA_VERSION = (
    "nucpred.mayr-joint-site-n-publication-transfer.v1"
)
PUBLICATION_RANKER_TRANSFER_SCHEMA_VERSION = (
    "nucpred.mayr-joint-site-n-publication-ranker-transfer.v1"
)
JOINT_RESIDUAL_PREFIXES = (
    "candidate_set_pooler.",
    "context_router.",
    "router_log_scale",
    "router_bias",
)


class JointEvidenceState(IntEnum):
    """Evidence states with two distinct zero-loss outcomes."""

    ONTOLOGY_OUT_OF_SCOPE = -2
    UNKNOWN = -1
    ENDPOINT_EXCLUDED = 0
    POSITIVE_EXACT = 1


@dataclass(frozen=True)
class CandidateSetFeatures:
    """Intermediate O(n) candidate-set features and independent logits."""

    candidate_hidden: torch.Tensor
    contextual_features: torch.Tensor
    base_membership_logits: torch.Tensor
    membership_logits: torch.Tensor


@dataclass(frozen=True)
class JointSiteNOutput:
    """Direct outputs of the joint network; no calibrated quantities live here."""

    n_prediction_standardized: torch.Tensor
    canonical_logits: torch.Tensor
    base_canonical_logits: torch.Tensor
    residual_canonical_logits: torch.Tensor
    membership_logits: torch.Tensor
    router_logits: torch.Tensor
    base_router_logits: torch.Tensor
    residual_router_logits: torch.Tensor
    router_selected_logits: torch.Tensor
    base_compatibility_logits: torch.Tensor
    node_embeddings: torch.Tensor
    graph_pool: torch.Tensor
    site_embeddings: torch.Tensor
    site_summary: torch.Tensor
    candidate_set_features: torch.Tensor


def _validate_index(
    index: torch.Tensor,
    *,
    row_count: int,
    upper_bound: int,
    name: str,
) -> None:
    if index.shape != (row_count,) or index.dtype != torch.long:
        raise ValueError(f"{name} must be a row-aligned int64 vector")
    if bool((index < 0).any()) or bool((index >= upper_bound).any()):
        raise ValueError(f"{name} is out of range")


def _segment_summaries(
    values: torch.Tensor,
    index: torch.Tensor,
    count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return mean, max, log-mean-exp, and log-count in linear memory."""

    if values.ndim != 2 or not values.shape[0] or count < 1:
        raise ValueError("Segment summaries require a non-empty matrix")
    _validate_index(
        index,
        row_count=int(values.shape[0]),
        upper_bound=count,
        name="segment index",
    )
    width = int(values.shape[1])
    counts = values.new_zeros((count, 1))
    counts.index_add_(
        0,
        index,
        torch.ones((values.shape[0], 1), dtype=values.dtype, device=values.device),
    )
    totals = values.new_zeros((count, width))
    totals.index_add_(0, index, values)
    mean = totals / counts.clamp_min(1.0)

    maximum = values.new_full((count, width), -torch.inf)
    maximum.scatter_reduce_(
        0,
        index[:, None].expand_as(values),
        values,
        reduce="amax",
        include_self=True,
    )
    observed = counts.squeeze(-1).gt(0)
    maximum = torch.where(observed[:, None], maximum, torch.zeros_like(maximum))

    shifted = torch.exp(values - maximum[index])
    exp_total = values.new_zeros((count, width))
    exp_total.index_add_(0, index, shifted)
    log_mean_exp = maximum + torch.log(
        (exp_total / counts.clamp_min(1.0)).clamp_min(
            torch.finfo(values.dtype).tiny
        )
    )
    log_mean_exp = torch.where(
        observed[:, None], log_mean_exp, torch.zeros_like(log_mean_exp)
    )
    return mean, maximum, log_mean_exp, torch.log1p(counts)


def _top_k_mean(
    values: torch.Tensor,
    index: torch.Tensor,
    count: int,
    *,
    k: int,
) -> torch.Tensor:
    """Differentiable fixed-k summaries without candidate-pair materialization."""

    if values.ndim != 1 or values.shape != index.shape or k < 1:
        raise ValueError("Top-k summary inputs are invalid")
    rows: list[torch.Tensor] = []
    zero = values.sum() * 0.0
    for segment in range(count):
        selected = values[index.eq(segment)]
        if not selected.numel():
            rows.append(zero)
            continue
        rows.append(torch.topk(selected, min(k, int(selected.numel()))).values.mean())
    return torch.stack(rows)


class TypedCandidateSetPooler(nn.Module):
    """Typed and global DeepSets-style summaries with relative score features."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        type_embedding_dim: int,
        dropout: float,
        top_k: int = 3,
        use_set_context: bool = True,
    ) -> None:
        super().__init__()
        if input_dim < 1 or hidden_dim < 1 or type_embedding_dim < 1 or top_k < 1:
            raise ValueError("Candidate-set pooling dimensions must be positive")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.type_embedding_dim = int(type_embedding_dim)
        self.top_k = int(top_k)
        self.use_set_context = bool(use_set_context)
        self.candidate_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.type_embedding = nn.Embedding(
            len(SITE_TYPE_NAMES), type_embedding_dim
        )
        self.base_membership_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.base_membership_head.weight)
        nn.init.zeros_(self.base_membership_head.bias)
        scalar_width = 6
        contextual_dim = 7 * hidden_dim + type_embedding_dim + scalar_width
        self.contextual_dim = contextual_dim
        self.contextual_residual = nn.Sequential(
            nn.Linear(contextual_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        final = self.contextual_residual[-1]
        if not isinstance(final, nn.Linear):
            raise AssertionError("Unexpected candidate-set residual head")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(
        self,
        candidate_features: torch.Tensor,
        graph_index: torch.Tensor,
        type_index: torch.Tensor,
        *,
        num_graphs: int,
        reference_logits: torch.Tensor | None = None,
    ) -> CandidateSetFeatures:
        row_count = int(candidate_features.shape[0])
        if candidate_features.shape != (row_count, self.input_dim) or not row_count:
            raise ValueError("Candidate feature shape changed")
        _validate_index(
            graph_index,
            row_count=row_count,
            upper_bound=num_graphs,
            name="candidate graph index",
        )
        _validate_index(
            type_index,
            row_count=row_count,
            upper_bound=len(SITE_TYPE_NAMES),
            name="candidate type index",
        )
        hidden = self.candidate_encoder(candidate_features)
        base = self.base_membership_head(hidden).squeeze(-1)
        if reference_logits is None:
            ranking_reference = base
        else:
            if reference_logits.shape != base.shape or not bool(
                torch.isfinite(reference_logits).all()
            ):
                raise ValueError("Reference logits must be finite and row-aligned")
            ranking_reference = reference_logits.to(
                device=base.device,
                dtype=base.dtype,
            )
        if not self.use_set_context:
            return CandidateSetFeatures(
                candidate_hidden=hidden,
                contextual_features=hidden,
                base_membership_logits=base,
                membership_logits=base,
            )
        group_count = num_graphs * len(SITE_TYPE_NAMES)
        group_index = graph_index * len(SITE_TYPE_NAMES) + type_index
        type_mean, type_max, type_lme, type_log_count = _segment_summaries(
            hidden, group_index, group_count
        )
        graph_mean, graph_max, graph_lme, graph_log_count = _segment_summaries(
            hidden, graph_index, num_graphs
        )

        type_best = base.new_full((group_count,), -torch.inf)
        type_best.scatter_reduce_(
            0, group_index, ranking_reference, reduce="amax", include_self=True
        )
        graph_best = base.new_full((num_graphs,), -torch.inf)
        graph_best.scatter_reduce_(
            0, graph_index, ranking_reference, reduce="amax", include_self=True
        )
        type_top_k = _top_k_mean(
            ranking_reference, group_index, group_count, k=self.top_k
        )
        graph_top_k = _top_k_mean(
            ranking_reference, graph_index, num_graphs, k=self.top_k
        )
        scalar = torch.stack(
            (
                type_log_count[group_index, 0],
                graph_log_count[graph_index, 0],
                type_best[group_index] - ranking_reference,
                graph_best[graph_index] - ranking_reference,
                type_top_k[group_index] - ranking_reference,
                graph_top_k[graph_index] - ranking_reference,
            ),
            dim=1,
        )
        contextual = torch.cat(
            (
                hidden,
                type_mean[group_index],
                type_max[group_index],
                type_lme[group_index],
                graph_mean[graph_index],
                graph_max[graph_index],
                graph_lme[graph_index],
                self.type_embedding(type_index),
                scalar,
            ),
            dim=1,
        )
        membership = base + self.contextual_residual(contextual).squeeze(-1)
        proposed = ranking_reference + membership
        reference_maximum = ranking_reference.new_full(
            (group_count,), -torch.inf
        )
        reference_maximum.scatter_reduce_(
            0,
            group_index,
            ranking_reference,
            reduce="amax",
            include_self=True,
        )
        proposed_maximum = proposed.new_full((group_count,), -torch.inf)
        proposed_maximum.scatter_reduce_(
            0,
            group_index,
            proposed,
            reduce="amax",
            include_self=True,
        )
        membership = (
            membership
            + reference_maximum[group_index]
            - proposed_maximum[group_index]
        )
        return CandidateSetFeatures(
            candidate_hidden=hidden,
            contextual_features=contextual,
            base_membership_logits=base,
            membership_logits=membership,
        )


class MayrJointSiteNModel(MayrSiteNModel):
    """End-to-end conditional N model with typed candidate-set retrieval."""

    def __init__(
        self,
        *,
        num_solvents: int,
        hidden_dim: int = 128,
        layers: int = 4,
        node_embedding_dim: int = 16,
        edge_embedding_dim: int = 16,
        solvent_embedding_dim: int = 16,
        type_embedding_dim: int = 16,
        router_hidden_dim: int = 64,
        router_logit_weight: float = 1.0,
        set_top_k: int = 3,
        use_candidate_set_context: bool = True,
        publication_n_lineage: bool = False,
        publication_ranker_architecture: Mapping[str, object] | None = None,
        publication_n_target_mean: float = 0.0,
        publication_n_target_scale: float = 1.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(
            num_solvents=num_solvents,
            hidden_dim=hidden_dim,
            layers=layers,
            node_embedding_dim=node_embedding_dim,
            edge_embedding_dim=edge_embedding_dim,
            solvent_embedding_dim=solvent_embedding_dim,
            dropout=dropout,
        )
        if not math.isfinite(router_logit_weight) or router_logit_weight < 0:
            raise ValueError("Router-logit weight must be finite and non-negative")
        self.site_n_architecture = dict(self.architecture)
        self.hidden_dim = int(hidden_dim)
        self.router_logit_weight = float(router_logit_weight)
        self.publication_n_lineage = bool(publication_n_lineage)
        if not math.isfinite(publication_n_target_mean) or not math.isfinite(
            publication_n_target_scale
        ) or publication_n_target_scale <= 0:
            raise ValueError("Publication N target transform is invalid")
        self.publication_n_target_mean = float(publication_n_target_mean)
        self.publication_n_target_scale = float(publication_n_target_scale)
        if self.publication_n_lineage:
            bottleneck_dim = max(32, hidden_dim // 2)
            self.publication_eb_residual_head = nn.Sequential(
                nn.Linear(4 * hidden_dim, bottleneck_dim),
                nn.SiLU(),
                nn.LayerNorm(bottleneck_dim),
                nn.Dropout(dropout),
                nn.Linear(bottleneck_dim, 1),
            )
            coordination_width = 2 * len(COORDINATION_ELEMENT_CHANNELS) + len(
                COORDINATION_BOND_TYPE_CHANNELS
            )
            self.publication_ec_coordination_projection = nn.Sequential(
                nn.Linear(coordination_width, hidden_dim),
                nn.SiLU(),
                nn.LayerNorm(hidden_dim),
            )
            self.publication_ec_residual_head = nn.Sequential(
                nn.Linear(8 * hidden_dim, bottleneck_dim),
                nn.SiLU(),
                nn.LayerNorm(bottleneck_dim),
                nn.Dropout(dropout),
                nn.Linear(bottleneck_dim, 1),
            )
        self.publication_site_ranker: StructuredSiteRanker | None = None
        if publication_ranker_architecture is not None:
            if (
                publication_ranker_architecture.get("schema_version")
                != STRUCTURED_RANKER_SCHEMA_VERSION
                or int(publication_ranker_architecture.get("block_dim", -1))
                != hidden_dim
                or int(publication_ranker_architecture.get("candidate_dim", -1))
                != 6 * hidden_dim + 8
                or int(publication_ranker_architecture.get("context_dim", -1))
                != 5 * hidden_dim + 5
                or not math.isclose(
                    float(
                        publication_ranker_architecture.get(
                            "router_logit_weight", math.nan
                        )
                    ),
                    self.router_logit_weight,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError("Publication site-ranker architecture is incompatible")
            self.publication_site_ranker = structured_ranker_from_architecture(
                publication_ranker_architecture
            )
        self.candidate_set_pooler = TypedCandidateSetPooler(
            input_dim=6 * hidden_dim + 1,
            hidden_dim=hidden_dim,
            type_embedding_dim=type_embedding_dim,
            dropout=dropout,
            top_k=set_top_k,
            use_set_context=use_candidate_set_context,
        )
        router_input_dim = 23 * hidden_dim + type_embedding_dim + 3
        self.context_router = nn.Sequential(
            nn.Linear(router_input_dim, router_hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(router_hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(router_hidden_dim, 1),
        )
        router_final = self.context_router[-1]
        if not isinstance(router_final, nn.Linear):
            raise AssertionError("Unexpected router output layer")
        nn.init.zeros_(router_final.weight)
        nn.init.zeros_(router_final.bias)
        self.router_log_scale = nn.Parameter(torch.zeros(len(SITE_TYPE_NAMES)))
        self.router_bias = nn.Parameter(torch.zeros(len(SITE_TYPE_NAMES)))
        self.architecture = {
            **self.site_n_architecture,
            "schema_version": JOINT_MODEL_SCHEMA_VERSION,
            "base_model_schema_version": MODEL_SCHEMA_VERSION,
            "candidate_set_pooling": (
                "typed_global_mean_max_logmeanexp_topk_type_max_preserving.v1"
                if use_candidate_set_context
                else "disabled_independent_candidate_head"
            ),
            "candidate_set_complexity": "linear_in_candidate_count",
            "canonical_site_logit": "frozen_base_plus_trainable_joint_residual",
            "frozen_base_logit_interface": "optional_row_aligned_direct_logit",
            "frozen_base_router_interface": "optional_row_aligned_selected_type_logit",
            "candidate_set_ranking_reference": "frozen_base_canonical_logit",
            "candidate_set_direct_n_feature": "conditional_N_standardized",
            "candidate_set_residual_contract": (
                "context_type_internal_reordering_with_frozen_type_maximum"
                if use_candidate_set_context
                else "not_applicable"
            ),
            "candidate_scores_independent": True,
            "candidate_softmax_used": False,
            "direct_outputs": ["canonical_logit", "conditional_N"],
            "conditional_n_initialization_lineage": (
                "publication_C2_to_E-B-N1_to_E-C-N3"
                if self.publication_n_lineage
                else "base_site_n"
            ),
            "conditional_n_parameters_trainable_after_warmup": True,
            "publication_site_ranker_warm_start": (
                self.publication_site_ranker is not None
            ),
            "publication_site_ranker_architecture": (
                None
                if self.publication_site_ranker is None
                else dict(self.publication_site_ranker.architecture)
            ),
            "publication_ranker_projection": (
                "single_member_plus_neutral_disagreement_channels"
                if self.publication_site_ranker is not None
                else None
            ),
            "publication_ranker_type_index_remap": {
                name: int(RANKER_TYPE_TO_INDEX[name]) for name in SITE_TYPE_NAMES
            },
            "publication_n_target_mean": self.publication_n_target_mean,
            "publication_n_target_scale": self.publication_n_target_scale,
            "type_embedding_dim": int(type_embedding_dim),
            "router_hidden_dim": int(router_hidden_dim),
            "router_input_dim": int(router_input_dim),
            "router_candidate_set_context": (
                "typed_fused_mean_max_logmeanexp_count_base_max_topk.v1"
            ),
            "router_logit_weight": self.router_logit_weight,
            "router_base_adapter": (
                "positive_per_type_affine_plus_context_residual_identity_initialized"
            ),
            "router_log_scale_bounds": [-2.0, 2.0],
            "set_top_k": int(set_top_k),
        }

    def train(self, mode: bool = True) -> "MayrJointSiteNModel":
        """Keep an optional publication base scorer deterministic when training."""

        super().train(mode)
        if self.publication_site_ranker is not None:
            self.publication_site_ranker.eval()
        return self

    def _conditional_n_prediction(
        self,
        encoded,
        inputs: SiteNModelInputs,
    ) -> torch.Tensor:
        base_prediction = self.regression_head(encoded.fused).squeeze(-1)
        if not self.publication_n_lineage:
            return base_prediction

        hidden = self.hidden_dim
        chunks = encoded.fused.split(hidden, dim=-1)
        if len(chunks) != 6:
            raise ValueError("Publication conditional-N lineage requires fused 6h")
        graph, site, solvent_continuous, solvent_embedding, charge, global_xtb = chunks
        eb_features = torch.cat(
            (
                site * solvent_continuous,
                global_xtb * solvent_continuous,
                solvent_continuous,
                solvent_embedding,
            ),
            dim=-1,
        )
        eb_raw = self.publication_eb_residual_head(eb_features).squeeze(-1)
        bond_or_region = (
            inputs.site_type_index.eq(SITE_TYPE_TO_INDEX["bond"])
            | inputs.site_type_index.eq(SITE_TYPE_TO_INDEX["delocalized_region"])
        )
        eb_prediction = base_prediction + eb_raw * bond_or_region.to(eb_raw.dtype)

        coordination = coordination_context_indicators(inputs)
        coordination_projection = self.publication_ec_coordination_projection(
            coordination
        )
        ec_features = torch.cat(
            (
                graph,
                site,
                global_xtb,
                coordination_projection,
                site * charge,
                site * solvent_continuous,
                global_xtb * charge,
                global_xtb * solvent_continuous,
            ),
            dim=-1,
        )
        ec_raw = self.publication_ec_residual_head(ec_features).squeeze(-1)
        ec_active = bond_or_region & coordination.bool().any(dim=-1)
        return eb_prediction + ec_raw * ec_active.to(ec_raw.dtype)

    def forward(
        self,
        inputs: SiteNModelInputs,
        *,
        base_canonical_logits: torch.Tensor | None = None,
        base_router_selected_logits: torch.Tensor | None = None,
    ) -> JointSiteNOutput:
        encoded = self.encode_fused_features(inputs)
        n_prediction = self._conditional_n_prediction(encoded, inputs)
        hidden = self.hidden_dim
        context_per_candidate = torch.cat(
            (encoded.fused[:, :hidden], encoded.fused[:, 2 * hidden :]),
            dim=1,
        )
        context_features = _segment_mean(
            context_per_candidate,
            inputs.site_graph_index,
            inputs.num_graphs,
        )
        publication_membership = encoded.fused.new_zeros(inputs.num_sites)
        publication_compatibility = encoded.fused.new_zeros(inputs.num_sites)
        publication_router = encoded.fused.new_zeros(
            (inputs.num_graphs, len(SITE_TYPE_NAMES))
        )
        if self.publication_site_ranker is not None:
            chunks = encoded.fused.split(hidden, dim=-1)
            if len(chunks) != 6:
                raise ValueError("Publication site ranker requires fused 6h")
            n_raw = (
                n_prediction * self.publication_n_target_scale
                + self.publication_n_target_mean
            )
            ranker = self.publication_site_ranker
            candidate_core = torch.cat((encoded.fused, n_raw[:, None]), dim=1)
            missing_candidate = ranker.candidate_dim - int(candidate_core.shape[1])
            context_core = torch.cat(
                (chunks[0], chunks[2], chunks[3], chunks[4], chunks[5]),
                dim=1,
            )
            missing_context = ranker.context_dim - int(context_core.shape[1])
            if missing_candidate != 7 or missing_context != 5:
                raise ValueError("Publication ranker projection width changed")
            candidate_view = torch.cat(
                (
                    candidate_core,
                    ranker.candidate_mean[-missing_candidate:][None, :].expand(
                        candidate_core.shape[0], -1
                    ),
                ),
                dim=1,
            )
            context_view = torch.cat(
                (
                    context_core,
                    ranker.context_mean[-missing_context:][None, :].expand(
                        context_core.shape[0], -1
                    ),
                ),
                dim=1,
            )
            ranker_type_order = inputs.site_type_index.new_tensor(
                [RANKER_TYPE_TO_INDEX[name] for name in SITE_TYPE_NAMES]
            )
            ranker_type_index = ranker_type_order[inputs.site_type_index]
            ranker_components = ranker.forward_components(
                candidate_view,
                context_view,
                ranker_type_index,
            )
            publication_membership = ranker_components["membership_logit"]
            publication_compatibility = ranker_components["compatibility_logit"]
            publication_router = _segment_mean(
                ranker_components["router_logits"][:, ranker_type_order],
                inputs.site_graph_index,
                inputs.num_graphs,
            )
        publication_router_selected = publication_router[
            inputs.site_graph_index, inputs.site_type_index
        ]
        internal_base = (
            publication_membership
            + self.router_logit_weight * publication_router_selected
        )
        if base_canonical_logits is None:
            external_base = torch.zeros_like(internal_base)
        else:
            if (
                base_canonical_logits.shape != internal_base.shape
                or base_canonical_logits.requires_grad
                or not bool(torch.isfinite(base_canonical_logits).all())
            ):
                raise ValueError(
                    "Base canonical logits must be finite, frozen, and row-aligned"
                )
            external_base = base_canonical_logits.to(
                device=internal_base.device,
                dtype=internal_base.dtype,
            )
        base_canonical = internal_base + external_base
        if base_router_selected_logits is None:
            external_router = torch.zeros_like(publication_router)
        else:
            if (
                base_router_selected_logits.shape != internal_base.shape
                or base_router_selected_logits.requires_grad
                or not bool(torch.isfinite(base_router_selected_logits).all())
            ):
                raise ValueError(
                    "Base router logits must be finite, frozen, and row-aligned"
                )
            selected_router = base_router_selected_logits.to(
                device=internal_base.device,
                dtype=internal_base.dtype,
            )
            group_index = (
                inputs.site_graph_index * len(SITE_TYPE_NAMES)
                + inputs.site_type_index
            )
            external_router = _segment_mean(
                selected_router[:, None],
                group_index,
                inputs.num_graphs * len(SITE_TYPE_NAMES),
            ).reshape(inputs.num_graphs, len(SITE_TYPE_NAMES))
        base_router = publication_router + external_router
        set_features = self.candidate_set_pooler(
            torch.cat((encoded.fused, n_prediction[:, None]), dim=1),
            inputs.site_graph_index,
            inputs.site_type_index,
            num_graphs=inputs.num_graphs,
            reference_logits=base_canonical,
        )
        group_count = inputs.num_graphs * len(SITE_TYPE_NAMES)
        group_index = (
            inputs.site_graph_index * len(SITE_TYPE_NAMES)
            + inputs.site_type_index
        )
        fused_mean, fused_maximum, fused_lme, group_log_count = _segment_summaries(
            encoded.fused,
            group_index,
            group_count,
        )
        type_best = base_canonical.new_full((group_count,), -torch.inf)
        type_best.scatter_reduce_(
            0,
            group_index,
            base_canonical,
            reduce="amax",
            include_self=True,
        )
        graph_best = base_canonical.new_full((inputs.num_graphs,), -torch.inf)
        graph_best.scatter_reduce_(
            0,
            inputs.site_graph_index,
            base_canonical,
            reduce="amax",
            include_self=True,
        )
        type_top_k = _top_k_mean(
            base_canonical,
            group_index,
            group_count,
            k=self.candidate_set_pooler.top_k,
        )
        observed_type = torch.isfinite(type_best)
        graph_grid = torch.arange(
            inputs.num_graphs, device=encoded.fused.device
        ).repeat_interleave(len(SITE_TYPE_NAMES))
        type_grid = torch.arange(
            len(SITE_TYPE_NAMES), device=encoded.fused.device
        ).repeat(inputs.num_graphs)
        relative_best = torch.where(
            observed_type,
            type_best - graph_best[graph_grid],
            torch.zeros_like(type_best),
        )
        relative_top_k = torch.where(
            observed_type,
            type_top_k - graph_best[graph_grid],
            torch.zeros_like(type_top_k),
        )
        router_set_scalar = torch.stack(
            (
                group_log_count[:, 0],
                relative_best,
                relative_top_k,
            ),
            dim=1,
        )
        router_input = torch.cat(
            (
                context_features[graph_grid],
                fused_mean,
                fused_maximum,
                fused_lme,
                self.candidate_set_pooler.type_embedding(type_grid),
                router_set_scalar,
            ),
            dim=1,
        )
        neural_router_residual = self.context_router(router_input).reshape(
            inputs.num_graphs, len(SITE_TYPE_NAMES)
        )
        router_scale = torch.exp(self.router_log_scale.clamp(-2.0, 2.0))
        residual_router_logits = (
            neural_router_residual
            + (router_scale[None, :] - 1.0) * base_router
            + self.router_bias[None, :]
        )
        residual_membership = set_features.membership_logits
        residual_router_selected = residual_router_logits[
            inputs.site_graph_index, inputs.site_type_index
        ]
        residual_canonical = (
            residual_membership
            + self.router_logit_weight * residual_router_selected
        )
        router_logits = residual_router_logits + base_router
        membership = residual_membership + publication_membership
        router_selected = router_logits[
            inputs.site_graph_index, inputs.site_type_index
        ]
        canonical = base_canonical + residual_canonical
        return JointSiteNOutput(
            n_prediction_standardized=n_prediction,
            canonical_logits=canonical,
            base_canonical_logits=base_canonical,
            residual_canonical_logits=residual_canonical,
            membership_logits=membership,
            router_logits=router_logits,
            base_router_logits=base_router,
            residual_router_logits=residual_router_logits,
            router_selected_logits=router_selected,
            base_compatibility_logits=publication_compatibility,
            node_embeddings=encoded.node_embeddings,
            graph_pool=encoded.graph_pool,
            site_embeddings=encoded.site_embeddings,
            site_summary=encoded.site_summary,
            candidate_set_features=set_features.contextual_features,
        )


@dataclass(frozen=True)
class JointSiteNTrainingBatch:
    """Candidate-aligned supervision for retrieval, evidence, and N."""

    inputs: SiteNModelInputs
    retrieval_mask: torch.Tensor
    retrieval_positive_mask: torch.Tensor
    evidence_state: torch.Tensor
    evidence_weight: torch.Tensor
    candidate_population_weight: torch.Tensor
    n_target_standardized: torch.Tensor
    n_supervision_mask: torch.Tensor
    candidate_n_harm: torch.Tensor
    base_canonical_logits: torch.Tensor
    base_router_selected_logits: torch.Tensor
    router_cell_weight: torch.Tensor
    site_context_weight: torch.Tensor
    context_weight: torch.Tensor

    def __post_init__(self) -> None:
        rows = self.inputs.num_sites
        expected = (rows,)
        aligned = {
            "retrieval mask": self.retrieval_mask,
            "retrieval positive mask": self.retrieval_positive_mask,
            "evidence state": self.evidence_state,
            "evidence weight": self.evidence_weight,
            "candidate population weight": self.candidate_population_weight,
            "N target": self.n_target_standardized,
            "N supervision mask": self.n_supervision_mask,
            "candidate N-harm": self.candidate_n_harm,
            "base canonical logits": self.base_canonical_logits,
            "base router-selected logits": self.base_router_selected_logits,
        }
        for name, value in aligned.items():
            if value.shape != expected:
                raise ValueError(f"{name} must align with candidate queries")
        boolean = (
            self.retrieval_mask,
            self.retrieval_positive_mask,
            self.n_supervision_mask,
        )
        if any(value.dtype != torch.bool for value in boolean):
            raise TypeError("Retrieval and N supervision masks must be boolean")
        if bool((self.retrieval_positive_mask & ~self.retrieval_mask).any()):
            raise ValueError("Retrieval positives must be eligible candidates")
        if self.evidence_state.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise TypeError("Evidence states must use an integer dtype")
        allowed = {int(state) for state in JointEvidenceState}
        observed = set(
            map(int, torch.unique(self.evidence_state).detach().cpu().tolist())
        )
        if not observed.issubset(allowed):
            raise ValueError(f"Unknown evidence states: {sorted(observed - allowed)}")
        positive_evidence = self.evidence_state.eq(
            int(JointEvidenceState.POSITIVE_EXACT)
        )
        endpoint_evidence = self.evidence_state.ge(
            int(JointEvidenceState.ENDPOINT_EXCLUDED)
        )
        if not torch.equal(self.retrieval_mask, endpoint_evidence):
            raise ValueError(
                "Retrieval eligibility must be exact positive or endpoint-excluded"
            )
        if not torch.equal(self.retrieval_positive_mask, positive_evidence):
            raise ValueError("Retrieval positives must equal exact-positive evidence")
        if bool((self.n_supervision_mask & ~positive_evidence).any()):
            raise ValueError("N supervision is allowed only for exact positives")
        if not bool(torch.isfinite(self.n_target_standardized[self.n_supervision_mask]).all()):
            raise ValueError("Supervised N targets must be finite")
        if not bool(torch.isfinite(self.evidence_weight).all()) or bool(
            (self.evidence_weight < 0).any()
        ):
            raise ValueError("Evidence weights must be finite and non-negative")
        if not bool(torch.isfinite(self.candidate_population_weight).all()) or bool(
            (self.candidate_population_weight < 0).any()
        ):
            raise ValueError(
                "Candidate population weights must be finite and non-negative"
            )
        if bool((self.candidate_population_weight[~self.retrieval_mask] != 0).any()):
            raise ValueError(
                "Unknown or out-of-scope candidates cannot carry population weight"
            )
        if bool((self.candidate_population_weight[self.retrieval_mask] <= 0).any()):
            raise ValueError("Every retrieval-eligible candidate needs population mass")
        if not bool(torch.isfinite(self.candidate_n_harm).all()) or bool(
            (self.candidate_n_harm < 0).any()
        ):
            raise ValueError("Candidate N-harm must be finite and non-negative")
        if self.candidate_n_harm.requires_grad:
            raise ValueError("Candidate N-harm must come from a frozen teacher")
        if not bool(torch.isfinite(self.base_canonical_logits).all()):
            raise ValueError("Base canonical logits must be finite")
        if self.base_canonical_logits.requires_grad:
            raise ValueError("Base canonical logits must be frozen")
        if not bool(torch.isfinite(self.base_router_selected_logits).all()):
            raise ValueError("Base router-selected logits must be finite")
        if self.base_router_selected_logits.requires_grad:
            raise ValueError("Base router-selected logits must be frozen")
        if bool((self.candidate_n_harm[~self.retrieval_mask] != 0).any()):
            raise ValueError("Unknown or out-of-scope candidates cannot carry N-harm")
        if self.router_cell_weight.shape != (
            self.inputs.num_graphs,
            len(SITE_TYPE_NAMES),
        ):
            raise ValueError("Router cell weights must align with context and type")
        if not bool(torch.isfinite(self.router_cell_weight).all()) or bool(
            (self.router_cell_weight < 0).any()
        ):
            raise ValueError("Router cell weights must be finite and non-negative")
        expected_router_mask = torch.zeros_like(
            self.router_cell_weight, dtype=torch.bool
        )
        eligible_rows = torch.nonzero(self.retrieval_mask, as_tuple=False).flatten()
        expected_router_mask[
            self.inputs.site_graph_index[eligible_rows],
            self.inputs.site_type_index[eligible_rows],
        ] = True
        if not torch.equal(self.router_cell_weight.gt(0), expected_router_mask):
            raise ValueError("Router weights must cover exactly the reviewed type cells")
        for name, value in (
            ("Site context weights", self.site_context_weight),
            ("N context weights", self.context_weight),
        ):
            if value.shape != (self.inputs.num_graphs,):
                raise ValueError(f"{name} must align with packed graphs")
            if not bool(torch.isfinite(value).all()) or bool((value <= 0).any()):
                raise ValueError(f"{name} must be finite and positive")
        for graph in range(self.inputs.num_graphs):
            graph_rows = self.inputs.site_graph_index.eq(graph)
            eligible = graph_rows & self.retrieval_mask
            if bool(eligible.any()) and not bool(
                (eligible & self.retrieval_positive_mask).any()
            ):
                raise ValueError("Every retrieval context requires a positive")

    def to(self, device: str | torch.device) -> "JointSiteNTrainingBatch":
        return JointSiteNTrainingBatch(
            inputs=self.inputs.to(device),
            retrieval_mask=self.retrieval_mask.to(device),
            retrieval_positive_mask=self.retrieval_positive_mask.to(device),
            evidence_state=self.evidence_state.to(device),
            evidence_weight=self.evidence_weight.to(device),
            candidate_population_weight=self.candidate_population_weight.to(device),
            n_target_standardized=self.n_target_standardized.to(device),
            n_supervision_mask=self.n_supervision_mask.to(device),
            candidate_n_harm=self.candidate_n_harm.to(device),
            base_canonical_logits=self.base_canonical_logits.to(device),
            base_router_selected_logits=self.base_router_selected_logits.to(device),
            router_cell_weight=self.router_cell_weight.to(device),
            site_context_weight=self.site_context_weight.to(device),
            context_weight=self.context_weight.to(device),
        )


def frozen_teacher_n_harm(
    teacher_n_prediction: torch.Tensor,
    positive_n_target: torch.Tensor,
    positive_mask: torch.Tensor,
    site_graph_index: torch.Tensor,
    *,
    cap_quantile: float = 0.95,
) -> torch.Tensor:
    """Compute detached extra N damage relative to the best true-site query."""

    if (
        teacher_n_prediction.ndim != 1
        or positive_n_target.shape != teacher_n_prediction.shape
        or positive_mask.shape != teacher_n_prediction.shape
    ):
        raise ValueError("Teacher predictions, positive targets, and mask must align")
    graph_count = int(site_graph_index.max().item()) + 1
    _validate_index(
        site_graph_index,
        row_count=int(teacher_n_prediction.numel()),
        upper_bound=graph_count,
        name="teacher site graph index",
    )
    if positive_mask.dtype != torch.bool:
        raise TypeError("Teacher positive mask must be boolean")
    if not 0.0 < cap_quantile <= 1.0:
        raise ValueError("N-harm cap quantile must be in (0, 1]")
    with torch.no_grad():
        prediction = teacher_n_prediction.detach()
        target = positive_n_target.detach()
        if not bool(torch.isfinite(prediction).all()) or not bool(
            torch.isfinite(target[positive_mask]).all()
        ):
            raise ValueError("Teacher N-harm inputs must be finite")
        harm = torch.zeros_like(prediction)
        for graph in range(graph_count):
            rows = site_graph_index.eq(graph)
            positives = rows & positive_mask
            if not bool(positives.any()):
                raise ValueError("Every teacher context requires a true site")
            targets = target[positives]
            reference_error = torch.abs(prediction[positives] - targets).min()
            candidate_error = torch.abs(
                prediction[rows, None] - targets[None, :]
            ).min(dim=1).values
            harm[rows] = torch.relu(candidate_error - reference_error)
        harm[positive_mask] = 0.0
        positive_harm = harm[harm > 0]
        if not positive_harm.numel():
            return harm
        cap = torch.quantile(positive_harm, float(cap_quantile)).clamp_min(1e-12)
        clipped = harm.clamp_max(cap)
        scale = clipped[clipped > 0].mean().clamp_min(1e-12)
        return (clipped / scale).detach()


def _weighted_context_mean(
    values: torch.Tensor,
    graph_index: torch.Tensor,
    graph_count: int,
    context_weight: torch.Tensor,
    *,
    row_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    if values.ndim != 1 or graph_index.shape != values.shape:
        raise ValueError("Context-balanced values must be row-aligned vectors")
    if not values.numel():
        return context_weight.sum() * 0.0
    weights = torch.ones_like(values) if row_weight is None else row_weight
    if weights.shape != values.shape:
        raise ValueError("Row weights must align with context-balanced values")
    totals = values.new_zeros(graph_count)
    masses = values.new_zeros(graph_count)
    totals.index_add_(0, graph_index, values * weights)
    masses.index_add_(0, graph_index, weights)
    observed = masses.gt(0)
    per_context = totals[observed] / masses[observed]
    selected_weight = context_weight[observed]
    return torch.sum(per_context * selected_weight) / selected_weight.sum().clamp_min(1e-12)


def joint_site_n_loss(
    output: JointSiteNOutput,
    batch: JointSiteNTrainingBatch,
    *,
    listwise_weight: float = 1.0,
    pairwise_weight: float = 1.0,
    evidence_weight: float = 1.0,
    router_weight: float = 1.0,
    n_weight: float = 1.0,
    pairwise_margin: float = 0.75,
    n_harm_multiplier: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | int]]:
    """Context-balanced joint objective with strictly masked evidence labels."""

    candidates = batch.inputs.num_sites
    if output.canonical_logits.shape != (candidates,):
        raise ValueError("Canonical logits must align with candidate queries")
    if output.n_prediction_standardized.shape != (candidates,):
        raise ValueError("Conditional N predictions must align with candidates")
    if output.router_logits.shape != (
        batch.inputs.num_graphs,
        len(SITE_TYPE_NAMES),
    ):
        raise ValueError("Router logits must align with contexts and site types")
    scalar_parameters = (
        listwise_weight,
        pairwise_weight,
        evidence_weight,
        router_weight,
        n_weight,
        n_harm_multiplier,
    )
    if any(not math.isfinite(value) or value < 0 for value in scalar_parameters):
        raise ValueError("Joint loss weights must be finite and non-negative")
    if not math.isfinite(pairwise_margin) or pairwise_margin < 0:
        raise ValueError("Pairwise margin must be finite and non-negative")

    graph_losses: list[torch.Tensor] = []
    graph_weights: list[torch.Tensor] = []
    pair_losses: list[torch.Tensor] = []
    pair_graphs: list[torch.Tensor] = []
    pair_row_weights: list[torch.Tensor] = []
    pair_count = 0
    for graph in range(batch.inputs.num_graphs):
        graph_rows = batch.inputs.site_graph_index.eq(graph)
        eligible = graph_rows & batch.retrieval_mask
        positives = eligible & batch.retrieval_positive_mask
        if not bool(eligible.any()):
            continue
        population = batch.candidate_population_weight
        graph_losses.append(
            torch.logsumexp(
                output.canonical_logits[eligible] + torch.log(population[eligible]),
                dim=0,
            )
            - torch.logsumexp(
                output.canonical_logits[positives] + torch.log(population[positives]),
                dim=0,
            )
        )
        graph_weights.append(batch.site_context_weight[graph])
        negatives = eligible & ~batch.retrieval_positive_mask
        positive_indices = torch.nonzero(positives, as_tuple=False).flatten()
        negative_indices = torch.nonzero(negatives, as_tuple=False).flatten()
        if not negative_indices.numel():
            continue
        positive_grid = positive_indices[:, None].expand(
            -1, negative_indices.numel()
        ).reshape(-1)
        negative_grid = negative_indices[None, :].expand(
            positive_indices.numel(), -1
        ).reshape(-1)
        losses = F.softplus(
            float(pairwise_margin)
            - (
                output.canonical_logits[positive_grid]
                - output.canonical_logits[negative_grid]
            )
        )
        harm_weight = batch.candidate_population_weight[negative_grid] * (
            1.0
            + float(n_harm_multiplier) * batch.candidate_n_harm[negative_grid]
        )
        pair_losses.append(losses)
        pair_graphs.append(
            torch.full_like(positive_grid, graph, dtype=torch.long)
        )
        pair_row_weights.append(harm_weight)
        pair_count += int(losses.numel())
    if graph_losses:
        listwise_values = torch.stack(graph_losses)
        listwise_context_weights = torch.stack(graph_weights)
        listwise = torch.sum(listwise_values * listwise_context_weights) / (
            listwise_context_weights.sum().clamp_min(1e-12)
        )
    else:
        listwise = output.canonical_logits.sum() * 0.0
    if pair_losses:
        pairwise = _weighted_context_mean(
            torch.cat(pair_losses),
            torch.cat(pair_graphs),
            batch.inputs.num_graphs,
            batch.site_context_weight,
            row_weight=torch.cat(pair_row_weights),
        )
    else:
        pairwise = output.canonical_logits.sum() * 0.0

    evidence_mask = batch.evidence_state.ge(
        int(JointEvidenceState.ENDPOINT_EXCLUDED)
    )
    evidence_target = batch.evidence_state[evidence_mask].to(
        output.canonical_logits.dtype
    )
    if bool(evidence_mask.any()):
        evidence_rows = F.binary_cross_entropy_with_logits(
            output.canonical_logits[evidence_mask],
            evidence_target,
            reduction="none",
        )
        evidence = _weighted_context_mean(
            evidence_rows,
            batch.inputs.site_graph_index[evidence_mask],
            batch.inputs.num_graphs,
            batch.site_context_weight,
            row_weight=batch.candidate_population_weight[evidence_mask],
        )
    else:
        evidence = output.canonical_logits.sum() * 0.0

    router_target = torch.zeros_like(output.router_logits)
    router_mask = torch.zeros_like(output.router_logits, dtype=torch.bool)
    eligible_rows = torch.nonzero(batch.retrieval_mask, as_tuple=False).flatten()
    router_mask[
        batch.inputs.site_graph_index[eligible_rows],
        batch.inputs.site_type_index[eligible_rows],
    ] = True
    positive_rows = torch.nonzero(
        batch.retrieval_positive_mask, as_tuple=False
    ).flatten()
    router_target[
        batch.inputs.site_graph_index[positive_rows],
        batch.inputs.site_type_index[positive_rows],
    ] = 1.0
    router_present = torch.zeros_like(output.router_logits, dtype=torch.bool)
    router_present[
        batch.inputs.site_graph_index,
        batch.inputs.site_type_index,
    ] = True
    endpoint_type_losses: list[torch.Tensor] = []
    endpoint_type_weights: list[torch.Tensor] = []
    for graph in range(batch.inputs.num_graphs):
        positive_types = router_target[graph].bool()
        present_types = router_present[graph]
        if not bool(positive_types.any()) or bool(
            (positive_types & ~present_types).any()
        ):
            raise ValueError(
                "Every context needs an exact endpoint type in its candidate set"
            )
        endpoint_type_losses.append(
            torch.logsumexp(output.router_logits[graph, present_types], dim=0)
            - torch.logsumexp(output.router_logits[graph, positive_types], dim=0)
        )
        endpoint_type_weights.append(batch.site_context_weight[graph])
    if endpoint_type_losses:
        endpoint_type_values = torch.stack(endpoint_type_losses)
        endpoint_type_weight = torch.stack(endpoint_type_weights)
        router_endpoint_type = torch.sum(
            endpoint_type_values * endpoint_type_weight
        ) / endpoint_type_weight.sum().clamp_min(1e-12)
    else:
        router_endpoint_type = output.router_logits.sum() * 0.0
    if not torch.equal(batch.router_cell_weight.gt(0), router_mask):
        raise ValueError("Router weight mask changed after batch validation")
    reviewed_router_rows = F.binary_cross_entropy_with_logits(
        output.router_logits[router_mask],
        router_target[router_mask],
        reduction="none",
    )
    reviewed_router_weights = batch.router_cell_weight[router_mask]
    if reviewed_router_rows.numel():
        router_bce = torch.sum(
            reviewed_router_rows * reviewed_router_weights
        ) / reviewed_router_weights.sum().clamp_min(1e-12)
    else:
        router_bce = output.router_logits.sum() * 0.0

    router_pair_losses: list[torch.Tensor] = []
    router_pair_weights: list[torch.Tensor] = []
    router_pair_count = 0
    for graph in range(batch.inputs.num_graphs):
        positive_types = torch.nonzero(
            router_mask[graph] & router_target[graph].bool(),
            as_tuple=False,
        ).flatten()
        negative_types = torch.nonzero(
            router_mask[graph] & ~router_target[graph].bool(),
            as_tuple=False,
        ).flatten()
        if not positive_types.numel() or not negative_types.numel():
            continue
        positive_grid = positive_types[:, None].expand(
            -1, negative_types.numel()
        ).reshape(-1)
        negative_grid = negative_types[None, :].expand(
            positive_types.numel(), -1
        ).reshape(-1)
        router_pair_losses.append(
            F.softplus(
                float(pairwise_margin)
                - (
                    output.router_logits[graph, positive_grid]
                    - output.router_logits[graph, negative_grid]
                )
            )
        )
        router_pair_weights.append(
            torch.sqrt(
                batch.router_cell_weight[graph, positive_grid]
                * batch.router_cell_weight[graph, negative_grid]
            )
        )
        router_pair_count += int(positive_grid.numel())
    if router_pair_losses:
        router_values = torch.cat(router_pair_losses)
        router_weights = torch.cat(router_pair_weights)
        router_pairwise = torch.sum(router_values * router_weights) / (
            router_weights.sum().clamp_min(1e-12)
        )
        reviewed_router = 0.5 * (router_bce + router_pairwise)
    else:
        router_pairwise = output.router_logits.sum() * 0.0
        reviewed_router = router_bce
    router = router_endpoint_type + reviewed_router

    if bool(batch.n_supervision_mask.any()):
        n_rows = F.mse_loss(
            output.n_prediction_standardized[batch.n_supervision_mask],
            batch.n_target_standardized[batch.n_supervision_mask],
            reduction="none",
        )
        n_regression = _weighted_context_mean(
            n_rows,
            batch.inputs.site_graph_index[batch.n_supervision_mask],
            batch.inputs.num_graphs,
            batch.context_weight,
        )
    else:
        n_regression = output.n_prediction_standardized.sum() * 0.0

    total = (
        float(listwise_weight) * listwise
        + float(pairwise_weight) * pairwise
        + float(evidence_weight) * evidence
        + float(router_weight) * router
        + float(n_weight) * n_regression
    )
    return total, {
        "listwise": listwise,
        "pairwise": pairwise,
        "evidence": evidence,
        "router": router,
        "router_endpoint_type": router_endpoint_type,
        "router_bce": router_bce,
        "router_pairwise": router_pairwise,
        "n_regression": n_regression,
        "retrieval_context_count": len(graph_losses),
        "pair_count": pair_count,
        "evidence_positive_count": int(
            batch.evidence_state.eq(int(JointEvidenceState.POSITIVE_EXACT)).sum()
        ),
        "evidence_negative_count": int(
            batch.evidence_state.eq(int(JointEvidenceState.ENDPOINT_EXCLUDED)).sum()
        ),
        "evidence_unknown_count": int(
            batch.evidence_state.eq(int(JointEvidenceState.UNKNOWN)).sum()
        ),
        "evidence_out_of_scope_count": int(
            batch.evidence_state.eq(
                int(JointEvidenceState.ONTOLOGY_OUT_OF_SCOPE)
            ).sum()
        ),
        "n_supervision_count": int(batch.n_supervision_mask.sum()),
        "router_pair_count": router_pair_count,
        "router_endpoint_context_count": len(endpoint_type_losses),
        "router_reviewed_cell_count": int(router_mask.sum()),
    }


def _tensor_mapping_sha256(values: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        tensor = values[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _publication_tensor_mapping_sha256(
    values: Mapping[str, torch.Tensor],
) -> str:
    """Match the delimiter-aware digest used by frozen publication rankers."""

    digest = hashlib.sha256()
    for name in sorted(values):
        tensor = values[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def transfer_site_n_checkpoint(
    model: MayrJointSiteNModel,
    source: MayrSiteNModel | Mapping[str, object],
) -> dict[str, object]:
    """Copy the complete legacy N path while preserving all new joint heads."""

    if not isinstance(model, MayrJointSiteNModel):
        raise TypeError("Target must be a MayrJointSiteNModel")
    if model.publication_n_lineage:
        raise ValueError("Use publication-lineage transfer for this joint model")
    if isinstance(source, MayrSiteNModel):
        source_architecture: object = source.architecture
        source_state = source.state_dict()
    else:
        source_architecture = source.get("model_architecture")
        source_state = source.get("model_state_dict")
        if not isinstance(source_state, Mapping):
            raise TypeError("Checkpoint has no model state mapping")
    if source_architecture != model.site_n_architecture:
        raise ValueError("Site-N checkpoint architecture does not match")
    if not all(isinstance(value, torch.Tensor) for value in source_state.values()):
        raise TypeError("Site-N checkpoint state values must be tensors")
    source_tensors = {str(name): value for name, value in source_state.items()}
    target_state = model.state_dict()
    head_prefixes = (*JOINT_RESIDUAL_PREFIXES, "publication_site_ranker.")
    expected_base = {
        name
        for name in target_state
        if not name.startswith(head_prefixes)
    }
    if set(source_tensors) != expected_base:
        raise ValueError("Site-N checkpoint key set changed")
    joint_keys = set(target_state) - expected_base
    joint_before = {name: target_state[name].detach().clone() for name in joint_keys}
    for name, source_tensor in source_tensors.items():
        if target_state[name].shape != source_tensor.shape:
            raise ValueError(f"Site-N checkpoint tensor shape changed: {name}")
        target_state[name].copy_(source_tensor.to(target_state[name].device))
    model.load_state_dict(target_state, strict=True)
    after = model.state_dict()
    exact = all(
        torch.equal(after[name].detach().cpu(), value.detach().cpu())
        for name, value in source_tensors.items()
    )
    joint_unchanged = all(
        torch.equal(after[name], joint_before[name]) for name in joint_keys
    )
    if not exact or not joint_unchanged:
        raise RuntimeError("Site-N checkpoint transfer was not exact")
    return {
        "schema_version": JOINT_TRANSFER_SCHEMA_VERSION,
        "status": "pass",
        "exact_transfer": True,
        "joint_heads_unchanged": True,
        "transferred_tensor_count": len(source_tensors),
        "source_state_sha256": _tensor_mapping_sha256(source_tensors),
    }


def _publication_transfer_key(source_name: str) -> str:
    mappings = (
        ("frozen_parent.frozen_base.", ""),
        ("frozen_parent.residual_head.", "publication_eb_residual_head."),
        ("coordination_projection.", "publication_ec_coordination_projection."),
        ("residual_head.", "publication_ec_residual_head."),
    )
    for old_prefix, new_prefix in mappings:
        if source_name.startswith(old_prefix):
            return new_prefix + source_name.removeprefix(old_prefix)
    raise ValueError(f"Unexpected publication conditional-N tensor: {source_name}")


def transfer_publication_conditional_n_checkpoint(
    model: MayrJointSiteNModel,
    source: MayrSiteNStageECExpertModel | Mapping[str, object],
) -> dict[str, object]:
    """Transfer the full C2 -> E-B-N1 -> E-C-N3 N lineage exactly."""

    if not isinstance(model, MayrJointSiteNModel) or not model.publication_n_lineage:
        raise TypeError("Target must enable the publication conditional-N lineage")
    if isinstance(source, MayrSiteNStageECExpertModel):
        source_architecture: object = source.architecture
        source_state: object = source.state_dict()
    else:
        source_architecture = source.get("model_architecture")
        source_state = source.get("model_state_dict")
    if not isinstance(source_architecture, Mapping):
        raise TypeError("Publication checkpoint has no model architecture")
    parent_architecture = source_architecture.get("frozen_parent_architecture")
    base_architecture = (
        parent_architecture.get("frozen_base_architecture")
        if isinstance(parent_architecture, Mapping)
        else None
    )
    if (
        source_architecture.get("arm") != E_C_N3
        or not isinstance(parent_architecture, Mapping)
        or parent_architecture.get("arm") != E_B_N1
        or base_architecture != model.site_n_architecture
    ):
        raise ValueError("Publication conditional-N architecture does not match")
    if not isinstance(source_state, Mapping) or not all(
        isinstance(value, torch.Tensor) for value in source_state.values()
    ):
        raise TypeError("Publication checkpoint has no tensor state mapping")
    source_tensors = {str(name): value for name, value in source_state.items()}
    mapped = {_publication_transfer_key(name): value for name, value in source_tensors.items()}
    if len(mapped) != len(source_tensors):
        raise ValueError("Publication checkpoint transfer keys collided")

    target_state = model.state_dict()
    head_prefixes = (*JOINT_RESIDUAL_PREFIXES, "publication_site_ranker.")
    expected = {name for name in target_state if not name.startswith(head_prefixes)}
    if set(mapped) != expected:
        missing = sorted(expected - set(mapped))
        unexpected = sorted(set(mapped) - expected)
        raise ValueError(
            f"Publication checkpoint key set changed; missing={missing[:3]}, "
            f"unexpected={unexpected[:3]}"
        )
    joint_before = {
        name: tensor.detach().clone()
        for name, tensor in target_state.items()
        if name.startswith(head_prefixes)
    }
    for name, source_tensor in mapped.items():
        if target_state[name].shape != source_tensor.shape:
            raise ValueError(f"Publication checkpoint tensor shape changed: {name}")
        target_state[name].copy_(source_tensor.to(target_state[name].device))
    model.load_state_dict(target_state, strict=True)
    after = model.state_dict()
    exact = all(
        torch.equal(after[name].detach().cpu(), value.detach().cpu())
        for name, value in mapped.items()
    )
    joint_unchanged = all(
        torch.equal(after[name], before) for name, before in joint_before.items()
    )
    if not exact or not joint_unchanged:
        raise RuntimeError("Publication conditional-N checkpoint transfer was not exact")
    return {
        "schema_version": PUBLICATION_TRANSFER_SCHEMA_VERSION,
        "status": "pass",
        "exact_transfer": True,
        "joint_heads_unchanged": True,
        "transferred_tensor_count": len(mapped),
        "source_state_sha256": _tensor_mapping_sha256(source_tensors),
        "mapped_state_sha256": _tensor_mapping_sha256(mapped),
    }


def transfer_publication_site_ranker_checkpoint(
    model: MayrJointSiteNModel,
    checkpoint: Mapping[str, object],
) -> dict[str, object]:
    """Load a split-safe structured ranker while preserving new residual heads."""

    ranker = model.publication_site_ranker
    if ranker is None:
        raise TypeError("Target does not enable publication site-ranker warm start")
    architecture = checkpoint.get("ranker_architecture")
    state = checkpoint.get("ranker_state_dict")
    if architecture != ranker.architecture:
        raise ValueError("Publication site-ranker architecture does not match")
    if not isinstance(state, Mapping) or not all(
        isinstance(value, torch.Tensor) for value in state.values()
    ):
        raise TypeError("Publication site-ranker checkpoint has no tensor state")
    source = {str(name): value for name, value in state.items()}
    publication_digest = _publication_tensor_mapping_sha256(source)
    if publication_digest != checkpoint.get("ranker_state_sha256"):
        raise ValueError("Publication site-ranker state hash changed")
    residual_prefixes = JOINT_RESIDUAL_PREFIXES
    before = {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
        if name.startswith(residual_prefixes)
    }
    ranker.load_state_dict(source, strict=True)
    for parameter in ranker.parameters():
        parameter.requires_grad_(False)
    ranker.eval()
    after = model.state_dict()
    exact = all(
        torch.equal(
            after[f"publication_site_ranker.{name}"].detach().cpu(),
            value.detach().cpu(),
        )
        for name, value in source.items()
    )
    residual_unchanged = all(
        torch.equal(after[name], tensor) for name, tensor in before.items()
    )
    if not exact or not residual_unchanged:
        raise RuntimeError("Publication site-ranker transfer was not exact")
    return {
        "schema_version": PUBLICATION_RANKER_TRANSFER_SCHEMA_VERSION,
        "status": "pass",
        "exact_transfer": True,
        "joint_residual_heads_unchanged": True,
        "transferred_tensor_count": len(source),
        "publication_source_state_sha256": publication_digest,
        "joint_transfer_state_sha256": _tensor_mapping_sha256(source),
        "projection": "single_member_plus_neutral_disagreement_channels",
        "frozen_base_offset": True,
    }


def joint_optimizer_parameter_groups(
    model: MayrJointSiteNModel,
    *,
    head_learning_rate: float,
    backbone_multiplier: float = 0.1,
) -> list[dict[str, object]]:
    """Return disjoint differential-LR groups for direct joint training."""

    if head_learning_rate <= 0 or not math.isfinite(head_learning_rate):
        raise ValueError("Head learning rate must be finite and positive")
    if backbone_multiplier <= 0 or not math.isfinite(backbone_multiplier):
        raise ValueError("Backbone multiplier must be finite and positive")
    head_prefixes = JOINT_RESIDUAL_PREFIXES
    backbone: list[nn.Parameter] = []
    heads: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if name.startswith("publication_site_ranker."):
            if parameter.requires_grad:
                raise ValueError("Publication site ranker must remain frozen")
            continue
        (heads if name.startswith(head_prefixes) else backbone).append(parameter)
    return [
        {
            "name": "pretrained_site_n",
            "params": backbone,
            "lr": float(head_learning_rate) * float(backbone_multiplier),
        },
        {
            "name": "joint_heads",
            "params": heads,
            "lr": float(head_learning_rate),
        },
    ]


def set_heads_only_warmup(model: MayrJointSiteNModel, *, enabled: bool) -> None:
    """Freeze or unfreeze the inherited N path for a short head warm-up."""

    head_prefixes = JOINT_RESIDUAL_PREFIXES
    for name, parameter in model.named_parameters():
        if name.startswith("publication_site_ranker."):
            parameter.requires_grad_(False)
        else:
            parameter.requires_grad_(not enabled or name.startswith(head_prefixes))
