"""Stage-E-C independent experts on a frozen Stage-E-B-N1 parent.

Each authorized arm adds one zero-start, fixed-gate residual path.  The entire
Stage-E-B-N1 parent remains frozen and is the exact epoch-zero prediction.
Inputs are target-independent RDKit/xTB graph features plus charge and solvent;
Mayr class labels, target values, errors, and split roles are never features.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

from nucpred.features.all_atom_graph import ELEMENT_VOCABULARY
from nucpred.training.mayr_site_n import SITE_TYPE_TO_INDEX, SiteNModelInputs
from nucpred.training.mayr_site_n_stage_e_a import StageEAOutput
from nucpred.training.mayr_site_n_stage_e_b import (
    E_B_N1,
    MayrSiteNStageEBResidualModel,
)


STAGE_E_C_MODEL_SCHEMA_VERSION = "nucpred.mayr-site-n-stage-e-c-expert.v1"
E_C_N1 = "e_c_n1_atom_electronic_interaction_expert"
E_C_N2 = "e_c_n2_transferable_h_parent_expert"
E_C_N3 = "e_c_n3_coordination_region_expert"
STAGE_E_C_ARMS = (E_C_N1, E_C_N2, E_C_N3)

COORDINATION_ELEMENT_CHANNELS = (
    "B",
    "Ge",
    "Hg",
    "K",
    "Li",
    "Na",
    "P",
    "Pb",
    "Si",
    "Sn",
    "Zn",
)
COORDINATION_BOND_TYPE_CHANNELS = ("DATIVE", "IONIC", "ZERO")
_BOND_TYPE_INDEX = {"DATIVE": 7, "IONIC": 8, "ZERO": 10}
_ELEMENT = {name: index for index, name in enumerate(ELEMENT_VOCABULARY)}


def _segment_sum(
    values: torch.Tensor,
    index: torch.Tensor,
    count: int,
) -> torch.Tensor:
    shape = (count,) + tuple(values.shape[1:])
    result = values.new_zeros(shape)
    if values.numel():
        result.index_add_(0, index, values)
    return result


def _segment_any(
    values: torch.Tensor,
    index: torch.Tensor,
    count: int,
) -> torch.Tensor:
    # The project CUDA build does not support bool scatter-reduce kernels.
    result = torch.zeros(count, dtype=torch.uint8, device=values.device)
    if values.numel():
        result.scatter_reduce_(
            0,
            index,
            values.to(dtype=torch.uint8),
            reduce="amax",
        )
    return result.bool()


def _site_membership(inputs: SiteNModelInputs) -> tuple[torch.Tensor, torch.Tensor]:
    counts = inputs.site_member_ptr[1:] - inputs.site_member_ptr[:-1]
    site_for_member = torch.repeat_interleave(
        torch.arange(inputs.num_sites, device=inputs.site_member_index.device),
        counts,
    )
    return counts, site_for_member


@dataclass(frozen=True, slots=True)
class HeavyParentSummary:
    """Exact heavy-parent summaries for each explicit-H typed site."""

    embedding: torch.Tensor
    local_xtb: torch.Tensor
    valid: torch.Tensor
    member_count: torch.Tensor
    recovered_parent_count: torch.Tensor


def exact_heavy_parent_summary(
    inputs: SiteNModelInputs,
    node_embeddings: torch.Tensor,
) -> HeavyParentSummary:
    """Recover one graph-neighbor heavy parent for every exact H member.

    Membership is handled per packed membership occurrence, so the helper
    remains deterministic even if a graph node appears in more than one query.
    Malformed or mixed-member sites receive ``valid=False`` and therefore the
    model's fixed parent fallback.
    """

    counts, site_for_member = _site_membership(inputs)
    element = inputs.node_categorical[:, 0]
    hydrogen = _ELEMENT["H"]
    source, destination = inputs.edge_index
    eligible = (element[source] == hydrogen) & (element[destination] != hydrogen)

    node_parent_count = torch.zeros(
        element.shape[0],
        dtype=torch.long,
        device=element.device,
    )
    if bool(eligible.any()):
        node_parent_count.index_add_(
            0,
            source[eligible],
            torch.ones_like(source[eligible], dtype=torch.long),
        )
    node_parent_index = torch.zeros(
        element.shape[0],
        dtype=torch.long,
        device=element.device,
    )
    if bool(eligible.any()):
        # Valid explicit H nodes have exactly one eligible directed edge.
        node_parent_index[source[eligible]] = destination[eligible]

    members = inputs.site_member_index
    member_is_h = element[members] == hydrogen
    member_has_exact_parent = member_is_h & (node_parent_count[members] == 1)
    safe_parent = node_parent_index[members]
    recovered = _segment_sum(
        member_has_exact_parent.long(),
        site_for_member,
        inputs.num_sites,
    )
    all_members_h = (
        _segment_sum(member_is_h.long(), site_for_member, inputs.num_sites) == counts
    )
    valid = all_members_h & (recovered == counts)

    mask = member_has_exact_parent.to(dtype=node_embeddings.dtype).unsqueeze(-1)
    embedding_sum = _segment_sum(
        node_embeddings[safe_parent] * mask,
        site_for_member,
        inputs.num_sites,
    )
    local_sum = _segment_sum(
        inputs.node_local[safe_parent]
        * member_has_exact_parent.to(dtype=inputs.node_local.dtype).unsqueeze(-1),
        site_for_member,
        inputs.num_sites,
    )
    denominator_embedding = (
        recovered.clamp_min(1).to(dtype=node_embeddings.dtype).unsqueeze(-1)
    )
    denominator_local = (
        recovered.clamp_min(1).to(dtype=inputs.node_local.dtype).unsqueeze(-1)
    )
    return HeavyParentSummary(
        embedding=embedding_sum / denominator_embedding,
        local_xtb=local_sum / denominator_local,
        valid=valid,
        member_count=counts,
        recovered_parent_count=recovered,
    )


def coordination_context_indicators(inputs: SiteNModelInputs) -> torch.Tensor:
    """Build target-independent graph/member coordination channels per site."""

    element = inputs.node_categorical[:, 0]
    site_graph = inputs.site_graph_index
    counts, site_for_member = _site_membership(inputs)
    del counts
    channels: list[torch.Tensor] = []

    for symbol in COORDINATION_ELEMENT_CHANNELS:
        node_mask = element == _ELEMENT[symbol]
        graph_any = _segment_any(
            node_mask,
            inputs.node_graph_index,
            inputs.num_graphs,
        )
        member_any = _segment_any(
            node_mask[inputs.site_member_index],
            site_for_member,
            inputs.num_sites,
        )
        channels.extend((graph_any[site_graph], member_any))

    edge_graph = inputs.node_graph_index[inputs.edge_index[0]]
    for label in COORDINATION_BOND_TYPE_CHANNELS:
        graph_any = _segment_any(
            inputs.edge_categorical[:, 0] == _BOND_TYPE_INDEX[label],
            edge_graph,
            inputs.num_graphs,
        )
        channels.append(graph_any[site_graph])
    return torch.stack(channels, dim=-1).to(dtype=torch.float32)


class MayrSiteNStageECExpertModel(nn.Module):
    """Frozen E-B-N1 plus exactly one authorized independent expert."""

    def __init__(
        self,
        *,
        frozen_parent: MayrSiteNStageEBResidualModel,
        arm: str,
        coordination_element_channels: Sequence[str] = (COORDINATION_ELEMENT_CHANNELS),
        coordination_bond_type_channels: Sequence[str] = (
            COORDINATION_BOND_TYPE_CHANNELS
        ),
    ) -> None:
        super().__init__()
        if arm not in STAGE_E_C_ARMS:
            raise ValueError(f"Unsupported Stage-E-C arm: {arm}")
        if frozen_parent.arm != E_B_N1:
            raise ValueError("Stage-E-C parent must be frozen Stage-E-B-N1")
        if tuple(coordination_element_channels) != COORDINATION_ELEMENT_CHANNELS:
            raise ValueError("Stage-E-C coordination element contract changed")
        if tuple(coordination_bond_type_channels) != (COORDINATION_BOND_TYPE_CHANNELS):
            raise ValueError("Stage-E-C coordination bond contract changed")

        self.frozen_parent = frozen_parent
        for parameter in self.frozen_parent.parameters():
            parameter.requires_grad_(False)
        self.frozen_parent.eval()
        self.arm = arm

        base_architecture = dict(self.frozen_parent.frozen_base.architecture)
        hidden_dim = int(base_architecture["hidden_dim"])
        dropout = float(base_architecture["dropout"])
        bottleneck_dim = max(32, hidden_dim // 2)
        if arm == E_C_N1:
            input_dim = 7 * hidden_dim
            residual_features = (
                "frozen_graph_pool",
                "frozen_site_embedding",
                "frozen_global_xtb_embedding",
                "frozen_site_times_charge",
                "frozen_site_times_continuous_solvent",
                "frozen_global_xtb_times_charge",
                "frozen_global_xtb_times_continuous_solvent",
            )
            gate = "fixed_one_for_atom_else_zero"
        elif arm == E_C_N2:
            self.parent_local_projection = nn.Sequential(
                nn.Linear(8, hidden_dim),
                nn.SiLU(),
                nn.LayerNorm(hidden_dim),
            )
            input_dim = 8 * hidden_dim
            residual_features = (
                "frozen_h_site_embedding",
                "frozen_exact_heavy_parent_embedding",
                "exact_heavy_parent_local_xtb_projection",
                "frozen_global_xtb_embedding",
                "frozen_parent_times_charge",
                "frozen_parent_times_continuous_solvent",
                "frozen_global_xtb_times_charge",
                "frozen_global_xtb_times_continuous_solvent",
            )
            gate = (
                "fixed_one_for_transferable_h_group_with_exact_heavy_parent_else_zero"
            )
        else:
            context_width = 2 * len(COORDINATION_ELEMENT_CHANNELS) + len(
                COORDINATION_BOND_TYPE_CHANNELS
            )
            self.coordination_projection = nn.Sequential(
                nn.Linear(context_width, hidden_dim),
                nn.SiLU(),
                nn.LayerNorm(hidden_dim),
            )
            input_dim = 8 * hidden_dim
            residual_features = (
                "frozen_graph_pool",
                "frozen_site_embedding",
                "frozen_global_xtb_embedding",
                "target_independent_coordination_context_projection",
                "frozen_site_times_charge",
                "frozen_site_times_continuous_solvent",
                "frozen_global_xtb_times_charge",
                "frozen_global_xtb_times_continuous_solvent",
            )
            gate = "fixed_one_for_bond_or_region_and_any_coordination_context_else_zero"

        self.residual_head = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim),
            nn.SiLU(),
            nn.LayerNorm(bottleneck_dim),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, 1),
        )
        final = self.residual_head[-1]
        if not isinstance(final, nn.Linear):
            raise AssertionError("Unexpected Stage-E-C residual output layer")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

        self.architecture = {
            "schema_version": STAGE_E_C_MODEL_SCHEMA_VERSION,
            "arm": arm,
            "frozen_parent_architecture": dict(self.frozen_parent.architecture),
            "hidden_dim": hidden_dim,
            "residual_features": list(residual_features),
            "residual_bottleneck_dim": bottleneck_dim,
            "residual_output_initialization": "exact_zero",
            "residual_gate": gate,
            "malformed_h_parent_policy": (
                "exact_frozen_parent_fallback" if arm == E_C_N2 else None
            ),
            "coordination_element_channels": (
                list(COORDINATION_ELEMENT_CHANNELS) if arm == E_C_N3 else []
            ),
            "coordination_bond_type_channels": (
                list(COORDINATION_BOND_TYPE_CHANNELS) if arm == E_C_N3 else []
            ),
            "mayr_class_label_input": False,
            "target_or_n_value_context_input": False,
            "parent_parameters_trainable": False,
            "parent_forced_eval_mode": True,
            "combined_arm": False,
        }

    def train(self, mode: bool = True) -> "MayrSiteNStageECExpertModel":
        super().train(mode)
        self.frozen_parent.eval()
        return self

    def _expert_features_and_gate(
        self,
        inputs: SiteNModelInputs,
        *,
        parent_output: StageEAOutput,
        solvent_continuous: torch.Tensor,
        charge: torch.Tensor,
        global_xtb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        graph = parent_output.graph_pool[inputs.site_graph_index].detach()
        site = parent_output.site_embeddings.detach()
        global_xtb = global_xtb.detach()
        solvent_continuous = solvent_continuous.detach()
        charge = charge.detach()
        if self.arm == E_C_N1:
            features = torch.cat(
                (
                    graph,
                    site,
                    global_xtb,
                    site * charge,
                    site * solvent_continuous,
                    global_xtb * charge,
                    global_xtb * solvent_continuous,
                ),
                dim=-1,
            )
            active = inputs.site_type_index == SITE_TYPE_TO_INDEX["atom"]
        elif self.arm == E_C_N2:
            parent = exact_heavy_parent_summary(
                inputs,
                parent_output.node_embeddings.detach(),
            )
            parent_embedding = parent.embedding.detach()
            parent_local = self.parent_local_projection(parent.local_xtb.detach())
            features = torch.cat(
                (
                    site,
                    parent_embedding,
                    parent_local,
                    global_xtb,
                    parent_embedding * charge,
                    parent_embedding * solvent_continuous,
                    global_xtb * charge,
                    global_xtb * solvent_continuous,
                ),
                dim=-1,
            )
            active = (
                inputs.site_type_index == SITE_TYPE_TO_INDEX["transferable_h_group"]
            ) & parent.valid
        else:
            context = coordination_context_indicators(inputs)
            context_projection = self.coordination_projection(context)
            features = torch.cat(
                (
                    graph,
                    site,
                    global_xtb,
                    context_projection,
                    site * charge,
                    site * solvent_continuous,
                    global_xtb * charge,
                    global_xtb * solvent_continuous,
                ),
                dim=-1,
            )
            typed = (inputs.site_type_index == SITE_TYPE_TO_INDEX["bond"]) | (
                inputs.site_type_index == SITE_TYPE_TO_INDEX["delocalized_region"]
            )
            active = typed & context.bool().any(dim=-1)
        return features, active

    def diagnostic_gate(self, inputs: SiteNModelInputs) -> torch.Tensor:
        """Return the fixed gate without using targets, scores, or split roles."""

        self.frozen_parent.eval()
        base = self.frozen_parent.frozen_base
        with torch.no_grad():
            parent_output = self.frozen_parent(inputs)
            site_graph = inputs.site_graph_index
            solvent_continuous = base.solvent_encoder(inputs.solvent_continuous)[
                site_graph
            ]
            charge = base.charge_encoder(inputs.molecular_formal_charge)[site_graph]
            global_xtb = base.global_xtb_encoder(inputs.global_xtb)[site_graph]
        _, active = self._expert_features_and_gate(
            inputs,
            parent_output=parent_output,
            solvent_continuous=solvent_continuous,
            charge=charge,
            global_xtb=global_xtb,
        )
        return active

    def forward(self, inputs: SiteNModelInputs) -> StageEAOutput:
        self.frozen_parent.eval()
        base = self.frozen_parent.frozen_base
        with torch.no_grad():
            parent_output = self.frozen_parent(inputs)
            site_graph = inputs.site_graph_index
            solvent_continuous = base.solvent_encoder(inputs.solvent_continuous)[
                site_graph
            ]
            charge = base.charge_encoder(inputs.molecular_formal_charge)[site_graph]
            global_xtb = base.global_xtb_encoder(inputs.global_xtb)[site_graph]
        features, active = self._expert_features_and_gate(
            inputs,
            parent_output=parent_output,
            solvent_continuous=solvent_continuous,
            charge=charge,
            global_xtb=global_xtb,
        )
        raw = self.residual_head(features).squeeze(-1)
        gate = active.to(dtype=raw.dtype)
        applied = raw * gate
        parent_prediction = parent_output.n_prediction_standardized.detach()
        return StageEAOutput(
            n_prediction_standardized=parent_prediction + applied,
            node_embeddings=parent_output.node_embeddings,
            graph_pool=parent_output.graph_pool,
            site_embeddings=parent_output.site_embeddings,
            site_summary=parent_output.site_summary,
            frozen_base_prediction_standardized=parent_prediction,
            raw_residual_standardized=raw,
            applied_residual_standardized=applied,
            residual_gate=gate,
            gate_is_learned=False,
        )


def zero_residual_output_is_exact(model: MayrSiteNStageECExpertModel) -> bool:
    final = model.residual_head[-1]
    return bool(
        isinstance(final, nn.Linear)
        and torch.count_nonzero(final.weight).item() == 0
        and torch.count_nonzero(final.bias).item() == 0
    )


def frozen_parent_parameters_are_frozen(
    model: MayrSiteNStageECExpertModel,
) -> bool:
    return all(
        not parameter.requires_grad for parameter in model.frozen_parent.parameters()
    )


def trainable_parameter_count(model: MayrSiteNStageECExpertModel) -> int:
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
