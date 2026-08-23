"""Stage-E-B frozen-C2 residual models for conditional Mayr N.

Both authorized arms retain the Stage-C C2 model bitwise frozen and start at
the exact C2 prediction.  E-B-N1 limits the Stage-E-A solvent residual to bond
and delocalized-region targets.  E-B-N2 uses only deterministic graph/query
structure-family indicators; Mayr class labels and target values are never
inputs.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from nucpred.features.all_atom_graph import ELEMENT_VOCABULARY
from nucpred.training.mayr_site_n import (
    SITE_TYPE_TO_INDEX,
    MayrSiteNModel,
    SiteNModelInputs,
)
from nucpred.training.mayr_site_n_stage_e_a import StageEAOutput


STAGE_E_B_MODEL_SCHEMA_VERSION = "nucpred.mayr-site-n-stage-e-b-residual.v1"
E_B_N1 = "e_b_n1_bond_region_masked_solvent_residual"
E_B_N2 = "e_b_n2_structural_family_residual"
STAGE_E_B_ARMS = (E_B_N1, E_B_N2)
FAMILY_CHANNELS = (
    "phenolate_like",
    "nho_like",
    "hydride_parent_carbon",
    "hydride_parent_boron",
    "hydride_parent_silicon_germanium_tin",
    "hydride_parent_phosphorus",
    "allyl_main_group_like",
    "neutral_amine_like",
    "oxygen_anion_nonphenolate",
)

_ELEMENT = {name: index for index, name in enumerate(ELEMENT_VOCABULARY)}


def _segment_any(
    values: torch.Tensor, graph_index: torch.Tensor, count: int
) -> torch.Tensor:
    # CUDA scatter-reduce does not implement bool kernels in the supported
    # PyTorch build, so reduce uint8 and convert back to bool.
    result = torch.zeros(count, dtype=torch.uint8, device=values.device)
    if values.numel():
        result.scatter_reduce_(
            0,
            graph_index,
            values.to(dtype=torch.uint8),
            reduce="amax",
        )
    return result.bool()


def _segment_sum(
    values: torch.Tensor, graph_index: torch.Tensor, count: int
) -> torch.Tensor:
    result = torch.zeros(count, dtype=values.dtype, device=values.device)
    if values.numel():
        result.scatter_add_(0, graph_index, values)
    return result


def _site_member_any(
    node_mask: torch.Tensor,
    inputs: SiteNModelInputs,
) -> torch.Tensor:
    """Reduce a node mask over each typed site's exact member set."""

    counts = inputs.site_member_ptr[1:] - inputs.site_member_ptr[:-1]
    site_for_member = torch.repeat_interleave(
        torch.arange(inputs.num_sites, device=node_mask.device),
        counts,
    )
    return _segment_any(
        node_mask[inputs.site_member_index],
        site_for_member,
        inputs.num_sites,
    )


def structural_family_indicators(inputs: SiteNModelInputs) -> torch.Tensor:
    """Return deterministic target-independent family channels per site.

    These are deliberately coarse structural contexts, not Mayr chemical-class
    annotations.  Every input is available from the molecular graph, exact
    query membership, explicit site type, or molecular formal charge encoded
    in the graph.
    """

    categorical = inputs.node_categorical
    element = categorical[:, 0]
    graph_index = inputs.node_graph_index
    graph_count = inputs.num_graphs
    site_graph = inputs.site_graph_index
    site_type = inputs.site_type_index

    aromatic_graph = _segment_any(categorical[:, 4] == 1, graph_index, graph_count)
    ring_graph = _segment_any(categorical[:, 5] == 1, graph_index, graph_count)
    nitrogen_count = _segment_sum(
        (element == _ELEMENT["N"]).long(),
        graph_index,
        graph_count,
    )
    # Formal charge category is bounded charge + 3.
    graph_charge = _segment_sum(
        categorical[:, 2].long() - 3,
        graph_index,
        graph_count,
    )
    has_main_group = _segment_any(
        (element == _ELEMENT["B"])
        | (element == _ELEMENT["Si"])
        | (element == _ELEMENT["Ge"])
        | (element == _ELEMENT["Sn"]),
        graph_index,
        graph_count,
    )

    member_o = _site_member_any(element == _ELEMENT["O"], inputs)
    member_n = _site_member_any(element == _ELEMENT["N"], inputs)
    member_c = _site_member_any(element == _ELEMENT["C"], inputs)
    member_aromatic = _site_member_any(categorical[:, 4] == 1, inputs)
    member_h = _site_member_any(element == _ELEMENT["H"], inputs)

    negative = graph_charge[site_graph] < 0
    neutral = graph_charge[site_graph] == 0
    aromatic = aromatic_graph[site_graph]
    transferable_h = site_type == SITE_TYPE_TO_INDEX["transferable_h_group"]

    # Identify exact heavy-atom parents for explicit H members using directed
    # graph edges. A valid explicit H has one heavy neighbor, but max reduction
    # keeps this deterministic even if malformed input reaches this function.
    source, destination = inputs.edge_index
    source_is_member_h = torch.zeros(
        categorical.shape[0],
        dtype=torch.bool,
        device=categorical.device,
    )
    source_is_member_h[inputs.site_member_index] = (
        element[inputs.site_member_index] == _ELEMENT["H"]
    )
    eligible_edge = source_is_member_h[source] & (element[destination] != _ELEMENT["H"])
    parent_element = element[destination]
    parent_masks: dict[str, torch.Tensor] = {}
    for label, elements in {
        "carbon": ("C",),
        "boron": ("B",),
        "silicon_germanium_tin": ("Si", "Ge", "Sn"),
        "phosphorus": ("P",),
    }.items():
        edge_mask = eligible_edge.clone()
        allowed = torch.zeros_like(edge_mask)
        for symbol in elements:
            allowed |= parent_element == _ELEMENT[symbol]
        edge_mask &= allowed
        node_mask = torch.zeros_like(source_is_member_h)
        if bool(edge_mask.any()):
            node_mask[source[edge_mask]] = True
        parent_masks[label] = _site_member_any(node_mask, inputs)

    channels = (
        member_o & negative & aromatic,
        member_c & neutral & (nitrogen_count[site_graph] >= 2) & ring_graph[site_graph],
        transferable_h & member_h & parent_masks["carbon"],
        transferable_h & member_h & parent_masks["boron"],
        transferable_h & member_h & parent_masks["silicon_germanium_tin"],
        transferable_h & member_h & parent_masks["phosphorus"],
        member_c & has_main_group[site_graph] & ~transferable_h,
        member_n & neutral & ~member_aromatic,
        member_o & negative & ~aromatic,
    )
    return torch.stack(channels, dim=-1).to(dtype=torch.float32)


class MayrSiteNStageEBResidualModel(nn.Module):
    """Frozen C2 plus one of the two authorized zero-start residual paths."""

    def __init__(
        self,
        *,
        frozen_base: MayrSiteNModel,
        arm: str,
        family_channels: Sequence[str] = FAMILY_CHANNELS,
    ) -> None:
        super().__init__()
        if arm not in STAGE_E_B_ARMS:
            raise ValueError(f"Unsupported Stage-E-B arm: {arm}")
        if tuple(family_channels) != FAMILY_CHANNELS:
            raise ValueError("Stage-E-B family channel contract changed")
        self.frozen_base = frozen_base
        for parameter in self.frozen_base.parameters():
            parameter.requires_grad_(False)
        self.frozen_base.eval()
        self.arm = arm

        architecture = dict(frozen_base.architecture)
        hidden_dim = int(architecture["hidden_dim"])
        dropout = float(architecture["dropout"])
        bottleneck_dim = max(32, hidden_dim // 2)
        if arm == E_B_N1:
            input_dim = 4 * hidden_dim
            residual_features = (
                "site_times_continuous_solvent",
                "global_xtb_times_continuous_solvent",
                "continuous_solvent",
                "solvent_embedding",
            )
        else:
            self.family_projection = nn.Sequential(
                nn.Linear(len(FAMILY_CHANNELS), hidden_dim),
                nn.SiLU(),
                nn.LayerNorm(hidden_dim),
            )
            input_dim = 4 * hidden_dim
            residual_features = (
                "frozen_graph_pool",
                "frozen_site_embedding",
                "frozen_global_xtb_embedding",
                "deterministic_family_projection",
            )
        self.residual_head = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim),
            nn.SiLU(),
            nn.LayerNorm(bottleneck_dim),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, 1),
        )
        final = self.residual_head[-1]
        if not isinstance(final, nn.Linear):
            raise AssertionError("Unexpected Stage-E-B residual output layer")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

        self.architecture = {
            "schema_version": STAGE_E_B_MODEL_SCHEMA_VERSION,
            "arm": arm,
            "frozen_base_architecture": architecture,
            "hidden_dim": hidden_dim,
            "residual_features": list(residual_features),
            "residual_output_initialization": "exact_zero",
            "residual_gate": (
                "fixed_one_for_bond_and_delocalized_region_else_zero"
                if arm == E_B_N1
                else "fixed_one_if_any_deterministic_family_channel_else_zero"
            ),
            "family_channels": list(FAMILY_CHANNELS) if arm == E_B_N2 else [],
            "mayr_class_label_input": False,
            "target_or_n_value_family_input": False,
            "base_parameters_trainable": False,
            "base_forced_eval_mode": True,
        }

    def train(self, mode: bool = True) -> "MayrSiteNStageEBResidualModel":
        super().train(mode)
        self.frozen_base.eval()
        return self

    def forward(self, inputs: SiteNModelInputs) -> StageEAOutput:
        self.frozen_base.eval()
        with torch.no_grad():
            encoded = self.frozen_base.encode_fused_features(inputs)
            base = self.frozen_base.regression_head(encoded.fused).squeeze(-1)
        hidden_dim = int(self.architecture["hidden_dim"])
        chunks = encoded.fused.split(hidden_dim, dim=-1)
        if len(chunks) != 6:
            raise ValueError("Stage-E-B requires frozen C2 fused 6h")
        graph, site, solvent_continuous, solvent_embedding, _, global_xtb = chunks
        if self.arm == E_B_N1:
            features = torch.cat(
                (
                    site * solvent_continuous,
                    global_xtb * solvent_continuous,
                    solvent_continuous,
                    solvent_embedding,
                ),
                dim=-1,
            )
            active = (inputs.site_type_index == SITE_TYPE_TO_INDEX["bond"]) | (
                inputs.site_type_index == SITE_TYPE_TO_INDEX["delocalized_region"]
            )
        else:
            indicators = structural_family_indicators(inputs)
            features = torch.cat(
                (
                    graph.detach(),
                    site.detach(),
                    global_xtb.detach(),
                    self.family_projection(indicators),
                ),
                dim=-1,
            )
            active = indicators.bool().any(dim=-1)
        raw = self.residual_head(
            features.detach() if self.arm == E_B_N1 else features
        ).squeeze(-1)
        gate = active.to(dtype=raw.dtype)
        applied = raw * gate
        return StageEAOutput(
            n_prediction_standardized=base + applied,
            node_embeddings=encoded.node_embeddings,
            graph_pool=encoded.graph_pool,
            site_embeddings=encoded.site_embeddings,
            site_summary=encoded.site_summary,
            frozen_base_prediction_standardized=base,
            raw_residual_standardized=raw,
            applied_residual_standardized=applied,
            residual_gate=gate,
            gate_is_learned=False,
        )


def zero_residual_output_is_exact(model: MayrSiteNStageEBResidualModel) -> bool:
    final = model.residual_head[-1]
    return bool(
        isinstance(final, nn.Linear)
        and torch.count_nonzero(final.weight).item() == 0
        and torch.count_nonzero(final.bias).item() == 0
    )


def frozen_base_parameters_are_frozen(model: MayrSiteNStageEBResidualModel) -> bool:
    return all(
        not parameter.requires_grad for parameter in model.frozen_base.parameters()
    )


def trainable_parameter_count(model: MayrSiteNStageEBResidualModel) -> int:
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
