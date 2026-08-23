"""Stage-E-A protected solvent residuals for conditional Mayr N.

The frozen Stage-C C2 model remains the complete structural predictor.  The
only trainable path is a zero-start solvent-dependent residual; the optional
E-N2 gate uses explicit site type and molecular formal charge.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn

from nucpred.training.mayr_site_n import (
    SITE_TYPE_NAMES,
    MayrSiteNModel,
    SiteNExample,
    SiteNModelInputs,
    SiteNTrainingBatch,
    within_context_ranking_loss,
)
from nucpred.training.mayr_site_n_stage_d import (
    PairedSolventDefinition,
    paired_solvent_delta_loss,
)


STAGE_E_A_MODEL_SCHEMA_VERSION = (
    "nucpred.mayr-site-n-stage-e-a-protected-solvent-residual.v1"
)


@dataclass(frozen=True, slots=True)
class StageEAOutput:
    """Prediction plus auditable base, residual, and gate components."""

    n_prediction_standardized: torch.Tensor
    node_embeddings: torch.Tensor
    graph_pool: torch.Tensor
    site_embeddings: torch.Tensor
    site_summary: torch.Tensor
    frozen_base_prediction_standardized: torch.Tensor
    raw_residual_standardized: torch.Tensor
    applied_residual_standardized: torch.Tensor
    residual_gate: torch.Tensor
    gate_is_learned: bool


class MayrSiteNProtectedSolventResidualModel(nn.Module):
    """Frozen C2 plus a centered solvent residual and optional shrinkage gate."""

    def __init__(
        self,
        *,
        frozen_base: MayrSiteNModel,
        charge_type_gate: bool,
        initial_gate_probability: float = 0.10,
    ) -> None:
        super().__init__()
        if not 0.0 < initial_gate_probability < 1.0:
            raise ValueError("initial_gate_probability must be in (0, 1)")
        self.frozen_base = frozen_base
        for parameter in self.frozen_base.parameters():
            parameter.requires_grad_(False)
        self.frozen_base.eval()

        architecture = dict(frozen_base.architecture)
        hidden_dim = int(architecture["hidden_dim"])
        dropout = float(architecture["dropout"])
        bottleneck_dim = max(32, hidden_dim // 2)
        self.solvent_residual_head = nn.Sequential(
            nn.Linear(4 * hidden_dim, bottleneck_dim),
            nn.SiLU(),
            nn.LayerNorm(bottleneck_dim),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, 1),
        )
        final = self.solvent_residual_head[-1]
        if not isinstance(final, nn.Linear):
            raise AssertionError("Unexpected Stage-E-A residual output layer")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

        self.charge_type_gate = bool(charge_type_gate)
        self.initial_gate_probability = float(initial_gate_probability)
        if self.charge_type_gate:
            initial_logit = math.log(
                initial_gate_probability / (1.0 - initial_gate_probability)
            )
            self.gate_parameters = nn.Embedding(len(SITE_TYPE_NAMES), 2)
            with torch.no_grad():
                self.gate_parameters.weight[:, 0].fill_(initial_logit)
                self.gate_parameters.weight[:, 1].zero_()
        else:
            self.gate_parameters = None

        self.architecture = {
            "schema_version": STAGE_E_A_MODEL_SCHEMA_VERSION,
            "frozen_base_architecture": architecture,
            "hidden_dim": hidden_dim,
            "residual_features": [
                "site_times_continuous_solvent",
                "global_xtb_times_continuous_solvent",
                "continuous_solvent",
                "solvent_embedding",
            ],
            "residual_head": (
                "four_h_to_bottleneck_silu_layernorm_dropout_to_zero_scalar"
            ),
            "residual_bottleneck_dim": bottleneck_dim,
            "residual_output_initialization": "exact_zero",
            "graph_only_or_site_only_residual_bias": False,
            "charge_type_gate": self.charge_type_gate,
            "gate_formula": (
                "sigmoid(type_intercept + "
                "type_charge_slope * standardized_formal_charge)"
                if self.charge_type_gate
                else "constant_one"
            ),
            "initial_gate_probability": (
                self.initial_gate_probability if self.charge_type_gate else 1.0
            ),
            "base_parameters_trainable": False,
            "base_forced_eval_mode": True,
        }

    def train(
        self,
        mode: bool = True,
    ) -> "MayrSiteNProtectedSolventResidualModel":
        """Train only the new path while keeping C2 dropout disabled."""

        super().train(mode)
        self.frozen_base.eval()
        return self

    def forward(self, inputs: SiteNModelInputs) -> StageEAOutput:
        self.frozen_base.eval()
        with torch.no_grad():
            encoded = self.frozen_base.encode_fused_features(inputs)
            base_prediction = self.frozen_base.regression_head(encoded.fused).squeeze(
                -1
            )
        hidden_dim = int(self.architecture["hidden_dim"])
        chunks = encoded.fused.split(hidden_dim, dim=-1)
        if len(chunks) != 6:
            raise ValueError("Stage-E-A requires frozen C2 fused 6h")
        _, site, solvent_continuous, solvent_embedding, _, global_xtb = chunks
        residual_features = torch.cat(
            (
                site * solvent_continuous,
                global_xtb * solvent_continuous,
                solvent_continuous,
                solvent_embedding,
            ),
            dim=-1,
        )
        raw_residual = self.solvent_residual_head(residual_features.detach()).squeeze(
            -1
        )
        if self.gate_parameters is None:
            gate = torch.ones_like(raw_residual)
        else:
            gate_coefficients = self.gate_parameters(inputs.site_type_index)
            standardized_charge = inputs.molecular_formal_charge[
                inputs.site_graph_index, 0
            ]
            gate = torch.sigmoid(
                gate_coefficients[:, 0] + gate_coefficients[:, 1] * standardized_charge
            )
        applied_residual = raw_residual * gate
        return StageEAOutput(
            n_prediction_standardized=base_prediction + applied_residual,
            node_embeddings=encoded.node_embeddings,
            graph_pool=encoded.graph_pool,
            site_embeddings=encoded.site_embeddings,
            site_summary=encoded.site_summary,
            frozen_base_prediction_standardized=base_prediction,
            raw_residual_standardized=raw_residual,
            applied_residual_standardized=applied_residual,
            residual_gate=gate,
            gate_is_learned=self.gate_parameters is not None,
        )


def zero_residual_output_is_exact(
    model: MayrSiteNProtectedSolventResidualModel,
) -> bool:
    """Return whether the residual output layer remains exact-zero."""

    final = model.solvent_residual_head[-1]
    return bool(
        isinstance(final, nn.Linear)
        and torch.count_nonzero(final.weight).item() == 0
        and torch.count_nonzero(final.bias).item() == 0
    )


def frozen_base_parameters_are_frozen(
    model: MayrSiteNProtectedSolventResidualModel,
) -> bool:
    """Return whether every C2 parameter is excluded from optimization."""

    return all(
        not parameter.requires_grad for parameter in model.frozen_base.parameters()
    )


def trainable_parameter_count(
    model: MayrSiteNProtectedSolventResidualModel,
) -> int:
    """Count only parameters eligible for Stage-E-A optimization."""

    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


@dataclass(frozen=True, slots=True)
class SolventCenterGroup:
    """Same typed site measured in at least two distinct solvents."""

    group_id: str
    connectivity_id: str
    site_object_id: str
    site_type: str
    target_ids: tuple[str, ...]
    context_ids: tuple[str, ...]
    solvents: tuple[str, ...]

    def to_json(self) -> dict[str, object]:
        return asdict(self)


def stage_e_a_solvent_center_groups(
    examples: Sequence[SiteNExample],
) -> tuple[tuple[SolventCenterGroup, ...], dict[str, object]]:
    """Build train-only groups used by the zero-mean residual constraint."""

    grouped: dict[
        tuple[str, str, str],
        list[tuple[str, str, str]],
    ] = defaultdict(list)
    seen_targets: set[str] = set()
    for example in examples:
        for target_id, site_object_id, site_type in zip(
            example.target_ids,
            example.site_object_ids,
            example.site_types,
            strict=True,
        ):
            target = str(target_id)
            if target in seen_targets:
                raise ValueError("Duplicate target id in Stage-E-A center input")
            seen_targets.add(target)
            grouped[
                (
                    str(example.connectivity_id),
                    str(site_object_id),
                    str(site_type),
                )
            ].append(
                (
                    target,
                    str(example.context_id),
                    str(example.solvent_raw),
                )
            )

    groups: list[SolventCenterGroup] = []
    for (connectivity, site_object, site_type), values in sorted(grouped.items()):
        ordered = sorted(values)
        solvents = tuple(value[2] for value in ordered)
        contexts = tuple(value[1] for value in ordered)
        if len(set(solvents)) < 2 or len(set(contexts)) < 2:
            continue
        groups.append(
            SolventCenterGroup(
                group_id=f"{connectivity}|{site_object}|{site_type}",
                connectivity_id=connectivity,
                site_object_id=site_object,
                site_type=site_type,
                target_ids=tuple(value[0] for value in ordered),
                context_ids=contexts,
                solvents=solvents,
            )
        )
    audit: dict[str, object] = {
        "schema_version": "nucpred.mayr-stage-e-a-center-groups.v1",
        "group_count": len(groups),
        "target_count": len(
            {target for group in groups for target in group.target_ids}
        ),
        "connectivity_count": len({group.connectivity_id for group in groups}),
        "site_type_group_counts": {
            site_type: sum(group.site_type == site_type for group in groups)
            for site_type in SITE_TYPE_NAMES
        },
        "fit_roles": ["train"],
        "validation_or_test_target_used": False,
        "same_connectivity_site_and_type_required": True,
        "different_context_and_solvent_required": True,
    }
    return tuple(groups), audit


def centered_residual_loss(
    output: StageEAOutput,
    batch: SiteNTrainingBatch,
    groups: Sequence[SolventCenterGroup],
) -> tuple[torch.Tensor, int]:
    """Penalize nonzero mean applied residual within complete train groups."""

    target_index = {
        target_id: index for index, target_id in enumerate(batch.target_ids)
    }
    losses: list[torch.Tensor] = []
    for group in groups:
        if not all(target_id in target_index for target_id in group.target_ids):
            continue
        indices = torch.tensor(
            [target_index[target_id] for target_id in group.target_ids],
            dtype=torch.long,
            device=output.applied_residual_standardized.device,
        )
        losses.append(output.applied_residual_standardized[indices].mean().square())
    if not losses:
        return output.n_prediction_standardized.sum() * 0.0, 0
    return torch.stack(losses).mean(), len(losses)


def stage_e_a_site_n_loss(
    output: StageEAOutput,
    batch: SiteNTrainingBatch,
    target_weights: Mapping[str, float],
    *,
    ranking_weight: float,
    paired_solvent_pairs: Sequence[PairedSolventDefinition],
    paired_solvent_weight: float,
    center_groups: Sequence[SolventCenterGroup],
    center_penalty_weight: float,
    residual_shrinkage_weight: float,
    gate_shrinkage_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | int]]:
    """Protected weighted N objective with solvent and centering terms."""

    penalties = (
        paired_solvent_weight,
        center_penalty_weight,
        residual_shrinkage_weight,
        gate_shrinkage_weight,
    )
    if any(float(value) < 0.0 for value in penalties):
        raise ValueError("Stage-E-A loss weights cannot be negative")
    weights = output.n_prediction_standardized.new_tensor(
        [float(target_weights[target_id]) for target_id in batch.target_ids]
    )
    if (
        weights.shape != output.n_prediction_standardized.shape
        or not bool(torch.isfinite(weights).all())
        or bool((weights <= 0.0).any())
    ):
        raise ValueError("Stage-E-A target weights are invalid")
    squared = (output.n_prediction_standardized - batch.n_target_standardized).square()
    regression = torch.sum(weights * squared) / weights.sum().clamp_min(1e-12)
    ranking, ranking_pair_count = within_context_ranking_loss(
        output.n_prediction_standardized,
        batch.n_target_standardized,
        batch.inputs.site_graph_index,
    )
    paired_delta, paired_delta_count = paired_solvent_delta_loss(
        output,
        batch,
        paired_solvent_pairs,
    )
    center, center_group_count = centered_residual_loss(
        output,
        batch,
        center_groups,
    )
    residual_shrinkage = output.raw_residual_standardized.square().mean()
    if output.gate_is_learned:
        gate_shrinkage = output.residual_gate.square().mean()
    else:
        gate_shrinkage = output.n_prediction_standardized.sum() * 0.0
    total = (
        regression
        + float(ranking_weight) * ranking
        + float(paired_solvent_weight) * paired_delta
        + float(center_penalty_weight) * center
        + float(residual_shrinkage_weight) * residual_shrinkage
        + float(gate_shrinkage_weight) * gate_shrinkage
    )
    if not bool(torch.isfinite(total)):
        raise ValueError("Stage-E-A loss became non-finite")
    return total, {
        "regression": regression,
        "ranking": ranking,
        "paired_solvent_delta": paired_delta,
        "center_penalty": center,
        "residual_shrinkage": residual_shrinkage,
        "gate_shrinkage": gate_shrinkage,
        "ranking_pairs": ranking_pair_count,
        "paired_solvent_pairs": paired_delta_count,
        "center_groups": center_group_count,
    }
