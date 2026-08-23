"""Stage-D development-only mechanisms for conditional Mayr N."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import combinations

import numpy as np
import torch
from torch import nn

from nucpred.training.mayr_site_n import (
    SITE_TYPE_NAMES,
    MayrSiteNModel,
    SiteNExample,
    SiteNOutput,
    SiteNTrainingBatch,
    within_context_ranking_loss,
)
from nucpred.training.mayr_site_n_stage_c import _bounded_mean_one


STAGE_D_TYPE_EXPERT_SCHEMA_VERSION = (
    "nucpred.mayr-site-n-stage-d-type-residual-model.v1"
)


class MayrSiteNTypeResidualModel(MayrSiteNModel):
    """Shared base model plus small zero-start residual experts by query type."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        hidden_dim = int(self.architecture["hidden_dim"])
        dropout = float(self.architecture["dropout"])
        bottleneck_dim = max(16, hidden_dim // 4)
        self.type_residual_experts = nn.ModuleDict(
            {
                site_type: nn.Sequential(
                    nn.Linear(6 * hidden_dim, bottleneck_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(bottleneck_dim, 1),
                )
                for site_type in SITE_TYPE_NAMES
            }
        )
        for expert in self.type_residual_experts.values():
            final = expert[-1]
            if not isinstance(final, nn.Linear):
                raise AssertionError("Unexpected type-residual output layer")
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
        self.architecture = {
            **self.architecture,
            "schema_version": STAGE_D_TYPE_EXPERT_SCHEMA_VERSION,
            "base_model_schema_version": self.architecture["schema_version"],
            "type_residual_expert": (
                "per_query_type_6h_to_bottleneck_silu_dropout_to_zero_scalar"
            ),
            "type_residual_bottleneck_dim": bottleneck_dim,
            "type_residual_output_initialization": "exact_zero",
            "query_type_is_explicit_non_oracle_input": True,
        }

    def forward(self, inputs):  # type annotation inherited conceptually
        encoded = self.encode_fused_features(inputs)
        prediction = self.regression_head(encoded.fused).squeeze(-1)
        residual = torch.zeros_like(prediction)
        for index, site_type in enumerate(SITE_TYPE_NAMES):
            selected = inputs.site_type_index.eq(index)
            if bool(selected.any()):
                residual[selected] = self.type_residual_experts[site_type](
                    encoded.fused[selected]
                ).squeeze(-1)
        return SiteNOutput(
            n_prediction_standardized=prediction + residual,
            node_embeddings=encoded.node_embeddings,
            graph_pool=encoded.graph_pool,
            site_embeddings=encoded.site_embeddings,
            site_summary=encoded.site_summary,
        )


def zero_type_residual_output_is_exact(
    model: MayrSiteNTypeResidualModel,
) -> bool:
    """Return whether every type expert still emits exact zero at initialization."""

    for expert in model.type_residual_experts.values():
        final = expert[-1]
        if not isinstance(final, nn.Linear):
            return False
        if (
            torch.count_nonzero(final.weight).item() != 0
            or torch.count_nonzero(final.bias).item() != 0
        ):
            return False
    return True


def type_residual_parameter_count(model: MayrSiteNTypeResidualModel) -> int:
    """Count only parameters introduced by the Stage-D type experts."""

    return sum(
        parameter.numel()
        for parameter in model.type_residual_experts.parameters()
    )


def stage_d_two_tail_target_weights(
    examples: Sequence[SiteNExample],
    *,
    tail_power: float = 0.5,
    maximum_weight: float = 3.0,
) -> tuple[dict[str, float], dict[str, object]]:
    """Fit train-only two-sided empirical-CDF rarity weights.

    The score changes smoothly by target rank instead of using the Stage-C
    binary N>=15 cut. Ties receive the same mid-CDF probability. Final weights
    are normalized to mean one under an exact upper cap.
    """

    if not 0.0 < tail_power <= 1.0:
        raise ValueError("tail_power must be in (0, 1]")
    target_ids: list[str] = []
    targets: list[float] = []
    for example in examples:
        for target_id, target in zip(
            example.target_ids,
            example.n_targets,
            strict=True,
        ):
            target_ids.append(str(target_id))
            targets.append(float(target))
    if not target_ids:
        raise ValueError("Cannot fit Stage-D weights without training targets")
    if len(target_ids) != len(set(target_ids)):
        raise ValueError("Stage-D target ids are not unique within train")
    values = np.asarray(targets, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Stage-D targets must be finite")

    unique, inverse, counts = np.unique(
        values,
        return_inverse=True,
        return_counts=True,
    )
    cumulative_before = np.concatenate(([0], np.cumsum(counts)[:-1]))
    mid_cdf_by_unique = (
        cumulative_before.astype(np.float64) + 0.5 * counts
    ) / len(values)
    mid_cdf = mid_cdf_by_unique[inverse]
    two_sided_tail_mass = 2.0 * np.minimum(mid_cdf, 1.0 - mid_cdf)
    minimum_mass = 1.0 / len(values)
    raw = np.maximum(two_sided_tail_mass, minimum_mass) ** (-tail_power)
    weights = _bounded_mean_one(raw, maximum_weight=maximum_weight)
    mapping = {
        target_id: float(weight)
        for target_id, weight in zip(target_ids, weights, strict=True)
    }

    quantiles = {
        str(probability): float(np.quantile(weights, probability))
        for probability in (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)
    }
    low = values < 0.0
    high = values >= 15.0
    audit: dict[str, object] = {
        "schema_version": "nucpred.mayr-stage-d-two-tail-weights.v1",
        "method": "train_only_two_sided_mid_ecdf_inverse_power",
        "target_count": len(values),
        "unique_target_value_count": len(unique),
        "tail_power": float(tail_power),
        "minimum_two_sided_tail_mass": float(minimum_mass),
        "configured_cap": float(maximum_weight),
        "minimum_weight": float(weights.min()),
        "mean_weight": float(weights.mean()),
        "maximum_weight": float(weights.max()),
        "weight_quantiles": quantiles,
        "N_lt_0_target_count": int(low.sum()),
        "N_lt_0_mean_weight": (
            float(weights[low].mean()) if bool(low.any()) else None
        ),
        "N_ge_15_target_count": int(high.sum()),
        "N_ge_15_mean_weight": (
            float(weights[high].mean()) if bool(high.any()) else None
        ),
        "fit_roles": ["train"],
        "validation_or_test_target_used_for_fit": False,
        "applies_to": "training_mse_only",
        "final_cap_satisfied": bool(
            weights.max() <= maximum_weight + 1e-12
        ),
    }
    return mapping, audit


@dataclass(frozen=True, slots=True)
class PairedSolventDefinition:
    """One train-only same-site, different-solvent target pair."""

    pair_id: str
    connectivity_id: str
    site_object_id: str
    site_type: str
    left_target_id: str
    right_target_id: str
    left_context_id: str
    right_context_id: str
    left_solvent: str
    right_solvent: str

    def to_json(self) -> dict[str, object]:
        return asdict(self)


def stage_d_paired_solvent_definitions(
    examples: Sequence[SiteNExample],
) -> tuple[tuple[PairedSolventDefinition, ...], dict[str, object]]:
    """Enumerate train-only same-connectivity/site cross-solvent target pairs."""

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
                raise ValueError("Duplicate target id in paired-solvent input")
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

    pairs: list[PairedSolventDefinition] = []
    for (connectivity, site_object, site_type), records in sorted(
        grouped.items()
    ):
        ordered = sorted(records, key=lambda record: (record[1], record[0]))
        for left, right in combinations(ordered, 2):
            if left[1] == right[1] or left[2] == right[2]:
                continue
            pair_id = (
                f"{connectivity}|{site_object}|{left[0]}|{right[0]}"
            )
            pairs.append(
                PairedSolventDefinition(
                    pair_id=pair_id,
                    connectivity_id=connectivity,
                    site_object_id=site_object,
                    site_type=site_type,
                    left_target_id=left[0],
                    right_target_id=right[0],
                    left_context_id=left[1],
                    right_context_id=right[1],
                    left_solvent=left[2],
                    right_solvent=right[2],
                )
            )
    observed_ids = [pair.pair_id for pair in pairs]
    if len(observed_ids) != len(set(observed_ids)):
        raise ValueError("Paired-solvent definitions are not unique")
    audit: dict[str, object] = {
        "schema_version": "nucpred.mayr-stage-d-paired-solvent-audit.v1",
        "pair_count": len(pairs),
        "site_object_count": len({pair.site_object_id for pair in pairs}),
        "connectivity_count": len({pair.connectivity_id for pair in pairs}),
        "site_type_pair_counts": {
            site_type: sum(pair.site_type == site_type for pair in pairs)
            for site_type in SITE_TYPE_NAMES
        },
        "fit_roles": ["train"],
        "validation_or_test_target_used_for_pairs": False,
        "same_connectivity_and_site_required": True,
        "different_context_and_solvent_required": True,
    }
    return tuple(pairs), audit


def paired_solvent_delta_loss(
    output: SiteNOutput,
    batch: SiteNTrainingBatch,
    pairs: Sequence[PairedSolventDefinition],
) -> tuple[torch.Tensor, int]:
    """Compute MSE on predicted versus observed standardized solvent deltas."""

    target_index = {
        target_id: index for index, target_id in enumerate(batch.target_ids)
    }
    selected = [
        (target_index[pair.left_target_id], target_index[pair.right_target_id])
        for pair in pairs
        if pair.left_target_id in target_index
        and pair.right_target_id in target_index
    ]
    if not selected:
        return output.n_prediction_standardized.sum() * 0.0, 0
    left = torch.tensor(
        [pair[0] for pair in selected],
        dtype=torch.long,
        device=output.n_prediction_standardized.device,
    )
    right = torch.tensor(
        [pair[1] for pair in selected],
        dtype=torch.long,
        device=output.n_prediction_standardized.device,
    )
    prediction_delta = (
        output.n_prediction_standardized[left]
        - output.n_prediction_standardized[right]
    )
    target_delta = (
        batch.n_target_standardized[left]
        - batch.n_target_standardized[right]
    )
    return torch.mean((prediction_delta - target_delta).square()), len(selected)


def stage_d_site_n_loss(
    output: SiteNOutput,
    batch: SiteNTrainingBatch,
    target_weights: Mapping[str, float],
    *,
    ranking_weight: float,
    paired_solvent_pairs: Sequence[PairedSolventDefinition] = (),
    paired_solvent_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | int]]:
    """Weighted N loss with optional train-only paired-solvent auxiliary."""

    if paired_solvent_weight < 0.0:
        raise ValueError("paired_solvent_weight cannot be negative")
    weights = output.n_prediction_standardized.new_tensor(
        [float(target_weights[target_id]) for target_id in batch.target_ids]
    )
    if (
        weights.shape != output.n_prediction_standardized.shape
        or not bool(torch.isfinite(weights).all())
        or bool((weights <= 0.0).any())
    ):
        raise ValueError("Stage-D target weights are invalid")
    squared = (
        output.n_prediction_standardized - batch.n_target_standardized
    ).square()
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
    total = (
        regression
        + float(ranking_weight) * ranking
        + float(paired_solvent_weight) * paired_delta
    )
    if not bool(torch.isfinite(total)):
        raise ValueError("Stage-D loss became non-finite")
    return total, {
        "regression": regression,
        "ranking": ranking,
        "paired_solvent_delta": paired_delta,
        "ranking_pairs": ranking_pair_count,
        "paired_solvent_pairs": paired_delta_count,
    }


def pair_aware_example_groups(
    examples: Sequence[SiteNExample],
    pairs: Sequence[PairedSolventDefinition],
    *,
    batch_size_contexts: int,
    shuffle_seed: int,
) -> list[list[SiteNExample]]:
    """Pack linked cross-solvent contexts together without dropping examples."""

    if batch_size_contexts <= 0:
        raise ValueError("batch_size_contexts must be positive")
    by_context = {str(example.context_id): example for example in examples}
    if len(by_context) != len(examples):
        raise ValueError("Context ids are not unique")
    parent = {context_id: context_id for context_id in by_context}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for pair in pairs:
        if (
            pair.left_context_id not in by_context
            or pair.right_context_id not in by_context
        ):
            raise ValueError("Pair references a context outside the input role")
        union(pair.left_context_id, pair.right_context_id)

    components: dict[str, list[SiteNExample]] = defaultdict(list)
    for context_id, example in by_context.items():
        components[find(context_id)].append(example)
    blocks = [
        sorted(block, key=lambda example: example.context_id)
        for _, block in sorted(components.items())
    ]
    np.random.default_rng(int(shuffle_seed)).shuffle(blocks)
    batches: list[list[SiteNExample]] = []
    current: list[SiteNExample] = []
    for block in blocks:
        if current and len(current) + len(block) > batch_size_contexts:
            batches.append(current)
            current = []
        current.extend(block)
    if current:
        batches.append(current)
    observed = [example.context_id for batch in batches for example in batch]
    if len(observed) != len(examples) or set(observed) != set(by_context):
        raise ValueError("Pair-aware batching changed the context population")
    return batches
