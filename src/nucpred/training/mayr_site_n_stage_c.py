"""Stage-C conditional-N model and preregistered difficult-group weighting."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import numpy as np
import torch
from torch import nn

from nucpred.training.mayr_site_n import (
    MayrSiteNModel,
    SiteNExample,
    SiteNOutput,
    SiteNTrainingBatch,
    within_context_ranking_loss,
)


STAGE_C_INTERACTION_SCHEMA_VERSION = "nucpred.mayr-site-n-stage-c-interaction-model.v1"


class MayrSiteNInteractionModel(MayrSiteNModel):
    """Base model plus a zero-start explicit encoded interaction residual."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        hidden_dim = int(self.architecture["hidden_dim"])
        dropout = float(self.architecture["dropout"])
        self.interaction_residual = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        final = self.interaction_residual[-1]
        if not isinstance(final, nn.Linear):
            raise AssertionError("Unexpected interaction residual output layer")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        self.architecture = {
            **self.architecture,
            "schema_version": STAGE_C_INTERACTION_SCHEMA_VERSION,
            "base_model_schema_version": self.architecture["schema_version"],
            "interaction_features": [
                "site_times_charge",
                "site_times_continuous_solvent",
                "global_xtb_times_charge",
                "global_xtb_times_continuous_solvent",
            ],
            "interaction_residual": "four_h_to_h_silu_dropout_to_zero_scalar",
            "interaction_output_initialization": "exact_zero",
        }

    def forward(self, inputs):  # type annotation inherited conceptually
        encoded = self.encode_fused_features(inputs)
        hidden_dim = int(self.architecture["hidden_dim"])
        chunks = encoded.fused.split(hidden_dim, dim=-1)
        if len(chunks) != 6:
            raise ValueError("Stage-C interaction model requires fused 6h")
        _, site, solvent_continuous, _, charge, global_xtb = chunks
        interactions = torch.cat(
            (
                site * charge,
                site * solvent_continuous,
                global_xtb * charge,
                global_xtb * solvent_continuous,
            ),
            dim=-1,
        )
        prediction = self.regression_head(encoded.fused).squeeze(-1)
        prediction = prediction + self.interaction_residual(interactions).squeeze(-1)
        return SiteNOutput(
            n_prediction_standardized=prediction,
            node_embeddings=encoded.node_embeddings,
            graph_pool=encoded.graph_pool,
            site_embeddings=encoded.site_embeddings,
            site_summary=encoded.site_summary,
        )


def _inverse_sqrt_binary_weights(mask: np.ndarray) -> np.ndarray:
    """Return mean-one inverse-sqrt group-frequency weights."""

    values = np.asarray(mask, dtype=bool).reshape(-1)
    total = int(values.size)
    positive = int(values.sum())
    negative = total - positive
    if total == 0 or positive == 0 or negative == 0:
        return np.ones(total, dtype=np.float64)
    result = np.empty(total, dtype=np.float64)
    result[values] = math.sqrt(total / (2.0 * positive))
    result[~values] = math.sqrt(total / (2.0 * negative))
    return result / float(result.mean())


def _bounded_mean_one(
    values: np.ndarray,
    *,
    maximum_weight: float,
) -> np.ndarray:
    """Scale positive weights to mean one without breaching the final cap."""

    weights = np.asarray(values, dtype=np.float64).reshape(-1)
    if weights.size == 0:
        return weights
    if not np.isfinite(weights).all() or bool((weights <= 0.0).any()):
        raise ValueError("Stage-C weights must be finite and positive")
    if maximum_weight < 1.0:
        raise ValueError("maximum_weight must be at least one")
    if maximum_weight == 1.0:
        return np.ones_like(weights)

    def bounded_mean(scale: float) -> float:
        return float(np.minimum(maximum_weight, scale * weights).mean())

    lower = 0.0
    upper = 1.0
    while bounded_mean(upper) < 1.0:
        upper *= 2.0
    for _ in range(100):
        midpoint = 0.5 * (lower + upper)
        if bounded_mean(midpoint) < 1.0:
            lower = midpoint
        else:
            upper = midpoint
    result = np.minimum(maximum_weight, upper * weights)
    if (
        not math.isclose(float(result.mean()), 1.0, abs_tol=1e-12)
        or float(result.max()) > maximum_weight + 1e-12
    ):
        raise ValueError("Bounded mean-one normalization did not converge")
    return result


def stage_c_target_weights(
    examples: Sequence[SiteNExample],
    *,
    use_h1: bool,
    use_h2: bool,
    maximum_weight: float = 3.0,
) -> tuple[dict[str, float], dict[str, object]]:
    """Build the frozen C1/C2/C4 target-weight map from train data only."""

    if maximum_weight < 1.0:
        raise ValueError("maximum_weight must be at least one")
    target_ids: list[str] = []
    h1: list[bool] = []
    h2: list[bool] = []
    for example in examples:
        small_negative = example.num_nodes <= 10 and example.model_formal_charge < 0.0
        for target_id, n_value in zip(
            example.target_ids,
            example.n_targets,
            strict=True,
        ):
            target_ids.append(str(target_id))
            h1.append(bool(small_negative))
            h2.append(bool(float(n_value) >= 15.0))
    weights = np.ones(len(target_ids), dtype=np.float64)
    if use_h1:
        weights *= _inverse_sqrt_binary_weights(np.asarray(h1, dtype=bool))
    if use_h2:
        weights *= _inverse_sqrt_binary_weights(np.asarray(h2, dtype=bool))
    weights = _bounded_mean_one(
        weights,
        maximum_weight=float(maximum_weight),
    )
    mapping = {
        target_id: float(weight)
        for target_id, weight in zip(target_ids, weights, strict=True)
    }
    if len(mapping) != len(target_ids):
        raise ValueError("Stage-C target ids are not unique within train")
    audit: dict[str, object] = {
        "schema_version": "nucpred.mayr-stage-c-target-weights.v1",
        "method": ("inverse_sqrt_binary_group_frequency_normalized_mean_one_cap_three"),
        "target_count": len(target_ids),
        "use_h1": bool(use_h1),
        "use_h2": bool(use_h2),
        "h1_target_count": int(sum(h1)),
        "h2_target_count": int(sum(h2)),
        "maximum_weight": (float(max(weights)) if len(weights) else float("nan")),
        "minimum_weight": (float(min(weights)) if len(weights) else float("nan")),
        "mean_weight": (float(np.mean(weights)) if len(weights) else float("nan")),
        "configured_cap": float(maximum_weight),
        "final_cap_satisfied": bool(
            not len(weights) or float(max(weights)) <= float(maximum_weight) + 1e-12
        ),
        "normalization_solver": "monotone_bounded_rescaling",
        "applies_to": "training_mse_only",
    }
    return mapping, audit


def weighted_site_n_loss(
    output: SiteNOutput,
    batch: SiteNTrainingBatch,
    target_weights: Mapping[str, float],
    *,
    ranking_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | int]]:
    """Weighted MSE plus the unchanged within-context ranking loss."""

    weights = output.n_prediction_standardized.new_tensor(
        [float(target_weights[target_id]) for target_id in batch.target_ids]
    )
    if weights.shape != output.n_prediction_standardized.shape:
        raise ValueError("Stage-C target weights do not align with predictions")
    squared = (output.n_prediction_standardized - batch.n_target_standardized).square()
    regression = torch.sum(weights * squared) / weights.sum().clamp_min(1e-12)
    ranking, pair_count = within_context_ranking_loss(
        output.n_prediction_standardized,
        batch.n_target_standardized,
        batch.inputs.site_graph_index,
    )
    total = regression + float(ranking_weight) * ranking
    if not bool(torch.isfinite(total)):
        raise ValueError("Stage-C weighted loss became non-finite")
    return total, {
        "regression": regression,
        "ranking": ranking,
        "ranking_pairs": pair_count,
    }


def zero_interaction_output_is_exact(
    model: MayrSiteNInteractionModel,
) -> bool:
    """Audit that the final residual layer still emits exact zero."""

    final = model.interaction_residual[-1]
    if not isinstance(final, nn.Linear):
        return False
    return bool(
        torch.count_nonzero(final.weight).item() == 0
        and torch.count_nonzero(final.bias).item() == 0
    )
