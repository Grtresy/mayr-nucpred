"""Independent multitype validity ranking for Mayr candidate sites.

The rankers in this module never normalize scores across a molecular context.
Reviewed negatives retain endpoint-relative semantics, and the pairwise loss
supports more than one reviewed positive per target.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


RANKER_SITE_TYPES = (
    "atom",
    "atom_group",
    "bond",
    "delocalized_region",
    "transferable_h_group",
)
RANKER_TYPE_TO_INDEX = {
    site_type: index for index, site_type in enumerate(RANKER_SITE_TYPES)
}
LINEAR_BCE = "linear_bce"
MULTITYPE_PAIRWISE = "multitype_pairwise"
RANKER_ARMS = (LINEAR_BCE, MULTITYPE_PAIRWISE)
RANKER_SCHEMA_VERSION = "nucpred.mayr-multitype-site-ranker.v1"
CALIBRATOR_SCHEMA_VERSION = "nucpred.mayr-type-aware-platt.v1"


@dataclass(frozen=True, slots=True)
class RankerFitResult:
    """Frozen development result for one preregistered ranker arm."""

    model: "IndependentSiteRanker"
    audit: dict[str, object]
    validation_logits: np.ndarray


def site_type_indices(values: Sequence[str]) -> torch.Tensor:
    """Encode canonical site type names, failing on ontology drift."""

    unknown = sorted(set(map(str, values)) - set(RANKER_SITE_TYPES))
    if unknown:
        raise ValueError(f"Unknown ranker site types: {unknown}")
    return torch.tensor(
        [RANKER_TYPE_TO_INDEX[str(value)] for value in values],
        dtype=torch.long,
    )


def fit_feature_normalizer(
    features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit a train-only finite mean/scale transform."""

    if features.ndim != 2 or not features.shape[0]:
        raise ValueError("Ranker features must be a non-empty matrix")
    if not bool(torch.isfinite(features).all()):
        raise ValueError("Ranker features contain non-finite values")
    mean = features.mean(dim=0)
    scale = features.std(dim=0, correction=0)
    scale = torch.where(scale > 1e-6, scale, torch.ones_like(scale))
    return mean.detach(), scale.detach()


class IndependentSiteRanker(nn.Module):
    """A linear baseline or shared-plus-type-residual independent logit."""

    def __init__(
        self,
        *,
        input_dim: int,
        arm: str,
        feature_mean: torch.Tensor,
        feature_scale: torch.Tensor,
        hidden_dim: int = 64,
        type_adapter_dim: int = 24,
    ) -> None:
        super().__init__()
        if arm not in RANKER_ARMS:
            raise ValueError(f"Unsupported ranker arm: {arm}")
        if tuple(feature_mean.shape) != (input_dim,) or tuple(feature_scale.shape) != (
            input_dim,
        ):
            raise ValueError("Ranker normalizer shape does not match input_dim")
        self.arm = arm
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.type_adapter_dim = int(type_adapter_dim)
        self.register_buffer("feature_mean", feature_mean.detach().clone())
        self.register_buffer("feature_scale", feature_scale.detach().clone())

        if arm == LINEAR_BCE:
            self.linear = nn.Linear(
                input_dim + len(RANKER_SITE_TYPES),
                1,
            )
        else:
            self.shared_projection = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.SiLU(),
                nn.LayerNorm(hidden_dim),
            )
            self.shared_logit = nn.Linear(hidden_dim, 1)
            self.type_adapters = nn.ModuleDict(
                {
                    site_type: nn.Sequential(
                        nn.Linear(hidden_dim, type_adapter_dim),
                        nn.SiLU(),
                        nn.Linear(type_adapter_dim, 1),
                    )
                    for site_type in RANKER_SITE_TYPES
                }
            )

        self.architecture = {
            "schema_version": RANKER_SCHEMA_VERSION,
            "arm": arm,
            "input_dim": int(input_dim),
            "hidden_dim": int(hidden_dim),
            "type_adapter_dim": int(type_adapter_dim),
            "site_types": list(RANKER_SITE_TYPES),
            "candidate_scores_independent": True,
            "candidate_softmax_used": False,
            "multi_positive_supported": True,
        }

    def forward(
        self,
        features: torch.Tensor,
        type_index: torch.Tensor,
    ) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError("Ranker feature shape changed")
        if type_index.shape != (features.shape[0],):
            raise ValueError("Ranker type-index shape changed")
        normalized = (features - self.feature_mean) / self.feature_scale
        if self.arm == LINEAR_BCE:
            one_hot = F.one_hot(
                type_index,
                num_classes=len(RANKER_SITE_TYPES),
            ).to(dtype=normalized.dtype)
            return self.linear(torch.cat((normalized, one_hot), dim=-1)).squeeze(-1)

        hidden = self.shared_projection(normalized)
        logits = self.shared_logit(hidden).squeeze(-1)
        residual = torch.zeros_like(logits)
        for index, site_type in enumerate(RANKER_SITE_TYPES):
            selected = type_index.eq(index)
            if bool(selected.any()):
                residual[selected] = self.type_adapters[site_type](
                    hidden[selected]
                ).squeeze(-1)
        return logits + residual


def reviewed_pairwise_logistic_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    target_group_index: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int]:
    """Rank every reviewed positive above every reviewed negative per target.

    A group can contain multiple positives. Groups without both classes are
    intentionally skipped because unknown candidates are never manufactured as
    negatives.
    """

    if logits.ndim != 1 or labels.shape != logits.shape:
        raise ValueError("Pairwise logits and labels must be aligned vectors")
    if target_group_index.shape != logits.shape:
        raise ValueError("Pairwise target groups must align with logits")
    if sample_weights is None:
        sample_weights = torch.ones_like(logits)
    if sample_weights.shape != logits.shape:
        raise ValueError("Pairwise sample weights must align with logits")

    losses: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    pair_count = 0
    for group_index in torch.unique(target_group_index, sorted=True):
        selected = target_group_index.eq(group_index)
        positives = torch.nonzero(
            selected & labels.eq(1),
            as_tuple=False,
        ).flatten()
        negatives = torch.nonzero(
            selected & labels.eq(0),
            as_tuple=False,
        ).flatten()
        if not len(positives) or not len(negatives):
            continue
        difference = logits[positives, None] - logits[negatives[None, :]]
        pair_weight = torch.sqrt(
            sample_weights[positives, None] * sample_weights[negatives[None, :]]
        )
        losses.append(F.softplus(-difference).reshape(-1))
        weights.append(pair_weight.reshape(-1))
        pair_count += int(difference.numel())
    if not losses:
        return logits.sum() * 0.0, 0
    loss_values = torch.cat(losses)
    pair_weights = torch.cat(weights)
    loss = torch.sum(loss_values * pair_weights) / pair_weights.sum().clamp_min(1e-12)
    return loss, pair_count


def balanced_site_label_connectivity_weights(
    *,
    labels: Sequence[int],
    site_types: Sequence[str],
    connectivity_ids: Sequence[str],
    sampling_weights: Sequence[float],
) -> torch.Tensor:
    """Give equal mass to type/label cells, then to connectivities.

    Review inverse-probability weights only distribute mass within a
    connectivity. This prevents large molecules or heavily sampled endpoints
    from dominating the objective.
    """

    labels_array = np.asarray(labels, dtype=int)
    types_array = np.asarray(site_types, dtype=str)
    connectivity_array = np.asarray(connectivity_ids, dtype=str)
    sampling_array = np.asarray(sampling_weights, dtype=float)
    length = len(labels_array)
    if not (
        len(types_array) == len(connectivity_array) == len(sampling_array) == length
    ):
        raise ValueError("Weight inputs have different lengths")
    if not length or not set(labels_array) <= {0, 1}:
        raise ValueError("Weights require non-empty binary labels")
    if not np.isfinite(sampling_array).all() or (sampling_array <= 0).any():
        raise ValueError("Sampling weights must be finite and positive")

    cells = sorted(set(zip(types_array, labels_array, strict=True)))
    values = np.zeros(length, dtype=np.float64)
    for site_type, label in cells:
        cell = (types_array == site_type) & (labels_array == label)
        connectivities = sorted(set(connectivity_array[cell]))
        for connectivity_id in connectivities:
            selected = cell & (connectivity_array == connectivity_id)
            within = sampling_array[selected]
            within = within / within.sum()
            values[selected] = within / len(connectivities) / len(cells)
    if not math.isclose(float(values.sum()), 1.0, abs_tol=1e-10):
        raise ValueError("Balanced ranker weights do not sum to one")
    return torch.tensor(values, dtype=torch.float32)


def retrieval_metrics(
    *,
    labels: Sequence[int],
    scores: Sequence[float],
    target_ids: Sequence[str],
) -> dict[str, float | int]:
    """Compute multi-positive exact retrieval metrics on reviewed candidates."""

    label_array = np.asarray(labels, dtype=int)
    score_array = np.asarray(scores, dtype=float)
    target_array = np.asarray(target_ids, dtype=str)
    reciprocal: list[float] = []
    top1 = 0
    top3 = 0
    eligible = 0
    for target_id in sorted(set(target_array)):
        selected = target_array == target_id
        group_labels = label_array[selected]
        if not (group_labels == 1).any() or not (group_labels == 0).any():
            continue
        group_scores = score_array[selected]
        order = np.argsort(-group_scores, kind="stable")
        ranks = np.flatnonzero(group_labels[order] == 1) + 1
        best_rank = int(ranks.min())
        reciprocal.append(1.0 / best_rank)
        top1 += int(best_rank <= 1)
        top3 += int(best_rank <= 3)
        eligible += 1
    return {
        "eligible_target_count": eligible,
        "mrr": float(np.mean(reciprocal)) if reciprocal else float("nan"),
        "top1_recall": top1 / eligible if eligible else float("nan"),
        "top3_recall": top3 / eligible if eligible else float("nan"),
    }


def _selection_key(
    metrics: Mapping[str, float | int],
    *,
    average_precision: float,
) -> tuple[float, float, float]:
    return (
        float(metrics["mrr"]),
        float(metrics["top1_recall"]),
        float(average_precision),
    )


def fit_ranker_arm(
    *,
    arm: str,
    train_features: torch.Tensor,
    train_type_index: torch.Tensor,
    train_labels: torch.Tensor,
    train_group_index: torch.Tensor,
    train_weights: torch.Tensor,
    validation_features: torch.Tensor,
    validation_type_index: torch.Tensor,
    validation_labels: np.ndarray,
    validation_target_ids: Sequence[str],
    validation_average_precision: Any,
    hidden_dim: int,
    type_adapter_dim: int,
    learning_rate: float,
    weight_decay: float,
    maximum_epochs: int,
    minimum_epochs: int,
    evaluation_interval: int,
    patience_evaluations: int,
    pairwise_loss_weight: float,
    gradient_clip_norm: float,
    seed: int,
) -> RankerFitResult:
    """Fit one arm on train and select its epoch on validation retrieval."""

    if arm not in RANKER_ARMS:
        raise ValueError(f"Unsupported ranker arm: {arm}")
    torch.manual_seed(int(seed))
    mean, scale = fit_feature_normalizer(train_features)
    model = IndependentSiteRanker(
        input_dim=int(train_features.shape[1]),
        arm=arm,
        feature_mean=mean,
        feature_scale=scale,
        hidden_dim=hidden_dim,
        type_adapter_dim=type_adapter_dim,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_key = (-math.inf, -math.inf, -math.inf)
    best_epoch = 0
    stale_evaluations = 0
    pair_count = 0
    final_bce = math.nan
    final_pairwise = math.nan
    epochs_completed = 0

    for epoch in range(1, maximum_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_features, train_type_index)
        bce_values = F.binary_cross_entropy_with_logits(
            logits,
            train_labels,
            reduction="none",
        )
        bce = torch.sum(bce_values * train_weights) / train_weights.sum()
        pairwise, pair_count = reviewed_pairwise_logistic_loss(
            logits,
            train_labels,
            train_group_index,
            train_weights,
        )
        pairwise_weight = pairwise_loss_weight if arm == MULTITYPE_PAIRWISE else 0.0
        loss = bce + pairwise_weight * pairwise
        if not bool(torch.isfinite(loss)):
            raise ValueError("Ranker training loss became non-finite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        epochs_completed = epoch
        final_bce = float(bce.detach())
        final_pairwise = float(pairwise.detach())

        should_evaluate = epoch % evaluation_interval == 0 or epoch == maximum_epochs
        if not should_evaluate:
            continue
        model.eval()
        with torch.no_grad():
            validation_logits_tensor = model(
                validation_features,
                validation_type_index,
            )
        validation_logits = validation_logits_tensor.cpu().numpy()
        retrieval = retrieval_metrics(
            labels=validation_labels,
            scores=validation_logits,
            target_ids=validation_target_ids,
        )
        average_precision = float(
            validation_average_precision(
                validation_labels,
                validation_logits,
            )
        )
        key = _selection_key(
            retrieval,
            average_precision=average_precision,
        )
        if key > best_key:
            best_key = key
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            stale_evaluations = 0
        else:
            stale_evaluations += 1
        if epoch >= minimum_epochs and stale_evaluations >= patience_evaluations:
            break

    if best_state is None:
        raise ValueError("Ranker fit did not produce a validation checkpoint")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    with torch.no_grad():
        validation_logits = (
            model(
                validation_features,
                validation_type_index,
            )
            .cpu()
            .numpy()
        )
    retrieval = retrieval_metrics(
        labels=validation_labels,
        scores=validation_logits,
        target_ids=validation_target_ids,
    )
    average_precision = float(
        validation_average_precision(validation_labels, validation_logits)
    )
    return RankerFitResult(
        model=model,
        validation_logits=validation_logits,
        audit={
            "schema_version": "nucpred.mayr-site-ranker-fit-audit.v1",
            "arm": arm,
            "seed": int(seed),
            "epochs_completed": epochs_completed,
            "best_epoch": best_epoch,
            "best_selection_key": list(best_key),
            "validation_retrieval": retrieval,
            "validation_average_precision": average_precision,
            "final_train_weighted_bce": final_bce,
            "final_train_pairwise_loss": final_pairwise,
            "reviewed_pair_count": pair_count,
            "pairwise_loss_weight": (
                pairwise_loss_weight if arm == MULTITYPE_PAIRWISE else 0.0
            ),
        },
    )


class TypeAwarePlattCalibrator(nn.Module):
    """Positive-slope Platt scaling with partially pooled type intercepts."""

    def __init__(self) -> None:
        super().__init__()
        self.raw_slope = nn.Parameter(
            torch.tensor(math.log(math.expm1(1.0)), dtype=torch.float64)
        )
        self.global_bias = nn.Parameter(torch.zeros((), dtype=torch.float64))
        self.type_offsets = nn.Parameter(
            torch.zeros(len(RANKER_SITE_TYPES), dtype=torch.float64)
        )

    @property
    def slope(self) -> torch.Tensor:
        return F.softplus(self.raw_slope) + 1e-6

    def calibrated_logits(
        self,
        logits: torch.Tensor,
        type_index: torch.Tensor,
    ) -> torch.Tensor:
        centered_offsets = self.type_offsets - self.type_offsets.mean()
        return (
            self.slope * logits.to(dtype=torch.float64)
            + self.global_bias
            + centered_offsets[type_index]
        )

    def forward(
        self,
        logits: torch.Tensor,
        type_index: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sigmoid(self.calibrated_logits(logits, type_index))

    def to_payload(self) -> dict[str, object]:
        centered = self.type_offsets.detach() - self.type_offsets.detach().mean()
        return {
            "schema_version": CALIBRATOR_SCHEMA_VERSION,
            "site_types": list(RANKER_SITE_TYPES),
            "positive_slope": float(self.slope.detach()),
            "global_bias": float(self.global_bias.detach()),
            "type_intercepts": {
                site_type: float(centered[index])
                for index, site_type in enumerate(RANKER_SITE_TYPES)
            },
            "probabilities_independent": True,
            "candidate_softmax_used": False,
        }

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, object],
    ) -> "TypeAwarePlattCalibrator":
        if payload.get("schema_version") != CALIBRATOR_SCHEMA_VERSION:
            raise ValueError("Calibrator schema changed")
        if tuple(payload.get("site_types", ())) != RANKER_SITE_TYPES:
            raise ValueError("Calibrator site types changed")
        slope = float(payload["positive_slope"])
        if not math.isfinite(slope) or slope <= 0:
            raise ValueError("Calibrator slope must be finite and positive")
        intercepts = payload["type_intercepts"]
        if not isinstance(intercepts, Mapping):
            raise ValueError("Calibrator type intercepts are not a mapping")
        model = cls()
        with torch.no_grad():
            model.raw_slope.copy_(
                torch.tensor(
                    math.log(math.expm1(max(slope - 1e-6, 1e-9))),
                    dtype=torch.float64,
                )
            )
            model.global_bias.copy_(
                torch.tensor(float(payload["global_bias"]), dtype=torch.float64)
            )
            model.type_offsets.copy_(
                torch.tensor(
                    [float(intercepts[site_type]) for site_type in RANKER_SITE_TYPES],
                    dtype=torch.float64,
                )
            )
        model.eval()
        return model


def fit_type_aware_platt(
    *,
    logits: torch.Tensor,
    type_index: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    l2_type_offset: float,
    l2_log_slope: float,
    maximum_iterations: int,
) -> tuple[TypeAwarePlattCalibrator, dict[str, object]]:
    """Fit the frozen validation-only calibrator."""

    calibrator = TypeAwarePlattCalibrator()
    labels64 = labels.to(dtype=torch.float64)
    weights64 = weights.to(dtype=torch.float64)
    logits64 = logits.detach().to(dtype=torch.float64)
    type_index = type_index.detach()
    prevalence = float(
        torch.sum(labels64 * weights64) / weights64.sum().clamp_min(1e-12)
    )
    prevalence = min(max(prevalence, 1e-6), 1.0 - 1e-6)
    with torch.no_grad():
        calibrator.global_bias.copy_(
            torch.tensor(
                math.log(prevalence / (1.0 - prevalence)),
                dtype=torch.float64,
            )
        )
    optimizer = torch.optim.LBFGS(
        calibrator.parameters(),
        lr=1.0,
        max_iter=int(maximum_iterations),
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-10,
        tolerance_change=1e-12,
    )
    evaluations = 0

    def closure() -> torch.Tensor:
        nonlocal evaluations
        optimizer.zero_grad(set_to_none=True)
        calibrated = calibrator.calibrated_logits(logits64, type_index)
        losses = F.binary_cross_entropy_with_logits(
            calibrated,
            labels64,
            reduction="none",
        )
        objective = torch.sum(losses * weights64) / weights64.sum().clamp_min(1e-12)
        centered = calibrator.type_offsets - calibrator.type_offsets.mean()
        objective = objective + l2_type_offset * centered.square().mean()
        objective = objective + l2_log_slope * torch.log(calibrator.slope).square()
        if not bool(torch.isfinite(objective)):
            raise ValueError("Calibration objective became non-finite")
        objective.backward()
        evaluations += 1
        return objective

    optimizer.step(closure)
    calibrator.eval()
    with torch.no_grad():
        probability = calibrator(logits64, type_index)
        brier = (
            torch.sum(weights64 * (probability - labels64).square()) / weights64.sum()
        )
    return calibrator, {
        "schema_version": "nucpred.mayr-type-aware-platt-fit-audit.v1",
        "optimizer": "LBFGS",
        "closure_evaluations": evaluations,
        "validation_weighted_brier": float(brier),
        "l2_type_offset": float(l2_type_offset),
        "l2_log_slope": float(l2_log_slope),
        **calibrator.to_payload(),
    }
