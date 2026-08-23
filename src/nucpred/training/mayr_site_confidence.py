"""Independent validity logits for typed Mayr site queries.

The confidence model reuses the exact fused representation and N-regression
head from :mod:`nucpred.training.mayr_site_n`.  Its validity head is parallel:
it never gates, normalizes, or otherwise changes the conditional N prediction.

Version 1 deliberately implements supervised positive/verified-negative loss
only.  Unlabeled candidates are unknown, not negatives, and make no loss
contribution.  Absolute probabilities are exposed only through an externally
fitted calibrator whose label evidence and artifact bindings all validate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntEnum
import hashlib
import math
from pathlib import Path
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from nucpred.training.mayr_site_n import (
    MODEL_SCHEMA_VERSION,
    SITE_TYPE_NAMES,
    MayrSiteNModel,
    SiteNModelInputs,
    within_context_ranking_loss,
)


CONFIDENCE_MODEL_SCHEMA_VERSION = "nucpred.mayr-site-confidence-model.v1"
CONFIDENCE_TRANSFER_SCHEMA_VERSION = (
    "nucpred.mayr-site-confidence-formal-transfer.v1"
)
CALIBRATOR_SCHEMA_VERSION = "nucpred.mayr-site-validity-calibrator.v2"
FORMAL_SITE_N_CHECKPOINT_SCHEMA_VERSION = (
    "nucpred.mayr-site-n-formal-checkpoint.v1"
)
SITE_VALIDITY_HEAD_KEYS = frozenset(
    {
        "site_validity_head.weight",
        "site_validity_head.bias",
    }
)


class SiteValidityState(IntEnum):
    """Three-state supervision; unlabeled does not mean negative."""

    UNLABELED = -1
    VERIFIED_NEGATIVE = 0
    POSITIVE = 1


@dataclass(frozen=True)
class SiteNConfidenceOutput:
    """Conditional N prediction and an independent raw validity logit."""

    n_prediction_standardized: torch.Tensor
    site_validity_logits: torch.Tensor
    node_embeddings: torch.Tensor
    graph_pool: torch.Tensor
    site_embeddings: torch.Tensor
    site_summary: torch.Tensor


class MayrSiteConfidenceModel(MayrSiteNModel):
    """Mayr Site-N model with a parallel, independently sigmoid-able head."""

    def __init__(
        self,
        *,
        num_solvents: int,
        hidden_dim: int = 128,
        layers: int = 4,
        node_embedding_dim: int = 16,
        edge_embedding_dim: int = 16,
        solvent_embedding_dim: int = 16,
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
        self.site_n_architecture = dict(self.architecture)
        self.site_validity_head = nn.Linear(6 * hidden_dim, 1)
        self.architecture = {
            **self.site_n_architecture,
            "schema_version": CONFIDENCE_MODEL_SCHEMA_VERSION,
            "base_model_schema_version": MODEL_SCHEMA_VERSION,
            "site_validity_output": "independent_logit_per_query",
            "site_validity_head": "linear_from_fused_6h",
            "validity_unknown_policy": "strictly_masked_no_pu",
        }

    def forward(self, inputs: SiteNModelInputs) -> SiteNConfidenceOutput:
        encoded = self.encode_fused_features(inputs)
        n_prediction = self.regression_head(encoded.fused).squeeze(-1)
        validity_logits = self.site_validity_head(encoded.fused).squeeze(-1)
        return SiteNConfidenceOutput(
            n_prediction_standardized=n_prediction,
            site_validity_logits=validity_logits,
            node_embeddings=encoded.node_embeddings,
            graph_pool=encoded.graph_pool,
            site_embeddings=encoded.site_embeddings,
            site_summary=encoded.site_summary,
        )


@dataclass(frozen=True)
class SiteConfidenceTrainingBatch:
    """Query-aligned targets and three-state validity supervision."""

    inputs: SiteNModelInputs
    n_target_standardized: torch.Tensor
    n_supervision_mask: torch.Tensor
    validity_state: torch.Tensor

    def __post_init__(self) -> None:
        query_count = self.inputs.num_sites
        expected_shape = (query_count,)
        if self.n_target_standardized.shape != expected_shape:
            raise ValueError("N targets must align with site queries")
        if self.n_supervision_mask.shape != expected_shape:
            raise ValueError("N supervision mask must align with site queries")
        if self.validity_state.shape != expected_shape:
            raise ValueError("Validity states must align with site queries")
        if self.n_supervision_mask.dtype != torch.bool:
            raise TypeError("N supervision mask must be boolean")
        integer_dtypes = {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }
        if self.validity_state.dtype not in integer_dtypes:
            raise TypeError("Validity states must use an integer dtype")
        allowed = {int(state) for state in SiteValidityState}
        observed = {
            int(value)
            for value in torch.unique(self.validity_state).detach().cpu().tolist()
        }
        if not observed.issubset(allowed):
            raise ValueError(f"Unknown validity states: {sorted(observed - allowed)}")
        if bool((self.n_supervision_mask & ~self.positive_mask).any()):
            raise ValueError("N supervision is allowed only for positive sites")
        supervised_targets = self.n_target_standardized[
            self.n_supervision_mask
        ]
        if not bool(torch.isfinite(supervised_targets).all()):
            raise ValueError("Supervised N targets must be finite")

    @property
    def positive_mask(self) -> torch.Tensor:
        return self.validity_state.eq(int(SiteValidityState.POSITIVE))

    @property
    def verified_negative_mask(self) -> torch.Tensor:
        return self.validity_state.eq(
            int(SiteValidityState.VERIFIED_NEGATIVE)
        )

    @property
    def unlabeled_mask(self) -> torch.Tensor:
        return self.validity_state.eq(int(SiteValidityState.UNLABELED))

    def to(
        self,
        device: str | torch.device,
    ) -> "SiteConfidenceTrainingBatch":
        return SiteConfidenceTrainingBatch(
            inputs=self.inputs.to(device),
            n_target_standardized=self.n_target_standardized.to(device),
            n_supervision_mask=self.n_supervision_mask.to(device),
            validity_state=self.validity_state.to(device),
        )


def _masked_binary_cross_entropy(
    logits: torch.Tensor,
    mask: torch.Tensor,
    target: float,
) -> torch.Tensor:
    selected = logits[mask]
    if selected.numel() == 0:
        return logits.new_zeros(())
    labels = torch.full_like(selected, float(target))
    return F.binary_cross_entropy_with_logits(selected, labels)


def site_n_confidence_loss(
    output: SiteNConfidenceOutput,
    batch: SiteConfidenceTrainingBatch,
    *,
    ranking_weight: float = 0.1,
    validity_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | int]]:
    """Joint conditional-N and supervised P/N validity loss.

    No positive-unlabeled estimator is implemented in v1.  Unlabeled logits
    and targets are never selected by a loss operation.
    """

    query_count = batch.inputs.num_sites
    if output.n_prediction_standardized.shape != (query_count,):
        raise ValueError("N predictions must align with site queries")
    if output.site_validity_logits.shape != (query_count,):
        raise ValueError("Validity logits must align with site queries")
    if ranking_weight < 0.0 or validity_weight < 0.0:
        raise ValueError("Loss weights cannot be negative")

    n_mask = batch.n_supervision_mask
    n_prediction = output.n_prediction_standardized[n_mask]
    n_target = batch.n_target_standardized[n_mask]
    if n_prediction.numel():
        regression = F.mse_loss(n_prediction, n_target)
        ranking, pair_count = within_context_ranking_loss(
            n_prediction,
            n_target,
            batch.inputs.site_graph_index[n_mask],
        )
    else:
        regression = output.n_prediction_standardized.new_zeros(())
        ranking = output.n_prediction_standardized.new_zeros(())
        pair_count = 0

    positive_mask = batch.positive_mask
    negative_mask = batch.verified_negative_mask
    supervised_validity_mask = positive_mask | negative_mask
    supervised_logits = output.site_validity_logits[
        supervised_validity_mask
    ]
    if supervised_logits.numel():
        supervised_labels = positive_mask[supervised_validity_mask].to(
            dtype=supervised_logits.dtype
        )
        validity = F.binary_cross_entropy_with_logits(
            supervised_logits,
            supervised_labels,
        )
    else:
        validity = output.site_validity_logits.new_zeros(())
    positive_validity = _masked_binary_cross_entropy(
        output.site_validity_logits,
        positive_mask,
        1.0,
    )
    negative_validity = _masked_binary_cross_entropy(
        output.site_validity_logits,
        negative_mask,
        0.0,
    )

    total = (
        regression
        + float(ranking_weight) * ranking
        + float(validity_weight) * validity
    )
    return total, {
        "n_regression": regression,
        "n_ranking": ranking,
        "n_ranking_pairs": pair_count,
        "validity": validity,
        "validity_positive": positive_validity,
        "validity_negative": negative_validity,
        "validity_positive_count": int(positive_mask.sum().item()),
        "validity_negative_count": int(negative_mask.sum().item()),
        "validity_unlabeled_count": int(
            batch.unlabeled_mask.sum().item()
        ),
        "n_supervision_count": int(n_mask.sum().item()),
    }


def tensor_mapping_sha256(values: Mapping[str, torch.Tensor]) -> str:
    """Hash names, dtypes, shapes, and exact tensor bytes deterministically."""

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


def _load_checkpoint_payload(
    checkpoint: str | Path | Mapping[str, object],
) -> Mapping[str, object]:
    if isinstance(checkpoint, (str, Path)):
        payload = torch.load(
            Path(checkpoint),
            map_location="cpu",
            weights_only=False,
        )
    else:
        payload = checkpoint
    if not isinstance(payload, Mapping):
        raise TypeError("Formal checkpoint payload must be a mapping")
    return payload


def transfer_formal_site_n_checkpoint(
    model: MayrSiteConfidenceModel,
    checkpoint: str | Path | Mapping[str, object],
) -> dict[str, object]:
    """Copy every legacy formal N tensor while leaving the new head untouched."""

    if not isinstance(model, MayrSiteConfidenceModel):
        raise TypeError("Target must be a MayrSiteConfidenceModel")
    payload = _load_checkpoint_payload(checkpoint)
    if (
        payload.get("schema_version")
        != FORMAL_SITE_N_CHECKPOINT_SCHEMA_VERSION
    ):
        raise ValueError("Unsupported formal Site-N checkpoint schema")
    source = payload.get("model_state_dict")
    if not isinstance(source, Mapping):
        raise TypeError("Formal checkpoint has no model state mapping")
    if not all(isinstance(value, torch.Tensor) for value in source.values()):
        raise TypeError("Formal checkpoint state values must be tensors")
    source_tensors = {
        str(name): value for name, value in source.items()
    }

    source_hash = tensor_mapping_sha256(source_tensors)
    if payload.get("model_state_sha256") != source_hash:
        raise ValueError("Formal checkpoint model-state hash changed")
    if payload.get("model_architecture") != model.site_n_architecture:
        raise ValueError("Formal checkpoint architecture does not match")

    target_state = model.state_dict()
    if not SITE_VALIDITY_HEAD_KEYS.issubset(target_state):
        raise ValueError("Target model has no complete validity head")
    expected_legacy_keys = set(target_state) - SITE_VALIDITY_HEAD_KEYS
    if set(source_tensors) != expected_legacy_keys:
        missing = sorted(expected_legacy_keys - set(source_tensors))
        unexpected = sorted(set(source_tensors) - expected_legacy_keys)
        raise ValueError(
            f"Formal checkpoint key mismatch; missing={missing}, "
            f"unexpected={unexpected}"
        )

    head_before = {
        name: target_state[name].detach().cpu().clone()
        for name in SITE_VALIDITY_HEAD_KEYS
    }
    for name in sorted(expected_legacy_keys):
        source_tensor = source_tensors[name]
        target_tensor = target_state[name]
        if (
            target_tensor.shape != source_tensor.shape
            or target_tensor.dtype != source_tensor.dtype
        ):
            raise ValueError(f"Formal checkpoint tensor changed: {name}")
    with torch.no_grad():
        for name in sorted(expected_legacy_keys):
            source_tensor = source_tensors[name]
            target_tensor = target_state[name]
            target_tensor.copy_(
                source_tensor.to(device=target_tensor.device)
            )
    model.load_state_dict(target_state, strict=True)

    after = model.state_dict()
    transferred = {
        name: after[name].detach().cpu()
        for name in expected_legacy_keys
    }
    exact = all(
        torch.equal(transferred[name], source_tensors[name].detach().cpu())
        for name in expected_legacy_keys
    )
    head_unchanged = all(
        torch.equal(after[name].detach().cpu(), head_before[name])
        for name in SITE_VALIDITY_HEAD_KEYS
    )
    transferred_hash = tensor_mapping_sha256(transferred)
    if not exact or not head_unchanged or transferred_hash != source_hash:
        raise RuntimeError("Formal checkpoint transfer was not exact")
    return {
        "schema_version": CONFIDENCE_TRANSFER_SCHEMA_VERSION,
        "status": "pass",
        "transferred_parameter_tensors": len(transferred),
        "transferred_parameter_numel": sum(
            int(tensor.numel()) for tensor in transferred.values()
        ),
        "source_state_sha256": source_hash,
        "transferred_state_sha256": transferred_hash,
        "exact_transfer": True,
        "validity_head_unchanged": True,
        "missing_target_keys": sorted(SITE_VALIDITY_HEAD_KEYS),
    }


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class SiteValidityCalibrator:
    """Affine-logit calibrator with fail-closed artifact bindings."""

    candidate_scope_sha256: str
    checkpoint_sha256: str
    label_evidence_sha256: str
    positive_count: int
    verified_negative_count: int
    unlabeled_count: int = 0
    slope: float = 1.0
    intercept: float = 0.0
    stage_gate_sha256: str = ""
    formal_negative_connectivity_count: int = 0
    calibration_negative_connectivity_count: int = 0
    test_negative_connectivity_count: int = 0
    supported_site_type_negative_connectivity_counts: tuple[
        tuple[str, int], ...
    ] = ()
    formal_probability_authorized: bool = False
    brier_gate_passed: bool = False
    ece_gate_passed: bool = False
    retrieval_gate_passed: bool = False
    method: str = "platt"
    fit_role: str = "validation"
    schema_version: str = CALIBRATOR_SCHEMA_VERSION

    def unavailable_reasons(
        self,
        *,
        candidate_scope_sha256: str,
        checkpoint_sha256: str,
        label_evidence_sha256: str,
        stage_gate_sha256: str,
    ) -> tuple[str, ...]:
        reasons: list[str] = []
        if self.schema_version != CALIBRATOR_SCHEMA_VERSION:
            reasons.append("unsupported_schema")
        if self.method != "platt":
            reasons.append("unsupported_method")
        if self.fit_role != "validation":
            reasons.append("invalid_fit_role")
        if self.positive_count <= 0 or self.verified_negative_count <= 0:
            reasons.append("insufficient_verified_labels")
        if self.unlabeled_count < 0:
            reasons.append("invalid_unlabeled_count")
        if not self.formal_probability_authorized:
            reasons.append("formal_probability_not_authorized")
        if self.formal_negative_connectivity_count < 60:
            reasons.append("insufficient_formal_negative_connectivity")
        if self.calibration_negative_connectivity_count < 15:
            reasons.append("insufficient_calibration_negative_connectivity")
        if self.test_negative_connectivity_count < 15:
            reasons.append("insufficient_test_negative_connectivity")
        if (
            self.formal_negative_connectivity_count
            > self.verified_negative_count
            or self.calibration_negative_connectivity_count
            > self.formal_negative_connectivity_count
            or self.test_negative_connectivity_count
            > self.formal_negative_connectivity_count
        ):
            reasons.append("inconsistent_negative_connectivity_counts")
        type_counts = self.supported_site_type_negative_connectivity_counts
        type_names = [str(name) for name, _ in type_counts]
        if (
            not type_counts
            or len(type_names) != len(set(type_names))
            or any(name not in SITE_TYPE_NAMES for name in type_names)
        ):
            reasons.append("invalid_supported_site_types")
        elif any(int(count) < 10 for _, count in type_counts):
            reasons.append("insufficient_supported_site_type_connectivity")
        if not self.brier_gate_passed:
            reasons.append("brier_gate_failed")
        if not self.ece_gate_passed:
            reasons.append("ece_gate_failed")
        if not self.retrieval_gate_passed:
            reasons.append("retrieval_gate_failed")
        if not math.isfinite(self.slope) or self.slope <= 0.0:
            reasons.append("invalid_slope")
        if not math.isfinite(self.intercept):
            reasons.append("invalid_intercept")
        bindings = (
            (
                "candidate_scope",
                self.candidate_scope_sha256,
                candidate_scope_sha256,
            ),
            ("checkpoint", self.checkpoint_sha256, checkpoint_sha256),
            (
                "label_evidence",
                self.label_evidence_sha256,
                label_evidence_sha256,
            ),
            ("stage_gate", self.stage_gate_sha256, stage_gate_sha256),
        )
        for name, expected, observed in bindings:
            if not _is_sha256(expected) or not _is_sha256(observed):
                reasons.append(f"invalid_{name}_hash")
            elif expected != observed:
                reasons.append(f"{name}_hash_mismatch")
        return tuple(dict.fromkeys(reasons))

    def transform(
        self,
        logits: torch.Tensor,
        *,
        candidate_scope_sha256: str,
        checkpoint_sha256: str,
        label_evidence_sha256: str,
        stage_gate_sha256: str,
    ) -> torch.Tensor | None:
        """Return absolute probabilities only when every binding is valid."""

        reasons = self.unavailable_reasons(
            candidate_scope_sha256=candidate_scope_sha256,
            checkpoint_sha256=checkpoint_sha256,
            label_evidence_sha256=label_evidence_sha256,
            stage_gate_sha256=stage_gate_sha256,
        )
        if reasons:
            return None
        if not logits.dtype.is_floating_point:
            raise TypeError("Calibration logits must use a floating dtype")
        if not bool(torch.isfinite(logits).all()):
            return None
        return torch.sigmoid(
            logits * float(self.slope) + float(self.intercept)
        )

    def to_json(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_json(
        cls,
        payload: Mapping[str, object],
    ) -> "SiteValidityCalibrator":
        return cls(
            candidate_scope_sha256=str(
                payload["candidate_scope_sha256"]
            ),
            checkpoint_sha256=str(payload["checkpoint_sha256"]),
            label_evidence_sha256=str(
                payload["label_evidence_sha256"]
            ),
            positive_count=int(payload["positive_count"]),
            verified_negative_count=int(
                payload["verified_negative_count"]
            ),
            unlabeled_count=int(payload.get("unlabeled_count", 0)),
            slope=float(payload.get("slope", 1.0)),
            intercept=float(payload.get("intercept", 0.0)),
            stage_gate_sha256=str(payload.get("stage_gate_sha256", "")),
            formal_negative_connectivity_count=int(
                payload.get("formal_negative_connectivity_count", 0)
            ),
            calibration_negative_connectivity_count=int(
                payload.get("calibration_negative_connectivity_count", 0)
            ),
            test_negative_connectivity_count=int(
                payload.get("test_negative_connectivity_count", 0)
            ),
            supported_site_type_negative_connectivity_counts=tuple(
                (str(item[0]), int(item[1]))
                for item in payload.get(
                    "supported_site_type_negative_connectivity_counts",
                    (),
                )
            ),
            formal_probability_authorized=(
                payload.get("formal_probability_authorized") is True
            ),
            brier_gate_passed=(payload.get("brier_gate_passed") is True),
            ece_gate_passed=(payload.get("ece_gate_passed") is True),
            retrieval_gate_passed=(
                payload.get("retrieval_gate_passed") is True
            ),
            method=str(payload.get("method", "platt")),
            fit_role=str(payload.get("fit_role", "validation")),
            schema_version=str(
                payload.get("schema_version", CALIBRATOR_SCHEMA_VERSION)
            ),
        )
