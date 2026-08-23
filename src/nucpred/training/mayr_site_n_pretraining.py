"""Matched ESNUEL pretraining for the independent typed site-N backbone.

ESNUEL proxy labels supervise only measured heavy-atom queries.  Explicit H
nodes still participate in message passing and masked reconstruction, while
bond, region, atom-group, and transferable-H-group adapters receive no fake
MCA/GCS labels.  No probability distribution or site softmax is constructed.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from nucpred.features.all_atom_graph import (
    EDGE_CATEGORY_SIZES,
    NODE_CATEGORY_SIZES,
)
from nucpred.training.mayr_node_xtb_pretraining import (
    GCS_DIM,
    HYDROGEN_ELEMENT_INDEX,
    EsnuelNodeXtbExample,
    MaskingConfig,
    PretrainingBatch,
    PretrainingNormalization,
    fit_pretraining_normalization,
    load_pretraining_examples,
    pack_pretraining_batch,
)
from nucpred.training.mayr_node_xtb_scratch import (
    GLOBAL_FEATURES,
    LOCAL_FEATURES,
)
from nucpred.training.mayr_site_n import (
    SITE_TYPE_NAMES,
    SITE_TYPE_TO_INDEX,
    MayrSiteNModel,
    seed_everything,
)


CHECKPOINT_SCHEMA_VERSION = "nucpred.mayr-site-n-pretraining-checkpoint.v1"
PRETRAINING_SCHEMA_VERSION = "nucpred.mayr-site-n-pretraining.v1"
ATOM_TYPE_INDEX = SITE_TYPE_TO_INDEX["atom"]
TRANSFER_PREFIXES = (
    "node_encoder.",
    "local_encoder.",
    "edge_encoder.",
    "message_layers.",
    "global_xtb_encoder.",
    "site_object_encoder.shared_encoder.",
    "site_object_encoder.type_adapters.atom.",
)
RESET_PREFIXES = (
    "site_object_encoder.type_adapters.bond.",
    "site_object_encoder.type_adapters.delocalized_region.",
    "site_object_encoder.type_adapters.atom_group.",
    "site_object_encoder.type_adapters.transferable_h_group.",
    "solvent_encoder.",
    "solvent_embedding.",
    "solvent_embedding_projection.",
    "charge_encoder.",
    "regression_head.",
)


def _prediction_head(
    input_dim: int,
    output_dim: int,
    dropout: float,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, input_dim),
        nn.ReLU(),
        nn.LayerNorm(input_dim),
        nn.Dropout(dropout),
        nn.Linear(input_dim, output_dim),
    )


def _tensor_mapping_sha256(values: Mapping[str, torch.Tensor]) -> str:
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


def _state_subset(
    values: Mapping[str, torch.Tensor],
    prefixes: Sequence[str],
) -> dict[str, torch.Tensor]:
    return {
        name: tensor
        for name, tensor in values.items()
        if any(name.startswith(prefix) for prefix in prefixes)
    }


@dataclass(frozen=True)
class SiteNPretrainingOutput:
    node_embeddings: torch.Tensor
    atom_query_embeddings: torch.Tensor
    edge_embeddings: torch.Tensor
    graph_pool: torch.Tensor
    global_embedding: torch.Tensor
    node_categorical_predictions: tuple[torch.Tensor, ...]
    edge_categorical_predictions: tuple[torch.Tensor, ...]
    local_prediction: torch.Tensor
    global_prediction: torch.Tensor
    mca_prediction: torch.Tensor
    gcs_prediction: torch.Tensor


class SiteNPretrainingModel(nn.Module):
    """Disposable pretraining heads around the exact downstream backbone."""

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
        gcs_dim: int = GCS_DIM,
        init_seed: int | None = None,
    ) -> None:
        super().__init__()
        if gcs_dim != GCS_DIM:
            raise ValueError(f"ESNUEL GCS must have {GCS_DIM} dimensions")
        if init_seed is not None:
            seed_everything(int(init_seed))
        self.initialization_seed = (
            None if init_seed is None else int(init_seed)
        )
        self.backbone = MayrSiteNModel(
            num_solvents=num_solvents,
            hidden_dim=hidden_dim,
            layers=layers,
            node_embedding_dim=node_embedding_dim,
            edge_embedding_dim=edge_embedding_dim,
            solvent_embedding_dim=solvent_embedding_dim,
            dropout=dropout,
        )
        self.architecture = {
            **self.backbone.architecture,
            "gcs_dim": int(gcs_dim),
            "pretraining_site_query_type": "atom",
            "site_probability_normalization": False,
        }
        self.node_reconstruction_heads = nn.ModuleList(
            _prediction_head(hidden_dim, size, dropout)
            for size in NODE_CATEGORY_SIZES
        )
        edge_input_dim = 3 * hidden_dim
        self.edge_reconstruction_heads = nn.ModuleList(
            _prediction_head(edge_input_dim, size, dropout)
            for size in EDGE_CATEGORY_SIZES
        )
        self.local_reconstruction_head = _prediction_head(
            hidden_dim,
            len(LOCAL_FEATURES),
            dropout,
        )
        self.global_reconstruction_head = _prediction_head(
            hidden_dim,
            len(GLOBAL_FEATURES),
            dropout,
        )
        self.mca_head = _prediction_head(hidden_dim, 1, dropout)
        self.gcs_head = _prediction_head(hidden_dim, gcs_dim, dropout)

    def forward(
        self,
        batch: PretrainingBatch,
    ) -> SiteNPretrainingOutput:
        inputs = batch.inputs
        nodes, graph_pool = self.backbone.encode_graph(inputs)  # type: ignore[arg-type]
        node_count = int(nodes.shape[0])
        member_index = torch.arange(node_count, device=nodes.device)
        member_ptr = torch.arange(node_count + 1, device=nodes.device)
        atom_types = torch.full(
            (node_count,),
            ATOM_TYPE_INDEX,
            dtype=torch.long,
            device=nodes.device,
        )
        atom_queries, _ = self.backbone.site_object_encoder(
            nodes,
            member_index,
            member_ptr,
            atom_types,
        )
        edges = self.backbone.edge_encoder(inputs.edge_categorical)
        global_embedding = self.backbone.global_xtb_encoder(inputs.global_xtb)
        if inputs.edge_index.shape[1]:
            source, destination = inputs.edge_index
            edge_representation = torch.cat(
                (nodes[source], nodes[destination], edges),
                dim=-1,
            )
        else:
            edge_representation = nodes.new_empty((0, 3 * nodes.shape[-1]))
        return SiteNPretrainingOutput(
            node_embeddings=nodes,
            atom_query_embeddings=atom_queries,
            edge_embeddings=edges,
            graph_pool=graph_pool,
            global_embedding=global_embedding,
            node_categorical_predictions=tuple(
                head(nodes) for head in self.node_reconstruction_heads
            ),
            edge_categorical_predictions=tuple(
                head(edge_representation)
                for head in self.edge_reconstruction_heads
            ),
            local_prediction=self.local_reconstruction_head(nodes),
            global_prediction=self.global_reconstruction_head(
                global_embedding
            ),
            mca_prediction=self.mca_head(atom_queries).squeeze(-1),
            gcs_prediction=self.gcs_head(atom_queries),
        )

    def transferable_state_dict(self) -> dict[str, torch.Tensor]:
        return _state_subset(
            self.backbone.state_dict(),
            TRANSFER_PREFIXES,
        )

    def pretraining_heads_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            name: tensor.detach().cpu()
            for name, tensor in self.state_dict().items()
            if not name.startswith("backbone.")
        }


@dataclass(frozen=True, slots=True)
class SiteNPretrainingLossConfig:
    node_categorical_weight: float = 1.0
    edge_categorical_weight: float = 1.0
    local_weight: float = 1.0
    global_weight: float = 1.0
    mca_weight: float = 1.0
    gcs_weight: float = 1.0
    ranking_weight: float = 0.25
    ranking_margin: float = 0.0

    def validate(self) -> None:
        for name, value in asdict(self).items():
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError(f"{name} must be finite")
            if name.endswith("_weight") and numeric < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.ranking_margin < 0:
            raise ValueError("ranking_margin must be non-negative")


@dataclass(frozen=True)
class SiteNPretrainingLossBreakdown:
    total: torch.Tensor
    components: Mapping[str, torch.Tensor]
    ranking_pairs: int
    eligible_mca_atoms: int
    eligible_gcs_atoms: int


def _masked_smooth_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    zero: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target shapes changed")
    if mask.shape == prediction.shape[:-1] and prediction.ndim == 2:
        selected = mask.unsqueeze(-1).expand_as(prediction)
    elif mask.shape == prediction.shape:
        selected = mask
    else:
        raise ValueError("Regression mask shape changed")
    if not bool(selected.any()):
        return zero
    return F.smooth_l1_loss(prediction[selected], target[selected])


def _categorical_reconstruction_loss(
    predictions: Sequence[torch.Tensor],
    targets: torch.Tensor,
    mask: torch.Tensor,
    *,
    zero: torch.Tensor,
) -> torch.Tensor:
    if len(predictions) != targets.shape[1] or targets.shape != mask.shape:
        raise ValueError("Categorical reconstruction shapes changed")
    losses: list[torch.Tensor] = []
    for column, prediction in enumerate(predictions):
        selected = mask[:, column]
        if bool(selected.any()):
            losses.append(
                F.cross_entropy(
                    prediction[selected],
                    targets[selected, column],
                )
            )
    return torch.stack(losses).mean() if losses else zero


def within_graph_delta_loss(
    prediction: torch.Tensor,
    targets: torch.Tensor,
    eligible: torch.Tensor,
    graph_ptr: torch.Tensor,
    *,
    margin: float = 0.0,
) -> tuple[torch.Tensor, int]:
    losses: list[torch.Tensor] = []
    pair_count = 0
    for graph_index in range(int(graph_ptr.shape[0]) - 1):
        start = int(graph_ptr[graph_index])
        end = int(graph_ptr[graph_index + 1])
        indices = (
            torch.nonzero(eligible[start:end], as_tuple=False).flatten()
            + start
        )
        if len(indices) < 2:
            continue
        pairs = torch.triu_indices(
            len(indices),
            len(indices),
            offset=1,
            device=indices.device,
        )
        left = indices[pairs[0]]
        right = indices[pairs[1]]
        target_delta = targets[left] - targets[right]
        non_ties = target_delta.ne(0)
        if not bool(non_ties.any()):
            continue
        prediction_delta = (
            prediction[left[non_ties]] - prediction[right[non_ties]]
        )
        difference = prediction_delta - target_delta[non_ties]
        losses.append(
            F.smooth_l1_loss(
                difference,
                torch.zeros_like(difference),
                beta=max(1.0, float(margin)),
                reduction="none",
            )
        )
        pair_count += int(non_ties.sum())
    if not losses:
        return prediction.sum() * 0.0, 0
    return torch.cat(losses).mean(), pair_count


def site_n_pretraining_loss(
    output: SiteNPretrainingOutput,
    batch: PretrainingBatch,
    config: SiteNPretrainingLossConfig = SiteNPretrainingLossConfig(),
) -> SiteNPretrainingLossBreakdown:
    """Compute proxy/reconstruction losses without a site-distribution target."""

    config.validate()
    zero = output.node_embeddings.sum() * 0.0
    eligible_mca = batch.mca_mask & ~batch.is_hydrogen
    eligible_gcs = batch.gcs_mask & ~batch.is_hydrogen
    components: dict[str, torch.Tensor] = {
        "node_categorical": _categorical_reconstruction_loss(
            output.node_categorical_predictions,
            batch.original_node_categorical,
            batch.node_reconstruction_mask,
            zero=zero,
        ),
        "edge_categorical": _categorical_reconstruction_loss(
            output.edge_categorical_predictions,
            batch.original_edge_categorical,
            batch.edge_reconstruction_mask,
            zero=zero,
        ),
        "local4": _masked_smooth_l1(
            output.local_prediction,
            batch.local_targets,
            batch.local_reconstruction_mask,
            zero=zero,
        ),
        "global6": _masked_smooth_l1(
            output.global_prediction,
            batch.global_targets,
            batch.global_reconstruction_mask,
            zero=zero,
        ),
        "mca": _masked_smooth_l1(
            output.mca_prediction,
            batch.mca_targets,
            eligible_mca,
            zero=zero,
        ),
        "gcs": _masked_smooth_l1(
            output.gcs_prediction,
            batch.gcs_targets,
            eligible_gcs,
            zero=zero,
        ),
    }
    ranking, pair_count = within_graph_delta_loss(
        output.mca_prediction,
        batch.mca_targets,
        eligible_mca,
        batch.inputs.graph_ptr,
        margin=config.ranking_margin,
    )
    components["ranking"] = ranking
    weights = {
        "node_categorical": config.node_categorical_weight,
        "edge_categorical": config.edge_categorical_weight,
        "local4": config.local_weight,
        "global6": config.global_weight,
        "mca": config.mca_weight,
        "gcs": config.gcs_weight,
        "ranking": config.ranking_weight,
    }
    total = sum(
        (
            float(weights[name]) * value
            for name, value in components.items()
        ),
        zero,
    )
    return SiteNPretrainingLossBreakdown(
        total=total,
        components=components,
        ranking_pairs=pair_count,
        eligible_mca_atoms=int(eligible_mca.sum()),
        eligible_gcs_atoms=int(eligible_gcs.sum()),
    )


def required_gradient_audit(
    model: SiteNPretrainingModel,
) -> dict[str, bool]:
    modules: dict[str, nn.Module] = {
        "node_encoder": model.backbone.node_encoder,
        "local_encoder": model.backbone.local_encoder,
        "edge_encoder": model.backbone.edge_encoder,
        "global_xtb_encoder": model.backbone.global_xtb_encoder,
        "site_object_encoder.shared_encoder": (
            model.backbone.site_object_encoder.shared_encoder
        ),
        "site_object_encoder.type_adapters.atom": (
            model.backbone.site_object_encoder.type_adapters["atom"]
        ),
        "node_reconstruction_heads": model.node_reconstruction_heads,
        "edge_reconstruction_heads": model.edge_reconstruction_heads,
        "local_reconstruction_head": model.local_reconstruction_head,
        "global_reconstruction_head": model.global_reconstruction_head,
        "mca_head": model.mca_head,
        "gcs_head": model.gcs_head,
        **{
            f"message_layers.{index}": layer
            for index, layer in enumerate(model.backbone.message_layers)
        },
    }
    audit: dict[str, bool] = {}
    for name, module in modules.items():
        gradients = [
            parameter.grad
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        audit[name] = bool(gradients) and any(
            gradient is not None
            and bool(torch.isfinite(gradient).all())
            and bool(torch.count_nonzero(gradient))
            for gradient in gradients
        )
    element_gradient = (
        model.backbone.node_encoder.embeddings[0].weight.grad
    )
    audit["ordinary_h_element_embedding"] = (
        element_gradient is not None
        and bool(
            torch.isfinite(
                element_gradient[HYDROGEN_ELEMENT_INDEX]
            ).all()
        )
        and bool(
            torch.count_nonzero(
                element_gradient[HYDROGEN_ELEMENT_INDEX]
            )
        )
    )
    return audit


def expected_unsupervised_gradient_audit(
    model: SiteNPretrainingModel,
) -> dict[str, bool]:
    """Confirm that unsupported site types and reset-only branches stay unused."""

    modules: dict[str, nn.Module] = {
        f"site_adapter.{site_type}": (
            model.backbone.site_object_encoder.type_adapters[site_type]
        )
        for site_type in SITE_TYPE_NAMES
        if site_type != "atom"
    }
    modules.update(
        {
            "solvent_encoder": model.backbone.solvent_encoder,
            "solvent_embedding": model.backbone.solvent_embedding,
            "solvent_embedding_projection": (
                model.backbone.solvent_embedding_projection
            ),
            "charge_encoder": model.backbone.charge_encoder,
            "regression_head": model.backbone.regression_head,
        }
    )
    return {
        name: all(
            parameter.grad is None
            or not bool(torch.count_nonzero(parameter.grad))
            for parameter in module.parameters()
        )
        for name, module in modules.items()
    }


def run_required_gradient_gate(
    model: SiteNPretrainingModel,
    examples: Sequence[EsnuelNodeXtbExample],
    *,
    normalization: PretrainingNormalization,
    loss_config: SiteNPretrainingLossConfig,
    seed: int,
    device: str | torch.device,
    max_molecules: int = 32,
    attempts: int = 32,
) -> dict[str, object]:
    if not examples:
        raise ValueError("Gradient gate requires training examples")
    diagnostic_masking = MaskingConfig(
        node_categorical_probability=0.5,
        edge_categorical_probability=0.5,
        local_probability=0.5,
        global_probability=0.5,
    )
    diagnostic = sorted(
        examples,
        key=lambda example: (
            -int(example.mca_mask.sum()),
            -int(example.is_hydrogen.sum()),
            example.source_id,
        ),
    )[:max_molecules]
    previous_mode = model.training
    model.eval()
    last_required: dict[str, bool] = {}
    last_unsupervised: dict[str, bool] = {}
    try:
        for attempt in range(attempts):
            model.zero_grad(set_to_none=True)
            batch = pack_pretraining_batch(
                diagnostic,
                normalization=normalization,
                masking=diagnostic_masking,
                mask_seed=int(seed) + attempt,
            ).to(device)
            output = model(batch)
            breakdown = site_n_pretraining_loss(
                output,
                batch,
                loss_config,
            )
            if not bool(torch.isfinite(breakdown.total)):
                raise RuntimeError("Gradient-gate loss is non-finite")
            breakdown.total.backward()
            last_required = required_gradient_audit(model)
            last_unsupervised = expected_unsupervised_gradient_audit(model)
            if (
                all(last_required.values())
                and all(last_unsupervised.values())
            ):
                return {
                    "required_paths": last_required,
                    "unsupported_and_reset_paths_unused": last_unsupervised,
                    "attempt": attempt + 1,
                    "status": "pass",
                }
    finally:
        model.zero_grad(set_to_none=True)
        model.train(previous_mode)
    failures = [
        name for name, passed in last_required.items() if not passed
    ]
    leakage = [
        name for name, passed in last_unsupervised.items() if not passed
    ]
    raise RuntimeError(
        f"Gradient gate failed; missing={failures}, leakage={leakage}"
    )


@dataclass(frozen=True)
class SiteNPretrainingResult:
    model: SiteNPretrainingModel
    optimizer: torch.optim.Optimizer
    normalization: PretrainingNormalization
    history: tuple[Mapping[str, object], ...]
    best_epoch: int
    best_validation_total: float
    audit_metrics: Mapping[str, float]
    gradient_audit: Mapping[str, object]


def _epoch_batches(
    examples: Sequence[EsnuelNodeXtbExample],
    *,
    batch_size: int,
    seed: int,
    shuffle: bool,
) -> list[list[EsnuelNodeXtbExample]]:
    indices = np.arange(len(examples))
    if shuffle:
        np.random.default_rng(int(seed)).shuffle(indices)
    return [
        [examples[int(index)] for index in indices[start : start + batch_size]]
        for start in range(0, len(indices), batch_size)
    ]


def _run_epoch(
    model: SiteNPretrainingModel,
    examples: Sequence[EsnuelNodeXtbExample],
    *,
    normalization: PretrainingNormalization,
    optimizer: torch.optim.Optimizer | None,
    batch_size: int,
    seed: int,
    device: str | torch.device,
    masking: MaskingConfig,
    loss_config: SiteNPretrainingLossConfig,
    gradient_clip_norm: float = 5.0,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    weight = 0
    for batch_index, batch_examples in enumerate(
        _epoch_batches(
            examples,
            batch_size=batch_size,
            seed=seed,
            shuffle=training,
        )
    ):
        batch = pack_pretraining_batch(
            batch_examples,
            normalization=normalization,
            masking=masking,
            mask_seed=seed * 1_000_003 + batch_index,
        ).to(device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            breakdown = site_n_pretraining_loss(
                model(batch),
                batch,
                loss_config,
            )
            if not bool(torch.isfinite(breakdown.total)):
                raise RuntimeError("Pretraining loss became non-finite")
            if optimizer is not None:
                breakdown.total.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(gradient_clip_norm),
                )
                optimizer.step()
        count = len(batch_examples)
        weight += count
        totals["total"] = totals.get("total", 0.0) + (
            float(breakdown.total.detach()) * count
        )
        for name, value in breakdown.components.items():
            totals[name] = totals.get(name, 0.0) + (
                float(value.detach()) * count
            )
    if not weight:
        raise ValueError("Cannot run an epoch on an empty dataset")
    return {name: value / weight for name, value in totals.items()}


def train_site_n_pretraining(
    train_examples: Sequence[EsnuelNodeXtbExample],
    validation_examples: Sequence[EsnuelNodeXtbExample],
    *,
    audit_test_examples: Sequence[EsnuelNodeXtbExample] = (),
    num_solvents: int,
    normalization: PretrainingNormalization | None = None,
    init_seed: int = 31001,
    epochs: int = 20,
    minimum_epochs: int = 5,
    patience: int = 5,
    minimum_delta: float = 1e-4,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    device: str | torch.device = "cpu",
    masking: MaskingConfig = MaskingConfig(),
    loss_config: SiteNPretrainingLossConfig = SiteNPretrainingLossConfig(),
    hidden_dim: int = 128,
    layers: int = 4,
    node_embedding_dim: int = 16,
    edge_embedding_dim: int = 16,
    solvent_embedding_dim: int = 16,
    dropout: float = 0.1,
    require_gradient_gate: bool = True,
) -> SiteNPretrainingResult:
    if not train_examples or not validation_examples:
        raise ValueError("Pretraining requires train and validation records")
    if not 1 <= minimum_epochs <= epochs:
        raise ValueError("minimum_epochs must be in [1, epochs]")
    seed_everything(init_seed)
    fitted = normalization or fit_pretraining_normalization(train_examples)
    model = SiteNPretrainingModel(
        num_solvents=num_solvents,
        hidden_dim=hidden_dim,
        layers=layers,
        node_embedding_dim=node_embedding_dim,
        edge_embedding_dim=edge_embedding_dim,
        solvent_embedding_dim=solvent_embedding_dim,
        dropout=dropout,
        init_seed=init_seed,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    gradient_audit = (
        run_required_gradient_gate(
            model,
            train_examples,
            normalization=fitted,
            loss_config=loss_config,
            seed=init_seed + 90_000_000,
            device=device,
        )
        if require_gradient_gate
        else {}
    )
    history: list[dict[str, object]] = []
    best_epoch = 0
    best_validation = math.inf
    best_model_state: dict[str, torch.Tensor] | None = None
    best_optimizer_state: dict[str, object] | None = None
    stale_epochs = 0
    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(
            model,
            train_examples,
            normalization=fitted,
            optimizer=optimizer,
            batch_size=batch_size,
            seed=init_seed + 2 * epoch,
            device=device,
            masking=masking,
            loss_config=loss_config,
        )
        validation_metrics = _run_epoch(
            model,
            validation_examples,
            normalization=fitted,
            optimizer=None,
            batch_size=batch_size,
            seed=init_seed + 10_000_000,
            device=device,
            masking=masking,
            loss_config=loss_config,
        )
        validation_total = float(validation_metrics["total"])
        improved = validation_total < best_validation - minimum_delta
        if improved:
            best_epoch = epoch
            best_validation = validation_total
            best_model_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            best_optimizer_state = copy.deepcopy(optimizer.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        history.append(
            {
                "epoch": epoch,
                "is_validation_best": improved,
                **{
                    f"train_{name}": value
                    for name, value in train_metrics.items()
                },
                **{
                    f"validation_{name}": value
                    for name, value in validation_metrics.items()
                },
            }
        )
        if epoch >= minimum_epochs and stale_epochs >= patience:
            break
    if best_model_state is None or best_optimizer_state is None:
        raise RuntimeError("Validation did not produce a best state")
    model.load_state_dict(best_model_state, strict=True)
    optimizer.load_state_dict(best_optimizer_state)
    audit_metrics: Mapping[str, float] = {}
    if audit_test_examples:
        audit_metrics = _run_epoch(
            model,
            audit_test_examples,
            normalization=fitted,
            optimizer=None,
            batch_size=batch_size,
            seed=init_seed + 20_000_000,
            device=device,
            masking=masking,
            loss_config=loss_config,
        )
    return SiteNPretrainingResult(
        model=model,
        optimizer=optimizer,
        normalization=fitted,
        history=tuple(history),
        best_epoch=best_epoch,
        best_validation_total=best_validation,
        audit_metrics=dict(audit_metrics),
        gradient_audit=dict(gradient_audit),
    )


def _checkpoint_payload(
    result: SiteNPretrainingResult,
    *,
    masking: MaskingConfig,
    loss_config: SiteNPretrainingLossConfig,
    dataset_contract: Mapping[str, object],
    selection: Mapping[str, object],
) -> dict[str, object]:
    model_state = {
        name: tensor.detach().cpu()
        for name, tensor in result.model.state_dict().items()
    }
    transferable = {
        name: tensor.detach().cpu()
        for name, tensor in result.model.transferable_state_dict().items()
    }
    heads = result.model.pretraining_heads_state_dict()
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "pretraining_schema_version": PRETRAINING_SCHEMA_VERSION,
        "model_architecture": result.model.architecture,
        "initialization_seed": result.model.initialization_seed,
        "model_state_dict": model_state,
        "model_state_sha256": _tensor_mapping_sha256(model_state),
        "transferable_state_dict": transferable,
        "transferable_state_sha256": _tensor_mapping_sha256(transferable),
        "pretraining_heads_state_sha256": _tensor_mapping_sha256(heads),
        "transfer_prefixes": list(TRANSFER_PREFIXES),
        "reset_prefixes": list(RESET_PREFIXES),
        "normalization": result.normalization.to_json(),
        "masking": asdict(masking),
        "loss": asdict(loss_config),
        "history": list(result.history),
        "best_epoch": result.best_epoch,
        "best_validation_total": result.best_validation_total,
        "audit_metrics": dict(result.audit_metrics),
        "gradient_audit": dict(result.gradient_audit),
        "dataset_contract": dict(dataset_contract),
        "selection": dict(selection),
        "tasks": [
            "masked_node_categorical_reconstruction_all_atoms",
            "masked_edge_categorical_reconstruction",
            "masked_local4_reconstruction_all_atoms",
            "masked_global6_reconstruction",
            "heavy_atom_pointwise_mca",
            "heavy_atom_pointwise_gcs53",
            "within_molecule_mca_delta_ranking",
        ],
        "site_probability_normalization": False,
        "fake_proxy_site_types": [],
    }


def save_site_n_pretraining_checkpoint(
    path: str | Path,
    result: SiteNPretrainingResult,
    *,
    masking: MaskingConfig,
    loss_config: SiteNPretrainingLossConfig,
    dataset_contract: Mapping[str, object],
    selection: Mapping[str, object],
) -> dict[str, object]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = _checkpoint_payload(
        result,
        masking=masking,
        loss_config=loss_config,
        dataset_contract=dataset_contract,
        selection=selection,
    )
    temporary = target.with_name(f".{target.name}.tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return payload


def load_site_n_pretraining_checkpoint(
    path: str | Path,
) -> dict[str, object]:
    payload = torch.load(
        Path(path),
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(payload, dict):
        raise ValueError("Pretraining checkpoint is not a mapping")
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Unsupported site-N pretraining checkpoint")
    model_state = payload.get("model_state_dict")
    transferable = payload.get("transferable_state_dict")
    if not isinstance(model_state, dict) or not isinstance(
        transferable, dict
    ):
        raise ValueError("Checkpoint state mappings are missing")
    if _tensor_mapping_sha256(model_state) != payload.get(
        "model_state_sha256"
    ):
        raise ValueError("Full pretraining state hash changed")
    if _tensor_mapping_sha256(transferable) != payload.get(
        "transferable_state_sha256"
    ):
        raise ValueError("Transferable pretraining state hash changed")
    if payload.get("site_probability_normalization") is not False:
        raise ValueError("Checkpoint unexpectedly contains site normalization")
    if payload.get("fake_proxy_site_types") != []:
        raise ValueError("Checkpoint declares fake proxy-site supervision")
    return payload


def transfer_pretrained_backbone(
    downstream: MayrSiteNModel,
    checkpoint: str | Path | Mapping[str, object],
) -> dict[str, object]:
    payload = (
        load_site_n_pretraining_checkpoint(checkpoint)
        if isinstance(checkpoint, (str, Path))
        else dict(checkpoint)
    )
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Unsupported transfer checkpoint")
    source = payload.get("transferable_state_dict")
    if not isinstance(source, dict):
        raise ValueError("Checkpoint has no transferable state")
    target_state = downstream.state_dict()
    expected_names = {
        name
        for name in target_state
        if any(name.startswith(prefix) for prefix in TRANSFER_PREFIXES)
    }
    if set(source) != expected_names:
        missing = sorted(expected_names - set(source))
        unexpected = sorted(set(source) - expected_names)
        raise ValueError(
            f"Transfer key mismatch; missing={missing}, "
            f"unexpected={unexpected}"
        )
    before_reset = {
        name: tensor.detach().cpu().clone()
        for name, tensor in target_state.items()
        if any(name.startswith(prefix) for prefix in RESET_PREFIXES)
    }
    for name in sorted(expected_names):
        source_tensor = source[name]
        if not isinstance(source_tensor, torch.Tensor):
            raise TypeError(f"Transfer value {name} is not a tensor")
        if target_state[name].shape != source_tensor.shape:
            raise ValueError(f"Transfer shape mismatch for {name}")
        target_state[name].copy_(
            source_tensor.to(
                device=target_state[name].device,
                dtype=target_state[name].dtype,
            )
        )
    downstream.load_state_dict(target_state, strict=True)
    after = downstream.state_dict()
    reset_unchanged = all(
        torch.equal(after[name].detach().cpu(), tensor)
        for name, tensor in before_reset.items()
    )
    loaded = {
        name: after[name].detach().cpu()
        for name in expected_names
    }
    source_cpu = {
        name: source[name].detach().cpu()
        for name in expected_names
    }
    exact = all(torch.equal(loaded[name], source_cpu[name]) for name in loaded)
    audit: dict[str, object] = {
        "schema_version": "nucpred.mayr-site-n-transfer-audit.v1",
        "status": "pass" if exact and reset_unchanged else "fail",
        "transferred_parameter_tensors": len(expected_names),
        "transferred_parameter_numel": sum(
            int(tensor.numel()) for tensor in loaded.values()
        ),
        "transferred_state_sha256": _tensor_mapping_sha256(loaded),
        "source_state_sha256": _tensor_mapping_sha256(source_cpu),
        "exact_transfer": exact,
        "reset_modules_unchanged": reset_unchanged,
        "transferred_prefixes": list(TRANSFER_PREFIXES),
        "reset_prefixes": list(RESET_PREFIXES),
        "non_atom_type_adapters_reset": True,
        "site_probability_normalization": False,
    }
    if audit["status"] != "pass":
        raise RuntimeError(f"Strict transfer audit failed: {audit}")
    return audit


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "RESET_PREFIXES",
    "TRANSFER_PREFIXES",
    "SiteNPretrainingLossConfig",
    "SiteNPretrainingModel",
    "SiteNPretrainingOutput",
    "SiteNPretrainingResult",
    "expected_unsupervised_gradient_audit",
    "fit_pretraining_normalization",
    "load_pretraining_examples",
    "load_site_n_pretraining_checkpoint",
    "pack_pretraining_batch",
    "required_gradient_audit",
    "run_required_gradient_gate",
    "save_site_n_pretraining_checkpoint",
    "site_n_pretraining_loss",
    "train_site_n_pretraining",
    "transfer_pretrained_backbone",
    "within_graph_delta_loss",
]
