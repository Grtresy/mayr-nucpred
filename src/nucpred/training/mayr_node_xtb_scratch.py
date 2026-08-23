"""All-atom ordinary-H Mayr model with pre-convolution node electronics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
import os
import random
from typing import Sequence

import numpy as np
import pandas as pd

# Required by CUDA deterministic algorithms for CuBLAS-backed linear layers.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import nn

from nucpred.features.all_atom_graph import (
    EDGE_CATEGORICAL_FEATURES,
    EDGE_CATEGORY_SIZES,
    ELEMENT_VOCABULARY,
    NODE_CATEGORICAL_FEATURES,
    NODE_CATEGORY_SIZES,
)
from nucpred.training.unified_sparse_gnn import (
    CategoricalFeatureEncoder,
    SparseMessageLayer,
)


LOCAL_FEATURES = (
    "alpb_fukui_minus_density",
    "alpb_cm5",
    "alpb_condensed_softness",
    "alpb_condensed_nucleophilicity_index_ev",
)
GLOBAL_FEATURES = (
    "alpb_homo_n_hartree",
    "alpb_softness_hartree_inverse",
    "alpb_vip_hartree",
    "alpb_nucleophilicity_index_hartree",
    "alpb_homo_n_minus_1_hartree",
    "delta_e_solv_hartree",
)
SOLVENT_FEATURES = (
    "solvent_nD",
    "solvent_f(n^2)",
    "solvent_epsilon_r",
    "solvent_ET(30)",
    "solvent_DI",
    "solvent_ES",
    "solvent_alpha_1",
    "solvent_beta_1",
    "solvent_alpha",
    "solvent_beta",
    "solvent_pi_*",
    "solvent_SPP",
    "solvent_SB",
    "solvent_SA",
    "solvent_delta_d",
    "solvent_delta_p",
    "solvent_delta_h",
    "solvent_delta",
)
ARMS = ("A", "B", "C", "D")


@dataclass(frozen=True, slots=True)
class SolventVocabulary:
    tokens: tuple[str, ...]

    @classmethod
    def from_values(cls, values: Sequence[object]) -> "SolventVocabulary":
        tokens = tuple(sorted({str(value).strip() for value in values}))
        return cls(("<UNK>", *tokens))

    def encode(self, value: object) -> int:
        token = str(value).strip()
        try:
            return self.tokens.index(token)
        except ValueError:
            return 0


@dataclass(frozen=True, slots=True)
class MayrNodeXtbExample:
    source_id: str
    model_canonical_smiles: str
    node_categorical: torch.Tensor
    edge_index: torch.Tensor
    edge_categorical: torch.Tensor
    local_values: np.ndarray
    local_mask: np.ndarray
    global_values: np.ndarray
    global_mask: np.ndarray
    solvent_values: np.ndarray
    solvent_raw: str
    model_formal_charge: float
    n_target: float
    site_target_indices: tuple[int, ...]
    site_target_mask: bool
    supervision_level: str
    spectator_stripped: bool

    @property
    def num_nodes(self) -> int:
        return int(self.node_categorical.shape[0])


@dataclass(frozen=True, slots=True)
class FoldPreprocessor:
    local_median: tuple[float, ...]
    local_mean: tuple[float, ...]
    local_scale: tuple[float, ...]
    global_median: tuple[float, ...]
    global_mean: tuple[float, ...]
    global_scale: tuple[float, ...]
    solvent_mean: tuple[float, ...]
    solvent_scale: tuple[float, ...]
    charge_mean: float
    charge_scale: float
    target_mean: float
    target_scale: float
    fit_source_id_sha256: str
    fit_record_count: int

    def to_json(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ModelInputs:
    node_categorical: torch.Tensor
    edge_index: torch.Tensor
    edge_categorical: torch.Tensor
    node_graph_index: torch.Tensor
    graph_ptr: torch.Tensor
    node_local: torch.Tensor
    solvent_continuous: torch.Tensor
    solvent_index: torch.Tensor
    molecular_formal_charge: torch.Tensor
    global_xtb: torch.Tensor

    @property
    def num_graphs(self) -> int:
        return int(self.graph_ptr.shape[0]) - 1

    def to(self, device: str | torch.device) -> "ModelInputs":
        return ModelInputs(
            node_categorical=self.node_categorical.to(device),
            edge_index=self.edge_index.to(device),
            edge_categorical=self.edge_categorical.to(device),
            node_graph_index=self.node_graph_index.to(device),
            graph_ptr=self.graph_ptr.to(device),
            node_local=self.node_local.to(device),
            solvent_continuous=self.solvent_continuous.to(device),
            solvent_index=self.solvent_index.to(device),
            molecular_formal_charge=self.molecular_formal_charge.to(device),
            global_xtb=self.global_xtb.to(device),
        )


@dataclass(frozen=True)
class TrainingBatch:
    inputs: ModelInputs
    n_target_standardized: torch.Tensor
    n_target_raw: torch.Tensor
    site_targets: tuple[tuple[int, ...], ...]
    site_target_mask: torch.Tensor
    source_ids: tuple[str, ...]

    def to(self, device: str | torch.device) -> "TrainingBatch":
        return TrainingBatch(
            inputs=self.inputs.to(device),
            n_target_standardized=self.n_target_standardized.to(device),
            n_target_raw=self.n_target_raw.to(device),
            site_targets=self.site_targets,
            site_target_mask=self.site_target_mask.to(device),
            source_ids=self.source_ids,
        )


@dataclass(frozen=True)
class MayrNodeXtbOutput:
    n_prediction_standardized: torch.Tensor
    site_logits: torch.Tensor
    site_distribution: torch.Tensor
    graph_pool: torch.Tensor
    site_pool: torch.Tensor


def _parse_json_matrix(value: object, *, width: int | None = None) -> list[list[int]]:
    parsed = json.loads(str(value)) if isinstance(value, str) else value
    if not isinstance(parsed, list):
        raise ValueError("Expected a JSON matrix")
    rows = [list(map(int, row)) for row in parsed]
    if width is not None and any(len(row) != width for row in rows):
        raise ValueError("JSON matrix width changed")
    return rows


def _parse_json_indices(value: object) -> tuple[int, ...]:
    parsed = json.loads(str(value)) if isinstance(value, str) else value
    if not isinstance(parsed, list):
        raise ValueError("Expected a JSON index list")
    return tuple(int(item) for item in parsed)


def _finite_or_nan(value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return math.nan
    return numeric if math.isfinite(numeric) else math.nan


def load_examples(
    records: pd.DataFrame,
    atom_features: pd.DataFrame,
    molecule_features: pd.DataFrame,
) -> tuple[list[MayrNodeXtbExample], SolventVocabulary]:
    """Load targets beside, never inside, the model-input dataclass."""

    if records["source_id"].nunique() != len(records):
        raise ValueError("Record source IDs must be unique")
    atom_groups = {
        str(source_id): group.sort_values("atom_index")
        for source_id, group in atom_features.groupby("source_id", sort=False)
    }
    molecule_by_id = molecule_features.set_index("source_id", drop=False)
    examples: list[MayrNodeXtbExample] = []
    for row in records.sort_values("cohort_index").itertuples(index=False):
        source_id = str(row.source_id)
        atoms = atom_groups[source_id]
        molecule = molecule_by_id.loc[source_id]
        node_rows = _parse_json_matrix(
            row.model_node_categorical_json,
            width=len(NODE_CATEGORICAL_FEATURES),
        )
        edges = _parse_json_matrix(row.model_directed_edges_json, width=2)
        edge_rows = _parse_json_matrix(
            row.model_edge_categorical_json,
            width=len(EDGE_CATEGORICAL_FEATURES),
        )
        num_nodes = int(row.model_all_atom_count)
        if (
            len(node_rows) != num_nodes
            or len(atoms) != num_nodes
            or len(edges) != len(edge_rows)
        ):
            raise ValueError(f"{source_id}: graph table alignment failed")
        edge_index = (
            torch.tensor(edges, dtype=torch.long).transpose(0, 1).contiguous()
            if edges
            else torch.empty((2, 0), dtype=torch.long)
        )
        local_values = atoms[list(LOCAL_FEATURES)].to_numpy(dtype=float)
        local_mask = atoms[
            [f"{name}__available" for name in LOCAL_FEATURES]
        ].to_numpy(dtype=bool)
        global_values = np.asarray(
            [_finite_or_nan(molecule[name]) for name in GLOBAL_FEATURES],
            dtype=float,
        )
        global_mask = np.asarray(
            [bool(molecule[f"{name}__available"]) for name in GLOBAL_FEATURES],
            dtype=bool,
        )
        solvent = np.asarray(
            [_finite_or_nan(molecule[name]) for name in SOLVENT_FEATURES],
            dtype=float,
        )
        targets = _parse_json_indices(row.site_target_atoms_model_json)
        if bool(row.site_target_mask_model) and not targets:
            raise ValueError(f"{source_id}: supervised target set is empty")
        if any(index < 0 or index >= num_nodes for index in targets):
            raise ValueError(f"{source_id}: site target is outside graph")
        examples.append(
            MayrNodeXtbExample(
                source_id=source_id,
                model_canonical_smiles=str(row.model_canonical_smiles),
                node_categorical=torch.tensor(node_rows, dtype=torch.long),
                edge_index=edge_index,
                edge_categorical=(
                    torch.tensor(edge_rows, dtype=torch.long)
                    if edge_rows
                    else torch.empty(
                        (0, len(EDGE_CATEGORICAL_FEATURES)),
                        dtype=torch.long,
                    )
                ),
                local_values=local_values,
                local_mask=local_mask,
                global_values=global_values,
                global_mask=global_mask,
                solvent_values=solvent,
                solvent_raw=str(molecule.solvent_raw),
                model_formal_charge=float(molecule.model_formal_charge),
                n_target=float(row.N),
                site_target_indices=targets,
                site_target_mask=bool(row.site_target_mask_model),
                supervision_level=str(row.supervision_level),
                spectator_stripped=bool(row.spectator_stripped),
            )
        )
    vocabulary = SolventVocabulary.from_values(
        [example.solvent_raw for example in examples]
    )
    if ELEMENT_VOCABULARY.index("H") == 0:
        raise ValueError("H must have an ordinary non-unknown element index")
    return examples, vocabulary


def _feature_statistics(
    values: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if values.shape != mask.shape or values.ndim != 2:
        raise ValueError("Feature values and masks must be aligned matrices")
    medians: list[float] = []
    means: list[float] = []
    scales: list[float] = []
    for index in range(values.shape[1]):
        observed = values[:, index][mask[:, index] & np.isfinite(values[:, index])]
        if observed.size == 0:
            raise ValueError(f"Training fold has no feature {index} observations")
        median = float(np.median(observed))
        mean = float(np.mean(observed))
        scale = float(np.std(observed))
        medians.append(median)
        means.append(mean)
        scales.append(scale if scale > 1e-8 else 1.0)
    return np.asarray(medians), np.asarray(means), np.asarray(scales)


def fit_preprocessor(examples: Sequence[MayrNodeXtbExample]) -> FoldPreprocessor:
    if not examples:
        raise ValueError("Cannot fit preprocessing on an empty fold")
    local_values = np.concatenate([example.local_values for example in examples])
    local_mask = np.concatenate([example.local_mask for example in examples])
    global_values = np.stack([example.global_values for example in examples])
    global_mask = np.stack([example.global_mask for example in examples])
    local_median, local_mean, local_scale = _feature_statistics(
        local_values, local_mask
    )
    global_median, global_mean, global_scale = _feature_statistics(
        global_values, global_mask
    )
    solvent_values = np.stack([example.solvent_values for example in examples])
    if not np.isfinite(solvent_values).all():
        raise ValueError("Solvent18 contains missing values in a training fold")
    solvent_mean = solvent_values.mean(axis=0)
    solvent_scale = solvent_values.std(axis=0)
    solvent_scale[solvent_scale <= 1e-8] = 1.0
    charges = np.asarray(
        [example.model_formal_charge for example in examples], dtype=float
    )
    targets = np.asarray([example.n_target for example in examples], dtype=float)
    charge_scale = float(charges.std())
    target_scale = float(targets.std())
    source_ids = sorted(example.source_id for example in examples)
    return FoldPreprocessor(
        local_median=tuple(map(float, local_median)),
        local_mean=tuple(map(float, local_mean)),
        local_scale=tuple(map(float, local_scale)),
        global_median=tuple(map(float, global_median)),
        global_mean=tuple(map(float, global_mean)),
        global_scale=tuple(map(float, global_scale)),
        solvent_mean=tuple(map(float, solvent_mean)),
        solvent_scale=tuple(map(float, solvent_scale)),
        charge_mean=float(charges.mean()),
        charge_scale=charge_scale if charge_scale > 1e-8 else 1.0,
        target_mean=float(targets.mean()),
        target_scale=target_scale if target_scale > 1e-8 else 1.0,
        fit_source_id_sha256=hashlib.sha256(
            ("\n".join(source_ids) + "\n").encode("utf-8")
        ).hexdigest(),
        fit_record_count=len(examples),
    )


def _transform(
    values: np.ndarray,
    mask: np.ndarray,
    *,
    median: Sequence[float],
    mean: Sequence[float],
    scale: Sequence[float],
) -> np.ndarray:
    medians = np.asarray(median, dtype=float)
    means = np.asarray(mean, dtype=float)
    scales = np.asarray(scale, dtype=float)
    filled = np.where(mask & np.isfinite(values), values, medians)
    return (filled - means) / scales


def pack_batch(
    examples: Sequence[MayrNodeXtbExample],
    *,
    preprocessor: FoldPreprocessor,
    solvent_vocabulary: SolventVocabulary,
    arm: str,
) -> TrainingBatch:
    if arm not in ARMS:
        raise ValueError(f"Unknown A--D arm {arm!r}")
    if not examples:
        raise ValueError("Cannot pack an empty batch")
    node_categorical: list[torch.Tensor] = []
    edge_indices: list[torch.Tensor] = []
    edge_categorical: list[torch.Tensor] = []
    graph_index: list[torch.Tensor] = []
    graph_ptr = [0]
    local_parts: list[torch.Tensor] = []
    solvent_parts: list[torch.Tensor] = []
    solvent_indices: list[int] = []
    charge_parts: list[float] = []
    global_parts: list[torch.Tensor] = []
    site_targets: list[tuple[int, ...]] = []
    site_masks: list[bool] = []
    targets: list[float] = []
    offset = 0
    local_enabled = arm in {"C", "D"}
    global_enabled = arm in {"B", "D"}
    for graph_number, example in enumerate(examples):
        node_categorical.append(example.node_categorical)
        edge_indices.append(example.edge_index + offset)
        edge_categorical.append(example.edge_categorical)
        graph_index.append(
            torch.full((example.num_nodes,), graph_number, dtype=torch.long)
        )
        local = _transform(
            example.local_values,
            example.local_mask,
            median=preprocessor.local_median,
            mean=preprocessor.local_mean,
            scale=preprocessor.local_scale,
        )
        local_with_mask = np.concatenate(
            (local, example.local_mask.astype(float)), axis=1
        )
        if not local_enabled:
            local_with_mask = np.zeros_like(local_with_mask)
        local_parts.append(torch.tensor(local_with_mask, dtype=torch.float32))
        solvent_parts.append(
            torch.tensor(
                (
                    example.solvent_values
                    - np.asarray(preprocessor.solvent_mean)
                )
                / np.asarray(preprocessor.solvent_scale),
                dtype=torch.float32,
            )
        )
        solvent_indices.append(
            solvent_vocabulary.encode(example.solvent_raw)
        )
        charge_parts.append(
            (
                example.model_formal_charge - preprocessor.charge_mean
            )
            / preprocessor.charge_scale
        )
        global_values = _transform(
            example.global_values.reshape(1, -1),
            example.global_mask.reshape(1, -1),
            median=preprocessor.global_median,
            mean=preprocessor.global_mean,
            scale=preprocessor.global_scale,
        )[0]
        global_with_mask = np.concatenate(
            (global_values, example.global_mask.astype(float))
        )
        if not global_enabled:
            global_with_mask = np.zeros_like(global_with_mask)
        global_parts.append(
            torch.tensor(global_with_mask, dtype=torch.float32)
        )
        site_targets.append(example.site_target_indices)
        site_masks.append(example.site_target_mask)
        targets.append(example.n_target)
        offset += example.num_nodes
        graph_ptr.append(offset)
    raw_target = torch.tensor(targets, dtype=torch.float32)
    standardized_target = (
        raw_target - preprocessor.target_mean
    ) / preprocessor.target_scale
    return TrainingBatch(
        inputs=ModelInputs(
            node_categorical=torch.cat(node_categorical),
            edge_index=torch.cat(edge_indices, dim=1),
            edge_categorical=torch.cat(edge_categorical),
            node_graph_index=torch.cat(graph_index),
            graph_ptr=torch.tensor(graph_ptr, dtype=torch.long),
            node_local=torch.cat(local_parts),
            solvent_continuous=torch.stack(solvent_parts),
            solvent_index=torch.tensor(solvent_indices, dtype=torch.long),
            molecular_formal_charge=torch.tensor(
                charge_parts, dtype=torch.float32
            ).unsqueeze(-1),
            global_xtb=torch.stack(global_parts),
        ),
        n_target_standardized=standardized_target,
        n_target_raw=raw_target,
        site_targets=tuple(site_targets),
        site_target_mask=torch.tensor(site_masks, dtype=torch.bool),
        source_ids=tuple(example.source_id for example in examples),
    )


def _segment_mean(
    values: torch.Tensor, graph_index: torch.Tensor, num_graphs: int
) -> torch.Tensor:
    pooled = values.new_zeros((num_graphs, values.shape[-1]))
    pooled.index_add_(0, graph_index, values)
    counts = values.new_zeros((num_graphs, 1))
    counts.index_add_(
        0,
        graph_index,
        torch.ones(
            (values.shape[0], 1),
            dtype=values.dtype,
            device=values.device,
        ),
    )
    return pooled / counts.clamp_min(1.0)


def _segment_softmax(
    logits: torch.Tensor, graph_ptr: torch.Tensor
) -> torch.Tensor:
    probabilities = torch.empty_like(logits)
    for graph_index in range(int(graph_ptr.shape[0]) - 1):
        start = int(graph_ptr[graph_index])
        end = int(graph_ptr[graph_index + 1])
        probabilities[start:end] = torch.softmax(logits[start:end], dim=0)
    return probabilities


class MayrOrdinaryHNodeXtbGNN(nn.Module):
    """Capacity-matched A--D model; only arm input values differ."""

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
        super().__init__()
        if num_solvents < 1:
            raise ValueError("Solvent vocabulary cannot be empty")
        self.node_encoder = CategoricalFeatureEncoder(
            NODE_CATEGORY_SIZES,
            embedding_dim=node_embedding_dim,
            output_dim=hidden_dim,
        )
        self.local_encoder = nn.Sequential(
            nn.Linear(2 * len(LOCAL_FEATURES), hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.edge_encoder = CategoricalFeatureEncoder(
            EDGE_CATEGORY_SIZES,
            embedding_dim=edge_embedding_dim,
            output_dim=hidden_dim,
        )
        self.message_layers = nn.ModuleList(
            SparseMessageLayer(hidden_dim, dropout=dropout)
            for _ in range(layers)
        )
        self.site_head = nn.Linear(hidden_dim, 1)
        self.solvent_encoder = nn.Sequential(
            nn.Linear(len(SOLVENT_FEATURES), hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.solvent_embedding = nn.Embedding(
            num_solvents, solvent_embedding_dim
        )
        self.solvent_embedding_projection = nn.Sequential(
            nn.Linear(solvent_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.charge_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.global_xtb_encoder = nn.Sequential(
            nn.Linear(2 * len(GLOBAL_FEATURES), hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.regression_head = nn.Sequential(
            nn.Linear(6 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, inputs: ModelInputs) -> MayrNodeXtbOutput:
        nodes = self.node_encoder(inputs.node_categorical)
        nodes = nodes + self.local_encoder(inputs.node_local)
        edges = self.edge_encoder(inputs.edge_categorical)
        for layer in self.message_layers:
            nodes = layer(nodes, inputs.edge_index, edges)
        graph_pool = _segment_mean(
            nodes, inputs.node_graph_index, inputs.num_graphs
        )
        site_logits = self.site_head(nodes).squeeze(-1)
        site_distribution = _segment_softmax(
            site_logits, inputs.graph_ptr
        )
        site_pool = nodes.new_zeros((inputs.num_graphs, nodes.shape[-1]))
        site_pool.index_add_(
            0,
            inputs.node_graph_index,
            nodes * site_distribution.unsqueeze(-1),
        )
        fused = torch.cat(
            (
                graph_pool,
                site_pool,
                self.solvent_encoder(inputs.solvent_continuous),
                self.solvent_embedding_projection(
                    self.solvent_embedding(inputs.solvent_index)
                ),
                self.charge_encoder(inputs.molecular_formal_charge),
                self.global_xtb_encoder(inputs.global_xtb),
            ),
            dim=-1,
        )
        prediction = self.regression_head(fused).squeeze(-1)
        return MayrNodeXtbOutput(
            n_prediction_standardized=prediction,
            site_logits=site_logits,
            site_distribution=site_distribution,
            graph_pool=graph_pool,
            site_pool=site_pool,
        )


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def initialization_sha256(model: nn.Module) -> str:
    buffer = io.BytesIO()
    torch.save(
        {key: value.detach().cpu() for key, value in model.state_dict().items()},
        buffer,
    )
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def group_site_nll(
    distribution: torch.Tensor,
    graph_ptr: torch.Tensor,
    targets: Sequence[Sequence[int]],
    mask: torch.Tensor,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for graph_index, indices in enumerate(targets):
        if not bool(mask[graph_index]):
            continue
        start = int(graph_ptr[graph_index])
        selected = torch.tensor(
            [start + int(index) for index in indices],
            dtype=torch.long,
            device=distribution.device,
        )
        mass = distribution[selected].sum().clamp_min(1e-12)
        losses.append(-torch.log(mass))
    if not losses:
        return distribution.sum() * 0.0
    return torch.stack(losses).mean()


def batch_loss(
    output: MayrNodeXtbOutput,
    batch: TrainingBatch,
    *,
    site_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    regression = torch.mean(
        (
            output.n_prediction_standardized
            - batch.n_target_standardized
        )
        ** 2
    )
    site = group_site_nll(
        output.site_distribution,
        batch.inputs.graph_ptr,
        batch.site_targets,
        batch.site_target_mask,
    )
    return regression + float(site_weight) * site, regression, site
