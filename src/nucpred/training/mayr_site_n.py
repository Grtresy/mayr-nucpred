"""Typed site-object GNN for independent Mayr N prediction.

One graph is encoded once per molecule/solvent context.  Atom, bond, region,
atom-group, and transferable-H-group queries are pooled independently and each
receives its own scalar N prediction.  There is intentionally no site softmax
or graph-level competition between site objects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Collection, Mapping, Sequence

import numpy as np
import pandas as pd

# Required by deterministic CUDA linear algebra.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import nn
from torch.nn import functional as F

from nucpred.datasets.mayr_site_n import SITE_TYPES
from nucpred.features.all_atom_graph import (
    EDGE_CATEGORY_SIZES,
    NODE_CATEGORY_SIZES,
)
from nucpred.training.mayr_node_xtb_scratch import (
    GLOBAL_FEATURES,
    LOCAL_FEATURES,
    SOLVENT_FEATURES,
    SolventVocabulary,
)
from nucpred.training.unified_sparse_gnn import (
    CategoricalFeatureEncoder,
    SparseMessageLayer,
)


SITE_TYPE_NAMES = tuple(SITE_TYPES)
SITE_TYPE_TO_INDEX = {
    name: index for index, name in enumerate(SITE_TYPE_NAMES)
}
MODEL_SCHEMA_VERSION = "nucpred.mayr-site-n-model.v1"


@dataclass(frozen=True, slots=True)
class SiteNExample:
    context_id: str
    species_id: str
    connectivity_id: str
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
    target_ids: tuple[str, ...]
    site_object_ids: tuple[str, ...]
    site_types: tuple[str, ...]
    site_members: tuple[tuple[int, ...], ...]
    n_targets: np.ndarray

    @property
    def num_nodes(self) -> int:
        return int(self.node_categorical.shape[0])

    @property
    def num_sites(self) -> int:
        return len(self.target_ids)


@dataclass(frozen=True, slots=True)
class SiteNFoldPreprocessor:
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
    fit_context_ids_sha256: str
    fit_context_count: int
    fit_target_count: int

    def to_json(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_json(cls, payload: Mapping[str, object]) -> "SiteNFoldPreprocessor":
        tuple_fields = (
            "local_median",
            "local_mean",
            "local_scale",
            "global_median",
            "global_mean",
            "global_scale",
            "solvent_mean",
            "solvent_scale",
        )
        values = dict(payload)
        for field in tuple_fields:
            values[field] = tuple(float(value) for value in values[field])
        return cls(**values)


@dataclass(frozen=True)
class SiteNModelInputs:
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
    site_member_index: torch.Tensor
    site_member_ptr: torch.Tensor
    site_graph_index: torch.Tensor
    site_type_index: torch.Tensor

    @property
    def num_graphs(self) -> int:
        return int(self.graph_ptr.shape[0]) - 1

    @property
    def num_sites(self) -> int:
        return int(self.site_graph_index.shape[0])

    def to(self, device: str | torch.device) -> "SiteNModelInputs":
        return SiteNModelInputs(
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
            site_member_index=self.site_member_index.to(device),
            site_member_ptr=self.site_member_ptr.to(device),
            site_graph_index=self.site_graph_index.to(device),
            site_type_index=self.site_type_index.to(device),
        )


@dataclass(frozen=True)
class SiteNTrainingBatch:
    inputs: SiteNModelInputs
    n_target_standardized: torch.Tensor
    n_target_raw: torch.Tensor
    context_ids: tuple[str, ...]
    target_ids: tuple[str, ...]
    site_object_ids: tuple[str, ...]

    def to(self, device: str | torch.device) -> "SiteNTrainingBatch":
        return SiteNTrainingBatch(
            inputs=self.inputs.to(device),
            n_target_standardized=self.n_target_standardized.to(device),
            n_target_raw=self.n_target_raw.to(device),
            context_ids=self.context_ids,
            target_ids=self.target_ids,
            site_object_ids=self.site_object_ids,
        )


@dataclass(frozen=True)
class SiteNOutput:
    n_prediction_standardized: torch.Tensor
    node_embeddings: torch.Tensor
    graph_pool: torch.Tensor
    site_embeddings: torch.Tensor
    site_summary: torch.Tensor


@dataclass(frozen=True)
class SiteNFusedFeatures:
    """Shared encoded features consumed by independent site-level heads."""

    node_embeddings: torch.Tensor
    graph_pool: torch.Tensor
    site_embeddings: torch.Tensor
    site_summary: torch.Tensor
    fused: torch.Tensor


def _parse_matrix(
    value: object,
    *,
    width: int,
    dtype: type = float,
) -> np.ndarray:
    parsed = json.loads(value) if isinstance(value, str) else value
    array = np.asarray(parsed, dtype=dtype)
    if array.size == 0:
        return np.empty((0, width), dtype=dtype)
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"Expected a matrix with width {width}")
    return array


def _parse_vector(
    value: object,
    *,
    width: int,
    dtype: type = float,
) -> np.ndarray:
    parsed = json.loads(value) if isinstance(value, str) else value
    array = np.asarray(parsed, dtype=dtype).reshape(-1)
    if array.shape != (width,):
        raise ValueError(f"Expected a vector with width {width}")
    return array


def _parse_int_list(value: object) -> tuple[int, ...]:
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, list):
        raise ValueError("Expected a JSON atom-index list")
    return tuple(int(item) for item in parsed)


def load_site_n_examples(
    dataset_directory: str | Path,
    *,
    split_seed: int | None = None,
    role: str | None = None,
    target_ids: Collection[str] | None = None,
    load_target_values: bool = True,
) -> list[SiteNExample]:
    root = Path(dataset_directory)
    contexts = pd.read_parquet(root / "contexts.parquet")
    selected_ids: set[str] | None = None
    if split_seed is not None or role is not None:
        if split_seed is None or role is None:
            raise ValueError("split_seed and role must be provided together")
        if target_ids is not None:
            raise ValueError("target_ids cannot be combined with split_seed/role")
        membership = pd.read_csv(root / "split_membership.csv")
        selected = membership.loc[
            membership["split_seed"].eq(int(split_seed))
            & membership["role"].eq(str(role)),
            "target_id",
        ].astype(str)
        selected_ids = set(selected)
        expected_count = len(selected)
    elif target_ids is not None:
        selected_ids = {str(value) for value in target_ids}
        if not selected_ids:
            raise ValueError("target_ids cannot be empty")
        expected_count = len(selected_ids)
    else:
        expected_count = None
    target_columns = [
        "context_id",
        "target_id",
        "site_object_id",
        "site_type",
        "member_atom_indices_json",
    ]
    if load_target_values:
        target_columns.append("N_mean")
    if selected_ids is None:
        targets = pd.read_parquet(
            root / "targets.parquet",
            columns=target_columns,
        )
    else:
        targets = pd.read_parquet(
            root / "targets.parquet",
            columns=target_columns,
            filters=[("target_id", "in", sorted(selected_ids))],
        )
        targets = targets.loc[
            targets["target_id"].astype(str).isin(selected_ids)
        ]
        if len(targets) != expected_count:
            raise ValueError("Split membership and target table changed")
    context_index = contexts.set_index("context_id", drop=False)
    examples: list[SiteNExample] = []
    for context_id, group in targets.groupby("context_id", sort=True):
        context = context_index.loc[str(context_id)]
        node_categorical = torch.tensor(
            _parse_matrix(
                context["model_node_categorical_json"],
                width=len(NODE_CATEGORY_SIZES),
                dtype=int,
            ),
            dtype=torch.long,
        )
        edge_pairs = _parse_matrix(
            context["model_directed_edges_json"],
            width=2,
            dtype=int,
        )
        edge_index = torch.tensor(edge_pairs.T, dtype=torch.long)
        edge_categorical = torch.tensor(
            _parse_matrix(
                context["model_edge_categorical_json"],
                width=len(EDGE_CATEGORY_SIZES),
                dtype=int,
            ),
            dtype=torch.long,
        )
        if edge_index.shape[1] != edge_categorical.shape[0]:
            raise ValueError(f"{context_id}: edge feature count changed")
        local_values = _parse_matrix(
            context["node_local4_json"],
            width=len(LOCAL_FEATURES),
            dtype=float,
        )
        local_mask = _parse_matrix(
            context["node_local4_available_json"],
            width=len(LOCAL_FEATURES),
            dtype=bool,
        ).astype(bool)
        if local_values.shape[0] != node_categorical.shape[0]:
            raise ValueError(f"{context_id}: local feature count changed")
        global_values = _parse_vector(
            context["molecule_global6_json"],
            width=len(GLOBAL_FEATURES),
            dtype=float,
        )
        global_mask = _parse_vector(
            context["molecule_global6_available_json"],
            width=len(GLOBAL_FEATURES),
            dtype=bool,
        ).astype(bool)
        solvent_values = np.asarray(
            [float(context[column]) for column in SOLVENT_FEATURES],
            dtype=float,
        )
        ordered = group.sort_values("target_id")
        site_members = tuple(
            _parse_int_list(value)
            for value in ordered["member_atom_indices_json"]
        )
        if any(
            not members
            or any(
                index < 0 or index >= node_categorical.shape[0]
                for index in members
            )
            for members in site_members
        ):
            raise ValueError(f"{context_id}: invalid site-object members")
        site_types = tuple(ordered["site_type"].astype(str))
        unknown = sorted(set(site_types) - set(SITE_TYPE_NAMES))
        if unknown:
            raise ValueError(f"{context_id}: unknown site types {unknown}")
        examples.append(
            SiteNExample(
                context_id=str(context_id),
                species_id=str(context["species_id"]),
                connectivity_id=str(context["connectivity_id"]),
                node_categorical=node_categorical,
                edge_index=edge_index,
                edge_categorical=edge_categorical,
                local_values=local_values,
                local_mask=local_mask,
                global_values=global_values,
                global_mask=global_mask,
                solvent_values=solvent_values,
                solvent_raw=str(context["solvent_raw"]),
                model_formal_charge=float(context["model_formal_charge"]),
                target_ids=tuple(ordered["target_id"].astype(str)),
                site_object_ids=tuple(ordered["site_object_id"].astype(str)),
                site_types=site_types,
                site_members=site_members,
                n_targets=(
                    ordered["N_mean"].to_numpy(dtype=float)
                    if load_target_values
                    else np.zeros(len(ordered), dtype=float)
                ),
            )
        )
    return examples


def _feature_statistics(
    values: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if values.shape != mask.shape or values.ndim != 2:
        raise ValueError("Feature values and masks must be aligned matrices")
    medians = np.zeros(values.shape[1], dtype=float)
    means = np.zeros(values.shape[1], dtype=float)
    scales = np.ones(values.shape[1], dtype=float)
    for column in range(values.shape[1]):
        selected = values[:, column][
            mask[:, column] & np.isfinite(values[:, column])
        ]
        if selected.size:
            medians[column] = float(np.median(selected))
            means[column] = float(np.mean(selected))
            scale = float(np.std(selected))
            scales[column] = scale if scale > 1e-8 else 1.0
    return medians, means, scales


def _safe_scalar_stats(values: np.ndarray) -> tuple[float, float]:
    mean = float(np.mean(values))
    scale = float(np.std(values))
    return mean, scale if scale > 1e-8 else 1.0


def fit_site_n_preprocessor(
    examples: Sequence[SiteNExample],
) -> SiteNFoldPreprocessor:
    if not examples:
        raise ValueError("Cannot fit a preprocessor on no contexts")
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
    solvent = np.stack([example.solvent_values for example in examples])
    solvent_mean = solvent.mean(axis=0)
    solvent_scale = solvent.std(axis=0)
    solvent_scale[solvent_scale <= 1e-8] = 1.0
    charge = np.asarray(
        [example.model_formal_charge for example in examples],
        dtype=float,
    )
    target = np.concatenate([example.n_targets for example in examples])
    charge_mean, charge_scale = _safe_scalar_stats(charge)
    target_mean, target_scale = _safe_scalar_stats(target)
    context_ids = sorted(example.context_id for example in examples)
    digest = hashlib.sha256(
        "\0".join(context_ids).encode("utf-8")
    ).hexdigest()
    return SiteNFoldPreprocessor(
        local_median=tuple(map(float, local_median)),
        local_mean=tuple(map(float, local_mean)),
        local_scale=tuple(map(float, local_scale)),
        global_median=tuple(map(float, global_median)),
        global_mean=tuple(map(float, global_mean)),
        global_scale=tuple(map(float, global_scale)),
        solvent_mean=tuple(map(float, solvent_mean)),
        solvent_scale=tuple(map(float, solvent_scale)),
        charge_mean=charge_mean,
        charge_scale=charge_scale,
        target_mean=target_mean,
        target_scale=target_scale,
        fit_context_ids_sha256=digest,
        fit_context_count=len(examples),
        fit_target_count=int(sum(example.num_sites for example in examples)),
    )


def _transform_masked(
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
    observed = mask & np.isfinite(values)
    filled = np.where(observed, values, medians)
    standard = (filled - means) / scales
    return np.concatenate((standard, observed.astype(float)), axis=-1)


def pack_site_n_batch(
    examples: Sequence[SiteNExample],
    *,
    preprocessor: SiteNFoldPreprocessor,
    solvent_vocabulary: SolventVocabulary,
) -> SiteNTrainingBatch:
    if not examples:
        raise ValueError("Cannot pack an empty site-N batch")
    node_parts: list[torch.Tensor] = []
    edge_parts: list[torch.Tensor] = []
    edge_feature_parts: list[torch.Tensor] = []
    node_graph_parts: list[torch.Tensor] = []
    local_parts: list[torch.Tensor] = []
    solvent_parts: list[torch.Tensor] = []
    solvent_indices: list[int] = []
    charge_parts: list[float] = []
    global_parts: list[torch.Tensor] = []
    site_member_parts: list[torch.Tensor] = []
    site_member_ptr = [0]
    site_graph_parts: list[int] = []
    site_type_parts: list[int] = []
    target_parts: list[torch.Tensor] = []
    context_ids: list[str] = []
    target_ids: list[str] = []
    site_object_ids: list[str] = []
    graph_ptr = [0]
    node_offset = 0
    for graph_index, example in enumerate(examples):
        node_parts.append(example.node_categorical)
        edge_parts.append(example.edge_index + node_offset)
        edge_feature_parts.append(example.edge_categorical)
        node_graph_parts.append(
            torch.full((example.num_nodes,), graph_index, dtype=torch.long)
        )
        local_parts.append(
            torch.tensor(
                _transform_masked(
                    example.local_values,
                    example.local_mask,
                    median=preprocessor.local_median,
                    mean=preprocessor.local_mean,
                    scale=preprocessor.local_scale,
                ),
                dtype=torch.float32,
            )
        )
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
        solvent_indices.append(solvent_vocabulary.encode(example.solvent_raw))
        charge_parts.append(
            (
                example.model_formal_charge - preprocessor.charge_mean
            )
            / preprocessor.charge_scale
        )
        global_parts.append(
            torch.tensor(
                _transform_masked(
                    example.global_values.reshape(1, -1),
                    example.global_mask.reshape(1, -1),
                    median=preprocessor.global_median,
                    mean=preprocessor.global_mean,
                    scale=preprocessor.global_scale,
                )[0],
                dtype=torch.float32,
            )
        )
        for site_type, members in zip(
            example.site_types,
            example.site_members,
            strict=True,
        ):
            global_members = torch.tensor(
                [node_offset + index for index in members],
                dtype=torch.long,
            )
            site_member_parts.append(global_members)
            site_member_ptr.append(
                site_member_ptr[-1] + int(global_members.shape[0])
            )
            site_graph_parts.append(graph_index)
            site_type_parts.append(SITE_TYPE_TO_INDEX[site_type])
        target_parts.append(torch.tensor(example.n_targets, dtype=torch.float32))
        context_ids.extend([example.context_id] * example.num_sites)
        target_ids.extend(example.target_ids)
        site_object_ids.extend(example.site_object_ids)
        node_offset += example.num_nodes
        graph_ptr.append(node_offset)
    raw_target = torch.cat(target_parts)
    standardized_target = (
        raw_target - float(preprocessor.target_mean)
    ) / float(preprocessor.target_scale)
    return SiteNTrainingBatch(
        inputs=SiteNModelInputs(
            node_categorical=torch.cat(node_parts),
            edge_index=torch.cat(edge_parts, dim=1),
            edge_categorical=torch.cat(edge_feature_parts),
            node_graph_index=torch.cat(node_graph_parts),
            graph_ptr=torch.tensor(graph_ptr, dtype=torch.long),
            node_local=torch.cat(local_parts),
            solvent_continuous=torch.stack(solvent_parts),
            solvent_index=torch.tensor(solvent_indices, dtype=torch.long),
            molecular_formal_charge=torch.tensor(
                charge_parts, dtype=torch.float32
            ).unsqueeze(-1),
            global_xtb=torch.stack(global_parts),
            site_member_index=torch.cat(site_member_parts),
            site_member_ptr=torch.tensor(site_member_ptr, dtype=torch.long),
            site_graph_index=torch.tensor(site_graph_parts, dtype=torch.long),
            site_type_index=torch.tensor(site_type_parts, dtype=torch.long),
        ),
        n_target_standardized=standardized_target,
        n_target_raw=raw_target,
        context_ids=tuple(context_ids),
        target_ids=tuple(target_ids),
        site_object_ids=tuple(site_object_ids),
    )


def _segment_mean(
    values: torch.Tensor,
    graph_index: torch.Tensor,
    count: int,
) -> torch.Tensor:
    pooled = values.new_zeros((count, values.shape[-1]))
    pooled.index_add_(0, graph_index, values)
    counts = values.new_zeros((count, 1))
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


class TypedSiteObjectEncoder(nn.Module):
    """Permutation-invariant site pooling with type-specific residual adapters."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        summary_dim = 2 * hidden_dim + 1
        self.shared_encoder = nn.Sequential(
            nn.Linear(summary_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.type_adapters = nn.ModuleDict(
            {
                site_type: nn.Sequential(
                    nn.Linear(summary_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for site_type in SITE_TYPE_NAMES
            }
        )
        for adapter in self.type_adapters.values():
            final = adapter[-1]
            if not isinstance(final, nn.Linear):
                raise AssertionError("Unexpected site-adapter output layer")
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        member_index: torch.Tensor,
        member_ptr: torch.Tensor,
        type_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        site_count = int(type_index.shape[0])
        counts = member_ptr[1:] - member_ptr[:-1]
        if (
            member_ptr.shape != (site_count + 1,)
            or bool((counts <= 0).any())
            or int(member_ptr[-1]) != int(member_index.shape[0])
        ):
            raise ValueError("Invalid packed site-object membership")
        membership_site = torch.repeat_interleave(
            torch.arange(site_count, device=member_index.device),
            counts,
        )
        member_values = node_embeddings[member_index]
        mean_pool = node_embeddings.new_zeros(
            (site_count, node_embeddings.shape[-1])
        )
        mean_pool.index_add_(0, membership_site, member_values)
        mean_pool = mean_pool / counts.to(node_embeddings.dtype).unsqueeze(-1)
        max_pool = node_embeddings.new_full(
            (site_count, node_embeddings.shape[-1]),
            -torch.inf,
        )
        max_pool.scatter_reduce_(
            0,
            membership_site.unsqueeze(-1).expand_as(member_values),
            member_values,
            reduce="amax",
            include_self=True,
        )
        size_feature = torch.log1p(
            counts.to(node_embeddings.dtype)
        ).unsqueeze(-1)
        summary = torch.cat((mean_pool, max_pool, size_feature), dim=-1)
        encoded = self.shared_encoder(summary)
        residual = torch.zeros_like(encoded)
        for index, site_type in enumerate(SITE_TYPE_NAMES):
            selected = type_index.eq(index)
            if bool(selected.any()):
                residual[selected] = self.type_adapters[site_type](
                    summary[selected]
                )
        return encoded + residual, summary


class MayrSiteNModel(nn.Module):
    """All-atom local4/global6 GNN with independent typed site queries."""

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
        self.architecture = {
            "schema_version": MODEL_SCHEMA_VERSION,
            "num_solvents": int(num_solvents),
            "hidden_dim": int(hidden_dim),
            "layers": int(layers),
            "node_embedding_dim": int(node_embedding_dim),
            "edge_embedding_dim": int(edge_embedding_dim),
            "solvent_embedding_dim": int(solvent_embedding_dim),
            "dropout": float(dropout),
            "site_types": list(SITE_TYPE_NAMES),
            "site_probability_normalization": False,
        }
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
        self.site_object_encoder = TypedSiteObjectEncoder(hidden_dim, dropout)
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

    def encode_graph(
        self,
        inputs: SiteNModelInputs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nodes = self.node_encoder(inputs.node_categorical)
        nodes = nodes + self.local_encoder(inputs.node_local)
        edges = self.edge_encoder(inputs.edge_categorical)
        for layer in self.message_layers:
            nodes = layer(nodes, inputs.edge_index, edges)
        graph_pool = _segment_mean(
            nodes,
            inputs.node_graph_index,
            inputs.num_graphs,
        )
        return nodes, graph_pool

    def encode_sites(
        self,
        node_embeddings: torch.Tensor,
        inputs: SiteNModelInputs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.site_object_encoder(
            node_embeddings,
            inputs.site_member_index,
            inputs.site_member_ptr,
            inputs.site_type_index,
        )

    def encode_fused_features(
        self,
        inputs: SiteNModelInputs,
    ) -> SiteNFusedFeatures:
        """Encode graph, typed site, solvent, charge, and global features."""

        nodes, graph_pool = self.encode_graph(inputs)
        site_embeddings, site_summary = self.encode_sites(nodes, inputs)
        site_graph = inputs.site_graph_index
        solvent_continuous = self.solvent_encoder(
            inputs.solvent_continuous
        )[site_graph]
        solvent_embedding = self.solvent_embedding_projection(
            self.solvent_embedding(inputs.solvent_index)
        )[site_graph]
        charge = self.charge_encoder(inputs.molecular_formal_charge)[site_graph]
        global_xtb = self.global_xtb_encoder(inputs.global_xtb)[site_graph]
        fused = torch.cat(
            (
                graph_pool[site_graph],
                site_embeddings,
                solvent_continuous,
                solvent_embedding,
                charge,
                global_xtb,
            ),
            dim=-1,
        )
        return SiteNFusedFeatures(
            node_embeddings=nodes,
            graph_pool=graph_pool,
            site_embeddings=site_embeddings,
            site_summary=site_summary,
            fused=fused,
        )

    def forward(self, inputs: SiteNModelInputs) -> SiteNOutput:
        encoded = self.encode_fused_features(inputs)
        prediction = self.regression_head(encoded.fused).squeeze(-1)
        return SiteNOutput(
            n_prediction_standardized=prediction,
            node_embeddings=encoded.node_embeddings,
            graph_pool=encoded.graph_pool,
            site_embeddings=encoded.site_embeddings,
            site_summary=encoded.site_summary,
        )


def within_context_ranking_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    site_graph_index: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    losses: list[torch.Tensor] = []
    pair_count = 0
    for graph_index in torch.unique(site_graph_index):
        indices = torch.nonzero(
            site_graph_index.eq(graph_index), as_tuple=False
        ).flatten()
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
        target_delta = target[left] - target[right]
        non_ties = target_delta.ne(0)
        if not bool(non_ties.any()):
            continue
        prediction_delta = (
            prediction[left[non_ties]] - prediction[right[non_ties]]
        )
        losses.append(
            F.smooth_l1_loss(
                prediction_delta,
                target_delta[non_ties],
                reduction="none",
            )
        )
        pair_count += int(non_ties.sum())
    if not losses:
        return prediction.sum() * 0.0, 0
    return torch.cat(losses).mean(), pair_count


def site_n_loss(
    output: SiteNOutput,
    batch: SiteNTrainingBatch,
    *,
    ranking_weight: float = 0.1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | int]]:
    regression = F.mse_loss(
        output.n_prediction_standardized,
        batch.n_target_standardized,
    )
    ranking, pair_count = within_context_ranking_loss(
        output.n_prediction_standardized,
        batch.n_target_standardized,
        batch.inputs.site_graph_index,
    )
    total = regression + float(ranking_weight) * ranking
    return total, {
        "regression": regression,
        "ranking": ranking,
        "ranking_pairs": pair_count,
    }


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
