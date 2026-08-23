"""Nie-inspired continuous-filter adapter for M7 heavy-atom hidden states."""

from __future__ import annotations

import math

import torch
from torch import nn


class ShiftedSoftplus(nn.Module):
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(values) - math.log(2.0)


class HeavyAtomContinuousFilter(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        *,
        radial_basis: int,
        cutoff: float,
    ) -> None:
        super().__init__()
        if hidden_dim < 1 or radial_basis < 2:
            raise ValueError("Spatial hidden/radial dimensions must be positive")
        if not math.isfinite(cutoff) or cutoff <= 0:
            raise ValueError("Spatial cutoff must be finite and positive")
        self.cutoff = float(cutoff)
        self.element_embedding = nn.Embedding(100, hidden_dim, padding_idx=0)
        self.radial_mlp = nn.Sequential(
            nn.Linear(radial_basis, hidden_dim),
            ShiftedSoftplus(),
            nn.Linear(hidden_dim, hidden_dim),
            ShiftedSoftplus(),
        )
        self.element_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            ShiftedSoftplus(),
            nn.Linear(hidden_dim, hidden_dim),
            ShiftedSoftplus(),
        )
        centers = torch.arange(radial_basis, dtype=torch.float32) * (
            self.cutoff / radial_basis
        )
        self.register_buffer("centers", centers)
        self.gamma = 10.0

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        atomic_numbers: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("Spatial edge_index must have shape [2,E]")
        source, destination = edge_index
        if source.numel() == 0:
            return torch.zeros_like(node_features)
        distance = torch.linalg.vector_norm(
            positions[source] - positions[destination], dim=-1
        )
        if not torch.isfinite(distance).all() or bool(
            (distance >= self.cutoff + 1e-5).any()
        ):
            raise ValueError("Spatial edges must contain finite distances below cutoff")
        radial = torch.exp(
            -self.gamma * (distance.unsqueeze(-1) - self.centers.unsqueeze(0)).square()
        )
        radial_filter = self.radial_mlp(radial)
        element_filter = self.element_mlp(
            self.element_embedding(atomic_numbers[source])
            * self.element_embedding(atomic_numbers[destination])
        )
        messages = node_features[source] * radial_filter * element_filter
        aggregated = torch.zeros_like(node_features)
        aggregated.index_add_(0, destination, messages)
        return aggregated


class SpatialInteractionBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        *,
        radial_basis: int,
        cutoff: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(hidden_dim, hidden_dim)
        self.convolution = HeavyAtomContinuousFilter(
            hidden_dim, radial_basis=radial_basis, cutoff=cutoff
        )
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            ShiftedSoftplus(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.normalization = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        atomic_numbers: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        messages = self.convolution(
            self.input_projection(node_features),
            edge_index,
            atomic_numbers,
            positions,
        )
        return self.normalization(node_features + self.output_mlp(messages))


class NieInspiredHeavyAtomSpatialAdapter(nn.Module):
    """Two or more 5 Å blocks with a zero-initialized residual gate."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        interactions: int = 2,
        radial_basis: int = 50,
        cutoff: float = 5.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if interactions < 1:
            raise ValueError("Spatial adapter needs at least one interaction")
        self.cutoff = float(cutoff)
        self.blocks = nn.ModuleList(
            SpatialInteractionBlock(
                hidden_dim,
                radial_basis=radial_basis,
                cutoff=cutoff,
                dropout=dropout,
            )
            for _ in range(interactions)
        )
        self.residual_gate = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        atomic_numbers: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        state = node_features
        for block in self.blocks:
            state = block(state, edge_index, atomic_numbers, positions)
        return node_features + self.residual_gate * (state - node_features)
