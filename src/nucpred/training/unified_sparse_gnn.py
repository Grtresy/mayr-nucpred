from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping, Sequence

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors
from torch import nn

from nucpred.training.nie_m7_spatial import NieInspiredHeavyAtomSpatialAdapter


SOLVENT_DESCRIPTOR_COLUMNS = (
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

# This vocabulary covers every element in the current 1,264-row Mayr graph set.
# Index zero is deliberately reserved for elements outside the fitted vocabulary.
MAYR_ELEMENT_VOCABULARY = (
    "<UNK>",
    "B",
    "Br",
    "C",
    "Cl",
    "F",
    "Ge",
    "Hg",
    "I",
    "K",
    "Li",
    "N",
    "Na",
    "O",
    "P",
    "Pb",
    "S",
    "Se",
    "Si",
    "Sn",
    "Te",
    "Zn",
)

NODE_CATEGORICAL_FEATURES = (
    "element",
    "degree",
    "formal_charge",
    "total_hydrogens",
    "aromatic",
    "ring",
    "hybridization",
    "chirality",
    "radical_electrons",
)
EDGE_CATEGORICAL_FEATURES = (
    "bond_type",
    "aromatic",
    "conjugated",
    "ring",
    "stereo",
)

EXPECTED_RDKIT_DESCRIPTOR_DIM = 217
RDKIT_DESCRIPTOR_NAMES = tuple(name for name, _ in Descriptors._descList)

_ELEMENT_TO_INDEX = {
    element: index for index, element in enumerate(MAYR_ELEMENT_VOCABULARY)
}
_HYBRIDIZATION_TO_INDEX = {
    "UNSPECIFIED": 1,
    "S": 2,
    "SP": 3,
    "SP2": 4,
    "SP3": 5,
    "SP3D": 6,
    "SP3D2": 7,
    "OTHER": 8,
}
_CHIRALITY_TO_INDEX = {
    "CHI_UNSPECIFIED": 1,
    "CHI_TETRAHEDRAL_CW": 2,
    "CHI_TETRAHEDRAL_CCW": 3,
    "CHI_OTHER": 4,
    "CHI_TETRAHEDRAL": 5,
    "CHI_ALLENE": 6,
    "CHI_SQUAREPLANAR": 7,
    "CHI_TRIGONALBIPYRAMIDAL": 8,
    "CHI_OCTAHEDRAL": 9,
}
_BOND_TYPE_TO_INDEX = {
    "UNSPECIFIED": 1,
    "SINGLE": 2,
    "DOUBLE": 3,
    "TRIPLE": 4,
    "QUADRUPLE": 5,
    "AROMATIC": 6,
    "DATIVE": 7,
    "IONIC": 8,
    "HYDROGEN": 9,
    "ZERO": 10,
}
_BOND_STEREO_TO_INDEX = {
    "STEREONONE": 1,
    "STEREOANY": 2,
    "STEREOZ": 3,
    "STEREOE": 4,
    "STEREOCIS": 5,
    "STEREOTRANS": 6,
}

NODE_CATEGORY_SIZES = (
    len(MAYR_ELEMENT_VOCABULARY),
    9,  # degree 0..7 plus out-of-range
    8,  # formal charge -3..3 plus out-of-range
    6,  # total H 0..4 plus out-of-range
    2,
    2,
    max(_HYBRIDIZATION_TO_INDEX.values()) + 1,
    max(_CHIRALITY_TO_INDEX.values()) + 1,
    5,  # radical electrons 0..3 plus out-of-range
)
EDGE_CATEGORY_SIZES = (
    max(_BOND_TYPE_TO_INDEX.values()) + 1,
    2,
    2,
    2,
    max(_BOND_STEREO_TO_INDEX.values()) + 1,
)


@dataclass(frozen=True)
class SolventVocabulary:
    tokens: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.tokens or self.tokens[0] != "<UNK>":
            raise ValueError("SolventVocabulary index zero must be '<UNK>'")
        if len(set(self.tokens)) != len(self.tokens):
            raise ValueError("SolventVocabulary tokens must be unique")

    @classmethod
    def from_values(cls, values: Sequence[object]) -> SolventVocabulary:
        tokens = sorted(
            {
                _normalise_category(value)
                for value in values
                if _normalise_category(value) != "<UNK>"
            }
        )
        return cls(("<UNK>", *tokens))

    def encode(self, value: object) -> int:
        token = _normalise_category(value)
        try:
            return self.tokens.index(token)
        except ValueError:
            return 0

    def __len__(self) -> int:
        return len(self.tokens)


@dataclass(frozen=True)
class MoleculeGraph:
    node_categorical: torch.Tensor
    edge_index: torch.Tensor
    edge_categorical: torch.Tensor
    solvent_continuous: torch.Tensor
    solvent_raw: str
    molecular_formal_charge: torch.Tensor
    canonical_smiles: str
    rdkit_descriptors: torch.Tensor | None = None
    reactivity_descriptors: torch.Tensor | None = None
    positions: torch.Tensor | None = None
    atomic_numbers: torch.Tensor | None = None
    spatial_edge_index: torch.Tensor | None = None

    @property
    def num_nodes(self) -> int:
        return int(self.node_categorical.shape[0])


@dataclass(frozen=True)
class PackedGraphBatch:
    node_categorical: torch.Tensor
    edge_index: torch.Tensor
    edge_categorical: torch.Tensor
    node_graph_index: torch.Tensor
    graph_ptr: torch.Tensor
    solvent_continuous: torch.Tensor
    solvent_index: torch.Tensor
    molecular_formal_charge: torch.Tensor
    canonical_smiles: tuple[str, ...]
    rdkit_descriptors: torch.Tensor | None = None
    reactivity_descriptors: torch.Tensor | None = None
    positions: torch.Tensor | None = None
    atomic_numbers: torch.Tensor | None = None
    spatial_edge_index: torch.Tensor | None = None

    @property
    def num_graphs(self) -> int:
        return len(self.canonical_smiles)

    @property
    def num_nodes(self) -> int:
        return int(self.node_categorical.shape[0])

    def to(self, device: str | torch.device) -> PackedGraphBatch:
        return PackedGraphBatch(
            node_categorical=self.node_categorical.to(device),
            edge_index=self.edge_index.to(device),
            edge_categorical=self.edge_categorical.to(device),
            node_graph_index=self.node_graph_index.to(device),
            graph_ptr=self.graph_ptr.to(device),
            solvent_continuous=self.solvent_continuous.to(device),
            solvent_index=self.solvent_index.to(device),
            molecular_formal_charge=self.molecular_formal_charge.to(device),
            canonical_smiles=self.canonical_smiles,
            rdkit_descriptors=None
            if self.rdkit_descriptors is None
            else self.rdkit_descriptors.to(device),
            reactivity_descriptors=None
            if self.reactivity_descriptors is None
            else self.reactivity_descriptors.to(device),
            positions=None if self.positions is None else self.positions.to(device),
            atomic_numbers=None
            if self.atomic_numbers is None
            else self.atomic_numbers.to(device),
            spatial_edge_index=None
            if self.spatial_edge_index is None
            else self.spatial_edge_index.to(device),
        )


def featurize_record(
    record: Mapping[str, object],
    *,
    include_rdkit_descriptors: bool = False,
    reactivity_feature_columns: Sequence[str] = (),
) -> MoleculeGraph:
    """Build inference inputs while intentionally ignoring all target/site fields."""

    smiles_value = record.get(
        "curated_canonical_smiles", record.get("canonical_smiles", "")
    )
    smiles = str(smiles_value).strip()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"Invalid or empty canonical SMILES: {smiles!r}")
    canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)

    node_rows = [_atom_categories(atom) for atom in mol.GetAtoms()]
    edge_pairs: list[tuple[int, int]] = []
    edge_rows: list[list[int]] = []
    for bond in mol.GetBonds():
        begin = int(bond.GetBeginAtomIdx())
        end = int(bond.GetEndAtomIdx())
        features = _bond_categories(bond)
        edge_pairs.extend(((begin, end), (end, begin)))
        edge_rows.extend((features, features))

    edge_index = (
        torch.tensor(edge_pairs, dtype=torch.long).transpose(0, 1).contiguous()
        if edge_pairs
        else torch.empty((2, 0), dtype=torch.long)
    )
    edge_categorical = (
        torch.tensor(edge_rows, dtype=torch.long)
        if edge_rows
        else torch.empty((0, len(EDGE_CATEGORICAL_FEATURES)), dtype=torch.long)
    )
    solvent_values = torch.tensor(
        [
            _finite_float(record.get(column, 0.0))
            for column in SOLVENT_DESCRIPTOR_COLUMNS
        ],
        dtype=torch.float32,
    )
    formal_charge = torch.tensor(
        [_finite_float(record.get("formal_charge", Chem.GetFormalCharge(mol)))],
        dtype=torch.float32,
    )
    descriptors = compute_rdkit_descriptors(mol) if include_rdkit_descriptors else None
    reactivity_descriptors = (
        torch.tensor(
            [
                _finite_float(record.get(column, 0.0))
                for column in reactivity_feature_columns
            ],
            dtype=torch.float32,
        )
        if reactivity_feature_columns
        else None
    )
    return MoleculeGraph(
        node_categorical=torch.tensor(node_rows, dtype=torch.long),
        edge_index=edge_index,
        edge_categorical=edge_categorical,
        solvent_continuous=solvent_values,
        solvent_raw=_normalise_category(record.get("solvent_raw", "<UNK>")),
        molecular_formal_charge=formal_charge,
        canonical_smiles=canonical_smiles,
        rdkit_descriptors=descriptors,
        reactivity_descriptors=reactivity_descriptors,
    )


def compute_rdkit_descriptors(mol: Chem.Mol) -> torch.Tensor:
    if len(RDKIT_DESCRIPTOR_NAMES) != EXPECTED_RDKIT_DESCRIPTOR_DIM:
        raise RuntimeError(
            "The active RDKit descriptor surface changed: "
            f"expected {EXPECTED_RDKIT_DESCRIPTOR_DIM}, got {len(RDKIT_DESCRIPTOR_NAMES)}"
        )
    functions = dict(Descriptors._descList)
    values = []
    for name in RDKIT_DESCRIPTOR_NAMES:
        try:
            value = float(functions[name](mol))
        except Exception:
            value = 0.0
        values.append(value if math.isfinite(value) else 0.0)
    return torch.tensor(values, dtype=torch.float32)


def collate_graphs(
    graphs: Sequence[MoleculeGraph],
    *,
    solvent_vocabulary: SolventVocabulary,
) -> PackedGraphBatch:
    if not graphs:
        raise ValueError("At least one graph is required for collation")
    descriptor_presence = {graph.rdkit_descriptors is not None for graph in graphs}
    if len(descriptor_presence) != 1:
        raise ValueError("A batch cannot mix graphs with and without RDKit descriptors")
    reactivity_presence = {graph.reactivity_descriptors is not None for graph in graphs}
    if len(reactivity_presence) != 1:
        raise ValueError(
            "A batch cannot mix graphs with and without reactivity descriptors"
        )
    atomic_number_presence = {graph.atomic_numbers is not None for graph in graphs}
    if len(atomic_number_presence) != 1:
        raise ValueError("A batch cannot mix graphs with and without atomic numbers")
    spatial_presence = {
        graph.positions is not None or graph.spatial_edge_index is not None
        for graph in graphs
    }
    if len(spatial_presence) != 1:
        raise ValueError("A batch cannot mix graphs with and without spatial geometry")

    node_parts: list[torch.Tensor] = []
    edge_index_parts: list[torch.Tensor] = []
    edge_parts: list[torch.Tensor] = []
    graph_index_parts: list[torch.Tensor] = []
    graph_ptr = [0]
    position_parts: list[torch.Tensor] = []
    atomic_number_parts: list[torch.Tensor] = []
    spatial_edge_parts: list[torch.Tensor] = []
    node_offset = 0
    for graph_index, graph in enumerate(graphs):
        _validate_graph(graph)
        node_parts.append(graph.node_categorical)
        edge_index_parts.append(graph.edge_index + node_offset)
        edge_parts.append(graph.edge_categorical)
        graph_index_parts.append(
            torch.full((graph.num_nodes,), graph_index, dtype=torch.long)
        )
        if True in atomic_number_presence:
            assert graph.atomic_numbers is not None
            atomic_number_parts.append(graph.atomic_numbers)
        if True in spatial_presence:
            if graph.positions is None or graph.spatial_edge_index is None:
                raise ValueError("Spatial positions and edges must be present together")
            position_parts.append(graph.positions)
            spatial_edge_parts.append(graph.spatial_edge_index + node_offset)
        node_offset += graph.num_nodes
        graph_ptr.append(node_offset)

    descriptors = (
        torch.stack(
            [
                graph.rdkit_descriptors
                for graph in graphs
                if graph.rdkit_descriptors is not None
            ]
        )
        if True in descriptor_presence
        else None
    )
    reactivity_descriptors = (
        torch.stack(
            [
                graph.reactivity_descriptors
                for graph in graphs
                if graph.reactivity_descriptors is not None
            ]
        )
        if True in reactivity_presence
        else None
    )
    return PackedGraphBatch(
        node_categorical=torch.cat(node_parts, dim=0),
        edge_index=torch.cat(edge_index_parts, dim=1),
        edge_categorical=torch.cat(edge_parts, dim=0),
        node_graph_index=torch.cat(graph_index_parts, dim=0),
        graph_ptr=torch.tensor(graph_ptr, dtype=torch.long),
        solvent_continuous=torch.stack([graph.solvent_continuous for graph in graphs]),
        solvent_index=torch.tensor(
            [solvent_vocabulary.encode(graph.solvent_raw) for graph in graphs],
            dtype=torch.long,
        ),
        molecular_formal_charge=torch.stack(
            [graph.molecular_formal_charge for graph in graphs]
        ),
        canonical_smiles=tuple(graph.canonical_smiles for graph in graphs),
        rdkit_descriptors=descriptors,
        reactivity_descriptors=reactivity_descriptors,
        positions=torch.cat(position_parts, dim=0)
        if True in spatial_presence
        else None,
        atomic_numbers=torch.cat(atomic_number_parts, dim=0)
        if True in atomic_number_presence
        else None,
        spatial_edge_index=torch.cat(spatial_edge_parts, dim=1)
        if True in spatial_presence
        else None,
    )


class CategoricalFeatureEncoder(nn.Module):
    def __init__(
        self,
        category_sizes: Sequence[int],
        *,
        embedding_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            nn.Embedding(int(size), embedding_dim) for size in category_sizes
        )
        self.projection = nn.Sequential(
            nn.Linear(len(category_sizes) * embedding_dim, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != len(self.embeddings):
            raise ValueError(
                f"Expected categorical tensor [N,{len(self.embeddings)}], "
                f"got {tuple(features.shape)}"
            )
        embedded = [
            embedding(features[:, index])
            for index, embedding in enumerate(self.embeddings)
        ]
        return self.projection(torch.cat(embedded, dim=-1))


class SparseMessageLayer(nn.Module):
    def __init__(self, hidden_dim: int, *, dropout: float) -> None:
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.normalization = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        edge_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        aggregated = torch.zeros_like(node_embeddings)
        degree = node_embeddings.new_zeros((node_embeddings.shape[0], 1))
        if edge_index.shape[1] > 0:
            source, destination = edge_index
            messages = self.message(
                torch.cat([node_embeddings[source], edge_embeddings], dim=-1)
            )
            aggregated.index_add_(0, destination, messages)
            degree.index_add_(
                0,
                destination,
                torch.ones(
                    (destination.shape[0], 1),
                    dtype=node_embeddings.dtype,
                    device=node_embeddings.device,
                ),
            )
            aggregated = aggregated / degree.clamp_min(1.0)
        update = self.update(torch.cat([node_embeddings, aggregated], dim=-1))
        return self.normalization(node_embeddings + self.dropout(update))


class PackedSparseGraphEncoder(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        layers: int,
        node_embedding_dim: int = 16,
        edge_embedding_dim: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be positive")
        self.hidden_dim = hidden_dim
        self.node_encoder = CategoricalFeatureEncoder(
            NODE_CATEGORY_SIZES,
            embedding_dim=node_embedding_dim,
            output_dim=hidden_dim,
        )
        self.edge_encoder = CategoricalFeatureEncoder(
            EDGE_CATEGORY_SIZES,
            embedding_dim=edge_embedding_dim,
            output_dim=hidden_dim,
        )
        self.message_layers = nn.ModuleList(
            SparseMessageLayer(hidden_dim, dropout=dropout) for _ in range(layers)
        )

    def forward(self, batch: PackedGraphBatch) -> tuple[torch.Tensor, torch.Tensor]:
        node_embeddings, graph_pool, _ = self.forward_with_layer_pools(batch)
        return node_embeddings, graph_pool

    def forward_with_layer_pools(
        self, batch: PackedGraphBatch
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        """Encode a batch and retain the graph state after every message layer.

        The ordinary forward path intentionally keeps its stable two-tensor API.
        The additional states support controlled graph-conditioned global-feature
        fusion without changing encoder parameters or checkpoint keys.
        """
        node_embeddings = self.node_encoder(batch.node_categorical)
        edge_embeddings = self.edge_encoder(batch.edge_categorical)
        layer_pools: list[torch.Tensor] = []
        for layer in self.message_layers:
            node_embeddings = layer(node_embeddings, batch.edge_index, edge_embeddings)
            layer_pools.append(
                _segment_mean(node_embeddings, batch.node_graph_index, batch.num_graphs)
            )
        graph_pool = _segment_mean(
            node_embeddings, batch.node_graph_index, batch.num_graphs
        )
        return node_embeddings, graph_pool, tuple(layer_pools)


@dataclass(frozen=True)
class SiteContext:
    logits: torch.Tensor
    distribution: torch.Tensor
    pooled: torch.Tensor


class UnifiedSparseBackbone(nn.Module):
    """Stable encoder/site-head surface shared by pretraining and fine-tuning."""

    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        layers: int = 4,
        node_embedding_dim: int = 16,
        edge_embedding_dim: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = PackedSparseGraphEncoder(
            hidden_dim=hidden_dim,
            layers=layers,
            node_embedding_dim=node_embedding_dim,
            edge_embedding_dim=edge_embedding_dim,
            dropout=dropout,
        )
        self.site_head = nn.Linear(hidden_dim, 1)

    def encode(self, batch: PackedGraphBatch) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(batch)

    def site_context(
        self, batch: PackedGraphBatch, node_embeddings: torch.Tensor
    ) -> SiteContext:
        logits = self.site_head(node_embeddings).squeeze(-1)
        distribution = _segment_softmax(logits, batch.graph_ptr)
        pooled = node_embeddings.new_zeros(
            (batch.num_graphs, node_embeddings.shape[-1])
        )
        pooled.index_add_(
            0,
            batch.node_graph_index,
            node_embeddings * distribution.unsqueeze(-1),
        )
        return SiteContext(logits=logits, distribution=distribution, pooled=pooled)


@dataclass(frozen=True)
class UnifiedSparseOutput:
    regression: torch.Tensor
    n_prediction: torch.Tensor
    sn_prediction: torch.Tensor | None
    node_embeddings: torch.Tensor
    graph_pool: torch.Tensor
    site_logits: torch.Tensor
    site_distribution: torch.Tensor
    site_pool: torch.Tensor


class GraphConditionedReactivityEncoder(nn.Module):
    """Nie-style residual global state updated by per-layer graph summaries."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.graph_update = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.state_update = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.merge_update = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.normalization = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        descriptors: torch.Tensor,
        layer_graph_pools: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        state = self.input_encoder(descriptors)
        for graph_pool in layer_graph_pools:
            update = self.merge_update(
                self.graph_update(graph_pool) + self.state_update(state)
            )
            state = self.normalization(state + update)
        return state


class PhysicsGroupedReactivityEncoder(nn.Module):
    """Compact Nie-CDFT encoder that respects descriptor provenance.

    The released vector contains five global frontier/solvation quantities,
    four exact-site local quantities, and one affine duplicate (``Nuc I`` is
    ``E_HOMO(N)`` shifted by the constant TCE reference energy).  Encoding the
    global and local blocks separately keeps the useful published derived
    local products while avoiding a dense 10-to-hidden expansion dominated by
    exact collinearity.
    """

    GLOBAL_INDICES = (0, 1, 2, 4, 9)
    LOCAL_INDICES = (5, 6, 7, 8)

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        group_dim = max(8, hidden_dim // 2)
        self.global_encoder = nn.Sequential(
            nn.Linear(len(self.GLOBAL_INDICES), group_dim),
            nn.SiLU(),
            nn.LayerNorm(group_dim),
        )
        self.local_encoder = nn.Sequential(
            nn.Linear(len(self.LOCAL_INDICES), group_dim),
            nn.SiLU(),
            nn.LayerNorm(group_dim),
        )
        self.merge = nn.Sequential(
            nn.Linear(2 * group_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, descriptors: torch.Tensor) -> torch.Tensor:
        if descriptors.ndim != 2 or descriptors.shape[1] != 10:
            raise ValueError(
                "physics_additive fusion requires the released 10D Nie CDFT vector"
            )
        global_values = descriptors[:, self.GLOBAL_INDICES]
        local_values = descriptors[:, self.LOCAL_INDICES]
        return self.merge(
            torch.cat(
                [
                    self.global_encoder(global_values),
                    self.local_encoder(local_values),
                ],
                dim=-1,
            )
        )


class UnifiedSparseSiteGNN(UnifiedSparseBackbone):
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
        predict_sn: bool = False,
        use_rdkit_descriptors: bool = False,
        reactivity_descriptor_dim: int = 0,
        reactivity_fusion_mode: str = "late",
        use_spatial_adapter: bool = False,
        spatial_interactions: int = 2,
        spatial_cutoff: float = 5.0,
        spatial_radial_basis: int = 50,
    ) -> None:
        super().__init__(
            hidden_dim=hidden_dim,
            layers=layers,
            node_embedding_dim=node_embedding_dim,
            edge_embedding_dim=edge_embedding_dim,
            dropout=dropout,
        )
        if num_solvents < 1:
            raise ValueError("num_solvents must include at least the unknown solvent")
        self.predict_sn = predict_sn
        self.use_rdkit_descriptors = use_rdkit_descriptors
        if reactivity_descriptor_dim < 0:
            raise ValueError("reactivity_descriptor_dim cannot be negative")
        self.reactivity_descriptor_dim = int(reactivity_descriptor_dim)
        if reactivity_fusion_mode not in {
            "late",
            "graph_conditioned",
            "physics_additive",
            "physics_site_additive",
        }:
            raise ValueError(
                "reactivity_fusion_mode must be 'late', 'graph_conditioned', "
                "'physics_additive', or 'physics_site_additive'"
            )
        self.reactivity_fusion_mode = str(reactivity_fusion_mode)
        self.solvent_continuous_encoder = _continuous_encoder(
            len(SOLVENT_DESCRIPTOR_COLUMNS), hidden_dim
        )
        self.solvent_embedding = nn.Embedding(num_solvents, solvent_embedding_dim)
        self.solvent_embedding_projection = _continuous_encoder(
            solvent_embedding_dim, hidden_dim
        )
        self.formal_charge_encoder = _continuous_encoder(1, hidden_dim)
        self.descriptor_encoder = (
            _continuous_encoder(EXPECTED_RDKIT_DESCRIPTOR_DIM, hidden_dim)
            if use_rdkit_descriptors
            else None
        )
        if not self.reactivity_descriptor_dim:
            self.reactivity_descriptor_encoder: nn.Module | None = None
        elif self.reactivity_fusion_mode == "late":
            self.reactivity_descriptor_encoder = _continuous_encoder(
                self.reactivity_descriptor_dim, hidden_dim
            )
        elif self.reactivity_fusion_mode == "graph_conditioned":
            self.reactivity_descriptor_encoder = GraphConditionedReactivityEncoder(
                self.reactivity_descriptor_dim, hidden_dim
            )
        else:
            if self.reactivity_descriptor_dim != 10:
                raise ValueError(
                    "physics additive fusion is defined only for Nie's 10D CDFT vector"
                )
            self.reactivity_descriptor_encoder = PhysicsGroupedReactivityEncoder(
                hidden_dim
            )
        output_dim = 2 if predict_sn else 1
        self.reactivity_site_projection = (
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
            )
            if self.reactivity_descriptor_dim
            and self.reactivity_fusion_mode == "physics_site_additive"
            else None
        )
        self.reactivity_head = (
            nn.Sequential(
                nn.Linear(hidden_dim, max(8, hidden_dim // 2)),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(max(8, hidden_dim // 2), output_dim),
            )
            if self.reactivity_descriptor_dim
            and self.reactivity_fusion_mode
            in {"physics_additive", "physics_site_additive"}
            else None
        )
        if self.reactivity_head is not None:
            final = self.reactivity_head[-1]
            if not isinstance(final, nn.Linear):
                raise AssertionError("Unexpected physics-additive output layer")
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
        fusion_blocks = (
            5
            + int(use_rdkit_descriptors)
            + int(
                self.reactivity_descriptor_dim > 0
                and self.reactivity_fusion_mode
                not in {"physics_additive", "physics_site_additive"}
            )
        )
        self.regression_head = nn.Sequential(
            nn.Linear(fusion_blocks * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.use_spatial_adapter = bool(use_spatial_adapter)
        self.spatial_adapter = (
            NieInspiredHeavyAtomSpatialAdapter(
                hidden_dim,
                interactions=spatial_interactions,
                radial_basis=spatial_radial_basis,
                cutoff=spatial_cutoff,
                dropout=dropout,
            )
            if self.use_spatial_adapter
            else None
        )

    def forward(self, batch: PackedGraphBatch) -> UnifiedSparseOutput:
        layer_graph_pools: tuple[torch.Tensor, ...] = ()
        if (
            self.reactivity_descriptor_dim
            and self.reactivity_fusion_mode == "graph_conditioned"
        ):
            node_embeddings, graph_pool, layer_graph_pools = (
                self.encoder.forward_with_layer_pools(batch)
            )
        else:
            node_embeddings, graph_pool = self.encode(batch)
        if self.use_spatial_adapter:
            if (
                self.spatial_adapter is None
                or batch.positions is None
                or batch.atomic_numbers is None
                or batch.spatial_edge_index is None
            ):
                raise ValueError(
                    "Spatial adapter requires positions, atomic numbers, and 5 Å edges"
                )
            node_embeddings = self.spatial_adapter(
                node_embeddings,
                batch.spatial_edge_index,
                batch.atomic_numbers,
                batch.positions,
            )
            graph_pool = _segment_mean(
                node_embeddings, batch.node_graph_index, batch.num_graphs
            )
        site = self.site_context(batch, node_embeddings)
        fusion = [
            graph_pool,
            site.pooled,
            self.solvent_continuous_encoder(batch.solvent_continuous),
            self.solvent_embedding_projection(
                self.solvent_embedding(batch.solvent_index)
            ),
            self.formal_charge_encoder(batch.molecular_formal_charge),
        ]
        if self.use_rdkit_descriptors:
            if batch.rdkit_descriptors is None or self.descriptor_encoder is None:
                raise ValueError(
                    "RDKit descriptor fusion was enabled but the batch has no descriptors"
                )
            fusion.append(self.descriptor_encoder(batch.rdkit_descriptors))
        elif batch.rdkit_descriptors is not None:
            raise ValueError(
                "The batch contains RDKit descriptors but descriptor fusion is disabled"
            )
        if self.reactivity_descriptor_dim:
            if (
                batch.reactivity_descriptors is None
                or self.reactivity_descriptor_encoder is None
                or batch.reactivity_descriptors.shape[1]
                != self.reactivity_descriptor_dim
            ):
                raise ValueError(
                    "Reactivity descriptor fusion was enabled but the batch has an "
                    "invalid descriptor tensor"
                )
            if self.reactivity_fusion_mode == "graph_conditioned":
                if not isinstance(
                    self.reactivity_descriptor_encoder,
                    GraphConditionedReactivityEncoder,
                ):
                    raise AssertionError(
                        "Graph-conditioned reactivity encoder is unavailable"
                    )
                fusion.append(
                    self.reactivity_descriptor_encoder(
                        batch.reactivity_descriptors, layer_graph_pools
                    )
                )
            elif self.reactivity_fusion_mode == "late":
                fusion.append(
                    self.reactivity_descriptor_encoder(batch.reactivity_descriptors)
                )
            elif self.reactivity_fusion_mode not in {
                "physics_additive",
                "physics_site_additive",
            }:
                raise AssertionError("Unknown reactivity fusion mode")
        elif batch.reactivity_descriptors is not None:
            raise ValueError(
                "The batch contains reactivity descriptors but fusion is disabled"
            )
        regression = self.regression_head(torch.cat(fusion, dim=-1))
        if self.reactivity_fusion_mode in {
            "physics_additive",
            "physics_site_additive",
        }:
            if (
                batch.reactivity_descriptors is None
                or self.reactivity_descriptor_encoder is None
                or self.reactivity_head is None
            ):
                raise ValueError("physics_additive fusion requires a valid CDFT branch")
            electronic = self.reactivity_descriptor_encoder(
                batch.reactivity_descriptors
            )
            if self.reactivity_fusion_mode == "physics_site_additive":
                if self.reactivity_site_projection is None:
                    raise AssertionError("Site-conditioned CDFT projection is missing")
                electronic = electronic * (
                    1.0 + self.reactivity_site_projection(site.pooled)
                )
            regression = regression + self.reactivity_head(electronic)
        return UnifiedSparseOutput(
            regression=regression,
            n_prediction=regression[:, 0],
            sn_prediction=regression[:, 1] if self.predict_sn else None,
            node_embeddings=node_embeddings,
            graph_pool=graph_pool,
            site_logits=site.logits,
            site_distribution=site.distribution,
            site_pool=site.pooled,
        )


class MayrSiteTargetKind(StrEnum):
    EXACT_ATOM = "exact_atom"
    WEIGHTED_DISTRIBUTION = "weighted_distribution"
    CANDIDATE_SET = "candidate_set"
    HYDRIDE_PROXY = "hydride_proxy"
    NONE = "none"


@dataclass(frozen=True)
class MayrSiteTarget:
    kind: MayrSiteTargetKind
    atom_indices: tuple[int, ...] = ()
    weights: tuple[float, ...] = ()

    def validate(self, num_atoms: int) -> None:
        if self.kind == MayrSiteTargetKind.NONE:
            if self.atom_indices or self.weights:
                raise ValueError("Masked site targets cannot contain atoms or weights")
            return
        if not self.atom_indices:
            raise ValueError(f"{self.kind} requires at least one atom index")
        if len(set(self.atom_indices)) != len(self.atom_indices):
            raise ValueError("Site target atom indices must be unique")
        if any(index < 0 or index >= num_atoms for index in self.atom_indices):
            raise ValueError("Site target atom index is outside the molecule")
        if self.kind == MayrSiteTargetKind.EXACT_ATOM:
            if len(self.atom_indices) != 1 or self.weights:
                raise ValueError(
                    "exact_atom requires one index and no explicit weights"
                )
        elif self.kind == MayrSiteTargetKind.WEIGHTED_DISTRIBUTION:
            if len(self.weights) != len(self.atom_indices):
                raise ValueError("weighted_distribution requires one weight per atom")
            if any(not math.isfinite(weight) or weight < 0 for weight in self.weights):
                raise ValueError(
                    "Site distribution weights must be finite and non-negative"
                )
            if not math.isclose(sum(self.weights), 1.0, abs_tol=1e-6):
                raise ValueError("Site distribution weights must sum to one")
        elif self.weights:
            raise ValueError(f"{self.kind} uses set likelihood and cannot take weights")


@dataclass(frozen=True)
class SiteLossResult:
    loss: torch.Tensor
    active_graphs: int
    counts_by_kind: Mapping[str, int]
    mean_loss_by_kind: Mapping[str, torch.Tensor]
    per_graph_loss: torch.Tensor


def mayr_site_loss(
    site_logits: torch.Tensor,
    graph_ptr: torch.Tensor,
    targets: Sequence[MayrSiteTarget],
    *,
    distribution_objective: str = "ce",
) -> SiteLossResult:
    if distribution_objective not in {"ce", "kl"}:
        raise ValueError("distribution_objective must be 'ce' or 'kl'")
    _validate_site_batch(site_logits, graph_ptr, targets)
    losses: list[torch.Tensor] = []
    active_losses: list[torch.Tensor] = []
    by_kind: dict[str, list[torch.Tensor]] = {}
    for graph_index, target in enumerate(targets):
        start = int(graph_ptr[graph_index])
        end = int(graph_ptr[graph_index + 1])
        target.validate(end - start)
        if target.kind == MayrSiteTargetKind.NONE:
            losses.append(site_logits.new_tensor(float("nan")))
            continue
        log_probabilities = torch.log_softmax(site_logits[start:end], dim=0)
        indices = torch.tensor(
            target.atom_indices, dtype=torch.long, device=site_logits.device
        )
        if target.kind == MayrSiteTargetKind.EXACT_ATOM:
            loss = -log_probabilities[indices[0]]
        elif target.kind == MayrSiteTargetKind.WEIGHTED_DISTRIBUTION:
            weights = site_logits.new_tensor(target.weights)
            loss = -(weights * log_probabilities[indices]).sum()
            if distribution_objective == "kl":
                positive = weights > 0
                loss = loss + (weights[positive] * weights[positive].log()).sum()
        else:
            # Candidate sets and hydride proxies are set-valued observations.  This
            # objective never invents a uniform atom-level ground truth.
            loss = -torch.logsumexp(log_probabilities[indices], dim=0)
        losses.append(loss)
        active_losses.append(loss)
        by_kind.setdefault(str(target.kind), []).append(loss)
    aggregate = (
        torch.stack(active_losses).mean() if active_losses else site_logits.sum() * 0.0
    )
    per_graph = torch.stack(losses) if losses else site_logits.new_empty((0,))
    return SiteLossResult(
        loss=aggregate,
        active_graphs=len(active_losses),
        counts_by_kind={kind: len(values) for kind, values in by_kind.items()},
        mean_loss_by_kind={
            kind: torch.stack(values).mean() for kind, values in by_kind.items()
        },
        per_graph_loss=per_graph,
    )


def compute_site_metrics(
    site_distribution: torch.Tensor,
    graph_ptr: torch.Tensor,
    targets: Sequence[MayrSiteTarget],
    *,
    calibration_bins: int = 10,
) -> dict[str, float | int]:
    if calibration_bins < 1:
        raise ValueError("calibration_bins must be positive")
    _validate_site_batch(site_distribution, graph_ptr, targets)
    distribution = site_distribution.detach().float().cpu()
    ptr = graph_ptr.detach().cpu()
    ce_values: list[float] = []
    kl_values: list[float] = []
    brier_values: list[float] = []
    entropy_values: list[float] = []
    set_nll_values: list[float] = []
    top1_hits: list[float] = []
    top3_hits: list[float] = []
    confidences: list[float] = []
    hydride_top1_hits: list[float] = []
    hydride_top3_hits: list[float] = []
    active_rows = 0
    distribution_rows = 0
    for graph_index, target in enumerate(targets):
        start = int(ptr[graph_index])
        end = int(ptr[graph_index + 1])
        target.validate(end - start)
        if target.kind == MayrSiteTargetKind.NONE:
            continue
        active_rows += 1
        probabilities = distribution[start:end]
        probabilities = probabilities / probabilities.sum().clamp_min(1e-12)
        support = set(target.atom_indices)
        ranking = torch.argsort(probabilities, descending=True).tolist()
        top1_hit = float(bool(ranking) and ranking[0] in support)
        top3_hit = float(bool(set(ranking[:3]) & support))
        top1_hits.append(top1_hit)
        top3_hits.append(top3_hit)
        confidences.append(float(probabilities.max()))
        entropy_values.append(
            float(-(probabilities * probabilities.clamp_min(1e-12).log()).sum())
        )
        if target.kind == MayrSiteTargetKind.HYDRIDE_PROXY:
            hydride_top1_hits.append(top1_hit)
            hydride_top3_hits.append(top3_hit)
        if target.kind in {
            MayrSiteTargetKind.EXACT_ATOM,
            MayrSiteTargetKind.WEIGHTED_DISTRIBUTION,
        }:
            target_distribution = torch.zeros_like(probabilities)
            if target.kind == MayrSiteTargetKind.EXACT_ATOM:
                target_distribution[target.atom_indices[0]] = 1.0
            else:
                target_distribution[list(target.atom_indices)] = torch.tensor(
                    target.weights, dtype=probabilities.dtype
                )
            log_probabilities = probabilities.clamp_min(1e-12).log()
            ce = float(-(target_distribution * log_probabilities).sum())
            target_entropy = float(
                -(
                    target_distribution * target_distribution.clamp_min(1e-12).log()
                ).sum()
            )
            ce_values.append(ce)
            kl_values.append(ce - target_entropy)
            brier_values.append(
                float(((probabilities - target_distribution) ** 2).sum())
            )
            distribution_rows += 1
        else:
            set_probability = float(probabilities[list(target.atom_indices)].sum())
            set_nll_values.append(-math.log(max(set_probability, 1e-12)))
    return {
        "site_eval_rows": active_rows,
        "site_distribution_rows": distribution_rows,
        "site_ce": _mean_or_nan(ce_values),
        "site_kl": _mean_or_nan(kl_values),
        "site_top1_accuracy": _mean_or_nan(top1_hits),
        "site_top3_set_hit_rate": _mean_or_nan(top3_hits),
        "site_brier": _mean_or_nan(brier_values),
        "site_calibration_error": _expected_calibration_error(
            confidences, top1_hits, bins=calibration_bins
        ),
        "site_prediction_entropy": _mean_or_nan(entropy_values),
        "site_set_nll": _mean_or_nan(set_nll_values),
        "hydride_proxy_rows": len(hydride_top1_hits),
        "hydride_proxy_top1_set_hit_rate": _mean_or_nan(hydride_top1_hits),
        "hydride_proxy_top3_set_hit_rate": _mean_or_nan(hydride_top3_hits),
    }


@dataclass(frozen=True)
class TransferReport:
    loaded_keys: tuple[str, ...]
    missing_keys: tuple[str, ...]
    shape_mismatches: tuple[str, ...]


def load_encoder_and_site_head(
    model: UnifiedSparseBackbone,
    pretrained_state: Mapping[str, torch.Tensor],
    *,
    strict: bool = True,
) -> TransferReport:
    target_state = model.state_dict()
    required_keys = tuple(
        key
        for key in target_state
        if key.startswith("encoder.") or key.startswith("site_head.")
    )
    loaded: list[str] = []
    missing: list[str] = []
    mismatched: list[str] = []
    for key in required_keys:
        value = pretrained_state.get(key)
        if value is None:
            missing.append(key)
        elif tuple(value.shape) != tuple(target_state[key].shape):
            mismatched.append(key)
        else:
            target_state[key] = value.detach().to(
                device=target_state[key].device, dtype=target_state[key].dtype
            )
            loaded.append(key)
    if strict and (missing or mismatched):
        raise ValueError(
            "Incomplete encoder/site-head transfer: "
            f"missing={missing[:5]}, shape_mismatches={mismatched[:5]}"
        )
    if not loaded:
        raise ValueError("No encoder/site-head parameters were transferred")
    model.load_state_dict(target_state)
    return TransferReport(
        loaded_keys=tuple(loaded),
        missing_keys=tuple(missing),
        shape_mismatches=tuple(mismatched),
    )


def parse_site_distribution(value: object) -> tuple[float, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(float(item) for item in value)
    text = str(value or "").strip()
    if not text:
        return ()
    parsed = json.loads(text)
    if not isinstance(parsed, list):
        raise ValueError("site distribution must be a JSON list")
    return tuple(float(item) for item in parsed)


def _atom_categories(atom: Chem.Atom) -> list[int]:
    return [
        _ELEMENT_TO_INDEX.get(atom.GetSymbol(), 0),
        _bounded_category(atom.GetTotalDegree(), minimum=0, maximum=7),
        _bounded_category(atom.GetFormalCharge(), minimum=-3, maximum=3),
        _bounded_category(atom.GetTotalNumHs(), minimum=0, maximum=4),
        int(atom.GetIsAromatic()),
        int(atom.IsInRing()),
        _HYBRIDIZATION_TO_INDEX.get(str(atom.GetHybridization()), 0),
        _CHIRALITY_TO_INDEX.get(str(atom.GetChiralTag()), 0),
        _bounded_category(atom.GetNumRadicalElectrons(), minimum=0, maximum=3),
    ]


def _bond_categories(bond: Chem.Bond) -> list[int]:
    return [
        _BOND_TYPE_TO_INDEX.get(str(bond.GetBondType()), 0),
        int(bond.GetIsAromatic()),
        int(bond.GetIsConjugated()),
        int(bond.IsInRing()),
        _BOND_STEREO_TO_INDEX.get(str(bond.GetStereo()), 0),
    ]


def _bounded_category(value: int, *, minimum: int, maximum: int) -> int:
    numeric = int(value)
    if minimum <= numeric <= maximum:
        return numeric - minimum
    return maximum - minimum + 1


def _validate_graph(graph: MoleculeGraph) -> None:
    if graph.node_categorical.shape != (
        graph.num_nodes,
        len(NODE_CATEGORICAL_FEATURES),
    ):
        raise ValueError("Invalid node categorical feature shape")
    if graph.edge_index.ndim != 2 or graph.edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2,E]")
    if graph.edge_categorical.shape != (
        graph.edge_index.shape[1],
        len(EDGE_CATEGORICAL_FEATURES),
    ):
        raise ValueError("Invalid edge categorical feature shape")
    if graph.solvent_continuous.shape != (len(SOLVENT_DESCRIPTOR_COLUMNS),):
        raise ValueError("Invalid solvent descriptor shape")
    if graph.molecular_formal_charge.shape != (1,):
        raise ValueError("Molecular formal charge must have shape [1]")
    if graph.rdkit_descriptors is not None and graph.rdkit_descriptors.shape != (
        EXPECTED_RDKIT_DESCRIPTOR_DIM,
    ):
        raise ValueError("RDKit descriptor vector must have 217 values")
    if (
        graph.reactivity_descriptors is not None
        and graph.reactivity_descriptors.ndim != 1
    ):
        raise ValueError("Reactivity descriptor vector must be one-dimensional")
    if graph.atomic_numbers is not None and graph.atomic_numbers.shape != (
        graph.num_nodes,
    ):
        raise ValueError("Atomic numbers must have shape [num_nodes]")
    spatial_fields = (graph.positions, graph.spatial_edge_index)
    if any(value is not None for value in spatial_fields):
        if not all(value is not None for value in spatial_fields):
            raise ValueError("Spatial positions and edges must be present together")
        if graph.atomic_numbers is None:
            raise ValueError("Spatial geometry requires atomic numbers")
        if graph.positions.shape != (graph.num_nodes, 3):
            raise ValueError("Spatial positions must have shape [num_nodes,3]")
        if graph.spatial_edge_index.ndim != 2 or graph.spatial_edge_index.shape[0] != 2:
            raise ValueError("Spatial edge_index must have shape [2,E]")
        if not torch.isfinite(graph.positions).all():
            raise ValueError("Spatial positions must be finite")
        if graph.spatial_edge_index.numel() and (
            int(graph.spatial_edge_index.min()) < 0
            or int(graph.spatial_edge_index.max()) >= graph.num_nodes
        ):
            raise ValueError("Spatial edge index is outside the graph")


def _continuous_encoder(input_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.LayerNorm(hidden_dim),
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
        torch.ones((values.shape[0], 1), dtype=values.dtype, device=values.device),
    )
    return pooled / counts.clamp_min(1.0)


def _segment_softmax(logits: torch.Tensor, graph_ptr: torch.Tensor) -> torch.Tensor:
    probabilities = torch.empty_like(logits)
    for graph_index in range(int(graph_ptr.shape[0]) - 1):
        start = int(graph_ptr[graph_index])
        end = int(graph_ptr[graph_index + 1])
        probabilities[start:end] = torch.softmax(logits[start:end], dim=0)
    return probabilities


def _validate_site_batch(
    values: torch.Tensor,
    graph_ptr: torch.Tensor,
    targets: Sequence[MayrSiteTarget],
) -> None:
    if values.ndim != 1:
        raise ValueError("Packed site values must be a one-dimensional tensor")
    if graph_ptr.ndim != 1 or len(graph_ptr) != len(targets) + 1:
        raise ValueError("graph_ptr must contain one boundary per site target")
    if int(graph_ptr[0]) != 0 or int(graph_ptr[-1]) != len(values):
        raise ValueError("graph_ptr does not span all packed site values")


def _normalise_category(value: object) -> str:
    if value is None:
        return "<UNK>"
    text = str(value).strip()
    return text if text and text.lower() != "nan" else "<UNK>"


def _finite_float(value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return numeric if math.isfinite(numeric) else 0.0


def _mean_or_nan(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _expected_calibration_error(
    confidences: Sequence[float], hits: Sequence[float], *, bins: int
) -> float:
    if not confidences:
        return float("nan")
    confidence_array = np.asarray(confidences, dtype=float)
    hit_array = np.asarray(hits, dtype=float)
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for index in range(bins):
        lower = boundaries[index]
        upper = boundaries[index + 1]
        in_bin = (confidence_array > lower) & (confidence_array <= upper)
        if index == 0:
            in_bin |= confidence_array == 0.0
        if not in_bin.any():
            continue
        weight = float(in_bin.mean())
        error += weight * abs(
            float(hit_array[in_bin].mean()) - float(confidence_array[in_bin].mean())
        )
    return float(error)
