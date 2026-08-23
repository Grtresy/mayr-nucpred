"""D-isomorphic, xTB-aware pretraining for the ordinary-H Mayr model.

The wrapper in this module owns an exact :class:`MayrOrdinaryHNodeXtbGNN`
instance.  It uses the downstream node, local4, edge, message-passing, site,
and global6 modules directly and adds only disposable pretraining heads.
Solvent, charge, and Mayr-N regression parameters are deliberately present in
the owned downstream model but are never optimized by a pretraining loss and
are never transferred.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import hashlib
import inspect
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from nucpred.features.all_atom_graph import (
    EDGE_CATEGORICAL_FEATURES,
    EDGE_CATEGORY_SIZES,
    ELEMENT_VOCABULARY,
    NODE_CATEGORICAL_FEATURES,
    NODE_CATEGORY_SIZES,
    assert_category_ranges,
)
from nucpred.training.mayr_node_xtb_scratch import (
    GLOBAL_FEATURES,
    LOCAL_FEATURES,
    MayrOrdinaryHNodeXtbGNN,
    ModelInputs,
    seed_everything,
)


CHECKPOINT_SCHEMA_VERSION = "mayr-node-xtb-d-isomorphic-pretraining-checkpoint.v1"
DATA_SCHEMA_VERSION = "mayr-node-xtb-esnuel-pretraining-batch.v1"
CONTRACT_SCHEMA_VERSION = "mayr-node-xtb-pretraining-contract.v1"
GCS_DIM = 53
HYDROGEN_ELEMENT_INDEX = ELEMENT_VOCABULARY.index("H")
EXPECTED_D_NUM_SOLVENTS = 10
EXPECTED_D_PARAMETER_TENSORS = 91
EXPECTED_D_PARAMETER_NUMEL = 535_362
EXPECTED_TRANSFER_PARAMETER_TENSORS = 72
EXPECTED_TRANSFER_PARAMETER_NUMEL = 430_753
DEFAULT_PRETRAINING_TASKS = (
    "masked_node_categorical_reconstruction_all_atoms",
    "masked_edge_categorical_reconstruction",
    "masked_local4_reconstruction_all_atoms",
    "masked_global6_reconstruction",
    "heavy_atom_mca_scalar",
    "heavy_atom_gcs_53d",
    "heavy_atom_mca_soft_site",
    "heavy_atom_mca_within_molecule_ranking",
)
PRODUCTION_ARCHITECTURE = {
    "num_solvents": EXPECTED_D_NUM_SOLVENTS,
    "hidden_dim": 128,
    "layers": 4,
    "node_embedding_dim": 16,
    "edge_embedding_dim": 16,
    "solvent_embedding_dim": 16,
    "dropout": 0.1,
    "gcs_dim": GCS_DIM,
}

TRANSFER_MODULES = (
    "node_encoder",
    "local_encoder",
    "edge_encoder",
    "message_layers.0",
    "message_layers.1",
    "message_layers.2",
    "message_layers.3",
    "site_head",
    "global_xtb_encoder",
)
RESET_MODULES = (
    "solvent_encoder",
    "solvent_embedding",
    "solvent_embedding_projection",
    "charge_encoder",
    "regression_head",
)
PRETRAINING_HEAD_MODULES = (
    "node_reconstruction_heads",
    "edge_reconstruction_heads",
    "local_reconstruction_head",
    "global_reconstruction_head",
    "mca_head",
    "gcs_head",
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: str | Path) -> str:
    handle = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            handle.update(chunk)
    return handle.hexdigest()


def _canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _tensor_mapping_sha256(values: Mapping[str, torch.Tensor]) -> str:
    handle = hashlib.sha256()
    for name in sorted(values):
        tensor = values[name].detach().cpu().contiguous()
        handle.update(name.encode("utf-8"))
        handle.update(b"\0")
        handle.update(str(tensor.dtype).encode("ascii"))
        handle.update(b"\0")
        handle.update(json.dumps(list(tensor.shape)).encode("ascii"))
        handle.update(b"\0")
        handle.update(tensor.numpy().tobytes())
    return handle.hexdigest()


def _state_subset(
    state: Mapping[str, torch.Tensor],
    modules: Sequence[str],
) -> dict[str, torch.Tensor]:
    prefixes = tuple(f"{module}." for module in modules)
    return {key: value for key, value in state.items() if key.startswith(prefixes)}


def _as_sequence(value: object, *, field: str) -> list[object]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{field} is not valid JSON") from exc
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, pd.Series):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be a sequence")
    return list(value)


def _matrix(
    value: object,
    *,
    field: str,
    rows: int | None = None,
    columns: int | None = None,
    dtype: type = float,
) -> np.ndarray:
    outer = _as_sequence(value, field=field)
    parsed: list[list[object]] = []
    for row in outer:
        parsed.append(_as_sequence(row, field=field))
    if rows is not None and len(parsed) != rows:
        raise ValueError(f"{field} row count changed")
    if columns is not None and any(len(row) != columns for row in parsed):
        raise ValueError(f"{field} width changed")
    try:
        matrix = np.asarray(parsed, dtype=dtype)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} contains invalid values") from exc
    if not parsed and columns is not None:
        matrix = matrix.reshape(0, columns)
    return matrix


def _vector(
    value: object,
    *,
    field: str,
    length: int,
    dtype: type = float,
) -> np.ndarray:
    parsed = _as_sequence(value, field=field)
    if len(parsed) != length:
        raise ValueError(f"{field} length changed")
    try:
        return np.asarray(parsed, dtype=dtype)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} contains invalid values") from exc


def _optional_float(value: object) -> float:
    if value is None:
        return math.nan
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return math.nan
    return numeric if math.isfinite(numeric) else math.nan


def _first_column(
    row: Mapping[str, object],
    names: Sequence[str],
    *,
    required: bool = True,
) -> object:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    if required:
        raise ValueError(f"Missing required column; expected one of {list(names)}")
    return None


def _record_id(row: Mapping[str, object]) -> str:
    value = _first_column(row, ("source_id", "molecule_id"))
    identifier = str(value).strip()
    if not identifier:
        raise ValueError("Pretraining record ID is empty")
    return identifier


@dataclass(frozen=True, slots=True)
class EsnuelNodeXtbExample:
    """One explicit-H ESNUEL molecule with xTB and heavy-site proxy labels."""

    source_id: str
    molecule_id: str
    model_canonical_smiles: str
    pretraining_role: str
    node_categorical: torch.Tensor
    edge_index: torch.Tensor
    edge_categorical: torch.Tensor
    is_hydrogen: torch.Tensor
    local_values: np.ndarray
    local_mask: np.ndarray
    global_values: np.ndarray
    global_mask: np.ndarray
    mca_targets: np.ndarray
    mca_mask: np.ndarray
    gcs_targets: np.ndarray
    gcs_mask: np.ndarray

    @property
    def num_nodes(self) -> int:
        return int(self.node_categorical.shape[0])


def _rows_by_source_id(
    table: pd.DataFrame | None,
    *,
    sort_column: str | None = None,
) -> dict[str, pd.DataFrame]:
    if table is None:
        return {}
    if "source_id" not in table:
        raise ValueError("Feature table is missing source_id")
    groups: dict[str, pd.DataFrame] = {}
    for source_id, group in table.groupby("source_id", sort=False):
        ordered = group
        if sort_column is not None:
            if sort_column not in group:
                raise ValueError(f"Feature table is missing {sort_column}")
            ordered = group.sort_values(sort_column)
        groups[str(source_id)] = ordered
    return groups


def _molecule_rows(
    table: pd.DataFrame | None,
) -> dict[str, Mapping[str, object]]:
    if table is None:
        return {}
    if "source_id" not in table:
        raise ValueError("Molecule feature table is missing source_id")
    if table["source_id"].astype(str).duplicated().any():
        raise ValueError("Molecule feature source IDs must be unique")
    return {str(row["source_id"]): row for row in table.to_dict(orient="records")}


def _load_local_features(
    row: Mapping[str, object],
    atoms: pd.DataFrame | None,
    *,
    num_nodes: int,
) -> tuple[np.ndarray, np.ndarray]:
    if atoms is not None:
        missing = [
            name
            for name in LOCAL_FEATURES
            if name not in atoms or f"{name}__available" not in atoms
        ]
        if missing:
            raise ValueError(f"Atom feature table is missing local4 fields: {missing}")
        values = atoms[list(LOCAL_FEATURES)].to_numpy(dtype=float)
        mask = atoms[[f"{name}__available" for name in LOCAL_FEATURES]].to_numpy(
            dtype=bool
        )
    else:
        values = _matrix(
            _first_column(
                row,
                (
                    "node_local4_json",
                    "node_local4",
                    "local4_values",
                ),
            ),
            field="node_local4",
            rows=num_nodes,
            columns=len(LOCAL_FEATURES),
        )
        mask = _matrix(
            _first_column(
                row,
                (
                    "node_local4_available_json",
                    "node_local4_available",
                    "local4_mask",
                ),
            ),
            field="node_local4_available",
            rows=num_nodes,
            columns=len(LOCAL_FEATURES),
            dtype=bool,
        )
    if values.shape != (num_nodes, len(LOCAL_FEATURES)):
        raise ValueError("local4 atom alignment failed")
    mask &= np.isfinite(values)
    return values, mask


def _load_global_features(
    row: Mapping[str, object],
    molecule: Mapping[str, object] | None,
) -> tuple[np.ndarray, np.ndarray]:
    if molecule is not None:
        missing = [
            name
            for name in GLOBAL_FEATURES
            if name not in molecule or f"{name}__available" not in molecule
        ]
        if missing:
            raise ValueError(
                f"Molecule feature table is missing global6 fields: {missing}"
            )
        values = np.asarray(
            [_optional_float(molecule[name]) for name in GLOBAL_FEATURES],
            dtype=float,
        )
        mask = np.asarray(
            [bool(molecule[f"{name}__available"]) for name in GLOBAL_FEATURES],
            dtype=bool,
        )
    else:
        values = _vector(
            _first_column(
                row,
                (
                    "molecule_global6_json",
                    "molecule_global6",
                    "global6_values",
                ),
            ),
            field="molecule_global6",
            length=len(GLOBAL_FEATURES),
        )
        mask = _vector(
            _first_column(
                row,
                (
                    "molecule_global6_available_json",
                    "molecule_global6_available",
                    "global6_mask",
                ),
            ),
            field="molecule_global6_available",
            length=len(GLOBAL_FEATURES),
            dtype=bool,
        )
    mask &= np.isfinite(values)
    return values, mask


def _nullable_target_vector(
    value: object,
    mask_value: object,
    *,
    field: str,
    length: int,
) -> tuple[np.ndarray, np.ndarray]:
    raw = _as_sequence(value, field=field)
    mask = _vector(
        mask_value,
        field=f"{field}_mask",
        length=length,
        dtype=bool,
    )
    if len(raw) != length:
        raise ValueError(f"{field} length changed")
    values = np.asarray([_optional_float(item) for item in raw], dtype=float)
    mask &= np.isfinite(values)
    return values, mask


def _nullable_gcs_matrix(
    value: object,
    mask_value: object,
    *,
    length: int,
    gcs_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    rows = _as_sequence(value, field="gcs_targets_all_atom")
    mask = _vector(
        mask_value,
        field="gcs_target_mask_all_atom",
        length=length,
        dtype=bool,
    )
    if len(rows) != length:
        raise ValueError("gcs_targets_all_atom length changed")
    matrix = np.full((length, gcs_dim), np.nan, dtype=float)
    for index, item in enumerate(rows):
        if item is None or (isinstance(item, float) and not math.isfinite(item)):
            mask[index] = False
            continue
        parsed = _vector(
            item,
            field=f"gcs_targets_all_atom[{index}]",
            length=gcs_dim,
        )
        if not np.isfinite(parsed).all():
            mask[index] = False
            continue
        matrix[index] = parsed
    return matrix, mask


def load_pretraining_examples(
    records: pd.DataFrame,
    atom_features: pd.DataFrame | None = None,
    molecule_features: pd.DataFrame | None = None,
    *,
    roles: Sequence[str] | None = None,
    gcs_dim: int = GCS_DIM,
) -> list[EsnuelNodeXtbExample]:
    """Load the future ESNUEL-xTB three-table contract.

    The canonical schema uses D graph column names, all-atom MCA/GCS arrays,
    atom-level local4 columns, and molecule-level global6 columns.  JSON-only
    local4/global6 records are also accepted so pilot assets can be compact.
    """

    if gcs_dim <= 0:
        raise ValueError("gcs_dim must be positive")
    rows = records.to_dict(orient="records")
    identifiers = [_record_id(row) for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Pretraining record IDs must be unique")
    atom_groups = _rows_by_source_id(atom_features, sort_column="atom_index")
    molecule_by_id = _molecule_rows(molecule_features)
    requested_roles = None if roles is None else {str(role) for role in roles}
    examples: list[EsnuelNodeXtbExample] = []
    for row, source_id in zip(rows, identifiers, strict=True):
        role = str(
            _first_column(
                row,
                (
                    "pretraining_role",
                    "native_pretraining_split",
                    "continuation_role",
                    "effective_split",
                ),
                required=False,
            )
            or "train"
        )
        if requested_roles is not None and role not in requested_roles:
            continue
        num_nodes = int(
            _first_column(
                row,
                ("model_all_atom_count", "all_atom_count", "atom_count"),
            )
        )
        if num_nodes <= 0:
            raise ValueError(f"{source_id}: graph must contain at least one atom")
        node_rows = _matrix(
            _first_column(
                row,
                (
                    "model_node_categorical_json",
                    "all_atom_node_categorical_json",
                    "node_categorical",
                ),
            ),
            field="model_node_categorical_json",
            rows=num_nodes,
            columns=len(NODE_CATEGORICAL_FEATURES),
            dtype=int,
        )
        edge_rows = _matrix(
            _first_column(
                row,
                (
                    "model_directed_edges_json",
                    "all_atom_directed_edges_json",
                    "directed_edges",
                ),
            ),
            field="model_directed_edges_json",
            columns=2,
            dtype=int,
        )
        edge_categories = _matrix(
            _first_column(
                row,
                (
                    "model_edge_categorical_json",
                    "all_atom_edge_categorical_json",
                    "edge_categorical",
                ),
            ),
            field="model_edge_categorical_json",
            rows=len(edge_rows),
            columns=len(EDGE_CATEGORICAL_FEATURES),
            dtype=int,
        )
        assert_category_ranges(node_rows.tolist(), NODE_CATEGORY_SIZES)
        assert_category_ranges(edge_categories.tolist(), EDGE_CATEGORY_SIZES)
        if edge_rows.size and (edge_rows.min() < 0 or edge_rows.max() >= num_nodes):
            raise ValueError(f"{source_id}: directed edge is outside graph")
        atoms = atom_groups.get(source_id)
        if atom_features is not None and atoms is None:
            raise ValueError(f"{source_id}: atom feature rows are missing")
        if atoms is not None and len(atoms) != num_nodes:
            raise ValueError(f"{source_id}: atom feature alignment failed")
        molecule = molecule_by_id.get(source_id)
        if molecule_features is not None and molecule is None:
            raise ValueError(f"{source_id}: molecule feature row is missing")
        local_values, local_mask = _load_local_features(
            row,
            atoms,
            num_nodes=num_nodes,
        )
        global_values, global_mask = _load_global_features(row, molecule)
        atomic_numbers_value = _first_column(
            row,
            (
                "model_atomic_numbers_json",
                "all_atom_atomic_numbers_json",
                "atomic_numbers",
            ),
            required=False,
        )
        if atomic_numbers_value is not None:
            atomic_numbers = _vector(
                atomic_numbers_value,
                field="model_atomic_numbers_json",
                length=num_nodes,
                dtype=int,
            )
            is_hydrogen = atomic_numbers == 1
        elif atoms is not None and "is_hydrogen" in atoms:
            is_hydrogen = atoms["is_hydrogen"].to_numpy(dtype=bool)
        else:
            is_hydrogen = node_rows[:, 0] == HYDROGEN_ELEMENT_INDEX
        encoded_hydrogen = node_rows[:, 0] == HYDROGEN_ELEMENT_INDEX
        if not np.array_equal(is_hydrogen, encoded_hydrogen):
            raise ValueError(
                f"{source_id}: atomic numbers disagree with element categories"
            )
        mca_targets, mca_mask = _nullable_target_vector(
            _first_column(
                row,
                ("mca_targets_all_atom", "mca_targets"),
            ),
            _first_column(
                row,
                (
                    "mca_target_mask_all_atom",
                    "mca_target_mask",
                    "site_mask_all_atom",
                ),
            ),
            field="mca_targets_all_atom",
            length=num_nodes,
        )
        gcs_targets, gcs_mask = _nullable_gcs_matrix(
            _first_column(
                row,
                ("gcs_targets_all_atom", "gcs_targets"),
            ),
            _first_column(
                row,
                (
                    "gcs_target_mask_all_atom",
                    "gcs_target_mask",
                    "site_mask_all_atom",
                ),
            ),
            length=num_nodes,
            gcs_dim=gcs_dim,
        )
        # ESNUEL supplies heavy-atom proxy labels only.  This defensive mask is
        # repeated in packing and loss calculation so H can never become a
        # proxy-site negative due to malformed upstream masks.
        mca_mask &= ~is_hydrogen
        gcs_mask &= ~is_hydrogen
        if not bool(mca_mask.any()):
            raise ValueError(f"{source_id}: no eligible heavy-atom MCA target")
        if not bool(gcs_mask.any()):
            raise ValueError(f"{source_id}: no eligible heavy-atom GCS target")
        edge_index = (
            torch.tensor(edge_rows, dtype=torch.long).transpose(0, 1).contiguous()
            if len(edge_rows)
            else torch.empty((2, 0), dtype=torch.long)
        )
        examples.append(
            EsnuelNodeXtbExample(
                source_id=source_id,
                molecule_id=str(row.get("molecule_id", source_id)),
                model_canonical_smiles=str(
                    row.get(
                        "model_canonical_smiles",
                        row.get("canonical_smiles", ""),
                    )
                ),
                pretraining_role=role,
                node_categorical=torch.tensor(node_rows, dtype=torch.long),
                edge_index=edge_index,
                edge_categorical=torch.tensor(edge_categories, dtype=torch.long),
                is_hydrogen=torch.tensor(is_hydrogen, dtype=torch.bool),
                local_values=local_values,
                local_mask=local_mask,
                global_values=global_values,
                global_mask=global_mask,
                mca_targets=mca_targets,
                mca_mask=mca_mask,
                gcs_targets=gcs_targets,
                gcs_mask=gcs_mask,
            )
        )
    return examples


def _feature_statistics(
    values: np.ndarray,
    mask: np.ndarray,
    *,
    allow_empty_columns: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if values.ndim != 2 or values.shape != mask.shape:
        raise ValueError("Values and masks must be aligned matrices")
    medians: list[float] = []
    means: list[float] = []
    scales: list[float] = []
    for column in range(values.shape[1]):
        observed = values[:, column][mask[:, column] & np.isfinite(values[:, column])]
        if not observed.size:
            if allow_empty_columns:
                medians.append(0.0)
                means.append(0.0)
                scales.append(1.0)
                continue
            raise ValueError(f"No observations for feature column {column}")
        median = float(np.median(observed))
        mean = float(np.mean(observed))
        scale = float(np.std(observed))
        medians.append(median)
        means.append(mean)
        scales.append(scale if scale > 1e-8 else 1.0)
    return (
        np.asarray(medians, dtype=float),
        np.asarray(means, dtype=float),
        np.asarray(scales, dtype=float),
    )


@dataclass(frozen=True, slots=True)
class PretrainingNormalization:
    local_median: tuple[float, ...]
    local_mean: tuple[float, ...]
    local_scale: tuple[float, ...]
    global_median: tuple[float, ...]
    global_mean: tuple[float, ...]
    global_scale: tuple[float, ...]
    mca_mean: float
    mca_scale: float
    gcs_mean: tuple[float, ...]
    gcs_scale: tuple[float, ...]
    fit_source_id_sha256: str
    fit_record_count: int

    def to_json(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_json(
        cls,
        payload: Mapping[str, object],
    ) -> "PretrainingNormalization":
        return cls(
            local_median=tuple(float(item) for item in payload["local_median"]),
            local_mean=tuple(float(item) for item in payload["local_mean"]),
            local_scale=tuple(float(item) for item in payload["local_scale"]),
            global_median=tuple(float(item) for item in payload["global_median"]),
            global_mean=tuple(float(item) for item in payload["global_mean"]),
            global_scale=tuple(float(item) for item in payload["global_scale"]),
            mca_mean=float(payload["mca_mean"]),
            mca_scale=float(payload["mca_scale"]),
            gcs_mean=tuple(float(item) for item in payload["gcs_mean"]),
            gcs_scale=tuple(float(item) for item in payload["gcs_scale"]),
            fit_source_id_sha256=str(payload["fit_source_id_sha256"]),
            fit_record_count=int(payload["fit_record_count"]),
        )


def fit_pretraining_normalization(
    examples: Sequence[EsnuelNodeXtbExample],
    *,
    allow_empty_xtb_features: bool = False,
) -> PretrainingNormalization:
    """Fit every statistic on the declared pretraining-fit records only."""

    if not examples:
        raise ValueError("Cannot fit normalization on an empty dataset")
    local_values = np.concatenate(
        [example.local_values for example in examples],
        axis=0,
    )
    local_mask = np.concatenate(
        [example.local_mask for example in examples],
        axis=0,
    )
    global_values = np.stack(
        [example.global_values for example in examples],
        axis=0,
    )
    global_mask = np.stack(
        [example.global_mask for example in examples],
        axis=0,
    )
    local_median, local_mean, local_scale = _feature_statistics(
        local_values,
        local_mask,
        allow_empty_columns=allow_empty_xtb_features,
    )
    global_median, global_mean, global_scale = _feature_statistics(
        global_values,
        global_mask,
        allow_empty_columns=allow_empty_xtb_features,
    )
    mca = np.concatenate(
        [example.mca_targets[example.mca_mask] for example in examples]
    )
    gcs = np.concatenate(
        [example.gcs_targets[example.gcs_mask] for example in examples],
        axis=0,
    )
    if not mca.size or not np.isfinite(mca).all():
        raise ValueError("MCA fit targets are empty or non-finite")
    if gcs.shape[1] != GCS_DIM or not np.isfinite(gcs).all():
        raise ValueError("GCS fit targets are empty, non-finite, or not 53D")
    mca_scale = float(np.std(mca))
    gcs_scale = np.std(gcs, axis=0)
    gcs_scale[gcs_scale <= 1e-8] = 1.0
    source_ids = sorted(example.source_id for example in examples)
    return PretrainingNormalization(
        local_median=tuple(map(float, local_median)),
        local_mean=tuple(map(float, local_mean)),
        local_scale=tuple(map(float, local_scale)),
        global_median=tuple(map(float, global_median)),
        global_mean=tuple(map(float, global_mean)),
        global_scale=tuple(map(float, global_scale)),
        mca_mean=float(np.mean(mca)),
        mca_scale=mca_scale if mca_scale > 1e-8 else 1.0,
        gcs_mean=tuple(map(float, np.mean(gcs, axis=0))),
        gcs_scale=tuple(map(float, gcs_scale)),
        fit_source_id_sha256=_sha256_bytes(
            ("\n".join(source_ids) + "\n").encode("utf-8")
        ),
        fit_record_count=len(examples),
    )


def _standardize(
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


@dataclass(frozen=True, slots=True)
class MaskingConfig:
    node_categorical_probability: float = 0.15
    edge_categorical_probability: float = 0.15
    local_probability: float = 0.15
    global_probability: float = 0.15

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")


@dataclass(frozen=True)
class PretrainingBatch:
    inputs: ModelInputs
    source_ids: tuple[str, ...]
    is_hydrogen: torch.Tensor
    original_node_categorical: torch.Tensor
    node_reconstruction_mask: torch.Tensor
    original_edge_categorical: torch.Tensor
    edge_reconstruction_mask: torch.Tensor
    local_targets: torch.Tensor
    local_reconstruction_mask: torch.Tensor
    global_targets: torch.Tensor
    global_reconstruction_mask: torch.Tensor
    mca_targets: torch.Tensor
    mca_mask: torch.Tensor
    gcs_targets: torch.Tensor
    gcs_mask: torch.Tensor

    def to(self, device: str | torch.device) -> "PretrainingBatch":
        return PretrainingBatch(
            inputs=self.inputs.to(device),
            source_ids=self.source_ids,
            is_hydrogen=self.is_hydrogen.to(device),
            original_node_categorical=self.original_node_categorical.to(device),
            node_reconstruction_mask=self.node_reconstruction_mask.to(device),
            original_edge_categorical=self.original_edge_categorical.to(device),
            edge_reconstruction_mask=self.edge_reconstruction_mask.to(device),
            local_targets=self.local_targets.to(device),
            local_reconstruction_mask=self.local_reconstruction_mask.to(device),
            global_targets=self.global_targets.to(device),
            global_reconstruction_mask=self.global_reconstruction_mask.to(device),
            mca_targets=self.mca_targets.to(device),
            mca_mask=self.mca_mask.to(device),
            gcs_targets=self.gcs_targets.to(device),
            gcs_mask=self.gcs_mask.to(device),
        )


def _bernoulli_mask(
    shape: Sequence[int],
    probability: float,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    if probability <= 0:
        return torch.zeros(tuple(shape), dtype=torch.bool)
    if probability >= 1:
        return torch.ones(tuple(shape), dtype=torch.bool)
    return torch.rand(tuple(shape), generator=generator) < probability


def pack_pretraining_batch(
    examples: Sequence[EsnuelNodeXtbExample],
    *,
    normalization: PretrainingNormalization,
    masking: MaskingConfig = MaskingConfig(),
    mask_seed: int,
) -> PretrainingBatch:
    """Pack and independently mask categorical, local4, and global6 inputs."""

    if not examples:
        raise ValueError("Cannot pack an empty pretraining batch")
    masking.validate()
    generator = torch.Generator()
    generator.manual_seed(int(mask_seed))
    node_parts: list[torch.Tensor] = []
    edge_index_parts: list[torch.Tensor] = []
    edge_parts: list[torch.Tensor] = []
    graph_index_parts: list[torch.Tensor] = []
    graph_ptr = [0]
    local_parts: list[torch.Tensor] = []
    global_parts: list[torch.Tensor] = []
    hydrogen_parts: list[torch.Tensor] = []
    node_mask_parts: list[torch.Tensor] = []
    edge_mask_parts: list[torch.Tensor] = []
    local_target_parts: list[torch.Tensor] = []
    local_mask_parts: list[torch.Tensor] = []
    global_target_parts: list[torch.Tensor] = []
    global_mask_parts: list[torch.Tensor] = []
    mca_target_parts: list[torch.Tensor] = []
    mca_mask_parts: list[torch.Tensor] = []
    gcs_target_parts: list[torch.Tensor] = []
    gcs_mask_parts: list[torch.Tensor] = []
    original_node_parts: list[torch.Tensor] = []
    original_edge_parts: list[torch.Tensor] = []
    offset = 0
    for graph_number, example in enumerate(examples):
        node_mask = _bernoulli_mask(
            example.node_categorical.shape,
            masking.node_categorical_probability,
            generator=generator,
        )
        masked_nodes = example.node_categorical.clone()
        masked_nodes[node_mask] = 0
        edge_mask = _bernoulli_mask(
            example.edge_categorical.shape,
            masking.edge_categorical_probability,
            generator=generator,
        )
        masked_edges = example.edge_categorical.clone()
        masked_edges[edge_mask] = 0
        local_target_np = _standardize(
            example.local_values,
            example.local_mask,
            median=normalization.local_median,
            mean=normalization.local_mean,
            scale=normalization.local_scale,
        )
        local_target = torch.tensor(local_target_np, dtype=torch.float32)
        local_available = torch.tensor(example.local_mask, dtype=torch.bool)
        local_mask = (
            _bernoulli_mask(
                local_target.shape,
                masking.local_probability,
                generator=generator,
            )
            & local_available
        )
        local_input = torch.cat(
            [local_target, local_available.to(torch.float32)],
            dim=-1,
        )
        local_input[:, : len(LOCAL_FEATURES)][local_mask] = 0.0
        local_input[:, len(LOCAL_FEATURES) :][local_mask] = 0.0
        global_target_np = _standardize(
            example.global_values.reshape(1, -1),
            example.global_mask.reshape(1, -1),
            median=normalization.global_median,
            mean=normalization.global_mean,
            scale=normalization.global_scale,
        )[0]
        global_target = torch.tensor(global_target_np, dtype=torch.float32)
        global_available = torch.tensor(example.global_mask, dtype=torch.bool)
        global_mask = (
            _bernoulli_mask(
                global_target.shape,
                masking.global_probability,
                generator=generator,
            )
            & global_available
        )
        global_input = torch.cat(
            [global_target, global_available.to(torch.float32)],
            dim=-1,
        )
        global_input[: len(GLOBAL_FEATURES)][global_mask] = 0.0
        global_input[len(GLOBAL_FEATURES) :][global_mask] = 0.0
        hydrogen = example.is_hydrogen.to(torch.bool)
        mca_mask = torch.tensor(example.mca_mask, dtype=torch.bool) & ~hydrogen
        gcs_mask = torch.tensor(example.gcs_mask, dtype=torch.bool) & ~hydrogen
        if not bool(mca_mask.any()) or not bool(gcs_mask.any()):
            raise ValueError(
                f"{example.source_id}: batch lost every eligible heavy proxy site"
            )
        mca_values = np.where(
            example.mca_mask & np.isfinite(example.mca_targets),
            example.mca_targets,
            normalization.mca_mean,
        )
        mca_target = torch.tensor(
            (mca_values - normalization.mca_mean) / normalization.mca_scale,
            dtype=torch.float32,
        )
        gcs_values = np.where(
            (example.gcs_mask & np.isfinite(example.gcs_targets).all(axis=1))[:, None],
            example.gcs_targets,
            np.asarray(normalization.gcs_mean)[None, :],
        )
        gcs_target = torch.tensor(
            (gcs_values - np.asarray(normalization.gcs_mean)[None, :])
            / np.asarray(normalization.gcs_scale)[None, :],
            dtype=torch.float32,
        )
        node_parts.append(masked_nodes)
        edge_index_parts.append(example.edge_index + offset)
        edge_parts.append(masked_edges)
        graph_index_parts.append(
            torch.full(
                (example.num_nodes,),
                graph_number,
                dtype=torch.long,
            )
        )
        local_parts.append(local_input)
        global_parts.append(global_input)
        hydrogen_parts.append(hydrogen)
        node_mask_parts.append(node_mask)
        edge_mask_parts.append(edge_mask)
        local_target_parts.append(local_target)
        local_mask_parts.append(local_mask)
        global_target_parts.append(global_target)
        global_mask_parts.append(global_mask)
        mca_target_parts.append(mca_target)
        mca_mask_parts.append(mca_mask)
        gcs_target_parts.append(gcs_target)
        gcs_mask_parts.append(gcs_mask)
        original_node_parts.append(example.node_categorical)
        original_edge_parts.append(example.edge_categorical)
        offset += example.num_nodes
        graph_ptr.append(offset)
    return PretrainingBatch(
        inputs=ModelInputs(
            node_categorical=torch.cat(node_parts, dim=0),
            edge_index=torch.cat(edge_index_parts, dim=1),
            edge_categorical=torch.cat(edge_parts, dim=0),
            node_graph_index=torch.cat(graph_index_parts, dim=0),
            graph_ptr=torch.tensor(graph_ptr, dtype=torch.long),
            node_local=torch.cat(local_parts, dim=0),
            # These reset-only branches are structurally present but unused.
            solvent_continuous=torch.zeros((len(examples), 18), dtype=torch.float32),
            solvent_index=torch.zeros(len(examples), dtype=torch.long),
            molecular_formal_charge=torch.zeros(
                (len(examples), 1), dtype=torch.float32
            ),
            global_xtb=torch.stack(global_parts, dim=0),
        ),
        source_ids=tuple(example.source_id for example in examples),
        is_hydrogen=torch.cat(hydrogen_parts, dim=0),
        original_node_categorical=torch.cat(original_node_parts, dim=0),
        node_reconstruction_mask=torch.cat(node_mask_parts, dim=0),
        original_edge_categorical=torch.cat(original_edge_parts, dim=0),
        edge_reconstruction_mask=torch.cat(edge_mask_parts, dim=0),
        local_targets=torch.cat(local_target_parts, dim=0),
        local_reconstruction_mask=torch.cat(local_mask_parts, dim=0),
        global_targets=torch.stack(global_target_parts, dim=0),
        global_reconstruction_mask=torch.stack(global_mask_parts, dim=0),
        mca_targets=torch.cat(mca_target_parts, dim=0),
        mca_mask=torch.cat(mca_mask_parts, dim=0),
        gcs_targets=torch.cat(gcs_target_parts, dim=0),
        gcs_mask=torch.cat(gcs_mask_parts, dim=0),
    )


def _segment_mean(
    values: torch.Tensor,
    graph_index: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    pooled = values.new_zeros((num_graphs, values.shape[-1]))
    pooled.index_add_(0, graph_index, values)
    counts = values.new_zeros((num_graphs, 1))
    counts.index_add_(
        0,
        graph_index,
        torch.ones(
            (len(values), 1),
            dtype=values.dtype,
            device=values.device,
        ),
    )
    return pooled / counts.clamp_min(1.0)


def _eligible_segment_softmax(
    logits: torch.Tensor,
    graph_ptr: torch.Tensor,
    eligible: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    distribution = torch.zeros_like(logits)
    graph_has_labels = torch.zeros(
        int(graph_ptr.shape[0]) - 1,
        dtype=torch.bool,
        device=logits.device,
    )
    for graph_index in range(int(graph_ptr.shape[0]) - 1):
        start = int(graph_ptr[graph_index])
        end = int(graph_ptr[graph_index + 1])
        support = (
            torch.nonzero(
                eligible[start:end],
                as_tuple=False,
            ).flatten()
            + start
        )
        if not len(support):
            continue
        distribution[support] = torch.softmax(logits[support], dim=0)
        graph_has_labels[graph_index] = True
    return distribution, graph_has_labels


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


@dataclass(frozen=True)
class MayrNodeXtbPretrainingOutput:
    node_embeddings: torch.Tensor
    edge_embeddings: torch.Tensor
    graph_pool: torch.Tensor
    global_embedding: torch.Tensor
    site_logits: torch.Tensor
    eligible_site_distribution: torch.Tensor
    graph_has_site_labels: torch.Tensor
    node_categorical_predictions: tuple[torch.Tensor, ...]
    edge_categorical_predictions: tuple[torch.Tensor, ...]
    local_prediction: torch.Tensor
    global_prediction: torch.Tensor
    mca_prediction: torch.Tensor
    gcs_prediction: torch.Tensor


class MayrNodeXtbPretrainingModel(nn.Module):
    """Disposable heads around an exact downstream D model instance."""

    def __init__(
        self,
        *,
        num_solvents: int = EXPECTED_D_NUM_SOLVENTS,
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
            raise ValueError(f"The ESNUEL GCS contract requires {GCS_DIM} dimensions")
        if init_seed is not None:
            seed_everything(int(init_seed))
        self.initialization_seed = None if init_seed is None else int(init_seed)
        self.architecture = {
            "num_solvents": int(num_solvents),
            "hidden_dim": int(hidden_dim),
            "layers": int(layers),
            "node_embedding_dim": int(node_embedding_dim),
            "edge_embedding_dim": int(edge_embedding_dim),
            "solvent_embedding_dim": int(solvent_embedding_dim),
            "dropout": float(dropout),
            "gcs_dim": int(gcs_dim),
        }
        self.backbone = MayrOrdinaryHNodeXtbGNN(
            num_solvents=num_solvents,
            hidden_dim=hidden_dim,
            layers=layers,
            node_embedding_dim=node_embedding_dim,
            edge_embedding_dim=edge_embedding_dim,
            solvent_embedding_dim=solvent_embedding_dim,
            dropout=dropout,
        )
        self.node_reconstruction_heads = nn.ModuleList(
            _prediction_head(hidden_dim, size, dropout) for size in NODE_CATEGORY_SIZES
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
    ) -> MayrNodeXtbPretrainingOutput:
        inputs = batch.inputs
        # This is deliberately the exact D path, invoked through the exact
        # downstream submodules rather than a separately reimplemented encoder.
        nodes = self.backbone.node_encoder(inputs.node_categorical)
        nodes = nodes + self.backbone.local_encoder(inputs.node_local)
        edges = self.backbone.edge_encoder(inputs.edge_categorical)
        for layer in self.backbone.message_layers:
            nodes = layer(nodes, inputs.edge_index, edges)
        graph_pool = _segment_mean(
            nodes,
            inputs.node_graph_index,
            inputs.num_graphs,
        )
        site_logits = self.backbone.site_head(nodes).squeeze(-1)
        eligible = batch.mca_mask & ~batch.is_hydrogen
        site_distribution, graph_has_site_labels = _eligible_segment_softmax(
            site_logits,
            inputs.graph_ptr,
            eligible,
        )
        global_embedding = self.backbone.global_xtb_encoder(inputs.global_xtb)
        if inputs.edge_index.shape[1]:
            source, destination = inputs.edge_index
            edge_representation = torch.cat(
                [nodes[source], nodes[destination], edges],
                dim=-1,
            )
        else:
            edge_representation = nodes.new_empty((0, 3 * nodes.shape[-1]))
        return MayrNodeXtbPretrainingOutput(
            node_embeddings=nodes,
            edge_embeddings=edges,
            graph_pool=graph_pool,
            global_embedding=global_embedding,
            site_logits=site_logits,
            eligible_site_distribution=site_distribution,
            graph_has_site_labels=graph_has_site_labels,
            node_categorical_predictions=tuple(
                head(nodes) for head in self.node_reconstruction_heads
            ),
            edge_categorical_predictions=tuple(
                head(edge_representation) for head in self.edge_reconstruction_heads
            ),
            local_prediction=self.local_reconstruction_head(nodes),
            global_prediction=self.global_reconstruction_head(global_embedding),
            mca_prediction=self.mca_head(nodes).squeeze(-1),
            gcs_prediction=self.gcs_head(nodes),
        )

    def pretraining_heads_state_dict(self) -> dict[str, torch.Tensor]:
        state = self.state_dict()
        return {
            key: value.detach().cpu()
            for key, value in state.items()
            if not key.startswith("backbone.")
        }


def assert_production_d_architecture(
    backbone: MayrOrdinaryHNodeXtbGNN,
) -> None:
    """Hard gate for the locked 4x128, ten-solvent downstream D instance."""

    if type(backbone) is not MayrOrdinaryHNodeXtbGNN:
        raise TypeError("Production backbone must be the exact downstream D class")
    parameters = dict(backbone.named_parameters())
    shared = _state_subset(parameters, TRANSFER_MODULES)
    observed = {
        "full_parameter_tensors": len(parameters),
        "full_parameter_numel": sum(
            parameter.numel() for parameter in parameters.values()
        ),
        "shared_parameter_tensors": len(shared),
        "shared_parameter_numel": sum(
            parameter.numel() for parameter in shared.values()
        ),
        "message_layers": len(backbone.message_layers),
        "num_solvents": int(backbone.solvent_embedding.num_embeddings),
        "message_dropout": tuple(
            float(layer.dropout.p) for layer in backbone.message_layers
        ),
        "regression_dropout": float(backbone.regression_head[3].p),
    }
    expected = {
        "full_parameter_tensors": EXPECTED_D_PARAMETER_TENSORS,
        "full_parameter_numel": EXPECTED_D_PARAMETER_NUMEL,
        "shared_parameter_tensors": EXPECTED_TRANSFER_PARAMETER_TENSORS,
        "shared_parameter_numel": EXPECTED_TRANSFER_PARAMETER_NUMEL,
        "message_layers": 4,
        "num_solvents": EXPECTED_D_NUM_SOLVENTS,
        "message_dropout": (0.1, 0.1, 0.1, 0.1),
        "regression_dropout": 0.1,
    }
    if observed != expected:
        raise ValueError(
            "Backbone is not the locked production D architecture: "
            f"observed={observed}, expected={expected}"
        )


@dataclass(frozen=True, slots=True)
class PretrainingLossConfig:
    node_categorical_weight: float = 1.0
    edge_categorical_weight: float = 1.0
    local_weight: float = 1.0
    global_weight: float = 1.0
    mca_weight: float = 1.0
    gcs_weight: float = 1.0
    site_weight: float = 0.5
    ranking_weight: float = 0.25
    site_temperature: float = 0.5
    ranking_margin: float = 0.0

    def validate(self) -> None:
        for name, value in asdict(self).items():
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError(f"{name} must be finite")
            if name.endswith("_weight") and numeric < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.site_temperature <= 0:
            raise ValueError("site_temperature must be positive")
        if self.ranking_margin < 0:
            raise ValueError("ranking_margin must be non-negative")


@dataclass(frozen=True)
class PretrainingLossBreakdown:
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


def _mca_site_cross_entropy(
    prediction: torch.Tensor,
    mca_targets: torch.Tensor,
    eligible: torch.Tensor,
    graph_ptr: torch.Tensor,
    *,
    temperature: float,
    zero: torch.Tensor,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for graph_index in range(int(graph_ptr.shape[0]) - 1):
        start = int(graph_ptr[graph_index])
        end = int(graph_ptr[graph_index + 1])
        support = (
            torch.nonzero(
                eligible[start:end],
                as_tuple=False,
            ).flatten()
            + start
        )
        if not len(support):
            continue
        target = torch.softmax(mca_targets[support] / temperature, dim=0)
        losses.append(-(target * prediction[support].clamp_min(1e-12).log()).sum())
    return torch.stack(losses).mean() if losses else zero


def _within_graph_ranking_loss(
    prediction: torch.Tensor,
    targets: torch.Tensor,
    eligible: torch.Tensor,
    graph_ptr: torch.Tensor,
    *,
    margin: float,
    zero: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    losses: list[torch.Tensor] = []
    pair_count = 0
    for graph_index in range(int(graph_ptr.shape[0]) - 1):
        start = int(graph_ptr[graph_index])
        end = int(graph_ptr[graph_index + 1])
        indices = (
            torch.nonzero(
                eligible[start:end],
                as_tuple=False,
            ).flatten()
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
        left = left[non_ties]
        right = right[non_ties]
        direction = torch.sign(target_delta[non_ties])
        losses.append(
            F.softplus(margin - direction * (prediction[left] - prediction[right]))
        )
        pair_count += int(len(left))
    if not losses:
        return zero, 0
    return torch.cat(losses).mean(), pair_count


def pretraining_loss(
    output: MayrNodeXtbPretrainingOutput,
    batch: PretrainingBatch,
    config: PretrainingLossConfig = PretrainingLossConfig(),
) -> PretrainingLossBreakdown:
    """All reconstruction/proxy tasks with an explicit no-H proxy policy."""

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
        "site": _mca_site_cross_entropy(
            output.eligible_site_distribution,
            batch.mca_targets,
            eligible_mca,
            batch.inputs.graph_ptr,
            temperature=config.site_temperature,
            zero=zero,
        ),
    }
    ranking, ranking_pairs = _within_graph_ranking_loss(
        output.mca_prediction,
        batch.mca_targets,
        eligible_mca,
        batch.inputs.graph_ptr,
        margin=config.ranking_margin,
        zero=zero,
    )
    components["ranking"] = ranking
    weights = {
        "node_categorical": config.node_categorical_weight,
        "edge_categorical": config.edge_categorical_weight,
        "local4": config.local_weight,
        "global6": config.global_weight,
        "mca": config.mca_weight,
        "gcs": config.gcs_weight,
        "site": config.site_weight,
        "ranking": config.ranking_weight,
    }
    total = sum(
        (float(weights[name]) * value for name, value in components.items()),
        zero,
    )
    return PretrainingLossBreakdown(
        total=total,
        components=components,
        ranking_pairs=ranking_pairs,
        eligible_mca_atoms=int(eligible_mca.sum()),
        eligible_gcs_atoms=int(eligible_gcs.sum()),
    )


def required_gradient_audit(
    model: MayrNodeXtbPretrainingModel,
) -> dict[str, bool]:
    """Report non-zero finite gradients for every required transferable path."""

    modules: dict[str, nn.Module] = {
        "node_encoder": model.backbone.node_encoder,
        "local_encoder": model.backbone.local_encoder,
        "edge_encoder": model.backbone.edge_encoder,
        "site_head": model.backbone.site_head,
        "global_xtb_encoder": model.backbone.global_xtb_encoder,
        **{
            f"message_layers.{index}": layer
            for index, layer in enumerate(model.backbone.message_layers)
        },
        **{name: getattr(model, name) for name in PRETRAINING_HEAD_MODULES},
        **{
            f"node_reconstruction_heads.{index}": head
            for index, head in enumerate(model.node_reconstruction_heads)
        },
        **{
            f"edge_reconstruction_heads.{index}": head
            for index, head in enumerate(model.edge_reconstruction_heads)
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
    element_gradient = model.backbone.node_encoder.embeddings[0].weight.grad
    audit["ordinary_h_element_embedding"] = (
        element_gradient is not None
        and bool(torch.isfinite(element_gradient[HYDROGEN_ELEMENT_INDEX]).all())
        and bool(torch.count_nonzero(element_gradient[HYDROGEN_ELEMENT_INDEX]))
    )
    return audit


def run_required_gradient_gate(
    model: MayrNodeXtbPretrainingModel,
    examples: Sequence[EsnuelNodeXtbExample],
    *,
    normalization: PretrainingNormalization,
    loss_config: PretrainingLossConfig = PretrainingLossConfig(),
    seed: int,
    device: str | torch.device,
    max_molecules: int = 32,
    attempts: int = 32,
) -> dict[str, bool]:
    """Fail unless every transferable module, head, and ordinary-H row learns.

    A fixed 50% diagnostic mask is intentionally independent of the campaign
    masking hyperparameter.  It checks connectivity of every objective without
    altering parameters or using validation/audit-test records.
    """

    if not examples:
        raise ValueError("Gradient gate requires training examples")
    if max_molecules <= 0 or attempts <= 0:
        raise ValueError("Gradient gate limits must be positive")
    diagnostic_masking = MaskingConfig(
        node_categorical_probability=0.5,
        edge_categorical_probability=0.5,
        local_probability=0.5,
        global_probability=0.5,
    )
    previous_mode = model.training
    model.eval()
    last_audit: dict[str, bool] = {}
    diagnostic_examples = sorted(
        examples,
        key=lambda example: (
            -int(example.mca_mask.sum()),
            -int(example.is_hydrogen.sum()),
            example.source_id,
        ),
    )[:max_molecules]
    try:
        for attempt in range(attempts):
            model.zero_grad(set_to_none=True)
            batch = pack_pretraining_batch(
                diagnostic_examples,
                normalization=normalization,
                masking=diagnostic_masking,
                mask_seed=int(seed) + attempt,
            ).to(device)
            breakdown = pretraining_loss(model(batch), batch, loss_config)
            if not bool(torch.isfinite(breakdown.total)):
                raise RuntimeError("Gradient-gate loss is non-finite")
            breakdown.total.backward()
            last_audit = required_gradient_audit(model)
            if last_audit and all(last_audit.values()):
                return last_audit
    finally:
        model.zero_grad(set_to_none=True)
        model.train(previous_mode)
    failures = sorted(name for name, passed in last_audit.items() if not passed)
    raise RuntimeError(
        "Required-gradient gate failed after "
        f"{attempts} deterministic masks: {failures}"
    )


def _source_hash_for_downstream_model() -> str:
    source = inspect.getsourcefile(MayrOrdinaryHNodeXtbGNN)
    if source is None:
        raise RuntimeError("Cannot locate downstream D model source")
    return _sha256_file(source)


def _feature_contract() -> dict[str, object]:
    return {
        "node_features": list(NODE_CATEGORICAL_FEATURES),
        "node_category_sizes": list(NODE_CATEGORY_SIZES),
        "edge_features": list(EDGE_CATEGORICAL_FEATURES),
        "edge_category_sizes": list(EDGE_CATEGORY_SIZES),
        "local4": list(LOCAL_FEATURES),
        "global6": list(GLOBAL_FEATURES),
        "gcs_dimension": GCS_DIM,
        "hydrogen_element_index": HYDROGEN_ELEMENT_INDEX,
        "hydrogen_proxy_policy": (
            "H participates in categorical/local/global reconstruction and "
            "message passing; H is excluded from MCA/GCS/site/ranking support"
        ),
    }


def _transfer_architecture_contract(
    model: MayrNodeXtbPretrainingModel,
) -> dict[str, object]:
    architecture = dict(model.architecture)
    # Solvent vocabulary size affects only a reset module.
    architecture.pop("num_solvents", None)
    return {
        "backbone_class": (
            "nucpred.training.mayr_node_xtb_scratch.MayrOrdinaryHNodeXtbGNN"
        ),
        "architecture": architecture,
        "transfer_modules": list(TRANSFER_MODULES),
        "reset_modules": list(RESET_MODULES),
    }


def build_pretraining_contract(
    model: MayrNodeXtbPretrainingModel,
    *,
    dataset_contract_hashes: Mapping[str, str] | None = None,
    tasks: Sequence[str] | None = None,
    variant_contract: Mapping[str, object] | None = None,
) -> tuple[dict[str, object], dict[str, str]]:
    feature_contract = _feature_contract()
    architecture_contract = _transfer_architecture_contract(model)
    pretraining_source = Path(__file__)
    contract: dict[str, object] = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "data_schema_version": DATA_SCHEMA_VERSION,
        "initialization_seed": model.initialization_seed,
        "feature_contract": feature_contract,
        "transfer_contract": architecture_contract,
        "tasks": list(tasks or DEFAULT_PRETRAINING_TASKS),
        "dataset_contract_hashes": dict(dataset_contract_hashes or {}),
    }
    if variant_contract is not None:
        contract["variant_contract"] = dict(variant_contract)
    hashes = {
        "downstream_source_sha256": _source_hash_for_downstream_model(),
        "pretraining_source_sha256": _sha256_file(pretraining_source),
        "feature_contract_sha256": _canonical_json_sha256(feature_contract),
        "transfer_architecture_sha256": _canonical_json_sha256(architecture_contract),
        "dataset_contract_sha256": _canonical_json_sha256(
            contract["dataset_contract_hashes"]
        ),
    }
    hashes["full_contract_sha256"] = _canonical_json_sha256(
        {"contract": contract, "hashes": hashes}
    )
    return contract, hashes


def _checkpoint_payload(
    model: MayrNodeXtbPretrainingModel,
    *,
    optimizer: torch.optim.Optimizer | None,
    history: Sequence[Mapping[str, object]],
    normalization: PretrainingNormalization,
    masking: MaskingConfig,
    loss_config: PretrainingLossConfig,
    dataset_contract_hashes: Mapping[str, str] | None,
    selection: Mapping[str, object] | None,
    audit_metrics: Mapping[str, float] | None,
    gradient_audit: Mapping[str, bool] | None,
    tasks: Sequence[str] | None,
    variant_contract: Mapping[str, object] | None,
) -> dict[str, object]:
    assert_production_d_architecture(model.backbone)
    if model.architecture != PRODUCTION_ARCHITECTURE:
        raise ValueError(
            "Checkpoint architecture metadata differs from locked production D"
        )
    if model.initialization_seed is None:
        raise ValueError(
            "Audited pretraining checkpoints require an initialization seed"
        )
    contract, contract_hashes = build_pretraining_contract(
        model,
        dataset_contract_hashes=dataset_contract_hashes,
        tasks=tasks,
        variant_contract=variant_contract,
    )
    backbone_state = {
        key: value.detach().cpu() for key, value in model.backbone.state_dict().items()
    }
    backbone_state_sha256 = _tensor_mapping_sha256(backbone_state)
    seed_state_binding_sha256 = _canonical_json_sha256(
        {
            "init_seed": model.initialization_seed,
            "backbone_state_sha256": backbone_state_sha256,
        }
    )
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "init_seed": model.initialization_seed,
        "backbone_class": (
            "nucpred.training.mayr_node_xtb_scratch.MayrOrdinaryHNodeXtbGNN"
        ),
        "architecture": dict(model.architecture),
        "backbone_state_dict": backbone_state,
        "pretraining_heads_state_dict": model.pretraining_heads_state_dict(),
        "optimizer_state_dict": (None if optimizer is None else optimizer.state_dict()),
        "history": [dict(row) for row in history],
        "selection": dict(selection or {}),
        "audit_test_metrics": dict(audit_metrics or {}),
        "required_gradient_audit": dict(gradient_audit or {}),
        "normalization": normalization.to_json(),
        "masking_config": asdict(masking),
        "loss_config": asdict(loss_config),
        "contract": contract,
        "contract_hashes": contract_hashes,
        "backbone_state_sha256": backbone_state_sha256,
        "seed_state_binding_sha256": seed_state_binding_sha256,
        "pretraining_heads_state_sha256": _tensor_mapping_sha256(
            model.pretraining_heads_state_dict()
        ),
    }


def save_pretraining_checkpoint(
    path: str | Path,
    model: MayrNodeXtbPretrainingModel,
    *,
    optimizer: torch.optim.Optimizer | None,
    history: Sequence[Mapping[str, object]],
    normalization: PretrainingNormalization,
    masking: MaskingConfig = MaskingConfig(),
    loss_config: PretrainingLossConfig = PretrainingLossConfig(),
    dataset_contract_hashes: Mapping[str, str] | None = None,
    selection: Mapping[str, object] | None = None,
    audit_metrics: Mapping[str, float] | None = None,
    gradient_audit: Mapping[str, bool] | None = None,
    tasks: Sequence[str] | None = None,
    variant_contract: Mapping[str, object] | None = None,
) -> Path:
    """Save complete evidence plus a full D state and disposable heads."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = _checkpoint_payload(
        model,
        optimizer=optimizer,
        history=history,
        normalization=normalization,
        masking=masking,
        loss_config=loss_config,
        dataset_contract_hashes=dataset_contract_hashes,
        selection=selection,
        audit_metrics=audit_metrics,
        gradient_audit=gradient_audit,
        tasks=tasks,
        variant_contract=variant_contract,
    )
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)
    return output


def load_pretraining_checkpoint(
    checkpoint: str | Path | Mapping[str, object],
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, object]:
    if isinstance(checkpoint, Mapping):
        payload = dict(checkpoint)
    else:
        payload = torch.load(
            checkpoint,
            map_location=map_location,
            weights_only=False,
        )
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Pretraining checkpoint schema changed")
    backbone_state = payload.get("backbone_state_dict")
    heads_state = payload.get("pretraining_heads_state_dict")
    if not isinstance(backbone_state, Mapping) or not isinstance(heads_state, Mapping):
        raise ValueError("Checkpoint is missing model state dictionaries")
    if _tensor_mapping_sha256(backbone_state) != payload.get("backbone_state_sha256"):
        raise ValueError("Backbone state hash mismatch")
    init_seed = payload.get("init_seed")
    if not isinstance(init_seed, int):
        raise ValueError("Checkpoint is missing its integer initialization seed")
    if payload.get("seed_state_binding_sha256") != _canonical_json_sha256(
        {
            "init_seed": init_seed,
            "backbone_state_sha256": payload["backbone_state_sha256"],
        }
    ):
        raise ValueError("Initialization-seed/state binding hash mismatch")
    if _tensor_mapping_sha256(heads_state) != payload.get(
        "pretraining_heads_state_sha256"
    ):
        raise ValueError("Pretraining-head state hash mismatch")
    contract = payload.get("contract")
    hashes = payload.get("contract_hashes")
    if not isinstance(contract, Mapping) or not isinstance(hashes, Mapping):
        raise ValueError("Checkpoint is missing contract evidence")
    if contract.get("initialization_seed") != init_seed:
        raise ValueError("Checkpoint contract/init seed mismatch")
    if payload.get("architecture") != PRODUCTION_ARCHITECTURE:
        raise ValueError("Checkpoint is not bound to the production architecture")
    recomputed_hashes = dict(hashes)
    declared_full = recomputed_hashes.pop("full_contract_sha256", None)
    if declared_full != _canonical_json_sha256(
        {"contract": dict(contract), "hashes": recomputed_hashes}
    ):
        raise ValueError("Checkpoint contract hash mismatch")
    return payload


@dataclass(frozen=True, slots=True)
class TransferAudit:
    status: str
    schema_version: str
    copied_modules: tuple[str, ...]
    copied_keys: tuple[str, ...]
    reset_modules: tuple[str, ...]
    reset_keys: tuple[str, ...]
    checkpoint_contract_hashes: Mapping[str, str]
    target_before_sha256: str
    target_after_sha256: str
    copied_state_sha256: str
    reset_state_sha256: str

    def to_json(self) -> dict[str, object]:
        return {
            "status": self.status,
            "schema_version": self.schema_version,
            "copied_modules": list(self.copied_modules),
            "copied_keys": list(self.copied_keys),
            "reset_modules": list(self.reset_modules),
            "reset_keys": list(self.reset_keys),
            "checkpoint_contract_hashes": dict(self.checkpoint_contract_hashes),
            "target_before_sha256": self.target_before_sha256,
            "target_after_sha256": self.target_after_sha256,
            "copied_state_sha256": self.copied_state_sha256,
            "reset_state_sha256": self.reset_state_sha256,
        }


def transfer_pretrained_backbone(
    checkpoint: str | Path | Mapping[str, object],
    target: MayrOrdinaryHNodeXtbGNN,
    *,
    expected_contract_hashes: Mapping[str, str] | None = None,
) -> TransferAudit:
    """Strictly copy D-shared paths and prove reset-only paths stayed pristine."""

    if type(target) is not MayrOrdinaryHNodeXtbGNN:
        raise TypeError("Transfer target must be an exact MayrOrdinaryHNodeXtbGNN")
    assert_production_d_architecture(target)
    payload = load_pretraining_checkpoint(checkpoint)
    contract_hashes = {
        str(key): str(value) for key, value in dict(payload["contract_hashes"]).items()
    }
    if (
        contract_hashes.get("downstream_source_sha256")
        != _source_hash_for_downstream_model()
    ):
        raise ValueError("Locked downstream D source hash changed")
    if contract_hashes.get("feature_contract_sha256") != _canonical_json_sha256(
        _feature_contract()
    ):
        raise ValueError("Pretraining/downstream feature contract changed")
    if expected_contract_hashes is not None:
        for key, expected in expected_contract_hashes.items():
            if contract_hashes.get(str(key)) != str(expected):
                raise ValueError(f"Checkpoint contract hash mismatch for {key}")
    source_state = payload["backbone_state_dict"]
    target_before = {
        key: value.detach().cpu().clone() for key, value in target.state_dict().items()
    }
    source_keys = set(source_state)
    target_keys = set(target_before)
    if source_keys != target_keys:
        raise ValueError(
            "Checkpoint and target D state keys differ; full-state parity failed"
        )
    for key in sorted(target_keys):
        source_tensor = source_state[key]
        target_tensor = target_before[key]
        if (
            source_tensor.shape != target_tensor.shape
            or source_tensor.dtype != target_tensor.dtype
        ):
            raise ValueError(f"Complete D tensor contract changed for {key}")
        if source_tensor.is_floating_point() and not bool(
            torch.isfinite(source_tensor).all()
        ):
            raise ValueError(f"Checkpoint D tensor is non-finite: {key}")
    copied = _state_subset(target_before, TRANSFER_MODULES)
    reset = _state_subset(target_before, RESET_MODULES)
    for module in TRANSFER_MODULES:
        if not any(key.startswith(f"{module}.") for key in copied):
            raise ValueError(f"Transfer module has no parameters: {module}")
    for module in RESET_MODULES:
        if not any(key.startswith(f"{module}.") for key in reset):
            raise ValueError(f"Reset module has no parameters: {module}")
    classified = set(copied) | set(reset)
    if classified != target_keys:
        unknown = sorted(target_keys - classified)
        raise ValueError(f"Unclassified D state keys: {unknown}")
    replacement = {
        key: value.detach().clone() for key, value in target.state_dict().items()
    }
    for key in sorted(copied):
        source_tensor = source_state[key]
        target_tensor = replacement[key]
        replacement[key] = source_tensor.to(
            device=target_tensor.device,
            dtype=target_tensor.dtype,
        ).clone()
    target.load_state_dict(replacement, strict=True)
    target_after = {
        key: value.detach().cpu().clone() for key, value in target.state_dict().items()
    }
    for key in copied:
        if not torch.equal(target_after[key], source_state[key].cpu()):
            raise RuntimeError(f"Transferred tensor verification failed for {key}")
    for key in reset:
        if not torch.equal(target_after[key], target_before[key]):
            raise RuntimeError(f"Reset-only tensor changed during transfer: {key}")
    return TransferAudit(
        status="pass",
        schema_version=CHECKPOINT_SCHEMA_VERSION,
        copied_modules=TRANSFER_MODULES,
        copied_keys=tuple(sorted(copied)),
        reset_modules=RESET_MODULES,
        reset_keys=tuple(sorted(reset)),
        checkpoint_contract_hashes=contract_hashes,
        target_before_sha256=_tensor_mapping_sha256(target_before),
        target_after_sha256=_tensor_mapping_sha256(target_after),
        copied_state_sha256=_tensor_mapping_sha256(
            {key: target_after[key] for key in copied}
        ),
        reset_state_sha256=_tensor_mapping_sha256(
            {key: target_after[key] for key in reset}
        ),
    )


@dataclass(frozen=True, slots=True)
class TrainingResult:
    model: MayrNodeXtbPretrainingModel
    optimizer: torch.optim.Optimizer
    history: tuple[Mapping[str, object], ...]
    best_epoch: int
    best_validation_total: float
    audit_metrics: Mapping[str, float]
    gradient_audit: Mapping[str, bool]


def _epoch_batches(
    examples: Sequence[EsnuelNodeXtbExample],
    *,
    batch_size: int,
    seed: int,
    shuffle: bool,
) -> list[list[EsnuelNodeXtbExample]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    order = np.arange(len(examples))
    if shuffle:
        order = np.random.default_rng(int(seed)).permutation(order)
    return [
        [examples[int(index)] for index in order[start : start + batch_size]]
        for start in range(0, len(order), batch_size)
    ]


def _run_epoch(
    model: MayrNodeXtbPretrainingModel,
    examples: Sequence[EsnuelNodeXtbExample],
    *,
    normalization: PretrainingNormalization,
    optimizer: torch.optim.Optimizer | None,
    batch_size: int,
    seed: int,
    device: str | torch.device,
    masking: MaskingConfig,
    loss_config: PretrainingLossConfig,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    weights = 0
    batches = _epoch_batches(
        examples,
        batch_size=batch_size,
        seed=seed,
        shuffle=training,
    )
    for batch_index, batch_examples in enumerate(batches):
        batch = pack_pretraining_batch(
            batch_examples,
            normalization=normalization,
            masking=masking,
            mask_seed=seed * 1_000_003 + batch_index,
        ).to(device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output = model(batch)
            breakdown = pretraining_loss(output, batch, loss_config)
            if optimizer is not None:
                breakdown.total.backward()
                optimizer.step()
        count = len(batch_examples)
        weights += count
        totals["total"] = totals.get("total", 0.0) + (
            float(breakdown.total.detach()) * count
        )
        for name, value in breakdown.components.items():
            totals[name] = totals.get(name, 0.0) + (float(value.detach()) * count)
    if not weights:
        raise ValueError("Cannot run an epoch on an empty dataset")
    return {name: value / weights for name, value in totals.items()}


def train_pretraining_pilot(
    train_examples: Sequence[EsnuelNodeXtbExample],
    validation_examples: Sequence[EsnuelNodeXtbExample],
    *,
    audit_test_examples: Sequence[EsnuelNodeXtbExample] = (),
    normalization: PretrainingNormalization | None = None,
    init_seed: int = 31001,
    epochs: int = 10,
    min_epochs: int = 1,
    patience: int | None = None,
    min_delta: float = 0.0,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    device: str | torch.device = "cpu",
    masking: MaskingConfig = MaskingConfig(),
    loss_config: PretrainingLossConfig = PretrainingLossConfig(),
    hidden_dim: int = 128,
    layers: int = 4,
    dropout: float = 0.1,
    require_gradient_gate: bool = True,
) -> TrainingResult:
    """Deterministic trainer with validation-only selection and audit holdout."""

    if not train_examples or not validation_examples:
        raise ValueError("Pilot requires non-empty train and validation records")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if not 1 <= min_epochs <= epochs:
        raise ValueError("min_epochs must be in [1, epochs]")
    if patience is not None and patience <= 0:
        raise ValueError("patience must be positive or None")
    if not math.isfinite(min_delta) or min_delta < 0:
        raise ValueError("min_delta must be finite and non-negative")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    seed_everything(init_seed)
    fitted = normalization or fit_pretraining_normalization(train_examples)
    model = MayrNodeXtbPretrainingModel(
        num_solvents=EXPECTED_D_NUM_SOLVENTS,
        hidden_dim=hidden_dim,
        layers=layers,
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
    best_validation_total = math.inf
    best_model_state: dict[str, torch.Tensor] | None = None
    best_optimizer_state: dict[str, object] | None = None
    stale_epochs = 0
    for epoch in range(epochs):
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
            # A fixed validation corruption makes epoch selection comparable.
            seed=init_seed + 10_000_000,
            device=device,
            masking=masking,
            loss_config=loss_config,
        )
        validation_total = float(validation_metrics["total"])
        improved = validation_total < (best_validation_total - float(min_delta))
        if improved:
            best_epoch = epoch + 1
            best_validation_total = validation_total
            best_model_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_optimizer_state = copy.deepcopy(optimizer.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        row: dict[str, object] = {
            "epoch": epoch + 1,
            "is_validation_best": improved,
        }
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update(
            {f"validation_{key}": value for key, value in validation_metrics.items()}
        )
        history.append(row)
        if (
            patience is not None
            and epoch + 1 >= min_epochs
            and stale_epochs >= patience
        ):
            break
    if best_model_state is None or best_optimizer_state is None:
        raise RuntimeError("Validation selection did not produce a finite best state")
    model.load_state_dict(best_model_state, strict=True)
    optimizer.load_state_dict(best_optimizer_state)
    audit_metrics: Mapping[str, float] = {}
    if audit_test_examples:
        # Audit-test is touched exactly once, after validation selection.
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
    return TrainingResult(
        model=model,
        optimizer=optimizer,
        history=tuple(history),
        best_epoch=best_epoch,
        best_validation_total=best_validation_total,
        audit_metrics=dict(audit_metrics),
        gradient_audit=dict(gradient_audit),
    )


def _read_table(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if source.suffix == ".parquet":
        return pd.read_parquet(source)
    if source.suffix in {".csv", ".gz"}:
        return pd.read_csv(source)
    raise ValueError(f"Unsupported table format: {source}")


def _path_hashes(paths: Mapping[str, Path]) -> dict[str, str]:
    return {f"{name}_sha256": _sha256_file(path) for name, path in paths.items()}


def _source_id_sha256(
    examples: Sequence[EsnuelNodeXtbExample],
) -> str:
    return _sha256_bytes(
        ("\n".join(sorted(example.source_id for example in examples)) + "\n").encode(
            "utf-8"
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for pilot or full, seed-bound pretraining."""

    parser = argparse.ArgumentParser(
        description="Run D-isomorphic ESNUEL-xTB pretraining"
    )
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--atom-features", type=Path, required=True)
    parser.add_argument("--molecule-features", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--history-json", type=Path)
    parser.add_argument("--history-csv", type=Path)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--min-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--init-seed", type=int, default=31001)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pilot-max-molecules", type=int, default=4000)
    args = parser.parse_args(argv)
    tables = {
        "records": args.records,
        "atom_features": args.atom_features,
        "molecule_features": args.molecule_features,
    }
    examples = load_pretraining_examples(
        _read_table(args.records),
        _read_table(args.atom_features),
        _read_table(args.molecule_features),
    )
    train_all = [example for example in examples if example.pretraining_role == "train"]
    validation_all = [
        example for example in examples if example.pretraining_role == "validation"
    ]
    audit_all = [
        example for example in examples if example.pretraining_role == "audit_test"
    ]
    if args.pilot_max_molecules > 0:
        train = train_all[: args.pilot_max_molecules]
        holdout_limit = max(1, args.pilot_max_molecules // 5)
        validation = validation_all[:holdout_limit]
        audit_test = audit_all[:holdout_limit]
    else:
        train = train_all
        validation = validation_all
        audit_test = audit_all
    if not audit_test:
        raise ValueError("An audit_test partition is required")
    normalization = fit_pretraining_normalization(train)
    result = train_pretraining_pilot(
        train,
        validation,
        audit_test_examples=audit_test,
        normalization=normalization,
        init_seed=args.init_seed,
        epochs=args.epochs,
        min_epochs=args.min_epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=args.device,
        hidden_dim=128,
        layers=4,
    )
    save_pretraining_checkpoint(
        args.output_checkpoint,
        result.model,
        optimizer=result.optimizer,
        history=result.history,
        normalization=normalization,
        dataset_contract_hashes=_path_hashes(tables),
        selection={
            "metric": "validation_total",
            "mode": "min",
            "best_epoch": result.best_epoch,
            "best_validation_total": result.best_validation_total,
            "trained_epochs": len(result.history),
            "min_epochs": args.min_epochs,
            "patience": args.patience,
            "min_delta": args.min_delta,
            "audit_test_used_for_selection": False,
            "partition_counts": {
                "train": len(train),
                "validation": len(validation),
                "audit_test": len(audit_test),
            },
            "partition_source_id_sha256": {
                "train": _source_id_sha256(train),
                "validation": _source_id_sha256(validation),
                "audit_test": _source_id_sha256(audit_test),
            },
        },
        audit_metrics=result.audit_metrics,
        gradient_audit=result.gradient_audit,
    )
    history_json = args.history_json or args.output_checkpoint.with_suffix(
        ".history.json"
    )
    history_csv = args.history_csv or args.output_checkpoint.with_suffix(".history.csv")
    history_json.parent.mkdir(parents=True, exist_ok=True)
    history_json.write_text(
        json.dumps(
            {
                "history": list(result.history),
                "selection": {
                    "best_epoch": result.best_epoch,
                    "best_validation_total": result.best_validation_total,
                },
                "audit_test_metrics": dict(result.audit_metrics),
                "required_gradient_audit": dict(result.gradient_audit),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    history_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result.history).to_csv(history_csv, index=False)
    print(
        json.dumps(
            {
                "status": "pass",
                "checkpoint": args.output_checkpoint.as_posix(),
                "init_seed": args.init_seed,
                "best_epoch": result.best_epoch,
                "best_validation_total": result.best_validation_total,
                "audit_test_metrics": dict(result.audit_metrics),
                "history_json": history_json.as_posix(),
                "history_csv": history_csv.as_posix(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
