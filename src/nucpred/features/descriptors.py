from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFingerprintGenerator
from sklearn.base import BaseEstimator, TransformerMixin


def _as_smiles_list(values: object) -> list[str]:
    if isinstance(values, pd.DataFrame):
        series = values.iloc[:, 0]
    elif isinstance(values, pd.Series):
        series = values
    else:
        array = np.asarray(values)
        if array.ndim == 2:
            array = array[:, 0]
        series = pd.Series(array)
    return series.astype(str).tolist()


class RDKitDescriptorTransformer(BaseEstimator, TransformerMixin):
    """Convert canonical SMILES into RDKit molecular descriptors."""

    def __init__(self, descriptor_names: tuple[str, ...] | None = None):
        self.descriptor_names = descriptor_names

    def fit(self, X: object, y: object | None = None) -> "RDKitDescriptorTransformer":
        del X, y
        if self.descriptor_names is None:
            self.descriptor_names_ = tuple(name for name, _ in Descriptors._descList)
        else:
            self.descriptor_names_ = tuple(self.descriptor_names)
        self._set_descriptor_functions()
        return self

    def transform(self, X: object) -> np.ndarray:
        rows: list[list[float]] = []
        for smiles in _as_smiles_list(X):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                rows.append([np.nan] * len(self.descriptor_names_))
                continue

            values = []
            for _, fn in self._descriptor_fns_:
                try:
                    value = float(fn(mol))
                except Exception:
                    value = np.nan
                if not np.isfinite(value):
                    value = np.nan
                values.append(value)
            rows.append(values)
        return np.asarray(rows, dtype=np.float64)

    def get_feature_names_out(self, input_features: Iterable[str] | None = None) -> np.ndarray:
        del input_features
        return np.asarray([f"rdkit_{name}" for name in self.descriptor_names_], dtype=object)

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state.pop("_descriptor_fns_", None)
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        if "descriptor_names_" in state:
            self._set_descriptor_functions()

    def _set_descriptor_functions(self) -> None:
        self._descriptor_fns_ = tuple((name, getattr(Descriptors, name)) for name in self.descriptor_names_)


class MorganFingerprintTransformer(BaseEstimator, TransformerMixin):
    """Convert canonical SMILES into Morgan fingerprint bit vectors."""

    def __init__(self, radius: int = 2, n_bits: int = 2048, include_chirality: bool = True):
        self.radius = radius
        self.n_bits = n_bits
        self.include_chirality = include_chirality

    def fit(self, X: object, y: object | None = None) -> "MorganFingerprintTransformer":
        del X, y
        self._set_generator()
        return self

    def transform(self, X: object) -> np.ndarray:
        rows = np.zeros((len(_as_smiles_list(X)), self.n_bits), dtype=np.float32)
        for row_idx, smiles in enumerate(_as_smiles_list(X)):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            fingerprint = self._generator_.GetFingerprint(mol)
            DataStructs.ConvertToNumpyArray(fingerprint, rows[row_idx])
        return rows

    def get_feature_names_out(self, input_features: Iterable[str] | None = None) -> np.ndarray:
        del input_features
        return np.asarray([f"morgan_{idx}" for idx in range(self.n_bits)], dtype=object)

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state.pop("_generator_", None)
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        if "radius" in state:
            self._set_generator()

    def _set_generator(self) -> None:
        self._generator_ = rdFingerprintGenerator.GetMorganGenerator(
            radius=self.radius,
            fpSize=self.n_bits,
            includeChirality=self.include_chirality,
        )
