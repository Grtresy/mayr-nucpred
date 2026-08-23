"""Target-independent categorical features for ordinary-H Mayr graphs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Sequence

from rdkit import Chem


ELEMENT_VOCABULARY = (
    "<UNK>",
    "B",
    "Br",
    "C",
    "Cl",
    "F",
    "Ge",
    "H",
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

_ELEMENT_TO_INDEX = {
    element: index for index, element in enumerate(ELEMENT_VOCABULARY)
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
    len(ELEMENT_VOCABULARY),
    9,
    8,
    6,
    2,
    2,
    max(_HYBRIDIZATION_TO_INDEX.values()) + 1,
    max(_CHIRALITY_TO_INDEX.values()) + 1,
    5,
)
EDGE_CATEGORY_SIZES = (
    max(_BOND_TYPE_TO_INDEX.values()) + 1,
    2,
    2,
    2,
    max(_BOND_STEREO_TO_INDEX.values()) + 1,
)


@dataclass(frozen=True, slots=True)
class AllAtomGraph:
    atomic_numbers: tuple[int, ...]
    node_categorical: tuple[tuple[int, ...], ...]
    directed_edges: tuple[tuple[int, int], ...]
    edge_categorical: tuple[tuple[int, ...], ...]
    hydrogen_parent_index: tuple[int, ...]
    source_atom_count: int
    mapping_sha256: str

    @property
    def atom_count(self) -> int:
        return len(self.atomic_numbers)


def _bounded_category(value: int, *, minimum: int, maximum: int) -> int:
    numeric = int(value)
    if minimum <= numeric <= maximum:
        return numeric - minimum
    return maximum - minimum + 1


def atom_categories(atom: Chem.Atom) -> tuple[int, ...]:
    """Use the heavy-atom model fields with H added to the shared vocabulary."""

    return (
        _ELEMENT_TO_INDEX.get(atom.GetSymbol(), 0),
        _bounded_category(atom.GetTotalDegree(), minimum=0, maximum=7),
        _bounded_category(atom.GetFormalCharge(), minimum=-3, maximum=3),
        _bounded_category(atom.GetTotalNumHs(), minimum=0, maximum=4),
        int(atom.GetIsAromatic()),
        int(atom.IsInRing()),
        _HYBRIDIZATION_TO_INDEX.get(str(atom.GetHybridization()), 0),
        _CHIRALITY_TO_INDEX.get(str(atom.GetChiralTag()), 0),
        _bounded_category(atom.GetNumRadicalElectrons(), minimum=0, maximum=3),
    )


def bond_categories(bond: Chem.Bond) -> tuple[int, ...]:
    return (
        _BOND_TYPE_TO_INDEX.get(str(bond.GetBondType()), 0),
        int(bond.GetIsAromatic()),
        int(bond.GetIsConjugated()),
        int(bond.IsInRing()),
        _BOND_STEREO_TO_INDEX.get(str(bond.GetStereo()), 0),
    )


def featurize_explicit_molecule(
    molecule: Chem.Mol,
    *,
    source_atom_count: int,
) -> AllAtomGraph:
    """Featurize an already-expanded molecule without consulting labels."""

    if molecule.GetNumAtoms() < 1:
        raise ValueError("Cannot featurize an empty molecule")
    if not 1 <= int(source_atom_count) <= molecule.GetNumAtoms():
        raise ValueError("Invalid source-atom count")
    atomic_numbers = tuple(atom.GetAtomicNum() for atom in molecule.GetAtoms())
    node_rows = tuple(atom_categories(atom) for atom in molecule.GetAtoms())
    edges: list[tuple[int, int]] = []
    edge_rows: list[tuple[int, ...]] = []
    neighbors: dict[int, list[int]] = {
        index: [] for index in range(molecule.GetNumAtoms())
    }
    for bond in molecule.GetBonds():
        left = int(bond.GetBeginAtomIdx())
        right = int(bond.GetEndAtomIdx())
        features = bond_categories(bond)
        edges.extend(((left, right), (right, left)))
        edge_rows.extend((features, features))
        neighbors[left].append(right)
        neighbors[right].append(left)
    parents = [-1] * molecule.GetNumAtoms()
    for index, atomic_number in enumerate(atomic_numbers):
        if atomic_number != 1:
            continue
        atom_neighbors = sorted(set(neighbors[index]))
        if len(atom_neighbors) != 1:
            raise ValueError(
                f"Hydrogen atom {index} has degree {len(atom_neighbors)}, expected one"
            )
        parent = atom_neighbors[0]
        if atomic_numbers[parent] == 1:
            raise ValueError(f"Hydrogen atom {index} has a hydrogen parent")
        parents[index] = parent
    payload = {
        "feature_schema": "nucpred.mayr-all-atom-rdkit-categorical.v1",
        "atomic_numbers": atomic_numbers,
        "node_categorical": node_rows,
        "directed_edges": edges,
        "edge_categorical": edge_rows,
        "source_atom_count": int(source_atom_count),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return AllAtomGraph(
        atomic_numbers=atomic_numbers,
        node_categorical=node_rows,
        directed_edges=tuple(edges),
        edge_categorical=tuple(edge_rows),
        hydrogen_parent_index=tuple(parents),
        source_atom_count=int(source_atom_count),
        mapping_sha256=digest,
    )


def assert_category_ranges(rows: Sequence[Sequence[int]], sizes: Sequence[int]) -> None:
    if any(len(row) != len(sizes) for row in rows):
        raise ValueError("Categorical feature width changed")
    for row in rows:
        for value, size in zip(row, sizes, strict=True):
            if not 0 <= int(value) < int(size):
                raise ValueError(f"Categorical value {value} is outside size {size}")
