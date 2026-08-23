"""Leak-free simple baselines for joint Mayr site retrieval and N prediction.

The command surface deliberately separates target-blind preparation, outer-fold
training/score freezing, and later evaluation.  ``run-fold`` may read Mayr labels
only for that fold's development targets.  It emits candidate scores for the
outer-test contexts without joining their site or N labels.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from scipy import sparse
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from nucpred.core.files import atomic_write_json, sha256_file
from nucpred.datasets.mayr_site_candidate_policy import (
    SITE_TYPES,
    select_deployment_candidates,
)
from nucpred.project import get_project_layout


ROOT = get_project_layout().root
DEFAULT_CONFIG = ROOT / "configs/mayr_simple_baselines_v1.toml"
CONFIG_SCHEMA = "nucpred.mayr-simple-baselines.v1"
CACHE_SCHEMA = "nucpred.mayr-simple-baseline-feature-cache.v1"
FOLD_SCHEMA = "nucpred.mayr-simple-baseline-fold-scores.v1"
FREEZE_SCHEMA = "nucpred.mayr-simple-baseline-score-freeze.v1"

MODEL_NAMES = (
    "site_type_prior_mean",
    "morgan_1nn",
    "linear",
    "random_forest_standard",
    "hist_gradient_boosting_standard",
    "random_forest_matched_inputs",
    "hist_gradient_boosting_matched_inputs",
)
STANDARD_TREE_MODELS = {
    "random_forest_standard": ("random_forest", False),
    "hist_gradient_boosting_standard": ("hist_gradient_boosting", False),
    "random_forest_matched_inputs": ("random_forest", True),
    "hist_gradient_boosting_matched_inputs": (
        "hist_gradient_boosting",
        True,
    ),
}
RDKIT_DESCRIPTOR_NAMES = tuple(name for name, _ in Descriptors._descList)
RDKIT_DESCRIPTOR_FUNCTIONS = tuple(
    (name, function) for name, function in Descriptors._descList
)
RDKIT_COLUMNS = tuple(f"rdkit_{name}" for name in RDKIT_DESCRIPTOR_NAMES)

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
SITE_NUMERIC_FEATURES = (
    "member_atom_count",
    "member_internal_bond_count",
    "member_atomic_number_mean",
    "member_atomic_number_min",
    "member_atomic_number_max",
    "member_H_count",
    "member_C_count",
    "member_N_count",
    "member_O_count",
    "member_P_count",
    "member_S_count",
    "member_halogen_count",
)

SITE_GRAPH_FEATURES = (
    *SITE_NUMERIC_FEATURES,
    "member_formal_charge_sum",
    "member_degree_mean",
    "member_degree_max",
    "member_aromatic_fraction",
    "member_ring_fraction",
    "boundary_bond_count",
    "radius1_neighbor_count",
    "radius1_neighbor_atomic_number_mean",
)
MATCHED_FEATURES = (
    *(f"local_xtb_{index}_mean" for index in range(4)),
    *(f"local_xtb_{index}_available_fraction" for index in range(4)),
    *(f"global_xtb_{index}" for index in range(6)),
    *(f"global_xtb_{index}_available" for index in range(6)),
)
STANDARD_NUMERIC_FEATURES = (
    *SITE_GRAPH_FEATURES,
    *SOLVENT_FEATURES,
    "model_formal_charge",
)


class SimpleBaselineError(RuntimeError):
    """Raised when a simple-baseline protocol or artifact invariant fails."""


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _project_path(value: object, *, label: str) -> Path:
    raw = Path(str(value))
    path = raw.resolve() if raw.is_absolute() else (ROOT / raw).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise SimpleBaselineError(f"{label} escapes the project root")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SimpleBaselineError(f"Expected JSON object: {path}")
    return payload


def _json_list(value: object, *, label: str) -> list[Any]:
    if isinstance(value, list):
        return value
    parsed = json.loads(str(value))
    if not isinstance(parsed, list):
        raise SimpleBaselineError(f"Expected JSON list for {label}")
    return parsed


def read_config(
    path: str | Path = DEFAULT_CONFIG,
) -> tuple[dict[str, Any], Path]:
    resolved = Path(path).resolve()
    with resolved.open("rb") as handle:
        config = tomllib.load(handle)
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise SimpleBaselineError("Unsupported simple-baseline config schema")
    required_false = (
        "selection_uses_outer_test_results",
        "outer_test_labels_may_be_read_before_fold_scores_are_frozen",
        "unknown_candidates_are_universal_negatives",
        "candidate_softmax_used",
        "independent_external_validation",
    )
    if any(config.get(key) is not False for key in required_false):
        raise SimpleBaselineError("A forbidden protocol flag is enabled")
    if config.get("non_selected_candidates_are_endpoint_relative_ranking_negatives") is not True:
        raise SimpleBaselineError("Endpoint-relative ranking semantics changed")
    if config.get("retrospective_grouped_oof") is not True:
        raise SimpleBaselineError("Retrospective OOF boundary changed")
    if int(config.get("outer_fold_count", -1)) != 5:
        raise SimpleBaselineError("Outer fold count changed")
    if set(config.get("models", {})) != {
        "random_uniform",
        "site_type_prior_mean",
        "morgan_1nn",
        "linear",
        "random_forest",
        "hist_gradient_boosting",
    }:
        raise SimpleBaselineError("Configured baseline family changed")
    execution = config["execution"]
    if (
        execution.get("formal_training_must_be_started_by_user") is not True
        or execution.get("codex_may_start_formal_training") is not False
        or execution.get("stop_after_score_freeze") is not True
        or int(execution.get("physical_gpu_index", -1)) != 0
    ):
        raise SimpleBaselineError("Manual GPU-0 phase gate changed")
    return config, resolved


def verify_bindings(
    config: Mapping[str, Any], config_path: Path
) -> dict[str, dict[str, object]]:
    bindings: dict[str, dict[str, object]] = {
        "config": {
            "path": config_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(config_path),
            "bytes": int(config_path.stat().st_size),
        }
    }
    raw = config["bindings"]
    for key in (
        "dataset_manifest",
        "contexts",
        "targets",
        "species",
        "candidate_sites",
        "outer_membership",
        "candidate_policy_source",
    ):
        path = _project_path(raw[f"{key}_path"], label=f"bindings.{key}_path")
        expected = str(raw[f"{key}_sha256"])
        if not path.is_file():
            raise SimpleBaselineError(f"Missing bound input: {path}")
        observed = sha256_file(path)
        if observed != expected:
            raise SimpleBaselineError(
                f"Bound input drifted for {key}: {observed} != {expected}"
            )
        bindings[key] = {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": observed,
            "bytes": int(path.stat().st_size),
        }
    return bindings


def _single_target_membership(membership: pd.DataFrame) -> pd.DataFrame:
    identities = membership[
        ["target_id", "context_id", "species_id", "connectivity_id"]
    ].drop_duplicates()
    if identities["target_id"].astype(str).duplicated().any():
        raise SimpleBaselineError("Target identity changes across outer folds")
    counts = identities.groupby("context_id")["target_id"].nunique()
    single_contexts = set(counts.loc[counts.eq(1)].index.astype(str))
    selected = identities.loc[
        identities["context_id"].astype(str).isin(single_contexts)
    ].copy()
    return selected.sort_values("target_id", kind="stable").reset_index(drop=True)


def _fold_membership(
    membership: pd.DataFrame,
    eligible: pd.DataFrame,
    outer_fold: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if outer_fold not in range(5):
        raise SimpleBaselineError(f"Invalid outer fold: {outer_fold}")
    eligible_ids = set(eligible["target_id"].astype(str))
    fold = membership.loc[
        membership["outer_fold"].eq(outer_fold)
        & membership["target_id"].astype(str).isin(eligible_ids)
    ].copy()
    development = fold.loc[fold["role"].eq("development")].copy()
    test = fold.loc[fold["role"].eq("test")].copy()
    if (
        not len(development)
        or not len(test)
        or set(development["target_id"].astype(str))
        & set(test["target_id"].astype(str))
    ):
        raise SimpleBaselineError(f"Invalid membership for outer fold {outer_fold}")
    development_groups = set(development["connectivity_id"].astype(str))
    test_groups = set(test["connectivity_id"].astype(str))
    if development_groups & test_groups:
        raise SimpleBaselineError(f"Connectivity leakage in outer fold {outer_fold}")
    return development.reset_index(drop=True), test.reset_index(drop=True)


def audit(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    write: bool = False,
) -> dict[str, object]:
    """Run a target-blind identity and candidate-policy audit."""

    config, resolved = read_config(config_path)
    bindings = verify_bindings(config, resolved)
    membership_path = _project_path(
        config["bindings"]["outer_membership_path"],
        label="outer membership",
    )
    membership = pd.read_csv(membership_path)
    eligible = _single_target_membership(membership)
    species = pd.read_parquet(
        _project_path(config["bindings"]["species_path"], label="species")
    )
    candidates = pd.read_parquet(
        _project_path(
            config["bindings"]["candidate_sites_path"],
            label="candidate sites",
        )
    )
    selected, policy_audit = select_deployment_candidates(candidates, species)
    contexts = pd.read_parquet(
        _project_path(config["bindings"]["contexts_path"], label="contexts"),
        columns=["context_id", "species_id", "connectivity_id", "complete_xtb10"],
    )
    eligible_contexts = contexts.loc[
        contexts["context_id"].astype(str).isin(
            set(eligible["context_id"].astype(str))
        )
    ].copy()
    if len(eligible_contexts) != int(config["expected_context_count"]):
        raise SimpleBaselineError("Single-target context count changed")
    if eligible_contexts["connectivity_id"].nunique() != int(
        config["expected_connectivity_count"]
    ):
        raise SimpleBaselineError("Single-target connectivity count changed")
    candidate_counts = (
        eligible_contexts[["context_id", "species_id"]]
        .merge(
            selected[["species_id", "candidate_site_id"]],
            on="species_id",
            how="left",
            validate="many_to_many",
        )
        .groupby("context_id")
        .size()
    )
    if len(candidate_counts) != len(eligible_contexts) or candidate_counts.lt(1).any():
        raise SimpleBaselineError("At least one eligible context lacks candidates")
    folds = []
    for outer_fold in range(int(config["outer_fold_count"])):
        development, test = _fold_membership(membership, eligible, outer_fold)
        folds.append(
            {
                "outer_fold": outer_fold,
                "development_context_count": int(development["context_id"].nunique()),
                "development_connectivity_count": int(
                    development["connectivity_id"].nunique()
                ),
                "test_context_count": int(test["context_id"].nunique()),
                "test_connectivity_count": int(test["connectivity_id"].nunique()),
            }
        )
    payload: dict[str, object] = {
        "schema_version": "nucpred.mayr-simple-baseline-audit.v1",
        "status": "pass",
        "campaign_id": config["campaign_id"],
        "verified_bindings": bindings,
        "context_count": len(eligible_contexts),
        "connectivity_count": int(eligible_contexts["connectivity_id"].nunique()),
        "deployment_candidate_count": len(selected),
        "candidate_query_count": int(candidate_counts.sum()),
        "minimum_candidates_per_context": int(candidate_counts.min()),
        "maximum_candidates_per_context": int(candidate_counts.max()),
        "complete_xtb10_context_count": int(
            eligible_contexts["complete_xtb10"].astype(bool).sum()
        ),
        "candidate_policy_audit": policy_audit,
        "folds": folds,
        "target_columns_requested": [],
        "target_site_labels_loaded": False,
        "target_n_values_loaded": False,
        "formal_training_started": False,
    }
    if write:
        output = _project_path(config["output_directory"], label="output")
        output.mkdir(parents=True, exist_ok=True)
        atomic_write_json(output / "preflight.json", payload, ensure_ascii=False)
    return payload


def _explicit_h_molecule(smiles: str, expected_atomic_numbers: object) -> Chem.Mol:
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise SimpleBaselineError(f"Cannot parse molecule: {smiles!r}")
    molecule = Chem.AddHs(molecule)
    observed = [atom.GetAtomicNum() for atom in molecule.GetAtoms()]
    expected = [
        int(value)
        for value in _json_list(
            expected_atomic_numbers,
            label="model_atomic_numbers_json",
        )
    ]
    if observed != expected:
        raise SimpleBaselineError("Explicit-H atom order differs from dataset order")
    return molecule


def _fingerprint_binary(
    generator: Any,
    molecule: Chem.Mol,
    *,
    members: Sequence[int] | None = None,
) -> bytes:
    kwargs = {} if members is None else {"fromAtoms": list(map(int, members))}
    fingerprint = generator.GetFingerprint(molecule, **kwargs)
    return bytes(DataStructs.BitVectToBinaryText(fingerprint))


def _molecular_descriptors(molecule: Chem.Mol) -> dict[str, float]:
    values: dict[str, float] = {}
    for name, function in RDKIT_DESCRIPTOR_FUNCTIONS:
        try:
            value = float(function(molecule))
        except Exception:
            value = math.nan
        values[f"rdkit_{name}"] = value if math.isfinite(value) else math.nan
    return values


def _candidate_features(row: Mapping[str, Any], molecule: Chem.Mol) -> dict[str, float]:
    members = tuple(
        sorted(
            set(
                int(value)
                for value in _json_list(
                    row["member_atom_indices_json"],
                    label="member_atom_indices_json",
                )
            )
        )
    )
    if not members or min(members) < 0 or max(members) >= molecule.GetNumAtoms():
        raise SimpleBaselineError("Candidate members are outside the explicit-H graph")
    atoms = [molecule.GetAtomWithIdx(index) for index in members]
    numbers = np.asarray([atom.GetAtomicNum() for atom in atoms], dtype=float)
    declared_numbers = [
        int(value)
        for value in _json_list(
            row["member_atomic_numbers_json"],
            label="member_atomic_numbers_json",
        )
    ]
    if sorted(map(int, numbers.tolist())) != sorted(declared_numbers):
        raise SimpleBaselineError("Candidate member atomic numbers drifted")
    member_set = set(members)
    internal_bonds = 0
    boundary_bonds = 0
    neighbours: set[int] = set()
    for bond in molecule.GetBonds():
        left = int(bond.GetBeginAtomIdx())
        right = int(bond.GetEndAtomIdx())
        if left in member_set and right in member_set:
            internal_bonds += 1
        elif left in member_set or right in member_set:
            boundary_bonds += 1
            neighbours.add(right if left in member_set else left)
    neighbour_numbers = np.asarray(
        [molecule.GetAtomWithIdx(index).GetAtomicNum() for index in sorted(neighbours)],
        dtype=float,
    )
    counts = {number: int(np.sum(numbers == number)) for number in (1, 6, 7, 8, 15, 16)}
    return {
        "member_atom_count": float(len(members)),
        "member_internal_bond_count": float(internal_bonds),
        "member_atomic_number_mean": float(numbers.mean()),
        "member_atomic_number_min": float(numbers.min()),
        "member_atomic_number_max": float(numbers.max()),
        "member_H_count": float(counts[1]),
        "member_C_count": float(counts[6]),
        "member_N_count": float(counts[7]),
        "member_O_count": float(counts[8]),
        "member_P_count": float(counts[15]),
        "member_S_count": float(counts[16]),
        "member_halogen_count": float(
            sum(int(np.sum(numbers == value)) for value in (9, 17, 35, 53))
        ),
        "member_formal_charge_sum": float(sum(atom.GetFormalCharge() for atom in atoms)),
        "member_degree_mean": float(np.mean([atom.GetDegree() for atom in atoms])),
        "member_degree_max": float(max(atom.GetDegree() for atom in atoms)),
        "member_aromatic_fraction": float(np.mean([atom.GetIsAromatic() for atom in atoms])),
        "member_ring_fraction": float(np.mean([atom.IsInRing() for atom in atoms])),
        "boundary_bond_count": float(boundary_bonds),
        "radius1_neighbor_count": float(len(neighbours)),
        "radius1_neighbor_atomic_number_mean": (
            float(neighbour_numbers.mean()) if len(neighbour_numbers) else 0.0
        ),
    }


def _write_parquet(frame: pd.DataFrame, path: Path) -> dict[str, object]:
    frame.to_parquet(path, index=False, engine="pyarrow", compression="zstd")
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "bytes": int(path.stat().st_size),
        "rows": len(frame),
    }


def prepare_feature_cache(
    config_path: str | Path = DEFAULT_CONFIG,
) -> dict[str, object]:
    """Create label-blind molecular and candidate feature caches."""

    config, resolved = read_config(config_path)
    verify_bindings(config, resolved)
    output = _project_path(config["output_directory"], label="output") / "feature_cache"
    summary_path = output / "summary.json"
    if summary_path.is_file():
        summary = _read_json(summary_path)
        if summary.get("status") != "complete":
            raise SimpleBaselineError("Feature cache summary is incomplete")
        for binding in summary["bindings"]:
            path = output / str(binding["path"])
            if not path.is_file() or sha256_file(path) != str(binding["sha256"]):
                raise SimpleBaselineError("Feature cache binding drifted")
        return summary
    if output.exists():
        raise SimpleBaselineError(f"Partial feature cache exists: {output}")
    stale_cache_staging = sorted(output.parent.glob(".simple-feature-cache-*"))
    if stale_cache_staging:
        raise SimpleBaselineError(
            f"Stale feature-cache staging exists: {stale_cache_staging[0]}"
        )

    membership = pd.read_csv(
        _project_path(
            config["bindings"]["outer_membership_path"],
            label="outer membership",
        )
    )
    eligible = _single_target_membership(membership)
    eligible_context_ids = set(eligible["context_id"].astype(str))
    contexts = pd.read_parquet(
        _project_path(config["bindings"]["contexts_path"], label="contexts")
    )
    contexts = contexts.loc[
        contexts["context_id"].astype(str).isin(eligible_context_ids)
    ].copy()
    species = pd.read_parquet(
        _project_path(config["bindings"]["species_path"], label="species")
    )
    candidates = pd.read_parquet(
        _project_path(
            config["bindings"]["candidate_sites_path"],
            label="candidate sites",
        )
    )
    candidates, policy_audit = select_deployment_candidates(candidates, species)
    required_species = set(contexts["species_id"].astype(str))
    species = species.loc[species["species_id"].astype(str).isin(required_species)].copy()
    candidates = candidates.loc[
        candidates["species_id"].astype(str).isin(required_species)
    ].copy()

    settings = config["features"]
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=int(settings["morgan_radius"]),
        fpSize=int(settings["morgan_bits"]),
        includeChirality=bool(settings["morgan_include_chirality"]),
    )
    molecule_cache: dict[str, Chem.Mol] = {}
    species_rows: list[dict[str, object]] = []
    for row in species.sort_values("species_id", kind="stable").to_dict(orient="records"):
        molecule = _explicit_h_molecule(
            str(row["model_canonical_smiles"]),
            row["model_atomic_numbers_json"],
        )
        species_id = str(row["species_id"])
        molecule_cache[species_id] = molecule
        heavy = Chem.RemoveHs(molecule)
        species_rows.append(
            {
                "species_id": species_id,
                "model_canonical_smiles": str(row["model_canonical_smiles"]),
                "molecule_morgan_binary": _fingerprint_binary(generator, heavy),
                **_molecular_descriptors(heavy),
            }
        )
    species_features = pd.DataFrame(species_rows)

    candidate_rows: list[dict[str, object]] = []
    for row in candidates.sort_values(
        ["species_id", "candidate_site_id"], kind="stable"
    ).to_dict(orient="records"):
        molecule = molecule_cache[str(row["species_id"])]
        members = tuple(
            int(value)
            for value in _json_list(
                row["member_atom_indices_json"],
                label="member_atom_indices_json",
            )
        )
        candidate_rows.append(
            {
                "candidate_site_id": str(row["candidate_site_id"]),
                "species_id": str(row["species_id"]),
                "site_type": str(row["site_type"]),
                "member_atom_indices_json": str(row["member_atom_indices_json"]),
                "member_bond_pairs_json": str(row["member_bond_pairs_json"]),
                "local_morgan_binary": _fingerprint_binary(
                    generator,
                    molecule,
                    members=members,
                ),
                **_candidate_features(row, molecule),
            }
        )
    candidate_features = pd.DataFrame(candidate_rows)
    if candidate_features["candidate_site_id"].duplicated().any():
        raise SimpleBaselineError("Candidate feature cache duplicated identities")

    keep_context_columns = [
        "context_id",
        "species_id",
        "connectivity_id",
        "model_formal_charge",
        "node_local4_json",
        "node_local4_available_json",
        "molecule_global6_json",
        "molecule_global6_available_json",
        "complete_xtb10",
        *SOLVENT_FEATURES,
    ]
    context_features = contexts[keep_context_columns].sort_values(
        "context_id", kind="stable"
    )
    if context_features["context_id"].duplicated().any():
        raise SimpleBaselineError("Context feature cache duplicated identities")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".simple-feature-cache-", dir=output.parent))
    try:
        bindings = [
            _write_parquet(species_features, staging / "species_features.parquet"),
            _write_parquet(candidate_features, staging / "candidate_features.parquet"),
            _write_parquet(context_features, staging / "context_features.parquet"),
        ]
        summary: dict[str, object] = {
            "schema_version": CACHE_SCHEMA,
            "status": "complete",
            "campaign_id": config["campaign_id"],
            "context_count": len(context_features),
            "connectivity_count": int(context_features["connectivity_id"].nunique()),
            "species_count": len(species_features),
            "candidate_count": len(candidate_features),
            "candidate_policy_audit": policy_audit,
            "bindings": bindings,
            "target_files_read": [],
            "target_site_labels_loaded": False,
            "target_n_values_loaded": False,
            "config_path": resolved.relative_to(ROOT).as_posix(),
            "config_sha256": sha256_file(resolved),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "created_at_utc": datetime.now(UTC).isoformat(),
        }
        summary["cache_sha256"] = _canonical_sha256(summary)
        atomic_write_json(staging / "summary.json", summary, ensure_ascii=False)
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return summary


def _load_cache(config: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = _project_path(config["output_directory"], label="output") / "feature_cache"
    summary = _read_json(root / "summary.json")
    if summary.get("status") != "complete" or summary.get("schema_version") != CACHE_SCHEMA:
        raise SimpleBaselineError("Feature cache is unavailable")
    for binding in summary["bindings"]:
        path = root / str(binding["path"])
        if sha256_file(path) != str(binding["sha256"]):
            raise SimpleBaselineError("Feature cache drifted")
    return (
        pd.read_parquet(root / "context_features.parquet"),
        pd.read_parquet(root / "candidate_features.parquet"),
        pd.read_parquet(root / "species_features.parquet"),
    )


def _load_development_targets(dataset: Path, target_ids: set[str]) -> pd.DataFrame:
    if not target_ids:
        raise SimpleBaselineError("Development target set is empty")
    columns = [
        "target_id",
        "context_id",
        "species_id",
        "connectivity_id",
        "site_object_id",
        "site_type",
        "N_mean",
    ]
    frame = pd.read_parquet(
        dataset / "targets.parquet",
        columns=columns,
        filters=[("target_id", "in", sorted(target_ids))],
    )
    frame["target_id"] = frame["target_id"].astype(str)
    if set(frame["target_id"]) != target_ids or frame["target_id"].duplicated().any():
        raise SimpleBaselineError("Filtered development labels are incomplete")
    if not np.isfinite(frame["N_mean"].to_numpy(dtype=float)).all():
        raise SimpleBaselineError("Development N labels are non-finite")
    return frame


def _query_frame(
    contexts: pd.DataFrame,
    candidates: pd.DataFrame,
    species: pd.DataFrame,
) -> pd.DataFrame:
    frame = contexts.merge(candidates, on="species_id", how="left", validate="many_to_many")
    frame = frame.merge(species, on="species_id", how="left", validate="many_to_one")
    if frame["candidate_site_id"].isna().any():
        raise SimpleBaselineError("A query context has no candidate")
    frame["query_id"] = (
        frame["context_id"].astype(str)
        + "|"
        + frame["candidate_site_id"].astype(str)
    )
    if frame["query_id"].duplicated().any():
        raise SimpleBaselineError("Query identities are duplicated")
    frame["candidate_count"] = frame.groupby("context_id")["query_id"].transform("size")
    return frame.sort_values(
        ["context_id", "site_type", "candidate_site_id"], kind="stable"
    ).reset_index(drop=True)


def _development_queries(
    frame: pd.DataFrame,
    targets: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_map = targets[["target_id", "context_id", "site_object_id", "N_mean", "site_type"]].copy()
    target_map = target_map.rename(columns={"site_type": "target_site_type"})
    if target_map["context_id"].duplicated().any():
        raise SimpleBaselineError("Development context has multiple targets")
    queries = frame.merge(target_map, on="context_id", how="left", validate="many_to_one")
    queries["is_positive"] = queries["candidate_site_id"].astype(str).eq(
        queries["site_object_id"].astype(str)
    )
    positive_counts = queries.groupby("context_id")["is_positive"].sum()
    if not positive_counts.eq(1).all():
        raise SimpleBaselineError("Development candidate universe loses a target")
    true_sites = queries.loc[queries["is_positive"]].copy()
    if not true_sites["site_type"].astype(str).eq(
        true_sites["target_site_type"].astype(str)
    ).all():
        raise SimpleBaselineError("Target and candidate site types disagree")
    return queries, true_sites


def _ranking_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame["candidate_count"].to_numpy(dtype=int)
    positive = frame["is_positive"].to_numpy(dtype=bool)
    weights = np.where(positive, 1.0, 1.0 / np.maximum(counts - 1, 1))
    weights[counts <= 1] = 0.0
    return weights.astype(float)


def _bit_vectors(values: Sequence[object]) -> list[Any]:
    return [DataStructs.CreateFromBinaryText(bytes(value)) for value in values]


def _fingerprint_sparse(values: Sequence[object], n_bits: int) -> sparse.csr_matrix:
    indices: list[int] = []
    indptr = [0]
    for fingerprint in _bit_vectors(values):
        indices.extend(map(int, fingerprint.GetOnBits()))
        indptr.append(len(indices))
    data = np.ones(len(indices), dtype=np.float32)
    return sparse.csr_matrix(
        (data, np.asarray(indices, dtype=np.int32), np.asarray(indptr, dtype=np.int64)),
        shape=(len(values), n_bits),
        dtype=np.float32,
    )


def _site_type_matrix(frame: pd.DataFrame) -> sparse.csr_matrix:
    order = {value: index for index, value in enumerate(SITE_TYPES)}
    indices = frame["site_type"].astype(str).map(order)
    if indices.isna().any():
        raise SimpleBaselineError("Unknown site type in model input")
    rows = np.arange(len(frame), dtype=np.int32)
    columns = indices.to_numpy(dtype=np.int32)
    data = np.ones(len(frame), dtype=np.float32)
    return sparse.csr_matrix(
        (data, (rows, columns)), shape=(len(frame), len(SITE_TYPES))
    )


def _scaled_numeric(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: Sequence[str],
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_values = imputer.fit_transform(train[list(columns)])
    test_values = imputer.transform(test[list(columns)])
    train_values = scaler.fit_transform(train_values).astype(np.float32)
    test_values = scaler.transform(test_values).astype(np.float32)
    return sparse.csr_matrix(train_values), sparse.csr_matrix(test_values)


def _linear_matrices(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    n_bits: int,
    include_molecule: bool,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    train_numeric, test_numeric = _scaled_numeric(
        train, test, STANDARD_NUMERIC_FEATURES
    )
    train_parts = [
        _fingerprint_sparse(train["local_morgan_binary"].tolist(), n_bits),
        _site_type_matrix(train),
        train_numeric,
    ]
    test_parts = [
        _fingerprint_sparse(test["local_morgan_binary"].tolist(), n_bits),
        _site_type_matrix(test),
        test_numeric,
    ]
    if include_molecule:
        train_parts.insert(
            1,
            _fingerprint_sparse(train["molecule_morgan_binary"].tolist(), n_bits),
        )
        test_parts.insert(
            1,
            _fingerprint_sparse(test["molecule_morgan_binary"].tolist(), n_bits),
        )
    return (
        sparse.hstack(train_parts, format="csr", dtype=np.float32),
        sparse.hstack(test_parts, format="csr", dtype=np.float32),
    )


def _matched_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    local_means = np.full((len(output), 4), np.nan, dtype=float)
    local_fraction = np.zeros((len(output), 4), dtype=float)
    global_values = np.full((len(output), 6), np.nan, dtype=float)
    global_available = np.zeros((len(output), 6), dtype=float)
    context_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for position, row in enumerate(output.itertuples(index=False)):
        context_id = str(row.context_id)
        cached = context_cache.get(context_id)
        if cached is None:
            local = np.asarray(json.loads(row.node_local4_json), dtype=float)
            available = np.asarray(json.loads(row.node_local4_available_json), dtype=bool)
            global_value = np.asarray(json.loads(row.molecule_global6_json), dtype=float)
            global_mask = np.asarray(
                json.loads(row.molecule_global6_available_json), dtype=bool
            )
            if local.ndim != 2 or local.shape[1] != 4 or available.shape != local.shape:
                raise SimpleBaselineError("Local xTB feature shape changed")
            if global_value.shape != (6,) or global_mask.shape != (6,):
                raise SimpleBaselineError("Global xTB feature shape changed")
            cached = (local, available, global_value, global_mask)
            context_cache[context_id] = cached
        local, available, global_value, global_mask = cached
        members = np.asarray(
            json.loads(row.member_atom_indices_json), dtype=int
        )
        if members.size == 0 or members.min() < 0 or members.max() >= len(local):
            raise SimpleBaselineError("Matched feature members are invalid")
        selected_values = local[members]
        selected_mask = available[members]
        for channel in range(4):
            mask = selected_mask[:, channel]
            local_fraction[position, channel] = float(mask.mean())
            if mask.any():
                local_means[position, channel] = float(
                    selected_values[mask, channel].mean()
                )
        global_values[position] = np.where(global_mask, global_value, np.nan)
        global_available[position] = global_mask.astype(float)
    for index in range(4):
        output[f"local_xtb_{index}_mean"] = local_means[:, index]
        output[f"local_xtb_{index}_available_fraction"] = local_fraction[:, index]
    for index in range(6):
        output[f"global_xtb_{index}"] = global_values[:, index]
        output[f"global_xtb_{index}_available"] = global_available[:, index]
    return output


def _dense_matrices(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    matched_inputs: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if matched_inputs:
        train = _matched_features(train)
        test = _matched_features(test)
    numeric = [*RDKIT_COLUMNS, *STANDARD_NUMERIC_FEATURES]
    if matched_inputs:
        numeric.extend(MATCHED_FEATURES)
    imputer = SimpleImputer(strategy="median")
    train_values = imputer.fit_transform(train[numeric]).astype(np.float32)
    test_values = imputer.transform(test[numeric]).astype(np.float32)
    train_type = _site_type_matrix(train).toarray().astype(np.float32)
    test_type = _site_type_matrix(test).toarray().astype(np.float32)
    return (
        np.concatenate([train_values, train_type], axis=1),
        np.concatenate([test_values, test_type], axis=1),
    )


def _base_score_frame(test: pd.DataFrame, model_name: str, feature_tier: str) -> pd.DataFrame:
    output = test[
        [
            "query_id",
            "context_id",
            "species_id",
            "connectivity_id",
            "candidate_site_id",
            "site_type",
            "candidate_count",
        ]
    ].copy()
    output["model_name"] = model_name
    output["feature_tier"] = feature_tier
    output["site_neighbor_target_id"] = ""
    output["n_neighbor_target_id"] = ""
    return output


def _prior_scores(
    train_true: pd.DataFrame,
    test: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    alpha = float(config["models"]["site_type_prior_mean"]["laplace_alpha"])
    counts = train_true["site_type"].astype(str).value_counts().to_dict()
    denominator = len(train_true) + alpha * len(SITE_TYPES)
    priors = {
        site_type: (float(counts.get(site_type, 0)) + alpha) / denominator
        for site_type in SITE_TYPES
    }
    output = _base_score_frame(test, "site_type_prior_mean", "standard_public")
    output["site_score"] = test["site_type"].astype(str).map(priors).map(math.log)
    output["N_pred"] = float(train_true["N_mean"].mean())
    return output


def _nearest_neighbor_scores(
    train_true: pd.DataFrame,
    test: pd.DataFrame,
) -> pd.DataFrame:
    output = _base_score_frame(test, "morgan_1nn", "standard_public")
    site_scores = np.full(len(test), -1.0, dtype=float)
    n_predictions = np.full(len(test), float(train_true["N_mean"].mean()), dtype=float)
    site_neighbors = np.full(len(test), "", dtype=object)
    n_neighbors = np.full(len(test), "", dtype=object)
    for site_type in SITE_TYPES:
        train_mask = train_true["site_type"].astype(str).eq(site_type).to_numpy()
        test_positions = np.flatnonzero(
            test["site_type"].astype(str).eq(site_type).to_numpy()
        )
        if not train_mask.any() or not len(test_positions):
            continue
        train_selected = train_true.loc[train_mask].sort_values(
            "target_id", kind="stable"
        )
        local_train = _bit_vectors(train_selected["local_morgan_binary"].tolist())
        molecule_train = _bit_vectors(
            train_selected["molecule_morgan_binary"].tolist()
        )
        target_ids = train_selected["target_id"].astype(str).tolist()
        target_values = train_selected["N_mean"].to_numpy(dtype=float)
        for position in test_positions:
            local = DataStructs.CreateFromBinaryText(
                bytes(test.iloc[position]["local_morgan_binary"])
            )
            similarities = np.asarray(
                DataStructs.BulkTanimotoSimilarity(local, local_train), dtype=float
            )
            best = int(np.argmax(similarities))
            site_scores[position] = float(similarities[best])
            site_neighbors[position] = target_ids[best]
            molecule = DataStructs.CreateFromBinaryText(
                bytes(test.iloc[position]["molecule_morgan_binary"])
            )
            molecule_similarities = np.asarray(
                DataStructs.BulkTanimotoSimilarity(molecule, molecule_train),
                dtype=float,
            )
            best_n = int(np.argmax(molecule_similarities))
            n_predictions[position] = float(target_values[best_n])
            n_neighbors[position] = target_ids[best_n]
    output["site_score"] = site_scores
    output["N_pred"] = n_predictions
    output["site_neighbor_target_id"] = site_neighbors
    output["n_neighbor_target_id"] = n_neighbors
    return output


def _linear_scores(
    train_queries: pd.DataFrame,
    train_true: pd.DataFrame,
    test: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    settings = config["models"]["linear"]
    n_bits = int(config["features"]["morgan_bits"])
    eligible = train_queries["candidate_count"].gt(1)
    rank_train = train_queries.loc[eligible].reset_index(drop=True)
    site_train, site_test = _linear_matrices(
        rank_train, test, n_bits=n_bits, include_molecule=False
    )
    classifier = LogisticRegression(
        C=float(settings["logistic_c"]),
        solver=str(settings["logistic_solver"]),
        max_iter=int(settings["logistic_max_iter"]),
        random_state=2026081901,
    )
    classifier.fit(
        site_train,
        rank_train["is_positive"].astype(int).to_numpy(),
        sample_weight=_ranking_weights(rank_train),
    )
    n_train, n_test = _linear_matrices(
        train_true, test, n_bits=n_bits, include_molecule=True
    )
    regressor = Ridge(alpha=float(settings["ridge_alpha"]), solver="lsqr")
    regressor.fit(n_train, train_true["N_mean"].to_numpy(dtype=float))
    output = _base_score_frame(test, "linear", "standard_public")
    output["site_score"] = classifier.decision_function(site_test).astype(float)
    output["N_pred"] = regressor.predict(n_test).astype(float)
    return output


def _tree_estimators(
    family: str,
    *,
    outer_fold: int,
    config: Mapping[str, Any],
) -> tuple[Any, Any]:
    if family == "random_forest":
        settings = config["models"]["random_forest"]
        seed = int(settings["seed_base"]) + outer_fold
        common = {
            "n_estimators": int(settings["n_estimators"]),
            "min_samples_leaf": int(settings["min_samples_leaf"]),
            "max_features": str(settings["max_features"]),
            "n_jobs": int(settings["n_jobs_per_fold"]),
            "random_state": seed,
        }
        return RandomForestClassifier(**common), RandomForestRegressor(**common)
    if family == "hist_gradient_boosting":
        settings = config["models"]["hist_gradient_boosting"]
        seed = int(settings["seed_base"]) + outer_fold
        common = {
            "max_iter": int(settings["max_iter"]),
            "learning_rate": float(settings["learning_rate"]),
            "l2_regularization": float(settings["l2_regularization"]),
            "random_state": seed,
        }
        return HistGradientBoostingClassifier(**common), HistGradientBoostingRegressor(**common)
    raise SimpleBaselineError(f"Unsupported tree family: {family}")


def _tree_scores(
    model_name: str,
    train_queries: pd.DataFrame,
    train_true: pd.DataFrame,
    test: pd.DataFrame,
    *,
    outer_fold: int,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    family, matched_inputs = STANDARD_TREE_MODELS[model_name]
    eligible = train_queries["candidate_count"].gt(1)
    rank_train = train_queries.loc[eligible].reset_index(drop=True)
    site_train, site_test = _dense_matrices(
        rank_train, test, matched_inputs=matched_inputs
    )
    n_train, n_test = _dense_matrices(
        train_true, test, matched_inputs=matched_inputs
    )
    classifier, regressor = _tree_estimators(
        family, outer_fold=outer_fold, config=config
    )
    classifier.fit(
        site_train,
        rank_train["is_positive"].astype(int).to_numpy(),
        sample_weight=_ranking_weights(rank_train),
    )
    regressor.fit(n_train, train_true["N_mean"].to_numpy(dtype=float))
    output = _base_score_frame(
        test,
        model_name,
        "matched_nonlearned" if matched_inputs else "standard_public",
    )
    output["site_score"] = classifier.predict_proba(site_test)[:, 1].astype(float)
    output["N_pred"] = regressor.predict(n_test).astype(float)
    return output


def run_fold(
    outer_fold: int,
    config_path: str | Path = DEFAULT_CONFIG,
) -> dict[str, object]:
    """Fit on one outer-development partition and freeze unlabeled test scores."""

    config, resolved = read_config(config_path)
    verify_bindings(config, resolved)
    root = _project_path(config["output_directory"], label="output") / "score_freeze"
    target = root / f"outer-{outer_fold}"
    summary_path = target / "summary.json"
    if summary_path.is_file():
        summary = _read_json(summary_path)
        binding = summary["score_binding"]
        score_path = target / str(binding["path"])
        if summary.get("status") == "complete" and score_path.is_file() and sha256_file(
            score_path
        ) == str(binding["sha256"]):
            return summary
        raise SimpleBaselineError(f"Existing fold is invalid: {target}")
    if target.exists():
        raise SimpleBaselineError(f"Partial fold output exists: {target}")
    stale_fold_staging = sorted(root.glob(f".outer-{outer_fold}-*"))
    if stale_fold_staging:
        raise SimpleBaselineError(
            f"Stale fold staging exists: {stale_fold_staging[0]}"
        )
    contexts, candidates, species = _load_cache(config)
    membership = pd.read_csv(
        _project_path(
            config["bindings"]["outer_membership_path"],
            label="outer membership",
        )
    )
    eligible = _single_target_membership(membership)
    development_membership, test_membership = _fold_membership(
        membership, eligible, outer_fold
    )
    development_ids = set(development_membership["target_id"].astype(str))
    dataset = _project_path(config["dataset_directory"], label="dataset")
    development_targets = _load_development_targets(dataset, development_ids)

    development_context_ids = set(development_membership["context_id"].astype(str))
    test_context_ids = set(test_membership["context_id"].astype(str))
    development_contexts = contexts.loc[
        contexts["context_id"].astype(str).isin(development_context_ids)
    ].copy()
    test_contexts = contexts.loc[
        contexts["context_id"].astype(str).isin(test_context_ids)
    ].copy()
    if set(development_contexts["context_id"].astype(str)) != development_context_ids:
        raise SimpleBaselineError("Development context cache is incomplete")
    if set(test_contexts["context_id"].astype(str)) != test_context_ids:
        raise SimpleBaselineError("Test context cache is incomplete")

    development_queries = _query_frame(development_contexts, candidates, species)
    test_queries = _query_frame(test_contexts, candidates, species)
    development_queries, development_true = _development_queries(
        development_queries, development_targets
    )

    started = time.monotonic()
    scores = [
        _prior_scores(development_true, test_queries, config),
        _nearest_neighbor_scores(development_true, test_queries),
        _linear_scores(development_queries, development_true, test_queries, config),
    ]
    for model_name in STANDARD_TREE_MODELS:
        scores.append(
            _tree_scores(
                model_name,
                development_queries,
                development_true,
                test_queries,
                outer_fold=outer_fold,
                config=config,
            )
        )
    score_frame = pd.concat(scores, ignore_index=True)
    score_frame.insert(0, "outer_fold", outer_fold)
    if set(score_frame["model_name"].astype(str)) != set(MODEL_NAMES):
        raise SimpleBaselineError("Fold score package loses a baseline")
    if score_frame[["site_score", "N_pred"]].isna().any().any():
        raise SimpleBaselineError("Fold scores contain missing values")
    expected_rows = len(test_queries) * len(MODEL_NAMES)
    if len(score_frame) != expected_rows:
        raise SimpleBaselineError("Fold score row count changed")
    if score_frame.duplicated(["model_name", "query_id"]).any():
        raise SimpleBaselineError("Fold scores duplicate model-query identities")

    root.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".outer-{outer_fold}-", dir=root)
    )
    try:
        score_path = staging / "baseline_candidate_scores.parquet"
        binding = _write_parquet(score_frame, score_path)
        summary: dict[str, object] = {
            "schema_version": FOLD_SCHEMA,
            "status": "complete",
            "campaign_id": config["campaign_id"],
            "outer_fold": outer_fold,
            "model_names": list(MODEL_NAMES),
            "development_context_count": len(development_contexts),
            "development_connectivity_count": int(
                development_contexts["connectivity_id"].nunique()
            ),
            "test_context_count": len(test_contexts),
            "test_connectivity_count": int(
                test_contexts["connectivity_id"].nunique()
            ),
            "test_candidate_query_count": len(test_queries),
            "development_target_id_sha256": _canonical_sha256(
                sorted(development_ids)
            ),
            "test_context_id_sha256": _canonical_sha256(
                sorted(test_context_ids)
            ),
            "score_binding": binding,
            "development_label_columns_requested": [
                "target_id",
                "context_id",
                "species_id",
                "connectivity_id",
                "site_object_id",
                "site_type",
                "N_mean",
            ],
            "outer_test_label_columns_requested": [],
            "outer_test_site_labels_loaded": False,
            "outer_test_n_values_loaded": False,
            "metrics_computed": False,
            "selection_uses_outer_test_results": False,
            "candidate_softmax_used": False,
            "training_seconds": time.monotonic() - started,
            "config_path": resolved.relative_to(ROOT).as_posix(),
            "config_sha256": sha256_file(resolved),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "completed_at_utc": datetime.now(UTC).isoformat(),
        }
        summary["fold_sha256"] = _canonical_sha256(summary)
        atomic_write_json(staging / "summary.json", summary, ensure_ascii=False)
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return summary


def assemble_score_freeze(
    config_path: str | Path = DEFAULT_CONFIG,
) -> dict[str, object]:
    """Verify all fold scores without loading any outer-test labels."""

    config, resolved = read_config(config_path)
    root = _project_path(config["output_directory"], label="output") / "score_freeze"
    summaries = []
    score_bindings = []
    context_ids_by_model = {model: set() for model in MODEL_NAMES}
    connectivity_ids: set[str] = set()
    for outer_fold in range(int(config["outer_fold_count"])):
        fold_root = root / f"outer-{outer_fold}"
        summary = _read_json(fold_root / "summary.json")
        if summary.get("status") != "complete" or summary.get("schema_version") != FOLD_SCHEMA:
            raise SimpleBaselineError(f"Fold {outer_fold} is incomplete")
        binding = summary["score_binding"]
        score_path = fold_root / str(binding["path"])
        if sha256_file(score_path) != str(binding["sha256"]):
            raise SimpleBaselineError(f"Fold {outer_fold} scores drifted")
        scores = pd.read_parquet(
            score_path,
            columns=["model_name", "context_id", "connectivity_id", "query_id"],
        )
        if scores.duplicated(["model_name", "query_id"]).any():
            raise SimpleBaselineError(f"Fold {outer_fold} duplicates queries")
        for model_name, group in scores.groupby("model_name"):
            contexts = set(group["context_id"].astype(str))
            if context_ids_by_model[str(model_name)] & contexts:
                raise SimpleBaselineError("A context appears in multiple outer tests")
            context_ids_by_model[str(model_name)].update(contexts)
        connectivity_ids.update(scores["connectivity_id"].astype(str))
        summaries.append(summary)
        score_bindings.append(
            {
                "outer_fold": outer_fold,
                "path": (
                    f"outer-{outer_fold}/{binding['path']}"
                ),
                "sha256": binding["sha256"],
                "bytes": binding["bytes"],
                "rows": binding["rows"],
            }
        )
    expected_contexts = int(config["expected_context_count"])
    if any(len(values) != expected_contexts for values in context_ids_by_model.values()):
        raise SimpleBaselineError("OOF context coverage changed")
    if len(connectivity_ids) != int(config["expected_connectivity_count"]):
        raise SimpleBaselineError("OOF connectivity coverage changed")
    payload: dict[str, object] = {
        "schema_version": FREEZE_SCHEMA,
        "status": "complete",
        "campaign_id": config["campaign_id"],
        "model_names": list(MODEL_NAMES),
        "analytic_reference_models": ["random_uniform"],
        "context_count": expected_contexts,
        "connectivity_count": len(connectivity_ids),
        "folds": summaries,
        "score_bindings": score_bindings,
        "all_outer_scores_frozen_before_any_outer_test_label_read": True,
        "outer_test_label_columns_requested": [],
        "outer_test_site_labels_loaded": False,
        "outer_test_n_values_loaded": False,
        "metrics_computed": False,
        "selection_uses_outer_test_results": False,
        "evaluation_requires_user_resume": True,
        "config_path": resolved.relative_to(ROOT).as_posix(),
        "config_sha256": sha256_file(resolved),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "frozen_at_utc": datetime.now(UTC).isoformat(),
    }
    payload["freeze_sha256"] = _canonical_sha256(payload)
    atomic_write_json(root / "summary.json", payload, ensure_ascii=False)
    return payload


def _tail(path: Path, lines: int = 60) -> str:
    if not path.is_file():
        return "<missing log>"
    return "\n".join(
        path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
    )


def run_all(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    workers: int | None = None,
) -> dict[str, object]:
    """Manual-only bounded outer-fold launcher; stops after score freezing."""

    config, resolved = read_config(config_path)
    expected_device = str(config["execution"]["physical_gpu_index"])
    if os.environ.get("CUDA_VISIBLE_DEVICES") != expected_device:
        raise SimpleBaselineError(
            f"CUDA_VISIBLE_DEVICES must be exactly {expected_device!r}"
        )
    audit(resolved, write=True)
    prepare_feature_cache(resolved)
    maximum = int(
        workers
        if workers is not None
        else config["execution"]["parallel_outer_folds"]
    )
    if maximum < 1 or maximum > int(config["outer_fold_count"]):
        raise SimpleBaselineError("Invalid outer-fold worker count")
    root = _project_path(config["output_directory"], label="output")
    log_root = root / "manual_training_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    pending = list(range(int(config["outer_fold_count"])))
    active: dict[int, tuple[subprocess.Popen[bytes], Any, Path, float]] = {}
    while pending or active:
        while pending and len(active) < maximum:
            outer_fold = pending.pop(0)
            summary_path = root / "score_freeze" / f"outer-{outer_fold}" / "summary.json"
            if summary_path.is_file():
                print(f"already complete: outer-{outer_fold}", flush=True)
                continue
            log_path = log_root / f"outer-{outer_fold}.log"
            handle = log_path.open("wb")
            command = (
                sys.executable,
                "-m",
                "nucpred.publication.mayr_simple_baselines",
                "--config",
                resolved.relative_to(ROOT).as_posix(),
                "run-fold",
                "--outer-fold",
                str(outer_fold),
            )
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=os.environ.copy(),
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            active[outer_fold] = (process, handle, log_path, time.monotonic())
            print(f"started outer-{outer_fold}: pid={process.pid}", flush=True)
        if not active:
            continue
        time.sleep(2.0)
        failures = []
        for outer_fold, item in list(active.items()):
            process, handle, log_path, started = item
            returncode = process.poll()
            if returncode is None:
                continue
            handle.close()
            del active[outer_fold]
            if returncode:
                failures.append((outer_fold, int(returncode), log_path))
            else:
                print(
                    f"completed outer-{outer_fold} in {time.monotonic() - started:.1f}s",
                    flush=True,
                )
        if failures:
            for process, handle, _path, _started in active.values():
                process.terminate()
                handle.close()
            for outer_fold, returncode, log_path in failures:
                print(
                    f"outer-{outer_fold} failed ({returncode})\n{_tail(log_path)}",
                    file=sys.stderr,
                )
            raise SimpleBaselineError("A parallel outer-fold job failed")
    return assemble_score_freeze(resolved)


def dry_run(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, object]:
    """Validate the immutable plan without creating caches or fitting models."""

    config, resolved = read_config(config_path)
    preflight = audit(resolved, write=False)
    commands = [
        [
            sys.executable,
            "-m",
            "nucpred.publication.mayr_simple_baselines",
            "--config",
            resolved.relative_to(ROOT).as_posix(),
            "run-fold",
            "--outer-fold",
            str(outer_fold),
        ]
        for outer_fold in range(int(config["outer_fold_count"]))
    ]
    return {
        "schema_version": "nucpred.mayr-simple-baseline-dry-run.v1",
        "status": "ready",
        "formal_training_started": False,
        "physical_gpu_index": int(config["execution"]["physical_gpu_index"]),
        "parallel_outer_folds": int(config["execution"]["parallel_outer_folds"]),
        "fold_jobs_are_cpu_only_sklearn": bool(
            config["execution"]["fold_jobs_are_cpu_only_sklearn"]
        ),
        "preflight": preflight,
        "planned_fold_commands": commands,
        "stop_after_score_freeze": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("audit")
    subparsers.add_parser("prepare-cache")
    fold = subparsers.add_parser("run-fold")
    fold.add_argument("--outer-fold", type=int, required=True)
    subparsers.add_parser("assemble")
    run = subparsers.add_parser("run-all")
    run.add_argument("--workers", type=int)
    subparsers.add_parser("dry-run")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "audit":
        payload = audit(args.config, write=True)
    elif args.command == "prepare-cache":
        payload = prepare_feature_cache(args.config)
    elif args.command == "run-fold":
        payload = run_fold(args.outer_fold, args.config)
    elif args.command == "assemble":
        payload = assemble_score_freeze(args.config)
    elif args.command == "run-all":
        payload = run_all(args.config, workers=args.workers)
    else:
        payload = dry_run(args.config)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
