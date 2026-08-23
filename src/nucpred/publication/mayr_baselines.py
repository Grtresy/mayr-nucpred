"""Leak-free outer-fold baselines for the Mayr N publication campaign."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import tomllib
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from nucpred.core.files import atomic_write_json, sha256_file
from nucpred.features.descriptors import RDKitDescriptorTransformer
from nucpred.project import get_project_layout
from nucpred.training.mayr_node_xtb_scratch import SOLVENT_FEATURES


ROOT = get_project_layout().root
DEFAULT_CONFIG = ROOT / "configs/mayr_n_publication_baselines_v1.toml"
DEFAULT_COMPARATORS = {
    "oracle_summary_path": (
        "artifacts/campaigns/mayr-n-publication-20260805-v1/modeling/"
        "oracle_evaluation/summary.json"
    ),
    "oracle_predictions_path": (
        "artifacts/campaigns/mayr-n-publication-20260805-v1/modeling/"
        "oracle_evaluation/oracle_oof_predictions_with_labels.parquet"
    ),
    "automatic_summary_path": (
        "artifacts/campaigns/mayr-n-publication-20260805-v1/modeling/"
        "automatic_site/outer_evaluation/summary.json"
    ),
}
SCHEMA = "nucpred.mayr-n-publication-baseline-scores.v1"
MODEL_NAMES = (
    "train_mean",
    "one_nearest_neighbor_morgan",
    "random_forest_standard_descriptors",
    "hist_gradient_boosting_standard_descriptors",
)
SITE_TYPES = (
    "atom",
    "atom_group",
    "bond",
    "delocalized_region",
    "transferable_h_group",
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


class PublicationBaselineError(RuntimeError):
    """Raised when a publication-baseline invariant is violated."""


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PublicationBaselineError(f"Expected JSON object: {path}")
    return payload


def _canonical_sha256(payload: object) -> str:
    import hashlib

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_config(path: str | Path = DEFAULT_CONFIG) -> tuple[dict[str, Any], Path]:
    resolved = Path(path).resolve()
    with resolved.open("rb") as handle:
        config = tomllib.load(handle)
    if config.get("schema_version") != "nucpred.mayr-n-publication-baselines.v1":
        raise PublicationBaselineError("Unsupported baseline configuration schema")
    if config.get("selection_uses_outer_test_results") is not False:
        raise PublicationBaselineError(
            "Baseline configuration permits test-set selection"
        )
    if (
        config.get("outer_test_labels_may_be_read_before_all_scores_frozen")
        is not False
    ):
        raise PublicationBaselineError(
            "Baseline configuration permits early test-label access"
        )
    return config, resolved


def _project_path(value: str, *, label: str) -> Path:
    path = (ROOT / value).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as exc:
        raise PublicationBaselineError(f"{label} escapes the project root") from exc
    return path


def _comparator_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    raw = config.get("comparators", DEFAULT_COMPARATORS)
    if not isinstance(raw, Mapping):
        raise PublicationBaselineError("Baseline comparator paths are invalid")
    if set(raw) != set(DEFAULT_COMPARATORS):
        raise PublicationBaselineError("Baseline comparator path set changed")
    return {
        key: _project_path(str(raw[key]), label=f"comparators.{key}")
        for key in DEFAULT_COMPARATORS
    }


def _parse_json_list(value: object, *, label: str) -> list[Any]:
    parsed = json.loads(str(value))
    if not isinstance(parsed, list):
        raise PublicationBaselineError(f"Expected JSON list for {label}")
    return parsed


def _site_features(row: pd.Series) -> dict[str, float]:
    members = [
        int(value)
        for value in _parse_json_list(
            row["member_atom_indices_json"],
            label="member_atom_indices_json",
        )
    ]
    atomic_numbers = [
        int(value)
        for value in _parse_json_list(
            row["model_atomic_numbers_json"],
            label="model_atomic_numbers_json",
        )
    ]
    if not members or any(
        index < 0 or index >= len(atomic_numbers) for index in members
    ):
        raise PublicationBaselineError(f"Invalid site members for {row['target_id']}")
    member_numbers = np.asarray(
        [atomic_numbers[index] for index in members], dtype=float
    )
    member_set = set(members)
    directed_edges = _parse_json_list(
        row["model_directed_edges_json"],
        label="model_directed_edges_json",
    )
    internal_pairs = {
        tuple(sorted((int(edge[0]), int(edge[1]))))
        for edge in directed_edges
        if len(edge) == 2
        and int(edge[0]) in member_set
        and int(edge[1]) in member_set
        and int(edge[0]) != int(edge[1])
    }
    counts = {
        number: int(np.sum(member_numbers == number)) for number in (1, 6, 7, 8, 15, 16)
    }
    return {
        "member_atom_count": float(len(members)),
        "member_internal_bond_count": float(len(internal_pairs)),
        "member_atomic_number_mean": float(member_numbers.mean()),
        "member_atomic_number_min": float(member_numbers.min()),
        "member_atomic_number_max": float(member_numbers.max()),
        "member_H_count": float(counts[1]),
        "member_C_count": float(counts[6]),
        "member_N_count": float(counts[7]),
        "member_O_count": float(counts[8]),
        "member_P_count": float(counts[15]),
        "member_S_count": float(counts[16]),
        "member_halogen_count": float(
            sum(int(np.sum(member_numbers == number)) for number in (9, 17, 35, 53))
        ),
    }


def load_label_blind_features(dataset: Path) -> pd.DataFrame:
    """Load target metadata without requesting any N target column."""

    target_columns = [
        "target_id",
        "context_id",
        "species_id",
        "connectivity_id",
        "site_type",
        "member_atom_indices_json",
        "model_canonical_smiles",
        "model_formal_charge",
        "solvent_raw",
    ]
    context_columns = [
        "context_id",
        "model_atomic_numbers_json",
        "model_directed_edges_json",
        *SOLVENT_FEATURES,
    ]
    targets = pd.read_parquet(dataset / "targets.parquet", columns=target_columns)
    contexts = pd.read_parquet(dataset / "contexts.parquet", columns=context_columns)
    frame = targets.merge(contexts, on="context_id", how="left", validate="many_to_one")
    if frame[list(context_columns[1:])].isna().all(axis=1).any():
        raise PublicationBaselineError(
            "Some target contexts lack label-independent features"
        )
    if not set(frame["site_type"].astype(str)).issubset(SITE_TYPES):
        raise PublicationBaselineError("Unknown site type in baseline input")
    site_frame = pd.DataFrame([_site_features(row) for _, row in frame.iterrows()])
    return pd.concat([frame.reset_index(drop=True), site_frame], axis=1)


def _load_development_labels(dataset: Path, target_ids: set[str]) -> pd.DataFrame:
    if not target_ids:
        raise PublicationBaselineError("Development target set cannot be empty")
    labels = pd.read_parquet(
        dataset / "targets.parquet",
        columns=["target_id", "N_mean"],
        filters=[("target_id", "in", sorted(target_ids))],
    )
    labels["target_id"] = labels["target_id"].astype(str)
    if set(labels["target_id"]) != target_ids or labels["target_id"].duplicated().any():
        raise PublicationBaselineError(
            "Filtered development label read is incomplete or duplicated"
        )
    if not np.isfinite(labels["N_mean"].to_numpy(dtype=float)).all():
        raise PublicationBaselineError("Development labels contain non-finite values")
    return labels


class SiteTypeMatchedMorgan1NN:
    """One-nearest-neighbor Morgan baseline, matched by known site type."""

    def __init__(self, *, radius: int, n_bits: int, include_chirality: bool):
        self.radius = radius
        self.n_bits = n_bits
        self.include_chirality = include_chirality
        self._generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius,
            fpSize=n_bits,
            includeChirality=include_chirality,
        )

    def _fingerprint(self, smiles: object):
        molecule = Chem.MolFromSmiles(str(smiles))
        return None if molecule is None else self._generator.GetFingerprint(molecule)

    def fit(
        self, features: pd.DataFrame, target: np.ndarray
    ) -> "SiteTypeMatchedMorgan1NN":
        self.target_ = np.asarray(target, dtype=float)
        self.site_types_ = features["site_type"].astype(str).to_numpy()
        self.target_ids_ = features["target_id"].astype(str).to_numpy()
        self.fingerprints_ = [
            self._fingerprint(value) for value in features["model_canonical_smiles"]
        ]
        self.global_mean_ = float(np.mean(self.target_))
        if not any(value is not None for value in self.fingerprints_):
            raise PublicationBaselineError("No valid development Morgan fingerprints")
        return self

    def predict(
        self, features: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        predictions: list[float] = []
        similarities: list[float] = []
        neighbor_ids: list[str] = []
        for _, row in features.iterrows():
            fingerprint = self._fingerprint(row["model_canonical_smiles"])
            positions = [
                index
                for index, (candidate, site_type) in enumerate(
                    zip(self.fingerprints_, self.site_types_, strict=True)
                )
                if candidate is not None and site_type == str(row["site_type"])
            ]
            if fingerprint is None or not positions:
                predictions.append(self.global_mean_)
                similarities.append(math.nan)
                neighbor_ids.append("")
                continue
            candidate_fingerprints = [self.fingerprints_[index] for index in positions]
            scores = DataStructs.BulkTanimotoSimilarity(
                fingerprint, candidate_fingerprints
            )
            best_score = max(float(value) for value in scores)
            tied = [
                positions[index]
                for index, value in enumerate(scores)
                if math.isclose(float(value), best_score, rel_tol=0.0, abs_tol=1e-12)
            ]
            best = min(tied, key=lambda index: self.target_ids_[index])
            predictions.append(float(self.target_[best]))
            similarities.append(best_score)
            neighbor_ids.append(str(self.target_ids_[best]))
        return np.asarray(predictions), np.asarray(similarities), neighbor_ids


def _descriptor_pipeline(
    model_name: str, *, seed: int, config: Mapping[str, Any]
) -> Pipeline:
    numeric_columns = ["model_formal_charge", *SOLVENT_FEATURES, *SITE_NUMERIC_FEATURES]
    features = ColumnTransformer(
        transformers=[
            (
                "molecule",
                Pipeline(
                    [
                        ("rdkit", RDKitDescriptorTransformer()),
                        ("impute", SimpleImputer(strategy="median")),
                    ]
                ),
                "model_canonical_smiles",
            ),
            (
                "numeric",
                SimpleImputer(strategy="median"),
                numeric_columns,
            ),
            (
                "site_type",
                OneHotEncoder(
                    categories=[list(SITE_TYPES)],
                    handle_unknown="error",
                    sparse_output=False,
                ),
                ["site_type"],
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    settings = config["models"][model_name]
    if model_name == "random_forest_standard_descriptors":
        regressor = RandomForestRegressor(
            n_estimators=int(settings["n_estimators"]),
            min_samples_leaf=int(settings["min_samples_leaf"]),
            max_features=float(settings["max_features"]),
            n_jobs=int(settings["n_jobs"]),
            random_state=seed,
        )
    elif model_name == "hist_gradient_boosting_standard_descriptors":
        regressor = HistGradientBoostingRegressor(
            max_iter=int(settings["max_iter"]),
            learning_rate=float(settings["learning_rate"]),
            l2_regularization=float(settings["l2_regularization"]),
            random_state=seed,
        )
    else:
        raise PublicationBaselineError(f"Unsupported descriptor baseline: {model_name}")
    return Pipeline([("features", features), ("regressor", regressor)])


def _fold_membership(
    membership: pd.DataFrame, outer_fold: int
) -> tuple[set[str], set[str]]:
    fold = membership.loc[membership["outer_fold"] == outer_fold].copy()
    development = set(fold.loc[fold["role"] == "development", "target_id"].astype(str))
    test = set(fold.loc[fold["role"] == "test", "target_id"].astype(str))
    if not development or not test or development & test:
        raise PublicationBaselineError(
            f"Invalid membership for outer fold {outer_fold}"
        )
    development_groups = set(
        fold.loc[fold["role"] == "development", "connectivity_id"].astype(str)
    )
    test_groups = set(fold.loc[fold["role"] == "test", "connectivity_id"].astype(str))
    if development_groups & test_groups:
        raise PublicationBaselineError(
            f"Connectivity leakage in outer fold {outer_fold}"
        )
    return development, test


def freeze_scores(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, object]:
    """Fit on outer development labels and freeze all outer-test scores label-blind."""

    config, resolved = read_config(config_path)
    dataset = _project_path(config["dataset_directory"], label="dataset")
    output = _project_path(config["output_directory"], label="output") / "score_freeze"
    if output.exists():
        summary = _read_json(output / "summary.json")
        if summary.get("status") == "complete":
            return summary
        raise PublicationBaselineError(f"Partial score freeze exists: {output}")
    features = load_label_blind_features(dataset)
    features["target_id"] = features["target_id"].astype(str)
    membership = pd.read_csv(dataset / "outer_fold_membership.csv")
    feature_ids = set(features["target_id"])
    fold_payloads: list[tuple[int, pd.DataFrame, dict[str, object]]] = []
    for outer_fold in range(int(config["outer_fold_count"])):
        development_ids, test_ids = _fold_membership(membership, outer_fold)
        if not development_ids | test_ids <= feature_ids:
            raise PublicationBaselineError(f"Unknown target in outer fold {outer_fold}")
        development = features.loc[features["target_id"].isin(development_ids)].copy()
        test = features.loc[features["target_id"].isin(test_ids)].copy()
        development = development.sort_values(
            "target_id", kind="mergesort"
        ).reset_index(drop=True)
        test = test.sort_values("target_id", kind="mergesort").reset_index(drop=True)
        labels = _load_development_labels(dataset, development_ids)
        development = development.merge(labels, on="target_id", validate="one_to_one")
        target = development["N_mean"].to_numpy(dtype=float)
        scores = test[
            [
                "target_id",
                "context_id",
                "species_id",
                "connectivity_id",
                "site_type",
                "solvent_raw",
            ]
        ].copy()
        scores["outer_fold"] = outer_fold
        scores["N_pred_train_mean"] = float(np.mean(target))
        knn_settings = config["models"]["one_nearest_neighbor_morgan"]
        knn = SiteTypeMatchedMorgan1NN(
            radius=int(knn_settings["radius"]),
            n_bits=int(knn_settings["n_bits"]),
            include_chirality=bool(knn_settings["include_chirality"]),
        ).fit(development, target)
        knn_predictions, knn_similarity, knn_neighbor = knn.predict(test)
        scores["N_pred_one_nearest_neighbor_morgan"] = knn_predictions
        scores["morgan_neighbor_similarity"] = knn_similarity
        scores["morgan_neighbor_target_id"] = knn_neighbor
        for model_name in (
            "random_forest_standard_descriptors",
            "hist_gradient_boosting_standard_descriptors",
        ):
            seed = int(config["models"][model_name]["seed_base"]) + outer_fold
            model = _descriptor_pipeline(model_name, seed=seed, config=config)
            model.fit(development, target)
            scores[f"N_pred_{model_name}"] = model.predict(test).astype(float)
        prediction_columns = [f"N_pred_{name}" for name in MODEL_NAMES]
        if scores[prediction_columns].isna().any().any():
            raise PublicationBaselineError(
                f"Non-finite baseline score in fold {outer_fold}"
            )
        fold_summary: dict[str, object] = {
            "outer_fold": outer_fold,
            "development_target_count": len(development),
            "test_target_count": len(test),
            "development_connectivity_count": int(
                development["connectivity_id"].nunique()
            ),
            "test_connectivity_count": int(test["connectivity_id"].nunique()),
            "development_target_id_sha256": _canonical_sha256(sorted(development_ids)),
            "test_target_id_sha256": _canonical_sha256(sorted(test_ids)),
            "outer_test_label_columns_requested": [],
            "outer_test_target_values_loaded": False,
            "metrics_computed": False,
            "sn_imported_or_predicted": False,
        }
        fold_payloads.append((outer_fold, scores, fold_summary))
    all_ids = [
        str(value) for _, scores, _ in fold_payloads for value in scores["target_id"]
    ]
    if (
        len(all_ids) != len(feature_ids)
        or set(all_ids) != feature_ids
        or len(all_ids) != len(set(all_ids))
    ):
        raise PublicationBaselineError(
            "Each target must receive exactly one frozen OOF score"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".baseline-score-freeze-", dir=output.parent)
    )
    try:
        bindings: list[dict[str, object]] = []
        folds: list[dict[str, object]] = []
        for outer_fold, scores, fold_summary in fold_payloads:
            fold_directory = staging / f"outer-{outer_fold}"
            fold_directory.mkdir()
            score_path = fold_directory / "baseline_predictions.parquet"
            scores.to_parquet(
                score_path, index=False, engine="pyarrow", compression="zstd"
            )
            binding = {
                "outer_fold": outer_fold,
                "path": f"outer-{outer_fold}/baseline_predictions.parquet",
                "bytes": score_path.stat().st_size,
                "sha256": sha256_file(score_path),
            }
            binding_summary = fold_summary | {"score_binding": binding}
            atomic_write_json(
                fold_directory / "summary.json", binding_summary, ensure_ascii=False
            )
            bindings.append(binding)
            folds.append(binding_summary)
        summary = {
            "schema_version": SCHEMA,
            "status": "complete",
            "campaign_id": config["campaign_id"],
            "scope": "known_site_oracle_baselines",
            "model_names": list(MODEL_NAMES),
            "target_count": len(all_ids),
            "connectivity_count": int(features["connectivity_id"].nunique()),
            "folds": folds,
            "score_bindings": bindings,
            "all_outer_scores_frozen_before_any_outer_test_label_read": True,
            "outer_test_label_columns_requested": [],
            "outer_test_target_values_loaded": False,
            "metrics_computed_before_score_freeze": False,
            "selection_uses_outer_test_results": False,
            "config_path": resolved.relative_to(ROOT).as_posix(),
            "config_sha256": sha256_file(resolved),
            "dataset_manifest_sha256": sha256_file(dataset / "dataset_manifest.json"),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "frozen_at_utc": datetime.now(UTC).isoformat(),
        }
        summary["freeze_sha256"] = _canonical_sha256(summary)
        atomic_write_json(staging / "summary.json", summary, ensure_ascii=False)
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return summary


def _regression_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    residual = prediction - truth
    truth_rank = pd.Series(truth).rank(method="average").to_numpy(dtype=float)
    prediction_rank = pd.Series(prediction).rank(method="average").to_numpy(dtype=float)
    spearman = (
        math.nan
        if np.std(truth_rank) == 0 or np.std(prediction_rank) == 0
        else float(np.corrcoef(truth_rank, prediction_rank)[0, 1])
    )
    return {
        "mae": float(mean_absolute_error(truth, prediction)),
        "rmse": float(math.sqrt(mean_squared_error(truth, prediction))),
        "r2": float(r2_score(truth, prediction)),
        "mean_error": float(np.mean(residual)),
        "spearman_rho": spearman,
    }


def _bootstrap(
    frame: pd.DataFrame,
    prediction_columns: Mapping[str, str],
    *,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    groups = {
        str(connectivity): group.index.to_numpy(dtype=int)
        for connectivity, group in frame.groupby("connectivity_id", sort=True)
    }
    keys = np.asarray(sorted(groups), dtype=object)
    truth = frame["N_true"].to_numpy(dtype=float)
    predictions = {
        name: frame[column].to_numpy(dtype=float)
        for name, column in prediction_columns.items()
    }
    rng = np.random.default_rng(seed)
    values = {
        name: {metric: [] for metric in ("mae", "rmse", "r2")}
        for name in prediction_columns
    }
    paired = {
        name: {metric: [] for metric in ("mae", "rmse", "r2")} for name in MODEL_NAMES
    }
    for _ in range(replicates):
        selected = rng.choice(keys, size=len(keys), replace=True)
        indices = np.concatenate([groups[str(key)] for key in selected])
        replicate = {
            name: _regression_metrics(truth[indices], prediction[indices])
            for name, prediction in predictions.items()
        }
        for name, metrics in replicate.items():
            for metric in values[name]:
                values[name][metric].append(metrics[metric])
        for name in MODEL_NAMES:
            for metric in paired[name]:
                paired[name][metric].append(
                    replicate[name][metric] - replicate["known_site_oracle_n"][metric]
                )

    def interval(items: Sequence[float]) -> dict[str, float]:
        array = np.asarray(items, dtype=float)
        return {
            "median": float(np.nanmedian(array)),
            "ci95_low": float(np.nanquantile(array, 0.025)),
            "ci95_high": float(np.nanquantile(array, 0.975)),
        }

    return {
        "unit": "connectivity_id",
        "replicates": replicates,
        "seed": seed,
        "connectivity_count": len(keys),
        "model_metric_intervals": {
            name: {metric: interval(items) for metric, items in metrics.items()}
            for name, metrics in values.items()
        },
        "baseline_minus_known_site_oracle_intervals": {
            name: {metric: interval(items) for metric, items in metrics.items()}
            for name, metrics in paired.items()
        },
    }


def evaluate_scores(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, object]:
    """Read labels only after the complete score package exists, then evaluate."""

    config, resolved = read_config(config_path)
    comparator_paths = _comparator_paths(config)
    dataset = _project_path(config["dataset_directory"], label="dataset")
    baseline_root = _project_path(config["output_directory"], label="output")
    freeze_root = baseline_root / "score_freeze"
    freeze_summary = _read_json(freeze_root / "summary.json")
    if (
        freeze_summary.get("status") != "complete"
        or freeze_summary.get(
            "all_outer_scores_frozen_before_any_outer_test_label_read"
        )
        is not True
    ):
        raise PublicationBaselineError("Complete label-blind score freeze is required")
    score_frames: list[pd.DataFrame] = []
    for binding in freeze_summary["score_bindings"]:
        path = freeze_root / str(binding["path"])
        if sha256_file(path) != binding["sha256"]:
            raise PublicationBaselineError(f"Frozen score binding changed: {path}")
        score_frames.append(pd.read_parquet(path))
    scores = pd.concat(score_frames, ignore_index=True)
    labels = pd.read_parquet(
        dataset / "targets.parquet", columns=["target_id", "N_mean"]
    )
    labels = labels.rename(columns={"N_mean": "N_true"})
    frame = scores.merge(labels, on="target_id", how="left", validate="one_to_one")
    oracle_path = comparator_paths["oracle_predictions_path"]
    oracle_summary_path = comparator_paths["oracle_summary_path"]
    oracle_summary = _read_json(oracle_summary_path)
    oracle = pd.read_parquet(
        oracle_path,
        columns=["target_id", "N_pred_stage_e_c_n3_ensemble"],
    ).rename(columns={"N_pred_stage_e_c_n3_ensemble": "N_pred_known_site_oracle_n"})
    frame = frame.merge(oracle, on="target_id", how="left", validate="one_to_one")
    if frame["N_true"].isna().any() or frame["N_pred_known_site_oracle_n"].isna().any():
        raise PublicationBaselineError(
            "Evaluation merge left missing labels or oracle predictions"
        )
    prediction_columns = {
        name: f"N_pred_{name}" for name in (*MODEL_NAMES, "known_site_oracle_n")
    }
    metrics = {
        name: _regression_metrics(
            frame["N_true"].to_numpy(dtype=float),
            frame[column].to_numpy(dtype=float),
        )
        for name, column in prediction_columns.items()
    }
    by_fold = {
        str(fold): {
            name: _regression_metrics(
                group["N_true"].to_numpy(dtype=float),
                group[column].to_numpy(dtype=float),
            )
            for name, column in prediction_columns.items()
        }
        for fold, group in frame.groupby("outer_fold", sort=True)
    }
    bootstrap = _bootstrap(
        frame,
        prediction_columns,
        replicates=int(config["bootstrap_replicates"]),
        seed=int(config["bootstrap_seed"]),
    )
    automatic_summary_path = comparator_paths["automatic_summary_path"]
    automatic_summary = _read_json(automatic_summary_path)
    output = baseline_root / "evaluation"
    if output.exists():
        summary = _read_json(output / "summary.json")
        if summary.get("status") == "complete":
            return summary
        raise PublicationBaselineError(f"Partial baseline evaluation exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".baseline-evaluation-", dir=output.parent))
    try:
        prediction_path = staging / "oof_predictions_with_labels.parquet"
        frame.to_parquet(
            prediction_path, index=False, engine="pyarrow", compression="zstd"
        )
        metric_rows = [
            {"model": name, "population": "known_site_all_targets", **values}
            for name, values in metrics.items()
        ]
        metric_rows.append(
            {
                "model": "automatic_site_plus_n",
                "population": "single_target_contexts",
                **automatic_summary["primary_automatic_site_N_metrics"],
            }
        )
        metrics_path = staging / "model_metrics.csv"
        pd.DataFrame(metric_rows).to_csv(
            metrics_path, index=False, float_format="%.12g"
        )
        bootstrap_path = staging / "bootstrap.json"
        atomic_write_json(bootstrap_path, bootstrap, ensure_ascii=False)
        summary = {
            "schema_version": "nucpred.mayr-n-publication-baseline-evaluation.v1",
            "status": "complete",
            "campaign_id": config["campaign_id"],
            "target_count": len(frame),
            "connectivity_count": int(frame["connectivity_id"].nunique()),
            "model_metrics": metrics,
            "by_outer_fold": by_fold,
            "bootstrap": bootstrap,
            "automatic_site_plus_n": {
                "population": "single_target_contexts",
                "target_count": automatic_summary["single_target_context_count"],
                "metrics": automatic_summary["primary_automatic_site_N_metrics"],
            },
            "required_baselines_accounted_for": [
                *MODEL_NAMES,
                "known_site_oracle_n",
                "automatic_site_plus_n",
            ],
            "scores_frozen_before_outer_test_labels_read": True,
            "model_or_score_updated_after_outer_test_labels_read": False,
            "selection_uses_outer_test_results": False,
            "known_site_baseline_scope": True,
            "automatic_endpoint_is_separate_population": True,
            "sn_imported_or_predicted": False,
            "input_bindings": {
                "score_freeze_summary": sha256_file(freeze_root / "summary.json"),
                "oracle_summary": sha256_file(oracle_summary_path),
                "oracle_predictions": sha256_file(oracle_path),
                "automatic_summary": sha256_file(automatic_summary_path),
                "config": sha256_file(resolved),
            },
            "output_bindings": {
                "predictions": sha256_file(prediction_path),
                "metrics": sha256_file(metrics_path),
                "bootstrap": sha256_file(bootstrap_path),
            },
            "oracle_metric_binding_matches": all(
                math.isclose(
                    metrics["known_site_oracle_n"][metric],
                    float(oracle_summary["metrics"]["stage_e_c_n3"][metric]),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
                for metric in ("mae", "rmse", "r2", "mean_error", "spearman_rho")
            ),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "evaluated_at_utc": datetime.now(UTC).isoformat(),
        }
        if summary["oracle_metric_binding_matches"] is not True:
            raise PublicationBaselineError("Oracle metric binding changed")
        atomic_write_json(staging / "summary.json", summary, ensure_ascii=False)
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("command", choices=("freeze-scores", "evaluate", "run-all"))
    args = parser.parse_args(argv)
    if args.command == "freeze-scores":
        result = freeze_scores(args.config)
    elif args.command == "evaluate":
        result = evaluate_scores(args.config)
    else:
        freeze_scores(args.config)
        result = evaluate_scores(args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
