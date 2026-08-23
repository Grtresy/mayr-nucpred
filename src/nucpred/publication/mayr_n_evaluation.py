"""Prediction-first evaluation for the publication Mayr N ensemble.

The command boundary is intentional:

``calibrate``
    Uses inner-validation labels and label-blind outer-development ensemble
    disagreement to freeze interval and applicability thresholds.
``freeze-predictions``
    Loads outer-test features without requesting ``N_mean`` and writes all
    C2/E-B-N1/E-C-N3 seed scores, intervals, and abstention decisions.
``evaluate``
    Runs only after all five score files exist, then joins labels and computes
    metrics.  It never changes a checkpoint, score, interval, or threshold.
"""

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
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch

from nucpred.core.files import atomic_write_json, sha256_file
from nucpred.experiments.mayr.nextgen_gate_a import _canonical_sha256
from nucpred.experiments.mayr.site_n import _iter_batches
from nucpred.publication.mayr_n_modeling import (
    DEFAULT_CONFIG,
    PublicationModelingError,
    _membership_tables,
    _project_path,
    _read_json,
    apply_input_ablation,
    read_config,
)
from nucpred.publication.mayr_n_outer import (
    _development_ids,
    load_outer_checkpoint,
)
from nucpred.training.mayr_site_n import load_site_n_examples


ROOT = Path(__file__).resolve().parents[3]
STAGES = ("base_c2", "stage_e_b_n1", "stage_e_c_n3")
CALIBRATION_SCHEMA = "nucpred.mayr-n-publication-calibration.v1"
PREDICTION_SCHEMA = "nucpred.mayr-n-publication-oracle-score-freeze.v1"
MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048,
)


def _outer_root(config: Mapping[str, Any]) -> Path:
    return _project_path(config["output_directory"], label="output directory")


def _checkpoint_paths(
    config: Mapping[str, Any], *, outer_fold: int
) -> list[tuple[int, Path]]:
    root = _outer_root(config) / "outer_refit" / f"outer-{outer_fold}"
    result = []
    for seed in map(int, config["outer_initialization_seeds"]):
        path = root / f"init-{seed}" / "model.pt"
        summary = _read_json(path.parent / "summary.json")
        if summary.get("status") != "pass":
            raise PublicationModelingError(f"Outer refit did not pass: {path}")
        if sha256_file(path) != summary.get("model_checkpoint_sha256"):
            raise PublicationModelingError(f"Outer checkpoint drifted: {path}")
        result.append((seed, path))
    return result


def _prediction_metadata(examples) -> dict[str, dict[str, object]]:
    return {
        target_id: {
            "context_id": example.context_id,
            "species_id": example.species_id,
            "connectivity_id": example.connectivity_id,
            "site_object_id": site_object_id,
            "site_type": site_type,
            "member_atom_indices_json": json.dumps(members, separators=(",", ":")),
            "solvent_raw": example.solvent_raw,
            "model_formal_charge": example.model_formal_charge,
        }
        for example in examples
        for target_id, site_object_id, site_type, members in zip(
            example.target_ids,
            example.site_object_ids,
            example.site_types,
            example.site_members,
            strict=True,
        )
    }


def _predict(
    model: torch.nn.Module,
    examples,
    *,
    metadata_examples=None,
    preprocessor,
    vocabulary,
    batch_size: int = 64,
    device: torch.device,
) -> pd.DataFrame:
    metadata = _prediction_metadata(metadata_examples or examples)
    rows: list[dict[str, object]] = []
    model.eval()
    with torch.no_grad():
        for raw in _iter_batches(
            examples,
            batch_size=batch_size,
            preprocessor=preprocessor,
            vocabulary=vocabulary,
            shuffle_seed=None,
        ):
            batch = raw.to(device)
            output = model(batch.inputs)
            values = output.n_prediction_standardized.detach().cpu().numpy() * float(
                preprocessor.target_scale
            ) + float(preprocessor.target_mean)
            for index, target_id in enumerate(batch.target_ids):
                rows.append(
                    {
                        "target_id": target_id,
                        **metadata[target_id],
                        "N_pred": float(values[index]),
                    }
                )
    frame = pd.DataFrame(rows).sort_values("target_id").reset_index(drop=True)
    if len(frame) != len(metadata) or frame["target_id"].duplicated().any():
        raise PublicationModelingError("Prediction rows changed target identity")
    return frame


def _stage_models(model: torch.nn.Module) -> dict[str, torch.nn.Module]:
    try:
        eb = model.frozen_parent
        c2 = eb.frozen_base
    except AttributeError as exc:
        raise PublicationModelingError("Outer model lineage changed") from exc
    return {
        "base_c2": c2,
        "stage_e_b_n1": eb,
        "stage_e_c_n3": model,
    }


def _ensemble_predictions(
    config: Mapping[str, Any],
    examples,
    *,
    outer_fold: int,
    config_path: str | Path = DEFAULT_CONFIG,
    include_stages: Sequence[str] = STAGES,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    selected_stages = tuple(map(str, include_stages))
    if not selected_stages or set(selected_stages) - set(STAGES):
        raise PublicationModelingError("Unknown prediction stage")
    model_examples = apply_input_ablation(examples, config)
    merged: pd.DataFrame | None = None
    checkpoint_audits: list[dict[str, object]] = []
    for seed, path in _checkpoint_paths(config, outer_fold=outer_fold):
        model, preprocessor, vocabulary, payload = load_outer_checkpoint(
            path,
            config_path=config_path,
            device="cpu",
        )
        stage_models = _stage_models(model)
        for stage in selected_stages:
            prediction = _predict(
                stage_models[stage],
                model_examples,
                metadata_examples=examples,
                preprocessor=preprocessor,
                vocabulary=vocabulary,
                device=torch.device("cpu"),
            )
            value_name = f"N_pred_{stage}_seed_{seed}"
            prediction = prediction.rename(columns={"N_pred": value_name})
            if merged is None:
                merged = prediction
            else:
                metadata_columns = [
                    column
                    for column in prediction.columns
                    if column not in {"target_id", value_name}
                ]
                right = prediction.drop(columns=metadata_columns)
                merged = merged.merge(
                    right,
                    on="target_id",
                    how="inner",
                    validate="one_to_one",
                )
        checkpoint_audits.append(
            {
                "initialization_seed": seed,
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(path),
                "model_state_sha256": payload["model_state_sha256"],
                "exact_cpu_load": True,
            }
        )
        del model
    if merged is None:
        raise PublicationModelingError("No ensemble predictions were produced")
    seeds = tuple(map(int, config["outer_initialization_seeds"]))
    for stage in selected_stages:
        columns = [f"N_pred_{stage}_seed_{seed}" for seed in seeds]
        values = merged[columns].to_numpy(dtype=float)
        merged[f"N_pred_{stage}_ensemble"] = values.mean(axis=1)
        merged[f"N_pred_{stage}_std"] = values.std(axis=1, ddof=0)
    return merged.sort_values("target_id").reset_index(drop=True), checkpoint_audits


def _conformal_radius(residuals: np.ndarray, coverage: float) -> float:
    values = np.sort(np.asarray(residuals, dtype=float))
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise PublicationModelingError("Cannot calibrate on no finite residuals")
    rank = min(values.size, int(math.ceil((values.size + 1) * coverage)))
    return float(values[rank - 1])


def _quantile_higher(values: Sequence[float], quantile: float) -> float:
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all() or array.size == 0:
        raise PublicationModelingError("Invalid development-only threshold values")
    return float(np.quantile(array, quantile, method="higher"))


def _fingerprint(smiles: str):
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise PublicationModelingError(f"Invalid dataset SMILES: {smiles}")
    return MORGAN_GENERATOR.GetFingerprint(molecule)


def _species_smiles(dataset: Path) -> dict[str, str]:
    species = pd.read_parquet(
        dataset / "species.parquet",
        columns=["species_id", "model_canonical_smiles"],
    )
    counts = species.groupby("species_id")["model_canonical_smiles"].nunique()
    if not counts.eq(1).all():
        raise PublicationModelingError("Species has multiple model SMILES")
    return (
        species.drop_duplicates("species_id")
        .set_index("species_id")["model_canonical_smiles"]
        .astype(str)
        .to_dict()
    )


def _leave_one_connectivity_distances(
    development: pd.DataFrame,
    smiles: Mapping[str, str],
) -> list[float]:
    entries = (
        development[["species_id", "connectivity_id"]]
        .drop_duplicates()
        .sort_values(["connectivity_id", "species_id"])
    )
    records = [
        (
            str(row.species_id),
            str(row.connectivity_id),
            _fingerprint(smiles[str(row.species_id)]),
        )
        for row in entries.itertuples(index=False)
    ]
    if len(records) < 2:
        raise PublicationModelingError("Need two development connectivities")
    distances: list[float] = []
    for _, connectivity, fingerprint in records:
        others = [item[2] for item in records if item[1] != connectivity]
        if not others:
            raise PublicationModelingError("No other connectivity for OOD reference")
        similarities = DataStructs.BulkTanimotoSimilarity(fingerprint, others)
        distances.append(1.0 - max(map(float, similarities)))
    return distances


def _distance_to_reference(
    query_species_ids: Sequence[str],
    reference_species_ids: Sequence[str],
    smiles: Mapping[str, str],
) -> dict[str, float]:
    references = sorted(set(map(str, reference_species_ids)))
    reference_fingerprints = [_fingerprint(smiles[item]) for item in references]
    result: dict[str, float] = {}
    for query in sorted(set(map(str, query_species_ids))):
        fingerprint = _fingerprint(smiles[query])
        similarities = DataStructs.BulkTanimotoSimilarity(
            fingerprint,
            reference_fingerprints,
        )
        result[query] = 1.0 - max(map(float, similarities))
    return result


def calibrate_outer(
    *,
    outer_fold: int,
    config_path: str | Path = DEFAULT_CONFIG,
) -> dict[str, object]:
    config, resolved = read_config(config_path)
    if outer_fold not in range(int(config["outer_fold_count"])):
        raise PublicationModelingError("Unregistered outer fold")
    outer, _ = _membership_tables(config)
    development_ids, split_audit = _development_ids(outer, outer_fold=outer_fold)
    root = _outer_root(config)
    inner_frames: list[pd.DataFrame] = []
    inner_hashes = []
    for inner_fold in range(int(config["inner_fold_count"])):
        path = (
            root
            / "nested_inner"
            / f"outer-{outer_fold}"
            / f"inner-{inner_fold}"
            / "validation_predictions.parquet"
        )
        inner_hashes.append(sha256_file(path))
        frame = pd.read_parquet(path)
        frame["inner_fold"] = inner_fold
        inner_frames.append(frame)
    calibration_frame = pd.concat(inner_frames, ignore_index=True)
    if calibration_frame["target_id"].duplicated().any():
        raise PublicationModelingError("Inner validation targets are not OOF")
    if set(calibration_frame["target_id"].astype(str)) != development_ids:
        raise PublicationModelingError("Inner OOF predictions do not cover development")
    calibration_frame["absolute_residual"] = (
        calibration_frame["N_pred"] - calibration_frame["N_true"]
    ).abs()
    coverages = (0.50, 0.80, 0.95)
    radii = {
        f"{coverage:.2f}": _conformal_radius(
            calibration_frame["absolute_residual"].to_numpy(dtype=float),
            coverage,
        )
        for coverage in coverages
    }
    dataset = _project_path(config["dataset"]["directory"], label="dataset")
    development = load_site_n_examples(
        dataset,
        target_ids=development_ids,
        load_target_values=False,
    )
    development_scores, checkpoint_audits = _ensemble_predictions(
        config,
        development,
        outer_fold=outer_fold,
        config_path=resolved,
        include_stages=("stage_e_c_n3",),
    )
    disagreement = development_scores["N_pred_stage_e_c_n3_std"].to_numpy(dtype=float)
    selected_outer = outer.loc[
        outer["outer_fold"].eq(outer_fold) & outer["role"].eq("development")
    ]
    smiles = _species_smiles(dataset)
    loo_distances = _leave_one_connectivity_distances(
        selected_outer,
        smiles,
    )
    thresholds = {
        "ensemble_std_q90": _quantile_higher(disagreement, 0.90),
        "ensemble_std_q95": _quantile_higher(disagreement, 0.95),
        "structure_distance_q90": _quantile_higher(loo_distances, 0.90),
        "structure_distance_q95": _quantile_higher(loo_distances, 0.95),
    }
    target = root / "calibration" / f"outer-{outer_fold}"
    if target.exists():
        existing = _read_json(target / "calibration.json")
        if existing.get("status") == "frozen":
            return existing
        raise PublicationModelingError(f"Partial calibration exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".outer-{outer_fold}.staging-", dir=target.parent)
    )
    try:
        calibration_frame.to_parquet(
            staging / "inner_oof_calibration.parquet",
            index=False,
            engine="pyarrow",
            compression="zstd",
        )
        development_scores.to_parquet(
            staging / "development_label_blind_scores.parquet",
            index=False,
            engine="pyarrow",
            compression="zstd",
        )
        payload: dict[str, object] = {
            "schema_version": CALIBRATION_SCHEMA,
            "status": "frozen",
            "campaign_id": config["campaign_id"],
            "outer_fold": outer_fold,
            "calibration_method": "finite_sample_absolute_residual_conformal",
            "calibration_population": "four_inner_validation_folds_exactly_once",
            "calibration_target_count": len(calibration_frame),
            "calibration_target_id_sha256": _canonical_sha256(
                sorted(calibration_frame["target_id"].astype(str))
            ),
            "interval_radii": radii,
            "applicability_thresholds": thresholds,
            "ensemble_disagreement_reference": (
                "three_outer_refit_seeds_on_label_blind_development_features"
            ),
            "structure_distance": "one_minus_nearest_Morgan_radius2_Tanimoto",
            "structure_reference_connectivity_count": int(
                selected_outer["connectivity_id"].nunique()
            ),
            "inner_prediction_sha256": inner_hashes,
            "checkpoint_audits": checkpoint_audits,
            "split_audit": split_audit,
            "config_sha256": sha256_file(resolved),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "label_blind_loader_sha256": sha256_file(
                ROOT / "src/nucpred/training/mayr_site_n.py"
            ),
            "outer_test_target_rows_loaded": 0,
            "outer_test_labels_read": False,
            "test_predictions_computed": 0,
            "frozen_at_utc": datetime.now(UTC).isoformat(),
        }
        payload["calibration_sha256"] = _canonical_sha256(payload)
        atomic_write_json(staging / "calibration.json", payload, ensure_ascii=False)
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return payload


def _outer_test_ids(
    outer: pd.DataFrame, *, outer_fold: int
) -> tuple[set[str], set[str]]:
    selected = outer.loc[outer["outer_fold"].eq(outer_fold) & outer["role"].eq("test")]
    target_ids = set(selected["target_id"].astype(str))
    connectivity_ids = set(selected["connectivity_id"].astype(str))
    if not target_ids or not connectivity_ids:
        raise PublicationModelingError("Outer test membership is empty")
    return target_ids, connectivity_ids


def freeze_outer_predictions(
    *,
    outer_fold: int,
    config_path: str | Path = DEFAULT_CONFIG,
) -> dict[str, object]:
    config, resolved = read_config(config_path)
    if outer_fold not in range(int(config["outer_fold_count"])):
        raise PublicationModelingError("Unregistered outer fold")
    root = _outer_root(config)
    calibration_path = root / "calibration" / f"outer-{outer_fold}" / "calibration.json"
    calibration = _read_json(calibration_path)
    if calibration.get("schema_version") != CALIBRATION_SCHEMA:
        raise PublicationModelingError("Calibration schema changed")
    if calibration.get("status") != "frozen":
        raise PublicationModelingError("Calibration is not frozen")
    outer, _ = _membership_tables(config)
    development_ids, _ = _development_ids(outer, outer_fold=outer_fold)
    test_ids, test_connectivity_ids = _outer_test_ids(outer, outer_fold=outer_fold)
    dataset = _project_path(config["dataset"]["directory"], label="dataset")
    # This call explicitly omits N_mean at the parquet-column boundary.
    test_examples = load_site_n_examples(
        dataset,
        target_ids=test_ids,
        load_target_values=False,
    )
    if sum(item.num_sites for item in test_examples) != len(test_ids):
        raise PublicationModelingError("Label-blind outer-test load changed count")
    scores, checkpoint_audits = _ensemble_predictions(
        config,
        test_examples,
        outer_fold=outer_fold,
        config_path=resolved,
    )
    if set(scores["target_id"].astype(str)) != test_ids:
        raise PublicationModelingError("Frozen scores do not cover outer test")
    selected_outer = outer.loc[outer["outer_fold"].eq(outer_fold)]
    development_species = set(
        selected_outer.loc[
            selected_outer["role"].eq("development"), "species_id"
        ].astype(str)
    )
    test_species = set(
        selected_outer.loc[selected_outer["role"].eq("test"), "species_id"].astype(str)
    )
    smiles = _species_smiles(dataset)
    distances = _distance_to_reference(
        sorted(test_species),
        sorted(development_species),
        smiles,
    )
    scores["structure_distance"] = scores["species_id"].map(distances)
    if scores["structure_distance"].isna().any():
        raise PublicationModelingError("Test structure-distance mapping failed")
    thresholds = calibration["applicability_thresholds"]
    scores["ensemble_disagreement"] = scores["N_pred_stage_e_c_n3_std"]
    scores["abstain_ensemble_disagreement"] = scores["ensemble_disagreement"].gt(
        float(thresholds["ensemble_std_q95"])
    )
    scores["abstain_structure_distance"] = scores["structure_distance"].gt(
        float(thresholds["structure_distance_q95"])
    )
    scores["abstain"] = (
        scores["abstain_ensemble_disagreement"] | scores["abstain_structure_distance"]
    )
    scores["applicability_status"] = np.where(
        scores["abstain"], "abstain", "within_development_thresholds"
    )
    std_scale = max(float(thresholds["ensemble_std_q95"]), 1e-12)
    distance_scale = max(float(thresholds["structure_distance_q95"]), 1e-12)
    scores["applicability_risk_score"] = np.maximum(
        scores["ensemble_disagreement"] / std_scale,
        scores["structure_distance"] / distance_scale,
    )
    final = scores["N_pred_stage_e_c_n3_ensemble"]
    for coverage, radius in calibration["interval_radii"].items():
        token = coverage.replace(".", "_")
        scores[f"N_interval_{token}_lower"] = final - float(radius)
        scores[f"N_interval_{token}_upper"] = final + float(radius)
    forbidden = {"N_true", "N_mean", "absolute_error", "sN", "sn"}
    if forbidden & set(scores.columns):
        raise PublicationModelingError("A label or sN leaked into score freeze")
    target = root / "oracle_score_freeze" / f"outer-{outer_fold}"
    if target.exists():
        existing = _read_json(target / "summary.json")
        if existing.get("status") == "frozen":
            return existing
        raise PublicationModelingError(f"Partial score freeze exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".outer-{outer_fold}.staging-", dir=target.parent)
    )
    try:
        score_path = staging / "oracle_predictions.parquet"
        scores.to_parquet(
            score_path,
            index=False,
            engine="pyarrow",
            compression="zstd",
        )
        summary: dict[str, object] = {
            "schema_version": PREDICTION_SCHEMA,
            "status": "frozen",
            "campaign_id": config["campaign_id"],
            "outer_fold": outer_fold,
            "prediction_scope": "known_site_oracle_diagnostic_only",
            "target_count": len(scores),
            "connectivity_count": int(scores["connectivity_id"].nunique()),
            "target_id_sha256": _canonical_sha256(sorted(test_ids)),
            "score_path": "oracle_predictions.parquet",
            "score_sha256": sha256_file(score_path),
            "calibration_path": calibration_path.relative_to(ROOT).as_posix(),
            "calibration_sha256": sha256_file(calibration_path),
            "checkpoint_audits": checkpoint_audits,
            "config_sha256": sha256_file(resolved),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "label_blind_loader_sha256": sha256_file(
                ROOT / "src/nucpred/training/mayr_site_n.py"
            ),
            "device": "cpu",
            "label_columns_requested": [],
            "target_values_loaded": False,
            "labels_read_before_score_freeze": False,
            "metrics_computed_before_score_freeze": False,
            "sn_imported_or_predicted": False,
            "frozen_at_utc": datetime.now(UTC).isoformat(),
        }
        summary["freeze_sha256"] = _canonical_sha256(summary)
        atomic_write_json(staging / "summary.json", summary, ensure_ascii=False)
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return summary


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    if truth.size == 0 or truth.shape != prediction.shape:
        raise PublicationModelingError("Invalid regression metric vectors")
    residual = prediction - truth
    truth_rank = pd.Series(truth).rank(method="average").to_numpy(dtype=float)
    prediction_rank = pd.Series(prediction).rank(method="average").to_numpy(dtype=float)
    if np.std(truth_rank) <= 0 or np.std(prediction_rank) <= 0:
        spearman = math.nan
    else:
        spearman = float(np.corrcoef(truth_rank, prediction_rank)[0, 1])
    return {
        "mae": float(mean_absolute_error(truth, prediction)),
        "rmse": float(math.sqrt(mean_squared_error(truth, prediction))),
        "r2": float(r2_score(truth, prediction)),
        "mean_error": float(np.mean(residual)),
        "spearman_rho": spearman,
    }


def _cluster_bootstrap(
    frame: pd.DataFrame,
    *,
    prediction_columns: Mapping[str, str],
    replicates: int,
    seed: int,
) -> dict[str, object]:
    groups = {
        str(connectivity): group.index.to_numpy(dtype=int)
        for connectivity, group in frame.groupby("connectivity_id", sort=True)
    }
    keys = np.asarray(sorted(groups), dtype=object)
    rng = np.random.default_rng(seed)
    samples: dict[str, dict[str, list[float]]] = {
        model: {metric: [] for metric in ("mae", "rmse", "r2")}
        for model in prediction_columns
    }
    paired: dict[str, dict[str, list[float]]] = {
        model: {metric: [] for metric in ("mae", "rmse", "r2")}
        for model in prediction_columns
        if model != "stage_e_c_n3"
    }
    truth_all = frame["N_true"].to_numpy(dtype=float)
    prediction_all = {
        model: frame[column].to_numpy(dtype=float)
        for model, column in prediction_columns.items()
    }
    for _ in range(replicates):
        selected = rng.choice(keys, size=len(keys), replace=True)
        indices = np.concatenate([groups[str(key)] for key in selected])
        truth = truth_all[indices]
        replicate_metrics: dict[str, dict[str, float]] = {}
        for model, values in prediction_all.items():
            metrics = _regression_metrics(truth, values[indices])
            replicate_metrics[model] = metrics
            for metric in samples[model]:
                samples[model][metric].append(float(metrics[metric]))
        final_metrics = replicate_metrics["stage_e_c_n3"]
        for model in paired:
            for metric in paired[model]:
                paired[model][metric].append(
                    float(replicate_metrics[model][metric] - final_metrics[metric])
                )

    def summarize(values: Sequence[float]) -> dict[str, float]:
        array = np.asarray(values, dtype=float)
        return {
            "median": float(np.nanmedian(array)),
            "ci95_low": float(np.nanquantile(array, 0.025)),
            "ci95_high": float(np.nanquantile(array, 0.975)),
        }

    return {
        "schema_version": "nucpred.mayr-n-connectivity-bootstrap.v1",
        "unit": "connectivity_id",
        "replicates": replicates,
        "seed": seed,
        "connectivity_count": len(keys),
        "model_metric_intervals": {
            model: {metric: summarize(values) for metric, values in metrics.items()}
            for model, metrics in samples.items()
        },
        "paired_minus_final_intervals": {
            model: {metric: summarize(values) for metric, values in metrics.items()}
            for model, metrics in paired.items()
        },
    }


def _risk_coverage(
    frame: pd.DataFrame, prediction_column: str
) -> list[dict[str, object]]:
    ordered = frame.sort_values(
        ["applicability_risk_score", "target_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    rows: list[dict[str, object]] = []
    for coverage in (1.00, 0.95, 0.90, 0.80, 0.70, 0.60, 0.50):
        count = max(1, int(math.floor(len(ordered) * coverage)))
        selected = ordered.iloc[:count]
        rows.append(
            {
                "nominal_retained_fraction": coverage,
                "retained_target_count": count,
                "observed_retained_fraction": count / len(ordered),
                "maximum_retained_risk_score": float(
                    selected["applicability_risk_score"].max()
                ),
                **_regression_metrics(
                    selected["N_true"].to_numpy(dtype=float),
                    selected[prediction_column].to_numpy(dtype=float),
                ),
            }
        )
    return rows


def evaluate_frozen_oracle(
    config_path: str | Path = DEFAULT_CONFIG,
) -> dict[str, object]:
    config, resolved = read_config(config_path)
    root = _outer_root(config)
    score_frames: list[pd.DataFrame] = []
    score_hashes: list[str] = []
    for outer_fold in range(int(config["outer_fold_count"])):
        directory = root / "oracle_score_freeze" / f"outer-{outer_fold}"
        summary = _read_json(directory / "summary.json")
        if summary.get("schema_version") != PREDICTION_SCHEMA:
            raise PublicationModelingError("Frozen prediction schema changed")
        if summary.get("status") != "frozen":
            raise PublicationModelingError("Outer scores are not frozen")
        score_path = directory / "oracle_predictions.parquet"
        observed = sha256_file(score_path)
        if observed != summary.get("score_sha256"):
            raise PublicationModelingError("Frozen score file drifted")
        frame = pd.read_parquet(score_path)
        frame["outer_fold"] = outer_fold
        score_frames.append(frame)
        score_hashes.append(observed)
    scores = pd.concat(score_frames, ignore_index=True)
    if scores["target_id"].duplicated().any() or len(scores) != 1038:
        raise PublicationModelingError("OOF score partition changed")
    dataset = _project_path(config["dataset"]["directory"], label="dataset")
    labels = pd.read_parquet(
        dataset / "targets.parquet",
        columns=["target_id", "N_mean"],
    ).rename(columns={"N_mean": "N_true"})
    frame = scores.merge(labels, on="target_id", how="inner", validate="one_to_one")
    if len(frame) != len(scores):
        raise PublicationModelingError("Frozen score/label join changed target count")
    prediction_columns = {stage: f"N_pred_{stage}_ensemble" for stage in STAGES}
    metrics = {
        stage: _regression_metrics(
            frame["N_true"].to_numpy(dtype=float),
            frame[column].to_numpy(dtype=float),
        )
        for stage, column in prediction_columns.items()
    }
    seeds = tuple(map(int, config["outer_initialization_seeds"]))
    single_seed_metrics = {
        str(seed): _regression_metrics(
            frame["N_true"].to_numpy(dtype=float),
            frame[f"N_pred_stage_e_c_n3_seed_{seed}"].to_numpy(dtype=float),
        )
        for seed in seeds
    }
    by_outer_fold = {
        str(fold): _regression_metrics(
            group["N_true"].to_numpy(dtype=float),
            group["N_pred_stage_e_c_n3_ensemble"].to_numpy(dtype=float),
        )
        | {
            "target_count": len(group),
            "connectivity_count": int(group["connectivity_id"].nunique()),
        }
        for fold, group in frame.groupby("outer_fold", sort=True)
    }
    by_site_type = {
        str(site_type): _regression_metrics(
            group["N_true"].to_numpy(dtype=float),
            group["N_pred_stage_e_c_n3_ensemble"].to_numpy(dtype=float),
        )
        | {
            "target_count": len(group),
            "connectivity_count": int(group["connectivity_id"].nunique()),
        }
        for site_type, group in frame.groupby("site_type", sort=True)
    }
    intervals: dict[str, object] = {}
    for coverage in ("0.50", "0.80", "0.95"):
        token = coverage.replace(".", "_")
        lower = frame[f"N_interval_{token}_lower"]
        upper = frame[f"N_interval_{token}_upper"]
        covered = frame["N_true"].between(lower, upper, inclusive="both")
        intervals[coverage] = {
            "nominal_coverage": float(coverage),
            "empirical_coverage": float(covered.mean()),
            "mean_width": float((upper - lower).mean()),
            "coverage_error": float(covered.mean() - float(coverage)),
            "target_count": len(frame),
        }
    retained = frame.loc[~frame["abstain"]]
    abstention = {
        "abstained_target_count": int(frame["abstain"].sum()),
        "abstention_fraction": float(frame["abstain"].mean()),
        "retained_target_count": len(retained),
        "retained_metrics": _regression_metrics(
            retained["N_true"].to_numpy(dtype=float),
            retained["N_pred_stage_e_c_n3_ensemble"].to_numpy(dtype=float),
        )
        if len(retained)
        else None,
        "without_abstention_metrics": metrics["stage_e_c_n3"],
        "coverage_error_curve": _risk_coverage(
            frame,
            "N_pred_stage_e_c_n3_ensemble",
        ),
    }
    bootstrap = _cluster_bootstrap(
        frame,
        prediction_columns=prediction_columns,
        replicates=int(config.get("bootstrap_replicates", 5000))
        if "bootstrap_replicates" in config
        else 5000,
        seed=2026080501,
    )
    output = root / "oracle_evaluation"
    if output.exists():
        existing = _read_json(output / "summary.json")
        if existing.get("status") == "complete":
            return existing
        raise PublicationModelingError(f"Partial oracle evaluation exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".oracle-evaluation.staging-", dir=output.parent)
    )
    try:
        frame.to_parquet(
            staging / "oracle_oof_predictions_with_labels.parquet",
            index=False,
            engine="pyarrow",
            compression="zstd",
        )
        atomic_write_json(staging / "bootstrap.json", bootstrap, ensure_ascii=False)
        summary: dict[str, object] = {
            "schema_version": "nucpred.mayr-n-publication-oracle-evaluation.v1",
            "status": "complete",
            "campaign_id": config["campaign_id"],
            "scope": "known_site_oracle_diagnostic_not_primary_automatic_model",
            "target_count": len(frame),
            "connectivity_count": int(frame["connectivity_id"].nunique()),
            "metrics": metrics,
            "single_seed_metrics": single_seed_metrics,
            "by_outer_fold": by_outer_fold,
            "by_site_type": by_site_type,
            "intervals": intervals,
            "abstention": abstention,
            "bootstrap": bootstrap,
            "score_sha256": score_hashes,
            "scores_frozen_before_label_read": True,
            "checkpoint_or_score_updated_after_label_read": False,
            "config_sha256": sha256_file(resolved),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "evaluated_at_utc": datetime.now(UTC).isoformat(),
        }
        atomic_write_json(staging / "summary.json", summary, ensure_ascii=False)
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return summary


def freeze_all(config_path: str | Path = DEFAULT_CONFIG) -> list[dict[str, object]]:
    config, _ = read_config(config_path)
    results = []
    for outer_fold in range(int(config["outer_fold_count"])):
        calibrate_outer(outer_fold=outer_fold, config_path=config_path)
        results.append(
            freeze_outer_predictions(outer_fold=outer_fold, config_path=config_path)
        )
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)
    calibrate_parser = subparsers.add_parser("calibrate")
    calibrate_parser.add_argument("--outer-fold", type=int, required=True)
    freeze_parser = subparsers.add_parser("freeze-predictions")
    freeze_parser.add_argument("--outer-fold", type=int, required=True)
    subparsers.add_parser("freeze-all")
    subparsers.add_parser("evaluate")
    args = parser.parse_args(argv)
    if args.command == "calibrate":
        result: object = calibrate_outer(
            outer_fold=args.outer_fold,
            config_path=args.config,
        )
    elif args.command == "freeze-predictions":
        result = freeze_outer_predictions(
            outer_fold=args.outer_fold,
            config_path=args.config,
        )
    elif args.command == "freeze-all":
        result = freeze_all(args.config)
    else:
        result = evaluate_frozen_oracle(args.config)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
