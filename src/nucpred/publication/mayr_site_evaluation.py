"""Post-freeze evaluation for automatic-site Mayr N prediction."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, r2_score

from nucpred.core.files import atomic_write_json, sha256_file
from nucpred.publication import mayr_site_publication as shared
from nucpred.publication.mayr_site_scoring import (
    FORBIDDEN_SCORE_COLUMNS,
    SCORE_SCHEMA,
    assert_label_blind_scores,
)


EVALUATION_SCHEMA = "nucpred.mayr-n-publication-automatic-site-evaluation.v1"


def _output_root(config: Mapping[str, Any]) -> Path:
    return shared.project_path(config["output_directory"], label="site output")


def _load_frozen_scores(
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    root = _output_root(config) / "outer_score_freeze"
    candidates: list[pd.DataFrame] = []
    contexts: list[pd.DataFrame] = []
    bindings: list[dict[str, object]] = []
    for outer_fold in range(int(config["outer_fold_count"])):
        directory = root / f"outer-{outer_fold}"
        summary_path = directory / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("schema_version") != SCORE_SCHEMA or summary.get("status") != "frozen":
            raise shared.PublicationSiteError("Automatic-site score package is not frozen")
        required_false = (
            "target_table_opened",
            "target_id_column_requested",
            "site_labels_read_before_score_freeze",
            "N_labels_read_before_score_freeze",
            "metrics_computed_before_score_freeze",
        )
        if any(summary.get(key) is not False for key in required_false):
            raise shared.PublicationSiteError("Score package phase separation changed")
        candidate_path = directory / "candidate_scores.parquet"
        context_path = directory / "context_scores.parquet"
        if sha256_file(candidate_path) != summary["candidate_score_sha256"]:
            raise shared.PublicationSiteError("Candidate score package drifted")
        if sha256_file(context_path) != summary["context_score_sha256"]:
            raise shared.PublicationSiteError("Context score package drifted")
        for binding in summary["source_bindings"].values():
            source_path = shared.project_path(binding["path"], label="score source")
            if sha256_file(source_path) != binding["sha256"]:
                raise shared.PublicationSiteError("Frozen scorer source drifted")
        candidate = pd.read_parquet(candidate_path)
        context = pd.read_parquet(context_path)
        assert_label_blind_scores(candidate)
        if FORBIDDEN_SCORE_COLUMNS & set(context.columns):
            raise shared.PublicationSiteError("Context score package contains labels")
        candidates.append(candidate)
        contexts.append(context)
        bindings.append(
            {
                "outer_fold": outer_fold,
                "summary_path": summary_path.relative_to(shared.ROOT).as_posix(),
                "summary_sha256": sha256_file(summary_path),
                "candidate_score_sha256": summary["candidate_score_sha256"],
                "context_score_sha256": summary["context_score_sha256"],
            }
        )
    candidate_scores = pd.concat(candidates, ignore_index=True)
    context_scores = pd.concat(contexts, ignore_index=True)
    if candidate_scores["query_id"].astype(str).duplicated().any():
        raise shared.PublicationSiteError("Outer candidate scores overlap across folds")
    if context_scores["context_id"].astype(str).duplicated().any():
        raise shared.PublicationSiteError("Outer context scores overlap across folds")
    return candidate_scores, context_scores, bindings


def _load_labels(config: Mapping[str, Any]) -> pd.DataFrame:
    dataset = shared.project_path(config["dataset"]["directory"], label="dataset")
    targets = pd.read_parquet(dataset / "targets.parquet")
    if len(targets) != 1038 or targets["target_id"].astype(str).duplicated().any():
        raise shared.PublicationSiteError("Frozen target labels changed")
    return targets


def _score_ranks(
    frame: pd.DataFrame, *, score_column: str, rank_column: str
) -> pd.DataFrame:
    ordered = frame.sort_values(
        ["context_id", score_column, "candidate_site_id"],
        ascending=[True, False, True],
        kind="stable",
    ).copy()
    ordered[rank_column] = ordered.groupby("context_id").cumcount() + 1
    return ordered.sort_values("query_id", kind="stable").reset_index(drop=True)


def _annotate_candidates(
    candidate_scores: pd.DataFrame, targets: pd.DataFrame
) -> pd.DataFrame:
    exact_by_context = (
        targets.groupby("context_id")["site_object_id"]
        .agg(lambda values: set(map(str, values)))
        .to_dict()
    )
    if set(candidate_scores["context_id"].astype(str)) != set(exact_by_context):
        raise shared.PublicationSiteError("Frozen scores and target contexts differ")
    frame = candidate_scores.copy()
    frame["exact_label"] = [
        str(candidate_id) in exact_by_context[str(context_id)]
        for context_id, candidate_id in zip(
            frame["context_id"], frame["candidate_site_id"], strict=True
        )
    ]
    covered_contexts = set(
        frame.loc[frame["exact_label"], "context_id"].astype(str)
    )
    if covered_contexts != set(exact_by_context):
        raise shared.PublicationSiteError("Exact candidate annotation changed")
    frame = _score_ranks(
        frame, score_column="base_canonical_logit", rank_column="base_candidate_rank"
    )
    frame = _score_ranks(
        frame,
        score_column="conditional_N_prediction",
        rank_column="nmax_candidate_rank",
    )
    return frame


def _truth_summary(targets: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for context_id, group in targets.groupby("context_id", sort=True):
        rows.append(
            {
                "context_id": str(context_id),
                "target_count": len(group),
                "true_site_ids_json": json.dumps(
                    sorted(set(group["site_object_id"].astype(str))),
                    separators=(",", ":"),
                ),
                "true_site_types_json": json.dumps(
                    sorted(set(group["site_type"].astype(str))),
                    separators=(",", ":"),
                ),
                "single_target_id": (
                    str(group.iloc[0]["target_id"]) if len(group) == 1 else None
                ),
                "single_true_site_id": (
                    str(group.iloc[0]["site_object_id"]) if len(group) == 1 else None
                ),
                "single_true_site_type": (
                    str(group.iloc[0]["site_type"]) if len(group) == 1 else None
                ),
                "single_N_true": (
                    float(group.iloc[0]["N_mean"]) if len(group) == 1 else np.nan
                ),
                "single_assignment_resolution": (
                    str(group.iloc[0]["assignment_resolution"])
                    if len(group) == 1
                    else None
                ),
            }
        )
    return pd.DataFrame(rows)


def _context_retrieval(
    candidates: pd.DataFrame,
    context_scores: pd.DataFrame,
    targets: pd.DataFrame,
) -> pd.DataFrame:
    truth = _truth_summary(targets)
    exact = candidates.loc[candidates["exact_label"]].copy()
    rank_summary = exact.groupby("context_id").agg(
        exact_rank=("candidate_rank", "min"),
        base_exact_rank=("base_candidate_rank", "min"),
        nmax_exact_rank=("nmax_candidate_rank", "min"),
    )
    candidate_count = candidates.groupby("context_id").size().rename("candidate_count")
    context = context_scores.merge(
        truth, on="context_id", how="inner", validate="one_to_one"
    ).merge(
        rank_summary,
        on="context_id",
        how="inner",
        validate="one_to_one",
    ).merge(
        candidate_count,
        on="context_id",
        how="inner",
        validate="one_to_one",
    )
    context["site_top1_correct"] = context["exact_rank"].eq(1)
    context["site_top3_correct"] = context["exact_rank"].le(3)
    context["site_top5_correct"] = context["exact_rank"].le(5)
    context["site_reciprocal_rank"] = 1.0 / context["exact_rank"]
    context["base_site_top1_correct"] = context["base_exact_rank"].eq(1)
    context["base_site_top3_correct"] = context["base_exact_rank"].le(3)
    context["base_site_top5_correct"] = context["base_exact_rank"].le(5)
    context["base_site_reciprocal_rank"] = 1.0 / context["base_exact_rank"]
    context["nmax_site_top1_correct"] = context["nmax_exact_rank"].eq(1)
    context["nmax_site_top3_correct"] = context["nmax_exact_rank"].le(3)
    context["nmax_site_top5_correct"] = context["nmax_exact_rank"].le(5)
    context["nmax_site_reciprocal_rank"] = 1.0 / context["nmax_exact_rank"]
    context["uniform_random_expected_top1"] = (
        context["target_count"] / context["candidate_count"]
    )
    true_type_sets = {
        str(context_id): set(group["site_type"].astype(str))
        for context_id, group in targets.groupby("context_id", sort=True)
    }
    context["predicted_type_correct"] = [
        str(site_type) in true_type_sets[str(context_id)]
        for context_id, site_type in zip(
            context["context_id"], context["predicted_site_type"], strict=True
        )
    ]
    return context.sort_values("context_id", kind="stable").reset_index(drop=True)


def _add_single_target_n(
    context: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    single = context.loc[context["target_count"].eq(1)].copy()
    exact_prediction = candidates.loc[
        candidates["exact_label"],
        [
            "context_id",
            "candidate_site_id",
            "conditional_N_prediction",
            "conditional_N_ensemble_std",
        ],
    ].rename(
        columns={
            "candidate_site_id": "oracle_candidate_site_id",
            "conditional_N_prediction": "oracle_site_predicted_N",
            "conditional_N_ensemble_std": "oracle_site_N_ensemble_std",
        }
    )
    if exact_prediction["context_id"].astype(str).duplicated().any():
        # This is allowed only outside the single-target primary population.
        exact_prediction = exact_prediction.loc[
            exact_prediction["context_id"].astype(str).isin(
                set(single["context_id"].astype(str))
            )
        ]
    single = single.merge(
        exact_prediction,
        on="context_id",
        how="inner",
        validate="one_to_one",
    )
    nmax = candidates.loc[candidates["nmax_candidate_rank"].eq(1), [
        "context_id",
        "candidate_site_id",
        "conditional_N_prediction",
    ]].rename(
        columns={
            "candidate_site_id": "nmax_candidate_site_id",
            "conditional_N_prediction": "nmax_predicted_N",
        }
    )
    single = single.merge(nmax, on="context_id", how="inner", validate="one_to_one")
    single["automatic_N_error"] = single["predicted_N"] - single["single_N_true"]
    single["oracle_site_N_error"] = (
        single["oracle_site_predicted_N"] - single["single_N_true"]
    )
    single["nmax_N_error"] = single["nmax_predicted_N"] - single["single_N_true"]
    return single


def _retrieval_metrics(frame: pd.DataFrame, *, prefix: str = "") -> dict[str, float | int]:
    column = lambda name: f"{prefix}{name}" if prefix else name
    return {
        "context_count": len(frame),
        "connectivity_count": int(frame["connectivity_id"].nunique()),
        "exact_top1_recall": float(frame[column("site_top1_correct")].mean()),
        "exact_top3_recall": float(frame[column("site_top3_correct")].mean()),
        "exact_top5_recall": float(
            frame[column("site_top5_correct")].mean()
            if column("site_top5_correct") in frame
            else frame[column("site_top3_correct")].mean()
        ),
        "mrr": float(frame[column("site_reciprocal_rank")].mean()),
        **(
            {
                "predicted_type_top1_recall": float(
                    frame["predicted_type_correct"].mean()
                )
            }
            if not prefix and "predicted_type_correct" in frame
            else {}
        ),
    }


def _n_metrics(frame: pd.DataFrame, *, prediction: str, truth: str) -> dict[str, float | int]:
    y = frame[truth].to_numpy(dtype=float)
    pred = frame[prediction].to_numpy(dtype=float)
    error = pred - y
    return {
        "count": len(frame),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "r2": float(r2_score(y, pred)) if len(frame) >= 2 else float("nan"),
        "mean_error": float(np.mean(error)),
        "spearman": float(pd.Series(y).corr(pd.Series(pred), method="spearman")),
    }


def _endpoint_calibration(
    candidates: pd.DataFrame, context: pd.DataFrame
) -> dict[str, object]:
    counts = candidates.groupby("context_id")["query_id"].transform("count")
    weights = 1.0 / counts.to_numpy(dtype=float)
    weights /= weights.sum()
    labels = candidates["exact_label"].to_numpy(dtype=int)
    probability = candidates["canonical_endpoint_probability"].to_numpy(dtype=float)
    candidate_brier = float(np.sum(weights * (probability - labels) ** 2))
    candidate_ap = float(average_precision_score(labels, probability, sample_weight=weights))
    top_probability = context["top1_endpoint_probability"].to_numpy(dtype=float)
    correct = context["site_top1_correct"].to_numpy(dtype=float)
    top1_brier = float(np.mean((top_probability - correct) ** 2))
    rows: list[dict[str, object]] = []
    ece = 0.0
    edges = np.linspace(0.0, 1.0, 11)
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        selected = (top_probability >= lower) & (
            (top_probability <= upper) if index == 9 else (top_probability < upper)
        )
        count = int(selected.sum())
        if count:
            confidence = float(top_probability[selected].mean())
            accuracy = float(correct[selected].mean())
            ece += count / len(context) * abs(confidence - accuracy)
        else:
            confidence = float("nan")
            accuracy = float("nan")
        rows.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "count": count,
                "mean_confidence": confidence,
                "accuracy": accuracy,
            }
        )
    return {
        "candidate_context_weighted_brier": candidate_brier,
        "candidate_context_weighted_average_precision": candidate_ap,
        "top1_brier": top1_brier,
        "top1_ece_10_equal_width_bins": float(ece),
        "top1_reliability_bins": rows,
        "probability_semantics": "canonical_endpoint_probability_not_universal_chemical_validity",
    }


def _bootstrap_means(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str],
    replicates: int,
    seed: int,
) -> dict[str, np.ndarray]:
    grouped = frame.groupby("connectivity_id", sort=True)
    counts = grouped.size().to_numpy(dtype=float)
    sums = {column: grouped[column].sum().to_numpy(dtype=float) for column in columns}
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(counts), size=(replicates, len(counts)))
    denominator = counts[draw].sum(axis=1)
    return {
        column: sums[column][draw].sum(axis=1) / denominator for column in columns
    }


def _bootstrap_n(
    frame: pd.DataFrame,
    *,
    prediction: str,
    truth: str,
    replicates: int,
    seed: int,
) -> dict[str, np.ndarray]:
    work = frame[["connectivity_id", prediction, truth]].copy()
    work["error"] = work[prediction] - work[truth]
    work["absolute_error"] = work["error"].abs()
    work["squared_error"] = work["error"] ** 2
    work["y"] = work[truth]
    work["y2"] = work[truth] ** 2
    grouped = work.groupby("connectivity_id", sort=True)
    count = grouped.size().to_numpy(dtype=float)
    arrays = {
        name: grouped[name].sum().to_numpy(dtype=float)
        for name in ["error", "absolute_error", "squared_error", "y", "y2"]
    }
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(count), size=(replicates, len(count)))
    n = count[draw].sum(axis=1)
    aggregated = {name: values[draw].sum(axis=1) for name, values in arrays.items()}
    sst = aggregated["y2"] - aggregated["y"] ** 2 / n
    return {
        "mae": aggregated["absolute_error"] / n,
        "rmse": np.sqrt(aggregated["squared_error"] / n),
        "r2": 1.0 - aggregated["squared_error"] / np.maximum(sst, 1e-12),
        "mean_error": aggregated["error"] / n,
    }


def _ci_rows(
    distributions: Mapping[str, np.ndarray],
    points: Mapping[str, float],
    *,
    family: str,
    population: str,
) -> list[dict[str, object]]:
    return [
        {
            "family": family,
            "population": population,
            "metric": metric,
            "point": float(points[metric]),
            "ci95_low": float(np.quantile(values, 0.025)),
            "ci95_high": float(np.quantile(values, 0.975)),
        }
        for metric, values in distributions.items()
    ]


def _site_type_rows(single: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for site_type, group in single.groupby("single_true_site_type", sort=True):
        rows.append(
            {
                "site_type": str(site_type),
                **_retrieval_metrics(group),
                **{
                    f"automatic_N_{key}": value
                    for key, value in _n_metrics(
                        group, prediction="predicted_N", truth="single_N_true"
                    ).items()
                },
            }
        )
    return rows


def evaluate(
    config_path: str | Path = shared.DEFAULT_CONFIG,
    *,
    output_root: str | Path | None = None,
) -> dict[str, object]:
    config, resolved = shared.read_config(config_path)
    candidate_scores, context_scores, score_bindings = _load_frozen_scores(config)
    # This is the first call in this workflow that opens targets.parquet.
    targets = _load_labels(config)
    candidates = _annotate_candidates(candidate_scores, targets)
    context = _context_retrieval(candidates, context_scores, targets)
    single = _add_single_target_n(context, candidates)
    multi = context.loc[context["target_count"].gt(1)].copy()
    if len(single) + len(multi) != len(context):
        raise shared.PublicationSiteError("Primary/multi-target populations do not partition")

    primary_site = _retrieval_metrics(single)
    base_site = _retrieval_metrics(single, prefix="base_")
    nmax_site = _retrieval_metrics(single, prefix="nmax_")
    multi_site = _retrieval_metrics(multi) if len(multi) else {}
    primary_n = _n_metrics(single, prediction="predicted_N", truth="single_N_true")
    oracle_n = _n_metrics(
        single, prediction="oracle_site_predicted_N", truth="single_N_true"
    )
    nmax_n = _n_metrics(single, prediction="nmax_predicted_N", truth="single_N_true")
    correct_site_n = _n_metrics(
        single.loc[single["site_top1_correct"]],
        prediction="predicted_N",
        truth="single_N_true",
    )
    wrong_site_n = _n_metrics(
        single.loc[~single["site_top1_correct"]],
        prediction="predicted_N",
        truth="single_N_true",
    )
    accepted = single.loc[single["accepted_by_margin"]]
    abstention = {
        "accepted_count": len(accepted),
        "coverage": float(len(accepted) / len(single)),
        "accepted_site_top1_recall": float(accepted["site_top1_correct"].mean()),
        "accepted_N_metrics": _n_metrics(
            accepted, prediction="predicted_N", truth="single_N_true"
        ),
    }
    calibration = _endpoint_calibration(candidates, context)
    replicate_count = int(config["evaluation"]["bootstrap_replicates"])
    bootstrap_seed = int(config["evaluation"]["bootstrap_seed"])
    site_columns = [
        "site_top1_correct",
        "site_top3_correct",
        "site_top5_correct",
        "site_reciprocal_rank",
        "predicted_type_correct",
    ]
    site_dist = _bootstrap_means(
        single,
        columns=site_columns,
        replicates=replicate_count,
        seed=bootstrap_seed,
    )
    site_points = {column: float(single[column].mean()) for column in site_columns}
    ci_rows = _ci_rows(
        site_dist, site_points, family="site_retrieval", population="single_target"
    )
    n_dist = _bootstrap_n(
        single,
        prediction="predicted_N",
        truth="single_N_true",
        replicates=replicate_count,
        seed=bootstrap_seed + 1,
    )
    ci_rows.extend(
        _ci_rows(
            n_dist,
            {key: float(primary_n[key]) for key in n_dist},
            family="automatic_site_N",
            population="single_target",
        )
    )
    comparison_columns = {
        "final_minus_base_top1": single["site_top1_correct"].astype(float)
        - single["base_site_top1_correct"].astype(float),
        "final_minus_base_top3": single["site_top3_correct"].astype(float)
        - single["base_site_top3_correct"].astype(float),
        "final_minus_base_mrr": single["site_reciprocal_rank"]
        - single["base_site_reciprocal_rank"],
        "final_minus_nmax_top1": single["site_top1_correct"].astype(float)
        - single["nmax_site_top1_correct"].astype(float),
    }
    paired = single[["connectivity_id"]].copy()
    for key, values in comparison_columns.items():
        paired[key] = values
    paired_dist = _bootstrap_means(
        paired,
        columns=list(comparison_columns),
        replicates=replicate_count,
        seed=bootstrap_seed + 2,
    )
    paired_rows = _ci_rows(
        paired_dist,
        {key: float(values.mean()) for key, values in comparison_columns.items()},
        family="paired_site_comparison",
        population="single_target",
    )
    for row in paired_rows:
        values = paired_dist[str(row["metric"])]
        row["probability_delta_gt_zero"] = float(np.mean(values > 0))
    outer_rows = []
    for outer_fold, group in single.groupby("outer_fold", sort=True):
        outer_rows.append(
            {
                "outer_fold": int(outer_fold),
                **_retrieval_metrics(group),
                **{
                    f"automatic_N_{key}": value
                    for key, value in _n_metrics(
                        group, prediction="predicted_N", truth="single_N_true"
                    ).items()
                },
                "abstention_coverage": float(group["accepted_by_margin"].mean()),
                "accepted_site_top1_recall": float(
                    group.loc[group["accepted_by_margin"], "site_top1_correct"].mean()
                ),
            }
        )
    distance_bins = pd.cut(
        single["structure_distance"],
        bins=[0.0, 0.25, 0.5, 0.75, 1.0000001],
        labels=["[0,0.25)", "[0.25,0.5)", "[0.5,0.75)", "[0.75,1]"],
        include_lowest=True,
        right=False,
    )
    ood_rows = []
    for label, group in single.assign(structure_distance_bin=distance_bins).groupby(
        "structure_distance_bin", observed=False, sort=True
    ):
        if group.empty:
            continue
        ood_rows.append(
            {
                "structure_distance_bin": str(label),
                **_retrieval_metrics(group),
                **{
                    f"automatic_N_{key}": value
                    for key, value in _n_metrics(
                        group, prediction="predicted_N", truth="single_N_true"
                    ).items()
                },
            }
        )
    root = Path(output_root).resolve() if output_root else _output_root(config)
    destination = root / "outer_evaluation"
    if destination.exists():
        raise shared.PublicationSiteError("Refusing to overwrite outer evaluation")
    destination.mkdir(parents=True)
    shared.atomic_parquet(destination / "candidate_evaluation.parquet", candidates)
    shared.atomic_parquet(destination / "context_evaluation.parquet", context)
    shared.atomic_parquet(destination / "single_target_evaluation.parquet", single)
    pd.DataFrame(outer_rows).to_csv(destination / "outer_fold_metrics.csv", index=False)
    pd.DataFrame(_site_type_rows(single)).to_csv(
        destination / "site_type_metrics.csv", index=False
    )
    pd.DataFrame(ood_rows).to_csv(destination / "structure_distance_metrics.csv", index=False)
    pd.DataFrame(ci_rows).to_csv(destination / "bootstrap_intervals.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(destination / "paired_comparisons.csv", index=False)
    error_cases = single.loc[~single["site_top1_correct"]].sort_values(
        ["structure_distance", "top1_margin"], ascending=[False, False], kind="stable"
    )
    error_cases.to_csv(destination / "site_error_cases.csv", index=False)
    artifact_paths = {
        "candidate_evaluation": destination / "candidate_evaluation.parquet",
        "context_evaluation": destination / "context_evaluation.parquet",
        "single_target_evaluation": destination / "single_target_evaluation.parquet",
        "outer_fold_metrics": destination / "outer_fold_metrics.csv",
        "site_type_metrics": destination / "site_type_metrics.csv",
        "structure_distance_metrics": destination / "structure_distance_metrics.csv",
        "bootstrap_intervals": destination / "bootstrap_intervals.csv",
        "paired_comparisons": destination / "paired_comparisons.csv",
        "site_error_cases": destination / "site_error_cases.csv",
    }
    payload: dict[str, object] = {
        "schema_version": EVALUATION_SCHEMA,
        "status": "pass",
        "campaign_id": config["campaign_id"],
        "config_sha256": sha256_file(resolved),
        "evaluation_source_path": Path(__file__).resolve().relative_to(
            shared.ROOT
        ).as_posix(),
        "evaluation_source_sha256": sha256_file(Path(__file__).resolve()),
        "score_bindings": score_bindings,
        "output_bindings": {
            name: {
                "path": path.relative_to(shared.ROOT).as_posix()
                if path.is_relative_to(shared.ROOT)
                else path.resolve().as_posix(),
                "sha256": sha256_file(path),
                "bytes": int(path.stat().st_size),
            }
            for name, path in artifact_paths.items()
        },
        "target_label_path": config["dataset"]["directory"] + "/targets.parquet",
        "target_label_sha256": sha256_file(
            shared.project_path(config["dataset"]["directory"], label="dataset")
            / "targets.parquet"
        ),
        "labels_read_only_after_all_score_packages_frozen": True,
        "total_target_count": len(targets),
        "total_context_count": len(context),
        "single_target_context_count": len(single),
        "multi_target_context_count": len(multi),
        "primary_population": "single_target_outer_test_contexts",
        "primary_site_metrics": primary_site,
        "multi_target_site_metrics": multi_site,
        "base_ranker_ablation_site_metrics": base_site,
        "conditional_N_max_baseline_site_metrics": nmax_site,
        "uniform_random_expected_top1": float(
            single["uniform_random_expected_top1"].mean()
        ),
        "primary_automatic_site_N_metrics": primary_n,
        "oracle_site_N_metrics_same_single_target_population": oracle_n,
        "conditional_N_max_baseline_N_metrics": nmax_n,
        "correct_site_only_N_metrics": correct_site_n,
        "wrong_site_only_N_metrics": wrong_site_n,
        "abstention": abstention,
        "endpoint_calibration": calibration,
        "bootstrap_unit": config["evaluation"]["bootstrap_unit"],
        "bootstrap_replicates": replicate_count,
        "bootstrap_seed": bootstrap_seed,
        "candidate_softmax_used": False,
        "unknown_as_universal_negative": False,
        "sn_imported_or_predicted": False,
        "automatic_site_N_interval_claimed": False,
        "automatic_site_N_interval_reason": (
            "conditional-N uncertainty does not include discrete site-selection error"
        ),
    }
    atomic_write_json(destination / "summary.json", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=shared.DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = evaluate(args.config, output_root=args.output_root)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
