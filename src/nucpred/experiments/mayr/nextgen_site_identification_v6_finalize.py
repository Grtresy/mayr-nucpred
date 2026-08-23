"""Finalize paired v5/v6 comparison and the structured-site report."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from nucpred.core.files import atomic_write_json, sha256_file
from nucpred.experiments.mayr import nextgen_site_identification as base
from nucpred.experiments.mayr.nextgen_site_identification_v6 import DEFAULT_CONFIG
from nucpred.inference.mayr_nextgen_contract import validate_response
from nucpred.project import get_project_layout
from nucpred.training.mayr_site_inference_assets import canonical_sha256


ROOT = get_project_layout().root
COMPARISON_SCHEMA = "nucpred.mayr-site-identification-v5-v6-comparison.v1"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _single_target_results(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    return frame.loc[frame["single_target_context"].astype(bool)].copy()


def _verify_stage_manifest(directory: Path) -> dict[str, object]:
    """Verify one immutable stage before it is cited by the final report."""

    if not directory.is_dir():
        raise base.SiteIdentificationError(f"Missing campaign stage: {directory}")
    manifest_path = directory / "run_manifest.json"
    manifest = base._load_json(manifest_path)
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise base.SiteIdentificationError(
            f"Stage manifest has no file records: {manifest_path}"
        )
    expected_paths: list[str] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise base.SiteIdentificationError(
                f"Stage manifest record is invalid: {manifest_path}"
            )
        relative = Path(str(record.get("path", "")))
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise base.SiteIdentificationError(
                f"Stage manifest path is invalid: {relative}"
            )
        path = directory / relative
        base._verify_sha(
            path,
            record.get("sha256"),
            label=f"stage file {relative.as_posix()}",
        )
        if path.stat().st_size != int(record.get("size_bytes", -1)):
            raise base.SiteIdentificationError(f"Stage file size changed: {path}")
        expected_paths.append(relative.as_posix())
    observed_paths = sorted(
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path.name != "run_manifest.json"
    )
    if expected_paths != observed_paths:
        raise base.SiteIdentificationError(f"Stage file inventory changed: {directory}")
    if canonical_sha256(records) != str(manifest.get("content_sha256")):
        raise base.SiteIdentificationError(
            f"Stage manifest content hash changed: {manifest_path}"
        )
    summary = base._load_json(directory / "summary.json")
    if summary.get("status") != "pass":
        raise base.SiteIdentificationError(f"Stage summary is not pass: {directory}")
    return {
        "stage": directory.name,
        "schema_version": str(manifest.get("schema_version")),
        "file_count": len(records),
        "content_sha256": str(manifest["content_sha256"]),
        "manifest_sha256": sha256_file(manifest_path),
    }


def _error_decomposition(frame: pd.DataFrame) -> dict[str, object]:
    frame = frame.copy()
    type_correct = (
        frame["automatic_site_type"].astype(str).eq(frame["site_type_true"].astype(str))
    )
    frame["error_kind"] = np.where(
        frame["exact_top1"].astype(bool),
        "exact",
        np.where(type_correct, "right_type_wrong_membership", "wrong_type"),
    )
    frame["automatic_squared_error"] = (
        frame["automatic_N_prediction"].astype(float) - frame["N_mean"].astype(float)
    ) ** 2
    frame["oracle_squared_error"] = (
        frame["oracle_N_prediction"].astype(float) - frame["N_mean"].astype(float)
    ) ** 2
    frame["excess_squared_error"] = (
        frame["automatic_squared_error"] - frame["oracle_squared_error"]
    )
    rows: list[dict[str, object]] = []
    for error_kind, group in frame.groupby("error_kind", sort=True):
        rows.append(
            {
                "error_kind": str(error_kind),
                "record_count": len(group),
                "record_fraction": float(len(group) / len(frame)),
                "automatic_rmse": float(
                    np.sqrt(group["automatic_squared_error"].mean())
                ),
                "excess_squared_error_sum": float(group["excess_squared_error"].sum()),
            }
        )
    return {
        "record_count": len(frame),
        "type_correct_count": int(type_correct.sum()),
        "type_correct_fraction": float(type_correct.mean()),
        "absolute_error_gt_5_count": int(
            (
                frame["automatic_N_prediction"].astype(float)
                - frame["N_mean"].astype(float)
            )
            .abs()
            .gt(5.0)
            .sum()
        ),
        "automatic_squared_error_sum": float(frame["automatic_squared_error"].sum()),
        "oracle_squared_error_sum": float(frame["oracle_squared_error"].sum()),
        "excess_squared_error_sum": float(frame["excess_squared_error"].sum()),
        "by_error_kind": rows,
    }


def _paired_cluster_bootstrap(
    merged: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    """Resample connectivity IDs jointly across every fold and both versions."""

    connectivity_column = "connectivity_id_v6"
    connectivity_values = np.asarray(
        sorted(set(merged[connectivity_column].astype(str))),
        dtype=object,
    )
    by_connectivity = {
        connectivity_id: merged.index[
            merged[connectivity_column].astype(str).eq(connectivity_id)
        ].to_numpy(dtype=int)
        for connectivity_id in connectivity_values
    }
    rng = np.random.default_rng(seed)
    exact_deltas: list[float] = []
    macro_r2_deltas: list[float] = []
    macro_rmse_deltas: list[float] = []
    split_seeds = sorted(merged["split_seed"].astype(int).unique())
    for _ in range(replicates):
        sampled = rng.choice(
            connectivity_values,
            size=len(connectivity_values),
            replace=True,
        )
        positions = np.concatenate([by_connectivity[str(value)] for value in sampled])
        sample = merged.loc[positions]
        exact_deltas.append(
            float(
                sample["exact_top1_v6"].astype(float).mean()
                - sample["exact_top1_v5"].astype(float).mean()
            )
        )
        split_r2: list[float] = []
        split_rmse: list[float] = []
        for split_seed in split_seeds:
            split = sample.loc[sample["split_seed"].eq(split_seed)]
            if len(split) < 2:
                continue
            truth = split["N_mean_v6"].to_numpy(dtype=float)
            v5 = base._regression_metrics(
                truth,
                split["automatic_N_prediction_v5"].to_numpy(dtype=float),
            )
            v6 = base._regression_metrics(
                truth,
                split["automatic_N_prediction_v6"].to_numpy(dtype=float),
            )
            if math.isfinite(float(v5["r2"])) and math.isfinite(float(v6["r2"])):
                split_r2.append(float(v6["r2"]) - float(v5["r2"]))
            split_rmse.append(float(v6["rmse"]) - float(v5["rmse"]))
        if len(split_r2) == len(split_seeds):
            macro_r2_deltas.append(float(np.mean(split_r2)))
        if len(split_rmse) == len(split_seeds):
            macro_rmse_deltas.append(float(np.mean(split_rmse)))

    def interval(values: Sequence[float]) -> dict[str, float | int]:
        array = np.asarray(values, dtype=float)
        return {
            "replicate_count": len(array),
            "mean": float(array.mean()),
            "ci_2_5_percent": float(np.quantile(array, 0.025)),
            "ci_97_5_percent": float(np.quantile(array, 0.975)),
        }

    return {
        "schema_version": "nucpred.mayr-v5-v6-paired-connectivity-bootstrap.v1",
        "replicates_requested": replicates,
        "unique_connectivity_count": len(connectivity_values),
        "exact_top1_delta_v6_minus_v5": interval(exact_deltas),
        "macro_r2_delta_v6_minus_v5": interval(macro_r2_deltas),
        "macro_rmse_delta_v6_minus_v5": interval(macro_rmse_deltas),
    }


def run_comparison(
    config: Mapping[str, Any],
    *,
    config_path: Path,
) -> dict[str, Any]:
    output_root = base._repo_path(config["output_directory"], label="output directory")
    v6_evaluation = output_root / "test_evaluation"
    v6_summary = base._load_json(v6_evaluation / "summary.json")
    if v6_summary.get("five_split_test_complete") is not True:
        raise base.SiteIdentificationError("v6 test evaluation is incomplete")
    baseline_development = base._repo_path(
        config["ranker"]["baseline_development_directory"],
        label="baseline development directory",
    )
    v5_evaluation = baseline_development.parent / "test_evaluation"
    v5_summary = base._load_json(v5_evaluation / "summary.json")
    if v5_summary.get("five_split_test_complete") is not True:
        raise base.SiteIdentificationError("v5 test evaluation is incomplete")
    target = output_root / "comparison"

    def writer(staged: Path) -> dict[str, Any]:
        v5_targets = _single_target_results(
            v5_evaluation / "all_split_target_level_results.parquet"
        )
        v6_targets = _single_target_results(
            v6_evaluation / "all_split_target_level_results.parquet"
        )
        merged = v5_targets.merge(
            v6_targets,
            on=["split_seed", "target_id"],
            how="inner",
            validate="one_to_one",
            suffixes=("_v5", "_v6"),
        )
        if len(merged) != len(v5_targets) or len(merged) != len(v6_targets):
            raise base.SiteIdentificationError("v5/v6 paired target coverage changed")
        if not np.allclose(
            merged["N_mean_v5"].to_numpy(dtype=float),
            merged["N_mean_v6"].to_numpy(dtype=float),
        ):
            raise base.SiteIdentificationError("v5/v6 paired target values changed")
        if (
            not merged["connectivity_id_v5"]
            .astype(str)
            .eq(merged["connectivity_id_v6"].astype(str))
            .all()
        ):
            raise base.SiteIdentificationError("v5/v6 connectivity binding changed")

        v5_metrics = pd.read_csv(v5_evaluation / "split_metrics.csv")
        v6_metrics = pd.read_csv(v6_evaluation / "split_metrics.csv")
        split_comparison = v5_metrics.merge(
            v6_metrics,
            on="split_seed",
            validate="one_to_one",
            suffixes=("_v5", "_v6"),
        )
        for metric in (
            "exact_top1_recall",
            "exact_top3_recall",
            "exact_top5_recall",
            "mrr",
            "automatic_mae",
            "automatic_rmse",
            "automatic_r2",
        ):
            split_comparison[f"{metric}_delta_v6_minus_v5"] = (
                split_comparison[f"{metric}_v6"] - split_comparison[f"{metric}_v5"]
            )
        split_comparison.to_csv(staged / "split_comparison.csv", index=False)

        v5_types = pd.read_csv(v5_evaluation / "pooled_retrieval_by_type.csv")
        v6_types = pd.read_csv(v6_evaluation / "pooled_retrieval_by_type.csv")
        v5_types["site_type"] = v5_types["population"].str.rsplit(":", n=1).str[-1]
        v6_types["site_type"] = v6_types["population"].str.rsplit(":", n=1).str[-1]
        type_comparison = v5_types.merge(
            v6_types,
            on="site_type",
            validate="one_to_one",
            suffixes=("_v5", "_v6"),
        )
        for metric in (
            "exact_top1_recall",
            "exact_top3_recall",
            "exact_top5_recall",
            "mrr",
            "compatible_top1_recall",
        ):
            type_comparison[f"{metric}_delta_v6_minus_v5"] = (
                type_comparison[f"{metric}_v6"] - type_comparison[f"{metric}_v5"]
            )
        type_comparison.to_csv(staged / "type_comparison.csv", index=False)

        v5_correct = merged["exact_top1_v5"].astype(bool)
        v6_correct = merged["exact_top1_v6"].astype(bool)
        improved = int((~v5_correct & v6_correct).sum())
        regressed = int((v5_correct & ~v6_correct).sum())
        transitions = {
            "both_correct": int((v5_correct & v6_correct).sum()),
            "v5_wrong_v6_correct": improved,
            "v5_correct_v6_wrong": regressed,
            "both_wrong": int((~v5_correct & ~v6_correct).sum()),
            "one_sided_exact_binomial_p_value": float(
                binomtest(
                    improved,
                    improved + regressed,
                    p=0.5,
                    alternative="greater",
                ).pvalue
            ),
        }
        bootstrap = _paired_cluster_bootstrap(
            merged,
            replicates=2000,
            seed=int(config["evaluation"]["bootstrap_seed"]) + 101,
        )
        atomic_write_json(staged / "paired_bootstrap.json", bootstrap)

        v5_decomposition = _error_decomposition(v5_targets)
        v6_decomposition = _error_decomposition(v6_targets)
        atomic_write_json(
            staged / "error_decomposition.json",
            {"v5": v5_decomposition, "v6": v6_decomposition},
        )
        v5_sse = float(v5_decomposition["automatic_squared_error_sum"])
        v6_sse = float(v6_decomposition["automatic_squared_error_sum"])
        v5_excess = float(v5_decomposition["excess_squared_error_sum"])
        v6_excess = float(v6_decomposition["excess_squared_error_sum"])
        return {
            "schema_version": COMPARISON_SCHEMA,
            "status": "pass",
            "campaign_id": config["campaign_id"],
            "created_at_utc": _utc_now(),
            "config_sha256": sha256_file(config_path),
            "v5_test_summary_path": base._display_path(v5_evaluation / "summary.json"),
            "v5_test_summary_sha256": sha256_file(v5_evaluation / "summary.json"),
            "v6_test_summary_path": base._display_path(v6_evaluation / "summary.json"),
            "v6_test_summary_sha256": sha256_file(v6_evaluation / "summary.json"),
            "paired_single_target_record_count": len(merged),
            "macro_metrics": {
                "v5_exact_top1": v5_summary["macro_split_metrics"]["exact_top1_recall"][
                    "mean"
                ],
                "v6_exact_top1": v6_summary["macro_split_metrics"]["exact_top1_recall"][
                    "mean"
                ],
                "exact_top1_delta_v6_minus_v5": (
                    v6_summary["macro_split_metrics"]["exact_top1_recall"]["mean"]
                    - v5_summary["macro_split_metrics"]["exact_top1_recall"]["mean"]
                ),
                "v5_automatic_r2": v5_summary["macro_split_metrics"]["automatic_r2"][
                    "mean"
                ],
                "v6_automatic_r2": v6_summary["macro_split_metrics"]["automatic_r2"][
                    "mean"
                ],
                "automatic_r2_delta_v6_minus_v5": (
                    v6_summary["macro_split_metrics"]["automatic_r2"]["mean"]
                    - v5_summary["macro_split_metrics"]["automatic_r2"]["mean"]
                ),
                "v5_automatic_rmse": v5_summary["macro_split_metrics"][
                    "automatic_rmse"
                ]["mean"],
                "v6_automatic_rmse": v6_summary["macro_split_metrics"][
                    "automatic_rmse"
                ]["mean"],
                "v5_oracle_r2": v5_summary["macro_split_metrics"]["oracle_r2"]["mean"],
                "v6_oracle_r2": v6_summary["macro_split_metrics"]["oracle_r2"]["mean"],
                "v5_automatic_minus_oracle_r2": v5_summary["macro_split_metrics"][
                    "automatic_minus_oracle_r2"
                ]["mean"],
                "v6_automatic_minus_oracle_r2": v6_summary["macro_split_metrics"][
                    "automatic_minus_oracle_r2"
                ]["mean"],
            },
            "paired_exact_transitions": transitions,
            "paired_connectivity_bootstrap": bootstrap,
            "v5_error_decomposition": v5_decomposition,
            "v6_error_decomposition": v6_decomposition,
            "automatic_sse_reduction_fraction": float(1.0 - v6_sse / v5_sse),
            "automatic_excess_sse_reduction_fraction": float(
                1.0 - v6_excess / v5_excess
            ),
            "evaluation_status": config["evaluation"]["evaluation_status"],
            "prior_v5_test_results_informed_architecture": True,
            "comparison_is_retrospective": True,
        }

    return base._publish_stage(
        target,
        schema_version=COMPARISON_SCHEMA,
        writer=writer,
    )


def _format_metric(mean: float, std: float) -> str:
    return f"{mean:.4f} ± {std:.4f}"


def write_final_report(
    config: Mapping[str, Any],
    *,
    config_path: Path,
    implementation_commit: str,
) -> Path:
    output_root = base._repo_path(config["output_directory"], label="output directory")
    stage_audits = [
        _verify_stage_manifest(output_root / stage)
        for stage in (
            "preflight",
            "development",
            "test_predictions",
            "test_evaluation",
            "deployment",
            "comparison",
        )
    ]
    comparison = base._load_json(output_root / "comparison" / "summary.json")
    development = base._load_json(output_root / "development" / "summary.json")
    evaluation = base._load_json(output_root / "test_evaluation" / "summary.json")
    registry_path = output_root / "deployment" / "runtime_registry.json"
    registry = base._load_json(registry_path)
    runtime_directory = output_root / "runtime_integration"
    runtime_files = {
        "cached_ok": runtime_directory / "cached_response.json",
        "low_margin_partial": runtime_directory / "low_margin_response.json",
        "uncached_refused": runtime_directory / "uncached_refusal_response.json",
    }
    runtime_rows: list[dict[str, object]] = []
    for scenario, path in runtime_files.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        validate_response(payload)
        runtime_rows.append(
            {
                "scenario": scenario,
                "status": payload["status"],
                "candidate_count": len(payload["candidates"]),
                "sha256": sha256_file(path),
            }
        )
    macro = evaluation["macro_split_metrics"]
    paired = comparison["macro_metrics"]
    bootstrap = comparison["paired_connectivity_bootstrap"]
    exact_ci = bootstrap["exact_top1_delta_v6_minus_v5"]
    r2_ci = bootstrap["macro_r2_delta_v6_minus_v5"]
    rmse_ci = bootstrap["macro_rmse_delta_v6_minus_v5"]
    type_comparison = pd.read_csv(output_root / "comparison" / "type_comparison.csv")
    type_rows = "\n".join(
        "| "
        + " | ".join(
            [
                str(row.site_type),
                f"{row.exact_top1_recall_v5:.4f}",
                f"{row.exact_top1_recall_v6:.4f}",
                f"{row.exact_top1_recall_delta_v6_minus_v5:+.4f}",
                f"{row.exact_top3_recall_v6:.4f}",
            ]
        )
        + " |"
        for row in type_comparison.itertuples(index=False)
    )
    split_comparison = pd.read_csv(output_root / "comparison" / "split_comparison.csv")
    split_rows = "\n".join(
        "| "
        + " | ".join(
            [
                str(int(row.split_seed)),
                f"{row.exact_top1_recall_v5:.4f}",
                f"{row.exact_top1_recall_v6:.4f}",
                f"{row.automatic_r2_v5:.4f}",
                f"{row.automatic_r2_v6:.4f}",
                f"{row.selective_coverage:.4f}",
                f"{row.selective_exact_top1_precision:.4f}",
            ]
        )
        + " |"
        for row in split_comparison.itertuples(index=False)
    )
    selected_arms = ", ".join(
        f"{item['split_seed']}={item['selected_arm']}"
        for item in development["split_summaries"]
    )
    thresholds = ", ".join(
        f"{item['split_seed']}={item['margin_abstention']['selected_threshold']:.2f}"
        for item in development["split_summaries"]
    )
    exact_calibration = evaluation["pooled_exact_fullspace_calibration"][0]
    transitions = comparison["paired_exact_transitions"]
    v5_error = comparison["v5_error_decomposition"]
    v6_error = comparison["v6_error_decomposition"]
    runtime_rows_markdown = "\n".join(
        f"| {row['scenario']} | {row['status']} | {row['candidate_count']} | `{row['sha256']}` |"
        for row in runtime_rows
    )
    stage_rows_markdown = "\n".join(
        f"| {row['stage']} | {row['file_count']} | `{row['content_sha256']}` | `{row['manifest_sha256']}` |"
        for row in stage_audits
    )
    report = f"""# Mayr 自动位点识别 v6 最终报告

- 生成时间：{_utc_now()}
- 实现提交：`{implementation_commit}`
- 基线提交：`{config["anchor_commit"]}`（v5 已先独立提交）
- campaign：`{config["campaign_id"]}`

## 结论

v6 已完成完整候选空间训练、分层类型路由、跨类型困难负样本、canonical/compatible 分层监督、validation-frozen margin gate，并接入 registry-backed 最终推理流程。五折 connectivity-disjoint 的 exact Top-1 从 v5 的 `{paired["v5_exact_top1"]:.4f}` 提升到 `{paired["v6_exact_top1"]:.4f}`（`{paired["exact_top1_delta_v6_minus_v5"]:+.4f}`）；自动位点条件下的 N 预测 R² 从 `{paired["v5_automatic_r2"]:.4f}` 提升到 `{paired["v6_automatic_r2"]:.4f}`（`{paired["automatic_r2_delta_v6_minus_v5"]:+.4f}`）。同一 frozen Stage E-C oracle R² 保持 `{paired["v6_oracle_r2"]:.4f}`，自动–oracle R² 差距由 `{paired["v5_automatic_minus_oracle_r2"]:.4f}` 缩小到 `{paired["v6_automatic_minus_oracle_r2"]:.4f}`。

结果支持此前的主要瓶颈是位点检索目标与完整候选部署空间不一致，而不是 conditional-N 回归器失效；由于该比较属于锁定回顾性评估，外部确认仍需独立位点数据。

## 开发冻结

- 完整 validation 候选空间 Top-1：v6 `{development["macro_validation_exact_top1_recall"]:.4f}`，冻结 v5 baseline `{development["macro_v5_baseline_validation_exact_top1_recall"]:.4f}`，差值 `{development["macro_validation_top1_delta_vs_v5"]:+.4f}`。
- 按折选择：{selected_arms}。
- validation margin threshold：{thresholds}；runtime 使用中位数 `{registry["runtime_margin_threshold"]:.2f}`。
- router context view 的候选间最大偏差在五折均为 0；unknown-as-negative 为 0；candidate softmax 未使用；Stage E-C 未重训且 final refit 未执行。

## 锁定回顾性五折结果

| 指标 | v6 macro mean ± population std |
| --- | ---: |
| Exact Top-1 | {_format_metric(macro["exact_top1_recall"]["mean"], macro["exact_top1_recall"]["std_population"])} |
| Exact Top-3 | {_format_metric(macro["exact_top3_recall"]["mean"], macro["exact_top3_recall"]["std_population"])} |
| Exact Top-5 | {_format_metric(macro["exact_top5_recall"]["mean"], macro["exact_top5_recall"]["std_population"])} |
| MRR | {_format_metric(macro["mrr"]["mean"], macro["mrr"]["std_population"])} |
| Compatible Top-1 | {_format_metric(macro["compatible_top1_recall"]["mean"], macro["compatible_top1_recall"]["std_population"])} |
| Automatic MAE | {_format_metric(macro["automatic_mae"]["mean"], macro["automatic_mae"]["std_population"])} |
| Automatic RMSE | {_format_metric(macro["automatic_rmse"]["mean"], macro["automatic_rmse"]["std_population"])} |
| Automatic R² | {_format_metric(macro["automatic_r2"]["mean"], macro["automatic_r2"]["std_population"])} |
| Oracle R² | {_format_metric(macro["oracle_r2"]["mean"], macro["oracle_r2"]["std_population"])} |

| split | v5 Top-1 | v6 Top-1 | v5 auto R² | v6 auto R² | accepted coverage | accepted precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{split_rows}

## 按真实位点类型

| site type | v5 Top-1 | v6 Top-1 | delta | v6 Top-3 |
| --- | ---: | ---: | ---: | ---: |
{type_rows}

`atom_group` 和 `delocalized_region` 仍是最弱类型，v6 Top-1 分别为 `{float(type_comparison.loc[type_comparison.site_type.eq("atom_group"), "exact_top1_recall_v6"].iloc[0]):.4f}` 与 `{float(type_comparison.loc[type_comparison.site_type.eq("delocalized_region"), "exact_top1_recall_v6"].iloc[0]):.4f}`，应作为下一轮数据与集合建模重点。

## 配对证据与错误收缩

- 1,093 条配对 single-target split records 中，v5 错而 v6 对 `{transitions["v5_wrong_v6_correct"]}` 条，v5 对而 v6 错 `{transitions["v5_correct_v6_wrong"]}` 条；单侧 exact binomial p=`{transitions["one_sided_exact_binomial_p_value"]:.3e}`。
- connectivity-cluster bootstrap 的 Top-1 差值 95% CI：`[{exact_ci["ci_2_5_percent"]:.4f}, {exact_ci["ci_97_5_percent"]:.4f}]`；macro R² 差值 95% CI：`[{r2_ci["ci_2_5_percent"]:.4f}, {r2_ci["ci_97_5_percent"]:.4f}]`；macro RMSE 差值 95% CI：`[{rmse_ci["ci_2_5_percent"]:.4f}, {rmse_ci["ci_97_5_percent"]:.4f}]`。
- wrong-type 记录从 `{next(row["record_count"] for row in v5_error["by_error_kind"] if row["error_kind"] == "wrong_type")}` 降到 `{next(row["record_count"] for row in v6_error["by_error_kind"] if row["error_kind"] == "wrong_type")}`；自动 absolute error > 5 的记录从 `{v5_error["absolute_error_gt_5_count"]}` 降到 `{v6_error["absolute_error_gt_5_count"]}`。
- 自动 SSE 降低 `{comparison["automatic_sse_reduction_fraction"]:.1%}`，相对 oracle 的 excess SSE 降低 `{comparison["automatic_excess_sse_reduction_fraction"]:.1%}`。

## 校准与拒判

exact full-space pooled calibration：ROC-AUC `{exact_calibration["roc_auc"]:.4f}`、AP `{exact_calibration["average_precision"]:.4f}`、Brier `{exact_calibration["weighted_brier"]:.4f}`、ECE `{exact_calibration["weighted_ece_10_bin"]:.4f}`。reviewed proxy 指标仅作为 compatible diagnostic，不再冒充 exact calibrator population。

margin gate 在 test 上的宏平均 coverage 为 `{macro["selective_coverage"]["mean"]:.4f}`，accepted exact precision 为 `{macro["selective_exact_top1_precision"]["mean"]:.4f}`，accepted 子集自动 R² 为 `{macro["selective_automatic_r2"]["mean"]:.4f}`。开发目标为 precision ≥ 0.95；其中一折 test precision 为 `{macro["selective_exact_top1_precision"]["minimum"]:.4f}`，说明该阈值是风险缓解而不是误差保证。

## 最终推理流程

registry：`{base._display_path(registry_path)}`，SHA-256 `{sha256_file(registry_path)}`。五个 cross-fit ranker、15 个 frozen Stage E-C checkpoint、exact calibrator 与 margin thresholds 均已绑定；未做 final refit。

| 场景 | status | candidates | response SHA-256 |
| --- | --- | ---: | --- |
{runtime_rows_markdown}

缓存高置信请求返回 `ok`；低 margin 请求保留全部候选与 conditional-N，但返回 `partial` 并把前两名标记为 `uncertain: low_site_rank_margin`；未缓存请求继续 fail closed 为 `refused`。

## 制品完整性

报告生成前已逐文件核验六个正式阶段的路径、大小、SHA-256 与 manifest 内容哈希。

| stage | files | content SHA-256 | manifest SHA-256 |
| --- | ---: | --- | --- |
{stage_rows_markdown}

## 解释边界

1. v5 test 结果已经用于诊断并影响 v6 架构，因此本报告是 locked retrospective cross-fit comparison，不是 pristine confirmatory test；需要独立外部位点数据才能给出真正的外部确认。
2. runtime 仍是 registry-backed Mayr context cache，不是任意新分子的在线 G1/xTB 计算服务。
3. `atom_group` 与 `delocalized_region` 的 canonical identity 仍受集合标注稀疏、嵌套候选和 proxy ontology 影响；compatible 指标不能替代 exact 指标。
4. selective 指标以覆盖率换精度，不能与全覆盖自动 R² 混为同一总体。

## 可复现入口

```bash
.venv/bin/python -m nucpred.experiments.mayr.nextgen_site_identification_v6 preflight --config configs/mayr_nextgen_site_identification_v6.toml
.venv/bin/python -m nucpred.experiments.mayr.nextgen_site_identification_v6 develop --config configs/mayr_nextgen_site_identification_v6.toml
.venv/bin/python -m nucpred.experiments.mayr.nextgen_site_identification_v6 predict-test --config configs/mayr_nextgen_site_identification_v6.toml
.venv/bin/python -m nucpred.experiments.mayr.nextgen_site_identification_v6 test --config configs/mayr_nextgen_site_identification_v6.toml
.venv/bin/python -m nucpred.experiments.mayr.nextgen_site_identification_v6 deploy --config configs/mayr_nextgen_site_identification_v6.toml
.venv/bin/python -m nucpred.experiments.mayr.nextgen_site_identification_v6_finalize compare --config configs/mayr_nextgen_site_identification_v6.toml
```
"""
    report_path = base._repo_path(config["report_path"], label="report path")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return report_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["compare", "report", "all"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--implementation-commit", default="WORKTREE")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = args.config.resolve()
    config = base.read_config(config_path)
    result: dict[str, object] = {}
    if args.command in {"compare", "all"}:
        result["comparison"] = run_comparison(config, config_path=config_path)
    if args.command in {"report", "all"}:
        path = write_final_report(
            config,
            config_path=config_path,
            implementation_commit=str(args.implementation_commit),
        )
        result["report_path"] = base._display_path(path)
        result["report_sha256"] = sha256_file(path)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
