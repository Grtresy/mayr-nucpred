"""Aggregate the mandatory pre-full-training site-N stage gate."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import pandas as pd

from nucpred.artifacts.managed import (
    ArtifactClassification,
    ManagedRun,
)
from nucpred.core.files import atomic_write_json, sha256_file
from nucpred.datasets.catalog import DatasetCatalog
from nucpred.project import get_project_layout

from .site_n import (
    DEFAULT_CONFIG,
    EXPERIMENT_ID,
    SiteNCampaignError,
    _display_path,
    _read_config,
    _write_manifest,
)


_LAYOUT = get_project_layout()
ROOT = _LAYOUT.root
PRETRAINING_SCOPES = ("pilot1024", "pilot4096")
PRETRAINING_SEEDS = (31001, 31002, 31003)
STAGE_GATE_RUN_ID = "mayr-site-n-stage-gate-20260726-v1"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SiteNCampaignError(f"Expected JSON object: {path}")
    return payload


def _verify_run_manifest(directory: Path) -> dict[str, object]:
    manifest_path = directory / "run_manifest.json"
    manifest = _load_json(manifest_path)
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise SiteNCampaignError(f"Run manifest has no files: {directory}")
    checked = 0
    for relative, raw_entry in files.items():
        if not isinstance(raw_entry, Mapping):
            raise SiteNCampaignError("Run manifest entry is not a mapping")
        path = directory / str(relative)
        if (
            not path.is_file()
            or path.stat().st_size != int(raw_entry["bytes"])
            or sha256_file(path) != str(raw_entry["sha256"])
        ):
            raise SiteNCampaignError(f"Run asset changed: {path}")
        checked += 1
    return {
        "status": "pass",
        "verified_file_count": checked,
        "manifest_sha256": sha256_file(manifest_path),
    }


def _source_parity(
    expected: Mapping[str, object],
    paths: Mapping[str, Path],
) -> dict[str, object]:
    observed = {name: sha256_file(path) for name, path in paths.items()}
    missing = sorted(set(expected) - set(observed))
    unexpected = sorted(set(observed) - set(expected))
    mismatched = sorted(
        name
        for name in set(expected) & set(observed)
        if str(expected[name]) != observed[name]
    )
    return {
        "status": (
            "pass"
            if not missing and not unexpected and not mismatched
            else "fail"
        ),
        "expected": dict(expected),
        "observed": observed,
        "missing": missing,
        "unexpected": unexpected,
        "mismatched": mismatched,
    }


def _database_gate(
    dataset_directory: Path,
) -> dict[str, object]:
    summary = _load_json(dataset_directory / "summary.json")
    coverage = summary["candidate_coverage_by_type"]
    overlap = summary["pretraining_connectivity_overlap"]
    passed = (
        int(summary["source_measurement_count"]) == 1136
        and int(summary["formal_target_count"]) == 1108
        and all(
            math.isclose(float(value["fraction"]), 1.0)
            for value in coverage.values()
        )
        and all(int(value) == 0 for value in overlap.values())
        and summary["site_probability_normalization"] is False
        and summary["unmeasured_candidates_are_negative"] is False
    )
    return {
        "status": "pass" if passed else "fail",
        "raw_measurement_count": int(summary["source_measurement_count"]),
        "species_count": int(summary["species_count"]),
        "context_count": int(summary["context_count"]),
        "target_count": int(summary["formal_target_count"]),
        "unresolved_measurement_count": int(
            summary["unresolved_measurement_count"]
        ),
        "candidate_coverage_by_type": coverage,
        "pretraining_connectivity_overlap": overlap,
        "multi_site_context_count": int(
            summary["multi_site_context_count"]
        ),
        "multi_site_pair_count": int(summary["multi_site_pair_count"]),
        "site_probability_normalization": False,
        "unmeasured_candidates_are_negative": False,
    }


def _pretraining_rows(
    campaign_root: Path,
    *,
    config_path: Path,
    config: Mapping[str, Any],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    verifications: dict[str, object] = {}
    for scope in PRETRAINING_SCOPES:
        for seed in PRETRAINING_SEEDS:
            directory = (
                campaign_root
                / "pretraining"
                / scope
                / f"seed-{seed}"
            )
            summary = _load_json(directory / "summary.json")
            verification = _verify_run_manifest(directory)
            dataset_directory = (
                ROOT
                / str(
                    config["pretraining"][f"{scope}_directory"]
                )
            ).resolve()
            parity = _source_parity(
                summary["contract"]["source_hashes"],
                {
                    "config": config_path,
                    "runner": (
                        ROOT
                        / "src/nucpred/experiments/mayr"
                        / "site_n_pretraining.py"
                    ),
                    "pretraining_model": (
                        ROOT
                        / "src/nucpred/training"
                        / "mayr_site_n_pretraining.py"
                    ),
                    "downstream_model": (
                        ROOT
                        / "src/nucpred/training/mayr_site_n.py"
                    ),
                    "esnuel_loader": (
                        ROOT
                        / "src/nucpred/training"
                        / "mayr_node_xtb_pretraining.py"
                    ),
                    "dataset_manifest": (
                        dataset_directory / "dataset_manifest.json"
                    ),
                },
            )
            verifications[f"{scope}/seed-{seed}"] = {
                "asset_verification": verification,
                "source_parity": parity,
            }
            required = summary["gradient_audit"]["required_paths"]
            unused = summary["gradient_audit"][
                "unsupported_and_reset_paths_unused"
            ]
            transfer = summary["transfer_audit"]
            rows.append(
                {
                    "scope": scope,
                    "seed": seed,
                    "status": summary["status"],
                    "epochs_completed": int(
                        summary["epochs_completed"]
                    ),
                    "best_epoch": int(summary["best_epoch"]),
                    "epoch1_validation_total": float(
                        summary["epoch1_validation_total"]
                    ),
                    "best_validation_total": float(
                        summary["best_validation_total"]
                    ),
                    "relative_validation_improvement": float(
                        summary["relative_validation_improvement"]
                    ),
                    "audit_total": float(
                        summary["audit_metrics"]["total"]
                    ),
                    "audit_metrics_finite": bool(
                        summary["audit_metrics_finite"]
                    ),
                    "required_gradient_paths_pass": bool(
                        required and all(required.values())
                    ),
                    "unsupported_paths_unused": bool(
                        unused and all(unused.values())
                    ),
                    "strict_transfer_pass": (
                        transfer["status"] == "pass"
                        and bool(transfer["exact_transfer"])
                        and bool(transfer["reset_modules_unchanged"])
                    ),
                    "source_parity_pass": parity["status"] == "pass",
                    "transferred_parameter_tensors": int(
                        transfer["transferred_parameter_tensors"]
                    ),
                    "transferred_parameter_numel": int(
                        transfer["transferred_parameter_numel"]
                    ),
                    "checkpoint_transferable_state_sha256": str(
                        summary[
                            "checkpoint_transferable_state_sha256"
                        ]
                    ),
                    "wall_seconds": float(summary["wall_seconds"]),
                }
            )
    frame = pd.DataFrame(rows)
    scope_summary = {
        scope: {
            "passing_seeds": int(
                frame.loc[frame["scope"].eq(scope), "status"]
                .eq("pass")
                .sum()
            ),
            "mean_relative_validation_improvement": float(
                frame.loc[
                    frame["scope"].eq(scope),
                    "relative_validation_improvement",
                ].mean()
            ),
            "mean_best_validation_total": float(
                frame.loc[
                    frame["scope"].eq(scope),
                    "best_validation_total",
                ].mean()
            ),
            "mean_audit_total": float(
                frame.loc[
                    frame["scope"].eq(scope),
                    "audit_total",
                ].mean()
            ),
        }
        for scope in PRETRAINING_SCOPES
    }
    return rows, {
        "run_asset_verifications": verifications,
        "scope_summary": scope_summary,
    }


def _report_markdown(summary: Mapping[str, Any]) -> str:
    database = summary["database_gate"]
    tiny = summary["tiny_overfit_gate"]
    scopes = summary["pretraining_gate"]["scope_summary"]
    rows = summary["pretraining_gate"]["runs"]
    lines = [
        "# 位点条件 Mayr N 实验：强制阶段门",
        "",
        f"- 技术门状态：`{summary['technical_gate_status']}`",
        f"- 审批状态：`{summary['approval_status']}`",
        "- 当前动作：暂停；未经用户明确批准，不启动 47,915 条三随机种子全量预训练或正式微调矩阵。",
        "",
        "## 数据库门",
        "",
        (
            f"1136 条原始测量保留为 {database['species_count']} 个物种、"
            f"{database['context_count']} 个分子-溶剂上下文和 "
            f"{database['target_count']} 个聚合位点 N 目标；"
            f"{database['unresolved_measurement_count']} 条 unresolved 测量保留证据但不正式监督。"
        ),
        (
            "atom、bond、delocalized_region、atom_group、"
            "transferable_h_group 五类正式位点的候选覆盖率均为 100%；"
            "ESNUEL pilot/full 与 Mayr 的 connectivity 重叠均为 0。"
        ),
        "",
        "## 新模型门",
        "",
        (
            f"tiny-overfit：{tiny['context_count']} 个上下文、"
            f"{tiny['target_count']} 个目标，N MAE={tiny['metrics']['mae']:.4f}，"
            f"R²={tiny['metrics']['r2']:.4f}；多位点 ΔN MAE="
            f"{tiny['metrics']['multi_site']['delta_mae']:.4f}，排序准确率="
            f"{tiny['metrics']['multi_site']['ranking_accuracy']:.1%}。"
        ),
        "模型逐候选独立输出标量 N，不存在位点 softmax；未测量候选不作为负样本。",
        "",
        "## 同构预训练门",
        "",
        "| scope | seed | best epoch | validation 改善 | audit total | 梯度 | 严格迁移 |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {scope} | {seed} | {best_epoch} | {improvement:.1%} | "
            "{audit:.4f} | {gradient} | {transfer} |".format(
                scope=row["scope"],
                seed=row["seed"],
                best_epoch=row["best_epoch"],
                improvement=row["relative_validation_improvement"],
                audit=row["audit_total"],
                gradient="pass"
                if row["required_gradient_paths_pass"]
                else "fail",
                transfer="pass" if row["strict_transfer_pass"] else "fail",
            )
        )
    lines.extend(
        [
            "",
            (
                "pilot1024 三种子平均验证改善 "
                f"{scopes['pilot1024']['mean_relative_validation_improvement']:.1%}；"
                "pilot4096 三种子平均验证改善 "
                f"{scopes['pilot4096']['mean_relative_validation_improvement']:.1%}。"
            ),
            (
                "ESNUEL 只监督有真实标签的重原子 atom 查询；显式 H 只参与图卷积和"
                "节点/边/local4/global6 重建，bond/region/group/H-group 未制造伪 MCA/GCS。"
            ),
            (
                "每个检查点仅迁移共享图骨干、global6 编码器、共享位点读取器和 atom "
                "适配器；其他类型适配器、溶剂、charge 与 N 回归头保持重新初始化。"
            ),
            "",
            "## 已知事件与处置",
            "",
            (
                "首轮 tiny-overfit 暴露方向型排序损失与点回归冲突；失败资产已保留。"
                "损失改为预测 ΔN 对真实 ΔN 的一致性损失后通过门禁。"
            ),
            (
                "pilot4096 的 mutable G1 缓存因代码哈希漂移被严格拒绝。没有重算或"
                "放宽校验；4096 资产由已验证 frozen full 按 source_id 做纯子集投影，"
                "并通过目录、组件、映射和覆盖率校验。"
            ),
            "",
            "## 阶段决策",
            "",
            (
                "所有技术门均已通过，建议下一阶段运行 47,915 条、3 个随机种子的"
                "全量同构预训练，随后执行受控下游对比：scratch、旧预训练共享编码器"
                "初始化、新同构预训练初始化。当前严格停在审批点。"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def build_stage_gate(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
) -> dict[str, object]:
    config = _read_config(config_path)
    config_file = Path(config_path).resolve()
    campaign_root = (ROOT / str(config["output_root"])).resolve()
    target = campaign_root / "stage_gate"
    if target.exists():
        summary = _load_json(target / "summary.json")
        if (
            summary.get("technical_gate_status") == "pass"
            and summary.get("approval_status") == "awaiting_user_approval"
        ):
            return summary
        raise SiteNCampaignError("Existing stage gate is stale or incomplete")

    dataset_directory = (ROOT / str(config["dataset_directory"])).resolve()
    database_gate = _database_gate(dataset_directory)
    tiny_directory = campaign_root / "tiny_overfit"
    tiny = _load_json(tiny_directory / "gate_summary.json")
    tiny_verification = _verify_run_manifest(tiny_directory)
    tiny_source_parity = _source_parity(
        tiny["contract"]["source_hashes"],
        {
            "config": config_file,
            "runner": ROOT / "src/nucpred/experiments/mayr/site_n.py",
            "model": ROOT / "src/nucpred/training/mayr_site_n.py",
            "dataset_builder": (
                ROOT / "src/nucpred/datasets/mayr_site_n.py"
            ),
            "all_atom_graph": (
                ROOT / "src/nucpred/features/all_atom_graph.py"
            ),
            "dataset_manifest": (
                dataset_directory / "dataset_manifest.json"
            ),
        },
    )
    pretraining_rows, pretraining_aux = _pretraining_rows(
        campaign_root,
        config_path=config_file,
        config=config,
    )
    pretraining_frame = pd.DataFrame(pretraining_rows)
    minimum_seeds = int(
        config["stage_gate"]["minimum_improved_pretraining_seeds"]
    )
    pretraining_pass = all(
        int(
            pretraining_frame.loc[
                pretraining_frame["scope"].eq(scope)
                & pretraining_frame["status"].eq("pass")
                & pretraining_frame["audit_metrics_finite"]
                & pretraining_frame["required_gradient_paths_pass"]
                & pretraining_frame["unsupported_paths_unused"]
                & pretraining_frame["strict_transfer_pass"]
                & pretraining_frame["source_parity_pass"]
            ].shape[0]
        )
        >= minimum_seeds
        for scope in PRETRAINING_SCOPES
    )
    catalog = DatasetCatalog()
    dataset_verifications = {
        dataset_id: catalog.verify(dataset_id)
        for dataset_id in (
            "mayr-site-n-20260726-v1",
            "esnuel-d-node-xtb-pretraining-20260726-v1-pilot1024",
            "esnuel-d-node-xtb-pretraining-20260726-v1-pilot4096",
            "esnuel-d-node-xtb-pretraining-20260726-v1-full",
        )
    }
    full_output = campaign_root / "pretraining" / "full"
    formal_output = campaign_root / "formal"
    forbidden_outputs_absent = (
        not full_output.exists() and not formal_output.exists()
    )
    technical_pass = (
        database_gate["status"] == "pass"
        and tiny["status"] == "pass"
        and float(tiny["metrics"]["mae"])
        <= float(tiny["maximum_N_MAE"])
        and tiny["site_probability_normalization"] is False
        and tiny["model_has_site_head"] is False
        and tiny_source_parity["status"] == "pass"
        and pretraining_pass
        and forbidden_outputs_absent
        and all(
            item["status"] == "pass"
            for item in dataset_verifications.values()
        )
    )
    if not technical_pass:
        raise SiteNCampaignError("Mandatory stage-gate condition failed")
    summary: dict[str, object] = {
        "schema_version": "nucpred.mayr-site-n-stage-gate.v1",
        "experiment_id": EXPERIMENT_ID,
        "technical_gate_status": "pass",
        "approval_status": "awaiting_user_approval",
        "authorized_next_stage": False,
        "database_gate": database_gate,
        "tiny_overfit_gate": {
            **tiny,
            "asset_verification": tiny_verification,
            "source_parity": tiny_source_parity,
        },
        "pretraining_gate": {
            "status": "pass",
            "minimum_improved_seeds": minimum_seeds,
            "runs": pretraining_rows,
            **pretraining_aux,
        },
        "dataset_catalog_verifications": dataset_verifications,
        "full_pretraining_output_absent": not full_output.exists(),
        "formal_finetuning_output_absent": not formal_output.exists(),
        "full_pretraining_forbidden_before_approval": True,
        "maximum_parallel_gpu_processes": 3,
        "recommended_next_stage": (
            "47,915-record matched pretraining with seeds "
            "31001/31002/31003, then controlled downstream comparison"
        ),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".stage-gate.staging-", dir=target.parent)
    )
    try:
        pretraining_frame.to_csv(
            staging / "pretraining_metrics.csv",
            index=False,
            lineterminator="\n",
        )
        atomic_write_json(
            staging / "summary.json", summary, ensure_ascii=False
        )
        (staging / "report.md").write_text(
            _report_markdown(summary),
            encoding="utf-8",
        )
        _write_manifest(staging)
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return summary


def _classification(relative: Path) -> ArtifactClassification:
    text = relative.as_posix()
    suffix = relative.suffix.lower()
    protocol_evidence = (
        text
        in {
            "report.md",
            "run_config.json",
            "environment.json",
            "campaign/stage_gate/summary.json",
            "campaign/tiny_overfit/gate_summary.json",
            "campaign/tiny_overfit/selection.json",
        }
        or text.endswith("/gradient_audit.json")
        or text.endswith("/transfer_audit.json")
        or text.endswith("/selection.json")
        or text.endswith("/normalization.json")
        or text.endswith("/preprocessor.json")
        or text.endswith("/run_manifest.json")
    )
    if protocol_evidence:
        role = "protocol_evidence"
    elif suffix == ".pt":
        role = "model_checkpoint"
    elif "prediction" in text:
        role = "predictions"
    elif "metric" in text:
        role = "metrics"
    elif "loss_curves" in text:
        role = "training_diagnostic"
    elif suffix == ".log":
        role = "execution_log"
    else:
        role = "supporting_artifact"
    media_type = {
        ".json": "application/json",
        ".csv": "text/csv",
        ".md": "text/markdown",
        ".parquet": "application/vnd.apache.parquet",
        ".pt": "application/octet-stream",
        ".toml": "application/toml",
        ".log": "text/plain",
    }.get(suffix)
    return ArtifactClassification(
        role=role,
        protocol_evidence=protocol_evidence,
        media_type=media_type,
    )


def finalize_stage_gate(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
) -> dict[str, object]:
    config_file = Path(config_path).resolve()
    config = _read_config(config_file)
    summary = build_stage_gate(config_path=config_file)
    if (
        summary["technical_gate_status"] != "pass"
        or summary["approval_status"] != "awaiting_user_approval"
        or summary["authorized_next_stage"] is not False
    ):
        raise SiteNCampaignError("Stage gate is not approval-blocked and passing")
    campaign_root = (ROOT / str(config["output_root"])).resolve()
    dataset_directory = (ROOT / str(config["dataset_directory"])).resolve()
    pretraining_directories = {
        scope: (
            ROOT
            / str(config["pretraining"][f"{scope}_directory"])
        ).resolve()
        for scope in PRETRAINING_SCOPES
    }
    full_directory = (
        ROOT / str(config["pretraining"]["full_directory"])
    ).resolve()
    managed = ManagedRun.start(
        experiment="mayr-site-n-stage-gate",
        protocol="mayr-site-n-independent-stage-gate-v1",
        dataset_ids=[
            "mayr-site-n-20260726-v1",
            "esnuel-d-node-xtb-pretraining-20260726-v1-pilot1024",
            "esnuel-d-node-xtb-pretraining-20260726-v1-pilot4096",
            "esnuel-d-node-xtb-pretraining-20260726-v1-full",
        ],
        config_path=config_file,
        source_paths=[
            Path(__file__).resolve(),
            ROOT / "src/nucpred/experiments/mayr/site_n.py",
            ROOT / "src/nucpred/experiments/mayr/site_n_pretraining.py",
            ROOT / "src/nucpred/training/mayr_site_n.py",
            ROOT / "src/nucpred/training/mayr_site_n_pretraining.py",
            ROOT / "src/nucpred/datasets/mayr_site_n.py",
            ROOT / "src/nucpred/datasets/esnuel_frozen_subset.py",
            ROOT / "src/nucpred/features/all_atom_graph.py",
            ROOT / "docs/protocols/mayr-site-n-independent.md",
            dataset_directory / "dataset_manifest.json",
            pretraining_directories["pilot1024"]
            / "dataset_manifest.json",
            pretraining_directories["pilot4096"]
            / "dataset_manifest.json",
            full_directory / "dataset_manifest.json",
        ],
        command=("nucpred", "report", "mayr-site-n-stage-gate"),
        run_id=STAGE_GATE_RUN_ID,
        metadata={
            "stage": "mandatory_pre_full_gate",
            "technical_gate_status": "pass",
            "approval_status": "awaiting_user_approval",
            "authorized_next_stage": False,
            "full_pretraining_started": False,
            "formal_finetuning_started": False,
            "maximum_parallel_gpu_processes": 3,
        },
    )
    try:
        shutil.copytree(campaign_root, managed.directory / "campaign")
        atomic_write_json(
            managed.directory / "run_config.json",
            {
                "schema_version": "nucpred.mayr-site-n-managed-config.v1",
                "config_path": _display_path(config_file),
                "config_sha256": sha256_file(config_file),
                "dataset_manifest_sha256": sha256_file(
                    dataset_directory / "dataset_manifest.json"
                ),
                "pretraining_dataset_manifest_sha256": {
                    scope: sha256_file(
                        directory / "dataset_manifest.json"
                    )
                    for scope, directory in pretraining_directories.items()
                },
                "full_dataset_manifest_sha256": sha256_file(
                    full_directory / "dataset_manifest.json"
                ),
                "technical_gate_status": "pass",
                "approval_status": "awaiting_user_approval",
                "authorized_next_stage": False,
            },
            ensure_ascii=False,
        )
        tiny = summary["tiny_overfit_gate"]
        atomic_write_json(
            managed.directory / "environment.json",
            {
                "python": os.sys.version,
                "torch": tiny["torch_version"],
                "cuda": tiny["cuda_version"],
                "tiny_device": tiny["device"],
                "maximum_parallel_gpu_processes": 3,
            },
            ensure_ascii=False,
        )
        shutil.copy2(
            campaign_root / "stage_gate/report.md",
            managed.directory / "report.md",
        )
        managed.register_tree(_classification)
        record = managed.complete(
            required_artifacts=(
                "campaign/stage_gate/pretraining_metrics.csv",
                "campaign/tiny_overfit/predictions.parquet",
                (
                    "campaign/pretraining/pilot1024/"
                    "seed-31001/checkpoint.pt"
                ),
                (
                    "campaign/pretraining/pilot4096/"
                    "seed-31001/checkpoint.pt"
                ),
            ),
            required_protocol_evidence=(
                "report.md",
                "run_config.json",
                "environment.json",
                "campaign/stage_gate/summary.json",
                "campaign/tiny_overfit/gate_summary.json",
                (
                    "campaign/pretraining/pilot1024/"
                    "seed-31001/gradient_audit.json"
                ),
                (
                    "campaign/pretraining/pilot4096/"
                    "seed-31001/transfer_audit.json"
                ),
            ),
        )
    except BaseException as exc:
        managed.fail(f"{type(exc).__name__}: {exc}")
        raise
    return {
        "status": "pass",
        "run_id": managed.run_id,
        "run_directory": _display_path(managed.directory),
        "catalog_status": record["status"],
        "approval_status": "awaiting_user_approval",
        "authorized_next_stage": False,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--finalize", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    payload = (
        finalize_stage_gate(config_path=arguments.config)
        if arguments.finalize
        else build_stage_gate(config_path=arguments.config)
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
