"""Matched ESNUEL pretraining with every xTB input and objective removed."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import tomllib
from typing import Any

import numpy as np
import pandas as pd
import torch

from nucpred.core.files import atomic_write_json, sha256_file
from nucpred.project import get_project_layout
from nucpred.training.mayr_node_xtb_pretraining import (
    EsnuelNodeXtbExample,
    MaskingConfig,
    MayrNodeXtbPretrainingModel,
    PretrainingLossConfig,
    _source_id_sha256,
    _tensor_mapping_sha256,
    fit_pretraining_normalization,
    load_pretraining_examples,
    save_pretraining_checkpoint,
    train_pretraining_pilot,
)


ROOT = get_project_layout().root
DEFAULT_CONFIG = ROOT / "configs/mayr_n_publication_no_xtb_pretraining_v1.toml"
SCHEMA = "nucpred.mayr-n-publication-no-xtb-pretraining.v1"


class NoXtbPretrainingError(RuntimeError):
    """Raised when the matched no-xTB pretraining contract is violated."""


def _read_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    resolved = Path(path).resolve()
    with resolved.open("rb") as handle:
        config = tomllib.load(handle)
    if config.get("schema_version") != SCHEMA or config.get("variant") != "without_xtb":
        raise NoXtbPretrainingError("Unsupported no-xTB pretraining config")
    ablation = config["input_ablation"]
    required = {
        "node_local4_values": "zero",
        "node_local4_availability": "all_false",
        "molecule_global6_values": "zero",
        "molecule_global6_availability": "all_false",
    }
    if any(ablation.get(key) != value for key, value in required.items()):
        raise NoXtbPretrainingError("No-xTB input masking contract changed")
    if bool(config.get("mayr_labels_used")):
        raise NoXtbPretrainingError("Mayr labels cannot enter ablation pretraining")
    return config, resolved


def _project_path(value: object, *, label: str) -> Path:
    path = (ROOT / str(value)).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as exc:
        raise NoXtbPretrainingError(f"{label} escapes the project root") from exc
    return path


def remove_xtb(example: EsnuelNodeXtbExample) -> EsnuelNodeXtbExample:
    """Return an example with both xTB families unavailable and zero-valued."""

    return replace(
        example,
        local_values=np.zeros_like(example.local_values, dtype=float),
        local_mask=np.zeros_like(example.local_mask, dtype=bool),
        global_values=np.zeros_like(example.global_values, dtype=float),
        global_mask=np.zeros_like(example.global_mask, dtype=bool),
    )


def _load_examples(
    config: Mapping[str, Any],
) -> tuple[list[EsnuelNodeXtbExample], dict[str, str]]:
    paths = {
        name: _project_path(config[name], label=name)
        for name in ("records", "atom_features", "molecule_features")
    }
    examples = load_pretraining_examples(
        pd.read_parquet(paths["records"]),
        pd.read_parquet(paths["atom_features"]),
        pd.read_parquet(paths["molecule_features"]),
    )
    masked = [remove_xtb(example) for example in examples]
    if any(example.local_mask.any() or example.global_mask.any() for example in masked):
        raise NoXtbPretrainingError("xTB availability survived the input transform")
    hashes = {f"{name}_sha256": sha256_file(path) for name, path in paths.items()}
    return masked, hashes


def _state_change_audit(
    trained: MayrNodeXtbPretrainingModel,
    *,
    seed: int,
) -> dict[str, object]:
    initial = MayrNodeXtbPretrainingModel(
        num_solvents=10,
        hidden_dim=128,
        layers=4,
        dropout=0.1,
        init_seed=seed,
    )
    before = initial.state_dict()
    after = trained.state_dict()
    module_changed: dict[str, bool] = {}
    for module in (
        "backbone.node_encoder",
        "backbone.local_encoder",
        "backbone.edge_encoder",
        "backbone.message_layers",
        "backbone.site_head",
        "backbone.global_xtb_encoder",
        "local_reconstruction_head",
        "global_reconstruction_head",
        "mca_head",
        "gcs_head",
    ):
        keys = [key for key in before if key.startswith(f"{module}.")]
        module_changed[module] = bool(keys) and any(
            not torch.equal(before[key].cpu(), after[key].detach().cpu())
            for key in keys
        )
    if module_changed["backbone.global_xtb_encoder"]:
        raise NoXtbPretrainingError(
            "Global xTB encoder changed despite zero xTB objectives"
        )
    if (
        module_changed["local_reconstruction_head"]
        or module_changed["global_reconstruction_head"]
    ):
        raise NoXtbPretrainingError("An excluded xTB reconstruction head changed")
    for required in (
        "backbone.node_encoder",
        "backbone.edge_encoder",
        "backbone.message_layers",
        "backbone.site_head",
        "mca_head",
        "gcs_head",
    ):
        if not module_changed[required]:
            raise NoXtbPretrainingError(f"Retained path did not train: {required}")
    return {
        "module_changed_from_deterministic_initialization": module_changed,
        "initial_state_sha256": _tensor_mapping_sha256(before),
        "trained_state_sha256": _tensor_mapping_sha256(after),
        "excluded_global_xtb_encoder_bitwise_unchanged": True,
        "excluded_reconstruction_heads_bitwise_unchanged": True,
    }


def run_seed(
    seed: int,
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    device: str | None = None,
) -> dict[str, object]:
    started = time.perf_counter()
    config, resolved = _read_config(config_path)
    if seed not in tuple(map(int, config["seeds"])):
        raise NoXtbPretrainingError("Unregistered no-xTB pretraining seed")
    selected_device = torch.device(device or str(config["device"]))
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise NoXtbPretrainingError("CUDA was requested but is unavailable")
    output = _project_path(config["output_root"], label="output_root") / f"seed-{seed}"
    if (output / "summary.json").is_file():
        payload = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        if payload.get("status") == "pass":
            return payload
    if output.exists():
        raise NoXtbPretrainingError(f"Partial output exists: {output}")
    examples, data_hashes = _load_examples(config)
    partitions = {
        role: [example for example in examples if example.pretraining_role == role]
        for role in ("train", "validation", "audit_test")
    }
    if {name: len(values) for name, values in partitions.items()} != {
        "train": 33475,
        "validation": 8243,
        "audit_test": 6197,
    }:
        raise NoXtbPretrainingError("ESNUEL partition counts changed")
    normalization = fit_pretraining_normalization(
        partitions["train"],
        allow_empty_xtb_features=True,
    )
    if (
        any(normalization.local_median)
        or any(normalization.local_mean)
        or tuple(normalization.local_scale) != (1.0, 1.0, 1.0, 1.0)
        or any(normalization.global_median)
        or any(normalization.global_mean)
        or tuple(normalization.global_scale) != (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    ):
        raise NoXtbPretrainingError("No-xTB normalization is not the zero/one identity")
    ablation = config["input_ablation"]
    masking = MaskingConfig(
        node_categorical_probability=0.15,
        edge_categorical_probability=0.15,
        local_probability=float(ablation["local_mask_probability"]),
        global_probability=float(ablation["global_mask_probability"]),
    )
    loss = PretrainingLossConfig(
        node_categorical_weight=1.0,
        edge_categorical_weight=1.0,
        local_weight=float(ablation["local_reconstruction_weight"]),
        global_weight=float(ablation["global_reconstruction_weight"]),
        mca_weight=1.0,
        gcs_weight=1.0,
        site_weight=0.5,
        ranking_weight=0.25,
        site_temperature=0.5,
        ranking_margin=0.0,
    )
    result = train_pretraining_pilot(
        partitions["train"],
        partitions["validation"],
        audit_test_examples=partitions["audit_test"],
        normalization=normalization,
        init_seed=seed,
        epochs=int(config["epochs"]),
        min_epochs=int(config["minimum_epochs"]),
        patience=int(config["early_stopping_patience"]),
        min_delta=float(config["minimum_validation_delta"]),
        batch_size=int(config["batch_size"]),
        learning_rate=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        device=selected_device,
        masking=masking,
        loss_config=loss,
        hidden_dim=128,
        layers=4,
        dropout=0.1,
        require_gradient_gate=False,
    )
    change_audit = _state_change_audit(result.model, seed=seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".seed-{seed}-", dir=output.parent))
    try:
        checkpoint_path = staging / "best.pt"
        retained_tasks = tuple(map(str, config["retained_tasks"]["names"]))
        variant_contract = {
            "variant": "without_xtb",
            "runtime_local4_input": False,
            "runtime_global6_input": False,
            "local4_pretraining_target": False,
            "global6_pretraining_target": False,
            "mayr_labels_used": False,
            "input_transform_source_sha256": sha256_file(Path(__file__).resolve()),
            "config_sha256": sha256_file(resolved),
        }
        save_pretraining_checkpoint(
            checkpoint_path,
            result.model,
            optimizer=result.optimizer,
            history=result.history,
            normalization=normalization,
            masking=masking,
            loss_config=loss,
            dataset_contract_hashes=data_hashes
            | {
                "runner_sha256": sha256_file(Path(__file__).resolve()),
                "config_sha256": sha256_file(resolved),
            },
            selection={
                "metric": "validation_total_without_xtb_objectives",
                "mode": "min",
                "best_epoch": result.best_epoch,
                "best_validation_total": result.best_validation_total,
                "trained_epochs": len(result.history),
                "minimum_epochs": int(config["minimum_epochs"]),
                "patience": int(config["early_stopping_patience"]),
                "audit_test_used_for_selection": False,
                "partition_counts": {
                    name: len(values) for name, values in partitions.items()
                },
                "partition_source_id_sha256": {
                    name: _source_id_sha256(values)
                    for name, values in partitions.items()
                },
            },
            audit_metrics=result.audit_metrics,
            gradient_audit={},
            tasks=retained_tasks,
            variant_contract=variant_contract,
        )
        pd.DataFrame(result.history).to_csv(
            staging / "loss_curves.csv", index=False, lineterminator="\n"
        )
        atomic_write_json(
            staging / "history.json",
            {
                "history": list(result.history),
                "selection": {
                    "best_epoch": result.best_epoch,
                    "best_validation_total": result.best_validation_total,
                },
                "audit_test_metrics": dict(result.audit_metrics),
                "state_change_audit": change_audit,
            },
            ensure_ascii=False,
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        summary = {
            "schema_version": "nucpred.mayr-n-no-xtb-pretraining-job.v1",
            "status": "pass",
            "campaign_id": config["campaign_id"],
            "variant": "without_xtb",
            "seed": seed,
            "device": str(selected_device),
            "partition_counts": {
                name: len(values) for name, values in partitions.items()
            },
            "best_epoch": result.best_epoch,
            "best_validation_total": result.best_validation_total,
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "backbone_state_sha256": checkpoint["backbone_state_sha256"],
            "contract_hashes": checkpoint["contract_hashes"],
            "retained_tasks": list(retained_tasks),
            "input_ablation": dict(ablation),
            "state_change_audit": change_audit,
            "mayr_labels_used": False,
            "audit_test_used_for_selection": False,
            "config_sha256": sha256_file(resolved),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "wall_seconds": time.perf_counter() - started,
            "completed_at_utc": datetime.now(UTC).isoformat(),
        }
        atomic_write_json(staging / "summary.json", summary, ensure_ascii=False)
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return summary


def aggregate(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, object]:
    config, resolved = _read_config(config_path)
    root = _project_path(config["output_root"], label="output_root")
    summaries = []
    for seed in map(int, config["seeds"]):
        path = root / f"seed-{seed}/summary.json"
        if not path.is_file():
            raise NoXtbPretrainingError(f"Missing seed output: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "pass":
            raise NoXtbPretrainingError(f"Seed did not pass: {path}")
        summaries.append(payload)
    aggregate_root = root / "aggregate"
    if aggregate_root.exists():
        existing = json.loads(
            (aggregate_root / "aggregate_summary.json").read_text(encoding="utf-8")
        )
        if existing.get("status") == "complete":
            return existing
        raise NoXtbPretrainingError(f"Partial aggregate exists: {aggregate_root}")
    aggregate_root.mkdir(parents=True)
    result = {
        "schema_version": "nucpred.mayr-n-no-xtb-pretraining-aggregate.v1",
        "status": "complete",
        "campaign_id": config["campaign_id"],
        "variant": "without_xtb",
        "dataset_record_count": 47915,
        "seed_count": len(summaries),
        "seed_summaries": summaries,
        "checkpoint_bindings": [
            {
                "seed": summary["seed"],
                "path": f"seed-{summary['seed']}/best.pt",
                "sha256": summary["checkpoint_sha256"],
            }
            for summary in summaries
        ],
        "all_xtb_inputs_masked": True,
        "all_xtb_objectives_removed": True,
        "mayr_labels_used": False,
        "config_sha256": sha256_file(resolved),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    atomic_write_json(
        aggregate_root / "aggregate_summary.json", result, ensure_ascii=False
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device")
    subparsers = parser.add_subparsers(dest="command", required=True)
    seed_parser = subparsers.add_parser("seed")
    seed_parser.add_argument("--seed", type=int, required=True)
    subparsers.add_parser("aggregate")
    args = parser.parse_args(argv)
    if args.command == "seed":
        result = run_seed(args.seed, config_path=args.config, device=args.device)
    else:
        result = aggregate(args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
