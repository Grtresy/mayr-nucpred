"""Matched ESNUEL pretraining for one removed electronic feature family."""

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
    DEFAULT_PRETRAINING_TASKS,
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
DEFAULT_CONFIG = ROOT / "configs/mayr_n_publication_pretraining_no_local_electronic_v1.toml"
SCHEMA = "nucpred.mayr-n-publication-component-pretraining.v1"
SUPPORTED = frozenset({"no_local_electronic", "no_global_electronic"})


class ComponentPretrainingError(RuntimeError):
    """Raised when a matched component-pretraining contract is violated."""


def _project_path(value: object, *, label: str) -> Path:
    path = (ROOT / str(value)).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as exc:
        raise ComponentPretrainingError(f"{label} escapes project root") from exc
    return path


def read_config(path: str | Path = DEFAULT_CONFIG) -> tuple[dict[str, Any], Path]:
    resolved = Path(path).resolve()
    config = tomllib.loads(resolved.read_text(encoding="utf-8"))
    component = str(config.get("component"))
    if config.get("schema_version") != SCHEMA or component not in SUPPORTED:
        raise ComponentPretrainingError("Unsupported component-pretraining config")
    if config.get("mayr_labels_used") is not False:
        raise ComponentPretrainingError("Mayr labels cannot enter pretraining")
    removed = config["input_ablation"]
    expected_family = "local4" if component == "no_local_electronic" else "global6"
    if (
        removed.get("removed_family") != expected_family
        or removed.get("values") != "zero"
        or removed.get("availability") != "all_false"
        or float(removed.get("mask_probability", -1)) != 0.0
        or float(removed.get("reconstruction_weight", -1)) != 0.0
    ):
        raise ComponentPretrainingError("Removed-family contract changed")
    expected_tasks = set(DEFAULT_PRETRAINING_TASKS)
    expected_tasks.remove(
        "masked_local4_reconstruction_all_atoms"
        if expected_family == "local4"
        else "masked_global6_reconstruction"
    )
    if set(map(str, config["retained_tasks"]["names"])) != expected_tasks:
        raise ComponentPretrainingError("Retained pretraining task set changed")
    return config, resolved


def remove_component(
    example: EsnuelNodeXtbExample, *, component: str
) -> EsnuelNodeXtbExample:
    if component == "no_local_electronic":
        return replace(
            example,
            local_values=np.zeros_like(example.local_values, dtype=float),
            local_mask=np.zeros_like(example.local_mask, dtype=bool),
        )
    if component == "no_global_electronic":
        return replace(
            example,
            global_values=np.zeros_like(example.global_values, dtype=float),
            global_mask=np.zeros_like(example.global_mask, dtype=bool),
        )
    raise ComponentPretrainingError(f"Unsupported component: {component}")


def _load_examples(
    config: Mapping[str, Any], *, component: str
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
    transformed = [remove_component(example, component=component) for example in examples]
    if component == "no_local_electronic" and any(
        example.local_mask.any() or example.local_values.any()
        for example in transformed
    ):
        raise ComponentPretrainingError("Local electronic input survived masking")
    if component == "no_global_electronic" and any(
        example.global_mask.any() or example.global_values.any()
        for example in transformed
    ):
        raise ComponentPretrainingError("Global electronic input survived masking")
    return transformed, {
        f"{name}_sha256": sha256_file(path) for name, path in paths.items()
    }


def _module_change_audit(
    trained: MayrNodeXtbPretrainingModel, *, component: str, seed: int
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
    modules = (
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
    )
    changed = {
        module: any(
            not torch.equal(before[key].cpu(), after[key].detach().cpu())
            for key in before
            if key.startswith(f"{module}.")
        )
        for module in modules
    }
    excluded_head = (
        "local_reconstruction_head"
        if component == "no_local_electronic"
        else "global_reconstruction_head"
    )
    retained_head = (
        "global_reconstruction_head"
        if component == "no_local_electronic"
        else "local_reconstruction_head"
    )
    if changed[excluded_head]:
        raise ComponentPretrainingError("Excluded reconstruction head changed")
    for required in (
        "backbone.node_encoder",
        "backbone.edge_encoder",
        "backbone.message_layers",
        "backbone.site_head",
        retained_head,
        "mca_head",
        "gcs_head",
    ):
        if not changed[required]:
            raise ComponentPretrainingError(f"Retained module did not train: {required}")
    return {
        "module_changed_from_deterministic_initialization": changed,
        "excluded_reconstruction_head": excluded_head,
        "excluded_reconstruction_head_bitwise_unchanged": True,
        "initial_state_sha256": _tensor_mapping_sha256(before),
        "trained_state_sha256": _tensor_mapping_sha256(after),
    }


def run_seed(
    seed: int,
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    device: str | None = None,
) -> dict[str, object]:
    started = time.perf_counter()
    config, resolved = read_config(config_path)
    component = str(config["component"])
    if seed not in tuple(map(int, config["seeds"])):
        raise ComponentPretrainingError("Unregistered pretraining seed")
    selected_device = torch.device(device or str(config["device"]))
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise ComponentPretrainingError("CUDA was requested but is unavailable")
    output = _project_path(config["output_root"], label="output root") / f"seed-{seed}"
    if (output / "summary.json").is_file():
        existing = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        if existing.get("status") == "pass":
            return existing
    if output.exists():
        raise ComponentPretrainingError(f"Partial output exists: {output}")
    examples, data_hashes = _load_examples(config, component=component)
    partitions = {
        role: [example for example in examples if example.pretraining_role == role]
        for role in ("train", "validation", "audit_test")
    }
    if {name: len(values) for name, values in partitions.items()} != {
        "train": 33_475,
        "validation": 8_243,
        "audit_test": 6_197,
    }:
        raise ComponentPretrainingError("ESNUEL partitions changed")
    normalization = fit_pretraining_normalization(
        partitions["train"], allow_empty_xtb_features=True
    )
    if component == "no_local_electronic":
        identity = (
            not any(normalization.local_median)
            and not any(normalization.local_mean)
            and tuple(normalization.local_scale) == (1.0,) * 4
        )
    else:
        identity = (
            not any(normalization.global_median)
            and not any(normalization.global_mean)
            and tuple(normalization.global_scale) == (1.0,) * 6
        )
    if not identity:
        raise ComponentPretrainingError("Removed-family normalization is not identity")
    local_removed = component == "no_local_electronic"
    global_removed = component == "no_global_electronic"
    masking = MaskingConfig(
        node_categorical_probability=0.15,
        edge_categorical_probability=0.15,
        local_probability=0.0 if local_removed else 0.15,
        global_probability=0.0 if global_removed else 0.15,
    )
    loss = PretrainingLossConfig(
        node_categorical_weight=1.0,
        edge_categorical_weight=1.0,
        local_weight=0.0 if local_removed else 1.0,
        global_weight=0.0 if global_removed else 1.0,
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
    change_audit = _module_change_audit(
        result.model, component=component, seed=seed
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".seed-{seed}-", dir=output.parent))
    try:
        checkpoint_path = staging / "best.pt"
        retained_tasks = tuple(map(str, config["retained_tasks"]["names"]))
        variant_contract = {
            "variant": component,
            "runtime_local4_input": not local_removed,
            "runtime_global6_input": not global_removed,
            "local4_pretraining_target": not local_removed,
            "global6_pretraining_target": not global_removed,
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
                "metric": f"validation_total_{component}",
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
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        summary = {
            "schema_version": "nucpred.mayr-n-component-pretraining-job.v1",
            "status": "pass",
            "campaign_id": config["campaign_id"],
            "component": component,
            "seed": seed,
            "device": str(selected_device),
            "partition_counts": {
                name: len(values) for name, values in partitions.items()
            },
            "best_epoch": result.best_epoch,
            "best_validation_total": result.best_validation_total,
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "backbone_state_sha256": checkpoint["backbone_state_sha256"],
            "retained_tasks": list(retained_tasks),
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
    config, resolved = read_config(config_path)
    component = str(config["component"])
    root = _project_path(config["output_root"], label="output root")
    summaries = []
    for seed in map(int, config["seeds"]):
        path = root / f"seed-{seed}/summary.json"
        if not path.is_file():
            raise ComponentPretrainingError(f"Missing seed output: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "pass" or payload.get("component") != component:
            raise ComponentPretrainingError(f"Seed did not pass: {path}")
        summaries.append(payload)
    target = root / "aggregate"
    if target.exists():
        existing = json.loads(
            (target / "aggregate_summary.json").read_text(encoding="utf-8")
        )
        if existing.get("status") == "complete":
            return existing
        raise ComponentPretrainingError(f"Partial aggregate exists: {target}")
    target.mkdir(parents=True)
    result = {
        "schema_version": "nucpred.mayr-n-component-pretraining-aggregate.v1",
        "status": "complete",
        "campaign_id": config["campaign_id"],
        "component": component,
        "dataset_record_count": 47_915,
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
        "removed_input_and_reconstruction_objective_matched": True,
        "mayr_labels_used": False,
        "config_sha256": sha256_file(resolved),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    atomic_write_json(target / "aggregate_summary.json", result, ensure_ascii=False)
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
    result = (
        run_seed(args.seed, config_path=args.config, device=args.device)
        if args.command == "seed"
        else aggregate(args.config)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
