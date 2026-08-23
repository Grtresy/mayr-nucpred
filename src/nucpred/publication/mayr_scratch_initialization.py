"""Create deterministic, unoptimized initialization checkpoints for ablation."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import shutil
import tempfile
import tomllib
from typing import Any, Sequence

import torch

from nucpred.core.files import atomic_write_json, sha256_file
from nucpred.project import get_project_layout
from nucpred.training.mayr_node_xtb_pretraining import (
    MaskingConfig,
    MayrNodeXtbPretrainingModel,
    PretrainingLossConfig,
    PretrainingNormalization,
    _canonical_json_sha256,
    _tensor_mapping_sha256,
    load_pretraining_checkpoint,
    save_pretraining_checkpoint,
)


ROOT = get_project_layout().root
DEFAULT_CONFIG = ROOT / "configs/mayr_n_publication_scratch_initialization_v1.toml"
SCHEMA = "nucpred.mayr-n-publication-scratch-initialization.v1"


class ScratchInitializationError(RuntimeError):
    """Raised when deterministic scratch initialization cannot be proven."""


def _declare_zero_pretraining_tasks(checkpoint_path: Path) -> None:
    """Make the scratch-only task contract explicit without changing frozen code.

    The shared checkpoint writer intentionally substitutes the production task
    set for a falsy task sequence.  Changing that implementation would alter the
    source hash bound into every frozen production checkpoint.  The scratch
    campaign therefore rewrites only its newly created payload and recomputes
    the payload's own full-contract hash before validation.
    """

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    contract = payload.get("contract")
    hashes = payload.get("contract_hashes")
    if not isinstance(contract, dict) or not isinstance(hashes, dict):
        raise ScratchInitializationError("Scratch checkpoint lacks contract evidence")
    contract["tasks"] = []
    hashes_without_full = {
        str(key): str(value)
        for key, value in hashes.items()
        if str(key) != "full_contract_sha256"
    }
    hashes["full_contract_sha256"] = _canonical_json_sha256(
        {"contract": contract, "hashes": hashes_without_full}
    )
    temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(checkpoint_path)


def _project_path(value: object, *, label: str) -> Path:
    path = (ROOT / str(value)).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as exc:
        raise ScratchInitializationError(f"{label} escapes project root") from exc
    return path


def read_config(path: str | Path = DEFAULT_CONFIG) -> tuple[dict[str, Any], Path]:
    resolved = Path(path).resolve()
    config = tomllib.loads(resolved.read_text(encoding="utf-8"))
    if (
        config.get("schema_version") != SCHEMA
        or config.get("component") != "no_pretraining"
        or config.get("mayr_labels_used") is not False
        or config.get("esnuel_optimization_steps") != 0
    ):
        raise ScratchInitializationError("Scratch initialization contract changed")
    return config, resolved


def build(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, object]:
    config, resolved = read_config(config_path)
    root = _project_path(config["output_root"], label="output root")
    aggregate_path = root / "aggregate" / "aggregate_summary.json"
    if aggregate_path.is_file():
        existing = json.loads(aggregate_path.read_text(encoding="utf-8"))
        if existing.get("status") == "complete":
            return existing
    if root.exists():
        raise ScratchInitializationError(f"Partial scratch output exists: {root}")
    reference_path = _project_path(
        config["reference_checkpoint"], label="reference checkpoint"
    )
    reference_sha256 = sha256_file(reference_path)
    if reference_sha256 != str(config["reference_checkpoint_sha256"]):
        raise ScratchInitializationError("Reference checkpoint drifted")
    reference = load_pretraining_checkpoint(reference_path)
    normalization = PretrainingNormalization.from_json(reference["normalization"])
    masking = MaskingConfig(**reference["masking_config"])
    loss = PretrainingLossConfig(**reference["loss_config"])
    root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=f".{root.name}-", dir=root.parent))
    summaries: list[dict[str, object]] = []
    try:
        for seed in map(int, config["downstream_initialization_seeds"]):
            directory = staging_root / f"seed-{seed}"
            directory.mkdir(parents=True)
            model = MayrNodeXtbPretrainingModel(
                num_solvents=10,
                hidden_dim=128,
                layers=4,
                dropout=0.1,
                init_seed=seed,
            )
            initial_state_sha256 = _tensor_mapping_sha256(model.state_dict())
            checkpoint_path = directory / "initialization.pt"
            save_pretraining_checkpoint(
                checkpoint_path,
                model,
                optimizer=None,
                history=(),
                normalization=normalization,
                masking=masking,
                loss_config=loss,
                dataset_contract_hashes={
                    "reference_checkpoint_sha256": reference_sha256,
                    "config_sha256": sha256_file(resolved),
                    "esnuel_records_loaded": "0",
                },
                selection={
                    "metric": "none",
                    "trained_epochs": 0,
                    "optimization_steps": 0,
                    "audit_test_used_for_selection": False,
                },
                audit_metrics={},
                gradient_audit={},
                tasks=(),
                variant_contract={
                    "variant": "no_pretraining",
                    "deterministic_scratch_initialization": True,
                    "pretraining_optimization_steps": 0,
                    "esnuel_records_loaded": 0,
                    "runtime_local4_input": True,
                    "runtime_global6_input": True,
                    "local4_pretraining_target": False,
                    "global6_pretraining_target": False,
                    "mayr_labels_used": False,
                    "config_sha256": sha256_file(resolved),
                },
            )
            _declare_zero_pretraining_tasks(checkpoint_path)
            payload = load_pretraining_checkpoint(checkpoint_path)
            if (
                int(payload["init_seed"]) != seed
                or payload["history"]
                or payload["contract"]["tasks"]
                or payload["backbone_state_sha256"]
                != _tensor_mapping_sha256(model.backbone.state_dict())
            ):
                raise ScratchInitializationError(
                    f"Scratch checkpoint changed deterministic state for seed {seed}"
                )
            summary = {
                "schema_version": "nucpred.mayr-n-scratch-initialization-job.v1",
                "status": "pass",
                "component": "no_pretraining",
                "downstream_initialization_seed": seed,
                "pretraining_optimization_steps": 0,
                "esnuel_records_loaded": 0,
                "pretraining_tasks": [],
                "initial_full_state_sha256": initial_state_sha256,
                "backbone_state_sha256": payload["backbone_state_sha256"],
                "checkpoint_path": f"seed-{seed}/initialization.pt",
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "mayr_labels_used": False,
            }
            atomic_write_json(directory / "summary.json", summary, ensure_ascii=False)
            summaries.append(summary)
        aggregate_dir = staging_root / "aggregate"
        aggregate_dir.mkdir()
        result = {
            "schema_version": "nucpred.mayr-n-scratch-initialization-aggregate.v1",
            "status": "complete",
            "component": "no_pretraining",
            "seed_count": len(summaries),
            "pretraining_optimization_steps": 0,
            "esnuel_records_loaded": 0,
            "pretraining_tasks": [],
            "mayr_labels_used": False,
            "reference_checkpoint_path": str(config["reference_checkpoint"]),
            "reference_checkpoint_sha256": reference_sha256,
            "config_sha256": sha256_file(resolved),
            "checkpoint_bindings": [
                {
                    "downstream_initialization_seed": summary[
                        "downstream_initialization_seed"
                    ],
                    "pretraining_seed": summary["downstream_initialization_seed"],
                    "path": summary["checkpoint_path"],
                    "sha256": summary["checkpoint_sha256"],
                }
                for summary in summaries
            ],
            "completed_at_utc": datetime.now(UTC).isoformat(),
        }
        atomic_write_json(
            aggregate_dir / "aggregate_summary.json", result, ensure_ascii=False
        )
        staging_root.replace(root)
    except BaseException:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    print(json.dumps(build(args.config), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
