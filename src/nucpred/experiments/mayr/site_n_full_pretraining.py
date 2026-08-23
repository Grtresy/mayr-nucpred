"""Run approved full-corpus matched pretraining for the typed site-N model."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import ctypes
from datetime import UTC, datetime
import gc
import json
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import tempfile
import time
import tomllib
import traceback
from typing import Any

import pandas as pd

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from nucpred.artifacts.catalog import ArtifactCatalog
from nucpred.core.files import atomic_write_json, sha256_file
from nucpred.datasets.esnuel_d_node_xtb_pretraining import (
    verify_dataset as verify_esnuel_dataset,
)
from nucpred.project import get_project_layout
from nucpred.training.mayr_site_n_pretraining import (
    EsnuelNodeXtbExample,
    load_pretraining_examples,
    save_site_n_pretraining_checkpoint,
    train_site_n_pretraining,
    transfer_pretrained_backbone,
)

from .site_n import (
    EXPERIMENT_ID,
    SiteNCampaignError,
    _canonical_sha256,
    _device,
    _display_path,
    _read_config,
    _write_manifest,
)
from .site_n_approval import verify_approval
from .site_n_pretraining import (
    ALLOWED_SEEDS,
    _downstream_solvent_vocabulary,
    _loss_config,
    _make_downstream,
    _masking,
    _source_id_sha256,
)


_LAYOUT = get_project_layout()
ROOT = _LAYOUT.root
DEFAULT_FULL_CONFIG = ROOT / "configs/mayr_site_n_full_pretraining.toml"
FULL_CONFIG_SCHEMA = "nucpred.mayr-site-n-full-pretraining-config.v1"
FULL_SCOPE = "full"


def _read_full_config(
    path: str | Path = DEFAULT_FULL_CONFIG,
) -> dict[str, Any]:
    config_path = Path(path).resolve()
    payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != FULL_CONFIG_SCHEMA:
        raise SiteNCampaignError("Unsupported full-pretraining config")
    if payload.get("experiment_id") != EXPERIMENT_ID:
        raise SiteNCampaignError("Full-pretraining experiment identity changed")
    if payload.get("scope") != FULL_SCOPE:
        raise SiteNCampaignError("Full-pretraining scope changed")
    if (
        payload.get("dataset_id")
        != "esnuel-d-node-xtb-pretraining-20260726-v1-full"
    ):
        raise SiteNCampaignError("Full-pretraining dataset identity changed")
    if tuple(map(int, payload["initialization_seeds"])) != ALLOWED_SEEDS:
        raise SiteNCampaignError("Full-pretraining seed axis changed")
    if int(payload["maximum_parallel_gpu_processes"]) != 3:
        raise SiteNCampaignError("Full-pretraining GPU concurrency changed")
    counts = {
        name: int(value)
        for name, value in payload["expected_counts"].items()
    }
    if counts != {
        "total": 47915,
        "train": 33475,
        "validation": 8243,
        "audit_test": 6197,
    }:
        raise SiteNCampaignError("Full-pretraining role counts changed")
    optimization = payload["optimization"]
    if not (
        1
        <= int(optimization["minimum_epochs"])
        <= int(optimization["maximum_epochs"])
    ):
        raise SiteNCampaignError("Invalid full-pretraining epoch contract")
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SiteNCampaignError(f"Expected JSON object: {path}")
    return payload


def _authorization_gate(
    full_config: Mapping[str, Any],
) -> dict[str, object]:
    section = full_config["authorization"]
    approval_path = (ROOT / str(section["approval_artifact"])).resolve()
    verification = verify_approval(
        approval_path,
        required_statement=str(section["required_approval_statement"]),
        required_goal_thread_id=str(section["required_goal_thread_id"]),
        stage_gate_run_id=str(section["stage_gate_run_id"]),
    )
    approval = verification["approval"]
    if not isinstance(approval, Mapping):
        raise SiteNCampaignError("Approval payload is not a mapping")

    catalog = ArtifactCatalog()
    stage_gate_run_id = str(section["stage_gate_run_id"])
    stage_verification = catalog.verify(stage_gate_run_id)
    stage_directory = catalog.run_directory(stage_gate_run_id)
    summary_path = (
        stage_directory / "campaign" / "stage_gate" / "summary.json"
    )
    report_path = stage_directory / "report.md"
    catalog_path = (
        ROOT / "artifacts" / "catalog" / "runs" / f"{stage_gate_run_id}.json"
    )
    parity = {
        "stage_gate_summary": (
            approval.get("stage_gate_summary_sha256")
            == sha256_file(summary_path)
        ),
        "stage_gate_report": (
            approval.get("stage_gate_report_sha256")
            == sha256_file(report_path)
        ),
        "stage_gate_catalog_manifest": (
            approval.get("stage_gate_catalog_manifest_sha256")
            == sha256_file(catalog_path)
        ),
        "stage_gate_catalog_verification": (
            stage_verification["status"] == "pass"
            and stage_verification["run_status"] == "complete"
        ),
    }
    if not all(parity.values()):
        failed = sorted(name for name, passed in parity.items() if not passed)
        raise SiteNCampaignError(
            f"Approval-to-stage-gate parity failed: {failed}"
        )
    return {
        "status": "pass",
        "approval_path": _display_path(approval_path),
        "approval_sha256": sha256_file(approval_path),
        "approval": dict(approval),
        "stage_gate_parity": parity,
        "stage_gate_catalog_verification": stage_verification,
    }


def _load_full_scope(
    full_config: Mapping[str, Any],
) -> tuple[
    list[EsnuelNodeXtbExample],
    list[EsnuelNodeXtbExample],
    list[EsnuelNodeXtbExample],
    Path,
    dict[str, object],
    dict[str, object],
]:
    directory = (ROOT / str(full_config["dataset_directory"])).resolve()
    if not directory.is_dir():
        raise SiteNCampaignError(
            f"Full pretraining dataset is missing: {_display_path(directory)}"
        )
    verification = verify_esnuel_dataset(directory)
    records = pd.read_parquet(directory / "records.parquet")
    atoms = pd.read_parquet(directory / "atom_features.parquet")
    molecules = pd.read_parquet(directory / "molecule_features.parquet")
    examples = load_pretraining_examples(records, atoms, molecules)
    del records, atoms, molecules
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (AttributeError, OSError):
        pass

    roles = {
        role: [
            example
            for example in examples
            if example.pretraining_role == role
        ]
        for role in ("train", "validation", "audit_test")
    }
    observed = {
        "total": len(examples),
        **{role: len(values) for role, values in roles.items()},
    }
    expected = {
        name: int(value)
        for name, value in full_config["expected_counts"].items()
    }
    if observed != expected:
        raise SiteNCampaignError(
            f"Full-pretraining role counts changed: {observed}"
        )
    selection = {
        "schema_version": "nucpred.mayr-site-n-pretraining-selection.v1",
        "scope": FULL_SCOPE,
        "counts": observed,
        "source_id_sha256": {
            role: _source_id_sha256(
                [example.source_id for example in values]
            )
            for role, values in roles.items()
        },
        "audit_test_used_for_selection": False,
        "native_esnuel_roles_preserved": True,
        "mayr_connectivity_overlap_count": 0,
    }
    del examples
    gc.collect()
    return (
        roles["train"],
        roles["validation"],
        roles["audit_test"],
        directory,
        verification,
        selection,
    )


def _full_contract(
    *,
    base_config_path: Path,
    full_config_path: Path,
    dataset_directory: Path,
    approval_path: Path,
    seed: int,
) -> dict[str, object]:
    paths = {
        "base_config": base_config_path,
        "full_config": full_config_path,
        "full_runner": Path(__file__).resolve(),
        "pilot_runner": (
            ROOT / "src/nucpred/experiments/mayr/site_n_pretraining.py"
        ).resolve(),
        "pretraining_model": (
            ROOT / "src/nucpred/training/mayr_site_n_pretraining.py"
        ).resolve(),
        "downstream_model": (
            ROOT / "src/nucpred/training/mayr_site_n.py"
        ).resolve(),
        "esnuel_loader": (
            ROOT / "src/nucpred/training/mayr_node_xtb_pretraining.py"
        ).resolve(),
        "dataset_manifest": dataset_directory / "dataset_manifest.json",
        "approval": approval_path,
        "stage_gate_catalog": (
            ROOT
            / "artifacts/catalog/runs"
            / "mayr-site-n-stage-gate-20260726-v1.json"
        ),
    }
    contract: dict[str, object] = {
        "schema_version": "nucpred.mayr-site-n-full-pretraining-contract.v1",
        "experiment_id": EXPERIMENT_ID,
        "scope": FULL_SCOPE,
        "initialization_seed": int(seed),
        "source_hashes": {
            name: sha256_file(path) for name, path in paths.items()
        },
        "tasks": [
            "masked_node_categorical_reconstruction_all_atoms",
            "masked_edge_categorical_reconstruction",
            "masked_local4_reconstruction_all_atoms",
            "masked_global6_reconstruction",
            "heavy_atom_pointwise_mca",
            "heavy_atom_pointwise_gcs53",
            "within_molecule_mca_delta_ranking",
        ],
        "site_probability_normalization": False,
        "fake_proxy_site_types": [],
        "maximum_parallel_gpu_processes": 3,
        "stage_gate_approval_required": True,
        "stage_gate_approval_sha256": sha256_file(approval_path),
    }
    contract["contract_sha256"] = _canonical_sha256(contract)
    return contract


def _running_marker(
    output_directory: Path,
    *,
    seed: int,
) -> Path:
    return output_directory / f".seed-{seed}.running.json"


def _write_failure(
    output_directory: Path,
    *,
    seed: int,
    exc: BaseException,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        output_directory / f"seed-{seed}.failure.json",
        {
            "schema_version": "nucpred.mayr-site-n-full-failure.v1",
            "status": "failed",
            "seed": int(seed),
            "failed_at_utc": datetime.now(UTC).isoformat(),
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        },
        ensure_ascii=False,
    )


def _run_loaded_seed(
    *,
    seed: int,
    base_config: Mapping[str, Any],
    base_config_path: Path,
    full_config: Mapping[str, Any],
    full_config_path: Path,
    train: Sequence[EsnuelNodeXtbExample],
    validation: Sequence[EsnuelNodeXtbExample],
    audit_test: Sequence[EsnuelNodeXtbExample],
    dataset_directory: Path,
    dataset_verification: Mapping[str, object],
    selection: Mapping[str, object],
    authorization: Mapping[str, object],
    device: str,
) -> dict[str, object]:
    started = time.perf_counter()
    if seed not in ALLOWED_SEEDS:
        raise SiteNCampaignError(f"Unregistered full-pretraining seed: {seed}")
    selected_device = _device(device)
    if selected_device.type != "cuda":
        raise SiteNCampaignError("Approved full pretraining requires CUDA")

    approval_path = (
        ROOT
        / str(full_config["authorization"]["approval_artifact"])
    ).resolve()
    contract = _full_contract(
        base_config_path=base_config_path,
        full_config_path=full_config_path,
        dataset_directory=dataset_directory,
        approval_path=approval_path,
        seed=seed,
    )
    output_directory = (
        ROOT / str(full_config["output_directory"])
    ).resolve()
    target = output_directory / f"seed-{seed}"
    summary_path = target / "summary.json"
    if summary_path.is_file():
        existing = _load_json(summary_path)
        if (
            existing.get("status") == "pass"
            and existing.get("contract") == contract
        ):
            return existing
        raise SiteNCampaignError(
            f"Existing full/seed-{seed} result is stale"
        )
    if target.exists():
        raise SiteNCampaignError(
            f"Partial full/seed-{seed} directory exists"
        )

    output_directory.mkdir(parents=True, exist_ok=True)
    marker = _running_marker(output_directory, seed=seed)
    atomic_write_json(
        marker,
        {
            "schema_version": "nucpred.mayr-site-n-full-running.v1",
            "status": "running",
            "seed": int(seed),
            "pid": os.getpid(),
            "started_at_utc": datetime.now(UTC).isoformat(),
            "device": str(selected_device),
            "contract_sha256": contract["contract_sha256"],
        },
        ensure_ascii=False,
    )

    vocabulary = _downstream_solvent_vocabulary(base_config)
    masking = _masking(base_config)
    loss_config = _loss_config(base_config)
    optimization = full_config["optimization"]
    model_section = base_config["model"]
    try:
        result = train_site_n_pretraining(
            train,
            validation,
            audit_test_examples=audit_test,
            num_solvents=len(vocabulary.tokens),
            init_seed=seed,
            epochs=int(optimization["maximum_epochs"]),
            minimum_epochs=int(optimization["minimum_epochs"]),
            patience=int(optimization["early_stopping_patience"]),
            minimum_delta=float(
                optimization["minimum_validation_delta"]
            ),
            batch_size=int(optimization["batch_size"]),
            learning_rate=float(optimization["learning_rate"]),
            weight_decay=float(optimization["weight_decay"]),
            device=selected_device,
            masking=masking,
            loss_config=loss_config,
            hidden_dim=int(model_section["hidden_dim"]),
            layers=int(model_section["message_passing_layers"]),
            node_embedding_dim=int(model_section["node_embedding_dim"]),
            edge_embedding_dim=int(model_section["edge_embedding_dim"]),
            solvent_embedding_dim=int(
                model_section["solvent_embedding_dim"]
            ),
            dropout=float(model_section["dropout"]),
            require_gradient_gate=True,
        )
        first_validation = float(
            result.history[0]["validation_total"]
        )
        relative_improvement = (
            (first_validation - result.best_validation_total)
            / max(abs(first_validation), 1e-12)
        )
        audit_finite = bool(result.audit_metrics) and all(
            math.isfinite(float(value))
            for value in result.audit_metrics.values()
        )
        threshold = float(
            optimization["minimum_relative_validation_improvement"]
        )
        status = (
            "pass"
            if relative_improvement >= threshold and audit_finite
            else "fail"
        )

        staging = Path(
            tempfile.mkdtemp(
                prefix=f".seed-{seed}.staging-",
                dir=output_directory,
            )
        )
        try:
            pd.DataFrame(result.history).to_csv(
                staging / "loss_curves.csv",
                index=False,
                lineterminator="\n",
            )
            atomic_write_json(
                staging / "normalization.json",
                result.normalization.to_json(),
                ensure_ascii=False,
            )
            atomic_write_json(
                staging / "selection.json",
                dict(selection),
                ensure_ascii=False,
            )
            atomic_write_json(
                staging / "gradient_audit.json",
                result.gradient_audit,
                ensure_ascii=False,
            )
            checkpoint_path = staging / "checkpoint.pt"
            payload = save_site_n_pretraining_checkpoint(
                checkpoint_path,
                result,
                masking=masking,
                loss_config=loss_config,
                dataset_contract={
                    "directory": _display_path(dataset_directory),
                    "verification": dict(dataset_verification),
                    "manifest_sha256": sha256_file(
                        dataset_directory / "dataset_manifest.json"
                    ),
                    "authorization_sha256": authorization[
                        "approval_sha256"
                    ],
                },
                selection=selection,
            )
            downstream = _make_downstream(
                base_config,
                num_solvents=len(vocabulary.tokens),
                seed=seed + 1_000_000,
            )
            transfer_audit = transfer_pretrained_backbone(
                downstream,
                payload,
            )
            atomic_write_json(
                staging / "transfer_audit.json",
                transfer_audit,
                ensure_ascii=False,
            )
            summary: dict[str, object] = {
                "schema_version": (
                    "nucpred.mayr-site-n-full-pretraining-run.v1"
                ),
                "status": status,
                "experiment_id": EXPERIMENT_ID,
                "scope": FULL_SCOPE,
                "initialization_seed": seed,
                "contract": contract,
                "authorization": dict(authorization),
                "dataset_verification": dict(dataset_verification),
                "selection": dict(selection),
                "device": str(selected_device),
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "epochs_requested": int(
                    optimization["maximum_epochs"]
                ),
                "epochs_completed": len(result.history),
                "best_epoch": result.best_epoch,
                "epoch1_validation_total": first_validation,
                "best_validation_total": (
                    result.best_validation_total
                ),
                "relative_validation_improvement": (
                    relative_improvement
                ),
                "minimum_relative_validation_improvement": threshold,
                "audit_metrics": dict(result.audit_metrics),
                "audit_metrics_finite": audit_finite,
                "gradient_audit": dict(result.gradient_audit),
                "transfer_audit": transfer_audit,
                "checkpoint_transferable_state_sha256": payload[
                    "transferable_state_sha256"
                ],
                "site_probability_normalization": False,
                "fake_proxy_site_types": [],
                "wall_seconds": time.perf_counter() - started,
                "optimizer_update_budget": (
                    math.ceil(len(train) / int(optimization["batch_size"]))
                    * len(result.history)
                ),
            }
            atomic_write_json(
                staging / "summary.json",
                summary,
                ensure_ascii=False,
            )
            (staging / "run.log").write_text(
                "\n".join(
                    (
                        f"status={status}",
                        "scope=full",
                        f"seed={seed}",
                        f"device={selected_device}",
                        f"epochs_completed={len(result.history)}",
                        f"best_epoch={result.best_epoch}",
                        (
                            "relative_validation_improvement="
                            f"{relative_improvement:.12f}"
                        ),
                        f"audit_metrics_finite={audit_finite}",
                        (
                            "contract_sha256="
                            f"{contract['contract_sha256']}"
                        ),
                        (
                            "approval_sha256="
                            f"{authorization['approval_sha256']}"
                        ),
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            _write_manifest(staging)
            os.replace(staging, target)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        marker.unlink(missing_ok=True)
        if status != "pass":
            raise SiteNCampaignError(
                f"full/seed-{seed} failed validation gate: "
                f"relative improvement={relative_improvement:.3%}, "
                f"audit finite={audit_finite}"
            )
        return summary
    except BaseException as exc:
        _write_failure(output_directory, seed=seed, exc=exc)
        marker.unlink(missing_ok=True)
        raise


def _worker(
    *,
    seed: int,
    base_config: Mapping[str, Any],
    base_config_path: Path,
    full_config: Mapping[str, Any],
    full_config_path: Path,
    train: Sequence[EsnuelNodeXtbExample],
    validation: Sequence[EsnuelNodeXtbExample],
    audit_test: Sequence[EsnuelNodeXtbExample],
    dataset_directory: Path,
    dataset_verification: Mapping[str, object],
    selection: Mapping[str, object],
    authorization: Mapping[str, object],
    device: str,
) -> None:
    try:
        summary = _run_loaded_seed(
            seed=seed,
            base_config=base_config,
            base_config_path=base_config_path,
            full_config=full_config,
            full_config_path=full_config_path,
            train=train,
            validation=validation,
            audit_test=audit_test,
            dataset_directory=dataset_directory,
            dataset_verification=dataset_verification,
            selection=selection,
            authorization=authorization,
            device=device,
        )
        print(
            json.dumps(
                {
                    "event": "full_seed_complete",
                    "seed": seed,
                    "status": summary["status"],
                    "best_epoch": summary["best_epoch"],
                    "best_validation_total": (
                        summary["best_validation_total"]
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    except BaseException:
        traceback.print_exc()
        raise


def _load_inputs(
    full_config_path: Path,
) -> tuple[
    dict[str, Any],
    Path,
    dict[str, Any],
    tuple[
        list[EsnuelNodeXtbExample],
        list[EsnuelNodeXtbExample],
        list[EsnuelNodeXtbExample],
        Path,
        dict[str, object],
        dict[str, object],
    ],
    dict[str, object],
]:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # A caller may have configured the process before invoking the API.
        pass
    full_config = _read_full_config(full_config_path)
    base_config_path = (ROOT / str(full_config["base_config"])).resolve()
    base_config = _read_config(base_config_path)
    authorization = _authorization_gate(full_config)
    loaded = _load_full_scope(full_config)
    return (
        full_config,
        base_config_path,
        base_config,
        loaded,
        authorization,
    )


def run_full_pretraining_seed(
    *,
    seed: int,
    full_config_path: str | Path = DEFAULT_FULL_CONFIG,
    device: str = "cuda:0",
) -> dict[str, object]:
    config_path = Path(full_config_path).resolve()
    (
        full_config,
        base_config_path,
        base_config,
        loaded,
        authorization,
    ) = _load_inputs(config_path)
    train, validation, audit_test, directory, verification, selection = loaded
    return _run_loaded_seed(
        seed=seed,
        base_config=base_config,
        base_config_path=base_config_path,
        full_config=full_config,
        full_config_path=config_path,
        train=train,
        validation=validation,
        audit_test=audit_test,
        dataset_directory=directory,
        dataset_verification=verification,
        selection=selection,
        authorization=authorization,
        device=device,
    )


def run_all_full_pretraining(
    *,
    full_config_path: str | Path = DEFAULT_FULL_CONFIG,
    device: str = "cuda:0",
) -> dict[str, object]:
    if not device.startswith("cuda"):
        raise SiteNCampaignError(
            "Three-process full pretraining requires a CUDA device"
        )
    config_path = Path(full_config_path).resolve()
    (
        full_config,
        base_config_path,
        base_config,
        loaded,
        authorization,
    ) = _load_inputs(config_path)
    train, validation, audit_test, directory, verification, selection = loaded
    output_directory = (
        ROOT / str(full_config["output_directory"])
    ).resolve()
    pending = [
        seed
        for seed in ALLOWED_SEEDS
        if not (output_directory / f"seed-{seed}" / "summary.json").is_file()
    ]
    if not pending:
        summaries = [
            _load_json(
                output_directory / f"seed-{seed}" / "summary.json"
            )
            for seed in ALLOWED_SEEDS
        ]
        return {
            "schema_version": (
                "nucpred.mayr-site-n-full-pretraining-coordinator.v1"
            ),
            "status": "pass",
            "parallel_workers": 0,
            "seeds": list(ALLOWED_SEEDS),
            "summaries": summaries,
        }
    if len(pending) > int(full_config["maximum_parallel_gpu_processes"]):
        raise SiteNCampaignError("Pending seeds exceed GPU process contract")

    gc.collect()
    gc.freeze()
    context = multiprocessing.get_context("fork")
    processes: list[multiprocessing.Process] = []
    for seed in pending:
        process = context.Process(
            name=f"mayr-site-n-full-{seed}",
            target=_worker,
            kwargs={
                "seed": seed,
                "base_config": base_config,
                "base_config_path": base_config_path,
                "full_config": full_config,
                "full_config_path": config_path,
                "train": train,
                "validation": validation,
                "audit_test": audit_test,
                "dataset_directory": directory,
                "dataset_verification": verification,
                "selection": selection,
                "authorization": authorization,
                "device": device,
            },
        )
        process.start()
        processes.append(process)
    print(
        json.dumps(
            {
                "event": "full_pretraining_started",
                "parallel_workers": len(processes),
                "pids": {
                    process.name: process.pid for process in processes
                },
                "seeds": pending,
                "shared_parent_pid": os.getpid(),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    last_heartbeat = 0.0
    while any(process.is_alive() for process in processes):
        for process in processes:
            process.join(timeout=1.0)
        now = time.monotonic()
        if now - last_heartbeat >= 30.0:
            print(
                json.dumps(
                    {
                        "event": "full_pretraining_heartbeat",
                        "workers": {
                            process.name: {
                                "pid": process.pid,
                                "alive": process.is_alive(),
                                "exitcode": process.exitcode,
                            }
                            for process in processes
                        },
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            last_heartbeat = now
    for process in processes:
        process.join()
    failures = {
        process.name: process.exitcode
        for process in processes
        if process.exitcode != 0
    }
    if failures:
        raise SiteNCampaignError(
            f"Full-pretraining workers failed: {failures}"
        )
    summaries = [
        _load_json(output_directory / f"seed-{seed}" / "summary.json")
        for seed in ALLOWED_SEEDS
    ]
    if any(summary.get("status") != "pass" for summary in summaries):
        raise SiteNCampaignError("A full-pretraining summary did not pass")
    return {
        "schema_version": (
            "nucpred.mayr-site-n-full-pretraining-coordinator.v1"
        ),
        "status": "pass",
        "parallel_workers": len(processes),
        "seeds": list(ALLOWED_SEEDS),
        "summaries": summaries,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--seed", type=int, choices=ALLOWED_SEEDS)
    selection.add_argument("--all-seeds", action="store_true")
    parser.add_argument(
        "--full-config",
        type=Path,
        default=DEFAULT_FULL_CONFIG,
    )
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    payload = (
        run_all_full_pretraining(
            full_config_path=arguments.full_config,
            device=arguments.device,
        )
        if arguments.all_seeds
        else run_full_pretraining_seed(
            seed=int(arguments.seed),
            full_config_path=arguments.full_config,
            device=arguments.device,
        )
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
