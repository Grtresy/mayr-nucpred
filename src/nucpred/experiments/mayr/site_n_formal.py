"""Run the approved controlled downstream matrix for typed site-N prediction."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import gc
import hashlib
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

from nucpred.core.files import atomic_write_json, sha256_file
from nucpred.datasets.mayr_site_n import verify_dataset
from nucpred.project import get_project_layout
from nucpred.training.mayr_node_xtb_pretraining import (
    load_pretraining_checkpoint as load_legacy_pretraining_checkpoint,
)
from nucpred.training.mayr_node_xtb_scratch import (
    SolventVocabulary,
    initialization_sha256,
)
from nucpred.training.mayr_site_n import (
    MayrSiteNModel,
    SiteNExample,
    SiteNFoldPreprocessor,
    fit_site_n_preprocessor,
    load_site_n_examples,
)
from nucpred.training.mayr_site_n_pretraining import (
    load_site_n_pretraining_checkpoint,
    transfer_pretrained_backbone,
)

from .site_n import (
    EXPERIMENT_ID,
    SiteNCampaignError,
    _canonical_sha256,
    _display_path,
    _evaluate,
    _iter_batches,
    _make_model,
    _read_config,
    _train_epoch,
    _write_manifest,
)
from .site_n_full_pretraining import (
    _authorization_gate,
    _full_contract,
    _read_full_config,
)
from .site_n_stage_gate import _verify_run_manifest


_LAYOUT = get_project_layout()
ROOT = _LAYOUT.root
DEFAULT_FORMAL_CONFIG = ROOT / "configs/mayr_site_n_formal.toml"
FORMAL_CONFIG_SCHEMA = "nucpred.mayr-site-n-formal-config.v1"
ARMS = ("scratch", "legacy_encoder", "matched_pretraining")
SPLIT_SEEDS = (20241220, 20241221, 20241222, 20241223, 20241224)
INITIALIZATION_SEEDS = (2026072601, 2026072602, 2026072603)
PRETRAINING_SEEDS = (31001, 31002, 31003)
LEGACY_TRANSFER_PREFIXES = (
    "node_encoder.",
    "local_encoder.",
    "edge_encoder.",
    "message_layers.",
    "global_xtb_encoder.",
)
EXPECTED_LEGACY_TRANSFER_TENSORS = 70
EXPECTED_LEGACY_TRANSFER_NUMEL = 430_624


@dataclass(frozen=True, slots=True)
class SelectionOutcome:
    model: MayrSiteNModel
    curves: pd.DataFrame
    best_epoch: int
    best_validation_mae: float
    validation_metrics: Mapping[str, object]
    validation_predictions: pd.DataFrame
    base_initialization_sha256: str
    post_transfer_initialization_sha256: str
    transfer_audit: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class FinalRefitOutcome:
    model: MayrSiteNModel
    curves: pd.DataFrame
    base_initialization_sha256: str
    post_transfer_initialization_sha256: str
    transfer_audit: Mapping[str, object]


def _read_formal_config(
    path: str | Path = DEFAULT_FORMAL_CONFIG,
) -> dict[str, Any]:
    config_path = Path(path).resolve()
    payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != FORMAL_CONFIG_SCHEMA:
        raise SiteNCampaignError("Unsupported formal site-N config")
    if payload.get("experiment_id") != EXPERIMENT_ID:
        raise SiteNCampaignError("Formal site-N experiment identity changed")
    if (
        payload.get("dataset_id") != "mayr-site-n-20260726-v1"
        or tuple(map(int, payload["split_seeds"])) != SPLIT_SEEDS
        or tuple(map(int, payload["initialization_seeds"]))
        != INITIALIZATION_SEEDS
        or tuple(map(int, payload["pretraining_seeds"]))
        != PRETRAINING_SEEDS
        or tuple(payload["arms"]) != ARMS
    ):
        raise SiteNCampaignError("Formal matrix axes changed")
    if int(payload["maximum_parallel_gpu_processes"]) != 3:
        raise SiteNCampaignError("Formal GPU concurrency changed")
    if payload.get("test_used_for_selection") is not False:
        raise SiteNCampaignError("Test selection is forbidden")
    optimization = payload["optimization"]
    if not (
        1
        <= int(optimization["minimum_epochs"])
        <= int(optimization["maximum_epochs"])
    ):
        raise SiteNCampaignError("Invalid formal epoch contract")
    for table_name in ("legacy_checkpoints", "matched_checkpoints"):
        rows = payload[table_name]
        pairs = {
            (
                int(row["pretraining_seed"]),
                int(row["downstream_initialization_seed"]),
            )
            for row in rows
        }
        if pairs != set(zip(PRETRAINING_SEEDS, INITIALIZATION_SEEDS, strict=True)):
            raise SiteNCampaignError(
                f"{table_name} seed binding changed"
            )
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SiteNCampaignError(f"Expected JSON object: {path}")
    return payload


def _tensor_mapping_sha256(
    values: Mapping[str, torch.Tensor],
) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        tensor = values[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _checkpoint_entry(
    config: Mapping[str, Any],
    table_name: str,
    initialization_seed: int,
) -> Mapping[str, object]:
    matches = [
        row
        for row in config[table_name]
        if int(row["downstream_initialization_seed"])
        == int(initialization_seed)
    ]
    if len(matches) != 1:
        raise SiteNCampaignError(
            f"{table_name} binding is not unique for {initialization_seed}"
        )
    return matches[0]


def _verify_pretraining_inputs(
    formal_config: Mapping[str, Any],
) -> dict[str, object]:
    full_config_path = (
        ROOT / str(formal_config["full_pretraining_config"])
    ).resolve()
    full_config = _read_full_config(full_config_path)
    base_config_path = (ROOT / str(full_config["base_config"])).resolve()
    dataset_directory = (
        ROOT / str(full_config["dataset_directory"])
    ).resolve()
    approval_path = (
        ROOT
        / str(full_config["authorization"]["approval_artifact"])
    ).resolve()
    authorization = _authorization_gate(full_config)

    matched: dict[str, object] = {}
    for row in formal_config["matched_checkpoints"]:
        seed = int(row["pretraining_seed"])
        path = (ROOT / str(row["path"])).resolve()
        run_directory = path.parent
        summary = _load_json(run_directory / "summary.json")
        verification = _verify_run_manifest(run_directory)
        expected_contract = _full_contract(
            base_config_path=base_config_path,
            full_config_path=full_config_path,
            dataset_directory=dataset_directory,
            approval_path=approval_path,
            seed=seed,
        )
        if (
            summary.get("status") != "pass"
            or summary.get("scope") != "full"
            or int(summary.get("initialization_seed", -1)) != seed
            or summary.get("contract") != expected_contract
            or summary.get("audit_metrics_finite") is not True
            or summary["transfer_audit"]["status"] != "pass"
        ):
            raise SiteNCampaignError(
                f"Matched full-pretraining seed {seed} did not verify"
            )
        checkpoint = load_site_n_pretraining_checkpoint(path)
        if (
            checkpoint["transferable_state_sha256"]
            != summary["checkpoint_transferable_state_sha256"]
        ):
            raise SiteNCampaignError(
                f"Matched checkpoint binding changed for seed {seed}"
            )
        matched[str(seed)] = {
            "path": _display_path(path),
            "sha256": sha256_file(path),
            "run_manifest_verification": verification,
            "contract_sha256": expected_contract["contract_sha256"],
            "transferable_state_sha256": checkpoint[
                "transferable_state_sha256"
            ],
        }

    legacy: dict[str, object] = {}
    for row in formal_config["legacy_checkpoints"]:
        seed = int(row["pretraining_seed"])
        path = (ROOT / str(row["path"])).resolve()
        observed_sha256 = sha256_file(path)
        if observed_sha256 != str(row["sha256"]):
            raise SiteNCampaignError(
                f"Legacy checkpoint hash changed for seed {seed}"
            )
        checkpoint = load_legacy_pretraining_checkpoint(path)
        if int(checkpoint["init_seed"]) != seed:
            raise SiteNCampaignError(
                f"Legacy checkpoint seed changed for {seed}"
            )
        legacy[str(seed)] = {
            "path": _display_path(path),
            "sha256": observed_sha256,
            "backbone_state_sha256": checkpoint[
                "backbone_state_sha256"
            ],
        }
    return {
        "status": "pass",
        "authorization": authorization,
        "matched": matched,
        "legacy": legacy,
    }


def _load_split_roles(
    examples: Sequence[SiteNExample],
    membership: pd.DataFrame,
    *,
    split_seed: int,
) -> tuple[
    list[SiteNExample],
    list[SiteNExample],
    list[SiteNExample],
    dict[str, object],
]:
    selected = membership.loc[
        membership["split_seed"].eq(int(split_seed))
    ].copy()
    if selected.empty:
        raise SiteNCampaignError(f"Unknown split seed: {split_seed}")
    if selected["target_id"].duplicated().any():
        raise SiteNCampaignError("Split membership target IDs are duplicated")
    target_role = {
        str(row.target_id): str(row.role)
        for row in selected.itertuples(index=False)
    }
    expected_targets = {
        target_id for example in examples for target_id in example.target_ids
    }
    if set(target_role) != expected_targets:
        raise SiteNCampaignError("Split membership target universe changed")

    roles: dict[str, list[SiteNExample]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for example in examples:
        observed_roles = {target_role[value] for value in example.target_ids}
        if len(observed_roles) != 1:
            raise SiteNCampaignError(
                f"Context crossed roles: {example.context_id}"
            )
        role = observed_roles.pop()
        if role not in roles:
            raise SiteNCampaignError(f"Unsupported split role: {role}")
        roles[role].append(example)

    role_audit: dict[str, object] = {}
    connectivity_sets: dict[str, set[str]] = {}
    for role, role_examples in roles.items():
        frame = selected.loc[selected["role"].eq(role)]
        connectivity = set(frame["connectivity_id"].astype(str))
        connectivity_sets[role] = connectivity
        role_audit[role] = {
            "context_count": len(role_examples),
            "target_count": sum(item.num_sites for item in role_examples),
            "connectivity_count": len(connectivity),
            "target_id_sha256": hashlib.sha256(
                (
                    "\n".join(sorted(frame["target_id"].astype(str)))
                    + "\n"
                ).encode("utf-8")
            ).hexdigest(),
        }
    overlap = {
        "train_validation": len(
            connectivity_sets["train"] & connectivity_sets["validation"]
        ),
        "train_test": len(
            connectivity_sets["train"] & connectivity_sets["test"]
        ),
        "validation_test": len(
            connectivity_sets["validation"] & connectivity_sets["test"]
        ),
    }
    if any(overlap.values()):
        raise SiteNCampaignError("Connectivity split leakage detected")
    return (
        roles["train"],
        roles["validation"],
        roles["test"],
        {
            "schema_version": "nucpred.mayr-site-n-formal-split-audit.v1",
            "split_seed": int(split_seed),
            "roles": role_audit,
            "connectivity_overlap": overlap,
            "test_used_for_selection": False,
        },
    )


def _legacy_transfer(
    model: MayrSiteNModel,
    checkpoint_path: Path,
) -> dict[str, object]:
    payload = load_legacy_pretraining_checkpoint(checkpoint_path)
    source = payload["backbone_state_dict"]
    if not isinstance(source, Mapping):
        raise SiteNCampaignError("Legacy checkpoint state is not a mapping")
    before = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    transfer_names = {
        name
        for name in before
        if any(name.startswith(prefix) for prefix in LEGACY_TRANSFER_PREFIXES)
    }
    if (
        len(transfer_names) != EXPECTED_LEGACY_TRANSFER_TENSORS
        or sum(before[name].numel() for name in transfer_names)
        != EXPECTED_LEGACY_TRANSFER_NUMEL
    ):
        raise SiteNCampaignError("Legacy transfer surface changed")
    missing = sorted(transfer_names - set(source))
    if missing:
        raise SiteNCampaignError(
            f"Legacy checkpoint lacks encoder tensors: {missing}"
        )
    state = model.state_dict()
    with torch.no_grad():
        for name in sorted(transfer_names):
            source_tensor = source[name]
            if (
                not isinstance(source_tensor, torch.Tensor)
                or source_tensor.shape != state[name].shape
                or source_tensor.dtype != state[name].dtype
            ):
                raise SiteNCampaignError(
                    f"Legacy transfer tensor changed: {name}"
                )
            state[name].copy_(
                source_tensor.to(
                    device=state[name].device,
                    dtype=state[name].dtype,
                )
            )
    model.load_state_dict(state, strict=True)
    after = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    reset_names = set(before) - transfer_names
    exact = all(
        torch.equal(after[name], source[name].detach().cpu())
        for name in transfer_names
    )
    reset_unchanged = all(
        torch.equal(after[name], before[name]) for name in reset_names
    )
    audit: dict[str, object] = {
        "schema_version": "nucpred.mayr-site-n-legacy-transfer-audit.v1",
        "status": "pass" if exact and reset_unchanged else "fail",
        "checkpoint": _display_path(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_backbone_state_sha256": payload[
            "backbone_state_sha256"
        ],
        "transferred_prefixes": list(LEGACY_TRANSFER_PREFIXES),
        "transferred_parameter_tensors": len(transfer_names),
        "transferred_parameter_numel": sum(
            int(after[name].numel()) for name in transfer_names
        ),
        "transferred_state_sha256": _tensor_mapping_sha256(
            {name: after[name] for name in transfer_names}
        ),
        "exact_transfer": exact,
        "reset_parameter_tensors": len(reset_names),
        "reset_modules_unchanged": reset_unchanged,
        "legacy_site_head_transferred": False,
        "shared_site_encoder_transferred": False,
        "site_type_adapters_transferred": False,
        "site_probability_normalization": False,
    }
    if audit["status"] != "pass":
        raise SiteNCampaignError("Legacy encoder transfer audit failed")
    return audit


def _scratch_audit(model: MayrSiteNModel) -> dict[str, object]:
    return {
        "schema_version": "nucpred.mayr-site-n-scratch-audit.v1",
        "status": "pass",
        "transferred_parameter_tensors": 0,
        "transferred_parameter_numel": 0,
        "reset_modules_unchanged": True,
        "site_probability_normalization": False,
        "state_sha256": initialization_sha256(model),
    }


def _initialize_model(
    *,
    arm: str,
    base_config: Mapping[str, Any],
    vocabulary: SolventVocabulary,
    initialization_seed: int,
    device: torch.device,
    legacy_checkpoint: Path | None,
    matched_checkpoint: Path | None,
) -> tuple[MayrSiteNModel, str, str, dict[str, object]]:
    model, base_hash = _make_model(
        base_config,
        vocabulary,
        seed=initialization_seed,
        device=device,
    )
    if arm == "scratch":
        audit = _scratch_audit(model)
    elif arm == "legacy_encoder":
        if legacy_checkpoint is None:
            raise SiteNCampaignError("Legacy checkpoint is required")
        audit = _legacy_transfer(model, legacy_checkpoint)
    elif arm == "matched_pretraining":
        if matched_checkpoint is None:
            raise SiteNCampaignError("Matched checkpoint is required")
        payload = load_site_n_pretraining_checkpoint(matched_checkpoint)
        audit = transfer_pretrained_backbone(model, payload)
        audit = {
            **audit,
            "checkpoint": _display_path(matched_checkpoint),
            "checkpoint_sha256": sha256_file(matched_checkpoint),
            "shared_site_encoder_transferred": True,
            "atom_type_adapter_transferred": True,
        }
    else:
        raise SiteNCampaignError(f"Unsupported formal arm: {arm}")
    post_hash = initialization_sha256(model)
    if arm == "scratch" and post_hash != base_hash:
        raise SiteNCampaignError("Scratch initialization changed")
    return model, base_hash, post_hash, audit


def _fit_selection(
    train: Sequence[SiteNExample],
    validation: Sequence[SiteNExample],
    *,
    arm: str,
    base_config: Mapping[str, Any],
    formal_config: Mapping[str, Any],
    preprocessor: SiteNFoldPreprocessor,
    vocabulary: SolventVocabulary,
    initialization_seed: int,
    device: torch.device,
    legacy_checkpoint: Path | None,
    matched_checkpoint: Path | None,
) -> SelectionOutcome:
    model, base_hash, post_hash, transfer_audit = _initialize_model(
        arm=arm,
        base_config=base_config,
        vocabulary=vocabulary,
        initialization_seed=initialization_seed,
        device=device,
        legacy_checkpoint=legacy_checkpoint,
        matched_checkpoint=matched_checkpoint,
    )
    optimization = formal_config["optimization"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
    )
    rows: list[dict[str, object]] = []
    best_epoch = 0
    best_mae = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    for epoch in range(1, int(optimization["maximum_epochs"]) + 1):
        training_metrics = _train_epoch(
            model,
            _iter_batches(
                train,
                batch_size=int(optimization["batch_size_contexts"]),
                preprocessor=preprocessor,
                vocabulary=vocabulary,
                shuffle_seed=initialization_seed + epoch,
            ),
            optimizer=optimizer,
            device=device,
            ranking_weight=float(optimization["ranking_weight"]),
            gradient_clip_norm=float(
                optimization["gradient_clip_norm"]
            ),
        )
        validation_metrics, _ = _evaluate(
            model,
            validation,
            preprocessor=preprocessor,
            vocabulary=vocabulary,
            batch_size=int(optimization["batch_size_contexts"]),
            device=device,
        )
        validation_mae = float(validation_metrics["mae"])
        improved = validation_mae < (
            best_mae
            - float(optimization["minimum_validation_mae_delta"])
        )
        if improved:
            best_epoch = epoch
            best_mae = validation_mae
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        rows.append(
            {
                "epoch": epoch,
                "is_validation_best": improved,
                **{
                    f"train_{name}": value
                    for name, value in training_metrics.items()
                },
                "validation_mae": validation_mae,
                "validation_rmse": float(validation_metrics["rmse"]),
                "validation_r2": float(validation_metrics["r2"]),
            }
        )
        if (
            epoch >= int(optimization["minimum_epochs"])
            and stale_epochs
            >= int(optimization["early_stopping_patience"])
        ):
            break
    if best_state is None or best_epoch <= 0:
        raise SiteNCampaignError("Selection did not produce a best state")
    model.load_state_dict(best_state, strict=True)
    validation_metrics, validation_predictions = _evaluate(
        model,
        validation,
        preprocessor=preprocessor,
        vocabulary=vocabulary,
        batch_size=int(optimization["batch_size_contexts"]),
        device=device,
    )
    if not math.isclose(
        float(validation_metrics["mae"]),
        best_mae,
        rel_tol=0.0,
        abs_tol=1e-7,
    ):
        raise SiteNCampaignError("Restored validation state changed")
    return SelectionOutcome(
        model=model,
        curves=pd.DataFrame(rows),
        best_epoch=best_epoch,
        best_validation_mae=best_mae,
        validation_metrics=validation_metrics,
        validation_predictions=validation_predictions,
        base_initialization_sha256=base_hash,
        post_transfer_initialization_sha256=post_hash,
        transfer_audit=transfer_audit,
    )


def _fit_final_refit(
    development: Sequence[SiteNExample],
    *,
    epochs: int,
    arm: str,
    base_config: Mapping[str, Any],
    formal_config: Mapping[str, Any],
    preprocessor: SiteNFoldPreprocessor,
    vocabulary: SolventVocabulary,
    initialization_seed: int,
    device: torch.device,
    legacy_checkpoint: Path | None,
    matched_checkpoint: Path | None,
) -> FinalRefitOutcome:
    model, base_hash, post_hash, transfer_audit = _initialize_model(
        arm=arm,
        base_config=base_config,
        vocabulary=vocabulary,
        initialization_seed=initialization_seed,
        device=device,
        legacy_checkpoint=legacy_checkpoint,
        matched_checkpoint=matched_checkpoint,
    )
    optimization = formal_config["optimization"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
    )
    rows: list[dict[str, object]] = []
    for epoch in range(1, int(epochs) + 1):
        metrics = _train_epoch(
            model,
            _iter_batches(
                development,
                batch_size=int(optimization["batch_size_contexts"]),
                preprocessor=preprocessor,
                vocabulary=vocabulary,
                shuffle_seed=initialization_seed + epoch,
            ),
            optimizer=optimizer,
            device=device,
            ranking_weight=float(optimization["ranking_weight"]),
            gradient_clip_norm=float(
                optimization["gradient_clip_norm"]
            ),
        )
        rows.append({"epoch": epoch, **metrics})
    return FinalRefitOutcome(
        model=model,
        curves=pd.DataFrame(rows),
        base_initialization_sha256=base_hash,
        post_transfer_initialization_sha256=post_hash,
        transfer_audit=transfer_audit,
    )


def _formal_contract(
    *,
    formal_config_path: Path,
    base_config_path: Path,
    dataset_directory: Path,
    approval_path: Path,
    split_seed: int,
    initialization_seed: int,
    arm: str,
    checkpoint_path: Path | None,
    split_audit: Mapping[str, object],
) -> dict[str, object]:
    paths = {
        "formal_config": formal_config_path,
        "base_config": base_config_path,
        "formal_runner": Path(__file__).resolve(),
        "downstream_model": (
            ROOT / "src/nucpred/training/mayr_site_n.py"
        ).resolve(),
        "dataset_manifest": dataset_directory / "dataset_manifest.json",
        "split_manifest": dataset_directory / "split_manifest.json",
        "approval": approval_path,
    }
    if checkpoint_path is not None:
        paths["pretraining_checkpoint"] = checkpoint_path
    contract: dict[str, object] = {
        "schema_version": "nucpred.mayr-site-n-formal-job-contract.v1",
        "experiment_id": EXPERIMENT_ID,
        "split_seed": int(split_seed),
        "initialization_seed": int(initialization_seed),
        "arm": arm,
        "source_hashes": {
            name: sha256_file(path) for name, path in paths.items()
        },
        "split_audit": dict(split_audit),
        "test_used_for_selection": False,
        "site_probability_normalization": False,
        "unmeasured_candidates_are_negative": False,
        "optimizer_schedule_shared_across_arms": True,
    }
    contract["contract_sha256"] = _canonical_sha256(contract)
    return contract


def _save_model_checkpoint(
    path: Path,
    *,
    model: MayrSiteNModel,
    phase: str,
    arm: str,
    split_seed: int,
    initialization_seed: int,
    preprocessor: SiteNFoldPreprocessor,
    vocabulary: SolventVocabulary,
    contract: Mapping[str, object],
    transfer_audit: Mapping[str, object],
) -> dict[str, object]:
    state = {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
    }
    payload: dict[str, object] = {
        "schema_version": "nucpred.mayr-site-n-formal-checkpoint.v1",
        "phase": phase,
        "arm": arm,
        "split_seed": int(split_seed),
        "initialization_seed": int(initialization_seed),
        "model_architecture": model.architecture,
        "model_state_dict": state,
        "model_state_sha256": _tensor_mapping_sha256(state),
        "preprocessor": preprocessor.to_json(),
        "solvent_vocabulary": list(vocabulary.tokens),
        "contract": dict(contract),
        "transfer_audit": dict(transfer_audit),
    }
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return payload


def _failure_marker(
    output_root: Path,
    *,
    split_seed: int,
    initialization_seed: int,
    arm: str,
    exc: BaseException,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    token = f"split-{split_seed}__init-{initialization_seed}__{arm}"
    atomic_write_json(
        output_root / f"{token}.failure.json",
        {
            "schema_version": "nucpred.mayr-site-n-formal-failure.v1",
            "status": "failed",
            "split_seed": int(split_seed),
            "initialization_seed": int(initialization_seed),
            "arm": arm,
            "failed_at_utc": datetime.now(UTC).isoformat(),
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        },
        ensure_ascii=False,
    )


def _run_job_loaded(
    *,
    split_seed: int,
    initialization_seed: int,
    arm: str,
    examples: Sequence[SiteNExample],
    membership: pd.DataFrame,
    base_config: Mapping[str, Any],
    base_config_path: Path,
    formal_config: Mapping[str, Any],
    formal_config_path: Path,
    dataset_directory: Path,
    dataset_verification: Mapping[str, object],
    pretraining_verification: Mapping[str, object],
    device: str,
) -> dict[str, object]:
    started = time.perf_counter()
    if (
        split_seed not in SPLIT_SEEDS
        or initialization_seed not in INITIALIZATION_SEEDS
        or arm not in ARMS
    ):
        raise SiteNCampaignError("Unregistered formal job axis")
    selected_device = torch.device(device)
    if selected_device.type != "cuda" or not torch.cuda.is_available():
        raise SiteNCampaignError("Formal matrix requires CUDA")

    train, validation, test, split_audit = _load_split_roles(
        examples,
        membership,
        split_seed=split_seed,
    )
    legacy_entry = _checkpoint_entry(
        formal_config, "legacy_checkpoints", initialization_seed
    )
    matched_entry = _checkpoint_entry(
        formal_config, "matched_checkpoints", initialization_seed
    )
    legacy_checkpoint = (ROOT / str(legacy_entry["path"])).resolve()
    matched_checkpoint = (ROOT / str(matched_entry["path"])).resolve()
    checkpoint_path = {
        "scratch": None,
        "legacy_encoder": legacy_checkpoint,
        "matched_pretraining": matched_checkpoint,
    }[arm]
    approval_path = (
        ROOT / str(formal_config["approval_artifact"])
    ).resolve()
    contract = _formal_contract(
        formal_config_path=formal_config_path,
        base_config_path=base_config_path,
        dataset_directory=dataset_directory,
        approval_path=approval_path,
        split_seed=split_seed,
        initialization_seed=initialization_seed,
        arm=arm,
        checkpoint_path=checkpoint_path,
        split_audit=split_audit,
    )
    output_root = (
        ROOT / str(formal_config["output_directory"])
    ).resolve()
    target = (
        output_root
        / f"split-{split_seed}"
        / f"init-{initialization_seed}"
        / arm
    )
    summary_path = target / "summary.json"
    if summary_path.is_file():
        existing = _load_json(summary_path)
        if (
            existing.get("status") == "pass"
            and existing.get("contract") == contract
        ):
            return existing
        raise SiteNCampaignError(f"Existing formal job is stale: {target}")
    if target.exists():
        raise SiteNCampaignError(f"Partial formal job exists: {target}")

    selection_preprocessor = fit_site_n_preprocessor(train)
    selection_vocabulary = SolventVocabulary.from_values(
        [example.solvent_raw for example in train]
    )
    development = [*train, *validation]
    final_preprocessor = fit_site_n_preprocessor(development)
    final_vocabulary = SolventVocabulary.from_values(
        [example.solvent_raw for example in development]
    )
    try:
        selection = _fit_selection(
            train,
            validation,
            arm=arm,
            base_config=base_config,
            formal_config=formal_config,
            preprocessor=selection_preprocessor,
            vocabulary=selection_vocabulary,
            initialization_seed=initialization_seed,
            device=selected_device,
            legacy_checkpoint=(
                legacy_checkpoint if arm == "legacy_encoder" else None
            ),
            matched_checkpoint=(
                matched_checkpoint
                if arm == "matched_pretraining"
                else None
            ),
        )
        final = _fit_final_refit(
            development,
            epochs=selection.best_epoch,
            arm=arm,
            base_config=base_config,
            formal_config=formal_config,
            preprocessor=final_preprocessor,
            vocabulary=final_vocabulary,
            initialization_seed=initialization_seed,
            device=selected_device,
            legacy_checkpoint=(
                legacy_checkpoint if arm == "legacy_encoder" else None
            ),
            matched_checkpoint=(
                matched_checkpoint
                if arm == "matched_pretraining"
                else None
            ),
        )
        test_metrics, test_predictions = _evaluate(
            final.model,
            test,
            preprocessor=final_preprocessor,
            vocabulary=final_vocabulary,
            batch_size=int(
                formal_config["optimization"]["batch_size_contexts"]
            ),
            device=selected_device,
        )
        if not all(
            math.isfinite(float(test_metrics[name]))
            for name in ("mae", "rmse", "r2")
        ):
            raise SiteNCampaignError("Formal test metrics are non-finite")

        target.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{arm}.staging-",
                dir=target.parent,
            )
        )
        try:
            selection.curves.to_csv(
                staging / "selection_loss_curves.csv",
                index=False,
                lineterminator="\n",
            )
            final.curves.to_csv(
                staging / "final_refit_loss_curves.csv",
                index=False,
                lineterminator="\n",
            )
            selection.validation_predictions.to_parquet(
                staging / "validation_predictions.parquet",
                index=False,
                engine="pyarrow",
                compression="zstd",
            )
            test_predictions.to_parquet(
                staging / "test_predictions.parquet",
                index=False,
                engine="pyarrow",
                compression="zstd",
            )
            atomic_write_json(
                staging / "validation_metrics.json",
                selection.validation_metrics,
                ensure_ascii=False,
            )
            atomic_write_json(
                staging / "test_metrics.json",
                test_metrics,
                ensure_ascii=False,
            )
            atomic_write_json(
                staging / "selection_preprocessor.json",
                selection_preprocessor.to_json(),
                ensure_ascii=False,
            )
            atomic_write_json(
                staging / "final_refit_preprocessor.json",
                final_preprocessor.to_json(),
                ensure_ascii=False,
            )
            atomic_write_json(
                staging / "selection_transfer_audit.json",
                selection.transfer_audit,
                ensure_ascii=False,
            )
            atomic_write_json(
                staging / "final_refit_transfer_audit.json",
                final.transfer_audit,
                ensure_ascii=False,
            )
            _save_model_checkpoint(
                staging / "selection_checkpoint.pt",
                model=selection.model,
                phase="selection",
                arm=arm,
                split_seed=split_seed,
                initialization_seed=initialization_seed,
                preprocessor=selection_preprocessor,
                vocabulary=selection_vocabulary,
                contract=contract,
                transfer_audit=selection.transfer_audit,
            )
            _save_model_checkpoint(
                staging / "final_refit_checkpoint.pt",
                model=final.model,
                phase="final_refit",
                arm=arm,
                split_seed=split_seed,
                initialization_seed=initialization_seed,
                preprocessor=final_preprocessor,
                vocabulary=final_vocabulary,
                contract=contract,
                transfer_audit=final.transfer_audit,
            )
            summary: dict[str, object] = {
                "schema_version": "nucpred.mayr-site-n-formal-job.v1",
                "status": "pass",
                "experiment_id": EXPERIMENT_ID,
                "split_seed": int(split_seed),
                "initialization_seed": int(initialization_seed),
                "pretraining_seed": int(
                    legacy_entry["pretraining_seed"]
                ),
                "arm": arm,
                "contract": contract,
                "dataset_verification": dict(dataset_verification),
                "pretraining_verification": dict(
                    pretraining_verification
                ),
                "split_audit": split_audit,
                "selection_best_epoch": selection.best_epoch,
                "selection_best_validation_mae": (
                    selection.best_validation_mae
                ),
                "validation_metrics": dict(
                    selection.validation_metrics
                ),
                "test_metrics": test_metrics,
                "selection_base_initialization_sha256": (
                    selection.base_initialization_sha256
                ),
                "selection_post_transfer_initialization_sha256": (
                    selection.post_transfer_initialization_sha256
                ),
                "final_base_initialization_sha256": (
                    final.base_initialization_sha256
                ),
                "final_post_transfer_initialization_sha256": (
                    final.post_transfer_initialization_sha256
                ),
                "selection_transfer_audit": dict(
                    selection.transfer_audit
                ),
                "final_transfer_audit": dict(final.transfer_audit),
                "optimizer_schedule_shared_across_arms": True,
                "test_used_for_selection": False,
                "test_evaluation_count": 1,
                "device": str(selected_device),
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "wall_seconds": time.perf_counter() - started,
            }
            atomic_write_json(
                staging / "summary.json", summary, ensure_ascii=False
            )
            (staging / "run.log").write_text(
                "\n".join(
                    (
                        "status=pass",
                        f"split_seed={split_seed}",
                        f"initialization_seed={initialization_seed}",
                        f"arm={arm}",
                        f"selection_best_epoch={selection.best_epoch}",
                        (
                            "selection_validation_mae="
                            f"{selection.best_validation_mae:.12f}"
                        ),
                        f"test_mae={float(test_metrics['mae']):.12f}",
                        f"test_r2={float(test_metrics['r2']):.12f}",
                        (
                            "contract_sha256="
                            f"{contract['contract_sha256']}"
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
        return summary
    except BaseException as exc:
        _failure_marker(
            output_root,
            split_seed=split_seed,
            initialization_seed=initialization_seed,
            arm=arm,
            exc=exc,
        )
        raise


def _lane_worker(
    *,
    initialization_seed: int,
    examples: Sequence[SiteNExample],
    membership: pd.DataFrame,
    base_config: Mapping[str, Any],
    base_config_path: Path,
    formal_config: Mapping[str, Any],
    formal_config_path: Path,
    dataset_directory: Path,
    dataset_verification: Mapping[str, object],
    pretraining_verification: Mapping[str, object],
    device: str,
) -> None:
    try:
        for split_seed in SPLIT_SEEDS:
            for arm in ARMS:
                summary = _run_job_loaded(
                    split_seed=split_seed,
                    initialization_seed=initialization_seed,
                    arm=arm,
                    examples=examples,
                    membership=membership,
                    base_config=base_config,
                    base_config_path=base_config_path,
                    formal_config=formal_config,
                    formal_config_path=formal_config_path,
                    dataset_directory=dataset_directory,
                    dataset_verification=dataset_verification,
                    pretraining_verification=pretraining_verification,
                    device=device,
                )
                print(
                    json.dumps(
                        {
                            "event": "formal_job_complete",
                            "split_seed": split_seed,
                            "initialization_seed": initialization_seed,
                            "arm": arm,
                            "best_epoch": summary[
                                "selection_best_epoch"
                            ],
                            "test_mae": summary["test_metrics"]["mae"],
                            "test_r2": summary["test_metrics"]["r2"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    except BaseException:
        traceback.print_exc()
        raise


def _load_inputs(
    formal_config_path: Path,
) -> tuple[
    dict[str, Any],
    Path,
    dict[str, Any],
    Path,
    dict[str, object],
    list[SiteNExample],
    pd.DataFrame,
    dict[str, object],
]:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    formal_config = _read_formal_config(formal_config_path)
    base_config_path = (
        ROOT / str(formal_config["base_config"])
    ).resolve()
    base_config = _read_config(base_config_path)
    dataset_directory = (
        ROOT / str(formal_config["dataset_directory"])
    ).resolve()
    dataset_verification = verify_dataset(dataset_directory)
    pretraining_verification = _verify_pretraining_inputs(formal_config)
    examples = load_site_n_examples(dataset_directory)
    membership = pd.read_csv(dataset_directory / "split_membership.csv")
    if (
        len(examples) != 1102
        or sum(item.num_sites for item in examples) != 1108
    ):
        raise SiteNCampaignError("Formal site-N population changed")
    return (
        formal_config,
        base_config_path,
        base_config,
        dataset_directory,
        dataset_verification,
        examples,
        membership,
        pretraining_verification,
    )


def run_formal_job(
    *,
    split_seed: int,
    initialization_seed: int,
    arm: str,
    formal_config_path: str | Path = DEFAULT_FORMAL_CONFIG,
    device: str = "cuda:0",
) -> dict[str, object]:
    config_path = Path(formal_config_path).resolve()
    (
        formal_config,
        base_config_path,
        base_config,
        dataset_directory,
        dataset_verification,
        examples,
        membership,
        pretraining_verification,
    ) = _load_inputs(config_path)
    return _run_job_loaded(
        split_seed=split_seed,
        initialization_seed=initialization_seed,
        arm=arm,
        examples=examples,
        membership=membership,
        base_config=base_config,
        base_config_path=base_config_path,
        formal_config=formal_config,
        formal_config_path=config_path,
        dataset_directory=dataset_directory,
        dataset_verification=dataset_verification,
        pretraining_verification=pretraining_verification,
        device=device,
    )


def run_all_formal_jobs(
    *,
    formal_config_path: str | Path = DEFAULT_FORMAL_CONFIG,
    device: str = "cuda:0",
) -> dict[str, object]:
    if not device.startswith("cuda"):
        raise SiteNCampaignError("Formal matrix requires CUDA")
    config_path = Path(formal_config_path).resolve()
    (
        formal_config,
        base_config_path,
        base_config,
        dataset_directory,
        dataset_verification,
        examples,
        membership,
        pretraining_verification,
    ) = _load_inputs(config_path)
    output_root = (
        ROOT / str(formal_config["output_directory"])
    ).resolve()
    expected_summaries = [
        output_root
        / f"split-{split_seed}"
        / f"init-{initialization_seed}"
        / arm
        / "summary.json"
        for split_seed in SPLIT_SEEDS
        for initialization_seed in INITIALIZATION_SEEDS
        for arm in ARMS
    ]
    if all(path.is_file() for path in expected_summaries):
        return {
            "schema_version": "nucpred.mayr-site-n-formal-coordinator.v1",
            "status": "pass",
            "parallel_workers": 0,
            "completed_job_count": len(expected_summaries),
        }

    gc.collect()
    gc.freeze()
    context = multiprocessing.get_context("fork")
    processes: list[multiprocessing.Process] = []
    for initialization_seed in INITIALIZATION_SEEDS:
        process = context.Process(
            name=f"mayr-site-n-formal-{initialization_seed}",
            target=_lane_worker,
            kwargs={
                "initialization_seed": initialization_seed,
                "examples": examples,
                "membership": membership,
                "base_config": base_config,
                "base_config_path": base_config_path,
                "formal_config": formal_config,
                "formal_config_path": config_path,
                "dataset_directory": dataset_directory,
                "dataset_verification": dataset_verification,
                "pretraining_verification": pretraining_verification,
                "device": device,
            },
        )
        process.start()
        processes.append(process)
    print(
        json.dumps(
            {
                "event": "formal_matrix_started",
                "parallel_workers": len(processes),
                "pids": {
                    process.name: process.pid for process in processes
                },
                "job_count": len(expected_summaries),
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
            completed = sum(path.is_file() for path in expected_summaries)
            print(
                json.dumps(
                    {
                        "event": "formal_matrix_heartbeat",
                        "completed_job_count": completed,
                        "total_job_count": len(expected_summaries),
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
        raise SiteNCampaignError(f"Formal workers failed: {failures}")
    missing = [
        _display_path(path)
        for path in expected_summaries
        if not path.is_file()
    ]
    if missing:
        raise SiteNCampaignError(f"Formal summaries are missing: {missing}")
    return {
        "schema_version": "nucpred.mayr-site-n-formal-coordinator.v1",
        "status": "pass",
        "parallel_workers": len(processes),
        "completed_job_count": len(expected_summaries),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--formal-config",
        type=Path,
        default=DEFAULT_FORMAL_CONFIG,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--all-jobs", action="store_true")
    parser.add_argument("--split-seed", type=int, choices=SPLIT_SEEDS)
    parser.add_argument(
        "--initialization-seed",
        type=int,
        choices=INITIALIZATION_SEEDS,
    )
    parser.add_argument("--arm", choices=ARMS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    if arguments.all_jobs:
        if any(
            value is not None
            for value in (
                arguments.split_seed,
                arguments.initialization_seed,
                arguments.arm,
            )
        ):
            raise SiteNCampaignError(
                "--all-jobs cannot be combined with single-job axes"
            )
        payload = run_all_formal_jobs(
            formal_config_path=arguments.formal_config,
            device=arguments.device,
        )
    else:
        if any(
            value is None
            for value in (
                arguments.split_seed,
                arguments.initialization_seed,
                arguments.arm,
            )
        ):
            raise SiteNCampaignError(
                "Single job requires --split-seed, "
                "--initialization-seed, and --arm"
            )
        payload = run_formal_job(
            split_seed=int(arguments.split_seed),
            initialization_seed=int(arguments.initialization_seed),
            arm=str(arguments.arm),
            formal_config_path=arguments.formal_config,
            device=arguments.device,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
