"""Run the preregistered validation-only Stage-C conditional-N matrix."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
    seed_everything,
)
from nucpred.training.mayr_site_n_stage_c import (
    MayrSiteNInteractionModel,
    stage_c_target_weights,
    weighted_site_n_loss,
    zero_interaction_output_is_exact,
)

from .nextgen_gate_a import _canonical_sha256, _verify_bound_file
from .site_n import (
    SiteNCampaignError,
    _display_path,
    _evaluate,
    _iter_batches,
    _read_config as _read_site_n_config,
    _write_manifest,
)
from .site_n_formal import (
    INITIALIZATION_SEEDS,
    SPLIT_SEEDS,
    _checkpoint_entry,
    _legacy_transfer,
    _tensor_mapping_sha256,
)


ROOT = get_project_layout().root
DEFAULT_CONFIG = ROOT / "configs/mayr_nextgen_stage_c.toml"
CONFIG_SCHEMA = "nucpred.mayr-nextgen-stage-c-config.v1"
ARMS = (
    "c0_current_source_control",
    "c1_h1_small_negative_weighting",
    "c2_h2_high_n_weighting",
    "c3_explicit_interaction_residual",
    "c4_combined",
)
INTERACTION_ARMS = frozenset({"c3_explicit_interaction_residual", "c4_combined"})
H1_ARMS = frozenset({"c1_h1_small_negative_weighting", "c4_combined"})
H2_ARMS = frozenset({"c2_h2_high_n_weighting", "c4_combined"})


@dataclass(frozen=True, slots=True)
class SelectionOutcome:
    model: MayrSiteNModel
    curves: pd.DataFrame
    best_epoch: int
    best_validation_rmse: float
    validation_metrics: Mapping[str, object]
    validation_predictions: pd.DataFrame
    base_initialization_sha256: str
    post_transfer_initialization_sha256: str
    transfer_audit: Mapping[str, object]
    weight_audit: Mapping[str, object]


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SiteNCampaignError(f"Expected JSON object: {path}")
    return value


def read_config(
    path: str | Path = DEFAULT_CONFIG,
) -> tuple[dict[str, Any], Path, dict[str, Any], Path, dict[str, Any], Path]:
    config_path = Path(path).resolve()
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise SiteNCampaignError("Stage-C config schema changed")
    r2 = config["r2"]
    if tuple(map(str, r2["arms"])) != ARMS:
        raise SiteNCampaignError("Stage-C R2 arms changed")
    if tuple(map(int, r2["split_seeds"])) != SPLIT_SEEDS:
        raise SiteNCampaignError("Stage-C split seeds changed")
    if tuple(map(int, r2["initialization_seeds"])) != INITIALIZATION_SEEDS:
        raise SiteNCampaignError("Stage-C initialization seeds changed")
    if (
        r2.get("selection_metric") != "rmse"
        or r2.get("test_examples_loaded") is not False
        or r2.get("test_predictions_written") is not False
    ):
        raise SiteNCampaignError("Stage-C validation-only contract changed")
    if int(config["maximum_parallel_gpu_processes"]) != 3:
        raise SiteNCampaignError("Stage-C GPU concurrency changed")
    if (
        config.get("test_labels_or_predictions_permitted_for_training_or_selection")
        is not False
        or config.get("dft_or_cdft_computation_permitted") is not False
    ):
        raise SiteNCampaignError("Stage-C authorization is broader than allowed")
    for section, pairs in {
        "authorization": (("path", "sha256"),),
        "stage_b": (
            ("config_path", "config_sha256"),
            ("results_manifest_path", "results_manifest_sha256"),
            ("results_summary_path", "results_summary_sha256"),
        ),
        "dataset": (
            ("candidate_policy_path", "candidate_policy_sha256"),
            ("formal_config_path", "formal_config_sha256"),
        ),
    }.items():
        for path_key, hash_key in pairs:
            _verify_bound_file(
                (ROOT / str(config[section][path_key])).resolve(),
                str(config[section][hash_key]),
            )
    stage_b_manifest = (
        ROOT / str(config["stage_b"]["pass_b_directory"]) / "run_manifest.json"
    ).resolve()
    _verify_bound_file(
        stage_b_manifest, str(config["stage_b"]["pass_b_manifest_sha256"])
    )
    authorization = _load_json((ROOT / str(config["authorization"]["path"])).resolve())
    contract = authorization.get("stage_contract", {})
    if (
        not isinstance(contract, Mapping)
        or contract.get("stage_c_authorized") is not True
        or contract.get("test_labels_or_predictions_authorized") is not False
        or contract.get("dft_or_cdft_computation_authorized") is not False
        or int(contract.get("maximum_parallel_gpu_processes", -1)) != 3
    ):
        raise SiteNCampaignError("Stage-C authorization artifact changed")
    formal_path = (ROOT / str(config["dataset"]["formal_config_path"])).resolve()
    formal = tomllib.loads(formal_path.read_text(encoding="utf-8"))
    base_path = (ROOT / str(formal["base_config"])).resolve()
    base = _read_site_n_config(base_path)
    return config, config_path, formal, formal_path, base, base_path


def _make_stage_c_model(
    *,
    arm: str,
    base_config: Mapping[str, Any],
    vocabulary: SolventVocabulary,
    initialization_seed: int,
    device: torch.device,
    checkpoint: Path,
) -> tuple[MayrSiteNModel, str, str, dict[str, object]]:
    seed_everything(initialization_seed)
    section = base_config["model"]
    kwargs = {
        "num_solvents": len(vocabulary.tokens),
        "hidden_dim": int(section["hidden_dim"]),
        "layers": int(section["message_passing_layers"]),
        "node_embedding_dim": int(section["node_embedding_dim"]),
        "edge_embedding_dim": int(section["edge_embedding_dim"]),
        "solvent_embedding_dim": int(section["solvent_embedding_dim"]),
        "dropout": float(section["dropout"]),
    }
    model: MayrSiteNModel
    if arm in INTERACTION_ARMS:
        model = MayrSiteNInteractionModel(**kwargs)
        if not zero_interaction_output_is_exact(model):
            raise SiteNCampaignError("Interaction residual is not exact-zero")
    else:
        model = MayrSiteNModel(**kwargs)
    base_hash = initialization_sha256(model)
    model = model.to(device)
    audit = _legacy_transfer(model, checkpoint)
    audit = {
        **audit,
        "stage_c_arm": arm,
        "base_transfer": "frozen_legacy_encoder",
        "interaction_residual_present": arm in INTERACTION_ARMS,
        "interaction_residual_exact_zero_after_transfer": (
            zero_interaction_output_is_exact(model)
            if isinstance(model, MayrSiteNInteractionModel)
            else None
        ),
    }
    if (
        arm in INTERACTION_ARMS
        and not audit["interaction_residual_exact_zero_after_transfer"]
    ):
        raise SiteNCampaignError("Transfer changed zero interaction residual")
    return model, base_hash, initialization_sha256(model), audit


def _train_epoch(
    model: MayrSiteNModel,
    batches,
    *,
    target_weights: Mapping[str, float],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    ranking_weight: float,
    gradient_clip_norm: float,
) -> dict[str, float]:
    model.train()
    totals = {"total": 0.0, "regression": 0.0, "ranking": 0.0}
    target_count = 0
    ranking_pairs = 0
    for raw_batch in batches:
        batch = raw_batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(batch.inputs)
        total, parts = weighted_site_n_loss(
            output,
            batch,
            target_weights,
            ranking_weight=ranking_weight,
        )
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip_norm))
        optimizer.step()
        count = batch.inputs.num_sites
        target_count += count
        for name in totals:
            value = total if name == "total" else parts[name]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Unexpected loss component: {name}")
            totals[name] += float(value.detach().cpu()) * count
        ranking_pairs += int(parts["ranking_pairs"])
    if target_count == 0:
        raise SiteNCampaignError("Stage-C epoch had no training targets")
    result = {name: value / target_count for name, value in totals.items()}
    result["ranking_pairs"] = float(ranking_pairs)
    return result


def _fit_selection(
    train: Sequence[SiteNExample],
    validation: Sequence[SiteNExample],
    *,
    arm: str,
    base_config: Mapping[str, Any],
    stage_config: Mapping[str, Any],
    preprocessor: SiteNFoldPreprocessor,
    vocabulary: SolventVocabulary,
    initialization_seed: int,
    device: torch.device,
    checkpoint: Path,
) -> SelectionOutcome:
    model, base_hash, post_hash, transfer_audit = _make_stage_c_model(
        arm=arm,
        base_config=base_config,
        vocabulary=vocabulary,
        initialization_seed=initialization_seed,
        device=device,
        checkpoint=checkpoint,
    )
    optimization = stage_config["r2"]["optimization"]
    target_weights, weight_audit = stage_c_target_weights(
        train,
        use_h1=arm in H1_ARMS,
        use_h2=arm in H2_ARMS,
        maximum_weight=float(optimization["maximum_target_weight"]),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
    )
    rows: list[dict[str, object]] = []
    best_epoch = 0
    best_rmse = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    for epoch in range(1, int(optimization["maximum_epochs"]) + 1):
        training = _train_epoch(
            model,
            _iter_batches(
                train,
                batch_size=int(optimization["batch_size_contexts"]),
                preprocessor=preprocessor,
                vocabulary=vocabulary,
                shuffle_seed=initialization_seed + epoch,
            ),
            target_weights=target_weights,
            optimizer=optimizer,
            device=device,
            ranking_weight=float(optimization["ranking_weight"]),
            gradient_clip_norm=float(optimization["gradient_clip_norm"]),
        )
        metrics, _ = _evaluate(
            model,
            validation,
            preprocessor=preprocessor,
            vocabulary=vocabulary,
            batch_size=int(optimization["batch_size_contexts"]),
            device=device,
        )
        rmse = float(metrics["rmse"])
        improved = rmse < (
            best_rmse - float(optimization["minimum_validation_metric_delta"])
        )
        if improved:
            best_epoch = epoch
            best_rmse = rmse
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        rows.append(
            {
                "epoch": epoch,
                "selection_metric": "rmse",
                "is_validation_best": improved,
                **{f"train_{name}": value for name, value in training.items()},
                "validation_mae": float(metrics["mae"]),
                "validation_rmse": rmse,
                "validation_r2": float(metrics["r2"]),
            }
        )
        if epoch >= int(optimization["minimum_epochs"]) and stale >= int(
            optimization["early_stopping_patience"]
        ):
            break
    if best_state is None or best_epoch <= 0:
        raise SiteNCampaignError("Stage-C selection produced no best state")
    model.load_state_dict(best_state, strict=True)
    metrics, predictions = _evaluate(
        model,
        validation,
        preprocessor=preprocessor,
        vocabulary=vocabulary,
        batch_size=int(optimization["batch_size_contexts"]),
        device=device,
    )
    if not math.isclose(float(metrics["rmse"]), best_rmse, rel_tol=0.0, abs_tol=1e-7):
        raise SiteNCampaignError("Restored Stage-C checkpoint changed")
    return SelectionOutcome(
        model=model,
        curves=pd.DataFrame(rows),
        best_epoch=best_epoch,
        best_validation_rmse=best_rmse,
        validation_metrics=metrics,
        validation_predictions=predictions,
        base_initialization_sha256=base_hash,
        post_transfer_initialization_sha256=post_hash,
        transfer_audit=transfer_audit,
        weight_audit=weight_audit,
    )


def _split_examples(
    dataset: Path,
    *,
    split_seed: int,
) -> tuple[list[SiteNExample], list[SiteNExample], dict[str, object]]:
    # The loader instantiates only the requested role; test examples are never built.
    train = load_site_n_examples(dataset, split_seed=split_seed, role="train")
    validation = load_site_n_examples(dataset, split_seed=split_seed, role="validation")
    membership = pd.read_csv(dataset / "split_membership.csv")
    selected = membership.loc[membership["split_seed"].eq(split_seed)].copy()
    train_conn = set(
        selected.loc[selected["role"].eq("train"), "connectivity_id"].astype(str)
    )
    validation_conn = set(
        selected.loc[selected["role"].eq("validation"), "connectivity_id"].astype(str)
    )
    test_conn = set(
        selected.loc[selected["role"].eq("test"), "connectivity_id"].astype(str)
    )
    if (
        train_conn & validation_conn
        or train_conn & test_conn
        or validation_conn & test_conn
    ):
        raise SiteNCampaignError("Parent split leaks connectivity roles")
    audit: dict[str, object] = {
        "schema_version": "nucpred.mayr-stage-c-validation-split-audit.v1",
        "split_seed": int(split_seed),
        "train_context_count": len(train),
        "train_target_count": sum(item.num_sites for item in train),
        "train_connectivity_count": len(train_conn),
        "validation_context_count": len(validation),
        "validation_target_count": sum(item.num_sites for item in validation),
        "validation_connectivity_count": len(validation_conn),
        "test_membership_connectivity_count": len(test_conn),
        "test_examples_instantiated": 0,
        "test_predictions_computed": 0,
        "connectivity_disjoint": True,
    }
    return train, validation, audit


def _checkpoint_contract(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    formal_path: Path,
    base_path: Path,
    dataset: Path,
    checkpoint: Path,
    split_seed: int,
    initialization_seed: int,
    arm: str,
    split_audit: Mapping[str, object],
) -> dict[str, object]:
    sources = {
        "stage_c_config": config_path,
        "formal_config": formal_path,
        "base_config": base_path,
        "runner": Path(__file__).resolve(),
        "stage_c_model": (
            ROOT / "src/nucpred/training/mayr_site_n_stage_c.py"
        ).resolve(),
        "base_model": (ROOT / "src/nucpred/training/mayr_site_n.py").resolve(),
        "base_runner": (ROOT / "src/nucpred/experiments/mayr/site_n.py").resolve(),
        "dataset_loader": (ROOT / "src/nucpred/datasets/mayr_site_n.py").resolve(),
        "dataset_manifest": dataset / "dataset_manifest.json",
        "split_manifest": dataset / "split_manifest.json",
        "authorization": (ROOT / str(config["authorization"]["path"])).resolve(),
        "legacy_checkpoint": checkpoint,
    }
    contract: dict[str, object] = {
        "schema_version": "nucpred.mayr-nextgen-stage-c-r2-job-contract.v1",
        "campaign_id": str(config["campaign_id"]),
        "split_seed": int(split_seed),
        "initialization_seed": int(initialization_seed),
        "arm": arm,
        "source_hashes": {name: sha256_file(path) for name, path in sources.items()},
        "split_audit": dict(split_audit),
        "selection_metric": "rmse",
        "selection_roles": ["train", "validation"],
        "test_examples_instantiated": 0,
        "test_predictions_computed": 0,
        "final_refit_performed": False,
        "formal_feature_scope": "strict_no_dft_rdkit_xtb",
        "dft_or_cdft_computation_authorized": False,
    }
    contract["contract_sha256"] = _canonical_sha256(contract)
    return contract


def _save_checkpoint(
    path: Path,
    *,
    outcome: SelectionOutcome,
    arm: str,
    split_seed: int,
    initialization_seed: int,
    preprocessor: SiteNFoldPreprocessor,
    vocabulary: SolventVocabulary,
    contract: Mapping[str, object],
) -> None:
    state = {
        name: tensor.detach().cpu()
        for name, tensor in outcome.model.state_dict().items()
    }
    payload = {
        "schema_version": "nucpred.mayr-nextgen-stage-c-r2-checkpoint.v1",
        "phase": "validation_selection",
        "arm": arm,
        "split_seed": int(split_seed),
        "initialization_seed": int(initialization_seed),
        "selection_metric": "rmse",
        "model_architecture": outcome.model.architecture,
        "model_state_dict": state,
        "model_state_sha256": _tensor_mapping_sha256(state),
        "preprocessor": preprocessor.to_json(),
        "solvent_vocabulary": list(vocabulary.tokens),
        "contract": dict(contract),
        "transfer_audit": dict(outcome.transfer_audit),
        "weight_audit": dict(outcome.weight_audit),
    }
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _verify_legacy_checkpoint(
    formal: Mapping[str, Any],
    initialization_seed: int,
) -> tuple[Mapping[str, Any], Path, dict[str, object]]:
    entry = _checkpoint_entry(formal, "legacy_checkpoints", initialization_seed)
    checkpoint = (ROOT / str(entry["path"])).resolve()
    observed = sha256_file(checkpoint)
    if observed != str(entry["sha256"]):
        raise SiteNCampaignError("Frozen legacy checkpoint hash changed")
    payload = load_legacy_pretraining_checkpoint(checkpoint)
    if int(payload["init_seed"]) != int(entry["pretraining_seed"]):
        raise SiteNCampaignError("Frozen legacy checkpoint seed changed")
    verification = {
        "schema_version": "nucpred.mayr-stage-c-legacy-checkpoint-audit.v1",
        "status": "pass",
        "path": _display_path(checkpoint),
        "sha256": observed,
        "pretraining_seed": int(entry["pretraining_seed"]),
        "backbone_state_sha256": payload["backbone_state_sha256"],
        "historical_source_parity_claimed": False,
        "reuse_semantics": "frozen_checkpoint_identity_and_internal_state_only",
    }
    return entry, checkpoint, verification


def run_job(
    *,
    split_seed: int,
    initialization_seed: int,
    arm: str,
    config_path: str | Path = DEFAULT_CONFIG,
    device: str = "cuda:0",
) -> dict[str, object]:
    started = time.perf_counter()
    if (
        split_seed not in SPLIT_SEEDS
        or initialization_seed not in INITIALIZATION_SEEDS
        or arm not in ARMS
    ):
        raise SiteNCampaignError("Unregistered Stage-C R2 job axis")
    selected_device = torch.device(device)
    if selected_device.type != "cuda" or not torch.cuda.is_available():
        raise SiteNCampaignError("Stage-C R2 matrix requires CUDA")
    config, config_file, formal, formal_path, base, base_path = read_config(config_path)
    dataset = (ROOT / str(config["dataset"]["directory"])).resolve()
    dataset_verification = verify_dataset(dataset)
    train, validation, split_audit = _split_examples(dataset, split_seed=split_seed)
    entry, checkpoint, checkpoint_verification = _verify_legacy_checkpoint(
        formal, initialization_seed
    )
    contract = _checkpoint_contract(
        config=config,
        config_path=config_file,
        formal_path=formal_path,
        base_path=base_path,
        dataset=dataset,
        checkpoint=checkpoint,
        split_seed=split_seed,
        initialization_seed=initialization_seed,
        arm=arm,
        split_audit=split_audit,
    )
    target = (
        (ROOT / str(config["r2"]["output_directory"])).resolve()
        / f"split-{split_seed}"
        / f"init-{initialization_seed}"
        / arm
    )
    summary_path = target / "summary.json"
    if summary_path.is_file():
        existing = _load_json(summary_path)
        if existing.get("status") == "pass" and existing.get("contract") == contract:
            return existing
        raise SiteNCampaignError(f"Existing Stage-C job is stale: {target}")
    if target.exists():
        raise SiteNCampaignError(f"Partial Stage-C job exists: {target}")
    preprocessor = fit_site_n_preprocessor(train)
    vocabulary = SolventVocabulary.from_values(
        [example.solvent_raw for example in train]
    )
    output_root = target.parents[2]
    try:
        outcome = _fit_selection(
            train,
            validation,
            arm=arm,
            base_config=base,
            stage_config=config,
            preprocessor=preprocessor,
            vocabulary=vocabulary,
            initialization_seed=initialization_seed,
            device=selected_device,
            checkpoint=checkpoint,
        )
        metadata = {
            target_id: example.connectivity_id
            for example in validation
            for target_id in example.target_ids
        }
        predictions = outcome.validation_predictions.copy()
        predictions["connectivity_id"] = predictions["target_id"].map(metadata)
        if predictions["connectivity_id"].isna().any():
            raise SiteNCampaignError("Validation connectivity mapping failed")
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{arm}.staging-", dir=target.parent))
        try:
            outcome.curves.to_csv(
                staging / "selection_loss_curves.csv",
                index=False,
                lineterminator="\n",
            )
            predictions.to_parquet(
                staging / "validation_predictions.parquet",
                index=False,
                engine="pyarrow",
                compression="zstd",
            )
            atomic_write_json(
                staging / "validation_metrics.json",
                outcome.validation_metrics,
                ensure_ascii=False,
            )
            atomic_write_json(
                staging / "preprocessor.json",
                preprocessor.to_json(),
                ensure_ascii=False,
            )
            atomic_write_json(
                staging / "transfer_audit.json",
                outcome.transfer_audit,
                ensure_ascii=False,
            )
            atomic_write_json(
                staging / "target_weight_audit.json",
                outcome.weight_audit,
                ensure_ascii=False,
            )
            _save_checkpoint(
                staging / "selection_checkpoint.pt",
                outcome=outcome,
                arm=arm,
                split_seed=split_seed,
                initialization_seed=initialization_seed,
                preprocessor=preprocessor,
                vocabulary=vocabulary,
                contract=contract,
            )
            summary: dict[str, object] = {
                "schema_version": "nucpred.mayr-nextgen-stage-c-r2-job.v1",
                "status": "pass",
                "campaign_id": config["campaign_id"],
                "split_seed": int(split_seed),
                "initialization_seed": int(initialization_seed),
                "pretraining_seed": int(entry["pretraining_seed"]),
                "arm": arm,
                "contract": contract,
                "dataset_verification": dataset_verification,
                "checkpoint_verification": checkpoint_verification,
                "split_audit": split_audit,
                "selection_metric": "rmse",
                "selection_best_epoch": outcome.best_epoch,
                "selection_best_validation_rmse": outcome.best_validation_rmse,
                "validation_metrics": dict(outcome.validation_metrics),
                "base_initialization_sha256": (outcome.base_initialization_sha256),
                "post_transfer_initialization_sha256": (
                    outcome.post_transfer_initialization_sha256
                ),
                "transfer_audit": dict(outcome.transfer_audit),
                "weight_audit": dict(outcome.weight_audit),
                "test_examples_instantiated": 0,
                "test_predictions_computed": 0,
                "test_prediction_files_written": 0,
                "final_refit_performed": False,
                "formal_feature_scope": "strict_no_dft_rdkit_xtb",
                "dft_or_cdft_computation_performed": False,
                "device": str(selected_device),
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "wall_seconds": time.perf_counter() - started,
            }
            atomic_write_json(staging / "summary.json", summary, ensure_ascii=False)
            (staging / "run.log").write_text(
                "\n".join(
                    (
                        "status=pass",
                        f"split_seed={split_seed}",
                        f"initialization_seed={initialization_seed}",
                        f"arm={arm}",
                        "selection_metric=rmse",
                        f"selection_best_epoch={outcome.best_epoch}",
                        (
                            "selection_validation_rmse="
                            f"{outcome.best_validation_rmse:.12f}"
                        ),
                        "test_examples_instantiated=0",
                        "test_predictions_computed=0",
                        "test_prediction_files_written=0",
                        "final_refit_performed=false",
                        f"contract_sha256={contract['contract_sha256']}",
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
        output_root.mkdir(parents=True, exist_ok=True)
        token = f"split-{split_seed}__init-{initialization_seed}__{arm}"
        atomic_write_json(
            output_root / f"{token}.failure.json",
            {
                "schema_version": "nucpred.mayr-nextgen-stage-c-r2-failure.v1",
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
        raise


def _lane_worker(
    *,
    initialization_seed: int,
    config_path: Path,
    device: str,
) -> None:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    for split_seed in SPLIT_SEEDS:
        for arm in ARMS:
            summary = run_job(
                split_seed=split_seed,
                initialization_seed=initialization_seed,
                arm=arm,
                config_path=config_path,
                device=device,
            )
            print(
                json.dumps(
                    {
                        "event": "stage_c_r2_job_complete",
                        "split_seed": split_seed,
                        "initialization_seed": initialization_seed,
                        "arm": arm,
                        "best_epoch": summary["selection_best_epoch"],
                        "validation_rmse": summary["selection_best_validation_rmse"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )


def run_all(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    device: str = "cuda:0",
) -> dict[str, object]:
    if not str(device).startswith("cuda"):
        raise SiteNCampaignError("Stage-C R2 matrix requires CUDA")
    config, config_file, *_ = read_config(config_path)
    output_root = (ROOT / str(config["r2"]["output_directory"])).resolve()
    expected = [
        output_root
        / f"split-{split_seed}"
        / f"init-{initialization_seed}"
        / arm
        / "summary.json"
        for split_seed in SPLIT_SEEDS
        for initialization_seed in INITIALIZATION_SEEDS
        for arm in ARMS
    ]
    if all(path.is_file() for path in expected):
        return {
            "schema_version": "nucpred.mayr-nextgen-stage-c-r2-coordinator.v1",
            "status": "pass",
            "parallel_workers": 0,
            "completed_job_count": len(expected),
        }
    gc.collect()
    gc.freeze()
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(
            name=f"mayr-stage-c-r2-{seed}",
            target=_lane_worker,
            kwargs={
                "initialization_seed": seed,
                "config_path": config_file,
                "device": device,
            },
        )
        for seed in INITIALIZATION_SEEDS
    ]
    for process in processes:
        process.start()
    print(
        json.dumps(
            {
                "event": "stage_c_r2_matrix_started",
                "parallel_workers": len(processes),
                "job_count": len(expected),
                "pids": {p.name: p.pid for p in processes},
            },
            sort_keys=True,
        ),
        flush=True,
    )
    last_heartbeat = 0.0
    while any(process.is_alive() for process in processes):
        for process in processes:
            process.join(timeout=1.0)
        if time.monotonic() - last_heartbeat >= 30.0:
            print(
                json.dumps(
                    {
                        "event": "stage_c_r2_matrix_heartbeat",
                        "completed_job_count": sum(path.is_file() for path in expected),
                        "total_job_count": len(expected),
                        "workers": {
                            p.name: {
                                "pid": p.pid,
                                "alive": p.is_alive(),
                                "exitcode": p.exitcode,
                            }
                            for p in processes
                        },
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            last_heartbeat = time.monotonic()
    for process in processes:
        process.join()
    failures = {p.name: p.exitcode for p in processes if p.exitcode != 0}
    if failures:
        raise SiteNCampaignError(f"Stage-C R2 workers failed: {failures}")
    missing = [_display_path(path) for path in expected if not path.is_file()]
    if missing:
        raise SiteNCampaignError(f"Stage-C R2 summaries are missing: {missing}")
    return {
        "schema_version": "nucpred.mayr-nextgen-stage-c-r2-coordinator.v1",
        "status": "pass",
        "parallel_workers": len(processes),
        "completed_job_count": len(expected),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--all-jobs", action="store_true")
    parser.add_argument("--split-seed", type=int, choices=SPLIT_SEEDS)
    parser.add_argument("--initialization-seed", type=int, choices=INITIALIZATION_SEEDS)
    parser.add_argument("--arm", choices=ARMS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.all_jobs:
        if any(
            value is not None
            for value in (args.split_seed, args.initialization_seed, args.arm)
        ):
            raise SiteNCampaignError(
                "--all-jobs cannot be combined with single-job axes"
            )
        payload = run_all(config_path=args.config, device=args.device)
    else:
        if any(
            value is None
            for value in (args.split_seed, args.initialization_seed, args.arm)
        ):
            raise SiteNCampaignError(
                "Single job requires split, initialization, and arm"
            )
        payload = run_job(
            split_seed=int(args.split_seed),
            initialization_seed=int(args.initialization_seed),
            arm=str(args.arm),
            config_path=args.config,
            device=args.device,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
