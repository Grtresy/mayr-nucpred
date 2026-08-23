"""Run the authorized Stage-E-B frozen-C2 development matrix."""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping, Sequence
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

import numpy as np
import pandas as pd

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from nucpred.artifacts.catalog import ArtifactCatalog
from nucpred.core.files import atomic_write_json, sha256_file
from nucpred.datasets.mayr_site_n import verify_dataset
from nucpred.project import get_project_layout
from nucpred.training.mayr_site_n import (
    SiteNExample,
    seed_everything,
)
from nucpred.training.mayr_site_n_stage_c import stage_c_target_weights
from nucpred.training.mayr_site_n_stage_d import (
    PairedSolventDefinition,
    stage_d_paired_solvent_definitions,
)
from nucpred.training.mayr_site_n_stage_e_a import (
    SolventCenterGroup,
    stage_e_a_site_n_loss,
    stage_e_a_solvent_center_groups,
)
from nucpred.training.mayr_site_n_stage_e_b import (
    E_B_N1,
    E_B_N2,
    FAMILY_CHANNELS,
    STAGE_E_B_ARMS,
    MayrSiteNStageEBResidualModel,
    frozen_base_parameters_are_frozen,
    structural_family_indicators,
    trainable_parameter_count,
    zero_residual_output_is_exact,
)

from .nextgen_gate_a import _canonical_sha256, _verify_bound_file
from .nextgen_stage_d_r2 import _split_examples, _validation_pair_predictions
from .nextgen_stage_e_a_r2 import (
    FrozenC2,
    _center_residual_table,
    _component_predictions,
    _load_frozen_c2,
    _load_json,
    _project_path,
    _training_batches,
)
from .site_n import (
    SiteNCampaignError,
    _evaluate,
    _iter_batches,
    _write_manifest,
)
from .site_n_formal import (
    INITIALIZATION_SEEDS,
    SPLIT_SEEDS,
    _tensor_mapping_sha256,
)


ROOT = get_project_layout().root
DEFAULT_CONFIG = ROOT / "configs/mayr_nextgen_stage_e_b.toml"
CONFIG_SCHEMA = "nucpred.mayr-nextgen-stage-e-b-config.v1"
EXPECTED_STATUS = "awaiting_stage_e_b_results_gate"
ARMS = STAGE_E_B_ARMS


@dataclass(frozen=True, slots=True)
class SelectionOutcome:
    model: MayrSiteNStageEBResidualModel
    curves: pd.DataFrame
    best_epoch: int
    best_validation_rmse: float
    validation_metrics: Mapping[str, object]
    validation_predictions: pd.DataFrame
    validation_components: pd.DataFrame
    validation_pair_predictions: pd.DataFrame
    validation_pair_metrics: Mapping[str, object]
    train_center_residuals: pd.DataFrame
    frozen_parent_metrics: Mapping[str, object]
    freeze_audit: Mapping[str, object]
    target_weight_audit: Mapping[str, object]
    paired_solvent_audit: Mapping[str, object]
    center_group_audit: Mapping[str, object]
    gate_audit: Mapping[str, object]


def read_config(path: str | Path = DEFAULT_CONFIG) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).resolve()
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise SiteNCampaignError("Stage-E-B config schema changed")
    if config.get("status_after_completion") != EXPECTED_STATUS:
        raise SiteNCampaignError("Stage-E-B result gate changed")
    r2 = config["r2"]
    if tuple(map(str, r2["arms"])) != ARMS:
        raise SiteNCampaignError("Stage-E-B arms changed")
    if tuple(map(int, r2["split_seeds"])) != SPLIT_SEEDS:
        raise SiteNCampaignError("Stage-E-B split seeds changed")
    if tuple(map(int, r2["initialization_seeds"])) != INITIALIZATION_SEEDS:
        raise SiteNCampaignError("Stage-E-B initialization seeds changed")
    forbidden = (
        "test_labels_examples_or_predictions_permitted",
        "formal_calibration_permitted",
        "absolute_probability_publication_permitted",
        "final_refit_permitted",
        "d_n4_combination_permitted",
        "dft_or_cdft_computation_permitted",
        "automatic_continuation_permitted",
    )
    if any(config.get(key) is not False for key in forbidden):
        raise SiteNCampaignError("Stage-E-B forbidden scope changed")
    if int(config.get("maximum_parallel_gpu_processes", -1)) != 3:
        raise SiteNCampaignError("Stage-E-B GPU ceiling changed")
    if (
        r2.get("selection_metric") != "rmse"
        or list(r2.get("selection_roles", [])) != ["train", "validation"]
        or r2.get("epoch_zero_frozen_c2_is_eligible") is not True
        or any(
            r2.get(key) is not False
            for key in (
                "test_examples_permitted",
                "test_predictions_permitted",
                "final_refit_permitted",
                "dft_or_cdft_permitted",
            )
        )
    ):
        raise SiteNCampaignError("Stage-E-B development-only contract changed")
    if tuple(r2["e_b_n1"]["active_site_types"]) != (
        "bond",
        "delocalized_region",
    ):
        raise SiteNCampaignError("E-B-N1 mask changed")
    if tuple(r2["e_b_n2"]["family_channels"]) != FAMILY_CHANNELS:
        raise SiteNCampaignError("E-B-N2 family channels changed")
    if (
        r2["e_b_n2"]["mayr_class_labels_as_model_input"] is not False
        or r2["e_b_n2"]["target_or_n_value_as_family_input"] is not False
    ):
        raise SiteNCampaignError("E-B-N2 target-independent contract changed")

    bindings = (
        ("authorization", "path", "sha256"),
        ("parents", "stage_e_a_catalog_path", "stage_e_a_catalog_sha256"),
        ("parents", "stage_e_a_config_path", "stage_e_a_config_sha256"),
        (
            "parents",
            "stage_e_a_results_manifest_path",
            "stage_e_a_results_manifest_sha256",
        ),
        ("parents", "stage_c_config_path", "stage_c_config_sha256"),
        (
            "parents",
            "stage_c_r2_manifest_path",
            "stage_c_r2_manifest_sha256",
        ),
        ("dataset", "manifest_path", "manifest_sha256"),
        ("dataset", "split_manifest_path", "split_manifest_sha256"),
    )
    for section, path_key, hash_key in bindings:
        _verify_bound_file(
            _project_path(config[section][path_key], label=f"{section}.{path_key}"),
            str(config[section][hash_key]),
        )
    decision = _load_json(
        _project_path(config["authorization"]["path"], label="authorization.path")
    )
    contract = decision.get("stage_contract")
    if (
        not isinstance(contract, Mapping)
        or contract.get("stage_e_b_development_training_authorized") is not True
        or contract.get("authorized_arms") != list(ARMS)
        or contract.get("maximum_parallel_gpu_processes") != 3
        or contract.get("status_after_completion") != EXPECTED_STATUS
        or any(
            contract.get(key) is not False
            for key in (
                "d_n4_combination_authorized",
                "test_labels_examples_or_predictions_authorized",
                "formal_calibration_authorized",
                "absolute_probability_publication_authorized",
                "final_refit_authorized",
                "dft_or_cdft_computation_authorized",
                "automatic_continuation_authorized",
            )
        )
    ):
        raise SiteNCampaignError("Stage-E-B authorization changed")
    parent = ArtifactCatalog().verify(str(config["parents"]["stage_e_a_run_id"]))
    if parent.get("status") != "pass":
        raise SiteNCampaignError("Stage-E-A parent catalog no longer verifies")
    return config, config_path


def _make_model(
    frozen: FrozenC2,
    *,
    arm: str,
    initialization_seed: int,
    device: torch.device,
) -> MayrSiteNStageEBResidualModel:
    seed_everything(initialization_seed)
    model = MayrSiteNStageEBResidualModel(frozen_base=frozen.model, arm=arm)
    if not zero_residual_output_is_exact(model):
        raise SiteNCampaignError("Stage-E-B residual is not exact-zero")
    if not frozen_base_parameters_are_frozen(model):
        raise SiteNCampaignError("Stage-E-B C2 is trainable")
    if trainable_parameter_count(model) <= 0:
        raise SiteNCampaignError("Stage-E-B has no trainable parameters")
    return model.to(device)


def _ordinary_batches(
    train: Sequence[SiteNExample],
    *,
    frozen: FrozenC2,
    batch_size: int,
    shuffle_seed: int,
) -> Iterator:
    yield from _iter_batches(
        train,
        batch_size=batch_size,
        preprocessor=frozen.preprocessor,
        vocabulary=frozen.vocabulary,
        shuffle_seed=shuffle_seed,
    )


def _train_epoch(
    model: MayrSiteNStageEBResidualModel,
    batches: Iterator,
    *,
    target_weights: Mapping[str, float],
    pairs: Sequence[PairedSolventDefinition],
    center_groups: Sequence[SolventCenterGroup],
    paired_weight: float,
    center_weight: float,
    config: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    names = (
        "total",
        "regression",
        "ranking",
        "paired_solvent_delta",
        "center_penalty",
        "residual_shrinkage",
        "gate_shrinkage",
    )
    totals = dict.fromkeys(names, 0.0)
    count_total = 0
    optimization = config["r2"]["optimization"]
    for raw_batch in batches:
        batch = raw_batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(batch.inputs)
        total, parts = stage_e_a_site_n_loss(
            output,
            batch,
            target_weights,
            ranking_weight=float(optimization["ranking_weight"]),
            paired_solvent_pairs=pairs,
            paired_solvent_weight=paired_weight,
            center_groups=center_groups,
            center_penalty_weight=center_weight,
            residual_shrinkage_weight=float(config["r2"]["residual_shrinkage_weight"]),
            gate_shrinkage_weight=0.0,
        )
        total.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            float(optimization["gradient_clip_norm"]),
        )
        optimizer.step()
        count = batch.inputs.num_sites
        count_total += count
        for name in names:
            value = total if name == "total" else parts[name]
            if isinstance(value, torch.Tensor):
                totals[name] += float(value.detach().cpu()) * count
    if count_total == 0:
        raise SiteNCampaignError("Stage-E-B epoch had no targets")
    return {name: value / count_total for name, value in totals.items()}


def _gate_audit(
    model: MayrSiteNStageEBResidualModel,
    examples: Sequence[SiteNExample],
    *,
    frozen: FrozenC2,
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    counts = {channel: 0 for channel in FAMILY_CHANNELS}
    active = 0
    targets = 0
    with torch.no_grad():
        for raw in _iter_batches(
            examples,
            batch_size=batch_size,
            preprocessor=frozen.preprocessor,
            vocabulary=frozen.vocabulary,
            shuffle_seed=None,
        ):
            batch = raw.to(device)
            output = model(batch.inputs)
            active += int(output.residual_gate.sum().item())
            targets += batch.inputs.num_sites
            if model.arm == E_B_N2:
                indicators = structural_family_indicators(batch.inputs)
                for index, channel in enumerate(FAMILY_CHANNELS):
                    counts[channel] += int(indicators[:, index].sum().item())
    return {
        "schema_version": "nucpred.mayr-stage-e-b-gate-audit.v1",
        "arm": model.arm,
        "target_count": targets,
        "active_target_count": active,
        "inactive_exact_c2_fallback_count": targets - active,
        "family_channel_counts": counts if model.arm == E_B_N2 else {},
        "gate_is_learned": False,
        "mayr_class_or_target_used": False,
    }


def _fit_selection(
    train: Sequence[SiteNExample],
    validation: Sequence[SiteNExample],
    *,
    arm: str,
    config: Mapping[str, Any],
    frozen: FrozenC2,
    initialization_seed: int,
    device: torch.device,
) -> SelectionOutcome:
    model = _make_model(
        frozen,
        arm=arm,
        initialization_seed=initialization_seed,
        device=device,
    )
    initial_hash = _tensor_mapping_sha256(model.frozen_base.state_dict())
    weights, weight_audit = stage_c_target_weights(
        train,
        use_h1=False,
        use_h2=True,
        maximum_weight=float(config["r2"]["optimization"]["maximum_target_weight"]),
    )
    weight_audit = {
        **weight_audit,
        "stage_e_b_reuse": "same_train_only_C2_H2_definition",
        "fit_roles": ["train"],
    }
    pairs, pair_audit = stage_d_paired_solvent_definitions(train)
    center_groups, center_audit = stage_e_a_solvent_center_groups(train)
    arm_config = config["r2"]["e_b_n1" if arm == E_B_N1 else "e_b_n2"]
    paired_weight = float(arm_config["paired_solvent_weight"])
    center_weight = float(arm_config["center_penalty_weight"])
    pair_audit = {
        **pair_audit,
        "arm": arm,
        "enabled": paired_weight > 0.0,
        "pair_aware_batching": bool(arm_config["pair_aware_batching"]),
    }
    optimization = config["r2"]["optimization"]
    batch_size = int(optimization["batch_size_contexts"])
    frozen_metrics, frozen_predictions = _evaluate(
        model.frozen_base,
        validation,
        preprocessor=frozen.preprocessor,
        vocabulary=frozen.vocabulary,
        batch_size=batch_size,
        device=device,
    )
    initial_metrics, initial_predictions = _evaluate(
        model,
        validation,
        preprocessor=frozen.preprocessor,
        vocabulary=frozen.vocabulary,
        batch_size=batch_size,
        device=device,
    )
    base_values = frozen_predictions.sort_values("target_id")["N_pred"].to_numpy()
    initial_values = initial_predictions.sort_values("target_id")["N_pred"].to_numpy()
    if not np.array_equal(base_values, initial_values):
        raise SiteNCampaignError("Stage-E-B epoch-zero differs from frozen C2")

    best_epoch = 0
    best_rmse = float(initial_metrics["rmse"])
    best_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    rows: list[dict[str, object]] = [
        {
            "epoch": 0,
            "selection_metric": "rmse",
            "is_validation_best": True,
            "epoch_zero_frozen_c2": True,
            "validation_mae": float(initial_metrics["mae"]),
            "validation_rmse": best_rmse,
            "validation_r2": float(initial_metrics["r2"]),
        }
    ]
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
    )
    stale = 0
    for epoch in range(1, int(optimization["maximum_epochs"]) + 1):
        if bool(arm_config["pair_aware_batching"]):
            batches = _training_batches(
                train,
                pairs=pairs,
                batch_size=batch_size,
                preprocessor=frozen.preprocessor,
                vocabulary=frozen.vocabulary,
                shuffle_seed=initialization_seed + epoch,
            )
        else:
            batches = _ordinary_batches(
                train,
                frozen=frozen,
                batch_size=batch_size,
                shuffle_seed=initialization_seed + epoch,
            )
        training = _train_epoch(
            model,
            batches,
            target_weights=weights,
            pairs=pairs,
            center_groups=center_groups,
            paired_weight=paired_weight,
            center_weight=center_weight,
            config=config,
            optimizer=optimizer,
            device=device,
        )
        metrics, _ = _evaluate(
            model,
            validation,
            preprocessor=frozen.preprocessor,
            vocabulary=frozen.vocabulary,
            batch_size=batch_size,
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
                "epoch_zero_frozen_c2": False,
                **{f"train_{key}": value for key, value in training.items()},
                "validation_mae": float(metrics["mae"]),
                "validation_rmse": rmse,
                "validation_r2": float(metrics["r2"]),
            }
        )
        if epoch >= int(optimization["minimum_epochs"]) and stale >= int(
            optimization["early_stopping_patience"]
        ):
            break
    model.load_state_dict(best_state, strict=True)
    model.frozen_base.eval()
    final_hash = _tensor_mapping_sha256(model.frozen_base.state_dict())
    if final_hash != initial_hash:
        raise SiteNCampaignError("Frozen C2 changed during Stage-E-B")
    metrics, predictions = _evaluate(
        model,
        validation,
        preprocessor=frozen.preprocessor,
        vocabulary=frozen.vocabulary,
        batch_size=batch_size,
        device=device,
    )
    if not math.isclose(float(metrics["rmse"]), best_rmse, abs_tol=1e-7):
        raise SiteNCampaignError("Restored Stage-E-B checkpoint changed")
    components = _component_predictions(
        model,
        validation,
        preprocessor=frozen.preprocessor,
        vocabulary=frozen.vocabulary,
        batch_size=batch_size,
        device=device,
    )
    train_components = _component_predictions(
        model,
        train,
        preprocessor=frozen.preprocessor,
        vocabulary=frozen.vocabulary,
        batch_size=batch_size,
        device=device,
    )
    pair_predictions, pair_metrics = _validation_pair_predictions(
        validation, predictions
    )
    freeze_audit = {
        "schema_version": "nucpred.mayr-stage-e-b-freeze-audit.v1",
        "status": "pass",
        "frozen_c2_checkpoint_sha256": frozen.checkpoint_sha256,
        "frozen_c2_internal_state_sha256": frozen.model_state_sha256,
        "base_state_before_training_sha256": initial_hash,
        "base_state_after_training_sha256": final_hash,
        "base_state_bitwise_unchanged": initial_hash == final_hash,
        "base_parameters_require_grad_false": frozen_base_parameters_are_frozen(model),
        "base_forced_eval_mode": not model.frozen_base.training,
        "epoch_zero_prediction_exactly_equal_to_c2": True,
        "trainable_parameter_count": trainable_parameter_count(model),
    }
    return SelectionOutcome(
        model=model,
        curves=pd.DataFrame(rows),
        best_epoch=best_epoch,
        best_validation_rmse=best_rmse,
        validation_metrics=metrics,
        validation_predictions=predictions,
        validation_components=components,
        validation_pair_predictions=pair_predictions,
        validation_pair_metrics=pair_metrics,
        train_center_residuals=_center_residual_table(train_components, center_groups),
        frozen_parent_metrics=frozen_metrics,
        freeze_audit=freeze_audit,
        target_weight_audit=weight_audit,
        paired_solvent_audit=pair_audit,
        center_group_audit=center_audit,
        gate_audit=_gate_audit(
            model,
            validation,
            frozen=frozen,
            batch_size=batch_size,
            device=device,
        ),
    )


def _job_contract(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    dataset: Path,
    frozen: FrozenC2,
    split_seed: int,
    initialization_seed: int,
    arm: str,
    split_audit: Mapping[str, object],
) -> dict[str, object]:
    sources = {
        "stage_e_b_config": config_path,
        "authorization": _project_path(
            config["authorization"]["path"], label="authorization.path"
        ),
        "runner": Path(__file__).resolve(),
        "stage_e_b_model": (
            ROOT / "src/nucpred/training/mayr_site_n_stage_e_b.py"
        ).resolve(),
        "stage_e_a_loss": (
            ROOT / "src/nucpred/training/mayr_site_n_stage_e_a.py"
        ).resolve(),
        "stage_c_model": (
            ROOT / "src/nucpred/training/mayr_site_n_stage_c.py"
        ).resolve(),
        "base_model": (ROOT / "src/nucpred/training/mayr_site_n.py").resolve(),
        "dataset_manifest": dataset / "dataset_manifest.json",
        "split_manifest": dataset / "split_manifest.json",
        "frozen_c2_checkpoint": frozen.checkpoint_path,
    }
    contract: dict[str, object] = {
        "schema_version": "nucpred.mayr-nextgen-stage-e-b-r2-job-contract.v1",
        "campaign_id": config["campaign_id"],
        "split_seed": split_seed,
        "initialization_seed": initialization_seed,
        "arm": arm,
        "source_hashes": {key: sha256_file(value) for key, value in sources.items()},
        "frozen_c2_internal_state_sha256": frozen.model_state_sha256,
        "split_audit": dict(split_audit),
        "selection_metric": "rmse",
        "selection_roles": ["train", "validation"],
        "epoch_zero_frozen_c2_is_eligible": True,
        "test_examples_instantiated": 0,
        "test_predictions_computed": 0,
        "formal_calibration_performed": False,
        "final_refit_performed": False,
        "d_n4_combination_performed": False,
        "dft_or_cdft_computation_performed": False,
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
    frozen: FrozenC2,
    contract: Mapping[str, object],
) -> None:
    state = {
        name: tensor.detach().cpu()
        for name, tensor in outcome.model.state_dict().items()
    }
    payload = {
        "schema_version": "nucpred.mayr-nextgen-stage-e-b-r2-checkpoint.v1",
        "phase": "development_validation_selection",
        "arm": arm,
        "split_seed": split_seed,
        "initialization_seed": initialization_seed,
        "selection_best_epoch": outcome.best_epoch,
        "model_architecture": outcome.model.architecture,
        "model_state_dict": state,
        "model_state_sha256": _tensor_mapping_sha256(state),
        "frozen_c2_checkpoint_sha256": frozen.checkpoint_sha256,
        "frozen_c2_internal_state_sha256": frozen.model_state_sha256,
        "preprocessor": frozen.preprocessor.to_json(),
        "solvent_vocabulary": list(frozen.vocabulary.tokens),
        "contract": dict(contract),
        "freeze_audit": dict(outcome.freeze_audit),
    }
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


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
        raise SiteNCampaignError("Unregistered Stage-E-B job axis")
    selected_device = torch.device(device)
    if selected_device.type != "cuda" or not torch.cuda.is_available():
        raise SiteNCampaignError("Stage-E-B R2 matrix requires CUDA")
    config, config_file = read_config(config_path)
    dataset = _project_path(config["dataset"]["directory"], label="dataset.directory")
    dataset_verification = verify_dataset(dataset)
    train, validation, split_audit = _split_examples(dataset, split_seed=split_seed)
    frozen = _load_frozen_c2(
        config=config,
        train=train,
        split_seed=split_seed,
        initialization_seed=initialization_seed,
        device=selected_device,
    )
    contract = _job_contract(
        config=config,
        config_path=config_file,
        dataset=dataset,
        frozen=frozen,
        split_seed=split_seed,
        initialization_seed=initialization_seed,
        arm=arm,
        split_audit=split_audit,
    )
    target = (
        _project_path(config["r2"]["output_directory"], label="r2.output_directory")
        / f"split-{split_seed}"
        / f"init-{initialization_seed}"
        / arm
    )
    if (target / "summary.json").is_file():
        existing = _load_json(target / "summary.json")
        if existing.get("status") == "pass" and existing.get("contract") == contract:
            return existing
        raise SiteNCampaignError(f"Existing Stage-E-B job is stale: {target}")
    if target.exists():
        raise SiteNCampaignError(f"Partial Stage-E-B job exists: {target}")
    output_root = target.parents[2]
    try:
        outcome = _fit_selection(
            train,
            validation,
            arm=arm,
            config=config,
            frozen=frozen,
            initialization_seed=initialization_seed,
            device=selected_device,
        )
        connectivity = {
            target_id: example.connectivity_id
            for example in validation
            for target_id in example.target_ids
        }
        predictions = outcome.validation_predictions.copy()
        predictions["connectivity_id"] = predictions["target_id"].map(connectivity)
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{arm}.staging-", dir=target.parent))
        try:
            outcome.curves.to_csv(staging / "selection_loss_curves.csv", index=False)
            parquet_assets = {
                "validation_predictions.parquet": predictions,
                "validation_components.parquet": outcome.validation_components,
                "validation_paired_solvent_predictions.parquet": (
                    outcome.validation_pair_predictions
                ),
                "train_center_group_residuals.parquet": (
                    outcome.train_center_residuals
                ),
            }
            for filename, frame in parquet_assets.items():
                frame.to_parquet(
                    staging / filename,
                    index=False,
                    engine="pyarrow",
                    compression="zstd",
                )
            json_assets = {
                "validation_metrics.json": outcome.validation_metrics,
                "frozen_c2_validation_metrics.json": outcome.frozen_parent_metrics,
                "validation_paired_solvent_metrics.json": (
                    outcome.validation_pair_metrics
                ),
                "frozen_c2_verification.json": frozen.verification,
                "freeze_audit.json": outcome.freeze_audit,
                "target_weight_audit.json": outcome.target_weight_audit,
                "paired_solvent_audit.json": outcome.paired_solvent_audit,
                "center_group_audit.json": outcome.center_group_audit,
                "gate_audit.json": outcome.gate_audit,
            }
            for filename, payload in json_assets.items():
                atomic_write_json(staging / filename, payload, ensure_ascii=False)
            _save_checkpoint(
                staging / "selection_checkpoint.pt",
                outcome=outcome,
                arm=arm,
                split_seed=split_seed,
                initialization_seed=initialization_seed,
                frozen=frozen,
                contract=contract,
            )
            summary: dict[str, object] = {
                "schema_version": "nucpred.mayr-nextgen-stage-e-b-r2-job.v1",
                "status": "pass",
                "campaign_id": config["campaign_id"],
                "split_seed": split_seed,
                "initialization_seed": initialization_seed,
                "arm": arm,
                "contract": contract,
                "dataset_verification": dataset_verification,
                "split_audit": split_audit,
                "selection_metric": "rmse",
                "selection_best_epoch": outcome.best_epoch,
                "selection_best_validation_rmse": outcome.best_validation_rmse,
                "epoch_zero_frozen_c2_was_selected": outcome.best_epoch == 0,
                "validation_metrics": dict(outcome.validation_metrics),
                "frozen_c2_validation_metrics": dict(outcome.frozen_parent_metrics),
                "freeze_audit": dict(outcome.freeze_audit),
                "gate_audit": dict(outcome.gate_audit),
                "test_examples_instantiated": 0,
                "test_predictions_computed": 0,
                "test_prediction_files_written": 0,
                "formal_calibration_performed": False,
                "absolute_probability_published": False,
                "final_refit_performed": False,
                "d_n4_combination_performed": False,
                "dft_or_cdft_computation_performed": False,
                "device": str(selected_device),
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
                        f"selection_best_epoch={outcome.best_epoch}",
                        "test_examples_instantiated=0",
                        "test_predictions_computed=0",
                        "formal_calibration_performed=false",
                        "final_refit_performed=false",
                        "d_n4_combination_performed=false",
                        "dft_or_cdft_computation_performed=false",
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
                "schema_version": "nucpred.mayr-nextgen-stage-e-b-r2-failure.v1",
                "status": "failed",
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
                        "event": "stage_e_b_r2_job_complete",
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
            gc.collect()
            torch.cuda.empty_cache()


def run_all(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    device: str = "cuda:0",
) -> dict[str, object]:
    config, config_file = read_config(config_path)
    output_root = _project_path(
        config["r2"]["output_directory"], label="r2.output_directory"
    )
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
        return {"status": "pass", "parallel_workers": 0, "completed_job_count": 30}
    gc.collect()
    gc.freeze()
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(
            name=f"mayr-stage-e-b-r2-{seed}",
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
                "event": "stage_e_b_r2_matrix_started",
                "parallel_workers": len(processes),
                "job_count": len(expected),
                "pids": {process.name: process.pid for process in processes},
            },
            sort_keys=True,
        ),
        flush=True,
    )
    while any(process.is_alive() for process in processes):
        for process in processes:
            process.join(timeout=1.0)
        print(
            json.dumps(
                {
                    "event": "stage_e_b_r2_matrix_heartbeat",
                    "completed_job_count": sum(path.is_file() for path in expected),
                    "total_job_count": len(expected),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        time.sleep(20)
    failures = {
        process.name: process.exitcode for process in processes if process.exitcode != 0
    }
    if failures:
        raise SiteNCampaignError(f"Stage-E-B R2 workers failed: {failures}")
    if not all(path.is_file() for path in expected):
        raise SiteNCampaignError("Stage-E-B R2 summaries are incomplete")
    atomic_write_json(
        output_root / "summary.json",
        {
            "schema_version": "nucpred.mayr-nextgen-stage-e-b-r2-matrix.v1",
            "status": "pass",
            "campaign_id": config["campaign_id"],
            "completed_job_count": 30,
            "expected_job_count": 30,
            "parallel_worker_count": 3,
            "selection_roles": ["train", "validation"],
            "test_examples_instantiated": 0,
            "test_predictions_computed": 0,
            "formal_calibration_performed": False,
            "final_refit_performed": False,
            "d_n4_combination_performed": False,
            "dft_or_cdft_computation_performed": False,
        },
        ensure_ascii=False,
    )
    _write_manifest(output_root)
    return {"status": "pass", "parallel_workers": 3, "completed_job_count": 30}


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
        payload = run_all(config_path=args.config, device=args.device)
    else:
        if None in (args.split_seed, args.initialization_seed, args.arm):
            raise SiteNCampaignError("Single job requires all axes")
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
