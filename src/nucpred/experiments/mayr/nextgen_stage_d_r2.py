"""Run the authorized validation-only Stage-D conditional-N matrix."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import gc
from itertools import combinations
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
    pack_site_n_batch,
    seed_everything,
)
from nucpred.training.mayr_site_n_stage_c import (
    MayrSiteNInteractionModel,
    stage_c_target_weights,
    zero_interaction_output_is_exact,
)
from nucpred.training.mayr_site_n_stage_d import (
    MayrSiteNTypeResidualModel,
    PairedSolventDefinition,
    pair_aware_example_groups,
    stage_d_paired_solvent_definitions,
    stage_d_site_n_loss,
    stage_d_two_tail_target_weights,
    zero_type_residual_output_is_exact,
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
DEFAULT_CONFIG = ROOT / "configs/mayr_nextgen_stage_d.toml"
CONFIG_SCHEMA = "nucpred.mayr-nextgen-stage-d-config.v1"
EXPECTED_STATUS = "awaiting_stage_d_results_gate"
ARMS = (
    "d_n1_two_tail_target_balancing",
    "d_n2_site_type_residual_experts",
    "d_n3_paired_solvent_delta_auxiliary",
)
D_N1 = ARMS[0]
D_N2 = ARMS[1]
D_N3 = ARMS[2]


@dataclass(frozen=True, slots=True)
class SelectionOutcome:
    model: MayrSiteNModel
    curves: pd.DataFrame
    best_epoch: int
    best_validation_rmse: float
    validation_metrics: Mapping[str, object]
    validation_predictions: pd.DataFrame
    validation_pair_predictions: pd.DataFrame
    validation_pair_metrics: Mapping[str, object]
    base_initialization_sha256: str
    post_transfer_initialization_sha256: str
    transfer_audit: Mapping[str, object]
    weight_audit: Mapping[str, object]
    paired_solvent_audit: Mapping[str, object]


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SiteNCampaignError(f"Expected JSON object: {path}")
    return value


def _project_path(value: object, *, label: str) -> Path:
    path = (ROOT / str(value)).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as exc:
        raise SiteNCampaignError(f"{label} escapes the project root") from exc
    return path


def read_config(
    path: str | Path = DEFAULT_CONFIG,
) -> tuple[dict[str, Any], Path, dict[str, Any], Path, dict[str, Any], Path]:
    """Read the authorized config and fail closed before loading model data."""

    config_path = Path(path).resolve()
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise SiteNCampaignError("Stage-D config schema changed")
    if config.get("status_after_completion") != EXPECTED_STATUS:
        raise SiteNCampaignError("Stage-D result gate changed")
    r2 = config["r2"]
    if tuple(map(str, r2["arms"])) != ARMS:
        raise SiteNCampaignError("Stage-D R2 arms changed")
    if tuple(map(int, r2["split_seeds"])) != SPLIT_SEEDS:
        raise SiteNCampaignError("Stage-D split seeds changed")
    if tuple(map(int, r2["initialization_seeds"])) != INITIALIZATION_SEEDS:
        raise SiteNCampaignError("Stage-D initialization seeds changed")
    if (
        r2.get("selection_metric") != "rmse"
        or list(r2.get("selection_roles", [])) != ["train", "validation"]
        or r2.get("test_examples_permitted") is not False
        or r2.get("test_predictions_permitted") is not False
        or r2.get("final_refit_permitted") is not False
        or r2.get("dft_or_cdft_permitted") is not False
    ):
        raise SiteNCampaignError("Stage-D validation-only contract changed")
    if int(config.get("maximum_parallel_gpu_processes", -1)) != 3:
        raise SiteNCampaignError("Stage-D GPU ceiling changed")
    forbidden = (
        "test_labels_or_predictions_permitted",
        "formal_calibration_permitted",
        "absolute_probability_publication_permitted",
        "final_refit_permitted",
        "d_n4_combination_permitted",
        "dft_or_cdft_computation_permitted",
    )
    if any(config.get(field) is not False for field in forbidden):
        raise SiteNCampaignError("Stage-D config exceeds user authorization")

    bound_specs = (
        ("authorization", "path", "sha256"),
        ("parents", "preflight_catalog_path", "preflight_catalog_sha256"),
        ("parents", "preflight_manifest_path", "preflight_manifest_sha256"),
        ("parents", "stage_c_config_path", "stage_c_config_sha256"),
        ("parents", "stage_c_r2_manifest_path", "stage_c_r2_manifest_sha256"),
        ("parents", "stage_c_r2_summary_path", "stage_c_r2_summary_sha256"),
        (
            "parents",
            "gate_c_results_manifest_path",
            "gate_c_results_manifest_sha256",
        ),
        (
            "parents",
            "gate_c_results_summary_path",
            "gate_c_results_summary_sha256",
        ),
        ("dataset", "manifest_path", "manifest_sha256"),
        ("dataset", "split_manifest_path", "split_manifest_sha256"),
        ("dataset", "formal_config_path", "formal_config_sha256"),
        ("dataset", "base_config_path", "base_config_sha256"),
        ("e3", "reference_inventory_path", "reference_inventory_sha256"),
        ("e3", "candidate_census_path", "candidate_census_sha256"),
        ("e3", "pending_queue_path", "pending_queue_sha256"),
    )
    for section, path_key, hash_key in bound_specs:
        _verify_bound_file(
            _project_path(config[section][path_key], label=f"{section}.{path_key}"),
            str(config[section][hash_key]),
        )

    authorization = _load_json(
        _project_path(config["authorization"]["path"], label="authorization.path")
    )
    contract = authorization.get("stage_contract")
    required_contract: dict[str, object] = {
        "stage_d_development_training_authorized": True,
        "authorized_arms": list(ARMS),
        "e3_isolated_two_pass_review_authorized": True,
        "e3_review_mode": "single_operator_two_isolated_passes",
        "claim_two_independent_reviewers": False,
        "e3_unknown_as_negative_authorized": False,
        "e3_role_reveal_authorized": False,
        "e3_model_score_access_authorized": False,
        "d_n4_combination_authorized": False,
        "test_labels_or_predictions_authorized": False,
        "formal_calibration_authorized": False,
        "absolute_probability_publication_authorized": False,
        "final_refit_authorized": False,
        "dft_or_cdft_computation_authorized": False,
        "maximum_parallel_gpu_processes": 3,
        "deployment_population": "mayr_like_molecules_not_arbitrary_molecules",
        "formal_feature_scope": "strict_no_dft_rdkit_xtb",
        "status_after_completion": EXPECTED_STATUS,
    }
    if not isinstance(contract, Mapping) or any(
        contract.get(key) != value for key, value in required_contract.items()
    ):
        raise SiteNCampaignError("Stage-D authorization artifact changed")
    preflight = ArtifactCatalog().verify(
        str(config["parents"]["preflight_run_id"])
    )
    if preflight.get("status") != "pass":
        raise SiteNCampaignError("Stage-D preflight catalog no longer verifies")

    formal_path = _project_path(
        config["dataset"]["formal_config_path"],
        label="dataset.formal_config_path",
    )
    formal = tomllib.loads(formal_path.read_text(encoding="utf-8"))
    base_path = _project_path(
        config["dataset"]["base_config_path"],
        label="dataset.base_config_path",
    )
    if (ROOT / str(formal["base_config"])).resolve() != base_path:
        raise SiteNCampaignError("Formal and bound base configs disagree")
    base = _read_site_n_config(base_path)
    return config, config_path, formal, formal_path, base, base_path


def _split_examples(
    dataset: Path,
    *,
    split_seed: int,
) -> tuple[list[SiteNExample], list[SiteNExample], dict[str, object]]:
    train = load_site_n_examples(dataset, split_seed=split_seed, role="train")
    validation = load_site_n_examples(
        dataset,
        split_seed=split_seed,
        role="validation",
    )
    membership = pd.read_csv(dataset / "split_membership.csv")
    selected = membership.loc[membership["split_seed"].eq(split_seed)].copy()
    role_connectivities = {
        role: set(
            selected.loc[selected["role"].eq(role), "connectivity_id"].astype(str)
        )
        for role in ("train", "validation", "test")
    }
    if (
        role_connectivities["train"] & role_connectivities["validation"]
        or role_connectivities["train"] & role_connectivities["test"]
        or role_connectivities["validation"] & role_connectivities["test"]
    ):
        raise SiteNCampaignError("Parent split leaks connectivity roles")
    return train, validation, {
        "schema_version": "nucpred.mayr-stage-d-validation-split-audit.v1",
        "split_seed": int(split_seed),
        "train_context_count": len(train),
        "train_target_count": sum(item.num_sites for item in train),
        "train_connectivity_count": len(role_connectivities["train"]),
        "validation_context_count": len(validation),
        "validation_target_count": sum(item.num_sites for item in validation),
        "validation_connectivity_count": len(
            role_connectivities["validation"]
        ),
        "test_membership_connectivity_count": len(role_connectivities["test"]),
        "test_examples_instantiated": 0,
        "test_predictions_computed": 0,
        "connectivity_disjoint": True,
    }


def _verify_legacy_checkpoint(
    formal: Mapping[str, Any],
    initialization_seed: int,
) -> tuple[Mapping[str, Any], Path, dict[str, object]]:
    entry = _checkpoint_entry(formal, "legacy_checkpoints", initialization_seed)
    checkpoint = _project_path(entry["path"], label="legacy_checkpoint")
    observed = sha256_file(checkpoint)
    if observed != str(entry["sha256"]):
        raise SiteNCampaignError("Frozen legacy checkpoint hash changed")
    payload = load_legacy_pretraining_checkpoint(checkpoint)
    if int(payload["init_seed"]) != int(entry["pretraining_seed"]):
        raise SiteNCampaignError("Frozen legacy checkpoint seed changed")
    return entry, checkpoint, {
        "schema_version": "nucpred.mayr-stage-d-legacy-checkpoint-audit.v1",
        "status": "pass",
        "path": _display_path(checkpoint),
        "sha256": observed,
        "pretraining_seed": int(entry["pretraining_seed"]),
        "backbone_state_sha256": payload["backbone_state_sha256"],
        "historical_source_parity_claimed": False,
        "reuse_semantics": "frozen_checkpoint_identity_and_internal_state_only",
    }


def _model_kwargs(
    base_config: Mapping[str, Any],
    vocabulary: SolventVocabulary,
) -> dict[str, object]:
    section = base_config["model"]
    return {
        "num_solvents": len(vocabulary.tokens),
        "hidden_dim": int(section["hidden_dim"]),
        "layers": int(section["message_passing_layers"]),
        "node_embedding_dim": int(section["node_embedding_dim"]),
        "edge_embedding_dim": int(section["edge_embedding_dim"]),
        "solvent_embedding_dim": int(section["solvent_embedding_dim"]),
        "dropout": float(section["dropout"]),
    }


def _make_stage_d_model(
    *,
    arm: str,
    base_config: Mapping[str, Any],
    vocabulary: SolventVocabulary,
    initialization_seed: int,
    device: torch.device,
    checkpoint: Path,
) -> tuple[MayrSiteNModel, str, str, dict[str, object]]:
    seed_everything(initialization_seed)
    kwargs = _model_kwargs(base_config, vocabulary)
    if arm == D_N1:
        model: MayrSiteNModel = MayrSiteNModel(**kwargs)
    elif arm == D_N2:
        model = MayrSiteNTypeResidualModel(**kwargs)
        if not zero_type_residual_output_is_exact(model):
            raise SiteNCampaignError("Type residual is not exact-zero")
    elif arm == D_N3:
        model = MayrSiteNInteractionModel(**kwargs)
        if not zero_interaction_output_is_exact(model):
            raise SiteNCampaignError("Interaction residual is not exact-zero")
    else:
        raise SiteNCampaignError(f"Unsupported Stage-D arm: {arm}")
    base_hash = initialization_sha256(model)
    model = model.to(device)
    transfer = _legacy_transfer(model, checkpoint)
    type_zero = (
        zero_type_residual_output_is_exact(model)
        if isinstance(model, MayrSiteNTypeResidualModel)
        else None
    )
    interaction_zero = (
        zero_interaction_output_is_exact(model)
        if isinstance(model, MayrSiteNInteractionModel)
        else None
    )
    if type_zero is False or interaction_zero is False:
        raise SiteNCampaignError("Legacy transfer changed zero residual")
    audit = {
        **transfer,
        "stage_d_arm": arm,
        "base_transfer": "frozen_legacy_encoder",
        "type_residual_present": arm == D_N2,
        "type_residual_exact_zero_after_transfer": type_zero,
        "interaction_residual_present": arm == D_N3,
        "interaction_residual_exact_zero_after_transfer": interaction_zero,
    }
    return model, base_hash, initialization_sha256(model), audit


def _arm_targets_and_pairs(
    train: Sequence[SiteNExample],
    *,
    arm: str,
    stage_config: Mapping[str, Any],
) -> tuple[
    dict[str, float],
    dict[str, object],
    tuple[PairedSolventDefinition, ...],
    dict[str, object],
]:
    optimization = stage_config["r2"]["optimization"]
    if arm == D_N1:
        weights, weight_audit = stage_d_two_tail_target_weights(
            train,
            tail_power=float(stage_config["r2"]["d_n1_tail_power"]),
            maximum_weight=float(optimization["maximum_target_weight"]),
        )
        pairs: tuple[PairedSolventDefinition, ...] = ()
        pair_audit: dict[str, object] = {
            "schema_version": "nucpred.mayr-stage-d-pair-use-audit.v1",
            "arm": arm,
            "pair_count": 0,
            "paired_solvent_auxiliary_enabled": False,
            "fit_roles": ["train"],
        }
    elif arm == D_N2:
        weights, weight_audit = stage_c_target_weights(
            train,
            use_h1=False,
            use_h2=True,
            maximum_weight=float(optimization["maximum_target_weight"]),
        )
        pairs = ()
        pair_audit = {
            "schema_version": "nucpred.mayr-stage-d-pair-use-audit.v1",
            "arm": arm,
            "pair_count": 0,
            "paired_solvent_auxiliary_enabled": False,
            "fit_roles": ["train"],
        }
    elif arm == D_N3:
        target_ids = [
            str(target_id)
            for example in train
            for target_id in example.target_ids
        ]
        if len(target_ids) != len(set(target_ids)):
            raise SiteNCampaignError("Stage-D train target ids are not unique")
        weights = dict.fromkeys(target_ids, 1.0)
        weight_audit = {
            "schema_version": "nucpred.mayr-stage-d-uniform-weights.v1",
            "method": "uniform",
            "target_count": len(target_ids),
            "minimum_weight": 1.0,
            "mean_weight": 1.0,
            "maximum_weight": 1.0,
            "fit_roles": ["train"],
            "validation_or_test_target_used_for_fit": False,
            "applies_to": "training_mse_only",
        }
        pairs, pair_audit = stage_d_paired_solvent_definitions(train)
        pair_audit = {
            **pair_audit,
            "arm": arm,
            "paired_solvent_auxiliary_enabled": True,
        }
    else:
        raise SiteNCampaignError(f"Unsupported Stage-D arm: {arm}")
    return weights, weight_audit, pairs, pair_audit


def _training_batches(
    train: Sequence[SiteNExample],
    *,
    arm: str,
    pairs: Sequence[PairedSolventDefinition],
    batch_size: int,
    preprocessor: SiteNFoldPreprocessor,
    vocabulary: SolventVocabulary,
    shuffle_seed: int,
) -> Iterator:
    if arm != D_N3:
        yield from _iter_batches(
            train,
            batch_size=batch_size,
            preprocessor=preprocessor,
            vocabulary=vocabulary,
            shuffle_seed=shuffle_seed,
        )
        return
    for group in pair_aware_example_groups(
        train,
        pairs,
        batch_size_contexts=batch_size,
        shuffle_seed=shuffle_seed,
    ):
        yield pack_site_n_batch(
            group,
            preprocessor=preprocessor,
            solvent_vocabulary=vocabulary,
        )


def _train_epoch(
    model: MayrSiteNModel,
    batches: Iterator,
    *,
    target_weights: Mapping[str, float],
    paired_solvent_pairs: Sequence[PairedSolventDefinition],
    paired_solvent_weight: float,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    ranking_weight: float,
    gradient_clip_norm: float,
) -> dict[str, float]:
    model.train()
    totals = {
        "total": 0.0,
        "regression": 0.0,
        "ranking": 0.0,
        "paired_solvent_delta": 0.0,
    }
    target_count = 0
    ranking_pairs = 0
    solvent_pairs = 0
    for raw_batch in batches:
        batch = raw_batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(batch.inputs)
        total, parts = stage_d_site_n_loss(
            output,
            batch,
            target_weights,
            ranking_weight=ranking_weight,
            paired_solvent_pairs=paired_solvent_pairs,
            paired_solvent_weight=paired_solvent_weight,
        )
        total.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            float(gradient_clip_norm),
        )
        optimizer.step()
        count = batch.inputs.num_sites
        target_count += count
        for name in totals:
            value = total if name == "total" else parts[name]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Unexpected loss component: {name}")
            totals[name] += float(value.detach().cpu()) * count
        ranking_pairs += int(parts["ranking_pairs"])
        solvent_pairs += int(parts["paired_solvent_pairs"])
    if target_count == 0:
        raise SiteNCampaignError("Stage-D epoch had no training targets")
    result = {name: value / target_count for name, value in totals.items()}
    result["ranking_pairs"] = float(ranking_pairs)
    result["paired_solvent_pairs"] = float(solvent_pairs)
    return result


def _validation_pair_predictions(
    validation: Sequence[SiteNExample],
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    prediction_index = predictions.set_index("target_id", drop=False)
    grouped: dict[
        tuple[str, str, str],
        list[tuple[str, str, str]],
    ] = defaultdict(list)
    for example in validation:
        for target_id, site_object_id, site_type in zip(
            example.target_ids,
            example.site_object_ids,
            example.site_types,
            strict=True,
        ):
            grouped[
                (
                    str(example.connectivity_id),
                    str(site_object_id),
                    str(site_type),
                )
            ].append(
                (
                    str(target_id),
                    str(example.context_id),
                    str(example.solvent_raw),
                )
            )
    rows: list[dict[str, object]] = []
    for (connectivity, site_object, site_type), records in sorted(
        grouped.items()
    ):
        for left, right in combinations(sorted(records), 2):
            if left[1] == right[1] or left[2] == right[2]:
                continue
            left_row = prediction_index.loc[left[0]]
            right_row = prediction_index.loc[right[0]]
            true_delta = float(left_row["N_true"]) - float(right_row["N_true"])
            predicted_delta = float(left_row["N_pred"]) - float(
                right_row["N_pred"]
            )
            rows.append(
                {
                    "pair_id": (
                        f"{connectivity}|{site_object}|{left[0]}|{right[0]}"
                    ),
                    "connectivity_id": connectivity,
                    "site_object_id": site_object,
                    "site_type": site_type,
                    "left_target_id": left[0],
                    "right_target_id": right[0],
                    "left_context_id": left[1],
                    "right_context_id": right[1],
                    "left_solvent": left[2],
                    "right_solvent": right[2],
                    "delta_N_true": true_delta,
                    "delta_N_pred": predicted_delta,
                    "delta_N_error": predicted_delta - true_delta,
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        metrics = {
            "pair_count": 0,
            "connectivity_count": 0,
            "mae": None,
            "rmse": None,
            "evaluation_role": "validation",
        }
    else:
        error = frame["delta_N_error"].to_numpy(dtype=float)
        metrics = {
            "pair_count": len(frame),
            "connectivity_count": int(frame["connectivity_id"].nunique()),
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error**2))),
            "evaluation_role": "validation",
        }
    return frame, metrics


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
    model, base_hash, post_hash, transfer_audit = _make_stage_d_model(
        arm=arm,
        base_config=base_config,
        vocabulary=vocabulary,
        initialization_seed=initialization_seed,
        device=device,
        checkpoint=checkpoint,
    )
    target_weights, weight_audit, pairs, pair_audit = _arm_targets_and_pairs(
        train,
        arm=arm,
        stage_config=stage_config,
    )
    optimization = stage_config["r2"]["optimization"]
    paired_weight = (
        float(stage_config["r2"]["d_n3_paired_solvent_weight"])
        if arm == D_N3
        else 0.0
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
            _training_batches(
                train,
                arm=arm,
                pairs=pairs,
                batch_size=int(optimization["batch_size_contexts"]),
                preprocessor=preprocessor,
                vocabulary=vocabulary,
                shuffle_seed=initialization_seed + epoch,
            ),
            target_weights=target_weights,
            paired_solvent_pairs=pairs,
            paired_solvent_weight=paired_weight,
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
        raise SiteNCampaignError("Stage-D selection produced no best state")
    model.load_state_dict(best_state, strict=True)
    metrics, predictions = _evaluate(
        model,
        validation,
        preprocessor=preprocessor,
        vocabulary=vocabulary,
        batch_size=int(optimization["batch_size_contexts"]),
        device=device,
    )
    if not math.isclose(
        float(metrics["rmse"]),
        best_rmse,
        rel_tol=0.0,
        abs_tol=1e-7,
    ):
        raise SiteNCampaignError("Restored Stage-D checkpoint changed")
    pair_predictions, pair_metrics = _validation_pair_predictions(
        validation,
        predictions,
    )
    return SelectionOutcome(
        model=model,
        curves=pd.DataFrame(rows),
        best_epoch=best_epoch,
        best_validation_rmse=best_rmse,
        validation_metrics=metrics,
        validation_predictions=predictions,
        validation_pair_predictions=pair_predictions,
        validation_pair_metrics=pair_metrics,
        base_initialization_sha256=base_hash,
        post_transfer_initialization_sha256=post_hash,
        transfer_audit=transfer_audit,
        weight_audit=weight_audit,
        paired_solvent_audit=pair_audit,
    )


def _job_contract(
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
        "stage_d_config": config_path,
        "authorization": _project_path(
            config["authorization"]["path"],
            label="authorization.path",
        ),
        "preflight_catalog": _project_path(
            config["parents"]["preflight_catalog_path"],
            label="parents.preflight_catalog_path",
        ),
        "formal_config": formal_path,
        "base_config": base_path,
        "runner": Path(__file__).resolve(),
        "stage_d_model": (
            ROOT / "src/nucpred/training/mayr_site_n_stage_d.py"
        ).resolve(),
        "stage_c_model": (
            ROOT / "src/nucpred/training/mayr_site_n_stage_c.py"
        ).resolve(),
        "base_model": (ROOT / "src/nucpred/training/mayr_site_n.py").resolve(),
        "base_runner": (
            ROOT / "src/nucpred/experiments/mayr/site_n.py"
        ).resolve(),
        "formal_runner": (
            ROOT / "src/nucpred/experiments/mayr/site_n_formal.py"
        ).resolve(),
        "dataset_loader": (
            ROOT / "src/nucpred/datasets/mayr_site_n.py"
        ).resolve(),
        "dataset_manifest": dataset / "dataset_manifest.json",
        "split_manifest": dataset / "split_manifest.json",
        "legacy_checkpoint": checkpoint,
    }
    contract: dict[str, object] = {
        "schema_version": "nucpred.mayr-nextgen-stage-d-r2-job-contract.v1",
        "campaign_id": str(config["campaign_id"]),
        "split_seed": int(split_seed),
        "initialization_seed": int(initialization_seed),
        "arm": arm,
        "source_hashes": {
            name: sha256_file(path) for name, path in sources.items()
        },
        "split_audit": dict(split_audit),
        "selection_metric": "rmse",
        "selection_roles": ["train", "validation"],
        "test_examples_instantiated": 0,
        "test_predictions_computed": 0,
        "final_refit_performed": False,
        "d_n4_combination_performed": False,
        "formal_calibration_performed": False,
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
        "schema_version": "nucpred.mayr-nextgen-stage-d-r2-checkpoint.v1",
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
        "paired_solvent_audit": dict(outcome.paired_solvent_audit),
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
        raise SiteNCampaignError("Unregistered Stage-D R2 job axis")
    selected_device = torch.device(device)
    if selected_device.type != "cuda" or not torch.cuda.is_available():
        raise SiteNCampaignError("Stage-D R2 matrix requires CUDA")
    config, config_file, formal, formal_path, base, base_path = read_config(
        config_path
    )
    dataset = _project_path(config["dataset"]["directory"], label="dataset.directory")
    dataset_verification = verify_dataset(dataset)
    train, validation, split_audit = _split_examples(
        dataset,
        split_seed=split_seed,
    )
    entry, checkpoint, checkpoint_verification = _verify_legacy_checkpoint(
        formal,
        initialization_seed,
    )
    contract = _job_contract(
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
        _project_path(config["r2"]["output_directory"], label="r2.output_directory")
        / f"split-{split_seed}"
        / f"init-{initialization_seed}"
        / arm
    )
    summary_path = target / "summary.json"
    if summary_path.is_file():
        existing = _load_json(summary_path)
        if existing.get("status") == "pass" and existing.get("contract") == contract:
            return existing
        raise SiteNCampaignError(f"Existing Stage-D job is stale: {target}")
    if target.exists():
        raise SiteNCampaignError(f"Partial Stage-D job exists: {target}")
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
        staging = Path(
            tempfile.mkdtemp(prefix=f".{arm}.staging-", dir=target.parent)
        )
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
            outcome.validation_pair_predictions.to_parquet(
                staging / "validation_paired_solvent_predictions.parquet",
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
                staging / "validation_paired_solvent_metrics.json",
                outcome.validation_pair_metrics,
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
            atomic_write_json(
                staging / "paired_solvent_audit.json",
                outcome.paired_solvent_audit,
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
                "schema_version": "nucpred.mayr-nextgen-stage-d-r2-job.v1",
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
                "selection_best_validation_rmse": (
                    outcome.best_validation_rmse
                ),
                "validation_metrics": dict(outcome.validation_metrics),
                "validation_paired_solvent_metrics": dict(
                    outcome.validation_pair_metrics
                ),
                "base_initialization_sha256": (
                    outcome.base_initialization_sha256
                ),
                "post_transfer_initialization_sha256": (
                    outcome.post_transfer_initialization_sha256
                ),
                "transfer_audit": dict(outcome.transfer_audit),
                "weight_audit": dict(outcome.weight_audit),
                "paired_solvent_audit": dict(outcome.paired_solvent_audit),
                "test_examples_instantiated": 0,
                "test_predictions_computed": 0,
                "test_prediction_files_written": 0,
                "final_refit_performed": False,
                "formal_calibration_performed": False,
                "d_n4_combination_performed": False,
                "formal_feature_scope": "strict_no_dft_rdkit_xtb",
                "dft_or_cdft_computation_performed": False,
                "device": str(selected_device),
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "wall_seconds": time.perf_counter() - started,
            }
            atomic_write_json(
                staging / "summary.json",
                summary,
                ensure_ascii=False,
            )
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
                        "formal_calibration_performed=false",
                        "final_refit_performed=false",
                        "d_n4_combination_performed=false",
                        "dft_or_cdft_computation_performed=false",
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
                "schema_version": "nucpred.mayr-nextgen-stage-d-r2-failure.v1",
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
                        "event": "stage_d_r2_job_complete",
                        "split_seed": split_seed,
                        "initialization_seed": initialization_seed,
                        "arm": arm,
                        "best_epoch": summary["selection_best_epoch"],
                        "validation_rmse": (
                            summary["selection_best_validation_rmse"]
                        ),
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
    if not str(device).startswith("cuda"):
        raise SiteNCampaignError("Stage-D R2 matrix requires CUDA")
    config, config_file, *_ = read_config(config_path)
    output_root = _project_path(
        config["r2"]["output_directory"],
        label="r2.output_directory",
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
        return {
            "schema_version": "nucpred.mayr-nextgen-stage-d-r2-coordinator.v1",
            "status": "pass",
            "parallel_workers": 0,
            "completed_job_count": len(expected),
        }
    gc.collect()
    gc.freeze()
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(
            name=f"mayr-stage-d-r2-{seed}",
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
                "event": "stage_d_r2_matrix_started",
                "parallel_workers": len(processes),
                "job_count": len(expected),
                "pids": {process.name: process.pid for process in processes},
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
                        "event": "stage_d_r2_matrix_heartbeat",
                        "completed_job_count": sum(
                            path.is_file() for path in expected
                        ),
                        "total_job_count": len(expected),
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
            last_heartbeat = time.monotonic()
    for process in processes:
        process.join()
    failures = {
        process.name: process.exitcode
        for process in processes
        if process.exitcode != 0
    }
    if failures:
        raise SiteNCampaignError(f"Stage-D R2 workers failed: {failures}")
    missing = [_display_path(path) for path in expected if not path.is_file()]
    if missing:
        raise SiteNCampaignError(f"Stage-D R2 summaries are missing: {missing}")
    atomic_write_json(
        output_root / "summary.json",
        {
            "schema_version": "nucpred.mayr-nextgen-stage-d-r2-matrix.v1",
            "status": "pass",
            "campaign_id": config["campaign_id"],
            "completed_job_count": len(expected),
            "expected_job_count": len(expected),
            "parallel_worker_count": len(processes),
            "selection_roles": ["train", "validation"],
            "test_examples_instantiated": 0,
            "test_predictions_computed": 0,
            "test_prediction_files_written": 0,
            "formal_calibration_performed": False,
            "final_refit_performed": False,
            "d_n4_combination_performed": False,
            "dft_or_cdft_computation_performed": False,
        },
        ensure_ascii=False,
    )
    _write_manifest(output_root)
    return {
        "schema_version": "nucpred.mayr-nextgen-stage-d-r2-coordinator.v1",
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
    parser.add_argument(
        "--initialization-seed",
        type=int,
        choices=INITIALIZATION_SEEDS,
    )
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
