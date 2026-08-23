"""Run the authorized Stage-E-A protected-residual development matrix."""

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
from nucpred.training.mayr_node_xtb_scratch import SolventVocabulary
from nucpred.training.mayr_site_n import (
    MayrSiteNModel,
    SiteNExample,
    SiteNFoldPreprocessor,
    fit_site_n_preprocessor,
    pack_site_n_batch,
    seed_everything,
)
from nucpred.training.mayr_site_n_stage_c import stage_c_target_weights
from nucpred.training.mayr_site_n_stage_d import (
    PairedSolventDefinition,
    pair_aware_example_groups,
    stage_d_paired_solvent_definitions,
)
from nucpred.training.mayr_site_n_stage_e_a import (
    MayrSiteNProtectedSolventResidualModel,
    SolventCenterGroup,
    frozen_base_parameters_are_frozen,
    stage_e_a_site_n_loss,
    stage_e_a_solvent_center_groups,
    trainable_parameter_count,
    zero_residual_output_is_exact,
)

from .nextgen_gate_a import _canonical_sha256, _verify_bound_file
from .nextgen_stage_d_r2 import (
    _split_examples,
    _validation_pair_predictions,
)
from .site_n import (
    SiteNCampaignError,
    _display_path,
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
DEFAULT_CONFIG = ROOT / "configs/mayr_nextgen_stage_e_a.toml"
CONFIG_SCHEMA = "nucpred.mayr-nextgen-stage-e-a-config.v1"
EXPECTED_STATUS = "awaiting_stage_e_a_results_gate"
ARMS = (
    "e_n1_centered_solvent_residual",
    "e_n2_charge_type_gated_centered_residual",
)
E_N1 = ARMS[0]
E_N2 = ARMS[1]


@dataclass(frozen=True, slots=True)
class FrozenC2:
    model: MayrSiteNModel
    preprocessor: SiteNFoldPreprocessor
    vocabulary: SolventVocabulary
    checkpoint_path: Path
    checkpoint_sha256: str
    model_state_sha256: str
    payload: Mapping[str, Any]
    verification: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class SelectionOutcome:
    model: MayrSiteNProtectedSolventResidualModel
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


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SiteNCampaignError(f"Expected JSON object: {path}")
    return payload


def _project_path(value: object, *, label: str) -> Path:
    path = (ROOT / str(value)).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as exc:
        raise SiteNCampaignError(f"{label} escapes the project root") from exc
    return path


def read_config(
    path: str | Path = DEFAULT_CONFIG,
) -> tuple[dict[str, Any], Path]:
    """Read Stage-E-A authority and reject any broadened scope."""

    config_path = Path(path).resolve()
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise SiteNCampaignError("Stage-E-A config schema changed")
    if config.get("status_after_completion") != EXPECTED_STATUS:
        raise SiteNCampaignError("Stage-E-A result gate changed")
    r2 = config["r2"]
    if tuple(map(str, r2["arms"])) != ARMS:
        raise SiteNCampaignError("Stage-E-A R2 arms changed")
    if tuple(map(int, r2["split_seeds"])) != SPLIT_SEEDS:
        raise SiteNCampaignError("Stage-E-A split seeds changed")
    if tuple(map(int, r2["initialization_seeds"])) != INITIALIZATION_SEEDS:
        raise SiteNCampaignError("Stage-E-A initialization seeds changed")
    if (
        r2.get("selection_metric") != "rmse"
        or list(r2.get("selection_roles", [])) != ["train", "validation"]
        or r2.get("epoch_zero_frozen_c2_is_eligible") is not True
        or r2.get("test_examples_permitted") is not False
        or r2.get("test_predictions_permitted") is not False
        or r2.get("final_refit_permitted") is not False
        or r2.get("dft_or_cdft_permitted") is not False
    ):
        raise SiteNCampaignError("Stage-E-A development-only contract changed")
    if int(config.get("maximum_parallel_gpu_processes", -1)) != 3:
        raise SiteNCampaignError("Stage-E-A GPU ceiling changed")
    forbidden = (
        "test_labels_examples_or_predictions_permitted",
        "formal_calibration_permitted",
        "absolute_probability_publication_permitted",
        "final_refit_permitted",
        "d_n4_combination_permitted",
        "dft_or_cdft_computation_permitted",
        "automatic_continuation_permitted",
    )
    if any(config.get(field) is not False for field in forbidden):
        raise SiteNCampaignError("Stage-E-A config exceeds user authority")

    bound_specs = (
        ("authorization", "path", "sha256"),
        ("parents", "stage_d_catalog_path", "stage_d_catalog_sha256"),
        ("parents", "stage_d_config_path", "stage_d_config_sha256"),
        (
            "parents",
            "stage_d_results_manifest_path",
            "stage_d_results_manifest_sha256",
        ),
        (
            "parents",
            "stage_d_results_summary_path",
            "stage_d_results_summary_sha256",
        ),
        ("parents", "stage_c_config_path", "stage_c_config_sha256"),
        (
            "parents",
            "stage_c_r2_manifest_path",
            "stage_c_r2_manifest_sha256",
        ),
        (
            "parents",
            "stage_c_r2_summary_path",
            "stage_c_r2_summary_sha256",
        ),
        ("dataset", "manifest_path", "manifest_sha256"),
        ("dataset", "split_manifest_path", "split_manifest_sha256"),
        (
            "evidence",
            "stage_b_projection_path",
            "stage_b_projection_sha256",
        ),
        (
            "evidence",
            "stage_b_manifest_path",
            "stage_b_manifest_sha256",
        ),
        (
            "evidence",
            "stage_d_e3_pass_a_manifest_path",
            "stage_d_e3_pass_a_manifest_sha256",
        ),
        (
            "evidence",
            "stage_d_e3_pass_b_manifest_path",
            "stage_d_e3_pass_b_manifest_sha256",
        ),
        (
            "evidence",
            "probability_protocol_path",
            "probability_protocol_sha256",
        ),
    )
    for section, path_key, hash_key in bound_specs:
        _verify_bound_file(
            _project_path(
                config[section][path_key],
                label=f"{section}.{path_key}",
            ),
            str(config[section][hash_key]),
        )

    decision = _load_json(
        _project_path(
            config["authorization"]["path"],
            label="authorization.path",
        )
    )
    contract = decision.get("stage_contract")
    required: dict[str, object] = {
        "stage_e_a_development_training_authorized": True,
        "authorized_arms": list(ARMS),
        "frozen_c2_parent_required": True,
        "positive_evidence_expansion_authorized": True,
        "evidence_types": ["atom_group", "delocalized_region"],
        "evidence_review_mode": "single_operator_two_isolated_passes",
        "claim_two_independent_reviewers": False,
        "evidence_must_be_frozen_before_split_role_reveal": True,
        "model_scores_visible_during_evidence_review": False,
        "unknown_as_negative_authorized": False,
        "automatic_continuation_authorized": False,
        "d_n4_combination_authorized": False,
        "test_labels_examples_or_predictions_authorized": False,
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
        contract.get(key) != value for key, value in required.items()
    ):
        raise SiteNCampaignError("Stage-E-A authorization artifact changed")
    minima = contract.get("minimum_new_positive_connectivities")
    if minima != {"atom_group": 27, "delocalized_region": 16}:
        raise SiteNCampaignError("Stage-E-A evidence minima changed")
    parent = ArtifactCatalog().verify(str(config["parents"]["stage_d_run_id"]))
    if parent.get("status") != "pass":
        raise SiteNCampaignError("Stage-D parent catalog no longer verifies")
    return config, config_path


def _checkpoint_relative_path(
    *,
    split_seed: int,
    initialization_seed: int,
    arm: str,
) -> str:
    return (
        f"split-{split_seed}/init-{initialization_seed}/{arm}/selection_checkpoint.pt"
    )


def _load_frozen_c2(
    *,
    config: Mapping[str, Any],
    train: Sequence[SiteNExample],
    split_seed: int,
    initialization_seed: int,
    device: torch.device,
) -> FrozenC2:
    """Load one manifest-bound C2 checkpoint without fitting any parent state."""

    r2 = config["r2"]
    relative = _checkpoint_relative_path(
        split_seed=split_seed,
        initialization_seed=initialization_seed,
        arm=str(r2["frozen_c2_arm"]),
    )
    checkpoint_root = _project_path(
        r2["frozen_c2_checkpoint_root"],
        label="r2.frozen_c2_checkpoint_root",
    )
    checkpoint = checkpoint_root / relative
    manifest = _load_json(
        _project_path(
            config["parents"]["stage_c_r2_manifest_path"],
            label="parents.stage_c_r2_manifest_path",
        )
    )
    entry = manifest.get("files", {}).get(relative)
    if not isinstance(entry, Mapping):
        raise SiteNCampaignError("C2 checkpoint missing from frozen manifest")
    observed_checkpoint_hash = sha256_file(checkpoint)
    if observed_checkpoint_hash != str(entry.get("sha256")):
        raise SiteNCampaignError("Frozen C2 checkpoint hash changed")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise SiteNCampaignError("Frozen C2 checkpoint payload changed")
    if (
        payload.get("schema_version") != r2["frozen_c2_checkpoint_schema"]
        or payload.get("phase") != "validation_selection"
        or payload.get("arm") != r2["frozen_c2_arm"]
        or int(payload.get("split_seed", -1)) != int(split_seed)
        or int(payload.get("initialization_seed", -1)) != int(initialization_seed)
    ):
        raise SiteNCampaignError("Frozen C2 checkpoint identity changed")
    architecture = payload.get("model_architecture")
    state = payload.get("model_state_dict")
    if not isinstance(architecture, Mapping) or not isinstance(state, Mapping):
        raise SiteNCampaignError("Frozen C2 model payload is incomplete")
    observed_state_hash = _tensor_mapping_sha256(state)
    if observed_state_hash != str(payload.get("model_state_sha256")):
        raise SiteNCampaignError("Frozen C2 internal state hash changed")
    if (
        architecture.get("schema_version") != "nucpred.mayr-site-n-model.v1"
        or architecture.get("site_probability_normalization") is not False
    ):
        raise SiteNCampaignError("Frozen C2 architecture changed")
    model = MayrSiteNModel(
        num_solvents=int(architecture["num_solvents"]),
        hidden_dim=int(architecture["hidden_dim"]),
        layers=int(architecture["layers"]),
        node_embedding_dim=int(architecture["node_embedding_dim"]),
        edge_embedding_dim=int(architecture["edge_embedding_dim"]),
        solvent_embedding_dim=int(architecture["solvent_embedding_dim"]),
        dropout=float(architecture["dropout"]),
    )
    model.load_state_dict(state, strict=True)
    if _tensor_mapping_sha256(model.state_dict()) != observed_state_hash:
        raise SiteNCampaignError("Exact C2 state load failed")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()

    preprocessor_payload = payload.get("preprocessor")
    vocabulary_payload = payload.get("solvent_vocabulary")
    if not isinstance(preprocessor_payload, Mapping) or not isinstance(
        vocabulary_payload, Sequence
    ):
        raise SiteNCampaignError("Frozen C2 preprocessing payload changed")
    preprocessor = SiteNFoldPreprocessor.from_json(preprocessor_payload)
    vocabulary = SolventVocabulary(tuple(str(value) for value in vocabulary_payload))
    independently_fitted = fit_site_n_preprocessor(train).to_json()
    if _canonical_sha256(independently_fitted) != _canonical_sha256(
        preprocessor.to_json()
    ):
        raise SiteNCampaignError("C2 preprocessor does not match train role")
    expected_vocabulary = SolventVocabulary.from_values(
        [example.solvent_raw for example in train]
    )
    if vocabulary != expected_vocabulary:
        raise SiteNCampaignError("C2 solvent vocabulary does not match train")
    if int(architecture["num_solvents"]) != len(vocabulary.tokens):
        raise SiteNCampaignError("C2 vocabulary width changed")
    verification = {
        "schema_version": "nucpred.mayr-stage-e-a-c2-verification.v1",
        "status": "pass",
        "path": _display_path(checkpoint),
        "checkpoint_sha256": observed_checkpoint_hash,
        "manifest_path": _display_path(
            _project_path(
                config["parents"]["stage_c_r2_manifest_path"],
                label="parents.stage_c_r2_manifest_path",
            )
        ),
        "manifest_sha256": config["parents"]["stage_c_r2_manifest_sha256"],
        "manifest_entry_verified": True,
        "model_state_sha256": observed_state_hash,
        "exact_state_load": True,
        "preprocessor_reused_exactly": True,
        "preprocessor_matches_train_role": True,
        "solvent_vocabulary_reused_exactly": True,
        "test_state_or_examples_read": False,
    }
    return FrozenC2(
        model=model.to(device),
        preprocessor=preprocessor,
        vocabulary=vocabulary,
        checkpoint_path=checkpoint,
        checkpoint_sha256=observed_checkpoint_hash,
        model_state_sha256=observed_state_hash,
        payload=payload,
        verification=verification,
    )


def _make_model(
    *,
    frozen: FrozenC2,
    arm: str,
    config: Mapping[str, Any],
    initialization_seed: int,
    device: torch.device,
) -> MayrSiteNProtectedSolventResidualModel:
    seed_everything(initialization_seed)
    model = MayrSiteNProtectedSolventResidualModel(
        frozen_base=frozen.model,
        charge_type_gate=arm == E_N2,
        initial_gate_probability=float(config["r2"]["e_n2_initial_gate_probability"]),
    )
    if not zero_residual_output_is_exact(model):
        raise SiteNCampaignError("Stage-E-A residual is not exact-zero")
    if not frozen_base_parameters_are_frozen(model):
        raise SiteNCampaignError("Stage-E-A C2 parameters remain trainable")
    if trainable_parameter_count(model) <= 0:
        raise SiteNCampaignError("Stage-E-A has no trainable residual state")
    return model.to(device)


def _training_batches(
    train: Sequence[SiteNExample],
    *,
    pairs: Sequence[PairedSolventDefinition],
    batch_size: int,
    preprocessor: SiteNFoldPreprocessor,
    vocabulary: SolventVocabulary,
    shuffle_seed: int,
) -> Iterator:
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
    model: MayrSiteNProtectedSolventResidualModel,
    batches: Iterator,
    *,
    target_weights: Mapping[str, float],
    pairs: Sequence[PairedSolventDefinition],
    center_groups: Sequence[SolventCenterGroup],
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
    target_count = 0
    ranking_pairs = 0
    solvent_pairs = 0
    centered_groups = 0
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
            paired_solvent_weight=float(config["r2"]["paired_solvent_weight"]),
            center_groups=center_groups,
            center_penalty_weight=float(config["r2"]["center_penalty_weight"]),
            residual_shrinkage_weight=float(config["r2"]["residual_shrinkage_weight"]),
            gate_shrinkage_weight=float(config["r2"]["gate_shrinkage_weight"]),
        )
        total.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            float(optimization["gradient_clip_norm"]),
        )
        optimizer.step()
        count = batch.inputs.num_sites
        target_count += count
        for name in names:
            value = total if name == "total" else parts[name]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Unexpected Stage-E-A loss part: {name}")
            totals[name] += float(value.detach().cpu()) * count
        ranking_pairs += int(parts["ranking_pairs"])
        solvent_pairs += int(parts["paired_solvent_pairs"])
        centered_groups += int(parts["center_groups"])
    if target_count == 0:
        raise SiteNCampaignError("Stage-E-A epoch had no targets")
    result = {name: value / target_count for name, value in totals.items()}
    result["ranking_pairs"] = float(ranking_pairs)
    result["paired_solvent_pairs"] = float(solvent_pairs)
    result["center_groups"] = float(centered_groups)
    return result


def _component_predictions(
    model: MayrSiteNProtectedSolventResidualModel,
    examples: Sequence[SiteNExample],
    *,
    preprocessor: SiteNFoldPreprocessor,
    vocabulary: SolventVocabulary,
    batch_size: int,
    device: torch.device,
) -> pd.DataFrame:
    metadata = {
        target_id: {
            "connectivity_id": example.connectivity_id,
            "site_type": site_type,
            "formal_charge": example.model_formal_charge,
            "solvent": example.solvent_raw,
        }
        for example in examples
        for target_id, site_type in zip(
            example.target_ids,
            example.site_types,
            strict=True,
        )
    }
    rows: list[dict[str, object]] = []
    model.eval()
    with torch.no_grad():
        for raw_batch in _iter_batches(
            examples,
            batch_size=batch_size,
            preprocessor=preprocessor,
            vocabulary=vocabulary,
            shuffle_seed=None,
        ):
            batch = raw_batch.to(device)
            output = model(batch.inputs)
            scale = float(preprocessor.target_scale)
            mean = float(preprocessor.target_mean)
            prediction = (
                output.n_prediction_standardized.detach().cpu().numpy() * scale + mean
            )
            base = (
                output.frozen_base_prediction_standardized.detach().cpu().numpy()
                * scale
                + mean
            )
            raw = output.raw_residual_standardized.detach().cpu().numpy() * scale
            applied = (
                output.applied_residual_standardized.detach().cpu().numpy() * scale
            )
            gate = output.residual_gate.detach().cpu().numpy()
            for index, target_id in enumerate(batch.target_ids):
                item = metadata[target_id]
                rows.append(
                    {
                        "target_id": target_id,
                        "connectivity_id": item["connectivity_id"],
                        "site_type": item["site_type"],
                        "solvent": item["solvent"],
                        "model_formal_charge": item["formal_charge"],
                        "N_frozen_c2": float(base[index]),
                        "N_pred": float(prediction[index]),
                        "raw_residual_N_units": float(raw[index]),
                        "applied_residual_N_units": float(applied[index]),
                        "residual_gate": float(gate[index]),
                    }
                )
    return pd.DataFrame(rows).sort_values("target_id").reset_index(drop=True)


def _center_residual_table(
    components: pd.DataFrame,
    groups: Sequence[SolventCenterGroup],
) -> pd.DataFrame:
    indexed = components.set_index("target_id", drop=False)
    rows: list[dict[str, object]] = []
    for group in groups:
        selected = indexed.loc[list(group.target_ids)]
        values = selected["applied_residual_N_units"].to_numpy(dtype=float)
        rows.append(
            {
                "group_id": group.group_id,
                "connectivity_id": group.connectivity_id,
                "site_object_id": group.site_object_id,
                "site_type": group.site_type,
                "target_count": len(group.target_ids),
                "solvent_count": len(set(group.solvents)),
                "mean_applied_residual_N_units": float(values.mean()),
                "maximum_absolute_applied_residual_N_units": float(
                    np.max(np.abs(values))
                ),
            }
        )
    return pd.DataFrame(rows)


def _gate_audit(
    *,
    arm: str,
    components: pd.DataFrame,
    model: MayrSiteNProtectedSolventResidualModel,
) -> dict[str, object]:
    by_type = {
        str(site_type): {
            "target_count": len(group),
            "minimum": float(group["residual_gate"].min()),
            "mean": float(group["residual_gate"].mean()),
            "maximum": float(group["residual_gate"].max()),
        }
        for site_type, group in components.groupby("site_type", sort=True)
    }
    parameters = None
    if model.gate_parameters is not None:
        values = model.gate_parameters.weight.detach().cpu().numpy()
        parameters = {
            site_type: {
                "intercept": float(values[index, 0]),
                "standardized_charge_slope": float(values[index, 1]),
            }
            for index, site_type in enumerate(
                model.frozen_base.architecture["site_types"]
            )
        }
    return {
        "schema_version": "nucpred.mayr-stage-e-a-gate-audit.v1",
        "arm": arm,
        "gate_is_learned": arm == E_N2,
        "gate_input_fields": (
            ["explicit_site_type", "standardized_molecular_formal_charge"]
            if arm == E_N2
            else []
        ),
        "validation_gate_by_site_type": by_type,
        "learned_parameters": parameters,
        "oracle_or_target_field_used": False,
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
        frozen=frozen,
        arm=arm,
        config=config,
        initialization_seed=initialization_seed,
        device=device,
    )
    initial_base_hash = _tensor_mapping_sha256(model.frozen_base.state_dict())
    if initial_base_hash != frozen.model_state_sha256:
        raise SiteNCampaignError("Wrapped C2 state changed before training")
    target_weights, target_weight_audit = stage_c_target_weights(
        train,
        use_h1=False,
        use_h2=True,
        maximum_weight=float(config["r2"]["optimization"]["maximum_target_weight"]),
    )
    target_weight_audit = {
        **target_weight_audit,
        "stage_e_a_reuse": "same_train_only_C2_H2_definition",
        "fit_roles": ["train"],
        "validation_or_test_target_used_for_fit": False,
    }
    pairs, paired_audit = stage_d_paired_solvent_definitions(train)
    paired_audit = {
        **paired_audit,
        "stage_e_a_arm": arm,
        "paired_solvent_auxiliary_enabled": True,
        "pair_aware_batching": True,
    }
    center_groups, center_audit = stage_e_a_solvent_center_groups(train)
    if not pairs or not center_groups:
        raise SiteNCampaignError("Stage-E-A has no train solvent groups")

    optimization = config["r2"]["optimization"]
    frozen_metrics, frozen_predictions = _evaluate(
        model.frozen_base,
        validation,
        preprocessor=frozen.preprocessor,
        vocabulary=frozen.vocabulary,
        batch_size=int(optimization["batch_size_contexts"]),
        device=device,
    )
    initial_metrics, initial_predictions = _evaluate(
        model,
        validation,
        preprocessor=frozen.preprocessor,
        vocabulary=frozen.vocabulary,
        batch_size=int(optimization["batch_size_contexts"]),
        device=device,
    )
    frozen_values = frozen_predictions.sort_values("target_id")["N_pred"].to_numpy(
        dtype=float
    )
    initial_values = initial_predictions.sort_values("target_id")["N_pred"].to_numpy(
        dtype=float
    )
    if not np.array_equal(frozen_values, initial_values):
        maximum = float(np.max(np.abs(frozen_values - initial_values)))
        raise SiteNCampaignError(
            f"Stage-E-A epoch-zero prediction differs from C2: {maximum}"
        )

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
            "train_total": None,
            "train_regression": None,
            "train_ranking": None,
            "train_paired_solvent_delta": None,
            "train_center_penalty": None,
            "train_residual_shrinkage": None,
            "train_gate_shrinkage": None,
            "train_ranking_pairs": 0,
            "train_paired_solvent_pairs": 0,
            "train_center_groups": 0,
            "validation_mae": float(initial_metrics["mae"]),
            "validation_rmse": float(initial_metrics["rmse"]),
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
        training = _train_epoch(
            model,
            _training_batches(
                train,
                pairs=pairs,
                batch_size=int(optimization["batch_size_contexts"]),
                preprocessor=frozen.preprocessor,
                vocabulary=frozen.vocabulary,
                shuffle_seed=initialization_seed + epoch,
            ),
            target_weights=target_weights,
            pairs=pairs,
            center_groups=center_groups,
            config=config,
            optimizer=optimizer,
            device=device,
        )
        metrics, _ = _evaluate(
            model,
            validation,
            preprocessor=frozen.preprocessor,
            vocabulary=frozen.vocabulary,
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
                "epoch_zero_frozen_c2": False,
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
    model.load_state_dict(best_state, strict=True)
    model.frozen_base.eval()
    final_base_hash = _tensor_mapping_sha256(model.frozen_base.state_dict())
    if final_base_hash != initial_base_hash:
        raise SiteNCampaignError("Frozen C2 changed during Stage-E-A fit")
    metrics, predictions = _evaluate(
        model,
        validation,
        preprocessor=frozen.preprocessor,
        vocabulary=frozen.vocabulary,
        batch_size=int(optimization["batch_size_contexts"]),
        device=device,
    )
    if not math.isclose(
        float(metrics["rmse"]),
        best_rmse,
        rel_tol=0.0,
        abs_tol=1e-7,
    ):
        raise SiteNCampaignError("Restored Stage-E-A checkpoint changed")
    components = _component_predictions(
        model,
        validation,
        preprocessor=frozen.preprocessor,
        vocabulary=frozen.vocabulary,
        batch_size=int(optimization["batch_size_contexts"]),
        device=device,
    )
    train_components = _component_predictions(
        model,
        train,
        preprocessor=frozen.preprocessor,
        vocabulary=frozen.vocabulary,
        batch_size=int(optimization["batch_size_contexts"]),
        device=device,
    )
    center_table = _center_residual_table(train_components, center_groups)
    pair_predictions, pair_metrics = _validation_pair_predictions(
        validation,
        predictions,
    )
    freeze_audit = {
        "schema_version": "nucpred.mayr-stage-e-a-freeze-audit.v1",
        "status": "pass",
        "frozen_c2_checkpoint_sha256": frozen.checkpoint_sha256,
        "frozen_c2_internal_state_sha256": frozen.model_state_sha256,
        "base_state_before_training_sha256": initial_base_hash,
        "base_state_after_training_sha256": final_base_hash,
        "base_state_bitwise_unchanged": initial_base_hash == final_base_hash,
        "base_parameters_require_grad_false": (
            frozen_base_parameters_are_frozen(model)
        ),
        "base_forced_eval_mode": not model.frozen_base.training,
        "epoch_zero_prediction_exactly_equal_to_c2": True,
        "epoch_zero_maximum_absolute_difference_N_units": 0.0,
        "trainable_parameter_count": trainable_parameter_count(model),
        "frozen_parameter_count": sum(
            parameter.numel() for parameter in model.frozen_base.parameters()
        ),
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
        train_center_residuals=center_table,
        frozen_parent_metrics=frozen_metrics,
        freeze_audit=freeze_audit,
        target_weight_audit=target_weight_audit,
        paired_solvent_audit=paired_audit,
        center_group_audit=center_audit,
        gate_audit=_gate_audit(
            arm=arm,
            components=components,
            model=model,
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
        "stage_e_a_config": config_path,
        "authorization": _project_path(
            config["authorization"]["path"],
            label="authorization.path",
        ),
        "stage_d_catalog": _project_path(
            config["parents"]["stage_d_catalog_path"],
            label="parents.stage_d_catalog_path",
        ),
        "stage_c_r2_manifest": _project_path(
            config["parents"]["stage_c_r2_manifest_path"],
            label="parents.stage_c_r2_manifest_path",
        ),
        "runner": Path(__file__).resolve(),
        "stage_e_a_model": (
            ROOT / "src/nucpred/training/mayr_site_n_stage_e_a.py"
        ).resolve(),
        "stage_d_model": (
            ROOT / "src/nucpred/training/mayr_site_n_stage_d.py"
        ).resolve(),
        "stage_d_runner": (
            ROOT / "src/nucpred/experiments/mayr/nextgen_stage_d_r2.py"
        ).resolve(),
        "stage_c_model": (
            ROOT / "src/nucpred/training/mayr_site_n_stage_c.py"
        ).resolve(),
        "base_model": (ROOT / "src/nucpred/training/mayr_site_n.py").resolve(),
        "base_runner": (ROOT / "src/nucpred/experiments/mayr/site_n.py").resolve(),
        "dataset_loader": (ROOT / "src/nucpred/datasets/mayr_site_n.py").resolve(),
        "dataset_manifest": dataset / "dataset_manifest.json",
        "split_manifest": dataset / "split_manifest.json",
        "frozen_c2_checkpoint": frozen.checkpoint_path,
    }
    contract: dict[str, object] = {
        "schema_version": "nucpred.mayr-nextgen-stage-e-a-r2-job-contract.v1",
        "campaign_id": str(config["campaign_id"]),
        "split_seed": int(split_seed),
        "initialization_seed": int(initialization_seed),
        "arm": arm,
        "source_hashes": {name: sha256_file(path) for name, path in sources.items()},
        "frozen_c2_internal_state_sha256": frozen.model_state_sha256,
        "split_audit": dict(split_audit),
        "selection_metric": "rmse",
        "selection_roles": ["train", "validation"],
        "epoch_zero_frozen_c2_is_eligible": True,
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
    frozen: FrozenC2,
    contract: Mapping[str, object],
) -> None:
    state = {
        name: tensor.detach().cpu()
        for name, tensor in outcome.model.state_dict().items()
    }
    trainable_state = {
        name: tensor.detach().cpu()
        for name, tensor in outcome.model.state_dict().items()
        if not name.startswith("frozen_base.")
    }
    payload = {
        "schema_version": "nucpred.mayr-nextgen-stage-e-a-r2-checkpoint.v1",
        "phase": "development_validation_selection",
        "arm": arm,
        "split_seed": int(split_seed),
        "initialization_seed": int(initialization_seed),
        "selection_metric": "rmse",
        "selection_best_epoch": int(outcome.best_epoch),
        "model_architecture": outcome.model.architecture,
        "model_state_dict": state,
        "model_state_sha256": _tensor_mapping_sha256(state),
        "trainable_state_sha256": _tensor_mapping_sha256(trainable_state),
        "frozen_c2_checkpoint": _display_path(frozen.checkpoint_path),
        "frozen_c2_checkpoint_sha256": frozen.checkpoint_sha256,
        "frozen_c2_internal_state_sha256": frozen.model_state_sha256,
        "preprocessor": frozen.preprocessor.to_json(),
        "solvent_vocabulary": list(frozen.vocabulary.tokens),
        "contract": dict(contract),
        "freeze_audit": dict(outcome.freeze_audit),
        "target_weight_audit": dict(outcome.target_weight_audit),
        "paired_solvent_audit": dict(outcome.paired_solvent_audit),
        "center_group_audit": dict(outcome.center_group_audit),
        "gate_audit": dict(outcome.gate_audit),
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
        raise SiteNCampaignError("Unregistered Stage-E-A job axis")
    selected_device = torch.device(device)
    if selected_device.type != "cuda" or not torch.cuda.is_available():
        raise SiteNCampaignError("Stage-E-A R2 matrix requires CUDA")
    config, config_file = read_config(config_path)
    dataset = _project_path(
        config["dataset"]["directory"],
        label="dataset.directory",
    )
    dataset_verification = verify_dataset(dataset)
    train, validation, split_audit = _split_examples(
        dataset,
        split_seed=split_seed,
    )
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
        _project_path(
            config["r2"]["output_directory"],
            label="r2.output_directory",
        )
        / f"split-{split_seed}"
        / f"init-{initialization_seed}"
        / arm
    )
    summary_path = target / "summary.json"
    if summary_path.is_file():
        existing = _load_json(summary_path)
        if existing.get("status") == "pass" and existing.get("contract") == contract:
            return existing
        raise SiteNCampaignError(f"Existing Stage-E-A job is stale: {target}")
    if target.exists():
        raise SiteNCampaignError(f"Partial Stage-E-A job exists: {target}")
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
        if predictions["connectivity_id"].isna().any():
            raise SiteNCampaignError("Stage-E-A validation connectivity mapping failed")
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
            outcome.validation_components.to_parquet(
                staging / "validation_components.parquet",
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
            outcome.train_center_residuals.to_parquet(
                staging / "train_center_group_residuals.parquet",
                index=False,
                engine="pyarrow",
                compression="zstd",
            )
            json_assets = {
                "validation_metrics.json": outcome.validation_metrics,
                "frozen_c2_validation_metrics.json": (outcome.frozen_parent_metrics),
                "validation_paired_solvent_metrics.json": (
                    outcome.validation_pair_metrics
                ),
                "frozen_c2_verification.json": frozen.verification,
                "freeze_audit.json": outcome.freeze_audit,
                "target_weight_audit.json": outcome.target_weight_audit,
                "paired_solvent_audit.json": outcome.paired_solvent_audit,
                "center_group_audit.json": outcome.center_group_audit,
                "gate_audit.json": outcome.gate_audit,
                "preprocessor.json": frozen.preprocessor.to_json(),
            }
            for filename, payload in json_assets.items():
                atomic_write_json(
                    staging / filename,
                    payload,
                    ensure_ascii=False,
                )
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
                "schema_version": "nucpred.mayr-nextgen-stage-e-a-r2-job.v1",
                "status": "pass",
                "campaign_id": config["campaign_id"],
                "split_seed": int(split_seed),
                "initialization_seed": int(initialization_seed),
                "arm": arm,
                "contract": contract,
                "dataset_verification": dataset_verification,
                "frozen_c2_verification": dict(frozen.verification),
                "split_audit": split_audit,
                "selection_metric": "rmse",
                "selection_best_epoch": outcome.best_epoch,
                "selection_best_validation_rmse": (outcome.best_validation_rmse),
                "epoch_zero_frozen_c2_was_selected": (outcome.best_epoch == 0),
                "validation_metrics": dict(outcome.validation_metrics),
                "frozen_c2_validation_metrics": dict(outcome.frozen_parent_metrics),
                "validation_paired_solvent_metrics": dict(
                    outcome.validation_pair_metrics
                ),
                "freeze_audit": dict(outcome.freeze_audit),
                "target_weight_audit": dict(outcome.target_weight_audit),
                "paired_solvent_audit": dict(outcome.paired_solvent_audit),
                "center_group_audit": dict(outcome.center_group_audit),
                "gate_audit": dict(outcome.gate_audit),
                "test_examples_instantiated": 0,
                "test_predictions_computed": 0,
                "test_prediction_files_written": 0,
                "formal_calibration_performed": False,
                "absolute_probability_published": False,
                "final_refit_performed": False,
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
                        "selection_roles=train,validation",
                        f"selection_best_epoch={outcome.best_epoch}",
                        (
                            "selection_validation_rmse="
                            f"{outcome.best_validation_rmse:.12f}"
                        ),
                        "test_examples_instantiated=0",
                        "test_predictions_computed=0",
                        "test_prediction_files_written=0",
                        "formal_calibration_performed=false",
                        "absolute_probability_published=false",
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
                "schema_version": ("nucpred.mayr-nextgen-stage-e-a-r2-failure.v1"),
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
                        "event": "stage_e_a_r2_job_complete",
                        "split_seed": split_seed,
                        "initialization_seed": initialization_seed,
                        "arm": arm,
                        "best_epoch": summary["selection_best_epoch"],
                        "validation_rmse": (summary["selection_best_validation_rmse"]),
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
        raise SiteNCampaignError("Stage-E-A R2 matrix requires CUDA")
    config, config_file = read_config(config_path)
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
            "schema_version": ("nucpred.mayr-nextgen-stage-e-a-r2-coordinator.v1"),
            "status": "pass",
            "parallel_workers": 0,
            "completed_job_count": len(expected),
        }
    gc.collect()
    gc.freeze()
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(
            name=f"mayr-stage-e-a-r2-{seed}",
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
                "event": "stage_e_a_r2_matrix_started",
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
                        "event": "stage_e_a_r2_matrix_heartbeat",
                        "completed_job_count": sum(path.is_file() for path in expected),
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
        process.name: process.exitcode for process in processes if process.exitcode != 0
    }
    if failures:
        raise SiteNCampaignError(f"Stage-E-A R2 workers failed: {failures}")
    missing = [_display_path(path) for path in expected if not path.is_file()]
    if missing:
        raise SiteNCampaignError(f"Stage-E-A R2 summaries are missing: {missing}")
    atomic_write_json(
        output_root / "summary.json",
        {
            "schema_version": ("nucpred.mayr-nextgen-stage-e-a-r2-matrix.v1"),
            "status": "pass",
            "campaign_id": config["campaign_id"],
            "completed_job_count": len(expected),
            "expected_job_count": len(expected),
            "parallel_worker_count": len(processes),
            "selection_roles": ["train", "validation"],
            "epoch_zero_frozen_c2_eligible": True,
            "test_examples_instantiated": 0,
            "test_predictions_computed": 0,
            "test_prediction_files_written": 0,
            "formal_calibration_performed": False,
            "absolute_probability_published": False,
            "final_refit_performed": False,
            "d_n4_combination_performed": False,
            "dft_or_cdft_computation_performed": False,
        },
        ensure_ascii=False,
    )
    _write_manifest(output_root)
    return {
        "schema_version": ("nucpred.mayr-nextgen-stage-e-a-r2-coordinator.v1"),
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
            for value in (
                args.split_seed,
                args.initialization_seed,
                args.arm,
            )
        ):
            raise SiteNCampaignError(
                "--all-jobs cannot be combined with single-job axes"
            )
        payload = run_all(config_path=args.config, device=args.device)
    else:
        if any(
            value is None
            for value in (
                args.split_seed,
                args.initialization_seed,
                args.arm,
            )
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
