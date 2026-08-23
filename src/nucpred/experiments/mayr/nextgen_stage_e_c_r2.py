"""Run the authorized Stage-E-C independent-expert development matrix."""

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
    MayrSiteNModel,
    SiteNExample,
    SiteNFoldPreprocessor,
    SolventVocabulary,
    fit_site_n_preprocessor,
    seed_everything,
)
from nucpred.training.mayr_site_n_stage_c import stage_c_target_weights
from nucpred.training.mayr_site_n_stage_e_a import stage_e_a_site_n_loss
from nucpred.training.mayr_site_n_stage_e_b import (
    E_B_N1,
    MayrSiteNStageEBResidualModel,
)
from nucpred.training.mayr_site_n_stage_e_c import (
    COORDINATION_BOND_TYPE_CHANNELS,
    COORDINATION_ELEMENT_CHANNELS,
    E_C_N2,
    E_C_N3,
    STAGE_E_C_ARMS,
    MayrSiteNStageECExpertModel,
    coordination_context_indicators,
    exact_heavy_parent_summary,
    frozen_parent_parameters_are_frozen,
    trainable_parameter_count,
    zero_residual_output_is_exact,
)

from .nextgen_gate_a import _canonical_sha256, _verify_bound_file
from .nextgen_stage_d_r2 import _split_examples, _validation_pair_predictions
from .nextgen_stage_e_a_r2 import _display_path, _load_json, _project_path
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
DEFAULT_CONFIG = ROOT / "configs/mayr_nextgen_stage_e_c.toml"
CONFIG_SCHEMA = "nucpred.mayr-nextgen-stage-e-c-config.v1"
EXPECTED_STATUS = "awaiting_stage_e_c_results_gate"
ARMS = STAGE_E_C_ARMS


@dataclass(frozen=True, slots=True)
class FrozenEBN1:
    model: MayrSiteNStageEBResidualModel
    preprocessor: SiteNFoldPreprocessor
    vocabulary: SolventVocabulary
    checkpoint_path: Path
    checkpoint_sha256: str
    model_state_sha256: str
    payload: Mapping[str, Any]
    verification: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class SelectionOutcome:
    model: MayrSiteNStageECExpertModel
    curves: pd.DataFrame
    best_epoch: int
    best_validation_rmse: float
    validation_metrics: Mapping[str, object]
    validation_predictions: pd.DataFrame
    validation_components: pd.DataFrame
    validation_pair_predictions: pd.DataFrame
    validation_pair_metrics: Mapping[str, object]
    frozen_parent_metrics: Mapping[str, object]
    freeze_audit: Mapping[str, object]
    target_weight_audit: Mapping[str, object]
    gate_audit: Mapping[str, object]


def read_config(path: str | Path = DEFAULT_CONFIG) -> tuple[dict[str, Any], Path]:
    """Read Stage-E-C authority while rejecting every forbidden scope change."""

    config_path = Path(path).resolve()
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise SiteNCampaignError("Stage-E-C config schema changed")
    if config.get("status_after_completion") != EXPECTED_STATUS:
        raise SiteNCampaignError("Stage-E-C result gate changed")
    if int(config.get("maximum_parallel_gpu_processes", -1)) != 3:
        raise SiteNCampaignError("Stage-E-C GPU ceiling changed")
    forbidden = (
        "test_labels_examples_or_predictions_permitted",
        "formal_calibration_permitted",
        "absolute_probability_publication_permitted",
        "final_refit_permitted",
        "combined_arm_permitted",
        "d_n4_combination_permitted",
        "dft_or_cdft_computation_permitted",
        "automatic_continuation_permitted",
    )
    if any(config.get(key) is not False for key in forbidden):
        raise SiteNCampaignError("Stage-E-C forbidden scope changed")

    r2 = config["r2"]
    if tuple(map(str, r2["arms"])) != ARMS:
        raise SiteNCampaignError("Stage-E-C independent arms changed")
    if tuple(map(int, r2["split_seeds"])) != SPLIT_SEEDS:
        raise SiteNCampaignError("Stage-E-C split seeds changed")
    if tuple(map(int, r2["initialization_seeds"])) != INITIALIZATION_SEEDS:
        raise SiteNCampaignError("Stage-E-C initialization seeds changed")
    if (
        r2.get("selection_metric") != "rmse"
        or list(r2.get("selection_roles", [])) != ["train", "validation"]
        or r2.get("epoch_zero_frozen_parent_is_eligible") is not True
        or r2.get("frozen_parent_arm") != E_B_N1
        or r2.get("pair_aware_batching") is not False
        or float(r2.get("paired_solvent_weight", -1.0)) != 0.0
        or float(r2.get("center_penalty_weight", -1.0)) != 0.0
        or any(
            r2.get(key) is not False
            for key in (
                "test_examples_permitted",
                "test_predictions_permitted",
                "final_refit_permitted",
                "combined_arm_permitted",
                "dft_or_cdft_permitted",
            )
        )
    ):
        raise SiteNCampaignError("Stage-E-C development-only contract changed")
    if tuple(r2["e_c_n1"]["active_site_types"]) != ("atom",):
        raise SiteNCampaignError("E-C-N1 fixed gate changed")
    if tuple(r2["e_c_n2"]["active_site_types"]) != ("transferable_h_group",):
        raise SiteNCampaignError("E-C-N2 fixed gate changed")
    if tuple(r2["e_c_n3"]["active_site_types"]) != (
        "bond",
        "delocalized_region",
    ):
        raise SiteNCampaignError("E-C-N3 fixed gate changed")
    if tuple(r2["e_c_n3"]["coordination_element_channels"]) != (
        COORDINATION_ELEMENT_CHANNELS
    ):
        raise SiteNCampaignError("E-C-N3 coordination element channels changed")
    if tuple(r2["e_c_n3"]["coordination_bond_type_channels"]) != (
        COORDINATION_BOND_TYPE_CHANNELS
    ):
        raise SiteNCampaignError("E-C-N3 coordination bond channels changed")
    if (
        r2["e_c_n3"]["mayr_class_labels_as_model_input"] is not False
        or r2["e_c_n3"]["target_or_n_value_as_context_input"] is not False
    ):
        raise SiteNCampaignError("E-C-N3 target-independent contract changed")

    evidence = config["evidence"]
    if (
        evidence.get("site_type") != "atom_group"
        or evidence.get("review_mode") != "single_operator_two_isolated_passes"
        or int(evidence.get("existing_confirmed_positive_connectivity_count", -1)) != 6
        or int(evidence.get("target_total_positive_connectivity_count", -1)) != 25
        or int(evidence.get("requested_new_positive_connectivity_count", -1)) != 19
        or evidence.get("primary_source_positive_required") is not True
        or evidence.get("pass_a_candidate_members_visible") is not False
        or evidence.get("pass_a_model_scores_visible") is not False
        or evidence.get("pass_a_split_roles_visible") is not False
        or evidence.get("pass_b_requires_frozen_pass_a_manifest") is not True
        or evidence.get("pass_b_model_scores_visible") is not False
        or evidence.get("pass_b_split_roles_visible") is not False
        or evidence.get("endpoint_family_relation") != "exact_same_members_only"
        or evidence.get("strict_containment_label_propagation") is not False
        or evidence.get("partial_overlap_label_propagation") is not False
        or evidence.get("new_binary_negative_labels_permitted") is not False
        or evidence.get("unknown_is_negative") is not False
    ):
        raise SiteNCampaignError("Stage-E-C atom_group evidence contract changed")

    bindings = (
        ("authorization", "path", "sha256"),
        ("parents", "stage_e_b_catalog_path", "stage_e_b_catalog_sha256"),
        ("parents", "stage_e_b_config_path", "stage_e_b_config_sha256"),
        ("parents", "stage_e_b_r2_manifest_path", "stage_e_b_r2_manifest_sha256"),
        (
            "parents",
            "stage_e_b_results_manifest_path",
            "stage_e_b_results_manifest_sha256",
        ),
        (
            "parents",
            "stage_e_b_results_summary_path",
            "stage_e_b_results_summary_sha256",
        ),
        (
            "parents",
            "stage_e_b_lineage_pass_a_manifest_path",
            "stage_e_b_lineage_pass_a_manifest_sha256",
        ),
        (
            "parents",
            "stage_e_b_lineage_pass_b_manifest_path",
            "stage_e_b_lineage_pass_b_manifest_sha256",
        ),
        ("parents", "stage_e_b_lineage_path", "stage_e_b_lineage_sha256"),
        ("dataset", "manifest_path", "manifest_sha256"),
        ("dataset", "split_manifest_path", "split_manifest_sha256"),
        (
            "evidence",
            "gate_a_preflight_manifest_path",
            "gate_a_preflight_manifest_sha256",
        ),
        (
            "evidence",
            "gate_a_candidate_policy_path",
            "gate_a_candidate_policy_sha256",
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
            "stage_d_typed_evidence_path",
            "stage_d_typed_evidence_sha256",
        ),
        (
            "evidence",
            "stage_e_a_pass_a_manifest_path",
            "stage_e_a_pass_a_manifest_sha256",
        ),
        (
            "evidence",
            "stage_e_a_pass_b_manifest_path",
            "stage_e_a_pass_b_manifest_sha256",
        ),
        (
            "evidence",
            "stage_e_a_positive_expansion_path",
            "stage_e_a_positive_expansion_sha256",
        ),
        (
            "evidence",
            "stage_e_b_lineage_pass_a_manifest_path",
            "stage_e_b_lineage_pass_a_manifest_sha256",
        ),
        (
            "evidence",
            "stage_e_b_lineage_pass_b_manifest_path",
            "stage_e_b_lineage_pass_b_manifest_sha256",
        ),
        (
            "evidence",
            "stage_e_b_lineage_path",
            "stage_e_b_lineage_sha256",
        ),
        (
            "evidence",
            "inference_contract_path",
            "inference_contract_sha256",
        ),
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
        or contract.get("stage_e_c_development_training_authorized") is not True
        or tuple(contract.get("authorized_independent_arms", ())) != ARMS
        or contract.get("frozen_stage_e_b_n1_parent_required") is not True
        or contract.get("epoch_zero_frozen_parent_must_be_eligible") is not True
        or contract.get("selection_roles") != ["train", "validation"]
        or contract.get("maximum_parallel_gpu_processes") != 3
        or contract.get("atom_group_primary_source_positive_expansion_authorized")
        is not True
        or contract.get("atom_group_existing_positive_connectivity_baseline") != 6
        or contract.get("atom_group_total_positive_connectivity_target") != 25
        or contract.get("atom_group_requested_increment_if_corpus_supports_it") != 19
        or contract.get("status_after_completion") != EXPECTED_STATUS
        or any(
            contract.get(key) is not False
            for key in (
                "combined_arm_authorized",
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
        raise SiteNCampaignError("Stage-E-C authorization changed")
    parent = ArtifactCatalog().verify(str(config["parents"]["stage_e_b_run_id"]))
    if parent.get("status") != "pass":
        raise SiteNCampaignError("Stage-E-B parent catalog no longer verifies")
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


def _load_frozen_parent(
    *,
    config: Mapping[str, Any],
    train: Sequence[SiteNExample],
    split_seed: int,
    initialization_seed: int,
    device: torch.device,
) -> FrozenEBN1:
    """Load one manifest-bound E-B-N1 checkpoint without fitting parent state."""

    r2 = config["r2"]
    relative = _checkpoint_relative_path(
        split_seed=split_seed,
        initialization_seed=initialization_seed,
        arm=str(r2["frozen_parent_arm"]),
    )
    checkpoint_root = _project_path(
        r2["frozen_parent_checkpoint_root"],
        label="r2.frozen_parent_checkpoint_root",
    )
    checkpoint = checkpoint_root / relative
    manifest_path = _project_path(
        config["parents"]["stage_e_b_r2_manifest_path"],
        label="parents.stage_e_b_r2_manifest_path",
    )
    manifest = _load_json(manifest_path)
    entry = manifest.get("files", {}).get(relative)
    if not isinstance(entry, Mapping):
        raise SiteNCampaignError("E-B-N1 checkpoint missing from frozen manifest")
    checkpoint_hash = sha256_file(checkpoint)
    if checkpoint_hash != str(entry.get("sha256")):
        raise SiteNCampaignError("Frozen E-B-N1 checkpoint hash changed")

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise SiteNCampaignError("Frozen E-B-N1 checkpoint payload changed")
    if (
        payload.get("schema_version") != r2["frozen_parent_checkpoint_schema"]
        or payload.get("phase") != "development_validation_selection"
        or payload.get("arm") != E_B_N1
        or int(payload.get("split_seed", -1)) != split_seed
        or int(payload.get("initialization_seed", -1)) != initialization_seed
    ):
        raise SiteNCampaignError("Frozen E-B-N1 checkpoint identity changed")
    architecture = payload.get("model_architecture")
    state = payload.get("model_state_dict")
    if not isinstance(architecture, Mapping) or not isinstance(state, Mapping):
        raise SiteNCampaignError("Frozen E-B-N1 model payload is incomplete")
    if (
        architecture.get("schema_version")
        != "nucpred.mayr-site-n-stage-e-b-residual.v1"
        or architecture.get("arm") != E_B_N1
    ):
        raise SiteNCampaignError("Frozen E-B-N1 architecture changed")
    base_architecture = architecture.get("frozen_base_architecture")
    if (
        not isinstance(base_architecture, Mapping)
        or base_architecture.get("schema_version") != "nucpred.mayr-site-n-model.v1"
        or base_architecture.get("site_probability_normalization") is not False
    ):
        raise SiteNCampaignError("Frozen E-B-N1 base architecture changed")
    state_hash = _tensor_mapping_sha256(state)
    if state_hash != str(payload.get("model_state_sha256")):
        raise SiteNCampaignError("Frozen E-B-N1 internal state hash changed")

    base = MayrSiteNModel(
        num_solvents=int(base_architecture["num_solvents"]),
        hidden_dim=int(base_architecture["hidden_dim"]),
        layers=int(base_architecture["layers"]),
        node_embedding_dim=int(base_architecture["node_embedding_dim"]),
        edge_embedding_dim=int(base_architecture["edge_embedding_dim"]),
        solvent_embedding_dim=int(base_architecture["solvent_embedding_dim"]),
        dropout=float(base_architecture["dropout"]),
    )
    model = MayrSiteNStageEBResidualModel(frozen_base=base, arm=E_B_N1)
    model.load_state_dict(state, strict=True)
    if _tensor_mapping_sha256(model.state_dict()) != state_hash:
        raise SiteNCampaignError("Exact E-B-N1 state load failed")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()

    preprocessor_payload = payload.get("preprocessor")
    vocabulary_payload = payload.get("solvent_vocabulary")
    if not isinstance(preprocessor_payload, Mapping) or not isinstance(
        vocabulary_payload, Sequence
    ):
        raise SiteNCampaignError("Frozen E-B-N1 preprocessing payload changed")
    preprocessor = SiteNFoldPreprocessor.from_json(preprocessor_payload)
    vocabulary = SolventVocabulary(tuple(str(value) for value in vocabulary_payload))
    if _canonical_sha256(fit_site_n_preprocessor(train).to_json()) != (
        _canonical_sha256(preprocessor.to_json())
    ):
        raise SiteNCampaignError("E-B-N1 preprocessor does not match train role")
    expected_vocabulary = SolventVocabulary.from_values(
        [example.solvent_raw for example in train]
    )
    if vocabulary != expected_vocabulary:
        raise SiteNCampaignError("E-B-N1 solvent vocabulary does not match train")
    if int(base_architecture["num_solvents"]) != len(vocabulary.tokens):
        raise SiteNCampaignError("E-B-N1 vocabulary width changed")

    verification = {
        "schema_version": "nucpred.mayr-stage-e-c-parent-verification.v1",
        "status": "pass",
        "path": _display_path(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "manifest_path": _display_path(manifest_path),
        "manifest_sha256": config["parents"]["stage_e_b_r2_manifest_sha256"],
        "manifest_entry_verified": True,
        "model_state_sha256": state_hash,
        "exact_state_load": True,
        "preprocessor_reused_exactly": True,
        "preprocessor_matches_train_role": True,
        "solvent_vocabulary_reused_exactly": True,
        "parent_arm": E_B_N1,
        "test_state_or_examples_read": False,
    }
    return FrozenEBN1(
        model=model.to(device),
        preprocessor=preprocessor,
        vocabulary=vocabulary,
        checkpoint_path=checkpoint,
        checkpoint_sha256=checkpoint_hash,
        model_state_sha256=state_hash,
        payload=payload,
        verification=verification,
    )


def _make_model(
    frozen: FrozenEBN1,
    *,
    arm: str,
    initialization_seed: int,
    device: torch.device,
) -> MayrSiteNStageECExpertModel:
    seed_everything(initialization_seed)
    model = MayrSiteNStageECExpertModel(
        frozen_parent=frozen.model,
        arm=arm,
    )
    if not zero_residual_output_is_exact(model):
        raise SiteNCampaignError("Stage-E-C residual is not exact-zero")
    if not frozen_parent_parameters_are_frozen(model):
        raise SiteNCampaignError("Stage-E-C E-B-N1 parent is trainable")
    if trainable_parameter_count(model) <= 0:
        raise SiteNCampaignError("Stage-E-C has no trainable parameters")
    return model.to(device)


def _ordinary_batches(
    train: Sequence[SiteNExample],
    *,
    frozen: FrozenEBN1,
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
    model: MayrSiteNStageECExpertModel,
    batches: Iterator,
    *,
    target_weights: Mapping[str, float],
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
            paired_solvent_pairs=(),
            paired_solvent_weight=0.0,
            center_groups=(),
            center_penalty_weight=0.0,
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
        target_count += count
        for name in names:
            value = total if name == "total" else parts[name]
            if isinstance(value, torch.Tensor):
                totals[name] += float(value.detach().cpu()) * count
    if target_count == 0:
        raise SiteNCampaignError("Stage-E-C epoch had no targets")
    return {name: value / target_count for name, value in totals.items()}


def _component_predictions(
    model: MayrSiteNStageECExpertModel,
    examples: Sequence[SiteNExample],
    *,
    frozen: FrozenEBN1,
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
        for raw in _iter_batches(
            examples,
            batch_size=batch_size,
            preprocessor=frozen.preprocessor,
            vocabulary=frozen.vocabulary,
            shuffle_seed=None,
        ):
            batch = raw.to(device)
            output = model(batch.inputs)
            scale = float(frozen.preprocessor.target_scale)
            mean = float(frozen.preprocessor.target_mean)
            prediction = (
                output.n_prediction_standardized.detach().cpu().numpy() * scale + mean
            )
            parent = (
                output.frozen_base_prediction_standardized.detach().cpu().numpy()
                * scale
                + mean
            )
            raw_residual = (
                output.raw_residual_standardized.detach().cpu().numpy() * scale
            )
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
                        "N_frozen_stage_e_b_n1": float(parent[index]),
                        "N_pred": float(prediction[index]),
                        "raw_residual_N_units": float(raw_residual[index]),
                        "applied_residual_N_units": float(applied[index]),
                        "residual_gate": float(gate[index]),
                    }
                )
    return pd.DataFrame(rows).sort_values("target_id").reset_index(drop=True)


def _gate_audit(
    model: MayrSiteNStageECExpertModel,
    examples: Sequence[SiteNExample],
    *,
    frozen: FrozenEBN1,
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    total = 0
    active = 0
    type_counts: dict[str, int] = {}
    h_malformed = 0
    context_counts: list[int] | None = None
    model.eval()
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
            gate = output.residual_gate.bool()
            total += batch.inputs.num_sites
            active += int(gate.sum().item())
            for index, name in enumerate(
                (
                    "atom",
                    "bond",
                    "delocalized_region",
                    "atom_group",
                    "transferable_h_group",
                )
            ):
                type_counts[name] = type_counts.get(name, 0) + int(
                    (batch.inputs.site_type_index == index).sum().item()
                )
            if model.arm == E_C_N2:
                parent = exact_heavy_parent_summary(
                    batch.inputs,
                    output.node_embeddings,
                )
                typed = batch.inputs.site_type_index == 4
                h_malformed += int((typed & ~parent.valid).sum().item())
            if model.arm == E_C_N3:
                context = coordination_context_indicators(batch.inputs)
                if context_counts is None:
                    context_counts = [0] * int(context.shape[1])
                for index in range(context.shape[1]):
                    context_counts[index] += int(context[:, index].sum().item())
    channel_names = [
        item
        for symbol in COORDINATION_ELEMENT_CHANNELS
        for item in (f"graph_element_{symbol}", f"member_element_{symbol}")
    ] + [f"graph_bond_{label}" for label in COORDINATION_BOND_TYPE_CHANNELS]
    return {
        "schema_version": "nucpred.mayr-stage-e-c-gate-audit.v1",
        "arm": model.arm,
        "target_count": total,
        "active_target_count": active,
        "inactive_exact_parent_fallback_count": total - active,
        "site_type_target_counts": type_counts,
        "malformed_transferable_h_parent_fallback_count": h_malformed,
        "coordination_channel_counts": (
            dict(zip(channel_names, context_counts, strict=True))
            if context_counts is not None
            else {}
        ),
        "gate_is_learned": False,
        "mayr_class_target_error_or_split_role_used": False,
        "combined_arm": False,
    }


def _fit_selection(
    train: Sequence[SiteNExample],
    validation: Sequence[SiteNExample],
    *,
    arm: str,
    config: Mapping[str, Any],
    frozen: FrozenEBN1,
    initialization_seed: int,
    device: torch.device,
) -> SelectionOutcome:
    model = _make_model(
        frozen,
        arm=arm,
        initialization_seed=initialization_seed,
        device=device,
    )
    initial_hash = _tensor_mapping_sha256(model.frozen_parent.state_dict())
    weights, weight_audit = stage_c_target_weights(
        train,
        use_h1=False,
        use_h2=True,
        maximum_weight=float(config["r2"]["optimization"]["maximum_target_weight"]),
    )
    weight_audit = {
        **weight_audit,
        "stage_e_c_reuse": "same_train_only_C2_H2_definition",
        "fit_roles": ["train"],
    }
    optimization = config["r2"]["optimization"]
    batch_size = int(optimization["batch_size_contexts"])
    parent_metrics, parent_predictions = _evaluate(
        model.frozen_parent,
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
    parent_values = parent_predictions.sort_values("target_id")["N_pred"].to_numpy()
    initial_values = initial_predictions.sort_values("target_id")["N_pred"].to_numpy()
    if not np.array_equal(parent_values, initial_values):
        raise SiteNCampaignError("Stage-E-C epoch-zero differs from frozen E-B-N1")

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
            "epoch_zero_frozen_parent": True,
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
        training = _train_epoch(
            model,
            _ordinary_batches(
                train,
                frozen=frozen,
                batch_size=batch_size,
                shuffle_seed=initialization_seed + epoch,
            ),
            target_weights=weights,
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
                "epoch_zero_frozen_parent": False,
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
    model.frozen_parent.eval()
    final_hash = _tensor_mapping_sha256(model.frozen_parent.state_dict())
    if final_hash != initial_hash:
        raise SiteNCampaignError("Frozen E-B-N1 changed during Stage-E-C")
    metrics, predictions = _evaluate(
        model,
        validation,
        preprocessor=frozen.preprocessor,
        vocabulary=frozen.vocabulary,
        batch_size=batch_size,
        device=device,
    )
    if not math.isclose(float(metrics["rmse"]), best_rmse, abs_tol=1e-7):
        raise SiteNCampaignError("Restored Stage-E-C checkpoint changed")
    components = _component_predictions(
        model,
        validation,
        frozen=frozen,
        batch_size=batch_size,
        device=device,
    )
    pair_predictions, pair_metrics = _validation_pair_predictions(
        validation,
        predictions,
    )
    freeze_audit = {
        "schema_version": "nucpred.mayr-stage-e-c-freeze-audit.v1",
        "status": "pass",
        "frozen_parent_checkpoint_sha256": frozen.checkpoint_sha256,
        "frozen_parent_internal_state_sha256": frozen.model_state_sha256,
        "parent_state_before_training_sha256": initial_hash,
        "parent_state_after_training_sha256": final_hash,
        "parent_state_bitwise_unchanged": initial_hash == final_hash,
        "parent_parameters_require_grad_false": (
            frozen_parent_parameters_are_frozen(model)
        ),
        "parent_forced_eval_mode": not model.frozen_parent.training,
        "epoch_zero_prediction_exactly_equal_to_stage_e_b_n1": True,
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
        frozen_parent_metrics=parent_metrics,
        freeze_audit=freeze_audit,
        target_weight_audit=weight_audit,
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
    frozen: FrozenEBN1,
    split_seed: int,
    initialization_seed: int,
    arm: str,
    split_audit: Mapping[str, object],
) -> dict[str, object]:
    sources = {
        "stage_e_c_config": config_path,
        "authorization": _project_path(
            config["authorization"]["path"],
            label="authorization.path",
        ),
        "runner": Path(__file__).resolve(),
        "stage_e_c_model": (
            ROOT / "src/nucpred/training/mayr_site_n_stage_e_c.py"
        ).resolve(),
        "stage_e_b_model": (
            ROOT / "src/nucpred/training/mayr_site_n_stage_e_b.py"
        ).resolve(),
        "stage_e_a_loss": (
            ROOT / "src/nucpred/training/mayr_site_n_stage_e_a.py"
        ).resolve(),
        "base_model": (ROOT / "src/nucpred/training/mayr_site_n.py").resolve(),
        "dataset_manifest": dataset / "dataset_manifest.json",
        "split_manifest": dataset / "split_manifest.json",
        "frozen_stage_e_b_n1_checkpoint": frozen.checkpoint_path,
    }
    contract: dict[str, object] = {
        "schema_version": "nucpred.mayr-nextgen-stage-e-c-r2-job-contract.v1",
        "campaign_id": config["campaign_id"],
        "split_seed": split_seed,
        "initialization_seed": initialization_seed,
        "arm": arm,
        "source_hashes": {key: sha256_file(value) for key, value in sources.items()},
        "frozen_parent_internal_state_sha256": frozen.model_state_sha256,
        "split_audit": dict(split_audit),
        "selection_metric": "rmse",
        "selection_roles": ["train", "validation"],
        "epoch_zero_frozen_parent_is_eligible": True,
        "test_examples_instantiated": 0,
        "test_predictions_computed": 0,
        "formal_calibration_performed": False,
        "final_refit_performed": False,
        "combined_arm_performed": False,
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
    frozen: FrozenEBN1,
    contract: Mapping[str, object],
) -> None:
    state = {
        name: tensor.detach().cpu()
        for name, tensor in outcome.model.state_dict().items()
    }
    payload = {
        "schema_version": "nucpred.mayr-nextgen-stage-e-c-r2-checkpoint.v1",
        "phase": "development_validation_selection",
        "arm": arm,
        "split_seed": split_seed,
        "initialization_seed": initialization_seed,
        "selection_best_epoch": outcome.best_epoch,
        "model_architecture": outcome.model.architecture,
        "model_state_dict": state,
        "model_state_sha256": _tensor_mapping_sha256(state),
        "frozen_parent_checkpoint_sha256": frozen.checkpoint_sha256,
        "frozen_parent_internal_state_sha256": frozen.model_state_sha256,
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
        raise SiteNCampaignError("Unregistered Stage-E-C job axis")
    selected_device = torch.device(device)
    if selected_device.type != "cuda" or not torch.cuda.is_available():
        raise SiteNCampaignError("Stage-E-C R2 matrix requires CUDA")
    config, config_file = read_config(config_path)
    dataset = _project_path(config["dataset"]["directory"], label="dataset.directory")
    dataset_verification = verify_dataset(dataset)
    train, validation, split_audit = _split_examples(
        dataset,
        split_seed=split_seed,
    )
    frozen = _load_frozen_parent(
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
        raise SiteNCampaignError(f"Existing Stage-E-C job is stale: {target}")
    if target.exists():
        raise SiteNCampaignError(f"Partial Stage-E-C job exists: {target}")
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
                "frozen_stage_e_b_n1_validation_metrics.json": (
                    outcome.frozen_parent_metrics
                ),
                "validation_paired_solvent_metrics.json": (
                    outcome.validation_pair_metrics
                ),
                "frozen_stage_e_b_n1_verification.json": frozen.verification,
                "freeze_audit.json": outcome.freeze_audit,
                "target_weight_audit.json": outcome.target_weight_audit,
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
                "schema_version": "nucpred.mayr-nextgen-stage-e-c-r2-job.v1",
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
                "epoch_zero_frozen_parent_was_selected": outcome.best_epoch == 0,
                "validation_metrics": dict(outcome.validation_metrics),
                "frozen_parent_validation_metrics": dict(outcome.frozen_parent_metrics),
                "freeze_audit": dict(outcome.freeze_audit),
                "gate_audit": dict(outcome.gate_audit),
                "test_examples_instantiated": 0,
                "test_predictions_computed": 0,
                "test_prediction_files_written": 0,
                "formal_calibration_performed": False,
                "absolute_probability_published": False,
                "final_refit_performed": False,
                "combined_arm_performed": False,
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
                        "combined_arm_performed=false",
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
                "schema_version": "nucpred.mayr-nextgen-stage-e-c-r2-failure.v1",
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
                        "event": "stage_e_c_r2_job_complete",
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
            "status": "pass",
            "parallel_workers": 0,
            "completed_job_count": 45,
        }
    gc.collect()
    gc.freeze()
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(
            name=f"mayr-stage-e-c-r2-{seed}",
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
                "event": "stage_e_c_r2_matrix_started",
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
                    "event": "stage_e_c_r2_matrix_heartbeat",
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
        raise SiteNCampaignError(f"Stage-E-C R2 workers failed: {failures}")
    if not all(path.is_file() for path in expected):
        raise SiteNCampaignError("Stage-E-C R2 summaries are incomplete")
    atomic_write_json(
        output_root / "summary.json",
        {
            "schema_version": "nucpred.mayr-nextgen-stage-e-c-r2-matrix.v1",
            "status": "pass",
            "campaign_id": config["campaign_id"],
            "completed_job_count": 45,
            "expected_job_count": 45,
            "parallel_worker_count": 3,
            "independent_arm_count": 3,
            "selection_roles": ["train", "validation"],
            "test_examples_instantiated": 0,
            "test_predictions_computed": 0,
            "formal_calibration_performed": False,
            "final_refit_performed": False,
            "combined_arm_performed": False,
            "d_n4_combination_performed": False,
            "dft_or_cdft_computation_performed": False,
        },
        ensure_ascii=False,
    )
    _write_manifest(output_root)
    return {
        "status": "pass",
        "parallel_workers": 3,
        "completed_job_count": 45,
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
