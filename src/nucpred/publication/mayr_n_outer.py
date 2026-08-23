"""Outer-development refit and frozen-checkpoint loading for Mayr N.

Outer-test target rows are intentionally outside this module's training path.
Each job reads only the target identifiers marked ``development`` for one
outer fold, then trains the preregistered C2 -> E-B-N1 -> E-C-N3 lineage for
the epoch counts selected by the four inner folds.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
import gc
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any

import pandas as pd

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from nucpred.core.files import atomic_write_json, sha256_file
from nucpred.experiments.mayr.nextgen_gate_a import _canonical_sha256
from nucpred.experiments.mayr.nextgen_stage_c_r2 import (
    _make_stage_c_model,
    _train_epoch as _train_c2_epoch,
)
from nucpred.experiments.mayr.nextgen_stage_e_a_r2 import (
    FrozenC2,
    _training_batches as _eb_pair_aware_batches,
)
from nucpred.experiments.mayr.nextgen_stage_e_b_r2 import (
    _make_model as _make_eb_model,
    _ordinary_batches as _eb_ordinary_batches,
    _train_epoch as _train_eb_epoch,
)
from nucpred.experiments.mayr.nextgen_stage_e_c_r2 import (
    FrozenEBN1,
    _make_model as _make_ec_model,
    _ordinary_batches as _ec_ordinary_batches,
    _train_epoch as _train_ec_epoch,
)
from nucpred.experiments.mayr.site_n import _iter_batches
from nucpred.experiments.mayr.site_n_formal import _tensor_mapping_sha256
from nucpred.publication.mayr_n_modeling import (
    DEFAULT_CONFIG,
    PublicationModelingError,
    _audit_splits,
    apply_input_ablation,
    _bound_inputs,
    _in_memory_c2,
    _in_memory_eb,
    _membership_tables,
    _pretraining_entry,
    _project_path,
    _read_json,
    _training_configs,
    read_config,
)
from nucpred.training.mayr_node_xtb_scratch import SolventVocabulary
from nucpred.training.mayr_site_n import (
    SiteNFoldPreprocessor,
    fit_site_n_preprocessor,
    load_site_n_examples,
)
from nucpred.training.mayr_site_n_stage_c import stage_c_target_weights
from nucpred.training.mayr_site_n_stage_d import (
    stage_d_paired_solvent_definitions,
)
from nucpred.training.mayr_site_n_stage_e_a import (
    stage_e_a_solvent_center_groups,
)
from nucpred.training.mayr_site_n_stage_e_b import E_B_N1
from nucpred.training.mayr_site_n_stage_e_c import E_C_N3


ROOT = Path(__file__).resolve().parents[3]
OUTER_CHECKPOINT_SCHEMA = "nucpred.mayr-n-publication-conditional-n-outer-checkpoint.v1"


def _development_ids(
    outer: pd.DataFrame, *, outer_fold: int
) -> tuple[set[str], dict[str, object]]:
    selected = outer.loc[outer["outer_fold"].eq(outer_fold)]
    development = selected.loc[selected["role"].eq("development")]
    test = selected.loc[selected["role"].eq("test")]
    development_ids = set(development["target_id"].astype(str))
    test_ids = set(test["target_id"].astype(str))
    development_connectivities = set(development["connectivity_id"].astype(str))
    test_connectivities = set(test["connectivity_id"].astype(str))
    if not development_ids or not test_ids:
        raise PublicationModelingError("Outer fold has an empty role")
    if development_ids & test_ids or development_connectivities & test_connectivities:
        raise PublicationModelingError("Outer development/test roles leak")
    return development_ids, {
        "schema_version": "nucpred.mayr-n-publication-outer-training-split-audit.v1",
        "outer_fold": outer_fold,
        "development_target_count": len(development_ids),
        "development_connectivity_count": len(development_connectivities),
        "outer_test_membership_target_count": len(test_ids),
        "outer_test_membership_connectivity_count": len(test_connectivities),
        "connectivity_disjoint": True,
        "outer_test_target_rows_loaded": 0,
    }


def _epoch_selection(
    config: Mapping[str, Any], *, outer_fold: int
) -> tuple[dict[str, Any], Path]:
    path = (
        _project_path(config["output_directory"], label="output directory")
        / "outer_epoch_selection"
        / f"outer-{outer_fold}.json"
    )
    payload = _read_json(path)
    if payload.get("status") != "frozen" or int(payload["outer_fold"]) != outer_fold:
        raise PublicationModelingError("Outer epoch selection is not frozen")
    if payload.get("rule") != "upper_median_of_four_inner_best_epochs":
        raise PublicationModelingError("Outer epoch rule changed")
    epochs = payload.get("selected_epochs")
    if not isinstance(epochs, Mapping) or set(epochs) != {
        "base_c2",
        "stage_e_b_n1",
        "stage_e_c_n3",
    }:
        raise PublicationModelingError("Selected epoch payload changed")
    if any(int(value) < 0 for value in epochs.values()):
        raise PublicationModelingError("Selected epochs cannot be negative")
    return payload, path


def _outer_contract(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    outer_fold: int,
    initialization_seed: int,
    development_ids: set[str],
    split_audit: Mapping[str, object],
    checkpoint_path: Path,
    selection_path: Path,
) -> dict[str, object]:
    sources = {
        "config": config_path,
        "runner": Path(__file__).resolve(),
        "modeling_runner": ROOT / "src/nucpred/publication/mayr_n_modeling.py",
        "dataset_manifest": _project_path(
            config["dataset"]["manifest_path"], label="dataset manifest"
        ),
        "outer_membership": _project_path(
            config["dataset"]["outer_membership_path"], label="outer membership"
        ),
        "nested_membership": _project_path(
            config["dataset"]["nested_membership_path"], label="nested membership"
        ),
        "epoch_selection": selection_path,
        "pretraining_checkpoint": checkpoint_path,
    }
    contract: dict[str, object] = {
        "schema_version": "nucpred.mayr-n-publication-outer-refit-contract.v1",
        "campaign_id": config["campaign_id"],
        "experiment_id": config["experiment_id"],
        "phase": "outer_development_refit",
        "outer_fold": outer_fold,
        "initialization_seed": initialization_seed,
        "architecture": [
            config["lineage"]["base_arm"],
            config["lineage"]["stage_e_b_arm"],
            config["lineage"]["stage_e_c_arm"],
        ],
        "ablation": dict(config["ablation"]) if "ablation" in config else None,
        "source_hashes": {name: sha256_file(path) for name, path in sources.items()},
        "development_target_id_sha256": _canonical_sha256(sorted(development_ids)),
        "development_target_count": len(development_ids),
        "split_audit": dict(split_audit),
        "outer_test_target_rows_loaded": 0,
        "outer_test_predictions_computed": 0,
        "outer_test_metrics_computed": 0,
        "sn_imported_or_predicted": False,
        "unknown_is_negative": False,
        "candidate_softmax_used": False,
    }
    contract["contract_sha256"] = _canonical_sha256(contract)
    return contract


def _train_base(
    development,
    *,
    config: Mapping[str, Any],
    base_config: Mapping[str, Any],
    stage_config: Mapping[str, Any],
    preprocessor: SiteNFoldPreprocessor,
    vocabulary: SolventVocabulary,
    initialization_seed: int,
    checkpoint: Path,
    epochs: int,
    device: torch.device,
) -> tuple[torch.nn.Module, pd.DataFrame, dict[str, object]]:
    model, base_hash, post_hash, transfer_audit = _make_stage_c_model(
        arm=str(config["lineage"]["base_arm"]),
        base_config=base_config,
        vocabulary=vocabulary,
        initialization_seed=initialization_seed,
        device=device,
        checkpoint=checkpoint,
    )
    optimization = stage_config["r2"]["optimization"]
    target_weights, weight_audit = stage_c_target_weights(
        development,
        use_h1=False,
        use_h2=True,
        maximum_weight=float(optimization["maximum_target_weight"]),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
    )
    rows: list[dict[str, object]] = []
    for epoch in range(1, epochs + 1):
        metrics = _train_c2_epoch(
            model,
            _iter_batches(
                development,
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
        rows.append({"epoch": epoch, **metrics})
    audit = {
        "base_initialization_sha256": base_hash,
        "post_transfer_initialization_sha256": post_hash,
        "transfer_audit": transfer_audit,
        "target_weight_audit": weight_audit,
    }
    return model, pd.DataFrame(rows), audit


def _train_eb(
    development,
    *,
    frozen: FrozenC2,
    config: Mapping[str, Any],
    initialization_seed: int,
    epochs: int,
    device: torch.device,
) -> tuple[torch.nn.Module, pd.DataFrame, dict[str, object]]:
    model = _make_eb_model(
        frozen,
        arm=E_B_N1,
        initialization_seed=initialization_seed,
        device=device,
    )
    parent_before = _tensor_mapping_sha256(model.frozen_base.state_dict())
    optimization = config["r2"]["optimization"]
    weights, weight_audit = stage_c_target_weights(
        development,
        use_h1=False,
        use_h2=True,
        maximum_weight=float(optimization["maximum_target_weight"]),
    )
    pairs, pair_audit = stage_d_paired_solvent_definitions(development)
    center_groups, center_audit = stage_e_a_solvent_center_groups(development)
    arm_config = config["r2"]["e_b_n1"]
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
    )
    rows: list[dict[str, object]] = []
    for epoch in range(1, epochs + 1):
        if bool(arm_config["pair_aware_batching"]):
            batches = _eb_pair_aware_batches(
                development,
                pairs=pairs,
                batch_size=int(optimization["batch_size_contexts"]),
                preprocessor=frozen.preprocessor,
                vocabulary=frozen.vocabulary,
                shuffle_seed=initialization_seed + epoch,
            )
        else:
            batches = _eb_ordinary_batches(
                development,
                frozen=frozen,
                batch_size=int(optimization["batch_size_contexts"]),
                shuffle_seed=initialization_seed + epoch,
            )
        metrics = _train_eb_epoch(
            model,
            batches,
            target_weights=weights,
            pairs=pairs,
            center_groups=center_groups,
            paired_weight=float(arm_config["paired_solvent_weight"]),
            center_weight=float(arm_config["center_penalty_weight"]),
            config=config,
            optimizer=optimizer,
            device=device,
        )
        rows.append({"epoch": epoch, **metrics})
    model.frozen_base.eval()
    parent_after = _tensor_mapping_sha256(model.frozen_base.state_dict())
    if parent_before != parent_after:
        raise PublicationModelingError("Frozen C2 changed during outer E-B refit")
    audit = {
        "parent_state_before_sha256": parent_before,
        "parent_state_after_sha256": parent_after,
        "parent_state_bitwise_unchanged": True,
        "target_weight_audit": weight_audit,
        "paired_solvent_audit": pair_audit,
        "center_group_audit": center_audit,
    }
    return model, pd.DataFrame(rows), audit


def _train_ec(
    development,
    *,
    frozen: FrozenEBN1,
    config: Mapping[str, Any],
    initialization_seed: int,
    epochs: int,
    device: torch.device,
) -> tuple[torch.nn.Module, pd.DataFrame, dict[str, object]]:
    model = _make_ec_model(
        frozen,
        arm=E_C_N3,
        initialization_seed=initialization_seed,
        device=device,
    )
    parent_before = _tensor_mapping_sha256(model.frozen_parent.state_dict())
    optimization = config["r2"]["optimization"]
    weights, weight_audit = stage_c_target_weights(
        development,
        use_h1=False,
        use_h2=True,
        maximum_weight=float(optimization["maximum_target_weight"]),
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
    )
    rows: list[dict[str, object]] = []
    for epoch in range(1, epochs + 1):
        metrics = _train_ec_epoch(
            model,
            _ec_ordinary_batches(
                development,
                frozen=frozen,
                batch_size=int(optimization["batch_size_contexts"]),
                shuffle_seed=initialization_seed + epoch,
            ),
            target_weights=weights,
            config=config,
            optimizer=optimizer,
            device=device,
        )
        rows.append({"epoch": epoch, **metrics})
    model.frozen_parent.eval()
    parent_after = _tensor_mapping_sha256(model.frozen_parent.state_dict())
    if parent_before != parent_after:
        raise PublicationModelingError("Frozen E-B-N1 changed during outer E-C refit")
    audit = {
        "parent_state_before_sha256": parent_before,
        "parent_state_after_sha256": parent_after,
        "parent_state_bitwise_unchanged": True,
        "target_weight_audit": weight_audit,
    }
    return model, pd.DataFrame(rows), audit


def _save_outer_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    preprocessor: SiteNFoldPreprocessor,
    vocabulary: SolventVocabulary,
    epochs: Mapping[str, int],
    contract: Mapping[str, object],
    stage_audits: Mapping[str, object],
) -> dict[str, object]:
    state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    payload: dict[str, object] = {
        "schema_version": OUTER_CHECKPOINT_SCHEMA,
        "phase": "outer_development_refit_frozen_before_test_prediction",
        "model_lineage": "pre-sN_C2_to_E-B-N1_to_E-C-N3",
        "model_architecture": model.architecture,
        "model_state_dict": state,
        "model_state_sha256": _tensor_mapping_sha256(state),
        "preprocessor": preprocessor.to_json(),
        "solvent_vocabulary": list(vocabulary.tokens),
        "trained_epochs": {name: int(value) for name, value in epochs.items()},
        "contract": dict(contract),
        "stage_audits": dict(stage_audits),
    }
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return payload


def run_outer_refit(
    *,
    outer_fold: int,
    initialization_seed: int,
    config_path: str | Path = DEFAULT_CONFIG,
    device: str | None = None,
) -> dict[str, object]:
    started = time.perf_counter()
    config, resolved = read_config(config_path)
    allowed_folds = range(int(config["outer_fold_count"]))
    allowed_seeds = tuple(map(int, config["outer_initialization_seeds"]))
    if outer_fold not in allowed_folds or initialization_seed not in allowed_seeds:
        raise PublicationModelingError("Unregistered outer refit axis")
    selected_device = torch.device(device or str(config["device"]))
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise PublicationModelingError("CUDA was requested but is unavailable")
    _bound_inputs(config, resolved)
    outer, nested = _membership_tables(config)
    _audit_splits(config, outer, nested)
    development_ids, split_audit = _development_ids(outer, outer_fold=outer_fold)
    selection, selection_path = _epoch_selection(config, outer_fold=outer_fold)
    epochs = {key: int(value) for key, value in selection["selected_epochs"].items()}
    entry, pretraining_checkpoint, checkpoint_audit = _pretraining_entry(
        config, initialization_seed
    )
    contract = _outer_contract(
        config=config,
        config_path=resolved,
        outer_fold=outer_fold,
        initialization_seed=initialization_seed,
        development_ids=development_ids,
        split_audit=split_audit,
        checkpoint_path=pretraining_checkpoint,
        selection_path=selection_path,
    )
    target = (
        _project_path(config["output_directory"], label="output directory")
        / "outer_refit"
        / f"outer-{outer_fold}"
        / f"init-{initialization_seed}"
    )
    if (target / "summary.json").is_file():
        existing = _read_json(target / "summary.json")
        if existing.get("status") == "pass" and existing.get("contract") == contract:
            return existing
        raise PublicationModelingError(f"Existing outer refit is stale: {target}")
    if target.exists():
        raise PublicationModelingError(f"Partial outer refit exists: {target}")
    dataset = _project_path(config["dataset"]["directory"], label="dataset")
    development = apply_input_ablation(
        load_site_n_examples(dataset, target_ids=development_ids), config
    )
    if sum(item.num_sites for item in development) != len(development_ids):
        raise PublicationModelingError("Outer development target count changed")
    preprocessor = fit_site_n_preprocessor(development)
    vocabulary = SolventVocabulary.from_values(
        [example.solvent_raw for example in development]
    )
    base_config, stage_c_config, stage_eb_config, stage_ec_config = _training_configs(
        config
    )
    try:
        c2_model, c2_curves, c2_audit = _train_base(
            development,
            config=config,
            base_config=base_config,
            stage_config=stage_c_config,
            preprocessor=preprocessor,
            vocabulary=vocabulary,
            initialization_seed=initialization_seed,
            checkpoint=pretraining_checkpoint,
            epochs=epochs["base_c2"],
            device=selected_device,
        )
        frozen_c2 = _in_memory_c2(c2_model, preprocessor, vocabulary)
        eb_model, eb_curves, eb_audit = _train_eb(
            development,
            frozen=frozen_c2,
            config=stage_eb_config,
            initialization_seed=initialization_seed,
            epochs=epochs["stage_e_b_n1"],
            device=selected_device,
        )
        frozen_eb = _in_memory_eb(eb_model, preprocessor, vocabulary)
        ec_model, ec_curves, ec_audit = _train_ec(
            development,
            frozen=frozen_eb,
            config=stage_ec_config,
            initialization_seed=initialization_seed,
            epochs=epochs["stage_e_c_n3"],
            device=selected_device,
        )
        stage_audits = {
            "base_c2": c2_audit,
            "stage_e_b_n1": eb_audit,
            "stage_e_c_n3": ec_audit,
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".init-{initialization_seed}.staging-", dir=target.parent
            )
        )
        try:
            for name, curves in (
                ("base_c2", c2_curves),
                ("stage_e_b_n1", eb_curves),
                ("stage_e_c_n3", ec_curves),
            ):
                curves.to_csv(
                    staging / f"{name}_training_curves.csv",
                    index=False,
                    lineterminator="\n",
                )
            payload = _save_outer_checkpoint(
                staging / "model.pt",
                model=ec_model,
                preprocessor=preprocessor,
                vocabulary=vocabulary,
                epochs=epochs,
                contract=contract,
                stage_audits=stage_audits,
            )
            summary: dict[str, object] = {
                "schema_version": "nucpred.mayr-n-publication-outer-refit-job.v1",
                "status": "pass",
                "campaign_id": config["campaign_id"],
                "experiment_id": config["experiment_id"],
                "outer_fold": outer_fold,
                "initialization_seed": initialization_seed,
                "pretraining_seed": int(entry["pretraining_seed"]),
                "pretraining_checkpoint_audit": checkpoint_audit,
                "contract": contract,
                "trained_epochs": epochs,
                "development_context_count": len(development),
                "development_target_count": len(development_ids),
                "model_state_sha256": payload["model_state_sha256"],
                "model_checkpoint_sha256": None,
                "outer_test_target_rows_loaded": 0,
                "outer_test_predictions_computed": 0,
                "outer_test_metrics_computed": 0,
                "sn_imported_or_predicted": False,
                "device": str(selected_device),
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "wall_seconds": time.perf_counter() - started,
                "completed_at_utc": datetime.now(UTC).isoformat(),
            }
            summary["model_checkpoint_sha256"] = sha256_file(staging / "model.pt")
            atomic_write_json(staging / "summary.json", summary, ensure_ascii=False)
            os.replace(staging, target)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return summary
    finally:
        gc.collect()
        if selected_device.type == "cuda":
            torch.cuda.empty_cache()


def load_outer_checkpoint(
    checkpoint: str | Path,
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    device: str | torch.device = "cpu",
) -> tuple[torch.nn.Module, SiteNFoldPreprocessor, SolventVocabulary, dict[str, Any]]:
    path = Path(checkpoint).resolve()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != OUTER_CHECKPOINT_SCHEMA:
        raise PublicationModelingError("Outer checkpoint schema changed")
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise PublicationModelingError("Outer checkpoint lacks model state")
    if _tensor_mapping_sha256(state) != payload.get("model_state_sha256"):
        raise PublicationModelingError("Outer checkpoint model-state hash changed")
    contract = payload.get("contract")
    if not isinstance(contract, Mapping):
        raise PublicationModelingError("Outer checkpoint lacks a contract")
    config, _ = read_config(config_path)
    initialization_seed = int(contract["initialization_seed"])
    _, pretraining_checkpoint, _ = _pretraining_entry(config, initialization_seed)
    preprocessor = SiteNFoldPreprocessor.from_json(payload["preprocessor"])
    vocabulary = SolventVocabulary(tuple(map(str, payload["solvent_vocabulary"])))
    base_config, _, _, _ = _training_configs(config)
    selected_device = torch.device(device)
    base, _, _, _ = _make_stage_c_model(
        arm=str(config["lineage"]["base_arm"]),
        base_config=base_config,
        vocabulary=vocabulary,
        initialization_seed=initialization_seed,
        device=selected_device,
        checkpoint=pretraining_checkpoint,
    )
    frozen_c2 = _in_memory_c2(base, preprocessor, vocabulary)
    eb = _make_eb_model(
        frozen_c2,
        arm=E_B_N1,
        initialization_seed=initialization_seed,
        device=selected_device,
    )
    frozen_eb = _in_memory_eb(eb, preprocessor, vocabulary)
    model = _make_ec_model(
        frozen_eb,
        arm=E_C_N3,
        initialization_seed=initialization_seed,
        device=selected_device,
    )
    model.load_state_dict(state, strict=True)
    if _tensor_mapping_sha256(model.state_dict()) != payload["model_state_sha256"]:
        raise PublicationModelingError("Exact outer checkpoint load failed")
    model.eval()
    return model, preprocessor, vocabulary, payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--outer-fold", type=int, required=True)
    parser.add_argument("--initialization-seed", type=int, required=True)
    parser.add_argument("--device")
    args = parser.parse_args(argv)
    result = run_outer_refit(
        outer_fold=args.outer_fold,
        initialization_seed=args.initialization_seed,
        config_path=args.config,
        device=args.device,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
