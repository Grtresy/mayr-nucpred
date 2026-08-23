"""Run stage-gated matched pretraining pilots for the typed site-N model."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
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
from nucpred.datasets.esnuel_d_node_xtb_pretraining import (
    verify_dataset as verify_esnuel_dataset,
)
from nucpred.project import get_project_layout
from nucpred.training.mayr_node_xtb_pretraining import (
    MaskingConfig,
)
from nucpred.training.mayr_node_xtb_scratch import SolventVocabulary
from nucpred.training.mayr_site_n import (
    MayrSiteNModel,
    load_site_n_examples,
    seed_everything,
)
from nucpred.training.mayr_site_n_pretraining import (
    SiteNPretrainingLossConfig,
    load_pretraining_examples,
    save_site_n_pretraining_checkpoint,
    train_site_n_pretraining,
    transfer_pretrained_backbone,
)

from .site_n import (
    CONFIG_SCHEMA,
    DEFAULT_CONFIG,
    EXPERIMENT_ID,
    SiteNCampaignError,
    _canonical_sha256,
    _device,
    _display_path,
    _read_config,
    _write_manifest,
)


_LAYOUT = get_project_layout()
ROOT = _LAYOUT.root
ALLOWED_SCOPES = ("pilot1024", "pilot4096")
ALLOWED_SEEDS = (31001, 31002, 31003)
EXPECTED_SCOPE_COUNTS = {
    "pilot1024": {
        "total": 1024,
        "train": 715,
        "validation": 176,
        "audit_test": 133,
    },
    "pilot4096": {
        "total": 4096,
        "train": 2861,
        "validation": 705,
        "audit_test": 530,
    },
}


def _source_id_sha256(values: Sequence[str]) -> str:
    return hashlib.sha256(
        ("\n".join(sorted(values)) + "\n").encode("utf-8")
    ).hexdigest()


def _pretraining_contract(
    *,
    config_path: Path,
    dataset_directory: Path,
    scope: str,
    seed: int,
) -> dict[str, object]:
    paths = {
        "config": config_path,
        "runner": Path(__file__).resolve(),
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
    }
    contract: dict[str, object] = {
        "schema_version": "nucpred.mayr-site-n-pretraining-contract.v1",
        "config_schema_version": CONFIG_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "scope": scope,
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
    }
    contract["contract_sha256"] = _canonical_sha256(contract)
    return contract


def _load_scope(
    config: Mapping[str, Any],
    *,
    scope: str,
) -> tuple[
    list,
    list,
    list,
    Path,
    dict[str, object],
    dict[str, object],
]:
    if scope not in ALLOWED_SCOPES:
        raise SiteNCampaignError(f"Unsupported pilot scope: {scope}")
    key = f"{scope}_directory"
    directory = (ROOT / str(config["pretraining"][key])).resolve()
    if not directory.is_dir():
        raise SiteNCampaignError(
            f"Pretraining dataset is not materialized: {_display_path(directory)}"
        )
    verification = verify_esnuel_dataset(directory)
    records = pd.read_parquet(directory / "records.parquet")
    atoms = pd.read_parquet(directory / "atom_features.parquet")
    molecules = pd.read_parquet(directory / "molecule_features.parquet")
    examples = load_pretraining_examples(records, atoms, molecules)
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
    if observed != EXPECTED_SCOPE_COUNTS[scope]:
        raise SiteNCampaignError(
            f"{scope} role counts changed: {observed}"
        )
    selection = {
        "schema_version": "nucpred.mayr-site-n-pretraining-selection.v1",
        "scope": scope,
        "counts": observed,
        "source_id_sha256": {
            role: _source_id_sha256(
                [example.source_id for example in values]
            )
            for role, values in roles.items()
        },
        "audit_test_used_for_selection": False,
    }
    return (
        roles["train"],
        roles["validation"],
        roles["audit_test"],
        directory,
        verification,
        selection,
    )


def _downstream_solvent_vocabulary(
    config: Mapping[str, Any],
) -> SolventVocabulary:
    directory = (ROOT / str(config["dataset_directory"])).resolve()
    examples = load_site_n_examples(directory)
    return SolventVocabulary.from_values(
        [example.solvent_raw for example in examples]
    )


def _masking(config: Mapping[str, Any]) -> MaskingConfig:
    section = config["pretraining"]["masking"]
    return MaskingConfig(
        node_categorical_probability=float(
            section["node_categorical_probability"]
        ),
        edge_categorical_probability=float(
            section["edge_categorical_probability"]
        ),
        local_probability=float(section["local_probability"]),
        global_probability=float(section["global_probability"]),
    )


def _loss_config(
    config: Mapping[str, Any],
) -> SiteNPretrainingLossConfig:
    section = config["pretraining"]["loss"]
    return SiteNPretrainingLossConfig(
        node_categorical_weight=float(
            section["node_categorical_weight"]
        ),
        edge_categorical_weight=float(
            section["edge_categorical_weight"]
        ),
        local_weight=float(section["local_weight"]),
        global_weight=float(section["global_weight"]),
        mca_weight=float(section["mca_weight"]),
        gcs_weight=float(section["gcs_weight"]),
        ranking_weight=float(section["ranking_weight"]),
        ranking_margin=float(section["ranking_margin"]),
    )


def _make_downstream(
    config: Mapping[str, Any],
    *,
    num_solvents: int,
    seed: int,
) -> MayrSiteNModel:
    seed_everything(seed)
    section = config["model"]
    return MayrSiteNModel(
        num_solvents=num_solvents,
        hidden_dim=int(section["hidden_dim"]),
        layers=int(section["message_passing_layers"]),
        node_embedding_dim=int(section["node_embedding_dim"]),
        edge_embedding_dim=int(section["edge_embedding_dim"]),
        solvent_embedding_dim=int(section["solvent_embedding_dim"]),
        dropout=float(section["dropout"]),
    )


def run_pretraining_pilot(
    *,
    scope: str,
    seed: int,
    config_path: str | Path = DEFAULT_CONFIG,
    device: str = "auto",
) -> dict[str, object]:
    started = time.perf_counter()
    if scope not in ALLOWED_SCOPES:
        raise SiteNCampaignError(f"Unsupported pilot scope: {scope}")
    if seed not in ALLOWED_SEEDS:
        raise SiteNCampaignError(f"Unregistered pretraining seed: {seed}")
    config_file = Path(config_path).resolve()
    config = _read_config(config_file)
    if list(map(int, config["pretraining"]["initialization_seeds"])) != list(
        ALLOWED_SEEDS
    ):
        raise SiteNCampaignError("Pretraining seed axis changed")
    selected_device = _device(device)
    (
        train,
        validation,
        audit_test,
        dataset_directory,
        verification,
        selection,
    ) = _load_scope(config, scope=scope)
    contract = _pretraining_contract(
        config_path=config_file,
        dataset_directory=dataset_directory,
        scope=scope,
        seed=seed,
    )
    output_root = (ROOT / str(config["output_root"])).resolve()
    target = output_root / "pretraining" / scope / f"seed-{seed}"
    summary_path = target / "summary.json"
    if summary_path.is_file():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            existing.get("status") == "pass"
            and existing.get("contract") == contract
        ):
            return existing
        raise SiteNCampaignError(
            f"Existing {scope}/seed-{seed} result is stale"
        )
    if target.exists():
        raise SiteNCampaignError(
            f"Partial {scope}/seed-{seed} directory exists"
        )
    vocabulary = _downstream_solvent_vocabulary(config)
    masking = _masking(config)
    loss_config = _loss_config(config)
    pretraining = config["pretraining"]
    epochs = int(pretraining[f"epochs_{scope}"])
    model_section = config["model"]
    result = train_site_n_pretraining(
        train,
        validation,
        audit_test_examples=audit_test,
        num_solvents=len(vocabulary.tokens),
        init_seed=seed,
        epochs=epochs,
        minimum_epochs=int(pretraining["minimum_epochs"]),
        patience=int(pretraining["early_stopping_patience"]),
        minimum_delta=float(
            pretraining["minimum_validation_delta"]
        ),
        batch_size=int(pretraining["batch_size"]),
        learning_rate=float(pretraining["learning_rate"]),
        weight_decay=float(pretraining["weight_decay"]),
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
    first_validation = float(result.history[0]["validation_total"])
    relative_improvement = (
        (first_validation - result.best_validation_total)
        / max(abs(first_validation), 1e-12)
    )
    audit_finite = bool(result.audit_metrics) and all(
        math.isfinite(float(value))
        for value in result.audit_metrics.values()
    )
    threshold = float(
        config["stage_gate"]["minimum_relative_validation_improvement"]
    )
    status = (
        "pass"
        if relative_improvement >= threshold and audit_finite
        else "fail"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".seed-{seed}.staging-",
            dir=target.parent,
        )
    )
    try:
        history = pd.DataFrame(result.history)
        history.to_csv(
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
            staging / "selection.json", selection, ensure_ascii=False
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
                "verification": verification,
                "manifest_sha256": sha256_file(
                    dataset_directory / "dataset_manifest.json"
                ),
            },
            selection=selection,
        )
        downstream = _make_downstream(
            config,
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
            "schema_version": "nucpred.mayr-site-n-pretraining-pilot.v1",
            "status": status,
            "experiment_id": EXPERIMENT_ID,
            "scope": scope,
            "initialization_seed": seed,
            "contract": contract,
            "dataset_verification": verification,
            "selection": selection,
            "device": str(selected_device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "epochs_requested": epochs,
            "epochs_completed": len(result.history),
            "best_epoch": result.best_epoch,
            "epoch1_validation_total": first_validation,
            "best_validation_total": result.best_validation_total,
            "relative_validation_improvement": relative_improvement,
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
        }
        atomic_write_json(
            staging / "summary.json", summary, ensure_ascii=False
        )
        (staging / "run.log").write_text(
            "\n".join(
                (
                    f"status={status}",
                    f"scope={scope}",
                    f"seed={seed}",
                    f"device={selected_device}",
                    f"epochs_completed={len(result.history)}",
                    f"best_epoch={result.best_epoch}",
                    (
                        "relative_validation_improvement="
                        f"{relative_improvement:.12f}"
                    ),
                    f"audit_metrics_finite={audit_finite}",
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
    if status != "pass":
        raise SiteNCampaignError(
            f"{scope}/seed-{seed} failed pilot gate: "
            f"relative improvement={relative_improvement:.3%}, "
            f"audit finite={audit_finite}"
        )
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=ALLOWED_SCOPES, required=True)
    parser.add_argument("--seed", type=int, choices=ALLOWED_SEEDS, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    payload = run_pretraining_pilot(
        scope=arguments.scope,
        seed=int(arguments.seed),
        config_path=arguments.config,
        device=arguments.device,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
