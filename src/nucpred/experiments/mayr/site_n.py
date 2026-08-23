"""Stage-gated training workflow for independent, typed Mayr site-N outputs."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
import tomllib
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from nucpred.core.files import atomic_write_json, sha256_file
from nucpred.datasets.mayr_site_n import verify_dataset
from nucpred.project import get_project_layout
from nucpred.training.mayr_node_xtb_scratch import (
    SolventVocabulary,
    initialization_sha256,
)
from nucpred.training.mayr_site_n import (
    SITE_TYPE_NAMES,
    MayrSiteNModel,
    SiteNExample,
    SiteNFoldPreprocessor,
    SiteNTrainingBatch,
    fit_site_n_preprocessor,
    load_site_n_examples,
    pack_site_n_batch,
    seed_everything,
    site_n_loss,
)


_LAYOUT = get_project_layout()
ROOT = _LAYOUT.root
DEFAULT_CONFIG = ROOT / "configs/mayr_site_n_experiment.toml"
CONFIG_SCHEMA = "nucpred.mayr-site-n-experiment-config.v1"
EXPERIMENT_ID = "mayr-site-n-independent-20260726-v1"


class SiteNCampaignError(RuntimeError):
    """Raised when a frozen site-N training contract is violated."""


@dataclass(frozen=True, slots=True)
class FitOutcome:
    model: MayrSiteNModel
    initialization_sha256: str
    curves: pd.DataFrame


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = Path(path).resolve()
    payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != CONFIG_SCHEMA:
        raise SiteNCampaignError("Unsupported site-N experiment config")
    if payload.get("experiment_id") != EXPERIMENT_ID:
        raise SiteNCampaignError("Site-N experiment identity changed")
    if payload.get("dataset_id") != "mayr-site-n-20260726-v1":
        raise SiteNCampaignError("Site-N dataset identity changed")
    if int(payload.get("maximum_parallel_gpu_processes", 0)) != 3:
        raise SiteNCampaignError("GPU concurrency contract changed")
    if not bool(payload.get("stage_gate_required_before_full_pretraining")):
        raise SiteNCampaignError("Mandatory full-pretraining gate was disabled")
    if bool(payload.get("site_probability_normalization")):
        raise SiteNCampaignError("Site softmax is forbidden")
    if bool(payload.get("unmeasured_candidates_are_negative")):
        raise SiteNCampaignError("Unmeasured candidates cannot be negatives")
    model = payload["model"]
    if (
        tuple(model["site_types"]) != SITE_TYPE_NAMES
        or model.get("site_output") != "independent_scalar_N_per_query"
        or model.get("hydrogen_policy")
        != "ordinary_element_in_shared_vocabulary"
    ):
        raise SiteNCampaignError("Typed site-query architecture changed")
    return payload


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected = torch.device(value)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise SiteNCampaignError("CUDA was requested but is unavailable")
    return selected


def _source_contract(
    *,
    config_path: Path,
    dataset_directory: Path,
) -> dict[str, object]:
    paths = {
        "config": config_path,
        "runner": Path(__file__).resolve(),
        "model": (
            ROOT / "src/nucpred/training/mayr_site_n.py"
        ).resolve(),
        "dataset_builder": (
            ROOT / "src/nucpred/datasets/mayr_site_n.py"
        ).resolve(),
        "all_atom_graph": (
            ROOT / "src/nucpred/features/all_atom_graph.py"
        ).resolve(),
        "dataset_manifest": dataset_directory / "dataset_manifest.json",
    }
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    contract: dict[str, object] = {
        "schema_version": "nucpred.mayr-site-n-run-contract.v1",
        "experiment_id": EXPERIMENT_ID,
        "dataset_id": "mayr-site-n-20260726-v1",
        "source_hashes": hashes,
        "site_probability_normalization": False,
        "site_types": list(SITE_TYPE_NAMES),
        "maximum_parallel_gpu_processes": 3,
    }
    contract["contract_sha256"] = _canonical_sha256(contract)
    return contract


def _make_model(
    config: Mapping[str, Any],
    vocabulary: SolventVocabulary,
    *,
    seed: int,
    device: torch.device,
) -> tuple[MayrSiteNModel, str]:
    seed_everything(seed)
    section = config["model"]
    model = MayrSiteNModel(
        num_solvents=len(vocabulary.tokens),
        hidden_dim=int(section["hidden_dim"]),
        layers=int(section["message_passing_layers"]),
        node_embedding_dim=int(section["node_embedding_dim"]),
        edge_embedding_dim=int(section["edge_embedding_dim"]),
        solvent_embedding_dim=int(section["solvent_embedding_dim"]),
        dropout=float(section["dropout"]),
    )
    digest = initialization_sha256(model)
    return model.to(device), digest


def _iter_batches(
    examples: Sequence[SiteNExample],
    *,
    batch_size: int,
    preprocessor: SiteNFoldPreprocessor,
    vocabulary: SolventVocabulary,
    shuffle_seed: int | None,
) -> list[SiteNTrainingBatch]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    indices = np.arange(len(examples))
    if shuffle_seed is not None:
        np.random.default_rng(int(shuffle_seed)).shuffle(indices)
    return [
        pack_site_n_batch(
            [
                examples[int(index)]
                for index in indices[start : start + batch_size]
            ],
            preprocessor=preprocessor,
            solvent_vocabulary=vocabulary,
        )
        for start in range(0, len(indices), batch_size)
    ]


def _train_epoch(
    model: MayrSiteNModel,
    batches: Sequence[SiteNTrainingBatch],
    *,
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
        total, parts = site_n_loss(
            output,
            batch,
            ranking_weight=ranking_weight,
        )
        if not bool(torch.isfinite(total)):
            raise SiteNCampaignError("Training loss became non-finite")
        total.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(gradient_clip_norm)
        )
        optimizer.step()
        count = batch.inputs.num_sites
        target_count += count
        for name in totals:
            value = total if name == "total" else parts[name]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Unexpected loss component {name}")
            totals[name] += float(value.detach().cpu()) * count
        ranking_pairs += int(parts["ranking_pairs"])
    if not target_count:
        raise SiteNCampaignError("Training epoch had no targets")
    result = {
        name: value / target_count for name, value in totals.items()
    }
    result["ranking_pairs"] = float(ranking_pairs)
    return result


def _regression_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float]:
    residual = prediction - target
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual**2)))
    denominator = float(np.sum((target - target.mean()) ** 2))
    r2 = (
        float(1.0 - np.sum(residual**2) / denominator)
        if denominator > 0.0
        else math.nan
    )
    return {"mae": mae, "rmse": rmse, "r2": r2}


def _evaluate(
    model: MayrSiteNModel,
    examples: Sequence[SiteNExample],
    *,
    preprocessor: SiteNFoldPreprocessor,
    vocabulary: SolventVocabulary,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, object], pd.DataFrame]:
    model.eval()
    metadata = {
        target_id: {
            "site_type": site_type,
            "site_members": members,
            "context_id": example.context_id,
        }
        for example in examples
        for target_id, site_type, members in zip(
            example.target_ids,
            example.site_types,
            example.site_members,
            strict=True,
        )
    }
    rows: list[dict[str, object]] = []
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
            predictions = (
                output.n_prediction_standardized.detach().cpu().numpy()
                * preprocessor.target_scale
                + preprocessor.target_mean
            )
            targets = batch.n_target_raw.detach().cpu().numpy()
            for index, target_id in enumerate(batch.target_ids):
                item = metadata[target_id]
                rows.append(
                    {
                        "target_id": target_id,
                        "context_id": item["context_id"],
                        "site_object_id": batch.site_object_ids[index],
                        "site_type": item["site_type"],
                        "member_atom_indices_json": json.dumps(
                            item["site_members"], separators=(",", ":")
                        ),
                        "N_true": float(targets[index]),
                        "N_pred": float(predictions[index]),
                        "absolute_error": abs(
                            float(predictions[index]) - float(targets[index])
                        ),
                    }
                )
    frame = pd.DataFrame(rows).sort_values("target_id").reset_index(drop=True)
    metrics: dict[str, object] = {
        **_regression_metrics(
            frame["N_true"].to_numpy(dtype=float),
            frame["N_pred"].to_numpy(dtype=float),
        ),
        "target_count": int(len(frame)),
        "context_count": int(frame["context_id"].nunique()),
    }
    metrics["by_site_type"] = {
        str(site_type): {
            **_regression_metrics(
                group["N_true"].to_numpy(dtype=float),
                group["N_pred"].to_numpy(dtype=float),
            ),
            "target_count": int(len(group)),
        }
        for site_type, group in frame.groupby("site_type", sort=True)
    }
    deltas: list[tuple[float, float]] = []
    correct = 0
    pair_count = 0
    for _, group in frame.groupby("context_id", sort=True):
        if len(group) < 2:
            continue
        values = group.reset_index(drop=True)
        for left in range(len(values)):
            for right in range(left + 1, len(values)):
                true_delta = float(
                    values.loc[left, "N_true"] - values.loc[right, "N_true"]
                )
                pred_delta = float(
                    values.loc[left, "N_pred"] - values.loc[right, "N_pred"]
                )
                if true_delta == 0:
                    continue
                deltas.append((true_delta, pred_delta))
                pair_count += 1
                correct += int(np.sign(true_delta) == np.sign(pred_delta))
    metrics["multi_site"] = {
        "pair_count": pair_count,
        "delta_mae": (
            float(np.mean([abs(pred - true) for true, pred in deltas]))
            if deltas
            else math.nan
        ),
        "ranking_accuracy": correct / pair_count if pair_count else math.nan,
    }
    return metrics, frame


def _select_tiny(
    examples: Sequence[SiteNExample],
    *,
    count: int,
) -> list[SiteNExample]:
    if count < len(SITE_TYPE_NAMES):
        raise ValueError("Tiny gate must cover every site type")
    ordered = sorted(examples, key=lambda example: example.context_id)
    selected: list[SiteNExample] = []

    def add(example: SiteNExample) -> None:
        if example not in selected:
            selected.append(example)

    for example in ordered:
        if example.num_sites > 1:
            add(example)
    for site_type in SITE_TYPE_NAMES:
        add(next(item for item in ordered if site_type in item.site_types))
    remaining = [item for item in ordered if item not in selected]
    remaining.sort(
        key=lambda item: (
            float(np.mean(item.n_targets)),
            item.context_id,
        )
    )
    needed = count - len(selected)
    if needed < 0:
        return selected[:count]
    if needed:
        positions = np.linspace(0, len(remaining) - 1, needed, dtype=int)
        for position in positions:
            add(remaining[int(position)])
    if len(selected) != count:
        raise SiteNCampaignError("Tiny selection did not reach exact size")
    return selected


def _fit_fixed(
    examples: Sequence[SiteNExample],
    *,
    preprocessor: SiteNFoldPreprocessor,
    vocabulary: SolventVocabulary,
    config: Mapping[str, Any],
    seed: int,
    device: torch.device,
) -> FitOutcome:
    model, init_hash = _make_model(
        config,
        vocabulary,
        seed=seed,
        device=device,
    )
    gate = config["tiny_overfit"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(gate["learning_rate"]),
        weight_decay=float(gate["weight_decay"]),
    )
    rows: list[dict[str, float | int]] = []
    for epoch in range(1, int(gate["epochs"]) + 1):
        metrics = _train_epoch(
            model,
            _iter_batches(
                examples,
                batch_size=int(gate["batch_size_contexts"]),
                preprocessor=preprocessor,
                vocabulary=vocabulary,
                shuffle_seed=seed + epoch,
            ),
            optimizer=optimizer,
            device=device,
            ranking_weight=float(config["optimization"]["ranking_weight"]),
            gradient_clip_norm=float(
                config["optimization"]["gradient_clip_norm"]
            ),
        )
        rows.append({"epoch": epoch, **metrics})
    return FitOutcome(
        model=model,
        initialization_sha256=init_hash,
        curves=pd.DataFrame(rows),
    )


def _write_manifest(directory: Path) -> dict[str, object]:
    files = {
        path.relative_to(directory).as_posix(): {
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != "run_manifest.json"
    }
    manifest: dict[str, object] = {
        "schema_version": "nucpred.mayr-site-n-run-manifest.v1",
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(int(item["bytes"]) for item in files.values()),
    }
    atomic_write_json(directory / "run_manifest.json", manifest)
    return manifest


def run_tiny_overfit(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    device: str = "auto",
) -> dict[str, object]:
    started = time.perf_counter()
    config_file = Path(config_path).resolve()
    config = _read_config(config_file)
    selected_device = _device(device)
    dataset_directory = (ROOT / str(config["dataset_directory"])).resolve()
    verification = verify_dataset(dataset_directory)
    contract = _source_contract(
        config_path=config_file,
        dataset_directory=dataset_directory,
    )
    output_root = (ROOT / str(config["output_root"])).resolve()
    target = output_root / "tiny_overfit"
    summary_path = target / "gate_summary.json"
    if summary_path.is_file():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            existing.get("status") == "pass"
            and existing.get("contract") == contract
        ):
            return existing
        raise SiteNCampaignError("Existing tiny-overfit gate is stale")
    if target.exists():
        raise SiteNCampaignError("Partial tiny-overfit directory exists")

    examples = load_site_n_examples(dataset_directory)
    vocabulary = SolventVocabulary.from_values(
        [example.solvent_raw for example in examples]
    )
    tiny = _select_tiny(
        examples,
        count=int(config["tiny_overfit"]["context_count"]),
    )
    preprocessor = fit_site_n_preprocessor(tiny)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".tiny-overfit.staging-", dir=target.parent)
    )
    try:
        outcome = _fit_fixed(
            tiny,
            preprocessor=preprocessor,
            vocabulary=vocabulary,
            config=config,
            seed=int(config["tiny_overfit"]["initialization_seed"]),
            device=selected_device,
        )
        metrics, predictions = _evaluate(
            outcome.model,
            tiny,
            preprocessor=preprocessor,
            vocabulary=vocabulary,
            batch_size=len(tiny),
            device=selected_device,
        )
        threshold = float(config["tiny_overfit"]["maximum_N_MAE"])
        status = "pass" if float(metrics["mae"]) <= threshold else "fail"
        outcome.curves.to_csv(
            staging / "loss_curves.csv",
            index=False,
            lineterminator="\n",
        )
        predictions.to_parquet(
            staging / "predictions.parquet",
            index=False,
            engine="pyarrow",
            compression="zstd",
        )
        atomic_write_json(
            staging / "preprocessor.json",
            preprocessor.to_json(),
            ensure_ascii=False,
        )
        atomic_write_json(
            staging / "selection.json",
            {
                "schema_version": "nucpred.mayr-site-n-tiny-selection.v1",
                "context_ids": [example.context_id for example in tiny],
                "target_ids": [
                    target_id
                    for example in tiny
                    for target_id in example.target_ids
                ],
                "site_type_target_counts": {
                    site_type: sum(
                        example.site_types.count(site_type)
                        for example in tiny
                    )
                    for site_type in SITE_TYPE_NAMES
                },
                "multi_site_context_count": sum(
                    example.num_sites > 1 for example in tiny
                ),
            },
            ensure_ascii=False,
        )
        final_state_hash = initialization_sha256(outcome.model)
        torch.save(
            {
                "schema_version": "nucpred.mayr-site-n-checkpoint.v1",
                "phase": "tiny_overfit",
                "model_architecture": outcome.model.architecture,
                "model_state_dict": {
                    name: tensor.detach().cpu()
                    for name, tensor in outcome.model.state_dict().items()
                },
                "initialization_sha256": outcome.initialization_sha256,
                "final_state_sha256": final_state_hash,
                "preprocessor": preprocessor.to_json(),
                "solvent_vocabulary": list(vocabulary.tokens),
                "contract": contract,
            },
            staging / "checkpoint.pt",
        )
        summary: dict[str, object] = {
            "schema_version": "nucpred.mayr-site-n-tiny-gate.v1",
            "status": status,
            "experiment_id": EXPERIMENT_ID,
            "contract": contract,
            "dataset_verification": verification,
            "device": str(selected_device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "context_count": len(tiny),
            "target_count": sum(example.num_sites for example in tiny),
            "metrics": metrics,
            "maximum_N_MAE": threshold,
            "initialization_sha256": outcome.initialization_sha256,
            "final_state_sha256": final_state_hash,
            "parameter_count": sum(
                parameter.numel()
                for parameter in outcome.model.parameters()
            ),
            "site_probability_normalization": False,
            "model_has_site_head": hasattr(outcome.model, "site_head"),
            "wall_seconds": time.perf_counter() - started,
        }
        atomic_write_json(
            staging / "gate_summary.json", summary, ensure_ascii=False
        )
        (staging / "run.log").write_text(
            "\n".join(
                (
                    f"status={status}",
                    f"device={selected_device}",
                    f"context_count={len(tiny)}",
                    f"target_count={sum(item.num_sites for item in tiny)}",
                    f"N_MAE={float(metrics['mae']):.12f}",
                    f"threshold={threshold:.12f}",
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
            f"Tiny-overfit N MAE {float(metrics['mae']):.4f} "
            f"exceeded {threshold:.4f}"
        )
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    tiny = commands.add_parser("tiny-overfit")
    tiny.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    tiny.add_argument("--device", default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    payload = run_tiny_overfit(
        config_path=arguments.config,
        device=arguments.device,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
