"""Leak-free nested training for the publication-scoped Mayr N model.

The runner deliberately separates four operations that are easy to conflate:

1. target-blind preflight and split auditing;
2. inner-fold epoch selection without loading an outer-test target row;
3. outer-development refitting with the selected epoch counts; and
4. post-freeze prediction/evaluation (implemented by later commands).

This module reuses the frozen pre-sN C2 -> E-B-N1 -> E-C-N3 architecture.  It
does not import or execute any sN, unknown-as-negative, or candidate-softmax
path.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime
import gc
import json
import os
from pathlib import Path
import shutil
import statistics
import tempfile
import time
import tomllib
from typing import Any

import pandas as pd
import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from nucpred.core.files import atomic_write_json, sha256_file
from nucpred.datasets.mayr_site_n_v2 import verify_dataset
from nucpred.experiments.mayr.nextgen_gate_a import _canonical_sha256
from nucpred.experiments.mayr.nextgen_stage_c_r2 import (
    _fit_selection as _fit_c2_selection,
)
from nucpred.experiments.mayr.nextgen_stage_e_a_r2 import FrozenC2
from nucpred.experiments.mayr.nextgen_stage_e_b_r2 import (
    _fit_selection as _fit_eb_selection,
)
from nucpred.experiments.mayr.nextgen_stage_e_c_r2 import (
    FrozenEBN1,
    _fit_selection as _fit_ec_selection,
)
from nucpred.experiments.mayr.site_n import SiteNCampaignError
from nucpred.experiments.mayr.site_n_formal import _tensor_mapping_sha256
from nucpred.project import get_project_layout
from nucpred.training.mayr_node_xtb_pretraining import (
    load_pretraining_checkpoint,
)
from nucpred.training.mayr_node_xtb_scratch import SolventVocabulary
from nucpred.training.mayr_site_n import (
    SiteNExample,
    fit_site_n_preprocessor,
    load_site_n_examples,
)
from nucpred.training.mayr_site_n_stage_e_b import E_B_N1
from nucpred.training.mayr_site_n_stage_e_c import E_C_N3


ROOT = get_project_layout().root
DEFAULT_CONFIG = ROOT / "configs/mayr_n_publication_experiment_v1.toml"
CONFIG_SCHEMA = "nucpred.mayr-n-publication-experiment-config.v1"
ABLATION_CONFIG_SCHEMA = "nucpred.mayr-n-publication-ablation-config.v1"
CHECKPOINT_SCHEMA = "nucpred.mayr-n-publication-conditional-n-checkpoint.v1"
EXPECTED_PRETRAINING_TASKS = frozenset(
    {
        "masked_node_categorical_reconstruction_all_atoms",
        "masked_edge_categorical_reconstruction",
        "masked_local4_reconstruction_all_atoms",
        "masked_global6_reconstruction",
        "heavy_atom_mca_scalar",
        "heavy_atom_gcs_53d",
        "heavy_atom_mca_soft_site",
        "heavy_atom_mca_within_molecule_ranking",
    }
)
ABLATION_NAMES = frozenset(
    {
        "without_xtb",
        "without_solvent",
        "without_site_type",
        "no_local_electronic",
        "no_global_electronic",
        "no_formal_charge",
        "no_pretraining",
    }
)


class PublicationModelingError(SiteNCampaignError):
    """Raised when publication modeling would violate its frozen contract."""


def _project_path(value: object, *, label: str) -> Path:
    path = (ROOT / str(value)).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as exc:
        raise PublicationModelingError(f"{label} escapes the project root") from exc
    return path


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PublicationModelingError(f"Expected JSON object: {path}")
    return payload


def _verify_bound_file(path: Path, expected: object, *, label: str) -> str:
    if not path.is_file():
        raise PublicationModelingError(f"Missing {label}: {path}")
    observed = sha256_file(path)
    if observed != str(expected):
        raise PublicationModelingError(
            f"Frozen {label} drifted: {observed} != {expected}"
        )
    return observed


def read_config(
    path: str | Path = DEFAULT_CONFIG,
) -> tuple[dict[str, Any], Path]:
    resolved = Path(path).resolve()
    raw = tomllib.loads(resolved.read_text(encoding="utf-8"))
    if raw.get("schema_version") == ABLATION_CONFIG_SCHEMA:
        parent_path = _project_path(
            raw["parent_config_path"], label="parent_config_path"
        )
        observed_parent_hash = sha256_file(parent_path)
        if observed_parent_hash != str(raw["parent_config_sha256"]):
            raise PublicationModelingError("Ablation parent config drifted")
        config = tomllib.loads(parent_path.read_text(encoding="utf-8"))
        for key in ("campaign_id", "experiment_id", "output_directory", "device"):
            if key in raw:
                config[key] = raw[key]
        config["ablation"] = dict(raw["ablation"])
        if "pretraining" in raw:
            config["pretraining"] = deepcopy(raw["pretraining"])
        for key, value in raw.get("lineage_overrides", {}).items():
            config["lineage"][key] = value
        config["_ablation_parent_config_path"] = parent_path.relative_to(
            ROOT
        ).as_posix()
        config["_ablation_parent_config_sha256"] = observed_parent_hash
    else:
        config = raw
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise PublicationModelingError("Unsupported publication experiment schema")
    if config.get("predict_sn") is not False:
        raise PublicationModelingError("Publication experiment must remain N-only")
    if config.get("unknown_is_negative") is not False:
        raise PublicationModelingError("Unknown candidates cannot become negatives")
    if config.get("candidate_softmax_used") is not False:
        raise PublicationModelingError("Candidate softmax is forbidden")
    if config.get("test_used_for_selection") is not False:
        raise PublicationModelingError("Outer test cannot select a model")
    if config.get("outer_epoch_rule") != "upper_median_of_four_inner_best_epochs":
        raise PublicationModelingError("Outer epoch-selection rule changed")
    if int(config.get("outer_fold_count", -1)) != 5:
        raise PublicationModelingError("Outer fold count changed")
    if int(config.get("inner_fold_count", -1)) != 4:
        raise PublicationModelingError("Inner fold count changed")
    if config["lineage"].get("pretraining_mayr_labels_used") is not False:
        raise PublicationModelingError("Mayr labels cannot enter pretraining")
    phases = config["phase_separation"]
    required_false = (
        "inner_jobs_may_read_outer_test_targets",
        "outer_training_may_read_outer_test_targets",
    )
    if any(phases.get(key) is not False for key in required_false):
        raise PublicationModelingError("Outer-test isolation changed")
    required_true = (
        "outer_test_prediction_requires_frozen_checkpoint",
        "oracle_site_prediction_is_diagnostic_only",
        "automatic_candidate_prediction_is_primary",
        "outer_test_metrics_after_prediction_freeze_only",
        "final_refit_after_outer_evaluation",
        "external_search_after_final_registry_freeze_only",
    )
    if any(phases.get(key) is not True for key in required_true):
        raise PublicationModelingError("Publication phase ordering changed")
    ablation = config.get("ablation")
    if ablation is not None:
        if (
            not isinstance(ablation, Mapping)
            or ablation.get("name") not in ABLATION_NAMES
        ):
            raise PublicationModelingError("Unsupported publication ablation")
        if ablation.get("outer_test_used_for_selection") is not False:
            raise PublicationModelingError("Ablation permits outer-test selection")
        if ablation.get("same_architecture_and_optimization") is not True:
            raise PublicationModelingError("Ablation architecture/optimization changed")
    return config, resolved


def apply_input_ablation(
    examples: Sequence[SiteNExample],
    config: Mapping[str, Any],
) -> list[SiteNExample]:
    """Apply one declared feature-family removal without touching target values."""

    ablation = config.get("ablation")
    if ablation is None:
        return list(examples)
    name = str(ablation["name"])
    transformed: list[SiteNExample] = []
    for example in examples:
        if name == "without_xtb":
            transformed.append(
                replace(
                    example,
                    local_values=np.zeros_like(example.local_values, dtype=float),
                    local_mask=np.zeros_like(example.local_mask, dtype=bool),
                    global_values=np.zeros_like(example.global_values, dtype=float),
                    global_mask=np.zeros_like(example.global_mask, dtype=bool),
                )
            )
        elif name == "without_solvent":
            transformed.append(
                replace(
                    example,
                    solvent_values=np.zeros_like(example.solvent_values, dtype=float),
                    solvent_raw="__NO_SOLVENT__",
                )
            )
        elif name == "without_site_type":
            transformed.append(
                replace(
                    example,
                    site_types=tuple("atom" for _ in example.site_types),
                )
            )
        elif name == "no_local_electronic":
            transformed.append(
                replace(
                    example,
                    local_values=np.zeros_like(example.local_values, dtype=float),
                    local_mask=np.zeros_like(example.local_mask, dtype=bool),
                )
            )
        elif name == "no_global_electronic":
            transformed.append(
                replace(
                    example,
                    global_values=np.zeros_like(example.global_values, dtype=float),
                    global_mask=np.zeros_like(example.global_mask, dtype=bool),
                )
            )
        elif name == "no_formal_charge":
            transformed.append(replace(example, model_formal_charge=0.0))
        elif name == "no_pretraining":
            transformed.append(example)
        else:
            raise PublicationModelingError(f"Unsupported input ablation: {name}")
    if [item.target_ids for item in transformed] != [
        item.target_ids for item in examples
    ]:
        raise PublicationModelingError("Ablation changed target identity")
    return transformed


def _bound_inputs(
    config: Mapping[str, Any], config_path: Path
) -> dict[str, dict[str, object]]:
    bindings: list[tuple[str, Path, object]] = [
        ("experiment_config", config_path, sha256_file(config_path)),
    ]
    if config.get("_ablation_parent_config_path"):
        bindings.append(
            (
                "ablation.parent_config",
                _project_path(
                    config["_ablation_parent_config_path"],
                    label="ablation.parent_config",
                ),
                config["_ablation_parent_config_sha256"],
            )
        )
    for section_name, pairs in {
        "dataset": (
            ("manifest_path", "manifest_sha256"),
            ("outer_membership_path", "outer_membership_sha256"),
            ("nested_membership_path", "nested_membership_sha256"),
        ),
        "lineage": (
            ("base_config_path", "base_config_sha256"),
            ("formal_config_path", "formal_config_sha256"),
            ("pretraining_config_path", "pretraining_config_sha256"),
            ("pretraining_protocol_path", "pretraining_protocol_sha256"),
            ("pretraining_aggregate_path", "pretraining_aggregate_sha256"),
            ("stage_c_config_path", "stage_c_config_sha256"),
            ("stage_e_b_config_path", "stage_e_b_config_sha256"),
            ("stage_e_c_config_path", "stage_e_c_config_sha256"),
        ),
    }.items():
        section = config[section_name]
        for path_key, hash_key in pairs:
            bindings.append(
                (
                    f"{section_name}.{path_key}",
                    _project_path(
                        section[path_key], label=f"{section_name}.{path_key}"
                    ),
                    section[hash_key],
                )
            )
    for index, entry in enumerate(config["pretraining"]["checkpoints"]):
        bindings.append(
            (
                f"pretraining.checkpoints[{index}]",
                _project_path(entry["path"], label=f"checkpoint[{index}].path"),
                entry["sha256"],
            )
        )
    ablation = config.get("ablation")
    if isinstance(ablation, Mapping) and ablation.get(
        "matched_pretraining_config_path"
    ):
        bindings.append(
            (
                "ablation.matched_pretraining_config",
                _project_path(
                    ablation["matched_pretraining_config_path"],
                    label="ablation.matched_pretraining_config_path",
                ),
                ablation["matched_pretraining_config_sha256"],
            )
        )
    verified: dict[str, dict[str, object]] = {}
    for label, path, expected in bindings:
        observed = _verify_bound_file(path, expected, label=label)
        verified[label] = {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": observed,
            "bytes": int(path.stat().st_size),
        }
    return verified


def _pretraining_entry(
    config: Mapping[str, Any], initialization_seed: int
) -> tuple[Mapping[str, Any], Path, dict[str, object]]:
    matches = [
        entry
        for entry in config["pretraining"]["checkpoints"]
        if int(entry["downstream_initialization_seed"]) == initialization_seed
    ]
    if len(matches) != 1:
        raise PublicationModelingError(
            f"Expected one pretraining checkpoint for seed {initialization_seed}"
        )
    entry = matches[0]
    path = _project_path(entry["path"], label="pretraining checkpoint")
    observed = _verify_bound_file(path, entry["sha256"], label="checkpoint")
    payload = load_pretraining_checkpoint(path)
    contract = payload.get("contract")
    if not isinstance(contract, Mapping):
        raise PublicationModelingError("Pretraining checkpoint lacks a contract")
    if contract.get("data_schema_version") != (
        "mayr-node-xtb-esnuel-pretraining-batch.v1"
    ):
        raise PublicationModelingError("Pretraining population is not ESNUEL-only")
    ablation = config.get("ablation")
    expected_tasks = EXPECTED_PRETRAINING_TASKS
    if isinstance(ablation, Mapping) and "expected_pretraining_tasks" in ablation:
        expected_tasks = frozenset(map(str, ablation["expected_pretraining_tasks"]))
    if frozenset(map(str, contract.get("tasks", ()))) != expected_tasks:
        raise PublicationModelingError("Pretraining task set changed")
    if isinstance(ablation, Mapping) and ablation.get("name") == "without_xtb":
        variant = contract.get("variant_contract")
        if not isinstance(variant, Mapping) or any(
            variant.get(key) is not False
            for key in (
                "runtime_local4_input",
                "runtime_global6_input",
                "local4_pretraining_target",
                "global6_pretraining_target",
            )
        ):
            raise PublicationModelingError("No-xTB pretraining contract changed")
    if isinstance(ablation, Mapping) and ablation.get("name") in {
        "no_local_electronic",
        "no_global_electronic",
    }:
        name = str(ablation["name"])
        local_retained = name == "no_global_electronic"
        global_retained = name == "no_local_electronic"
        variant = contract.get("variant_contract")
        expected = {
            "variant": name,
            "runtime_local4_input": local_retained,
            "runtime_global6_input": global_retained,
            "local4_pretraining_target": local_retained,
            "global6_pretraining_target": global_retained,
            "mayr_labels_used": False,
        }
        if not isinstance(variant, Mapping) or any(
            variant.get(key) != value for key, value in expected.items()
        ):
            raise PublicationModelingError(
                "Electronic-family pretraining contract changed"
            )
    if isinstance(ablation, Mapping) and ablation.get("name") == "no_pretraining":
        variant = contract.get("variant_contract")
        expected = {
            "variant": "no_pretraining",
            "deterministic_scratch_initialization": True,
            "pretraining_optimization_steps": 0,
            "esnuel_records_loaded": 0,
            "runtime_local4_input": True,
            "runtime_global6_input": True,
            "mayr_labels_used": False,
        }
        if not isinstance(variant, Mapping) or any(
            variant.get(key) != value for key, value in expected.items()
        ):
            raise PublicationModelingError("Scratch initialization contract changed")
    if int(payload["init_seed"]) != int(entry["pretraining_seed"]):
        raise PublicationModelingError("Pretraining checkpoint seed changed")
    return (
        entry,
        path,
        {
            "schema_version": "nucpred.mayr-n-pretraining-checkpoint-audit.v1",
            "status": "pass",
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": observed,
            "pretraining_seed": int(entry["pretraining_seed"]),
            "downstream_initialization_seed": initialization_seed,
            "data_schema_version": contract["data_schema_version"],
            "task_count": len(expected_tasks),
            "tasks": sorted(expected_tasks),
            "variant_contract": contract.get("variant_contract"),
            "mayr_labels_used": False,
            "backbone_state_sha256": payload["backbone_state_sha256"],
        },
    )


def _pretraining_lineage_summary(
    config: Mapping[str, Any],
) -> dict[str, object]:
    """Validate the exact pretraining lineage used by one conditional-N arm.

    The publication baseline and the older matched ablations point to the
    canonical ESNUEL pretraining configuration.  The new component study also
    has two deliberately different lineage schemas: zero-step deterministic
    initialization for ``no_pretraining`` and input/objective-matched ESNUEL
    reruns for the local/global electronic-family removals.  Treating all three
    schemas as the canonical configuration made the scratch arm fail before
    training and would also reject the electronic-family arms after their
    checkpoints were produced.
    """

    lineage = config["lineage"]
    pretraining_path = _project_path(
        lineage["pretraining_config_path"], label="pretraining config"
    )
    pretraining_config = tomllib.loads(pretraining_path.read_text(encoding="utf-8"))
    aggregate = _read_json(
        _project_path(
            lineage["pretraining_aggregate_path"],
            label="pretraining aggregate",
        )
    )
    ablation = config.get("ablation")
    name = str(ablation.get("name")) if isinstance(ablation, Mapping) else None
    population = str(
        lineage.get(
            "pretraining_population",
            "ESNUEL-only after six connectivity exclusions",
        )
    )

    if name == "no_pretraining":
        if (
            pretraining_config.get("schema_version")
            != "nucpred.mayr-n-publication-scratch-initialization.v1"
            or pretraining_config.get("component") != name
            or int(pretraining_config.get("esnuel_optimization_steps", -1)) != 0
            or pretraining_config.get("mayr_labels_used") is not False
        ):
            raise PublicationModelingError(
                "Scratch-initialization pretraining config changed"
            )
        if (
            aggregate.get("schema_version")
            != "nucpred.mayr-n-scratch-initialization-aggregate.v1"
            or aggregate.get("status") != "complete"
            or aggregate.get("component") != name
            or int(aggregate.get("seed_count", -1)) != 3
            or int(aggregate.get("pretraining_optimization_steps", -1)) != 0
            or int(aggregate.get("esnuel_records_loaded", -1)) != 0
            or aggregate.get("pretraining_tasks") != []
            or aggregate.get("mayr_labels_used") is not False
            or aggregate.get("config_sha256") != sha256_file(pretraining_path)
        ):
            raise PublicationModelingError(
                "Scratch-initialization pretraining aggregate changed"
            )
        return {
            "pretraining_population": population,
            "pretraining_record_count": 0,
            "pretraining_optimization_steps": 0,
            "pretraining_task_count": 0,
            "pretraining_component_ablation": name,
            "pretraining_mayr_labels_used": False,
        }

    if name in {"no_local_electronic", "no_global_electronic"}:
        expected_family = "local4" if name == "no_local_electronic" else "global6"
        input_ablation = pretraining_config.get("input_ablation")
        retained_tasks = pretraining_config.get("retained_tasks")
        expected_tasks = tuple(
            map(str, ablation.get("expected_pretraining_tasks", ()))
        )
        if (
            pretraining_config.get("schema_version")
            != "nucpred.mayr-n-publication-component-pretraining.v1"
            or pretraining_config.get("component") != name
            or pretraining_config.get("mayr_labels_used") is not False
            or pretraining_config.get("audit_test_used_for_selection") is not False
            or not isinstance(input_ablation, Mapping)
            or input_ablation.get("removed_family") != expected_family
            or input_ablation.get("values") != "zero"
            or input_ablation.get("availability") != "all_false"
            or float(input_ablation.get("mask_probability", -1)) != 0.0
            or float(input_ablation.get("reconstruction_weight", -1)) != 0.0
            or not isinstance(retained_tasks, Mapping)
            or frozenset(map(str, retained_tasks.get("names", ())))
            != frozenset(expected_tasks)
        ):
            raise PublicationModelingError(
                "Electronic-family pretraining config changed"
            )
        if (
            aggregate.get("schema_version")
            != "nucpred.mayr-n-component-pretraining-aggregate.v1"
            or aggregate.get("status") != "complete"
            or aggregate.get("component") != name
            or int(aggregate.get("dataset_record_count", -1)) != 47_915
            or int(aggregate.get("seed_count", -1)) != 3
            or aggregate.get("removed_input_and_reconstruction_objective_matched")
            is not True
            or aggregate.get("mayr_labels_used") is not False
            or aggregate.get("config_sha256") != sha256_file(pretraining_path)
        ):
            raise PublicationModelingError(
                "Electronic-family pretraining aggregate changed"
            )
        return {
            "pretraining_population": population,
            "pretraining_record_count": 47_915,
            "pretraining_task_count": len(expected_tasks),
            "pretraining_component_ablation": name,
            "pretraining_mayr_labels_used": False,
        }

    if pretraining_config["parents"]["m7_esnuel"].get("forbid_mayr_branch") is not True:
        raise PublicationModelingError("Pretraining Mayr branch is not forbidden")
    if int(pretraining_config["overlap"]["expected_eligible_esnuel_records"]) != 47_915:
        raise PublicationModelingError("Pretraining eligible population changed")
    if int(pretraining_config["overlap"]["expected_excluded_esnuel_records"]) != 6:
        raise PublicationModelingError("Pretraining overlap exclusion changed")
    if int(aggregate.get("dataset_record_count", -1)) != 47_915:
        raise PublicationModelingError("Pretraining aggregate population changed")
    return {
        "pretraining_population": population,
        "pretraining_record_count": 47_915,
        "pretraining_task_count": len(
            tuple(
                map(
                    str,
                    ablation.get("expected_pretraining_tasks", EXPECTED_PRETRAINING_TASKS)
                    if isinstance(ablation, Mapping)
                    else EXPECTED_PRETRAINING_TASKS,
                )
            )
        ),
        "pretraining_mayr_labels_used": False,
    }


def _membership_tables(config: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset = config["dataset"]
    outer = pd.read_csv(
        _project_path(dataset["outer_membership_path"], label="outer membership")
    )
    nested = pd.read_csv(
        _project_path(dataset["nested_membership_path"], label="nested membership")
    )
    return outer, nested


def _audit_splits(
    config: Mapping[str, Any], outer: pd.DataFrame, nested: pd.DataFrame
) -> dict[str, object]:
    required_outer = {
        "outer_fold",
        "role",
        "target_id",
        "context_id",
        "species_id",
        "connectivity_id",
    }
    required_nested = required_outer | {"inner_fold"}
    if set(outer.columns) != required_outer:
        raise PublicationModelingError("Outer membership columns changed")
    if set(nested.columns) != required_nested:
        raise PublicationModelingError("Nested membership columns changed")
    all_targets = set(outer["target_id"].astype(str))
    test_counts = (
        outer.loc[outer["role"].eq("test")].groupby("target_id", sort=False).size()
    )
    if set(test_counts.index.astype(str)) != all_targets or not test_counts.eq(1).all():
        raise PublicationModelingError("Outer folds do not partition targets")
    fold_rows: list[dict[str, object]] = []
    for outer_fold in range(int(config["outer_fold_count"])):
        selected_outer = outer.loc[outer["outer_fold"].eq(outer_fold)]
        development = selected_outer.loc[selected_outer["role"].eq("development")]
        test = selected_outer.loc[selected_outer["role"].eq("test")]
        dev_ids = set(development["target_id"].astype(str))
        test_ids = set(test["target_id"].astype(str))
        dev_conn = set(development["connectivity_id"].astype(str))
        test_conn = set(test["connectivity_id"].astype(str))
        if dev_ids & test_ids or dev_conn & test_conn:
            raise PublicationModelingError(f"Outer fold {outer_fold} leaks")
        for inner_fold in range(int(config["inner_fold_count"])):
            selected_inner = nested.loc[
                nested["outer_fold"].eq(outer_fold)
                & nested["inner_fold"].eq(inner_fold)
            ]
            train = selected_inner.loc[selected_inner["role"].eq("train")]
            validation = selected_inner.loc[selected_inner["role"].eq("validation")]
            train_ids = set(train["target_id"].astype(str))
            val_ids = set(validation["target_id"].astype(str))
            train_conn = set(train["connectivity_id"].astype(str))
            val_conn = set(validation["connectivity_id"].astype(str))
            if train_ids | val_ids != dev_ids:
                raise PublicationModelingError(
                    f"Inner {outer_fold}/{inner_fold} is not the outer development set"
                )
            if train_ids & val_ids or train_conn & val_conn:
                raise PublicationModelingError(
                    f"Inner {outer_fold}/{inner_fold} leaks connectivity"
                )
            if (train_ids | val_ids) & test_ids or (train_conn | val_conn) & test_conn:
                raise PublicationModelingError(
                    f"Inner {outer_fold}/{inner_fold} contains outer-test data"
                )
            fold_rows.append(
                {
                    "outer_fold": outer_fold,
                    "inner_fold": inner_fold,
                    "train_target_count": len(train_ids),
                    "train_connectivity_count": len(train_conn),
                    "validation_target_count": len(val_ids),
                    "validation_connectivity_count": len(val_conn),
                    "outer_test_target_count": len(test_ids),
                    "outer_test_connectivity_count": len(test_conn),
                }
            )
    return {
        "schema_version": "nucpred.mayr-n-publication-split-audit.v1",
        "status": "pass",
        "target_count": len(all_targets),
        "outer_fold_count": int(config["outer_fold_count"]),
        "inner_fold_count": int(config["inner_fold_count"]),
        "each_target_exactly_one_outer_test": True,
        "all_roles_connectivity_disjoint": True,
        "inner_rows_exclude_outer_test": True,
        "folds": fold_rows,
    }


def preflight(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    output_directory: str | Path | None = None,
) -> dict[str, object]:
    config, resolved = read_config(config_path)
    verified = _bound_inputs(config, resolved)
    dataset = _project_path(config["dataset"]["directory"], label="dataset")
    dataset_verification = verify_dataset(dataset)
    outer, nested = _membership_tables(config)
    split_audit = _audit_splits(config, outer, nested)
    checkpoint_audits = []
    for seed in map(int, config["outer_initialization_seeds"]):
        _, _, audit = _pretraining_entry(config, seed)
        checkpoint_audits.append(audit)
    pretraining_lineage = _pretraining_lineage_summary(config)
    result: dict[str, object] = {
        "schema_version": "nucpred.mayr-n-publication-modeling-preflight.v1",
        "status": "pass",
        "campaign_id": config["campaign_id"],
        "experiment_id": config["experiment_id"],
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "config_path": resolved.relative_to(ROOT).as_posix(),
        "config_sha256": sha256_file(resolved),
        "bound_inputs": verified,
        "dataset_verification": dataset_verification,
        "split_audit": split_audit,
        "pretraining_checkpoint_audits": checkpoint_audits,
        **pretraining_lineage,
        "sn_imported_or_predicted": False,
        "unknown_is_negative": False,
        "candidate_softmax_used": False,
        "outer_test_target_rows_loaded": 0,
    }
    result["contract_sha256"] = _canonical_sha256(result)
    root = (
        Path(output_directory).resolve()
        if output_directory is not None
        else _project_path(config["output_directory"], label="output directory")
        / "preflight"
    )
    root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(root / "preflight.json", result, ensure_ascii=False)
    return result


def _nested_ids(
    nested: pd.DataFrame, *, outer_fold: int, inner_fold: int, role: str
) -> set[str]:
    selected = nested.loc[
        nested["outer_fold"].eq(outer_fold)
        & nested["inner_fold"].eq(inner_fold)
        & nested["role"].eq(role),
        "target_id",
    ].astype(str)
    result = set(selected)
    if not result or len(result) != len(selected):
        raise PublicationModelingError(
            f"Invalid target membership for {outer_fold}/{inner_fold}/{role}"
        )
    return result


def _training_configs(
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    lineage = config["lineage"]
    base = tomllib.loads(
        _project_path(lineage["base_config_path"], label="base config").read_text(
            encoding="utf-8"
        )
    )
    stage_c = tomllib.loads(
        _project_path(lineage["stage_c_config_path"], label="Stage-C config").read_text(
            encoding="utf-8"
        )
    )
    stage_eb = tomllib.loads(
        _project_path(
            lineage["stage_e_b_config_path"], label="Stage-E-B config"
        ).read_text(encoding="utf-8")
    )
    stage_ec = tomllib.loads(
        _project_path(
            lineage["stage_e_c_config_path"], label="Stage-E-C config"
        ).read_text(encoding="utf-8")
    )
    training = config["training"]
    shared = training
    mappings = (
        (stage_c, training["base"]),
        (stage_eb, training["stage_e_b"]),
        (stage_ec, training["stage_e_c"]),
    )
    for stage, source in mappings:
        optimization = stage["r2"]["optimization"]
        for key in (
            "maximum_epochs",
            "minimum_epochs",
            "early_stopping_patience",
            "minimum_validation_metric_delta",
            "learning_rate",
            "weight_decay",
        ):
            optimization[key] = source[key]
        optimization["batch_size_contexts"] = shared["batch_size_contexts"]
        optimization["ranking_weight"] = shared["ranking_weight"]
        optimization["gradient_clip_norm"] = shared["gradient_clip_norm"]
        optimization["maximum_target_weight"] = shared["maximum_target_weight"]
    stage_eb["r2"]["residual_shrinkage_weight"] = shared["residual_shrinkage_weight"]
    stage_eb["r2"]["e_b_n1"]["pair_aware_batching"] = training["stage_e_b"][
        "pair_aware_batching"
    ]
    stage_eb["r2"]["e_b_n1"]["paired_solvent_weight"] = training["stage_e_b"][
        "paired_solvent_weight"
    ]
    stage_eb["r2"]["e_b_n1"]["center_penalty_weight"] = training["stage_e_b"][
        "center_penalty_weight"
    ]
    stage_ec["r2"]["residual_shrinkage_weight"] = shared["residual_shrinkage_weight"]
    return base, stage_c, stage_eb, stage_ec


def _in_memory_c2(model: torch.nn.Module, preprocessor, vocabulary) -> FrozenC2:
    state_hash = _tensor_mapping_sha256(model.state_dict())
    return FrozenC2(
        model=model,
        preprocessor=preprocessor,
        vocabulary=vocabulary,
        checkpoint_path=Path("<in-memory-c2>"),
        checkpoint_sha256=state_hash,
        model_state_sha256=state_hash,
        payload={"phase": "nested_inner_selection"},
        verification={"status": "pass", "in_memory": True},
    )


def _in_memory_eb(model: torch.nn.Module, preprocessor, vocabulary) -> FrozenEBN1:
    state_hash = _tensor_mapping_sha256(model.state_dict())
    return FrozenEBN1(
        model=model,
        preprocessor=preprocessor,
        vocabulary=vocabulary,
        checkpoint_path=Path("<in-memory-stage-e-b-n1>"),
        checkpoint_sha256=state_hash,
        model_state_sha256=state_hash,
        payload={"phase": "nested_inner_selection"},
        verification={"status": "pass", "in_memory": True},
    )


def _job_contract(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    outer_fold: int,
    inner_fold: int,
    train_ids: set[str],
    validation_ids: set[str],
    checkpoint_path: Path,
) -> dict[str, object]:
    sources = {
        "config": config_path,
        "runner": Path(__file__).resolve(),
        "dataset_manifest": _project_path(
            config["dataset"]["manifest_path"], label="dataset manifest"
        ),
        "outer_membership": _project_path(
            config["dataset"]["outer_membership_path"], label="outer membership"
        ),
        "nested_membership": _project_path(
            config["dataset"]["nested_membership_path"], label="nested membership"
        ),
        "pretraining_checkpoint": checkpoint_path,
    }
    contract: dict[str, object] = {
        "schema_version": "nucpred.mayr-n-publication-inner-job-contract.v1",
        "campaign_id": config["campaign_id"],
        "experiment_id": config["experiment_id"],
        "phase": "nested_inner_selection",
        "outer_fold": outer_fold,
        "inner_fold": inner_fold,
        "initialization_seed": int(config["inner_initialization_seed"]),
        "architecture": [
            config["lineage"]["base_arm"],
            config["lineage"]["stage_e_b_arm"],
            config["lineage"]["stage_e_c_arm"],
        ],
        "ablation": dict(config["ablation"]) if "ablation" in config else None,
        "source_hashes": {name: sha256_file(path) for name, path in sources.items()},
        "train_target_id_sha256": _canonical_sha256(sorted(train_ids)),
        "train_target_count": len(train_ids),
        "validation_target_id_sha256": _canonical_sha256(sorted(validation_ids)),
        "validation_target_count": len(validation_ids),
        "outer_test_target_rows_loaded": 0,
        "outer_test_predictions_computed": 0,
        "outer_test_metrics_computed": 0,
        "selection_metric": "rmse",
        "sn_imported_or_predicted": False,
        "unknown_is_negative": False,
        "candidate_softmax_used": False,
    }
    contract["contract_sha256"] = _canonical_sha256(contract)
    return contract


def _save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    preprocessor,
    vocabulary: SolventVocabulary,
    contract: Mapping[str, object],
    best_epochs: Mapping[str, int],
) -> dict[str, object]:
    state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    payload: dict[str, object] = {
        "schema_version": CHECKPOINT_SCHEMA,
        "phase": "nested_inner_selection",
        "model_lineage": "pre-sN_C2_to_E-B-N1_to_E-C-N3",
        "model_architecture": model.architecture,
        "model_state_dict": state,
        "model_state_sha256": _tensor_mapping_sha256(state),
        "preprocessor": preprocessor.to_json(),
        "solvent_vocabulary": list(vocabulary.tokens),
        "best_epochs": {name: int(value) for name, value in best_epochs.items()},
        "contract": dict(contract),
    }
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return payload


def run_inner(
    *,
    outer_fold: int,
    inner_fold: int,
    config_path: str | Path = DEFAULT_CONFIG,
    device: str | None = None,
) -> dict[str, object]:
    started = time.perf_counter()
    config, resolved = read_config(config_path)
    if outer_fold not in range(int(config["outer_fold_count"])):
        raise PublicationModelingError("Unregistered outer fold")
    if inner_fold not in range(int(config["inner_fold_count"])):
        raise PublicationModelingError("Unregistered inner fold")
    selected_device = torch.device(device or str(config["device"]))
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise PublicationModelingError("CUDA was requested but is unavailable")
    _bound_inputs(config, resolved)
    outer, nested = _membership_tables(config)
    _audit_splits(config, outer, nested)
    train_ids = _nested_ids(
        nested, outer_fold=outer_fold, inner_fold=inner_fold, role="train"
    )
    validation_ids = _nested_ids(
        nested, outer_fold=outer_fold, inner_fold=inner_fold, role="validation"
    )
    initialization_seed = int(config["inner_initialization_seed"])
    entry, checkpoint, checkpoint_audit = _pretraining_entry(
        config, initialization_seed
    )
    contract = _job_contract(
        config=config,
        config_path=resolved,
        outer_fold=outer_fold,
        inner_fold=inner_fold,
        train_ids=train_ids,
        validation_ids=validation_ids,
        checkpoint_path=checkpoint,
    )
    target = (
        _project_path(config["output_directory"], label="output directory")
        / "nested_inner"
        / f"outer-{outer_fold}"
        / f"inner-{inner_fold}"
    )
    summary_path = target / "summary.json"
    if summary_path.is_file():
        existing = _read_json(summary_path)
        if existing.get("status") == "pass" and existing.get("contract") == contract:
            return existing
        raise PublicationModelingError(f"Existing inner job is stale: {target}")
    if target.exists():
        raise PublicationModelingError(f"Partial inner job exists: {target}")
    dataset = _project_path(config["dataset"]["directory"], label="dataset")
    # These two filtered reads are the only Mayr target-table reads in this phase.
    train = apply_input_ablation(
        load_site_n_examples(dataset, target_ids=train_ids), config
    )
    validation = apply_input_ablation(
        load_site_n_examples(dataset, target_ids=validation_ids), config
    )
    if sum(item.num_sites for item in train) != len(train_ids):
        raise PublicationModelingError("Inner train loader changed target count")
    if sum(item.num_sites for item in validation) != len(validation_ids):
        raise PublicationModelingError("Inner validation loader changed target count")
    preprocessor = fit_site_n_preprocessor(train)
    vocabulary = SolventVocabulary.from_values(
        [example.solvent_raw for example in train]
    )
    base_config, stage_c_config, stage_eb_config, stage_ec_config = _training_configs(
        config
    )
    try:
        c2 = _fit_c2_selection(
            train,
            validation,
            arm=str(config["lineage"]["base_arm"]),
            base_config=base_config,
            stage_config=stage_c_config,
            preprocessor=preprocessor,
            vocabulary=vocabulary,
            initialization_seed=initialization_seed,
            device=selected_device,
            checkpoint=checkpoint,
        )
        frozen_c2 = _in_memory_c2(c2.model, preprocessor, vocabulary)
        eb = _fit_eb_selection(
            train,
            validation,
            arm=E_B_N1,
            config=stage_eb_config,
            frozen=frozen_c2,
            initialization_seed=initialization_seed,
            device=selected_device,
        )
        frozen_eb = _in_memory_eb(eb.model, preprocessor, vocabulary)
        ec = _fit_ec_selection(
            train,
            validation,
            arm=E_C_N3,
            config=stage_ec_config,
            frozen=frozen_eb,
            initialization_seed=initialization_seed,
            device=selected_device,
        )
        best_epochs = {
            "base_c2": int(c2.best_epoch),
            "stage_e_b_n1": int(eb.best_epoch),
            "stage_e_c_n3": int(ec.best_epoch),
        }
        connectivity = {
            target_id: example.connectivity_id
            for example in validation
            for target_id in example.target_ids
        }
        predictions = ec.validation_predictions.copy()
        predictions["connectivity_id"] = predictions["target_id"].map(connectivity)
        if predictions["connectivity_id"].isna().any():
            raise PublicationModelingError("Validation connectivity mapping failed")
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(prefix=f".inner-{inner_fold}.staging-", dir=target.parent)
        )
        try:
            for name, outcome in (
                ("base_c2", c2),
                ("stage_e_b_n1", eb),
                ("stage_e_c_n3", ec),
            ):
                outcome.curves.to_csv(
                    staging / f"{name}_selection_curves.csv",
                    index=False,
                    lineterminator="\n",
                )
            predictions.to_parquet(
                staging / "validation_predictions.parquet",
                index=False,
                engine="pyarrow",
                compression="zstd",
            )
            validation_metrics = {
                "base_c2": dict(c2.validation_metrics),
                "stage_e_b_n1": dict(eb.validation_metrics),
                "stage_e_c_n3": dict(ec.validation_metrics),
            }
            atomic_write_json(
                staging / "validation_metrics.json",
                validation_metrics,
                ensure_ascii=False,
            )
            checkpoint_payload = _save_checkpoint(
                staging / "selection_checkpoint.pt",
                model=ec.model,
                preprocessor=preprocessor,
                vocabulary=vocabulary,
                contract=contract,
                best_epochs=best_epochs,
            )
            summary: dict[str, object] = {
                "schema_version": "nucpred.mayr-n-publication-inner-job.v1",
                "status": "pass",
                "campaign_id": config["campaign_id"],
                "experiment_id": config["experiment_id"],
                "outer_fold": outer_fold,
                "inner_fold": inner_fold,
                "initialization_seed": initialization_seed,
                "pretraining_seed": int(entry["pretraining_seed"]),
                "pretraining_checkpoint_audit": checkpoint_audit,
                "contract": contract,
                "train_context_count": len(train),
                "train_target_count": len(train_ids),
                "validation_context_count": len(validation),
                "validation_target_count": len(validation_ids),
                "best_epochs": best_epochs,
                "validation_metrics": validation_metrics,
                "model_state_sha256": checkpoint_payload["model_state_sha256"],
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


def select_outer_epochs(
    *,
    outer_fold: int,
    config_path: str | Path = DEFAULT_CONFIG,
) -> dict[str, object]:
    config, resolved = read_config(config_path)
    if outer_fold not in range(int(config["outer_fold_count"])):
        raise PublicationModelingError("Unregistered outer fold")
    root = _project_path(config["output_directory"], label="output directory")
    summaries = []
    for inner_fold in range(int(config["inner_fold_count"])):
        path = (
            root
            / "nested_inner"
            / f"outer-{outer_fold}"
            / f"inner-{inner_fold}"
            / "summary.json"
        )
        summary = _read_json(path)
        if summary.get("status") != "pass":
            raise PublicationModelingError(f"Inner job did not pass: {path}")
        summaries.append(summary)
    stages = ("base_c2", "stage_e_b_n1", "stage_e_c_n3")
    selected: dict[str, int] = {}
    observed: dict[str, list[int]] = {}
    for stage in stages:
        values = sorted(int(item["best_epochs"][stage]) for item in summaries)
        observed[stage] = values
        # statistics.median_high is the preregistered upper median for four folds.
        selected[stage] = int(statistics.median_high(values))
    payload: dict[str, object] = {
        "schema_version": "nucpred.mayr-n-publication-outer-epoch-selection.v1",
        "status": "frozen",
        "campaign_id": config["campaign_id"],
        "experiment_id": config["experiment_id"],
        "outer_fold": outer_fold,
        "rule": "upper_median_of_four_inner_best_epochs",
        "inner_best_epochs_sorted": observed,
        "selected_epochs": selected,
        "inner_summary_sha256": [
            sha256_file(
                root
                / "nested_inner"
                / f"outer-{outer_fold}"
                / f"inner-{index}"
                / "summary.json"
            )
            for index in range(int(config["inner_fold_count"]))
        ],
        "config_sha256": sha256_file(resolved),
        "outer_test_target_rows_loaded": 0,
        "outer_test_predictions_computed": 0,
        "outer_test_metrics_computed": 0,
        "selected_at_utc": datetime.now(UTC).isoformat(),
    }
    payload["selection_sha256"] = _canonical_sha256(payload)
    target = root / "outer_epoch_selection" / f"outer-{outer_fold}.json"
    if target.exists():
        existing = _read_json(target)
        comparable = deepcopy(existing)
        comparable.pop("selected_at_utc", None)
        candidate = deepcopy(payload)
        candidate.pop("selected_at_utc", None)
        if comparable != candidate:
            raise PublicationModelingError(
                f"Frozen outer epoch selection drifted: {target}"
            )
        return existing
    atomic_write_json(target, payload, ensure_ascii=False)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--output-directory", type=Path)
    inner_parser = subparsers.add_parser("inner")
    inner_parser.add_argument("--outer-fold", type=int, required=True)
    inner_parser.add_argument("--inner-fold", type=int, required=True)
    inner_parser.add_argument("--device")
    select_parser = subparsers.add_parser("select-outer-epochs")
    select_parser.add_argument("--outer-fold", type=int, required=True)
    args = parser.parse_args(argv)
    if args.command == "preflight":
        result = preflight(args.config, output_directory=args.output_directory)
    elif args.command == "inner":
        result = run_inner(
            outer_fold=args.outer_fold,
            inner_fold=args.inner_fold,
            config_path=args.config,
            device=args.device,
        )
    else:
        result = select_outer_epochs(
            outer_fold=args.outer_fold,
            config_path=args.config,
        )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
