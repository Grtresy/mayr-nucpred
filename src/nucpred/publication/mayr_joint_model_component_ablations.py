"""Prepare hash-bound, end-to-end component ablations for the joint model.

The final joint learner inherits a conditional-N teacher and a site-ranking
offset.  A component ablation is therefore scientifically matched only when
the same input removal is present in (i) those upstream models and (ii) the
joint learner's own inputs.  This module creates immutable derived input views
and binds them to completed matched upstream artifacts without changing labels,
candidate identities, or grouped folds.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import tomllib
from typing import Any

import numpy as np
import pandas as pd

from nucpred.core.files import atomic_write_json, sha256_file
from nucpred.project import get_project_layout
from nucpred.publication.mayr_retraining_config import _atomic_write_text, _toml_string
from nucpred.training.mayr_node_xtb_scratch import SOLVENT_FEATURES


ROOT = get_project_layout().root
DEFAULT_JOINT_CONFIG = ROOT / "configs/mayr_joint_site_n_v1.toml"
PARENT_DATASET = ROOT / "data/processed/mayr_site_n/mayr-site-n-20260805-v2"
CAMPAIGN_ROOT = ROOT / (
    "artifacts/campaigns/mayr-joint-model-component-ablations-20260815-v1"
)
REPORTING_BASELINE_ROOT = ROOT / (
    "artifacts/campaigns/mayr-wsp-reporting-baseline-20260811-v1/baseline_freeze"
)
REPORTING_MODEL_VARIANT = "without_set_pooling"
DATASET_SCHEMA = "nucpred.mayr-site-n-derived-input-ablation.v1"
PREPARATION_SCHEMA = "nucpred.mayr-joint-model-component-ablation-preparation.v1"


COMPONENTS: dict[str, dict[str, str]] = {
    "no_pretraining": {
        "public_label": "without transferable pretraining",
        "dataset_suffix": "no-pretraining",
        "conditional_config": "configs/mayr_n_publication_ablation_no_pretraining_v1.toml",
        "site_config": "configs/mayr_n_publication_site_ablation_no_pretraining_v1.toml",
        "config": "configs/mayr_joint_model_ablation_no_pretraining_v2.toml",
    },
    "no_local_electronic": {
        "public_label": "without local electronic inputs",
        "dataset_suffix": "no-local-electronic",
        "conditional_config": "configs/mayr_n_publication_ablation_no_local_electronic_v1.toml",
        "site_config": "configs/mayr_n_publication_site_ablation_no_local_electronic_v1.toml",
        "config": "configs/mayr_joint_model_ablation_no_local_electronic_v2.toml",
    },
    "no_global_electronic": {
        "public_label": "without global electronic inputs",
        "dataset_suffix": "no-global-electronic",
        "conditional_config": "configs/mayr_n_publication_ablation_no_global_electronic_v1.toml",
        "site_config": "configs/mayr_n_publication_site_ablation_no_global_electronic_v1.toml",
        "config": "configs/mayr_joint_model_ablation_no_global_electronic_v2.toml",
    },
    "no_formal_charge": {
        "public_label": "without context-level molecular formal charge",
        "dataset_suffix": "no-formal-charge",
        "conditional_config": "configs/mayr_n_publication_ablation_no_formal_charge_v1.toml",
        "site_config": "configs/mayr_n_publication_site_ablation_no_formal_charge_v1.toml",
        "config": "configs/mayr_joint_model_ablation_no_formal_charge_v2.toml",
    },
    "no_xtb": {
        "public_label": "without xTB-derived electronic inputs",
        "dataset_suffix": "no-xtb",
        "conditional_config": "configs/mayr_n_publication_ablation_without_xtb_v1.toml",
        "site_config": "configs/mayr_n_publication_site_ablation_without_xtb_v1.toml",
        "config": "configs/mayr_joint_model_ablation_no_xtb_v2.toml",
    },
    "no_solvent": {
        "public_label": "without explicit solvent inputs",
        "dataset_suffix": "no-solvent",
        "conditional_config": "configs/mayr_n_publication_ablation_without_solvent_v1.toml",
        "site_config": "configs/mayr_n_publication_site_ablation_without_solvent_v1.toml",
        "config": "configs/mayr_joint_model_ablation_no_solvent_v2.toml",
    },
}


class ComponentAblationPreparationError(RuntimeError):
    """Raised when a matched ablation view cannot be frozen safely."""


def _reporting_baseline_binding() -> dict[str, str]:
    summary_path = REPORTING_BASELINE_ROOT / "summary.json"
    manifest_path = REPORTING_BASELINE_ROOT / "manifest.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        summary.get("status") != "frozen_pre_audit_reporting_baseline"
        or summary.get("selected_architecture_status") != "final_wsp_architecture"
        or summary.get("source_variant") != REPORTING_MODEL_VARIANT
        or summary.get("current_result_may_be_called_independent_confirmation")
        is not False
        or int(summary.get("metrics", {}).get("context_count", -1)) != 1_026
        or manifest.get("status") != "frozen"
    ):
        raise ComponentAblationPreparationError(
            "Frozen reporting-model identity or statistical role changed"
        )
    return {
        "variant": REPORTING_MODEL_VARIANT,
        "summary_path": _relative(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "manifest_path": _relative(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
    }


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _project_path(value: str | Path, *, label: str) -> Path:
    raw = Path(value)
    path = raw.resolve() if raw.is_absolute() else (ROOT / raw).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as exc:
        raise ComponentAblationPreparationError(
            f"{label} escapes the project root"
        ) from exc
    return path


def _canonical_frame_identity(frame: pd.DataFrame) -> str:
    columns = ["context_id", "species_id", "connectivity_id"]
    payload = frame[columns].sort_values(columns, kind="stable").to_dict("records")
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _zero_json(value: object, *, boolean: bool) -> str:
    parsed = json.loads(str(value))

    def replace(item: object) -> object:
        if isinstance(item, list):
            return [replace(value) for value in item]
        return False if boolean else 0.0

    return json.dumps(replace(parsed), separators=(",", ":"))


def transform_context_inputs(
    contexts: pd.DataFrame, *, component: str
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Return a model-input-only transformation and its explicit contract."""

    if component not in COMPONENTS:
        raise ComponentAblationPreparationError(f"Unsupported component: {component}")
    result = contexts.copy(deep=True)
    identity_before = _canonical_frame_identity(result)
    changed: list[str] = []
    if component == "no_pretraining":
        pass
    elif component == "no_local_electronic":
        for column, boolean in (
            ("node_local4_json", False),
            ("node_local4_available_json", True),
        ):
            result[column] = result[column].map(
                lambda value, flag=boolean: _zero_json(value, boolean=flag)
            )
            changed.append(column)
    elif component == "no_global_electronic":
        for column, boolean in (
            ("molecule_global6_json", False),
            ("molecule_global6_available_json", True),
        ):
            result[column] = result[column].map(
                lambda value, flag=boolean: _zero_json(value, boolean=flag)
            )
            changed.append(column)
    elif component == "no_formal_charge":
        for column in ("formal_charge", "model_formal_charge"):
            result[column] = 0.0
            changed.append(column)
    elif component == "no_xtb":
        for column, boolean in (
            ("node_local4_json", False),
            ("node_local4_available_json", True),
            ("molecule_global6_json", False),
            ("molecule_global6_available_json", True),
        ):
            result[column] = result[column].map(
                lambda value, flag=boolean: _zero_json(value, boolean=flag)
            )
            changed.append(column)
    elif component == "no_solvent":
        for column in SOLVENT_FEATURES:
            result[column] = 0.0
            changed.append(column)
        result["solvent_raw"] = "__NO_SOLVENT__"
        changed.append("solvent_raw")
    else:  # pragma: no cover - guarded by the registry above
        raise AssertionError(component)
    if _canonical_frame_identity(result) != identity_before:
        raise ComponentAblationPreparationError(
            "Input ablation changed context/species/connectivity identity"
        )
    untouched = sorted(set(contexts.columns) - set(changed))
    for column in untouched:
        if not contexts[column].equals(result[column]):
            raise ComponentAblationPreparationError(
                f"Input ablation unexpectedly changed {column}"
            )
    if component == "no_pretraining":
        if not result.equals(contexts):
            raise ComponentAblationPreparationError(
                "No-pretraining unexpectedly changed runtime inputs"
            )
    elif component == "no_local_electronic":
        local_available = result["node_local4_available_json"].map(json.loads)
        if any(np.asarray(value, dtype=bool).any() for value in local_available):
            raise ComponentAblationPreparationError("A local electronic mask remains enabled")
    elif component == "no_global_electronic":
        global_available = result["molecule_global6_available_json"].map(json.loads)
        if any(np.asarray(value, dtype=bool).any() for value in global_available):
            raise ComponentAblationPreparationError("A global electronic mask remains enabled")
    elif component == "no_formal_charge":
        if result[["formal_charge", "model_formal_charge"]].to_numpy(dtype=float).any():
            raise ComponentAblationPreparationError(
                "A context-level formal-charge scalar remains nonzero"
            )
    elif component == "no_xtb":
        local_available = result["node_local4_available_json"].map(json.loads)
        global_available = result["molecule_global6_available_json"].map(json.loads)
        if any(np.asarray(value, dtype=bool).any() for value in local_available):
            raise ComponentAblationPreparationError("A local xTB mask remains enabled")
        if any(np.asarray(value, dtype=bool).any() for value in global_available):
            raise ComponentAblationPreparationError("A global xTB mask remains enabled")
    elif result[list(SOLVENT_FEATURES)].to_numpy(dtype=float).any():
        raise ComponentAblationPreparationError("A solvent descriptor remains nonzero")
    return result, {
        "component": component,
        "public_label": COMPONENTS[component]["public_label"],
        "changed_columns": changed,
        "unchanged_column_count": len(untouched),
        "context_count": int(len(result)),
        "context_identity_sha256": identity_before,
        "target_or_split_column_read": False,
    }


def _verify_derived_dataset(path: Path, *, component: str) -> dict[str, Any]:
    manifest_path = path / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise ComponentAblationPreparationError(
            f"Derived dataset has no manifest: {path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != DATASET_SCHEMA
        or manifest.get("status") != "frozen"
        or manifest.get("component") != component
        or manifest.get("labels_candidates_and_splits_byte_identical") is not True
    ):
        raise ComponentAblationPreparationError(
            f"Derived dataset contract changed: {path}"
        )
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise ComponentAblationPreparationError("Derived manifest lacks file bindings")
    for name, raw in files.items():
        file_path = path / str(name)
        if (
            not isinstance(raw, Mapping)
            or not file_path.is_file()
            or int(file_path.stat().st_size) != int(raw["bytes"])
            or sha256_file(file_path) != str(raw["sha256"])
        ):
            raise ComponentAblationPreparationError(
                f"Derived dataset file drifted: {file_path}"
            )
    return manifest


def build_derived_dataset(
    component: str,
    *,
    parent: str | Path = PARENT_DATASET,
    output: str | Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Freeze a derived feature view while retaining exact labels and folds."""

    if component not in COMPONENTS:
        raise ComponentAblationPreparationError(f"Unsupported component: {component}")
    source = _project_path(parent, label="parent dataset")
    target = (
        _project_path(output, label="derived dataset")
        if output is not None
        else source.with_name(
            f"{source.name}-ablation-{COMPONENTS[component]['dataset_suffix']}"
        )
    )
    if target.exists():
        return target, _verify_derived_dataset(target, component=component)
    if not (source / "dataset_manifest.json").is_file():
        raise ComponentAblationPreparationError("Parent dataset is not frozen")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent)
    )
    try:
        source_contexts = pd.read_parquet(source / "contexts.parquet")
        transformed, transform_audit = transform_context_inputs(
            source_contexts, component=component
        )
        transformed.to_parquet(
            staging / "contexts.parquet",
            index=False,
            engine="pyarrow",
            compression="zstd",
        )
        for source_file in sorted(source.iterdir()):
            if (
                not source_file.is_file()
                or source_file.name in {"contexts.parquet", "dataset_manifest.json"}
            ):
                continue
            destination = staging / source_file.name
            try:
                os.link(source_file, destination)
            except OSError:
                shutil.copy2(source_file, destination)
        files = {
            path.name: {
                "path": path.name,
                "sha256": sha256_file(path),
                "bytes": int(path.stat().st_size),
            }
            for path in sorted(staging.iterdir())
            if path.is_file()
        }
        parent_manifest = source / "dataset_manifest.json"
        immutable_names = sorted(set(files) - {"contexts.parquet"})
        if any(
            sha256_file(staging / name) != sha256_file(source / name)
            for name in immutable_names
        ):
            raise ComponentAblationPreparationError(
                "A label, candidate, or split artifact changed"
            )
        manifest: dict[str, Any] = {
            "schema_version": DATASET_SCHEMA,
            "status": "frozen",
            "dataset_id": f"mayr-site-n-20260805-v2-ablation-{COMPONENTS[component]['dataset_suffix']}",
            "component": component,
            "public_label": COMPONENTS[component]["public_label"],
            "parent_dataset_path": _relative(source),
            "parent_manifest_sha256": sha256_file(parent_manifest),
            "model_input_transform": transform_audit,
            "labels_candidates_and_splits_byte_identical": True,
            "immutable_parent_files": immutable_names,
            "files": files,
        }
        atomic_write_json(staging / "dataset_manifest.json", manifest, ensure_ascii=False)
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target, _verify_derived_dataset(target, component=component)


def _render_toml_value(value: object) -> list[str]:
    if isinstance(value, str):
        return [_toml_string(value)]
    if isinstance(value, bool):
        return ["true" if value else "false"]
    if isinstance(value, int):
        return [str(value)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return ["[", *(f"  {_toml_string(str(item))}," for item in value), "]"]
    raise TypeError(f"Unsupported TOML value: {value!r}")


def replace_toml_section_values(
    text: str, *, section: str, replacements: Mapping[str, object]
) -> str:
    """Replace existing assignments without reserializing the frozen config."""

    lines = text.splitlines()
    header = f"[{section}]"
    starts = [index for index, line in enumerate(lines) if line.strip() == header]
    if len(starts) != 1:
        raise ComponentAblationPreparationError(
            f"Expected one TOML section {header}, found {len(starts)}"
        )
    start = starts[0] + 1
    end = next(
        (
            index
            for index in range(start, len(lines))
            if lines[index].startswith("[")
        ),
        len(lines),
    )
    spans: list[tuple[int, int, str, list[str]]] = []
    for key, value in replacements.items():
        matches = [
            index
            for index in range(start, end)
            if re.match(rf"^{re.escape(key)}\s*=", lines[index])
        ]
        if len(matches) != 1:
            raise ComponentAblationPreparationError(
                f"Expected one assignment [{section}].{key}, found {len(matches)}"
            )
        first = matches[0]
        last = first + 1
        if lines[first].split("=", 1)[1].strip().startswith("[") and not lines[
            first
        ].rstrip().endswith("]"):
            while last < end and lines[last].strip() != "]":
                last += 1
            if last >= end:
                raise ComponentAblationPreparationError(
                    f"Unterminated array [{section}].{key}"
                )
            last += 1
        rendered = _render_toml_value(value)
        replacement = [f"{key} = {rendered[0]}", *rendered[1:]]
        spans.append((first, last, key, replacement))
    for first, last, _key, replacement in sorted(spans, reverse=True):
        lines[first:last] = replacement
    return "\n".join(lines) + "\n"


def _upstream_bindings(component: str) -> dict[str, Any]:
    from nucpred.publication import mayr_site_publication as site_publication

    spec = COMPONENTS[component]
    site_path = _project_path(spec["site_config"], label="site config")
    conditional_path = _project_path(
        spec["conditional_config"], label="conditional-N config"
    )
    site_config, _ = site_publication.read_config(site_path)
    conditional_config, resolved_conditional = site_publication.conditional_config(
        site_config
    )
    if resolved_conditional != conditional_path:
        raise ComponentAblationPreparationError(
            "Site and conditional-N ablation configs are not matched"
        )
    site_publication.verify_bindings(site_config, site_path)
    site_root = _project_path(site_config["output_directory"], label="site output")
    conditional_root = _project_path(
        conditional_config["output_directory"], label="conditional-N output"
    )
    evaluation = site_root / "outer_evaluation"
    required_evaluation = (
        "summary.json",
        "single_target_evaluation.parquet",
        "site_type_metrics.csv",
        "outer_fold_metrics.csv",
        "paired_comparisons.csv",
    )
    if any(not (evaluation / name).is_file() for name in required_evaluation):
        raise ComponentAblationPreparationError(
            f"Matched upstream evaluation is incomplete: {evaluation}"
        )
    development_root = site_root / "outer_refit"
    score_root = site_root / "outer_score_freeze"
    development_hashes = [
        sha256_file(
            development_root / f"outer-{outer_fold}" / "development_oof_predictions.parquet"
        )
        for outer_fold in range(5)
    ]
    development_summary_hashes = [
        sha256_file(development_root / f"outer-{outer_fold}" / "summary.json")
        for outer_fold in range(5)
    ]
    score_hashes = [
        sha256_file(score_root / f"outer-{outer_fold}" / "candidate_scores.parquet")
        for outer_fold in range(5)
    ]
    score_summary_hashes = [
        sha256_file(score_root / f"outer-{outer_fold}" / "summary.json")
        for outer_fold in range(5)
    ]
    return {
        "automatic_site_config_path": _relative(site_path),
        "automatic_site_config_sha256": sha256_file(site_path),
        "evaluation": evaluation,
        "conditional_root": conditional_root,
        "site_root": site_root,
        "development_root": development_root,
        "score_root": score_root,
        "development_hashes": development_hashes,
        "development_summary_hashes": development_summary_hashes,
        "score_hashes": score_hashes,
        "score_summary_hashes": score_summary_hashes,
        "conditional_config_path": _relative(conditional_path),
        "conditional_config_sha256": sha256_file(conditional_path),
    }


def render_joint_component_config(
    component: str,
    *,
    base_text: str,
    dataset: Path,
    dataset_manifest: Mapping[str, Any],
    upstream: Mapping[str, Any],
    reporting: Mapping[str, str],
) -> str:
    spec = COMPONENTS[component]
    files = dataset_manifest["files"]
    result = replace_toml_section_values(
        base_text,
        section="dataset",
        replacements={
            "dataset_id": dataset_manifest["dataset_id"],
            "directory": _relative(dataset),
            "manifest_path": _relative(dataset / "dataset_manifest.json"),
            "manifest_sha256": sha256_file(dataset / "dataset_manifest.json"),
            "contexts_path": _relative(dataset / "contexts.parquet"),
            "contexts_sha256": files["contexts.parquet"]["sha256"],
            "targets_path": _relative(dataset / "targets.parquet"),
            "targets_sha256": files["targets.parquet"]["sha256"],
            "candidates_path": _relative(dataset / "candidate_sites.parquet"),
            "candidates_sha256": files["candidate_sites.parquet"]["sha256"],
            "species_path": _relative(dataset / "species.parquet"),
            "species_sha256": files["species.parquet"]["sha256"],
            "outer_membership_path": _relative(dataset / "outer_fold_membership.csv"),
            "outer_membership_sha256": files["outer_fold_membership.csv"]["sha256"],
            "nested_membership_path": _relative(
                dataset / "nested_split_membership.csv"
            ),
            "nested_membership_sha256": files["nested_split_membership.csv"][
                "sha256"
            ],
        },
    )
    result = replace_toml_section_values(
        result,
        section="ablations",
        replacements={"required": [reporting["variant"]]},
    )
    evaluation = upstream["evaluation"]
    result = replace_toml_section_values(
        result,
        section="baseline",
        replacements={
            "automatic_site_config_path": upstream["automatic_site_config_path"],
            "automatic_site_config_sha256": upstream[
                "automatic_site_config_sha256"
            ],
            "evaluation_summary_path": _relative(evaluation / "summary.json"),
            "evaluation_summary_sha256": sha256_file(evaluation / "summary.json"),
            "single_target_evaluation_path": _relative(
                evaluation / "single_target_evaluation.parquet"
            ),
            "single_target_evaluation_sha256": sha256_file(
                evaluation / "single_target_evaluation.parquet"
            ),
            "site_type_metrics_path": _relative(evaluation / "site_type_metrics.csv"),
            "site_type_metrics_sha256": sha256_file(
                evaluation / "site_type_metrics.csv"
            ),
            "outer_fold_metrics_path": _relative(
                evaluation / "outer_fold_metrics.csv"
            ),
            "outer_fold_metrics_sha256": sha256_file(
                evaluation / "outer_fold_metrics.csv"
            ),
            "paired_comparisons_path": _relative(
                evaluation / "paired_comparisons.csv"
            ),
            "paired_comparisons_sha256": sha256_file(
                evaluation / "paired_comparisons.csv"
            ),
            "inner_checkpoint_root": _relative(
                upstream["conditional_root"] / "nested_inner"
            ),
            "outer_checkpoint_root": _relative(
                upstream["conditional_root"] / "outer_refit"
            ),
            "inner_site_ranker_root": _relative(
                upstream["site_root"] / "nested_inner"
            ),
            "outer_site_ranker_root": _relative(
                upstream["site_root"] / "outer_refit"
            ),
            "outer_development_oof_root": _relative(upstream["development_root"]),
            "outer_development_oof_sha256": upstream["development_hashes"],
            "outer_development_summary_sha256": upstream[
                "development_summary_hashes"
            ],
            "outer_score_freeze_root": _relative(upstream["score_root"]),
            "outer_score_candidate_sha256": upstream["score_hashes"],
            "outer_score_summary_sha256": upstream["score_summary_hashes"],
        },
    )
    output_directory = _relative(CAMPAIGN_ROOT / component)
    result = re.sub(
        r'^output_directory = ".*"$',
        f"output_directory = {_toml_string(output_directory)}",
        result,
        count=1,
        flags=re.MULTILINE,
    )
    result += (
        "\n[component_ablation]\n"
        f"schema_version = {_toml_string(PREPARATION_SCHEMA)}\n"
        f"name = {_toml_string(component)}\n"
        f"public_label = {_toml_string(spec['public_label'])}\n"
        f"base_model_internal_variant = {_toml_string(reporting['variant'])}\n"
        f"reporting_baseline_summary_path = {_toml_string(reporting['summary_path'])}\n"
        f"reporting_baseline_summary_sha256 = {_toml_string(reporting['summary_sha256'])}\n"
        f"reporting_baseline_manifest_path = {_toml_string(reporting['manifest_path'])}\n"
        f"reporting_baseline_manifest_sha256 = {_toml_string(reporting['manifest_sha256'])}\n"
        "same_outer_folds = true\n"
        "same_initialization_seeds = true\n"
        "same_joint_architecture_and_optimization = true\n"
        "matched_upstream_conditional_n_and_site_models = true\n"
        f"conditional_n_config_path = {_toml_string(upstream['conditional_config_path'])}\n"
        f"conditional_n_config_sha256 = {_toml_string(upstream['conditional_config_sha256'])}\n"
        "base_runtime_registry_role = \"candidate-policy contract only; its model weights are not transferred\"\n"
    )
    parsed = tomllib.loads(result)
    if parsed["component_ablation"]["name"] != component:
        raise ComponentAblationPreparationError("Rendered component identity changed")
    if parsed["ablations"]["required"] != [REPORTING_MODEL_VARIANT]:
        raise ComponentAblationPreparationError("Reporting-model variant changed")
    return result


def prepare_component(component: str) -> dict[str, Any]:
    """Create and validate one complete joint-component training contract."""

    if component not in COMPONENTS:
        raise ComponentAblationPreparationError(f"Unsupported component: {component}")
    dataset, dataset_manifest = build_derived_dataset(component)
    upstream = _upstream_bindings(component)
    reporting = _reporting_baseline_binding()
    base_text = DEFAULT_JOINT_CONFIG.read_text(encoding="utf-8")
    rendered = render_joint_component_config(
        component,
        base_text=base_text,
        dataset=dataset,
        dataset_manifest=dataset_manifest,
        upstream=upstream,
        reporting=reporting,
    )
    output = _project_path(COMPONENTS[component]["config"], label="output config")
    if output.exists():
        if output.read_text(encoding="utf-8") != rendered:
            raise ComponentAblationPreparationError(
                f"Existing component config drifted: {output}"
            )
    else:
        _atomic_write_text(output, rendered)

    from nucpred.experiments.mayr.joint_site_n import (
        read_config,
        verify_input_bindings,
    )

    config, resolved = read_config(output)
    bindings = verify_input_bindings(config, resolved)
    result = {
        "schema_version": PREPARATION_SCHEMA,
        "status": "pass",
        "component": component,
        "public_label": COMPONENTS[component]["public_label"],
        "config_path": _relative(resolved),
        "config_sha256": sha256_file(resolved),
        "dataset_path": _relative(dataset),
        "dataset_manifest_sha256": sha256_file(dataset / "dataset_manifest.json"),
        "matched_upstream": {
            key: upstream[key]
            for key in (
                "conditional_config_path",
                "conditional_config_sha256",
                "automatic_site_config_path",
                "automatic_site_config_sha256",
            )
        },
        "reporting_model": reporting,
        "verified_joint_input_binding_count": len(bindings),
        "ready_for_diagnostic_training": True,
    }
    manifest = output.with_suffix(".manifest.json")
    if manifest.exists():
        existing = json.loads(manifest.read_text(encoding="utf-8"))
        if existing != result:
            raise ComponentAblationPreparationError(
                f"Existing preparation manifest drifted: {manifest}"
            )
    else:
        atomic_write_json(manifest, result, ensure_ascii=False)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--components",
        nargs="+",
        choices=tuple(COMPONENTS),
        default=tuple(COMPONENTS),
    )
    args = parser.parse_args(argv)
    results = [prepare_component(component) for component in args.components]
    print(json.dumps(results, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
