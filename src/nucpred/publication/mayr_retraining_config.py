"""Bind locally regenerated ESNUEL checkpoints to an independent replication.

The frozen publication configs are evidence for the exact paper weights and
must not be edited after release.  A from-scratch user necessarily creates new
checkpoint and aggregate hashes, so this module writes separate conditional-N
and automatic-site configs plus compact solvent/site-type ablation overlays.
Generated outputs are marked as independent replications and may not be
presented as the frozen paper run.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import re
import tempfile
import tomllib
from typing import Any

from nucpred.core.files import atomic_write_json, sha256_file
from nucpred.project import get_project_layout
from nucpred.publication import mayr_baselines as baselines
from nucpred.publication import mayr_n_modeling as modeling
from nucpred.publication import mayr_site_publication as site
from nucpred.training.mayr_node_xtb_pretraining import load_pretraining_checkpoint


ROOT = get_project_layout().root
DEFAULT_PARENT = ROOT / "configs/mayr_n_publication_experiment_v1.toml"
DEFAULT_SITE_PARENT = ROOT / "configs/mayr_n_publication_site_v1.toml"
DEFAULT_BASELINE_PARENT = ROOT / "configs/mayr_n_publication_baselines_v1.toml"
DEFAULT_AGGREGATE = ROOT / (
    "artifacts/campaigns/mayr-explicit-h-node-xtb-pretraining-20260726-v1/"
    "full/aggregate/aggregate_summary.json"
)
DEFAULT_OUTPUT = ROOT / (
    "artifacts/reproduction/mayr-n-publication-independent-v1/configs/"
    "mayr_n_publication_retrained_v1.toml"
)
STANDARD_ABLATIONS = {
    "without_solvent": ROOT
    / "configs/mayr_n_publication_ablation_without_solvent_v1.toml",
    "without_site_type": ROOT
    / "configs/mayr_n_publication_ablation_without_site_type_v1.toml",
}
AGGREGATE_SCHEMA = "nucpred.mayr-node-xtb-pretraining-aggregate.v1"
MANIFEST_SCHEMA = "nucpred.mayr-n-independent-retraining-config-manifest.v1"


class RetrainingConfigError(RuntimeError):
    """Raised when regenerated weights cannot be safely rebound."""


def _project_path(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError as exc:
        raise RetrainingConfigError(f"{label} escapes the project root") from exc
    return resolved


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _toml_string(value: object) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def _replace_key(
    text: str,
    *,
    section: str | None,
    key: str,
    replacement: str,
) -> str:
    lines = text.splitlines()
    current: str | None = None
    matches = 0
    key_pattern = re.compile(rf"^(\s*){re.escape(key)}\s*=.*$")
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            current = stripped
        expected = section if section is None else f"[{section}]"
        if current == expected or (section is None and current is None):
            match = key_pattern.match(line)
            if match:
                lines[index] = f"{match.group(1)}{key} = {replacement}"
                matches += 1
    if matches != 1:
        raise RetrainingConfigError(
            f"Expected one {section or 'top-level'}.{key} binding, found {matches}"
        )
    return "\n".join(lines) + "\n"


def _replace_literal_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RetrainingConfigError(f"Expected one {label} binding, found {count}")
    return text.replace(old, new, 1)


def inspect_standard_pretraining(
    aggregate_path: str | Path = DEFAULT_AGGREGATE,
    *,
    parent_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate an ESNUEL-only aggregate and its three checkpoint contracts."""

    aggregate = _project_path(aggregate_path, label="aggregate")
    payload = json.loads(aggregate.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != AGGREGATE_SCHEMA
        or payload.get("status") != "pass"
        or payload.get("scope") != "full"
        or int(payload.get("dataset_record_count", -1)) != 47_915
        or payload.get("dataset_id")
        != "esnuel-d-node-xtb-pretraining-20260726-v1-full"
    ):
        raise RetrainingConfigError("Regenerated aggregate is not the full ESNUEL run")
    raw_bindings = payload.get("checkpoint_bindings")
    if not isinstance(raw_bindings, Mapping):
        raise RetrainingConfigError("Aggregate lacks checkpoint bindings")

    bindings: list[dict[str, object]] = []
    for entry in parent_config["pretraining"]["checkpoints"]:
        downstream_seed = int(entry["downstream_initialization_seed"])
        pretraining_seed = int(entry["pretraining_seed"])
        raw = raw_bindings.get(str(pretraining_seed))
        if not isinstance(raw, Mapping):
            raise RetrainingConfigError(
                f"Aggregate lacks pretraining seed {pretraining_seed}"
            )
        checkpoint = _project_path(str(raw.get("path")), label="checkpoint")
        observed = sha256_file(checkpoint)
        if observed != str(raw.get("sha256")):
            raise RetrainingConfigError(
                f"Aggregate checkpoint hash drifted for seed {pretraining_seed}"
            )
        checkpoint_payload = load_pretraining_checkpoint(checkpoint)
        contract = checkpoint_payload.get("contract")
        if not isinstance(contract, Mapping):
            raise RetrainingConfigError("Checkpoint lacks a pretraining contract")
        if (
            contract.get("data_schema_version")
            != "mayr-node-xtb-esnuel-pretraining-batch.v1"
            or frozenset(map(str, contract.get("tasks", ())))
            != modeling.EXPECTED_PRETRAINING_TASKS
            or int(checkpoint_payload.get("init_seed", -1)) != pretraining_seed
        ):
            raise RetrainingConfigError(
                f"Checkpoint contract changed for seed {pretraining_seed}"
            )
        bindings.append(
            {
                "downstream_initialization_seed": downstream_seed,
                "pretraining_seed": pretraining_seed,
                "path": _relative(checkpoint),
                "sha256": observed,
            }
        )
    if len(bindings) != 3:
        raise RetrainingConfigError("Expected exactly three checkpoint bindings")
    return {
        "aggregate_path": _relative(aggregate),
        "aggregate_sha256": sha256_file(aggregate),
        "dataset_manifest_sha256": payload.get("dataset_manifest_sha256"),
        "checkpoints": bindings,
    }


def render_retrained_parent(
    parent_text: str,
    parent_config: Mapping[str, Any],
    inspected: Mapping[str, Any],
    *,
    reproduction_id: str,
    output_directory: str,
    parent_path: str,
    parent_sha256: str,
) -> str:
    """Render a full config while changing only local-output and weight bindings."""

    text = _replace_key(
        parent_text,
        section=None,
        key="campaign_id",
        replacement=_toml_string(reproduction_id),
    )
    text = _replace_key(
        text,
        section=None,
        key="experiment_id",
        replacement=_toml_string(f"{reproduction_id}-nested"),
    )
    text = _replace_key(
        text,
        section=None,
        key="output_directory",
        replacement=_toml_string(output_directory),
    )
    text = _replace_key(
        text,
        section="lineage",
        key="pretraining_aggregate_path",
        replacement=_toml_string(inspected["aggregate_path"]),
    )
    text = _replace_key(
        text,
        section="lineage",
        key="pretraining_aggregate_sha256",
        replacement=_toml_string(inspected["aggregate_sha256"]),
    )
    by_downstream = {
        int(value["downstream_initialization_seed"]): value
        for value in inspected["checkpoints"]
    }
    for original in parent_config["pretraining"]["checkpoints"]:
        seed = int(original["downstream_initialization_seed"])
        replacement = by_downstream.get(seed)
        if replacement is None:
            raise RetrainingConfigError(f"No regenerated binding for seed {seed}")
        old_path = f'path = {_toml_string(original["path"])}'
        new_path = f'path = {_toml_string(replacement["path"])}'
        if old_path != new_path:
            text = _replace_literal_once(
                text, old_path, new_path, label=f"checkpoint path {seed}"
            )
        text = _replace_literal_once(
            text,
            f'sha256 = {_toml_string(original["sha256"])}',
            f'sha256 = {_toml_string(replacement["sha256"])}',
            label=f"checkpoint hash {seed}",
        )
    text += (
        "\n[reproduction]\n"
        f"source_config_path = {_toml_string(parent_path)}\n"
        f"source_config_sha256 = {_toml_string(parent_sha256)}\n"
        'binding_mode = "locally_regenerated_esnuel_pretraining"\n'
        "independent_replication = true\n"
        "frozen_paper_metrics_claimed = false\n"
    )
    parsed = tomllib.loads(text)
    if parsed.get("schema_version") != modeling.CONFIG_SCHEMA:
        raise RetrainingConfigError("Rendered config changed the experiment schema")
    return text


def _render_ablation(
    *,
    parent_path: str,
    parent_sha256: str,
    output_directory: str,
    reproduction_id: str,
    source: Mapping[str, Any],
) -> str:
    ablation = source["ablation"]
    name = str(ablation["name"])
    lines = [
        f'schema_version = "{modeling.ABLATION_CONFIG_SCHEMA}"',
        f"parent_config_path = {_toml_string(parent_path)}",
        f"parent_config_sha256 = {_toml_string(parent_sha256)}",
        f"campaign_id = {_toml_string(reproduction_id)}",
        f"experiment_id = {_toml_string(f'{reproduction_id}-{name}')}",
        f"output_directory = {_toml_string(f'{output_directory}/ablations/{name}')}",
        f"device = {_toml_string(source.get('device', 'cuda:0'))}",
        "",
        "[ablation]",
    ]
    for key, value in ablation.items():
        if isinstance(value, bool):
            rendered = "true" if value else "false"
        elif isinstance(value, str):
            rendered = _toml_string(value)
        else:
            raise RetrainingConfigError(f"Unsupported ablation value: {key}")
        lines.append(f"{key} = {rendered}")
    return "\n".join(lines) + "\n"


def render_retrained_site_parent(
    parent_text: str,
    *,
    reproduction_id: str,
    output_directory: str,
    conditional_config_path: str,
    conditional_config_sha256: str,
    parent_path: str,
    parent_sha256: str,
) -> str:
    """Bind an automatic-site replication to regenerated conditional outputs."""

    text = _replace_key(
        parent_text,
        section=None,
        key="campaign_id",
        replacement=_toml_string(reproduction_id),
    )
    text = _replace_key(
        text,
        section=None,
        key="experiment_id",
        replacement=_toml_string(f"{reproduction_id}-automatic-site"),
    )
    text = _replace_key(
        text,
        section=None,
        key="output_directory",
        replacement=_toml_string(f"{output_directory}/automatic_site"),
    )
    lineage_replacements = {
        "conditional_n_config_path": conditional_config_path,
        "conditional_n_config_sha256": conditional_config_sha256,
        "inner_conditional_checkpoint_root": f"{output_directory}/nested_inner",
        "outer_conditional_checkpoint_root": f"{output_directory}/outer_refit",
    }
    for key, value in lineage_replacements.items():
        text = _replace_key(
            text,
            section="lineage",
            key=key,
            replacement=_toml_string(value),
        )
    text += (
        "\n[reproduction]\n"
        f"source_config_path = {_toml_string(parent_path)}\n"
        f"source_config_sha256 = {_toml_string(parent_sha256)}\n"
        'binding_mode = "locally_regenerated_conditional_n"\n'
        "independent_replication = true\n"
        "frozen_paper_metrics_claimed = false\n"
    )
    if tomllib.loads(text).get("schema_version") != site.CONFIG_SCHEMA:
        raise RetrainingConfigError("Rendered site config changed its schema")
    return text


def render_retrained_baseline_config(
    parent_text: str,
    *,
    reproduction_id: str,
    output_directory: str,
    parent_path: str,
    parent_sha256: str,
) -> str:
    """Bind baseline evaluation to one independent model-output family."""

    text = _replace_key(
        parent_text,
        section=None,
        key="campaign_id",
        replacement=_toml_string(reproduction_id),
    )
    text = _replace_key(
        text,
        section=None,
        key="output_directory",
        replacement=_toml_string(f"{output_directory}/baselines"),
    )
    text += (
        "\n[comparators]\n"
        "oracle_summary_path = "
        f"{_toml_string(f'{output_directory}/oracle_evaluation/summary.json')}\n"
        "oracle_predictions_path = "
        f"{_toml_string(f'{output_directory}/oracle_evaluation/oracle_oof_predictions_with_labels.parquet')}\n"
        "automatic_summary_path = "
        f"{_toml_string(f'{output_directory}/automatic_site/outer_evaluation/summary.json')}\n"
        "\n[reproduction]\n"
        f"source_config_path = {_toml_string(parent_path)}\n"
        f"source_config_sha256 = {_toml_string(parent_sha256)}\n"
        'binding_mode = "independent_model_output_family"\n'
        "independent_replication = true\n"
        "frozen_paper_metrics_claimed = false\n"
    )
    if tomllib.loads(text).get("schema_version") != "nucpred.mayr-n-publication-baselines.v1":
        raise RetrainingConfigError("Rendered baseline config changed its schema")
    return text


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def generate_retraining_configs(
    *,
    parent_path: str | Path = DEFAULT_PARENT,
    aggregate_path: str | Path = DEFAULT_AGGREGATE,
    site_parent_path: str | Path = DEFAULT_SITE_PARENT,
    baseline_parent_path: str | Path = DEFAULT_BASELINE_PARENT,
    output_config: str | Path = DEFAULT_OUTPUT,
    reproduction_id: str = "mayr-n-publication-independent-v1",
) -> dict[str, Any]:
    parent = _project_path(parent_path, label="parent config")
    output = _project_path(output_config, label="output config")
    if output.exists():
        raise RetrainingConfigError(f"Refusing to overwrite generated config: {output}")
    parent_config = tomllib.loads(parent.read_text(encoding="utf-8"))
    if parent_config.get("schema_version") != modeling.CONFIG_SCHEMA:
        raise RetrainingConfigError("Parent is not a full publication config")
    inspected = inspect_standard_pretraining(
        aggregate_path, parent_config=parent_config
    )
    output_directory = f"artifacts/reproduction/{reproduction_id}/modeling"
    parent_relative = _relative(parent)
    parent_hash = sha256_file(parent)
    rendered = render_retrained_parent(
        parent.read_text(encoding="utf-8"),
        parent_config,
        inspected,
        reproduction_id=reproduction_id,
        output_directory=output_directory,
        parent_path=parent_relative,
        parent_sha256=parent_hash,
    )
    _atomic_write_text(output, rendered)

    generated = [
        {"name": "main", "kind": "conditional_n", "path": _relative(output)}
    ]
    output_hash = sha256_file(output)
    for name, source_path in STANDARD_ABLATIONS.items():
        source = tomllib.loads(source_path.read_text(encoding="utf-8"))
        ablation_path = output.with_name(f"{output.stem}_{name}.toml")
        if ablation_path.exists():
            raise RetrainingConfigError(
                f"Refusing to overwrite generated config: {ablation_path}"
            )
        _atomic_write_text(
            ablation_path,
            _render_ablation(
                parent_path=_relative(output),
                parent_sha256=output_hash,
                output_directory=output_directory,
                reproduction_id=reproduction_id,
                source=source,
            ),
        )
        generated.append(
            {"name": name, "kind": "conditional_n", "path": _relative(ablation_path)}
        )

    site_parent = _project_path(site_parent_path, label="site parent config")
    site_output = output.with_name(f"{output.stem}_site.toml")
    if site_output.exists():
        raise RetrainingConfigError(
            f"Refusing to overwrite generated config: {site_output}"
        )
    _atomic_write_text(
        site_output,
        render_retrained_site_parent(
            site_parent.read_text(encoding="utf-8"),
            reproduction_id=reproduction_id,
            output_directory=output_directory,
            conditional_config_path=_relative(output),
            conditional_config_sha256=output_hash,
            parent_path=_relative(site_parent),
            parent_sha256=sha256_file(site_parent),
        ),
    )
    generated.append(
        {
            "name": "automatic_site",
            "kind": "automatic_site",
            "path": _relative(site_output),
        }
    )

    # Local import avoids a module-level cycle: the generic site-ablation
    # generator reuses this module's atomic TOML writer.
    from nucpred.publication import mayr_site_ablation_config as site_ablation

    for name in STANDARD_ABLATIONS:
        conditional_path = output.with_name(f"{output.stem}_{name}.toml")
        site_ablation_path = output.with_name(f"{output.stem}_site_{name}.toml")
        site_ablation.generate_site_ablation_config(
            conditional_path,
            parent_path=site_output,
            output_config=site_ablation_path,
            output_directory=(
                f"{output_directory}/automatic_site_ablations/{name}"
            ),
        )
        generated.append(
            {
                "name": f"automatic_site_{name}",
                "kind": "automatic_site",
                "path": _relative(site_ablation_path),
            }
        )

    baseline_parent = _project_path(
        baseline_parent_path, label="baseline parent config"
    )
    baseline_output = output.with_name(f"{output.stem}_baselines.toml")
    if baseline_output.exists():
        raise RetrainingConfigError(
            f"Refusing to overwrite generated config: {baseline_output}"
        )
    _atomic_write_text(
        baseline_output,
        render_retrained_baseline_config(
            baseline_parent.read_text(encoding="utf-8"),
            reproduction_id=reproduction_id,
            output_directory=output_directory,
            parent_path=_relative(baseline_parent),
            parent_sha256=sha256_file(baseline_parent),
        ),
    )
    generated.append(
        {
            "name": "baselines",
            "kind": "baselines",
            "path": _relative(baseline_output),
        }
    )

    # Reuse the formal loader and checkpoint auditor on every generated config.
    for entry in generated:
        if entry["kind"] == "conditional_n":
            config, resolved = modeling.read_config(ROOT / str(entry["path"]))
            for seed in map(int, config["outer_initialization_seeds"]):
                modeling._pretraining_entry(config, seed)
        elif entry["kind"] == "automatic_site":
            config, resolved = site.read_config(ROOT / str(entry["path"]))
            site.verify_bindings(config, resolved)
        else:
            config, resolved = baselines.read_config(ROOT / str(entry["path"]))
            baselines._comparator_paths(config)
        entry["sha256"] = sha256_file(resolved)

    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "status": "pass",
        "reproduction_id": reproduction_id,
        "independent_replication": True,
        "frozen_paper_metrics_claimed": False,
        "source_config_path": parent_relative,
        "source_config_sha256": parent_hash,
        "pretraining": inspected,
        "generated_configs": generated,
    }
    manifest_path = output.parent / "retraining_config_manifest.json"
    atomic_write_json(manifest_path, manifest, ensure_ascii=False)
    manifest["manifest_path"] = _relative(manifest_path)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-config", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--aggregate", type=Path, default=DEFAULT_AGGREGATE)
    parser.add_argument("--site-parent-config", type=Path, default=DEFAULT_SITE_PARENT)
    parser.add_argument(
        "--baseline-parent-config", type=Path, default=DEFAULT_BASELINE_PARENT
    )
    parser.add_argument("--output-config", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--reproduction-id", default="mayr-n-publication-independent-v1"
    )
    args = parser.parse_args(argv)
    result = generate_retraining_configs(
        parent_path=args.parent_config,
        aggregate_path=args.aggregate,
        site_parent_path=args.site_parent_config,
        baseline_parent_path=args.baseline_parent_config,
        output_config=args.output_config,
        reproduction_id=args.reproduction_id,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
