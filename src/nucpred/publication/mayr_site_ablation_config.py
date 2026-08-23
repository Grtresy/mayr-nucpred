"""Generate hash-bound automatic-site configs for matched input ablations."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import tomllib
from typing import Any

from nucpred.core.files import sha256_file
from nucpred.project import get_project_layout
from nucpred.publication import mayr_n_modeling as conditional
from nucpred.publication import mayr_site_publication as site
from nucpred.publication.mayr_retraining_config import (
    _atomic_write_text,
    _relative,
    _toml_string,
)


ROOT = get_project_layout().root
DEFAULT_PARENT = ROOT / "configs/mayr_n_publication_site_v1.toml"


class SiteAblationConfigError(RuntimeError):
    """Raised when an automatic-site ablation cannot be bound exactly."""


def _project_path(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError as exc:
        raise SiteAblationConfigError(f"{label} escapes the project root") from exc
    return resolved


def _variant_contract(
    conditional_config_path: str | Path,
) -> tuple[dict[str, Any], Path, str]:
    config, resolved = conditional.read_config(conditional_config_path)
    ablation = config.get("ablation")
    if not isinstance(ablation, Mapping):
        raise SiteAblationConfigError("Conditional config is not an ablation")
    name = str(ablation.get("name"))
    if name not in conditional.ABLATION_NAMES:
        raise SiteAblationConfigError("Conditional ablation identity changed")
    return config, resolved, name


def render_site_ablation_config(
    *,
    parent_path: Path,
    conditional_config: Mapping[str, Any],
    conditional_path: Path,
    name: str,
    experiment_id: str,
    output_directory: str,
) -> str:
    conditional_root = str(conditional_config["output_directory"]).rstrip("/")
    without_type = name == "without_site_type"
    interpretation = str(conditional_config["ablation"]["interpretation"])
    lines = [
        f'schema_version = "{site.ABLATION_CONFIG_SCHEMA}"',
        f"parent_config_path = {_toml_string(_relative(parent_path))}",
        f"parent_config_sha256 = {_toml_string(sha256_file(parent_path))}",
        f"campaign_id = {_toml_string(str(conditional_config['campaign_id']))}",
        f"experiment_id = {_toml_string(experiment_id)}",
        f"output_directory = {_toml_string(output_directory)}",
        f"device = {_toml_string(str(conditional_config.get('device', 'cuda:0')))}",
        "",
        "[lineage_overrides]",
        f"conditional_n_config_path = {_toml_string(_relative(conditional_path))}",
        f"conditional_n_config_sha256 = {_toml_string(sha256_file(conditional_path))}",
        "inner_conditional_checkpoint_root = "
        f"{_toml_string(conditional_root + '/nested_inner')}",
        "outer_conditional_checkpoint_root = "
        f"{_toml_string(conditional_root + '/outer_refit')}",
        "",
        "[ablation]",
        f"name = {_toml_string(name)}",
        f"interpretation = {_toml_string(interpretation)}",
        "conditional_n_input_ablation_matched = true",
        "ranker_architecture_and_optimization_matched = true",
        "frozen_candidate_universe_and_mining_retained = true",
        "true_metadata_retained_only_outside_predictor = true",
        "outer_test_used_for_selection = false",
        "model_facing_site_type = "
        + _toml_string("constant_atom" if without_type else "frozen_true_metadata"),
        "type_dependent_region_residual_enabled = "
        + ("false" if without_type else "true"),
        "type_dependent_calibration_enabled = " + ("false" if without_type else "true"),
        "type_router_supervision = "
        + _toml_string("constant_atom" if without_type else "frozen_true_metadata"),
    ]
    rendered = "\n".join(lines) + "\n"
    parsed = tomllib.loads(rendered)
    if parsed["ablation"]["name"] != name:
        raise SiteAblationConfigError("Rendered ablation identity changed")
    return rendered


def generate_site_ablation_config(
    conditional_config_path: str | Path,
    *,
    parent_path: str | Path = DEFAULT_PARENT,
    output_config: str | Path | None = None,
    output_directory: str | None = None,
) -> dict[str, object]:
    parent = _project_path(parent_path, label="site parent config")
    parent_config, parent_resolved = site.read_config(parent)
    if parent_config.get("ablation") is not None:
        raise SiteAblationConfigError("Site parent must be the unabated protocol")
    conditional_config, conditional_path, name = _variant_contract(
        conditional_config_path
    )
    destination = (
        _project_path(output_config, label="output config")
        if output_config is not None
        else ROOT / f"configs/mayr_n_publication_site_ablation_{name}_v1.toml"
    )
    site_output = output_directory or (
        "artifacts/campaigns/mayr-n-publication-20260805-v1/modeling/"
        f"automatic_site_ablations/{name}"
    )
    experiment_id = (
        "mayr-n-publication-automatic-site-ablation-"
        f"{name.replace('_', '-')}-20260805-v1"
    )
    rendered = render_site_ablation_config(
        parent_path=parent_resolved,
        conditional_config=conditional_config,
        conditional_path=conditional_path,
        name=name,
        experiment_id=experiment_id,
        output_directory=site_output,
    )
    if destination.exists():
        raise SiteAblationConfigError(f"Refusing to overwrite {destination}")
    _atomic_write_text(destination, rendered)
    parsed, resolved = site.read_config(destination)
    site.verify_bindings(parsed, resolved)
    return {
        "schema_version": "nucpred.mayr-n-publication-site-ablation-config-manifest.v1",
        "status": "frozen",
        "ablation": name,
        "config_path": _relative(destination),
        "config_sha256": sha256_file(destination),
        "parent_config_path": _relative(parent_resolved),
        "parent_config_sha256": sha256_file(parent_resolved),
        "conditional_config_path": _relative(conditional_path),
        "conditional_config_sha256": sha256_file(conditional_path),
        "output_directory": str(parsed["output_directory"]),
        "true_site_type_available_to_predictor": not (name == "without_site_type"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditional-config", type=Path, required=True)
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--output-config", type=Path)
    parser.add_argument("--output-directory")
    args = parser.parse_args(argv)
    result = generate_site_ablation_config(
        args.conditional_config,
        parent_path=args.parent,
        output_config=args.output_config,
        output_directory=args.output_directory,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
