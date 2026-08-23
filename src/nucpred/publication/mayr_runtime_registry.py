"""Freeze and audit the publication all-data Mayr runtime registry."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import joblib
import pandas as pd
import torch

from nucpred.core.files import atomic_write_json, sha256_file
from nucpred.inference import mayr_nextgen_runtime as runtime
from nucpred.inference.mayr_nextgen_contract import RESPONSE_SCHEMA_PATH, SITE_TYPES
from nucpred.publication import mayr_site_publication as shared
from nucpred.publication.mayr_n_outer import load_outer_checkpoint
from nucpred.training.mayr_site_inference_assets import (
    canonical_sha256,
    ranker_from_checkpoint,
)


REGISTRY_BUILD_SCHEMA = "nucpred.mayr-n-publication-runtime-registry-build.v1"
PUBLICATION_PROTOCOL_CONFIG = shared.ROOT / "configs/mayr_n_publication_v1.toml"
CANDIDATE_GENERATOR = shared.ROOT / "src/nucpred/datasets/mayr_site_n.py"
CANDIDATE_ONTOLOGY = shared.ROOT / "configs/mayr_nextgen_gate_a.toml"


def _binding(
    path: Path,
    *,
    extras: Mapping[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "path": path.resolve().relative_to(shared.ROOT).as_posix(),
        "sha256": sha256_file(path),
        "bytes": int(path.stat().st_size),
    }
    if extras:
        payload.update(extras)
    return payload


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise shared.PublicationSiteError(f"Expected JSON object: {path}")
    return payload


def _model_assets(
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    output = shared.project_path(config["output_directory"], label="site output")
    ranker_path = output / "final_refit/site_ranker/ranker_checkpoint.pt"
    summary_path = output / "final_refit/site_ranker/summary.json"
    summary = _read_json(summary_path)
    checkpoint = torch.load(ranker_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise shared.PublicationSiteError("Final ranker checkpoint is not a mapping")
    expected = {
        "schema_version": "nucpred.mayr-n-publication-site-ranker-checkpoint.v1",
        "phase": "post_outer_evaluation_all_data_final_refit",
        "campaign_id": config["campaign_id"],
        "outer_fold": -1,
        "selected_arm": "hierarchical_exact",
        "final_refit_performed": True,
        "all_corrected_v2_targets_used": True,
        "reported_outer_metrics_modified": False,
        "candidate_softmax_used": False,
        "fixed_epoch_in_sample_monitoring_metrics_reported": False,
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise shared.PublicationSiteError(
                f"Final ranker checkpoint boundary changed: {key}"
            )
    ranker_from_checkpoint(checkpoint)
    if (
        summary.get("status") != "pass"
        or summary.get("ranker_checkpoint_sha256") != sha256_file(ranker_path)
        or summary.get("ranker_state_sha256")
        != checkpoint.get("ranker_state_sha256")
        or summary.get("external_sources_or_labels_used") is not False
    ):
        raise shared.PublicationSiteError("Final ranker summary changed")
    return checkpoint, summary, ranker_path, summary_path


def _conditional_bindings(
    checkpoint: Mapping[str, Any],
    *,
    conditional_config: Path,
) -> list[dict[str, object]]:
    raw = checkpoint.get("conditional_n_bindings")
    if not isinstance(raw, list) or len(raw) != 3:
        raise shared.PublicationSiteError("Final N ensemble must contain three models")
    audited: list[dict[str, object]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise shared.PublicationSiteError("Final N binding is invalid")
        path = shared.project_path(item["path"], label="final N checkpoint")
        if sha256_file(path) != item.get("sha256"):
            raise shared.PublicationSiteError("Final N checkpoint hash changed")
        model, preprocessor, _, payload = load_outer_checkpoint(
            path,
            config_path=conditional_config,
            device="cpu",
        )
        del model
        contract = payload.get("contract")
        if (
            not isinstance(contract, Mapping)
            or contract.get("all_corrected_v2_targets_used") is not True
            or contract.get("external_sources_or_labels_used") is not False
            or int(contract.get("target_count", -1)) != 1038
            or int(contract.get("initialization_seed", -1))
            != int(item["initialization_seed"])
            or payload.get("model_state_sha256") != item.get("model_state_sha256")
            or preprocessor.fit_context_count != 1032
            or preprocessor.fit_target_count != 1038
        ):
            raise shared.PublicationSiteError("Final N refit contract changed")
        audited.append(dict(item))
    if len({int(item["initialization_seed"]) for item in audited}) != 3:
        raise shared.PublicationSiteError("Final N initialization seeds overlap")
    return audited


def _region_binding(checkpoint: Mapping[str, Any]) -> dict[str, object]:
    raw = checkpoint.get("region_membership_residual")
    if not isinstance(raw, Mapping):
        raise shared.PublicationSiteError("Final region residual binding is missing")
    path = shared.project_path(raw["path"], label="final region residual")
    if sha256_file(path) != raw.get("sha256"):
        raise shared.PublicationSiteError("Final region residual hash changed")
    bundle = joblib.load(path)
    if (
        not isinstance(bundle, Mapping)
        or bundle.get("schema_version") != raw.get("schema_version")
        or bundle.get("feature_schema_version") != raw.get("feature_schema_version")
        or list(bundle.get("feature_names", ())) != list(raw.get("feature_names", ()))
        or list(bundle.get("origin_vocabulary", ()))
        != list(raw.get("origin_vocabulary", ()))
        or bundle.get("candidate_softmax_used") is not False
        or bundle.get("type_level_maximum_preserved") is not True
    ):
        raise shared.PublicationSiteError("Final region residual artifact changed")
    return dict(raw)


def build_registry(
    config_path: str | Path = shared.DEFAULT_CONFIG,
    *,
    output_directory: str | Path | None = None,
) -> dict[str, object]:
    config, resolved = shared.read_config(config_path)
    shared.verify_bindings(config, resolved)
    output = shared.project_path(config["output_directory"], label="site output")
    destination = (
        Path(output_directory).resolve()
        if output_directory is not None
        else output / "deployment"
    )
    if destination.exists():
        raise shared.PublicationSiteError("Refusing to overwrite runtime deployment")

    checkpoint, final_summary, ranker_path, final_summary_path = _model_assets(config)
    conditional_config = shared.project_path(
        config["lineage"]["conditional_n_config_path"], label="conditional N config"
    )
    conditional_bindings = _conditional_bindings(
        checkpoint,
        conditional_config=conditional_config,
    )
    residual = _region_binding(checkpoint)
    margin = checkpoint.get("margin_abstention")
    if (
        not isinstance(margin, Mapping)
        or margin.get("constraint_met") is not True
        or margin.get("selection_uses_outer_oof_labels") is not True
        or margin.get("reported_outer_metrics_modified") is not False
        or margin.get("undefined_singleton_policy")
        != "abstain_margin_undefined"
    ):
        raise shared.PublicationSiteError("Final margin gate changed")

    selection_path = output / "final_selection/selection.json"
    selection = _read_json(selection_path)
    oracle_evaluation = output.parent / "oracle_evaluation/summary.json"
    site_evaluation = output / "outer_evaluation/summary.json"
    if (
        selection.get("status") != "frozen"
        or selection.get("selection_uses_outer_test_metrics") is not False
        or selection.get("selection_uses_external_labels") is not False
        or _read_json(oracle_evaluation).get("status") != "complete"
        or _read_json(site_evaluation).get("status") != "pass"
    ):
        raise shared.PublicationSiteError("Sealed evaluation or final selection changed")

    dataset = shared.project_path(config["dataset"]["directory"], label="dataset")
    species = pd.read_parquet(dataset / "species.parquet")
    deployment_candidates, candidate_audit = shared.deployment_candidates(
        config, species
    )
    if len(deployment_candidates) != 119402:
        raise shared.PublicationSiteError("Publication deployment candidates changed")

    registry: dict[str, object] = {
        "schema_version": runtime.PUBLICATION_RUNTIME_REGISTRY_SCHEMA,
        "campaign_id": config["campaign_id"],
        "created_at_utc": datetime.now(UTC).isoformat(),
        "runtime_mode": "publication_all_data_final_refit",
        "dataset_directory": config["dataset"]["directory"],
        "dataset_manifest_sha256": sha256_file(dataset / "dataset_manifest.json"),
        "candidate_types": list(SITE_TYPES),
        "candidate_policy_filter": (
            "corrected_v2_label_independent_deployment_and_response_contract"
        ),
        "deployment_candidate_count": len(deployment_candidates),
        "deployment_candidate_audit": candidate_audit,
        "publication_config_binding": _binding(resolved),
        "publication_protocol_config_binding": _binding(
            PUBLICATION_PROTOCOL_CONFIG
        ),
        "conditional_n_config_binding": _binding(conditional_config),
        "candidate_table_binding": _binding(
            shared.project_path(
                config["dataset"]["candidate_path"], label="candidate table"
            )
        ),
        "candidate_generator_binding": _binding(CANDIDATE_GENERATOR),
        "candidate_ontology_binding": _binding(CANDIDATE_ONTOLOGY),
        "candidate_policy_source_binding": _binding(
            shared.project_path(
                config["lineage"]["candidate_policy_source_path"],
                label="candidate policy source",
            )
        ),
        "runtime_source_binding": _binding(Path(runtime.__file__).resolve()),
        "registry_builder_binding": _binding(Path(__file__).resolve()),
        "response_schema_path": RESPONSE_SCHEMA_PATH.relative_to(shared.ROOT).as_posix(),
        "response_schema_sha256": sha256_file(RESPONSE_SCHEMA_PATH),
        "final_selection_binding": _binding(selection_path),
        "oracle_outer_evaluation_binding": _binding(oracle_evaluation),
        "automatic_site_outer_evaluation_binding": _binding(site_evaluation),
        "final_site_summary_binding": _binding(final_summary_path),
        "publication_model": {
            "ranker_checkpoint": _binding(
                ranker_path,
                extras={
                    "ranker_state_sha256": checkpoint["ranker_state_sha256"],
                    "deployment_member_count": 1,
                    "logit_std_semantics": (
                        "zero_over_one_deterministic_all_data_ranker_not_uncertainty"
                    ),
                },
            ),
            "conditional_n_bindings": conditional_bindings,
            "conditional_n_ensemble_member_count": 3,
            "region_membership_residual": residual,
            "margin_abstention": dict(margin),
            "internal_oof_canonical_endpoint_calibrator": checkpoint["calibrator"],
            "internal_calibrator_not_absolute_probability": True,
        },
        "conditional_n_ensemble_semantics": (
            "mean_and_population_std_of_three_all_data_initializations"
        ),
        "validity_ensemble_semantics": (
            "single_deterministic_all_data_ranker_over_three_backbone_features"
        ),
        "candidate_scores_independent": True,
        "candidate_softmax_used": False,
        "candidate_set_conditioned_structural_residual": True,
        "region_type_level_maximum_preserved": True,
        "conditional_n_backbone_frozen_during_ranker_training": True,
        "final_refit_performed": True,
        "all_corrected_v2_targets_used": True,
        "final_refit_target_count": int(final_summary["target_count"]),
        "target_or_site_label_read_at_inference": False,
        "no_site_claim_permitted": False,
        "margin_abstention_enabled": True,
        "margin_threshold_aggregation": "post_evaluation_outer_oof_global",
        "runtime_margin_threshold": float(margin["selected_threshold"]),
        "singleton_margin_policy": margin["undefined_singleton_policy"],
        "low_margin_runtime_status": "partial_uncertain",
        "absolute_site_probability_enabled": False,
        "absolute_site_probability_unavailable_reason": (
            "external_absolute_probability_calibration_not_established"
        ),
        "internal_calibration_semantics": (
            "cross_fitted_probability_of_reported_canonical_endpoint_within_"
            "the_enumerated_mayr_like_candidate_set"
        ),
        "external_confirmation_status_at_freeze": "not_started",
        "external_sources_or_labels_used_before_registry_freeze": False,
        "external_search_permitted_only_after_registry_freeze": True,
        "code_public_intent": True,
        "source_data_public_intent": True,
        "curated_training_data_public_intent": False,
        "data_release_rights_audit_status": "confirmed_for_scoped_source_data",
        "weights_public": True,
        "weight_license": "Apache-2.0",
        "weights_confidential_editor_reviewer_access": False,
    }
    registry["registry_sha256"] = canonical_sha256(registry)

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".deployment.staging-", dir=destination.parent)
    )
    try:
        atomic_write_json(staging / "runtime_registry.json", registry)
        loaded = runtime._load_registry(staging / "runtime_registry.json")
        if loaded.get("registry_sha256") != registry["registry_sha256"]:
            raise shared.PublicationSiteError("Publication registry round-trip changed")
        summary: dict[str, object] = {
            "schema_version": REGISTRY_BUILD_SCHEMA,
            "status": "pass",
            "campaign_id": config["campaign_id"],
            "registry_path": (destination / "runtime_registry.json")
            .relative_to(shared.ROOT)
            .as_posix()
            if destination.is_relative_to(shared.ROOT)
            else (destination / "runtime_registry.json").as_posix(),
            "registry_file_sha256": sha256_file(staging / "runtime_registry.json"),
            "registry_internal_sha256": registry["registry_sha256"],
            "registered_conditional_n_model_count": len(conditional_bindings),
            "registered_ranker_count": 1,
            "registered_region_residual_count": 1,
            "target_or_site_label_read_at_inference": False,
            "absolute_site_probability_enabled": False,
            "external_confirmation_status_at_freeze": "not_started",
            "source_path": Path(__file__).resolve().relative_to(shared.ROOT).as_posix(),
            "source_sha256": sha256_file(Path(__file__).resolve()),
            "completed_at_utc": datetime.now(UTC).isoformat(),
        }
        atomic_write_json(staging / "summary.json", summary)
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=shared.DEFAULT_CONFIG)
    parser.add_argument("--output-directory", type=Path)
    args = parser.parse_args(argv)
    result = build_registry(args.config, output_directory=args.output_directory)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
