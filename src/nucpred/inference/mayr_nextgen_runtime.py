"""Target-blind registry-backed Mayr next-generation inference runtime."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
import json
from pathlib import Path
import tomllib
from typing import Any

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
import torch

from nucpred.core.files import atomic_write_json, sha256_file
from nucpred.datasets.mayr_site_candidate_policy import (
    CandidatePolicyError,
    select_deployment_candidates,
)
from nucpred.datasets.mayr_site_n import enumerate_candidate_sites
from nucpred.inference.mayr_external_context import (
    DEFAULT_REFERENCE_CONTEXTS,
    DEFAULT_XTB_CONFIG,
    ExternalMayrContextError,
    build_external_pure_contexts,
)
from nucpred.inference.mayr_nextgen_contract import (
    RESPONSE_SCHEMA_PATH,
    RESPONSE_SCHEMA_VERSION,
    SITE_TYPES,
    validate_request,
    validate_response,
)
from nucpred.project import get_project_layout
from nucpred.publication import mayr_site_publication as publication_site
from nucpred.publication.mayr_n_outer import load_outer_checkpoint
from nucpred.training.mayr_site_inference_assets import (
    RUNTIME_REGISTRY_SCHEMA,
    candidate_universe as _candidate_universe,
    canonical_sha256 as _canonical_sha256,
    deployment_candidates as _deployment_candidates,
    encode_split_ensemble as _encode_split_ensemble,
    load_ranker_checkpoint as _load_ranker_checkpoint,
    ranker_from_checkpoint as _ranker_from_checkpoint,
    read_site_identification_config as _read_legacy_config,
    score_ranker_from_source_features as _score_ranker_from_source_features,
)
from nucpred.training.mayr_site_ranker import (
    TypeAwarePlattCalibrator,
    site_type_indices,
)
from nucpred.training.mayr_site_region_residual import (
    RegionResidualError,
    apply_region_residual,
    region_feature_matrix,
    score_region_residual,
)


ROOT = get_project_layout().root
DEFAULT_CONFIG = ROOT / "configs/mayr_nextgen_site_identification_v7.toml"
DEFAULT_REGISTRY = (
    ROOT
    / "artifacts/campaigns/mayr-nextgen-site-identification-20260731-v7"
    / "deployment/runtime_registry.json"
)
DEFAULT_UNSEEN_FEATURE_CACHE = (
    ROOT / "data/interim/mayr_unseen_runtime/mayr-unseen-runtime-v1"
)
PUBLICATION_RUNTIME_REGISTRY_SCHEMA = "nucpred.mayr-n-publication-runtime-registry.v1"
DEFAULT_PUBLICATION_CONFIG = ROOT / "configs/mayr_n_publication_site_v1.toml"
DEFAULT_PUBLICATION_REGISTRY = (
    ROOT
    / "artifacts/campaigns/mayr-n-publication-20260805-v1/modeling/automatic_site"
    / "deployment/runtime_registry.json"
)


class MayrNextgenRuntimeError(RuntimeError):
    """Raised when a registered inference asset violates its frozen contract."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MayrNextgenRuntimeError(f"Cannot read JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise MayrNextgenRuntimeError(f"Expected JSON object: {path}")
    return payload


def _repo_path(raw: object, *, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise MayrNextgenRuntimeError(f"{label} must be repo-relative")
    path = Path(raw)
    if path.is_absolute():
        raise MayrNextgenRuntimeError(f"{label} must be repo-relative")
    resolved = (ROOT / path).resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise MayrNextgenRuntimeError(f"{label} escapes repository") from exc
    return resolved


def _read_runtime_config(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    try:
        with resolved.open("rb") as handle:
            schema = tomllib.load(handle).get("schema_version")
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise MayrNextgenRuntimeError(
            f"Cannot read runtime config: {resolved}"
        ) from exc
    if schema == publication_site.CONFIG_SCHEMA:
        config, _ = publication_site.read_config(resolved)
        return config
    return _read_legacy_config(resolved)


def _publication_registry(registry: Mapping[str, Any]) -> bool:
    return registry.get("schema_version") == PUBLICATION_RUNTIME_REGISTRY_SCHEMA


def _absolute_probability_enabled(registry: Mapping[str, Any] | None) -> bool:
    if registry is None:
        return False
    if _publication_registry(registry):
        return registry.get("absolute_site_probability_enabled") is True
    return True


def _verify_registered_file(
    binding: Mapping[str, Any],
    *,
    label: str,
) -> Path:
    path = _repo_path(binding["path"], label=label)
    if sha256_file(path) != binding.get("sha256"):
        raise MayrNextgenRuntimeError(f"Registered {label} changed")
    return path


def _validate_publication_registry(registry: dict[str, Any]) -> dict[str, Any]:
    semantic_boundary = {
        "runtime_mode": "publication_all_data_final_refit",
        "final_refit_performed": True,
        "all_corrected_v2_targets_used": True,
        "target_or_site_label_read_at_inference": False,
        "candidate_scores_independent": True,
        "candidate_softmax_used": False,
        "no_site_claim_permitted": False,
        "absolute_site_probability_enabled": False,
        "absolute_site_probability_unavailable_reason": (
            "external_absolute_probability_calibration_not_established"
        ),
    }
    for key, expected in semantic_boundary.items():
        if registry.get(key) != expected:
            raise MayrNextgenRuntimeError(
                f"Publication runtime semantic boundary changed: {key}"
            )
    if tuple(registry.get("candidate_types", ())) != SITE_TYPES:
        raise MayrNextgenRuntimeError("Publication candidate type order changed")

    dataset_root = _repo_path(registry["dataset_directory"], label="runtime dataset")
    if sha256_file(dataset_root / "dataset_manifest.json") != registry.get(
        "dataset_manifest_sha256"
    ):
        raise MayrNextgenRuntimeError("Publication dataset manifest changed")
    if sha256_file(RESPONSE_SCHEMA_PATH) != registry.get("response_schema_sha256"):
        raise MayrNextgenRuntimeError("Publication response schema changed")
    for key, label in (
        ("publication_config_binding", "publication config"),
        ("publication_protocol_config_binding", "publication protocol config"),
        ("conditional_n_config_binding", "conditional-N config"),
        ("candidate_table_binding", "candidate table"),
        ("candidate_generator_binding", "candidate generator"),
        ("candidate_ontology_binding", "candidate ontology"),
        ("candidate_policy_source_binding", "candidate policy source"),
        ("runtime_source_binding", "runtime source"),
        ("registry_builder_binding", "registry builder"),
        ("final_selection_binding", "final selection"),
        ("oracle_outer_evaluation_binding", "oracle outer evaluation"),
        ("automatic_site_outer_evaluation_binding", "automatic-site outer evaluation"),
        ("final_site_summary_binding", "final site summary"),
    ):
        binding = registry.get(key)
        if not isinstance(binding, Mapping):
            raise MayrNextgenRuntimeError(f"Missing publication binding: {key}")
        _verify_registered_file(binding, label=label)

    model = registry.get("publication_model")
    if not isinstance(model, Mapping):
        raise MayrNextgenRuntimeError("Publication model binding is missing")
    ranker_binding = model.get("ranker_checkpoint")
    if not isinstance(ranker_binding, Mapping):
        raise MayrNextgenRuntimeError("Publication ranker binding is missing")
    _verify_registered_file(ranker_binding, label="publication ranker checkpoint")
    backbones = model.get("conditional_n_bindings")
    if not isinstance(backbones, list) or len(backbones) != 3:
        raise MayrNextgenRuntimeError(
            "Publication runtime must register three conditional-N models"
        )
    seeds = []
    for binding in backbones:
        if not isinstance(binding, Mapping):
            raise MayrNextgenRuntimeError("Invalid publication N binding")
        _verify_registered_file(binding, label="publication conditional-N checkpoint")
        seeds.append(int(binding["initialization_seed"]))
    if len(set(seeds)) != 3:
        raise MayrNextgenRuntimeError("Publication N initialization seeds overlap")
    residual = model.get("region_membership_residual")
    if (
        not isinstance(residual, Mapping)
        or residual.get("enabled") is not True
        or residual.get("candidate_set_conditioned") is not True
        or residual.get("type_level_maximum_preserved") is not True
        or residual.get("candidate_softmax_used") is not False
    ):
        raise MayrNextgenRuntimeError("Publication region residual boundary changed")
    _verify_registered_file(residual, label="publication region residual")

    threshold = registry.get("runtime_margin_threshold")
    if (
        registry.get("margin_abstention_enabled") is not True
        or not isinstance(threshold, (int, float))
        or not np.isfinite(float(threshold))
        or float(threshold) < 0
        or registry.get("margin_threshold_aggregation")
        != "post_evaluation_outer_oof_global"
        or registry.get("singleton_margin_policy") != "abstain_margin_undefined"
        or registry.get("low_margin_runtime_status") != "partial_uncertain"
    ):
        raise MayrNextgenRuntimeError("Publication runtime margin gate changed")
    return registry


def _load_registry(path: Path) -> dict[str, Any]:
    registry = _load_json(path)
    if registry.get("schema_version") not in {
        RUNTIME_REGISTRY_SCHEMA,
        PUBLICATION_RUNTIME_REGISTRY_SCHEMA,
    }:
        raise MayrNextgenRuntimeError("Runtime registry schema changed")
    claimed = str(registry.get("registry_sha256"))
    content = dict(registry)
    content.pop("registry_sha256", None)
    if _canonical_sha256(content) != claimed:
        raise MayrNextgenRuntimeError("Runtime registry internal hash changed")
    if _publication_registry(registry):
        return _validate_publication_registry(registry)
    if (
        registry.get("target_or_site_label_read_at_inference") is not False
        or registry.get("candidate_scores_independent") is not True
        or registry.get("candidate_softmax_used") is not False
        or registry.get("final_refit_performed") is not False
        or tuple(registry.get("candidate_types", ())) != SITE_TYPES
    ):
        raise MayrNextgenRuntimeError("Runtime registry semantic boundary changed")
    dataset_root = _repo_path(
        registry["dataset_directory"],
        label="runtime dataset",
    )
    manifest_path = dataset_root / "dataset_manifest.json"
    if sha256_file(manifest_path) != registry["dataset_manifest_sha256"]:
        raise MayrNextgenRuntimeError("Runtime dataset manifest changed")
    if sha256_file(RESPONSE_SCHEMA_PATH) != registry["response_schema_sha256"]:
        raise MayrNextgenRuntimeError("Runtime response schema changed")
    generator_path = _repo_path(
        registry["candidate_generator_path"],
        label="candidate generator",
    )
    if sha256_file(generator_path) != registry["candidate_generator_sha256"]:
        raise MayrNextgenRuntimeError("Runtime candidate generator changed")
    policy_path = _repo_path(
        registry["candidate_policy_path"],
        label="candidate policy",
    )
    if (
        sha256_file(policy_path) != registry["candidate_policy_sha256"]
        or registry.get("candidate_policy_filter")
        != "gate_a_deployment_and_response_membership_contract"
    ):
        raise MayrNextgenRuntimeError("Runtime candidate policy changed")
    split_models = registry.get("split_models")
    if not isinstance(split_models, list) or len(split_models) != 5:
        raise MayrNextgenRuntimeError("Runtime must register five split models")
    region_residual_required = bool(
        registry.get("candidate_set_conditioned_structural_residual", False)
    )
    if region_residual_required and (
        registry.get("region_type_level_maximum_preserved") is not True
        or registry.get("candidate_softmax_used") is not False
    ):
        raise MayrNextgenRuntimeError("Runtime region residual boundary changed")
    for split_model in split_models:
        if not isinstance(split_model, Mapping):
            raise MayrNextgenRuntimeError("Invalid runtime split-model binding")
        ranker_path = _repo_path(
            split_model["ranker_checkpoint_path"],
            label="ranker checkpoint",
        )
        if sha256_file(ranker_path) != split_model["ranker_checkpoint_sha256"]:
            raise MayrNextgenRuntimeError("Registered ranker checkpoint changed")
        bindings = split_model.get("backbone_bindings")
        if not isinstance(bindings, list) or len(bindings) != 3:
            raise MayrNextgenRuntimeError(
                "Each split must register three backbone checkpoints"
            )
        for binding in bindings:
            checkpoint_path = _repo_path(
                binding["path"],
                label="conditional-N checkpoint",
            )
            if sha256_file(checkpoint_path) != binding["sha256"]:
                raise MayrNextgenRuntimeError(
                    "Registered conditional-N checkpoint changed"
                )
        residual = split_model.get("region_membership_residual")
        if region_residual_required:
            if (
                not isinstance(residual, Mapping)
                or residual.get("enabled") is not True
                or residual.get("candidate_set_conditioned") is not True
                or residual.get("type_level_maximum_preserved") is not True
                or residual.get("candidate_softmax_used") is not False
            ):
                raise MayrNextgenRuntimeError(
                    "Registered region residual boundary changed"
                )
            residual_path = _repo_path(
                residual["path"],
                label="region residual checkpoint",
            )
            if sha256_file(residual_path) != residual["sha256"]:
                raise MayrNextgenRuntimeError(
                    "Registered region residual checkpoint changed"
                )
        elif residual is not None:
            raise MayrNextgenRuntimeError(
                "Unexpected region residual in legacy registry"
            )
    margin_enabled = bool(registry.get("margin_abstention_enabled", False))
    if margin_enabled:
        threshold = registry.get("runtime_margin_threshold")
        if (
            not isinstance(threshold, (int, float))
            or not np.isfinite(float(threshold))
            or float(threshold) < 0
            or registry.get("margin_threshold_aggregation")
            != "median_of_five_split_thresholds"
            or registry.get("low_margin_runtime_status") != "partial_uncertain"
        ):
            raise MayrNextgenRuntimeError("Runtime margin gate changed")
    return registry


def _apply_registered_region_residual(
    *,
    split_model: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    ordered: pd.DataFrame,
    score_components: Mapping[str, torch.Tensor],
    conditional_n_mean: np.ndarray,
    conditional_n_std: np.ndarray,
) -> torch.Tensor:
    """Apply one hash-bound target-blind v7 residual, or return legacy logits."""

    binding = split_model.get("region_membership_residual")
    if binding is None:
        return score_components["canonical_logit"]
    if not isinstance(binding, Mapping) or binding.get("enabled") is not True:
        raise MayrNextgenRuntimeError("Runtime region residual binding is invalid")
    checkpoint_binding = checkpoint.get("region_membership_residual")
    if not isinstance(checkpoint_binding, Mapping) or dict(checkpoint_binding) != dict(
        binding
    ):
        raise MayrNextgenRuntimeError(
            "Runtime checkpoint and registry residual bindings differ"
        )
    path = _repo_path(binding["path"], label="region residual checkpoint")
    if sha256_file(path) != binding["sha256"]:
        raise MayrNextgenRuntimeError("Runtime region residual hash changed")
    bundle = joblib.load(path)
    if not isinstance(bundle, Mapping):
        raise MayrNextgenRuntimeError("Runtime region residual artifact is invalid")
    if not ordered["site_type"].astype(str).eq("delocalized_region").any():
        return score_components["canonical_logit"]
    try:
        positions, features, feature_names = region_feature_matrix(
            ordered,
            membership_logits=score_components["membership_logit"].cpu().numpy(),
            compatibility_logits=score_components["compatibility_logit"].cpu().numpy(),
            conditional_n_mean=conditional_n_mean,
            conditional_n_std=conditional_n_std,
            origin_vocabulary_values=binding["origin_vocabulary"],
        )
        residual_probability = score_region_residual(
            bundle,
            features,
            expected_feature_names=feature_names,
        )
        logits, _ = apply_region_residual(
            ordered,
            base_logits=score_components["canonical_logit"].cpu().numpy(),
            region_positions=positions,
            residual_probabilities=residual_probability,
            residual_weight=float(binding["residual_weight"]),
            maximum_base_margin=(
                float(binding["maximum_base_margin"])
                if binding.get("maximum_base_margin") is not None
                else None
            ),
            top_k=(int(binding["top_k"]) if binding.get("top_k") is not None else None),
        )
    except RegionResidualError as exc:
        raise MayrNextgenRuntimeError(str(exc)) from exc
    return torch.tensor(
        logits,
        dtype=torch.float32,
        device=score_components["canonical_logit"].device,
    )


def _scope(
    *,
    calibrated: bool,
) -> dict[str, object]:
    return {
        "deployment_population": "mayr_like_molecules_not_arbitrary_molecules",
        "feature_scope": "strict_no_dft_rdkit_gfn1_xtb_solvent_charge",
        "candidate_types": list(SITE_TYPES),
        "candidate_scores_independent": True,
        "candidate_softmax_used": False,
        "no_site_claim_permitted": False,
        "calibrator_scope": "all_site_types" if calibrated else "unavailable",
        "calibrated_site_types": list(SITE_TYPES) if calibrated else [],
    }


def _aggregate(candidates: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    counts = Counter(str(candidate["site_type"]) for candidate in candidates)
    return {
        "returned_candidate_count": len(candidates),
        "candidate_count_by_type": {
            site_type: counts[site_type] for site_type in SITE_TYPES
        },
        "raw_score_available_candidate_count": sum(
            candidate["validity"]["raw_sigmoid_score"] is not None
            for candidate in candidates
        ),
        "probability_available_candidate_count": sum(
            candidate["validity"]["absolute_site_probability"] is not None
            for candidate in candidates
        ),
        "conditional_N_available_candidate_count": sum(
            candidate["conditional_N"]["status"] == "available"
            for candidate in candidates
        ),
    }


def _interpretation() -> dict[str, object]:
    return {
        "candidate_scores_form_joint_distribution": False,
        "multiple_candidates_may_be_high": True,
        "low_scores_mean_no_nucleophilic_site": False,
        "low_score_interpretation": (
            "uncertainty_or_out_of_domain_not_evidence_of_absence"
        ),
        "validity_gates_conditional_N": False,
    }


def _provenance(
    *,
    config: Mapping[str, Any],
    registry: Mapping[str, Any] | None,
    calibrated: bool,
) -> dict[str, object]:
    if registry is not None:
        generator_raw = (
            registry["candidate_generator_binding"]["path"]
            if _publication_registry(registry)
            else registry["candidate_generator_path"]
        )
        campaign_id = str(registry["campaign_id"])
    elif config.get("schema_version") == publication_site.CONFIG_SCHEMA:
        generator_raw = "src/nucpred/datasets/mayr_site_n.py"
        campaign_id = str(config["campaign_id"])
    else:
        generator_raw = config["contract"]["candidate_generator_path"]
        campaign_id = str(config["campaign_id"])
    generator_path = _repo_path(generator_raw, label="candidate generator")
    publication = registry is not None and _publication_registry(registry)
    ontology_path = (
        _repo_path(
            registry["candidate_ontology_binding"]["path"],
            label="candidate ontology",
        )
        if publication
        else ROOT / "configs/mayr_nextgen_gate_a.toml"
    )
    return {
        "contract_schema_sha256": sha256_file(RESPONSE_SCHEMA_PATH),
        "candidate_ontology_id": "mayr-nextgen-gate-a-ontology-v1",
        "candidate_ontology_sha256": sha256_file(ontology_path),
        "candidate_generator_id": "mayr-site-candidate-generator-v1",
        "candidate_generator_sha256": sha256_file(generator_path),
        "N_model_run_id": (
            (
                f"{campaign_id}:all-data-three-initialization-conditional-n"
                if publication
                else f"{campaign_id}:stage-e-c-15-checkpoint-ensemble"
            )
            if registry is not None
            else None
        ),
        "validity_model_run_id": (
            (
                f"{campaign_id}:all-data-automatic-site-ranker"
                if publication
                else f"{campaign_id}:five-split-validity-ensemble"
            )
            if registry is not None
            else None
        ),
        "calibrator_run_id": (
            (
                f"{campaign_id}:external-absolute-probability-calibrator"
                if publication
                else f"{campaign_id}:five-split-platt-ensemble"
            )
            if calibrated
            else None
        ),
        "source_run_ids": (
            (
                [campaign_id, "mayr-site-n-20260805-v2"]
                if publication
                else [
                    campaign_id,
                    "mayr-nextgen-stage-e-c-20260728-v1",
                    "mayr-nextgen-stage-d-20260727-v1",
                ]
            )
            if registry is not None
            else []
        ),
        "feature_families": [
            "rdkit_2d_graph",
            "g1_force_field_geometry",
            "gfn1_xtb_gas",
            "gfn1_xtb_alpb",
            "solvent_descriptors",
            "formal_charge",
        ],
        "dft_or_cdft_used": False,
        "target_or_site_label_read": False,
        "created_at_utc": _utc_now(),
    }


def _refusal_response(
    request: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    registry: Mapping[str, Any] | None,
    reasons: Sequence[str],
    structure: str,
    solvent: str,
    applicability_status: str,
    candidate_generation: str = "failed",
    g1_geometry: str = "not_run",
    xtb_electronic: str = "not_run",
) -> dict[str, Any]:
    calibrated = _absolute_probability_enabled(registry)
    response = {
        "schema_version": RESPONSE_SCHEMA_VERSION,
        "request_id": str(request["request_id"]),
        "input": {
            "smiles": str(request["smiles"]),
            "solvent": str(request["solvent"]),
            "formal_charge": int(request["formal_charge"]),
        },
        "status": "refused",
        "scope": _scope(calibrated=calibrated),
        "feature_status": {
            "structure": structure,
            "candidate_generation": candidate_generation,
            "g1_geometry": g1_geometry,
            "xtb_electronic": xtb_electronic,
            "solvent": solvent,
            "applicability_status": applicability_status,
            "refusal_reasons": list(dict.fromkeys(reasons)),
        },
        "candidates": [],
        "aggregate": _aggregate([]),
        "interpretation": _interpretation(),
        "provenance": _provenance(
            config=config,
            registry=registry,
            calibrated=calibrated,
        ),
    }
    validate_response(response, request=request)
    return response


def _normalize_solvent(value: object) -> str:
    return "".join(str(value).casefold().split())


def _resolve_context(
    request: Mapping[str, Any],
    *,
    contexts: pd.DataFrame,
) -> tuple[pd.Series | None, list[str], str, str, str]:
    molecule = Chem.MolFromSmiles(str(request["smiles"]))
    if molecule is None:
        return None, ["invalid_smiles"], "failed", "supported", "feature_failure"
    observed_charge = sum(atom.GetFormalCharge() for atom in molecule.GetAtoms())
    if observed_charge != int(request["formal_charge"]):
        return (
            None,
            ["formal_charge_mismatch"],
            "complete",
            "supported",
            "out_of_domain",
        )
    canonical = Chem.MolToSmiles(
        molecule,
        canonical=True,
        isomericSmiles=True,
    )
    solvent_key = _normalize_solvent(request["solvent"])
    known_solvents = {
        _normalize_solvent(value)
        for value in pd.concat(
            [contexts["solvent_raw"], contexts["xtb_alpb_solvent"]],
            ignore_index=True,
        )
    }
    solvent_supported = solvent_key in known_solvents
    if not solvent_supported:
        return (
            None,
            ["unsupported_solvent"],
            "complete",
            "unsupported",
            "out_of_domain",
        )
    # Cached memberships are indexed against the stored canonical SMILES.
    # Equivalent but noncanonical input is refused rather than returning
    # memberships grounded in a different atom order.
    if canonical != str(request["smiles"]):
        return (
            None,
            ["outside_mayr_like_domain", "model_asset_unavailable"],
            "complete",
            "supported",
            "out_of_domain",
        )
    structure_match = contexts["model_canonical_smiles"].astype(str).eq(canonical)
    charge_match = (
        contexts["model_formal_charge"].astype(int).eq(int(request["formal_charge"]))
    )
    solvent_match = contexts.apply(
        lambda row: (
            solvent_key
            in {
                _normalize_solvent(row["solvent_raw"]),
                _normalize_solvent(row["xtb_alpb_solvent"]),
            }
        ),
        axis=1,
    )
    matched = contexts.loc[structure_match & charge_match & solvent_match]
    if len(matched) != 1:
        return (
            None,
            ["outside_mayr_like_domain", "model_asset_unavailable"],
            "complete",
            "supported",
            "out_of_domain",
        )
    return matched.iloc[0], [], "complete", "supported", "in_domain"


def _runtime_feature_query(request: Mapping[str, Any]) -> pd.DataFrame:
    """Build a stable label-free external query for one canonical molecule."""

    molecule = Chem.MolFromSmiles(str(request["smiles"]))
    if molecule is None:
        raise ExternalMayrContextError("Invalid runtime SMILES")
    canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    if canonical != str(request["smiles"]):
        raise ExternalMayrContextError(
            "Runtime SMILES must be canonical to preserve atom-index grounding"
        )
    identity = _canonical_sha256(
        {
            "canonical_smiles": canonical,
            "formal_charge": int(request["formal_charge"]),
            "solvent": _normalize_solvent(request["solvent"]),
        }
    )
    return pd.DataFrame(
        [
            {
                "fit_id": f"runtime:{identity[:24]}",
                "canonical_smiles": canonical,
                "formal_charge": int(request["formal_charge"]),
                "solvent_raw": str(request["solvent"]),
            }
        ]
    )


def _dynamic_deployment_candidates(
    context: Mapping[str, object],
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Enumerate and policy-filter candidates for an arbitrary new species."""

    records = enumerate_candidate_sites(
        context,
        species_id=str(context["species_id"]),
    )
    candidates = pd.DataFrame(records)
    species = pd.DataFrame(
        [
            {
                "species_id": str(context["species_id"]),
                "model_canonical_smiles": str(context["model_canonical_smiles"]),
            }
        ]
    )
    return select_deployment_candidates(candidates, species)


def _build_unseen_context(
    request: Mapping[str, Any],
    *,
    cached_contexts: pd.DataFrame,
    feature_cache_directory: Path,
    xtb_config_path: Path,
    reference_contexts_path: Path,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, str]:
    """Construct model features and candidates without consulting any label."""

    built = build_external_pure_contexts(
        _runtime_feature_query(request),
        cache_directory=feature_cache_directory,
        config_path=xtb_config_path,
        reference_contexts_path=reference_contexts_path,
    )
    if len(built.contexts) != 1:
        raise ExternalMayrContextError("Runtime feature adapter returned !=1 context")
    context = built.contexts.iloc[0]
    candidates, _ = _dynamic_deployment_candidates(context)
    connectivity_seen = str(context["connectivity_id"]) in set(
        cached_contexts["connectivity_id"].astype(str)
    )
    species_seen = str(context["species_id"]) in set(
        cached_contexts["species_id"].astype(str)
    )
    if not connectivity_seen:
        novelty_reason = "unseen_connectivity"
    elif not species_seen:
        novelty_reason = "unseen_species_state"
    else:
        novelty_reason = "unseen_molecule_solvent_context"
    scoring_contexts = pd.concat(
        [cached_contexts, built.contexts],
        ignore_index=True,
    )
    return context, scoring_contexts, candidates, novelty_reason


def _hydrogen_parents(
    context: pd.Series,
    members: Sequence[int],
) -> list[int]:
    parents = json.loads(str(context["model_hydrogen_parent_index_json"]))
    return [int(parents[index]) for index in members]


def _g1_runtime_status(context: Mapping[str, object]) -> str:
    if str(context["g1_status"]) != "success":
        return "failed"
    fallback = context.get("g1_fallback_reason")
    if fallback is not None and not pd.isna(fallback) and str(fallback):
        return "fallback"
    return "complete"


def _apply_margin_abstention(
    candidates: list[dict[str, Any]],
    *,
    registry: Mapping[str, Any],
) -> bool:
    """Mark the two leading candidates uncertain when the frozen gate rejects."""

    if not candidates:
        raise MayrNextgenRuntimeError("Margin gate received no candidates")
    if not bool(registry.get("margin_abstention_enabled", False)):
        return False
    singleton_abstention = (
        len(candidates) == 1
        and registry.get("singleton_margin_policy") == "abstain_margin_undefined"
    )
    top1_top2_margin = (
        float(candidates[0]["validity"]["logit_mean"])
        - float(candidates[1]["validity"]["logit_mean"])
        if len(candidates) > 1
        else None
    )
    low_margin = singleton_abstention or (
        top1_top2_margin is not None
        and top1_top2_margin < float(registry["runtime_margin_threshold"])
    )
    if low_margin:
        reason = (
            "undefined_singleton_site_rank_margin"
            if singleton_abstention
            else "low_site_rank_margin"
        )
        for candidate in candidates[:2]:
            applicability = candidate["applicability"]
            applicability["status"] = "uncertain"
            applicability["reasons"] = list(
                dict.fromkeys([*applicability["reasons"], reason])
            )
    return low_margin


def _score_publication_candidate_universe(
    *,
    config: Mapping[str, Any],
    registry: Mapping[str, Any],
    universe: pd.DataFrame,
    contexts: pd.DataFrame,
    device: str,
) -> pd.DataFrame:
    if config.get("schema_version") != publication_site.CONFIG_SCHEMA:
        raise MayrNextgenRuntimeError(
            "Publication registry requires the publication site config"
        )
    model_binding = registry["publication_model"]
    registered_backbones = model_binding["conditional_n_bindings"]
    runtime_device = torch.device(device)
    models = []
    conditional_config = _repo_path(
        registry["conditional_n_config_binding"]["path"],
        label="conditional-N config",
    )
    try:
        for binding in registered_backbones:
            path = _repo_path(
                binding["path"], label="publication conditional-N checkpoint"
            )
            loaded_model, preprocessor, vocabulary, payload = load_outer_checkpoint(
                path,
                config_path=conditional_config,
                device=runtime_device,
            )
            seed = int(binding["initialization_seed"])
            contract = payload.get("contract")
            if (
                not isinstance(contract, Mapping)
                or int(contract.get("initialization_seed", -1)) != seed
                or contract.get("all_corrected_v2_targets_used") is not True
                or contract.get("external_sources_or_labels_used") is not False
                or payload.get("phase") != "post_outer_evaluation_all_data_final_refit"
                or payload.get("model_state_sha256")
                != binding.get("model_state_sha256")
            ):
                raise MayrNextgenRuntimeError(
                    "Publication conditional-N checkpoint boundary changed"
                )
            models.append(
                (
                    seed,
                    loaded_model,
                    preprocessor,
                    vocabulary,
                    payload,
                    path,
                )
            )
        query_ids, source_features, n_mean, n_std, observed_bindings = (
            publication_site.encode_queries(
                models=models,
                queries=universe,
                contexts=contexts,
                config=config,
                device=runtime_device,
            )
        )
    finally:
        publication_site.release_models(models, device=runtime_device)
    if observed_bindings != registered_backbones:
        raise MayrNextgenRuntimeError(
            "Publication conditional-N registry bindings changed"
        )

    ranker_binding = model_binding["ranker_checkpoint"]
    ranker_path = _repo_path(
        ranker_binding["path"], label="publication ranker checkpoint"
    )
    checkpoint = torch.load(ranker_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise MayrNextgenRuntimeError("Publication ranker checkpoint is invalid")
    expected = {
        "schema_version": "nucpred.mayr-n-publication-site-ranker-checkpoint.v1",
        "phase": "post_outer_evaluation_all_data_final_refit",
        "campaign_id": registry["campaign_id"],
        "outer_fold": -1,
        "selected_arm": "hierarchical_exact",
        "final_refit_performed": True,
        "all_corrected_v2_targets_used": True,
        "reported_outer_metrics_modified": False,
        "candidate_softmax_used": False,
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise MayrNextgenRuntimeError(
                f"Publication ranker checkpoint boundary changed: {key}"
            )
    if (
        checkpoint.get("conditional_n_bindings") != registered_backbones
        or checkpoint.get("region_membership_residual")
        != model_binding.get("region_membership_residual")
        or checkpoint.get("margin_abstention") != model_binding.get("margin_abstention")
    ):
        raise MayrNextgenRuntimeError("Publication ranker and registry bindings differ")
    ranker = _ranker_from_checkpoint(checkpoint)
    ordered = (
        universe.set_index("query_id", drop=False).loc[query_ids].reset_index(drop=True)
    )
    type_index = site_type_indices(ordered["site_type"].astype(str))
    with torch.no_grad():
        score_components = _score_ranker_from_source_features(
            ranker=ranker,
            checkpoint=checkpoint,
            source_features=source_features,
            type_index=type_index,
        )
        logits = _apply_registered_region_residual(
            split_model=model_binding,
            checkpoint=checkpoint,
            ordered=ordered,
            score_components=score_components,
            conditional_n_mean=n_mean,
            conditional_n_std=n_std,
        )
        calibrator = TypeAwarePlattCalibrator.from_payload(checkpoint["calibrator"])
        probability = calibrator(logits, type_index)

    logit_values = logits.cpu().numpy()
    scored = ordered.copy()
    scored["validity_logit_mean"] = logit_values
    # The publication deployment contains one deterministic all-data ranker.
    # Zero is the standard deviation over that singleton deployment member; it
    # must not be interpreted as epistemic certainty.
    scored["validity_logit_std"] = np.zeros(len(scored), dtype=float)
    scored["validity_raw_sigmoid"] = 1.0 / (
        1.0 + np.exp(-np.clip(logit_values, -60.0, 60.0))
    )
    scored["validity_probability_mean"] = probability.cpu().numpy()
    scored["conditional_N_mean"] = n_mean
    scored["conditional_N_std"] = n_std
    return scored


def score_candidate_universe(
    *,
    config: Mapping[str, Any],
    registry: Mapping[str, Any],
    universe: pd.DataFrame,
    contexts: pd.DataFrame,
    device: str = "cpu",
) -> pd.DataFrame:
    """Score a prebuilt target-blind context/candidate universe.

    This is the model-bearing portion of the registry-backed runtime.  Keeping
    it independent from cached-context resolution lets audited external query
    adapters provide model-ready features without changing any frozen weights,
    rankers, calibrators, or residual bindings.
    """

    if _publication_registry(registry):
        return _score_publication_candidate_universe(
            config=config,
            registry=registry,
            universe=universe,
            contexts=contexts,
            device=device,
        )

    logits_by_split: list[np.ndarray] = []
    probability_by_split: list[np.ndarray] = []
    n_mean_by_split: list[np.ndarray] = []
    n_variance_by_split: list[np.ndarray] = []
    reference_ids: list[str] | None = None
    runtime_device = torch.device(device)
    for split_model in registry["split_models"]:
        split_seed = int(split_model["split_seed"])
        query_ids, features, n_mean, n_std, _ = _encode_split_ensemble(
            config=config,
            split_seed=split_seed,
            queries=universe,
            contexts=contexts,
            device=runtime_device,
        )
        if reference_ids is None:
            reference_ids = query_ids
        elif reference_ids != query_ids:
            raise MayrNextgenRuntimeError("Runtime candidate order changed")
        ranker_path = _repo_path(
            split_model["ranker_checkpoint_path"],
            label="ranker checkpoint",
        )
        checkpoint = _load_ranker_checkpoint(
            ranker_path,
            split_seed=split_seed,
            config=config,
        )
        ranker = _ranker_from_checkpoint(checkpoint)
        ordered = (
            universe.set_index("query_id", drop=False)
            .loc[query_ids]
            .reset_index(drop=True)
        )
        type_index = site_type_indices(ordered["site_type"].astype(str))
        with torch.no_grad():
            score_components = _score_ranker_from_source_features(
                ranker=ranker,
                checkpoint=checkpoint,
                source_features=features,
                type_index=type_index,
            )
            logits = _apply_registered_region_residual(
                split_model=split_model,
                checkpoint=checkpoint,
                ordered=ordered,
                score_components=score_components,
                conditional_n_mean=n_mean,
                conditional_n_std=n_std,
            )
            calibrator = TypeAwarePlattCalibrator.from_payload(checkpoint["calibrator"])
            probability = calibrator(logits, type_index)
        logits_by_split.append(logits.cpu().numpy())
        probability_by_split.append(probability.cpu().numpy())
        n_mean_by_split.append(n_mean)
        n_variance_by_split.append(n_std**2)
    if reference_ids is None:
        raise MayrNextgenRuntimeError("Runtime registry encoded no split")

    ordered = (
        universe.set_index("query_id", drop=False)
        .loc[reference_ids]
        .reset_index(drop=True)
    )
    logits_matrix = np.stack(logits_by_split, axis=1)
    probability_matrix = np.stack(probability_by_split, axis=1)
    n_mean_matrix = np.stack(n_mean_by_split, axis=1)
    n_variance_matrix = np.stack(n_variance_by_split, axis=1)
    logit_mean = logits_matrix.mean(axis=1)
    n_mean = n_mean_matrix.mean(axis=1)
    n_second_moment = (n_variance_matrix + n_mean_matrix**2).mean(axis=1)

    scored = ordered.copy()
    scored["validity_logit_mean"] = logit_mean
    scored["validity_logit_std"] = logits_matrix.std(axis=1)
    scored["validity_raw_sigmoid"] = 1.0 / (
        1.0 + np.exp(-np.clip(logit_mean, -60.0, 60.0))
    )
    scored["validity_probability_mean"] = probability_matrix.mean(axis=1)
    scored["conditional_N_mean"] = n_mean
    scored["conditional_N_std"] = np.sqrt(np.maximum(n_second_moment - n_mean**2, 0.0))
    return scored


def infer(
    request: Mapping[str, Any],
    *,
    registry_path: Path = DEFAULT_REGISTRY,
    config_path: Path = DEFAULT_CONFIG,
    device: str = "cpu",
    feature_cache_directory: Path = DEFAULT_UNSEEN_FEATURE_CACHE,
    xtb_config_path: Path = DEFAULT_XTB_CONFIG,
    reference_contexts_path: Path = DEFAULT_REFERENCE_CONTEXTS,
) -> dict[str, Any]:
    """Run target-blind inference for a cached or genuinely unseen molecule."""

    validate_request(request)
    config = _read_runtime_config(config_path)
    try:
        registry = _load_registry(registry_path)
    except (OSError, KeyError, TypeError, MayrNextgenRuntimeError):
        return _refusal_response(
            request,
            config=config,
            registry=None,
            reasons=["model_asset_unavailable"],
            structure="complete",
            solvent="supported",
            applicability_status="feature_failure",
        )

    dataset_root = _repo_path(
        registry["dataset_directory"],
        label="runtime dataset",
    )
    contexts = pd.read_parquet(dataset_root / "contexts.parquet")
    if _publication_registry(registry):
        species = pd.read_parquet(dataset_root / "species.parquet")
        candidates, _ = publication_site.deployment_candidates(config, species)
    else:
        candidates, _ = _deployment_candidates(config)
    scoring_contexts = contexts
    novelty_reason: str | None = None
    context, reasons, structure, solvent, applicability = _resolve_context(
        request,
        contexts=contexts,
    )
    if context is None:
        dynamic_eligible = reasons == [
            "outside_mayr_like_domain",
            "model_asset_unavailable",
        ]
        dynamic_candidate_generation = "failed"
        dynamic_g1 = "not_run"
        dynamic_xtb = "not_run"
        if dynamic_eligible:
            try:
                context, scoring_contexts, candidates, novelty_reason = (
                    _build_unseen_context(
                        request,
                        cached_contexts=contexts,
                        feature_cache_directory=feature_cache_directory.resolve(),
                        xtb_config_path=xtb_config_path.resolve(),
                        reference_contexts_path=reference_contexts_path.resolve(),
                    )
                )
                reasons = []
                applicability = "uncertain"
            except CandidatePolicyError:
                reasons = ["candidate_generation_failed"]
                applicability = "feature_failure"
                dynamic_g1 = "complete"
                dynamic_xtb = "complete"
            except ExternalMayrContextError as exc:
                if "canonical" in str(exc) or "one fragment" in str(exc):
                    reasons = ["outside_mayr_like_domain"]
                    applicability = "out_of_domain"
                elif "G1 geometry failed" in str(exc):
                    reasons = ["xtb_feature_failure"]
                    applicability = "feature_failure"
                    dynamic_g1 = "failed"
                else:
                    reasons = ["xtb_feature_failure"]
                    applicability = "feature_failure"
                    dynamic_g1 = "complete"
                    dynamic_xtb = "failed"
        if context is None:
            return _refusal_response(
                request,
                config=config,
                registry=registry,
                reasons=reasons,
                structure=structure,
                solvent=solvent,
                applicability_status=applicability,
                candidate_generation=dynamic_candidate_generation,
                g1_geometry=dynamic_g1,
                xtb_electronic=dynamic_xtb,
            )
    if not bool(context["complete_xtb10"]):
        return _refusal_response(
            request,
            config=config,
            registry=registry,
            reasons=["xtb_feature_failure"],
            structure="complete",
            solvent="supported",
            applicability_status="feature_failure",
            g1_geometry=_g1_runtime_status(context),
            xtb_electronic="failed",
        )

    context_frame = pd.DataFrame(
        [
            {
                "context_id": str(context["context_id"]),
                "species_id": str(context["species_id"]),
                "connectivity_id": str(context["connectivity_id"]),
            }
        ]
    )
    universe = _candidate_universe(
        test_contexts=context_frame,
        candidates=candidates,
    )
    scored = score_candidate_universe(
        config=config,
        registry=registry,
        universe=universe,
        contexts=scoring_contexts,
        device=device,
    )
    absolute_probability_enabled = _absolute_probability_enabled(registry)

    atomic_numbers = json.loads(str(context["model_atomic_numbers_json"]))
    response_candidates: list[dict[str, Any]] = []
    for row in scored.itertuples(index=False):
        members = [int(value) for value in json.loads(row.member_atom_indices_json)]
        site_type = str(row.site_type)
        response_candidates.append(
            {
                "candidate_site_id": str(row.candidate_site_id),
                "site_type": site_type,
                "membership": {
                    "member_atom_indices": members,
                    "member_atomic_numbers": [
                        int(atomic_numbers[value]) for value in members
                    ],
                    "member_bond_pairs": [
                        [int(value) for value in pair]
                        for pair in json.loads(row.member_bond_pairs_json)
                    ],
                    "hydrogen_parent_atom_indices": (
                        _hydrogen_parents(context, members)
                        if site_type == "transferable_h_group"
                        else []
                    ),
                },
                "candidate_origins": [
                    str(value) for value in json.loads(row.candidate_origins_json)
                ],
                "validity": {
                    "logit_mean": float(row.validity_logit_mean),
                    "logit_std": float(row.validity_logit_std),
                    "raw_sigmoid_score": float(row.validity_raw_sigmoid),
                    "raw_score_semantics": (
                        "uncalibrated_independent_sigmoid_not_probability"
                    ),
                    "support_status": "formal_ranking",
                    "absolute_site_probability": (
                        float(row.validity_probability_mean)
                        if absolute_probability_enabled
                        else None
                    ),
                    "probability_status": (
                        "calibrated"
                        if absolute_probability_enabled
                        else "unavailable_no_formal_calibrator"
                    ),
                    "probability_unavailable_reason": (
                        None
                        if absolute_probability_enabled
                        else str(
                            registry["absolute_site_probability_unavailable_reason"]
                        )
                    ),
                },
                "conditional_N": {
                    "mean": float(row.conditional_N_mean),
                    "std": float(row.conditional_N_std),
                    "status": "available",
                    "conditioning": "candidate_site_query",
                },
                "applicability": {
                    "status": "uncertain" if novelty_reason else "supported",
                    "reasons": [novelty_reason] if novelty_reason else [],
                },
            }
        )
    response_candidates.sort(
        key=lambda candidate: (
            -float(candidate["validity"]["logit_mean"]),
            str(candidate["candidate_site_id"]),
        )
    )
    low_margin = _apply_margin_abstention(
        response_candidates,
        registry=registry,
    )
    ranking_or_domain_uncertain = low_margin or novelty_reason is not None
    partial = ranking_or_domain_uncertain or not absolute_probability_enabled
    response = {
        "schema_version": RESPONSE_SCHEMA_VERSION,
        "request_id": str(request["request_id"]),
        "input": {
            "smiles": str(request["smiles"]),
            "solvent": str(request["solvent"]),
            "formal_charge": int(request["formal_charge"]),
        },
        "status": "partial" if partial else "ok",
        "scope": _scope(calibrated=absolute_probability_enabled),
        "feature_status": {
            "structure": "complete",
            "candidate_generation": "complete",
            "g1_geometry": _g1_runtime_status(context),
            "xtb_electronic": "complete",
            "solvent": "supported",
            "applicability_status": (
                "uncertain" if ranking_or_domain_uncertain else "in_domain"
            ),
            "refusal_reasons": [],
        },
        "candidates": response_candidates,
        "aggregate": _aggregate(response_candidates),
        "interpretation": _interpretation(),
        "provenance": _provenance(
            config=config,
            registry=registry,
            calibrated=absolute_probability_enabled,
        ),
    }
    validate_response(response, request=request)
    return response


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--feature-cache",
        type=Path,
        default=DEFAULT_UNSEEN_FEATURE_CACHE,
    )
    parser.add_argument("--xtb-config", type=Path, default=DEFAULT_XTB_CONFIG)
    parser.add_argument(
        "--reference-contexts",
        type=Path,
        default=DEFAULT_REFERENCE_CONTEXTS,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    response = infer(
        _load_json(args.request),
        registry_path=args.registry.resolve(),
        config_path=args.config.resolve(),
        device=str(args.device),
        feature_cache_directory=args.feature_cache.resolve(),
        xtb_config_path=args.xtb_config.resolve(),
        reference_contexts_path=args.reference_contexts.resolve(),
    )
    if args.output is None:
        print(json.dumps(response, ensure_ascii=False, indent=2))
    else:
        atomic_write_json(args.output.resolve(), response)
        print(
            json.dumps(
                {
                    "status": response["status"],
                    "request_id": response["request_id"],
                    "candidate_count": len(response["candidates"]),
                    "output": str(args.output.resolve()),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
