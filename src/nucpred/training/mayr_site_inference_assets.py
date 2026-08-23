"""Shared frozen assets for Mayr site-ranking evaluation and inference.

This module sits below both the experiment and inference layers so that the
two paths use the same candidate policy, query encoder, and checkpoint
validation without introducing a package dependency cycle.
"""

from __future__ import annotations

from collections.abc import Mapping
import gc
import hashlib
import json
from pathlib import Path
import tomllib
from typing import Any

import numpy as np
import pandas as pd
import torch

from nucpred.core.files import sha256_file
from nucpred.project import get_project_layout
from nucpred.training.mayr_node_xtb_scratch import SolventVocabulary
from nucpred.training.mayr_site_confidence import tensor_mapping_sha256
from nucpred.training.mayr_site_n import (
    MayrSiteNModel,
    SiteNFoldPreprocessor,
    pack_site_n_batch,
)
from nucpred.training.mayr_site_n_stage_e_b import (
    E_B_N1,
    MayrSiteNStageEBResidualModel,
)
from nucpred.training.mayr_site_n_stage_e_c import (
    E_C_N3,
    MayrSiteNStageECExpertModel,
)
from nucpred.training.mayr_site_queries import site_n_examples_from_queries
from nucpred.training.mayr_site_ranker import (
    RANKER_ARMS,
    RANKER_SITE_TYPES,
    IndependentSiteRanker,
)
from nucpred.training.mayr_site_structured_ranker import (
    STRUCTURED_CAMPAIGN_ARMS,
    STRUCTURED_RANKER_SCHEMA_VERSION,
    StructuredSiteRanker,
    reduce_frozen_ensemble_features,
    structured_ranker_from_architecture,
)


ROOT = get_project_layout().root
CONFIG_SCHEMA = "nucpred.mayr-nextgen-site-identification-config.v1"
RUNTIME_REGISTRY_SCHEMA = "nucpred.mayr-site-identification-runtime-registry.v1"
REGION_RESIDUAL_CAMPAIGN_ARMS = ("frozen_v6", "region_structural_residual")


class MayrSiteInferenceAssetError(RuntimeError):
    """Raised when a shared site-identification asset violates its contract."""


def canonical_sha256(value: object) -> str:
    """Hash a JSON-compatible value with canonical serialization."""

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _repo_path(raw: object, *, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise MayrSiteInferenceAssetError(f"{label} must be a repo-relative path")
    relative = Path(raw)
    if relative.is_absolute():
        raise MayrSiteInferenceAssetError(f"{label} must be a repo-relative path")
    resolved = (ROOT / relative).resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise MayrSiteInferenceAssetError(f"{label} escapes repository") from exc
    return resolved


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def read_site_identification_config(path: str | Path) -> dict[str, Any]:
    """Read and validate the frozen site-identification authority boundary."""

    config_path = Path(path).resolve()
    try:
        with config_path.open("rb") as handle:
            payload = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise MayrSiteInferenceAssetError(
            f"Cannot read site-identification config: {config_path}"
        ) from exc
    if payload.get("schema_version") != CONFIG_SCHEMA:
        raise MayrSiteInferenceAssetError("Site-identification config schema changed")
    if tuple(payload["ranker"]["candidate_types"]) != RANKER_SITE_TYPES:
        raise MayrSiteInferenceAssetError("Configured candidate type order changed")
    training_population = str(
        payload["ranker"].get("training_population", "reviewed_evidence_only")
    )
    if "region_residual" in payload["ranker"]:
        expected_arms = REGION_RESIDUAL_CAMPAIGN_ARMS
        residual = payload["ranker"]["region_residual"]
        if (
            residual.get("target_site_type") != "delocalized_region"
            or residual.get("type_level_maximum_preserved") is not True
            or residual.get("unknown_as_negative") is not False
            or residual.get("candidate_softmax_used") is not False
        ):
            raise MayrSiteInferenceAssetError(
                "Configured region residual boundary changed"
            )
    else:
        expected_arms = (
            STRUCTURED_CAMPAIGN_ARMS
            if training_population == "full_candidate_endpoint_retrieval"
            else RANKER_ARMS
        )
    if tuple(payload["ranker"]["arms"]) != expected_arms:
        raise MayrSiteInferenceAssetError("Configured ranker arms changed")
    authority = payload["authority"]
    expected = {
        "new_validity_head_training_permitted": True,
        "frozen_conditional_n_backbone_training_permitted": False,
        "test_prediction_permitted_after_development_freeze": True,
        "test_labels_permitted_only_in_test_phase": True,
        "formal_calibration_permitted": True,
        "final_refit_permitted": False,
        "dft_or_cdft_computation_permitted": False,
        "unknown_as_negative_permitted": False,
        "candidate_softmax_permitted": False,
        "no_site_claim_permitted": False,
    }
    for key, value in expected.items():
        if authority.get(key) is not value:
            raise MayrSiteInferenceAssetError(f"Authority boundary changed: {key}")
    return payload


def deployment_candidates(
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Apply the target-independent Gate A and response-contract intersection."""

    policy_config = config["candidate_policy"]
    policy = pd.read_parquet(
        _repo_path(policy_config["path"], label="candidate policy")
    )
    if (
        policy_config.get("deployment_eligible_required") is not True
        or policy_config.get("response_membership_contract_required") is not True
        or policy_config.get("delocalized_region_requires_internal_bond") is not True
        or not policy["label_independent"].astype(bool).all()
    ):
        raise MayrSiteInferenceAssetError("Candidate deployment policy changed")
    gate_a_deployment = policy["deployment_eligible"].astype(bool)
    no_internal_bond_region = policy["site_type"].astype(str).eq(
        "delocalized_region"
    ) & policy["member_bond_pairs_json"].astype(str).eq("[]")
    selected = policy.loc[gate_a_deployment & ~no_internal_bond_region].copy()
    if selected["candidate_site_id"].astype(str).duplicated().any():
        raise MayrSiteInferenceAssetError("Deployment candidate IDs are duplicated")
    if set(selected["site_type"].astype(str)) != set(RANKER_SITE_TYPES):
        raise MayrSiteInferenceAssetError(
            "Contract-compatible deployment loses a candidate type"
        )
    audit = {
        "candidate_policy_id": "mayr-multitype-deployment-candidates-v1",
        "audit_population_count": len(policy),
        "gate_a_deployment_count": int(gate_a_deployment.sum()),
        "contract_incompatible_no_bond_region_count": int(
            (gate_a_deployment & no_internal_bond_region).sum()
        ),
        "final_deployment_candidate_count": len(selected),
        "filter_target_independent": True,
    }
    return selected.reset_index(drop=True), audit


def candidate_universe(
    *,
    test_contexts: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    """Build the complete label-independent context-candidate query universe."""

    required_context = {"context_id", "species_id", "connectivity_id"}
    if not required_context <= set(test_contexts.columns):
        raise MayrSiteInferenceAssetError("Unlabeled test contexts changed")
    if test_contexts["context_id"].astype(str).duplicated().any():
        raise MayrSiteInferenceAssetError("Unlabeled test context IDs are duplicated")
    columns = [
        "candidate_site_id",
        "species_id",
        "site_type",
        "member_atom_indices_json",
        "member_bond_pairs_json",
        "member_atomic_numbers_json",
        "candidate_origins_json",
        "label_independent",
    ]
    universe = test_contexts.merge(
        candidates[columns],
        on="species_id",
        how="left",
        validate="many_to_many",
    )
    if universe["candidate_site_id"].isna().any():
        raise MayrSiteInferenceAssetError("Test context has no generated candidates")
    if not universe["label_independent"].astype(bool).all():
        raise MayrSiteInferenceAssetError("Candidate generator exposed a label")
    if set(universe["site_type"].astype(str)) - set(RANKER_SITE_TYPES):
        raise MayrSiteInferenceAssetError("Candidate universe contains unknown type")
    universe["query_id"] = (
        universe["context_id"].astype(str)
        + "|"
        + universe["candidate_site_id"].astype(str)
    )
    if universe["query_id"].duplicated().any():
        raise MayrSiteInferenceAssetError("Candidate universe query IDs are duplicated")
    universe["N_value"] = np.nan
    type_order = {value: index for index, value in enumerate(RANKER_SITE_TYPES)}
    universe["_type_order"] = universe["site_type"].map(type_order)
    return (
        universe.sort_values(
            ["context_id", "_type_order", "candidate_site_id"],
            kind="stable",
        )
        .drop(columns="_type_order")
        .reset_index(drop=True)
    )


def _checkpoint_path(
    config: Mapping[str, Any],
    *,
    split_seed: int,
    initialization_seed: int,
) -> Path:
    return (
        _repo_path(
            config["backbone"]["checkpoint_root"],
            label="checkpoint root",
        )
        / f"split-{split_seed}"
        / f"init-{initialization_seed}"
        / str(config["backbone"]["arm"])
        / str(config["backbone"]["checkpoint_filename"])
    )


def _load_ec_model(
    config: Mapping[str, Any],
    *,
    split_seed: int,
    initialization_seed: int,
    device: torch.device,
) -> tuple[MayrSiteNStageECExpertModel, Mapping[str, Any], Path]:
    path = _checkpoint_path(
        config,
        split_seed=split_seed,
        initialization_seed=initialization_seed,
    )
    if not path.is_file():
        raise MayrSiteInferenceAssetError(f"Missing Stage E-C checkpoint: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise MayrSiteInferenceAssetError("Stage E-C checkpoint is not a mapping")
    expected = {
        "schema_version": config["backbone"]["checkpoint_schema"],
        "phase": "development_validation_selection",
        "arm": E_C_N3,
        "split_seed": split_seed,
        "initialization_seed": initialization_seed,
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise MayrSiteInferenceAssetError(
                f"Stage E-C checkpoint {key} changed at {path}"
            )
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping) or tensor_mapping_sha256(state) != str(
        checkpoint.get("model_state_sha256")
    ):
        raise MayrSiteInferenceAssetError("Stage E-C checkpoint state hash changed")
    contract = checkpoint.get("contract")
    if (
        not isinstance(contract, Mapping)
        or int(contract.get("test_examples_instantiated", -1)) != 0
        or int(contract.get("test_predictions_computed", -1)) != 0
        or contract.get("final_refit_performed") is not False
    ):
        raise MayrSiteInferenceAssetError("Stage E-C checkpoint boundary changed")

    architecture = checkpoint.get("model_architecture")
    if not isinstance(architecture, Mapping):
        raise MayrSiteInferenceAssetError("Stage E-C architecture is missing")
    parent_architecture = architecture.get("frozen_parent_architecture")
    if not isinstance(parent_architecture, Mapping):
        raise MayrSiteInferenceAssetError("Stage E-C parent architecture is missing")
    base_architecture = parent_architecture.get("frozen_base_architecture")
    if not isinstance(base_architecture, Mapping):
        raise MayrSiteInferenceAssetError("Stage E-C base architecture is missing")
    base = MayrSiteNModel(
        num_solvents=len(checkpoint["solvent_vocabulary"]),
        hidden_dim=int(base_architecture["hidden_dim"]),
        layers=int(base_architecture["layers"]),
        node_embedding_dim=int(base_architecture["node_embedding_dim"]),
        edge_embedding_dim=int(base_architecture["edge_embedding_dim"]),
        solvent_embedding_dim=int(base_architecture["solvent_embedding_dim"]),
        dropout=float(base_architecture["dropout"]),
    )
    parent = MayrSiteNStageEBResidualModel(
        frozen_base=base,
        arm=E_B_N1,
    )
    model = MayrSiteNStageECExpertModel(
        frozen_parent=parent,
        arm=E_C_N3,
    )
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint, path


def _encoded_fused_from_output(
    model: MayrSiteNStageECExpertModel,
    inputs: Any,
    output: Any,
) -> torch.Tensor:
    base = model.frozen_parent.frozen_base
    site_graph = inputs.site_graph_index
    solvent_continuous = base.solvent_encoder(inputs.solvent_continuous)[site_graph]
    solvent_embedding = base.solvent_embedding_projection(
        base.solvent_embedding(inputs.solvent_index)
    )[site_graph]
    charge = base.charge_encoder(inputs.molecular_formal_charge)[site_graph]
    global_xtb = base.global_xtb_encoder(inputs.global_xtb)[site_graph]
    return torch.cat(
        (
            output.graph_pool[site_graph],
            output.site_embeddings,
            solvent_continuous,
            solvent_embedding,
            charge,
            global_xtb,
        ),
        dim=-1,
    )


def _encode_queries(
    *,
    model: MayrSiteNStageECExpertModel,
    checkpoint: Mapping[str, Any],
    queries: pd.DataFrame,
    contexts: pd.DataFrame,
    device: torch.device,
    batch_context_count: int = 24,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    preprocessor = SiteNFoldPreprocessor.from_json(checkpoint["preprocessor"])
    vocabulary = SolventVocabulary(
        tuple(str(value) for value in checkpoint["solvent_vocabulary"])
    )
    context_ids = sorted(set(queries["context_id"].astype(str)))
    query_ids: list[str] = []
    features: list[np.ndarray] = []
    n_predictions: list[np.ndarray] = []
    query_index = queries.set_index("query_id", drop=False)
    if query_index.index.duplicated().any():
        raise MayrSiteInferenceAssetError("Encoding query IDs are not unique")
    for start in range(0, len(context_ids), batch_context_count):
        batch_contexts = set(context_ids[start : start + batch_context_count])
        frame = queries.loc[
            queries["context_id"].astype(str).isin(batch_contexts)
        ].copy()
        examples = site_n_examples_from_queries(frame, contexts=contexts)
        packed = pack_site_n_batch(
            examples,
            preprocessor=preprocessor,
            solvent_vocabulary=vocabulary,
        )
        ordered_ids = list(map(str, packed.target_ids))
        query_index.loc[ordered_ids]
        inputs = packed.inputs.to(device)
        with torch.no_grad():
            output = model(inputs)
            fused = _encoded_fused_from_output(model, inputs, output)
            n_raw = output.n_prediction_standardized * float(
                preprocessor.target_scale
            ) + float(preprocessor.target_mean)
            augmented = torch.cat((fused, n_raw.unsqueeze(-1)), dim=-1)
        query_ids.extend(ordered_ids)
        features.append(augmented.cpu().numpy().astype(np.float32))
        n_predictions.append(n_raw.cpu().numpy().astype(np.float32))
    if len(query_ids) != len(queries) or len(set(query_ids)) != len(query_ids):
        raise MayrSiteInferenceAssetError("Encoded query coverage changed")
    return (
        query_ids,
        np.concatenate(features, axis=0),
        np.concatenate(n_predictions, axis=0),
    )


def encode_split_ensemble(
    *,
    config: Mapping[str, Any],
    split_seed: int,
    queries: pd.DataFrame,
    contexts: pd.DataFrame,
    device: torch.device,
) -> tuple[list[str], torch.Tensor, np.ndarray, np.ndarray, list[dict[str, object]]]:
    """Encode queries with all frozen Stage E-C initializations for a split."""

    ensemble_features: list[np.ndarray] = []
    ensemble_n: list[np.ndarray] = []
    reference_ids: list[str] | None = None
    bindings: list[dict[str, object]] = []
    for initialization_seed in config["backbone"]["initialization_seeds"]:
        initialization_seed = int(initialization_seed)
        model, checkpoint, path = _load_ec_model(
            config,
            split_seed=split_seed,
            initialization_seed=initialization_seed,
            device=device,
        )
        query_ids, features, n_prediction = _encode_queries(
            model=model,
            checkpoint=checkpoint,
            queries=queries,
            contexts=contexts,
            device=device,
        )
        if reference_ids is None:
            reference_ids = query_ids
        elif query_ids != reference_ids:
            raise MayrSiteInferenceAssetError("Checkpoint query order changed")
        ensemble_features.append(features)
        ensemble_n.append(n_prediction)
        bindings.append(
            {
                "initialization_seed": initialization_seed,
                "path": _display_path(path),
                "sha256": sha256_file(path),
                "model_state_sha256": checkpoint["model_state_sha256"],
            }
        )
        del model, checkpoint, features, n_prediction
        gc.collect()
    if reference_ids is None:
        raise MayrSiteInferenceAssetError("No Stage E-C checkpoints were encoded")
    n_matrix = np.stack(ensemble_n, axis=1)
    return (
        reference_ids,
        torch.from_numpy(np.concatenate(ensemble_features, axis=1)),
        n_matrix.mean(axis=1),
        n_matrix.std(axis=1),
        bindings,
    )


def ranker_from_checkpoint(
    checkpoint: Mapping[str, Any],
) -> IndependentSiteRanker | StructuredSiteRanker:
    """Instantiate and hash-verify a frozen independent site ranker."""

    architecture = checkpoint["ranker_architecture"]
    if not isinstance(architecture, Mapping):
        raise MayrSiteInferenceAssetError("Ranker architecture is missing")
    if architecture.get("schema_version") == STRUCTURED_RANKER_SCHEMA_VERSION:
        model: IndependentSiteRanker | StructuredSiteRanker = (
            structured_ranker_from_architecture(architecture)
        )
    else:
        input_dim = int(architecture["input_dim"])
        model = IndependentSiteRanker(
            input_dim=input_dim,
            arm=str(architecture["arm"]),
            feature_mean=torch.zeros(input_dim),
            feature_scale=torch.ones(input_dim),
            hidden_dim=int(architecture["hidden_dim"]),
            type_adapter_dim=int(architecture["type_adapter_dim"]),
        )
    state = checkpoint["ranker_state_dict"]
    if not isinstance(state, Mapping) or tensor_mapping_sha256(state) != str(
        checkpoint["ranker_state_sha256"]
    ):
        raise MayrSiteInferenceAssetError("Ranker checkpoint state hash changed")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def score_ranker_from_source_features(
    *,
    ranker: IndependentSiteRanker | StructuredSiteRanker,
    checkpoint: Mapping[str, Any],
    source_features: torch.Tensor,
    type_index: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Score frozen source features with either v1 or structured-v2 heads."""

    architecture = checkpoint.get("ranker_architecture")
    if not isinstance(architecture, Mapping):
        raise MayrSiteInferenceAssetError("Ranker architecture is missing")
    if isinstance(ranker, StructuredSiteRanker):
        try:
            views = reduce_frozen_ensemble_features(
                source_features,
                ensemble_size=int(architecture["ensemble_size"]),
                block_dim=int(architecture["block_dim"]),
            )
            return ranker.forward_components(
                views.candidate,
                views.context,
                type_index,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise MayrSiteInferenceAssetError(
                "Cannot score structured ranker source features"
            ) from exc
    logits = ranker(source_features, type_index)
    return {
        "canonical_logit": logits,
        "membership_logit": logits,
        "router_selected_logit": torch.zeros_like(logits),
        "compatibility_logit": logits,
    }


def load_ranker_checkpoint(
    path: Path,
    *,
    split_seed: int,
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Load a development-frozen ranker without crossing the test boundary."""

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise MayrSiteInferenceAssetError("Ranker checkpoint is not a mapping")
    expected = {
        "schema_version": str(
            config["ranker"].get(
                "checkpoint_schema",
                "nucpred.mayr-site-ranker-checkpoint.v1",
            )
        ),
        "phase": "development_frozen",
        "campaign_id": config["campaign_id"],
        "split_seed": split_seed,
        "test_labels_read": False,
        "test_predictions_computed": False,
        "conditional_n_backbone_frozen": True,
        "unknown_as_negative_count": 0,
        "candidate_softmax_used": False,
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise MayrSiteInferenceAssetError(
                f"Ranker checkpoint boundary changed: {key}"
            )
    return checkpoint
