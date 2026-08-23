"""Leak-free inner-fold training for the joint Mayr site-N prototype."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any

import numpy as np
import pandas as pd
import joblib

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from nucpred.core.files import atomic_write_json, sha256_file
from nucpred.experiments.mayr.joint_site_n import (
    DEFAULT_CONFIG,
    ROOT,
    JointSiteNExperimentError,
    _display_path,
    _project_path,
    read_config,
    verify_input_bindings,
)
from nucpred.experiments.mayr.site_n_formal import _tensor_mapping_sha256
from nucpred.publication.mayr_site_publication import (
    load_inner_conditional_checkpoint,
    read_config as read_site_config,
)
from nucpred.training.mayr_joint_site_n import (
    JointEvidenceState,
    MayrJointSiteNModel,
    frozen_teacher_n_harm,
    joint_optimizer_parameter_groups,
    joint_site_n_loss,
    set_heads_only_warmup,
    transfer_publication_conditional_n_checkpoint,
    transfer_publication_site_ranker_checkpoint,
)
from nucpred.training.mayr_joint_site_n_data import (
    JointSiteNCorpus,
    balanced_router_cell_weights,
    load_joint_site_n_corpus,
    pack_joint_site_n_batch,
    site_type_balanced_context_weights,
)
from nucpred.training.mayr_node_xtb_scratch import SolventVocabulary
from nucpred.training.mayr_site_region_residual import (
    apply_region_residual,
    region_feature_matrix,
    score_region_residual,
)
from nucpred.training.mayr_site_n import (
    SiteNExample,
    SiteNFoldPreprocessor,
    seed_everything,
)


INNER_CHECKPOINT_SCHEMA = "nucpred.mayr-joint-site-n-inner-checkpoint.v1"
INNER_SUMMARY_SCHEMA = "nucpred.mayr-joint-site-n-inner-summary.v1"
TRAINABLE_VARIANTS = (
    "joint_full",
    "frozen_backbone",
    "without_set_pooling",
    "without_evidence_bce",
    "without_n_harm",
    "without_historical_evidence",
)


@dataclass(frozen=True, slots=True)
class VariantSettings:
    name: str
    train_backbone_after_warmup: bool
    use_candidate_set_context: bool
    use_evidence_bce: bool
    use_n_harm: bool
    use_historical_evidence: bool


def variant_settings(name: str) -> VariantSettings:
    if name not in TRAINABLE_VARIANTS:
        raise JointSiteNExperimentError(f"Unsupported joint prototype variant: {name}")
    return VariantSettings(
        name=name,
        train_backbone_after_warmup=name != "frozen_backbone",
        use_candidate_set_context=name != "without_set_pooling",
        use_evidence_bce=name != "without_evidence_bce",
        use_n_harm=name != "without_n_harm",
        use_historical_evidence=name != "without_historical_evidence",
    )


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected = torch.device(value)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise JointSiteNExperimentError("CUDA was requested but is unavailable")
    return selected


def _membership_tables(config: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset = config["dataset"]
    return (
        pd.read_csv(
            _project_path(dataset["outer_membership_path"], label="outer membership")
        ),
        pd.read_csv(
            _project_path(dataset["nested_membership_path"], label="nested membership")
        ),
    )


def inner_target_ids(
    config: Mapping[str, Any],
    *,
    outer_fold: int,
    inner_fold: int,
) -> tuple[set[str], set[str], dict[str, object]]:
    """Return development-only inner roles and prove outer-test exclusion."""

    outer, nested = _membership_tables(config)
    selected_outer = outer.loc[outer["outer_fold"].eq(outer_fold)]
    outer_test = selected_outer.loc[selected_outer["role"].eq("test")]
    selected = nested.loc[
        nested["outer_fold"].eq(outer_fold)
        & nested["inner_fold"].eq(inner_fold)
    ]
    train = selected.loc[selected["role"].eq("train")]
    validation = selected.loc[selected["role"].eq("validation")]
    train_ids = set(train["target_id"].astype(str))
    validation_ids = set(validation["target_id"].astype(str))
    outer_test_ids = set(outer_test["target_id"].astype(str))
    train_connectivity = set(train["connectivity_id"].astype(str))
    validation_connectivity = set(validation["connectivity_id"].astype(str))
    outer_test_connectivity = set(outer_test["connectivity_id"].astype(str))
    if (
        not train_ids
        or not validation_ids
        or train_ids & validation_ids
        or (train_ids | validation_ids) & outer_test_ids
        or train_connectivity & validation_connectivity
        or train_connectivity & outer_test_connectivity
        or validation_connectivity & outer_test_connectivity
    ):
        raise JointSiteNExperimentError("Inner split leaks outer-test membership")
    return train_ids, validation_ids, {
        "schema_version": "nucpred.mayr-joint-site-n-inner-split-audit.v1",
        "outer_fold": outer_fold,
        "inner_fold": inner_fold,
        "train_target_count": len(train_ids),
        "validation_target_count": len(validation_ids),
        "outer_test_membership_target_count": len(outer_test_ids),
        "train_connectivity_count": len(train_connectivity),
        "validation_connectivity_count": len(validation_connectivity),
        "outer_test_connectivity_count": len(outer_test_connectivity),
        "all_pairwise_connectivity_overlap": 0,
        "outer_test_target_rows_loaded": 0,
    }


def _site_config(config: Mapping[str, Any]) -> tuple[dict[str, Any], Path]:
    path = _project_path(
        config["baseline"]["automatic_site_config_path"],
        label="automatic-site config",
    )
    return read_site_config(path)


def _joint_model_from_teacher(
    teacher: torch.nn.Module,
    *,
    config: Mapping[str, Any],
    vocabulary: SolventVocabulary,
    preprocessor: SiteNFoldPreprocessor,
    ranker_checkpoint: Mapping[str, object] | None,
    initialization_seed: int,
    settings: VariantSettings,
    device: torch.device,
) -> tuple[MayrJointSiteNModel, dict[str, object]]:
    base = teacher.frozen_parent.frozen_base
    architecture = base.architecture
    model_config = config["model"]
    expected = {
        "hidden_dim": int(model_config["hidden_dim"]),
        "layers": int(model_config["message_passing_layers"]),
        "node_embedding_dim": int(model_config["node_embedding_dim"]),
        "edge_embedding_dim": int(model_config["edge_embedding_dim"]),
        "solvent_embedding_dim": int(model_config["solvent_embedding_dim"]),
    }
    if any(int(architecture[name]) != value for name, value in expected.items()):
        raise JointSiteNExperimentError("Conditional teacher architecture drifted")
    if not math.isclose(
        float(architecture["dropout"]), float(model_config["dropout"]), abs_tol=1e-12
    ):
        raise JointSiteNExperimentError("Conditional teacher dropout drifted")
    model = new_joint_model(
        config=config,
        vocabulary=vocabulary,
        preprocessor=preprocessor,
        publication_ranker_architecture=(
            None
            if ranker_checkpoint is None
            else ranker_checkpoint["ranker_architecture"]
        ),
        initialization_seed=initialization_seed,
        settings=settings,
        device=device,
    )
    conditional_audit = transfer_publication_conditional_n_checkpoint(model, teacher)
    ranker_audit = (
        None
        if ranker_checkpoint is None
        else transfer_publication_site_ranker_checkpoint(model, ranker_checkpoint)
    )
    return model, {
        "schema_version": "nucpred.mayr-joint-site-n-combined-transfer.v1",
        "status": "pass",
        "exact_transfer": True,
        "conditional_n": conditional_audit,
        "site_ranker": ranker_audit,
        "site_logit_base_mode": (
            "external_split_safe_frozen_logits"
            if ranker_checkpoint is None
            else "internal_split_safe_frozen_publication_ranker"
        ),
    }


def new_joint_model(
    *,
    config: Mapping[str, Any],
    vocabulary: SolventVocabulary,
    preprocessor: SiteNFoldPreprocessor,
    publication_ranker_architecture: Mapping[str, object] | None,
    initialization_seed: int,
    settings: VariantSettings,
    device: torch.device,
) -> MayrJointSiteNModel:
    """Instantiate a deterministic joint model before exact state transfer."""

    model_config = config["model"]
    seed_everything(initialization_seed)
    model = MayrJointSiteNModel(
        num_solvents=len(vocabulary.tokens),
        hidden_dim=int(model_config["hidden_dim"]),
        layers=int(model_config["message_passing_layers"]),
        node_embedding_dim=int(model_config["node_embedding_dim"]),
        edge_embedding_dim=int(model_config["edge_embedding_dim"]),
        solvent_embedding_dim=int(model_config["solvent_embedding_dim"]),
        type_embedding_dim=int(model_config["type_embedding_dim"]),
        router_hidden_dim=int(model_config["router_hidden_dim"]),
        router_logit_weight=float(model_config["router_logit_weight"]),
        set_top_k=int(model_config["set_top_k"]),
        use_candidate_set_context=settings.use_candidate_set_context,
        publication_n_lineage=True,
        publication_ranker_architecture=publication_ranker_architecture,
        publication_n_target_mean=float(preprocessor.target_mean),
        publication_n_target_scale=float(preprocessor.target_scale),
        dropout=float(model_config["dropout"]),
    )
    return model.to(device)


def load_split_safe_site_ranker(
    config: Mapping[str, Any],
    *,
    outer_fold: int,
    inner_fold: int | None,
    conditional_teacher_path: Path,
) -> tuple[dict[str, object], Path, dict[str, object]]:
    """Load the publication ranker matching the current development boundary."""

    if inner_fold is None:
        path = (
            _project_path(
                config["baseline"]["outer_site_ranker_root"],
                label="outer site-ranker root",
            )
            / f"outer-{outer_fold}"
            / "ranker_checkpoint.pt"
        )
        phase = "outer_development_refit"
    else:
        path = (
            _project_path(
                config["baseline"]["inner_site_ranker_root"],
                label="inner site-ranker root",
            )
            / f"outer-{outer_fold}"
            / f"inner-{inner_fold}"
            / "ranker_checkpoint.pt"
        )
        phase = "nested_inner_selection"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise JointSiteNExperimentError("Site-ranker checkpoint is not a mapping")
    state = checkpoint.get("ranker_state_dict")
    bindings = checkpoint.get("conditional_n_bindings")
    if (
        checkpoint.get("schema_version")
        != "nucpred.mayr-n-publication-site-ranker-checkpoint.v1"
        or checkpoint.get("phase") != phase
        or int(checkpoint.get("outer_fold", -1)) != outer_fold
        or checkpoint.get("inner_fold") != inner_fold
        or checkpoint.get("selected_arm") != "hierarchical_exact"
        or checkpoint.get("candidate_softmax_used") is not False
        or int(checkpoint.get("unknown_as_negative_count", -1)) != 0
        or int(checkpoint.get("outer_test_target_rows_loaded", -1)) != 0
        or checkpoint.get("outer_test_predictions_computed") is not False
        or not isinstance(state, Mapping)
        or not all(isinstance(value, torch.Tensor) for value in state.values())
        or _tensor_mapping_sha256(state) != checkpoint.get("ranker_state_sha256")
        or not isinstance(bindings, list)
    ):
        raise JointSiteNExperimentError("Split-safe site-ranker contract changed")
    teacher_sha256 = sha256_file(conditional_teacher_path)
    matching = [
        binding
        for binding in bindings
        if isinstance(binding, Mapping)
        and _project_path(
            binding.get("path"), label="ranker conditional-N binding"
        )
        == conditional_teacher_path.resolve()
        and binding.get("sha256") == teacher_sha256
    ]
    if len(matching) != 1:
        raise JointSiteNExperimentError(
            "Site ranker is not bound to the conditional-N teacher"
        )
    return checkpoint, path, {
        "schema_version": "nucpred.mayr-joint-site-n-ranker-binding-audit.v1",
        "status": "pass",
        "phase": phase,
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(path),
        "ranker_state_sha256": checkpoint["ranker_state_sha256"],
        "conditional_teacher_path": conditional_teacher_path.relative_to(
            ROOT
        ).as_posix(),
        "conditional_teacher_sha256": teacher_sha256,
        "outer_test_target_rows_loaded": 0,
        "outer_test_predictions_computed": False,
    }


def audit_inner_ranker_initialization(
    candidate_scores: pd.DataFrame,
    *,
    reference_path: Path,
    maximum_absolute_delta: float = 5e-5,
) -> dict[str, object]:
    """Prove epoch-0 logits reproduce the split-safe publication base ranker."""

    reference = pd.read_parquet(
        reference_path,
        columns=["query_id", "validity_logit"],
    )
    if candidate_scores["query_id"].astype(str).duplicated().any() or reference[
        "query_id"
    ].astype(str).duplicated().any():
        raise JointSiteNExperimentError("Ranker reproduction query IDs are duplicated")
    reference["publication_canonical_logit"] = reference["validity_logit"]
    compared = candidate_scores[["query_id", "canonical_logit"]].merge(
        reference[["query_id", "publication_canonical_logit"]],
        on="query_id",
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    if not compared["_merge"].eq("both").all():
        raise JointSiteNExperimentError(
            "Joint and publication initialization candidate universes differ"
        )
    difference = (
        compared["canonical_logit"] - compared["publication_canonical_logit"]
    ).abs()
    observed_maximum = float(difference.max())
    if observed_maximum > maximum_absolute_delta:
        raise JointSiteNExperimentError(
            "Split-safe publication ranker initialization was not reproduced"
        )
    return {
        "schema_version": "nucpred.mayr-joint-site-n-ranker-reproduction.v1",
        "status": "pass",
        "candidate_count": int(len(compared)),
        "candidate_identity_exact": True,
        "site_type_indices_remapped_by_name": True,
        "maximum_absolute_logit_delta": observed_maximum,
        "mean_absolute_logit_delta": float(difference.mean()),
        "maximum_allowed_absolute_logit_delta": maximum_absolute_delta,
        "reference_path": _display_path(reference_path),
        "reference_sha256": sha256_file(reference_path),
        "reference_uses_labels": False,
        "region_residual_included": True,
    }


def transfer_inner_validation_base_logits(
    base_logits: Mapping[str, float],
    base_router_logits: Mapping[str, float],
    *,
    reference_path: Path,
    expected_query_ids: Sequence[str],
) -> tuple[dict[str, float], dict[str, float], dict[str, object]]:
    """Transfer frozen publication validation scores by exact query identity.

    Recomputing the region residual can change which nearly tied candidate lies
    on its top-k boundary.  The publication validation predictions are the
    authoritative, already-frozen label-blind model outputs for this role, so
    use them directly rather than treating a fresh floating-point forward pass
    as the frozen artifact.
    """

    expected = tuple(map(str, expected_query_ids))
    if not expected or len(set(expected)) != len(expected):
        raise JointSiteNExperimentError(
            "Inner validation reference query IDs are empty or duplicated"
        )
    expected_set = set(expected)
    if not expected_set <= set(base_logits) or not expected_set <= set(
        base_router_logits
    ):
        raise JointSiteNExperimentError(
            "Inner validation reference is outside the materialized base universe"
        )
    columns = ["query_id", "validity_logit", "router_selected_logit"]
    reference = pd.read_parquet(reference_path, columns=columns)
    reference["query_id"] = reference["query_id"].astype(str)
    if (
        reference["query_id"].duplicated().any()
        or set(reference["query_id"]) != expected_set
    ):
        raise JointSiteNExperimentError(
            "Frozen publication validation candidate identity changed"
        )
    reference = reference.set_index("query_id").loc[list(expected)]
    validity = reference["validity_logit"].to_numpy(dtype=float)
    router = reference["router_selected_logit"].to_numpy(dtype=float)
    if not np.isfinite(validity).all() or not np.isfinite(router).all():
        raise JointSiteNExperimentError(
            "Frozen publication validation logits are non-finite"
        )

    recomputed = np.asarray([float(base_logits[query_id]) for query_id in expected])
    recomputed_router = np.asarray(
        [float(base_router_logits[query_id]) for query_id in expected]
    )
    canonical_delta = np.abs(recomputed - validity)
    router_delta = np.abs(recomputed_router - router)
    transferred = dict(base_logits)
    transferred_router = dict(base_router_logits)
    transferred.update(dict(zip(expected, map(float, validity), strict=True)))
    transferred_router.update(dict(zip(expected, map(float, router), strict=True)))
    return transferred, transferred_router, {
        "schema_version": (
            "nucpred.mayr-joint-site-n-inner-validation-base-transfer.v1"
        ),
        "status": "pass",
        "source": "frozen_publication_inner_validation_predictions_by_query_id",
        "candidate_count": len(expected),
        "candidate_identity_exact": True,
        "columns_read": columns,
        "N_value_column_requested": False,
        "exact_label_column_requested": False,
        "role_column_requested": False,
        "reference_path": _display_path(reference_path),
        "reference_sha256": sha256_file(reference_path),
        "recomputed_maximum_absolute_logit_delta": float(canonical_delta.max()),
        "recomputed_mean_absolute_logit_delta": float(canonical_delta.mean()),
        "recomputed_count_above_5e_5": int((canonical_delta > 5e-5).sum()),
        "recomputed_router_maximum_absolute_logit_delta": float(
            router_delta.max()
        ),
        "combined_mapping_sha256": _frozen_logit_sha256(transferred),
        "combined_router_mapping_sha256": _frozen_logit_sha256(
            transferred_router
        ),
        "outer_test_target_rows_loaded": 0,
        "outer_test_predictions_computed": False,
    }


def _example_batches(
    examples: Sequence[SiteNExample],
    *,
    batch_size: int,
    shuffle_seed: int | None,
) -> list[list[SiteNExample]]:
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    indices = np.arange(len(examples))
    if shuffle_seed is not None:
        np.random.default_rng(int(shuffle_seed)).shuffle(indices)
    return [
        [
            examples[int(index)]
            for index in indices[start : start + batch_size]
        ]
        for start in range(0, len(indices), batch_size)
    ]


def _batch_query_ids(examples: Sequence[SiteNExample]) -> tuple[str, ...]:
    return tuple(query_id for example in examples for query_id in example.target_ids)


def _frozen_base_logit_mapping(
    corpus: JointSiteNCorpus,
) -> Mapping[str, float] | None:
    """Return a complete, finite external base offset when one is attached."""

    column = "frozen_base_canonical_logit"
    if column not in corpus.queries:
        return None
    query_ids = corpus.queries["query_id"].astype(str)
    if query_ids.duplicated().any():
        raise JointSiteNExperimentError("Frozen base-logit query IDs are duplicated")
    values = corpus.queries[column].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise JointSiteNExperimentError("Frozen base logits are not finite")
    return dict(zip(query_ids, map(float, values), strict=True))


def _frozen_base_router_logit_mapping(
    corpus: JointSiteNCorpus,
) -> Mapping[str, float] | None:
    column = "frozen_base_router_selected_logit"
    if column not in corpus.queries:
        return None
    query_ids = corpus.queries["query_id"].astype(str)
    values = corpus.queries[column].to_numpy(dtype=float)
    if query_ids.duplicated().any() or not np.isfinite(values).all():
        raise JointSiteNExperimentError("Frozen base router logits are invalid")
    return dict(zip(query_ids, map(float, values), strict=True))


def _frozen_logit_sha256(values: Mapping[str, float]) -> str:
    digest = hashlib.sha256()
    for query_id in sorted(values):
        value = float(values[query_id])
        if not math.isfinite(value):
            raise JointSiteNExperimentError("Frozen base logits are not finite")
        digest.update(query_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(np.float64(value).tobytes())
    return digest.hexdigest()


def attach_frozen_base_logits(
    corpus: JointSiteNCorpus,
    values: Mapping[str, float],
    router_selected_values: Mapping[str, float],
) -> JointSiteNCorpus:
    """Attach an exact query-aligned frozen base score vector to a corpus."""

    query_ids = corpus.queries["query_id"].astype(str)
    expected = set(query_ids)
    observed = set(map(str, values))
    observed_router = set(map(str, router_selected_values))
    if (
        query_ids.duplicated().any()
        or expected != observed
        or expected != observed_router
    ):
        raise JointSiteNExperimentError(
            "Frozen base-logit candidate identity does not match the corpus"
        )
    queries = corpus.queries.copy()
    queries["frozen_base_canonical_logit"] = np.asarray(
        [float(values[query_id]) for query_id in query_ids],
        dtype=float,
    )
    queries["frozen_base_router_selected_logit"] = np.asarray(
        [float(router_selected_values[query_id]) for query_id in query_ids],
        dtype=float,
    )
    if not np.isfinite(
        queries[
            [
                "frozen_base_canonical_logit",
                "frozen_base_router_selected_logit",
            ]
        ].to_numpy(dtype=float)
    ).all():
        raise JointSiteNExperimentError("Frozen base logits are not finite")
    return replace(corpus, queries=queries)


def freeze_publication_base_logits(
    model: MayrJointSiteNModel,
    examples: Sequence[SiteNExample],
    *,
    corpus: JointSiteNCorpus,
    preprocessor: SiteNFoldPreprocessor,
    vocabulary: SolventVocabulary,
    batch_size: int,
    device: torch.device,
    region_residual_path: Path,
    ranker_summary_path: Path,
    outer_fold: int,
    inner_fold: int,
) -> tuple[dict[str, float], dict[str, float], dict[str, object]]:
    """Materialize an immutable epoch-0 publication ranker score for every query."""

    ranker = model.publication_site_ranker
    if ranker is None or any(parameter.requires_grad for parameter in ranker.parameters()):
        raise JointSiteNExperimentError(
            "Publication base scorer must exist and be parameter-frozen"
        )
    model.eval()
    values: dict[str, float] = {}
    router_values: dict[str, float] = {}
    membership_values: dict[str, float] = {}
    compatibility_values: dict[str, float] = {}
    conditional_n_values: dict[str, float] = {}
    maximum_residual = 0.0
    with torch.no_grad():
        for selected in _example_batches(
            examples,
            batch_size=batch_size,
            shuffle_seed=None,
        ):
            batch = pack_joint_site_n_batch(
                selected,
                query_table=corpus.queries,
                preprocessor=preprocessor,
                solvent_vocabulary=vocabulary,
            ).to(device)
            output = model(batch.inputs)
            residual = output.residual_canonical_logits.detach().abs()
            if residual.numel():
                maximum_residual = max(
                    maximum_residual,
                    float(residual.max().cpu()),
                )
            selected_router = output.base_router_logits[
                batch.inputs.site_graph_index,
                batch.inputs.site_type_index,
            ]
            conditional_n = (
                output.n_prediction_standardized * float(preprocessor.target_scale)
                + float(preprocessor.target_mean)
            )
            for (
                query_id,
                value,
                router_value,
                membership_value,
                compatibility_value,
                conditional_n_value,
            ) in zip(
                _batch_query_ids(selected),
                output.base_canonical_logits.detach().cpu().tolist(),
                selected_router.detach().cpu().tolist(),
                output.membership_logits.detach().cpu().tolist(),
                output.base_compatibility_logits.detach().cpu().tolist(),
                conditional_n.detach().cpu().tolist(),
                strict=True,
            ):
                if str(query_id) in values:
                    raise JointSiteNExperimentError(
                        "Publication base scorer emitted a duplicate query"
                    )
                values[str(query_id)] = float(value)
                router_values[str(query_id)] = float(router_value)
                membership_values[str(query_id)] = float(membership_value)
                compatibility_values[str(query_id)] = float(compatibility_value)
                conditional_n_values[str(query_id)] = float(conditional_n_value)
    expected = set(corpus.queries["query_id"].astype(str))
    component_mappings = (
        values,
        router_values,
        membership_values,
        compatibility_values,
        conditional_n_values,
    )
    if any(set(mapping) != expected for mapping in component_mappings):
        raise JointSiteNExperimentError(
            "Publication base components do not cover the candidate corpus"
        )

    summary = json.loads(ranker_summary_path.read_text(encoding="utf-8"))
    selected_region = summary.get("selected_region")
    if (
        summary.get("schema_version")
        != "nucpred.mayr-n-publication-site-inner-fit.v1"
        or summary.get("status") != "pass"
        or int(summary.get("outer_fold", -1)) != outer_fold
        or int(summary.get("inner_fold", -1)) != inner_fold
        or summary.get("candidate_softmax_used") is not False
        or int(summary.get("unknown_as_negative_count", -1)) != 0
        or int(summary.get("outer_test_target_rows_loaded", -1)) != 0
        or summary.get("outer_test_predictions_computed") is not False
        or not isinstance(selected_region, Mapping)
    ):
        raise JointSiteNExperimentError("Split-safe inner region contract changed")
    base_before_region_sha256 = _frozen_logit_sha256(values)
    region_audit: dict[str, object]
    if selected_region.get("arm") == "region_structural_residual":
        if not region_residual_path.is_file():
            raise JointSiteNExperimentError("Split-safe inner region bundle is missing")
        bundle = joblib.load(region_residual_path)
        if not isinstance(bundle, Mapping):
            raise JointSiteNExperimentError("Inner region residual is not a mapping")
        query_ids = tuple(corpus.queries["query_id"].astype(str))
        frame = corpus.queries.reset_index(drop=True)
        positions, features, feature_names = region_feature_matrix(
            frame,
            membership_logits=[membership_values[query_id] for query_id in query_ids],
            compatibility_logits=[
                compatibility_values[query_id] for query_id in query_ids
            ],
            conditional_n_mean=[
                conditional_n_values[query_id] for query_id in query_ids
            ],
            conditional_n_std=np.zeros(len(query_ids), dtype=float),
            origin_vocabulary_values=tuple(map(str, bundle["origin_vocabulary"])),
        )
        probabilities = score_region_residual(
            bundle,
            features,
            expected_feature_names=feature_names,
        )
        adjusted, application = apply_region_residual(
            frame,
            base_logits=[values[query_id] for query_id in query_ids],
            region_positions=positions,
            residual_probabilities=probabilities,
            residual_weight=float(selected_region["residual_weight"]),
            maximum_base_margin=(
                None
                if selected_region.get("maximum_base_margin") is None
                else float(selected_region["maximum_base_margin"])
            ),
            top_k=(
                None
                if selected_region.get("top_k") is None
                else int(selected_region["top_k"])
            ),
        )
        values = dict(zip(query_ids, map(float, adjusted), strict=True))
        region_audit = {
            "status": "pass",
            "applied": True,
            "selected_region": dict(selected_region),
            "application": application,
            "bundle_path": _display_path(region_residual_path),
            "bundle_sha256": sha256_file(region_residual_path),
            "feature_names": list(feature_names),
            "region_candidate_count": int(len(positions)),
        }
    elif selected_region.get("arm") == "frozen_hierarchical_exact":
        region_audit = {
            "status": "pass",
            "applied": False,
            "selected_region": dict(selected_region),
        }
    else:
        raise JointSiteNExperimentError("Unknown split-safe inner region arm")

    router_frame = corpus.queries[["query_id", "context_id", "site_type"]].copy()
    router_frame["router_selected_logit"] = router_frame["query_id"].astype(str).map(
        router_values
    )
    router_spread = router_frame.groupby(
        ["context_id", "site_type"], sort=False
    )["router_selected_logit"].agg(lambda rows: float(rows.max() - rows.min()))
    router_maximum_delta = (
        float(router_spread.max()) if len(router_spread) else 0.0
    )
    if (
        set(values) != expected
        or maximum_residual > 1e-8
        or not math.isfinite(router_maximum_delta)
        or router_maximum_delta > 5e-5
    ):
        raise JointSiteNExperimentError(
            "Publication base-score materialization changed candidate coverage or residual"
        )
    return values, router_values, {
        "schema_version": "nucpred.mayr-joint-site-n-frozen-base-materialization.v1",
        "status": "pass",
        "source": "split_safe_publication_inner_ranker_plus_region_residual",
        "candidate_count": len(values),
        "candidate_identity_exact": True,
        "frozen_before_optimizer_step": True,
        "ranker_parameters_trainable": False,
        "base_logits_require_grad": False,
        "maximum_initial_residual_absolute_logit": maximum_residual,
        "mapping_sha256": _frozen_logit_sha256(values),
        "base_before_region_mapping_sha256": base_before_region_sha256,
        "router_mapping_sha256": _frozen_logit_sha256(router_values),
        "router_context_type_maximum_delta": router_maximum_delta,
        "region_residual_audit": region_audit,
        "ranker_summary_path": _display_path(ranker_summary_path),
        "ranker_summary_sha256": sha256_file(ranker_summary_path),
        "outer_test_target_rows_loaded": 0,
        "outer_test_predictions_computed": False,
    }


def _region_residual_source_binding(
    *,
    region_residual_path: Path,
    frozen_base_audit: Mapping[str, object],
) -> dict[str, object]:
    """Bind a selected region bundle, or prove that no bundle was selected."""

    region_audit = frozen_base_audit.get("region_residual_audit")
    if not isinstance(region_audit, Mapping):
        raise JointSiteNExperimentError("Frozen base audit has no region contract")
    selected_region = region_audit.get("selected_region")
    if (
        region_audit.get("status") != "pass"
        or not isinstance(selected_region, Mapping)
    ):
        raise JointSiteNExperimentError("Frozen base region contract changed")

    arm = selected_region.get("arm")
    if arm == "region_structural_residual":
        if region_audit.get("applied") is not True:
            raise JointSiteNExperimentError(
                "Selected region residual was not applied"
            )
        if not region_residual_path.is_file():
            raise JointSiteNExperimentError("Selected region residual is missing")
        digest = sha256_file(region_residual_path)
        if region_audit.get("bundle_sha256") != digest:
            raise JointSiteNExperimentError("Selected region residual hash changed")
        return {
            "status": "selected_and_bound",
            "selected_arm": str(arm),
            "path": _display_path(region_residual_path),
            "sha256": digest,
        }

    if arm == "frozen_hierarchical_exact":
        if region_audit.get("applied") is not False:
            raise JointSiteNExperimentError(
                "Unselected region residual has an invalid application state"
            )
        if region_residual_path.exists():
            raise JointSiteNExperimentError(
                "Unselected region residual artifact is unexpectedly present"
            )
        if (
            region_audit.get("bundle_sha256") is not None
            or region_audit.get("bundle_path") is not None
        ):
            raise JointSiteNExperimentError(
                "Unselected region residual carries a stale bundle binding"
            )
        return {
            "status": "not_selected",
            "selected_arm": str(arm),
            "path": _display_path(region_residual_path),
            "sha256": None,
        }

    raise JointSiteNExperimentError("Unknown frozen base region arm")


def compute_teacher_n_harm(
    model: MayrJointSiteNModel,
    examples: Sequence[SiteNExample],
    *,
    corpus: JointSiteNCorpus,
    preprocessor: SiteNFoldPreprocessor,
    vocabulary: SolventVocabulary,
    batch_size: int,
    device: torch.device,
    cap_quantile: float,
    use_historical_evidence: bool,
) -> tuple[dict[str, float], dict[str, object]]:
    """Freeze candidate N-harm before the first optimizer step."""

    if not use_historical_evidence:
        return {}, {
            "status": "disabled_without_historical_evidence",
            "stored_endpoint_exclusion_count": 0,
            "unknown_nonzero_count": 0,
        }
    model.eval()
    frozen_base_logits = _frozen_base_logit_mapping(corpus)
    frozen_base_router_logits = _frozen_base_router_logit_mapping(corpus)
    values: dict[str, float] = {}
    positive_count = 0
    excluded_count = 0
    with torch.no_grad():
        for selected in _example_batches(
            examples,
            batch_size=batch_size,
            shuffle_seed=None,
        ):
            raw_batch = pack_joint_site_n_batch(
                selected,
                query_table=corpus.queries,
                preprocessor=preprocessor,
                solvent_vocabulary=vocabulary,
                base_canonical_logits=frozen_base_logits,
                base_router_selected_logits=frozen_base_router_logits,
            )
            batch = raw_batch.to(device)
            output = model(
                batch.inputs,
                base_canonical_logits=batch.base_canonical_logits,
                base_router_selected_logits=batch.base_router_selected_logits,
            )
            harm = frozen_teacher_n_harm(
                output.n_prediction_standardized,
                batch.n_target_standardized,
                batch.retrieval_positive_mask,
                batch.inputs.site_graph_index,
                cap_quantile=cap_quantile,
            )
            excluded = batch.evidence_state.eq(
                int(JointEvidenceState.ENDPOINT_EXCLUDED)
            )
            ids = _batch_query_ids(selected)
            for query_id, value, keep in zip(
                ids,
                harm.detach().cpu().tolist(),
                excluded.detach().cpu().tolist(),
                strict=True,
            ):
                if keep:
                    values[str(query_id)] = float(value)
            positive_count += int(batch.retrieval_positive_mask.sum())
            excluded_count += int(excluded.sum())
    array = np.asarray(list(values.values()), dtype=float)
    if len(values) != excluded_count or not np.isfinite(array).all():
        raise JointSiteNExperimentError("Frozen teacher N-harm coverage changed")
    return values, {
        "schema_version": "nucpred.mayr-joint-site-n-teacher-harm-audit.v1",
        "status": "pass",
        "stop_gradient": True,
        "cap_quantile": float(cap_quantile),
        "positive_count": positive_count,
        "stored_endpoint_exclusion_count": excluded_count,
        "unknown_nonzero_count": 0,
        "minimum": float(array.min()) if array.size else 0.0,
        "mean": float(array.mean()) if array.size else 0.0,
        "maximum": float(array.max()) if array.size else 0.0,
    }


def _context_metric_summary(frame: pd.DataFrame) -> dict[str, float | int]:
    if frame.empty:
        raise JointSiteNExperimentError("No single-target contexts were scored")
    correct = frame["site_top1_correct"].astype(bool)
    wrong = ~correct
    automatic = frame["automatic_n_error"].abs()
    known = frame["known_site_n_error"].abs()
    return {
        "context_count": int(len(frame)),
        "connectivity_count": int(frame["connectivity_id"].astype(str).nunique()),
        "exact_top1": float(correct.mean()),
        "exact_top3": float(frame["site_top3_correct"].mean()),
        "exact_top5": float(frame["site_top5_correct"].mean()),
        "mrr": float(frame["site_reciprocal_rank"].mean()),
        "automatic_n_mae": float(automatic.mean()),
        "known_site_n_mae": float(known.mean()),
        "site_addressable_n_mae_gap": float(automatic.mean() - known.mean()),
        "correct_site_count": int(correct.sum()),
        "correct_site_n_mae": (
            float(automatic.loc[correct].mean()) if bool(correct.any()) else math.nan
        ),
        "wrong_site_count": int(wrong.sum()),
        "wrong_site_n_mae": (
            float(automatic.loc[wrong].mean()) if bool(wrong.any()) else math.nan
        ),
        "candidate_recall": 1.0,
    }


def _type_top1_curve_columns(frame: pd.DataFrame) -> dict[str, float]:
    return {
        f"validation_type_{row.site_type}_exact_top1": float(row.exact_top1)
        for row in frame.itertuples(index=False)
    }


def evaluate_labeled(
    model: MayrJointSiteNModel,
    examples: Sequence[SiteNExample],
    *,
    corpus: JointSiteNCorpus,
    preprocessor: SiteNFoldPreprocessor,
    vocabulary: SolventVocabulary,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate raw logits and conditional N on labeled development data."""

    model.eval()
    frozen_base_logits = _frozen_base_logit_mapping(corpus)
    frozen_base_router_logits = _frozen_base_router_logit_mapping(corpus)
    score_rows: list[dict[str, object]] = []
    metadata = corpus.queries.set_index("query_id", drop=False)
    with torch.no_grad():
        for selected in _example_batches(
            examples,
            batch_size=batch_size,
            shuffle_seed=None,
        ):
            batch = pack_joint_site_n_batch(
                selected,
                query_table=corpus.queries,
                preprocessor=preprocessor,
                solvent_vocabulary=vocabulary,
                base_canonical_logits=frozen_base_logits,
                base_router_selected_logits=frozen_base_router_logits,
            ).to(device)
            output = model(
                batch.inputs,
                base_canonical_logits=batch.base_canonical_logits,
                base_router_selected_logits=batch.base_router_selected_logits,
            )
            logits = output.canonical_logits.detach().cpu().numpy()
            base_output = output.base_canonical_logits.detach().cpu().numpy()
            residual_logits = output.residual_canonical_logits.detach().cpu().numpy()
            residual_router = output.residual_router_logits[
                batch.inputs.site_graph_index,
                batch.inputs.site_type_index,
            ].detach().cpu().numpy()
            residual_membership = (
                output.residual_canonical_logits
                - float(model.router_logit_weight)
                * output.residual_router_logits[
                    batch.inputs.site_graph_index,
                    batch.inputs.site_type_index,
                ]
            ).detach().cpu().numpy()
            predictions = (
                output.n_prediction_standardized.detach().cpu().numpy()
                * float(preprocessor.target_scale)
                + float(preprocessor.target_mean)
            )
            for (
                query_id,
                logit,
                base_logit,
                residual_logit,
                residual_membership_logit,
                residual_router_logit,
                prediction,
            ) in zip(
                _batch_query_ids(selected),
                logits,
                base_output,
                residual_logits,
                residual_membership,
                residual_router,
                predictions,
                strict=True,
            ):
                row = metadata.loc[str(query_id)]
                score_rows.append(
                    {
                        "query_id": str(query_id),
                        "context_id": str(row["context_id"]),
                        "species_id": str(row["species_id"]),
                        "connectivity_id": str(row["connectivity_id"]),
                        "candidate_site_id": str(row["candidate_site_id"]),
                        "site_type": str(row["site_type"]),
                        "canonical_logit": float(logit),
                        "base_canonical_logit": float(base_logit),
                        "residual_canonical_logit": float(residual_logit),
                        "residual_membership_logit": float(
                            residual_membership_logit
                        ),
                        "residual_router_selected_logit": float(
                            residual_router_logit
                        ),
                        "conditional_n_prediction": float(prediction),
                        "evidence_state": int(row["evidence_state"]),
                        "N_value": float(row["N_value"]),
                    }
                )
    scores = pd.DataFrame(score_rows)
    if len(scores) != sum(example.num_sites for example in examples):
        raise JointSiteNExperimentError("Evaluation candidate coverage changed")
    if not np.allclose(
        scores["canonical_logit"].to_numpy(dtype=float),
        scores["base_canonical_logit"].to_numpy(dtype=float)
        + scores["residual_canonical_logit"].to_numpy(dtype=float),
        rtol=0.0,
        atol=2e-6,
    ):
        raise JointSiteNExperimentError("Labeled score decomposition changed")
    scores = scores.sort_values(
        ["context_id", "canonical_logit", "query_id"],
        ascending=[True, False, True],
        kind="stable",
    ).reset_index(drop=True)
    scores["candidate_rank"] = scores.groupby("context_id").cumcount() + 1
    context_rows: list[dict[str, object]] = []
    for context_id, group in scores.groupby("context_id", sort=True):
        positive = group.loc[
            group["evidence_state"].eq(int(JointEvidenceState.POSITIVE_EXACT))
        ]
        if len(positive) != 1:
            continue
        truth = positive.iloc[0]
        top = group.iloc[0]
        exact_rank = int(truth["candidate_rank"])
        n_true = float(truth["N_value"])
        context_rows.append(
            {
                "context_id": str(context_id),
                "species_id": str(top["species_id"]),
                "connectivity_id": str(top["connectivity_id"]),
                "true_candidate_site_id": str(truth["candidate_site_id"]),
                "true_site_type": str(truth["site_type"]),
                "predicted_candidate_site_id": str(top["candidate_site_id"]),
                "predicted_site_type": str(top["site_type"]),
                "candidate_count": int(len(group)),
                "exact_rank": exact_rank,
                "site_top1_correct": exact_rank <= 1,
                "site_top3_correct": exact_rank <= 3,
                "site_top5_correct": exact_rank <= 5,
                "site_reciprocal_rank": 1.0 / exact_rank,
                "N_true": n_true,
                "automatic_n_prediction": float(top["conditional_n_prediction"]),
                "known_site_n_prediction": float(
                    truth["conditional_n_prediction"]
                ),
                "automatic_n_error": float(top["conditional_n_prediction"]) - n_true,
                "known_site_n_error": (
                    float(truth["conditional_n_prediction"]) - n_true
                ),
                "top1_canonical_logit": float(top["canonical_logit"]),
                "top1_margin": (
                    float(top["canonical_logit"] - group.iloc[1]["canonical_logit"])
                    if len(group) > 1
                    else math.nan
                ),
            }
        )
    contexts = pd.DataFrame(context_rows)
    overall = _context_metric_summary(contexts)
    type_rows = [
        {"site_type": str(site_type), **_context_metric_summary(selected)}
        for site_type, selected in contexts.groupby("true_site_type", sort=True)
    ]
    by_type = pd.DataFrame(type_rows)
    residual_absolute = scores["residual_canonical_logit"].abs()
    return (
        {
            "schema_version": "nucpred.mayr-joint-site-n-labeled-evaluation.v1",
            "primary_population": "single_target_contexts",
            "overall": overall,
            "single_target_context_count": int(len(contexts)),
            "multi_target_context_count": int(
                scores["context_id"].nunique() - len(contexts)
            ),
            "direct_outputs_only": True,
            "calibrated_probability_computed": False,
            "conformal_p_value_computed": False,
            "score_diagnostics": {
                "residual_absolute_mean": float(residual_absolute.mean()),
                "residual_absolute_p95": float(residual_absolute.quantile(0.95)),
                "residual_absolute_maximum": float(residual_absolute.max()),
            },
        },
        scores,
        contexts,
        by_type,
    )


def _train_epoch(
    model: MayrJointSiteNModel,
    examples: Sequence[SiteNExample],
    *,
    corpus: JointSiteNCorpus,
    preprocessor: SiteNFoldPreprocessor,
    vocabulary: SolventVocabulary,
    candidate_n_harm: Mapping[str, float],
    router_cell_weights: Mapping[tuple[str, str], float],
    site_context_weights: Mapping[str, float],
    settings: VariantSettings,
    config: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    shuffle_seed: int,
) -> dict[str, float | int]:
    model.train()
    training = config["training"]
    loss_config = config["loss"]
    totals = {
        "total": 0.0,
        "listwise": 0.0,
        "pairwise": 0.0,
        "evidence": 0.0,
        "router": 0.0,
        "router_endpoint_type": 0.0,
        "router_bce": 0.0,
        "router_pairwise": 0.0,
        "n_regression": 0.0,
    }
    contexts = 0
    counts = {
        "pair_count": 0,
        "evidence_positive_count": 0,
        "evidence_negative_count": 0,
        "evidence_unknown_count": 0,
        "evidence_out_of_scope_count": 0,
        "n_supervision_count": 0,
        "router_pair_count": 0,
        "router_endpoint_context_count": 0,
        "router_reviewed_cell_count": 0,
    }
    base_logits = _frozen_base_logit_mapping(corpus)
    base_router_logits = _frozen_base_router_logit_mapping(corpus)
    for selected in _example_batches(
        examples,
        batch_size=int(training["batch_size_contexts"]),
        shuffle_seed=shuffle_seed,
    ):
        batch = pack_joint_site_n_batch(
            selected,
            query_table=corpus.queries,
            preprocessor=preprocessor,
            solvent_vocabulary=vocabulary,
            candidate_n_harm=candidate_n_harm,
            base_canonical_logits=base_logits,
            base_router_selected_logits=base_router_logits,
            router_cell_weights=router_cell_weights,
            site_context_weights=site_context_weights,
            use_historical_evidence=settings.use_historical_evidence,
        ).to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(
            batch.inputs,
            base_canonical_logits=batch.base_canonical_logits,
            base_router_selected_logits=batch.base_router_selected_logits,
        )
        total, parts = joint_site_n_loss(
            output,
            batch,
            listwise_weight=float(loss_config["listwise_weight"]),
            pairwise_weight=float(loss_config["pairwise_weight"]),
            evidence_weight=(
                float(loss_config["evidence_bce_weight"])
                if settings.use_evidence_bce
                else 0.0
            ),
            router_weight=float(loss_config["type_router_weight"]),
            n_weight=float(loss_config["n_regression_weight"]),
            pairwise_margin=float(loss_config["pairwise_margin"]),
            n_harm_multiplier=(
                float(loss_config["n_harm_multiplier"])
                if settings.use_n_harm
                else 0.0
            ),
        )
        if not bool(torch.isfinite(total)):
            raise JointSiteNExperimentError("Joint training loss became non-finite")
        total.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            float(training["gradient_clip_norm"]),
        )
        optimizer.step()
        mass = batch.inputs.num_graphs
        contexts += mass
        totals["total"] += float(total.detach().cpu()) * mass
        for name in totals:
            if name == "total":
                continue
            value = parts[name]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Unexpected loss part: {name}")
            totals[name] += float(value.detach().cpu()) * mass
        for name in counts:
            counts[name] += int(parts[name])
    if not contexts:
        raise JointSiteNExperimentError("Joint training epoch had no contexts")
    return {
        **{name: value / contexts for name, value in totals.items()},
        **counts,
        "context_count": contexts,
        "unknown_direct_loss": 0.0,
        "ontology_out_of_scope_direct_loss": 0.0,
    }


def _selection_better(
    candidate: Mapping[str, object],
    incumbent: Mapping[str, object] | None,
    *,
    minimum_top1_delta: float,
) -> bool:
    if incumbent is None:
        return True
    candidate_top1 = float(candidate["exact_top1"])
    incumbent_top1 = float(incumbent["exact_top1"])
    if candidate_top1 > incumbent_top1 + minimum_top1_delta:
        return True
    if abs(candidate_top1 - incumbent_top1) <= minimum_top1_delta:
        return float(candidate["automatic_n_mae"]) < float(
            incumbent["automatic_n_mae"]
        )
    return False


def _save_checkpoint(
    path: Path,
    *,
    model: MayrJointSiteNModel,
    preprocessor: SiteNFoldPreprocessor,
    vocabulary: SolventVocabulary,
    payload: Mapping[str, object],
    schema_version: str = INNER_CHECKPOINT_SCHEMA,
) -> dict[str, object]:
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    complete = {
        **dict(payload),
        "schema_version": schema_version,
        "model_architecture": dict(model.architecture),
        "model_state_dict": state,
        "model_state_sha256": _tensor_mapping_sha256(state),
        "preprocessor": preprocessor.to_json(),
        "solvent_vocabulary": list(vocabulary.tokens),
    }
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(complete, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return complete


def run_inner(
    *,
    outer_fold: int,
    inner_fold: int,
    initialization_seed: int,
    variant: str = "joint_full",
    config_path: str | Path = DEFAULT_CONFIG,
    device: str | None = None,
    maximum_epochs: int | None = None,
    head_learning_rate: float | None = None,
    output_directory: str | Path | None = None,
) -> dict[str, object]:
    """Train one inner-fold/seed job without loading any outer-test targets."""

    started = time.perf_counter()
    config, resolved = read_config(config_path)
    verify_input_bindings(config, resolved)
    if outer_fold not in range(int(config["outer_fold_count"])) or inner_fold not in range(
        int(config["inner_fold_count"])
    ):
        raise JointSiteNExperimentError("Inner fold axis is outside the frozen split")
    if initialization_seed not in tuple(map(int, config["initialization_seeds"])):
        raise JointSiteNExperimentError("Initialization seed is not registered")
    settings = variant_settings(variant)
    selected_device = _device(device or str(config["device"]))
    train_ids, validation_ids, split_audit = inner_target_ids(
        config,
        outer_fold=outer_fold,
        inner_fold=inner_fold,
    )
    corpus = load_joint_site_n_corpus(
        _project_path(config["dataset"]["directory"], label="dataset"),
        evidence_path=_project_path(
            config["prototype_evidence"]["path"], label="prototype evidence"
        ),
        target_ids=sorted(train_ids | validation_ids),
    )
    train_context_ids = corpus.context_ids_for_target_ids(sorted(train_ids))
    validation_context_ids = corpus.context_ids_for_target_ids(sorted(validation_ids))
    if set(train_context_ids) & set(validation_context_ids):
        raise JointSiteNExperimentError("Inner context roles overlap")
    train_examples = corpus.examples(train_context_ids)
    validation_examples = corpus.examples(validation_context_ids)
    site_context_weights, site_type_balance_audit = (
        site_type_balanced_context_weights(
            corpus.queries,
            context_ids=train_context_ids,
            exponent=float(
                config["training"]["site_type_context_balance_exponent"]
            ),
        )
    )
    router_cell_weights, router_balance_audit = balanced_router_cell_weights(
        corpus.queries,
        context_ids=train_context_ids,
        use_historical_evidence=settings.use_historical_evidence,
    )

    site_config, _ = _site_config(config)
    teacher, preprocessor, vocabulary, teacher_payload, teacher_path = (
        load_inner_conditional_checkpoint(
            site_config,
            outer_fold=outer_fold,
            inner_fold=inner_fold,
            device=selected_device,
        )
    )
    ranker_checkpoint, ranker_path, ranker_binding_audit = (
        load_split_safe_site_ranker(
            config,
            outer_fold=outer_fold,
            inner_fold=inner_fold,
            conditional_teacher_path=teacher_path,
        )
    )
    base_scorer, base_scorer_transfer_audit = _joint_model_from_teacher(
        teacher,
        config=config,
        vocabulary=vocabulary,
        preprocessor=preprocessor,
        ranker_checkpoint=ranker_checkpoint,
        initialization_seed=initialization_seed,
        settings=settings,
        device=selected_device,
    )
    (
        frozen_base_logits,
        frozen_base_router_logits,
        frozen_base_audit,
    ) = freeze_publication_base_logits(
        base_scorer,
        [*train_examples, *validation_examples],
        corpus=corpus,
        preprocessor=preprocessor,
        vocabulary=vocabulary,
        batch_size=int(config["training"]["batch_size_contexts"]),
        device=selected_device,
        region_residual_path=ranker_path.parent / "region_residual.joblib",
        ranker_summary_path=ranker_path.parent / "summary.json",
        outer_fold=outer_fold,
        inner_fold=inner_fold,
    )
    reference_path = ranker_path.parent / "validation_predictions.parquet"
    (
        frozen_base_logits,
        frozen_base_router_logits,
        validation_base_transfer_audit,
    ) = transfer_inner_validation_base_logits(
        frozen_base_logits,
        frozen_base_router_logits,
        reference_path=reference_path,
        expected_query_ids=_batch_query_ids(validation_examples),
    )
    frozen_base_audit = {
        **frozen_base_audit,
        "mapping_sha256_before_validation_reference_transfer": frozen_base_audit[
            "mapping_sha256"
        ],
        "router_mapping_sha256_before_validation_reference_transfer": (
            frozen_base_audit["router_mapping_sha256"]
        ),
        "mapping_sha256": validation_base_transfer_audit[
            "combined_mapping_sha256"
        ],
        "router_mapping_sha256": validation_base_transfer_audit[
            "combined_router_mapping_sha256"
        ],
        "validation_reference_transfer_audit": validation_base_transfer_audit,
    }
    del base_scorer
    corpus = attach_frozen_base_logits(
        corpus,
        frozen_base_logits,
        frozen_base_router_logits,
    )
    model, transfer_audit = _joint_model_from_teacher(
        teacher,
        config=config,
        vocabulary=vocabulary,
        preprocessor=preprocessor,
        ranker_checkpoint=None,
        initialization_seed=initialization_seed,
        settings=settings,
        device=selected_device,
    )
    del teacher
    harm, harm_audit = compute_teacher_n_harm(
        model,
        train_examples,
        corpus=corpus,
        preprocessor=preprocessor,
        vocabulary=vocabulary,
        batch_size=int(config["training"]["batch_size_contexts"]),
        device=selected_device,
        cap_quantile=float(config["loss"]["n_harm_cap_quantile"]),
        use_historical_evidence=(
            settings.use_historical_evidence and settings.use_n_harm
        ),
    )
    configured_head_learning_rate = float(
        config["training"]["head_learning_rate"]
    )
    effective_head_learning_rate = (
        configured_head_learning_rate
        if head_learning_rate is None
        else float(head_learning_rate)
    )
    if (
        not math.isfinite(effective_head_learning_rate)
        or effective_head_learning_rate <= 0
    ):
        raise JointSiteNExperimentError(
            "Diagnostic head learning rate must be finite and positive"
        )
    groups = joint_optimizer_parameter_groups(
        model,
        head_learning_rate=effective_head_learning_rate,
        backbone_multiplier=float(
            config["training"]["backbone_learning_rate_multiplier"]
        ),
    )
    optimizer = torch.optim.AdamW(
        groups,
        weight_decay=float(config["training"]["weight_decay"]),
    )
    configured_maximum = int(config["training"]["maximum_epochs"])
    epochs = configured_maximum if maximum_epochs is None else int(maximum_epochs)
    if epochs < 1 or epochs > configured_maximum:
        raise JointSiteNExperimentError("Diagnostic epoch override is outside (0, max]")
    diagnostic_override = (
        maximum_epochs is not None or head_learning_rate is not None
    )
    default_target = (
        _project_path(config["output_directory"], label="output directory")
        / "prototype"
        / variant
        / "inner"
        / f"outer-{outer_fold}"
        / f"inner-{inner_fold}"
        / f"seed-{initialization_seed}"
    )
    if diagnostic_override:
        suffix = f"diagnostic-{epochs}ep"
        if head_learning_rate is not None:
            suffix += f"-lr-{effective_head_learning_rate:g}"
        default_target = default_target.with_name(
            f"seed-{initialization_seed}-{suffix}"
        )
    target = Path(output_directory).resolve() if output_directory else default_target
    if target.exists():
        raise JointSiteNExperimentError(f"Refusing to overwrite inner run: {target}")

    (
        initial_evaluation,
        initial_candidate_scores,
        initial_context_scores,
        initial_by_type,
    ) = evaluate_labeled(
        model,
        validation_examples,
        corpus=corpus,
        preprocessor=preprocessor,
        vocabulary=vocabulary,
        batch_size=int(config["training"]["batch_size_contexts"]),
        device=selected_device,
    )
    initialization_reproduction = audit_inner_ranker_initialization(
        initial_candidate_scores,
        reference_path=reference_path,
    )
    curves: list[dict[str, object]] = [
        {
            "epoch": 0,
            "phase": "transferred_initialization",
            **{f"validation_{key}": value for key, value in initial_evaluation["overall"].items()},
            **{
                f"validation_{key}": value
                for key, value in initial_evaluation["score_diagnostics"].items()
            },
            **_type_top1_curve_columns(initial_by_type),
        }
    ]
    best_metrics: Mapping[str, object] | None = dict(initial_evaluation["overall"])
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    stale = 0
    last_validation = initial_evaluation
    last_candidate_scores = initial_candidate_scores
    last_context_scores = initial_context_scores
    last_by_type = initial_by_type
    training_config = config["training"]
    warmup_epochs = int(training_config["heads_only_warmup_epochs"])
    for epoch in range(1, epochs + 1):
        heads_only = epoch <= warmup_epochs or not settings.train_backbone_after_warmup
        set_heads_only_warmup(model, enabled=heads_only)
        train_metrics = _train_epoch(
            model,
            train_examples,
            corpus=corpus,
            preprocessor=preprocessor,
            vocabulary=vocabulary,
            candidate_n_harm=harm,
            router_cell_weights=router_cell_weights,
            site_context_weights=site_context_weights,
            settings=settings,
            config=config,
            optimizer=optimizer,
            device=selected_device,
            shuffle_seed=initialization_seed + epoch,
        )
        (
            validation,
            validation_candidate_scores,
            validation_context_scores,
            validation_by_type,
        ) = evaluate_labeled(
            model,
            validation_examples,
            corpus=corpus,
            preprocessor=preprocessor,
            vocabulary=vocabulary,
            batch_size=int(training_config["batch_size_contexts"]),
            device=selected_device,
        )
        metrics = validation["overall"]
        last_validation = validation
        last_candidate_scores = validation_candidate_scores
        last_context_scores = validation_context_scores
        last_by_type = validation_by_type
        curves.append(
            {
                "epoch": epoch,
                "phase": "heads_only" if heads_only else "joint_finetune",
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"validation_{key}": value for key, value in metrics.items()},
                **{
                    f"validation_{key}": value
                    for key, value in validation["score_diagnostics"].items()
                },
                **_type_top1_curve_columns(validation_by_type),
            }
        )
        if _selection_better(
            metrics,
            best_metrics,
            minimum_top1_delta=float(training_config["minimum_validation_top1_delta"]),
        ):
            best_metrics = dict(metrics)
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if (
            not diagnostic_override
            and epoch >= int(training_config["minimum_epochs"])
            and stale >= int(training_config["early_stopping_patience"])
        ):
            break
    if best_state is None or best_metrics is None:
        raise JointSiteNExperimentError("Inner training produced no selectable epoch")
    model.load_state_dict(best_state, strict=True)
    final_evaluation, candidate_scores, context_scores, by_type = evaluate_labeled(
        model,
        validation_examples,
        corpus=corpus,
        preprocessor=preprocessor,
        vocabulary=vocabulary,
        batch_size=int(training_config["batch_size_contexts"]),
        device=selected_device,
    )
    if final_evaluation["overall"] != dict(best_metrics):
        raise JointSiteNExperimentError("Restored best checkpoint metrics changed")
    region_residual_binding = _region_residual_source_binding(
        region_residual_path=ranker_path.parent / "region_residual.joblib",
        frozen_base_audit=frozen_base_audit,
    )
    source_hashes = {
        "config": sha256_file(resolved),
        "runner": sha256_file(Path(__file__).resolve()),
        "model": sha256_file(
            ROOT / "src/nucpred/training/mayr_joint_site_n.py"
        ),
        "data_adapter": sha256_file(
            ROOT / "src/nucpred/training/mayr_joint_site_n_data.py"
        ),
        "type_router": sha256_file(
            ROOT / "src/nucpred/training/mayr_joint_site_type_router.py"
        ),
        "publication_structured_ranker": sha256_file(
            ROOT / "src/nucpred/training/mayr_site_structured_ranker.py"
        ),
        "publication_ranker_type_contract": sha256_file(
            ROOT / "src/nucpred/training/mayr_site_ranker.py"
        ),
        "dataset_manifest": str(config["dataset"]["manifest_sha256"]),
        "outer_membership": str(config["dataset"]["outer_membership_sha256"]),
        "nested_membership": str(config["dataset"]["nested_membership_sha256"]),
        "prototype_evidence": str(config["prototype_evidence"]["sha256"]),
        "conditional_teacher_checkpoint": sha256_file(teacher_path),
        "site_ranker_checkpoint": sha256_file(ranker_path),
        "site_ranker_summary": sha256_file(ranker_path.parent / "summary.json"),
        "site_region_residual": region_residual_binding["sha256"],
        "site_ranker_reference_predictions": initialization_reproduction[
            "reference_sha256"
        ],
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        checkpoint_payload = {
            "phase": "inner_development_selection",
            "experiment_id": config["experiment_id"],
            "variant": variant,
            "outer_fold": outer_fold,
            "inner_fold": inner_fold,
            "initialization_seed": initialization_seed,
            "best_epoch": best_epoch,
            "best_validation_metrics": dict(best_metrics),
            "variant_settings": asdict(settings),
            "split_audit": split_audit,
            "source_hashes": source_hashes,
            "teacher_model_state_sha256": teacher_payload["model_state_sha256"],
            "transfer_audit": transfer_audit,
            "base_scorer_transfer_audit": base_scorer_transfer_audit,
            "frozen_base_audit": frozen_base_audit,
            "ranker_binding_audit": ranker_binding_audit,
            "ranker_initialization_reproduction": initialization_reproduction,
            "site_region_residual_binding": region_residual_binding,
            "teacher_harm_audit": harm_audit,
            "site_type_balance_audit": site_type_balance_audit,
            "router_balance_audit": router_balance_audit,
            "diagnostic_epoch_override": diagnostic_override,
            "configured_head_learning_rate": configured_head_learning_rate,
            "effective_head_learning_rate": effective_head_learning_rate,
            "diagnostic_head_learning_rate_override": (
                head_learning_rate is not None
            ),
            "eligible_for_formal_inner_selection": not diagnostic_override,
            "outer_test_target_rows_loaded": 0,
            "outer_test_predictions_computed": 0,
        }
        saved = _save_checkpoint(
            staging / "selection_checkpoint.pt",
            model=model,
            preprocessor=preprocessor,
            vocabulary=vocabulary,
            payload=checkpoint_payload,
        )
        pd.DataFrame(curves).to_csv(staging / "training_curves.csv", index=False)
        candidate_scores.to_parquet(
            staging / "validation_candidate_scores.parquet", index=False
        )
        context_scores.to_parquet(
            staging / "validation_context_scores.parquet", index=False
        )
        by_type.to_csv(staging / "validation_site_type_metrics.csv", index=False)
        if diagnostic_override:
            last_candidate_scores.to_parquet(
                staging / "last_epoch_validation_candidate_scores.parquet",
                index=False,
            )
            last_context_scores.to_parquet(
                staging / "last_epoch_validation_context_scores.parquet",
                index=False,
            )
            last_by_type.to_csv(
                staging / "last_epoch_validation_site_type_metrics.csv",
                index=False,
            )
        summary: dict[str, object] = {
            "schema_version": INNER_SUMMARY_SCHEMA,
            "status": "pass",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "experiment_id": config["experiment_id"],
            "variant": variant,
            "variant_settings": asdict(settings),
            "outer_fold": outer_fold,
            "inner_fold": inner_fold,
            "initialization_seed": initialization_seed,
            "device": str(selected_device),
            "best_epoch": best_epoch,
            "epochs_completed": int(curves[-1]["epoch"]),
            "best_validation": final_evaluation,
            "initial_validation": initial_evaluation,
            "last_validation": last_validation,
            "split_audit": split_audit,
            "corpus_audit": dict(corpus.audit),
            "transfer_audit": transfer_audit,
            "base_scorer_transfer_audit": base_scorer_transfer_audit,
            "frozen_base_audit": frozen_base_audit,
            "ranker_binding_audit": ranker_binding_audit,
            "ranker_initialization_reproduction": initialization_reproduction,
            "site_region_residual_binding": region_residual_binding,
            "teacher_harm_audit": harm_audit,
            "site_type_balance_audit": site_type_balance_audit,
            "router_balance_audit": router_balance_audit,
            "source_hashes": source_hashes,
            "model_state_sha256": saved["model_state_sha256"],
            "diagnostic_epoch_override": diagnostic_override,
            "configured_head_learning_rate": configured_head_learning_rate,
            "effective_head_learning_rate": effective_head_learning_rate,
            "diagnostic_head_learning_rate_override": (
                head_learning_rate is not None
            ),
            "eligible_for_formal_inner_selection": not diagnostic_override,
            "unknown_direct_loss": 0.0,
            "ontology_out_of_scope_direct_loss": 0.0,
            "outer_test_target_rows_loaded": 0,
            "outer_test_predictions_computed": 0,
            "elapsed_seconds": time.perf_counter() - started,
        }
        atomic_write_json(staging / "summary.json", summary, ensure_ascii=False)
        output_bindings = {
            path.name: {
                "sha256": sha256_file(path),
                "bytes": int(path.stat().st_size),
            }
            for path in sorted(staging.iterdir())
        }
        atomic_write_json(
            staging / "manifest.json",
            {
                "schema_version": "nucpred.mayr-joint-site-n-inner-manifest.v1",
                "status": "frozen",
                "files": output_bindings,
            },
            ensure_ascii=False,
        )
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        gc.collect()
        if selected_device.type == "cuda":
            torch.cuda.empty_cache()
    summary["output_directory"] = _display_path(target)
    summary["manifest_sha256"] = sha256_file(target / "manifest.json")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--outer-fold", type=int, required=True)
    parser.add_argument("--inner-fold", type=int, required=True)
    parser.add_argument("--initialization-seed", type=int, required=True)
    parser.add_argument("--variant", choices=TRAINABLE_VARIANTS, default="joint_full")
    parser.add_argument("--device")
    parser.add_argument("--maximum-epochs", type=int)
    parser.add_argument("--head-learning-rate", type=float)
    parser.add_argument("--output-directory")
    args = parser.parse_args(argv)
    result = run_inner(
        outer_fold=args.outer_fold,
        inner_fold=args.inner_fold,
        initialization_seed=args.initialization_seed,
        variant=args.variant,
        config_path=args.config,
        device=args.device,
        maximum_epochs=args.maximum_epochs,
        head_learning_rate=args.head_learning_rate,
        output_directory=args.output_directory,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
