"""Strict prototype data adapter for the joint Mayr site-N model.

The complete label-independent deployment candidate set is retained as model
input.  Direct supervision is emitted only for corrected-v2 exact endpoints
and evidence-bound historical endpoint exclusions.  Historical exclusions are
prototype-only and carry no authority for a later v3 calibration or test set.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from nucpred.datasets.mayr_site_candidate_policy import (
    select_deployment_candidates,
)
from nucpred.training.mayr_joint_site_n import (
    JointEvidenceState,
    JointSiteNTrainingBatch,
)
from nucpred.training.mayr_node_xtb_scratch import SolventVocabulary
from nucpred.training.mayr_site_n import (
    SITE_TYPE_NAMES,
    SiteNExample,
    SiteNFoldPreprocessor,
    pack_site_n_batch,
)
from nucpred.training.mayr_site_queries import site_n_examples_from_queries


PROTOTYPE_EVIDENCE_ROLE = "prototype_train_and_diagnostic_only"


class JointSiteNDataError(RuntimeError):
    """Raised when evidence or candidate identities are not safely alignable."""


@dataclass(frozen=True)
class JointSiteNCorpus:
    """In-memory label-independent queries plus strictly masked supervision."""

    contexts: pd.DataFrame
    targets: pd.DataFrame
    queries: pd.DataFrame
    evidence: pd.DataFrame
    audit: Mapping[str, object]

    def context_ids_for_target_ids(self, target_ids: Sequence[str]) -> tuple[str, ...]:
        selected = set(map(str, target_ids))
        if not selected:
            raise ValueError("Target IDs cannot be empty")
        rows = self.targets.loc[self.targets["target_id"].astype(str).isin(selected)]
        if len(rows) != len(selected):
            raise JointSiteNDataError("Target membership references unavailable targets")
        return tuple(sorted(set(rows["context_id"].astype(str))))

    def examples(self, context_ids: Sequence[str]) -> list[SiteNExample]:
        selected = set(map(str, context_ids))
        if not selected:
            raise ValueError("Context IDs cannot be empty")
        frame = self.queries.loc[
            self.queries["context_id"].astype(str).isin(selected)
        ]
        if frame["context_id"].astype(str).nunique() != len(selected):
            raise JointSiteNDataError("Requested context is unavailable")
        return site_n_examples_from_queries(frame, contexts=self.contexts)


@dataclass(frozen=True)
class JointCandidateUniverse:
    """Label-blind context-candidate queries used for outer score freeze."""

    contexts: pd.DataFrame
    queries: pd.DataFrame
    audit: Mapping[str, object]

    def examples(self) -> list[SiteNExample]:
        return site_n_examples_from_queries(self.queries, contexts=self.contexts)


def site_type_balanced_context_weights(
    query_table: pd.DataFrame,
    *,
    context_ids: Sequence[str],
    exponent: float,
) -> tuple[dict[str, float], dict[str, object]]:
    """Fit train-only inverse-frequency endpoint-type context weights."""

    if not 0.0 <= exponent <= 1.0:
        raise ValueError("Site-type balance exponent must be in [0, 1]")
    selected_ids = set(map(str, context_ids))
    if not selected_ids:
        raise ValueError("Context IDs cannot be empty")
    selected = query_table.loc[
        query_table["context_id"].astype(str).isin(selected_ids)
        & query_table["evidence_state"].eq(int(JointEvidenceState.POSITIVE_EXACT)),
        ["context_id", "site_type"],
    ].drop_duplicates()
    observed_ids = set(selected["context_id"].astype(str))
    if observed_ids != selected_ids:
        raise JointSiteNDataError(
            "Every weighted training context must have an exact endpoint"
        )
    unknown_types = sorted(set(selected["site_type"].astype(str)) - set(SITE_TYPE_NAMES))
    if unknown_types:
        raise JointSiteNDataError(
            f"Cannot balance unknown endpoint types: {unknown_types}"
        )
    counts = {
        str(site_type): int(group["context_id"].astype(str).nunique())
        for site_type, group in selected.groupby("site_type", sort=True)
    }
    raw_by_type = {
        site_type: float(count) ** (-float(exponent))
        for site_type, count in counts.items()
    }
    raw_by_context = {
        str(context_id): float(
            np.mean([raw_by_type[str(site_type)] for site_type in group["site_type"]])
        )
        for context_id, group in selected.groupby("context_id", sort=True)
    }
    normalization = len(raw_by_context) / float(sum(raw_by_context.values()))
    weights = {
        context_id: raw * normalization
        for context_id, raw in raw_by_context.items()
    }
    values = np.asarray(list(weights.values()), dtype=float)
    if (
        not np.isfinite(values).all()
        or (values <= 0).any()
        or not np.isclose(values.mean(), 1.0, rtol=0.0, atol=1e-10)
    ):
        raise JointSiteNDataError("Site-type context balancing is invalid")
    return weights, {
        "schema_version": "nucpred.mayr-joint-site-n-type-balance.v1",
        "status": "pass",
        "fit_role": "training_contexts_only",
        "exponent": float(exponent),
        "context_count": len(weights),
        "endpoint_context_counts_by_type": counts,
        "normalized_context_weight_by_type": {
            site_type: raw * normalization
            for site_type, raw in raw_by_type.items()
        },
        "mean_context_weight": float(values.mean()),
        "minimum_context_weight": float(values.min()),
        "maximum_context_weight": float(values.max()),
    }


def balanced_router_cell_weights(
    query_table: pd.DataFrame,
    *,
    context_ids: Sequence[str],
    use_historical_evidence: bool = True,
) -> tuple[dict[tuple[str, str], float], dict[str, object]]:
    """Give equal router mass to type/label cells, then connectivities."""

    selected_ids = set(map(str, context_ids))
    if not selected_ids:
        raise ValueError("Context IDs cannot be empty")
    eligible_state = (
        query_table["evidence_state"].ge(
            int(JointEvidenceState.ENDPOINT_EXCLUDED)
        )
        if use_historical_evidence
        else query_table["evidence_state"].eq(
            int(JointEvidenceState.POSITIVE_EXACT)
        )
    )
    eligible = query_table.loc[
        query_table["context_id"].astype(str).isin(selected_ids)
        & eligible_state,
        ["context_id", "connectivity_id", "site_type", "evidence_state"],
    ].copy()
    cells = (
        eligible.assign(
            router_target=eligible["evidence_state"].eq(
                int(JointEvidenceState.POSITIVE_EXACT)
            )
        )
        .groupby(["context_id", "connectivity_id", "site_type"], as_index=False)
        ["router_target"]
        .max()
    )
    if set(cells["context_id"].astype(str)) != selected_ids:
        raise JointSiteNDataError("Every router training context needs reviewed cells")
    observed_types = set(cells["site_type"].astype(str))
    if not observed_types <= set(SITE_TYPE_NAMES):
        raise JointSiteNDataError("Router cells contain an unknown site type")
    observed_type_labels = sorted(
        set(
            zip(
                cells["site_type"].astype(str),
                cells["router_target"].astype(int),
                strict=True,
            )
        )
    )
    values: dict[tuple[str, str], float] = {}
    connectivity_counts: dict[str, int] = {}
    context_counts: dict[str, int] = {}
    for site_type, target in observed_type_labels:
        selected = cells.loc[
            cells["site_type"].astype(str).eq(site_type)
            & cells["router_target"].astype(int).eq(target)
        ]
        connectivity_ids = tuple(sorted(set(selected["connectivity_id"].astype(str))))
        cell_name = f"{site_type}|{target}"
        connectivity_counts[cell_name] = len(connectivity_ids)
        context_counts[cell_name] = len(selected)
        for connectivity_id in connectivity_ids:
            same_connectivity = selected.loc[
                selected["connectivity_id"].astype(str).eq(connectivity_id)
            ]
            mass = (
                1.0
                / len(observed_type_labels)
                / len(connectivity_ids)
                / len(same_connectivity)
            )
            for row in same_connectivity.itertuples(index=False):
                identity = (str(row.context_id), str(row.site_type))
                if identity in values:
                    raise JointSiteNDataError("Router cell identity is duplicated")
                values[identity] = mass
    array = np.asarray(list(values.values()), dtype=float)
    if (
        len(values) != len(cells)
        or not np.isfinite(array).all()
        or (array <= 0).any()
        or not np.isclose(array.sum(), 1.0, rtol=0.0, atol=1e-10)
    ):
        raise JointSiteNDataError("Balanced router cell weights are invalid")
    return values, {
        "schema_version": "nucpred.mayr-joint-site-n-router-balance.v1",
        "status": "pass",
        "fit_role": "training_contexts_only",
        "historical_evidence_enabled": bool(use_historical_evidence),
        "balance_axes": (
            "site_type_x_binary_label_then_connectivity"
            if use_historical_evidence
            else "site_type_x_positive_label_then_connectivity"
        ),
        "observed_type_label_cell_count": len(observed_type_labels),
        "router_context_type_cell_count": len(values),
        "context_counts_by_type_label": context_counts,
        "connectivity_counts_by_type_label": connectivity_counts,
        "total_weight": float(array.sum()),
    }


def _required_columns(
    frame: pd.DataFrame, required: set[str], *, label: str
) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise JointSiteNDataError(f"{label} lacks columns: {missing}")


def _candidate_queries(
    contexts: pd.DataFrame,
    deployment: pd.DataFrame,
) -> pd.DataFrame:
    query_columns = [
        "candidate_site_id",
        "species_id",
        "site_type",
        "member_atom_indices_json",
        "member_bond_pairs_json",
        "member_atomic_numbers_json",
        "candidate_origins_json",
        "label_independent",
    ]
    queries = contexts[
        ["context_id", "species_id", "connectivity_id", "solvent_raw"]
    ].merge(
        deployment[query_columns],
        on="species_id",
        how="left",
        validate="many_to_many",
    )
    if queries["candidate_site_id"].isna().any():
        raise JointSiteNDataError("A requested context has no deployment candidates")
    if not queries["label_independent"].astype(bool).all():
        raise JointSiteNDataError("Candidate generation exposed labels")
    queries["query_id"] = (
        queries["context_id"].astype(str)
        + "|"
        + queries["candidate_site_id"].astype(str)
    )
    if queries["query_id"].duplicated().any():
        raise JointSiteNDataError("Candidate query IDs are duplicated")
    type_order = {name: index for index, name in enumerate(SITE_TYPE_NAMES)}
    queries["_site_type_order"] = queries["site_type"].map(type_order)
    if queries["_site_type_order"].isna().any():
        raise JointSiteNDataError("Candidate ontology contains an unknown type")
    return (
        queries.sort_values(
            ["context_id", "_site_type_order", "candidate_site_id"],
            kind="stable",
        )
        .drop(columns="_site_type_order")
        .reset_index(drop=True)
    )


def load_joint_candidate_universe(
    dataset_directory: str | Path,
    *,
    context_ids: Sequence[str],
) -> JointCandidateUniverse:
    """Load candidate queries without reading targets, evidence, or split roles."""

    selected_ids = set(map(str, context_ids))
    if not selected_ids:
        raise ValueError("Context IDs cannot be empty")
    root = Path(dataset_directory)
    contexts = pd.read_parquet(root / "contexts.parquet")
    contexts = contexts.loc[contexts["context_id"].astype(str).isin(selected_ids)]
    if contexts["context_id"].astype(str).nunique() != len(selected_ids):
        raise JointSiteNDataError("Requested score-freeze context is unavailable")
    species = pd.read_parquet(root / "species.parquet")
    candidates = pd.read_parquet(root / "candidate_sites.parquet")
    deployment, policy_audit = select_deployment_candidates(candidates, species)
    queries = _candidate_queries(contexts, deployment)
    queries["N_value"] = np.nan
    forbidden = {
        "target_id",
        "site_object_id",
        "N_mean",
        "evidence_state",
        "validity_label",
        "split_role",
    }
    exposed = sorted(forbidden & set(queries.columns))
    if exposed:
        raise JointSiteNDataError(f"Label-blind query universe exposes {exposed}")
    return JointCandidateUniverse(
        contexts=contexts,
        queries=queries,
        audit={
            "schema_version": "nucpred.mayr-joint-site-n-unlabeled-universe-audit.v1",
            "status": "pass",
            "context_count": int(queries["context_id"].nunique()),
            "candidate_query_count": int(len(queries)),
            "target_rows_loaded": 0,
            "evidence_rows_loaded": 0,
            "split_roles_exposed_to_model": False,
            "candidate_policy_audit": policy_audit,
        },
    )


def load_joint_site_n_corpus(
    dataset_directory: str | Path,
    *,
    evidence_path: str | Path,
    target_ids: Sequence[str] | None = None,
) -> JointSiteNCorpus:
    """Load corrected v2 candidates and map historical evidence conservatively."""

    root = Path(dataset_directory)
    contexts = pd.read_parquet(root / "contexts.parquet")
    selected_target_ids = None if target_ids is None else set(map(str, target_ids))
    if selected_target_ids is not None and not selected_target_ids:
        raise ValueError("Target IDs cannot be empty")
    targets = pd.read_parquet(
        root / "targets.parquet",
        filters=(
            None
            if selected_target_ids is None
            else [("target_id", "in", sorted(selected_target_ids))]
        ),
    )
    if selected_target_ids is not None:
        targets = targets.loc[
            targets["target_id"].astype(str).isin(selected_target_ids)
        ]
        if len(targets) != len(selected_target_ids):
            raise JointSiteNDataError("Requested target IDs are unavailable")
    species = pd.read_parquet(root / "species.parquet")
    candidates = pd.read_parquet(root / "candidate_sites.parquet")
    selected_context_ids = sorted(set(targets["context_id"].astype(str)))
    evidence = pd.read_parquet(
        evidence_path,
        filters=(
            None
            if selected_target_ids is None
            else [("context_id", "in", selected_context_ids)]
        ),
    )
    if selected_target_ids is not None:
        evidence = evidence.loc[
            evidence["context_id"].astype(str).isin(selected_context_ids)
        ]
    _required_columns(
        targets,
        {"target_id", "context_id", "site_object_id", "site_type", "N_mean"},
        label="v2 targets",
    )
    _required_columns(
        evidence,
        {
            "target_id",
            "context_id",
            "candidate_site_id",
            "site_type",
            "validity_label",
            "evidence_strength_weight",
            "candidate_sampling_probability",
            "candidate_inverse_probability_weight",
            "label_source",
            "endpoint_relative",
            "unknown_is_negative",
        },
        label="historical evidence",
    )
    if targets["target_id"].astype(str).duplicated().any():
        raise JointSiteNDataError("v2 target IDs are duplicated")
    if evidence.duplicated(["context_id", "candidate_site_id"]).any():
        raise JointSiteNDataError("Historical evidence duplicates a context candidate")
    labels = set(evidence["validity_label"].astype(int))
    if not labels <= {0, 1}:
        raise JointSiteNDataError("Historical evidence is not strictly binary")
    if not evidence["endpoint_relative"].astype(bool).all():
        raise JointSiteNDataError("Historical evidence lost endpoint-relative semantics")
    if evidence["unknown_is_negative"].astype(bool).any():
        raise JointSiteNDataError("Historical evidence converts unknowns to negatives")
    sampling_probability = evidence["candidate_sampling_probability"].to_numpy(
        dtype=float
    )
    inverse_probability = evidence[
        "candidate_inverse_probability_weight"
    ].to_numpy(dtype=float)
    if (
        not np.isfinite(sampling_probability).all()
        or not np.isfinite(inverse_probability).all()
        or (sampling_probability <= 0).any()
        or (sampling_probability > 1).any()
        or (inverse_probability < 1).any()
        or not np.allclose(
            sampling_probability * inverse_probability,
            np.ones(len(evidence), dtype=float),
            rtol=1e-7,
            atol=1e-7,
        )
    ):
        raise JointSiteNDataError("Historical evidence sampling/IPW contract changed")

    deployment, policy_audit = select_deployment_candidates(candidates, species)
    target_context_ids = set(selected_context_ids)
    selected_contexts = contexts.loc[
        contexts["context_id"].astype(str).isin(target_context_ids)
    ].copy()
    if selected_contexts["context_id"].astype(str).nunique() != len(target_context_ids):
        raise JointSiteNDataError("v2 targets reference unavailable contexts")
    queries = _candidate_queries(selected_contexts, deployment)

    query_identity = set(
        zip(
            queries["context_id"].astype(str),
            queries["candidate_site_id"].astype(str),
            strict=True,
        )
    )
    target_identity = set(
        zip(
            targets["context_id"].astype(str),
            targets["site_object_id"].astype(str),
            strict=True,
        )
    )
    missing_targets = sorted(target_identity - query_identity)
    if missing_targets:
        raise JointSiteNDataError(
            f"Deployment candidates miss corrected v2 targets: {missing_targets[:3]}"
        )

    historical_positive = evidence.loc[evidence["validity_label"].astype(int).eq(1)]
    historical_positive_identity = set(
        zip(
            historical_positive["context_id"].astype(str),
            historical_positive["candidate_site_id"].astype(str),
            strict=True,
        )
    )
    if not historical_positive_identity <= target_identity:
        raise JointSiteNDataError(
            "Historical positive endpoint no longer matches corrected v2"
        )
    negative = evidence.loc[evidence["validity_label"].astype(int).eq(0)].copy()
    negative_identity = set(
        zip(
            negative["context_id"].astype(str),
            negative["candidate_site_id"].astype(str),
            strict=True,
        )
    )
    if negative_identity & target_identity:
        raise JointSiteNDataError("Historical exclusion collides with a v2 endpoint")
    if not negative_identity <= query_identity:
        raise JointSiteNDataError("Historical exclusion is outside deployment candidates")

    target_values = {
        (str(row.context_id), str(row.site_object_id)): float(row.N_mean)
        for row in targets.itertuples(index=False)
    }
    negative_rows = {
        (str(row.context_id), str(row.candidate_site_id)): row
        for row in negative.itertuples(index=False)
    }
    states: list[int] = []
    n_values: list[float] = []
    evidence_weights: list[float] = []
    sampling_probabilities: list[float] = []
    inverse_probability_weights: list[float] = []
    label_sources: list[str] = []
    for row in queries.itertuples(index=False):
        identity = (str(row.context_id), str(row.candidate_site_id))
        if identity in target_values:
            states.append(int(JointEvidenceState.POSITIVE_EXACT))
            n_values.append(target_values[identity])
            evidence_weights.append(1.0)
            sampling_probabilities.append(1.0)
            inverse_probability_weights.append(1.0)
            label_sources.append("corrected_v2_exact_endpoint")
        elif identity in negative_rows:
            source = negative_rows[identity]
            states.append(int(JointEvidenceState.ENDPOINT_EXCLUDED))
            n_values.append(np.nan)
            evidence_weights.append(float(source.evidence_strength_weight))
            sampling_probabilities.append(float(source.candidate_sampling_probability))
            inverse_probability_weights.append(
                float(source.candidate_inverse_probability_weight)
            )
            label_sources.append(str(source.label_source))
        else:
            states.append(int(JointEvidenceState.UNKNOWN))
            n_values.append(np.nan)
            evidence_weights.append(0.0)
            sampling_probabilities.append(np.nan)
            inverse_probability_weights.append(0.0)
            label_sources.append("unreviewed_unknown")
    queries["evidence_state"] = np.asarray(states, dtype=np.int8)
    queries["N_value"] = np.asarray(n_values, dtype=float)
    queries["evidence_weight"] = np.asarray(evidence_weights, dtype=float)
    queries["candidate_sampling_probability"] = np.asarray(
        sampling_probabilities, dtype=float
    )
    queries["candidate_inverse_probability_weight"] = np.asarray(
        inverse_probability_weights, dtype=float
    )
    queries["label_source"] = label_sources
    queries = queries.reset_index(drop=True)
    state_counts = queries["evidence_state"].value_counts().to_dict()
    audit: dict[str, object] = {
        "schema_version": "nucpred.mayr-joint-site-n-prototype-corpus-audit.v1",
        "status": "pass",
        "evidence_role": PROTOTYPE_EVIDENCE_ROLE,
        "historical_evidence_may_enter_v3_formal_calibration_or_test": False,
        "context_count": int(queries["context_id"].nunique()),
        "candidate_query_count": int(len(queries)),
        "positive_exact_count": int(
            state_counts.get(int(JointEvidenceState.POSITIVE_EXACT), 0)
        ),
        "endpoint_excluded_count": int(
            state_counts.get(int(JointEvidenceState.ENDPOINT_EXCLUDED), 0)
        ),
        "unknown_count": int(state_counts.get(int(JointEvidenceState.UNKNOWN), 0)),
        "ontology_out_of_scope_count": 0,
        "candidate_recall": 1.0,
        "unknown_direct_loss_weight": 0.0,
        "ontology_out_of_scope_direct_loss_weight": 0.0,
        "historical_positive_alignment_count": int(len(historical_positive)),
        "candidate_population_weighting": (
            "evidence_strength_times_inverse_inclusion_probability"
        ),
        "maximum_candidate_inverse_probability_weight": float(
            inverse_probability.max()
        ),
        "target_rows_loaded": int(len(targets)),
        "target_filter_applied": selected_target_ids is not None,
        "negative_counts_by_type": {
            str(name): int(value)
            for name, value in negative.groupby("site_type").size().items()
        },
        "candidate_policy_audit": policy_audit,
    }
    return JointSiteNCorpus(
        contexts=contexts,
        targets=targets,
        queries=queries,
        evidence=evidence,
        audit=audit,
    )


def pack_joint_site_n_batch(
    examples: Sequence[SiteNExample],
    *,
    query_table: pd.DataFrame,
    preprocessor: SiteNFoldPreprocessor,
    solvent_vocabulary: SolventVocabulary,
    candidate_n_harm: Mapping[str, float] | None = None,
    base_canonical_logits: Mapping[str, float] | None = None,
    base_router_selected_logits: Mapping[str, float] | None = None,
    router_cell_weights: Mapping[tuple[str, str], float] | None = None,
    site_context_weights: Mapping[str, float] | None = None,
    use_historical_evidence: bool = True,
) -> JointSiteNTrainingBatch:
    """Pack candidate inputs while preserving zero loss for unknown states."""

    packed = pack_site_n_batch(
        examples,
        preprocessor=preprocessor,
        solvent_vocabulary=solvent_vocabulary,
    )
    index = query_table.set_index("query_id", drop=False)
    if index.index.duplicated().any():
        raise JointSiteNDataError("Query supervision table has duplicate IDs")
    try:
        aligned = index.loc[list(packed.target_ids)]
    except KeyError as exc:
        raise JointSiteNDataError("Packed query lacks supervision metadata") from exc
    states = aligned["evidence_state"].to_numpy(dtype=np.int8)
    if not use_historical_evidence:
        states = np.where(
            states == int(JointEvidenceState.ENDPOINT_EXCLUDED),
            int(JointEvidenceState.UNKNOWN),
            states,
        ).astype(np.int8)
    state_tensor = torch.tensor(states, dtype=torch.int8)
    positive = state_tensor.eq(int(JointEvidenceState.POSITIVE_EXACT))
    retrieval = state_tensor.ge(int(JointEvidenceState.ENDPOINT_EXCLUDED))
    n_raw = aligned["N_value"].to_numpy(dtype=float)
    n_standardized = (n_raw - float(preprocessor.target_mean)) / float(
        preprocessor.target_scale
    )
    evidence_weight = aligned["evidence_weight"].to_numpy(dtype=float)
    evidence_weight[states < int(JointEvidenceState.ENDPOINT_EXCLUDED)] = 0.0
    inverse_probability_weight = aligned[
        "candidate_inverse_probability_weight"
    ].to_numpy(dtype=float)
    population_weight = evidence_weight * inverse_probability_weight
    population_weight[states < int(JointEvidenceState.ENDPOINT_EXCLUDED)] = 0.0
    eligible_population = population_weight[
        states >= int(JointEvidenceState.ENDPOINT_EXCLUDED)
    ]
    if (
        not np.isfinite(eligible_population).all()
        or (eligible_population <= 0).any()
    ):
        raise JointSiteNDataError(
            "Eligible candidate population weights must be finite and positive"
        )
    harm_source = candidate_n_harm or {}
    harm = np.asarray(
        [float(harm_source.get(str(query_id), 0.0)) for query_id in packed.target_ids],
        dtype=np.float32,
    )
    harm[states != int(JointEvidenceState.ENDPOINT_EXCLUDED)] = 0.0
    base_source = {} if base_canonical_logits is None else base_canonical_logits
    base = np.asarray(
        [float(base_source.get(str(query_id), 0.0)) for query_id in packed.target_ids],
        dtype=np.float32,
    )
    if not np.isfinite(base).all():
        raise JointSiteNDataError("Frozen base logits are not finite")
    router_source = (
        {} if base_router_selected_logits is None else base_router_selected_logits
    )
    base_router = np.asarray(
        [float(router_source.get(str(query_id), 0.0)) for query_id in packed.target_ids],
        dtype=np.float32,
    )
    if not np.isfinite(base_router).all():
        raise JointSiteNDataError("Frozen base router logits are not finite")
    router_weight = np.zeros(
        (packed.inputs.num_graphs, len(SITE_TYPE_NAMES)), dtype=np.float32
    )
    router_source = router_cell_weights or {}
    for graph, example in enumerate(examples):
        graph_rows = packed.inputs.site_graph_index.eq(graph)
        eligible_types = set(
            packed.inputs.site_type_index[graph_rows & retrieval].tolist()
        )
        for type_index in eligible_types:
            site_type = SITE_TYPE_NAMES[int(type_index)]
            value = float(router_source.get((example.context_id, site_type), 1.0))
            if not np.isfinite(value) or value <= 0:
                raise JointSiteNDataError(
                    "Eligible router cell weights must be finite and positive"
                )
            router_weight[graph, int(type_index)] = value
        unexpected = [
            site_type
            for context_id, site_type in router_source
            if context_id == example.context_id
            and SITE_TYPE_NAMES.index(site_type) not in eligible_types
        ]
        if unexpected:
            raise JointSiteNDataError("Router weights expose an unreviewed type cell")
    context_weight_source = site_context_weights or {}
    site_context_weight = np.asarray(
        [float(context_weight_source.get(example.context_id, 1.0)) for example in examples],
        dtype=np.float32,
    )
    if not np.isfinite(site_context_weight).all() or (site_context_weight <= 0).any():
        raise JointSiteNDataError("Site context weights must be finite and positive")
    return JointSiteNTrainingBatch(
        inputs=packed.inputs,
        retrieval_mask=retrieval,
        retrieval_positive_mask=positive,
        evidence_state=state_tensor,
        evidence_weight=torch.tensor(evidence_weight, dtype=torch.float32),
        candidate_population_weight=torch.tensor(
            population_weight, dtype=torch.float32
        ),
        n_target_standardized=torch.tensor(n_standardized, dtype=torch.float32),
        n_supervision_mask=positive & torch.tensor(np.isfinite(n_raw)),
        candidate_n_harm=torch.tensor(harm, dtype=torch.float32),
        base_canonical_logits=torch.tensor(base, dtype=torch.float32),
        base_router_selected_logits=torch.tensor(base_router, dtype=torch.float32),
        router_cell_weight=torch.tensor(router_weight, dtype=torch.float32),
        site_context_weight=torch.tensor(site_context_weight, dtype=torch.float32),
        context_weight=torch.ones(packed.inputs.num_graphs, dtype=torch.float32),
    )
