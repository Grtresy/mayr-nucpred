"""Build model-ready Mayr examples from target-independent site queries."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from nucpred.features.all_atom_graph import (
    EDGE_CATEGORY_SIZES,
    NODE_CATEGORY_SIZES,
)
from nucpred.training.mayr_node_xtb_scratch import (
    GLOBAL_FEATURES,
    LOCAL_FEATURES,
    SOLVENT_FEATURES,
)
from nucpred.training.mayr_site_n import (
    SITE_TYPE_NAMES,
    SiteNExample,
    _parse_int_list,
    _parse_matrix,
    _parse_vector,
)


def site_n_examples_from_queries(
    queries: pd.DataFrame,
    *,
    contexts: pd.DataFrame,
) -> list[SiteNExample]:
    """Build examples without reading target values or site labels.

    Required query columns are ``query_id``, ``context_id``,
    ``candidate_site_id``, ``site_type`` and ``member_atom_indices_json``.
    ``N_value`` is optional and becomes NaN for inference queries.
    """

    required = {
        "query_id",
        "context_id",
        "candidate_site_id",
        "site_type",
        "member_atom_indices_json",
    }
    missing = sorted(required - set(queries.columns))
    if missing:
        raise ValueError(f"Candidate queries lack columns: {missing}")
    if queries["query_id"].astype(str).duplicated().any():
        raise ValueError("Candidate query IDs must be unique")
    context_index = contexts.set_index("context_id", drop=False)
    if context_index.index.duplicated().any():
        raise ValueError("Context IDs must be unique")

    examples: list[SiteNExample] = []
    for context_id, group in queries.groupby("context_id", sort=True):
        if str(context_id) not in context_index.index:
            raise ValueError(f"Candidate query references unknown context {context_id}")
        context = context_index.loc[str(context_id)]
        node_categorical = torch.tensor(
            _parse_matrix(
                context["model_node_categorical_json"],
                width=len(NODE_CATEGORY_SIZES),
                dtype=int,
            ),
            dtype=torch.long,
        )
        edge_pairs = _parse_matrix(
            context["model_directed_edges_json"],
            width=2,
            dtype=int,
        )
        edge_index = torch.tensor(edge_pairs.T, dtype=torch.long).contiguous()
        edge_categorical = torch.tensor(
            _parse_matrix(
                context["model_edge_categorical_json"],
                width=len(EDGE_CATEGORY_SIZES),
                dtype=int,
            ),
            dtype=torch.long,
        )
        local_values = _parse_matrix(
            context["node_local4_json"],
            width=len(LOCAL_FEATURES),
            dtype=float,
        )
        local_mask = _parse_matrix(
            context["node_local4_available_json"],
            width=len(LOCAL_FEATURES),
            dtype=bool,
        ).astype(bool)
        global_values = _parse_vector(
            context["molecule_global6_json"],
            width=len(GLOBAL_FEATURES),
            dtype=float,
        )
        global_mask = _parse_vector(
            context["molecule_global6_available_json"],
            width=len(GLOBAL_FEATURES),
            dtype=bool,
        ).astype(bool)
        if (
            node_categorical.shape[0] != local_values.shape[0]
            or edge_index.shape[1] != edge_categorical.shape[0]
        ):
            raise ValueError(f"{context_id}: graph feature alignment changed")

        ordered = group.sort_values("query_id", kind="stable")
        site_members = tuple(
            _parse_int_list(value) for value in ordered["member_atom_indices_json"]
        )
        if any(
            not members
            or any(index < 0 or index >= node_categorical.shape[0] for index in members)
            for members in site_members
        ):
            raise ValueError(f"{context_id}: invalid candidate membership")
        site_types = tuple(ordered["site_type"].astype(str))
        unknown = sorted(set(site_types) - set(SITE_TYPE_NAMES))
        if unknown:
            raise ValueError(f"{context_id}: unknown site types {unknown}")
        if "N_value" in ordered:
            n_targets = np.asarray(
                [
                    float(value) if pd.notna(value) else np.nan
                    for value in ordered["N_value"]
                ],
                dtype=float,
            )
        else:
            n_targets = np.full(len(ordered), np.nan, dtype=float)
        examples.append(
            SiteNExample(
                context_id=str(context_id),
                species_id=str(context["species_id"]),
                connectivity_id=str(context["connectivity_id"]),
                node_categorical=node_categorical,
                edge_index=edge_index,
                edge_categorical=edge_categorical,
                local_values=local_values,
                local_mask=local_mask,
                global_values=global_values,
                global_mask=global_mask,
                solvent_values=np.asarray(
                    [float(context[column]) for column in SOLVENT_FEATURES],
                    dtype=float,
                ),
                solvent_raw=str(context["solvent_raw"]),
                model_formal_charge=float(context["model_formal_charge"]),
                target_ids=tuple(ordered["query_id"].astype(str)),
                site_object_ids=tuple(ordered["candidate_site_id"].astype(str)),
                site_types=site_types,
                site_members=site_members,
                n_targets=n_targets,
            )
        )
    return examples
