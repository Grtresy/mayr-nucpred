"""Label-blind outer-test score freeze for automatic Mayr site prediction."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
import gc
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
import torch

from nucpred.core.files import atomic_write_json, sha256_file
from nucpred.publication import mayr_site_publication as shared
from nucpred.publication.mayr_site_training import (
    OUTER_REFIT_SCHEMA,
    RANKER_CHECKPOINT_SCHEMA,
)
from nucpred.training.mayr_site_inference_assets import (
    ranker_from_checkpoint,
    score_ranker_from_source_features,
)
from nucpred.training.mayr_site_ranker import (
    TypeAwarePlattCalibrator,
)
from nucpred.training.mayr_site_region_residual import (
    apply_region_residual,
    origin_vocabulary,
    region_feature_matrix,
    score_region_residual,
)


SCORE_SCHEMA = "nucpred.mayr-n-publication-automatic-site-score-freeze.v1"
FORBIDDEN_SCORE_COLUMNS = frozenset(
    {
        "target_id",
        "site_object_id",
        "exact_label",
        "N_value",
        "N_true",
        "N_target",
        "N_label",
        "measurement_count",
        "source_ids_json",
        "paper_keys_json",
    }
)
MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _output_root(config: Mapping[str, Any]) -> Path:
    return shared.project_path(config["output_directory"], label="site output")


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(shared.ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _source_bindings(config_path: Path) -> dict[str, dict[str, object]]:
    files = {
        "config": config_path,
        "scoring_source": Path(__file__).resolve(),
        "shared_source": Path(shared.__file__).resolve(),
    }
    return {
        key: {
            "path": path.relative_to(shared.ROOT).as_posix(),
            "sha256": sha256_file(path),
            "bytes": int(path.stat().st_size),
        }
        for key, path in files.items()
    }


def _unlabeled_outer_contexts(
    config: Mapping[str, Any], *, outer_fold: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, set[str]]:
    dataset = shared.project_path(config["dataset"]["directory"], label="dataset")
    # Read only identity/split columns.  target_id is intentionally not requested.
    membership = pd.read_csv(
        dataset / "outer_fold_membership.csv",
        usecols=["outer_fold", "role", "context_id", "species_id", "connectivity_id"],
    )
    selected = membership.loc[membership["outer_fold"].eq(outer_fold)]
    test_identity = (
        selected.loc[
            selected["role"].eq("test"),
            ["context_id", "species_id", "connectivity_id"],
        ]
        .drop_duplicates()
        .sort_values("context_id", kind="stable")
        .reset_index(drop=True)
    )
    if test_identity["context_id"].astype(str).duplicated().any():
        raise shared.PublicationSiteError("Unlabeled outer context is duplicated")
    development_connectivities = set(
        selected.loc[selected["role"].eq("development"), "connectivity_id"].astype(str)
    )
    if set(test_identity["connectivity_id"].astype(str)) & development_connectivities:
        raise shared.PublicationSiteError("Outer scoring split leaks connectivity")
    contexts = pd.read_parquet(dataset / "contexts.parquet")
    selected_contexts = contexts.merge(
        test_identity,
        on=["context_id", "species_id", "connectivity_id"],
        how="inner",
        validate="one_to_one",
    )
    if len(selected_contexts) != len(test_identity):
        raise shared.PublicationSiteError("Unlabeled outer context coverage changed")
    species = pd.read_parquet(dataset / "species.parquet")
    return selected_contexts, species, test_identity, development_connectivities


def _fingerprint(smiles: str):
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise shared.PublicationSiteError(f"Invalid frozen species SMILES: {smiles}")
    return MORGAN_GENERATOR.GetFingerprint(molecule)


def _structure_distances(
    *,
    test_contexts: pd.DataFrame,
    species: pd.DataFrame,
    development_connectivities: set[str],
) -> dict[str, float]:
    identity = species[
        ["species_id", "connectivity_id", "model_canonical_smiles"]
    ].drop_duplicates("species_id")
    development = identity.loc[
        identity["connectivity_id"].astype(str).isin(development_connectivities)
    ]
    if development.empty:
        raise shared.PublicationSiteError("No development structures for OOD distance")
    development_fingerprints = [
        _fingerprint(value) for value in development["model_canonical_smiles"]
    ]
    smiles_by_species = identity.set_index("species_id")[
        "model_canonical_smiles"
    ].astype(str)
    result: dict[str, float] = {}
    for row in test_contexts.itertuples(index=False):
        fingerprint = _fingerprint(smiles_by_species.loc[str(row.species_id)])
        similarities = DataStructs.BulkTanimotoSimilarity(
            fingerprint, development_fingerprints
        )
        result[str(row.context_id)] = 1.0 - float(max(similarities))
    return result


def _load_ranker(
    config: Mapping[str, Any], *, outer_fold: int
) -> tuple[torch.nn.Module, dict[str, Any], Path, Path, dict[str, object]]:
    directory = _output_root(config) / "outer_refit" / f"outer-{outer_fold}"
    summary_path = directory / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("schema_version") != OUTER_REFIT_SCHEMA
        or summary.get("status") != "pass"
    ):
        raise shared.PublicationSiteError("Outer site refit is not frozen")
    if int(summary.get("outer_test_target_rows_loaded", -1)) != 0:
        raise shared.PublicationSiteError("Outer site refit read test targets")
    checkpoint_path = directory / "ranker_checkpoint.pt"
    if sha256_file(checkpoint_path) != summary.get("ranker_checkpoint_sha256"):
        raise shared.PublicationSiteError("Outer ranker checkpoint drifted")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected = {
        "schema_version": RANKER_CHECKPOINT_SCHEMA,
        "phase": "outer_development_refit",
        "outer_fold": outer_fold,
        "outer_test_target_rows_loaded": 0,
        "outer_test_predictions_computed": False,
        "unknown_as_negative_count": 0,
        "candidate_softmax_used": False,
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise shared.PublicationSiteError(f"Outer ranker contract changed: {key}")
    expected_ablation = (
        dict(config["ablation"])
        if isinstance(config.get("ablation"), Mapping)
        else None
    )
    if checkpoint.get("input_ablation") != expected_ablation:
        raise shared.PublicationSiteError("Outer ranker ablation binding changed")
    model = ranker_from_checkpoint(checkpoint)
    residual_path = directory / "region_residual.joblib"
    if (
        str(checkpoint["region_membership_residual"]["arm"])
        == "region_structural_residual"
    ):
        if not residual_path.is_file():
            raise shared.PublicationSiteError("Outer region residual is missing")
    return model, checkpoint, checkpoint_path, residual_path, summary


def _apply_ranker(
    *,
    config: Mapping[str, Any],
    ordered: pd.DataFrame,
    source_features: torch.Tensor,
    n_mean: np.ndarray,
    n_std: np.ndarray,
    candidates: pd.DataFrame,
    ranker: torch.nn.Module,
    checkpoint: Mapping[str, Any],
    residual_path: Path,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    type_index = shared.model_facing_site_type_indices(ordered, config)
    with torch.no_grad():
        tensors = score_ranker_from_source_features(
            ranker=ranker,
            checkpoint=checkpoint,
            source_features=source_features,
            type_index=type_index,
        )
    components = {key: value.detach().cpu().numpy() for key, value in tensors.items()}
    logits = components["canonical_logit"].copy()
    region_probability = np.full(len(ordered), np.nan, dtype=float)
    selection = checkpoint["region_membership_residual"]
    if (
        not shared.type_dependent_region_residual_enabled(config)
        and str(selection["arm"]) == "region_structural_residual"
    ):
        raise shared.PublicationSiteError(
            "Site-type ablation checkpoint retained a type-dependent residual"
        )
    if str(selection["arm"]) == "region_structural_residual":
        bundle = joblib.load(residual_path)
        if int(bundle["minimum_samples_leaf"]) != int(
            selection["minimum_samples_leaf"]
        ):
            raise shared.PublicationSiteError("Region residual leaf size drifted")
        origins = origin_vocabulary(candidates)
        positions, features, names = region_feature_matrix(
            ordered.reset_index(drop=True),
            membership_logits=components["membership_logit"],
            compatibility_logits=components["compatibility_logit"],
            conditional_n_mean=n_mean,
            conditional_n_std=n_std,
            origin_vocabulary_values=origins,
        )
        probabilities = score_region_residual(
            bundle, features, expected_feature_names=names
        )
        region_probability[positions] = probabilities
        logits, _ = apply_region_residual(
            ordered.reset_index(drop=True),
            base_logits=components["canonical_logit"],
            region_positions=positions,
            residual_probabilities=probabilities,
            residual_weight=float(selection["residual_weight"]),
            maximum_base_margin=(
                float(selection["maximum_base_margin"])
                if selection.get("maximum_base_margin") is not None
                else None
            ),
            top_k=(
                int(selection["top_k"]) if selection.get("top_k") is not None else None
            ),
        )
    return components, logits, region_probability


def assert_label_blind_scores(frame: pd.DataFrame) -> None:
    overlap = sorted(FORBIDDEN_SCORE_COLUMNS & set(frame.columns))
    if overlap:
        raise shared.PublicationSiteError(
            f"Score package contains forbidden label columns: {overlap}"
        )
    if frame.empty or frame["query_id"].astype(str).duplicated().any():
        raise shared.PublicationSiteError("Score package query identity changed")
    prediction_columns = [
        "conditional_N_prediction",
        "conditional_N_ensemble_std",
        "base_canonical_logit",
        "canonical_logit",
        "canonical_endpoint_probability",
    ]
    if not np.isfinite(frame[prediction_columns].to_numpy(dtype=float)).all():
        raise shared.PublicationSiteError("Score package has non-finite predictions")


def _ranked_outputs(
    frame: pd.DataFrame,
    *,
    margin_threshold: float,
    structure_distances: Mapping[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ranked = frame.sort_values(
        ["context_id", "canonical_logit", "candidate_site_id"],
        ascending=[True, False, True],
        kind="stable",
    ).copy()
    ranked["candidate_rank"] = ranked.groupby("context_id").cumcount() + 1
    top_two = ranked.loc[ranked["candidate_rank"].le(2)].pivot(
        index="context_id", columns="candidate_rank", values="canonical_logit"
    )
    margins = (top_two[1] - top_two.get(2, top_two[1])).astype(float)
    ranked["top1_margin"] = ranked["context_id"].map(margins)
    ranked["accepted_by_margin"] = ranked["top1_margin"].ge(margin_threshold)
    ranked["structure_distance"] = ranked["context_id"].map(structure_distances)
    ranked = ranked.sort_values("query_id", kind="stable").reset_index(drop=True)
    top = ranked.loc[ranked["candidate_rank"].eq(1)].copy()
    contexts = top[
        [
            "outer_fold",
            "context_id",
            "species_id",
            "connectivity_id",
            "candidate_site_id",
            "site_type",
            "member_atom_indices_json",
            "conditional_N_prediction",
            "conditional_N_ensemble_std",
            "canonical_logit",
            "canonical_endpoint_probability",
            "top1_margin",
            "accepted_by_margin",
            "structure_distance",
        ]
    ].rename(
        columns={
            "candidate_site_id": "predicted_candidate_site_id",
            "site_type": "predicted_site_type",
            "member_atom_indices_json": "predicted_member_atom_indices_json",
            "conditional_N_prediction": "predicted_N",
            "conditional_N_ensemble_std": "predicted_N_ensemble_std",
            "canonical_logit": "top1_canonical_logit",
            "canonical_endpoint_probability": "top1_endpoint_probability",
        }
    )
    return ranked, contexts.reset_index(drop=True)


def freeze_outer_scores(
    config_path: str | Path,
    *,
    outer_fold: int,
    output_root: str | Path | None = None,
) -> dict[str, object]:
    config, resolved = shared.read_config(config_path)
    shared.verify_bindings(config, resolved)
    if not 0 <= outer_fold < int(config["outer_fold_count"]):
        raise shared.PublicationSiteError("Outer scoring fold is out of range")
    root = Path(output_root).resolve() if output_root else _output_root(config)
    destination = root / "outer_score_freeze" / f"outer-{outer_fold}"
    if destination.exists():
        raise shared.PublicationSiteError("Refusing to overwrite frozen outer scores")
    destination.mkdir(parents=True)
    contexts, species, test_identity, development_connectivities = (
        _unlabeled_outer_contexts(config, outer_fold=outer_fold)
    )
    candidates, policy_audit = shared.deployment_candidates(config, species)
    queries = shared.candidate_universe(
        test_contexts=test_identity,
        candidates=candidates,
    )
    device = torch.device(str(config["device"]))
    models = shared.load_outer_conditional_ensemble(
        config, outer_fold=outer_fold, device=device
    )
    query_ids, source_features, n_mean, n_std, conditional_bindings = (
        shared.encode_queries(
            models=models,
            queries=queries,
            contexts=contexts,
            config=config,
            device=device,
        )
    )
    shared.release_models(models, device=device)
    del models
    ordered = (
        queries.set_index("query_id", drop=False).loc[query_ids].reset_index(drop=True)
    )
    ordered["outer_fold"] = outer_fold
    ordered["conditional_N_prediction"] = n_mean
    ordered["conditional_N_ensemble_std"] = n_std
    ranker, checkpoint, checkpoint_path, residual_path, refit_summary = _load_ranker(
        config, outer_fold=outer_fold
    )
    components, logits, region_probability = _apply_ranker(
        config=config,
        ordered=ordered,
        source_features=source_features,
        n_mean=n_mean,
        n_std=n_std,
        candidates=candidates,
        ranker=ranker,
        checkpoint=checkpoint,
        residual_path=residual_path,
    )
    calibrator = TypeAwarePlattCalibrator.from_payload(checkpoint["calibrator"])
    type_index = shared.model_facing_site_type_indices(ordered, config)
    with torch.no_grad():
        endpoint_probability = calibrator(
            torch.tensor(logits, dtype=torch.float32), type_index
        ).numpy()
    score = ordered[
        [
            "outer_fold",
            "query_id",
            "context_id",
            "species_id",
            "connectivity_id",
            "candidate_site_id",
            "site_type",
            "member_atom_indices_json",
            "member_bond_pairs_json",
            "member_atomic_numbers_json",
            "candidate_origins_json",
            "label_independent",
            "conditional_N_prediction",
            "conditional_N_ensemble_std",
        ]
    ].copy()
    score["base_canonical_logit"] = components["canonical_logit"]
    score["canonical_logit"] = logits
    score["membership_logit"] = components["membership_logit"]
    score["router_selected_logit"] = components["router_selected_logit"]
    score["compatibility_logit"] = components["compatibility_logit"]
    score["region_residual_probability"] = region_probability
    score["canonical_endpoint_probability"] = endpoint_probability
    distances = _structure_distances(
        test_contexts=test_identity,
        species=species,
        development_connectivities=development_connectivities,
    )
    score, context_scores = _ranked_outputs(
        score,
        margin_threshold=float(checkpoint["margin_abstention"]["selected_threshold"]),
        structure_distances=distances,
    )
    assert_label_blind_scores(score)
    assert not (FORBIDDEN_SCORE_COLUMNS & set(context_scores.columns))
    score_path = destination / "candidate_scores.parquet"
    context_path = destination / "context_scores.parquet"
    shared.atomic_parquet(score_path, score)
    shared.atomic_parquet(context_path, context_scores)
    payload: dict[str, object] = {
        "schema_version": SCORE_SCHEMA,
        "status": "frozen",
        "campaign_id": config["campaign_id"],
        "outer_fold": outer_fold,
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "candidate_score_path": _display_path(score_path),
        "candidate_score_sha256": sha256_file(score_path),
        "context_score_path": _display_path(context_path),
        "context_score_sha256": sha256_file(context_path),
        "candidate_score_count": len(score),
        "context_score_count": len(context_scores),
        "connectivity_count": int(context_scores["connectivity_id"].nunique()),
        "ranker_checkpoint_path": checkpoint_path.relative_to(shared.ROOT).as_posix(),
        "ranker_checkpoint_sha256": sha256_file(checkpoint_path),
        "ranker_refit_summary_sha256": sha256_file(
            checkpoint_path.parent / "summary.json"
        ),
        "region_residual_path": (
            residual_path.relative_to(shared.ROOT).as_posix()
            if residual_path.is_file()
            else None
        ),
        "region_residual_sha256": (
            sha256_file(residual_path) if residual_path.is_file() else None
        ),
        "conditional_n_bindings": conditional_bindings,
        "candidate_policy_audit": policy_audit,
        "source_bindings": _source_bindings(resolved),
        "forbidden_score_column_overlap": [],
        "target_table_opened": False,
        "target_id_column_requested": False,
        "site_labels_read_before_score_freeze": False,
        "N_labels_read_before_score_freeze": False,
        "metrics_computed_before_score_freeze": False,
        "candidate_softmax_used": False,
        "sn_imported_or_predicted": False,
        "input_ablation": (
            dict(config["ablation"])
            if isinstance(config.get("ablation"), Mapping)
            else None
        ),
        "true_site_type_available_to_predictor": (
            shared.ablation_name(config) != "without_site_type"
        ),
        "outer_refit_outer_test_rows_loaded": refit_summary[
            "outer_test_target_rows_loaded"
        ],
    }
    atomic_write_json(destination / "summary.json", payload)
    del source_features, ranker, checkpoint, components
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(
        f"outer={outer_fold} contexts={len(context_scores)} "
        f"candidates={len(score)} score_sha={payload['candidate_score_sha256']}",
        flush=True,
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=shared.DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--outer-fold", type=int)
    parser.add_argument("--all", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config, _ = shared.read_config(args.config)
    if args.all == (args.outer_fold is not None):
        raise shared.PublicationSiteError("Choose exactly one of --all or --outer-fold")
    folds = (
        range(int(config["outer_fold_count"])) if args.all else [int(args.outer_fold)]
    )
    for outer_fold in folds:
        freeze_outer_scores(
            args.config,
            outer_fold=outer_fold,
            output_root=args.output_root,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
