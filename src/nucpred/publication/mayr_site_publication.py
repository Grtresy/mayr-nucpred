"""Shared assets for publication-grade automatic Mayr site prediction."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
import gc
import json
from pathlib import Path
import tempfile
import tomllib
from typing import Any

import numpy as np
import pandas as pd
import torch

from nucpred.core.files import atomic_write_json, sha256_file
from nucpred.datasets.mayr_site_candidate_policy import (
    select_deployment_candidates,
)
from nucpred.experiments.mayr.nextgen_stage_c_r2 import _make_stage_c_model
from nucpred.experiments.mayr.nextgen_stage_e_b_r2 import (
    _make_model as _make_eb_model,
)
from nucpred.experiments.mayr.nextgen_stage_e_c_r2 import (
    _make_model as _make_ec_model,
)
from nucpred.experiments.mayr.site_n_formal import _tensor_mapping_sha256
from nucpred.project import get_project_layout
from nucpred.publication.mayr_n_modeling import (
    ABLATION_NAMES,
    CHECKPOINT_SCHEMA,
    PublicationModelingError,
    _in_memory_c2,
    _in_memory_eb,
    _pretraining_entry,
    _training_configs,
    apply_input_ablation,
    read_config as read_conditional_config,
)
from nucpred.publication.mayr_n_outer import load_outer_checkpoint
from nucpred.training.mayr_node_xtb_scratch import SolventVocabulary
from nucpred.training.mayr_site_inference_assets import (
    _encoded_fused_from_output,
    candidate_universe as candidate_universe,
)
from nucpred.training.mayr_site_n import (
    SiteNFoldPreprocessor,
    pack_site_n_batch,
)
from nucpred.training.mayr_site_n_stage_e_b import E_B_N1
from nucpred.training.mayr_site_n_stage_e_c import E_C_N3
from nucpred.training.mayr_site_queries import site_n_examples_from_queries


ROOT = get_project_layout().root
DEFAULT_CONFIG = ROOT / "configs/mayr_n_publication_site_v1.toml"
CONFIG_SCHEMA = "nucpred.mayr-n-publication-site-config.v1"
ABLATION_CONFIG_SCHEMA = "nucpred.mayr-n-publication-site-ablation-config.v1"


class PublicationSiteError(PublicationModelingError):
    """Raised when automatic-site publication contracts are violated."""


def project_path(value: object, *, label: str) -> Path:
    path = (ROOT / str(value)).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as exc:
        raise PublicationSiteError(f"{label} escapes project root") from exc
    return path


def read_config(
    path: str | Path = DEFAULT_CONFIG,
) -> tuple[dict[str, Any], Path]:
    resolved = Path(path).resolve()
    raw = tomllib.loads(resolved.read_text(encoding="utf-8"))
    if raw.get("schema_version") == ABLATION_CONFIG_SCHEMA:
        parent_path = project_path(
            raw["parent_config_path"], label="ablation parent config"
        )
        observed_parent_hash = sha256_file(parent_path)
        if observed_parent_hash != str(raw["parent_config_sha256"]):
            raise PublicationSiteError("Automatic-site ablation parent config drifted")
        config = tomllib.loads(parent_path.read_text(encoding="utf-8"))
        for key in ("campaign_id", "experiment_id", "output_directory", "device"):
            if key in raw:
                config[key] = raw[key]
        for key, value in raw.get("lineage_overrides", {}).items():
            config["lineage"][key] = value
        config["ablation"] = deepcopy(raw["ablation"])
        config["_ablation_parent_config_path"] = parent_path.relative_to(
            ROOT
        ).as_posix()
        config["_ablation_parent_config_sha256"] = observed_parent_hash
    else:
        config = raw
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise PublicationSiteError("Unsupported publication site config schema")
    if config.get("unknown_is_negative") is not False:
        raise PublicationSiteError("Unknown candidates cannot be chemical negatives")
    if config.get("candidate_softmax_used") is not False:
        raise PublicationSiteError("Candidate softmax is forbidden")
    if config.get("no_site_class_used") is not False:
        raise PublicationSiteError("No-site class is outside scope")
    if config.get("outer_test_used_for_selection") is not False:
        raise PublicationSiteError("Outer test cannot select the site model")
    if config["lineage"].get("base_ranker_arm") != "hierarchical_exact":
        raise PublicationSiteError("Base ranker arm changed")
    if config["region_residual"].get("unknown_as_negative") is not False:
        raise PublicationSiteError("Region residual unknown boundary changed")
    if config["region_residual"].get("candidate_softmax_used") is not False:
        raise PublicationSiteError("Region residual softmax boundary changed")
    phases = config["phase_separation"]
    if phases.get("inner_ranker_may_read_outer_test_targets") is not False:
        raise PublicationSiteError("Inner ranker can read outer test")
    if phases.get("outer_ranker_refit_may_read_outer_test_targets") is not False:
        raise PublicationSiteError("Outer ranker refit can read outer test")
    if phases.get("outer_test_scoring_reads_target_or_site_labels") is not False:
        raise PublicationSiteError("Outer score freeze can read labels")
    ablation = config.get("ablation")
    if ablation is not None:
        if (
            not isinstance(ablation, Mapping)
            or ablation.get("name") not in ABLATION_NAMES
        ):
            raise PublicationSiteError("Unsupported automatic-site ablation")
        if ablation.get("outer_test_used_for_selection") is not False:
            raise PublicationSiteError("Automatic-site ablation can use outer test")
        if ablation.get("ranker_architecture_and_optimization_matched") is not True:
            raise PublicationSiteError("Automatic-site ablation changed the ranker")
        if ablation.get("true_metadata_retained_only_outside_predictor") is not True:
            raise PublicationSiteError(
                "Automatic-site ablation metadata boundary changed"
            )
        conditional, _ = conditional_config(config)
        conditional_ablation = conditional.get("ablation")
        if not isinstance(conditional_ablation, Mapping) or conditional_ablation.get(
            "name"
        ) != ablation.get("name"):
            raise PublicationSiteError(
                "Automatic-site and conditional-N ablations are not matched"
            )
        if ablation.get("name") == "without_site_type":
            if ablation.get("model_facing_site_type") != "constant_atom":
                raise PublicationSiteError("Site-type ablation is not constant-token")
            if ablation.get("type_dependent_region_residual_enabled") is not False:
                raise PublicationSiteError(
                    "Site-type ablation retains region type leakage"
                )
    return config, resolved


def ablation_name(config: Mapping[str, Any]) -> str | None:
    ablation = config.get("ablation")
    return str(ablation["name"]) if isinstance(ablation, Mapping) else None


def conditional_config(
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], Path]:
    """Load the exact conditional-N config bound by one site experiment."""

    path = project_path(
        config["lineage"]["conditional_n_config_path"],
        label="lineage.conditional_n_config_path",
    )
    conditional, resolved = read_conditional_config(path)
    return conditional, resolved


def model_facing_frame(frame: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    """Return predictor inputs while preserving true metadata in the source frame."""

    result = frame.copy()
    if ablation_name(config) == "without_site_type":
        result["site_type"] = "atom"
    return result


def model_facing_site_type_indices(
    frame: pd.DataFrame, config: Mapping[str, Any]
) -> torch.Tensor:
    from nucpred.training.mayr_site_ranker import site_type_indices

    return site_type_indices(model_facing_frame(frame, config)["site_type"].astype(str))


def type_dependent_region_residual_enabled(config: Mapping[str, Any]) -> bool:
    return ablation_name(config) != "without_site_type"


def _verify(path: Path, expected: object, *, label: str) -> dict[str, object]:
    if not path.is_file():
        raise PublicationSiteError(f"Missing {label}: {path}")
    observed = sha256_file(path)
    if observed != str(expected):
        raise PublicationSiteError(f"Frozen {label} drifted: {observed} != {expected}")
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": observed,
        "bytes": int(path.stat().st_size),
    }


def verify_bindings(
    config: Mapping[str, Any], config_path: Path
) -> dict[str, dict[str, object]]:
    verified = {
        "config": {
            "path": config_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(config_path),
            "bytes": int(config_path.stat().st_size),
        }
    }
    if config.get("_ablation_parent_config_path"):
        parent_path = project_path(
            config["_ablation_parent_config_path"], label="ablation parent config"
        )
        verified["ablation.parent_config"] = _verify(
            parent_path,
            config["_ablation_parent_config_sha256"],
            label="ablation parent config",
        )
    for section_name, pairs in {
        "dataset": (
            ("manifest_path", "manifest_sha256"),
            ("candidate_path", "candidate_sha256"),
            ("outer_membership_path", "outer_membership_sha256"),
            ("nested_membership_path", "nested_membership_sha256"),
        ),
        "lineage": (
            ("conditional_n_config_path", "conditional_n_config_sha256"),
            ("historical_v6_config_path", "historical_v6_config_sha256"),
            ("historical_v7_config_path", "historical_v7_config_sha256"),
            ("candidate_policy_source_path", "candidate_policy_source_sha256"),
        ),
    }.items():
        section = config[section_name]
        for path_key, hash_key in pairs:
            label = f"{section_name}.{path_key}"
            verified[label] = _verify(
                project_path(section[path_key], label=label),
                section[hash_key],
                label=label,
            )
    return verified


def dataset_tables(
    config: Mapping[str, Any],
    *,
    include_targets: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = project_path(config["dataset"]["directory"], label="dataset")
    contexts = pd.read_parquet(root / "contexts.parquet")
    targets = pd.read_parquet(root / "targets.parquet") if include_targets else None
    species = pd.read_parquet(root / "species.parquet")
    outer = pd.read_csv(root / "outer_fold_membership.csv")
    nested = pd.read_csv(root / "nested_split_membership.csv")
    return contexts, targets, species, outer, nested


def deployment_candidates(
    config: Mapping[str, Any], species: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, object]]:
    candidates = pd.read_parquet(
        project_path(config["dataset"]["candidate_path"], label="candidate table")
    )
    selected, audit = select_deployment_candidates(candidates, species)
    if set(selected["site_type"].astype(str)) != set(
        config["ranker"]["candidate_types"]
    ):
        raise PublicationSiteError("Deployment candidates lose a site type")
    return selected, audit


def preflight(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    output_directory: str | Path | None = None,
) -> dict[str, object]:
    config, resolved = read_config(config_path)
    verified = verify_bindings(config, resolved)
    contexts, targets, species, outer, nested = dataset_tables(config)
    assert targets is not None
    candidates, policy_audit = deployment_candidates(config, species)
    candidate_ids = set(candidates["candidate_site_id"].astype(str))
    covered = targets["site_object_id"].astype(str).isin(candidate_ids)
    if not covered.all():
        missing = targets.loc[~covered, "target_id"].astype(str).tolist()
        raise PublicationSiteError(f"Deployment policy misses targets: {missing[:5]}")
    if not candidates["label_independent"].astype(bool).all():
        raise PublicationSiteError("Candidate policy exposed labels")
    if outer.loc[outer["role"].eq("test"), "target_id"].duplicated().any():
        raise PublicationSiteError("Outer test target is duplicated")
    test_counts = outer.loc[outer["role"].eq("test")].groupby("target_id").size()
    if len(test_counts) != len(targets) or not test_counts.eq(1).all():
        raise PublicationSiteError("Outer tests do not partition targets")
    split_rows = []
    for outer_fold in range(int(config["outer_fold_count"])):
        selected_outer = outer.loc[outer["outer_fold"].eq(outer_fold)]
        test_connectivity = set(
            selected_outer.loc[
                selected_outer["role"].eq("test"), "connectivity_id"
            ].astype(str)
        )
        for inner_fold in range(int(config["inner_fold_count"])):
            selected_inner = nested.loc[
                nested["outer_fold"].eq(outer_fold)
                & nested["inner_fold"].eq(inner_fold)
            ]
            train_connectivity = set(
                selected_inner.loc[
                    selected_inner["role"].eq("train"), "connectivity_id"
                ].astype(str)
            )
            validation_connectivity = set(
                selected_inner.loc[
                    selected_inner["role"].eq("validation"), "connectivity_id"
                ].astype(str)
            )
            if (
                train_connectivity & validation_connectivity
                or train_connectivity & test_connectivity
                or validation_connectivity & test_connectivity
            ):
                raise PublicationSiteError("Nested site split leaks connectivity")
            split_rows.append(
                {
                    "outer_fold": outer_fold,
                    "inner_fold": inner_fold,
                    "train_connectivity_count": len(train_connectivity),
                    "validation_connectivity_count": len(validation_connectivity),
                    "outer_test_connectivity_count": len(test_connectivity),
                }
            )
    payload: dict[str, object] = {
        "schema_version": "nucpred.mayr-n-publication-site-preflight.v1",
        "status": "pass",
        "campaign_id": config["campaign_id"],
        "experiment_id": config["experiment_id"],
        "verified_bindings": verified,
        "context_count": len(contexts),
        "target_count": len(targets),
        "deployment_candidate_count": len(candidates),
        "deployment_candidate_context_independent": True,
        "known_target_coverage_fraction": float(covered.mean()),
        "candidate_policy_audit": policy_audit,
        "split_audit": split_rows,
        "unknown_as_negative_count": 0,
        "endpoint_relative_noncanonical_candidates_are_not_universal_negatives": True,
        "candidate_softmax_used": False,
        "sn_imported_or_predicted": False,
        "outer_test_target_rows_loaded_by_training": 0,
    }
    target = (
        Path(output_directory).resolve()
        if output_directory is not None
        else project_path(config["output_directory"], label="output directory")
        / "preflight"
    )
    target.mkdir(parents=True, exist_ok=True)
    atomic_write_json(target / "preflight.json", payload, ensure_ascii=False)
    return payload


def _build_conditional_model(
    *,
    payload: Mapping[str, Any],
    initialization_seed: int,
    conditional_config: Mapping[str, Any],
    device: torch.device,
) -> tuple[torch.nn.Module, SiteNFoldPreprocessor, SolventVocabulary]:
    _, pretraining_checkpoint, _ = _pretraining_entry(
        conditional_config,
        initialization_seed,
    )
    preprocessor = SiteNFoldPreprocessor.from_json(payload["preprocessor"])
    vocabulary = SolventVocabulary(tuple(map(str, payload["solvent_vocabulary"])))
    base_config, _, _, _ = _training_configs(conditional_config)
    base, _, _, _ = _make_stage_c_model(
        arm=str(conditional_config["lineage"]["base_arm"]),
        base_config=base_config,
        vocabulary=vocabulary,
        initialization_seed=initialization_seed,
        device=device,
        checkpoint=pretraining_checkpoint,
    )
    frozen_c2 = _in_memory_c2(base, preprocessor, vocabulary)
    eb = _make_eb_model(
        frozen_c2,
        arm=E_B_N1,
        initialization_seed=initialization_seed,
        device=device,
    )
    frozen_eb = _in_memory_eb(eb, preprocessor, vocabulary)
    model = _make_ec_model(
        frozen_eb,
        arm=E_C_N3,
        initialization_seed=initialization_seed,
        device=device,
    )
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise PublicationSiteError("Conditional checkpoint lacks state")
    if _tensor_mapping_sha256(state) != payload.get("model_state_sha256"):
        raise PublicationSiteError("Conditional checkpoint state hash changed")
    model.load_state_dict(state, strict=True)
    if _tensor_mapping_sha256(model.state_dict()) != payload["model_state_sha256"]:
        raise PublicationSiteError("Conditional checkpoint exact load failed")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, preprocessor, vocabulary


def load_inner_conditional_checkpoint(
    config: Mapping[str, Any],
    *,
    outer_fold: int,
    inner_fold: int,
    device: torch.device,
) -> tuple[
    torch.nn.Module, SiteNFoldPreprocessor, SolventVocabulary, dict[str, Any], Path
]:
    path = (
        project_path(
            config["lineage"]["inner_conditional_checkpoint_root"],
            label="inner conditional root",
        )
        / f"outer-{outer_fold}"
        / f"inner-{inner_fold}"
        / "selection_checkpoint.pt"
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != CHECKPOINT_SCHEMA:
        raise PublicationSiteError("Inner conditional checkpoint schema changed")
    contract = payload.get("contract")
    if (
        not isinstance(contract, Mapping)
        or int(contract.get("outer_fold", -1)) != outer_fold
        or int(contract.get("inner_fold", -1)) != inner_fold
        or int(contract.get("outer_test_target_rows_loaded", -1)) != 0
    ):
        raise PublicationSiteError("Inner conditional checkpoint contract changed")
    initialization_seed = int(contract["initialization_seed"])
    bound_conditional_config, _ = conditional_config(config)
    model, preprocessor, vocabulary = _build_conditional_model(
        payload=payload,
        initialization_seed=initialization_seed,
        conditional_config=bound_conditional_config,
        device=device,
    )
    return model, preprocessor, vocabulary, payload, path


def load_outer_conditional_ensemble(
    config: Mapping[str, Any],
    *,
    outer_fold: int,
    device: torch.device,
) -> list[
    tuple[
        int,
        torch.nn.Module,
        SiteNFoldPreprocessor,
        SolventVocabulary,
        dict[str, Any],
        Path,
    ]
]:
    root = project_path(
        config["lineage"]["outer_conditional_checkpoint_root"],
        label="outer conditional root",
    )
    _, bound_conditional_path = conditional_config(config)
    result = []
    for seed in map(int, config["initialization_seeds"]):
        path = root / f"outer-{outer_fold}" / f"init-{seed}" / "model.pt"
        model, preprocessor, vocabulary, payload = load_outer_checkpoint(
            path,
            config_path=bound_conditional_path,
            device=device,
        )
        result.append((seed, model, preprocessor, vocabulary, payload, path))
    return result


def _encode_one_model(
    *,
    model: torch.nn.Module,
    preprocessor: SiteNFoldPreprocessor,
    vocabulary: SolventVocabulary,
    queries: pd.DataFrame,
    contexts: pd.DataFrame,
    input_config: Mapping[str, Any],
    device: torch.device,
    batch_context_count: int = 24,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    context_ids = sorted(set(queries["context_id"].astype(str)))
    query_ids: list[str] = []
    features: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    for start in range(0, len(context_ids), batch_context_count):
        selected_contexts = set(context_ids[start : start + batch_context_count])
        frame = queries.loc[
            queries["context_id"].astype(str).isin(selected_contexts)
        ].copy()
        examples = apply_input_ablation(
            site_n_examples_from_queries(frame, contexts=contexts), input_config
        )
        packed = pack_site_n_batch(
            examples,
            preprocessor=preprocessor,
            solvent_vocabulary=vocabulary,
        )
        inputs = packed.inputs.to(device)
        with torch.no_grad():
            output = model(inputs)
            fused = _encoded_fused_from_output(model, inputs, output)
            n_raw = output.n_prediction_standardized * float(
                preprocessor.target_scale
            ) + float(preprocessor.target_mean)
            augmented = torch.cat((fused, n_raw[:, None]), dim=1)
        query_ids.extend(map(str, packed.target_ids))
        features.append(augmented.detach().cpu().numpy().astype(np.float32))
        predictions.append(n_raw.detach().cpu().numpy().astype(np.float32))
    if len(query_ids) != len(queries) or len(set(query_ids)) != len(query_ids):
        raise PublicationSiteError("Conditional query encoding changed coverage")
    return query_ids, np.concatenate(features), np.concatenate(predictions)


def encode_queries(
    *,
    models: Sequence[
        tuple[
            int,
            torch.nn.Module,
            SiteNFoldPreprocessor,
            SolventVocabulary,
            Mapping[str, Any],
            Path,
        ]
    ],
    queries: pd.DataFrame,
    contexts: pd.DataFrame,
    config: Mapping[str, Any],
    device: torch.device,
) -> tuple[list[str], torch.Tensor, np.ndarray, np.ndarray, list[dict[str, object]]]:
    reference_ids: list[str] | None = None
    features: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    bindings: list[dict[str, object]] = []
    bound_conditional_config, _ = conditional_config(config)
    for seed, model, preprocessor, vocabulary, payload, path in models:
        ids, member_features, n_values = _encode_one_model(
            model=model,
            preprocessor=preprocessor,
            vocabulary=vocabulary,
            queries=queries,
            contexts=contexts,
            input_config=bound_conditional_config,
            device=device,
        )
        if reference_ids is None:
            reference_ids = ids
        elif reference_ids != ids:
            raise PublicationSiteError("Conditional ensemble query order changed")
        features.append(member_features)
        predictions.append(n_values)
        bindings.append(
            {
                "initialization_seed": seed,
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(path),
                "model_state_sha256": payload["model_state_sha256"],
            }
        )
    if reference_ids is None:
        raise PublicationSiteError("No conditional model was encoded")
    matrix = np.stack(predictions, axis=1)
    return (
        reference_ids,
        torch.from_numpy(np.concatenate(features, axis=1)),
        matrix.mean(axis=1),
        matrix.std(axis=1, ddof=0),
        bindings,
    )


def inner_models_for_encoding(
    config: Mapping[str, Any],
    *,
    outer_fold: int,
    inner_fold: int,
    device: torch.device,
) -> list[
    tuple[
        int,
        torch.nn.Module,
        SiteNFoldPreprocessor,
        SolventVocabulary,
        Mapping[str, Any],
        Path,
    ]
]:
    model, preprocessor, vocabulary, payload, path = load_inner_conditional_checkpoint(
        config,
        outer_fold=outer_fold,
        inner_fold=inner_fold,
        device=device,
    )
    seed = int(payload["contract"]["initialization_seed"])
    return [(seed, model, preprocessor, vocabulary, payload, path)]


def release_models(models: Sequence[tuple[Any, ...]], *, device: torch.device) -> None:
    for item in models:
        for value in item:
            if isinstance(value, torch.nn.Module):
                value.to("cpu")
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_parquet(
            temporary,
            index=False,
            engine="pyarrow",
            compression="zstd",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preflight the frozen automatic-site Mayr publication protocol."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-directory", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = preflight(
        args.config,
        output_directory=args.output_directory,
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
