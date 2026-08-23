"""Run label-free operational checks against the frozen publication runtime."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import shutil
import tempfile

import pandas as pd
from rdkit import Chem

from nucpred.core.files import atomic_write_json, sha256_file
from nucpred.inference import mayr_nextgen_runtime as runtime
from nucpred.training.mayr_site_inference_assets import canonical_sha256


VERIFICATION_SCHEMA = "nucpred.mayr-n-publication-runtime-verification.v1"
DEFAULT_OUTPUT = (
    runtime.DEFAULT_PUBLICATION_REGISTRY.parent.parent / "runtime_verification"
)
DEFAULT_CACHE = (
    runtime.ROOT
    / "data/interim/mayr_unseen_runtime/publication-operational-20260805-v1"
)


def _request(
    request_id: str,
    smiles: str,
    solvent: str,
    formal_charge: int,
) -> dict[str, object]:
    return {
        "schema_version": "nucpred.mayr-nextgen-inference-request.v1",
        "request_id": request_id,
        "smiles": smiles,
        "solvent": solvent,
        "formal_charge": formal_charge,
    }


def _scientific_response_sha256(response: Mapping[str, object]) -> str:
    payload = deepcopy(dict(response))
    provenance = payload.get("provenance")
    if isinstance(provenance, dict):
        provenance.pop("created_at_utc", None)
    return canonical_sha256(payload)


def _connectivity_block(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise runtime.MayrNextgenRuntimeError("Invalid verification SMILES")
    return Chem.MolToInchiKey(molecule).split("-", maxsplit=1)[0]


def _response_audit(response: Mapping[str, object]) -> dict[str, object]:
    candidates = list(response["candidates"])
    counts = Counter(str(candidate["site_type"]) for candidate in candidates)
    top = candidates[0] if candidates else None
    return {
        "status": response["status"],
        "candidate_count": len(candidates),
        "candidate_count_by_type": dict(sorted(counts.items())),
        "all_raw_rank_scores_available": all(
            candidate["validity"]["raw_sigmoid_score"] is not None
            for candidate in candidates
        ),
        "all_conditional_n_available": all(
            candidate["conditional_N"]["status"] == "available"
            for candidate in candidates
        ),
        "all_absolute_probabilities_withheld": all(
            candidate["validity"]["absolute_site_probability"] is None
            for candidate in candidates
        ),
        "top_candidate_site_id": top["candidate_site_id"] if top else None,
        "top_candidate_type": top["site_type"] if top else None,
        "top_conditional_N": top["conditional_N"] if top else None,
        "feature_status": response["feature_status"],
        "scientific_response_sha256": _scientific_response_sha256(response),
    }


def run_verification(
    *,
    registry_path: str | Path = runtime.DEFAULT_PUBLICATION_REGISTRY,
    config_path: str | Path = runtime.DEFAULT_PUBLICATION_CONFIG,
    output_directory: str | Path = DEFAULT_OUTPUT,
    feature_cache_directory: str | Path = DEFAULT_CACHE,
    device: str = "cpu",
) -> dict[str, object]:
    registry_path = Path(registry_path).resolve()
    config_path = Path(config_path).resolve()
    output = Path(output_directory).resolve()
    feature_cache = Path(feature_cache_directory).resolve()
    if output.exists():
        raise runtime.MayrNextgenRuntimeError(
            "Refusing to overwrite runtime verification"
        )
    registry = runtime._load_registry(registry_path)
    if not runtime._publication_registry(registry):
        raise runtime.MayrNextgenRuntimeError(
            "Operational verification requires the publication registry"
        )
    dataset = runtime._repo_path(
        registry["dataset_directory"], label="verification dataset"
    )
    contexts = pd.read_parquet(dataset / "contexts.parquet")
    outer_membership = pd.read_csv(dataset / "outer_fold_membership.csv")

    requests = {
        "cached_singleton": _request(
            "publication-cached-singleton", "[Cl-]", "MeOH", -1
        ),
        "cached_multitype": _request(
            "publication-cached-multitype", "C1CCSC1", "MeCN", 0
        ),
        "unseen_connectivity": _request(
            "publication-unseen-operational", "Nc1ccsc1", "MeOH", 0
        ),
    }
    unseen_block = _connectivity_block(str(requests["unseen_connectivity"]["smiles"]))
    runtime_inventory_blocks = set(contexts["connectivity_id"].astype(str))
    supervised_training_blocks = set(
        outer_membership["connectivity_id"].astype(str)
    )
    if (
        unseen_block in runtime_inventory_blocks
        or unseen_block in supervised_training_blocks
    ):
        raise runtime.MayrNextgenRuntimeError(
            "Operational unseen probe overlaps the publication dataset"
        )

    responses = {
        name: runtime.infer(
            request,
            registry_path=registry_path,
            config_path=config_path,
            device=device,
            feature_cache_directory=feature_cache,
        )
        for name, request in requests.items()
    }
    repeat = runtime.infer(
        requests["unseen_connectivity"],
        registry_path=registry_path,
        config_path=config_path,
        device=device,
        feature_cache_directory=feature_cache,
    )
    unseen = responses["unseen_connectivity"]
    repeat_equal = _scientific_response_sha256(unseen) == _scientific_response_sha256(
        repeat
    )
    if not repeat_equal:
        raise runtime.MayrNextgenRuntimeError(
            "CPU publication runtime is not scientifically byte-deterministic"
        )
    if (
        unseen["status"] != "partial"
        or unseen["feature_status"]["xtb_electronic"] != "complete"
        or "unseen_connectivity"
        not in {
            reason
            for candidate in unseen["candidates"]
            for reason in candidate["applicability"]["reasons"]
        }
        or set(candidate["site_type"] for candidate in unseen["candidates"])
        != set(runtime.SITE_TYPES)
    ):
        raise runtime.MayrNextgenRuntimeError(
            "Publication unseen operational probe changed"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".runtime-verification.staging-", dir=output.parent)
    )
    try:
        for name, request in requests.items():
            atomic_write_json(staging / f"{name}_request.json", request)
            atomic_write_json(staging / f"{name}_response.json", responses[name])
        summary: dict[str, object] = {
            "schema_version": VERIFICATION_SCHEMA,
            "status": "pass",
            "campaign_id": registry["campaign_id"],
            "registry_file_sha256": sha256_file(registry_path),
            "registry_internal_sha256": registry["registry_sha256"],
            "device": device,
            "target_or_site_labels_read": False,
            "external_scientific_evidence_claimed": False,
            "operational_probe_only": True,
            "cached_singleton": _response_audit(responses["cached_singleton"]),
            "cached_multitype": _response_audit(responses["cached_multitype"]),
            "unseen_connectivity": {
                **_response_audit(unseen),
                "connectivity_block": unseen_block,
                "supervised_training_connectivity_count": len(
                    supervised_training_blocks
                ),
                "supervised_training_connectivity_overlap": False,
                "runtime_context_inventory_connectivity_count": len(
                    runtime_inventory_blocks
                ),
                "runtime_context_inventory_connectivity_overlap": False,
                "repeat_scientific_response_identical": repeat_equal,
                "previously_used_legacy_operational_probe": True,
                "eligible_for_pristine_external_confirmation": False,
            },
            "source_path": Path(__file__).resolve().relative_to(runtime.ROOT).as_posix(),
            "source_sha256": sha256_file(Path(__file__).resolve()),
            "completed_at_utc": datetime.now(UTC).isoformat(),
        }
        atomic_write_json(staging / "summary.json", summary)
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=runtime.DEFAULT_PUBLICATION_REGISTRY)
    parser.add_argument("--config", type=Path, default=runtime.DEFAULT_PUBLICATION_CONFIG)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--feature-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    result = run_verification(
        registry_path=args.registry,
        config_path=args.config,
        output_directory=args.output_directory,
        feature_cache_directory=args.feature_cache,
        device=args.device,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
