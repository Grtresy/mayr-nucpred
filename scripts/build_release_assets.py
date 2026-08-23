"""Build deterministic weight archives and their public manifest.

This maintainer utility reads the frozen private source checkout, verifies every
binary against the runtime/score-freeze registries, and writes public release
metadata plus two deterministic tar.gz assets. It never modifies source files.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CAMPAIGN_ID = "mayr-n-publication-20260805-v1"
RELEASE_VERSION = "1.0.0"
CAMPAIGN_REL = Path("artifacts/campaigns") / CAMPAIGN_ID
MODELING_REL = CAMPAIGN_REL / "modeling"
DEPLOYMENT_ARCHIVE = f"mayr-nucpred-v{RELEASE_VERSION}-deployment-weights.tar.gz"
OOF_ARCHIVE = f"mayr-nucpred-v{RELEASE_VERSION}-oof-weights.tar.gz"


@dataclass(frozen=True)
class Artifact:
    path: str
    release_layer: str
    role: str
    outer_fold: int | None
    initialization_seed: int | None
    expected_sha256: str
    model_state_sha256: str | None
    archive: str


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(
    source_root: Path,
    *,
    path: str,
    release_layer: str,
    role: str,
    outer_fold: int | None,
    initialization_seed: int | None,
    expected_sha256: str,
    model_state_sha256: str | None,
    archive: str,
) -> Artifact:
    source = source_root / path
    if not source.is_file():
        raise FileNotFoundError(source)
    observed = _sha256(source)
    if observed != expected_sha256:
        raise ValueError(
            f"Frozen artifact drifted: {path}; expected {expected_sha256}, got {observed}"
        )
    return Artifact(
        path=path,
        release_layer=release_layer,
        role=role,
        outer_fold=outer_fold,
        initialization_seed=initialization_seed,
        expected_sha256=expected_sha256,
        model_state_sha256=model_state_sha256,
        archive=archive,
    )


def _deployment_artifacts(source_root: Path) -> tuple[list[Artifact], Path]:
    registry_rel = MODELING_REL / "automatic_site/deployment/runtime_registry.json"
    registry = _read_json(source_root / registry_rel)
    if registry.get("campaign_id") != CAMPAIGN_ID:
        raise ValueError("Deployment campaign identity changed")
    if registry.get("final_refit_performed") is not True:
        raise ValueError("Deployment final refit is not asserted")
    if registry.get("all_corrected_v2_targets_used") is not True:
        raise ValueError("Deployment does not bind all corrected-v2 targets")
    if int(registry.get("final_refit_target_count", -1)) != 1038:
        raise ValueError("Deployment target count changed")

    model = registry["publication_model"]
    output: list[Artifact] = []
    for binding in model["conditional_n_bindings"]:
        output.append(
            _artifact(
                source_root,
                path=str(binding["path"]),
                release_layer="deployment",
                role="conditional_n",
                outer_fold=None,
                initialization_seed=int(binding["initialization_seed"]),
                expected_sha256=str(binding["sha256"]),
                model_state_sha256=str(binding["model_state_sha256"]),
                archive=DEPLOYMENT_ARCHIVE,
            )
        )
    ranker = model["ranker_checkpoint"]
    output.append(
        _artifact(
            source_root,
            path=str(ranker["path"]),
            release_layer="deployment",
            role="site_ranker",
            outer_fold=None,
            initialization_seed=None,
            expected_sha256=str(ranker["sha256"]),
            model_state_sha256=str(ranker["ranker_state_sha256"]),
            archive=DEPLOYMENT_ARCHIVE,
        )
    )
    residual = model["region_membership_residual"]
    output.append(
        _artifact(
            source_root,
            path=str(residual["path"]),
            release_layer="deployment",
            role="region_residual",
            outer_fold=None,
            initialization_seed=None,
            expected_sha256=str(residual["sha256"]),
            model_state_sha256=None,
            archive=DEPLOYMENT_ARCHIVE,
        )
    )
    return output, registry_rel


def _oof_artifacts(source_root: Path) -> tuple[list[Artifact], list[Path]]:
    output: list[Artifact] = []
    summary_paths: list[Path] = []
    for outer_fold in range(5):
        summary_rel = (
            MODELING_REL
            / "automatic_site/outer_score_freeze"
            / f"outer-{outer_fold}/summary.json"
        )
        summary_paths.append(summary_rel)
        summary = _read_json(source_root / summary_rel)
        if summary.get("status") != "frozen":
            raise ValueError(f"OOF score package is not frozen for outer-{outer_fold}")
        if int(summary.get("outer_fold", -1)) != outer_fold:
            raise ValueError(f"OOF fold identity changed for outer-{outer_fold}")
        if summary.get("metrics_computed_before_score_freeze") is not False:
            raise ValueError(f"Metrics were visible before score freeze for outer-{outer_fold}")
        if summary.get("N_labels_read_before_score_freeze") is not False:
            raise ValueError(f"N labels were visible before score freeze for outer-{outer_fold}")

        bindings = summary["conditional_n_bindings"]
        if len(bindings) != 3:
            raise ValueError(f"Expected three conditional-N members in outer-{outer_fold}")
        for binding in bindings:
            output.append(
                _artifact(
                    source_root,
                    path=str(binding["path"]),
                    release_layer="oof",
                    role="conditional_n",
                    outer_fold=outer_fold,
                    initialization_seed=int(binding["initialization_seed"]),
                    expected_sha256=str(binding["sha256"]),
                    model_state_sha256=str(binding["model_state_sha256"]),
                    archive=OOF_ARCHIVE,
                )
            )
        output.append(
            _artifact(
                source_root,
                path=str(summary["ranker_checkpoint_path"]),
                release_layer="oof",
                role="site_ranker",
                outer_fold=outer_fold,
                initialization_seed=None,
                expected_sha256=str(summary["ranker_checkpoint_sha256"]),
                model_state_sha256=None,
                archive=OOF_ARCHIVE,
            )
        )
        output.append(
            _artifact(
                source_root,
                path=str(summary["region_residual_path"]),
                release_layer="oof",
                role="region_residual",
                outer_fold=outer_fold,
                initialization_seed=None,
                expected_sha256=str(summary["region_residual_sha256"]),
                model_state_sha256=None,
                archive=OOF_ARCHIVE,
            )
        )
    return output, summary_paths


def _companion_metadata(
    source_root: Path,
    artifacts: list[Artifact],
    deployment_registry: Path,
    score_summaries: list[Path],
) -> dict[str, list[Path]]:
    deployment: set[Path] = {deployment_registry}
    oof: set[Path] = set(score_summaries)
    deployment.add(MODELING_REL / "automatic_site/deployment/summary.json")

    for artifact in artifacts:
        path = Path(artifact.path)
        destination = deployment if artifact.release_layer == "deployment" else oof
        if artifact.role == "conditional_n":
            destination.add(path.with_name("summary.json"))
        elif artifact.role in {"site_ranker", "region_residual"}:
            destination.add(path.parent / "summary.json")
            if artifact.release_layer == "oof":
                destination.add(path.parent / "calibration_audit.json")
                destination.add(path.parent / "region_refit_audit.json")

    for paths in (deployment, oof):
        missing = [path for path in paths if not (source_root / path).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing companion metadata: {missing}")
    return {"deployment": sorted(deployment), "oof": sorted(oof)}


def _write_tar_gz(output: Path, source_root: Path, members: list[Path]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for relative in sorted(set(members), key=lambda path: path.as_posix()):
                    source = source_root / relative
                    info = archive.gettarinfo(str(source), arcname=relative.as_posix())
                    info.mtime = 0
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mode = 0o644
                    with source.open("rb") as handle:
                        archive.addfile(info, handle)


def _copy_metadata(
    release_root: Path,
    source_root: Path,
    metadata: dict[str, list[Path]],
) -> None:
    metadata_root = release_root / "weights/metadata"
    for layer, paths in metadata.items():
        for relative in paths:
            destination = metadata_root / layer / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_root / relative, destination)


def build(source_root: Path, release_root: Path) -> None:
    deployment, registry_rel = _deployment_artifacts(source_root)
    oof, score_summaries = _oof_artifacts(source_root)
    artifacts = sorted(deployment + oof, key=lambda item: item.path)
    if len(artifacts) != 30 or len({item.path for item in artifacts}) != 30:
        raise ValueError("Release must contain exactly 30 unique binary artifacts")
    if len(deployment) != 5 or len(oof) != 25:
        raise ValueError("Expected 5 deployment and 25 OOF artifacts")

    records = []
    for artifact in artifacts:
        source = source_root / artifact.path
        records.append(
            {
                "path": artifact.path,
                "bytes": source.stat().st_size,
                "sha256": artifact.expected_sha256,
                "model_state_sha256": artifact.model_state_sha256,
                "release_layer": artifact.release_layer,
                "role": artifact.role,
                "outer_fold": artifact.outer_fold,
                "initialization_seed": artifact.initialization_seed,
                "archive": artifact.archive,
                "license": "Apache-2.0",
            }
        )

    manifest = {
        "schema_version": "mayr-nucpred.weight-release-manifest.v1",
        "release_version": RELEASE_VERSION,
        "campaign_id": CAMPAIGN_ID,
        "release_date": "2026-08-22",
        "artifact_count": len(records),
        "deployment_artifact_count": len(deployment),
        "oof_artifact_count": len(oof),
        "total_uncompressed_bytes": sum(record["bytes"] for record in records),
        "license": "Apache-2.0",
        "post_review_retraining_release": False,
        "artifacts": records,
    }
    weights_root = release_root / "weights"
    weights_root.mkdir(parents=True, exist_ok=True)
    (weights_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (weights_root / "SHA256SUMS").write_text(
        "".join(f"{item.expected_sha256}  {item.path}\n" for item in artifacts),
        encoding="utf-8",
    )

    metadata = _companion_metadata(
        source_root, artifacts, registry_rel, score_summaries
    )
    _copy_metadata(release_root, source_root, metadata)

    deployment_members = [Path(item.path) for item in deployment] + metadata["deployment"]
    oof_members = [Path(item.path) for item in oof] + metadata["oof"]
    dist = release_root / "dist"
    _write_tar_gz(dist / DEPLOYMENT_ARCHIVE, source_root, deployment_members)
    _write_tar_gz(dist / OOF_ARCHIVE, source_root, oof_members)
    archive_paths = [dist / DEPLOYMENT_ARCHIVE, dist / OOF_ARCHIVE]
    (dist / "SHA256SUMS").write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in archive_paths),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Private source checkout containing the frozen campaign artifacts.",
    )
    parser.add_argument(
        "--release-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Public release repository root (defaults to this checkout).",
    )
    args = parser.parse_args()
    build(args.source_root.resolve(), args.release_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
