"""Reusable run lifecycle for catalogued experiment output trees."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re

from nucpred.artifacts import ArtifactCatalog, ArtifactCatalogError


class ManagedRunError(RuntimeError):
    """Raised when an experiment cannot satisfy the managed-run contract."""


@dataclass(frozen=True, slots=True)
class ArtifactClassification:
    role: str
    protocol_evidence: bool = False
    media_type: str | None = None


class ManagedRun:
    """Bind one output directory to one catalog lifecycle record."""

    def __init__(
        self,
        catalog: ArtifactCatalog,
        record: dict[str, object],
    ) -> None:
        self.catalog = catalog
        self.record = record
        self.run_id = str(record["run_id"])
        self.directory = catalog.run_directory(self.run_id)
        self._registered: dict[str, tuple[str, bool]] = {}

    @classmethod
    def start(
        cls,
        *,
        experiment: str,
        protocol: str,
        dataset_ids: Sequence[str],
        config_path: str | Path,
        source_paths: Sequence[str | Path],
        command: Sequence[str],
        run_id: str | None = None,
        supersedes: Sequence[str] = (),
        metadata: Mapping[str, object] | None = None,
        catalog: ArtifactCatalog | None = None,
    ) -> "ManagedRun":
        selected_catalog = catalog or ArtifactCatalog()
        record = selected_catalog.create_run(
            experiment=experiment,
            protocol=protocol,
            dataset_ids=dataset_ids,
            run_id=run_id,
            config_path=config_path,
            source_paths=source_paths,
            command=command,
            supersedes=supersedes,
            metadata=metadata,
        )
        return cls(selected_catalog, record)

    def register_tree(
        self,
        classify: Callable[[Path], ArtifactClassification],
    ) -> dict[str, tuple[str, bool]]:
        """Hash and register every file under the run directory exactly once."""
        if self._registered:
            raise ManagedRunError("Run tree has already been registered")
        for path in sorted(self.directory.rglob("*")):
            if path.is_symlink():
                raise ManagedRunError(f"Run outputs cannot contain symlinks: {path}")
            if not path.is_file():
                continue
            relative = path.relative_to(self.directory)
            classification = classify(relative)
            catalog_name = _catalog_name(relative)
            if catalog_name in {name for name, _ in self._registered.values()}:
                raise ManagedRunError(f"Artifact catalog name collision: {relative}")
            self.catalog.add_file(
                self.run_id,
                name=catalog_name,
                path=path,
                role=classification.role,
                protocol_evidence=classification.protocol_evidence,
                media_type=classification.media_type,
            )
            self._registered[relative.as_posix()] = (
                catalog_name,
                classification.protocol_evidence,
            )
        if not self._registered:
            raise ManagedRunError("Run produced no files to register")
        return dict(self._registered)

    def complete(
        self,
        *,
        required_artifacts: Sequence[str | Path],
        required_protocol_evidence: Sequence[str | Path],
    ) -> dict[str, object]:
        artifact_names = self._required_names(
            required_artifacts,
            protocol_evidence=False,
        )
        evidence_names = self._required_names(
            required_protocol_evidence,
            protocol_evidence=True,
        )
        self.record = self.catalog.complete_run(
            self.run_id,
            required_artifacts=artifact_names,
            required_protocol_evidence=evidence_names,
        )
        return self.record

    def fail(self, reason: str) -> dict[str, object]:
        try:
            current = self.catalog.get(self.run_id)
        except ArtifactCatalogError as exc:
            raise ManagedRunError(str(exc)) from exc
        if current.get("status") != "running":
            return current
        self.record = self.catalog.fail_run(self.run_id, reason=reason)
        return self.record

    def _required_names(
        self,
        paths: Sequence[str | Path],
        *,
        protocol_evidence: bool,
    ) -> list[str]:
        names: list[str] = []
        for value in paths:
            relative = Path(value).as_posix()
            registered = self._registered.get(relative)
            if registered is None:
                raise ManagedRunError(f"Required run file was not registered: {relative}")
            name, observed_evidence = registered
            if observed_evidence is not protocol_evidence:
                expected = "protocol evidence" if protocol_evidence else "artifact"
                raise ManagedRunError(
                    f"Required file {relative} was not classified as {expected}"
                )
            names.append(name)
        return names


def _catalog_name(relative: Path) -> str:
    text = relative.as_posix()
    readable = re.sub(r"[^A-Za-z0-9._:-]+", "-", text).strip("-.")
    readable = readable[:96] or "file"
    suffix = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return f"{readable}-{suffix}"
