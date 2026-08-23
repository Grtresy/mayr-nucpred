"""Canonical catalog for managed experiment runs and frozen result trees."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import shutil
import sys
from typing import Any
from uuid import uuid4

from nucpred.core.files import atomic_write_json, sha256_file
from nucpred.project import ProjectLayout, get_project_layout


RUN_MANIFEST_SCHEMA = "nucpred.run-manifest.v1"
RUN_STATUSES = frozenset({"running", "complete", "failed", "archived"})
TERMINAL_STATUSES = frozenset({"complete", "failed", "archived"})
_SLUG_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")


class ArtifactCatalogError(RuntimeError):
    """Raised when a run or artifact would violate the catalog contract."""


class ArtifactCatalog:
    """Store one authoritative manifest per run under the artifact root.

    Managed output files live below ``artifacts/runs``. Historical results are
    indexed in place as frozen storage and are never modified by this class.
    """

    def __init__(self, layout: ProjectLayout | None = None) -> None:
        self.layout = layout or get_project_layout()
        self.catalog_dir = self.layout.artifact_root / "catalog" / "runs"
        self.runs_dir = self.layout.artifact_root / "runs"
        self.archive_dir = self.layout.artifact_root / "archive"

    def create_run(
        self,
        *,
        experiment: str,
        protocol: str,
        dataset_ids: Sequence[str],
        run_id: str | None = None,
        config_path: str | Path | None = None,
        source_paths: Sequence[str | Path] = (),
        command: Sequence[str] = (),
        supersedes: Sequence[str] = (),
        metadata: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        experiment = _slug(experiment, "experiment")
        protocol = _slug(protocol, "protocol")
        identifier = _slug(run_id or _new_run_id(), "run_id")
        datasets = _unique_tokens(dataset_ids, "dataset_ids")
        predecessors = _unique_tokens(supersedes, "supersedes", allow_empty=True)
        if identifier in predecessors:
            raise ArtifactCatalogError("A run cannot supersede itself")
        self._require_terminal_predecessors(predecessors)

        manifest_path = self._manifest_path(identifier)
        if manifest_path.exists():
            raise ArtifactCatalogError(f"Run already exists: {identifier}")
        run_dir = self.runs_dir / experiment / identifier
        if run_dir.exists():
            raise ArtifactCatalogError(f"Run directory already exists: {run_dir}")
        run_dir.mkdir(parents=True)

        try:
            now = _utc_now()
            config = (
                self._project_file_record(config_path, role="config")
                if config_path is not None
                else None
            )
            sources = {
                record["path"]: record
                for record in (
                    self._project_file_record(path, role="source")
                    for path in source_paths
                )
            }
            record: dict[str, object] = {
                "schema_version": RUN_MANIFEST_SCHEMA,
                "run_id": identifier,
                "experiment": experiment,
                "protocol": protocol,
                "dataset_ids": datasets,
                "status": "running",
                "created_at_utc": now,
                "updated_at_utc": now,
                "storage": {
                    "kind": "managed",
                    "root": "artifact",
                    "path": run_dir.relative_to(self.layout.artifact_root).as_posix(),
                    "read_only": False,
                    "snapshot": None,
                },
                "provenance": {
                    "command": [str(value) for value in command],
                    "config": config,
                    "sources": sources,
                    "runtime": {
                        "python": platform.python_version(),
                        "python_executable": sys.executable,
                        "platform": platform.platform(),
                    },
                    "metadata": dict(metadata or {}),
                },
                "artifacts": {},
                "protocol_evidence": {},
                "completion": {
                    "status": "pending",
                    "required_artifacts": [],
                    "required_protocol_evidence": [],
                },
                "supersedes": predecessors,
                "archive": None,
            }
            self._save(record)
        except Exception:
            if run_dir.exists() and not any(run_dir.iterdir()):
                run_dir.rmdir()
            raise
        return record

    def import_frozen_run(
        self,
        *,
        run_id: str,
        experiment: str,
        protocol: str,
        dataset_ids: Sequence[str],
        source_dir: str | Path,
        artifacts: Mapping[str, str | Path],
        protocol_evidence: Mapping[str, str | Path] | None = None,
        supersedes: Sequence[str] = (),
        reason: str = "pre-migration frozen scientific result",
    ) -> dict[str, object]:
        identifier = _slug(run_id, "run_id")
        experiment = _slug(experiment, "experiment")
        protocol = _slug(protocol, "protocol")
        datasets = _unique_tokens(dataset_ids, "dataset_ids")
        predecessors = _unique_tokens(supersedes, "supersedes", allow_empty=True)
        self._require_terminal_predecessors(predecessors)
        if self._manifest_path(identifier).exists():
            raise ArtifactCatalogError(f"Run already exists: {identifier}")
        if not artifacts:
            raise ArtifactCatalogError("A frozen run must declare at least one artifact")

        frozen_root = self._project_directory(source_dir)
        frozen_snapshot = _directory_snapshot(frozen_root)
        artifact_records = {
            _slug(name, "artifact name"): _file_record(
                frozen_root / path,
                base=frozen_root,
                role="artifact",
            )
            for name, path in artifacts.items()
        }
        evidence_records = {
            _slug(name, "protocol evidence name"): _file_record(
                frozen_root / path,
                base=frozen_root,
                role="protocol_evidence",
            )
            for name, path in (protocol_evidence or {}).items()
        }
        now = _utc_now()
        record: dict[str, object] = {
            "schema_version": RUN_MANIFEST_SCHEMA,
            "run_id": identifier,
            "experiment": experiment,
            "protocol": protocol,
            "dataset_ids": datasets,
            "status": "archived",
            "created_at_utc": now,
            "updated_at_utc": now,
            "storage": {
                "kind": "frozen",
                "root": "project",
                "path": frozen_root.relative_to(self.layout.root).as_posix(),
                "read_only": True,
                "snapshot": frozen_snapshot,
            },
            "provenance": {
                "command": [],
                "config": None,
                "sources": {},
                "runtime": {},
                "metadata": {"imported_from_existing_result": True},
            },
            "artifacts": artifact_records,
            "protocol_evidence": evidence_records,
            "completion": {
                "status": "complete",
                "completed_at_utc": None,
                "verified_at_utc": now,
                "required_artifacts": sorted(artifact_records),
                "required_protocol_evidence": sorted(evidence_records),
                "verified_tree_file_count": frozen_snapshot["file_count"],
            },
            "supersedes": predecessors,
            "archive": {
                "archived_at_utc": now,
                "reason": _nonempty(reason, "archive reason"),
                "source_is_read_only": True,
            },
        }
        self._save(record)
        return record

    def add_file(
        self,
        run_id: str,
        *,
        name: str,
        path: str | Path,
        role: str,
        protocol_evidence: bool = False,
        media_type: str | None = None,
    ) -> dict[str, object]:
        record = self.get(run_id)
        self._require_running_managed(record)
        collection_name = "protocol_evidence" if protocol_evidence else "artifacts"
        collection = _mapping(record, collection_name)
        key = _slug(name, f"{collection_name} name")
        if key in collection:
            raise ArtifactCatalogError(f"{collection_name} already contains {key!r}")
        file_record = _file_record(
            path,
            base=self._storage_directory(record),
            role=_nonempty(role, "role"),
        )
        if media_type is not None:
            file_record["media_type"] = _nonempty(media_type, "media_type")
        collection[key] = file_record
        self._save(record)
        return record

    def complete_run(
        self,
        run_id: str,
        *,
        required_artifacts: Sequence[str],
        required_protocol_evidence: Sequence[str] = (),
    ) -> dict[str, object]:
        record = self.get(run_id)
        self._require_running_managed(record)
        required_outputs = _unique_tokens(required_artifacts, "required_artifacts")
        required_evidence = _unique_tokens(
            required_protocol_evidence,
            "required_protocol_evidence",
            allow_empty=True,
        )
        artifacts = _mapping(record, "artifacts")
        evidence = _mapping(record, "protocol_evidence")
        _require_keys(artifacts, required_outputs, "artifacts")
        _require_keys(evidence, required_evidence, "protocol evidence")
        verification = self._verify_record(record)
        now = _utc_now()
        record["status"] = "complete"
        record["completion"] = {
            "status": "complete",
            "completed_at_utc": now,
            "verified_at_utc": now,
            "required_artifacts": required_outputs,
            "required_protocol_evidence": required_evidence,
            "verified_file_count": verification["verified_file_count"],
        }
        self._save(record)
        return record

    def fail_run(self, run_id: str, *, reason: str) -> dict[str, object]:
        record = self.get(run_id)
        self._require_running_managed(record)
        now = _utc_now()
        record["status"] = "failed"
        record["completion"] = {
            "status": "failed",
            "failed_at_utc": now,
            "reason": _nonempty(reason, "failure reason"),
            "required_artifacts": [],
            "required_protocol_evidence": [],
        }
        self._save(record)
        return record

    def reopen_failed_run(
        self,
        run_id: str,
        *,
        reason: str,
    ) -> dict[str, object]:
        """Reopen an unregistered failed managed run after provenance verification."""

        record = self.get(run_id)
        if record.get("status") != "failed":
            raise ArtifactCatalogError("Only failed managed runs can be reopened")
        storage = _mapping(record, "storage")
        if storage.get("kind") != "managed" or storage.get("read_only") is not False:
            raise ArtifactCatalogError("Frozen or archived run storage cannot be reopened")
        if _mapping(record, "artifacts") or _mapping(record, "protocol_evidence"):
            raise ArtifactCatalogError(
                "A failed run with registered files cannot be reopened in place"
            )
        verification = self._verify_record(record)
        now = _utc_now()
        history = record.setdefault("recovery_history", [])
        if not isinstance(history, list):
            raise ArtifactCatalogError("Run recovery_history must be a list")
        history.append(
            {
                "failed_completion": dict(_mapping(record, "completion")),
                "reason": _nonempty(reason, "reopen reason"),
                "reopened_at_utc": now,
                "verified_file_count": verification["verified_file_count"],
            }
        )
        record["status"] = "running"
        record["completion"] = {
            "status": "pending",
            "reopened_at_utc": now,
            "reopen_reason": _nonempty(reason, "reopen reason"),
            "required_artifacts": [],
            "required_protocol_evidence": [],
        }
        self._save(record)
        return record

    def archive_run(self, run_id: str, *, reason: str) -> dict[str, object]:
        record = self.get(run_id)
        storage = _mapping(record, "storage")
        if storage.get("kind") == "frozen":
            raise ArtifactCatalogError("Frozen historical runs are already read-only")
        if record.get("status") not in {"complete", "failed"}:
            raise ArtifactCatalogError("Only complete or failed managed runs can be archived")
        source = self._storage_directory(record)
        destination = self.archive_dir / str(record["experiment"]) / str(record["run_id"])
        if destination.exists():
            raise ArtifactCatalogError(f"Archive destination already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(source.as_posix(), destination.as_posix())
        now = _utc_now()
        record["status"] = "archived"
        record["storage"] = {
            "kind": "managed_archive",
            "root": "artifact",
            "path": destination.relative_to(self.layout.artifact_root).as_posix(),
            "read_only": True,
            "snapshot": _directory_snapshot(destination),
        }
        record["archive"] = {
            "archived_at_utc": now,
            "reason": _nonempty(reason, "archive reason"),
            "source_is_read_only": True,
        }
        self._save(record)
        return record

    def get(self, run_id: str) -> dict[str, object]:
        path = self._manifest_path(_slug(run_id, "run_id"))
        if not path.is_file():
            raise ArtifactCatalogError(f"Unknown run: {run_id}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ArtifactCatalogError(f"Cannot read run manifest {path}: {exc}") from exc
        record = _mapping(payload)
        self._validate_record(record)
        return record

    def list_runs(
        self,
        *,
        experiment: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, object]]:
        if status is not None and status not in RUN_STATUSES:
            raise ArtifactCatalogError(f"Unsupported run status: {status}")
        records = []
        if not self.catalog_dir.exists():
            return records
        for path in sorted(self.catalog_dir.glob("*.json")):
            record = self.get(path.stem)
            if experiment is not None and record["experiment"] != experiment:
                continue
            if status is not None and record["status"] != status:
                continue
            records.append(record)
        return records

    def verify(self, run_id: str) -> dict[str, object]:
        return self._verify_record(self.get(run_id))

    def run_directory(self, run_id: str) -> Path:
        return self._storage_directory(self.get(run_id))

    def _project_file_record(self, path: str | Path, *, role: str) -> dict[str, object]:
        resolved = self.layout.resolve(path)
        return _file_record(resolved, base=self.layout.root, role=role)

    def _project_directory(self, path: str | Path) -> Path:
        resolved = self.layout.resolve(path).resolve()
        _relative_to(resolved, self.layout.root, "frozen source directory")
        if not resolved.is_dir():
            raise ArtifactCatalogError(f"Frozen source directory is missing: {resolved}")
        return resolved

    def _storage_directory(self, record: Mapping[str, object]) -> Path:
        storage = _mapping(record, "storage")
        root_name = storage.get("root")
        if root_name == "artifact":
            root = self.layout.artifact_root
        elif root_name == "project":
            root = self.layout.root
        else:
            raise ArtifactCatalogError(f"Unsupported storage root: {root_name!r}")
        path = root / _required_string(storage, "path")
        resolved = path.resolve()
        _relative_to(resolved, root, "storage path")
        if not resolved.is_dir():
            raise ArtifactCatalogError(f"Run storage directory is missing: {resolved}")
        return resolved

    def _verify_record(self, record: Mapping[str, object]) -> dict[str, object]:
        self._validate_record(record)
        storage_dir = self._storage_directory(record)
        storage = _mapping(record, "storage")
        tree_file_count = 0
        if storage.get("kind") in {"frozen", "managed_archive"}:
            expected_snapshot = _mapping(storage, "snapshot")
            actual_snapshot = _directory_snapshot(storage_dir)
            if actual_snapshot != expected_snapshot:
                raise ArtifactCatalogError(
                    f"Read-only directory snapshot changed: {storage_dir}"
                )
            tree_file_count = int(actual_snapshot["file_count"])
        verified = 0
        for collection_name in ("artifacts", "protocol_evidence"):
            for file_record in _mapping(record, collection_name).values():
                _verify_file_record(_mapping(file_record), base=storage_dir)
                verified += 1
        provenance = _mapping(record, "provenance")
        config = provenance.get("config")
        if config is not None:
            _verify_file_record(_mapping(config), base=self.layout.root)
            verified += 1
        for source in _mapping(provenance, "sources").values():
            _verify_file_record(_mapping(source), base=self.layout.root)
            verified += 1
        return {
            "schema_version": "nucpred.run-verification.v1",
            "run_id": record["run_id"],
            "status": "pass",
            "run_status": record["status"],
            "verified_file_count": verified,
            "verified_tree_file_count": tree_file_count,
            "verified_at_utc": _utc_now(),
        }

    def _validate_record(self, record: Mapping[str, object]) -> None:
        if record.get("schema_version") != RUN_MANIFEST_SCHEMA:
            raise ArtifactCatalogError(
                f"Unsupported run manifest schema: {record.get('schema_version')!r}"
            )
        _slug(_required_string(record, "run_id"), "run_id")
        _slug(_required_string(record, "experiment"), "experiment")
        _slug(_required_string(record, "protocol"), "protocol")
        status = record.get("status")
        if status not in RUN_STATUSES:
            raise ArtifactCatalogError(f"Unsupported run status: {status!r}")
        storage = _mapping(record, "storage")
        kind = storage.get("kind")
        snapshot = storage.get("snapshot")
        if kind in {"frozen", "managed_archive"}:
            _mapping(storage, "snapshot")
        elif kind == "managed" and snapshot is not None:
            raise ArtifactCatalogError("A mutable managed run cannot have a tree snapshot")
        _mapping(record, "provenance")
        _mapping(record, "artifacts")
        _mapping(record, "protocol_evidence")
        _mapping(record, "completion")

    def _require_running_managed(self, record: Mapping[str, object]) -> None:
        if record.get("status") != "running":
            raise ArtifactCatalogError("Only running runs can register files or finish")
        storage = _mapping(record, "storage")
        if storage.get("kind") != "managed" or storage.get("read_only") is not False:
            raise ArtifactCatalogError("Frozen or archived run storage cannot be mutated")

    def _require_terminal_predecessors(self, run_ids: Sequence[str]) -> None:
        for run_id in run_ids:
            predecessor = self.get(run_id)
            if predecessor.get("status") not in TERMINAL_STATUSES:
                raise ArtifactCatalogError(
                    f"Cannot supersede non-terminal run {run_id!r}"
                )

    def _manifest_path(self, run_id: str) -> Path:
        return self.catalog_dir / f"{run_id}.json"

    def _save(self, record: dict[str, object]) -> None:
        self._validate_record(record)
        record["updated_at_utc"] = _utc_now()
        atomic_write_json(self._manifest_path(str(record["run_id"])), record)


def _new_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid4().hex[:8]}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(value: str, label: str) -> str:
    if not isinstance(value, str) or _SLUG_PATTERN.fullmatch(value) is None:
        raise ArtifactCatalogError(
            f"{label} must match {_SLUG_PATTERN.pattern!r}: {value!r}"
        )
    return value


def _nonempty(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ArtifactCatalogError(f"{label} must be a non-empty string")
    return value.strip()


def _unique_tokens(
    values: Sequence[str],
    label: str,
    *,
    allow_empty: bool = False,
) -> list[str]:
    tokens = list(dict.fromkeys(_slug(value, label) for value in values))
    if not tokens and not allow_empty:
        raise ArtifactCatalogError(f"{label} must contain at least one identifier")
    return tokens


def _require_keys(
    collection: Mapping[str, object],
    required: Sequence[str],
    label: str,
) -> None:
    missing = sorted(set(required).difference(collection))
    if missing:
        raise ArtifactCatalogError(f"Required {label} are missing: {missing}")


def _file_record(
    path: str | Path,
    *,
    base: Path,
    role: str,
) -> dict[str, object]:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = base / resolved
    resolved = resolved.resolve()
    relative = _relative_to(resolved, base, "file")
    if not resolved.is_file():
        raise ArtifactCatalogError(f"Artifact is missing or not a file: {resolved}")
    return {
        "path": relative.as_posix(),
        "role": role,
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _verify_file_record(record: Mapping[str, object], *, base: Path) -> None:
    path = base / _required_string(record, "path")
    resolved = path.resolve()
    _relative_to(resolved, base, "recorded file")
    if not resolved.is_file():
        raise ArtifactCatalogError(f"Recorded file is missing: {resolved}")
    expected_bytes = record.get("bytes")
    if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int):
        raise ArtifactCatalogError(f"Recorded byte count is invalid: {expected_bytes!r}")
    if resolved.stat().st_size != expected_bytes:
        raise ArtifactCatalogError(f"Recorded file size changed: {resolved}")
    if sha256_file(resolved) != _required_string(record, "sha256"):
        raise ArtifactCatalogError(f"Recorded file hash changed: {resolved}")


def _directory_snapshot(root: Path) -> dict[str, object]:
    """Hash every directory entry in a self-contained read-only result tree."""

    resolved_root = root.resolve()
    digest = hashlib.sha256()
    file_count = 0
    directory_count = 0
    total_bytes = 0
    entries = sorted(
        resolved_root.rglob("*"),
        key=lambda path: path.relative_to(resolved_root).as_posix(),
    )
    for path in entries:
        relative = path.relative_to(resolved_root).as_posix()
        if path.is_symlink():
            raise ArtifactCatalogError(
                f"Read-only result trees cannot contain symlinks: {path}"
            )
        if path.is_dir():
            directory_count += 1
            descriptor: list[object] = ["directory", relative]
        elif path.is_file():
            size = path.stat().st_size
            file_count += 1
            total_bytes += size
            descriptor = ["file", relative, size, sha256_file(path)]
        else:
            raise ArtifactCatalogError(
                f"Read-only result trees cannot contain special files: {path}"
            )
        digest.update(
            (json.dumps(descriptor, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
                "utf-8"
            )
        )
    return {
        "schema_version": "nucpred.read-only-tree-snapshot.v1",
        "file_count": file_count,
        "directory_count": directory_count,
        "total_bytes": total_bytes,
        "tree_sha256": digest.hexdigest(),
    }


def _relative_to(path: Path, base: Path, label: str) -> Path:
    try:
        return path.relative_to(base.resolve())
    except ValueError as exc:
        raise ArtifactCatalogError(f"{label} escapes its declared root: {path}") from exc


def _mapping(value: object, key: str | None = None) -> dict[str, Any]:
    selected = value.get(key) if key is not None and isinstance(value, Mapping) else value
    if not isinstance(selected, dict):
        suffix = "" if key is None else f"[{key!r}]"
        raise ArtifactCatalogError(f"Expected mapping{suffix}")
    return selected


def _required_string(record: Mapping[str, object], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise ArtifactCatalogError(f"{key} must be a non-empty string")
    return value
