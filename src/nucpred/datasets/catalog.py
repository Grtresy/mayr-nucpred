"""Authoritative logical dataset catalog and integrity verifier."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from nucpred.project import ProjectLayout, get_project_layout


CATALOG_SCHEMA = "nucpred.dataset-catalog.v2"
LAYERS = ("raw", "manual", "curated", "processed", "static")
RELATION_TYPES = frozenset(
    {
        "records_from",
        "subset_of",
        "labels_from",
        "curated_from",
        "structure_corrections_from",
        "identity_corrections_from",
    }
)
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")


class DatasetCatalogError(RuntimeError):
    """Raised when dataset identity, lineage, or integrity is invalid."""


class DatasetCatalog:
    def __init__(
        self,
        catalog_path: str | Path | None = None,
        *,
        layout: ProjectLayout | None = None,
    ) -> None:
        self.layout = layout or get_project_layout()
        self.path = (
            self.layout.resolve(catalog_path)
            if catalog_path is not None
            else self.layout.data_path("catalog", "datasets.json")
        )
        self.payload = self._load()
        self.records = self._index_records()
        self._validate_lineage()

    def get(self, dataset_id: str) -> dict[str, Any]:
        try:
            return self.records[dataset_id]
        except KeyError as exc:
            raise DatasetCatalogError(f"Unknown dataset: {dataset_id}") from exc

    def list(self, *, layer: str | None = None) -> list[dict[str, Any]]:
        if layer is not None and layer not in LAYERS:
            raise DatasetCatalogError(f"Unknown dataset layer: {layer}")
        return [
            self.records[identifier]
            for identifier in sorted(self.records)
            if layer is None or self.records[identifier]["layer"] == layer
        ]

    def lineage(self, dataset_id: str) -> dict[str, object]:
        self.get(dataset_id)
        edges: list[dict[str, str]] = []
        visited: set[str] = set()

        def visit(child: str) -> None:
            if child in visited:
                return
            visited.add(child)
            provenance = _mapping(self.records[child], "provenance")
            relations = {
                str(_mapping(value)["dataset_id"]): str(_mapping(value)["type"])
                for value in _sequence(provenance, "relationships")
            }
            for parent in provenance.get("parents", []):
                edges.append(
                    {
                        "parent": str(parent),
                        "child": child,
                        "relation": relations[str(parent)],
                    }
                )
                visit(str(parent))

        visit(dataset_id)
        return {
            "schema_version": "nucpred.dataset-lineage.v1",
            "dataset_id": dataset_id,
            "dataset_ids": sorted(visited),
            "edges": edges,
        }

    def verify(self, dataset_id: str) -> dict[str, object]:
        record = self.get(dataset_id)
        checked = self._verify_asset(_mapping(record, "asset"))
        for component in _sequence(record, "components"):
            checked += self._verify_asset(_mapping(component))
        return {
            "schema_version": "nucpred.dataset-verification.v1",
            "dataset_id": dataset_id,
            "status": "pass",
            "verified_file_count": checked,
        }

    def _load(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise DatasetCatalogError(
                f"Cannot load dataset catalog {self.path}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise DatasetCatalogError("Dataset catalog must contain an object")
        if payload.get("schema_version") != CATALOG_SCHEMA:
            raise DatasetCatalogError(
                f"Unsupported dataset catalog schema: {payload.get('schema_version')!r}"
            )
        if payload.get("authority") != "data/catalog/datasets.json":
            raise DatasetCatalogError("Dataset catalog authority path is not canonical")
        if payload.get("layers") != list(LAYERS):
            raise DatasetCatalogError("Dataset catalog layers are not canonical")
        return payload

    def _index_records(self) -> dict[str, dict[str, Any]]:
        values = self.payload.get("datasets")
        if not isinstance(values, list):
            raise DatasetCatalogError("datasets must be a list")
        records: dict[str, dict[str, Any]] = {}
        for value in values:
            record = _mapping(value)
            identifier = _required_string(record, "id")
            if _IDENTIFIER.fullmatch(identifier) is None:
                raise DatasetCatalogError(f"Invalid dataset id: {identifier!r}")
            if identifier in records:
                raise DatasetCatalogError(f"Duplicate dataset id: {identifier}")
            if record.get("layer") not in LAYERS:
                raise DatasetCatalogError(f"{identifier}: invalid layer")
            if record.get("status") != "active":
                raise DatasetCatalogError(f"{identifier}: unsupported status")
            _mapping(record, "asset")
            _sequence(record, "components")
            scope = _mapping(record, "scientific_scope")
            _required_string(scope, "population")
            _mapping(record, "provenance")
            records[identifier] = record
        return records

    def _validate_lineage(self) -> None:
        layer_index = {name: index for index, name in enumerate(LAYERS)}
        for identifier, record in self.records.items():
            provenance = _mapping(record, "provenance")
            parents = provenance.get("parents")
            if not isinstance(parents, list) or not all(
                isinstance(parent, str) for parent in parents
            ):
                raise DatasetCatalogError(f"{identifier}: parents must be string IDs")
            for parent in parents:
                if parent not in self.records:
                    raise DatasetCatalogError(
                        f"{identifier}: unknown parent {parent!r}"
                    )
                if (
                    layer_index[self.records[parent]["layer"]]
                    > layer_index[record["layer"]]
                ):
                    raise DatasetCatalogError(
                        f"{identifier}: parent {parent!r} is from a later layer"
                    )
            relationships = [
                _mapping(value) for value in _sequence(provenance, "relationships")
            ]
            related_ids: list[str] = []
            for relationship in relationships:
                related_id = _required_string(relationship, "dataset_id")
                relation_type = _required_string(relationship, "type")
                if relation_type not in RELATION_TYPES:
                    raise DatasetCatalogError(
                        f"{identifier}: invalid relationship type {relation_type!r}"
                    )
                related_ids.append(related_id)
            if len(related_ids) != len(set(related_ids)):
                raise DatasetCatalogError(
                    f"{identifier}: relationship dataset IDs must be unique"
                )
            if set(related_ids) != set(parents):
                raise DatasetCatalogError(
                    f"{identifier}: relationships must type every parent exactly once"
                )

        visiting: set[str] = set()
        complete: set[str] = set()

        def visit(identifier: str) -> None:
            if identifier in complete:
                return
            if identifier in visiting:
                raise DatasetCatalogError(f"Dataset lineage cycle at {identifier}")
            visiting.add(identifier)
            for parent in _mapping(self.records[identifier], "provenance")["parents"]:
                visit(str(parent))
            visiting.remove(identifier)
            complete.add(identifier)

        for identifier in self.records:
            visit(identifier)

    def _verify_asset(self, asset: Mapping[str, object]) -> int:
        kind = asset.get("kind")
        relative = Path(_required_string(asset, "path"))
        if relative.is_absolute() or ".." in relative.parts:
            raise DatasetCatalogError(f"Unsafe dataset path: {relative}")
        path = (self.layout.root / relative).resolve()
        try:
            path.relative_to(self.layout.root)
        except ValueError as exc:
            raise DatasetCatalogError(
                f"Dataset path escapes project root: {path}"
            ) from exc
        integrity = _mapping(asset, "integrity")
        if kind == "file":
            if not path.is_file():
                raise DatasetCatalogError(f"Dataset file is missing: {path}")
            _check_size(path.stat().st_size, integrity, path)
            _check_digest(_sha256(path), integrity.get("sha256"), path)
            return 1
        if kind == "directory":
            if asset.get("format") not in {"html-collection", "file-collection"}:
                raise DatasetCatalogError(
                    f"Unsupported directory dataset format: {asset.get('format')!r}"
                )
            if not path.is_dir():
                raise DatasetCatalogError(f"Dataset directory is missing: {path}")
            digest, file_count, total_bytes = _directory_tree_digest(
                path, format_name=str(asset.get("format"))
            )
            if integrity.get("file_count") != file_count:
                raise DatasetCatalogError(f"Dataset file count changed: {path}")
            _check_size(total_bytes, integrity, path)
            _check_digest(digest, integrity.get("tree_sha256"), path)
            return file_count
        raise DatasetCatalogError(f"Unsupported dataset asset kind: {kind!r}")


def _directory_tree_digest(root: Path, *, format_name: str) -> tuple[str, int, int]:
    if format_name == "html-collection":
        files = sorted(root.rglob("*.html"))
    elif format_name == "file-collection":
        files = []
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise DatasetCatalogError(
                    f"Directory dataset contains a symlink: {path}"
                )
            if path.is_dir():
                continue
            if not path.is_file():
                raise DatasetCatalogError(
                    f"Directory dataset contains a special file: {path}"
                )
            files.append(path)
    else:  # guarded by the caller
        raise DatasetCatalogError(
            f"Unsupported directory dataset format: {format_name!r}"
        )
    digest = hashlib.sha256()
    total_bytes = 0
    for path in files:
        total_bytes += path.stat().st_size
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest(), len(files), total_bytes


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_size(actual: int, integrity: Mapping[str, object], path: Path) -> None:
    expected = integrity.get("bytes")
    if (
        isinstance(expected, bool)
        or not isinstance(expected, int)
        or expected != actual
    ):
        raise DatasetCatalogError(
            f"Dataset byte size changed: {path} expected={expected!r} actual={actual}"
        )


def _check_digest(actual: str, expected: object, path: Path) -> None:
    if not isinstance(expected, str) or expected != actual:
        raise DatasetCatalogError(
            f"Dataset digest changed: {path} expected={expected!r} actual={actual}"
        )


def _mapping(value: object, key: str | None = None) -> dict[str, Any]:
    selected = (
        value.get(key) if key is not None and isinstance(value, Mapping) else value
    )
    if not isinstance(selected, dict):
        suffix = "" if key is None else f"[{key!r}]"
        raise DatasetCatalogError(f"Expected mapping{suffix}")
    return selected


def _sequence(value: Mapping[str, object], key: str) -> list[object]:
    selected = value.get(key)
    if not isinstance(selected, list):
        raise DatasetCatalogError(f"{key} must be a list")
    return selected


def _required_string(value: Mapping[str, object], key: str) -> str:
    selected = value.get(key)
    if not isinstance(selected, str) or not selected:
        raise DatasetCatalogError(f"{key} must be a non-empty string")
    return selected


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect the authoritative dataset catalog."
    )
    commands = parser.add_subparsers(dest="action", required=True)
    listing = commands.add_parser("list")
    listing.add_argument("--layer", choices=LAYERS)
    for action in ("show", "verify", "lineage"):
        command = commands.add_parser(action)
        command.add_argument("dataset_id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        catalog = DatasetCatalog()
        if args.action == "list":
            payload: object = catalog.list(layer=args.layer)
        elif args.action == "show":
            payload = catalog.get(args.dataset_id)
        elif args.action == "verify":
            payload = catalog.verify(args.dataset_id)
        else:
            payload = catalog.lineage(args.dataset_id)
    except DatasetCatalogError as exc:
        parser.error(str(exc))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
