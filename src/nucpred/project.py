"""Project-local path policy for source data, references, and generated artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


PROJECT_ROOT_ENV = "NUCPRED_PROJECT_ROOT"
ARTIFACT_ROOT_ENV = "NUCPRED_ARTIFACT_ROOT"
REFERENCE_ROOT_ENV = "NUCPRED_REFERENCE_ROOT"


def _configured_path(name: str, default: Path, *, relative_to: Path) -> Path:
    value = os.environ.get(name)
    if not value:
        return default
    path = Path(value).expanduser()
    return path if path.is_absolute() else relative_to / path


@dataclass(frozen=True)
class ProjectLayout:
    """Resolved repository boundaries without depending on the process CWD."""

    root: Path
    artifact_root: Path
    reference_root: Path

    @classmethod
    def discover(cls) -> "ProjectLayout":
        package_root = Path(__file__).resolve().parents[2]
        root = _configured_path(PROJECT_ROOT_ENV, package_root, relative_to=package_root)
        artifact_root = _configured_path(
            ARTIFACT_ROOT_ENV,
            root / "artifacts",
            relative_to=root,
        )
        reference_root = _configured_path(
            REFERENCE_ROOT_ENV,
            root / "reference",
            relative_to=root,
        )
        return cls(
            root=root.resolve(),
            artifact_root=artifact_root.resolve(),
            reference_root=reference_root.resolve(),
        )

    @property
    def data_root(self) -> Path:
        return self.root / "data"

    @property
    def config_root(self) -> Path:
        return self.root / "configs"

    @property
    def docs_root(self) -> Path:
        return self.root / "docs"

    @property
    def reports_root(self) -> Path:
        """Version-controlled, human-readable reports and review material."""
        return self.root / "reports"

    @property
    def frozen_artifact_root(self) -> Path:
        """Project-local pre-migration artifacts with fixed historical identity."""
        return self.root / "artifacts" / "frozen" / "pre-migration"

    def resolve(self, path: str | Path) -> Path:
        candidate = Path(path).expanduser()
        return candidate if candidate.is_absolute() else self.root / candidate

    def data_path(self, *parts: str) -> Path:
        return self.data_root.joinpath(*parts)

    def artifact_path(self, *parts: str) -> Path:
        return self.artifact_root.joinpath(*parts)

    def frozen_artifact_path(self, collection: str, *parts: str) -> Path:
        return self.frozen_artifact_root.joinpath(collection, *parts)

    def reference_path(self, *parts: str) -> Path:
        return self.reference_root.joinpath(*parts)


def get_project_layout() -> ProjectLayout:
    return ProjectLayout.discover()
