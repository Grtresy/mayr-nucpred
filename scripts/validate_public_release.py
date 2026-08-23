"""Fail-closed validation for the slim public-release candidate."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import tarfile
from pathlib import Path


ARCHIVES = {
    "mayr-nucpred-v1.0.0-deployment-weights.tar.gz":
        "7471feca4d5b14f1303c42a197646acd2d9ed8d9258c8940f92125be9a640d05",
    "mayr-nucpred-v1.0.0-oof-weights.tar.gz":
        "81f63065318adaafc7f38f1e7c9cd6cd21f7bbdd9a04bc81579aa3b398525eca",
}
FORBIDDEN_SUFFIXES = {
    ".bib", ".doc", ".docx", ".drawio", ".eps", ".jpg", ".jpeg",
    ".pdf", ".png", ".tex", ".tif", ".tiff", ".svg", ".xlsx",
}
FORBIDDEN_PARTS = {
    "manuscript", "portal", "verification", "curation", "frontend",
    "reference", "submissions",
}
IGNORED_TOP_LEVEL = {".git", ".venv", ".pytest_cache", ".ruff_cache", "dist"}
SECRET_PATTERNS = {
    "private key": re.compile(r"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY"),
    "GitHub token": re.compile(r"\bgh[opsu]_[A-Za-z0-9]{30,}\b"),
    "OpenAI-style key": re.compile(r"\bsk-[A-Za-z0-9_-]{24,}\b"),
    "private home path": re.compile(r"/home/[A-Za-z0-9._-]+/"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_scope(root: Path) -> None:
    required = {
        "README.md", "CITATION.cff", "CONTRIBUTORS.md",
        "DATA_AVAILABILITY.md", "LICENSE", "LICENSES/Apache-2.0.txt",
        "LICENSE_SCOPE.md", "MODEL_CARD.md", "SOURCE_SNAPSHOT.json",
        "THIRD_PARTY_NOTICES.md", "pyproject.toml", "uv.lock",
        "toolchain/xtb-runtime.json", "weights/manifest.json",
        "weights/SHA256SUMS", "results/source_data/README.md",
    }
    missing = sorted(item for item in required if not (root / item).is_file())
    if missing:
        raise ValueError(f"Required release files are missing: {missing}")

    offenders: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if relative.parts[0] in IGNORED_TOP_LEVEL:
            continue
        if FORBIDDEN_PARTS.intersection(relative.parts):
            offenders.append(relative.as_posix())
        elif path.suffix.lower() in FORBIDDEN_SUFFIXES:
            offenders.append(relative.as_posix())
        elif path.suffix.lower() in {".parquet", ".pt", ".joblib"}:
            offenders.append(relative.as_posix())
    if offenders:
        raise ValueError(f"Out-of-scope files entered the release: {offenders[:20]}")

    source_data = list((root / "results/source_data").glob("*"))
    csv_count = sum(path.suffix == ".csv" for path in source_data)
    json_count = sum(path.suffix == ".json" for path in source_data)
    if (csv_count, json_count) != (42, 2):
        raise ValueError(
            f"Expected 42 CSV and 2 JSON Source Data files; got {csv_count}, {json_count}"
        )


def check_source_data(root: Path) -> None:
    source_root = root / "results/source_data"
    for manifest_name in ("baseline_comparison_manifest.json", "si_results_manifest.json"):
        manifest = json.loads((source_root / manifest_name).read_text())
        files = manifest.get("files", {})
        if manifest.get("file_count") != len(files):
            raise ValueError(f"Source Data count changed in {manifest_name}")
        for name, record in files.items():
            path = source_root / name
            if record.get("path") != f"results/source_data/{name}":
                raise ValueError(f"Outdated Source Data path in {manifest_name}: {name}")
            if record.get("bytes") != path.stat().st_size or record.get("sha256") != sha256(path):
                raise ValueError(f"Source Data hash changed in {manifest_name}: {name}")


def check_internal_imports(root: Path) -> None:
    source_root = root / "src"
    modules: set[str] = set()
    for path in (source_root / "nucpred").rglob("*.py"):
        parts = list(path.relative_to(source_root).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        modules.add(".".join(parts))
    missing: set[str] = set()
    for path in (source_root / "nucpred").rglob("*.py"):
        current = list(path.relative_to(source_root).with_suffix("").parts)
        if current[-1] == "__init__":
            current.pop()
        else:
            current.pop()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = current[: len(current) - node.level + 1]
                    if node.module:
                        base.extend(node.module.split("."))
                    names = [".".join(base)]
                else:
                    names = [node.module or ""]
            else:
                continue
            for name in names:
                if not name.startswith("nucpred."):
                    continue
                candidate = name
                while candidate and candidate not in modules and candidate.startswith("nucpred."):
                    candidate = candidate.rsplit(".", 1)[0]
                if candidate not in modules:
                    missing.add(name)
    if missing:
        raise ValueError(f"Internal import closure is incomplete: {sorted(missing)}")


def check_weights(root: Path) -> None:
    manifest = json.loads((root / "weights/manifest.json").read_text())
    records = manifest.get("artifacts", [])
    if len(records) != 30 or manifest.get("artifact_count") != 30:
        raise ValueError("Weight manifest must bind exactly 30 artifacts")
    if manifest.get("deployment_artifact_count") != 5:
        raise ValueError("Expected 5 deployment artifacts")
    if manifest.get("oof_artifact_count") != 25:
        raise ValueError("Expected 25 OOF artifacts")
    if len({record["path"] for record in records}) != 30:
        raise ValueError("Weight manifest paths are not unique")
    if any(record.get("license") != "Apache-2.0" for record in records):
        raise ValueError("Every released model artifact must be Apache-2.0")
    for fold in range(5):
        roles = sorted(
            record["role"] for record in records
            if record.get("release_layer") == "oof" and record.get("outer_fold") == fold
        )
        expected = ["conditional_n"] * 3 + ["region_residual", "site_ranker"]
        if roles != sorted(expected):
            raise ValueError(f"Outer fold {fold} has the wrong 3+1+1 composition")

    by_archive: dict[str, dict[str, str]] = {}
    for record in records:
        by_archive.setdefault(record["archive"], {})[record["path"]] = record["sha256"]
    for name, expected_archive_hash in ARCHIVES.items():
        archive_path = root / "dist" / name
        if sha256(archive_path) != expected_archive_hash:
            raise ValueError(f"Release archive hash changed: {name}")
        with tarfile.open(archive_path, "r:gz") as archive:
            for member_name, expected_member_hash in by_archive[name].items():
                handle = archive.extractfile(member_name)
                if handle is None:
                    raise ValueError(f"Archive member is missing: {member_name}")
                if hashlib.sha256(handle.read()).hexdigest() != expected_member_hash:
                    raise ValueError(f"Archive member hash changed: {member_name}")


def check_text(root: Path) -> None:
    findings: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.relative_to(root).parts[0] in IGNORED_TOP_LEVEL:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                findings.append(f"{label} in {path.relative_to(root)}")
    if findings:
        raise ValueError("; ".join(findings))

    xtb = json.loads((root / "toolchain/xtb-runtime.json").read_text())
    if (xtb.get("version"), xtb.get("build_identifier")) != ("6.7.0", "08769fc"):
        raise ValueError("Frozen xTB version/build identity changed")
    if xtb.get("redistribution", {}).get("binary_included") is not False:
        raise ValueError("The third-party xTB binary must not be bundled")


def validate(root: Path) -> None:
    check_scope(root)
    check_source_data(root)
    check_internal_imports(root)
    check_weights(root)
    check_text(root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    validate(args.root.resolve())
    print("public release validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
