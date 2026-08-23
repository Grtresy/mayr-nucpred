"""Verify hashes and deserialize every trusted frozen weight artifact.

Only use this utility with the release archives produced by
``build_release_assets.py``. PyTorch and joblib formats use pickle internally;
the script verifies each member against ``weights/manifest.json`` before it
deserializes the payload.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from collections import Counter
from pathlib import Path

import joblib
import torch


def validate(root: Path) -> Counter[str]:
    manifest = json.loads((root / "weights/manifest.json").read_text(encoding="utf-8"))
    records = manifest["artifacts"]
    archives: dict[str, tarfile.TarFile] = {}
    observed_types: Counter[str] = Counter()
    try:
        for record in records:
            archive_name = record["archive"]
            archive = archives.setdefault(
                archive_name,
                tarfile.open(root / "dist" / archive_name, mode="r:gz"),
            )
            handle = archive.extractfile(record["path"])
            if handle is None:
                raise ValueError(f"Cannot read {record['path']} from {archive_name}")
            payload = handle.read()
            digest = hashlib.sha256(payload).hexdigest()
            if digest != record["sha256"]:
                raise ValueError(f"Hash mismatch before deserialization: {record['path']}")

            stream = io.BytesIO(payload)
            if record["path"].endswith(".pt"):
                value = torch.load(stream, map_location="cpu", weights_only=False)
            elif record["path"].endswith(".joblib"):
                value = joblib.load(stream)
            else:
                raise ValueError(f"Unsupported serialized artifact: {record['path']}")
            observed_types[type(value).__name__] += 1
    finally:
        for archive in archives.values():
            archive.close()
    return observed_types


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()
    observed = validate(args.root.resolve())
    print(f"weight loadability validation: PASS ({sum(observed.values())} artifacts)")
    for name, count in sorted(observed.items()):
        print(f"  {name}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
