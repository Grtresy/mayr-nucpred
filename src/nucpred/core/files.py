"""File integrity and atomic persistence primitives.

This module deliberately has no scientific, dataframe, or training
dependencies. Dataset, experiment, evaluation, and reporting code should use
these primitives directly rather than growing subsystem-local copies.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4


DEFAULT_HASH_BLOCK_SIZE = 1024 * 1024


def _json_scalar_default(value: object) -> object:
    """Convert scalar-like third-party values to JSON-native primitives."""

    item = getattr(value, "item", None)
    if callable(item) and getattr(value, "ndim", 0) == 0:
        scalar = item()
        if isinstance(scalar, str | int | float | bool) or scalar is None:
            return scalar
    raise TypeError(
        f"Object of type {value.__class__.__name__} is not JSON serializable"
    )


def sha256_file(
    path: str | Path,
    *,
    block_size: int = DEFAULT_HASH_BLOCK_SIZE,
) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""

    if block_size < 1:
        raise ValueError("block_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(
    path: str | Path,
    payload: object,
    *,
    ensure_ascii: bool = True,
    indent: int | None = 2,
    sort_keys: bool = True,
) -> None:
    """Durably replace a JSON file using a unique same-directory temporary."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=ensure_ascii,
                indent=indent,
                sort_keys=sort_keys,
                default=_json_scalar_default,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
