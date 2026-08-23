# Frozen model artifacts

The public Git history contains manifests and metadata, not binary checkpoints.
The 30 model artifacts are distributed in two `v1.0.0` release assets:

- `mayr-nucpred-v1.0.0-oof-weights.tar.gz`: 25 OOF artifacts;
- `mayr-nucpred-v1.0.0-deployment-weights.tar.gz`: 5 all-data deployment artifacts.

Extract both archives at the repository root. Archive members retain their
original `artifacts/campaigns/...` locations so the frozen configuration and
runtime registry resolve without path rewriting.

`manifest.json` is authoritative for each artifact's role, outer fold,
initialization seed, byte count, original path, archive, SHA-256 digest, and
Apache-2.0 licence. `SHA256SUMS` contains hashes for the 30 uncompressed binary
members. `dist/SHA256SUMS` records the two archive hashes.

The OOF layer contains five independent outer-fold stacks. Each stack has three
conditional-\(N\) models, one site ranker, and one region residual. The
deployment layer has the same 3 + 1 + 1 composition refitted on all 1,038
corrected-v2 supervised targets.

The release intentionally excludes nested-inner selection checkpoints,
component-ablation checkpoints, and three upstream ESNUEL pretraining
checkpoints. Their identities remain part of the training provenance, while
the public code provides the corresponding training workflows.

## Security

PyTorch `.pt` and joblib files use pickle-based serialization. A malicious file
can execute code when loaded. Verify the full SHA-256 digest before loading any
artifact and obtain archives only from the official GitHub/Zenodo release.

Maintainers can verify every hash and deserialize all 30 trusted artifacts with:

```bash
uv run --no-sync python scripts/validate_weight_loadability.py
```

## Licence

Only the 30 files enumerated in `manifest.json` are licensed here under
Apache-2.0. The grant does not cover datasets or third-party source material.
