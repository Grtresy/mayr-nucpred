# Testing the public release

The repository separates tests that can run from the public checkout from
tests that intentionally exercise curated inputs that are not redistributed.

## Public-checkout checks

After installing the locked development environment, run:

```bash
ruff check scripts/build_release_assets.py \
  scripts/validate_public_release.py \
  scripts/validate_weight_loadability.py
pytest -q tests/test_mayr_final_refit.py
```

Maintainers with the two local release archives under `dist/` should also run:

```bash
python scripts/validate_public_release.py
python scripts/validate_weight_loadability.py
```

The second command deliberately deserializes trusted pickle-based artifacts
only after each archive member matches the published SHA-256 manifest.

## Data-dependent tests

The broader suite includes scientific-contract tests that read the frozen
curated Mayr/ESNUEL Parquet inputs and deeper campaign artifacts. Those files
are not part of the public repository. Such tests are expected to fail with an
explicit missing-file error in a code-only checkout; this is not permission to
replace the missing records with synthetic data.

Researchers who lawfully obtain the inputs can restore their catalogued paths
and run the full suite with `pytest`. See `DATA_AVAILABILITY.md` for the request
policy and `docs/REPRODUCIBILITY.md` for the provenance boundary.
