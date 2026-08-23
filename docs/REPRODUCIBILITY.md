# Reproducibility guide

This release separates three reproducibility targets that require different
inputs.

## 1. Exact frozen deployment inference

Required:

- the public code checkout;
- `mayr-nucpred-v1.0.0-deployment-weights.tar.gz`;
- supported molecular structures and solvent contexts;
- the frozen feature-generation dependencies, including xTB for uncached inputs.

Extract the deployment asset at the repository root. It restores the original
campaign paths, including the runtime registry and five binary model artifacts.

```bash
tar -xzf dist/mayr-nucpred-v1.0.0-deployment-weights.tar.gz
sha256sum -c weights/SHA256SUMS
```

The deployment stack is the current all-data refit for 1,038 supervised
corrected-v2 targets. It must be identified as
`mayr-n-publication-20260805-v1`, not as a later post-review rerun.

For uncached electronic descriptors, use xTB 6.7.0, build `08769fc`. Verify
both the official Linux x86-64 archive and the extracted executable against
`toolchain/xtb-runtime.json`. The archived protocol used GFN1-xTB on frozen
input geometries without xTB geometry optimization. No separate container
image identifier was captured; the runtime is therefore bound by the printed
build identifier and both SHA-256 digests.

## 2. Exact OOF score regeneration

Required:

- everything required for inference;
- `mayr-nucpred-v1.0.0-oof-weights.tar.gz`;
- lawfully obtained processed model inputs and the frozen outer-fold membership;
- the publication and automatic-site configuration files.

The OOF archive contains five outer-fold stacks. Each stack has three
conditional-\(N\) members, one ranker checkpoint, and one region residual. A
held-out context is scored only by its corresponding outer-fold stack.

The frozen workflow is:

```bash
NCONFIG=configs/mayr_n_publication_v1.toml
SCONFIG=configs/mayr_n_publication_site_v1.toml

uv run --extra train-gpu python -m nucpred.publication.mayr_n_evaluation \
  --config "$NCONFIG" freeze-all

uv run --extra train-gpu python -m nucpred.publication.mayr_site_scoring \
  --config "$SCONFIG" --all
```

Evaluation joins target labels only after every fold's score package is frozen.
Do not describe the pooled result as independent external validation.

## 3. From-scratch training

The code and configuration cover the full pretraining, nested selection,
outer-refit, automatic-site, final-refit, baseline, ablation, and manuscript
aggregation workflow. From-scratch execution additionally requires the curated
Mayr and ESNUEL inputs, which are not redistributed in this repository because
of third-party source-specific reuse conditions.

Researchers who lawfully obtain the required records should preserve the
catalogued paths and verify their manifests before training. Additional
author-generated derived records may be requested through corresponding author
Shaoguang Zhang at `sgzhang@tsinghua.edu.cn`, subject to redistribution rights.
Software-release inquiries may be sent to
`ssk23@mails.tsinghua.edu.cn`.

### Environment

```bash
uv sync --frozen \
  --extra train-gpu \
  --extra dev
```

Formal training targets Linux x86-64, Python 3.13, PyTorch 2.8.0, and the CUDA
12.9 PyG wheel set pinned in `pyproject.toml` and `uv.lock`.

### Standard three-seed pretraining

```bash
DATA=data/processed/esnuel_d_node_xtb_pretraining/esnuel-d-node-xtb-pretraining-20260726-v1-full
OUT=artifacts/campaigns/mayr-explicit-h-node-xtb-pretraining-20260726-v1/full

for SEED in 31001 31002 31003; do
  uv run --extra train-gpu python -m nucpred.training.mayr_node_xtb_pretraining \
    --records "$DATA/records.parquet" \
    --atom-features "$DATA/atom_features.parquet" \
    --molecule-features "$DATA/molecule_features.parquet" \
    --output-checkpoint "$OUT/seed-$SEED/best.pt" \
    --history-json "$OUT/seed-$SEED/history.json" \
    --history-csv "$OUT/seed-$SEED/history.csv" \
    --epochs 60 --min-epochs 10 --patience 8 --min-delta 0.0001 \
    --batch-size 128 --learning-rate 0.0003 --weight-decay 0.0001 \
    --init-seed "$SEED" --device cuda:0 --pilot-max-molecules 0
done
```

### Conditional-N nested evaluation

```bash
NCONFIG=configs/mayr_n_publication_v1.toml

uv run --extra train-gpu python -m nucpred.publication.mayr_n_modeling \
  --config "$NCONFIG" preflight

for OUTER in 0 1 2 3 4; do
  for INNER in 0 1 2 3; do
    uv run --extra train-gpu python -m nucpred.publication.mayr_n_modeling \
      --config "$NCONFIG" inner \
      --outer-fold "$OUTER" --inner-fold "$INNER" --device cuda:0
  done
  uv run --extra train-gpu python -m nucpred.publication.mayr_n_modeling \
    --config "$NCONFIG" select-outer-epochs --outer-fold "$OUTER"
done

for OUTER in 0 1 2 3 4; do
  for SEED in 2026072601 2026072602 2026072603; do
    uv run --extra train-gpu python -m nucpred.publication.mayr_n_outer \
      --config "$NCONFIG" --outer-fold "$OUTER" \
      --initialization-seed "$SEED" --device cuda:0
  done
done
```

### Automatic-site nested evaluation

```bash
SCONFIG=configs/mayr_n_publication_site_v1.toml

uv run --extra train-gpu python -m nucpred.publication.mayr_site_publication \
  --config "$SCONFIG"
uv run --extra train-gpu python -m nucpred.publication.mayr_site_training \
  --config "$SCONFIG" inner-all
uv run --extra train-gpu python -m nucpred.publication.mayr_site_training \
  --config "$SCONFIG" select-all
uv run --extra train-gpu python -m nucpred.publication.mayr_site_training \
  --config "$SCONFIG" outer-all
uv run --extra train-gpu python -m nucpred.publication.mayr_site_scoring \
  --config "$SCONFIG" --all
uv run --extra train-gpu python -m nucpred.publication.mayr_site_evaluation \
  --config "$SCONFIG"
```

### All-data deployment refit

The frozen final-refit implementation validates that all 1,038 supervised
corrected-v2 targets are used and records the resulting 3 + 1 + 1 stack in the
deployment runtime registry. Use the CLI exposed by the publication modules and
retain a new campaign identity for any independently regenerated weights. Never
edit the frozen paper hashes merely to accept a local rerun.

## Figure and table Source Data

Final chart- and table-facing values are versioned in `results/source_data/`.
The manuscript, plotting sources, draw.io files, and generated visual assets
are intentionally outside this software-and-weights release. Some analyses
require deeper per-context artifacts or permitted input records in addition to
the provided Source Data. A missing restricted input must fail explicitly; it
must not be replaced by an invented or silently modified record.
