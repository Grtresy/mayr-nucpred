# mayr-nucpred

Research code, final figure/table Source Data, and frozen model artifacts for
joint nucleophilic-site ranking and conditional prediction of the Mayr
nucleophilicity parameter, \(N\).

Shangkun Shi is the lead developer, release maintainer, and primary software
contributor. The archive creators are Shangkun Shi, Junliang He, Zikang Dong,
and Shaoguang Zhang; Shaoguang Zhang is the corresponding author.

## Frozen release

Version `v1.0.0` contains 30 binary model artifacts from campaign
`mayr-n-publication-20260805-v1`:

| Layer | Composition | Files |
| --- | --- | ---: |
| Five-fold OOF | Five outer folds, each with 3 conditional-\(N\) models + 1 site ranker + 1 region residual | 25 |
| Deployment | One all-data refit with the same 3 + 1 + 1 structure | 5 |

The deployment stack was refitted on all 1,038 corrected-v2 supervised targets.
Nested-inner, ablation, and upstream pretraining checkpoints are not released.

## Deliberately narrow repository scope

- `src/nucpred/`: the dependency-closed paper training, evaluation, final-refit,
  and frozen-inference implementation.
- `configs/`: frozen primary, baseline, and matched-ablation configurations.
- `results/source_data/`: final chart- and table-facing numerical Source Data.
- `weights/`: the 30-file manifest, member hashes, and extraction guidance.
- `toolchain/xtb-runtime.json`: the archived xTB version/build identity.

This repository does **not** contain the manuscript, LaTeX, BibTeX, draw.io
sources, generated figures, PDFs, submission files, review workflows, portals,
literature-curation tools, or the underlying curated Mayr/ESNUEL datasets.

## Installation

The frozen environment uses Python 3.13 and `uv`.

```bash
uv sync --frozen --extra train-gpu --extra dev
```

The formal GPU setup uses PyTorch 2.8.0 and the CUDA 12.9 PyG wheels recorded
in `pyproject.toml` and `uv.lock`.

## Frozen weights

The Git history contains weight metadata, not checkpoint binaries. The matching
release carries:

```text
mayr-nucpred-v1.0.0-deployment-weights.tar.gz
mayr-nucpred-v1.0.0-oof-weights.tar.gz
```

Verify the release archive checksums, extract at the repository root, and then
verify the 30 member hashes in `weights/SHA256SUMS`. Only deserialize the
pickle-based `.pt` and `.joblib` files after verification.

## Reproducibility and data boundary

See `docs/REPRODUCIBILITY.md` for frozen inference, OOF regeneration, and
from-scratch training. Figure/table Source Data are under
`results/source_data/`. Grouped OOF results are retrospective model-evaluation
evidence, not independent external confirmation.

The curated training datasets are not redistributed because their records
derive from third-party literature and database sources with source-specific
reuse conditions. Additional redistributable author-derived records may be
requested from corresponding author Shaoguang Zhang
(`sgzhang@tsinghua.edu.cn`) for research and reproducibility, subject to the
underlying rights. See `DATA_AVAILABILITY.md`.

The reserved Zenodo DOI is
[`10.5281/zenodo.22059011`](https://doi.org/10.5281/zenodo.22059011). A saved
Zenodo draft is not a published archive; verify that the DOI resolves before
citing it as publicly available.

## Licences

- Code and software configuration: MIT (`LICENSE`).
- The 30 model artifacts: Apache-2.0 (`LICENSES/Apache-2.0.txt`).
- Source Data and third-party-derived fields: `LICENSE_SCOPE.md` and
  `results/source_data/README.md`.
