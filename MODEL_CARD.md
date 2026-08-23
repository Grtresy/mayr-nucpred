# Model Card: mayr-nucpred v1.0.0 release candidate

## Model summary

`mayr-nucpred` is a joint site-ranking and conditional-\(N\) workflow. Given a
molecular structure and supported solvent context, it enumerates typed
nucleophilic-site candidates, ranks them without access to the reported target
site, and predicts the Mayr nucleophilicity parameter \(N\) conditional on each
candidate.

- Developer and primary contributor: Shangkun Shi
- Contact: ssk23@mails.tsinghua.edu.cn
- Campaign: `mayr-n-publication-20260805-v1`
- Dataset lineage: `mayr-site-n-20260805-v2`
- Release status: current frozen campaign weights; not post-review retraining

## Frozen model composition

Each prediction stack contains:

1. three conditional-\(N\) neural models with initialization seeds
   `2026072601`, `2026072602`, and `2026072603`;
2. one deterministic structured site-ranker checkpoint; and
3. one candidate-set-conditioned region residual for delocalized-region
   candidates.

The release provides five connectivity-grouped OOF stacks (25 binary artifacts)
and one all-data deployment stack (5 binary artifacts). Exact paths, sizes,
roles, folds, seeds, and SHA-256 digests are authoritative in
`weights/manifest.json`.

The conditional ensemble reports the mean and population standard deviation
across three all-data initializations. Ranker dispersion is not a calibrated
uncertainty measure. The internal endpoint calibrator is not an absolute
chemical-reactivity probability.

## Training and evaluation populations

The corrected-v2 corpus contains 1,038 supervised context-site targets and 872
connectivity groups. The primary automatic-site evaluation contains 1,026
single-target contexts and 866 connectivity groups. Seventy unresolved targets
remain outside supervision and are not treated as negatives.

Five outer folds are connectivity-disjoint. For every fold, model fitting and
prediction generation do not read that fold's target labels. Labels are joined
only after the score files are frozen. The resulting evidence is retrospective
grouped OOF evaluation, not independent external confirmation.

Current Source Data report:

| Endpoint | Estimate | 95% connectivity-bootstrap interval |
| --- | ---: | ---: |
| Top-1 site accuracy | 0.9308 | 0.9137–0.9470 |
| Top-3 site accuracy | 0.9805 | 0.9715–0.9884 |
| Automatic-site \(N\) MAE | 1.2200 | 1.1290–1.3147 |
| Automatic-site \(N\) \(R^2\) | 0.9374 | 0.9229–0.9494 |
| Known-site \(N\) MAE | 1.1506 | 1.0746–1.2296 |
| Known-site \(N\) \(R^2\) | 0.9502 | 0.9411–0.9580 |

These values come from `results/source_data/figure3_overall_performance.csv`
and must remain bound to that file rather than copied into a new evaluation
claim.

## Intended uses

- reproduce the frozen OOF predictions and paper analyses after obtaining the
  permitted input records;
- run research inference within the documented molecular, solvent, feature,
  and candidate-site contracts;
- inspect or extend site-resolved chemical machine-learning methods; and
- retrain an independently identified model family using the released workflow
  and lawfully obtained data.

## Out-of-scope uses

- safety-critical, clinical, regulatory, or autonomous synthesis decisions;
- claims of reaction feasibility, yield, selectivity, or chemical nonreactivity;
- unsupported solvents, mixtures, charge states, or candidate types;
- treating ranker/calibrator outputs as universal probabilities; or
- describing the retrospective OOF evidence as pristine external validation.

## Limitations

The training corpus reflects the coverage and reporting conventions of the
Mayr literature. Atom endpoints dominate; atom-group and delocalized-region
strata are smaller. Connectivity grouping prevents exact connectivity overlap
but does not eliminate all scaffold similarity. Candidate coverage on the
curated corpus does not guarantee coverage on new chemistry. Uncached inference
also depends on deterministic geometry and xTB feature generation.

## Data and rights

Figure/table Source Data are included in this repository. The underlying curated
Mayr and ESNUEL records are not redistributed in this release because they
include third-party-source material with source-specific reuse conditions. See
`DATA_AVAILABILITY.md`.

## Model licence and loading security

The 30 model artifacts enumerated in `weights/manifest.json` are licensed under
Apache-2.0. This licence does not extend to the underlying third-party data.

The `.pt` and `.joblib` formats can execute code during deserialization. Load
only the official release files after verifying their SHA-256 hashes. Do not
load renamed or third-party-modified copies solely because their filenames
match this release.
