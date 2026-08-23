# Licence scope

This repository contains several classes of material with different rights.

| Material | Scope | Licence or rights statement |
| --- | --- | --- |
| Python code and maintainer utilities | `src/`, `scripts/`, `tests/`, and software configuration | MIT License in `LICENSE` |
| Released model artifacts | The 30 binary files enumerated by `weights/manifest.json` and distributed as release assets | Apache License 2.0 in `LICENSES/Apache-2.0.txt` |
| Figure and table Source Data | `results/source_data/` | Author-generated numerical results may be reused with attribution; third-party-derived fields remain governed by their source terms, as stated in `results/source_data/README.md` |
| Third-party datasets, publications, trademarks, and software | Not redistributed unless explicitly identified | Governed by their original terms; no rights are granted here |

Manuscript text and editable figure sources are not part of this release.

The Apache-2.0 grant covers only the author-controlled model artifacts. It does
not grant access to, ownership of, or redistribution rights in the underlying
Mayr, ESNUEL, publisher, or other third-party source materials.

Dependency names in `pyproject.toml` identify separately distributed software.
Their own licences apply when installed.

The frozen xTB runtime is identified in `toolchain/xtb-runtime.json` but its
binary distribution is not included. xTB is separately distributed by its
upstream authors under LGPL-3.0-or-later; see `THIRD_PARTY_NOTICES.md`.
