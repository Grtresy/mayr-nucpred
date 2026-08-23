# Third-party notices

## xTB

Electronic descriptors in the frozen workflow were generated with GFN1-xTB
using **xTB 6.7.0 (build `08769fc`)** from the official Linux x86-64 binary
distribution. Exact archive and executable checksums, plus official retrieval
links, are recorded in `toolchain/xtb-runtime.json`.

xTB is developed by the Grimme group and distributed separately under
LGPL-3.0-or-later. The xTB archive and executable are **not** included in this
repository or its model-weight archives, and are not covered by this
repository's MIT or Apache-2.0 licences. Obtain xTB from its official release
page and verify the checksum before use.

- Official project: <https://github.com/grimme-lab/xtb>
- Frozen release: <https://github.com/grimme-lab/xtb/releases/tag/v6.7.0>

Other dependencies named in `pyproject.toml` and `uv.lock` remain governed by
their respective upstream licences.
