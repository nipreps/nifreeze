---
title: "Garyfallidis et al. (2014) — Dipy, a library for the analysis of diffusion MRI data"
entity_type: paper
doi: 10.3389/fninf.2014.00008
s2_id: a89d23db5695b754bf735ce7c4a9c538aca0760e
citation_count_at_verify: 1389
source_urls:
  - https://doi.org/10.3389/fninf.2014.00008
  - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3931231/
  - https://docs.dipy.org
last_verified: 2026-07-13
confidence_score: 0.9
---

# Garyfallidis et al. (2014) — Dipy, a library for the analysis of diffusion MRI data

Garyfallidis, E., Brett, M., Amirbekian, B., Rokem, A., van der Walt, S.,
Descoteaux, M., Nimmo-Smith, I. and Dipy Contributors (2014). *Dipy, a library
for the analysis of diffusion MRI data.* Frontiers in Neuroinformatics **8**:8.

## Why relevant

**Claim.** DIPY is a free, open-source, community-maintained Python library that
provides reference implementations of the core diffusion-MRI methods —
reconstruction models (DTI, DKI, q-space methods), gradient/b-value handling,
sphere and direction machinery, and signal simulation — behind a small number of
stable programmatic contracts (notably the `ReconstModel`/`ReconstFit`
fit→predict interface and the `GradientTable` object). It is the canonical
software substrate that nifreeze builds its diffusion models on.

**Evidence.** The paper documents a tested, versioned library with a broad method
suite and an emphasis on interoperability and reproducibility; DIPY has since
become one of the most widely used dMRI toolkits (large citation count, active
development). Its API stability is what allows a downstream project to depend on
`dipy.reconst.*` and `dipy.core.gradients.*` as fixed contracts.

**Location in the project.** nifreeze depends on DIPY as a runtime library
(`pyproject.toml`, pinned to commit `2ecd3655`). The integration surface it
consumes is cached in this wiki as the DIPY *tool* pages
([[tool-dipy]] and its facets: [[tool-dipy-gradient-table]],
[[tool-dipy-reconst-contract]], [[tool-dipy-reconst-models]],
[[tool-dipy-sphere-directions]], [[tool-dipy-sims-voxel]]). This is the software
citation those pages descend from.

**Limits.** A software citation dates quickly: the *paper* describes DIPY circa
2014, but nifreeze pins an unreleased 2025-era commit, so the authoritative
description of the consumed API is the pinned source, not this paper. The tool
pages — not this reference — hold the version-specific signatures, and they carry
`refresh_needed_if` keyed to the pin advancing.
