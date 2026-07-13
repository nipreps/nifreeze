---
title: "DIPY — nifreeze's diffusion-MRI dependency (integration surface)"
entity_type: tool
last_verified: 2026-07-13
confidence_score: 0.9
refresh_needed_if: "DIPY pin in pyproject.toml advances past 2ecd3655"
source_urls:
  - https://github.com/dipy/dipy/tree/2ecd3655
  - https://docs.dipy.org
---

# DIPY — nifreeze's diffusion-MRI dependency

[DIPY](https://dipy.org) ([[garyfallidis-2014-dipy]]) is the open-source Python
library nifreeze builds its diffusion-MRI models on. This is the **umbrella tool
page**: it records *what nifreeze consumes from DIPY and at which pinned version*,
and anchors the per-area facet pages. It documents the **integration surface**, not
DIPY's internals and not nifreeze's own wrapping code (see the schema's
dependency-integration scope clause).

## Pinned version

nifreeze pins DIPY to an **unreleased git commit**, not a PyPI release
(`pyproject.toml`):

```
dipy @ git+https://github.com/dipy/dipy.git@2ecd3655
```

Every signature/behaviour on the facet pages describes DIPY **at commit
`2ecd3655`**. When that pin moves, this page and its facets go stale by design;
the sync procedure lives in [[dipy-version-pin-fragility]].

## The surface nifreeze imports

Grouped by DIPY area; each area has a facet page transcribing the exact contract.

| DIPY area | Symbols nifreeze imports | Facet page |
|---|---|---|
| `dipy.core.gradients` | `GradientTable`, `gradient_table`, `gradient_table_from_bvals_bvecs`, `check_multi_b`, `unique_bvals_magnitude` | [[tool-dipy-gradient-table]] |
| `dipy.reconst.base` | `ReconstModel`, `ReconstFit` | [[tool-dipy-reconst-contract]] |
| `dipy.reconst.{dti,dki,gqi}` | `TensorModel`, `DiffusionKurtosisModel`, `GeneralizedQSamplingModel` | [[tool-dipy-reconst-models]] |
| `dipy.core.sphere` / `dipy.core.subdivide_octahedron` / `dipy.core.geometry` | `Sphere`, `HemiSphere`, `disperse_charges`, `create_unit_sphere`, `sphere2cart`, `normalized_vector` | [[tool-dipy-sphere-directions]] |
| `dipy.sims.voxel` | `single_tensor`, `multi_tensor`, `all_tensor_evecs` | [[tool-dipy-sims-voxel]] |

**Production dependencies** (on the estimator's path): `dipy.core.gradients`,
`dipy.reconst.base`, and the reconstruction models. The single most load-bearing
object is the `GradientTable`, rebuilt on every leave-one-volume-out fit and
per-volume prediction (see [[gradient-table-interop-hotpath]]).

**Test/simulation-only surface** (not on the correction path, so not given facet
pages): `dipy.io.gradients.read_bvals_bvecs`, `dipy.data.read_stanford_hardi`,
`dipy.segment.mask.median_otsu`. The `dipy.sims.voxel` and sphere/direction
machinery are used to synthesize ground-truth phantoms in
`src/nifreeze/testing/simulations.py`.

## What nifreeze does NOT use

nifreeze does **not** import DIPY's spherical-harmonics machinery
(`dipy.reconst.shm`): its vendored GQI model reconstructs with an explicit
$\operatorname{sinc}$ quadrature over sphere vertices, not an SH basis. It also
does not use DIPY's registration, denoising, or tractography subsystems (those are
candidate forward-looking areas, not part of the current cache).

## Licensing and vendoring

DIPY is BSD-3-Clause. nifreeze **vendors** one DIPY module: `dipy.reconst.gqi`
is copied into `src/nifreeze/model/gqi.py` (with a statement-of-changes header and
the reproduced BSD-3 license), piloting a variant proposed upstream as
[dipy/dipy#3553](https://github.com/dipy/dipy/pull/3553). Consequently the model
nifreeze resolves for GQI is `nifreeze.model.gqi.GeneralizedQSamplingModel` (the
vendored copy), **not** `dipy.reconst.gqi` — the divergence risk this creates is
tracked in [[vendored-gqi-lineage]]. The other models (`TensorModel`,
`DiffusionKurtosisModel`) are used from DIPY directly (DKI via a thin subclass in
`src/nifreeze/model/dki.py`).

## Relation to the theory pages

The DIPY tool pages `implement` the algorithm theory this wiki already grounds:
[[concept-diffusion-tensor-imaging]], [[concept-diffusion-kurtosis-imaging]],
[[concept-generalized-q-sampling-imaging]],
[[concept-sphere-sampling-electrostatic-repulsion]],
[[concept-multi-tensor-signal-simulation]], and the acquisition physics in
[[concept-diffusion-mri-signal]]. Read a facet page for *what the code contract is*;
read the linked concept page for *why the algorithm is what it is*.
