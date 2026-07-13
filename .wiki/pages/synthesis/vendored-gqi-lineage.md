---
title: "The vendored GQI model: lineage and divergence from DIPY"
entity_type: synthesis
last_verified: 2026-07-13
confidence_score: 0.9
derived_from:
  - pages/entity/tool-dipy-reconst-models.md
  - pages/entity/concept-generalized-q-sampling-imaging.md
  - refs/yeh-2010-generalized-q-sampling-imaging.md
---

# The vendored GQI model: lineage and divergence from DIPY

**Technical claim.** nifreeze's GQI predictor is **not** DIPY's — it is a vendored
copy of `dipy.reconst.gqi` (`src/nifreeze/model/gqi.py`, carrying a
statement-of-changes header and the reproduced BSD-3 license), piloting a variant
proposed upstream as [dipy/dipy#3553](https://github.com/dipy/dipy/pull/3553). It
`replicates` DIPY's GQI and `implements` the GQI theory
([[concept-generalized-q-sampling-imaging]], [[yeh-2010-generalized-q-sampling-imaging]]),
but at the pinned commit it has diverged in ways that matter for maintenance.

## Why vendor at all

Upstream DIPY's `GeneralizedQSamplingModel` at commit `2ecd3655` is an
`OdfModel`/`Cache` subclass built for **ODF estimation** (its `fit`/`odf` produce
orientation functions); it has **no `predict` method**. nifreeze needs GQI as a
*signal predictor* inside the leave-one-volume-out loop, i.e. it needs the reverse
map ODF → signal. The vendored copy re-bases the class on a plain `ReconstModel`
and **adds a `predict`** built from a regularised pseudo-inverse of the GQI kernel
(`prediction_kernel`: $\left(K K^{\mathsf T} + \varepsilon I\right)^{-1}K$, with
`INVERSE_LAMBDA = 1e-6`). That extension is what lets GQI satisfy the fit/predict
contract ([[nifreeze-dipy-reconst-contract]]).

## Divergence at the pin

| aspect | upstream `dipy.reconst.gqi` @ `2ecd3655` | vendored `nifreeze.model.gqi` |
|---|---|---|
| base classes | `OdfModel, Cache` | plain `ReconstModel` / `ReconstFit` |
| default `method` | `"gqi2"` | `"standard"` |
| `predict` | absent | present (signal prediction, nifreeze extension) |
| `odf` | `odf(sphere)` — sphere passed per call | `odf()` — sphere stored on the model |
| default sphere | caller-supplied | `create_unit_sphere(recursion_level=5)` (1026 vertices) |

The kernels agree on the core math: `standard` uses
`sinc(⟨b_vector, vertices⟩ · λ / π)` with
`b_vector = bvecs · sqrt(bvals · 0.01506)` and `0.01506 = 6·D_free-water` — the
b-value form of the SDF quadrature (Eq. 9 of the concept page); `gqi2` uses the
`squared_radial_component` basis (the $L^2$-weighted basis, Eq. 8). The **default
method differs** (`"standard"` vendored vs `"gqi2"` upstream), so a naive drop-in
swap would change reconstructions.

## The maintenance hazard

A vendored copy is a **fork frozen at a moment**: upstream fixes and changes to
`dipy.reconst.gqi` do **not** propagate. As DIPY evolves, the copy drifts — and
because nifreeze resolves GQI by the dotted string
`nifreeze.model.gqi.GeneralizedQSamplingModel` ([[tool-dipy-reconst-contract]]),
the drift is invisible until someone diffs the two. On any DIPY pin bump, the GQI
module is the first place to reconcile: check whether upstream gained a `predict`
(making the vendoring unnecessary), changed the kernel scaling constant, or altered
the `method` defaults. This reconciliation is step 2 of the sync procedure in
[[dipy-version-pin-fragility]].
