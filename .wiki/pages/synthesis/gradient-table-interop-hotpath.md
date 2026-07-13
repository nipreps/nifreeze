---
title: "The GradientTable rebuild is nifreeze's DIPY interop seam and a hot path"
entity_type: synthesis
last_verified: 2026-07-13
confidence_score: 0.9
derived_from:
  - pages/entity/tool-dipy-gradient-table.md
  - pages/entity/concept-leave-one-volume-out.md
  - pages/entity/concept-diffusion-mri-signal.md
---

# The GradientTable rebuild is nifreeze's DIPY interop seam and a hot path

**Technical claim.** nifreeze keeps its gradients in a lightweight internal
`(N, 4)` array (b-vectors in columns `0:3`, b-values in column `3`) and converts to
a DIPY `GradientTable` **only at the boundary** where it calls a DIPY model. Because
leave-one-volume-out ([[concept-leave-one-volume-out]]) fits and predicts once *per
volume*, this conversion sits on the estimator's hot path — it is both the
interoperability seam with DIPY and a recurring per-iteration cost.

## Where the rebuild happens

Every conversion is a call to `gradient_table_from_bvals_bvecs`
([[tool-dipy-gradient-table]]):

1. **At each fit** — `BaseDWIModel._fit` rebuilds a `GradientTable` from the
   training subset's `(N, 4)` slice before instantiating the model.
2. **At each prediction** — `fit_predict` builds a **one-row** gradient table for
   the held-out volume's single $(\mathbf{g}_k, b_k)$ and passes it to
   `fit.predict`.
3. **For the b0 append (DKI)** — `_append_bzero` concatenates a $b=0$ row and
   rebuilds the table, because DIPY's DKI fit assumes a b0 reference is present
   (see [[concept-diffusion-kurtosis-imaging]]).

Over a full pass the model is fit/predicted for all $N$ volumes, so the round-trip
recurs $\mathcal{O}(N)$ times (more with voxel-chunk parallelism).

## Why the internal array exists at all

The internal `(N, 4)` representation lets nifreeze slice, mask, and split the
gradient scheme with plain NumPy (train/test splitting, shell selection) without
DIPY's validation overhead — and keeps the *data* layer independent of the *model*
layer's dependency. DIPY's `GradientTable` is richer (it derives `b0s_mask`,
`bvals`, `bvecs` from scaled gradients, and can carry acquisition timings), but
nifreeze only needs it at the moment it calls a model. Converting at the seam,
rather than storing a `GradientTable` throughout, is the deliberate design.

## Consequences

- **Coupling.** The seam is a single, well-defined dependency point — good for
  isolation, but it means the positional signature of
  `gradient_table_from_bvals_bvecs` (which nifreeze calls positionally) is
  load-bearing; a change to it silently mis-maps every fit
  ([[dipy-version-pin-fragility]]).
- **b0 semantics.** DIPY's `b0s_mask` uses `bvals <= 50`; nifreeze's b0 append and
  S0 handling inherit that threshold ([[tool-dipy-gradient-table]]).
- **Cost.** The rebuild is cheap per call but non-zero and repeated; it is a known
  place to optimise (e.g. caching the table for target-independent predictions)
  without changing the interface.
