---
title: "When single-fit mode is admissible (and when it is not)"
entity_type: synthesis
last_verified: 2026-07-10
confidence_score: 0.85
derived_from:
  - refs/andersson-2015-gp-dmri.md
  - refs/andersson-2016-integrated-eddy.md
---

# When single-fit mode is admissible (and when it is not)

**Technical claim.** nifreeze's *single-fit mode* — fit the model **once** on
all available volumes and reuse that one locked prediction as the registration
target for every index, instead of refitting leave-one-out per volume — trades
away the held-out independence that makes Leave-One-Volume-Out valid
([[concept-leave-one-volume-out]]). It is therefore **not** a lossless ~$N\times$
speed-up for data-driven predictors, but it *is* a legitimate tool with a real
and currently under-documented role. The distinction is about **whether the
target depends on the volume it is registered against**, not about speed.

## Why it is not a free speed-up

For a data-driven model (DTI, DKI, GP), single-fit includes volume $k$ in the
fit that generates $k$'s own target. Per [[concept-leave-one-volume-out]] that
is a self-registration whose transform is biased toward the identity —
systematically *under*-estimating motion. The "leaving 1 of $N$ out barely
changes the fit" argument does not rescue it: the bias comes from the target
being correlated with its own volume, not from the size of the fit change.
Making single-fit the default for DKI (or multi-shell head-motion correction)
on the grounds that it "removes the $\times N$ refit at negligible accuracy
cost" reproduces exactly this error and should be rejected.

## Where single-fit is admissible

1. **Target-independent references.** When the predictor does not depend on the
   moving volume — a fixed $b\!=\!0$ / reference map (nifreeze's `TrivialModel`)
   — the independence invariant holds *by construction*, so fitting once is not
   an approximation at all. This is single-fit's principled home.
2. **Development, CI and integration tests.** A locked prediction makes the
   estimator loop cheap and deterministic to exercise end-to-end without paying
   the per-volume refit, which is what you want when testing the *plumbing*
   (iteration, registration wiring, I/O) rather than measuring accuracy.
3. **Coarse, low-DOF initialisation.** A single locked target is an acceptable
   seed for a rough, low-degree-of-freedom linear pre-alignment whose only job
   is to get volumes approximately aligned before a subsequent, independence-
   respecting LOVO stage refines them. The identity bias is tolerable here
   because an later independent stage corrects it — the coarse stage is not the
   final estimate.

The common thread: single-fit is fine wherever the resulting bias either cannot
occur (case 1) or does not reach the final motion estimate (cases 2–3). It is
inadmissible precisely when the locked prediction *is* the accuracy-critical
target for a data-driven model.

## The real levers for the DKI slowness

The slowness that motivates reaching for single-fit (a full-brain 22-parameter
kurtosis refit $\times N$ volumes) has lossless remedies that keep LOVO
independence intact:

- **Exact / down-dated closed-form leave-one-out** for the linear (weighted)
  least-squares DTI/DKI fits: the leave-one-out prediction is available from a
  single full fit via the hat-matrix / PRESS identity, giving *true* LOVO at
  roughly one-fit cost.
- **Voxel-chunked parallelism** (the joblib / `multi_voxel_fit` path).
- **BLAS thread control** to avoid oversubscription between inner linear-algebra
  threads and outer parallel workers.

> **Method note (not a cited wiki reference).** The exact/down-dated LOO result
> is standard linear-algebra plus a nifreeze/DIPY code fact about the linear WLS
> fit; it is engineering knowledge, not literature grounded in this wiki's refs.
> Treat it as a lead to implement and verify, not as an established citation.

## Secondary loss

A single locked GP fit also collapses the per-volume predictive **variance**
that [[concept-outlier-detection-replacement]] relies on — so single-fit is
doubly inappropriate for pipelines that also do GP-based outlier detection.
