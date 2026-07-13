---
title: "GP prediction underpins Leave-One-Volume-Out estimation"
entity_type: synthesis
last_verified: 2026-07-08
confidence_score: 0.9
derived_from:
  - refs/andersson-2015-gp-dmri.md
  - refs/andersson-2016-integrated-eddy.md
---

# GP prediction underpins Leave-One-Volume-Out estimation

**Technical claim.** The Leave-One-Volume-Out (LOVO) estimation strategy — fit a
signal model on all volumes but one, predict the held-out volume, and register
the observed held-out volume to that prediction — is a direct instantiation of
Gaussian-process regression used as a *reference-free* target generator. The
predictive mean of a GP (Eq. 7,
[[concept-gaussian-process-regression]]) *is* the predicted volume, and the
angular/multi-shell covariance (Eqs. 9–16,
[[concept-dmri-angular-covariance]]) is what makes that prediction faithful for
diffusion contrast.

## Why LOVO needs a predictor, not a template

Diffusion volumes have direction-dependent contrast
([[concept-diffusion-mri-signal]]), so there is no fixed template to register
against and volume-to-volume registration is ill-posed. LOVO solves this by
*synthesising* the target: leave volume $k$ out, fit the GP to the rest,
evaluate the predictive mean at $(\mathbf{g}_k, b_k)$, and register volume $k$
to it ([[concept-image-registration]]). Every volume thus gets a
same-contrast target derived from the data itself. That the target is built
*without* volume $k$ is not incidental — the held-out independence is the
validity guarantee of the whole scheme; see [[concept-leave-one-volume-out]].

## The loop is necessarily iterative

Good predictions require approximately-aligned inputs, and good alignment
requires good predictions. Both [[andersson-2015-gp-dmri]] and
[[andersson-2016-integrated-eddy]] therefore alternate: predict → register each
volume → refit → repeat. This is exactly nifreeze's
*Data → Model → Estimator* loop: `ModelFactory` builds the GP (or DTI/DKI/average)
model, the model's `fit_predict(index)` returns the LOVO prediction, and
`Estimator.run` registers each volume with ANTs and iterates (optionally chaining
via `prev_model`).

## Consequences for refactors / new models

- Any model plugged into the estimator only has to satisfy the contract "given
  all volumes but one, predict the left-out volume." DTI, DKI, average-DWI, and
  the GP are interchangeable *predictors* behind the same LOVO interface;
  correctness of the loop does not depend on which one is used.
- The GP additionally yields a predictive **variance** (Eq. 8,
  [[concept-gaussian-process-regression]]), which is the natural signal for
  outlier detection ([[concept-outlier-detection-replacement]]) — a capability
  simpler predictors lack.
- Registration quality caps the achievable prediction; using a well-validated
  registration engine (ANTs) rather than a bespoke solver is a deliberate
  divergence from the paper — see [[andersson-eddy-framework-lineage]].
