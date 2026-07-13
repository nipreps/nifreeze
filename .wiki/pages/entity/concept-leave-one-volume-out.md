---
title: "Leave-One-Volume-Out (LOVO) estimation"
entity_type: concept
namespace: paper
last_verified: 2026-07-10
confidence_score: 0.9
flag_ids:
  - F0001
---

# Leave-One-Volume-Out (LOVO) estimation

The estimation strategy at the core of nifreeze's motion/distortion correction:
to estimate the transform for volume $k$, fit a signal model on **every volume
but $k$**, predict the held-out volume from that fit, and register the observed
volume $k$ to the prediction. Notation and rationale follow
[[andersson-2015-gp-dmri]] and [[andersson-2016-integrated-eddy]]. This page
states the *validity principle* — why the "leave-one-out" is load-bearing and
not a removable optimisation.

## Why volume-to-volume registration needs a synthesised target

Diffusion volumes have direction-dependent contrast
([[concept-diffusion-mri-signal]]): a $b\!=\!0$ image, a volume sensitised along
$\mathbf{g}_1$, and a volume along $\mathbf{g}_2$ do not look alike, so there is
no fixed template that all volumes can be registered against, and pairwise
volume-to-volume registration is ill-posed. LOVO resolves this by
*synthesising* a same-contrast target for each volume: leave volume $k$ out, fit
the model to the rest, evaluate the prediction at $k$'s acquisition parameters
$(\mathbf{g}_k, b_k)$, and register volume $k$ to that prediction
([[concept-image-registration]]). The target is generated *from the data
itself*, but crucially **not from the datum it will be compared against**.

## The independence invariant

The defining property is: **the prediction used as volume $k$'s registration
target must not be a function of volume $k$'s own signal.** This is what makes
the target an *independent* reference and the estimated transform meaningful.

Violating it — letting volume $k$ enter the fit that generates $k$'s own target
— turns the step into a *self-registration*. A volume compared against a
reference partly built from itself is, to first order, being registered to a
blurred copy of its own (still-misaligned) content, so the estimated transform
is biased **toward the identity**: the estimator systematically reports *less*
motion than is present ("no motion detected"). This is a **circularity**, not a
small numerical approximation. In particular the intuition "with $N$ volumes,
leaving one out barely changes a full-brain fit, so reuse one fit for all $k$"
does **not** rescue it: the error is in the *dependence structure* of the
target, not in the magnitude of the fit perturbation. Even a negligible change
to the fitted field leaves the target correlated with its own volume, and it is
that correlation — not the field's stability — that biases the transform.

## The loop is necessarily iterative

Good predictions require approximately-aligned inputs, and good alignment
requires good predictions. Both [[andersson-2015-gp-dmri]] and
[[andersson-2016-integrated-eddy]] therefore alternate predict → register each
volume → refit → repeat until convergence. The held-out independence invariant
must hold at *every* iteration, not merely the first.

## Consequences for models and refactors

- Any predictor plugged into the estimator only has to satisfy the LOVO
  contract "given all volumes but one, predict the left-out volume." DTI, DKI,
  average-DWI and the GP ([[concept-gaussian-process-regression]]) are
  interchangeable predictors behind this interface; the loop's *correctness*
  does not depend on which is used, but it *does* depend on each predictor
  honouring the independence invariant.
- A predictor that is genuinely **independent of the moving volume** (a fixed
  reference such as a $b\!=\!0$ template) trivially satisfies the invariant and
  may be fit once — see the admissibility analysis in
  [[single-fit-mode-admissibility]].
- The GP additionally yields a predictive **variance**, the natural signal for
  outlier detection ([[concept-outlier-detection-replacement]]); a target that
  collapses the per-volume fit forfeits it.

## Grounding in the project

LOVO is realised by nifreeze's *Data → Model → Estimator* loop: a model's
`fit_predict(index)` returns the held-out prediction for `index` (masking that
volume out of the fit), and `Estimator.run` registers each observed volume to it
with ANTs and iterates. See [[gp-prediction-underpins-lovo]] for how the GP
specialises this, and [[single-fit-mode-admissibility]] for when the per-volume
refit may (and may not) be skipped.
