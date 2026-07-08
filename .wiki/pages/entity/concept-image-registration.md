---
title: "Image registration to a model prediction"
entity_type: concept
namespace: paper
last_verified: 2026-07-08
confidence_score: 0.9
---

# Image registration to a model prediction

The optimisation that recovers motion and distortion parameters by aligning each
observed volume to a *predicted* volume. This is the computer-vision core of the
Andersson framework and the pattern nifreeze's estimator implements. Notation
follows [[andersson-2016-integrated-eddy]].

## Register to a prediction, not to another volume

Because diffusion volumes differ in contrast ([[concept-diffusion-mri-signal]]),
the framework does **not** register volume-to-volume. Instead it registers each
observed volume $f_i$ to a **model-free prediction** $\bar f_i$ of what that
volume should look like — the Gaussian-process prediction of
[[andersson-2015-gp-dmri]] ([[concept-gaussian-process-regression]]). This makes
the two images have the *same* expected contrast, so an intensity-based metric
is meaningful even at high b-value.

## Forward and inverse resampling with Jacobian modulation

Given a reference-space image $s_i$, the parameters (susceptibility $h$, eddy
$\beta_i$, motion $r_i$, acquisition $a_i$) define the sampling map $x'$
(Eqs. 2–3, [[concept-rigid-body-motion]]). Resampling to predict the *observed*
(distorted) image, and its inverse, are (Eqs. 4–5)

$$ \bar{s}_i(x) = f_i(x')\, J_x(h, \beta_i, r_i, a_i), \qquad \bar{f}_i(x) = s_i(x')\, J_x^{-1}(h, \beta_i, r_i, a_i), $$

where $J_x$ is the **Jacobian determinant** of the transform. Multiplying by
$J_x$ conserves signal: compression concentrates intensity, stretching dilutes
it. Omitting this modulation biases the estimate.

## The registration algorithm (Gauss–Newton)

Parameters are updated to minimise the sum-of-squared difference between the
predicted distorted volume $\bar f_i$ and the observed $f_i$. With the residual
$(\bar f_i - f_i)$ and the Jacobian of the model wrt the parameters (Eq. 7)

$$ D = \left[\, \frac{\partial \bar f_i}{\partial \beta_{1i}} \;\cdots\; \frac{\partial \bar f_i}{\partial \beta_{ni}} \;\; \frac{\partial \bar f_i}{\partial r_{1i}} \;\cdots\; \frac{\partial \bar f_i}{\partial r_{6i}} \,\right], $$

the Gauss–Newton update is (Eq. 6)

$$ \begin{bmatrix} \beta_i^{(k+1)} \\ r_i^{(k+1)} \end{bmatrix} = \begin{bmatrix} \beta_i^{(k)} \\ r_i^{(k)} \end{bmatrix} - (D^{\mathsf{T}} D)^{-1} D^{\mathsf{T}} \big(\bar f_i - f_i\big). $$

The whole procedure is **iterated**: fit the model → predict each volume →
re-register → refit, because a good prediction needs approximately-aligned
inputs and good alignment needs a good prediction.

## Grounding in the project

This *predict → register-each-volume → update* loop is nifreeze's
`Estimator.run` (a model's `fit_predict` + ANTs registration per volume, run
with joblib). nifreeze uses ANTs rather than the paper's bespoke Gauss–Newton
solver, but the objective — align observed to predicted — is identical. See
[[gp-prediction-underpins-lovo]] and [[andersson-eddy-framework-lineage]].
