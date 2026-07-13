---
title: "Gaussian-process regression"
entity_type: concept
namespace: paper
last_verified: 2026-07-08
confidence_score: 0.95
---

# Gaussian-process regression

The general Bayesian non-parametric regression machinery that, specialised with
a spherical covariance ([[concept-dmri-angular-covariance]]), becomes the
diffusion-signal predictor of [[andersson-2015-gp-dmri]]. Equation numbers below
follow that paper (the source nifreeze's `gpr.py` cites).

## Definition

A **Gaussian process** is a distribution over functions such that any finite set
of function values is jointly Gaussian. Where a Gaussian random variable is
specified by a mean and variance (Eq. 1),

$$ Y \sim \mathcal{N}(\mu, \sigma^2), $$

a GP is specified by a **mean function** $m(x)$ and a **covariance (kernel)
function** $k(x, x')$ (Eq. 2):

$$ f(x) \sim \mathcal{GP}\big(m(x),\, k(x, x')\big). $$

Evaluated at a finite set of inputs it reduces to a multivariate normal (Eq. 3),
$Y \sim \mathcal{N}_p(\mu, \Sigma)$, with $\Sigma_{ij} = k(x_i, x_j)$.

## Prediction (regression)

Assume zero mean. The training outputs $f$ (at inputs $x$) and a test output
$f^{*}$ (at $x^{*}$) are jointly Gaussian (Eq. 4):

$$ \begin{bmatrix} f \\ f^{*} \end{bmatrix} \sim \mathcal{N}_{n+1}\!\left(0,\; \begin{bmatrix} K(x, x) & k(x, x^{*}) \\ k(x^{*}, x) & k(x^{*}, x^{*}) \end{bmatrix}\right). $$

Conditioning on the observed $f$ gives the predictive distribution (Eq. 5):

$$ p(f^{*} \mid x^{*}, x, f) = \mathcal{N}\!\Big( k(x^{*}, x)\,K(x,x)^{-1} f,\;\; k(x^{*}, x^{*}) - k(x^{*}, x)\,K(x,x)^{-1} k(x, x^{*}) \Big). $$

## Observation noise

Real measurements carry noise, $y = f + \varepsilon$ with
$\varepsilon \sim \mathcal{N}(0, \sigma^2)$. The joint becomes (Eq. 6)

$$ \begin{bmatrix} f \\ f^{*} \end{bmatrix} \sim \mathcal{N}_{n+1}\!\left(0,\; \begin{bmatrix} K(x, x) + \sigma^2 I & k(x, x^{*}) \\ k(x^{*}, x) & k(x^{*}, x^{*}) \end{bmatrix}\right), $$

so the predictive **mean** and **covariance** are (Eqs. 7–8)

$$ \bar{f}(x^{*}) = k(x^{*}, x)\big(K(x, x) + \sigma^2 I\big)^{-1} f, $$

$$ \operatorname{Cov}\!\big(\bar{f}(x^{*})\big) = k(x^{*}, x^{*}) - k(x^{*}, x)\big(K(x, x) + \sigma^2 I\big)^{-1} k(x, x^{*}). $$

The mean (Eq. 7) is the **prediction** used to synthesise a left-out volume; the
covariance (Eq. 8) gives a principled *expected error* — the basis for outlier
detection in [[concept-outlier-detection-replacement]].

## Choosing the model: hyperparameters and evidence

The kernel has hyperparameters $\beta$ (e.g. length scales, noise variance).
They are set by maximising the **log marginal likelihood** (Eq. 12), where
$K_y = K + \sigma^2 I$:

$$ \log p(y \mid \beta, M) = -\tfrac{1}{2} y^{\mathsf{T}} K_y^{-1} y - \tfrac{1}{2} \log |K_y| + c. $$

Competing covariance forms $M_i$ are compared by their **model evidence**
(Eq. 13),

$$ p(M_i \mid y) \propto p(y \mid M_i) = \int_{\beta} p(y \mid \beta, M_i)\, p(\beta \mid M_i)\, d\beta, $$

which the paper evaluates with Laplace's approximation (its Appendix, Eqs.
A1–A8) and whose gradient/Hessian for optimisation are given by Eqs. B1–B2.

## Grounding in the project

The predictive mean (Eq. 7) is precisely what nifreeze's GP model returns for a
held-out volume in Leave-One-Volume-Out mode; see
[[gp-prediction-underpins-lovo]]. The specific kernel that makes this work for
diffusion data is [[concept-dmri-angular-covariance]].
