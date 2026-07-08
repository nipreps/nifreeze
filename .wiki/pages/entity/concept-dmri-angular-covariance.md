---
title: "Angular covariance functions for diffusion MRI"
entity_type: concept
namespace: paper
last_verified: 2026-07-08
confidence_score: 0.95
---

# Angular covariance functions for diffusion MRI

The kernel that specialises generic GP regression
([[concept-gaussian-process-regression]]) to the diffusion signal. These are the
equations nifreeze's `src/nifreeze/model/gpr.py` cites by number (Eqs. 9, 10,
14, 16 of [[andersson-2015-gp-dmri]]).

## The idea

The DW signal is a smooth, antipodally-symmetric function on the sphere of
gradient directions ([[concept-diffusion-mri-signal]]). Its covariance can
therefore depend only on the **angle** between two directions,

$$ \theta(\mathbf{g}, \mathbf{g}') = \arccos\big|\langle \mathbf{g}, \mathbf{g}'\rangle\big| \qquad\text{(Eq. 11),} $$

with the absolute value enforcing $S(\mathbf{g}) = S(-\mathbf{g})$, so
$\theta \in [0, \pi/2]$.

## Two angular covariance families

The paper proposes and compares two covariance functions $C(\theta)$ with a
single length-scale hyperparameter $a$.

**Exponential (Eq. 9):**

$$ C(\theta) = e^{-\theta / a}, \qquad 0 \le \theta \le \pi. $$

**Spherical (compact support, Eq. 10):**

$$ C(\theta) = \begin{cases} 1 - \dfrac{3\theta}{2a} + \dfrac{\theta^{3}}{2a^{3}} & \text{if } \theta \le a, \\[4pt] 0 & \text{if } \theta > a. \end{cases} $$

The spherical form has **compact support**: directions farther apart than $a$
are treated as uncorrelated, giving a sparser, better-conditioned covariance
matrix. Model selection (Eqs. 12–13) chooses between them from the data.

## Multi-shell extension

For multi-shell data the kernel is the product of an angular term and a
**b-value** term (Eq. 14):

$$ k(x, x') = C_{\theta}\big(\theta(\mathbf{g}, \mathbf{g}');\, a\big)\; C_{b}\big(|b - b'|;\, \ell\big), $$

where the b-value covariance is a squared-exponential in log-b (Eq. 15):

$$ C_{b}(b, b', \ell) = \exp\!\left(-\frac{(\log b - \log b')^{2}}{2\ell^{2}}\right). $$

Because $C_b \neq 0$ across shells, measurements in one shell **inform**
predictions in another — the multi-shell coupling the paper highlights.

## Assembled covariance matrix

For two shells with gradient sets $G_1, G_2$, per-shell noise variances
$\sigma_1^2, \sigma_2^2$, and a signal-scaling hyperparameter $\lambda$, the
full (noise-augmented) covariance is (Eq. 16):

$$ K = \begin{bmatrix} \lambda\, C_{\theta}(\theta(G_1); a) + \sigma_1^{2} I & \lambda\, C_{\theta}(\theta(G_2, G_1); a)\, C_{b}(b_2, b_1, \ell) \\[4pt] \lambda\, C_{\theta}(\theta(G_1, G_2); a)\, C_{b}(b_1, b_2, \ell) & \lambda\, C_{\theta}(\theta(G_2); a) + \sigma_2^{2} I \end{bmatrix}. $$

This $K$ is the $K(x,x)+\sigma^2 I$ that enters the predictive mean (Eq. 7) in
[[concept-gaussian-process-regression]]. The diagonal blocks are within-shell
covariances; the off-diagonal blocks carry the cross-shell coupling.

## Grounding in the project

`gpr.py` implements $C_\theta$ (Eqs. 9 and 10), the combined kernel (Eq. 14),
and the noise-augmented $K$ (Eq. 16); the angle helper corresponds to Eq. 11.
The predictions this kernel yields drive the correction loop in
[[gp-prediction-underpins-lovo]].
