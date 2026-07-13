---
title: "Diffusion tensor imaging (DTI)"
entity_type: concept
namespace: paper
last_verified: 2026-07-13
confidence_score: 0.9
---

# Diffusion tensor imaging (DTI)

The first-order model of the diffusion-weighted signal: water displacement in a
voxel is treated as a single anisotropic Gaussian, summarised by a symmetric
$3\times3$ **diffusion tensor** $\mathbf{D}$ ([[basser-1994-estimation-effective-self-diffusion-tensor-nmr-spin]]).
This page is domain theory at the fidelity of the original paper; it does not
describe nifreeze code. It inherits the acquisition notation of
[[concept-diffusion-mri-signal]] ($\mathbf{g}$, $b$, $S_0$).

## The model

Under the mono-exponential (Stejskal–Tanner) approximation, the signal in
direction $\mathbf{g}$ at b-value $b$ is

$$ S(\mathbf{g}, b) = S_0 \, \exp\!\left(-b\,\mathbf{g}^{\mathsf{T}}\mathbf{D}\,\mathbf{g}\right). $$

$\mathbf{D}$ is symmetric positive (semi-)definite, so it has **6 unique
parameters** ($D_{xx}, D_{yy}, D_{zz}, D_{xy}, D_{xz}, D_{yz}$). With $S_0$ that
is **7 unknowns**, requiring at minimum one $b=0$ measurement and **6 non-collinear
diffusion directions**.

## Estimation

Taking logs linearises the model in the tensor elements:

$$ \ln S(\mathbf{g}, b) = \ln S_0 - b\,\mathbf{g}^{\mathsf{T}}\mathbf{D}\,\mathbf{g}
 = \ln S_0 - b\sum_{i,j} g_i g_j D_{ij}. $$

Stacking all measurements gives a linear system $\mathbf{y} = \mathbf{B}\,\mathbf{d}$,
where $\mathbf{d} = (\ln S_0, D_{xx}, D_{yy}, D_{zz}, D_{xy}, D_{xz}, D_{yz})^{\mathsf{T}}$
and each row of the **design matrix** $\mathbf{B}$ holds
$(1, -b g_x^2, -b g_y^2, -b g_z^2, -2b g_x g_y, -2b g_x g_z, -2b g_y g_z)$.
Common estimators:

- **Ordinary log-linear least squares (OLS)** — solve
  $\hat{\mathbf{d}} = (\mathbf{B}^{\mathsf{T}}\mathbf{B})^{-1}\mathbf{B}^{\mathsf{T}}\mathbf{y}$.
  Fast, but the log transform makes the noise heteroscedastic (low-signal, high-$b$
  points are over-weighted).
- **Weighted least squares (WLS)** — reweight rows by $S^2$ (or $\hat S^2$) to
  undo the log-induced heteroscedasticity; the standard default.
- **Non-linear least squares (NLLS)** — fit in the signal domain directly,
  optionally with positivity constraints on $\mathbf{D}$.

Robust variants (e.g. RESTORE) reject outlier measurements during the fit — see
[[concept-outlier-detection-replacement]].

## Derived scalars

Diagonalising $\mathbf{D} = \sum_i \lambda_i \mathbf{e}_i \mathbf{e}_i^{\mathsf{T}}$
(eigenvalues $\lambda_1 \ge \lambda_2 \ge \lambda_3 \ge 0$) yields
rotation-invariant summaries:

$$ \mathrm{MD} = \frac{\lambda_1 + \lambda_2 + \lambda_3}{3}, \qquad
\mathrm{FA} = \sqrt{\tfrac{3}{2}}\,
\frac{\sqrt{(\lambda_1-\mathrm{MD})^2 + (\lambda_2-\mathrm{MD})^2 + (\lambda_3-\mathrm{MD})^2}}
{\sqrt{\lambda_1^2 + \lambda_2^2 + \lambda_3^2}}. $$

**MD** (mean diffusivity) is the trace/3; **FA** (fractional anisotropy) ranges
from 0 (isotropic) to 1 (perfectly linear). The principal eigenvector
$\mathbf{e}_1$ estimates the dominant fibre orientation.

## Use as a signal predictor

For motion/eddy correction the interesting output of DTI is not FA but its
**predictive** power: once $\mathbf{D}$ (and $S_0$) are fit from a set of volumes,
the model predicts the expected signal at *any* new $(\mathbf{g}, b)$ via the
model equation. This is exactly how nifreeze uses DIPY's `TensorModel` — as the
predictor of the left-out volume in leave-one-volume-out estimation
([[concept-leave-one-volume-out]]). DTI's limitation as a predictor is that a
single Gaussian cannot represent **crossing fibres**, which motivates richer
predictors (DKI, GQI, and the model-free Gaussian process,
[[concept-gaussian-process-regression]]).

## Relation to nifreeze's dependency

DIPY implements this model as `dipy.reconst.dti.TensorModel`; the exact contract
nifreeze consumes is cached in [[tool-dipy-reconst-models]]. The kurtosis
extension that captures non-Gaussian diffusion is
[[concept-diffusion-kurtosis-imaging]].
