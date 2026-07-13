---
title: "Diffusion kurtosis imaging (DKI)"
entity_type: concept
namespace: paper
last_verified: 2026-07-13
confidence_score: 0.9
---

# Diffusion kurtosis imaging (DKI)

The natural extension of the diffusion tensor to **non-Gaussian** diffusion: DKI
adds the next term of the cumulant expansion of the log-signal, a fourth-order
**kurtosis tensor** $\mathbf{W}$ that quantifies how far tissue-water
displacement departs from a Gaussian ([[jensen-2005-diffusional-kurtosis]]). This
page is domain theory at the fidelity of the original paper; it does not describe
nifreeze code. Notation follows [[concept-diffusion-mri-signal]] and
[[concept-diffusion-tensor-imaging]].

## The model

The DTI model is the leading term of a cumulant expansion of $\ln S$ in the
b-value. Keeping the next (quadratic) term gives the DKI signal model:

$$ \ln S(\mathbf{g}, b) = \ln S_0
 - b\,D_{\mathrm{app}}(\mathbf{g})
 + \tfrac{1}{6}\,b^2\,D_{\mathrm{app}}(\mathbf{g})^2\,K_{\mathrm{app}}(\mathbf{g})
 + \mathcal{O}(b^3), $$

where along direction $\mathbf{g}$:

$$ D_{\mathrm{app}}(\mathbf{g}) = \mathbf{g}^{\mathsf{T}}\mathbf{D}\,\mathbf{g}, \qquad
K_{\mathrm{app}}(\mathbf{g}) =
\frac{M_D^2}{D_{\mathrm{app}}(\mathbf{g})^2}\sum_{i,j,k,l} g_i g_j g_k g_l\,W_{ijkl}. $$

$\mathbf{D}$ is the usual $3\times3$ diffusion tensor (6 unique parameters);
$\mathbf{W}$ is a fully symmetric $3\times3\times3\times3$ tensor with **15 unique
parameters**; $M_D$ is the mean diffusivity. The apparent kurtosis
$K_{\mathrm{app}}$ is dimensionless and, for a Gaussian, is exactly zero — so DKI
degrades gracefully to DTI when diffusion *is* Gaussian.

## Why ≥3 b-value levels are required

DTI fits one slope per direction (just $D_{\mathrm{app}}$), so two b-value levels
($b=0$ plus one shell) suffice. DKI additionally fits the **curvature** of
$\ln S$ vs. $b$ (the $K_{\mathrm{app}}$ term), which needs at least a third
distinct b-value level to be identifiable. Hence any DKI fit requires **at least
three b-values, which may include $b=0$** — a hard prerequisite, not a
recommendation.

This is the exact gate nifreeze enforces before instantiating a DKI model, via
`check_multi_b(gtab, 3, non_zero=False)` (the `non_zero=False` admits $b=0$ as one
of the three); see [[tool-dipy-gradient-table]]. Because DIPY's DKI fit assumes a
$b=0$ reference is present, nifreeze appends the acquired b-zero to the data and
gradient table before fitting ([[tool-dipy-reconst-models]],
[[gradient-table-interop-hotpath]]).

## Derived scalars

Diagonalisation is less direct than in DTI (kurtosis is a fourth-order quantity),
but rotationally-invariant summaries are standard: **mean kurtosis (MK)** — the
kurtosis averaged over all directions; **axial kurtosis (AK)** — along the
principal diffusion eigenvector; **radial kurtosis (RK)** — perpendicular to it.
Elevated kurtosis marks microstructural heterogeneity/restriction (membranes,
crossing fibres, compartmentation) that the Gaussian tensor cannot express.

## Use as a signal predictor

Like DTI, DKI's role for correction is prediction: fit $(\mathbf{D}, \mathbf{W},
S_0)$ from a set of volumes, then predict the left-out volume's signal at its
$(\mathbf{g}, b)$. DKI predicts high-$b$ signal more faithfully than DTI because
it captures the log-signal curvature, at the cost of the multi-shell requirement
and greater noise sensitivity.

## Limits

The cumulant expansion is a low-b approximation; it is valid only over a limited
range (roughly $b \lesssim 3000\,\mathrm{s/mm^2}$) and diverges at very high $b$.
The 21-parameter fit (6 + 15) is markedly more sensitive to noise than DTI's,
motivating weighted/robust estimators. DIPY implements this model as
`dipy.reconst.dki.DiffusionKurtosisModel`; the consumed contract (and nifreeze's
thin subclass) is cached in [[tool-dipy-reconst-models]].
