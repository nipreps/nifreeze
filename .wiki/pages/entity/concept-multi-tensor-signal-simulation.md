---
title: "Multi-tensor signal simulation"
entity_type: concept
namespace: paper
last_verified: 2026-07-13
confidence_score: 0.9
---

# Multi-tensor signal simulation

The standard forward model for synthesising a **ground-truth** diffusion-weighted
signal: a voxel is modelled as a mixture of a few non-exchanging Gaussian
compartments, each a diffusion tensor, and the observed signal is their
volume-fraction-weighted sum plus noise. This is the generator behind every
phantom nifreeze validates its motion estimation against. Domain theory; not a
description of nifreeze code. Notation follows [[concept-diffusion-mri-signal]]
and [[concept-diffusion-tensor-imaging]].

## The mixture model

For $K$ fibre compartments with unit gradient direction $\mathbf{g}$ at b-value
$b$:

$$ S(\mathbf{g}, b) = S_0 \sum_{k=1}^{K} f_k \,
 \exp\!\left(-b\,\mathbf{g}^{\mathsf{T}}\mathbf{D}_k\,\mathbf{g}\right),
 \qquad \sum_{k=1}^{K} f_k = 1, \quad f_k \ge 0. $$

Each compartment $k$ is a tensor $\mathbf{D}_k = \mathbf{R}_k \,
\mathrm{diag}(\lambda_1, \lambda_2, \lambda_3)\, \mathbf{R}_k^{\mathsf{T}}$, where
the eigenvalues set the compartment's diffusivity/anisotropy and the rotation
$\mathbf{R}_k$ (built from the fibre's principal direction) sets its orientation.
The $f_k$ are **volume fractions**. A **single fibre** is the $K=1$ special case;
**crossing fibres** are $K \ge 2$ with different orientations — the configuration
that a single diffusion tensor cannot represent and that stresses a predictor's
angular fidelity.

This is the same mixture used, e.g., in the GQI validation study (Eq. 12 of
[[concept-generalized-q-sampling-imaging]]), there written with an explicit
isotropic compartment $f_0$; setting one compartment's tensor isotropic recovers
that form.

## Noise

Simulated signal is made realistic by adding noise at a specified
**signal-to-noise ratio (SNR)**, defined against $S_0$. MR magnitude noise is
**Rician** (the magnitude of complex Gaussian noise): at high SNR it is
approximately Gaussian, but near the noise floor (high $b$, low signal) it is
biased upward — a property that matters when validating corrections at high
b-value. The noise is drawn from a pseudo-random generator seeded for
reproducibility.

## Why it underpins validation

Motion/eddy estimation can only be validated against a **known** ground truth:
one simulates a clean multi-fibre signal on a chosen gradient scheme, applies
known motion/distortion, runs the estimator, and checks that the recovered
transforms match the applied ones. The multi-tensor model supplies that clean
signal with controllable fibre count, orientation, anisotropy, and SNR.

## Relation to nifreeze's dependency

DIPY implements this model as `dipy.sims.voxel.single_tensor` ($K=1$) and
`multi_tensor` ($K\ge1$, returning both the signal and the compartment sticks),
with `all_tensor_evecs` building a compartment's eigenvector frame from its
principal direction. The exact contracts nifreeze consumes are cached in
[[tool-dipy-sims-voxel]]; the directions the signal is simulated on come from
[[concept-sphere-sampling-electrostatic-repulsion]].
