---
title: "Diffusion MRI signal formation"
entity_type: concept
namespace: paper
last_verified: 2026-07-08
confidence_score: 0.9
---

# Diffusion MRI signal formation

The theory of *what a diffusion-weighted (DW) measurement is* — the foundation
on which every predictor and correction in nifreeze operates. This page is
domain theory; it does not describe nifreeze's code.

## The measurement

A diffusion-weighted acquisition applies a pair of gradient pulses that sensitise
the MR signal to the displacement of water molecules along a direction
$\mathbf{g}$ (a unit vector). The attenuation of the signal relative to the
non-weighted ($b=0$) signal follows, in the mono-exponential (Stejskal–Tanner)
approximation,

$$ S(\mathbf{g}, b) = S_0 \, \exp\!\left(-b \, \mathbf{g}^{\mathsf{T}} \mathbf{D}\, \mathbf{g}\right), $$

where $S_0$ is the non-attenuated signal, $\mathbf{D}$ is the $3\times3$
diffusion tensor, and $b$ (the **b-value**, units s·mm⁻²) encodes the strength
and timing of the diffusion gradients,

$$ b = \gamma^2 G^2 \delta^2 \left(\Delta - \tfrac{\delta}{3}\right), $$

with $\gamma$ the gyromagnetic ratio, $G$ the gradient amplitude, $\delta$ the
pulse duration and $\Delta$ the separation between pulses.

## Sampling: q-space, shells, and gradient tables

- A **gradient table** (b-vectors + b-values) specifies the set of
  measurements. Each DW volume corresponds to one $(\mathbf{g}, b)$ pair.
- Because attenuation depends on $\mathbf{g}^{\mathsf{T}}\mathbf{D}\,\mathbf{g}$,
  the signal is **antipodally symmetric**: $S(\mathbf{g}) = S(-\mathbf{g})$. The
  relevant geometry is therefore the *angle between directions*, taken over the
  half-sphere.
- A **shell** is a set of directions acquired at one fixed $b$. **Single-shell**
  sampling uses one non-zero $b$; **multi-shell** uses several. Higher $b$
  probes finer microstructure but has lower SNR and very different contrast
  volume-to-volume.

## Why this matters for correction

Two properties make DWI unusually hard to align and denoise, and motivate the
whole Andersson framework:

1. **Contrast varies between volumes.** Two volumes with different $\mathbf{g}$
   look genuinely different, so naive volume-to-volume registration fails. A
   *predictor* of each volume's expected appearance is needed — supplied by the
   Gaussian process ([[concept-gaussian-process-regression]],
   [[concept-dmri-angular-covariance]], [[andersson-2015-gp-dmri]]).
2. **The signal is fragile.** Bulk motion during the diffusion-encoding period
   causes whole-slice **signal dropout**; eddy currents and susceptibility warp
   the geometry ([[concept-eddy-current-distortion]],
   [[concept-epi-off-resonance-distortion]]).

The angle between two gradient directions, used pervasively downstream, is

$$ \theta(\mathbf{g}, \mathbf{g}') = \arccos\big|\langle \mathbf{g}, \mathbf{g}'\rangle\big|, $$

the absolute value enforcing the antipodal symmetry (see Eq. 11 in
[[concept-dmri-angular-covariance]]).
