---
title: "The q-space reconstruction landscape: DSI, QBI, and GQI"
entity_type: synthesis
last_verified: 2026-07-12
confidence_score: 0.9
derived_from:
  - refs/yeh-2010-generalized-q-sampling-imaging.md
  - refs/tuch-2004-q-ball-imaging.md
---

# The q-space reconstruction landscape: DSI, QBI, and GQI

Model-free diffusion reconstruction methods all exploit the same Fourier-
transform relation between the diffusion MR signal and the underlying diffusion
displacement (Eq. 1 in [[concept-generalized-q-sampling-imaging]]). They differ
in *what they reconstruct*, *what sampling they require*, and *how they turn
samples into an orientation function*. Generalized q-sampling imaging
([[yeh-2010-generalized-q-sampling-imaging]]) is best understood as the method
that unifies the two earlier families.

## The three methods

| Method | Reconstructs | Native sampling | Core operation | Regularizer |
|--------|--------------|-----------------|----------------|-------------|
| **DSI** (Wedeen) | diffusion ODF (probability) | Cartesian **grid** in q-space | Fourier transform → numerical ODF integration on the grid | Hanning filter on truncation artifacts |
| **QBI** ([[tuch-2004-q-ball-imaging]]) | diffusion ODF (probability) | single **shell** (HARDI) | **Funk–Radon transform** of the shell | spherical-harmonic order + λ (e.g. [[descoteaux-2006-apparent-diffusion-coefficients-high-angular-resolutio]]) |
| **GQI** ([[yeh-2010-generalized-q-sampling-imaging]]) | **spin distribution function (SDF)** | shell **or** grid (any balanced scheme) | one quadrature with a $\operatorname{sinc}$ (or $L^2$) kernel | explicit diffusion sampling length $L_\Delta$ |

## Two claims that make GQI a generalization, not just another method

**1. QBI is the infinite-$L_\Delta$ limit of GQI.** GQI reconstructs the SDF as a
$\operatorname{sinc}$-weighted quadrature over q-space samples (Eq. 5–6). As the
diffusion sampling length $L_\Delta \to \infty$, the $\operatorname{sinc}$ kernel
tends to a delta function and the reconstruction collapses to the Funk–Radon
transform that defines QBI. QBI is therefore a special (infinite-length) case;
GQI is the finite-length family that keeps $L_\Delta$ as a tunable parameter.

**2. GQI and DSI share the theory but not the numerics.** Both rest on the
Fourier relation of Eq. 1, so on grid data they produce similar diffusion
patterns. But DSI performs an explicit Fourier transform followed by ODF
integration on the transformed grid, and suppresses the resulting spiky
truncation artifact with a Hanning filter. GQI collapses the Fourier transform
and the ODF integral into a single mathematical reduction — no forward
transform, no grid interpolation, no Hanning filter — and controls the
artifact/resolution trade-off through the explicit length $L_\Delta$ instead.

## SDF vs ODF: the load-bearing distinction

QBI and DSI both normalize the ODF *per voxel* so that each is a probability
density; consequently the same ODF value in two voxels need not represent the
same physical quantity, and ODFs are not comparable across voxels. GQI's SDF is
instead the propagator scaled by spin density (Eq. 4) and rescaled by a single
constant $Z_0$ across all voxels, giving SDF values a consistent physical
meaning **voxel-to-voxel**. This is what lets GQI define a per-fiber
quantitative anisotropy (QA, Eq. 11) that correlates with fiber volume fraction
($r = 0.86$ in the simulation) — a comparison the per-voxel-normalized ODF
cannot support. Normalizing the SDF per voxel would recover a diffusion ODF.

## Empirical bottom line

In the paper's simulation and in-vivo study, GQI's accuracy in resolving major
and minor fibers was **comparable to QBI on shell data and to DSI on grid
data** — no method dominated on angular resolution, and GQI's accuracy varied
with $L_\Delta$. The distinguishing advantages of GQI are practical:
deconvolution-free implementation, applicability to arbitrary (balanced)
sampling schemes, and inter-voxel-comparable SDF/QA values — not a claim of
superior fiber-orientation accuracy.
