---
title: "F.-C. Yeh, V. J. Wedeen, W.-Y. I. Tseng (2010) — Generalized q-Sampling Imaging"
entity_type: paper
doi: 10.1109/TMI.2010.2045126
source_urls:
  - https://doi.org/10.1109/TMI.2010.2045126
last_verified: 2026-07-12
confidence_score: 0.9
---

# F.-C. Yeh, V. J. Wedeen, W.-Y. I. Tseng (2010)

*Generalized q-Sampling Imaging* — IEEE Transactions on Medical Imaging,
29(9), 1626–1635.

## Claim

From the Fourier-transform relation between the diffusion MR signal and the
underlying diffusion displacement, the authors derive a new relation that
estimates the **spin distribution function (SDF)** directly from the signal.
This yields a reconstruction method — generalized q-sampling imaging (GQI) —
that obtains the SDF from **either** shell (q-ball) **or** grid (DSI) sampling,
via a single quadrature over the acquired q-space samples with no Fourier
transform or grid interpolation.

## Why relevant

- **Claim.** GQI is a model-free q-space reconstruction giving an SDF whose
  values are *comparable across voxels* (unlike per-voxel-normalized ODFs), and
  a per-fiber anisotropy index, quantitative anisotropy (QA), that correlates
  with fiber volume fraction.
- **Evidence.** Mixed-Gaussian simulation (two fibers + isotropic, Rician noise,
  b0-SNR = 30) and an in-vivo 3T experiment. Major-fiber angular deviation and
  minor-fiber resolving rate were **comparable to QBI (shell) and DSI (grid)**;
  QA correlated with volume fraction ($r = 0.86$). The paper does not claim
  better angular resolution than QBI/DSI.
- **Location in the project.** Domain theory in the same q-space family as the
  signal models nifreeze fits/predicts; not a motion or distortion correction
  step. Fully transcribed with equations in
  [[concept-generalized-q-sampling-imaging]]; positioned against QBI/DSI in
  [[q-space-reconstruction-landscape]].
- **Limits (acknowledged).** Correct SDF requires a *balanced* sampling scheme
  (Eq. 10); the measured spin density is biased by $T_1$/$T_2$/B1
  inhomogeneity; the SDF pattern depends on the sampling scheme, so an optimum
  scheme is still needed.

Primary source read in full (IEEE TMI PDF); equations 1–12 captured verbatim in
the concept page.
