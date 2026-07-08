---
title: "Outlier (signal-dropout) detection and replacement"
entity_type: concept
namespace: paper
last_verified: 2026-07-08
confidence_score: 0.95
---

# Outlier (signal-dropout) detection and replacement

Detecting and repairing whole-slice **signal dropout** — the most common and
damaging dMRI artefact — inside the non-parametric framework. Equations follow
[[andersson-2016-outlier-replacement]] (numbering as in the paper; captured from
the full text).

## The artefact

Bulk (subject or physiological) motion *during the diffusion-encoding period*
spoils the spin phase and causes **signal loss** over a whole slice, or a large
connected region of a slice, in the affected volume. Those measurements are
unusable and, left in, bias motion/distortion estimation and tensor metrics
(FA, MD).

## Detection: compare the observed scan to the GP prediction

The predictor is the Gaussian process ([[concept-gaussian-process-regression]],
[[andersson-2015-gp-dmri]]): for any measurement point $x^{*} = [\theta\ \phi\ b]$
(direction + b-value) it returns the expected signal (Eq. 1 — identical to Eq. 7
of the 2015 GP paper),

$$ \hat{y}(x^{*}) = k(x^{*}, X)\,\big(K(X, X) + \sigma^2 I\big)^{-1} y. $$

The per-voxel **deviation from expectation** is $d_i = y_i - \hat{y}_i$; a
*negative* $d_i$ means less signal than predicted — a potential dropout. Because
dropout affects a whole slice (or a large connected subset), the natural test
statistic is the **slice mean deviation** (Eq. 2), for slice $s$ of volume $g$
over its $n_s$ brain voxels:

$$ d_{gs} = \frac{1}{n_s} \sum_{i \in s} \big( y_{gi} - \hat{y}_{gi} \big). $$

From the collection of all $d_{gs}$ (thousands to >50 000), the mean $\bar d$
(close to zero) and a pooled voxel-wise standard deviation are estimated
(Eq. 3), with $N_s$ slices per volume and $N_g$ volumes:

$$ \sigma_d^2 = \frac{1}{N_s - 1} \sum_{s=1}^{N_s} n_s \, \frac{1}{N_g - 1} \sum_{g=1}^{N_g} \big( d_{gs} - \bar d \big)^2 . $$

Each slice is then converted to a **z-score** (Eq. 4); the $\sqrt{n_s}$ factor
rescales the standard error of a mean over $n_s$ voxels:

$$ z_{gs} = \frac{\sqrt{n_s}\,\big( d_{gs} - \bar d \big)}{\sigma_d}. $$

A slice is declared an **outlier** when $z_{gs}$ exceeds a chosen threshold
(the paper uses 3–4 standard deviations; typically only *negative* $z$, i.e.
signal loss, is rejected). Higher thresholds (3.5–4.5) gave essentially no false
positives in simulation.

## Detection in scan space

A key design choice: the observed scan is compared to the prediction **in
acquisition (scan) space**, where the GP model in reference space is transformed
*to* the scan rather than the scan being resampled to reference space. This
avoids interpolation mixing valid and outlier voxels, and it removes the
chicken-and-egg **order-of-operations** problem — outlier detection and
movement/distortion correction ([[andersson-2016-integrated-eddy]]) share one
prediction and need not be sequenced. Statistics can be computed per slice or
per multiband (MB) slice-group, since simultaneous-multislice acquisitions drop
out as a group.

## Replacement

A flagged slice is **replaced by its GP prediction** $\hat y$ before use, so
corrupted data no longer contribute to the fit. Detection, replacement, and
correction of the other distortions all live in one integrated pass — the same
prediction serves registration and repair. In simulation, replacement almost
completely reverses the deleterious effect of outliers on registration accuracy
and on FA/MD, and outperforms tensor-level robust estimators (RESTORE /
iRESTORE, [[chang-2005-restore-robust-estimation-tensors-outlier-rejection]]).

## Trade-offs

- Injecting model-predicted values means a systematic **model bias** would
  propagate into "corrected" slices — replacement is only as good as the
  predictor.
- Detection targets *slice-* (or MB-group-) level dropout, not voxel-level
  artefacts or gross geometric failure.
- The threshold trades type-1 (discarding good data) against type-2 (missed
  corruption) errors, characterised with realistic simulations.

## Grounding in the project

Outlier-aware fitting is the third pillar nifreeze inherits from the `eddy`
family: reject/replace corrupted measurements before they bias the model fit.
The GP predictive variance (Eq. 8 in [[concept-gaussian-process-regression]])
is what makes the z-score's expected-error normalisation possible, so this
capability is cheapest to build on the GP model —
see [[andersson-eddy-framework-lineage]] and [[gp-prediction-underpins-lovo]].
