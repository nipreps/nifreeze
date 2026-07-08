---
title: "Andersson, Graham, Zsoldos & Sotiropoulos (2016) — Incorporating outlier detection and replacement into a non-parametric framework for movement and distortion correction"
entity_type: paper
doi: 10.1016/j.neuroimage.2016.06.058
s2_id: 61a359b763d169e7c8988e9db401a709fa34bce6
citation_count_at_verify: 469
source_urls:
  - https://doi.org/10.1016/j.neuroimage.2016.06.058
  - https://nottingham-repository.worktribe.com/output/819688
last_verified: 2026-07-08
confidence_score: 0.95
---

# Andersson et al. (2016) — outlier detection and replacement

Andersson, J. L. R., Graham, M. S., Zsoldos, E. and Sotiropoulos, S. N. (2016).
*Incorporating outlier detection and replacement into a non-parametric
framework for movement and distortion correction of diffusion MR images.*
NeuroImage **141**:556–572.

## Why relevant

**Claim.** Whole-slice (or large connected-region) **signal dropout** caused by
bulk motion during diffusion encoding can be detected and replaced within the
same non-parametric framework used for distortion/motion correction. A slice is
flagged as an outlier when its observed intensity deviates from the GP
prediction by more than a data-derived threshold; flagged slices are replaced by
the GP prediction so they no longer corrupt the downstream estimate
([[concept-outlier-detection-replacement]]).

**Evidence.** Using highly-realistic simulations the paper characterizes
detection sensitivity/specificity (type-1 and type-2 error), the effect of
outliers on retrospective motion/distortion estimation, and the effect on
tensor-derived metrics (FA, MD); it validates on the Whitehall imaging
sub-study, and outperforms tensor-level robust estimators (RESTORE/iRESTORE,
[[chang-2005-restore-robust-estimation-tensors-outlier-rejection]]). Detection
aggregates per-voxel deviations $d_i = y_i - \hat y_i$ into a slice mean
$d_{gs}$ (Eq. 2), pools a voxel-wise variance $\sigma_d^2$ (Eq. 3), and forms a
$\sqrt{n_s}$-scaled z-score $z_{gs}$ (Eq. 4) thresholded at 3–4 s.d.; the exact
equations are transcribed in [[concept-outlier-detection-replacement]].

**Location in the project.** This is the third pillar of the `eddy` family that
nifreeze draws on: outlier-aware model fitting. Where nifreeze fits a model and
registers, this paper motivates rejecting/replacing corrupted measurements
before they bias the fit ([[andersson-eddy-framework-lineage]]). It extends
[[andersson-2016-integrated-eddy]] and reuses the predictor of
[[andersson-2015-gp-dmri]].

**Limits.** Detection targets slice- (or multiband-group-) level dropout from
spin-history/bulk-motion signal loss; it is not designed for voxel-level
artefacts or gross geometric errors. Replacement injects model-predicted values,
so a systematic model bias would propagate into the "corrected" slices.

*Provenance: full text read from the published NeuroImage article and the
authors' May-2016 preprint (both provided by the user); equations and algorithm
verified against the source rather than reconstructed.*
