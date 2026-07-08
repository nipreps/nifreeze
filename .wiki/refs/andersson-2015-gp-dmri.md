---
title: "Andersson & Sotiropoulos (2015) — Non-parametric representation and prediction of DWI using Gaussian processes"
entity_type: paper
doi: 10.1016/j.neuroimage.2015.07.067
s2_id: b29a8160021f1485bffe63f9eb9933a83fed788a
citation_count_at_verify: 270
source_urls:
  - https://doi.org/10.1016/j.neuroimage.2015.07.067
  - https://europepmc.org/articles/PMC4627362
last_verified: 2026-07-08
confidence_score: 0.95
---

# Andersson & Sotiropoulos (2015) — Gaussian-process representation of DWI

Andersson, J. L. R. and Sotiropoulos, S. N. (2015). *Non-parametric
representation and prediction of single- and multi-shell diffusion-weighted MRI
data using Gaussian processes.* NeuroImage **122**:166–176.

## Why relevant

**Claim.** The diffusion-weighted signal at a voxel, viewed as a function on the
sphere of gradient directions, can be represented as a **Gaussian process (GP)**
whose covariance depends only on the angle between gradient directions (and, for
multi-shell data, on the b-value separation). This yields a *model-free*
predictor of what any measurement should look like given all the others — the
predictive mean and variance follow in closed form from GP regression
([[concept-gaussian-process-regression]], [[concept-dmri-angular-covariance]]).

**Evidence.** The paper demonstrates that a GP with the proposed angular
covariance predicts held-out diffusion volumes accurately even in voxels with
crossing fibres, outperforming the diffusion tensor as a predictor, and that a
cross-shell covariance lets one shell inform predictions in another. Model
selection (via the Laplace-approximated evidence, Eqs. 12–13) picks the
covariance form and its hyperparameters from the data itself.

**Location in the project.** This is the theoretical basis of nifreeze's
Gaussian-process model layer (`src/nifreeze/model/gpr.py`, `_dipy.py`). The
covariance functions and hyperparameter-estimation equations cited in the code
(Eqs. 9, 10, 14, 16) are transcribed in
[[concept-dmri-angular-covariance]]; the general GP machinery is in
[[concept-gaussian-process-regression]]. The predict-then-register loop it
enables is captured in [[gp-prediction-underpins-lovo]].

**Limits.** The GP is fit per-voxel (or per neighbourhood) and assumes the
signal is a smooth, antipodally-symmetric function of direction; sharp
angular features from very high b-value data may be under-fit. The predictor is
only as good as the alignment of the input volumes — it presupposes that motion
and distortion have (approximately) been removed, which is why it is used
iteratively inside a correction loop rather than once.
