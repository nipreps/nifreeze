---
title: "Andersson & Sotiropoulos (2016) — An integrated approach to correction for off-resonance effects and subject movement in diffusion MR imaging"
entity_type: paper
doi: 10.1016/j.neuroimage.2015.10.019
s2_id: 8f8b7314fedc501c5ab5b38bcc44997873849fa9
citation_count_at_verify: 3420
source_urls:
  - https://doi.org/10.1016/j.neuroimage.2015.10.019
  - https://europepmc.org/articles/PMC4692656
last_verified: 2026-07-08
confidence_score: 0.95
---

# Andersson & Sotiropoulos (2016) — the `eddy` framework

Andersson, J. L. R. and Sotiropoulos, S. N. (2016). *An integrated approach to
correction for off-resonance effects and subject movement in diffusion MR
imaging.* NeuroImage **125**:1063–1078. (The method distributed as FSL `eddy`.)

## Why relevant

**Claim.** Eddy-current-induced distortion, susceptibility-induced distortion,
and subject movement can be estimated **jointly and retrospectively** by
registering each observed diffusion volume to a *model-free prediction* of what
that volume should look like — the GP prediction of
[[andersson-2015-gp-dmri]]. Because the prediction accounts for the expected
contrast of each volume, registration works even at high b-value where volumes
look nothing alike. The three fields are combined in a physically-correct way:
the susceptibility field is fixed in scanner space while the eddy field moves
with the subject ([[concept-epi-off-resonance-distortion]],
[[concept-eddy-current-distortion]], [[concept-rigid-body-motion]]).

**Evidence.** On high angular/spatial-resolution data (HCP, 3 T and 7 T
Siemens; Whitehall), the paper shows a linear eddy-current model is insufficient
and a quadratic model performs significantly better, that registering to the GP
prediction beats pairwise `eddy_correct`, and that a "second-level" GLM relating
eddy parameters to gradient direction both regularizes the estimate and reveals
scanner-specific systematic fields. The forward model, resampling with Jacobian
intensity modulation (Eqs. 4–5), and the Gauss–Newton update (Eqs. 6–7) are
given explicitly.

**Location in the project.** This is the methodological template nifreeze
generalizes: the *predict → register-each-volume → update* loop is nifreeze's
`Estimator.run` orchestration over a model plus ANTs registration
([[gp-prediction-underpins-lovo]], [[andersson-eddy-framework-lineage]]). The
distortion/motion parameterizations are captured in
[[concept-image-registration]] and the concept pages above.

**Limits.** It requires enough well-distributed gradient directions to
constrain the second-level model, assumes distortions are smooth and
representable by low-order spatial polynomials, and (in its original form)
models eddy fields as static per volume rather than time-varying within the
readout. Susceptibility field must be supplied from elsewhere (e.g. `topup`).
