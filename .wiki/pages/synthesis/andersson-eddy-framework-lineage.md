---
title: "The Andersson eddy framework and nifreeze's lineage from it"
entity_type: synthesis
last_verified: 2026-07-08
confidence_score: 0.9
derived_from:
  - refs/andersson-2015-gp-dmri.md
  - refs/andersson-2016-integrated-eddy.md
  - refs/andersson-2016-outlier-replacement.md
---

# The Andersson eddy framework and nifreeze's lineage from it

**Technical claim.** nifreeze is a generalisation of the three-paper Andersson
framework — GP prediction, integrated distortion/motion correction, and outlier
replacement — re-expressed as a modular *Data → Model → Estimator* pipeline that
extends beyond dMRI to fMRI and PET. Understanding what each paper contributes,
and where nifreeze deliberately diverges, is the grounding for extending it.

## The three pillars

1. **Prediction** — [[andersson-2015-gp-dmri]] gives a model-free predictor of
   the diffusion signal on the sphere via Gaussian processes
   ([[concept-gaussian-process-regression]], [[concept-dmri-angular-covariance]]).
2. **Integrated correction** — [[andersson-2016-integrated-eddy]] registers each
   volume to that prediction to jointly estimate susceptibility distortion
   ([[concept-epi-off-resonance-distortion]]), eddy-current distortion
   ([[concept-eddy-current-distortion]]), and rigid-body motion
   ([[concept-rigid-body-motion]]), with correct field/motion composition and
   Jacobian intensity modulation ([[concept-image-registration]]).
3. **Robustness** — [[andersson-2016-outlier-replacement]] detects slice
   dropout against the prediction and replaces it
   ([[concept-outlier-detection-replacement]]).

Together these form FSL `eddy`. The load-bearing idea shared by all three is
"compare each observed volume to what a data-driven model says it should be."

## What nifreeze keeps and what it changes

| Aspect | Andersson `eddy` | nifreeze |
|---|---|---|
| Prediction target | GP (Eqs. 7–16) | GP **and** interchangeable models (DTI, DKI, average-DWI; B-spline for PET) behind one `fit_predict` contract |
| Alignment | bespoke Gauss–Newton (Eqs. 6–7) | ANTs registration per volume (well-validated engine) |
| Loop | predict → register → refit, iterated | `Estimator.run`, joblib-parallel, chainable via `prev_model` |
| Distortion model | susceptibility + eddy + motion, coupled | motion/artifact estimation; susceptibility handled upstream (NiPreps/SDCFlows) |
| Modality | diffusion only | dMRI, fMRI, PET (4D neuroimaging) |

The divergences are principled: swapping the bespoke solver for ANTs trades a
task-specific optimiser for a maintained, general one; abstracting the predictor
behind a model interface ([[gp-prediction-underpins-lovo]]) lets the same
estimator serve modalities the original framework never targeted.

## Implications for new models / large refactors

- A new model is admissible iff it can predict a left-out volume from the rest;
  it inherits the whole correction loop for free.
- Because the eddy field is fixed in scanner space while the susceptibility
  field moves with the subject ([[concept-eddy-current-distortion]],
  [[concept-epi-off-resonance-distortion]]), any distortion handling added to
  nifreeze must respect that frame distinction to stay physically correct.
- Outlier handling ([[concept-outlier-detection-replacement]]) depends on a
  predictor that also yields an expected error; GP provides it natively (Eq. 8),
  so robustness features are cheapest to build on the GP model.
