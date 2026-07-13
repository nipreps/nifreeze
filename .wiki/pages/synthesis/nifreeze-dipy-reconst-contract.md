---
title: "nifreeze implements DIPY's ReconstModel fit/predict contract"
entity_type: synthesis
last_verified: 2026-07-13
confidence_score: 0.9
derived_from:
  - pages/entity/tool-dipy-reconst-contract.md
  - pages/entity/tool-dipy-reconst-models.md
  - pages/entity/concept-leave-one-volume-out.md
---

# nifreeze implements DIPY's ReconstModel fit/predict contract

**Technical claim.** nifreeze standardises *every* diffusion signal predictor onto
DIPY's `ReconstModel` → `ReconstFit` **fit-then-predict** contract
([[tool-dipy-reconst-contract]]), so that the estimator loop can treat a tensor
fit, a kurtosis fit, a q-space model, a Gaussian process, and a trivial average
through one uniform interface. This is the design decision that lets nifreeze's
`BaseDWIModel` be model-agnostic.

## How the standardisation works

`BaseDWIModel._fit` builds a DIPY `GradientTable` from the training volumes,
resolves the model class from a **dotted string** (`_model_class`), instantiates it
as `Model(gtab, **kwargs)`, and calls DIPY's `model.fit(data)`. `fit_predict` then
constructs a one-row gradient table for the held-out volume's $(\mathbf{g}_k, b_k)$
and calls `fit.predict(gtab=…, S0=…)`. Both halves are exactly the DIPY contract.

Crucially, nifreeze extends the same contract to models DIPY does not provide:

- `GaussianProcessModel` (`src/nifreeze/model/_dipy.py`) subclasses DIPY's
  `ReconstModel` **specifically** to expose `fit`/`predict`, so the GP predictor
  drops into the loop unchanged.
- The vendored GQI model ([[tool-dipy-reconst-models]], [[vendored-gqi-lineage]])
  adds a `predict` that upstream DIPY's GQI lacks, precisely to satisfy this
  contract.
- Trivial/average models present the same signature.

## Why the contract matters for correction

The predictor's job in [[concept-leave-one-volume-out]] is to synthesise the
expected (motion-free) appearance of the left-out volume. Because every predictor
speaks the same `fit`/`predict` language, the estimator can **swap models without
touching the registration loop** — the model is a plug-in, chosen per dataset
(DTI by default, DKI for multi-shell, GQI/GP for richer angular content). The
uniform contract is what makes nifreeze's Data → Model → Estimator separation real
rather than nominal.

## The cost of leaning on the contract

The contract is a *convention*, not an enforced abstract base class: DIPY's
`ReconstModel.fit` is a concrete stub and there is **no `predict` on the base
class** ([[tool-dipy-reconst-contract]]). Conformance is therefore checked only at
call time, and the models are wired in by dotted string, so a mismatch surfaces as
a runtime error inside the estimator, not at import. That fragility is analysed in
[[dipy-version-pin-fragility]]; the per-fit rebuild cost it implies is in
[[gradient-table-interop-hotpath]].
