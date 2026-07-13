---
title: "DIPY reconstruction models: TensorModel, DiffusionKurtosisModel, GQI"
entity_type: tool
last_verified: 2026-07-13
confidence_score: 0.9
refresh_needed_if: "DIPY pin in pyproject.toml advances past 2ecd3655"
source_urls:
  - https://github.com/dipy/dipy/blob/2ecd3655/dipy/reconst/dti.py
  - https://github.com/dipy/dipy/blob/2ecd3655/dipy/reconst/dki.py
  - https://github.com/dipy/dipy/blob/2ecd3655/dipy/reconst/gqi.py
---

# DIPY reconstruction models nifreeze uses

The concrete diffusion models nifreeze instantiates as signal predictors, cached
at DIPY commit `2ecd3655`. Each `implements` a theory concept; read this page for
the *code contract*, the concept page for the *algorithm*. Facet of [[tool-dipy]];
all three obey the [[tool-dipy-reconst-contract]].

## `TensorModel` — `dipy.reconst.dti.TensorModel`

```python
TensorModel(gtab, *args, fit_method="WLS", return_S0_hat=False, **kwargs)
fit(data, *, mask=None)                       # -> TensorFit
TensorFit.predict(gtab, *, S0=None, step=None)  # predicted signal at new directions
```

- Default fit is **weighted least squares** (`"WLS"`); `fit_method` may also be a
  callable `f(design_matrix, data, *args, **kwargs)`. `min_signal` is popped from
  `kwargs` (must be strictly positive if given).
- Implements [[concept-diffusion-tensor-imaging]]. This is nifreeze's default
  predictor (`DTIModel._model_class = "dipy.reconst.dti.TensorModel"`), used
  directly from DIPY. `DTIModel._modelargs = (min_signal, return_S0_hat,
  fit_method, weighting, sigma, jac)`.

## `DiffusionKurtosisModel` — `dipy.reconst.dki.DiffusionKurtosisModel`

```python
DiffusionKurtosisModel(gtab, *args, fit_method="WLS", return_S0_hat=False, **kwargs)
fit(data, *, mask=None)          # -> DiffusionKurtosisFit
multi_fit(data, *, mask=None, **kwargs)
predict(dki_params, *, S0=1.0)
```

- **Requires ≥3 b-values**: `__init__` runs `check_multi_b(self.gtab, 3,
  non_zero=False)` and raises `ValueError` otherwise (see
  [[tool-dipy-gradient-table]], [[concept-diffusion-kurtosis-imaging]]).
- Attributes nifreeze's subclass depends on, set from the *string* `fit_method`:
  `is_multi_method` (bool), `is_iter_method` (`True` iff `weights_method` in
  kwargs), `min_signal` (defaults to `MIN_POSITIVE_SIGNAL`, **not** `None`),
  `weights` (bool: `fit_method in {WLS, WLLS, CWLS}`), `kwargs`.
- **Caveat:** those attributes are computed from string method names. If a
  **callable** `fit_method` is passed, `is_multi_method`/`weights`/
  `convexity_constraint` all evaluate `False` and `self.fit_method` is left unset —
  a trap for the nifreeze subclass, which branches on `is_multi_method`/
  `is_iter_method`.
- nifreeze wraps this in `src/nifreeze/model/dki.py` (thin subclass adding a
  normalised `fit(data, *, mask=None, **kwargs)` that delegates to `multi_fit`
  when orchestration kwargs are present). `DKIModel._model_class =
  "nifreeze.model.dki.DiffusionKurtosisModel"`.

## `GeneralizedQSamplingModel` — vendored, **not** DIPY's

nifreeze resolves GQI to `nifreeze.model.gqi.GeneralizedQSamplingModel` — a
**vendored copy** of `dipy.reconst.gqi`, not the upstream class. The differences at
this pin are load-bearing (full detail in [[vendored-gqi-lineage]]):

| | upstream `dipy.reconst.gqi` @ `2ecd3655` | vendored `nifreeze.model.gqi` |
|---|---|---|
| base classes | `OdfModel, Cache` | plain `ReconstModel` |
| default `method` | `"gqi2"` | `"standard"` |
| default `sampling_length` | `1.2` | `1.2` |
| `predict` | **absent** | **present** (nifreeze extension: `prediction_kernel` via a regularised pseudo-inverse) |
| `odf` | `odf(sphere)` — sphere passed in | `odf()` — sphere stored on the model |
| sphere | caller supplies to `.odf` | `create_unit_sphere(recursion_level=5)` by default |

- Vendored signature: `GeneralizedQSamplingModel(gtab, *, method="standard",
  sampling_length=1.2, normalize_peaks=False, sphere=None)`. `GQIModel._modelargs =
  (method, sampling_length, normalize_peaks)`.
- Implements [[concept-generalized-q-sampling-imaging]]. The vendored kernel is
  `sinc(⟨b_vector, vertices⟩ · λ / π)` with `b_vector = bvecs · sqrt(bvals · 0.01506)`,
  where `0.01506 = 6·D_free-water` — the b-value form of the SDF quadrature (Eq. 9
  of the concept page).

## Common failure modes on a pin bump

- Rename/relocation of `dipy.reconst.dti.TensorModel` breaks nifreeze's
  dotted-string resolution (see [[tool-dipy-reconst-contract]]).
- Change to DKI's `is_multi_method`/`is_iter_method`/`weights`/`min_signal`
  attributes breaks nifreeze's DKI subclass.
- Upstream GQI drift (e.g. it gains a `predict`, or changes `method` default)
  widens the gap with the vendored copy — reconcile via [[vendored-gqi-lineage]].
