---
title: "DIPY ReconstModel / ReconstFit contract (dipy.reconst.base)"
entity_type: tool
last_verified: 2026-07-13
confidence_score: 0.9
refresh_needed_if: "DIPY pin in pyproject.toml advances past 2ecd3655"
source_urls:
  - https://github.com/dipy/dipy/blob/2ecd3655/dipy/reconst/base.py
  - https://docs.dipy.org/stable/reference/dipy.reconst.html
---

# DIPY `ReconstModel` / `ReconstFit` contract (`dipy.reconst.base`)

The two-object **fit → predict** interface every DIPY diffusion model presents,
and the contract nifreeze standardises *all* of its signal predictors on. This
page caches the contract at DIPY commit `2ecd3655`. Facet of [[tool-dipy]].

## The base classes

```python
class ReconstModel:
    def __init__(self, gtab): ...  # stores self.gtab = gtab
    def fit(self, data, *, mask=None, **kwargs):  # concrete stub -> ReconstFit(self, data)
        ...


class ReconstFit:
    def __init__(self, model, data): ...  # self.model, self.data
```

Two facts about this pinned version matter:

1. **`fit`/`predict` are not abstract.** `ReconstModel.fit` is a concrete stub
   returning a bare `ReconstFit(self, data)`, and **the base class defines no
   `predict` at all**. The real interface is a *convention* subclasses implement,
   not an enforced ABC. A subclass that forgets `predict` fails only at call time.
2. **A model is constructed from a `GradientTable`** (`__init__(self, gtab)`), fit
   to voxel `data`, and the resulting fit object predicts signal for a *new*
   gradient table.

## The de-facto contract nifreeze relies on

Across DIPY's models the shape is:

```python
model = SomeModel(gtab, **model_kwargs)  # construct from a gradient table
fit = model.fit(data, mask=None)  # fit to (voxels × directions) signal
pred = fit.predict(gtab_new, S0=...)  # predict signal at new directions
```

nifreeze's `BaseDWIModel` treats this as the uniform interface for every
predictor — the DIPY tensor/kurtosis/GQI models **and** nifreeze's own
`GaussianProcessModel` and `TrivialModel`, which subclass `ReconstModel`
specifically to present the same API. The synthesis page
[[nifreeze-dipy-reconst-contract]] describes how nifreeze standardises on it.

## Dotted-string model resolution (`_model_class` / `_modelargs`)

nifreeze does not import its models directly; each `BaseDWIModel` subclass names
its model by a **dotted string** and nifreeze resolves it at fit time:

```python
module_name, class_name = model_str.rsplit(".", 1)
model = getattr(import_module(module_name), class_name)(gtab, **kwargs)
```

The `_model_class` strings in use:

| nifreeze model | `_model_class` | resolves to |
|---|---|---|
| `DTIModel` | `dipy.reconst.dti.TensorModel` | DIPY directly |
| `DKIModel` | `nifreeze.model.dki.DiffusionKurtosisModel` | nifreeze subclass of DIPY DKI |
| `GQIModel` | `nifreeze.model.gqi.GeneralizedQSamplingModel` | **vendored** DIPY GQI |
| `GPModel` | `nifreeze.model._dipy.GaussianProcessModel` | nifreeze's own `ReconstModel` |

`_modelargs` is the whitelist of constructor kwargs each model accepts (e.g. DTI:
`min_signal, return_S0_hat, fit_method, weighting, sigma, jac`; GQI:
`method, sampling_length, normalize_peaks`). See [[tool-dipy-reconst-models]].

**Why this is fragile.** Because resolution is by *string*, an upstream refactor
that renames or relocates `dipy.reconst.dti.TensorModel` — or changes the
`(gtab, **kwargs)` constructor convention — breaks nifreeze only when the model is
instantiated, with an `ImportError`/`AttributeError`/`TypeError` rather than at
import. This silent-break surface is the subject of
[[dipy-version-pin-fragility]].
