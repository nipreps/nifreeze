---
title: "DIPY gradient-table API (dipy.core.gradients)"
entity_type: tool
last_verified: 2026-07-13
confidence_score: 0.9
refresh_needed_if: "DIPY pin in pyproject.toml advances past 2ecd3655"
source_urls:
  - https://github.com/dipy/dipy/blob/2ecd3655/dipy/core/gradients.py
  - https://docs.dipy.org/stable/reference/dipy.core.html
---

# DIPY gradient-table API (`dipy.core.gradients`)

The `GradientTable` is DIPY's canonical description of a diffusion-encoding scheme
and the object nifreeze converts *into* every time it hands data to a diffusion
model. This page caches the contract at DIPY commit `2ecd3655`; the theory it
encodes is [[concept-diffusion-mri-signal]]. Facet of [[tool-dipy]].

## `GradientTable`

```python
GradientTable(gradients, *, big_delta=None, small_delta=None,
              b0_threshold=50, btens=None)
```

Constructor takes a single positional `gradients` array of shape **(N, 3)** — the
gradient directions already **scaled by their b-value** (raises `ValueError` if
not 2-D with 3 columns). Everything else is keyword-only. Derived quantities are
lazily cached (`@auto_attr`) on first access:

| attribute | meaning |
|---|---|
| `bvals` | (N,) — row-wise L2 norms of `gradients` (b-values recovered from the scaled directions) |
| `bvecs` | (N, 3) — `gradients / bvals` (unit directions; divides by 1 where `bval == 0`) |
| `b0s_mask` | (N,) bool — `bvals <= b0_threshold`. **Note the `<=`**, and the default `b0_threshold = 50` |
| `gradients` | (N, 3) — the stored scaled gradients |

**Gotcha (b0 threshold).** Any volume with b-value $\le 50$ counts as a b0. A
"b=0" acquired with a small residual b (e.g. 5–20) is therefore treated as
non-weighted; a genuinely low but weighted shell below 50 would be misclassified.

## Factories

```python
gradient_table(bvals, *, bvecs=None, big_delta=None, small_delta=None,
               b0_threshold=50, atol=1e-2, btens=None)          # -> GradientTable

gradient_table_from_bvals_bvecs(bvals, bvecs, *, b0_threshold=50,
                                atol=1e-2, btens=None, **kwargs)  # -> GradientTable
```

- `gradient_table` is the general factory; at this pin **`bvecs` is keyword-only**
  (older DIPY allowed `gradient_table(bvals, bvecs)` positionally — a breaking
  change to watch on the next pin bump). It validates and can also read file paths
  or an (N,4) b-table.
- `gradient_table_from_bvals_bvecs` is the lower-level factory: `bvals` **and**
  `bvecs` are both positional, everything else keyword-only. It validates b-vectors
  are unit-norm within `atol`, warns if `b0_threshold >= 200`, and sets
  `b0s_mask = ~(bvals > b0_threshold)`.

**How nifreeze uses these.** nifreeze stores gradients internally as a plain
`(N, 4)` array (columns `0:3` = b-vectors, column `3` = b-values) and rebuilds a
DIPY `GradientTable` on demand by calling
`gradient_table_from_bvals_bvecs(gradients[:, -1], gradients[:, :-1])`
**positionally** — so it is insulated from the `gradient_table` keyword-only
change, but *not* from any change to `gradient_table_from_bvals_bvecs`'s positional
order. This conversion happens at every leave-one-volume-out fit and every
per-volume prediction — the hot path documented in
[[gradient-table-interop-hotpath]].

## Validation helpers

```python
check_multi_b(gtab, n_bvals, *, non_zero=True, bmag=None)          # -> bool
unique_bvals_magnitude(bvals, *, bmag=None, rbvals=False)          # -> ndarray (or tuple)
```

- `check_multi_b` returns `True` iff the table has at least `n_bvals` distinct
  (magnitude-rounded) b-values. With `non_zero=True` (default) it first drops b0s,
  counting only weighted shells; with `non_zero=False` it counts **all** b-values
  including $b=0$. `bmag` controls the rounding order of magnitude.
- **nifreeze's DKI gate** calls `check_multi_b(gtab, 3, non_zero=False)`: DKI needs
  three distinct b-value levels, and $b=0$ is allowed to be one of them (see
  [[concept-diffusion-kurtosis-imaging]]). This runs both at `DKIModel`
  construction and inside DIPY's own `DiffusionKurtosisModel.__init__`.
- `unique_bvals_magnitude` returns the sorted unique rounded b-values (used by
  nifreeze's tests to count shells).

## Failure modes to watch on a pin bump

1. Positional-order change to `gradient_table_from_bvals_bvecs(bvals, bvecs, ...)`
   would silently mis-map nifreeze's positional call.
2. A change to the `b0s_mask` threshold semantics (`<=` vs `<`, or the default 50)
   would shift which volumes count as b0, affecting the b0 append and S0 handling.
3. `check_multi_b`'s `non_zero` default or signature changing would move the DKI
   admissibility boundary.
