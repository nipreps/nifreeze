---
title: "DIPY signal simulation API (dipy.sims.voxel)"
entity_type: tool
last_verified: 2026-07-13
confidence_score: 0.9
refresh_needed_if: "DIPY pin in pyproject.toml advances past 2ecd3655"
source_urls:
  - https://github.com/dipy/dipy/blob/2ecd3655/dipy/sims/voxel.py
  - https://github.com/dipy/dipy/blob/2ecd3655/dipy/core/geometry.py
---

# DIPY signal simulation API (`dipy.sims.voxel`)

The forward-model functions nifreeze uses to synthesise ground-truth DWI signal
for validating motion estimation, cached at DIPY commit `2ecd3655`. Implements
[[concept-multi-tensor-signal-simulation]]. Facet of [[tool-dipy]].

## `single_tensor`

```python
single_tensor(gtab, S0=1, *, evals=None, evecs=None, snr=None, rng=None)   # -> (N,) signal
```

- Implements the single-compartment Stejskal–Tanner model
  `S(b, g) = S0 · exp(-b · gᵀ (R D Rᵀ) g)`, with `D = diag(evals)` and
  `R = evecs` (or a full b-tensor contraction `-B:D` when `gtab.btens` is set).
- Defaults: `S0=1`; `evals` → a prolate white-matter default when `None`;
  `evecs` → `np.eye(3)`; `snr=None` (noise-free, else Rician noise via `add_noise`);
  `rng=None` → a fresh `np.random.default_rng()`. **`evals/evecs/snr/rng` are
  keyword-only.** Uses a `np.random.Generator` (`rng`), not the legacy global.
- **nifreeze usage:** `single_tensor(gtab, S0=S0, evals=evals, evecs=evecs,
  snr=snr, rng=rng)` for one-fibre voxels.

## `multi_tensor`

```python
multi_tensor(gtab, mevals, *, S0=1.0, angles=((0, 0), (90, 0)),
             fractions=(50, 50), snr=20, rng=None)   # -> (signal, sticks)
```

- Returns a **2-tuple** `(S, sticks)`: `S` is the `(N,)` volume-fraction-weighted
  sum of per-compartment `single_tensor` signals (noise added once at the end);
  `sticks` is the `(M, 3)` array of compartment directions. Raises `ValueError`
  unless `fractions` sum to **100**.
- **Gotcha:** `S0` default is **`1.0`**, not `100` as older docs show; `mevals` is a
  required positional `(K, 3)` eigenvalue array; `S0/angles/fractions/snr/rng` are
  keyword-only (default `snr=20`).
- **nifreeze usage:** `multi_tensor(gtab, evals, S0=S0, angles=angles,
  fractions=fractions, snr=snr, rng=rng)[0]` — it keeps only the signal, discarding
  `sticks`, and passes `fractions` scaled to sum to 100.

## `all_tensor_evecs`

```python
all_tensor_evecs(e0)   # -> (3, 3) eigenvector frame (columns)
```

- Given a principal axis `e0` (3,), builds an orthonormal eigenvector frame by
  rotating the canonical axes onto `e0` (`vec2vec_rotmat`), returning the three
  eigenvectors **column-wise**. nifreeze feeds it a Cartesian stick from
  `sphere2cart` to get a single fibre's `evecs` (`create_single_fiber_evecs`).

## Failure modes on a pin bump

- `multi_tensor`'s `(signal, sticks)` return-tuple order/arity, or the `S0`/
  `fractions`-sum-to-100 conventions, changing would break nifreeze's `[0]` unpack
  and fraction scaling.
- `single_tensor` switching noise model or the `rng` interface would change
  simulated-phantom statistics used across the test suite.
- `all_tensor_evecs` returning row-wise instead of column-wise vectors would rotate
  every simulated fibre — a silent correctness bug in ground-truth generation.
