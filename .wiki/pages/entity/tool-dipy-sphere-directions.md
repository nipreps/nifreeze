---
title: "DIPY sphere & direction machinery (sphere, subdivide_octahedron, geometry)"
entity_type: tool
last_verified: 2026-07-13
confidence_score: 0.9
refresh_needed_if: "DIPY pin in pyproject.toml advances past 2ecd3655"
source_urls:
  - https://github.com/dipy/dipy/blob/2ecd3655/dipy/core/sphere.py
  - https://github.com/dipy/dipy/blob/2ecd3655/dipy/core/subdivide_octahedron.py
  - https://github.com/dipy/dipy/blob/2ecd3655/dipy/core/geometry.py
---

# DIPY sphere & direction machinery

The objects nifreeze uses to build diffusion-encoding direction sets (for phantom
simulation) and the discrete ODF sphere (inside the vendored GQI model), cached at
DIPY commit `2ecd3655`. Implements [[concept-sphere-sampling-electrostatic-repulsion]].
Facet of [[tool-dipy]].

## Spheres (`dipy.core.sphere`)

```python
Sphere(*, x=None, y=None, z=None, theta=None, phi=None, xyz=None,
       faces=None, edges=None)
HemiSphere(*, x=None, y=None, z=None, theta=None, phi=None, xyz=None,
           faces=None, edges=None, tol=1e-5)
HemiSphere.from_sphere(sphere, *, tol=1e-5)
```

- **Both constructors are fully keyword-only** at this pin (older DIPY allowed
  positional `Sphere(x, y, z)`). Supply exactly one coordinate spec: `(x, y, z)`,
  `(theta, phi)`, or `xyz`; otherwise `ValueError`.
- `HemiSphere` represents antipodally-symmetric points (folds vertices to one
  hemisphere), de-duplicating within `tol`.
- **nifreeze usage:** `HemiSphere(theta=..., phi=...)` to seed random directions,
  then `Sphere(xyz=np.vstack((v, -v)))` to mirror the dispersed hemisphere into a
  full sphere (`create_diffusion_encoding_gradient_dirs` in
  `testing/simulations.py`).

## Electrostatic repulsion (`disperse_charges`)

```python
disperse_charges(hemi, iters, *, const=0.2)   # -> (HemiSphere, potential)
```

- Returns a **2-tuple**: a new `HemiSphere` with charge-repulsion-distributed
  points, and a `(iters,)` array of the electrostatic potential per iteration.
  `hemi` must be a `HemiSphere`; `const` (keyword-only, default `0.2`) is the
  gradient-descent step scale.
- **nifreeze usage:** `hsph_updated, _ = disperse_charges(hsph_initial, iterations)`
  — it keeps only the sphere, discarding the potential trace. This is the
  acquisition-direction optimisation of
  [[concept-sphere-sampling-electrostatic-repulsion]] (Problem 1).

## Octahedral subdivision (`dipy.core.subdivide_octahedron`)

```python
create_unit_sphere(*, recursion_level=2)       # -> Sphere,     4**rl + 2 vertices
create_unit_hemisphere(*, recursion_level=2)   # -> HemiSphere, (4**rl + 2)/2 vertices
```

- **`recursion_level` is keyword-only**, default `2`, valid range `1..7`. Returns a
  deterministic near-uniform tessellation (Problem 2 of the concept page).
- **nifreeze usage:** the vendored GQI model defaults its ODF sphere to
  `create_unit_sphere(recursion_level=5)` → **1026 vertices**
  (`DEFAULT_SPHERE_RECURSION_LEVEL = 5`; see [[tool-dipy-reconst-models]]).

## Geometry helpers (`dipy.core.geometry`)

```python
sphere2cart(r, theta, phi)              # -> (x, y, z), physics convention
normalized_vector(vec, *, axis=-1)      # -> vec / ||vec||
```

- `sphere2cart` uses the **physics convention**: `theta` is inclination
  (co-latitude from +z), `phi` is azimuth. Plain positional signature (not
  keyword-only). nifreeze uses it to turn a fibre's polar angles into a Cartesian
  stick (`create_single_fiber_evecs`).
- `normalized_vector`'s `axis` is keyword-only (default `-1`). Test-only usage
  (b-vector normalisation).

## Failure modes on a pin bump

- The keyword-only migration means any nifreeze call that were positional would
  already fail; watch for *reversion* or further tightening.
- `create_unit_sphere`'s vertex-count formula ($4^{\ell}+2$) or default
  `recursion_level` changing would alter the GQI ODF resolution.
- `disperse_charges` return-shape change (tuple arity) would break the
  `hsph_updated, _ = ...` unpack.
