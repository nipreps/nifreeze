---
title: "DIPY version-pin fragility and the wiki-cache sync procedure"
entity_type: synthesis
last_verified: 2026-07-13
confidence_score: 0.9
derived_from:
  - pages/entity/tool-dipy.md
  - pages/entity/tool-dipy-reconst-contract.md
  - pages/entity/tool-dipy-gradient-table.md
---

# DIPY version-pin fragility and the wiki-cache sync procedure

**Technical claim.** nifreeze combines two choices that are each reasonable but
together create a **silent-break surface**: (1) it pins DIPY to an *unreleased git
commit* (`2ecd3655`), and (2) it resolves its diffusion models by **dotted-string
import** at fit time ([[tool-dipy-reconst-contract]]). An upstream refactor that
renames, relocates, or re-signatures any consumed symbol breaks nifreeze **at model
instantiation**, not at import — so a green test collection can still fail deep in
the estimator loop. This page states the risk and defines how the DIPY tool-page
*cache* is kept in sync when the pin moves.

## Why the risk is real

- **Unreleased pin.** `dipy @ git+https://github.com/dipy/dipy.git@2ecd3655` tracks
  a moving upstream, not a stability-guaranteed release. At this commit DIPY has
  already moved most optional arguments to **keyword-only** (a bare `*` in
  signatures) — a category of change that breaks positional callers silently.
- **String resolution.** `getattr(import_module(module_name), class_name)(gtab,
  **kwargs)` means the strings `dipy.reconst.dti.TensorModel` etc. are effectively
  untyped, uncheckable references until executed.
- **Positional hot-path call.** nifreeze calls
  `gradient_table_from_bvals_bvecs(bvals, bvecs)` positionally on every fit/predict
  ([[gradient-table-interop-hotpath]]); a positional-order change would mis-map
  b-values and b-vectors without raising.
- **Vendored fork.** The GQI copy ([[vendored-gqi-lineage]]) never receives upstream
  fixes automatically.

## The wiki cache as a mitigation

The DIPY tool pages ([[tool-dipy]] and facets) are a **cache** of the consumed
surface *at the pin*. Their value is diagnostic: when the pin moves, the cache is
the checklist of exactly what to re-verify. Each tool page therefore:

- embeds the commit SHA `2ecd3655` in its `source_urls` blob paths, making the pin
  machine-diffable and each transcribed signature click-through-able; and
- declares `refresh_needed_if: "DIPY pin in pyproject.toml advances past 2ecd3655"`,
  which suppresses the 180-day staleness warning — correct, because a tool page's
  truth is keyed to the pin, not the calendar.

## Sync procedure (run when the pin advances)

1. **Detect.** A PR bumps the `dipy @ …@<sha>` pin in `pyproject.toml`. The
   `refresh_needed_if` predicate fires: the new SHA no longer matches the
   `2ecd3655` embedded in the tool pages' `source_urls`.
2. **Diff.** Compare the consumed surface across the two commits:
   ```
   git diff 2ecd3655..<newpin> -- \
     dipy/core/gradients.py dipy/reconst/base.py dipy/reconst/dti.py \
     dipy/reconst/dki.py dipy/reconst/gqi.py \
     dipy/core/sphere.py dipy/core/subdivide_octahedron.py \
     dipy/sims/voxel.py dipy/core/geometry.py
   ```
   against each tool page's documented signatures and the vendored GQI copy.
3. **Update.** Revise any drifted tool page, bump its `last_verified`, and replace
   `2ecd3655` with the new SHA in every `source_urls` entry and in the schema
   addendum note. If a symbol nifreeze imports changed signature/location, open a
   `/wiki-flags` entry (kind `content`) flagging the potential silent break of the
   dotted-string resolution, and cross-check `src/nifreeze/model/{dmri,dki,gqi,_dipy}.py`
   and `src/nifreeze/testing/simulations.py`.
4. **Log.** Append a `## YYYY-MM-DD — DIPY sync — pin 2ecd3655→<newpin>` block to
   `log.md` summarising what changed.

## Standing recommendations for the code (not the wiki)

Out of scope for the cache but worth noting as pointers: converting the positional
`gradient_table_from_bvals_bvecs` call to keyword arguments, and adding an
import-time smoke check that each `_model_class` string resolves, would convert two
silent-break modes into loud ones. These are code changes for nifreeze, recorded
here only as the fragility this cache exists to contain.
