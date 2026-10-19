---
title: "DIPY version-pin fragility and the wiki-cache sync procedure"
entity_type: synthesis
last_verified: 2026-10-19
confidence_score: 0.9
derived_from:
  - pages/entity/tool-dipy.md
  - pages/entity/tool-dipy-reconst-contract.md
  - pages/entity/tool-dipy-gradient-table.md
---

# DIPY version-pin fragility and the wiki-cache sync procedure

**Technical claim.** nifreeze combines two choices that are each reasonable but
together create a **silent-break surface**: (1) it accepts any DIPY release from
an *open-ended floor* (`dipy>=1.12.1`), and (2) it resolves its diffusion models by **dotted-string
import** at fit time ([[tool-dipy-reconst-contract]]). An upstream refactor that
renames, relocates, or re-signatures any consumed symbol breaks nifreeze **at model
instantiation**, not at import — so a green test collection can still fail deep in
the estimator loop. This page states the risk and defines how the DIPY tool-page
*cache* is kept in sync when the floor moves.

## Why the risk is real

- **Open-ended floor.** `dipy>=1.12.1` admits every future release unreviewed. At
  `1.12.1` DIPY has already moved most optional arguments to **keyword-only** (a
  bare `*` in signatures), a category of change that breaks positional callers
  silently.
- **No git pins.** An unreleased commit cannot be the requirement: PyPI refuses URL
  dependencies, so the `2ecd3655` pin (2026-07-09 to 2026-10-19) failed every
  package upload (nifreeze#504). An unreleased fix is backported in nifreeze
  instead, behind a feature check, with a test that fails once a DIPY release
  ships it.
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
surface *at the floor*. Their value is diagnostic: when the floor moves, the cache
is the checklist of exactly what to re-verify. Each tool page therefore:

- embeds the floor's release tag `1.12.1` in its `source_urls` blob paths, making
  the floor machine-diffable and each transcribed signature click-through-able; and
- declares `refresh_needed_if: "DIPY floor in pyproject.toml rises above 1.12.1"`,
  which suppresses the 180-day staleness warning. That is correct, because a tool
  page's truth is keyed to the floor, not the calendar.

## Sync procedure (run when the floor rises)

1. **Detect.** A PR raises the `dipy>=` floor in `pyproject.toml`, for example
   when `test_dipy_orchestration_backport_still_needed` fails on a new release.
   The `refresh_needed_if` predicate fires: the new floor no longer matches the
   `1.12.1` embedded in the tool pages' `source_urls`.
2. **Diff.** Compare the consumed surface across the two releases:
   ```
   git diff 1.12.1..<newfloor> -- \
     dipy/core/gradients.py dipy/reconst/base.py dipy/reconst/dti.py \
     dipy/reconst/dki.py dipy/reconst/gqi.py dipy/reconst/multi_voxel.py \
     dipy/core/sphere.py dipy/core/subdivide_octahedron.py \
     dipy/sims/voxel.py dipy/core/geometry.py
   ```
   against each tool page's documented signatures and the vendored GQI copy.
3. **Update.** Revise any drifted tool page, bump its `last_verified`, and replace
   `1.12.1` with the new floor in every `source_urls` entry and in the schema
   addendum note. If a symbol nifreeze imports changed signature/location, open a
   `/wiki-flags` entry (kind `content`) flagging the potential silent break of the
   dotted-string resolution, and cross-check `src/nifreeze/model/{dmri,dki,gqi,_dipy}.py`
   and `src/nifreeze/testing/simulations.py`.
4. **Log.** Append a `## YYYY-MM-DD — DIPY sync — floor 1.12.1→<newfloor>` block to
   `log.md` summarising what changed.

## Standing recommendations for the code (not the wiki)

Out of scope for the cache but worth noting as pointers: converting the positional
`gradient_table_from_bvals_bvecs` call to keyword arguments, and adding an
import-time smoke check that each `_model_class` string resolves, would convert two
silent-break modes into loud ones. These are code changes for nifreeze, recorded
here only as the fragility this cache exists to contain.
