# Wiki Log

Append-only record of `/wiki-*` skill runs against this wiki. Each block is a
level-2 heading of the form `## YYYY-MM-DD — /<skill-name> — <summary>`
followed by bullets. See [`schema.md`](schema.md) for the format contract
(W006).

## 2026-07-08 — /wiki-init — bootstrapped empty wiki
- created `.wiki/{schema.md, index.md, _MAP.md, log.md}` from `~/.claude/skills/wiki-init/templates/`
- created `.wiki/pages/{entity,synthesis}/` and `.wiki/{refs,_inbox}/` with `.gitkeep` placeholders

## 2026-07-08 — wiki-init expansion — theory-grounding scope + Andersson first-expansion
- added nifreeze per-project addendum to `schema.md` Scope (theory-grounding KB; no code/docs; `namespace: paper` for concepts)
- seeded 3 Andersson reference pages (2015 GP `b29a8160…`; 2016 integrated eddy `8f8b7314…`; 2016 outlier `61a359b7…`), resolved via `/s2-query`
- captured equations from open-access full text (Europe PMC MathML → LaTeX): GP paper PMC4627362 (Eqs. 1–16, A1–A8, B1–B2); integrated-eddy paper PMC4692656 (Eqs. 1–7, A1–A3, B1–B6). Outlier paper had no open PMC full text — captured from abstract + framework (confidence 0.8, `refresh_needed_if` set)
- wrote 8 concept entity pages (dMRI signal, GP regression, angular covariance, EPI/off-resonance, eddy-current, rigid-body motion, image registration, outlier detection) and 2 synthesis pages (GP↔LOVO; eddy-framework lineage)
- cross-checked Eqs. 9/10/14/16 against `src/nifreeze/model/gpr.py`

## 2026-07-08 — wiki-init expansion — reference investigation
- pulled reference lists of all 3 seed papers from the S2 graph API (gp2015: 32; eddy2016: 74; outlier2016: 60); **deduplicated to 134 unique** cited works (22 cited by 2 of the 3, 5 by all 3)
- created 134 `refs/*.md` pages from S2 metadata (title/authors/venue/DOI/`s2_id`/citation count + TL;DR/abstract as Claim); 20 without DOI use `status: doi-not-found`; confidence 0.4–0.6 (metadata-verified, not read in full)
- wired `index.md` (seeds + 134 cited works) and `_MAP.md` (166 `cites` edges + 13 framework `extends`/`informs`/`depends_on` edges)
- no refs dropped or truncated

## 2026-07-08 — /wiki-inbox-sweep — full-text intake of the outlier paper
- consumed `_inbox/1-s2.0-S1053811916303068-main.pdf` (published NeuroImage version) and `_inbox/olr.pdf` (authors' May-2016 preprint), both of Andersson et al. (2016) outlier detection & replacement
- extracted exact detection equations from the preprint (Eqs. 1–4: GP prediction, slice-mean deviation $d_{gs}$, pooled variance $\sigma_d^2$, $\sqrt{n_s}$-scaled z-score $z_{gs}$); rewrote [[concept-outlier-detection-replacement]] to full fidelity (scan-space detection, MB-group vs slice-wise, order-of-operations, RESTORE comparison)
- upgraded `refs/andersson-2016-outlier-replacement.md`: confidence 0.8→0.95, removed `refresh_needed_if`, added source provenance
- **reference reconciliation**: parsed the paper's 55-entry bibliography from the PDF and cross-checked against `refs/`; all entries already covered (6 initial mismatches were year-parse artefacts — Rasmussen 2003/2006, Van Essen→essen-2013, Vu 2015, Zwiers 2010, Graham 2016 — or a sentence fragment). No new refs needed.
- deleted both consumed PDFs from `_inbox/` (content fully captured; papers are published/retrievable)

## 2026-07-10 — wiki-expansion — LOVO validity + single-fit admissibility
- added entity page [[concept-leave-one-volume-out]] stating the **held-out independence invariant**: the target for volume $k$ must not depend on $k$'s own signal; violating it (single-fit on a data-driven model) is a self-registration biased toward the identity ("no motion detected"), a circularity rather than a numerical approximation
- added synthesis page [[single-fit-mode-admissibility]]: single-fit is **not** a lossless $\times N$ speed-up, but IS admissible for target-independent references (`TrivialModel`), development/CI/integration tests, and coarse low-DOF initialisation; the real DKI levers (exact closed-form LOO for linear WLS fits, voxel-chunk parallelism, BLAS control) are captured as a flagged method note
- cross-linked [[gp-prediction-underpins-lovo]] to the new concept; registered both pages in `index.md`; wired 5 `_MAP.md` edges (`informs`/`depends_on`)
- filed flag on the manuscript/wiki gap (held-out independence must be stated explicitly; exact LOO exists for the linear models)

## 2026-07-12 — /wiki-inbox-sweep — GQI deep expansion from dropped PDF
- consumed `_inbox/Generalized__q-Sampling_Imaging.pdf` (Yeh, Wedeen & Tseng, IEEE TMI 2010; DOI 10.1109/TMI.2010.2045126), read in full
- wrote deep concept page [[concept-generalized-q-sampling-imaging]] transcribing equations 1–12 verbatim (Fourier foundation, cosine transform, SDF, GQI sinc quadrature Eq. 6, L² basis Eq. 8, diffusion sampling length, b-value form Eq. 9, balanced requirement Eq. 10, QA Eq. 11, simulation model Eq. 12) plus SDF-vs-ODF distinction, QBI/DSI relation, and validation status
- added ref page `refs/yeh-2010-generalized-q-sampling-imaging.md` (confidence 0.9, primary source read) and a metadata stub `refs/tuch-2004-q-ball-imaging.md` (QBI, referenced as the L→∞ limit of GQI)
- added synthesis page [[q-space-reconstruction-landscape]] (DSI vs QBI vs GQI: what each reconstructs, sampling, core operation, regularizer; GQI as unifier; inter-voxel SDF comparability)
- wired `index.md` (1 entity, 1 synthesis, 2 refs) and `_MAP.md` (6 edges: `informs`/`extends`/`depends_on`/`cites`)
- parked the source PDF at `.wiki/_sources/yeh-2010-generalized-q-sampling-imaging.pdf` and added `.wiki/_sources/` to `.gitignore` (verified via `git check-ignore`); `_inbox/` cleared

## 2026-07-10 — /wiki-flags — opened F0001 (content)
- summary: State explicitly that held-out independence is load-bearing for LOVO validity (target for volume k must not depend on k's own signal); note exact closed-form LOO exists for the linear DTI/DKI WLS fits.
- target: pages/entity/concept-leave-one-volume-out.md

## 2026-07-13 — correction — single-fit locks the *fit*, not the *prediction*
- fixed a misconception (surfaced in PR #454 review) that single-fit "reuses one locked prediction / the same volume for every index": single-fit locks the **fit** (shared parameters), but data-driven models (DTI/DKI/GQI/GP) still evaluate a **distinct** per-index prediction at each volume's $(\mathbf{g}_k, b_k)$; only target-independent models (`TrivialModel`) return the same volume for all indices. The bias is from the queried volume having informed the fit (not out-of-sample), not from an identical target.
- reworded [[single-fit-mode-admissibility]] (L13-15 technical claim + added clarifying sentence; L40/L44/L53 "locked prediction/target" → "locked fit"); bumped `last_verified` to 2026-07-13. Argument structure and admissibility cases unchanged; [[concept-leave-one-volume-out]] and [[gp-prediction-underpins-lovo]] were already correct and untouched.
- same correction applied outside the wiki (PR #454): `docs/design.rst`, `docs/usage.rst`, `src/nifreeze/model/base.py`, `src/nifreeze/estimator.py`, `src/nifreeze/cli/parser.py` (×2).

## 2026-07-13 — wiki-expansion — DIPY dependency cache (pinned to commit 2ecd3655)
- extended `schema.md` scope addendum with a **dependency-integration knowledge** clause: a dependency's API surface is in scope as `entity_type: tool` pages (pinned revision + `refresh_needed_if` + upstream `source_urls`), distinct from nifreeze's own code (still out of scope); no `schema_version` bump (`tool` already legal)
- mapped nifreeze's full DIPY footprint from `src/nifreeze/model/{dmri,dki,gqi,_dipy}.py` and `src/nifreeze/testing/simulations.py`; DIPY pinned in `pyproject.toml` to git commit `2ecd3655` (full SHA `2ecd365584cc344c61e3543d8d7d4a498739a51c`)
- added 2 refs via `/s2-query`: `garyfallidis-2014-dipy` (DOI 10.3389/fninf.2014.00008, s2 `a89d23db…`, 1389 cites) and `jensen-2005-diffusional-kurtosis` (DOI 10.1002/mrm.20508, s2 `62f5553e…`, 2424 cites)
- wrote 6 **tool** entity pages (`tool-dipy` umbrella + gradient-table, reconst-contract, reconst-models, sphere-directions, sims-voxel), transcribing verbatim signatures fetched from DIPY source at the pinned SHA; captured the pervasive keyword-only migration, `b0s_mask` `<=50`, DKI `check_multi_b(gtab,3,non_zero=False)` gate, and the upstream-vs-vendored GQI divergence (upstream default `method="gqi2"`, no `predict`; vendored default `"standard"`, adds `predict`)
- wrote 4 **concept** entity pages (Layer-A theory): DTI (Basser 1994), DKI (Jensen 2005), sphere-sampling electrostatic-repulsion vs octahedral-subdivision (Jones 1999), multi-tensor signal simulation
- wrote 4 **synthesis** pages: nifreeze↔ReconstModel contract, GradientTable interop hot-path, vendored-GQI lineage, and DIPY version-pin fragility (which **hosts the cache-sync procedure**)
- wired `index.md` (4 concepts + a DIPY tool-page subsection + 4 synthesis + a DIPY refs group) and `_MAP.md` (28 edges, legal enum only; `implements`/`depends_on`/`informs`/`replicates`/`critiques`); facet→umbrella grouping via prose + index (no illegal `part_of`)
- cache maintenance: each tool page carries `refresh_needed_if: "DIPY pin in pyproject.toml advances past 2ecd3655"` (suppresses W001) and `source_urls` embedding the SHA; no new `/wiki-*` skill
