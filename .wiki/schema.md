# LLM-Wiki Schema (v1)

This document is the contract that every page in `.wiki/` must satisfy. It is
read by `/wiki-audit`, `/wiki-health`, `/wiki-flags`, `/wiki-refresh`,
`/wiki-inbox-sweep`, `/wiki-diff-since`, and `/wiki-reconcile-zotero`. The
audit script exits with code 2 if this file is missing — its presence is the
marker that "this repo has a wiki".

The schema is versioned (`schema_version: "1"`). Changes here must be
coordinated with the `/wiki-*` skill scripts.

---

## Directory layout

```
.wiki/
├── schema.md         (this file)
├── index.md          (curated navigation: every page that should not be an orphan)
├── _MAP.md           (relationship graph)
├── log.md            (append-only changelog of skill runs)
├── flags.md          (optional; created by /wiki-flags)
├── pages/
│   ├── entity/       (one .md per entity: person, initiative, event, paper, …)
│   └── synthesis/    (multi-source claims that cite ≥1 entity or ref)
├── refs/             (bibliographic entries; one .md per cited work)
└── _inbox/           (unstructured human drops; consolidated by /wiki-inbox-sweep)
```

The four meta files (`schema.md`, `index.md`, `_MAP.md`, `log.md`) live at
the wiki root. They are exempt from the orphan rule and from the `pages/` /
`refs/` front-matter requirement.

---

## Page kinds

| Kind                  | Subtypes (`entity_type:`)                                                                  | Lives in                |
|-----------------------|--------------------------------------------------------------------------------------------|-------------------------|
| Entity                | `paper`, `person`, `venue`, `initiative`, `concept`, `dataset`, `tool`, `event`            | `pages/entity/`         |
| Synthesis             | `synthesis`                                                                                | `pages/synthesis/`      |
| Reference             | `paper` (with bibliographic frontmatter)                                                   | `refs/`                 |
| Meta                  | `index`, `log`, `map`, `schema`                                                            | wiki root               |

Every entity / synthesis / ref page **must** carry frontmatter (see below).
Meta pages are not required to.

---

## Scope: project knowledge base

This wiki describes a **project** — its components, the technologies it
integrates, the people building it, the literature it builds on, and the
empirical-validation status of its claims. It is not a grant proposal,
a manuscript, a review report, or any other document being written
*about* the project. Documents about the project live elsewhere in the
host repository (e.g. `drafts/`, `reviews/`, `manuscripts/`).

The distinction is load-bearing because the wiki sits inside repositories
whose primary product is often a document. Without an explicit boundary,
content drifts: claims authored to support a *document's* argument leak
into wiki entries, which then describe the proposal/paper/report rather
than the project itself.

### What belongs in the wiki

- **Entity pages** describe stable objects in the project's world: people,
  institutions, papers, datasets, tools, controlled-vocabulary concepts,
  initiatives the project participates in, events the project attends.
- **Synthesis pages** compare or integrate multiple entities/refs to
  state a project-level claim — typically about technique, design choice,
  empirical-validation status, or methodological lineage.
- **Reference pages** document what a cited paper/article claims and
  whether the technique it demonstrates is empirically validated, at
  what scale, and with what failure modes. Each reference page should
  answer "does this technique work?" — not "where should we cite this?"

### What does not belong

The following categories belong to the document(s) being written *about*
the project, not to the project KB itself:

- **Document-structure references.** Mentions of section numbers
  (`§1`, `§2`, `Section 3`), chapter labels, page counts, sub-headings
  in a containing document. The wiki has its own structure and does
  not co-reference an external document's structure.
- **Citation-routing advice.** Phrasings like "cite X for Y", "must cite",
  "should cite", "do not cite for §N". Whether a paper should be cited
  in a particular passage is a drafting concern, not a project-knowledge
  concern.
- **Narrative-meta posture.** Phrasings like "the proposal argues",
  "the rebuttal addresses", "the load-bearing argument", "the
  reviewer's strongest objection". These describe the *document* being
  written, not the *project* the document describes.
- **Administrative-context framing.** Treating an external venue
  (funder, working group, conference, journal) as a structural
  *component* of the project rather than as a counterpart the project
  engages with. The project is what the team builds; venues are where
  it is presented or where partners are met.

### Recommended prose shape

- **Reference pages**: a Why-relevant block structured as Claim /
  Evidence / Location-in-the-project / Limits — what the paper
  demonstrates, what empirical evidence supports it, where the
  technique appears in the project's own design, and what failure
  modes the paper acknowledges.
- **Synthesis pages**: a technical-claim opener (project-level,
  not document-level), the technical comparison or argument in the
  middle, and a technical-conclusion closer or empirical-validation
  status. No grant-narrative posture, no "the proposal must…",
  no citation routing.

### Per-project addendum

Skills that bootstrap a wiki inside a more specific kind of repo
(e.g. a grant skill, a manuscript skill) may extend this section with
a project-specific sub-section recording the project's name, technical
components, and the external venues it engages with. The generic
boundary above always applies; the addendum tightens it.

#### nifreeze — theory-grounding knowledge base

**Project.** nifreeze estimates volume-to-volume head motion and volume-wise
artifacts (eddy-current distortion in dMRI, and analogous effects in fMRI/PET)
in 4D neuroimaging. It is part of the NiPreps ecosystem. Its architecture is a
three-layer *Data → Model → Estimator* pipeline: signal models predict a
left-out volume (Leave-One-Volume-Out), and an ANTs-based registration step
aligns the observed volume to that prediction.

**Purpose of this wiki.** This wiki is the **theory** nifreeze rests on — not a
description of nifreeze itself. It is the first-principles grounding an author
(human or agent) consults before a large refactor or a new-model
implementation. It captures the mathematics and physics of the techniques
nifreeze reimplements, at the fidelity of the original papers.

**Theory areas in scope** (grow as needed):

- Diffusion-MRI signal formation and acquisition physics (the Stejskal–Tanner
  model, b-values/b-vectors, single- and multi-shell sampling).
- Gaussian-process regression / kriging, and its use as a non-parametric
  predictor of the dMRI signal on the sphere.
- Image registration and computer-vision fundamentals (rigid-body transforms,
  similarity metrics, interpolation, optimization).
- EPI / off-resonance geometric distortion and its correction.
- Eddy-current-induced distortion and its parameterization.
- Rigid-body head motion and its estimation/correction.
- Outlier (dropout/signal-loss) detection and replacement.

**Boundary (tightens the generic rule above).** This wiki does **not** memorize
nifreeze's source code or its API/user documentation. Those are the *artifacts
this theory grounds*, not wiki content. A page must be understandable — and
correct — to someone who has never read nifreeze's code. References to specific
nifreeze modules/equations are allowed only as pointers ("this is where the
project uses the technique"), never as the substance of a page.

Concept pages in this wiki use `namespace: paper` (concepts are
literature-sourced). If a future concept is purely internal to the project
rather than drawn from the literature, revisit the E007 `{rr, paper}` enum.

---

## Required frontmatter (entity / synthesis / ref pages)

```yaml
---
title: "Human-readable page name"
entity_type: person                  # one of the subtypes above
last_verified: 2026-04-25            # ISO date — bumped when a human/skill re-verifies
confidence_score: 0.9                # 0.0 (rumour) to 1.0 (primary source seen)
---
```

`/wiki-audit` raises **E002** if any of `title`, `entity_type`,
`last_verified`, or `confidence_score` is missing.

### Conditional / typed fields

| Page kind                      | Additional required fields                                                                 | Audit code |
|--------------------------------|--------------------------------------------------------------------------------------------|------------|
| `entity_type: paper`           | `doi:` **OR** `status:` ∈ {`unpublished`, `doi-not-found`, `preprint`, `provisional`}      | E003       |
| `entity_type: concept`         | `namespace:` ∈ {`rr`, `paper`}                                                             | E007       |
| `pages/synthesis/*.md`         | At least one Markdown link to `pages/entity/*.md` or `refs/*.md` (or in `derived_from:`)   | E006       |

### Optional but recognised fields

- `source_urls:` — list of canonical URLs (used by `/wiki-refresh` for entity re-verification).
- `s2_id:` — Semantic Scholar paperId (used by `/wiki-refresh` for refs).
- `citation_count_at_verify:` — set/bumped by `/wiki-refresh`.
- `refresh_needed_if:` — free-form predicate; suppresses **W001** staleness even if `last_verified` is older than the freshness threshold.
- `conflicts_with:` — list of relative paths to pages whose claims contradict this one. Must be **bidirectional** (W004).
- `flag_ids:` — list of `F####` ids from `flags.md` (forward-compatible).

---

## Freshness rule

A page's `last_verified` more than **180 days** older than `today` raises
**W001** unless the page declares `refresh_needed_if:` with an explicit
predicate (e.g., `refresh_needed_if: "DOI changes or paper retracted"`).

The 180-day window is the same threshold `/wiki-refresh --stale` uses to
pick targets for re-verification.

---

## Relationship graph (`_MAP.md`)

`_MAP.md` is a Markdown file containing one or more pipe tables. Each row is
an edge: source page → relationship type → target page → optional note.

```markdown
| source                                        | type           | target                                            | note            |
|-----------------------------------------------|----------------|---------------------------------------------------|-----------------|
| pages/synthesis/research-idea.md              | informs        | pages/entity/person-pi.md                         | author          |
| pages/synthesis/research-idea.md              | depends_on     | pages/entity/event-call-2026.md                   | submission slot |
| pages/entity/event-call-2026.md               | operationalizes| pages/entity/initiative-funder.md                 | programme       |
```

The header row (cell starting with the literal word `source`, case-insensitive)
and the divider row (cells consisting of `-`/`:`/space only) are skipped.

### Allowed relationship types (closed enum)

`cites`, `critiques`, `supersedes`, `extends`, `depends_on`, `informs`,
`implements`, `operationalizes`, `replicates`, `contradicts`.

`/wiki-audit` raises **E004** for any other value.

---

## Index (`index.md`)

`index.md` is the curated navigation. A page is **not** an orphan (W002) if
it is either:

- linked from `index.md` as a Markdown link, **or**
- the target of an inbound edge in `_MAP.md`.

Convention: `index.md` carries three top-level sections — `## Entities`,
`## Synthesis`, `## References` — and every page added to the wiki gets a
one-line link under the appropriate section. New pages added by skills
(`/wiki-init`, `/wiki-inbox-sweep`) append to the relevant section.

---

## Glossary (`pages/synthesis/glossary-paper-vs-rr.md`)

Optional. If present, every `` `paper.<term>` `` token used in any page's
prose must appear in the glossary; otherwise **W003** is raised. The glossary
itself is exempt from synthesis-grounding (E006).

---

## Log (`log.md`)

Append-only changelog of skill runs. Every block starts with a level-2 heading:

```markdown
## YYYY-MM-DD — /<skill-name> — <one-line summary>
- bullet describing what changed
- another bullet
```

The header line must match `^## \d{4}-\d{2}-\d{2}\b` or `/wiki-audit` raises
**W006**. Plain section headings (e.g., `## Deferred work`) and code-fenced
example blocks are ignored. Lines inside the wiki frontmatter (between two
`---` delimiters at the top) are also ignored.

---

## Inbox (`_inbox/`)

Drop unstructured prose (raw notes, screenshots' worth of text, brain dumps)
as `<slug>.md` files. `/wiki-audit` raises **W005** for each non-draft inbox
entry to remind you to run `/wiki-inbox-sweep`. Files ending in `.draft.md`
are silently skipped (use them for in-progress notes you don't want flagged).

---

## File naming

- Entities: `<subtype>-<slug>.md` — e.g. `person-jane-doe.md`, `initiative-erc-adg.md`, `event-call-2026.md`, `paper-smith-2024-replication.md`.
- Synthesis: `<topic-slug>.md` — e.g. `research-idea.md`, `glossary-paper-vs-rr.md`.
- Refs: `<first-author>-<year>-<short-slug>.md` — e.g. `scheel-2021-positivity-gap.md`.

Slugs are lowercase, ASCII, hyphen-separated. The audit doesn't enforce file
names — but the `/wiki-*` ecosystem assumes this convention.

---

## Schema version

`schema_version: "1"`. The audit script hard-codes this; bump both together
when extending the schema.
