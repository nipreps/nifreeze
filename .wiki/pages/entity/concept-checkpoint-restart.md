---
title: "Checkpoint-restart for long-running volume-wise estimation"
entity_type: concept
namespace: rr
last_verified: 2026-07-15
confidence_score: 0.9
refresh_needed_if: "the estimator's per-volume loop contract or the NFDH5 container format changes"
---

# Checkpoint-restart

Checkpoint-restart is the ability to persist enough intermediate state during a
long computation that, after a crash or an interruption, the work resumes from
the last persisted point instead of restarting from zero. For nifreeze this is
the second dividend of a disk-backed data layer (the first is memory:
[[concept-memory-mapped-io]]): the same [[tool-hdf5-storage-layout]] container
that bounds RAM also holds the estimation's progress.

## Why nifreeze needs it

Head-motion estimation fits a signal model and runs an ANTs registration **per
volume** (see [[concept-leave-one-volume-out]] and
[[concept-image-registration]]). For a large series each volume is minutes of
work; a run can be hours. An out-of-memory kill, a preempted HPC job, or a node
failure partway through currently discards *all* completed volumes. Because the
per-volume result (a rigid transform, and for some modalities the resampled
volume) is independent and write-once, the loop is a natural checkpoint boundary.

## The persisted-state contract

State sufficient to resume must capture **what is done**, **the results so far**,
and **enough of the plan to continue deterministically**:

- **Completion vector** — a per-volume boolean (`converged[i]`) marking volumes
  already estimated. Resume skips these.
- **Results so far** — the per-volume motion transforms, and, for modalities that
  mutate the data in place (PET frame resampling), the updated volumes. Writing
  the result of volume *k* and setting `converged[k]` must be effectively atomic
  from the resumer's point of view (write payload first, flag last).
- **Plan/ordering** — the traversal order and its parameters (strategy, seed,
  start/stop bounds). Iteration order can affect results when a step initializes
  from the previous one; persisting it makes a resumed run reproduce the original
  ordering rather than a fresh shuffle.
- **Provenance** — model name, software version, timestamp — so a resume can
  refuse to continue a checkpoint written by an incompatible configuration.

## Invariants

- **Idempotent resume.** Re-running from a checkpoint must produce the same final
  result as an uninterrupted run. This requires that a volume's estimate does not
  depend on transient state lost at the crash beyond what is persisted.
- **Flush per unit, not per run.** The result of each volume must be durable
  before the next begins; a buffer flushed only at the end provides no crash
  safety. This trades a small per-volume I/O cost for restart guarantees.
- **Checkpoint outlives the process.** The store must live at a durable,
  caller-chosen path — never a `TemporaryDirectory` that the crashing process's
  cleanup would remove. It is user-owned and never auto-deleted. This is distinct
  from the *scratch* memory-map cache, which is transient and process-scoped.
- **A finished checkpoint is just a dataset.** On completion the container is a
  valid, fully-populated data file, re-openable by the normal load path — the
  checkpoint format and the serialization format are one and the same.

## Precedent already in the codebase

The design generalizes an existing pattern rather than inventing one: nifreeze's
PET path already writes its dataset to an HDF5 file on first transform, reads a
pristine frame back from it on demand, resamples, and writes the frame in place.
That is a per-volume disk round-trip through the same container — the checkpoint
writer is the same operation, plus a completion flag and resume-time skip logic.
The unifying artifact is the NFDH5 container documented in
[[tool-hdf5-storage-layout]]; the end-to-end rationale is
[[storage-backend-decision-memmap-hdf5]].
