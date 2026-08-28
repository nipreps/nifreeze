---
title: "Memory-mapped I/O and demand paging for 4D neuroimaging"
entity_type: concept
namespace: rr
last_verified: 2026-07-15
confidence_score: 0.9
refresh_needed_if: "OS mmap semantics or numpy's memmap page-cache behavior change materially"
---

# Memory-mapped I/O and demand paging

A memory-mapped file exposes bytes on disk as if they were an in-process array,
without an eager read. This is the mechanism nifreeze uses to stop holding an
entire 4D volume series resident in RAM. The concrete API is
[[tool-numpy-memmap]]; the container that backs it is [[tool-hdf5-storage-layout]].
This page is the vendor-neutral *idea*.

## The core mechanism

`mmap(2)` maps a file (or a region of it) into the process's virtual address
space. No bytes are read at map time — only page-table entries are created.
A read of address *A* that has no resident physical page triggers a **page
fault**; the kernel reads the corresponding disk page (typically 4 KiB) into the
**page cache**, wires it into the process, and the load completes. This is
**demand paging**: RAM is populated only for the bytes actually touched.

Consequences that matter for a motion-estimation loop over 4D data:

- **Working set, not total size, bounds RAM.** Leave-One-Volume-Out iterates one
  volume at a time (see [[concept-leave-one-volume-out]]). Only the pages of the
  volume(s) currently being read/predicted/registered fault in; the rest of the
  series stays on disk. Peak resident memory scales with the *volume*, not the
  *series*.
- **Clean pages are free to evict.** Under memory pressure the kernel can drop
  clean (unmodified) mapped pages without writing anything back — they can be
  re-read from the file on next fault. The page cache acts as an automatic LRU
  cache sized by the OS, not the application.
- **Shared pages across processes.** Two processes mapping the same file share
  the same physical page-cache pages for read-only access — relevant if data are
  ever consumed by multiple workers reading (not pickling) the same backing file.

## Read-only vs writable maps

- **Read-only (`r`)** maps forbid writes; ideal for the pristine input series.
- **Copy-on-write (`c`)** maps let a process write, but changes stay private and
  are never flushed to the file — useful for scratch transforms that must not
  corrupt the source.
- **Read-write (`r+`)** maps propagate writes back to the file. A modified
  ("dirty") page is written back on `flush`/`msync`, on unmap, or when the kernel
  reclaims it. This is what makes a writable map double as a persistence device:
  writing volume *k* in place *is* the act of saving it. That property is the
  hinge of the checkpoint design in [[concept-checkpoint-restart]].

## Why this fits nifreeze's access pattern specifically

nifreeze's data are dense, whole-brain 4D arrays accessed **volume-wise** on a
**single node**. The access is sequential-ish and local (one orientation at a
time), which is exactly the pattern demand paging serves best: high locality,
low working set, no need for chunked/compressed random access. The alternative
storage idioms that shine for *cloud, chunked, or out-of-core-random* workloads
([[tool-zarr-array-store]]) buy nothing here and cost API compatibility — see the
decision record [[storage-backend-decision-memmap-hdf5]].

## Failure modes and gotchas

- **Touching everything defeats the point.** Any operation that reads the whole
  array (a global reduction, an `np.asarray` copy, a full-array equality check)
  faults every page resident and reproduces the original memory cost. Streaming
  reductions volume-by-volume preserve the benefit. This is also why a peak-RSS
  *test* must not read the whole array after the operation it measures.
- **Fancy/boolean indexing returns copies.** Indexing a mapped array with a
  boolean mask or an index array materializes a new in-RAM array (the selected
  subset), not a view. Bounded when the subset is one volume/shell; unbounded
  when it is "all but one volume" (the LOVO train split).
- **`flush()`/close discipline.** Dirty pages are not guaranteed on disk until an
  explicit flush or unmap. Crash-before-flush loses unpersisted writes — the
  checkpoint contract must flush per volume, not per run.
- **Portability.** POSIX unmaps lazily; Windows forbids deleting a file while it
  is mapped, so cache cleanup must unmap (drop all references, let GC/`del` run)
  *before* unlinking.
- **Network/overlay filesystems** may implement mmap with weaker or slower
  semantics; the safe fallback is streaming reads rather than a direct map.
