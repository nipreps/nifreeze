---
title: "Storage backend decision: memmap working array + HDF5 container"
entity_type: synthesis
last_verified: 2026-10-20
confidence_score: 0.9
refresh_needed_if: "a second storage backend is adopted, or the model consumer surface stops requiring ndarray semantics"
derived_from:
  - pages/entity/tool-numpy-memmap.md
  - pages/entity/tool-hdf5-storage-layout.md
  - pages/entity/tool-zarr-array-store.md
  - pages/entity/concept-memory-mapped-io.md
  - pages/entity/concept-checkpoint-restart.md
---

# Storage backend decision: memmap + HDF5

**Claim.** For nifreeze's access pattern, the data layer should keep its working
array as a `np.memmap` ([[tool-numpy-memmap]]) and use a single HDF5 file per
dataset ([[tool-hdf5-storage-layout]]) as both the serialization format and the
crash-safe checkpoint store ([[concept-checkpoint-restart]]). Live `zarr.Array`
([[tool-zarr-array-store]]) and live `h5py.Dataset` backends are rejected, and a
generic swappable-backend protocol is deferred as premature abstraction. This is
the durable resolution of the long-standing "HDF5 is a memory hog" problem.

## The problem being solved

nifreeze historically held the entire 4D series resident: the HDF5 load path did
`np.asanyarray` per field, DWI b0 removal copied the full array, and `to_nifti`
allocated a second full array. Peak RAM scaled with the *series*, not the
*volume* — untenable for whole-brain, many-orientation data. The goal is to bound
peak RAM to the working set (one volume at a time; see
[[concept-memory-mapped-io]] and [[concept-leave-one-volume-out]]) and to survive
crashes without losing completed volumes.

## Why memmap, not a lazy non-ndarray backend

The decisive constraint is the **model consumer contract**. nifreeze's models
perform, on `dataobj`, operations that require true `np.ndarray` semantics:

- 3-D boolean masking of a 4-D block (`data[:, brainmask]` since the
  volumes-first layout below),
- `.reshape(n_volumes, -1)` returning a view,
- in-place per-volume assignment (PET frame resampling),
- `np.array_split(...)` before handing chunks to `joblib.Parallel`.

`np.memmap` **is an ndarray subclass**, so all of these work unchanged — the
memory fix needs *zero* rewrites to the model/estimator surface, and the
`dataobj` validator can remain strict (`isinstance(value, np.ndarray)`).
`zarr.Array` and `h5py.Dataset` are **not** ndarrays: each of those operations
would raise or change copy/view semantics, forcing a broad, fragile rewrite of
every consumer — for cloud/chunking/compression benefits nifreeze's single-node,
volume-wise, dense-4D access does not exercise. A lowest-common-denominator
"backend protocol" over {memmap, h5py, zarr} could not even express reshape /
fancy-index, so it would leak immediately; it is deferred until a second backend
is genuinely demanded. Decoupling *format* (HDF5) from *in-memory representation*
(memmap) keeps that future door open without paying for it now.

## Why HDF5 as the container

HDF5 is already the nifreeze serialization format (the NFDH5 layout), is
self-describing and versioned, and — written contiguous/uncompressed/native-
endian — exposes a raw data block that `np.memmap` can map **directly** at the
byte offset from `Dataset.id.get_offset()`. The direct map is opened copy-on-write,
so in-place writes (PET frame resampling) stay private and never modify the
source file; a checkpoint store would need an explicit writable map. The robust primary load path is streaming the dataset into
a `.npy` memmap (works for any layout); direct-mmap is an opportunistic zero-copy
optimization on top.

## Volumes-first layout and cache lifecycle (2026-10-20)

Decided with the maintainer while preparing nifreeze#461: `dataobj` is
C-contiguous `(N, X, Y, Z)` in memory as well as on disk, and `dataset[i]` is
`dataobj[i]`, one contiguous block. Registration and closed-form smoothers
(nifreeze#499) read one volume at a time; model fits gather the masked voxels of
the training volumes one volume at a time into a voxels-by-volumes matrix, as
DIPY expects, which removes the whole-field-of-view copy each fit used to make.
Bounding the per-fit copy itself (a masked cache plus block-wise fits) is
nifreeze#513. NIfTI input and output convert at the boundary.

Lifecycle rules, each from a defect found in review:
- Temporary caches (legacy v1 conversion, compressed v2, b0 removal from a
  memory-mapped DWI) are removed when the last array referencing them is
  released, or at exit. Earlier, every conversion left a full-size `.npy` behind.
- Caches are filled through file writes, and v1 files are read in x-slabs:
  loading a 135 MB v1 file went from +145 MB to +8 MB peak RSS.
- b0 removal from an in-memory DWI stays in memory (no temporary file).
- `to_filename` writes a temporary file and renames it over the target, so a
  dataset can be written onto the file its memory map comes from. `h5py.File(path,
  "w")` truncates in place: on macOS the mapped pages survived, but Linux raises
  SIGBUS when a mapping is read past the end of the file.

## Empirical-validation status

The memory claim is validated by peak-RSS tests (child-process `ru_maxrss`) that
assert `from_filename`+index, DWI construction, and `to_nifti` each stay well
under the full-array size on a ~1 GB fixture — measurements that *fail* on the
pre-change eager-load code and *pass* after. tracemalloc is unsuitable (blind to
mmap page cache and h5py C buffers). The checkpoint claim is validated by an
interrupt-and-resume test asserting identical final motion transforms to an
uninterrupted run. Prior attempts (a memmap+HDF5-caching PR) foundered on cache
lifecycle and a duplicative write path; this design fixes that by making the
container the single source of truth and defining cache/checkpoint lifecycles
explicitly.
