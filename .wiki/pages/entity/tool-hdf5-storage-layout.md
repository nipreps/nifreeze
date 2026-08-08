---
title: "HDF5 storage layout and the direct-mmap technique (h5py)"
entity_type: tool
last_verified: 2026-07-15
confidence_score: 0.9
source_urls:
  - https://docs.h5py.org/en/3.12.1/high/dataset.html
  - https://docs.h5py.org/en/3.12.1/high/lowlevel.html
  - https://support.hdfgroup.org/documentation/hdf5/latest/_l_b_dset_layout.html
refresh_needed_if: "h5py dependency advances past 3.12.x / HDF5 past 1.14.x, or the NFDH5 container schema changes"
---

# HDF5 storage layout and the direct-mmap technique

The container nifreeze writes/reads (via **h5py 3.12.1**, HDF5 **1.14.4**). It is
both the serialization format *and* the checkpoint store
([[concept-checkpoint-restart]]), and — when written the right way — a file whose
raw data block can be memory-mapped directly ([[tool-numpy-memmap]],
[[concept-memory-mapped-io]]).

## NFDH5 container format (what nifreeze writes)

- File attrs `Format="NFDH5"`, `Version` (uint16). A single group `/0` with a
  group attr `Type` (`"base dataset"` / `"dmri"` / `"EMC/PET"`).
- One dataset per data field under `/0` (`dataobj`, `affine`, `brainmask`,
  `motion_affines`, and modality extras `gradients`/`midframe`). The 4D `dataobj`
  dominates size.
- The load path historically did `np.asanyarray(dset)` per field — an eager full
  read (the memory hog fixed by lazy `dataobj` loading).

## Dataset layouts (why layout decides mappability)

| layout | on-disk arrangement | contiguous raw block? | directly mmap-able? |
|---|---|---|---|
| **contiguous** | one flat C-ordered block | yes | **yes** |
| **chunked** | independent (optionally filtered) chunks scattered in the file | no | no |
| **compact** | stored inside the object header (tiny datasets) | no | no |

`h5py.create_dataset(...)` with **no** `chunks=` and **no** `compression=` yields
a **contiguous** dataset — the default nifreeze relies on. Requesting compression
or chunking forces a non-contiguous layout.

## The direct-mmap technique

HDF5 stores a contiguous dataset as a single C-ordered raw block at a fixed byte
offset in the file. That offset can be read and handed to `np.memmap`:

```python
dset = h5f["/0/dataobj"]
offset = dset.id.get_offset()          # absolute byte offset, or None
if offset is not None and _is_directly_mappable(dset):
    arr = np.memmap(path, dtype=dset.dtype, mode=mode,
                    offset=offset, shape=dset.shape, order="C")
```

`np.memmap` opens its own file descriptor against `path`, so the `h5py.File` may
be **closed immediately** after reading `offset`/`dtype`/`shape`. HDF5's
contiguous layout is row-major C-order, matching numpy's default — no transpose.
A writable (`r+`) map means writes land back inside the HDF5 container, which is
what lets the checkpoint store be updated in place volume-by-volume.

### `_is_directly_mappable` — the guard (all must hold)

- `dset.id.get_offset()` is not `None` (returns `None` for chunked, compact,
  virtual, or unallocated datasets).
- No filters/compression: `dset.id.get_create_plist().get_nfilters() == 0`.
- Native byte order: `dset.dtype.byteorder in ("=", "|")` (or equals the
  platform). A non-native dtype technically maps but risks surprising casts under
  `NIFREEZE_WERRORS`; treat as "fall back".
- Not virtual/external storage; single contiguous block.

## Streaming fallback (the robust primary path)

When the guard fails (compressed, chunked, non-native), stream the dataset into a
`.npy` memmap without a full intermediate:

```python
mm = np.lib.format.open_memmap(cache_npy, mode="w+", dtype=dset.dtype, shape=dset.shape)
for i in range(dset.shape[-1]):
    dset.read_direct(mm, np.s_[..., i], np.s_[..., i])   # ~one volume transient
mm.flush()
```

`read_direct` copies straight into the destination buffer (here, mapped pages) —
peak extra RAM ≈ one volume + the HDF5 chunk cache. Design choice: direct-mmap is
an *opportunistic* zero-copy fast path; **streaming is the primary mechanism**
because it works for every layout, dtype, and filesystem.

## Failure modes to watch

- Writing with `compression=`/`chunks=` makes the file non-directly-mappable
  (streaming fallback still works). nifreeze writes contiguous+uncompressed+
  native-endian by default to preserve mappability.
- A userblock or later file edits do not break `get_offset()` (it is absolute).
- **Writable direct-mmap into a compressed dataset would corrupt it** — the guard
  forbids this.
- h5py file handles are **not picklable** — never send a live `Dataset`/`File`
  into a worker; extract arrays first (mirrors the memmap-pickling hazard in
  [[tool-numpy-memmap]]).
- Network/overlay filesystems may weaken mmap semantics — fall back to streaming.
