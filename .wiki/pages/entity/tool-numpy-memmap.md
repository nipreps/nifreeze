---
title: "NumPy memory-mapped arrays (numpy.memmap, np.lib.format.open_memmap)"
entity_type: tool
last_verified: 2026-07-15
confidence_score: 0.9
source_urls:
  - https://numpy.org/doc/stable/reference/generated/numpy.memmap.html
  - https://numpy.org/doc/stable/reference/generated/numpy.lib.format.open_memmap.html
refresh_needed_if: "numpy dependency in pyproject.toml (>=1.21.3,<2.4.0) advances past 2.3.x or memmap __reduce__ semantics change"
---

# NumPy memory-mapped arrays

The concrete API nifreeze uses to realize [[concept-memory-mapped-io]]. Verified
against **numpy 2.3.5** (pin `numpy>=1.21.3,<2.4.0`).

## `numpy.memmap`

```python
np.memmap(filename, dtype=uint8, mode='r+', offset=0, shape=None, order='C')
```

- **`np.memmap` is a subclass of `np.ndarray`.** This is the load-bearing fact
  for nifreeze: a `memmap` passes `isinstance(x, np.ndarray)`, supports
  `.reshape`, boolean/fancy indexing, in-place assignment, `np.array_split`, and
  every ufunc — so the entire model/estimator consumer surface works on it with
  **zero code changes**. This is why the `dataobj` field validator can stay
  strict (`isinstance(value, np.ndarray)`) and still accept the lazy backend.
- **`mode`**: `'r'` read-only, `'r+'` read-write existing file, `'w+'` create/
  overwrite, `'c'` copy-on-write (writes stay private, never hit disk).
- **`offset`**: byte offset into the file where the array starts. This is the
  hook that lets a `memmap` point *directly at the raw data block inside an HDF5
  file* — see the `get_offset()` technique in [[tool-hdf5-storage-layout]].
- **`shape`/`dtype`/`order`**: interpret the mapped bytes; must match how they
  were written (C-order, native endianness for zero-copy).
- `.flush()` writes dirty pages back (mode `r+`/`w+`); deletion/GC of the object
  unmaps.

### Raw `np.memmap` vs `.npy` files

A bare `np.memmap` file is **headerless raw bytes**: nothing on disk records
dtype/shape/order, so the reader must supply them exactly. Prefer:

## `np.lib.format.open_memmap`

```python
np.lib.format.open_memmap(filename, mode='r+', dtype=None, shape=None,
                          fortran_order=False, version=None)
```

Creates/opens a **`.npy`** file (self-describing header carrying dtype, shape,
order) and returns a `np.memmap` view of its data region. This is the preferred
scratch backing store: self-describing, reloadable without out-of-band metadata,
and still a true `memmap`. nifreeze uses it for the streamed-copy fallback and
for scratch outputs (filtered DWI series, resampled `to_nifti` buffer).

Streaming a large array into one without a full intermediate:

```python
mm = np.lib.format.open_memmap(path, mode='w+', dtype=src.dtype, shape=src.shape)
for i in range(src.shape[-1]):
    mm[..., i] = src[..., i]     # one volume of transient RAM
mm.flush()
```

## Gotchas / failure modes

- **Pickling materializes.** `np.memmap.__reduce__` serializes the **full array
  contents**, not the file reference. Passing a `memmap` into `joblib.Parallel`/
  loky would copy the whole array into every worker (or joblib's own array
  reducer would remap it) — either way a memory hazard. **Invariant: extract a
  plain `np.ndarray` (e.g. via a boolean-mask copy or `np.asarray`) before
  handing data to workers.** nifreeze's DWI model already does this
  (`np.array_split` on an already-masked in-RAM array) *before* `Parallel`.
- **Fancy/boolean indexing copies** into RAM (ndarray semantics) — bounded for a
  single volume/shell, unbounded for an all-but-one train split.
- **Whole-array ops fault everything resident**, negating the memory win.
- **Dtype/endianness must match the bytes.** A non-native-endian source mapped
  with a native dtype yields garbage; either map with the correct byteorder dtype
  or fall back to a streamed copy (nifreeze chooses the latter for simplicity).
- **`w+` truncates.** Opening an existing file `w+` destroys it; use `r+` to
  update in place.
- **Windows** cannot unlink a file while mapped — release the object before
  deleting the scratch file (see [[concept-memory-mapped-io]] portability note).
