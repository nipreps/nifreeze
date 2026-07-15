---
title: "NiBabel ArrayProxy lazy image data (nibabel)"
entity_type: tool
last_verified: 2026-07-15
confidence_score: 0.85
source_urls:
  - https://nipy.org/nibabel/reference/nibabel.arrayproxy.html
  - https://nipy.org/nibabel/images_and_memory.html
refresh_needed_if: "nibabel dependency advances past 5.3.x, or nifreeze's load() stops calling np.asanyarray on img.dataobj"
---

# NiBabel ArrayProxy

nibabel (imported as `nb`; verified against **nibabel 5.3.2**) already provides a
lazy access layer for on-disk images — relevant both as prior art for
[[concept-memory-mapped-io]] and because nifreeze's NIfTI ingestion currently
**discards** that laziness.

## What an ArrayProxy is

`img.dataobj` on a NIfTI image is an `ArrayProxy`, not an ndarray: a lightweight
handle that reads pixel data **on indexing**, applying scaling
(`scl_slope`/`scl_inter`) and dtype conversion lazily. Two access idioms:

- `np.asarray(img.dataobj)` / `img.get_fdata()` — **materialize the whole array**
  in RAM (get_fdata additionally caches a float64 copy on the image).
- `img.dataobj[..., i]` — read **only** volume `i` from disk.
- For uncompressed NIfTI (`.nii`, not `.nii.gz`), nibabel can memory-map the file,
  so slice reads are demand-paged rather than re-parsed.

## Where nifreeze defeats it (the injection point)

nifreeze's loaders convert the proxy to a dense array eagerly at ingestion —
`np.asanyarray(img.dataobj, ...)` in the NIfTI read path — which is the NIfTI-side
analogue of the HDF5 eager-read hog ([[tool-hdf5-storage-layout]]). This is the
natural place to instead stream volume-by-volume into a `.npy` memmap
([[tool-numpy-memmap]]) so a freshly loaded NIfTI series is disk-backed from the
start, not just after a round-trip through the NFDH5 container.

## Comparison to numpy.memmap

| | ArrayProxy | np.memmap |
|---|---|---|
| type | proxy object (not ndarray) | **ndarray subclass** |
| indexing | lazy read, returns ndarray | view (basic) / copy (fancy) |
| ndarray ops (`.reshape`, ufuncs, in-place) | **no** (must materialize first) | **yes** |
| scaling applied | yes (slope/inter) | no (raw bytes) |
| writable-in-place | no | yes (`r+`) |

The takeaway mirrors the Zarr analysis ([[tool-zarr-array-store]]): a lazy proxy
is fine for *ingestion* (read one volume at a time) but cannot be the live
`dataobj` the models operate on, because it is not an ndarray. nifreeze keeps
nibabel for I/O and header handling, and uses a `np.memmap` as the working array
the models see. The scaling caveat matters: when converting a proxy to a memmap,
apply slope/intercept during the streamed copy so the on-disk memmap holds true
signal values, not raw integers.
