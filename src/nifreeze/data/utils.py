from __future__ import annotations

import sys
from pathlib import Path
from tempfile import mkdtemp

import attrs
import h5py
import nibabel as nb
import nitransforms as nt
import numpy as np

_NATIVE_BYTEORDER = "<" if sys.byteorder == "little" else ">"

#: NFDH5 marks a volume-major ``dataobj`` (volumes are the *first*, slowest-varying
#: axis on disk, so a single volume is a contiguous block that memory-maps locally).
#: Legacy files store the last axis as volumes and carry no such attribute.
VOLUME_AXIS_ATTR = "volume_axis"


def _is_directly_mappable(dset: h5py.Dataset) -> bool:
    """Return whether an HDF5 dataset's raw block can be mapped with ``np.memmap``.

    A dataset is directly mappable only if it is stored as a single contiguous,
    unfiltered (uncompressed) block in native byte order — the layout
    :func:`save_dataobj_volume_major` writes by default. See the wiki page
    ``tool-hdf5-storage-layout`` for the full rationale.
    """
    if dset.id.get_offset() is None:  # None for chunked / compact / virtual / unallocated
        return False
    if dset.id.get_create_plist().get_nfilters() != 0:  # compression/filters
        return False
    byteorder = dset.dtype.byteorder
    return byteorder in ("=", "|") or byteorder == _NATIVE_BYTEORDER


def save_dataobj_volume_major(
    root: h5py.Group,
    dataobj: np.ndarray,
    compression: str | None = None,
    compression_opts=None,
) -> None:
    """Write a 4D ``dataobj`` volume-major, streaming one volume at a time.

    The on-disk dataset has shape ``(N, X, Y, Z)`` (volumes first) so that each
    volume is a contiguous block, which memory-maps with per-volume locality on
    read (see :func:`open_dataobj_memmap`). Writing proceeds volume-by-volume to
    avoid materializing a transposed copy of the whole array.
    """
    n = dataobj.shape[-1]
    dset = root.create_dataset(
        "dataobj",
        shape=(n, *dataobj.shape[:-1]),
        dtype=dataobj.dtype,
        compression=compression,
        compression_opts=compression_opts,
    )
    dset.attrs[VOLUME_AXIS_ATTR] = 0
    for i in range(n):
        dset[i] = dataobj[..., i]


def open_dataobj_memmap(
    filename: Path | str,
    key: str = "/0/dataobj",
    mode: str = "c",
    cache_dir: Path | str | None = None,
) -> np.memmap:
    """Open a 4D ``dataobj`` as a lazy, per-volume-local :obj:`~numpy.memmap`.

    The returned array always has the in-memory shape ``(X, Y, Z, N)`` so that
    ``dataobj[..., k]`` selects volume ``k``. Two on-disk layouts are handled:

    - **volume-major** (``VOLUME_AXIS_ATTR == 0``, written by
      :func:`save_dataobj_volume_major`): directly mapped zero-copy when
      contiguous/uncompressed/native-endian, then transposed to ``(X, Y, Z, N)``
      — each volume is a contiguous block.
    - **legacy** ``(X, Y, Z, N)`` (no attribute): streamed once into a
      Fortran-ordered ``.npy`` cache so that volumes become contiguous. This has
      a one-time O(N) cost; new files never take this path.

    ``mode="c"`` (copy-on-write) is the default: reads are lazy and shared with
    the file, while writes fault private pages and never mutate the source.
    """
    filename = Path(filename)
    with h5py.File(filename, "r") as in_file:
        dset = in_file[key]
        shape = dset.shape
        dtype = dset.dtype
        volume_major = int(dset.attrs.get(VOLUME_AXIS_ATTR, -1)) == 0
        mappable = _is_directly_mappable(dset)
        offset = dset.id.get_offset() if mappable else None

    if volume_major:
        if mappable:
            mm = np.memmap(filename, dtype=dtype, mode=mode, offset=offset, shape=shape, order="C")
            return np.moveaxis(mm, 0, -1)
        # Compressed/chunked volume-major: stream (N, X, Y, Z) then transpose.
        cache = _cache_npy(cache_dir, key)
        mm = np.lib.format.open_memmap(cache, mode="w+", dtype=dtype, shape=shape)
        with h5py.File(filename, "r") as in_file:
            dset = in_file[key]
            for i in range(shape[0]):
                dset.read_direct(mm, np.s_[i], np.s_[i])  # C-contiguous slice
        mm.flush()
        return np.moveaxis(mm, 0, -1)

    # Legacy (X, Y, Z, N): stream into a Fortran-ordered cache for volume locality.
    # The Fortran-ordered ``[..., i]`` slice is not C-contiguous, so plain
    # assignment (not ``read_direct``) is used; each step reads one volume.
    cache = _cache_npy(cache_dir, key)
    mm = np.lib.format.open_memmap(cache, mode="w+", dtype=dtype, shape=shape, fortran_order=True)
    with h5py.File(filename, "r") as in_file:
        dset = in_file[key]
        for i in range(shape[-1]):
            mm[..., i] = dset[..., i]
    mm.flush()
    return mm


def stream_select_last_axis(
    src: np.ndarray,
    keep_mask: np.ndarray,
    cache_dir: Path | str | None = None,
    name: str = "dataobj",
) -> np.memmap:
    """Select volumes along the last axis into a disk-backed memmap.

    Copies ``src[..., keep_mask]`` one volume at a time into a Fortran-ordered
    ``.npy`` memmap (per-volume contiguous). The result is disk-backed, so once
    the source is released the selection holds no resident RAM — used to drop
    b=0 volumes from a DWI without retaining a full in-memory copy.
    """
    keep = np.flatnonzero(keep_mask)
    out_shape = (*src.shape[:-1], keep.size)
    cache = _cache_npy(cache_dir, name)
    mm = np.lib.format.open_memmap(
        cache, mode="w+", dtype=src.dtype, shape=out_shape, fortran_order=True
    )
    for j, i in enumerate(keep):
        mm[..., j] = src[..., i]
    mm.flush()
    return mm


def _cache_npy(cache_dir: Path | str | None, key: str) -> Path:
    cache_dir = Path(cache_dir) if cache_dir is not None else Path(mkdtemp())
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{key.strip('/').replace('/', '_')}.npy"


def save_nfdh5_fields(
    root: h5py.Group,
    dataset,
    compression: str | None = None,
    compression_opts=None,
) -> None:
    """Write a dataset's public attrs fields into the ``/0`` group.

    ``dataobj`` is written volume-major (:func:`save_dataobj_volume_major`); all
    other non-``None``, non-private fields are written as plain datasets.
    """
    for field in attrs.fields(dataset.__class__):
        if field.name.startswith("_"):
            continue
        value = getattr(dataset, field.name)
        if value is None:
            continue
        if field.name == "dataobj":
            save_dataobj_volume_major(root, value, compression, compression_opts)
        else:
            root.create_dataset(
                field.name,
                data=value,
                compression=compression,
                compression_opts=compression_opts,
            )


def load_nfdh5_fields(filename: Path | str, lazy_keys: tuple[str, ...] = ("dataobj",)) -> dict:
    """Read the ``/0`` group of an NFDH5 file into a field dict.

    ``dataobj`` (named in ``lazy_keys``) is returned as a lazy, per-volume-local
    :obj:`~numpy.memmap` (via :func:`open_dataobj_memmap`); every other field is
    read eagerly (they are small: affine, brain mask, gradients, timings).
    """
    filename = Path(filename)
    with h5py.File(filename, "r") as in_file:
        root = in_file["/0"]
        data = {
            k: np.asanyarray(v)
            for k, v in root.items()
            if not k.startswith("_") and k not in lazy_keys
        }
        present = [k for k in lazy_keys if k in root]
    for key in present:
        data[key] = open_dataobj_memmap(filename, f"/0/{key}")
    return data


def apply_affines(nii, em_affines, output_filename=None) -> nb.nifti1.Nifti1Image:
    """
    Apply affines to supplied nii data

    Parameters
    ----------
    nii : :obj:`~nibabel.nifti1.Nifti1Image`
        Nifti1Image data to be transformed
    em_affines : :obj:`ndarray`
        Nx4x4 array
    output_filename : :obj:`str`, optional
        String specifying filepath to which to save transformed Nifti1Image data

    Returns
    -------
    nii_t_img : :obj:`~nibabel.nifti1.Nifti1Image`
        Transformed Nifti1Image data

    """
    transformed_nii = np.zeros_like(np.asanyarray(nii.dataobj))

    for ii, bvecnii in enumerate(nb.four_to_three(nii)):
        xfms = nt.linear.Affine(em_affines[ii])
        transformed_nii[..., ii] = np.asanyarray(
            nt.resampling.apply(~xfms, bvecnii, reference=nii).dataobj
        )

    nii_t_img = nii.__class__(transformed_nii, nii.affine, nii.header)

    if output_filename is not None:
        # Ensure directories in output_filename exist
        Path(output_filename).parent.mkdir(exist_ok=True)

        # Save as .nii
        nii_t_img.to_filename(output_filename)

    return nii_t_img
