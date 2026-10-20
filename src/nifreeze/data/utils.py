from __future__ import annotations

import os
import shutil
import sys
import weakref
from contextlib import contextmanager
from pathlib import Path
from tempfile import mkdtemp
from typing import Iterator, Literal
from uuid import uuid4

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

CONVERSION_SLAB_BYTES = 2**20
"""Bytes read per step when converting a legacy ``dataobj`` (at least one x-plane)."""


def _is_directly_mappable(dset: h5py.Dataset) -> bool:
    """Return whether an HDF5 dataset's raw block can be mapped with ``np.memmap``.

    A dataset is directly mappable only if it is stored as a single contiguous
    block in native byte order — the layout :func:`save_dataobj_volume_major`
    writes by default. ``get_offset()`` returns ``None`` for chunked, compact,
    virtual, or unallocated datasets; because HDF5 forbids filters on a
    contiguous layout, a non-``None`` offset also guarantees the block is
    uncompressed. See the wiki page ``tool-hdf5-storage-layout``.
    """
    if dset.id.get_offset() is None:  # None for chunked / compact / virtual / unallocated
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


class _VolumeCache:
    """A volume-major ``(N, X, Y, Z)`` ``.npy`` file filled through file I/O.

    Writing through ``write`` instead of the memory map keeps the resident set
    to the chunk being written: dirty pages of a mapping count toward the
    process's RSS until the kernel writes them back.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: np.dtype,
        name: str = "dataobj",
        cache_dir: Path | str | None = None,
    ):
        self.owned_dir = None
        if cache_dir is None:
            cache_dir = self.owned_dir = Path(mkdtemp(prefix="nifreeze-"))
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.path = cache_dir / f"{name.strip('/').replace('/', '_')}.npy"
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        # Write the header and size the file without touching any data page
        header = np.lib.format.open_memmap(
            self.path, mode="w+", dtype=self.dtype, shape=self.shape
        )
        self.offset = header.offset
        del header
        self._file = open(self.path, "r+b")

    def __enter__(self) -> _VolumeCache:
        return self

    def __exit__(self, exc_type, *_) -> None:
        # On success the file stays for ``open_memmap``; on failure nothing is left.
        if exc_type is not None:
            self._file.close()
            _remove_cache(self.path, self.owned_dir, os.getpid())

    def write(self, start: tuple[int, ...], values: np.ndarray) -> None:
        """Write ``values``, which must be C-contiguous in the cache, at index ``start``."""
        position = int(np.ravel_multi_index(start, self.shape))
        self._file.seek(self.offset + position * self.dtype.itemsize)
        self._file.write(np.ascontiguousarray(values, dtype=self.dtype).data)

    def open_memmap(self) -> np.memmap:
        """Close the writer and map the cache; the file is removed with the map."""
        self._file.close()
        mm = np.lib.format.open_memmap(self.path, mode="r+")
        weakref.finalize(mm, _remove_cache, self.path, self.owned_dir, os.getpid())
        return mm


def _remove_cache(path: Path, owned_dir: Path | None, pid: int) -> None:
    # A forked child inherits the finalizer: only the creating process cleans up.
    if os.getpid() != pid:
        return
    path.unlink(missing_ok=True)
    if owned_dir is not None:
        shutil.rmtree(owned_dir, ignore_errors=True)


def open_dataobj_memmap(
    filename: Path | str,
    key: str = "/0/dataobj",
    mode: Literal["r", "c", "r+"] = "c",
    cache_dir: Path | str | None = None,
) -> np.ndarray:
    """Open a 4D ``dataobj`` as a lazy, per-volume-local :obj:`~numpy.memmap`.

    The returned array always has the in-memory shape ``(X, Y, Z, N)`` so that
    ``dataobj[..., k]`` selects volume ``k``. Two on-disk layouts are handled:

    - **volume-major** (``VOLUME_AXIS_ATTR == 0``, written by
      :func:`save_dataobj_volume_major`): directly mapped zero-copy when
      contiguous/uncompressed/native-endian, then transposed to ``(X, Y, Z, N)``
      — each volume is a contiguous block. Compressed or chunked datasets are
      decompressed one volume at a time into a temporary cache.
    - **legacy** ``(X, Y, Z, N)`` (no attribute): converted into a volume-major
      temporary cache in a single sequential pass over the file, reading slabs of
      at most :data:`CONVERSION_SLAB_BYTES`. Re-save the dataset with
      ``to_filename`` to skip this conversion on later loads.

    Temporary caches are deleted when the last array referencing them is
    released (or at interpreter exit).

    ``mode`` applies to the direct mapping: ``"c"`` (copy-on-write, the default)
    reads lazily from the file, while writes fault private pages and never mutate
    the source. Caches are always opened ``"r+"``.
    """
    filename = Path(filename)
    with h5py.File(filename, "r") as in_file:
        dset = in_file[key]
        shape = dset.shape
        dtype = dset.dtype
        volume_major = int(dset.attrs.get(VOLUME_AXIS_ATTR, -1)) == 0
        mappable = volume_major and _is_directly_mappable(dset)
        offset = dset.id.get_offset() if mappable else None

        if not mappable:
            if volume_major:
                with _VolumeCache(shape, dtype, key, cache_dir) as cache:
                    for i in range(shape[0]):
                        cache.write((i, 0, 0, 0), dset[i])
            else:
                # Legacy (X, Y, Z, N): an x-slab is contiguous in the file and
                # holds a contiguous run of every output volume.
                with _VolumeCache((shape[-1], *shape[:-1]), dtype, key, cache_dir) as cache:
                    plane_bytes = int(np.prod(shape[1:])) * dtype.itemsize
                    step = max(1, CONVERSION_SLAB_BYTES // plane_bytes)
                    for x0 in range(0, shape[0], step):
                        slab = dset[x0 : x0 + step]
                        for i in range(shape[-1]):
                            cache.write((i, x0, 0, 0), slab[..., i])
            return np.moveaxis(cache.open_memmap(), 0, -1)

    assert offset is not None  # guaranteed by mappable
    mm = np.memmap(filename, dtype=dtype, mode=mode, offset=offset, shape=shape, order="C")
    return np.moveaxis(mm, 0, -1)


def stream_select_last_axis(
    src: np.ndarray,
    keep_mask: np.ndarray,
    cache_dir: Path | str | None = None,
    name: str = "dataobj",
) -> np.ndarray:
    """Select volumes along the last axis into a disk-backed memmap.

    Copies ``src[..., keep_mask]`` one volume at a time into a volume-major
    temporary cache, which is deleted when the result is released. Used to drop
    b=0 volumes from a memory-mapped DWI without materializing it.
    """
    keep = np.flatnonzero(keep_mask)
    with _VolumeCache((keep.size, *src.shape[:-1]), src.dtype, name, cache_dir) as cache:
        for j, i in enumerate(keep):
            cache.write((j, 0, 0, 0), src[..., i])
    return np.moveaxis(cache.open_memmap(), 0, -1)


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


@contextmanager
def replace_on_success(filename: Path | str) -> Iterator[Path]:
    """Yield a temporary path next to ``filename`` and move it onto ``filename`` on success.

    ``h5py.File(filename, "w")`` truncates the file in place. If that file backs a
    memory-mapped ``dataobj`` (for example, writing a dataset back to the file
    it was loaded from), the mapping would lose its data while it is being
    written out. Renaming a new file over it keeps the old inode, and thus the
    mapping, valid.
    """
    filename = Path(filename)
    tmp = filename.with_name(f".{filename.name}.{uuid4().hex}.tmp")
    try:
        yield tmp
        os.replace(tmp, filename)
    finally:
        tmp.unlink(missing_ok=True)


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
