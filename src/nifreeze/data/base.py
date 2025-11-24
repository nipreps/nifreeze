# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
"""Four-dimensional data representation in hard-disk and memory."""

from __future__ import annotations

from collections import namedtuple
from pathlib import Path
from tempfile import mkdtemp
from typing import Any, Generic, Protocol, runtime_checkable
from warnings import warn

import attrs
import h5py
import nibabel as nb
import numpy as np
from nitransforms.linear import LinearTransformsMapping
from nitransforms.resampling import apply
from typing_extensions import Self, TypeVarTuple, Unpack

Ts = TypeVarTuple("Ts")

NFDH5_EXT = ".h5"

ImageGrid = namedtuple("ImageGrid", ("shape", "affine"))

DATAOBJ_ABSENCE_ERROR_MSG = "BaseDataset 'dataobj' may not be None"
"""BaseDataset initialization dataobj absence error message."""

DATAOBJ_OBJECT_ERROR_MSG = "BaseDataset 'dataobj' must be an array-like object."
"""BaseDataset initialization dataobj object error message."""

DATAOBJ_NDIM_ERROR_MSG = "BaseDataset 'dataobj' must be a 4-D array-like object"
"""BaseDataset initialization dataobj dimensionality error message."""

AFFINE_ABSENCE_ERROR_MSG = "BaseDataset 'affine' may not be None"
"""BaseDataset initialization affine absence error message."""

AFFINE_OBJECT_ERROR_MSG = "BaseDataset 'affine' must be a numpy array."
"""BaseDataset initialization affine object error message."""

AFFINE_NDIM_ERROR_MSG = "BaseDataset 'affine' must be a 2D array"
"""Affine dimensionality error message."""

AFFINE_SHAPE_ERROR_MSG = "BaseDataset 'affine' must be a 2D numpy array (4 x 4)"
"""BaseDataset initialization affine shape error message."""

BRAINMASK_SHAPE_MISMATCH_ERROR_MSG = "BaseDataset 'brainmask' shape ({brainmask_shape}) does not match dataset volumes ({data_shape})."
"""BaseDataset brainmask shape mismatch error message."""


def _has_dim_size(value: Any, size: int) -> bool:
    """Return :obj:`True` if ``value`` has a ``.shape`` attribute and one of its
     dimensions equals ``size``.

    This is useful for checks where at least one axis must match an expected
    length. It does not require a specific axis index; it only verifies presence
    of the size in any axis in ``.shape``.

    Parameters
    ----------
    value : :obj:`Any`
        Object to inspect. Typical inputs are NumPy arrays or objects exposing
        ``.shape``.
    size : :obj:`int`
        The required dimension size to look for in ``value.shape``.

    Returns
    -------
     :obj:`bool`
        :obj:`True` if ``.shape`` exists and any of its integers equals ``size``,
        :obj:`False` otherwise.

    Examples
    --------
    >>> _has_dim_size(np.zeros((10, 3)), 3)
    True
    >>> _has_dim_size(np.zeros((4, 5)), 6)
    False
    """

    shape = getattr(value, "shape", None)
    if shape is None:
        return False
    # Shape may be an object that is not iterable; handle TypeError explicitly
    try:
        return size in tuple(shape)
    except TypeError:
        return False


def _has_ndim(value: Any, ndim: int) -> bool:
    """Check if ``value`` has ``ndim`` dimensionality.

    Returns :obj:`True` if `value` has an ``.ndim`` attribute equal to ``ndim``,
    or if it has a ``.shape`` attribute whose length equals ``ndim``.

    This helper is tolerant: it accepts objects that either:
    - expose an integer ``.ndim`` attribute (e.g., NumPy arrays), or
    - expose a ``.shape`` attribute (sequence/tuple-like) whose length equals
     ``ndim``.

    Parameters
    ----------
    value : :obj:`Any`
        Object to inspect for dimensionality. Typical inputs are NumPy arrays,
        array-likes, or objects that provide ``.ndim`` / ``.shape``.
    ndim : :obj:`int`
        The required dimensionality.

    Returns
    -------
    :obj:`bool`
        :obj:`True` if ``value`` appears to have ``ndim`` dimensions,
        :obj:`False` otherwise.

    Examples
    --------
    >>> _has_ndim(np.zeros((2, 3)), 2)
    True
    >>> _has_ndim(np.zeros((3,)), 2)
    False
    >>> class WithShape:
    ...     shape = (2, 2, 2)
    >>> _has_ndim(WithShape(), 3)
    True
    """

    # Prefer .ndim if available
    ndim_attr = getattr(value, "ndim", None)
    if ndim_attr is not None:
        try:
            return int(ndim_attr) == ndim
        except (TypeError, ValueError):
            return False

    # Fallback to checking shape length
    shape = getattr(value, "shape", None)
    if shape is None:
        return False
    try:
        return len(tuple(shape)) == ndim
    except TypeError:
        return False


def _data_repr(value: Any) -> str:
    if value is None:
        return "None"

    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is None:
        return repr(value)

    return f"<{'x'.join(str(v) for v in tuple(shape))} ({dtype})>"


def _cmp(lh: Any, rh: Any) -> bool:
    lh_is_array = _has_ndim(lh, 0) or hasattr(lh, "shape")
    rh_is_array = _has_ndim(rh, 0) or hasattr(rh, "shape")
    if lh_is_array and rh_is_array:
        try:
            return np.allclose(np.asarray(lh), np.asarray(rh))
        except Exception:
            return False

    return lh == rh


@runtime_checkable
class _ArrayLike(Protocol):
    """Minimal protocol for array-like objects used by :class:`BaseDataset`."""

    shape: tuple[int, ...]
    dtype: Any

    def __getitem__(self, key: Any) -> Any:  # pragma: no cover - structural protocol
        ...


def _is_array_like(value: Any) -> bool:
    """Return ``True`` when ``value`` looks like an array."""

    if value is None:
        return False

    return hasattr(value, "__getitem__") and hasattr(value, "shape") and hasattr(value, "dtype")


def validate_dataobj(inst: BaseDataset, attr: attrs.Attribute, value: Any) -> None:
    """Strict validator for data objects.

    Enforces that ``value`` is present and exposes array-like properties with
    exactly 4 dimensions (``ndim == 4``).

    This function is intended for use as an attrs-style validator.

    Parameters
    ----------
    inst : :obj:`~nifreeze.data.base.BaseDataset`
        The instance being validated (unused, present for validator signature).
    attr : :obj:`~attrs.Attribute`
        The attribute being validated (unused, present for validator signature).
    value : :obj:`Any`
        The value to validate.

    Raises
    ------
    exc:`TypeError`
        If the input cannot be converted to a float :obj:`~numpy.ndarray`.
    exc:`ValueError`
        If the value is :obj:`None`, or not 4-dimensional.
    """
    if value is None:
        raise ValueError(DATAOBJ_ABSENCE_ERROR_MSG)

    if not _is_array_like(value):
        raise TypeError(DATAOBJ_OBJECT_ERROR_MSG)

    if not _has_ndim(value, 4):
        raise ValueError(DATAOBJ_NDIM_ERROR_MSG)


def validate_affine(inst: BaseDataset, attr: attrs.Attribute, value: Any) -> None:
    """Strict validator for affine matrices.

    Enforces that ``value`` is present and is a 4x4 NumPy array.

    This function is intended for use as an attrs-style validator.

    Parameters
    ----------
    inst : :obj:`~nifreeze.data.base.BaseDataset`
        The instance being validated (unused, present for validator signature).
    attr : :obj:`~attrs.Attribute`
        The attribute being validated (unused, present for validator signature).
    value : :obj:`Any`
        The value to validate.

    Raises
    ------
    exc:`TypeError`
        If the input cannot be converted to a float :obj:`~numpy.ndarray`.
    exc:`ValueError`
        If the value is :obj:`None`, or not shaped ``(4, 4)``.
    """
    if value is None:
        raise ValueError(AFFINE_ABSENCE_ERROR_MSG)

    if not isinstance(value, np.ndarray):
        raise TypeError(AFFINE_OBJECT_ERROR_MSG)

    if not _has_ndim(value, 2):
        raise ValueError(AFFINE_NDIM_ERROR_MSG)

    if value.shape != (4, 4):
        raise ValueError(AFFINE_SHAPE_ERROR_MSG)


@attrs.define(slots=True, eq=False)
class BaseDataset(Generic[Unpack[Ts]]):
    """
    Base dataset representation structure.

    A general data structure to represent 4D images and the necessary metadata
    for head motion estimation (that is, potentially a brain mask and the head
    motion estimates).

    The data structure has a direct HDF5 mapping to facilitate memory efficiency.
    For modalities requiring additional metadata such as DWI (which requires the gradient table
    and potentially a b=0 reference), this class may be derived to override certain behaviors
    (in the case of DWIs, the indexed access should also return the corresponding gradient
    specification).

    """

    dataobj: _ArrayLike = attrs.field(
        default=None, repr=_data_repr, eq=attrs.cmp_using(eq=_cmp), validator=validate_dataobj
    )
    """A 4D array-like object for the data array."""
    affine: np.ndarray = attrs.field(
        default=None, repr=_data_repr, eq=attrs.cmp_using(eq=_cmp), validator=validate_affine
    )
    """Best affine for RAS-to-voxel conversion of coordinates (NIfTI header)."""
    brainmask: np.ndarray | None = attrs.field(
        default=None, repr=_data_repr, eq=attrs.cmp_using(eq=_cmp)
    )
    """A boolean ndarray object containing a corresponding brainmask."""
    motion_affines: np.ndarray | None = attrs.field(default=None, eq=attrs.cmp_using(eq=_cmp))
    """Array of :obj:`~nitransforms.linear.Affine` realigning the dataset."""
    datahdr: nb.Nifti1Header | None = attrs.field(default=None)
    """A :obj:`~nibabel.Nifti1Header` header corresponding to the data."""

    _filepath: Path = attrs.field(
        factory=lambda: Path(mkdtemp()) / "hmxfms_cache.h5",
        repr=False,
        eq=False,
    )
    """A path to an HDF5 file to store the whole dataset."""
    _file_handle: h5py.File | None = attrs.field(default=None, repr=False, eq=False)
    """An open HDF5 file handle keeping :attr:`dataobj` alive when using HDF5-backed arrays."""
    _mmap_path: Path | None = attrs.field(default=None, repr=False, eq=False)
    """Path to a memory-mapped file backing :attr:`dataobj`, when available."""

    def __attrs_post_init__(self) -> None:
        """Enforce basic consistency of base dataset fields at instantiation
        time.

        Specifically, the brainmask (if present) must match spatial shape of
        dataobj.
        """

        if self.brainmask is not None:
            if self.brainmask.shape != tuple(self.dataobj.shape[:3]):
                raise ValueError(
                    BRAINMASK_SHAPE_MISMATCH_ERROR_MSG.format(
                        brainmask_shape=self.brainmask.shape, data_shape=self.dataobj.shape[:3]
                    )
                )

        if self._file_handle is None and isinstance(self.dataobj, h5py.Dataset):
            # Pin the open handle to ensure slices remain readable
            try:
                self._file_handle = self.dataobj.file  # type: ignore[assignment]
            except Exception:
                self._file_handle = None

        if self._mmap_path is None and isinstance(self.dataobj, np.memmap):
            try:
                self._mmap_path = Path(self.dataobj.filename)
            except Exception:
                self._mmap_path = None

    def __len__(self) -> int:
        """Obtain the number of volumes/frames in the dataset."""
        return self.dataobj.shape[-1]

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented

        assert isinstance(other, BaseDataset)

        base_equal = all(
            (
                _cmp(self.dataobj, other.dataobj),
                _cmp(self.affine, other.affine),
                _cmp(self.brainmask, other.brainmask),
                _cmp(self.motion_affines, other.motion_affines),
                self.datahdr == other.datahdr,
            )
        )

        return base_equal and self._eq_extras(other)

    def _eq_extras(self, other: BaseDataset) -> bool:
        """Additional equality checks implemented by subclasses."""

        return True

    def _getextra(self, idx: int | slice | tuple | np.ndarray) -> tuple[Unpack[Ts]]:
        """
        Extract extra fields for a given index of the corresponding data object.

        Parameters
        ----------
        idx : :obj:`int` or :obj:`slice` or :obj:`tuple` or :obj:`~numpy.ndarray`
            Index (or indexing type/object) for which extra information will be extracted.

        Returns
        -------
        :obj:`tuple`
            A tuple with the extra fields (may be an empty tuple if no extra fields are defined).

        """
        _ = idx  # Avoid unused parameter warning
        return ()  # type: ignore[return-value]

    def __getitem__(
        self, idx: int | slice | tuple | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray | None, Unpack[Ts]]:
        """
        Returns volume(s) and corresponding affine(s) through fancy indexing.

        Parameters
        ----------
        idx : :obj:`int` or :obj:`slice` or :obj:`tuple` or :obj:`~numpy.ndarray`
            Indexer for the last dimension (or possibly other dimensions if extended).

        Returns
        -------
        :obj:`~numpy.ndarray`
            The selected data subset.
            If ``idx`` is a single integer, this will have shape ``(X, Y, Z)``,
            otherwise it may have shape ``(X, Y, Z, k)``.
        affine : :obj:`~numpy.ndarray` or :obj:`None`
            The corresponding per-volume motion affine(s) or :obj:`None` if identity transform(s).
        Unpack[:obj:`~nifreeze.data.base.Ts`]
            Zero or more additional per-volume fields returned as unpacked
            trailing elements. The exact number, order, and types of elements
            are determined by the type variables :obj:`~nifreeze.data.base.Ts`
            and by the values returned from :meth:`_getextra`. Subclasses
            provide these values by implementing :meth:`_getextra`. If no extra
            fields are defined, no element is returned.
            Example usages:
            - vols, aff, *extras = dataset[0:10]
            - vol, aff, bvecs, bvals = dataset[0]  # when two extras are present

        """

        affine = self.motion_affines[idx] if self.motion_affines is not None else None
        return self.dataobj[..., idx], affine, *self._getextra(idx)

    @property
    def shape3d(self):
        """Get the shape of the 3D volume."""
        return self.dataobj.shape[:3]

    @property
    def size3d(self):
        """Get the number of voxels in the 3D volume."""
        return np.prod(self.dataobj.shape[:3])

    @classmethod
    def from_filename(cls, filename: Path | str, *, keep_file_open: bool = True) -> Self:
        """
        Read an HDF5 file from disk and create a BaseDataset.

        Parameters
        ----------
        filename : :obj:`os.pathlike`
            The HDF5 file path to read.
        keep_file_open : :obj:`bool`, optional
            When ``True`` (default), keep the HDF5 file handle open and store
            datasets directly to enable on-demand slicing without loading the
            full array into memory. When ``False``, datasets are eagerly read
            into NumPy arrays.

        Returns
        -------
        :obj:`~nifreeze.data.base.BaseDataset`
            The constructed dataset with data loaded from the file.

        """
        filename = Path(filename)
        if keep_file_open:
            in_file = h5py.File(filename, "r")
            root = in_file["/0"]
            data = {k: v for k, v in root.items() if not k.startswith("_")}
            data["_file_handle"] = in_file
            data["_filepath"] = filename
            return cls(**data)

        with h5py.File(filename, "r") as in_file:
            root = in_file["/0"]
            data = {k: np.asanyarray(v) for k, v in root.items() if not k.startswith("_")}
            data["_filepath"] = filename

        return cls(**data)

    def get_filename(self) -> Path:
        """Get the filepath of the HDF5 file."""
        return self._filepath

    def close(self) -> None:
        """Close any open backing resources held by this dataset."""

        if self._file_handle is not None:
            try:
                self._file_handle.close()
            finally:
                self._file_handle = None

    def set_transform(self, index: int, affine: np.ndarray) -> None:
        """
        Set an affine transform for a particular index and update the data object.

        Parameters
        ----------
        index : :obj:`int`
            The volume index to transform.
        affine : :obj:`~numpy.ndarray`
            The 4x4 affine matrix to be applied.

        """
        # if head motion affines are to be used, initialized to identities
        if self.motion_affines is None:
            self.motion_affines = np.repeat(np.eye(4)[None, ...], len(self), axis=0)

        self.motion_affines[index] = affine

    def to_filename(
        self, filename: Path | str, compression: str | None = None, compression_opts: Any = None
    ) -> None:
        """
        Write an HDF5 file to disk.

        Parameters
        ----------
        filename : :obj:`os.pathlike`
            The HDF5 file path to write to.
        compression : :obj:`str`, optional
            Compression strategy.
            See :obj:`~h5py.Group.create_dataset` documentation.
        compression_opts : :obj:`typing.Any`, optional
            Parameters for compression
            `filters <https://docs.h5py.org/en/stable/high/dataset.html#dataset-compression>`__.

        """
        filename = Path(filename)
        if not filename.name.endswith(NFDH5_EXT):
            filename = filename.parent / f"{filename.name}.h5"

        with h5py.File(filename, "w") as out_file:
            out_file.attrs["Format"] = "NFDH5"  # NiFreeze Data HDF5
            out_file.attrs["Version"] = np.uint16(1)
            root = out_file.create_group("/0")
            root.attrs["Type"] = "base dataset"
            for f in attrs.fields(self.__class__):
                if f.name.startswith("_"):
                    continue

                value = getattr(self, f.name)
                if value is not None:
                    root.create_dataset(
                        f.name,
                        data=value,
                        compression=compression,
                        compression_opts=compression_opts,
                    )

    def to_nifti(
        self,
        filename: Path | str | None = None,
        write_hmxfms: bool = False,
        order: int = 3,
    ) -> nb.nifti1.Nifti1Image:
        """
        Write a NIfTI file to disk.

        Volumes are resampled to the reference affine if motion affines have
        been set, otherwise the original data are written.

        Parameters
        ----------
        filename : :obj:`os.pathlike`, optional
            The output NIfTI file path.
        write_hmxfms : :obj:`bool`, optional
            If :obj:`True`, the head motion affines will be written out to
            filesystem with BIDS' X5 format.
        order : :obj:`int`, optional
            The interpolation order to use when resampling the data.
            Defaults to 3 (cubic interpolation).

        Returns
        -------
        :obj:`~nibabel.nifti1.Nifti1Image`
            NIfTI image written to disk.
        """

        return to_nifti(
            self,
            filename=filename,
            write_hmxfms=write_hmxfms,
            order=order,
        )


def to_nifti(
    dataset: BaseDataset,
    filename: Path | str | None = None,
    write_hmxfms: bool = False,
    order: int = 3,
) -> nb.nifti1.Nifti1Image:
    """
    Write a NIfTI file to disk.

    Volumes are resampled to the reference affine if motion affines have
    been set, otherwise the original data are written.

    Parameters
    ----------
    dataset : :obj:`~nifreeze.data.base.BaseDataset`
        The dataset to serialize.
    filename : :obj:`os.pathlike`, optional
        The output NIfTI file path.
    write_hmxfms : :obj:`bool`, optional
        If :obj:`True`, the head motion affines will be written out to filesystem
        with BIDS' X5 format.
    order : :obj:`int`, optional
        The interpolation order to use when resampling the data.
        Defaults to 3 (cubic interpolation).

    Returns
    -------
    :obj:`~nibabel.nifti1.Nifti1Image`
        NIfTI image written to disk.
    """

    if filename is None and write_hmxfms:
        warn("write_hmxfms is set to True, but no filename was provided.", stacklevel=2)
        write_hmxfms = False

    if dataset.motion_affines is not None:  # resampling is needed
        reference = ImageGrid(shape=dataset.dataobj.shape[:3], affine=dataset.affine)
        resampled = np.empty(dataset.dataobj.shape, dtype=dataset.dataobj.dtype)
        xforms = LinearTransformsMapping(dataset.motion_affines, reference=reference)

        # This loop could be replaced by nitransforms.resampling.apply() when
        # it is fixed (bug should only affect datasets with less than 9 orientations)
        for i, xform in enumerate(xforms):
            frame = dataset[i]
            datamoving = nb.Nifti1Image(frame[0], dataset.affine, dataset.datahdr)
            # resample at index
            resampled[..., i] = np.asanyarray(
                apply(xform, datamoving, order=order).dataobj,
                dtype=dataset.dataobj.dtype,
            )

            if filename is not None and write_hmxfms:
                # Prepare filename and write out
                out_root = Path(filename).absolute()
                out_root = out_root.parent / out_root.name.replace("".join(out_root.suffixes), "")
                xform.to_filename(out_root.with_suffix(".x5"))
    else:
        resampled = np.asanyarray(dataset.dataobj)

        if write_hmxfms:
            warn(
                "write_hmxfms is set to True, but no motion affines were found. Skipping.",
                stacklevel=2,
            )

    if dataset.datahdr is None:
        hdr = nb.Nifti1Header()
        hdr.set_xyzt_units("mm")
        hdr.set_data_dtype(dataset.dataobj.dtype)
    else:
        hdr = dataset.datahdr.copy()

    nii = nb.Nifti1Image(resampled, dataset.affine, hdr)
    if filename is not None:
        nii.to_filename(filename)

    return nii
