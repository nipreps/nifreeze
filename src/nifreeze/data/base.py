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
from typing import Any, Generic
from warnings import warn

import attrs
import h5py
import nibabel as nb
import numpy as np
from nibabel.spatialimages import SpatialHeader
from nitransforms.linear import LinearTransformsMapping
from typing_extensions import Self, TypeVarTuple, Unpack

Ts = TypeVarTuple("Ts")

NFDH5_EXT = ".h5"

ImageGrid = namedtuple("ImageGrid", ("shape", "affine"))


def _data_repr(value: np.ndarray | None) -> str:
    if value is None:
        return "None"
    return f"<{'x'.join(str(v) for v in value.shape)} ({value.dtype})>"


def _cmp(lh: Any, rh: Any) -> bool:
    if isinstance(lh, np.ndarray) and isinstance(rh, np.ndarray):
        return np.allclose(lh, rh)

    return lh == rh


@attrs.define(slots=True)
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

    dataobj: np.ndarray = attrs.field(default=None, repr=_data_repr, eq=attrs.cmp_using(eq=_cmp))
    """A :obj:`~numpy.ndarray` object for the data array."""
    affine: np.ndarray = attrs.field(default=None, repr=_data_repr, eq=attrs.cmp_using(eq=_cmp))
    """Best affine for RAS-to-voxel conversion of coordinates (NIfTI header)."""
    brainmask: np.ndarray = attrs.field(default=None, repr=_data_repr, eq=attrs.cmp_using(eq=_cmp))
    """A boolean ndarray object containing a corresponding brainmask."""
    motion_affines: np.ndarray = attrs.field(default=None, eq=attrs.cmp_using(eq=_cmp))
    """List of :obj:`~nitransforms.linear.Affine` realigning the dataset."""
    datahdr: SpatialHeader = attrs.field(default=None)
    """A :obj:`~nibabel.spatialimages.SpatialHeader` header corresponding to the data."""

    _filepath: Path = attrs.field(
        factory=lambda: Path(mkdtemp()) / "hmxfms_cache.h5",
        repr=False,
        eq=False,
    )
    """A path to an HDF5 file to store the whole dataset."""

    def __len__(self) -> int:
        """Obtain the number of volumes/frames in the dataset."""
        if self.dataobj is None:
            return 0

        return self.dataobj.shape[-1]

    def _getextra(self, idx: int | slice | tuple | np.ndarray) -> tuple[Unpack[Ts]]:
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
        volumes : :obj:`~numpy.ndarray`
            The selected data subset.
            If ``idx`` is a single integer, this will have shape ``(X, Y, Z)``,
            otherwise it may have shape ``(X, Y, Z, k)``.
        motion_affine : :obj:`~numpy.ndarray` or ``None``
            The corresponding per-volume motion affine(s) or ``None`` if identity transform(s).

        """
        if self.dataobj is None:
            raise ValueError("No data available (dataobj is None).")

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
    def from_filename(cls, filename: Path | str) -> Self:
        """
        Read an HDF5 file from disk and create a BaseDataset.

        Parameters
        ----------
        filename : :obj:`os.pathlike`
            The HDF5 file path to read.

        Returns
        -------
        :obj:`~nifreeze.data.base.BaseDataset`
            The constructed dataset with data loaded from the file.

        """
        with h5py.File(filename, "r") as in_file:
            root = in_file["/0"]
            data = {k: np.asanyarray(v) for k, v in root.items() if not k.startswith("_")}
        return cls(**data)

    def get_filename(self) -> Path:
        """Get the filepath of the HDF5 file."""
        return self._filepath

    def set_transform(self, index: int, affine: np.ndarray) -> None:
        """
        Set an affine transform for a particular index and update the data object.

        Parameters
        ----------
        index : :obj:`int`
            The volume index to transform.
        affine : :obj:`numpy.ndarray`
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
    ) -> nb.Nifti1Image:
        """
        Write a NIfTI file to disk.

        Volumes are resampled to the reference affine if motion affines have
        been set, otherwise the original data are written.

        Parameters
        ----------
        filename : :obj:`os.pathlike`, optional
            The output NIfTI file path.
        write_hmxfms : :obj:`bool`, optional
            If ``True``, the head motion affines will be written out to filesystem
            with BIDS' X5 format.
        order : :obj:`int`, optional
            The interpolation order to use when resampling the data.
            Defaults to 3 (cubic interpolation).

        """

        if filename is None and write_hmxfms:
            warn("write_hmxfms is set to True, but no filename was provided.", stacklevel=2)
            write_hmxfms = False

        if self.motion_affines is not None:  # resampling is needed
            reference = ImageGrid(shape=self.dataobj.shape[:3], affine=self.affine)
            resampled = np.empty_like(self.dataobj, dtype=self.dataobj.dtype)
            xforms = LinearTransformsMapping(self.motion_affines, reference=reference)

            # This loop could be replaced by nitransforms.resampling.apply() when
            # it is fixed (bug should only affect datasets with less than 9 orientations)
            for i, xform in enumerate(xforms):
                frame = self[i]
                datamoving = nb.Nifti1Image(frame[0], self.affine, self.datahdr)
                # resample at index
                resampled[..., i] = np.asanyarray(
                    xform.apply(datamoving, order=order).dataobj,
                    dtype=self.dataobj.dtype,
                )

                if filename is not None and write_hmxfms:
                    # Prepare filename and write out
                    out_root = Path(filename).absolute()
                    out_root = out_root.parent / out_root.name.replace(
                        "".join(out_root.suffixes), ""
                    )
                    xform.to_filename(out_root.with_suffix(".x5"))
        else:
            resampled = self.dataobj

            if write_hmxfms:
                warn(
                    "write_hmxfms is set to True, but no motion affines were found. Skipping.",
                    stacklevel=2,
                )

        if self.datahdr is None:
            hdr = nb.Nifti1Header()
            hdr.set_xyzt_units("mm")
            hdr.set_data_dtype(self.dataobj.dtype)
        else:
            hdr = self.datahdr.copy()

        nii = nb.Nifti1Image(resampled, self.affine, hdr)
        if filename is not None:
            nii.to_filename(filename)

        return nii
