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

import attr
import h5py
import nibabel as nb
import numpy as np
from nibabel.spatialimages import SpatialHeader, SpatialImage
from nitransforms.linear import Affine
from typing_extensions import TypeVarTuple, Unpack

from nifreeze.utils.ndimage import load_api

NFDH5_EXT = ".h5"


Ts = TypeVarTuple("Ts")


def _data_repr(value: np.ndarray | None) -> str:
    if value is None:
        return "None"
    return f"<{'x'.join(str(v) for v in value.shape)} ({value.dtype})>"


def _cmp(lh: Any, rh: Any) -> bool:
    if isinstance(lh, np.ndarray) and isinstance(rh, np.ndarray):
        return np.allclose(lh, rh)

    return lh == rh


@attr.s(slots=True)
class BaseDataset(Generic[Unpack[Ts]]):
    """
    Base dataset representation structure.

    A general data structure to represent 4D images and the necessary metadata
    for head-motion estimation (that is, potentially a brain mask and the head-motion
    estimates).

    The data structure has a direct HDF5 mapping to facilitate memory efficiency.
    For modalities requiring additional metadata such as DWI (which requires the gradient table
    and potentially a b=0 reference), this class may be derived to override certain behaviors
    (in the case of DWIs, the indexed access should also return the corresponding gradient
    specification).

    """

    dataobj: np.ndarray = attr.ib(default=None, repr=_data_repr, eq=attr.cmp_using(eq=_cmp))
    """A :obj:`~numpy.ndarray` object for the data array."""
    affine: np.ndarray = attr.ib(default=None, repr=_data_repr, eq=attr.cmp_using(eq=_cmp))
    """Best affine for RAS-to-voxel conversion of coordinates (NIfTI header)."""
    brainmask: np.ndarray = attr.ib(default=None, repr=_data_repr, eq=attr.cmp_using(eq=_cmp))
    """A boolean ndarray object containing a corresponding brainmask."""
    motion_affines: np.ndarray = attr.ib(default=None, eq=attr.cmp_using(eq=_cmp))
    """List of :obj:`~nitransforms.linear.Affine` realigning the dataset."""
    datahdr: SpatialHeader = attr.ib(default=None)
    """A :obj:`~nibabel.spatialimages.SpatialHeader` header corresponding to the data."""

    _filepath = attr.ib(
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

    @classmethod
    def from_filename(cls, filename: Path | str) -> BaseDataset:
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

    def set_transform(self, index: int, affine: np.ndarray, order: int = 3) -> None:
        """
        Set an affine transform for a particular index and update the data object.

        Parameters
        ----------
        index : :obj:`int`
            The volume index to transform.
        affine : :obj:`numpy.ndarray`
            The 4x4 affine matrix to be applied.
        order : :obj:`int`, optional
            The order of the spline interpolation.

        """
        ImageGrid = namedtuple("ImageGrid", ("shape", "affine"))
        reference = ImageGrid(shape=self.dataobj.shape[:3], affine=self.affine)

        xform = Affine(matrix=affine, reference=reference)

        if not Path(self._filepath).exists():
            self.to_filename(self._filepath)

        # read original DWI data & b-vector
        with h5py.File(self._filepath, "r") as in_file:
            root = in_file["/0"]
            dataframe = np.asanyarray(root["dataobj"][..., index])

        datamoving = nb.Nifti1Image(dataframe, self.affine, None)

        # resample and update orientation at index
        self.dataobj[..., index] = np.asanyarray(
            xform.apply(datamoving, order=order).dataobj,
            dtype=self.dataobj.dtype,
        )

        # if head motion affines are to be used, initialized to identities
        if self.motion_affines is None:
            self.motion_affines = np.repeat(np.zeros((4, 4))[None, ...], len(self), axis=0)

        self.motion_affines[index] = xform.matrix

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
            for f in attr.fields(self.__class__):
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

    def to_nifti(self, filename: Path | str) -> None:
        """
        Write a NIfTI file to disk.

        Parameters
        ----------
        filename : :obj:`os.pathlike`
            The output NIfTI file path.

        """
        nii = nb.Nifti1Image(self.dataobj, self.affine, self.datahdr)
        if self.datahdr is None:
            nii.header.set_xyzt_units("mm")
        nii.to_filename(filename)


def load(
    filename: Path | str,
    brainmask_file: Path | str | None = None,
    motion_file: Path | str | None = None,
) -> BaseDataset[()]:
    """
    Load 4D data from a filename or an HDF5 file.

    Parameters
    ----------
    filename : :obj:`os.pathlike`
        The NIfTI or HDF5 file.
    brainmask_file : :obj:`os.pathlike`, optional
        A brainmask NIfTI file. If provided, will be loaded and
        stored in the returned dataset.
    motion_file : :obj:`os.pathlike`
        A file containing head-motion affine matrices (linear).

    Returns
    -------
    :obj:`~nifreeze.data.base.BaseDataset`
        The loaded dataset.

    Raises
    ------
    ValueError
        If the file extension is not supported or the file cannot be loaded.

    """
    if motion_file:
        raise NotImplementedError

    filename = Path(filename)
    if filename.name.endswith(NFDH5_EXT):
        return BaseDataset.from_filename(filename)

    img = load_api(filename, SpatialImage)
    retval: BaseDataset[()] = BaseDataset(dataobj=np.asanyarray(img.dataobj), affine=img.affine)

    if brainmask_file:
        mask = load_api(brainmask_file, SpatialImage)
        retval.brainmask = np.asanyarray(mask.dataobj)

    return retval
