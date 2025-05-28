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
"""PET data representation."""

from __future__ import annotations
from collections import namedtuple

import json
from pathlib import Path
from typing import Any, Union

import attr
import h5py
import nibabel as nib
import numpy as np
from nibabel.spatialimages import SpatialImage
from nitransforms.linear import Affine

from nifreeze.data.base import BaseDataset, _cmp, _data_repr
from nifreeze.utils.ndimage import load_api


@attr.s(slots=True)
class PET(BaseDataset[np.ndarray | None]):
    """Data representation structure for PET data."""

    midframe_time: np.ndarray | None = attr.ib(
        default=None, repr=_data_repr, eq=attr.cmp_using(eq=_cmp)
    )
    """A (N,) numpy array specifying the midpoint timing of each sample or frame."""
    total_duration: float | None = attr.ib(default=None, repr=True)
    """A float representing the total duration of the dataset."""

    def _getextra(self, idx: int | slice | tuple | np.ndarray) -> tuple[np.ndarray | None]:
        return (self.midframe_time[idx] if self.midframe_time is not None else None,)

    # For the sake of the docstring
    def __getitem__(
        self, idx: int | slice | tuple | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """
        Returns volume(s) and corresponding affine(s) and timing(s) through fancy indexing.

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
        time : :obj:`float` or ``None``
            The frame time corresponding to the index(es).

        """
        return super().__getitem__(idx)

    def lofo_split(self, index, order: int = 3, pad_mode: str = "edge"):
        """
        Leave-one-frame-out (LOFO) for PET data.

        Parameters
        ----------
        index : int
            Index of the PET frame to be left out in this fold.
        pad_mode : :obj:`str`, optional
            Padding mode to be passed to :func:`numpy.pad` when extending the
            training data and timing arrays. Default is ``"edge"``.

        pad_mode : str
            Padding mode passed to :func:`numpy.pad`.

        Returns
        -------
        (train_data, train_timings) : tuple
            Training data and corresponding timings, excluding the left-out frame.
        (test_data, test_timing) : tuple
            Test data (one PET frame) and corresponding timing.
        """
        if not Path(self._filepath).exists():
            self.to_filename(self._filepath)

        # Read original PET data
        with h5py.File(self._filepath, "r") as in_file:
            root = in_file["/0"]
            pet_frame = np.asanyarray(root["dataobj"][..., index])
            if self.midframe_time is not None:
                timing_frame = np.asanyarray(root["midframe_time"][..., index])

        # Mask to exclude the selected frame
        mask = np.ones(self.dataobj.shape[-1], dtype=bool)
        mask[index] = False

        train_data = self.dataobj[..., mask]
        train_timings = (
            self.midframe_time[mask] if self.midframe_time is not None else None
        )
        if train_timings is not None:
            train_data = np.pad(
                train_data,
                [(0, 0), (0, 0), (0, 0), (order, order)],
                mode=pad_mode,
            )
            train_timings = np.pad(train_timings, (order, order), mode=pad_mode)

        test_data = pet_frame
        test_timing = timing_frame if self.midframe_time is not None else None

        return ((train_data, train_timings), (test_data, test_timing))

    def set_transform(self, index, affine, order=3):
        """Set an affine, and update data object and gradients."""
        reference = namedtuple("ImageGrid", ("shape", "affine"))(
            shape=self.dataobj.shape[:3], affine=self.affine
        )
        xform = Affine(matrix=affine, reference=reference)

        if not Path(self._filepath).exists():
            self.to_filename(self._filepath)

        # read original PET
        with h5py.File(self._filepath, "r") as in_file:
            root = in_file["/0"]
            dframe = np.asanyarray(root["dataobj"][..., index])

        dmoving = nib.Nifti1Image(dframe, self.affine, None)

        # resample and update orientation at index
        self.dataobj[..., index] = np.asanyarray(
            xform.apply(dmoving, order=order).dataobj,
            dtype=self.dataobj.dtype,
        )

        # update transform
        if self.em_affines is None:
            self.em_affines = [None] * len(self)

        self.em_affines[index] = xform

    def to_filename(self, filename, compression=None, compression_opts=None):
        """Write an HDF5 file to disk."""
        filename = Path(filename)
        if not filename.name.endswith(".h5"):
            filename = filename.parent / f"{filename.name}.h5"

        with h5py.File(filename, "w") as out_file:
            out_file.attrs["Format"] = "EMC/PET"
            out_file.attrs["Version"] = np.uint16(1)
            root = out_file.create_group("/0")
            root.attrs["Type"] = "pet"
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

    def to_nifti(self, filename, *_):
        """Write a NIfTI 1.0 file to disk."""
        nii = nib.Nifti1Image(self.dataobj, self.affine, None)
        nii.header.set_xyzt_units("mm")
        nii.to_filename(filename)

    @classmethod
    def from_filename(cls, filename):
        """Read an HDF5 file from disk."""
        with h5py.File(filename, "r") as in_file:
            root = in_file["/0"]
            data = {k: np.asanyarray(v) for k, v in root.items() if
                    not k.startswith("_")}
        return cls(**data)

    def load(
        filename,
        json_file,
        brainmask_file=None
    ):
        """Load PET data."""
        filename = Path(filename)
        if filename.name.endswith(".h5"):
            return PET.from_filename(filename)

        img = nib.load(filename)
        retval = PET(
            dataobj=img.get_fdata(dtype="float32"),
            affine=img.affine,
        )

        # Load metadata
        with open(json_file, 'r') as f:
            metadata = json.load(f)

        frame_duration = np.array(metadata['FrameDuration'])
        frame_times_start = np.array(metadata['FrameTimesStart'])
        midframe_time = frame_times_start + frame_duration / 2

        retval.midframe_time = midframe_time
        retval.total_duration = float(frame_times_start[-1] + frame_duration[-1])

        assert len(retval.midframe_time) == retval.dataobj.shape[-1]

        if brainmask_file:
            mask = nib.load(brainmask_file)
            retval.brainmask = np.asanyarray(mask.dataobj)

        return retval


def load(
    filename: Path | str,
    brainmask_file: Path | str | None = None,
    motion_file: Path | str | None = None,
    midframe_time: np.ndarray | list[float] | None = None,
    frame_duration: np.ndarray | list[float] | None = None,
) -> PET:
    """
    Load PET data from HDF5 or NIfTI, creating a PET object with appropriate metadata.

    Parameters
    ----------
    filename : :obj:`os.pathlike`
        The NIfTI or HDF5 file.
    brainmask_file : :obj:`os.pathlike`, optional
        A brainmask NIfTI file. If provided, will be loaded and
        stored in the returned dataset.
    motion_file : :obj:`os.pathlike`
        A file containing head-motion affine matrices (linear).
    midframe : :obj:`numpy.ndarray` or :obj:`list` of :obj:`float`, optional
        The midframe times of each frame relative to the beginning of the acquisition.
        If ``None``, an error is raised (since BIDS requires ``FrameTimesStart``).
    frame_duration : :obj:`numpy.ndarray` or :obj:`list` of :obj:`float`, optional
        The duration of each frame.
        If ``None``, it is derived by the difference of consecutive frame times,
        defaulting the last frame to match the second-last.

    Returns
    -------
    :obj:`~nifreeze.data.pet.PET`
        A PET object storing the data, metadata, and any optional mask.

    Raises
    ------
    RuntimeError
        If ``midframe_time`` is not provided (BIDS requires it).

    """
    if motion_file:
        raise NotImplementedError

    filename = Path(filename)
    if filename.suffix == ".h5":
        # Load from HDF5
        pet_obj = PET.from_filename(filename)
    else:
        # Load from NIfTI
        img = load_api(filename, SpatialImage)
        data = img.get_fdata(dtype=np.float32)
        pet_obj = PET(
            dataobj=data,
            affine=img.affine,
        )

    # Verify the user provided frame_time if not already in the PET object
    if pet_obj.midframe_time is None and midframe_time is None:
        raise RuntimeError(
            "The `midframe_time` is mandatory for PET data to comply with BIDS. "
            "See https://bids-specification.readthedocs.io for details."
        )

    # If the user supplied new values, set them
    if midframe_time is not None:
        # Convert to a float32 numpy array and zero out the earliest time
        midframe_time_arr = np.array(midframe_time, dtype=np.float32)
        midframe_time_arr -= midframe_time_arr[0]
        pet_obj.midframe_time = midframe_time_arr

    # If the user doesn't provide frame_duration, we derive it:
    if frame_duration is None:
        if pet_obj.midframe_time is not None:
            midframe_time_arr = pet_obj.midframe_time
            # If shape is e.g. (N,), then we can do
            durations = np.diff(midframe_time_arr)
            if len(durations) == (len(midframe_time_arr) - 1):
                durations = np.append(durations, durations[-1])  # last frame same as second-last
    else:
        durations = np.array(frame_duration, dtype=np.float32)

    # Set total_duration and shift frame_time to the midpoint
    pet_obj.total_duration = float(midframe_time_arr[-1] + durations[-1])
    pet_obj.midframe_time = midframe_time_arr + 0.5 * durations

    # If a brain mask is provided, load and attach
    if brainmask_file is not None:
        mask_img = load_api(brainmask_file, SpatialImage)
        pet_obj.brainmask = np.asanyarray(mask_img.dataobj, dtype=bool)

    return pet_obj
