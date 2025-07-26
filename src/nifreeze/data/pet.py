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

import json
from collections import namedtuple
from pathlib import Path

import attrs
import h5py
import nibabel as nb
import numpy as np
from nibabel.spatialimages import SpatialImage
from nitransforms.linear import Affine

from nifreeze.data.base import BaseDataset, _cmp, _data_repr
from nifreeze.utils.ndimage import load_api


@attrs.define(slots=True)
class PET(BaseDataset[np.ndarray | None]):
    """Data representation structure for PET data."""

    midframe: np.ndarray | None = attrs.field(
        default=None, repr=_data_repr, eq=attrs.cmp_using(eq=_cmp)
    )
    """A (N,) numpy array specifying the midpoint timing of each sample or frame."""
    total_duration: float | None = attrs.field(default=None, repr=True)
    """A float representing the total duration of the dataset."""

    def _getextra(self, idx: int | slice | tuple | np.ndarray) -> tuple[np.ndarray | None]:
        return (self.midframe[idx] if self.midframe is not None else None,)

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

    def lofo_split(self, index):
        """
        Leave-one-frame-out (LOFO) for PET data.

        Parameters
        ----------
        index : int
            Index of the PET frame to be left out in this fold.

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
            if self.midframe is not None:
                timing_frame = np.asanyarray(root["midframe"][..., index])

        # Mask to exclude the selected frame
        mask = np.ones(self.dataobj.shape[-1], dtype=bool)
        mask[index] = False

        train_data = self.dataobj[..., mask]
        train_timings = self.midframe[mask] if self.midframe is not None else None

        test_data = pet_frame
        test_timing = timing_frame if self.midframe is not None else None

        return (train_data, train_timings), (test_data, test_timing)

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

        dmoving = nb.Nifti1Image(dframe, self.affine, None)

        # resample and update orientation at index
        self.dataobj[..., index] = np.asanyarray(
            xform.apply(dmoving, order=order).dataobj,
            dtype=self.dataobj.dtype,
        )

        # update transform
        if self.motion_affines is None:
            self.motion_affines = [None] * len(self)

        self.motion_affines[index] = xform

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

    def to_nifti(self, filename, *_):
        """Write a NIfTI 1.0 file to disk."""
        nii = nb.Nifti1Image(self.dataobj, self.affine, None)
        nii.header.set_xyzt_units("mm")
        nii.to_filename(filename)

    @classmethod
    def from_filename(cls, filename):
        """Read an HDF5 file from disk."""
        with h5py.File(filename, "r") as in_file:
            root = in_file["/0"]
            data = {k: np.asanyarray(v) for k, v in root.items() if not k.startswith("_")}
        return cls(**data)

    @classmethod
    def load(cls, filename, json_file, brainmask_file=None):
        """Load PET data."""
        filename = Path(filename)
        if filename.name.endswith(".h5"):
            return cls.from_filename(filename)

        img = nb.load(filename)
        retval = cls(
            dataobj=img.get_fdata(dtype="float32"),
            affine=img.affine,
        )

        # Load metadata
        with open(json_file, "r") as f:
            metadata = json.load(f)

        frame_duration = np.array(metadata["FrameDuration"])
        frame_times_start = np.array(metadata["FrameTimesStart"])
        midframe = frame_times_start + frame_duration / 2

        retval.midframe = midframe
        retval.total_duration = float(frame_times_start[-1] + frame_duration[-1])

        assert len(retval.midframe) == retval.dataobj.shape[-1]

        if brainmask_file:
            mask = nb.load(brainmask_file)
            retval.brainmask = np.asanyarray(mask.dataobj)

        return retval


def from_nii(
    filename: Path | str,
    brainmask_file: Path | str | None = None,
    motion_file: Path | str | None = None,
    frame_time: np.ndarray | list[float] | None = None,
    frame_duration: np.ndarray | list[float] | None = None,
) -> PET:
    """
    Load PET data from NIfTI, creating a PET object with appropriate metadata.

    Parameters
    ----------
    filename : :obj:`os.pathlike`
        The NIfTI file.
    brainmask_file : :obj:`os.pathlike`, optional
        A brainmask NIfTI file. If provided, will be loaded and
        stored in the returned dataset.
    motion_file : :obj:`os.pathlike`
        A file containing head motion affine matrices (linear).
    frame_time : :obj:`numpy.ndarray` or :obj:`list` of :obj:`float`, optional
        The start times of each frame relative to the beginning of the acquisition.
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
        If ``frame_time`` is not provided (BIDS requires it).

    """
    if frame_time is None:
        raise RuntimeError("frame_time must be provided")
    if motion_file:
        raise NotImplementedError

    filename = Path(filename)
    # Load from NIfTI
    img = load_api(filename, SpatialImage)
    data = img.get_fdata(dtype=np.float32)
    pet_obj = PET(
        dataobj=data,
        affine=img.affine,
    )

    # If the user supplied new values, set them
    if frame_time is not None:
        # Convert to a float32 numpy array and zero out the earliest time
        frame_time_arr = np.array(frame_time, dtype=np.float32)
        frame_time_arr -= frame_time_arr[0]
        pet_obj.midframe = frame_time_arr

    # If the user doesn't provide frame_duration, we derive it:
    if frame_duration is None:
        if pet_obj.midframe is not None:
            frame_time_arr = pet_obj.midframe
            # If shape is e.g. (N,), then we can do
            durations = np.diff(frame_time_arr)
            if len(durations) == (len(frame_time_arr) - 1):
                durations = np.append(durations, durations[-1])  # last frame same as second-last
    else:
        durations = np.array(frame_duration, dtype=np.float32)

    # Set total_duration and shift frame_time to the midpoint
    pet_obj.total_duration = float(frame_time_arr[-1] + durations[-1])
    pet_obj.midframe = frame_time_arr + 0.5 * durations

    # If a brain mask is provided, load and attach
    if brainmask_file is not None:
        mask_img = load_api(brainmask_file, SpatialImage)
        pet_obj.brainmask = np.asanyarray(mask_img.dataobj, dtype=bool)

    return pet_obj
