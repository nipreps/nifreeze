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
from typing import Any, Callable

import attrs
import h5py
import nibabel as nb
import numpy as np
from nibabel.spatialimages import SpatialImage
from nitransforms.linear import Affine
from nitransforms.resampling import apply
from typing_extensions import Self

from nifreeze.data.base import BaseDataset, _cmp, _data_repr, _has_ndim
from nifreeze.utils.ndimage import load_api

ARRAY_ATTRIBUTE_ABSENCE_ERROR_MSG = "PET '{attribute}' may not be None"
"""PET initialization array attribute absence error message."""

ARRAY_ATTRIBUTE_OBJECT_ERROR_MSG = "PET '{attribute}' must be a numpy array."
"""PET initialization array attribute object error message."""

ARRAY_ATTRIBUTE_NDIM_ERROR_MSG = "PET '{attribute}' must be a 1D numpy array."
"""PET initialization array attribute ndim error message."""

ATTRIBUTE_VOLUME_DIMENSIONALITY_MISMATCH_ERROR = """\
PET '{attribute}' length does not match number of frames: \
expected {n_frames} values, found {attr_len}."""
"""PET attribute shape mismatch error message."""


def validate_1d_array(inst: PET, attr: attrs.Attribute, value: Any) -> None:
    """Strict validator to ensure an attribute is a 1D NumPy array.

    Enforces that ``value`` is a :obj:`~numpy.ndarray` and that it has exactly
    one dimension (``value.ndim == 1``).

    This function is intended for use as an attrs-style validator.

    Parameters
    ----------
    inst : :obj:`~nifreeze.data.pet.PET`
        The instance being validated (unused; present for validator signature).
    attr : :obj:`~attrs.Attribute`
        The attribute being validated; ``attr.name`` is used in the error message.
    value : :obj:`Any`
        The value to validate.

    Raises
    ------
    exc:`TypeError`
        If the input cannot be converted to a float :obj:`~numpy.ndarray`.
    exc:`ValueError`
        If the value is ``None``, or not 1D.
    """

    if value is None:
        raise ValueError(ARRAY_ATTRIBUTE_ABSENCE_ERROR_MSG.format(attribute=attr.name))

    if not isinstance(value, np.ndarray):
        raise TypeError(ARRAY_ATTRIBUTE_OBJECT_ERROR_MSG.format(attribute=attr.name))

    if not _has_ndim(value, 1):
        raise ValueError(ARRAY_ATTRIBUTE_NDIM_ERROR_MSG.format(attribute=attr.name))


@attrs.define(slots=True)
class PET(BaseDataset[np.ndarray]):
    """Data representation structure for PET data.

    If not provided, frame duration data are computed as differences between
    consecutive midframe times. The last interval is duplicated.
    """

    frame_time: np.ndarray = attrs.field(
        default=None, repr=_data_repr, eq=attrs.cmp_using(eq=_cmp), validator=validate_1d_array
    )
    """A (N,) numpy array specifying the timing of each sample or frame."""
    uptake: np.ndarray = attrs.field(
        default=None, repr=_data_repr, eq=attrs.cmp_using(eq=_cmp), validator=validate_1d_array
    )
    """A (N,) numpy array specifying the uptake value of each sample or frame."""
    frame_duration: np.ndarray | None = attrs.field(
        default=None, repr=_data_repr, eq=attrs.cmp_using(eq=_cmp)
    )
    """A (N,) numpy array specifying the frame duration."""
    midframe: np.ndarray = attrs.field(
        default=None, repr=_data_repr, init=False, eq=attrs.cmp_using(eq=_cmp)
    )
    """A (N,) numpy array specifying the midpoint timing of each sample or frame."""
    total_duration: float = attrs.field(default=None, repr=True, init=False)
    """A float representing the total duration of the dataset."""

    def __attrs_post_init__(self) -> None:
        """Enforce presence and basic consistency of PET data fields at
        instantiation time.

        Specifically, the length of the frame_time and uptake attributes must
        match the last dimension of the data (number of frames).

        Computes the values for the private attributes.
        """
        n_frames = int(self.dataobj.shape[-1])

        if len(self.frame_time) != n_frames:
            raise ValueError(
                ATTRIBUTE_VOLUME_DIMENSIONALITY_MISMATCH_ERROR.format(
                    attribute=attrs.fields_dict(self.__class__)["frame_time"].name,
                    n_frames=n_frames,
                    attr_len=len(self.frame_time),
                )
            )

        if len(self.uptake) != n_frames:
            raise ValueError(
                ATTRIBUTE_VOLUME_DIMENSIONALITY_MISMATCH_ERROR.format(
                    attribute=attrs.fields_dict(self.__class__)["uptake"].name,
                    n_frames=n_frames,
                    attr_len=len(self.uptake),
                )
            )

        # Compute temporal attributes

        # Convert to a float32 numpy array and zero out the earliest time
        frame_time_arr = np.array(self.frame_time, dtype=np.float32)
        frame_time_arr -= frame_time_arr[0]
        self.midframe = frame_time_arr

        # If the user did not provide frame duration values,compute them
        if self.frame_duration:
            durations = np.array(self.frame_duration, dtype=np.float32)
        else:
            durations = _compute_frame_duration(self.midframe)

        # Compute total duration and shift midframe to the midpoint
        self.total_duration = float(self.midframe[-1] + durations[-1])
        self.midframe = self.midframe + 0.5 * durations

    def _getextra(self, idx: int | slice | tuple | np.ndarray) -> tuple[np.ndarray]:
        return (self.midframe[idx],)

    # For the sake of the docstring
    def __getitem__(
        self, idx: int | slice | tuple | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
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
        time : :obj:`~numpy.ndarray`
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
            timing_frame = np.asanyarray(root["midframe"][..., index])

        # Mask to exclude the selected frame
        mask = np.ones(self.dataobj.shape[-1], dtype=bool)
        mask[index] = False

        train_data = self.dataobj[..., mask]
        train_timings = self.midframe[mask]

        test_data = pet_frame
        test_timing = timing_frame

        return (train_data, train_timings), (test_data, test_timing)

    def set_transform(self, index: int, affine: np.ndarray, order: int = 3) -> None:
        """Set an affine, and update data object and gradients."""
        ImageGrid = namedtuple("ImageGrid", ("shape", "affine"))
        reference = ImageGrid(self.dataobj.shape[:3], self.affine)
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
            apply(xform, dmoving, order=order).dataobj,
            dtype=self.dataobj.dtype,
        )

        # update transform
        if self.motion_affines is None:
            self.motion_affines = np.asarray([None] * len(self))

        self.motion_affines[index] = xform

    def to_filename(
        self, filename: Path | str, compression: str | None = None, compression_opts: Any = None
    ) -> None:
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

    @classmethod
    def from_filename(cls, filename: Path | str) -> Self:
        """Read an HDF5 file from disk."""
        with h5py.File(filename, "r") as in_file:
            root = in_file["/0"]
            data = {k: np.asanyarray(v) for k, v in root.items() if not k.startswith("_")}
        return cls(**data)

    @classmethod
    def load(
        cls, filename: Path | str, json_file: Path | str, brainmask_file: Path | str | None = None
    ) -> Self:
        """Load PET data."""
        filename = Path(filename)
        if filename.name.endswith(".h5"):
            return cls.from_filename(filename)

        img = load_api(filename, SpatialImage)
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
            mask = load_api(brainmask_file, SpatialImage)
            retval.brainmask = np.asanyarray(mask.dataobj)

        return retval


def from_nii(
    filename: Path | str,
    frame_time: np.ndarray | list[float],
    brainmask_file: Path | str | None = None,
    motion_file: Path | str | None = None,
    frame_duration: np.ndarray | list[float] | None = None,
    uptake_stat_func: Callable[..., np.ndarray] = np.sum,
) -> PET:
    """
    Load PET data from NIfTI, creating a PET object with appropriate metadata.

    Parameters
    ----------
    filename : :obj:`os.pathlike`
        The NIfTI file.
    frame_time : :obj:`numpy.ndarray` or :obj:`list` of :obj:`float`
        The start times of each frame relative to the beginning of the acquisition.
    brainmask_file : :obj:`os.pathlike`, optional
        A brainmask NIfTI file. If provided, will be loaded and
        stored in the returned dataset.
    motion_file : :obj:`os.pathlike`, optional
        A file containing head motion affine matrices (linear).
    frame_duration : :obj:`numpy.ndarray` or :obj:`list` of :obj:`float`, optional
        The duration of each frame.
        If ``None``, it is derived by the difference of consecutive frame times,
        defaulting the last frame to match the second-last.
    uptake_stat_func : :obj:`Callable`, optional
        The statistic function to be used to compute the uptake value.

    Returns
    -------
    :obj:`~nifreeze.data.pet.PET`
        A PET object storing the data, metadata, and any optional mask.

    Raises
    ------
    :exc:`RuntimeError`
        If ``frame_time`` is not provided (BIDS requires it).

    """
    if motion_file:
        raise NotImplementedError

    filename = Path(filename)

    # 1) Load a NIfTI
    img = load_api(filename, SpatialImage)
    fulldata = img.get_fdata(dtype=np.float32)

    # 2) Determine uptake value
    uptake = _compute_uptake_statistic(fulldata, stat_func=uptake_stat_func)

    # 3) If a brainmask_file was provided, load it
    brainmask_data = None
    if brainmask_file is not None:
        mask_img = load_api(brainmask_file, SpatialImage)
        brainmask_data = np.asanyarray(mask_img.dataobj, dtype=bool)

    # 4) Create and return the PET instance
    return PET(
        dataobj=fulldata,
        affine=img.affine,
        brainmask=brainmask_data,
        frame_time=np.asarray(frame_time),
        frame_duration=np.asarray(frame_duration),
        uptake=uptake,
    )


def _compute_frame_duration(midframe: np.ndarray) -> np.ndarray:
    """Compute the frame duration from the midframe values.

    Parameters
    ----------
    midframe : :obj:`~numpy.ndarray`
        Midframe time values.

    Returns
    -------
    durations : :obj:`~numpy.ndarray`
        Frame duration.
    """

    # If shape is e.g. (N,), then we can do
    durations = np.diff(midframe)
    if len(durations) == (len(midframe) - 1):
        durations = np.append(durations, durations[-1])  # last frame same as second-last

    return durations


def _compute_uptake_statistic(data: np.ndarray, stat_func: Callable[..., np.ndarray] = np.sum):
    """Compute a statistic over all voxels for each frame on a PET sequence.

    Assumes the last dimension corresponds to the number of frames in the
    sequence.

    Parameters
    ----------
    data : :obj:`~numpy.ndarray`
        PET data.
    stat_func : :obj:`Callable`, optional
        Function to apply over voxels (e.g., :obj:`~numpy.sum`,
        :obj:`~numpy.mean`, :obj:`~numpy.np.std`)

    Returns
    -------
    :obj:`~numpy.ndarray`
        1D array of statistic values for each frame.
    """

    return stat_func(data.reshape(-1, data.shape[-1]), axis=0)
