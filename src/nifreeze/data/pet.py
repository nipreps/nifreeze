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
from collections.abc import Callable
from pathlib import Path
from typing import Any, Tuple

import attrs
import h5py
import nibabel as nb
import numpy as np
from nibabel.spatialimages import SpatialImage
from nitransforms.linear import Affine
from nitransforms.resampling import apply
from typing_extensions import Self

from nifreeze.data.base import BaseDataset, _cmp, _data_repr, _has_ndim
from nifreeze.utils.ndimage import get_data, load_api

ATTRIBUTE_ABSENCE_ERROR_MSG = "PET '{attribute}' may not be None"
"""PET initialization array attribute absence error message."""

ARRAY_ATTRIBUTE_OBJECT_ERROR_MSG = (
    "PET '{attribute}' must be a numeric homogeneous array-like object."
)
"""PET initialization array attribute object error message."""

ARRAY_ATTRIBUTE_NDIM_ERROR_MSG = "PET '{attribute}' must be a 1D numpy array."
"""PET initialization array attribute ndim error message."""

ATTRIBUTE_VOLUME_DIMENSIONALITY_MISMATCH_ERROR_MSG = """\
PET '{attribute}' length does not match number of frames: \
expected {n_frames} values, found {attr_len}."""
"""PET attribute shape mismatch error message."""

TEMPORAL_ATTRIBUTE_INCONSISTENCY_ERROR_MSG = """\
PET 'total_duration' cannot be smaller than last 'midframe' value: \
found {total_duration} and {last_midframe}."""
"""PET attribute inconsistency error message."""

SCALAR_ATTRIBUTE_ERROR_MSG = (
    "PET '{attribute}' must be a numeric or single-element sequence object."
)
"""PET initialization scalar attribute object error message."""

TEMPORAL_FILE_KEY_ERROR_MSG = "{key} key not found in temporal file"
"""PET temporal file key error message."""

FRAME_TIME_START_KEY = "FrameTimesStart"
"""PET frame time start key."""


def format_scalar_like(value: Any, attr: attrs.Attribute) -> float:
    """Convert ``value`` to a scalar.

    Accepts:
      - :obj:`float` or :obj:`int` (but rejects :obj:`bool`)
      - Numpy scalar (:obj:`~numpy.generic`, e.g. :obj:`~numpy.floating`, :obj:`~numpy.integer`)
      - :obj:`~numpy.ndarray` of size 1
      - :obj:`list`/:obj:`tuple` of length 1

    This function is intended for use as an attrs-style formatter.

    Parameters
    ----------
    value : :obj:`Any`
        The value to format.
    attr : :obj:`~attrs.Attribute`
        The attribute being initialized; ``attr.name`` is used in the error message.

    Returns
    -------
    formatted : :obj:`float`
        The formatted value.

    Raises
    ------
    exc:`TypeError`
        If the input cannot be converted to a scalar.
    exc:`ValueError`
        If the value is ``None``, is of type :obj:`bool` or has not size/length 1.
    """

    if value is None:
        raise ValueError(ATTRIBUTE_ABSENCE_ERROR_MSG.format(attribute=attr.name))

    # Reject bool explicitly (bool is subclass of int)
    if isinstance(value, bool):
        raise ValueError(SCALAR_ATTRIBUTE_ERROR_MSG.format(attribute=attr.name))

    # Numpy scalar (np.generic) or numpy 0-d array
    if np is not None and isinstance(value, np.generic):
        return float(value.item())

    # Numpy ndarray (ndarray)
    if np is not None and isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError(SCALAR_ATTRIBUTE_ERROR_MSG.format(attribute=attr.name))
        return float(value.ravel()[0])

    # List/tuple with single element
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(SCALAR_ATTRIBUTE_ERROR_MSG.format(attribute=attr.name))
        return float(value[0])

    # Plain int/float (but not bool)
    if isinstance(value, (int, float)):
        return float(value)

    # Fallback: try to use .item() if present
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return float(item())
        except Exception:
            pass

    raise TypeError(f"Cannot convert {type(value)!r} to float")


def format_array_like(value: Any, attr: attrs.Attribute) -> np.ndarray:
    """Convert ``value`` to a :obj:`~numpy.ndarray`.

    This function is intended for use as an attrs-style formatter.

    Parameters
    ----------
    value : :obj:`Any`
        The value to format.
    attr : :obj:`~attrs.Attribute`
        The attribute being initialized; ``attr.name`` is used in the error message.

    Returns
    -------
    formatted : :obj:`~numpy.ndarray`
        The formatted value.

    Raises
    ------
    exc:`TypeError`
        If the input cannot be converted to a float :obj:`~numpy.ndarray`.
    exc:`ValueError`
        If the value is ``None``.
    """

    if value is None:
        raise ValueError(ATTRIBUTE_ABSENCE_ERROR_MSG.format(attribute=attr.name))

    try:
        formatted = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        # Conversion failed (e.g. nested ragged objects, non-numeric)
        raise TypeError(ARRAY_ATTRIBUTE_OBJECT_ERROR_MSG.format(attribute=attr.name)) from exc

    return formatted


def validate_1d_array(inst: PET, attr: attrs.Attribute, value: Any) -> None:
    """Strict validator to ensure an attribute is a 1D NumPy array.

    Enforces that ``value`` has exactly one dimension (``value.ndim == 1``).

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
    exc:`ValueError`
        If the value is  not 1D.
    """

    if not _has_ndim(value, 1):
        raise ValueError(ARRAY_ATTRIBUTE_NDIM_ERROR_MSG.format(attribute=attr.name))


@attrs.define(slots=True)
class PET(BaseDataset[np.ndarray]):
    """Data representation structure for PET data."""

    midframe: np.ndarray = attrs.field(
        default=None,
        repr=_data_repr,
        eq=attrs.cmp_using(eq=_cmp),
        converter=attrs.Converter(format_array_like, takes_field=True),  # type: ignore
        validator=validate_1d_array,
    )
    """A (N,) numpy array specifying the midpoint timing of each sample or frame."""
    total_duration: float = attrs.field(
        default=None,
        repr=True,
        converter=attrs.Converter(format_scalar_like, takes_field=True),  # type: ignore
        validator=attrs.validators.optional(attrs.validators.instance_of(float)),
    )
    """A float representing the total duration of the dataset."""

    def __attrs_post_init__(self) -> None:
        """Enforce presence and basic consistency of PET data fields at
        instantiation time.

        Specifically, the length of the frame_time and uptake attributes must
        match the last dimension of the data (number of frames).

        Computes the values for the private attributes.
        """

        def _check_attr_vol_length_match(
            _attr_name: str, _value: np.ndarray | None, _n_frames: int
        ) -> None:
            if _value is not None and len(_value) != _n_frames:
                raise ValueError(
                    ATTRIBUTE_VOLUME_DIMENSIONALITY_MISMATCH_ERROR_MSG.format(
                        attribute=_attr_name,
                        n_frames=_n_frames,
                        attr_len=len(_value),
                    )
                )

        n_frames = int(self.dataobj.shape[-1])
        _check_attr_vol_length_match("midframe", self.midframe, n_frames)

        # Ensure that the total duration is larger than last midframe
        if self.total_duration <= self.midframe[-1]:
            raise ValueError(
                TEMPORAL_ATTRIBUTE_INCONSISTENCY_ERROR_MSG.format(
                    total_duration=self.total_duration,
                    last_midframe=self.midframe[-1],
                )
            )

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
        motion_affine : :obj:`~numpy.ndarray` or :obj:`None`
            The corresponding per-volume motion affine(s) or :obj:`None` if identity transform(s).
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


def from_nii(
    filename: Path | str,
    temporal_file: Path | str,
    brainmask_file: Path | str | None = None,
) -> PET:
    """
    Load PET data from NIfTI, creating a PET object with appropriate metadata.

    Parameters
    ----------
    filename : :obj:`os.pathlike`
        The NIfTI file.
    temporal_file : :obj:`os.pathlike`
        A JSON file containing temporal data. It must at least contain
        ``frame_time`` data.
    brainmask_file : :obj:`os.pathlike`, optional
        A brainmask NIfTI file. If provided, will be loaded and
        stored in the returned dataset.

    Returns
    -------
    :obj:`~nifreeze.data.pet.PET`
        A PET object storing the data, metadata, and any optional mask.

    Raises
    ------
    :exc:`RuntimeError`
        If ``frame_time`` is not provided (BIDS requires it).

    """

    filename = Path(filename)

    # 1) Load a NIfTI
    img = load_api(filename, SpatialImage)
    fulldata = get_data(img)

    # 2) Load the temporal data
    with open(temporal_file, "r") as f:
        temporal_attrs = json.load(f)

    frame_time = temporal_attrs.get(FRAME_TIME_START_KEY, None)
    if frame_time is None:
        raise RuntimeError(TEMPORAL_FILE_KEY_ERROR_MSG.format(key=FRAME_TIME_START_KEY))

    # 3) If a brainmask_file was provided, load it
    brainmask_data = None
    if brainmask_file is not None:
        mask_img = load_api(brainmask_file, SpatialImage)
        brainmask_data = np.asanyarray(mask_img.dataobj, dtype=bool)

    # 4) Compute temporal attributes
    midframe, total_duration = _compute_temporal_markers(np.asarray(frame_time))

    # 5) Create and return the PET instance
    return PET(
        dataobj=fulldata,
        affine=img.affine,
        brainmask=brainmask_data,
        midframe=midframe,
        total_duration=total_duration,
    )


def _compute_temporal_markers(frame_time: np.ndarray) -> Tuple[np.ndarray, float]:
    """Compute the frame temporal markers from the frame time values.

    Computes the midframe times and the total duration following the principles
    detailed below.

    Let :math:`K` be the number of frames and :math:`t_{k}` be the :math:`k`-th
    (start) frame time. For each frame :math:`k`, the frame duration
    :math:`d_{k}` is defined as the difference between consecutive frame times:

    .. math::
       d_{k} = t_{k+1} - t_{k}

    If necessary, the last frame duration is set to the value of the second to
    last frame to match the appropriate dimensionality in this implementation.

    Per-frame midpoints :math:`m_{k}` are computed as:

    .. math::
       m_{k} = t_{k} + \\frac{d_k}{2}

    The total duration :math:`D` of the acquisition is a scalar computed as the
    sum of the frame durations:

    .. math::
       D = \\sum_{k=1}^{K} d_{k}

    or, equivalently, the difference between the last frame start and its
    duration once the frame times have been time-origin shifted:

    .. math::
       D = t_{K} - d_{K}

    Frame times are time-origin shifted (i.e. the earliest time is zeroed out)
    if not already done at the beginning of the process for the sake of
    simplicity.

    Parameters
    ----------
    frame_time : :obj:`~numpy.ndarray`
        Frame time values.

    Returns
    -------
    :obj:`tuple`
        Midpoint timing of each frame and total duration
    """

    # Time-origin shift: zero out the earliest time if necessary
    # Flatten the array in case it is not a 1D array
    if not np.isclose(frame_time.ravel()[0], 0):
        frame_time -= frame_time.flat[0]

    # If shape is e.g. (N,), then we can do
    frame_duration = np.diff(frame_time)
    if len(frame_duration) == (len(frame_time) - 1):
        frame_duration = np.append(
            frame_duration, frame_duration[-1]
        )  # last frame same as second-last

    midframe = frame_time + frame_duration / 2
    total_duration = float(frame_time[-1] + frame_duration[-1])

    return midframe, total_duration


def compute_uptake_statistic(data: np.ndarray, stat_func: Callable[..., np.ndarray] = np.sum):
    """Compute a statistic over all voxels for each frame on a PET sequence.

    Assumes the last dimension corresponds to the number of frames in the
    sequence.

    Parameters
    ----------
    data : :obj:`~numpy.ndarray`
        PET data.
    stat_func : :obj:`~collections.abc.Callable`, optional
        Function to apply over voxels (e.g., :func:`numpy.sum`,
        :func:`numpy.mean`, :func:`numpy.std`)

    Returns
    -------
    :obj:`~numpy.ndarray`
        1D array of statistic values for each frame.
    """

    return stat_func(data.reshape(-1, data.shape[-1]), axis=0)
