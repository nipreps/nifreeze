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
from pathlib import Path
from typing import Any

import attrs
import h5py
import nibabel as nb
import numpy as np
from nitransforms.linear import Affine
from nitransforms.resampling import apply
from typing_extensions import Self

from nifreeze.data.base import BaseDataset, _cmp, _data_repr, _has_ndim

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
