# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2022 The NiPreps Developers <nipreps@gmail.com>
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

from pathlib import Path
from typing import Any, Union

import attr
import h5py
import nibabel as nb
import numpy as np

from nifreeze.data.base import BaseDataset, _cmp, _data_repr


@attr.s(slots=True)
class PET(BaseDataset):
    """
    Data representation structure for PET data, inheriting from BaseDataset.

    In addition to the base attributes (e.g., dataobj, affine), this PET class stores:
    - frame_time: a 1D array specifying the midpoint timing of each frame.
    - total_duration: a float specifying the total acquisition duration.

    """

    frame_time: np.ndarray | None = attr.ib(
        default=None, repr=_data_repr, eq=attr.cmp_using(eq=_cmp)
    )
    """
    A 1D numpy array specifying the midpoint timing of each sample or frame.
    Typically shape (N,).
    """
    total_duration: float | None = attr.ib(default=None, repr=True)
    """
    A float representing the total duration of the entire PET acquisition.
    """

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
        volumes : np.ndarray
            The selected data subset. If `idx` is a single integer, this will have shape
            ``(X, Y, Z)``, otherwise it may have shape ``(X, Y, Z, k)``.
        motion_affine : np.ndarray or None
            The corresponding per-volume motion affine(s) or `None` if identity transform(s).
        time : float
            The corresponding frame time.

        """

        data, affine = super().__getitem__(idx)
        return data, affine, self.frame_time[idx]

    @classmethod
    def from_filename(cls, filename: Union[str, Path]) -> PET:
        """
        Read an HDF5 file from disk and create a PET object.

        Parameters
        ----------
        filename : str or Path
            The HDF5 file path to read.

        Returns
        -------
        PET
            A PET dataset with data loaded from the specified file.
        """
        import attr

        filename = Path(filename)
        data: dict[str, Any] = {}

        with h5py.File(filename, "r") as in_file:
            root = in_file["/0"]
            for f in attr.fields(cls):
                # skip private attributes (start with '_')
                if f.name.startswith("_"):
                    continue
                if f.name in root:
                    data[f.name] = np.asanyarray(root[f.name])
                else:
                    data[f.name] = None

        return cls(**data)

    def to_filename(
        self,
        filename: Path | str,
        compression: str | None = None,
        compression_opts: Any = None,
    ) -> None:
        """
        Write the PET dataset to an HDF5 file on disk.

        Parameters
        ----------
        filename : Path or str
            Path to the output HDF5 file.
        compression : str, optional
            Compression filter, e.g. 'gzip'. Default is None (no compression).
        compression_opts : Any, optional
            Compression level or other parameters for the HDF5 dataset.
        """
        super().to_filename(filename, compression=compression, compression_opts=compression_opts)
        # Overriding if you'd like to set a custom attribute, for example:
        with h5py.File(filename, "r+") as out_file:
            out_file.attrs["Type"] = "pet"


def load(
    filename: Path | str,
    brainmask_file: Path | str | None = None,
    frame_time: np.ndarray | list[float] | None = None,
    frame_duration: np.ndarray | list[float] | None = None,
) -> PET:
    """
    Load PET data from HDF5 or NIfTI, creating a PET object with appropriate metadata.

    Parameters
    ----------
    filename : Path or str
        Path to the PET data (HDF5 or NIfTI).
    brainmask_file : Path or str, optional
        An optional brain mask NIfTI file.
    frame_time : np.ndarray or list of float, optional
        The start times of each frame relative to the beginning of the acquisition.
        If None, an error is raised (since BIDS requires FrameTimesStart).
    frame_duration : np.ndarray or list of float, optional
        The duration of each frame. If None, it is derived by the difference
        of consecutive frame_times, defaulting the last frame to match the second-last.

    Returns
    -------
    PET
        A PET object storing the data, metadata, and any optional mask.

    Raises
    ------
    RuntimeError
        If `frame_time` is not provided (BIDS requires it).
    """
    filename = Path(filename)
    if filename.suffix == ".h5":
        # Load from HDF5
        pet_obj = PET.from_filename(filename)
    else:
        # Load from NIfTI
        img = nb.load(str(filename))
        data = img.get_fdata(dtype=np.float32)
        pet_obj = PET(
            dataobj=data,
            affine=img.affine,
        )

    # Verify the user provided frame_time if not already in the PET object
    if pet_obj.frame_time is None and frame_time is None:
        raise RuntimeError(
            "The `frame_time` is mandatory for PET data to comply with BIDS. "
            "See https://bids-specification.readthedocs.io for details."
        )

    # If the user supplied new values, set them
    if frame_time is not None:
        # Convert to a float32 numpy array and zero out the earliest time
        frame_time_arr = np.array(frame_time, dtype=np.float32)
        frame_time_arr -= frame_time_arr[0]
        pet_obj.frame_time = frame_time_arr

    # If the user doesn't provide frame_duration, we derive it:
    if frame_duration is None:
        frame_time_arr = pet_obj.frame_time
        # If shape is e.g. (N,), then we can do
        durations = np.diff(frame_time_arr)
        if len(durations) == (len(frame_time_arr) - 1):
            durations = np.append(durations, durations[-1])  # last frame same as second-last
    else:
        durations = np.array(frame_duration, dtype=np.float32)

    # Set total_duration and shift frame_time to the midpoint
    pet_obj.total_duration = float(frame_time_arr[-1] + durations[-1])
    pet_obj.frame_time = frame_time_arr + 0.5 * durations

    # If a brain mask is provided, load and attach
    if brainmask_file is not None:
        mask_img = nb.load(str(brainmask_file))
        pet_obj.brainmask = np.asanyarray(mask_img.dataobj, dtype=bool)

    return pet_obj
