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
"""Input/Output utilities for PET objects."""

import json
from pathlib import Path

import numpy as np
from nibabel.spatialimages import SpatialImage

from nifreeze.data.pet.base import PET
from nifreeze.data.pet.utils import compute_temporal_markers
from nifreeze.utils.ndimage import get_data, load_api

TEMPORAL_FILE_KEY_ERROR_MSG = "{key} key not found in temporal file"
"""PET temporal file key error message."""

FRAME_TIME_START_KEY = "FrameTimesStart"
"""PET frame time start key."""


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
        ``FrameTimesStart`` data.
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
        If ``FrameTimesStart`` is not provided (BIDS requires it).

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
    midframe, total_duration = compute_temporal_markers(np.asarray(frame_time))

    # 5) Create and return the PET instance
    return PET(
        dataobj=fulldata,
        affine=img.affine,
        brainmask=brainmask_data,
        midframe=midframe,
        total_duration=total_duration,
    )
