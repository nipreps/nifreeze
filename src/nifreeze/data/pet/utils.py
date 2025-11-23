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
"""Utilities for handling PET temporal and activity attributes."""

from collections.abc import Callable
from typing import Tuple

import numpy as np


def compute_temporal_markers(frame_time: np.ndarray) -> Tuple[np.ndarray, float]:
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
