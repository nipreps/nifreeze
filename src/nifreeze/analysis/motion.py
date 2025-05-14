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
"""Motion analysis."""

import numpy as np
from scipy.stats import zscore


def compute_percentage_change(
    reference: np.ndarray,
    test: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Compute motion change between reference and test as a percentage.

    If a mask is provided, the computation is only provided within the mask.
    Also, null values are ignored.

    Parameters
    ----------
    reference : :obj:`~numpy.ndarray`
        Reference imaging volume.
    test : :obj:`~numpy.ndarray`
        Test (shifted) imaging volume.
    mask : :obj:`~numpy.ndarray`
        Mask for value consideration.

    Returns
    -------
    rel_diff : :obj:`~numpy.ndarray`
        Motion change between reference and test.
    """

    # Avoid divide-by-zero errors
    eps = 1e-5
    rel_diff = np.zeros_like(reference)
    mask = mask.copy()
    mask[reference <= eps] = False
    rel_diff[mask] = 100 * (test[mask] - reference[mask]) / reference[mask]

    return rel_diff


def identify_spikes(fd: np.ndarray, threshold: float = 2.0):
    """Identify motion spikes in framewise displacement data.

    Identifies high-motion frames as timepoint exceeding a given threshold value
    based on z-score normalized framewise displacement (FD) values.

    Parameters
    ----------
    fd : :obj:`~numpy.ndarray`
        Framewise displacement data.
    threshold : :obj:`float`, optional
        Threshold value to determine motion spikes.

    Returns
    -------
    indices : :obj:`~numpy.ndarray`
        Indices of identified motion spikes.
    mask : :obj:`~numpy.ndarray`
        Mask of identified motion spikes.
    """

    # Normalize (z-score)
    fd_norm = zscore(fd)

    mask = fd_norm > threshold
    indices = np.where(mask)[0]

    return indices, mask
