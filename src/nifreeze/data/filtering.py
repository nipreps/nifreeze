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
"""Filtering data."""

from __future__ import annotations

import copy

import numpy as np
from scipy.ndimage import median_filter
from skimage.morphology import ball

from nifreeze.data.dmri import DEFAULT_CLIP_PERCENTILE, DWI

DEFAULT_DTYPE = "int16"
"""The default image's data type."""


def advanced_clip(
    data: np.ndarray,
    p_min: float = 35,
    p_max: float = 99.98,
    nonnegative: bool = True,
    dtype: str | np.dtype = DEFAULT_DTYPE,
    invert: bool = False,
) -> np.ndarray:
    """
    Clips outliers from a n-dimensional array and scales/casts to a specified data type.

    This function removes outliers from both ends of the intensity distribution
    in a n-dimensional array using percentiles. It optionally enforces non-negative
    values and scales the data to fit within a specified data type (e.g., uint8
    for image registration). To remove outliers more robustly, the function
    first applies a median filter to the data before calculating clipping thresholds.

    Parameters
    ----------
    data : :obj:`~numpy.ndarray`
        The input n-dimensional data array.
    p_min : :obj:`float`, optional
        The lower percentile threshold for clipping. Values below this percentile
        are set to the threshold value.
    p_max : :obj:`float`, optional
        The upper percentile threshold for clipping. Values above this percentile
        are set to the threshold value.
    nonnegative : :obj:`bool`, optional
        If True, only consider non-negative values when calculating thresholds.
    dtype : :obj:`str` or :obj:`~numpy.dtype`, optional
        The desired data type for the output array. Supported types are "uint8"
        and "int16".
    invert : :obj:`bool`, optional
        If ``True``, inverts the intensity values after scaling (1.0 - ``data``).

    Returns
    -------
    :obj:`~numpy.ndarray`
        The clipped and scaled data array with the specified data type.

    """

    # Calculate stats on denoised version to avoid outlier bias
    denoised = median_filter(data, footprint=ball(3))

    a_min = np.percentile(
        np.asarray([denoised[denoised >= 0] if nonnegative else denoised]), p_min
    )
    a_max = np.percentile(
        np.asarray([denoised[denoised >= 0] if nonnegative else denoised]), p_max
    )

    # Clip and scale data
    data = np.clip(data, a_min=a_min, a_max=a_max)
    data -= data.min()
    data /= data.max()

    if invert:
        data = 1.0 - data

    if dtype in ("uint8", "int16"):
        data = np.round(255 * data).astype(dtype)

    return data


def detrend_data_percentile(data: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    r"""Detrend data.

    Regresses out global signal differences so that its values are centered around the middle 90%
    of the data following:

    .. math::
        \text{data}_{\text{detrended}} = \frac{(\text{data} - p_{5}) \cdot p_{\text{mean}}}{p_{\text{range}}} + p_{5}^{\text{mean}}

    where

    .. math::
        p_{\text{range}} = p_{95} - p_{5}, \quad p_{\text{mean}} = \frac{1}{N} \sum_{i=1}^N p_{\text{range}_i}, \quad p_{5}^{\text{mean}} = \frac{1}{N} \sum_{i=1}^N p_{5_i}

    :math:`p_{5}` and :math:`p_{95}` being the 5th percentile and the 95th percentile of the data,
    respectively.

    If a mask is provided, only the data within the mask are considered.

    Parameters
    ----------
    data : :obj:`~numpy.ndarray`
        Data to be detrended.
    mask : :obj:`~numpy.ndarray`, optional
        Mask. If provided, only the data within the mask are considered.

    Returns
    -------
    :obj:`~numpy.ndarray`
        Detrended data.
    """

    data = data.copy().astype("float32")
    reshaped_data = data.reshape((-1, data.shape[-1])) if mask is None else data[mask]
    p5 = np.percentile(reshaped_data, 5.0, axis=0)
    p95 = np.percentile(reshaped_data, 95.0, axis=0) - p5
    return (data - p5) * p95.mean() / p95 + p5.mean()


def detrend_dwi_median(data: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Detrend DWI data.

    Regresses out global DWI signal differences so that its standardized and centered around the
    :data:`src.nifreeze.model.base.DEFAULT_CLIP_PERCENTILE` percentile.

    If a mask is provided, only the data within the mask are considered.

    Parameters
    ----------
    data : :obj:`~numpy.ndarray`
        Data to be detrended.
    mask : :obj:`~numpy.ndarray`, optional
        Mask. If provided, only the data within the mask are considered.

    Returns
    -------
    :obj:`~numpy.ndarray`
        Detrended data.
    """

    shelldata = data[..., mask]

    centers = np.median(shelldata, axis=(0, 1, 2))
    reference = np.percentile(centers[centers >= 1.0], DEFAULT_CLIP_PERCENTILE)
    centers[centers < 1.0] = reference
    drift = reference / centers
    return shelldata * drift


def clip_dwi_shell_data(dataset: DWI, index: int, th_low: int = 100, th_high: int = 100) -> DWI:
    """Clip DWI shell data around the given index and lower and upper b-value bounds.

    Clip DWI data around the given index with the provided lower and upper bound b-values.

    Parameters
    ----------
    dataset : :obj:`~nifreeze.data.dmri.DWI`
        Reference to a DWI object.
    index  : :obj:`int`
        Index of the shell data.
    th_low : :obj:`numbers.Number`, optional
        A lower bound for the b-value.
    th_high : :obj:`numbers.Number`, optional
        An upper bound for the b-value.

    Returns
    -------
    clipped_dataset : :obj:`~nifreeze.data.dmri.DWI`
        Clipped dataset.
    """

    clipped_dataset = copy.deepcopy(dataset)

    bvalues = clipped_dataset.gradients[:, -1]
    bcenter = bvalues[index]

    shellmask = np.ones(len(clipped_dataset._dataset), dtype=bool)

    # Keep only bvalues within the range defined by th_high and th_low
    shellmask[index] = False
    shellmask[bvalues > (bcenter + th_high)] = False
    shellmask[bvalues < (bcenter - th_low)] = False

    if not shellmask.sum():
        raise RuntimeError(f"Shell corresponding to index {index} (b={bcenter}) is empty.")

    clipped_dataset._dataset = clipped_dataset._dataset.dataobj[..., shellmask]

    return clipped_dataset
