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
"""Filtering data."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter
from skimage.morphology import ball

from nifreeze.data.dmri import DEFAULT_CLIP_PERCENTILE

DEFAULT_DTYPE = "int16"
"""The default image's data type."""
BVAL_ATOL = 100.0
"""b-value tolerance value."""


def advanced_clip(
    data: np.ndarray,
    p_min: float = 35.0,
    p_max: float = 99.98,
    nonnegative: bool = True,
    dtype: str | np.dtype = DEFAULT_DTYPE,
    invert: bool = False,
    inplace: bool = False,
) -> np.ndarray | None:
    """
    Clips outliers from an n-dimensional array and scales/casts to a specified data type.

    This function removes outliers from both ends of the intensity distribution
    in an n-dimensional array using percentiles. It optionally enforces non-negative
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
    inplace : :obj:`bool`, optional
        If ``True``, the normalization is performed on the original data.

    Returns
    -------
    :obj:`~numpy.ndarray` or None
        The clipped and scaled data array with the specified data type or
        ``None`` if ``inplace`` is ``True``.

    """

    if not inplace:
        data = data.copy()

    # Cast to float32 before modifying in-place: clipping a float array and
    # writing the result in-place to an integer array is disallowed by NumPy due
    # to risk of data loss
    if not inplace or not np.issubdtype(data.dtype, np.floating):
        data = data.astype(np.float32, copy=not inplace)

    # Calculate stats on denoised version to avoid outlier bias
    denoised = median_filter(data, footprint=ball(3))

    a_min = np.percentile(
        np.asarray([denoised[denoised >= 0] if nonnegative else denoised]), p_min
    )
    a_max = np.percentile(
        np.asarray([denoised[denoised >= 0] if nonnegative else denoised]), p_max
    )

    # Clip and scale data
    np.clip(data, a_min=a_min, a_max=a_max, out=data)
    data -= data.min()
    data /= data.max()

    if invert:
        np.subtract(1.0, data, out=data)

    if dtype in ("uint8", "int16"):
        np.round(255 * data, out=data).astype(dtype)

    if inplace:
        return None

    return data


def robust_minmax_normalization(
    data: np.ndarray,
    mask: np.ndarray | None = None,
    p_min: float = 5.0,
    p_max: float = 95.0,
    inplace: bool = False,
) -> np.ndarray | None:
    r"""Normalize min-max percentiles of each volume to the grand min-max
    percentiles.

    Robust min/max normalization of the volumes in the dataset following:

    .. math::
        \text{data}_{\text{normalized}} = \frac{(\text{data} - p_{min}) \cdot p_{\text{mean}}}{p_{\text{range}}} + p_{min}^{\text{mean}}

    where

    .. math::
        p_{\text{range}} = p_{max} - p_{min}, \quad p_{\text{mean}} = \frac{1}{N} \sum_{i=1}^N p_{\text{range}_i}, \quad p_{min}^{\text{mean}} = \frac{1}{N} \sum_{i=1}^N p_{5_i}

    If a mask is provided, only the data within the mask are considered.

    Parameters
    ----------
    data : :obj:`~numpy.ndarray`
        Data to be normalized.
    mask : :obj:`~numpy.ndarray`, optional
        Mask. If provided, only the data within the mask are considered.
    p_min : :obj:`float`, optional
        The lower percentile value for normalization.
    p_max : :obj:`float`, optional
        The upper percentile value for normalization.
    inplace : :obj:`bool`, optional
        If ``False``, the normalization is performed on the original data.

    Returns
    -------
    data : :obj:`~numpy.ndarray` or None
        Normalized data or ``None`` if ``inplace`` is ``True``.
    """

    normalized = data if inplace else data.copy()

    mask = mask if mask is not None else np.ones(data.shape[-1], dtype=bool)
    volumes = data[..., mask]
    reshape_shape = (-1, volumes.shape[-1]) if mask is None else (-1, sum(mask))
    reshaped_data = volumes.reshape(reshape_shape)
    p5 = np.percentile(reshaped_data, p_min, axis=0)
    p95 = np.percentile(reshaped_data, p_max, axis=0) - p5
    normalized[..., mask] = (volumes - p5) * p95.mean() / p95 + p5.mean()

    if inplace:
        return None

    return normalized


def grand_mean_normalization(
    data: np.ndarray,
    mask: np.ndarray | None = None,
    center: float = DEFAULT_CLIP_PERCENTILE,
    inplace: bool = False,
) -> np.ndarray | None:
    """Robust grand mean normalization.

    Regresses out global signal differences so that data are normalized and
    centered around a given value.

    If a mask is provided, only the data within the mask are considered.

    Parameters
    ----------
    data : :obj:`~numpy.ndarray`
        Data to be normalized.
    mask : :obj:`~numpy.ndarray`, optional
        Mask. If provided, only the data within the mask are considered.
    center : float, optional
        Central value around which to normalize the data.
    inplace : :obj:`bool`, optional
        If ``False``, the normalization is performed on the original data.

    Returns
    -------
    data : :obj:`~numpy.ndarray` or None
        Normalized data or ``None`` if ``inplace`` is ``True``.
    """

    normalized = data if inplace else data.copy()

    mask = mask if mask is not None else np.ones(data.shape[-1], dtype=bool)
    volumes = data[..., mask]

    centers = np.median(volumes, axis=(0, 1, 2))
    reference = np.percentile(centers[centers >= 1.0], center)
    centers[centers < 1.0] = reference
    drift = reference / centers
    normalized[..., mask] = volumes * drift

    if inplace:
        return None

    return normalized


def dwi_select_shells(
    gradients: np.ndarray,
    index: int,
    atol_low: float | None = None,
    atol_high: float | None = None,
) -> np.ndarray:
    """Select DWI shells around the given index and lower and upper b-value
    bounds.

    Computes a boolean mask of the DWI shells around the given index with the
    provided lower and upper bound b-values.

    If ``atol_low`` and ``atol_high`` are both ``None``, the returned shell mask
    corresponds to the lengths of the diffusion-sensitizing gradients.

    Parameters
    ----------
    gradients : :obj:`~numpy.ndarray`
        Gradients.
    index : :obj:`int`
        Index of the shell data.
    atol_low : :obj:`float`, optional
        A lower bound for the b-value.
    atol_high : :obj:`float`, optional
        An upper bound for the b-value.

    Returns
    -------
    shellmask : :obj:`~numpy.ndarray`
        Shell mask.
    """

    bvalues = gradients[:, -1]
    bcenter = bvalues[index]

    shellmask = np.ones(len(bvalues), dtype=bool)
    shellmask[index] = False  # Drop the held-out index

    if atol_low is None and atol_high is None:
        return shellmask

    atol_low = 0 if atol_low is None else atol_low
    atol_high = gradients[:, -1].max() if atol_high is None else atol_high

    # Keep only b-values within the range defined by atol_high and atol_low
    shellmask[bvalues > (bcenter + atol_high)] = False
    shellmask[bvalues < (bcenter - atol_low)] = False

    if not shellmask.sum():
        raise RuntimeError(f"Shell corresponding to index {index} (b={bcenter}) is empty.")

    return shellmask
