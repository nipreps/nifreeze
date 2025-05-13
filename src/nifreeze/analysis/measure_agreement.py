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
"""Measure agreement computation."""

import enum

import numpy as np
import scipy.stats as stats


class BASalientEntity(enum.Enum):
    RELIABILITY_INDICES = "reliability_indices"
    RELIABILITY_MASK = "reliability_mask"
    LEFT_INDICES = "left_indices"
    LEFT_MASK = "left_mask"
    RIGHT_INDICES = "right_indices"
    RIGHT_MASK = "right_mask"


def _check_ci(ci: float) -> None:
    """Check that the confidence interval size is in the [0, 1] range.

    Parameters
    ----------
    ci : :obj:`float`
        Confidence interval size.
    """

    if ci < 0 or ci > 1:
        raise ValueError("Confidence interval size must be between 0 and 1")


def _check_bland_altman_data(data1: np.ndarray, data2: np.ndarray) -> None:
    """Check that the data for the Bland-Altman agreement analysis are compliant.

    Checks that
        - The data are one-dimensional.
        - The data have the same dimensionality.
        - There is no missing values.

    Parameters
    ----------
    data1 : :obj:`numpy.ndarray`
        Data values.
    data2 : :obj:`numpy.ndarray`
        Data values.
    """

    if data1.ndim != 1 or data2.ndim != 1:
        raise ValueError("Data arrays must be 1D")
    if data1.size != data2.size:
        raise ValueError("Data arrays must have equal size")
    if np.isnan(data1).any() or np.isnan(data2).any():
        raise ValueError("Missing values are not supported")


def compute_z_score(ci: float) -> float:
    """Compute the critical z-score for being outside a confidence interval.

    Parameters
    ----------
    ci : :obj:`float`
        Confidence interval size. Must be in the [0, 1] range.

    Returns
    -------
    :obj:`float`
        Z-score value.
    """

    _check_ci(ci)

    # Compute the z-score for confidence interval (two-tailed)
    p = (1 - ci) / 2
    q = 1 - p
    return float(stats.norm.ppf(q))


def compute_bland_altman_features(
    data1: np.ndarray, data2: np.ndarray, ci: float
) -> tuple[np.ndarray, np.ndarray, float, float, float, float, float, float]:
    """Compute quantities of interest for the Bland-Altman plot.

    Parameters
    ----------
    data1 : :obj:`numpy.ndarray`
        Data values.
    data2 : :obj:`numpy.ndarray`
        Data values.
    ci : :obj:`float`
        Confidence interval size. Must be in the [0, 1] range.

    Returns
    -------
    diff : :obj:`numpy.ndarray`
        Differences.
    mean : :obj:`numpy.ndarray`
        Mean values (across both data arrays).
    mean_diff : :obj:`float`
        Mean differences.
    std_diff : :obj:`float`
        Standard deviation of differences.
    loa_lower : :obj:`float`
        Lower limit of agreement.
    loa_upper : :obj:`float`
        Upper limit of agreement.
    ci_mean : :obj:`float`
        Confidence interval of mean values.
    ci_loa : :obj:`float`
        Confidence interval of limits of agreement.
    """

    _check_bland_altman_data(data1, data2)
    _check_ci(ci)

    axis = 0
    # Compute mean
    mean = np.mean([data1, data2], axis=axis)

    # Compute differences, mean difference, and std dev of differences
    diff = data1 - data2
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, axis=axis)
    # std_diff = np.std(diff, ddof=1)  # Use Bessel's correction

    # Compute confidence interval limits of agreement (LoA)
    z_score = compute_z_score(ci)
    loa_lower = mean_diff - z_score * std_diff
    loa_upper = mean_diff + z_score * std_diff

    n = len(diff)

    # Compute the t-distribution critical value for the confidence intervals
    # (CIs)
    t_val = stats.t.ppf((1 + ci) / 2, n - 1)

    # Compute confidence intervals

    # Confidence interval for the mean difference
    std_err_mean = std_diff / np.sqrt(n)
    ci_mean = t_val * std_err_mean

    # Confidence interval for the LoA
    # Follows Bland-Altman 1999 and Altman 1983, where the standard error of LoA
    # SE_{LoA} = \sigma_{d} \sqrt{2/n}
    # where \sigma_{d} is the standard deviation of the differences and n is the
    # sample size.
    # The confidence interval for the LoA is then calculated using the
    # t-distribution critical value
    std_err_loa = std_diff * np.sqrt(2 / n)
    ci_loa = t_val * std_err_loa

    return diff, mean, mean_diff, std_diff, loa_lower, loa_upper, ci_mean, ci_loa


def get_reliability_mask(diff: np.ndarray, loa_lower: float, loa_upper: float) -> np.ndarray:
    """Get reliability mask as the data within the lower and upper limits of
    agreement.

    Boundaries are inclusive.

    Parameters
    ----------
    diff : :obj:`numpy.ndarray`
        Differences data.
    loa_lower : :obj:`float`
        Lower limit of agreement.
    loa_upper : :obj:`float`
        Upper limit of agreement.

    Returns
    -------
    :obj:`numpy.ndarray`
        Reliability mask.
    """

    return (diff >= loa_lower) & (diff <= loa_upper)


def identify_bland_altman_salient_data(
    data1: np.ndarray, data2: np.ndarray, ci: float, top_n: int, percentile: float = 0.75
) -> dict:
    """Identify the Bland-Altman (BA) plot salient data.

    Given the Bland-Altman data arrays, identifies the left- and right-most
    `top_n` data points from the BA plot.

    Once the left-most data points identified, the right-most `percentile` data
    points are considered from the remaining data points.
    The ``top_n`` data points closest to the zero mean difference are
    identified among these.

    Parameters
    ----------
    data1 : :obj:`numpy.ndarray`
        Data array 1.
    data2 : :obj:`numpy.ndarray`
        Data array 2.
    ci : :obj:`float`
        Confidence interval.
    top_n : :obj:`float`
        Number of top-N salient data points to identify.
    percentile: :obj:`float`, optional
        Percentile of right-most salient data points to identify.

    Returns
    -------
    :obj:`dict`
        Reliability, left- and right-most data point indices, and corresponding
        data masks as specified by the `:obj:`~nifreeze.analysis.measure_agreement.BASalientEntity`
        keys.
    """

    (
        diff,
        mean,
        mean_diff,
        std_diff,
        loa_lower,
        loa_upper,
        _,
        _,
    ) = compute_bland_altman_features(data1, data2, ci)

    # Filter data outside the confidence intervals
    reliability_mask = get_reliability_mask(diff, loa_lower, loa_upper)
    reliability_idx = np.where(reliability_mask)[0]

    # Check that there are enough data points left to identify the requested
    # number of salient data points
    reliability_point_count = len(reliability_idx)
    salient_point_count = 2 * top_n
    if reliability_point_count < salient_point_count:
        raise ValueError(
            f"Too few reliable data points ({reliability_point_count}) to "
            f"identify the requested Bland-Altman salient points "
            f"(2 * {top_n}). Reduce the number of salient data points "
            f"requested ({top_n})"
        )

    # Select the top_n lowest median values from the left side of the BA plot
    lower_idx = np.argsort(mean[reliability_idx])[:top_n]
    left_indices = reliability_idx[lower_idx]
    left_mask = np.zeros_like(reliability_mask, dtype=bool)
    left_mask[left_indices] = True

    # Select the top_n highest median values from the right side of the BA plot
    # which are also closest to the zero mean difference value
    remaining_idx = np.setdiff1d(reliability_idx, left_indices)

    # Sort indices by descending mean (rightmost values first)
    right_sort_mean = remaining_idx[np.argsort(mean[remaining_idx])[::-1]]

    # Take a percentile of the rightmost points
    top_p_count = int(percentile * len(right_sort_mean))
    top_p_sorted = right_sort_mean[:top_p_count]

    # Check that there are enough data points left to identify the requested
    # number of rightmost points
    if top_p_count < top_n:
        raise ValueError(
            f"Too few data points ({top_p_count}) to identify the requested "
            f"Bland-Altman right-most salient points ({top_n}). Increase the "
            f"percentile requested ({top_n})"
        )

    # Get absolute difference from mean_diff (closeness to zero mean difference)
    diff_distance = np.abs(diff[top_p_sorted] - mean_diff)

    # Sort rightmost points by closeness to zero diff
    top_p_idx = np.argsort(diff_distance)

    # Take top_n of them
    upper_idx = top_p_sorted[top_p_idx][:top_n]
    right_mask = np.zeros_like(reliability_mask, dtype=bool)
    right_mask[upper_idx] = True

    return {
        BASalientEntity.RELIABILITY_INDICES.value: reliability_idx,
        BASalientEntity.RELIABILITY_MASK.value: reliability_mask,
        BASalientEntity.LEFT_INDICES.value: lower_idx,
        BASalientEntity.LEFT_MASK.value: left_mask,
        BASalientEntity.RIGHT_INDICES.value: upper_idx,
        BASalientEntity.RIGHT_MASK.value: right_mask,
    }
