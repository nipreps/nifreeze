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

import numpy as np
import scipy.stats as stats


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
    # SE_{LoA} = \sigma_{d} \ sqrt{2/n}
    # The confidence interval (CI) for LoA is then calculated using the
    # t-distribution critical value.
    std_err_loa = std_diff * np.sqrt(2 / n)
    ci_loa = t_val * std_err_loa

    return diff, mean, mean_diff, std_diff, loa_lower, loa_upper, ci_mean, ci_loa
