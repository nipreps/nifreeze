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
"""Bland-Altman plot."""

import matplotlib.pyplot as plt
import numpy as np

from nifreeze.analysis.measure_agreement import (
    compute_bland_altman_features,
    compute_z_score,
)


def plot_bland_altman(
    data1: np.ndarray,
    data2: np.ndarray,
    ci: float = 0.95,
    figsize: tuple | None = (15, 10),
) -> plt.Figure:
    """Create a Bland-Altman plot.

    Create a Bland-Altman plot [Bland86]_ and highlight ``size`` lower and upper
    extremes along the X coordinates.

    Parameters
    ----------
    data1 : :obj:`numpy.ndarray`
        Data values.
    data2 : :obj:`numpy.ndarray`
        Data values.
    ci : :obj:`float`, optional
        Confidence interval value. Must be in the [0, 1] range.
    figsize : :obj:`tuple`, optional
        Figure size.

    Returns
    -------
    fig : :obj:`matplotlib.pyplot.Figure`
        Matplotlib figure.

    References
    ----------
    .. [Bland86] J. Martin Bland and Douglas G. Altman, Statistical methods for
       assessing agreement between two methods of clinical measurement, The
       Lancet 327(8476) (1986) 307-310

    """

    diff, mean, mean_diff, std_diff, loa_lower, loa_upper, ci_mean, ci_loa = (
        compute_bland_altman_features(data1, data2, ci)
    )

    fig, ax = plt.subplots(figsize=figsize)

    mean_color = "blue"
    loa_color = "red"

    strict = True

    # Plot the mean and limit of agreement lines
    alpha_l = 0.2
    linestyle = "--"

    horiz_line_y = np.asarray([mean_diff, loa_upper, loa_lower])
    horiz_line_color = [mean_color, loa_color, loa_color]
    labels = ["Mean difference", "Upper LoA", "Lower LoA"]
    for line_data, color, label in zip(horiz_line_y, horiz_line_color, labels, strict=strict):
        plt.axhline(line_data, color=color, alpha=alpha_l, linestyle=linestyle, label=label)

    # Plot confidence intervals as shaded regions
    ci_label = f"{ci * 100:.0f}"

    horiz_span = [
        [mean_diff - ci_mean, mean_diff + ci_mean],
        [loa_lower - ci_loa, loa_lower + ci_loa],
        [loa_upper - ci_loa, loa_upper + ci_loa],
    ]
    horiz_span_color = [mean_color, loa_color, loa_color]
    labels = [
        f"{ci_label}% CI (Mean)",
        f"{ci_label}% CI (lower LoA)",
        f"{ci_label}% CI (upper LoA)",
    ]
    for span, color, label in zip(horiz_span, horiz_span_color, labels, strict=strict):
        plt.axhspan(span[0], span[1], color=color, alpha=alpha_l, label=label)

    # Plot the BA data points
    alpha_p = 0.5
    plt.scatter(mean, diff, alpha=alpha_p, label="Data", color="gray")

    # Add the mean and limit of agreement text
    factor = 1.02
    x_out_plot = np.min(mean) + (np.max(mean) - np.min(mean)) * factor

    z_score = compute_z_score(ci)
    z_score_label = f"{z_score:.2f}"

    ha = "center"
    va = "center"

    mean_loa_vals = np.asarray([mean_diff, loa_lower, loa_upper])
    mean_loa_text = [
        f"Mean\n{mean_diff:.2f}",
        f"-{str(z_score_label)}SD\n{loa_lower:.2f}",
        f"+{str(z_score_label)}SD\n{loa_upper:.2f}",
    ]
    for val, _text in zip(mean_loa_vals, mean_loa_text, strict=strict):
        plt.text(x_out_plot, val, _text, ha=ha, va=va)

    plt.xlabel("Average")
    plt.ylabel("Difference")
    plt.legend()

    plt.tight_layout()

    return fig
