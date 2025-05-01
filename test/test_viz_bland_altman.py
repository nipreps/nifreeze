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
"""Unit tests exercising the Bland-Altman plot."""

import numpy as np
import pytest
from matplotlib import pyplot as plt

from nifreeze.analysis.measure_agreement import BASalientEntity, identify_bland_altman_salient_data
from nifreeze.viz.bland_altman import BASalientEntityColor, plot_bland_altman


def test_plot_bland_altman(request, tmp_path):
    rng = request.node.rng

    n_samples = 450

    # Verify that the data is compliant

    # Data must be the same size
    _data1 = rng.normal(0, 5, n_samples)
    _data2 = rng.normal(-8, 10, (n_samples + 2))
    with pytest.raises(ValueError):
        plot_bland_altman(_data1, _data2)

    # Data must be 1D
    _data1 = rng.normal(0, 5, (n_samples, 1))
    _data2 = rng.normal(-8, 10, (n_samples, 1))
    with pytest.raises(ValueError):
        plot_bland_altman(_data1, _data2)

    # No missing data is allowed
    _data1 = rng.normal(0, 5, n_samples)
    _data2 = rng.normal(-8, 10, n_samples)
    _data2[-1] = np.nan
    with pytest.raises(ValueError):
        plot_bland_altman(_data1, _data2)

    # Generate measurements

    # True values
    true_values = rng.normal(100, 10, n_samples)

    _data1 = true_values + rng.normal(0, 5, n_samples)
    _data2 = true_values + rng.normal(-8, 10, n_samples)

    # Verify that a proper confidence interval value is required
    ci = 1.01
    with pytest.raises(ValueError):
        plot_bland_altman(_data1, _data2, ci=ci)

    ci = 0.95
    fig = plot_bland_altman(_data1, _data2, ci=ci)

    out_svg = tmp_path / "bland-altman.svg"
    fig.savefig(out_svg, format="svg")

    top_n = 100
    percentile = 0.75
    salient_data = identify_bland_altman_salient_data(
        _data1, _data2, ci, top_n=top_n, percentile=percentile
    )

    cmap = plt.get_cmap("cividis")
    left_color = cmap(0)
    right_color = cmap(cmap.N - 1)

    salient_data = {
        BASalientEntity.RELIABILITY_MASK.value: salient_data[
            BASalientEntity.RELIABILITY_MASK.value
        ],
        BASalientEntityColor.RELIABLE_COLOR.value: "powderblue",
        BASalientEntity.LEFT_MASK.value: salient_data[BASalientEntity.LEFT_MASK.value],
        BASalientEntityColor.LEFT_COLOR.value: left_color,
        BASalientEntity.RIGHT_MASK.value: salient_data[BASalientEntity.RIGHT_MASK.value],
        BASalientEntityColor.RIGHT_COLOR.value: right_color,
    }
    fig = plot_bland_altman(_data1, _data2, ci=ci, salient_data=salient_data)

    out_svg = tmp_path / "bland-altman_salient_data.svg"
    fig.savefig(out_svg, format="svg")
