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
"""Unit tests exercising the analysis."""

import numpy as np
import pytest

from nifreeze.analysis.measure_agreement import (
    BASalientEntity,
    compute_bland_altman_features,
    compute_z_score,
    identify_bland_altman_salient_data,
)
from nifreeze.analysis.motion import identify_spikes


def test_compute_z_score():
    # Verify that a proper confidence interval value is required
    ci = 1.01
    with pytest.raises(ValueError):
        compute_z_score(ci)

    ci = 0.95
    expected_val = 1.96
    z_score = compute_z_score(ci)

    np.allclose(z_score, expected_val, atol=1e-2)


def test_compute_bland_altman_features(request):
    rng = request.node.rng

    n_samples = 350
    ci = 0.95

    # Verify that the data is compliant

    # Data must be the same size
    _data1 = rng.normal(0, 5, n_samples)
    _data2 = rng.normal(-8, 10, (n_samples + 2))
    with pytest.raises(ValueError):
        compute_bland_altman_features(_data1, _data2, ci)

    # Data must be 1D
    _data1 = rng.normal(0, 5, (n_samples, 1))
    _data2 = rng.normal(-8, 10, (n_samples, 1))
    with pytest.raises(ValueError):
        compute_bland_altman_features(_data1, _data2, ci)

    # No missing data is allowed
    _data1 = rng.normal(0, 5, n_samples)
    _data2 = rng.normal(-8, 10, n_samples)
    _data2[-1] = np.nan
    with pytest.raises(ValueError):
        compute_bland_altman_features(_data1, _data2, ci)

    # Generate measurements

    # True values
    true_values = rng.normal(100, 10, n_samples)

    _data1 = true_values + rng.normal(0, 5, n_samples)
    _data2 = true_values + rng.normal(-8, 10, n_samples)

    # Verify that a proper confidence interval value is required
    ci = 1.01
    with pytest.raises(ValueError):
        compute_bland_altman_features(_data1, _data2, ci)

    ci = 0.95
    (
        diff,
        mean,
        mean_diff,
        std_diff,
        loa_lower,
        loa_upper,
        ci_mean,
        ci_loa,
    ) = compute_bland_altman_features(_data1, _data2, ci=ci)

    assert len(diff) == n_samples
    assert len(mean) == n_samples
    assert np.isscalar(mean_diff)
    assert np.isscalar(std_diff)
    assert np.isscalar(loa_lower)
    assert np.isscalar(loa_upper)
    assert loa_lower < loa_upper  # type: ignore
    assert np.isscalar(ci_mean)
    assert np.isscalar(ci_loa)


def test_identify_bland_altman_salient_data():
    _data1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    _data2 = np.array([1.1, 2.1, 1.1, 2.7, 3.4, 5.1, 2.2, 6.3, 7.6, 8.2])

    ci = 0.95

    # Verify that a sufficient number of data points exists to get the requested
    # number of salient data points exists
    top_n = 6
    with pytest.raises(ValueError):
        identify_bland_altman_salient_data(_data1, _data2, ci, top_n)

    top_n = 4

    # Verify that the percentile is not restrictive enough to get the requested
    # number of rightmost salient data points exists
    percentile = 0.75
    with pytest.raises(ValueError):
        identify_bland_altman_salient_data(_data1, _data2, ci, top_n, percentile=percentile)

    percentile = 0.8
    salient_data = identify_bland_altman_salient_data(
        _data1, _data2, ci, top_n, percentile=percentile
    )

    assert len(salient_data[BASalientEntity.RELIABILITY_MASK.value]) == len(_data1)

    assert len(salient_data[BASalientEntity.LEFT_INDICES.value]) == top_n
    assert len(salient_data[BASalientEntity.LEFT_MASK.value]) == len(_data1)

    assert len(salient_data[BASalientEntity.RIGHT_INDICES.value]) == top_n
    assert len(salient_data[BASalientEntity.RIGHT_MASK.value]) == len(_data1)


def test_identify_spikes():
    rng = np.random.default_rng(1234)

    n_samples = 450

    fd = rng.normal(0, 5, n_samples)
    threshold = 2.0

    expected_indices = np.asarray([5, 57, 85, 100, 127, 180, 191, 202, 335, 393, 409])
    expected_mask = np.zeros(n_samples, dtype=bool)
    expected_mask[expected_indices] = True

    obtained_indices, obtained_mask = identify_spikes(fd, threshold=threshold)

    assert np.array_equal(obtained_indices, expected_indices)
    assert np.array_equal(obtained_mask, expected_mask)
