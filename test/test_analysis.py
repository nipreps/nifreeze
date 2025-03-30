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
    compute_bland_altman_features,
    compute_z_score,
)


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
    assert loa_lower < loa_upper
    assert np.isscalar(ci_mean)
    assert np.isscalar(ci_loa)
