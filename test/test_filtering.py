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

import numpy as np
import pytest

from nifreeze.data.filtering import (
    BVAL_ATOL,
    dwi_select_shells,
    grand_mean_normalization,
    robust_minmax_normalization,
)


def _generate_random_choices(request, values, count):
    rng = request.node.rng

    num_elements = len(values)

    # Randomly distribute N among the given values
    partitions = rng.multinomial(count, np.ones(num_elements) / num_elements)

    # Create a list of selected values
    selected_values = [
        val for val, count in zip(values, partitions, strict=True) for _ in range(count)
    ]

    return sorted(selected_values)


def _create_random_gtab_dataobj(request, n_gradients=10, shells=(1000, 2000, 3000), b0s=1):
    rng = request.node.rng

    # Generate a random number of elements for each shell
    bvals_shells = _generate_random_choices(request, shells, n_gradients)

    bvals = np.hstack([b0s * [0], bvals_shells])
    bvecs = np.hstack([np.zeros((3, b0s)), rng.random((3, n_gradients))])

    return bvals, bvecs


def _random_uniform_4d_data(request, size=(32, 32, 32, 5), a=0.0, b=1.0) -> np.ndarray:
    """Create 4D random uniform data for testing."""

    rng = request.node.rng
    data = rng.random(size=size).astype(np.float32)
    return (b - a) * data + a


@pytest.mark.parametrize(
    "n_gradients, shells, index, expected_output",
    [(5, (1000, 2000, 3000), 3, np.asarray([False, True, True, True, True, False]))],
)
def test_dwi_select_shells(request, n_gradients, shells, index, expected_output):
    bvals, bvecs = _create_random_gtab_dataobj(request, n_gradients=n_gradients, shells=shells)

    gradients = np.vstack([bvecs, bvals[np.newaxis, :]], dtype="float32")

    shell_mask = dwi_select_shells(
        gradients.T,
        index,
        atol_low=BVAL_ATOL,
        atol_high=BVAL_ATOL,
    )

    assert np.all(shell_mask == expected_output)


@pytest.mark.parametrize("a, b, mask, center", [(0.0, 2.0, None, 1)])
def test_grand_mean_normalization(request, a, b, mask, center):
    data = _random_uniform_4d_data(request, a=a, b=b)

    centers = np.median(data, axis=(0, 1, 2))
    reference = np.percentile(centers[centers >= 1.0], center)
    centers[centers < 1.0] = reference
    drift = reference / centers
    expected_output = data * drift

    normalized_data = grand_mean_normalization(data, mask=mask, center=center)

    assert np.allclose(normalized_data, expected_output, atol=1e-6)


@pytest.mark.parametrize("a, b, mask, p_min, p_max", [(0.0, 2.0, None, 5.0, 95.0)])
def test_robust_minmax_normalization(request, a, b, mask, p_min, p_max):
    data = _random_uniform_4d_data(request, a=a, b=b)

    reshaped_data = data.reshape((-1, data.shape[-1]))
    p5 = np.percentile(reshaped_data, p_min, axis=0)
    p95 = np.percentile(reshaped_data, p_max, axis=0)
    p_range = p95 - p5
    p_mean = np.mean(p_range)
    p5_mean = np.mean(p5)
    expected_output = (data - p5) * p_mean / p_range + p5_mean

    normalized_data = robust_minmax_normalization(data, mask=mask, p_min=p_min, p_max=p_max)
    assert np.allclose(normalized_data, expected_output, atol=1e-6)
