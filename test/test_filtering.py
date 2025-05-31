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

import copy

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
    "n_gradients, shells, index, expect_exception, expected_output",
    [
        (5, (1000, 2000, 3000), 3, False, np.asarray([False, True, True, False, True, False])),
        (5, (1000, 2000, 3000), 0, True, np.asarray([])),
    ],
)
def test_dwi_select_shells(request, n_gradients, shells, index, expect_exception, expected_output):
    bvals, bvecs = _create_random_gtab_dataobj(request, n_gradients=n_gradients, shells=shells)

    gradients = np.vstack([bvecs, bvals[np.newaxis, :]], dtype="float32")

    if expect_exception:
        with pytest.raises(RuntimeError):
            _ = dwi_select_shells(
                gradients.T,
                index,
                atol_low=BVAL_ATOL,
                atol_high=BVAL_ATOL,
            )
    else:
        shell_mask = dwi_select_shells(
            gradients.T,
            index,
            atol_low=BVAL_ATOL,
            atol_high=BVAL_ATOL,
        )

        assert np.all(shell_mask == expected_output)


@pytest.mark.parametrize(
    "a, b, use_mask, center",
    [
        (0.0, 2.0, False, 1),
        (0.0, 2.0, True, 1),
    ],
)
def test_grand_mean_normalization(request, a, b, use_mask, center):
    data = _random_uniform_4d_data(request, a=a, b=b)

    mask = None
    # Mask the last volume for testing purposes
    if use_mask:
        mask = np.ones(data.shape[-1], dtype=bool)
        mask[-1] = False

    expected_output = copy.deepcopy(data)

    _mask = mask if mask is not None else np.ones(data.shape[-1], dtype=bool)
    volumes = data[..., _mask]
    centers = np.median(volumes, axis=(0, 1, 2))
    reference = np.percentile(centers[centers >= 1.0], center)
    centers[centers < 1.0] = reference
    drift = reference / centers
    expected_output[..., _mask] = volumes * drift

    normalized_data = grand_mean_normalization(data, mask=mask, center=center)

    assert np.allclose(normalized_data, expected_output, atol=1e-6)


@pytest.mark.parametrize(
    "a, b, use_mask, p_min, p_max",
    [
        (0.0, 2.0, False, 5.0, 95.0),
        (0.0, 2.0, True, 5.0, 95.0),
    ],
)
def test_robust_minmax_normalization(request, a, b, use_mask, p_min, p_max):
    data = _random_uniform_4d_data(request, a=a, b=b)

    mask = None
    # Mask the last volume for testing purposes
    if use_mask:
        mask = np.ones(data.shape[-1], dtype=bool)
        mask[-1] = False

    expected_output = copy.deepcopy(data)

    _mask = mask if mask is not None else np.ones(data.shape[-1], dtype=bool)
    volumes = data[..., _mask]
    reshape_shape = (-1, volumes.shape[-1]) if _mask is None else (-1, sum(_mask))
    reshaped_data = volumes.reshape(reshape_shape)
    p5 = np.percentile(reshaped_data, p_min, axis=0)
    p95 = np.percentile(reshaped_data, p_max, axis=0)
    p_range = p95 - p5
    p_mean = np.mean(p_range)
    p5_mean = np.mean(p5)
    expected_output[..., _mask] = (volumes - p5) * p_mean / p_range + p5_mean

    normalized_data = robust_minmax_normalization(data, mask=mask, p_min=p_min, p_max=p_max)
    assert np.allclose(normalized_data, expected_output, atol=1e-6)
