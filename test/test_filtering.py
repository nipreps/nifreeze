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


@pytest.mark.random_gtab_data(5, (1000, 2000, 3000), 1)
@pytest.mark.parametrize(
    "index, expect_exception, expected_output",
    [
        (3, False, np.asarray([False, True, True, False, False, False])),
        (0, True, np.asarray([])),
    ],
)
def test_dwi_select_shells(setup_random_gtab_data, index, expect_exception, expected_output):
    bvals, bvecs = setup_random_gtab_data

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


@pytest.mark.random_uniform_ndim_data((32, 32, 32, 5), 0.0, 2.0)
@pytest.mark.parametrize(
    "use_mask, center, inplace",
    [
        (False, 1, True),
        (True, 1, False),
    ],
)
def test_grand_mean_normalization(setup_random_uniform_ndim_data, use_mask, center, inplace):
    data = setup_random_uniform_ndim_data

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

    normalized_data = grand_mean_normalization(data, mask=mask, center=center, inplace=inplace)

    assert (normalized_data is None) if inplace else isinstance(normalized_data, np.ndarray)
    assert not np.shares_memory(normalized_data, data) if not inplace else True
    assert (
        np.allclose(data, expected_output, atol=1e-6)
        if inplace
        else np.allclose(normalized_data, expected_output, atol=1e-6)
    )


@pytest.mark.random_uniform_ndim_data((32, 32, 32, 5), 0.0, 2.0)
@pytest.mark.parametrize(
    "use_mask, p_min, p_max, inplace",
    [
        (False, 5.0, 95.0, True),
        (True, 5.0, 95.0, False),
    ],
)
def test_robust_minmax_normalization(
    setup_random_uniform_ndim_data, use_mask, p_min, p_max, inplace
):
    data = setup_random_uniform_ndim_data

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

    normalized_data = robust_minmax_normalization(
        data, mask=mask, p_min=p_min, p_max=p_max, inplace=inplace
    )

    assert (normalized_data is None) if inplace else isinstance(normalized_data, np.ndarray)
    assert not np.shares_memory(normalized_data, data) if not inplace else True
    assert (
        np.allclose(data, expected_output, atol=1e-6)
        if inplace
        else np.allclose(normalized_data, expected_output, atol=1e-6)
    )
