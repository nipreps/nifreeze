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

from nifreeze.data.filtering import BVAL_THRESHOLD, clip_dwi_shell_data, detrend_data_percentile


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


def _create_dwi_random_dataobj(request, bvals, bvecs, b0_thres=50, vol_size=(2, 2, 2)):
    rng = request.node.rng

    n_gradients = np.count_nonzero(bvecs)
    b0s = len(bvals) - n_gradients
    volumes = n_gradients + b0s

    dwi_dataobj = rng.random((*vol_size, volumes), dtype="float32")
    affine = np.eye(4, dtype="float32")
    brainmask_dataobj = rng.random(vol_size, dtype="float32")
    b0_dataobj = rng.random(vol_size, dtype="float32")
    gradients = np.vstack([bvecs, bvals[np.newaxis, :]], dtype="float32")

    return (
        dwi_dataobj,
        affine,
        brainmask_dataobj,
        b0_dataobj,
        gradients,
        b0_thres,
    )


def test_clip_dwi_shell_data(request):
    n_gradients = 5
    shells = (1000, 2000, 3000)
    bvals, bvecs = _create_random_gtab_dataobj(request, n_gradients=n_gradients, shells=shells)

    (
        dwi_dataobj,
        affine,
        brainmask_dataobj,
        b0_dataobj,
        gradients,
        _,
    ) = _create_dwi_random_dataobj(request, bvals, bvecs)

    index = 3

    expected_output = np.asarray(
        [
            [
                [[0.22275364, 0.8670869, 0.7887176], [0.06013864, 0.16233343, 0.04156363]],
                [[0.61101794, 0.54873836, 0.17848688], [0.4389516, 0.45328808, 0.19304818]],
            ],
            [
                [[0.25126708, 0.65764445, 0.43090707], [0.735819, 0.69890034, 0.7759642]],
                [[0.9922592, 0.35762864, 0.8519457], [0.08688301, 0.87896436, 0.43176556]],
            ],
        ]
    )

    clipped_data = clip_dwi_shell_data(
        dwi_dataobj,
        gradients.T,
        index,
        th_low=BVAL_THRESHOLD,
        th_high=BVAL_THRESHOLD,
    )

    assert np.allclose(clipped_data, expected_output, atol=1e-6)


def test_detrend_data_percentile():
    data = np.array(
        [[[10, 20, 30], [40, 50, 60]], [[70, 80, 90], [100, 110, 120]]], dtype=np.float32
    )

    reshaped_data = data.reshape((-1, data.shape[-1]))
    p5 = np.percentile(reshaped_data, 5.0, axis=0)
    p95 = np.percentile(reshaped_data, 95.0, axis=0)
    p_range = p95 - p5
    p_mean = np.mean(p_range)
    p5_mean = np.mean(p5)
    expected_output = (data - p5) * p_mean / p_range + p5_mean

    normalized_data = detrend_data_percentile(data)
    assert np.allclose(normalized_data, expected_output, atol=1e-6)
