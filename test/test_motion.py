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
"""Unit tests exercising motion analysis."""

import numpy as np

from nifreeze.analysis.motion import identify_spikes


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
