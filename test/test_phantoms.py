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
"""Noise of the synthetic reference datasets."""

import numpy as np
import pytest

from nifreeze.testing.phantoms import add_noise, noise_sigma

SHAPE = (24, 24, 24)


def test_noise_follows_the_volume_wherever_it_is_stored():
    """A volume gets the same noise alone, in a subset, or in the full dataset."""
    volumes = np.random.default_rng(0).uniform(0, 100, size=(5, *SHAPE))
    full = add_noise(volumes, 3.0, seed=11)

    subset = add_noise(volumes[[4, 1]], 3.0, seed=11, indices=[4, 1])

    assert np.array_equal(subset, full[[4, 1]])
    assert not np.array_equal(full[0] - volumes[0], full[1] - volumes[1])
    assert not np.array_equal(add_noise(volumes, 3.0, seed=12), full)


def test_noise_is_rician_with_the_requested_sigma():
    """Without signal the magnitude is Rayleigh; with high signal, Gaussian about it."""
    sigma = 2.0
    background = add_noise(np.zeros((2, *SHAPE)), sigma, seed=0)
    assert background.mean() == pytest.approx(sigma * np.sqrt(np.pi / 2), rel=0.02)
    assert (background**2).mean() == pytest.approx(2 * sigma**2, rel=0.02)

    bright = add_noise(np.full((2, *SHAPE), 200.0), sigma, seed=0)
    assert bright.std() == pytest.approx(sigma, rel=0.03)
    assert bright.mean() == pytest.approx(200.0, abs=0.1)


def test_sigma_gives_the_white_matter_snr():
    bzero = np.full(SHAPE, 50.0)
    white = np.zeros(SHAPE, dtype=bool)
    white[8:16, 8:16, 8:16] = True
    bzero[white] = 30.0

    sigma = noise_sigma(bzero, white, snr=15.0)

    assert sigma == pytest.approx(2.0)
    noisy = add_noise(bzero[np.newaxis], sigma, seed=3)[0]
    assert noisy[white].mean() / sigma == pytest.approx(15.0, rel=0.02)
