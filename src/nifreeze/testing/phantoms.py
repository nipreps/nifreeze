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
"""
Noise of the synthetic reference datasets (see :doc:`/phantoms`).

The simulator writes noise-free images. Noise is added afterwards, so that one
noise-free volume can be stored once and given the same noise in every dataset
that contains it, perturbed or not: the noise of a volume is drawn from a
generator keyed on a seed and on the volume's index in the full dataset.

"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def noise_sigma(bzero: np.ndarray, white_matter: np.ndarray, snr: float) -> float:
    """
    Standard deviation of the noise in each channel for a given b=0 SNR.

    Parameters
    ----------
    bzero : :obj:`~numpy.ndarray`
        A noise-free b=0 volume.
    white_matter : :obj:`~numpy.ndarray`
        Boolean mask of white matter, where the SNR is defined.
    snr : :obj:`float`
        Mean white-matter b=0 signal over the noise standard deviation.

    Returns
    -------
    :obj:`float`
        The standard deviation of the real and of the imaginary noise.

    Examples
    --------
    >>> noise_sigma(np.full((2, 2, 2), 30.0), np.ones((2, 2, 2), dtype=bool), 10.0)
    3.0

    """
    return float(np.mean(np.asarray(bzero, dtype=float)[white_matter]) / snr)


def add_noise(
    volumes: np.ndarray,
    sigma: float,
    seed: int,
    indices: Iterable[int] | None = None,
) -> np.ndarray:
    """
    Add complex Gaussian noise to noise-free volumes, and take the magnitude.

    The noise-free volumes are real (their phase is zero). Each channel gets
    noise of standard deviation ``sigma``, so the magnitude follows a Rice
    distribution, and a Rayleigh one where there is no signal.

    Parameters
    ----------
    volumes : :obj:`~numpy.ndarray`
        Noise-free volumes, of shape ``(N, X, Y, Z)``.
    sigma : :obj:`float`
        Standard deviation of the real and of the imaginary noise.
    seed : :obj:`int`
        Seed of the noise realization.
    indices : :obj:`~collections.abc.Iterable`, optional
        Index of each volume in the full dataset (``0, 1, ...`` by default),
        which keys the noise of that volume.

    Returns
    -------
    :obj:`~numpy.ndarray`
        The noisy magnitudes, ``float32``, of the same shape.

    Examples
    --------
    >>> volumes = np.zeros((3, 4, 4, 4))
    >>> noisy = add_noise(volumes, 1.0, seed=7)
    >>> np.array_equal(add_noise(volumes[1:], 1.0, seed=7, indices=[1, 2]), noisy[1:])
    True

    """
    volumes = np.asarray(volumes)
    indices = range(len(volumes)) if indices is None else [int(i) for i in indices]
    noisy = np.empty(volumes.shape, dtype=np.float32)
    for k, index in enumerate(indices):
        noise = np.random.default_rng([seed, index]).normal(0.0, sigma, (2, *volumes.shape[1:]))
        noisy[k] = np.hypot(volumes[k] + noise[0], noise[1])
    return noisy
