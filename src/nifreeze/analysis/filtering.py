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
"""Analysis data filtering."""

import numpy as np


def normalize(x: np.ndarray):
    r"""Normalize data using the z-score.

    The z-score normalization is computed as:

    .. math::

        z_i = \frac{x_i - \mu}{\sigma}

    where :math:`x_i` is the framewise displacement at point :math:`i`,
    :math:`\mu` is the mean of all values, :math:`\sigma` is the standard
    deviation of the values, and :math:`z_i` is the normalized z-score.

    Parameters
    ----------
    x : :obj:`~numpy.ndarray`
        Data to be normalized.

    Returns
    -------
    :obj:`~numpy.ndarray`
        Normalized data.
    """

    return (x - np.mean(x)) / np.std(x)
