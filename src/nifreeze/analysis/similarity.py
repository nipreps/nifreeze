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
"""Similarity between a predicted and an actual volume, within a mask."""

from __future__ import annotations

import numpy as np

DEFAULT_MI_BINS = 32
"""Number of intensity bins of the mutual information histogram."""


def _masked_pair(
    predicted: np.ndarray, reference: np.ndarray, mask: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    predicted = np.asarray(predicted, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if predicted.shape != reference.shape:
        raise ValueError(f"Shapes differ: {predicted.shape} and {reference.shape}.")
    if mask is None:
        return predicted.ravel(), reference.ravel()
    mask = np.asarray(mask, dtype=bool)
    return predicted[mask], reference[mask]


def masked_cc(
    predicted: np.ndarray, reference: np.ndarray, mask: np.ndarray | None = None
) -> float:
    """
    Pearson correlation between two volumes, within a mask.

    Parameters
    ----------
    predicted : :obj:`~numpy.ndarray`
        The predicted volume.
    reference : :obj:`~numpy.ndarray`
        The actual volume.
    mask : :obj:`~numpy.ndarray`, optional
        Boolean mask of the voxels compared (all if :obj:`None`).

    Returns
    -------
    :obj:`float`
        The correlation, or 0 if either volume is constant within the mask.

    Examples
    --------
    >>> x = np.arange(10.0)
    >>> round(masked_cc(3 * x + 2, x), 6)
    1.0
    >>> round(masked_cc(-x, x), 6)
    -1.0

    """
    a, b = _masked_pair(predicted, reference, mask)
    a = a - a.mean()
    b = b - b.mean()
    denominator = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denominator) if denominator > 0 else 0.0


def mutual_information(
    predicted: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray | None = None,
    bins: int = DEFAULT_MI_BINS,
) -> float:
    """
    Mutual information between two volumes, within a mask.

    The joint histogram has ``bins`` equal-width bins spanning each volume's
    range within the mask.

    Parameters
    ----------
    predicted : :obj:`~numpy.ndarray`
        The predicted volume.
    reference : :obj:`~numpy.ndarray`
        The actual volume.
    mask : :obj:`~numpy.ndarray`, optional
        Boolean mask of the voxels compared (all if :obj:`None`).
    bins : :obj:`int`, optional
        Number of intensity bins per volume.

    Returns
    -------
    :obj:`float`
        The mutual information, in nats. A volume shares with itself its own
        (binned) entropy, the largest value any prediction can reach.

    Examples
    --------
    >>> x = np.repeat(np.arange(4.0), 25)
    >>> bool(np.isclose(mutual_information(x, x, bins=4), np.log(4)))
    True
    >>> rng = np.random.default_rng(0)
    >>> mutual_information(rng.random(10000), rng.random(10000), bins=4) < 0.01
    True

    """
    a, b = _masked_pair(predicted, reference, mask)
    joint, _, _ = np.histogram2d(a, b, bins=bins)
    joint /= joint.sum()
    marginal_a = joint.sum(axis=1, keepdims=True)
    marginal_b = joint.sum(axis=0, keepdims=True)
    nonzero = joint > 0
    ratio = joint[nonzero] / (marginal_a @ marginal_b)[nonzero]
    return float((joint[nonzero] * np.log(ratio)).sum())


def nrmse(
    predicted: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray | None = None,
    scale: float | None = None,
) -> float:
    """
    Root-mean-square error, normalized by the standard deviation of the reference.

    Parameters
    ----------
    predicted : :obj:`~numpy.ndarray`
        The predicted volume.
    reference : :obj:`~numpy.ndarray`
        The actual volume.
    mask : :obj:`~numpy.ndarray`, optional
        Boolean mask of the voxels compared (all if :obj:`None`).
    scale : :obj:`float`, optional
        Denominator replacing the reference's standard deviation, so that errors
        measured against different references share one scale.

    Returns
    -------
    :obj:`float`
        The normalized error: 0 for a perfect prediction, 1 for predicting the
        reference's mean everywhere.

    Examples
    --------
    >>> x = np.arange(10.0)
    >>> nrmse(x, x)
    0.0
    >>> round(nrmse(np.full(10, x.mean()), x), 6)
    1.0
    >>> float(nrmse(x + 1, x, scale=2.0))
    0.5

    """
    a, b = _masked_pair(predicted, reference, mask)
    if scale is None:
        scale = b.std()
        if scale == 0:
            raise ValueError("The reference is constant within the mask.")
    elif scale <= 0:
        raise ValueError(f"The scale must be positive, got {scale}.")
    return float(np.sqrt(np.mean((a - b) ** 2)) / scale)


METRICS = {"cc": masked_cc, "mi": mutual_information, "nrmse": nrmse}
"""Similarity metrics by name."""
