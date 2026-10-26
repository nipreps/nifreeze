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
Known perturbations of a dataset, for the controls of :mod:`nifreeze.analysis.sensitivity`.

The sensitivity analysis compares predictions on a reference dataset with
predictions on perturbed copies of it, whose ground truth it is given (a
:class:`~nifreeze.analysis.sensitivity.Perturbation`). Realistic copies come from
a simulator; these helpers resample a dataset through rigid motions and
eddy-current-like distortions instead, so that tests can check the analysis
against answers known in closed form.

"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from copy import copy
from typing import Any

import nibabel as nb
import numpy as np
from numpy.typing import ArrayLike
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation

from nifreeze.analysis.sensitivity import Distortion, Perturbation, _monomials, _resolve_mask


def rigid_motion(
    magnitude: float,
    axis: ArrayLike,
    direction: ArrayLike,
    center: ArrayLike = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """
    Rotate by ``magnitude`` degrees about ``axis`` and shift by ``magnitude`` mm.

    Parameters
    ----------
    magnitude : :obj:`float`
        Rotation angle (degrees) and translation length (mm).
    axis : :obj:`~numpy.typing.ArrayLike`
        Rotation axis (normalized internally).
    direction : :obj:`~numpy.typing.ArrayLike`
        Translation direction (normalized internally).
    center : :obj:`~numpy.typing.ArrayLike`, optional
        World coordinates of a point on the rotation axis.

    Returns
    -------
    :obj:`~numpy.ndarray`
        The ``(4, 4)`` motion, mapping head positions before the motion to
        positions after it.

    Examples
    --------
    >>> motion = rigid_motion(90.0, (0, 0, 1), (1, 0, 0), center=(10, 0, 0))
    >>> np.round(motion @ (10, 0, 5, 1), 6).tolist()
    [100.0, 0.0, 5.0, 1.0]
    >>> np.round(motion @ (11, 0, 0, 1), 6).tolist()
    [100.0, 1.0, 0.0, 1.0]

    """
    unit_axis = np.asarray(axis, dtype=float)
    unit_axis /= np.linalg.norm(unit_axis)
    shift = np.asarray(direction, dtype=float)
    shift *= magnitude / np.linalg.norm(shift)
    pivot = np.asarray(center, dtype=float)

    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_rotvec(np.deg2rad(magnitude) * unit_axis).as_matrix()
    matrix[:3, 3] = pivot - matrix[:3, :3] @ pivot + shift
    return matrix


def mask_centroid(dataset: Any, mask: np.ndarray | None = None) -> np.ndarray:
    """
    World coordinates of the centroid of a mask (the brain mask by default).

    Parameters
    ----------
    dataset : :obj:`~nifreeze.data.base.BaseDataset`
        The dataset.
    mask : :obj:`~numpy.ndarray`, optional
        Boolean mask; the dataset's brain mask, or the whole volume, if :obj:`None`.

    Returns
    -------
    :obj:`~numpy.ndarray`
        The centroid, of shape ``(3,)``.

    """
    mask = _resolve_mask(dataset, mask)
    ijk = np.argwhere(mask).mean(axis=0)
    return nb.affines.apply_affine(dataset.affine, ijk)


def perturbation_plan(
    n_volumes: int,
    magnitudes: Sequence[float],
    counts: Sequence[int],
    seeds: int | Sequence[int] = 5,
    center: ArrayLike = (0.0, 0.0, 0.0),
    fixed: Iterable[int] = (),
) -> list[Perturbation]:
    """
    Draw the perturbations of a sensitivity experiment.

    For each seed, the moved volumes are the first ``count`` of a random ordering
    of the volumes not listed in ``fixed``, each with a random rotation axis and
    translation direction. Volumes moved with a smaller count
    are thus also moved with a larger one, and they follow the same axis and
    direction at every magnitude, so the curves over magnitude and count are
    paired.

    Parameters
    ----------
    n_volumes : :obj:`int`
        Number of volumes in the dataset.
    magnitudes : :obj:`~collections.abc.Sequence`
        Motion magnitudes (degrees and mm, see :func:`rigid_motion`).
    counts : :obj:`~collections.abc.Sequence`
        Numbers of moved volumes.
    seeds : :obj:`int` or :obj:`~collections.abc.Sequence`, optional
        Number of seeds (``0, 1, ...``), or the seeds themselves.
    center : :obj:`~numpy.typing.ArrayLike`, optional
        World coordinates of a point on every rotation axis (see
        :func:`mask_centroid`).
    fixed : :obj:`~collections.abc.Iterable`, optional
        Volumes that never move.

    Returns
    -------
    :obj:`list` of :obj:`~nifreeze.analysis.sensitivity.Perturbation`
        One perturbation per seed, count and magnitude.

    Examples
    --------
    >>> plan = perturbation_plan(10, magnitudes=[0.5, 1.0], counts=[1, 3], seeds=2, fixed=[0])
    >>> len(plan)
    8
    >>> any(0 in p.moved for p in plan)
    False
    >>> set(plan[0].moved) <= set(plan[2].moved)
    True

    """
    candidates = np.array(sorted(set(range(n_volumes)) - set(fixed)), dtype=int)
    if max(counts) > len(candidates):
        raise ValueError(f"Cannot move {max(counts)} volumes: only {len(candidates)} may move.")
    seeds = range(seeds) if isinstance(seeds, int) else seeds

    plan = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        order = rng.permutation(candidates)
        axes = rng.normal(size=(len(order), 3))
        directions = rng.normal(size=(len(order), 3))
        for count in counts:
            for magnitude in magnitudes:
                motions = tuple(
                    rigid_motion(magnitude, axes[k], directions[k], center) for k in range(count)
                )
                moved = tuple(int(j) for j in order[:count])
                plan.append(Perturbation(float(magnitude), seed, moved, motions))
    return plan


def one_at_a_time(
    volumes: Iterable[int],
    magnitude: float,
    seed: int = 0,
    center: ArrayLike = (0.0, 0.0, 0.0),
) -> list[Perturbation]:
    """
    Move each volume alone, to rank the volumes by their influence.

    Parameters
    ----------
    volumes : :obj:`~collections.abc.Iterable`
        The volumes to move, one per perturbation.
    magnitude : :obj:`float`
        Motion magnitude (degrees and mm, see :func:`rigid_motion`).
    seed : :obj:`int`, optional
        Seed for the rotation axes and translation directions.
    center : :obj:`~numpy.typing.ArrayLike`, optional
        World coordinates of a point on every rotation axis.

    Returns
    -------
    :obj:`list` of :obj:`~nifreeze.analysis.sensitivity.Perturbation`
        One perturbation per volume.

    """
    volumes = [int(j) for j in volumes]
    rng = np.random.default_rng(seed)
    axes = rng.normal(size=(len(volumes), 3))
    directions = rng.normal(size=(len(volumes), 3))
    return [
        Perturbation(
            float(magnitude),
            seed,
            (j,),
            (rigid_motion(magnitude, axes[k], directions[k], center),),
        )
        for k, j in enumerate(volumes)
    ]


def eddy_distortion(
    dataset: Any,
    magnitude: float,
    axis: int,
    degree: int = 1,
    seed: int = 0,
    gradients: ArrayLike | None = None,
    mask: np.ndarray | None = None,
) -> Distortion:
    r"""
    Draw distortions that follow the diffusion gradients, as eddy currents do.

    The fields that a diffusion gradient leaves behind scale with its amplitude
    and change sign with its direction. Each volume's coefficients are therefore
    one fixed random linear map of its gradient vector, scaled by
    :math:`\sqrt{b / b_{max}}` (the gradient amplitude, at fixed timing). b=0
    volumes are not distorted, and antipodal gradients get opposite distortions.

    Parameters
    ----------
    dataset : :obj:`~nifreeze.data.base.BaseDataset`
        The aligned dataset.
    magnitude : :obj:`float`
        Largest displacement (mm) of any voxel of the mask, over all volumes.
    axis : :obj:`int`
        Voxel axis of phase encoding.
    degree : :obj:`int`, optional
        Degree of the polynomial (1: translation, scaling and shears).
    seed : :obj:`int`, optional
        Seed for the linear map.
    gradients : :obj:`~numpy.typing.ArrayLike`, optional
        Gradient table, ``(n_volumes, 4)`` with the b-value last; the dataset's
        if :obj:`None`.
    mask : :obj:`~numpy.ndarray`, optional
        Voxels over which the polynomial is normalized and ``magnitude`` reached.

    Returns
    -------
    :obj:`~nifreeze.analysis.sensitivity.Distortion`
        The distortions.

    """
    table = np.asarray(dataset.gradients if gradients is None else gradients, dtype=float)
    vectors, bvals = table[:, :3], np.clip(table[:, 3], 0.0, None)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit = np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)
    peak = bvals.max()
    amplitude = unit * np.sqrt(bvals / peak)[:, np.newaxis] if peak > 0 else 0.0 * unit
    if not amplitude.any():
        raise ValueError("No volume has a diffusion gradient.")

    mask = _resolve_mask(dataset, mask)
    affine = np.asarray(dataset.affine, dtype=float)
    frame = affine[:3, :3] / np.linalg.norm(affine[:3, :3], axis=0)
    points = nb.affines.apply_affine(affine, np.argwhere(mask)).T
    center = points.mean(axis=1)
    radius = np.abs(frame.T @ (points - center[:, np.newaxis])).max(axis=1)

    n_terms = len(_monomials(degree))
    linear_map = np.random.default_rng(seed).normal(size=(3, n_terms))
    distortion = Distortion(
        float(magnitude), axis, seed, degree, center, radius, frame, amplitude @ linear_map
    )
    values, _ = distortion.basis(points)
    largest = max(np.abs(row @ values).max() for row in distortion.coefficients)
    distortion.coefficients[:] *= magnitude / largest
    return distortion


def perturb(dataset: Any, perturbation: Perturbation, order: int = 3) -> Any:
    """
    Copy a dataset with some of its volumes moved or distorted.

    Each moved or distorted volume is resampled once from the aligned original,
    through its motion and then its distortion. A distortion with an EPI readout
    (:attr:`~nifreeze.analysis.sensitivity.Distortion.readout`) instead forms the
    moved volume as that readout would. Everything else (gradients, b=0
    reference, mask) is left as it was, as a model would see it when the motion
    and the distortion go uncorrected.

    Parameters
    ----------
    dataset : :obj:`~nifreeze.data.base.BaseDataset`
        The aligned dataset.
    perturbation : :obj:`~nifreeze.analysis.sensitivity.Perturbation`
        The volumes to move, their motions, and the distortion.
    order : :obj:`int`, optional
        Interpolation order.

    Returns
    -------
    :obj:`~nifreeze.data.base.BaseDataset`
        A copy of the dataset with ``float32`` volumes.

    """
    distortion = perturbation.distortion
    motions = dict(zip(perturbation.moved, perturbation.motions, strict=True))
    displaced = set(motions)
    if distortion is not None:
        displaced |= {int(i) for i in np.flatnonzero(distortion.coefficients.any(axis=1))}

    readout = distortion is not None and distortion.readout is not None
    resampled = None if readout else distortion
    dataobj = np.array(dataset.dataobj, dtype=np.float32)
    for index in sorted(displaced):
        motion = motions.get(index)
        if motion is not None or resampled is not None:
            dataobj[index] = _displace(dataset, index, dataset.affine, motion, resampled, order)
        if readout:
            assert distortion is not None
            dataobj[index] = distortion.distort(
                dataobj[index], dataset.affine, distortion.coefficients[index], order
            )

    moved = copy(dataset)
    moved.dataobj = dataobj
    return moved


def _displace(
    dataset: Any,
    index: int,
    affine: np.ndarray,
    motion: np.ndarray | None,
    distortion: Distortion | None,
    order: int,
) -> np.ndarray:
    """Resample one aligned volume onto a grid, through its motion and distortion."""
    shape = dataset.shape3d
    ijk = np.indices(shape, dtype=float).reshape(3, -1)
    points = affine[:3, :3] @ ijk + affine[:3, 3:]
    jacobian: float | np.ndarray = 1.0
    if distortion is not None:
        values, derivatives = distortion.basis(points)
        coefficients = distortion.coefficients[index]
        # The scanner shows at each point what lies further along phase encoding
        points = points + np.outer(distortion.direction, coefficients @ values)
        jacobian = 1.0 + coefficients @ derivatives
    if motion is not None:
        # The moved head shows, at each position, what was at motion⁻¹(position)
        inverse = np.linalg.inv(motion)
        points = inverse[:3, :3] @ points + inverse[:3, 3:]
    to_voxels = np.linalg.inv(dataset.affine)
    coordinates = to_voxels[:3, :3] @ points + to_voxels[:3, 3:]
    volume = np.asarray(dataset.dataobj[index], dtype=np.float32)
    resampled = map_coordinates(volume, coordinates, order=order)  # type: ignore[call-overload]
    return (resampled * jacobian).reshape(shape).astype(np.float32)
