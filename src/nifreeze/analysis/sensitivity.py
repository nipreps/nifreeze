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
Sensitivity of model predictions to motion of their training volumes.

Two questions are asked of a model, on an aligned dataset:

- :func:`fidelity`: how well does it predict each held-out volume?
- :func:`sensitivity`: how far do its predictions drift from those on the aligned
  dataset when some volumes move by known rigid motions, or are distorted as by
  eddy currents (a *perturbation*)?

A fresh model is built on each perturbed copy. A held-out prediction never uses
its own volume, so every volume can be evaluated, the moved ones included: a
moved volume's prediction sees the motion of the others only. Eddy-current
distortions follow the diffusion gradient, so a prediction may still resemble
its volume's own distortion through volumes with similar gradients:
:func:`inherited_distortion` measures how much. Registration, the estimator and
its intensity window are not involved.

"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import copy
from typing import Any

import attrs
import nibabel as nb
import numpy as np
from nitransforms.linear import Affine
from nitransforms.resampling import apply
from numpy.typing import ArrayLike
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation

from nifreeze.analysis.similarity import METRICS, nrmse


@attrs.define(frozen=True)
class Distortion:
    r"""
    Displacements along the phase-encoding axis, polynomial in space, one per volume.

    Volume ``i`` shows at each point what lies :math:`d_i` mm further along the
    phase-encoding axis, with its intensity scaled by the Jacobian
    :math:`1 + \partial d_i / \partial x_{PE}` so that no signal is created or lost.
    :math:`d_i` is a polynomial of the voxel-axis coordinates, centered on the
    mask and divided by its half-extent along each axis.
    """

    magnitude: float
    """Largest displacement (mm) of any voxel of the mask, over all volumes."""
    axis: int
    """Voxel axis of phase encoding."""
    seed: int | None
    """Seed that drew the map from gradients to coefficients."""
    degree: int
    """Degree of the polynomial."""
    center: np.ndarray = attrs.field(eq=False)
    """World coordinates of the origin of the polynomial."""
    radius: np.ndarray = attrs.field(eq=False)
    """Half-extent (mm) of the mask along each voxel axis."""
    frame: np.ndarray = attrs.field(eq=False)
    """Direction cosines of the voxel axes, as columns."""
    coefficients: np.ndarray = attrs.field(eq=False)
    """Displacement (mm) carried by each term, of shape ``(n_volumes, n_terms)``."""

    @property
    def direction(self) -> np.ndarray:
        """World direction of the phase-encoding axis."""
        return self.frame[:, self.axis]

    def basis(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluate each term, and its derivative along the phase-encoding axis.

        Parameters
        ----------
        points : :obj:`~numpy.ndarray`
            World coordinates, of shape ``(3, n_points)``.

        Returns
        -------
        :obj:`tuple` of :obj:`~numpy.ndarray`
            The terms and their derivatives (per mm), each of shape
            ``(n_terms, n_points)``.

        """
        local = self.frame.T @ (points - self.center[:, np.newaxis])
        normalized = local / self.radius[:, np.newaxis]
        values, derivatives = [], []
        for exponents in _monomials(self.degree):
            values.append(np.prod(normalized ** exponents[:, np.newaxis], axis=0))
            power = exponents[self.axis]
            if power == 0:
                derivatives.append(np.zeros(points.shape[1]))
                continue
            lowered = exponents.copy()
            lowered[self.axis] -= 1
            derivatives.append(
                power
                * np.prod(normalized ** lowered[:, np.newaxis], axis=0)
                / self.radius[self.axis]
            )
        return np.array(values), np.array(derivatives)


@attrs.define(frozen=True)
class Perturbation:
    """Rigid motions of some volumes of a dataset, and optionally a distortion of all."""

    magnitude: float
    """Rotation (degrees) and translation (mm) of every moved volume."""
    seed: int | None
    """Seed that drew the moved volumes and their motions."""
    moved: tuple[int, ...] = ()
    """Indices of the moved volumes."""
    motions: tuple[np.ndarray, ...] = attrs.field(default=(), eq=False)
    """Head motion of each moved volume, as a ``(4, 4)`` matrix in world coordinates."""
    distortion: Distortion | None = None
    """Distortion of every volume, applied after its motion, in the scanner's frame."""

    @property
    def count(self) -> int:
        """Number of moved volumes."""
        return len(self.moved)


NO_PERTURBATION = Perturbation(magnitude=0.0, seed=None)
"""The aligned dataset."""

DEFAULT_GRID_SHIFT = 0.25
"""Offset (voxels, along each axis) of the grid all volumes are resampled onto."""


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
    :obj:`list` of :obj:`Perturbation`
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
    :obj:`list` of :obj:`Perturbation`
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
    :obj:`Distortion`
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


def shift_grid(dataset: Any, shift: float | ArrayLike = DEFAULT_GRID_SHIFT, order: int = 3) -> Any:
    """
    Resample a whole dataset onto a grid offset by a fraction of a voxel.

    Resampling smooths noise, which real motion does not. On the shifted grid
    every volume goes through one interpolation, moved or not (see
    :func:`perturb`), so moved and unmoved volumes are smoothed alike. Only the
    sampling grid moves: the anatomy stays in place. A quarter of a voxel along
    the diagonal smooths white noise as much as resampling through a rigid
    motion of 0.5 to 4 degrees and millimeters (within about 3%), whereas half a
    voxel smooths noticeably more.

    Parameters
    ----------
    dataset : :obj:`~nifreeze.data.base.BaseDataset`
        The aligned dataset.
    shift : :obj:`float` or :obj:`~numpy.typing.ArrayLike`, optional
        Offset of the grid, in voxels, along each axis.
    order : :obj:`int`, optional
        Interpolation order.

    Returns
    -------
    :obj:`~nifreeze.data.base.BaseDataset`
        A copy with ``float32`` volumes on the shifted grid. The brain mask and
        the b=0 reference, if any, are resampled too.

    """
    offset = np.broadcast_to(np.asarray(shift, dtype=float), (3,))
    affine = np.array(dataset.affine, dtype=float)
    affine[:3, 3] += affine[:3, :3] @ offset
    reference = nb.Nifti1Image(np.zeros(dataset.shape3d, dtype=np.uint8), affine)

    def _resample(volume: np.ndarray, interpolation: int) -> np.ndarray:
        image = nb.Nifti1Image(np.asarray(volume, dtype=np.float32), dataset.affine)
        resampled = apply(Affine(np.eye(4)), image, reference=reference, order=interpolation)
        return np.asanyarray(resampled.dataobj)

    shifted = copy(dataset)
    shifted.dataobj = np.stack([_resample(volume, order) for volume in dataset.dataobj])
    shifted.affine = affine
    if getattr(dataset, "brainmask", None) is not None:
        shifted.brainmask = _resample(dataset.brainmask, 1) > 0.5
    if getattr(dataset, "bzero", None) is not None:
        shifted.bzero = _resample(dataset.bzero, order)
    return shifted


def perturb(
    dataset: Any,
    perturbation: Perturbation,
    order: int = 3,
    round_trip: bool = False,
    grid: Any = None,
) -> Any:
    """
    Copy a dataset with some of its volumes moved or distorted.

    Each moved or distorted volume is resampled once from the aligned original,
    through its motion and then its distortion. Everything else (gradients, b=0
    reference, mask) is left as it was, as a model would see it when the motion
    and the distortion go uncorrected.

    Parameters
    ----------
    dataset : :obj:`~nifreeze.data.base.BaseDataset`
        The aligned dataset.
    perturbation : :obj:`Perturbation`
        The volumes to move, their motions, and the distortion.
    order : :obj:`int`, optional
        Interpolation order.
    round_trip : :obj:`bool`, optional
        Move each volume and move it back, which leaves it in place but
        interpolated twice (rigid motions only).
    grid : :obj:`~nifreeze.data.base.BaseDataset`, optional
        The dataset on a shifted grid (see :func:`shift_grid`). Moved volumes are
        then resampled from the original onto that grid, through their motion,
        and everything else is taken from ``grid``.

    Returns
    -------
    :obj:`~nifreeze.data.base.BaseDataset`
        A copy of the dataset (or of ``grid``) with ``float32`` volumes.

    """
    distortion = perturbation.distortion
    if round_trip and grid is not None:
        raise ValueError("A round trip cannot be combined with a shifted grid.")
    if round_trip and distortion is not None:
        raise ValueError("A round trip is defined for rigid motions only.")
    source = dataset if grid is None else grid
    motions = dict(zip(perturbation.moved, perturbation.motions, strict=True))
    displaced = set(motions)
    if distortion is not None:
        displaced |= {int(i) for i in np.flatnonzero(distortion.coefficients.any(axis=1))}

    dataobj = np.array(source.dataobj, dtype=np.float32)
    if round_trip:
        reference = nb.Nifti1Image(np.zeros(dataset.shape3d, dtype=np.uint8), source.affine)
        for index, motion in motions.items():
            image = nb.Nifti1Image(
                np.asarray(dataset.dataobj[index], dtype=np.float32), dataset.affine
            )
            for step in (np.linalg.inv(motion), motion):
                image = apply(Affine(step), image, reference=reference, order=order)
            dataobj[index] = np.asanyarray(image.dataobj)
    else:
        for index in sorted(displaced):
            dataobj[index] = _displace(
                dataset, index, source.affine, motions.get(index), distortion, order
            )

    moved = copy(source)
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


def held_out_predictions(
    model: str | Callable[..., Any],
    dataset: Any,
    evaluate: Iterable[int] | None = None,
    **model_kwargs: Any,
) -> dict[int, np.ndarray]:
    """
    Predict each evaluated volume from all the others, with a fresh model.

    Parameters
    ----------
    model : :obj:`str` or :obj:`~collections.abc.Callable`
        A model name for :class:`~nifreeze.model.base.ModelFactory`, or a callable
        building a model from a dataset (such as a model class).
    dataset : :obj:`~nifreeze.data.base.BaseDataset`
        The dataset the model is built on.
    evaluate : :obj:`~collections.abc.Iterable`, optional
        Volumes to predict (all if :obj:`None`).
    **model_kwargs
        Arguments for the model.

    Returns
    -------
    :obj:`dict`
        The prediction of each evaluated volume.

    """
    if isinstance(model, str):
        from nifreeze.model.base import ModelFactory

        instance = ModelFactory.init(model, dataset=dataset, **model_kwargs)
    else:
        instance = model(dataset, **model_kwargs)
    evaluate = range(len(dataset)) if evaluate is None else evaluate
    return {int(index): np.asarray(instance.fit_predict(index)) for index in evaluate}


def fidelity(
    model: str | Callable[..., Any],
    dataset: Any,
    evaluate: Iterable[int] | None = None,
    mask: np.ndarray | None = None,
    metrics: Mapping[str, Callable[..., float]] = METRICS,
    **model_kwargs: Any,
) -> list[dict[str, Any]]:
    """
    Compare each held-out prediction with its volume, on an aligned dataset.

    Parameters
    ----------
    model : :obj:`str` or :obj:`~collections.abc.Callable`
        The model (see :func:`held_out_predictions`).
    dataset : :obj:`~nifreeze.data.base.BaseDataset`
        The aligned dataset.
    evaluate : :obj:`~collections.abc.Iterable`, optional
        Volumes to predict (all if :obj:`None`).
    mask : :obj:`~numpy.ndarray`, optional
        Voxels compared; the dataset's brain mask, or the whole volume, if :obj:`None`.
    metrics : :obj:`~collections.abc.Mapping`, optional
        Similarity metrics by name, with the signature of
        :func:`~nifreeze.analysis.similarity.masked_cc`.
    **model_kwargs
        Arguments for the model.

    Returns
    -------
    :obj:`list` of :obj:`dict`
        One row per evaluated volume, with each metric.

    """
    mask = _resolve_mask(dataset, mask)
    predicted = held_out_predictions(model, dataset, evaluate, **model_kwargs)
    return [
        {
            "index": index,
            **{
                name: metric(prediction, np.asarray(dataset.dataobj[index], dtype=float), mask)
                for name, metric in metrics.items()
            },
        }
        for index, prediction in predicted.items()
    ]


def sensitivity(
    model: str | Callable[..., Any],
    dataset: Any,
    plan: Iterable[Perturbation],
    evaluate: Iterable[int] | None = None,
    mask: np.ndarray | None = None,
    metrics: Mapping[str, Callable[..., float]] | None = None,
    order: int = 3,
    grid_shift: float | ArrayLike | None = DEFAULT_GRID_SHIFT,
    **model_kwargs: Any,
) -> list[dict[str, Any]]:
    """
    Measure how far held-out predictions drift from the aligned ones under a plan.

    Each prediction on a perturbed copy is compared with the prediction of the
    same volume on the aligned dataset. The change is reported as an NRMSE
    divided by the standard deviation of the actual volume, not by that of the
    aligned prediction: a smooth prediction would otherwise inflate its own
    error, and models could not be compared. By default, the whole dataset is first
    resampled onto a shifted grid (see :func:`shift_grid`), so that every volume
    is interpolated once and moved volumes differ from the others only by their
    displacement.

    Parameters
    ----------
    model : :obj:`str` or :obj:`~collections.abc.Callable`
        The model (see :func:`held_out_predictions`).
    dataset : :obj:`~nifreeze.data.base.BaseDataset`
        The aligned dataset.
    plan : :obj:`~collections.abc.Iterable` of :obj:`Perturbation`
        The perturbations (see :func:`perturbation_plan`).
    evaluate : :obj:`~collections.abc.Iterable`, optional
        Volumes to predict (all if :obj:`None`), moved or not.
    mask : :obj:`~numpy.ndarray`, optional
        Voxels compared, on the grid of the compared volumes (the shifted grid,
        unless ``grid_shift`` is :obj:`None`); the brain mask on that grid, or
        the whole volume, if :obj:`None`.
    metrics : :obj:`~collections.abc.Mapping`, optional
        Further metrics by name, with the signature of
        :func:`~nifreeze.analysis.similarity.masked_cc`, called as
        ``metric(perturbed, aligned, mask)``.
    order : :obj:`int`, optional
        Interpolation order.
    grid_shift : :obj:`float` or :obj:`~numpy.typing.ArrayLike`, optional
        Offset of the shifted grid, in voxels; :obj:`None` resamples only the
        moved volumes, which then come out smoother than the others.
    **model_kwargs
        Arguments for the model.

    Returns
    -------
    :obj:`list` of :obj:`dict`
        One row per perturbation and evaluated volume, with the ``nrmse``, each
        further metric, whether that volume was itself ``moved``, the magnitude
        of the ``distortion`` (0 without one) and the share of the volume's own
        distortion that its prediction ``inherited`` (the projection of the
        coefficients fitted by :func:`inherited_distortion` onto the volume's
        own; NaN without a distortion, or for an undistorted volume). With
        motions too, the share is fitted to the change from the prediction with
        the same motions and no distortion, since a motion alone also projects
        onto the distortion's terms.

    """
    metrics = {} if metrics is None else metrics
    grid = None if grid_shift is None else shift_grid(dataset, grid_shift, order=order)
    aligned = dataset if grid is None else grid
    mask = _resolve_mask(aligned, mask)
    reference = held_out_predictions(model, aligned, evaluate, **model_kwargs)
    scale = {
        index: np.asarray(aligned.dataobj[index], dtype=float)[mask].std() for index in reference
    }

    rows = []
    for perturbation in plan:
        moved = perturb(dataset, perturbation, order=order, grid=grid)
        predicted = held_out_predictions(model, moved, reference, **model_kwargs)
        distortion = perturbation.distortion
        undistorted = reference
        if distortion is not None and perturbation.motions:
            still = perturb(
                dataset, attrs.evolve(perturbation, distortion=None), order=order, grid=grid
            )
            undistorted = held_out_predictions(model, still, reference, **model_kwargs)
        for index, prediction in predicted.items():
            inherited = float("nan")
            if distortion is not None and distortion.coefficients[index].any():
                own = distortion.coefficients[index]
                fitted = inherited_distortion(
                    undistorted[index], prediction, distortion, aligned.affine, mask
                )
                inherited = float(fitted @ own / (own @ own))
            rows.append(
                {
                    "magnitude": perturbation.magnitude,
                    "count": perturbation.count,
                    "seed": perturbation.seed,
                    "index": index,
                    "moved": index in perturbation.moved,
                    "distortion": 0.0 if distortion is None else distortion.magnitude,
                    "nrmse": nrmse(prediction, reference[index], mask, scale=scale[index]),
                    "inherited": inherited,
                    **{
                        name: metric(prediction, reference[index], mask)
                        for name, metric in metrics.items()
                    },
                }
            )
    return rows


def inherited_distortion(
    aligned: np.ndarray,
    perturbed: np.ndarray,
    distortion: Distortion,
    affine: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    r"""
    Distortion coefficients that best explain how a prediction changed.

    To first order, displacing an image :math:`p` by :math:`d` along the
    phase-encoding axis (with its Jacobian) changes it by
    :math:`\partial_{PE} p \, d + p \, \partial_{PE} d`, which is linear in the
    coefficients of :math:`d`. Least squares over the mask give the coefficients
    of the distortion that the change resembles; whatever the distortion cannot
    explain (noise) stays in the residual. A rigid motion does not: a
    translation along phase encoding is the distortion's constant term, so
    ``aligned`` must carry the same motions as ``perturbed``.

    Parameters
    ----------
    aligned : :obj:`~numpy.ndarray`
        The prediction without the distortion.
    perturbed : :obj:`~numpy.ndarray`
        The prediction with the distortion.
    distortion : :obj:`Distortion`
        The distortion, whose terms are fitted.
    affine : :obj:`~numpy.ndarray`
        Grid of the predictions.
    mask : :obj:`~numpy.ndarray`
        Voxels fitted.

    Returns
    -------
    :obj:`~numpy.ndarray`
        The fitted coefficients, comparable to a row of
        :attr:`Distortion.coefficients`.

    """
    aligned = np.asarray(aligned, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    ijk = np.argwhere(mask).T.astype(float)
    # Derivative of the cubic spline that resampling interpolates with
    step = np.zeros((3, 1))
    step[distortion.axis] = 1e-2
    ahead, behind = (map_coordinates(aligned, ijk + s, order=3) for s in (step, -step))
    spacing = np.linalg.norm(np.asarray(affine)[:3, distortion.axis])
    slope = (ahead - behind) / (2e-2 * spacing)
    points = nb.affines.apply_affine(affine, ijk.T).T
    values, derivatives = distortion.basis(points)
    design = (values * slope + derivatives * aligned[mask]).T
    change = (np.asarray(perturbed, dtype=float) - aligned)[mask]
    return np.linalg.lstsq(design, change, rcond=None)[0]


def influence(model: Any) -> np.ndarray:
    r"""
    Analytic influence of each volume on the predictions of a linear smoother.

    The closed-form models predict volume :math:`i` as :math:`\mathbf{S}_{i}
    \mathbf{Y}`, so moving volume :math:`j` by :math:`\Delta\mathbf{y}_j`
    changes every prediction by :math:`\mathbf{S}_{:, j} \Delta\mathbf{y}_j`.

    Parameters
    ----------
    model : :obj:`~nifreeze.model.lovo.ClosedFormLOVOMixin`
        A closed-form model.

    Returns
    -------
    :obj:`~numpy.ndarray`
        The norm of each column of the smoother, :math:`\|\mathbf{S}_{:, j}\|_2`.

    """
    return np.linalg.norm(model.smoother, axis=0)


def _monomials(degree: int) -> list[np.ndarray]:
    """Exponents of the monomials of three variables up to ``degree``, by degree."""
    return [
        np.array(exponents)
        for total in range(degree + 1)
        for exponents in sorted(
            ((i, j, total - i - j) for i in range(total + 1) for j in range(total + 1 - i)),
            reverse=True,
        )
    ]


def _resolve_mask(dataset: Any, mask: np.ndarray | None) -> np.ndarray:
    if mask is not None:
        return np.asarray(mask, dtype=bool)
    if getattr(dataset, "brainmask", None) is not None:
        return np.asarray(dataset.brainmask, dtype=bool)
    return np.ones(dataset.shape3d, dtype=bool)
