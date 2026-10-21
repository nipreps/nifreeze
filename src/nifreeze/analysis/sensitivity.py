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
  dataset when some volumes move by known rigid motions (a *perturbation*)?

A fresh model is built on each perturbed copy. A held-out prediction never uses
its own volume, so every volume can be evaluated, the moved ones included: a
moved volume's prediction sees the motion of the others only. Registration, the
estimator and its intensity window are not involved.

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
from scipy.spatial.transform import Rotation

from nifreeze.analysis.similarity import METRICS, nrmse


@attrs.define(frozen=True)
class Perturbation:
    """Rigid motions applied to some volumes of a dataset."""

    magnitude: float
    """Rotation (degrees) and translation (mm) of every moved volume."""
    seed: int | None
    """Seed that drew the moved volumes and their motions."""
    moved: tuple[int, ...] = ()
    """Indices of the moved volumes."""
    motions: tuple[np.ndarray, ...] = attrs.field(default=(), eq=False)
    """Head motion of each moved volume, as a ``(4, 4)`` matrix in world coordinates."""

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
    Copy a dataset with some of its volumes moved.

    Each moved volume is resampled once from the aligned original. Everything
    else (gradients, b=0 reference, mask) is left as it was, as a model would
    see it when the motion goes uncorrected.

    Parameters
    ----------
    dataset : :obj:`~nifreeze.data.base.BaseDataset`
        The aligned dataset.
    perturbation : :obj:`Perturbation`
        The volumes to move and their motions.
    order : :obj:`int`, optional
        Interpolation order.
    round_trip : :obj:`bool`, optional
        Move each volume and move it back, which leaves it in place but
        interpolated twice.
    grid : :obj:`~nifreeze.data.base.BaseDataset`, optional
        The dataset on a shifted grid (see :func:`shift_grid`). Moved volumes are
        then resampled from the original onto that grid, through their motion,
        and everything else is taken from ``grid``.

    Returns
    -------
    :obj:`~nifreeze.data.base.BaseDataset`
        A copy of the dataset (or of ``grid``) with ``float32`` volumes.

    """
    if round_trip and grid is not None:
        raise ValueError("A round trip cannot be combined with a shifted grid.")
    source = dataset if grid is None else grid
    reference = nb.Nifti1Image(np.zeros(dataset.shape3d, dtype=np.uint8), source.affine)

    dataobj = np.array(source.dataobj, dtype=np.float32)
    for index, motion in zip(perturbation.moved, perturbation.motions, strict=True):
        image = nb.Nifti1Image(
            np.asarray(dataset.dataobj[index], dtype=np.float32), dataset.affine
        )
        # The moved head shows, at each position, what was at motion⁻¹(position)
        steps = (np.linalg.inv(motion), motion) if round_trip else (np.linalg.inv(motion),)
        for step in steps:
            image = apply(Affine(step), image, reference=reference, order=order)
        dataobj[index] = np.asanyarray(image.dataobj)

    moved = copy(source)
    moved.dataobj = dataobj
    return moved


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
        further metric, and whether that volume was itself ``moved``.

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
        for index, prediction in predicted.items():
            rows.append(
                {
                    "magnitude": perturbation.magnitude,
                    "count": perturbation.count,
                    "seed": perturbation.seed,
                    "index": index,
                    "moved": index in perturbation.moved,
                    "nrmse": nrmse(prediction, reference[index], mask, scale=scale[index]),
                    **{
                        name: metric(prediction, reference[index], mask)
                        for name, metric in metrics.items()
                    },
                }
            )
    return rows


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


def _resolve_mask(dataset: Any, mask: np.ndarray | None) -> np.ndarray:
    if mask is not None:
        return np.asarray(mask, dtype=bool)
    if getattr(dataset, "brainmask", None) is not None:
        return np.asarray(dataset.brainmask, dtype=bool)
    return np.ones(dataset.shape3d, dtype=bool)
