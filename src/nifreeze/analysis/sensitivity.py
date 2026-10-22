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

The perturbed copies are given, each with its ground truth (a
:class:`Perturbation`): a simulator produces realistic ones, and
:mod:`nifreeze.testing.perturbations` resamples a dataset for tests. A fresh
model is built on each copy. A held-out prediction never uses its own volume, so
every volume can be evaluated, the moved ones included: a moved volume's
prediction sees the motion of the others only. Eddy-current distortions follow
the diffusion gradient, so a prediction may still resemble its volume's own
distortion through volumes with similar gradients: :func:`inherited_distortion`
measures how much. Registration, the estimator and its intensity window are not
involved.

"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

import attrs
import nibabel as nb
import numpy as np
from scipy.ndimage import map_coordinates

from nifreeze.analysis.similarity import METRICS, nrmse


@attrs.define(frozen=True)
class Distortion:
    r"""
    Displacements along the phase-encoding axis, polynomial in space, one per volume.

    Volume ``i`` shows at each point what lies :math:`d_i` mm further along the
    phase-encoding axis, with its intensity scaled by the Jacobian
    :math:`1 + \partial d_i / \partial x_{PE}` so that no signal is created or lost.
    :math:`d_i` is a polynomial of the voxel-axis coordinates, centered on
    ``center`` and divided by ``radius`` along each axis.
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
    """Scale (mm) of the polynomial's variables along each voxel axis."""
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
    """
    The ground truth of a perturbed dataset.

    Some volumes moved by rigid motions, and optionally every volume distorted.
    """

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
"""The ground truth of the aligned dataset."""


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
    reference: Any,
    cells: Iterable[tuple[Perturbation, Any]],
    evaluate: Iterable[int] | None = None,
    mask: np.ndarray | None = None,
    metrics: Mapping[str, Callable[..., float]] | None = None,
    **model_kwargs: Any,
) -> list[dict[str, Any]]:
    """
    Measure how far held-out predictions drift from those on a reference dataset.

    Each cell pairs a perturbed copy of ``reference`` (same grid, same volumes,
    some moved or distorted) with the :class:`Perturbation` that made it. Each
    prediction on a copy is compared with the prediction of the same volume on
    the reference. The change is reported as an NRMSE divided by the standard
    deviation of the actual volume, not by that of the reference prediction: a
    smooth prediction would otherwise inflate its own error, and models could not
    be compared.

    Parameters
    ----------
    model : :obj:`str` or :obj:`~collections.abc.Callable`
        The model (see :func:`held_out_predictions`).
    reference : :obj:`~nifreeze.data.base.BaseDataset`
        The aligned dataset.
    cells : :obj:`~collections.abc.Iterable`
        Pairs of a :class:`Perturbation` and the perturbed copy of ``reference``.
        A copy with both motions and a distortion needs the copy with the same
        motions alone among the cells: its inherited share is measured on top of
        them.
    evaluate : :obj:`~collections.abc.Iterable`, optional
        Volumes to predict (all if :obj:`None`), moved or not.
    mask : :obj:`~numpy.ndarray`, optional
        Voxels compared; the reference's brain mask, or the whole volume, if
        :obj:`None`.
    metrics : :obj:`~collections.abc.Mapping`, optional
        Further metrics by name, with the signature of
        :func:`~nifreeze.analysis.similarity.masked_cc`, called as
        ``metric(perturbed, aligned, mask)``.
    **model_kwargs
        Arguments for the model.

    Returns
    -------
    :obj:`list` of :obj:`dict`
        One row per cell and evaluated volume, with the ``nrmse``, each further
        metric, whether that volume was itself ``moved``, the magnitude of the
        ``distortion`` (0 without one) and the share of the volume's own
        distortion that its prediction ``inherited`` (the projection of the
        coefficients fitted by :func:`inherited_distortion` onto the volume's
        own; NaN without a distortion, or for an undistorted volume). With
        motions too, the share is fitted to the change from the prediction with
        the same motions and no distortion, since a motion alone also projects
        onto the distortion's terms.

    """
    cells = list(cells)
    metrics = {} if metrics is None else metrics
    mask = _resolve_mask(reference, mask)
    aligned = held_out_predictions(model, reference, evaluate, **model_kwargs)
    scale = {
        index: np.asarray(reference.dataobj[index], dtype=float)[mask].std() for index in aligned
    }

    still = {_motions_of(p): data for p, data in cells if p.distortion is None and p.motions}
    needed = {_motions_of(p) for p, _ in cells if p.distortion is not None and p.motions}
    if missing := needed - still.keys():
        raise ValueError(
            f"{len(missing)} cell(s) with motions and a distortion lack the cell with the same "
            "motions alone, on top of which their inherited share is measured."
        )
    kept: dict[tuple, dict[int, np.ndarray]] = {}

    def predict(perturbation: Perturbation, perturbed: Any) -> dict[int, np.ndarray]:
        key = _motions_of(perturbation) if perturbation.distortion is None else None
        if key in kept:
            return kept[key]
        predicted = held_out_predictions(model, perturbed, aligned, **model_kwargs)
        if key in needed:
            kept[key] = predicted
        return predicted

    rows = []
    for perturbation, perturbed in cells:
        predicted = predict(perturbation, perturbed)
        distortion = perturbation.distortion
        undistorted = aligned
        if distortion is not None and perturbation.motions:
            key = _motions_of(perturbation)
            undistorted = predict(attrs.evolve(perturbation, distortion=None), still[key])
        for index, prediction in predicted.items():
            inherited = float("nan")
            if distortion is not None and distortion.coefficients[index].any():
                own = distortion.coefficients[index]
                fitted = inherited_distortion(
                    undistorted[index], prediction, distortion, reference.affine, mask
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
                    "nrmse": nrmse(prediction, aligned[index], mask, scale=scale[index]),
                    "inherited": inherited,
                    **{
                        name: metric(prediction, aligned[index], mask)
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


def _motions_of(perturbation: Perturbation) -> tuple:
    """Identify the motions of a perturbation, whatever its distortion."""
    return perturbation.moved, tuple(np.asarray(m).tobytes() for m in perturbation.motions)


def _resolve_mask(dataset: Any, mask: np.ndarray | None) -> np.ndarray:
    if mask is not None:
        return np.asarray(mask, dtype=bool)
    if getattr(dataset, "brainmask", None) is not None:
        return np.asarray(dataset.brainmask, dtype=bool)
    return np.ones(dataset.shape3d, dtype=bool)
