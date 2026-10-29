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
:class:`Perturbation`): a simulator produces realistic ones
(:mod:`nifreeze.analysis.phantoms` reads them), and
:mod:`nifreeze.testing.perturbations` resamples a dataset for tests. A fresh
model is built on each copy. A held-out prediction never uses its own volume, so
every volume can be evaluated, the moved ones included: a moved volume's
prediction sees the motion of the others only. Registration, the estimator and
its intensity window are not involved.

"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import attrs
import numpy as np
from scipy.ndimage import map_coordinates

from nifreeze.analysis.similarity import METRICS, nrmse


@attrs.define(frozen=True)
class Distortion:
    r"""
    Displacements along the phase-encoding axis, polynomial in space, one per volume.

    Volume ``i`` shows at each point what lies :math:`d_i` mm further along the
    phase-encoding axis. :math:`d_i` is a polynomial of the voxel-axis
    coordinates, centered on ``center`` and divided by ``radius`` along each axis.

    Without ``readout``, a distorted volume is resampled at the displaced points,
    with its intensity scaled by the Jacobian
    :math:`1 + \partial d_i / \partial x_{PE}` so that no signal is created or lost.
    With it, the volume is formed as an EPI readout forms it, which is how TRXScan
    simulates eddy currents: the displacement is that of an off-resonance field
    :math:`f_i = d_i / (T \, \delta)` (Hz), with :math:`T` the total readout time
    and :math:`\delta` the voxel size along phase encoding, and each
    phase-encoding line accrues the field's phase at the time it is acquired.
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
    readout: np.ndarray | None = attrs.field(default=None, eq=False)
    """Acquisition time (s, from the echo) of each phase-encoding line: the ``n``-th
    of ``N`` lines has the spatial frequency ``N // 2 - n``. :obj:`None` to resample."""
    readout_time: float | None = None
    """Total readout time (s), which turns displacements into fields."""

    @classmethod
    def from_sidecar(
        cls, eddy: Mapping[str, Any], indices: Sequence[int], n_volumes: int
    ) -> Distortion:
        """
        The eddy currents of a perturbed copy, from its sidecar.

        Parameters
        ----------
        eddy : :obj:`~collections.abc.Mapping`
            The ``Eddy`` entry of the sidecar written by ``tools/phantoms/phantom.py``.
        indices : :obj:`~collections.abc.Sequence`
            The dataset's index of each volume whose coefficients the sidecar lists.
        n_volumes : :obj:`int`
            Number of volumes of the dataset; the others are not distorted.

        Returns
        -------
        :obj:`Distortion`
            The distortion, formed as an EPI readout forms it.

        """
        terms = eddy["Distortion"]
        listed = np.asarray(terms["Coefficients"], dtype=float)
        if len(listed) != len(indices):
            raise ValueError(
                f"The sidecar lists {len(listed)} volumes, but {len(indices)} indices were given."
            )
        coefficients = np.zeros((n_volumes, listed.shape[1]))
        coefficients[list(indices)] = listed
        return cls(
            magnitude=float(eddy["MagnitudeMm"]),
            axis=int(terms["Axis"]),
            seed=None,
            degree=int(terms["Degree"]),
            center=np.asarray(terms["Center"], dtype=float),
            radius=np.asarray(terms["Radius"], dtype=float),
            frame=np.asarray(terms["Frame"], dtype=float),
            coefficients=coefficients,
            readout=np.asarray(eddy["ReadoutTimes"], dtype=float),
            readout_time=float(eddy["TotalReadoutTime"]),
        )

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

    def distort(
        self,
        volume: np.ndarray,
        affine: np.ndarray,
        coefficients: np.ndarray,
        order: int = 3,
    ) -> np.ndarray:
        """
        Distort a volume by the displacement that some coefficients give.

        Parameters
        ----------
        volume : :obj:`~numpy.ndarray`
            The undistorted volume.
        affine : :obj:`~numpy.ndarray`
            Its grid, whose voxel axis :attr:`axis` is the phase-encoding axis.
        coefficients : :obj:`~numpy.ndarray`
            The displacement (mm) carried by each term, as a row of
            :attr:`coefficients`.
        order : :obj:`int`, optional
            Interpolation order, when the volume is resampled.

        Returns
        -------
        :obj:`~numpy.ndarray`
            The distorted volume, on the same grid.

        """
        volume = np.asarray(volume, dtype=float)
        coefficients = np.asarray(coefficients, dtype=float)
        affine = np.asarray(affine, dtype=float)
        if not coefficients.any():
            # No distortion at all: exactly the volume, without interpolation error
            return volume.copy()
        ijk = np.indices(volume.shape, dtype=float).reshape(3, -1)
        values, derivatives = self.basis(affine[:3, :3] @ ijk + affine[:3, 3:])
        displacement = coefficients @ values
        if self.readout is None:
            return self._resample(
                volume, affine, ijk, derivatives, coefficients, displacement, order
            )
        return self._readout(volume, affine, displacement)

    def _resample(
        self,
        volume: np.ndarray,
        affine: np.ndarray,
        ijk: np.ndarray,
        derivatives: np.ndarray,
        coefficients: np.ndarray,
        displacement: np.ndarray,
        order: int,
    ) -> np.ndarray:
        """Resample at the displaced points, scaled by the Jacobian."""
        steps = np.linalg.inv(affine[:3, :3]) @ self.direction
        coordinates = ijk + np.outer(steps, displacement)
        resampled = map_coordinates(volume, coordinates, order=order)  # type: ignore[call-overload]
        jacobian = 1.0 + coefficients @ derivatives
        return (resampled * jacobian).reshape(volume.shape)

    def _readout(
        self,
        volume: np.ndarray,
        affine: np.ndarray,
        displacement: np.ndarray,
    ) -> np.ndarray:
        """Form the volume as an EPI readout does, line by line of k-space."""
        if self.readout is None or self.readout_time is None:
            raise ValueError("An EPI readout needs its line times and total readout time.")
        n_lines = volume.shape[self.axis]
        if len(self.readout) != n_lines:
            raise ValueError(f"The readout has {len(self.readout)} lines, the grid {n_lines}.")
        hertz = 1.0 / (self.readout_time * np.linalg.norm(affine[:3, self.axis]))

        def columns(image: np.ndarray) -> np.ndarray:
            """The image as columns along phase encoding, ``(..., n_lines)``."""
            return np.moveaxis(image.reshape(volume.shape), self.axis, -1).reshape(-1, n_lines)

        signal, field = columns(volume), columns(displacement * hertz)
        frequencies = n_lines // 2 - np.arange(n_lines)
        forward = np.exp(-2j * np.pi * np.outer(frequencies, np.arange(n_lines)) / n_lines)
        inverse = np.exp(2j * np.pi * np.outer(np.arange(n_lines), frequencies) / n_lines)
        inverse /= n_lines
        times = np.asarray(self.readout, dtype=float)

        image = np.zeros((len(signal), n_lines))
        # Columns without signal stay empty; the others go in batches of bounded memory
        active = np.flatnonzero(signal.any(axis=1))
        batch = max(1, 2**22 // n_lines**2)
        for start in range(0, len(active), batch):
            rows = active[start : start + batch]
            encoding = forward * np.exp(2j * np.pi * times[:, np.newaxis] * field[rows, None, :])
            kspace = (encoding @ signal[rows, :, None])[..., 0]
            image[rows] = np.abs(kspace @ inverse.T)

        shape = np.moveaxis(np.empty(volume.shape), self.axis, -1).shape
        return np.moveaxis(image.reshape(shape), -1, self.axis)


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
        Pairs of a :class:`Perturbation` and the perturbed copy of ``reference``,
        or a function without arguments that loads it (each copy is then held in
        memory only while it is analyzed).
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
        One row per cell and evaluated volume, with the volume's ``shell`` (its
        b-value to the nearest 100 s/mm², NaN without gradients), the ``nrmse``,
        each further metric, whether that volume was itself ``moved``, and the
        magnitude of the ``distortion`` (0 without one).

    """
    cells = list(cells)
    metrics = {} if metrics is None else metrics
    mask = _resolve_mask(reference, mask)
    aligned = held_out_predictions(model, reference, evaluate, **model_kwargs)
    scale = {
        index: np.asarray(reference.dataobj[index], dtype=float)[mask].std() for index in aligned
    }
    gradients = getattr(reference, "gradients", None)
    shells = {
        index: float("nan") if gradients is None else float(np.round(gradients[index, -1], -2))
        for index in aligned
    }

    rows = []
    for perturbation, perturbed in cells:
        if callable(perturbed):
            perturbed = perturbed()
        predicted = held_out_predictions(model, perturbed, aligned, **model_kwargs)
        distortion = perturbation.distortion
        for index, prediction in predicted.items():
            rows.append(
                {
                    "magnitude": perturbation.magnitude,
                    "count": perturbation.count,
                    "seed": perturbation.seed,
                    "index": index,
                    "shell": shells[index],
                    "moved": index in perturbation.moved,
                    "distortion": 0.0 if distortion is None else distortion.magnitude,
                    "nrmse": nrmse(prediction, aligned[index], mask, scale=scale[index]),
                    **{
                        name: metric(prediction, aligned[index], mask)
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
