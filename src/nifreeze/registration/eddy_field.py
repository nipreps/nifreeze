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
r"""
A global, physics-based model of the eddy-current fields of a DWI (nipreps/nifreeze#745).

A field of eddy currents linear in position moves points only along the
phase-encoding direction :math:`u`: :math:`E = I - u s^T`, with three slopes :math:`s`
(see :func:`~nifreeze.registration.utils.split_phase_encoding`). The field comes from
the diffusion gradient, so its slopes follow it. In volume :math:`v`, with the unit
b-vector :math:`g_v` and the b-value :math:`b_v`,

.. math::

    s_v = \left(\frac{b_v}{b_\text{ref}}\right)^p C g_v,

where the matrix :math:`C` and the exponent :math:`p` are the same for every volume
(:class:`EddyField`). The exponent is 1/2 when the field follows the gradient's
strength at a fixed timing, and 1 when it follows the b-value, as in the phantoms that
TRXScan makes (PennLINC/TRXScan#27). Alternatively, each segment of b-values has its own
matrix, with the exponent held (:class:`SegmentedEddyField`). The model can also be
fitted on the volumes of a range of b-values only, and give every volume its field
(``fit_range`` of :class:`EddyFieldFit`). A stretch that every
volume shares has no parameter to go into, as long as the b-vectors average out; fitted
on images, where volumes weigh a little differently in the metric, a small part of it
still reaches the fields.

The fields are zero at one point, by default the centre of the field of view
(:func:`field_of_view_centre`). Only the model is estimated: each volume's slopes follow
from it, and so does its shift along phase encoding, the field's displacement away from
that point. The shift is head motion by the convention of
:func:`~nifreeze.registration.utils.split_phase_encoding`; each volume keeps the rest of
the head motion it is given.

Estimating the model is one registration of all the volumes at once, with one set of
parameters, in which each volume moves by its own field. ITK's transforms map a point
without knowing which volume it belongs to, so :class:`EddyFieldFit` has ITK's metric
evaluate each volume's similarity and its derivative, and scipy estimates the
parameters.

FSL eddy's second-level model (``--slm linear``) is the antecedent: it replaces each
volume's estimated field coefficients by their least-squares fit on
:math:`(b_v / 1000)\, g_v`, which is this model with :math:`p = 1`, fitted to the
estimates of the volumes registered one by one (:meth:`EddyField.from_slopes` does the
same, with the exponent free). Here, the model is fitted on the images; the zero point is
given, whereas eddy regresses the field's offset on the gradients; and each segment of
b-values may have its own matrix.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from os import PathLike
from typing import Any
from warnings import warn

import attrs
import numpy as np

from nifreeze.data.dmri.utils import format_phase_encoding
from nifreeze.data.filtering import ClippingValueError, robust_window
from nifreeze.registration.base import CLIPPING
from nifreeze.registration.itk_backend import (
    _frame,
    _matched,
    _prepare,
    _threads,
    _voxel_image,
    get_settings,
)
from nifreeze.registration.utils import phase_encoding_vector, split_phase_encoding

DEFAULT_EXPONENT = 0.5
"""The exponent of a field that follows the gradient's strength at a fixed timing."""
EXPONENT_BOUNDS = (0.0, 2.0)
"""The range the exponent is estimated in."""
TABLE_ERROR_MSG = "The gradient table must be (N, 4): a b-vector, then the b-value."
"""Malformed gradient table error message."""
BVALUE_ERROR_MSG = "Every b-value must be positive: the field of a b=0 volume is not modelled."
"""A b-value that is not positive error message."""
SPAN_ERROR_MSG = "The b-vectors do not span three dimensions, so the matrix C is not determined."
"""B-vectors in a plane error message."""
SLOPES_ERROR_MSG = "The slopes must be (N, 3), one row per volume of the gradient table."
"""Malformed slopes error message."""
MATRIX_ERROR_MSG = "The matrix C must be 3 x 3."
"""Malformed matrix error message."""
B_REF_ERROR_MSG = "The reference b-value must be positive."
"""A reference b-value that is not positive error message."""
ONE_BVALUE_WARN_MSG = (
    "Every volume has the same b-value, so the exponent is not determined: "
    "it is held at {exponent:g}."
)
"""A single b-value warning message."""
EDGES_ERROR_MSG = "The b-values that split the segments must be positive and increasing."
"""Malformed segment edges error message."""
PARTS_ERROR_MSG = "A segmented field needs one field for each segment: {parts} for {edges} edges."
"""A segmented field with the wrong number of fields error message."""
EMPTY_SEGMENT_ERROR_MSG = "No volume has a b-value in segment {segment} of the edges {edges}."
"""A segment without volumes error message."""
EDGES_B_REF_ERROR_MSG = (
    "b_ref applies to one model for all volumes: with edges, each segment's reference "
    "is the geometric mean of its b-values."
)
"""A reference b-value given with segments error message."""
FIT_RANGE_ERROR_MSG = "fit_range must be two b-values, low then high, with 0 <= low < high."
"""Malformed range of b-values error message."""
FIT_RANGE_EDGES_ERROR_MSG = (
    "fit_range fits one model on some of the volumes, and edges one model for each "
    "segment: give one of them."
)
"""A range of b-values given with segments error message."""
EMPTY_RANGE_ERROR_MSG = "No volume added has a b-value in the fit range [{low:g}, {high:g})."
"""A range of b-values without volumes error message."""
NO_PHASE_ENCODING_ERROR_MSG = (
    "The eddy-current field moves points along phase encoding, whose direction is unknown: "
    "give the fit phase_encoding (the dataset's, as BIDS writes it)."
)
"""Unknown phase-encoding direction error message."""
GRID_ERROR_MSG = "Volume {index} is not on the grid and mask of the volumes added before it."
"""A volume on another grid error message."""
NOTHING_ADDED_ERROR_MSG = "No volume was added to the fit."
"""A fit without volumes error message."""
NOT_FITTED_ERROR_MSG = "The field is not fitted yet: call fit() first."
"""Mappings before the fit error message."""
TABLE_LENGTH_ERROR_MSG = "The gradient table has {rows} rows, but volume {index} was added."
"""A gradient table without a row for every volume error message."""


def _table(gradients: Any) -> np.ndarray:
    """A gradient table, checked: ``(N, 4)``, with positive b-values."""
    table = np.asarray(gradients, dtype=float)
    if table.ndim != 2 or table.shape[1] != 4:
        raise ValueError(TABLE_ERROR_MSG)
    if np.any(table[:, 3] <= 0):
        raise ValueError(BVALUE_ERROR_MSG)
    return table


def _unit(direction: Any) -> np.ndarray:
    vector = np.asarray(direction, dtype=float)
    return vector / np.linalg.norm(vector)


def _check_span(table: np.ndarray) -> None:
    if np.linalg.matrix_rank(table[:, :3]) < 3:
        raise ValueError(SPAN_ERROR_MSG)


def _held_exponent(table: np.ndarray, exponent: float | None, stacklevel: int) -> float | None:
    """
    The exponent to hold: the one given, or :data:`DEFAULT_EXPONENT` when a single
    b-value leaves it undetermined; :obj:`None` to estimate it.
    """
    if exponent is not None:
        return float(exponent)
    if np.ptp(np.log(table[:, 3])) < 1e-9:
        warn(ONE_BVALUE_WARN_MSG.format(exponent=DEFAULT_EXPONENT), stacklevel=stacklevel)
        return DEFAULT_EXPONENT
    return None


def _from_slopes(slopes: np.ndarray, direction: Any) -> np.ndarray:
    """The fields :math:`E = I - u s^T` of slopes ``(N, 3)``, ``(N, 4, 4)``."""
    fields = np.repeat(np.eye(4)[np.newaxis], len(slopes), axis=0)
    fields[:, :3, :3] -= _unit(direction)[np.newaxis, :, np.newaxis] * slopes[:, np.newaxis]
    return fields


def _as_edges(value: Any) -> tuple[float, ...]:
    edges = tuple(float(edge) for edge in np.atleast_1d(np.asarray(value, dtype=float)))
    if not edges or edges[0] <= 0 or np.any(np.diff(edges) <= 0):
        raise ValueError(EDGES_ERROR_MSG)
    return edges


def _as_range(value: Any) -> tuple[float, float]:
    bounds = np.asarray(value, dtype=float)
    if bounds.shape != (2,) or not 0 <= bounds[0] < bounds[1]:
        raise ValueError(FIT_RANGE_ERROR_MSG)
    return float(bounds[0]), float(bounds[1])


def _segment_of(edges: Sequence[float], bvals: Any) -> np.ndarray:
    """The segment of each b-value: ``k`` when it lies in ``[edges[k - 1], edges[k])``."""
    return np.searchsorted(np.asarray(edges, dtype=float), np.asarray(bvals, dtype=float), "right")


def field_of_view_centre(affine: Any, shape: Sequence[int]) -> np.ndarray:
    """
    The centre of a grid's field of view: the middle of its physical extent (RAS+, mm).

    Samples lie at the centres of the voxels, so the extent runs from index -0.5 to
    ``n - 0.5`` along each axis, and its middle is index ``(n - 1) / 2``, whether ``n``
    is odd or even.

    >>> field_of_view_centre(np.diag([2.0, 2.0, 2.0, 1.0]), (4, 5, 6)).tolist()
    [3.0, 4.0, 5.0]

    """
    middle = (np.asarray(shape[:3], dtype=float) - 1) / 2
    return np.asarray(affine, dtype=float)[:3] @ np.r_[middle, 1.0]


def geometric_mean(bvals: Any) -> float:
    """
    The geometric mean of the b-values, the default reference.

    >>> round(geometric_mean([1000.0, 4000.0]), 6)
    2000.0

    """
    return float(np.exp(np.mean(np.log(np.asarray(bvals, dtype=float)))))


def field_slopes(fields: Any, direction: Any) -> np.ndarray:
    """
    The slopes :math:`s` of fields :math:`E = I - u s^T` along ``direction``.

    For a mapping outside that family, they are those of its row along :math:`u`.

    Parameters
    ----------
    fields : :obj:`~numpy.ndarray`
        The fields, ``(N, 4, 4)`` RAS+ matrices (:attr:`~nifreeze.data.dmri.base.DWI.eddy`).
    direction : :obj:`~numpy.ndarray`
        The phase-encoding direction :math:`u` in world coordinates.

    Returns
    -------
    :obj:`~numpy.ndarray`
        The slopes, ``(N, 3)``.

    Examples
    --------
    >>> field = np.eye(4)
    >>> field[1, :3] = [0.02, 0.97, -0.01]
    >>> field_slopes(field[np.newaxis], [0.0, 1.0, 0.0]).round(6).tolist()
    [[-0.02, 0.03, 0.01]]

    """
    fields = np.asarray(fields, dtype=float)
    return np.einsum("i,nij->nj", _unit(direction), np.eye(3) - fields[:, :3, :3])


def _as_matrix(value: Any) -> np.ndarray:
    matrix = np.array(value, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError(MATRIX_ERROR_MSG)
    matrix.flags.writeable = False
    return matrix


@attrs.frozen
class EddyField:
    r"""
    The eddy-current field of every volume, from its b-vector and b-value.

    In volume :math:`v`, the field's slopes are
    :math:`s_v = (b_v / b_\text{ref})^p C g_v`, and the field is
    :math:`E_v = I - u s_v^T`.

    Examples
    --------
    >>> field = EddyField(0.02 * np.eye(3), exponent=0.5, b_ref=1000.0)
    >>> gradients = [[1.0, 0.0, 0.0, 1000.0], [0.0, -1.0, 0.0, 4000.0]]
    >>> field.slopes(gradients).round(6).tolist()
    [[0.02, 0.0, 0.0], [0.0, -0.04, 0.0]]
    >>> field.fields(gradients, [0.0, 1.0, 0.0])[1, :3, :3].round(6).tolist()
    [[1.0, 0.0, 0.0], [0.0, 1.04, 0.0], [0.0, 0.0, 1.0]]

    """

    matrix: np.ndarray = attrs.field(
        converter=_as_matrix, eq=attrs.cmp_using(eq=np.array_equal), hash=False
    )
    """:math:`C`: the slopes of a unit b-vector at the reference b-value."""
    exponent: float = attrs.field(default=DEFAULT_EXPONENT, converter=float)
    """:math:`p`: the slopes scale with the b-value to this power."""
    b_ref: float = attrs.field(default=1000.0, converter=float)
    """The reference b-value :math:`b_\\text{ref}` (s/mm²)."""

    @b_ref.validator
    def _check_b_ref(self, attribute: Any, value: float) -> None:
        if not value > 0:
            raise ValueError(B_REF_ERROR_MSG)

    def scales(self, bvals: Any) -> np.ndarray:
        """How much each b-value scales the slopes: :math:`(b / b_\\text{ref})^p`."""
        return (np.asarray(bvals, dtype=float) / self.b_ref) ** self.exponent

    def rescaled(self, b_ref: float) -> EddyField:
        """The same field, with :math:`C` given at another reference b-value."""
        return EddyField(self.matrix * float(self.scales(b_ref)), self.exponent, b_ref)

    def slopes(self, gradients: Any) -> np.ndarray:
        """
        The slopes of each volume's field, ``(N, 3)``.

        Parameters
        ----------
        gradients : :obj:`~numpy.ndarray`
            The gradient table, ``(N, 4)``: a unit b-vector in world RAS+ coordinates, as
            the scanner applied it (eddy currents do not turn with the head), then the
            b-value.

        """
        table = _table(gradients)
        return self.scales(table[:, 3])[:, np.newaxis] * (table[:, :3] @ self.matrix.T)

    def fields(self, gradients: Any, direction: Any) -> np.ndarray:
        """
        Each volume's field :math:`E_v = I - u s_v^T`, ``(N, 4, 4)`` RAS+ matrices.

        This is how :func:`~nifreeze.registration.utils.split_phase_encoding` gives a
        field: without a translation, which belongs to the head motion.
        """
        return _from_slopes(self.slopes(gradients), direction)

    @classmethod
    def from_slopes(
        cls,
        slopes: Any,
        gradients: Any,
        weights: Any = None,
        exponent: float | None = None,
        b_ref: float | None = None,
    ) -> EddyField:
        """
        Fit :math:`C` and :math:`p` to each volume's slopes, by least squares.

        For a fixed exponent, :math:`C` is linear least squares; the exponent is the
        one, within :data:`EXPONENT_BOUNDS`, that leaves the least residual. With the
        exponent held at 1, this is FSL eddy's linear second-level model.

        Parameters
        ----------
        slopes : :obj:`~numpy.ndarray`
            The slopes of each volume's field, ``(N, 3)`` (see :func:`field_slopes`).
        gradients : :obj:`~numpy.ndarray`
            The gradient table, ``(N, 4)`` (see :meth:`slopes`).
        weights : :obj:`~numpy.ndarray`, optional
            A weight for each volume.
        exponent : :obj:`float`, optional
            Hold the exponent at this value. With a single b-value, it is held at
            :data:`DEFAULT_EXPONENT`, with a warning.
        b_ref : :obj:`float`, optional
            The reference b-value (default: the b-values' :func:`geometric_mean`).

        Raises
        ------
        ValueError
            If the shapes do not match, a b-value is not positive, or the b-vectors do
            not span three dimensions.

        Examples
        --------
        >>> rng = np.random.default_rng(0)
        >>> b_vectors = rng.normal(size=(30, 3))
        >>> b_vectors /= np.linalg.norm(b_vectors, axis=1, keepdims=True)
        >>> gradients = np.column_stack([b_vectors, np.repeat([1000.0, 2000.0, 3000.0], 10)])
        >>> truth = EddyField(rng.normal(scale=0.02, size=(3, 3)), exponent=1.0, b_ref=1000.0)
        >>> found = EddyField.from_slopes(truth.slopes(gradients), gradients, b_ref=1000.0)
        >>> round(found.exponent, 6), np.allclose(found.matrix, truth.matrix)
        (1.0, True)

        """
        from scipy.optimize import minimize_scalar

        table = _table(gradients)
        slopes = np.asarray(slopes, dtype=float)
        if slopes.shape != (len(table), 3):
            raise ValueError(SLOPES_ERROR_MSG)
        _check_span(table)
        w = np.ones(len(table)) if weights is None else np.asarray(weights, dtype=float)
        b_ref = geometric_mean(table[:, 3]) if b_ref is None else float(b_ref)
        log_b = np.log(table[:, 3] / b_ref)

        def solve(p: float) -> tuple[np.ndarray, float]:
            scaled = np.exp(p * log_b)[:, np.newaxis] * table[:, :3]
            # C (sum w q q^T) = sum w s q^T, with q the scaled b-vector
            matrix = np.linalg.solve(
                (w[:, None] * scaled).T @ scaled, scaled.T @ (w[:, None] * slopes)
            ).T
            residual = slopes - scaled @ matrix.T
            return matrix, float(np.sum(w * np.sum(residual**2, axis=1)))

        held = _held_exponent(table, exponent, stacklevel=3)
        if held is None:
            found = minimize_scalar(
                lambda p: solve(p)[1],
                bounds=EXPONENT_BOUNDS,
                method="bounded",
                options={"xatol": 1e-12},
            )
            held = float(found.x)
        return cls(solve(held)[0], held, b_ref)


def _as_parts(value: Any) -> tuple[EddyField, ...]:
    return tuple(value)


@attrs.frozen
class SegmentedEddyField:
    r"""
    The eddy-current field of every volume, with a matrix :math:`C` for each segment of
    b-values.

    ``edges`` split the b-values into segments: segment :math:`k` holds the b-values from
    ``edges[k - 1]`` up to, not including, ``edges[k]`` (the first, those below
    ``edges[0]``; the last, those from ``edges[-1]`` on). Each segment has its own
    :class:`EddyField`, so the slopes need not follow one power of the b-value: within a
    single shell, its matrix takes up the shell's scale.

    Examples
    --------
    >>> field = SegmentedEddyField(
    ...     [1500.0],
    ...     [EddyField(0.02 * np.eye(3), b_ref=1000.0), EddyField(0.05 * np.eye(3), b_ref=2000.0)],
    ... )
    >>> gradients = [[1.0, 0.0, 0.0, 1000.0], [1.0, 0.0, 0.0, 2000.0]]
    >>> field.slopes(gradients).round(6).tolist()
    [[0.02, 0.0, 0.0], [0.05, 0.0, 0.0]]
    >>> field.segments([1000.0, 1500.0, 3000.0]).tolist()
    [0, 1, 1]

    """

    edges: tuple[float, ...] = attrs.field(converter=_as_edges)
    """The b-values that split the segments, increasing."""
    parts: tuple[EddyField, ...] = attrs.field(converter=_as_parts)
    """The field of each segment, one more than ``edges``."""

    @parts.validator
    def _check_parts(self, attribute: Any, value: tuple[EddyField, ...]) -> None:
        if len(value) != len(self.edges) + 1:
            raise ValueError(PARTS_ERROR_MSG.format(parts=len(value), edges=len(self.edges)))

    def segments(self, bvals: Any) -> np.ndarray:
        """The segment of each b-value."""
        return _segment_of(self.edges, bvals)

    def slopes(self, gradients: Any) -> np.ndarray:
        """The slopes of each volume's field, ``(N, 3)`` (see :meth:`EddyField.slopes`)."""
        table = _table(gradients)
        segments = self.segments(table[:, 3])
        slopes = np.zeros((len(table), 3))
        for k, part in enumerate(self.parts):
            chosen = segments == k
            if np.any(chosen):
                slopes[chosen] = part.slopes(table[chosen])
        return slopes

    def fields(self, gradients: Any, direction: Any) -> np.ndarray:
        """Each volume's field, ``(N, 4, 4)`` RAS+ matrices (see :meth:`EddyField.fields`)."""
        return _from_slopes(self.slopes(gradients), direction)


class _Problem:
    r"""
    The cost of a stage and its derivative, as a function of the estimated parameters.

    The parameters are :math:`C` times the mask's radius (so that a unit moves its edge
    about 1 mm), then the exponent. Each volume's mapping is computed in the voxel frame
    of the images (see :func:`~nifreeze.registration.itk_backend._frame`), where phase
    encoding runs along the data axis ``axis``: :math:`W_v = E_v M_v`, with
    :math:`E_v(y) = y - e\, r_v^T (y - z)`, the voxel-frame slopes :math:`r_v` and the
    fields' zero point :math:`z`. ``evaluators`` give, for one volume's matrix and
    translation, the metric's value and its derivative with respect to both.
    """

    def __init__(
        self,
        table: np.ndarray,
        b_ref: float,
        motions: np.ndarray,
        zero: np.ndarray,
        radius: float,
        axis: int,
        sign: float,
        rotation: np.ndarray,
        evaluators: Sequence[Callable[[np.ndarray, np.ndarray], tuple]],
    ) -> None:
        self.bvecs = table[:, :3]
        self.log_b = np.log(table[:, 3] / b_ref)
        self.motions = motions
        self.zero = np.asarray(zero, dtype=float)
        self.radius = radius
        self.axis = axis
        self.sign = sign
        self.rotation = rotation
        self.evaluators = evaluators

    def unpack(self, theta: np.ndarray) -> tuple[np.ndarray, float]:
        """The matrix :math:`C` and the exponent."""
        return theta[:9].reshape(3, 3) / self.radius, float(theta[9])

    def pack(self, matrix: np.ndarray, exponent: float) -> np.ndarray:
        """The parameters of a matrix :math:`C` and an exponent."""
        return np.r_[np.ravel(matrix) * self.radius, exponent]

    def _rows(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The matrix :math:`C`, each volume's scale and its voxel-frame slopes."""
        matrix, exponent = self.unpack(theta)
        scales = np.exp(exponent * self.log_b)
        slopes = scales[:, np.newaxis] * (self.bvecs @ matrix.T)  # world
        return matrix, scales, self.sign * slopes @ self.rotation

    def matrices(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Each volume's mapping in the voxel frame: its matrix and translation."""
        rows = self._rows(theta)[2]
        rotations, translations = self.motions[:, :3, :3], self.motions[:, :3, 3]
        linear = rotations.copy()
        linear[:, self.axis] -= np.einsum("nk,nkj->nj", rows, rotations)
        offset = translations.copy()
        offset[:, self.axis] -= np.einsum("nk,nk->n", rows, translations - self.zero)
        return linear, offset

    def mappings(self, theta: np.ndarray) -> np.ndarray:
        """Each volume's mapping in the voxel frame, ``(N, 4, 4)``."""
        linear, offset = self.matrices(theta)
        mappings = np.repeat(np.eye(4)[np.newaxis], len(linear), axis=0)
        mappings[:, :3, :3], mappings[:, :3, 3] = linear, offset
        return mappings

    def cost(self, theta: np.ndarray) -> tuple[float, np.ndarray]:
        """The summed metric and its derivative with respect to the parameters."""
        matrix, scales, _ = self._rows(theta)
        linear, offset = self.matrices(theta)
        each = [evaluate(linear[v], offset[v]) for v, evaluate in enumerate(self.evaluators)]
        values = [value for value, _, _ in each]
        by_linear = np.stack([gradient for _, gradient, _ in each])
        by_offset = np.stack([gradient for _, _, gradient in each])
        rotations, translations = self.motions[:, :3, :3], self.motions[:, :3, 3]
        by_rows = -np.einsum("nkj,nj->nk", rotations, by_linear[:, self.axis]) - by_offset[
            :, self.axis, np.newaxis
        ] * (translations - self.zero)
        by_slopes = self.sign * by_rows @ self.rotation.T
        by_matrix = (scales[:, np.newaxis] * by_slopes).T @ self.bvecs
        by_exponent = np.sum(
            self.log_b * scales * np.einsum("nj,nj->n", by_slopes, self.bvecs @ matrix.T)
        )
        derivative = np.r_[by_matrix.ravel() / self.radius, by_exponent]
        return float(np.sum(values)), derivative


def _itk_image(values: np.ndarray, spacing: np.ndarray, origin: np.ndarray) -> Any:
    """A view of an array in ITK's order as an ITK image in the voxel frame, at ``origin``."""
    import itk

    image = itk.image_view_from_array(values)
    image.SetSpacing(spacing.tolist())
    image.SetOrigin(origin.tolist())
    return image


def _stage_image(image: Any, sigma: float, shrink: int) -> Any:
    """An image smoothed (``sigma`` in voxels) and shrunk as a stage sets."""
    import itk

    kind = type(image)
    if sigma > 0:
        smoother = itk.DiscreteGaussianImageFilter[kind, kind].New(
            image, Variance=sigma**2, UseImageSpacing=False, MaximumError=0.01
        )
        smoother.Update()
        image = smoother.GetOutput()
        image.DisconnectPipeline()
    if shrink > 1:
        shrinker = itk.ShrinkImageFilter[kind, kind].New(image, ShrinkFactors=shrink)
        shrinker.Update()
        image = shrinker.GetOutput()
        image.DisconnectPipeline()
    return image


class _Metric:
    """
    One volume's ITK metric, as a function of its mapping's matrix and translation.

    It holds the volume's arrays, which its stage images may only view.
    """

    def __init__(
        self,
        stage: Mapping[str, Any],
        arrays: tuple[np.ndarray, np.ndarray],
        spacing: np.ndarray,
        origin: np.ndarray,
        points: Any,
    ) -> None:
        import itk

        self.arrays = arrays
        sigma = float(stage.get("smoothing_sigma", 0.0))
        shrink = int(stage.get("shrink_factor", 1))
        fixed, moving = (
            _stage_image(_itk_image(values, spacing, origin), sigma, shrink) for values in arrays
        )
        kind = type(fixed)
        if stage["metric"] == "mattes":
            metric = itk.MattesMutualInformationImageToImageMetricv4[kind, kind].New()
            metric.SetNumberOfHistogramBins(int(stage.get("bins", 32)))
        else:
            metric = itk.CorrelationImageToImageMetricv4[kind, kind].New()
        metric.SetFixedImage(fixed)
        metric.SetMovingImage(moving)
        # Central differences of the moving image keep the derivative within a few percent
        # of the metric's own (ITK's default gradient filter makes it about 1.6 times larger)
        metric.SetUseMovingImageGradientFilter(False)
        self.transform = itk.AffineTransform[itk.D, 3].New()
        metric.SetMovingTransform(self.transform)
        metric.SetVirtualDomainFromImage(fixed)
        metric.SetFixedSampledPointSet(points)
        metric.SetUseSampledPointSet(True)
        metric.Initialize()
        self.metric = metric
        self.images = (fixed, moving)
        self.derivative = itk.Array[itk.D](self.transform.GetNumberOfParameters())

    def __call__(self, linear: np.ndarray, offset: np.ndarray) -> tuple:
        import itk

        self.transform.SetParameters(itk.OptimizerParameters[itk.D](np.r_[linear.ravel(), offset]))
        self.metric.GetDerivative(self.derivative)
        # ITK gives the direction of descent: the derivative with its sign reversed
        gradient = -itk.array_from_vnl_vector(self.derivative)
        return self.metric.GetCurrentValue(), gradient[:9].reshape(3, 3), gradient[9:]


class EddyFieldFit:
    """
    Estimate the eddy-current field of all volumes, from their images, with one model.

    Each volume is added (:meth:`add`) with its target (the image it should match once its
    field is undone), its sample and the head motion it is given; :meth:`fit` then
    estimates the model, and :meth:`mappings` gives each volume's head motion and field.
    Without ``edges``, the model is one :class:`EddyField` (:math:`C` and :math:`p`);
    with them, a :class:`SegmentedEddyField`, one :math:`C` for each segment of
    b-values, with the exponent held. The segments share no parameter, so each is fitted
    on its own. With ``fit_range``, the model is fitted on the volumes of a range of
    b-values only, and every volume added gets its field from it. Each volume's shift
    along phase encoding is derived from its slopes and the fields' zero point.

    The cost is the sum over volumes of ITK's metric between the target and the
    sample, run in the stages of the settings (the ``itk`` section of
    :data:`~nifreeze.registration.base.AFFINE` by default), and minimized by
    L-BFGS-B. A stage takes its metric, histogram bins, sampling, smoothing (in
    voxels), shrink factor and iterations from the settings; their step settings
    (``learning_rate`` and the like) belong to ITK's optimizer and do not apply. Both
    images of a stage are smoothed and then shrunk, and its points, drawn once, are the
    same for every volume. The images are prepared as
    :class:`~nifreeze.registration.itk_backend.ITK` prepares them (clipped to the
    sample's window, winsorized, histogram-matched as the settings ask), in their voxel
    frame, and kept within the mask's box and ``margin``; a point mapped beyond it does
    not count.

    Parameters
    ----------
    align_config : :obj:`~collections.abc.Mapping` or :obj:`os.pathlike`, optional
        ITK's settings, or a file with an ``itk`` section (see
        :func:`~nifreeze.registration.itk_backend.get_settings`).
    phase_encoding : :obj:`str`, optional
        The phase-encoding direction, as BIDS writes it.
    exponent : :obj:`float`, optional
        Hold the exponent :math:`p` at this value. Without ``edges``, it is estimated by
        default; with them, it is held at :data:`DEFAULT_EXPONENT` by default.
    b_ref : :obj:`float`, optional
        The reference b-value, without ``edges`` (default: the :func:`geometric_mean` of
        the b-values). With them, each segment's is the geometric mean of its b-values.
    edges : :obj:`~collections.abc.Sequence` of :obj:`float`, optional
        The b-values that split the segments, each with its own matrix :math:`C` (see
        :class:`SegmentedEddyField`); one model for all volumes if :obj:`None`.
    fit_range : :obj:`~collections.abc.Sequence` of :obj:`float`, optional
        Fit the model on the volumes whose b-value is in ``[low, high)`` only, and give
        every volume added its field from it (default: fit on all of them). With a
        single b-value in the range, the exponent is held: at ``exponent``, or at
        :data:`DEFAULT_EXPONENT` with a warning. Not with ``edges``.
    zero_point : :obj:`~numpy.ndarray`, optional
        Where the fields are zero (RAS+, mm; default: the :func:`field_of_view_centre`
        of the volumes' grid).
    seed : :obj:`int`, optional
        Seed of the points drawn in each stage.
    num_threads : :obj:`int`, optional
        ITK's threads.
    clip : :obj:`str`, optional
        Clip ``"sample"``, ``"predicted"`` (the target), ``"both"`` (default) or
        ``"none"`` of the images to the sample's intensity window.
    margin : :obj:`float`, optional
        How far (mm) beyond the mask's box each volume's images are kept.

    """

    def __init__(
        self,
        align_config: Mapping[str, Any] | str | PathLike | None = None,
        *,
        phase_encoding: str | None = None,
        exponent: float | None = None,
        b_ref: float | None = None,
        edges: Sequence[float] | None = None,
        fit_range: Sequence[float] | None = None,
        zero_point: Sequence[float] | np.ndarray | None = None,
        seed: int | None = 12345,
        num_threads: int | None = None,
        clip: str | bool | None = "both",
        margin: float = 15.0,
    ) -> None:
        if edges is not None and b_ref is not None:
            raise ValueError(EDGES_B_REF_ERROR_MSG)
        if edges is not None and fit_range is not None:
            raise ValueError(FIT_RANGE_EDGES_ERROR_MSG)
        self.settings = get_settings(align_config)
        self.phase_encoding = format_phase_encoding(phase_encoding)
        self.exponent = None if exponent is None else float(exponent)
        self.b_ref = None if b_ref is None else float(b_ref)
        self.edges = None if edges is None else _as_edges(edges)
        self.fit_range = None if fit_range is None else _as_range(fit_range)
        self.zero_point = None if zero_point is None else np.asarray(zero_point, dtype=float)
        """Where the fields are zero (RAS+, mm); once fitted, the one used."""
        self.seed = seed
        self.num_threads = num_threads
        clip = str(clip).lower()
        self.clip = {"true": "both", "false": "none"}.get(clip, clip)
        self.margin = float(margin)
        self.field: EddyField | SegmentedEddyField | None = None
        """The estimated field, once fitted."""
        self.evaluations: list[list[int]] = []
        """How many times each stage evaluated the cost, for each segment (one list
        without ``edges``)."""
        self.messages: list[list[str]] = []
        """How each stage's minimization stopped, for each segment."""
        self._grid: tuple[np.ndarray, np.ndarray, tuple[slice, ...]] | None = None
        self._fixed: dict[int, np.ndarray] = {}
        self._moving: dict[int, np.ndarray] = {}
        self._motion: dict[int, np.ndarray] = {}
        self._mappings: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    @property
    def indices(self) -> list[int]:
        """The volumes added, in order."""
        return sorted(self._fixed)

    def add(
        self,
        index: int,
        fixed: np.ndarray,
        moving: np.ndarray,
        affine: np.ndarray,
        mask: np.ndarray | None = None,
        motion: np.ndarray | None = None,
    ) -> None:
        """
        Add one volume.

        Parameters
        ----------
        index : :obj:`int`
            The volume's index, its row in the gradient table.
        fixed, moving : :obj:`~numpy.ndarray`
            The volume's target and its sample.
        affine : :obj:`~numpy.ndarray`
            The voxel-to-RAS+ affine of their grid, the same for every volume.
        mask : :obj:`~numpy.ndarray`, optional
            The mask of the target, the same for every volume.
        motion : :obj:`~numpy.ndarray`, optional
            The head motion the volume is given (a RAS+ matrix that maps the reference
            into it; default: none).

        """
        affine = np.asarray(affine, dtype=float)
        mask = np.ones(fixed.shape, dtype=bool) if mask is None else np.asanyarray(mask, bool)
        if self._grid is None:
            spacing = _frame(affine)[1]
            inside = np.argwhere(mask)
            pad = np.ceil(self.margin / spacing).astype(int)
            low = np.maximum(inside.min(axis=0) - pad, 0)
            high = np.minimum(inside.max(axis=0) + 1 + pad, mask.shape)
            box = tuple(slice(a, b) for a, b in zip(low, high, strict=True))
            self._grid = (affine, mask, box)
        elif not (
            np.allclose(affine, self._grid[0])
            and mask.shape == self._grid[1].shape
            and np.array_equal(mask, self._grid[1])
        ):
            raise ValueError(GRID_ERROR_MSG.format(index=index))

        window = None
        if self.clip in CLIPPING:
            try:
                window = robust_window(moving, mask)
            except ClippingValueError as exc:
                warn(f"Adding volume {index} without clipping: {exc}", stacklevel=2)
        fixed_values, moving_values = _prepare(fixed, moving, window, self.clip, self.settings)
        if self.settings.get("use_histogram_matching", False):
            import itk

            spacing = _frame(affine)[1]
            moving_values = np.transpose(
                itk.array_from_image(
                    _matched(
                        _voxel_image(moving_values, spacing), _voxel_image(fixed_values, spacing)
                    )
                ),
                (2, 1, 0),
            )
        box = self._grid[2]
        # Kept in ITK's order, so that each stage views them without a copy
        self._fixed[index] = np.ascontiguousarray(fixed_values[box].transpose(2, 1, 0))
        self._moving[index] = np.ascontiguousarray(moving_values[box].transpose(2, 1, 0))
        self._motion[index] = np.eye(4) if motion is None else np.array(motion, dtype=float)

    def _points(self, stage: Mapping[str, Any], number: int) -> np.ndarray:
        """A stage's points in the voxel frame (mm), within the mask."""
        assert self._grid is not None
        affine, mask, _ = self._grid
        spacing = _frame(affine)[1]
        shrink = int(stage.get("shrink_factor", 1))
        index = np.argwhere(mask)
        index = index[np.all(index % shrink == 0, axis=1)]
        points = (index + (shrink - 1) / 2) * spacing  # the centres of the shrunk voxels
        sampling = stage.get("sampling", "none")
        if sampling == "none":
            return points
        rng = np.random.default_rng(None if self.seed is None else self.seed + number)
        count = len(points)
        if "samples" in stage:
            count = min(count, int(stage["samples"]))
        else:
            count = max(1, round(float(stage.get("sampling_percentage", 1.0)) * count))
        if sampling == "regular":
            chosen = np.linspace(0, len(points) - 1, count).round().astype(int)
        else:
            chosen = np.sort(rng.choice(len(points), count, replace=False))
        # As ITK's samplers do, each point moves at random within its (shrunk) voxel
        jitter = rng.uniform(-0.5, 0.5, size=(count, 3)) * shrink * spacing
        return points[chosen] + jitter

    def _problem(
        self,
        indices: Sequence[int],
        table: np.ndarray,
        b_ref: float,
        zero: np.ndarray,
        number: int,
    ) -> _Problem:
        """The cost of stage ``number`` over the volumes ``indices``."""
        import itk

        assert self._grid is not None
        affine, _, box = self._grid
        spacing = _frame(affine)[1]
        stage = self.settings["stages"][number]
        origin = np.array([part.start for part in box]) * spacing

        points = itk.PointSet[itk.F, 3].New()
        for n, point in enumerate(self._points(stage, number)):
            points.SetPoint(n, itk.Point[itk.F, 3](point.tolist()))
        evaluators = [
            _Metric(stage, (self._fixed[index], self._moving[index]), spacing, origin, points)
            for index in indices
        ]
        return self._model(indices, table, b_ref, zero, evaluators)

    def _model(
        self,
        indices: Sequence[int],
        table: np.ndarray,
        b_ref: float,
        zero: np.ndarray,
        evaluators: Sequence[Callable[[np.ndarray, np.ndarray], tuple]] = (),
    ) -> _Problem:
        """The model over the volumes ``indices``: their mappings, and a cost if evaluated."""
        assert self._grid is not None and self.phase_encoding is not None
        affine, mask, _ = self._grid
        frame, spacing = _frame(affine)
        inverse = np.linalg.inv(frame)
        each = [inverse @ self._motion[index] @ frame for index in indices]
        motions = np.reshape(each, (-1, 4, 4))  # (0, 4, 4) without volumes
        inside = np.argwhere(mask) * spacing
        radius = float(np.sqrt(np.mean(np.sum((inside - inside.mean(axis=0)) ** 2, axis=1))))
        return _Problem(
            table,
            b_ref,
            motions,
            zero,
            radius,
            axis="ijk".index(self.phase_encoding[0]),
            sign=-1.0 if self.phase_encoding.endswith("-") else 1.0,
            rotation=frame[:3, :3],
            evaluators=evaluators,
        )

    def _table(self, gradients: Any) -> np.ndarray:
        """The rows of the volumes added, from the dataset's gradient table."""
        gradients = np.asarray(gradients, dtype=float)
        if len(gradients) <= self.indices[-1]:
            raise ValueError(
                TABLE_LENGTH_ERROR_MSG.format(rows=len(gradients), index=self.indices[-1])
            )
        return _table(gradients[self.indices])

    def _fit_group(
        self, indices: Sequence[int], table: np.ndarray, zero: np.ndarray
    ) -> tuple[EddyField, np.ndarray]:
        """Fit one model to the volumes ``indices``; its field and their voxel mappings."""
        from scipy.optimize import minimize

        _check_span(table)
        if self.edges is None:
            b_ref = geometric_mean(table[:, 3]) if self.b_ref is None else self.b_ref
            held = _held_exponent(table, self.exponent, stacklevel=4)
        else:
            b_ref = geometric_mean(table[:, 3])
            held = DEFAULT_EXPONENT if self.exponent is None else self.exponent
        bounds = [(None, None)] * 9 + [EXPONENT_BOUNDS if held is None else (held, held)]

        evaluations: list[int] = []
        messages: list[str] = []
        problem, theta = None, None
        for number, stage in enumerate(self.settings["stages"]):
            problem = self._problem(indices, table, b_ref, zero, number)
            if theta is None:
                theta = problem.pack(np.zeros((3, 3)), DEFAULT_EXPONENT if held is None else held)
            result = minimize(
                problem.cost,
                theta,
                jac=True,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": int(stage["iterations"])},
            )
            theta = result.x
            evaluations.append(int(result.nfev))
            messages.append(str(result.message))
            problem.evaluators = []  # release the stage's images and metrics
        assert problem is not None and theta is not None
        self.evaluations.append(evaluations)
        self.messages.append(messages)
        matrix, exponent = problem.unpack(theta)
        return EddyField(matrix, exponent, b_ref), problem.mappings(theta)

    def _groups(self, table: np.ndarray) -> list[np.ndarray]:
        """The rows of ``table`` fitted together: all, those in the range, or each segment's."""
        if self.fit_range is not None:
            low, high = self.fit_range
            inside = np.flatnonzero((table[:, 3] >= low) & (table[:, 3] < high))
            if not len(inside):
                raise ValueError(EMPTY_RANGE_ERROR_MSG.format(low=low, high=high))
            return [inside]
        if self.edges is None:
            return [np.arange(len(table))]
        segments = _segment_of(self.edges, table[:, 3])
        groups = [np.flatnonzero(segments == k) for k in range(len(self.edges) + 1)]
        for k, group in enumerate(groups):
            if not len(group):
                raise ValueError(EMPTY_SEGMENT_ERROR_MSG.format(segment=k, edges=self.edges))
        return groups

    def fit(self, gradients: Any) -> EddyField | SegmentedEddyField:
        """
        Estimate the field of every volume added.

        The estimation starts from no field, with the exponent at
        :data:`DEFAULT_EXPONENT` unless it is held.

        Parameters
        ----------
        gradients : :obj:`~numpy.ndarray`
            The dataset's gradient table, ``(N, 4)``, with a row for each volume index
            (see :meth:`EddyField.slopes`).

        Returns
        -------
        :class:`EddyField` or :class:`SegmentedEddyField`
            The estimate (segmented with ``edges``), also kept as :attr:`field`. The
            images are released.

        """
        if not self._fixed:
            raise ValueError(NOTHING_ADDED_ERROR_MSG)
        if self.phase_encoding is None:
            raise ValueError(NO_PHASE_ENCODING_ERROR_MSG)
        table = self._table(gradients)
        indices = self.indices
        groups = self._groups(table)

        assert self._grid is not None
        affine, mask, _ = self._grid
        frame = _frame(affine)[0]
        if self.zero_point is None:
            world = field_of_view_centre(affine, mask.shape)
        else:
            world = self.zero_point
        zero = (np.linalg.inv(frame) @ np.r_[world, 1.0])[:3]
        u = phase_encoding_vector(self.phase_encoding, affine)

        self.evaluations, self.messages = [], []
        parts: list[EddyField] = []
        mappings: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        def keep(chosen: list[int], voxel: np.ndarray) -> None:
            for index, mapping in zip(chosen, voxel, strict=True):
                world_mapping = frame @ mapping @ np.linalg.inv(frame)
                mappings[index] = split_phase_encoding(world_mapping, u)

        with _threads(self.num_threads):
            for group in groups:
                chosen = [indices[position] for position in group]
                part, voxel = self._fit_group(chosen, table[group], zero)
                parts.append(part)
                keep(chosen, voxel)
        if self.fit_range is not None:
            # The volumes outside the range (if any) get their field from the model
            rest = np.setdiff1d(np.arange(len(indices)), groups[0])
            chosen = [indices[position] for position in rest]
            model = self._model(chosen, table[rest], parts[0].b_ref, zero)
            keep(chosen, model.mappings(model.pack(parts[0].matrix, parts[0].exponent)))
        self.zero_point = np.asarray(world, dtype=float)
        self.field = parts[0] if self.edges is None else SegmentedEddyField(self.edges, parts)
        self._mappings = mappings
        self._fixed, self._moving = {}, {}
        return self.field

    def mappings(self) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """
        Each volume's head motion and field, as the dataset stores them.

        Returns
        -------
        :obj:`dict`
            By volume index, the head motion (with the shift along phase encoding derived
            from the field) and the field :math:`E_v = I - u s_v^T`, both RAS+ matrices;
            the field applies after the motion.

        """
        if self.field is None:
            raise ValueError(NOT_FITTED_ERROR_MSG)
        return dict(self._mappings)
