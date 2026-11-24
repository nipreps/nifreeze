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
"""A global, physics-based model of the eddy-current fields (#745): one matrix C and the
b-value's exponent for all volumes, or a matrix for each segment of b-values, with every
field zero at one point."""

import itertools
import re
from collections.abc import Callable
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from test_registration_conventions import SKIP_ELASTIX, _error, _motion, _move, _volume
from test_registration_itk import CENTRE, _across, _field, _off_centre

from nifreeze.registration.eddy_field import (
    B_REF_ERROR_MSG,
    BVALUE_ERROR_MSG,
    DEFAULT_EXPONENT,
    EDGES_B_REF_ERROR_MSG,
    EDGES_ERROR_MSG,
    MATRIX_ERROR_MSG,
    NO_PHASE_ENCODING_ERROR_MSG,
    NOT_FITTED_ERROR_MSG,
    NOTHING_ADDED_ERROR_MSG,
    SLOPES_ERROR_MSG,
    SPAN_ERROR_MSG,
    TABLE_ERROR_MSG,
    EddyField,
    EddyFieldFit,
    SegmentedEddyField,
    _Problem,
    field_of_view_centre,
    field_slopes,
)
from nifreeze.registration.itk_backend import _frame
from nifreeze.registration.utils import phase_encoding_vector, split_phase_encoding

SHELLS = (1000.0, 2000.0, 3000.0)
"""The b-values of the synthetic schemes."""
EDGES = (1500.0, 2500.0)
"""Edges that put each shell in a segment of its own."""
ZERO = CENTRE + np.array([9.0, -7.0, 6.0])
"""Where the synthetic fields are zero (RAS+, mm): 13 mm from the grid's centre."""


def _unit_vectors(rng: np.random.Generator, count: int) -> np.ndarray:
    vectors = rng.normal(size=(count, 3))
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def _balanced(rng: np.random.Generator, pairs: int = 3) -> np.ndarray:
    """Shells of antipodal pairs of b-vectors: within each, the b-vectors sum to zero."""
    rows = []
    for bval in SHELLS:
        vectors = _unit_vectors(rng, pairs)
        rows.append(np.column_stack([np.vstack([vectors, -vectors]), np.full(2 * pairs, bval)]))
    return np.vstack(rows)


def _half_sphere(rng: np.random.Generator) -> np.ndarray:
    """Three shells of 16 b-vectors each, all on one half of the sphere."""
    vectors = _unit_vectors(rng, 48)
    vectors[:, 2] = np.abs(vectors[:, 2])
    return np.column_stack([vectors, np.repeat(SHELLS, 16)])


def _dsi(rng: np.random.Generator) -> np.ndarray:
    """Many b-values, a few b-vectors each, as on a DSI grid."""
    bvals = np.repeat(np.linspace(300.0, 4000.0, 20), 3)
    return np.column_stack([_unit_vectors(rng, len(bvals)), bvals])


def _truth(rng: np.random.Generator, exponent: float) -> EddyField:
    return EddyField(rng.normal(scale=0.012, size=(3, 3)), exponent=exponent, b_ref=1000.0)


def _segmented(rng: np.random.Generator) -> SegmentedEddyField:
    """A matrix for each shell, unrelated to the others' (no power of the b-value)."""
    parts = [EddyField(rng.normal(scale=0.015, size=(3, 3)), b_ref=bval) for bval in SHELLS]
    return SegmentedEddyField(EDGES, parts)


def _about(fields: np.ndarray, zero: np.ndarray) -> np.ndarray:
    """The fields ``I - u s^T``, made zero at ``zero``: :math:`E(y) = y - u s^T (y - z)`."""
    about = np.array(fields, dtype=float)
    about[:, :3, 3] = (np.eye(3) - about[:, :3, :3]) @ zero
    return about


# The model


@pytest.mark.parametrize("exponent", [0.5, 1.0])
def test_slopes_follow_the_gradient_and_its_strength(exponent):
    rng = np.random.default_rng(0)
    field = _truth(rng, exponent)
    g = _unit_vectors(rng, 1)[0]
    gradients = [[*g, 1000.0], [*(-g), 1000.0], [*g, 4000.0]]

    slopes = field.slopes(gradients)

    assert np.allclose(slopes[0], field.matrix @ g)
    assert np.allclose(slopes[1], -slopes[0])  # the field flips with the gradient
    assert np.allclose(slopes[2], 4.0**exponent * slopes[0])  # 2x at p=1/2, 4x at p=1
    rescaled = field.rescaled(2000.0)
    assert rescaled.b_ref == 2000.0 and np.allclose(rescaled.slopes(gradients), slopes)


def test_fields_split_back_into_motion_and_field():
    """Each field moves points only along u, and splits from head motion exactly."""
    rng = np.random.default_rng(1)
    table = _balanced(rng)
    field = _truth(rng, 0.5)
    for seed in range(6):
        voxels = Rotation.random(random_state=seed).as_matrix()
        direction = voxels[:, seed % 3] * (-1.0 if seed % 2 else 1.0)
        fields = field.fields(table, direction)
        assert np.allclose(field_slopes(fields, direction), field.slopes(table))
        for eddy in fields:
            assert _across(eddy, direction) < 1e-12
            motion = _motion(degrees=2.0, shift=rng.normal(size=3))
            found_motion, found_eddy = split_phase_encoding(eddy @ motion, direction)
            assert np.allclose(found_motion, motion, atol=1e-12)
            assert np.allclose(found_eddy, eddy, atol=1e-12)


@pytest.mark.parametrize("exponent", [0.5, 1.0])
@pytest.mark.parametrize("scheme", [_balanced, _half_sphere, _dsi])
def test_from_slopes_recovers_the_matrix_and_exponent(scheme, exponent):
    rng = np.random.default_rng(2)
    table = scheme(rng)
    truth = _truth(rng, exponent)

    found = EddyField.from_slopes(truth.slopes(table), table)

    assert abs(found.exponent - exponent) < 1e-8
    assert np.allclose(found.rescaled(1000.0).matrix, truth.matrix, atol=1e-10)
    weighted = EddyField.from_slopes(truth.slopes(table), table, weights=rng.random(len(table)))
    assert abs(weighted.exponent - exponent) < 1e-8


def test_from_slopes_leaves_out_a_slope_every_volume_shares():
    """A shared slope stays out of C when the b-vectors sum to zero, and leaks otherwise."""
    rng = np.random.default_rng(3)
    shared = np.array([0.004, -0.006, 0.005])
    truth = _truth(rng, 0.5)

    balanced = _balanced(rng)
    found = EddyField.from_slopes(truth.slopes(balanced) + shared, balanced, exponent=0.5)
    assert np.allclose(found.rescaled(1000.0).matrix, truth.matrix, atol=1e-10)

    half = _half_sphere(rng)
    leaked = EddyField.from_slopes(truth.slopes(half) + shared, half, exponent=0.5)
    assert not np.allclose(leaked.rescaled(1000.0).matrix, truth.matrix, atol=1e-4)


def test_one_b_value_holds_the_exponent():
    rng = np.random.default_rng(4)
    table = _balanced(rng)
    table[:, 3] = 1000.0
    truth = _truth(rng, DEFAULT_EXPONENT)

    with pytest.warns(UserWarning, match="the same b-value"):
        found = EddyField.from_slopes(truth.slopes(table), table)

    assert found.exponent == DEFAULT_EXPONENT
    assert np.allclose(found.rescaled(1000.0).matrix, truth.matrix)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("plane", SPAN_ERROR_MSG),
        ("zero", BVALUE_ERROR_MSG),
        ("columns", TABLE_ERROR_MSG),
        ("slopes", SLOPES_ERROR_MSG),
    ],
)
def test_from_slopes_validates_its_inputs(change, message):
    rng = np.random.default_rng(5)
    table = _balanced(rng)
    slopes = _truth(rng, 0.5).slopes(table)
    if change == "plane":
        table[:, 2] = 0.0
    elif change == "zero":
        table[0, 3] = 0.0
    elif change == "columns":
        table = table[:, :3]
    else:
        slopes = slopes[:-1]
    with pytest.raises(ValueError, match=re.escape(message)):
        EddyField.from_slopes(slopes, table)


def test_the_model_validates_its_parameters():
    with pytest.raises(ValueError, match=re.escape(MATRIX_ERROR_MSG)):
        EddyField(np.eye(2))
    with pytest.raises(ValueError, match=re.escape(B_REF_ERROR_MSG)):
        EddyField(np.eye(3), b_ref=0.0)
    assert EddyField(np.eye(3)) == EddyField(np.eye(3))


def test_a_segmented_field_gives_each_segment_its_own_matrix():
    rng = np.random.default_rng(13)
    field = _segmented(rng)
    table = _balanced(rng)
    direction = np.array([0.0, 1.0, 0.0])

    slopes = field.slopes(table)

    for k, bval in enumerate(SHELLS):
        chosen = table[:, 3] == bval
        assert np.allclose(slopes[chosen], field.parts[k].slopes(table[chosen]))
    assert np.allclose(field_slopes(field.fields(table, direction), direction), slopes)
    # A b-value on an edge opens the next segment
    assert field.segments([1499.0, 1500.0, 2500.0, 9000.0]).tolist() == [0, 1, 2, 2]
    # Segments without volumes leave the slopes of the others
    first = table[table[:, 3] == SHELLS[0]]
    assert np.allclose(field.slopes(first), field.parts[0].slopes(first))


@pytest.mark.parametrize("edges", [[], [0.0, 1000.0], [2000.0, 1000.0], [1000.0, 1000.0]])
def test_segment_edges_are_positive_and_increasing(edges):
    with pytest.raises(ValueError, match=re.escape(EDGES_ERROR_MSG)):
        SegmentedEddyField(edges, [EddyField(np.eye(3))] * (len(edges) + 1))
    with pytest.raises(ValueError, match=re.escape(EDGES_ERROR_MSG)):
        EddyFieldFit(edges=edges)


def test_a_segmented_field_needs_a_field_for_each_segment():
    with pytest.raises(ValueError, match="2 for 2 edges"):
        SegmentedEddyField(EDGES, [EddyField(np.eye(3))] * 2)


@pytest.mark.parametrize(
    "linear",
    [
        np.diag([2.0, 2.0, 2.0]),
        np.diag([-1.25, -1.25, 2.5]),
        Rotation.from_rotvec([0.2, -0.3, 0.1]).as_matrix() @ np.diag([1.5, 2.0, 2.5]),
    ],
)
@pytest.mark.parametrize("shape", [(5, 6, 7), (64, 64, 41), (2, 3, 4, 10)])
def test_the_centre_of_the_field_of_view_is_the_middle_of_its_extent(linear, shape):
    """The middle of the extent, from index -0.5 to n - 0.5 on each axis, odd or even."""
    affine = np.eye(4)
    affine[:3, :3], affine[:3, 3] = linear, [-30.0, 12.5, 4.0]
    corners = np.array(list(itertools.product(*[(-0.5, n - 0.5) for n in shape[:3]])), dtype=float)
    world = corners @ affine[:3, :3].T + affine[:3, 3]

    assert np.allclose(field_of_view_centre(affine, shape), world.mean(axis=0))


# The cost and its derivative


def _quadratic(rng: np.random.Generator) -> Callable:
    """A smooth stand-in for one volume's metric, with its exact derivative."""
    linear0, offset0 = rng.normal(size=(3, 3)), rng.normal(size=3)
    weights = rng.random(12) + 0.5

    def evaluate(linear, offset):
        residual = np.r_[(linear - linear0).ravel(), offset - offset0]
        value = float(np.sum(weights * np.sin(residual) ** 2))
        gradient = weights * np.sin(2 * residual)
        return value, gradient[:9].reshape(3, 3), gradient[9:]

    return evaluate


def _central(cost, theta: np.ndarray, step: float) -> np.ndarray:
    return np.array(
        [
            (cost(theta + step * e)[0] - cost(theta - step * e)[0]) / (2 * step)
            for e in np.eye(len(theta))
        ]
    )


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_the_chain_rule_gives_the_derivative_of_the_cost(sign):
    """From each volume's derivative to C and the exponent, with the zero point off the
    grid's centre."""
    rng = np.random.default_rng(6)
    table = _balanced(rng)
    motions = np.stack([_motion(degrees=3.0, shift=rng.normal(size=3)) for _ in table])
    problem = _Problem(
        table,
        b_ref=1500.0,
        motions=motions,
        zero=np.array([12.0, 50.0, 20.0]),
        radius=25.0,
        axis=1,
        sign=sign,
        rotation=Rotation.random(random_state=6).as_matrix(),
        evaluators=[_quadratic(rng) for _ in table],
    )
    theta = problem.pack(rng.normal(scale=0.01, size=(3, 3)), 0.7)

    _, derivative = problem.cost(theta)

    assert np.allclose(derivative, _central(problem.cost, theta, 1e-6), rtol=1e-6, atol=1e-8)


def test_the_mappings_are_the_model_in_the_voxel_frame():
    """Each voxel-frame mapping is the head motion, then the field, zero at the zero point."""
    rng = np.random.default_rng(7)
    table = _balanced(rng)
    motions = np.stack([_motion(degrees=3.0, shift=rng.normal(size=3)) for _ in table])
    zero = np.array([12.0, 50.0, 20.0])
    problem = _Problem(table, 1000.0, motions, zero, 25.0, 2, -1.0, np.eye(3), [])
    field = _truth(rng, 0.5)

    mappings = problem.mappings(problem.pack(field.matrix, 0.5))

    u = np.array([0.0, 0.0, -1.0])  # "k-" in an identity frame
    fields = _about(field.fields(table, u), zero)
    assert np.allclose(mappings, fields @ motions)
    assert np.allclose(fields @ np.r_[zero, 1.0], np.r_[zero, 1.0])  # zero at the zero point


# The fit on images (ITK)


def _images(
    exponent: float = 1.0,
    phase_encoding: str = "j-",
    shared: np.ndarray | None = None,
    zero: np.ndarray = ZERO,
    segmented: bool = False,
):
    """Volumes moved by known head motion and known fields, all zero at ``zero``."""
    rng = np.random.default_rng(8)
    data, mask = _volume()
    affine = _off_centre()
    u = phase_encoding_vector(phase_encoding, affine)
    table = _balanced(rng)
    truth = _segmented(rng) if segmented else _truth(rng, exponent)
    fields = _about(truth.fields(table, u), zero)
    motions = np.stack(
        [_motion(degrees=rng.uniform(0.0, 2.0), shift=rng.normal(size=3)) for _ in table]
    )
    mappings = fields @ motions
    if shared is not None:
        mappings = _about(_field(u, shared)[np.newaxis], zero)[0] @ mappings
    samples = np.stack([_move(data, affine, mapping) for mapping in mappings])
    return SimpleNamespace(
        data=data,
        mask=mask,
        affine=affine,
        u=u,
        table=table,
        truth=truth,
        fields=fields,
        motions=motions,
        mappings=mappings,
        samples=samples,
    )


def _fitted(sim, phase_encoding: str = "j-", **kwargs) -> EddyFieldFit:
    fit = EddyFieldFit(phase_encoding=phase_encoding, num_threads=1, clip="none", **kwargs)
    for index, (sample, motion) in enumerate(zip(sim.samples, sim.motions, strict=True)):
        fit.add(index, sim.data, sample, sim.affine, sim.mask, motion)
    return fit


def _voxel_zero(affine: np.ndarray, zero: np.ndarray = ZERO) -> np.ndarray:
    """A world point in the grid's voxel frame (see the backend's ``_frame``)."""
    return (np.linalg.inv(_frame(affine)[0]) @ np.r_[zero, 1.0])[:3]


def _mapping_errors(fit: EddyFieldFit, sim) -> np.ndarray:
    """Each volume's mean displacement error (mm, in the mask) of the whole mapping."""
    return np.array(
        [
            _error(sim.mask, sim.affine, eddy @ motion, sim.mappings[index])
            for index, (motion, eddy) in sorted(fit.mappings().items())
        ]
    )


@SKIP_ELASTIX
@pytest.mark.parametrize(("exponent", "phase_encoding"), [(0.5, "k"), (1.0, "j-")])
def test_one_model_recovers_the_field_from_images(exponent, phase_encoding):
    sim = _images(exponent, phase_encoding)
    fit = _fitted(sim, phase_encoding, zero_point=ZERO)

    found = fit.fit(sim.table)

    assert isinstance(found, EddyField)
    assert abs(found.exponent - exponent) < 0.05
    error = np.abs(found.rescaled(1000.0).matrix - sim.truth.matrix).max()
    assert error < 0.05 * np.abs(sim.truth.matrix).max()
    for index, (motion, eddy) in fit.mappings().items():
        assert np.allclose(eddy, found.fields(sim.table[[index]], sim.u)[0])
        assert np.allclose(motion[:3, :3], sim.motions[index][:3, :3])  # rotations held
    assert _mapping_errors(fit, sim).max() < 0.05
    assert fit.zero_point is not None and np.allclose(fit.zero_point, ZERO)
    assert len(fit.evaluations) == 1 and len(fit.evaluations[0]) == 2
    assert min(fit.evaluations[0]) > 0 and len(fit.messages[0]) == 2


@SKIP_ELASTIX
def test_a_matrix_for_each_segment_recovers_each_shells_field():
    sim = _images(segmented=True)
    fit = _fitted(sim, edges=EDGES, zero_point=ZERO)

    found = fit.fit(sim.table)

    assert isinstance(found, SegmentedEddyField) and found.edges == EDGES
    for part, truth in zip(found.parts, sim.truth.parts, strict=True):
        assert part.exponent == DEFAULT_EXPONENT and part.b_ref == pytest.approx(truth.b_ref)
        assert np.abs(part.matrix - truth.matrix).max() < 0.05 * np.abs(truth.matrix).max()
    assert _mapping_errors(fit, sim).max() < 0.05
    assert len(fit.evaluations) == len(SHELLS)


@SKIP_ELASTIX
def test_the_zero_point_given_is_the_one_fitted():
    """Data whose fields are zero 13 mm from the centre: the fit about the centre misses."""
    sim = _images()

    given = _fitted(sim, zero_point=ZERO)
    given.fit(sim.table)
    centred = _fitted(sim)
    centred.fit(sim.table)

    assert centred.zero_point is not None
    assert np.allclose(centred.zero_point, field_of_view_centre(sim.affine, sim.data.shape))
    assert np.median(_mapping_errors(centred, sim)) > 2 * np.median(_mapping_errors(given, sim))


@SKIP_ELASTIX
def test_the_zero_point_defaults_to_the_centre_of_the_field_of_view():
    centre = field_of_view_centre(_off_centre(), _volume()[0].shape)
    sim = _images(zero=centre)
    fit = _fitted(sim)

    fit.fit(sim.table)

    assert fit.zero_point is not None and np.allclose(fit.zero_point, centre)
    assert _mapping_errors(fit, sim).max() < 0.05


@SKIP_ELASTIX
def test_the_fit_leaves_out_most_of_a_stretch_every_volume_shares():
    """
    A stretch shared by every volume, on top of the field, mostly stays out of the fields.

    A registration of each volume alone takes all of it. On slopes, the b-vectors of each
    shell cancel it exactly (see the test of :meth:`EddyField.from_slopes`); on images,
    the volumes of an antipodal pair weigh a little differently in the metric, so some of
    it leaks, in proportion to its size (about a tenth here, at most a fifth).
    """
    shared = np.array([0.01, -0.015, 0.012])
    sim = _images(shared=shared)

    found = _fitted(sim, zero_point=ZERO).fit(sim.table)

    assert isinstance(found, EddyField)
    assert abs(found.exponent - 1.0) < 0.05
    error = found.slopes(sim.table) - sim.truth.slopes(sim.table)
    share = np.linalg.norm(error, axis=1) / np.linalg.norm(shared)
    assert np.median(share) < 0.15 and share.max() < 0.25


@SKIP_ELASTIX
def test_the_fit_holds_the_exponent_given():
    sim = _images()
    fit = _fitted(sim, exponent=0.8, b_ref=1000.0, zero_point=ZERO)

    found = fit.fit(sim.table)

    assert isinstance(found, EddyField) and found.exponent == 0.8 and found.b_ref == 1000.0
    segmented = _fitted(sim, exponent=1.0, edges=EDGES, zero_point=ZERO).fit(sim.table)
    assert isinstance(segmented, SegmentedEddyField)
    assert [part.exponent for part in segmented.parts] == [1.0] * len(SHELLS)


@SKIP_ELASTIX
def test_one_b_value_holds_the_exponent_of_the_fit():
    sim = _images()
    sim.table[:, 3] = 1000.0
    fit = _fitted(sim, zero_point=ZERO)

    with pytest.warns(UserWarning, match="the same b-value"):
        found = fit.fit(sim.table)

    assert isinstance(found, EddyField) and found.exponent == DEFAULT_EXPONENT


@SKIP_ELASTIX
def test_itks_derivative_points_downhill():
    """ITK's derivative of the summed metric has the sign and size of the cost's slope."""
    sim = _images()
    fit = _fitted(sim)
    problem = fit._problem(fit.indices, sim.table, 2000.0, _voxel_zero(sim.affine), 1)
    theta = problem.pack(0.5 * sim.truth.matrix, 0.8)

    _, derivative = problem.cost(theta)
    central = _central(problem.cost, theta, 1e-3)

    cosine = derivative @ central / (np.linalg.norm(derivative) * np.linalg.norm(central))
    assert cosine > 0.99
    assert 0.9 < np.linalg.norm(derivative) / np.linalg.norm(central) < 1.1


@SKIP_ELASTIX
def test_keeping_the_mask_box_leaves_the_correlation_unchanged():
    sim = _images()
    zero = _voxel_zero(sim.affine)
    theta = None
    values = []
    for margin in (8.0, 1000.0):  # the mask's box and 4 voxels, and the whole grid
        fit = _fitted(sim, margin=margin)
        kept = fit._fixed[0].shape
        problem = fit._problem(fit.indices, sim.table, 2000.0, zero, 1)
        del fit  # the stage's metrics hold the arrays they view
        if theta is None:
            assert np.all(np.array(kept) < np.array(sim.data.shape))
            theta = problem.pack(sim.truth.matrix, 1.0)
        values.append(problem.cost(theta)[0])
    assert values[0] == pytest.approx(values[1], rel=1e-6)


@SKIP_ELASTIX
def test_the_fit_prepares_the_images_as_the_backend_does():
    """Clipped to the sample's window, winsorized and histogram-matched, as by the backend."""
    import itk

    from nifreeze.data.filtering import robust_window
    from nifreeze.registration.itk_backend import _matched, _prepare, _voxel_image, get_settings

    sim = _images()
    settings = get_settings() | {"use_histogram_matching": True}
    fit = EddyFieldFit(settings, phase_encoding="j-", num_threads=1, margin=1000.0)
    sample = sim.samples[0]

    fit.add(0, sim.data, sample, sim.affine, sim.mask, sim.motions[0])

    window = robust_window(sample, sim.mask)
    fixed, moving = _prepare(sim.data, sample, window, "both", settings)
    spacing = np.linalg.norm(sim.affine[:3, :3], axis=0)
    matched = _matched(_voxel_image(moving, spacing), _voxel_image(fixed, spacing))
    assert np.array_equal(fit._fixed[0], fixed.transpose(2, 1, 0))
    assert np.array_equal(fit._moving[0], itk.array_from_image(matched))
    with pytest.warns(UserWarning, match="without clipping"):
        fit.add(1, sim.data, np.zeros_like(sample), sim.affine, sim.mask)


def test_a_stage_draws_its_points_in_the_mask():
    data, mask = _volume()
    fit = EddyFieldFit(phase_encoding="j", clip="none")
    fit.add(0, data, data, _off_centre(), mask)
    stage = {"metric": "correlation"}

    every = fit._points(stage | {"sampling": "none", "shrink_factor": 2}, 0)
    share = fit._points(stage | {"sampling": "regular", "sampling_percentage": 0.1}, 0)
    count = fit._points(stage | {"sampling": "random", "samples": 500}, 0)

    assert len(every) == np.sum(np.all(np.argwhere(mask) % 2 == 0, axis=1))
    assert len(share) == round(0.1 * mask.sum())
    assert len(count) == 500
    assert np.array_equal(count, fit._points(stage | {"sampling": "random", "samples": 500}, 0))


def test_the_fit_validates_its_inputs():
    data, mask = _volume()
    affine = _off_centre()
    table = _balanced(np.random.default_rng(10))
    with pytest.raises(ValueError, match=NOTHING_ADDED_ERROR_MSG):
        EddyFieldFit(phase_encoding="j").fit(table)
    with pytest.raises(ValueError, match=re.escape(NOT_FITTED_ERROR_MSG)):
        EddyFieldFit(phase_encoding="j").mappings()
    with pytest.raises(ValueError, match=re.escape(EDGES_B_REF_ERROR_MSG)):
        EddyFieldFit(edges=EDGES, b_ref=1000.0)

    fit = EddyFieldFit(clip="none")
    fit.add(0, data, data, affine, mask)
    with pytest.raises(ValueError, match=re.escape(NO_PHASE_ENCODING_ERROR_MSG)):
        fit.fit(table)
    fit.phase_encoding = "j"
    with pytest.raises(ValueError, match="has 0 rows"):
        fit.fit(table[:0])
    zero = table.copy()
    zero[0, 3] = 0.0
    with pytest.raises(ValueError, match=re.escape(BVALUE_ERROR_MSG)):
        fit.fit(zero)
    with pytest.raises(ValueError, match="grid and mask"):
        fit.add(1, data, data, 2 * affine, mask)
    with pytest.raises(ValueError, match="grid and mask"):
        fit.add(1, data, data, affine, ~mask)

    segmented = EddyFieldFit(phase_encoding="j", clip="none", edges=EDGES)
    for index in range(len(table)):
        segmented.add(index, data, data, affine, mask)
    middle = table.copy()
    middle[middle[:, 3] == 2000.0, 3] = 1000.0
    with pytest.raises(ValueError, match=r"segment 1 of the edges \(1500.0, 2500.0\)"):
        segmented.fit(middle)
