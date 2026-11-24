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
"""One eddy-current field model for all volumes: C and the b-value's exponent (#740), or a
C for each segment of b-values, with free or derived shifts."""

import re
from collections.abc import Callable
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from test_registration_conventions import SKIP_ELASTIX, _error, _motion, _move, _volume
from test_registration_itk import _across, _field, _off_centre

from nifreeze.data.dmri import DWI
from nifreeze.estimator import Estimator
from nifreeze.registration.eddy_field import (
    BVALUE_ERROR_MSG,
    DEFAULT_EXPONENT,
    EDGES_ERROR_MSG,
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
    field_slopes,
)
from nifreeze.registration.utils import phase_encoding_vector, split_phase_encoding

SHELLS = (1000.0, 2000.0, 3000.0)
"""The b-values of the synthetic schemes."""
EDGES = (1500.0, 2500.0)
"""The b-values that split :data:`SHELLS` into one segment each."""
ZERO_POINT_MM = 1.0
"""How close derived shifts find the fields' zero point: 24 mm from the grid's centre, it
comes within 0.27 mm; with slopes of 0.01 to 0.06 per mm, 1 mm moves a shift by at most
0.06 mm."""


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
    assert np.allclose(field.matrix_at(4000.0) @ g, slopes[2])
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
    with pytest.raises(ValueError, match="3 x 3"):
        EddyField(np.eye(2))
    with pytest.raises(ValueError, match="must be positive"):
        EddyField(np.eye(3), b_ref=0.0)
    assert EddyField(np.eye(3)) == EddyField(np.eye(3))


def test_a_segmented_field_gives_each_segment_its_own_matrix():
    rng = np.random.default_rng(13)
    table = _balanced(rng)
    parts = [_truth(rng, 0.5) for _ in SHELLS]
    field = SegmentedEddyField(EDGES, parts)

    slopes = field.slopes(table)

    for bval, part in zip(SHELLS, parts, strict=True):
        chosen = table[:, 3] == bval
        assert np.allclose(slopes[chosen], part.slopes(table[chosen]))
    u = np.array([0.0, 1.0, 0.0])
    assert np.allclose(field_slopes(field.fields(table, u), u), slopes)
    assert field.segments([1499.0, 1500.0, 2500.0]).tolist() == [0, 1, 2]


@pytest.mark.parametrize(
    ("edges", "count", "message"),
    [
        (EDGES, 2, "one field for each segment"),
        ((2500.0, 1500.0), 3, re.escape(EDGES_ERROR_MSG)),
        ((0.0, 1500.0), 3, re.escape(EDGES_ERROR_MSG)),
    ],
)
def test_a_segmented_field_validates_its_parameters(edges, count, message):
    with pytest.raises(ValueError, match=message):
        SegmentedEddyField(edges, [EddyField(np.zeros((3, 3)))] * count)


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


@pytest.mark.parametrize("derived", [False, True])
@pytest.mark.parametrize("segmented", [False, True])
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_the_chain_rule_gives_the_derivative_of_the_cost(sign, segmented, derived):
    """From each volume's derivative to each C, the exponent, and the shifts or the zero
    point, exactly."""
    rng = np.random.default_rng(6)
    table = _balanced(rng)
    motions = np.stack([_motion(degrees=3.0, shift=rng.normal(size=3)) for _ in table])
    problem = _Problem(
        table,
        b_ref=[900.0, 2100.0, 2800.0] if segmented else 1500.0,
        motions=motions,
        centre=np.array([30.0, 41.0, 35.0]),
        radius=25.0,
        axis=1,
        sign=sign,
        rotation=Rotation.random(random_state=6).as_matrix(),
        evaluators=[_quadratic(rng) for _ in table],
        groups=np.searchsorted(EDGES, table[:, 3], "right") if segmented else None,
        derived=derived,
    )
    matrices = rng.normal(scale=0.01, size=(3 if segmented else 1, 3, 3))
    rest = rng.normal(scale=0.5, size=3) if derived else rng.normal(size=len(table))
    theta = problem.pack(matrices, 0.7, rest)

    _, derivative = problem.cost(theta)

    assert np.allclose(derivative, _central(problem.cost, theta, 1e-6), rtol=1e-6, atol=1e-8)


def test_the_mappings_are_the_model_in_the_voxel_frame():
    """Each voxel-frame mapping is the head motion, then the field and the shift."""
    rng = np.random.default_rng(7)
    table = _balanced(rng)
    motions = np.stack([_motion(degrees=3.0, shift=rng.normal(size=3)) for _ in table])
    centre = np.array([30.0, 41.0, 35.0])
    problem = _Problem(table, 1000.0, motions, centre, 25.0, 2, -1.0, np.eye(3), [])
    field = _truth(rng, 0.5)
    shifts = rng.normal(size=len(table))

    mappings = problem.mappings(problem.pack(field.matrix, 0.5, shifts))

    u = np.array([0.0, 0.0, -1.0])  # "k-" in an identity frame
    for mapping, eddy, motion, shift in zip(
        mappings, field.fields(table, u), motions, shifts, strict=True
    ):
        about = eddy.copy()
        about[:3, 3] = (np.eye(3) - eddy[:3, :3]) @ centre - u * shift
        assert np.allclose(mapping, about @ motion)


def test_derived_shifts_put_the_zero_of_every_field_at_one_point():
    rng = np.random.default_rng(12)
    table = _balanced(rng)
    motions = np.stack([_motion(degrees=3.0, shift=rng.normal(size=3)) for _ in table])
    centre = np.array([30.0, 41.0, 35.0])
    derived = _Problem(table, 1000.0, motions, centre, 25.0, 2, -1.0, np.eye(3), [], derived=True)
    field = _truth(rng, 0.5)
    theta = derived.pack(field.matrix, 0.5, rng.normal(scale=0.5, size=3))
    zero = derived.zero_point(theta)

    mappings = derived.mappings(theta)

    u = np.array([0.0, 0.0, -1.0])  # "k-" in an identity frame
    for mapping, eddy, motion in zip(mappings, field.fields(table, u), motions, strict=True):
        about = eddy.copy()
        about[:3, 3] = (np.eye(3) - eddy[:3, :3]) @ zero
        assert np.allclose(mapping, about @ motion)
    # The same mappings as free shifts of those sizes
    free = _Problem(table, 1000.0, motions, centre, 25.0, 2, -1.0, np.eye(3), [])
    assert np.allclose(
        free.mappings(free.pack(field.matrix, 0.5, derived.shifts(theta))), mappings
    )


# The fit on images (ITK)


def _images(exponent: float, phase_encoding: str, shared: np.ndarray | None = None):
    """Volumes moved by known head motion and fields of known C and exponent."""
    rng = np.random.default_rng(8)
    data, mask = _volume()
    affine = _off_centre()
    u = phase_encoding_vector(phase_encoding, affine)
    table = _balanced(rng)
    truth = _truth(rng, exponent)
    # The fields are zero at the world's origin, away from the grid's centre: the
    # shifts along phase encoding take that up
    fields = truth.fields(table, u)
    motions = np.stack(
        [_motion(degrees=rng.uniform(0.0, 2.0), shift=rng.normal(size=3)) for _ in table]
    )
    mappings = fields @ motions
    if shared is not None:
        mappings = _field(u, shared) @ mappings
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


def _fitted(sim, phase_encoding: str, **kwargs) -> EddyFieldFit:
    fit = EddyFieldFit(phase_encoding=phase_encoding, num_threads=1, clip="none", **kwargs)
    for index, (sample, motion) in enumerate(zip(sim.samples, sim.motions, strict=True)):
        fit.add(index, sim.data, sample, sim.affine, sim.mask, motion)
    return fit


@SKIP_ELASTIX
@pytest.mark.parametrize(("exponent", "phase_encoding"), [(0.5, "k"), (1.0, "j-")])
def test_the_fit_recovers_the_field_from_images(exponent, phase_encoding):
    sim = _images(exponent, phase_encoding)
    fit = _fitted(sim, phase_encoding)

    found = fit.fit(sim.table)

    assert abs(found.exponent - exponent) < 0.05
    error = np.abs(found.rescaled(1000.0).matrix - sim.truth.matrix).max()
    assert error < 0.05 * np.abs(sim.truth.matrix).max()
    mappings = fit.mappings()
    for index, (motion, eddy) in mappings.items():
        assert np.allclose(eddy, found.fields(sim.table[[index]], sim.u)[0])
        assert np.allclose(motion[:3, :3], sim.motions[index][:3, :3])  # rotations held
        assert _error(sim.mask, sim.affine, eddy @ motion, sim.mappings[index]) < 0.05
    assert len(fit.evaluations) == 2 and min(fit.evaluations) > 0
    assert len(fit.shifts) == len(sim.table)


@SKIP_ELASTIX
def test_the_fit_leaves_out_a_stretch_every_volume_shares():
    """A stretch shared by every volume, on top of the field, stays out of C."""
    shared = np.array([0.01, -0.015, 0.012])
    sim = _images(1.0, "j-", shared=shared)

    found = _fitted(sim, "j-").fit(sim.table)

    assert abs(found.exponent - 1.0) < 0.05
    error = np.abs(found.rescaled(1000.0).matrix - sim.truth.matrix).max()
    assert error < 0.05 * np.abs(sim.truth.matrix).max()


@SKIP_ELASTIX
def test_the_fit_with_a_held_exponent_and_from_a_start():
    sim = _images(1.0, "j-")
    start = EddyField(0.5 * sim.truth.matrix, exponent=1.0, b_ref=1000.0)

    found = _fitted(sim, "j-", exponent=0.8, b_ref=1000.0).fit(sim.table, start=start)

    assert found.exponent == 0.8 and found.b_ref == 1000.0


@SKIP_ELASTIX
def test_the_fit_recovers_the_matrix_of_each_segment():
    """With a C for each shell, each shell's slopes, whatever power of the b-value they follow."""
    sim = _images(1.0, "j-")
    fit = _fitted(sim, "j-", segments=EDGES)

    found = fit.fit(sim.table)

    assert isinstance(found, SegmentedEddyField)
    for part, bval in zip(found.parts, SHELLS, strict=True):
        assert part.b_ref == pytest.approx(bval)
        expected = sim.truth.matrix_at(bval)
        assert np.abs(part.matrix - expected).max() < 0.1 * np.abs(expected).max()
    for index, (motion, eddy) in fit.mappings().items():
        assert np.allclose(eddy, found.fields(sim.table[[index]], sim.u)[0])
        assert _error(sim.mask, sim.affine, eddy @ motion, sim.mappings[index]) < 0.05
    assert fit.zero_point is None


@SKIP_ELASTIX
def test_derived_shifts_find_the_fields_zero_point():
    """The fields of the images are zero at the world's origin, away from the grid."""
    sim = _images(0.5, "k")
    fit = _fitted(sim, "k", shifts="derived")

    fit.fit(sim.table)

    assert fit.zero_point is not None
    assert np.linalg.norm(fit.zero_point) < ZERO_POINT_MM
    for index, (motion, eddy) in fit.mappings().items():
        assert np.allclose(motion[:3, :3], sim.motions[index][:3, :3])  # rotations held
        assert _error(sim.mask, sim.affine, eddy @ motion, sim.mappings[index]) < 0.05
    assert len(fit.shifts) == len(sim.table)


@SKIP_ELASTIX
@pytest.mark.parametrize(("segments", "shifts"), [(None, "free"), (EDGES, "derived")])
def test_itks_derivative_points_downhill(segments, shifts):
    """ITK's derivative of the summed metric has the sign and size of the cost's slope."""
    sim = _images(1.0, "j-")
    fit = _fitted(sim, "j-", segments=segments, shifts=shifts)
    groups, b_refs = fit._segments(sim.table)
    problem = fit._problem(sim.table, b_refs, groups, 1)  # the correlation stage
    rng = np.random.default_rng(9)
    matrices = np.stack([0.5 * sim.truth.matrix_at(b_ref) for b_ref in b_refs])
    rest = rng.normal(scale=0.5, size=3 if shifts == "derived" else len(sim.table))
    theta = problem.pack(matrices, 0.8, rest)

    _, derivative = problem.cost(theta)
    central = _central(problem.cost, theta, 1e-3)

    cosine = derivative @ central / (np.linalg.norm(derivative) * np.linalg.norm(central))
    assert cosine > 0.99
    assert 0.9 < np.linalg.norm(derivative) / np.linalg.norm(central) < 1.1


@SKIP_ELASTIX
def test_keeping_the_mask_box_leaves_the_correlation_unchanged():
    sim = _images(1.0, "j-")
    theta = None
    values = []
    for margin in (8.0, 1000.0):  # the mask's box and 4 voxels, and the whole grid
        fit = _fitted(sim, "j-", margin=margin)
        kept = fit._fixed[0].shape
        problem = fit._problem(sim.table, np.array([2000.0]), np.zeros(len(sim.table), int), 1)
        del fit  # the stage's metrics hold the arrays they view
        if theta is None:
            assert np.all(np.array(kept) < np.array(sim.data.shape))
            theta = problem.pack(sim.truth.matrix, 1.0, np.zeros(len(sim.table)))
        values.append(problem.cost(theta)[0])
    assert values[0] == pytest.approx(values[1], rel=1e-6)


@SKIP_ELASTIX
def test_the_fit_prepares_the_images_as_the_backend_does():
    """Clipped to the sample's window, winsorized and histogram-matched, as by the backend."""
    import itk

    from nifreeze.data.filtering import robust_window
    from nifreeze.registration.itk_backend import _matched, _prepare, _voxel_image, get_settings

    sim = _images(1.0, "j-")
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

    fit = EddyFieldFit(clip="none")
    fit.add(0, data, data, affine, mask)
    with pytest.raises(ValueError, match=re.escape(NO_PHASE_ENCODING_ERROR_MSG)):
        fit.fit(table)
    fit.phase_encoding = "j"
    with pytest.raises(ValueError, match="has 0 rows"):
        fit.fit(table[:0])
    with pytest.raises(ValueError, match="'free' or 'derived'"):
        EddyFieldFit(shifts="each")
    with pytest.raises(ValueError, match=re.escape(EDGES_ERROR_MSG)):
        EddyFieldFit(segments=[2500.0, 1500.0])
    fit.segments = (1500.0, 2500.0, 5000.0)
    with pytest.raises(ValueError, match="No volume has a b-value in segment 0"):
        fit.fit(table[[0]] * [1, 1, 1, 4])
    with pytest.raises(ValueError, match="grid and mask"):
        fit.add(1, data, data, 2 * affine, mask)
    with pytest.raises(ValueError, match="grid and mask"):
        fit.add(1, data, data, affine, ~mask)


# The eddy-current pass


def _dwi(sim, phase_encoding: str) -> DWI:
    dwi = DWI(
        dataobj=sim.samples,
        affine=sim.affine,
        brainmask=sim.mask,
        bzero=sim.data,
        gradients=sim.table,
        phase_encoding=phase_encoding,
    )
    dwi.motion = sim.motions.copy()
    return dwi


@SKIP_ELASTIX
def test_the_eddy_pass_fits_one_field_for_every_volume():
    """Through the estimator: the field of each volume, and its head motion's rotation kept."""
    sim = _images(1.0, "j-")
    dwi = _dwi(sim, "j-")
    fit = EddyFieldFit(num_threads=1, clip="none")

    Estimator("trivial", estimates="eddy", registration=fit).run(dwi, n_jobs=1, omp_nthreads=1)

    assert fit.phase_encoding == "j-"  # the dataset's
    assert fit.field is not None and abs(fit.field.exponent - 1.0) < 0.05
    assert dwi.motion is not None and dwi.eddy is not None
    assert np.allclose(dwi.motion[:, :3, :3], sim.motions[:, :3, :3])
    assert np.abs(dwi.eddy - sim.fields).max() < 2e-3
    for index, mapping in enumerate(sim.mappings):
        assert _error(sim.mask, sim.affine, dwi.get_transform(index), mapping) < 0.05


@SKIP_ELASTIX
def test_the_eddy_pass_starts_from_the_fields_stored():
    sim = _images(1.0, "j-")
    dwi = _dwi(sim, "j-")
    dwi.eddy = sim.fields.copy()  # as a previous pass would leave them
    fit = EddyFieldFit(num_threads=1, clip="none", b_ref=1000.0)
    starts = []
    original = fit.fit

    def fit_and_keep(gradients, start=None):
        starts.append(start)
        return original(gradients, start=start)

    fit.fit = fit_and_keep  # type: ignore[method-assign]
    Estimator("trivial", estimates="eddy", registration=fit).run(dwi, n_jobs=1, omp_nthreads=1)

    assert np.allclose(starts[0].matrix, sim.truth.matrix)
    assert abs(starts[0].exponent - 1.0) < 1e-8


@SKIP_ELASTIX
def test_the_eddy_pass_realigns_the_view_once_fitted():
    """With the instant update, the view of the chain is realigned with the fitted mappings."""
    sim = _images(1.0, "j-")
    dwi = _dwi(sim, "j-")
    views: dict = {}
    fit = EddyFieldFit(phase_encoding="j-", clip="none")

    Estimator("avg", estimates="eddy", registration=fit).run(
        dwi, n_jobs=1, omp_nthreads=1, realigned=views
    )

    assert fit.num_threads == 1  # the run's
    assert np.allclose(views[id(dwi)].dataobj, dwi.realigned().dataobj, atol=1e-4)


def test_the_field_fit_estimates_only_the_eddy_currents():
    sim = SimpleNamespace(
        samples=np.zeros((18, 6, 6, 4), dtype="float32"),
        affine=np.diag([2.0, 2.0, 2.0, 1.0]),
        mask=np.ones((6, 6, 4), dtype=bool),
        data=np.zeros((6, 6, 4), dtype="float32"),
        table=_balanced(np.random.default_rng(11)),
        motions=np.repeat(np.eye(4)[np.newaxis], 18, axis=0),
    )
    with pytest.raises(ValueError, match="estimates='eddy', not 'motion'"):
        Estimator("trivial", registration=EddyFieldFit()).run(_dwi(sim, "j"))
