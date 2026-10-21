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
"""Sensitivity of model predictions to motion of their training volumes."""

import numpy as np
import pytest
from scipy.ndimage import center_of_mass, gaussian_filter
from sklearn.gaussian_process.kernels import WhiteKernel

from nifreeze.analysis import sensitivity as sens
from nifreeze.analysis.similarity import masked_cc, mutual_information, nrmse
from nifreeze.data.base import BaseDataset
from nifreeze.data.dmri import DWI
from nifreeze.model import lovo
from nifreeze.model.base import ExpectationModel, TrivialModel
from nifreeze.model.dmri import GPLOOModel, GQIKRRModel
from nifreeze.model.gpr import SphericalKriging
from nifreeze.testing import simulations as _sim

ZOOMS = 2.0
SHAPE = (20, 20, 16)


def _smooth_head(rng, shape=SHAPE):
    """A smooth ellipsoid with smooth texture, and a mask of its interior."""
    grid = np.indices(shape, dtype=float)
    center = (np.array(shape) - 1) / 2
    semiaxes = 0.32 * np.array(shape)
    radius = np.sqrt(
        (((grid - center[:, None, None, None]) / semiaxes[:, None, None, None]) ** 2).sum(0)
    )
    texture = gaussian_filter(rng.normal(size=shape), 1.5)
    texture /= texture.std()
    # The edge falls below 1% of the head's intensity at the field of view
    head = gaussian_filter((radius < 1.0).astype(float), 1.0) * 1000.0 * (1.0 + 0.1 * texture)
    return head.astype(np.float32), radius < 0.7


def _copies(rng, n_volumes=9):
    """One b=0 volume, copied ``n_volumes`` times."""
    bzero, mask = _smooth_head(rng)
    dataset: BaseDataset = BaseDataset(
        dataobj=np.repeat(bzero[np.newaxis], n_volumes, axis=0),
        affine=np.diag([ZOOMS, ZOOMS, ZOOMS, 1.0]),
        brainmask=mask,
    )
    return dataset, bzero


def _dwi(rng, n_directions=30):
    """A single-shell DWI with spatially smooth, positive volumes."""
    s0, mask = _smooth_head(rng)
    gtab = _sim.create_single_shell_gradient_table(n_directions, 1000)
    volumes = [
        s0 * (0.3 + 0.1 * gaussian_filter(rng.normal(size=SHAPE), 1.5))
        for _ in range(int((~gtab.b0s_mask).sum()))
    ]
    return DWI(
        dataobj=np.stack(volumes).astype(np.float32),
        affine=np.diag([ZOOMS, ZOOMS, ZOOMS, 1.0]),
        brainmask=mask,
        gradients=np.column_stack((gtab.bvecs, gtab.bvals))[~gtab.b0s_mask].astype(np.float32),
    )


def _gploo_record():
    kernel = SphericalKriging(beta_a=1.6, beta_l=0.5) + WhiteKernel(noise_level=0.05)
    values = np.exp(kernel.theta).tolist()
    return {
        "schema": lovo.HYPERPARAMETERS_SCHEMA,
        "kernel_model": "spherical",
        "hyperparameters": dict(zip(lovo._hyperparameter_names(kernel), values, strict=True)),
    }


# Similarity metrics


def test_metrics_of_a_perfect_prediction():
    """CC is invariant to positive affine maps; MI of a volume with itself is its entropy."""
    volume, mask = _smooth_head(np.random.default_rng(1234))

    assert masked_cc(3.0 * volume + 7.0, volume, mask) == pytest.approx(1.0)
    assert nrmse(volume, volume, mask) == 0.0

    counts, _ = np.histogram(volume[mask], bins=32)
    p = counts[counts > 0] / counts.sum()
    assert mutual_information(volume, volume, mask) == pytest.approx(-(p * np.log(p)).sum())


def test_metrics_ignore_voxels_outside_the_mask():
    volume, mask = _smooth_head(np.random.default_rng(1234))
    corrupted = volume.copy()
    corrupted[~mask] = -1e6

    assert masked_cc(corrupted, volume, mask) == pytest.approx(1.0)
    assert nrmse(corrupted, volume, mask) == 0.0


def test_metrics_degrade_with_noise():
    rng = np.random.default_rng(1234)
    volume, mask = _smooth_head(rng)
    noise = rng.normal(size=volume.shape) * volume[mask].std()
    noisy = [volume + level * noise for level in (0.1, 0.5, 2.0)]

    assert np.all(np.diff([masked_cc(v, volume, mask) for v in noisy]) < 0)
    assert np.all(np.diff([mutual_information(v, volume, mask) for v in noisy]) < 0)
    assert np.all(np.diff([nrmse(v, volume, mask) for v in noisy]) > 0)


def test_nrmse_rejects_a_constant_reference():
    with pytest.raises(ValueError, match="constant"):
        nrmse(np.ones(10), np.ones(10))


# Perturbations


def test_perturb_moves_the_head_and_nothing_else():
    """A 3 mm shift along x moves the head's center of mass by 3 mm along x."""
    dataset, bzero = _copies(np.random.default_rng(1234))
    shift = np.eye(4)
    shift[0, 3] = 3.0
    original = np.array(dataset.dataobj)

    moved = sens.perturb(dataset, sens.Perturbation(3.0, None, (2,), (shift,)))

    displacement = np.subtract(center_of_mass(moved.dataobj[2]), center_of_mass(bzero)) * ZOOMS
    assert displacement == pytest.approx([3.0, 0.0, 0.0], abs=0.05)
    others = np.arange(len(dataset)) != 2
    assert np.array_equal(moved.dataobj[others], original[others])
    assert np.array_equal(dataset.dataobj, original)


def test_round_trip_leaves_the_volume_in_place():
    dataset, bzero = _copies(np.random.default_rng(1234))
    motion = sens.rigid_motion(2.0, (1, 1, 0), (0, 1, 1), sens.mask_centroid(dataset))
    perturbation = sens.Perturbation(2.0, None, (0,), (motion,))

    moved = sens.perturb(dataset, perturbation)
    back = sens.perturb(dataset, perturbation, round_trip=True)

    mask = dataset.brainmask
    assert nrmse(back.dataobj[0], bzero, mask) < 0.1 * nrmse(moved.dataobj[0], bzero, mask)


def test_shift_grid_keeps_the_anatomy_in_place():
    """Only the sampling grid moves: the head's center of mass stays put in world coordinates."""
    dataset, bzero = _copies(np.random.default_rng(1234))

    shifted = sens.shift_grid(dataset, 0.25)

    assert np.allclose(shifted.affine[:3, 3] - dataset.affine[:3, 3], 0.25 * ZOOMS)
    world = [
        np.asarray(center_of_mass(volume)) * ZOOMS + origin
        for volume, origin in (
            (bzero, dataset.affine[:3, 3]),
            (shifted.dataobj[0], shifted.affine[:3, 3]),
        )
    ]
    assert world[1] == pytest.approx(world[0], abs=0.02)
    assert np.array_equal(shifted.dataobj[0], shifted.dataobj[-1])
    assert shifted.brainmask.sum() == pytest.approx(dataset.brainmask.sum(), rel=0.05)


def test_grid_smooths_moved_and_unmoved_volumes_alike():
    """On the shifted grid, moved and unmoved volumes keep the same noise variance."""
    rng = np.random.default_rng(1234)
    shape = (40, 40, 32)
    dataset: BaseDataset = BaseDataset(
        dataobj=rng.normal(size=(4, *shape)).astype(np.float32),
        affine=np.diag([ZOOMS, ZOOMS, ZOOMS, 1.0]),
        brainmask=np.ones(shape, dtype=bool),
    )
    plan = sens.one_at_a_time([1, 2, 3], 2.0, center=sens.mask_centroid(dataset))
    inner = (slice(None), *(slice(8, -8),) * 3)

    def variances(grid):
        moved = [sens.perturb(dataset, p, grid=grid).dataobj[inner] for p in plan]
        unmoved = moved[0][0].var()
        return unmoved, np.mean([m[p.moved[0]].var() for m, p in zip(moved, plan, strict=True)])

    unmoved, moved = variances(None)
    assert unmoved / moved > 1.3  # Without the grid, resampling smooths only the moved ones
    unmoved, moved = variances(sens.shift_grid(dataset))
    assert unmoved / moved == pytest.approx(1.0, abs=0.08)

    with pytest.raises(ValueError, match="round trip"):
        sens.perturb(dataset, plan[0], round_trip=True, grid=sens.shift_grid(dataset))


def test_plan_never_moves_fixed_volumes_and_pairs_its_cells():
    fixed = [0, 5]
    plan = sens.perturbation_plan(12, magnitudes=(0.5, 2.0), counts=(1, 3), seeds=3, fixed=fixed)

    assert len(plan) == 3 * 2 * 2
    assert not any(set(fixed) & set(p.moved) for p in plan)
    for seed in range(3):
        cells = [p for p in plan if p.seed == seed]
        small, large = (p for p in cells if p.magnitude == 0.5)
        assert set(small.moved) < set(large.moved)
        # The same volume follows the same axis at every magnitude
        weak, strong = (p for p in cells if p.count == 1)
        assert np.allclose(4 * _rotvec(weak.motions[0]), _rotvec(strong.motions[0]))

    with pytest.raises(ValueError, match="Cannot move"):
        sens.perturbation_plan(12, magnitudes=(1.0,), counts=(11,), fixed=fixed)


def test_a_volume_motion_never_reaches_its_own_prediction():
    """Moving only volume j changes every prediction but that of volume j."""
    dataset, _ = _copies(np.random.default_rng(1234))
    plan = sens.one_at_a_time([3], 2.0, center=sens.mask_centroid(dataset))

    rows = sens.sensitivity(ExpectationModel, dataset, plan, stat="mean")

    assert len(rows) == len(dataset)
    for row in rows:
        assert row["moved"] == (row["index"] == 3)
        assert (row["nrmse"] == 0.0) == row["moved"]


def _rotvec(motion):
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(motion[:3, :3]).as_rotvec()


# Controls


def test_trivial_on_copies_is_perfect_under_any_plan():
    """The trivial model predicts the b=0 its training copies came from, wherever they move."""
    dataset, bzero = _copies(np.random.default_rng(1234))
    plan = sens.perturbation_plan(
        len(dataset),
        magnitudes=(0.5, 2.0),
        counts=(1, 4),
        seeds=2,
        center=sens.mask_centroid(dataset),
    )
    # Every volume is compared on the shifted grid
    grid = sens.shift_grid(dataset)

    aligned = sens.fidelity(TrivialModel, grid, predicted=grid.dataobj[0])
    rows = sens.sensitivity(TrivialModel, dataset, plan, predicted=grid.dataobj[0])

    entropy = mutual_information(grid.dataobj[0], grid.dataobj[0], grid.brainmask)
    assert len(aligned) == len(dataset)
    for row in aligned:
        assert row["cc"] == pytest.approx(1.0)
        assert row["nrmse"] == 0.0
        assert row["mi"] == pytest.approx(entropy)
    assert len(rows) == len(plan) * len(dataset)
    assert all(row["nrmse"] == 0.0 for row in rows)

    # The control can fail: a 1 mm motion of the held-out volume is visible
    motion = sens.rigid_motion(1.0, (0, 0, 1), (1, 0, 0), sens.mask_centroid(dataset))
    moved = sens.perturb(dataset, sens.Perturbation(1.0, None, (0,), (motion,)))
    assert masked_cc(bzero, moved.dataobj[0], dataset.brainmask) < 0.99
    assert nrmse(bzero, moved.dataobj[0], dataset.brainmask) > 0.1


def test_mean_of_copies_matches_its_closed_form():
    """The mean of copies has a closed form; the median ignores a moved minority."""
    dataset, _ = _copies(np.random.default_rng(1234))
    n_volumes, evaluate = len(dataset), [0]
    plan = sens.perturbation_plan(
        n_volumes,
        magnitudes=(1.0,),
        counts=(1, 3, 6),
        seeds=2,
        center=sens.mask_centroid(dataset),
        fixed=evaluate,
    )

    grid = sens.shift_grid(dataset)
    rows = sens.sensitivity(ExpectationModel, dataset, plan, evaluate, stat="mean")
    for perturbation, row in zip(plan, rows, strict=True):
        moved = sens.perturb(dataset, perturbation, grid=grid)
        expected = (
            (n_volumes - 1 - perturbation.count) * grid.dataobj[0]
            + moved.dataobj[list(perturbation.moved)].sum(axis=0)
        ) / (n_volumes - 1)
        predicted = sens.held_out_predictions(ExpectationModel, moved, evaluate, stat="mean")
        assert np.allclose(predicted[0], expected, rtol=1e-5, atol=1e-3)
        assert row["nrmse"] == pytest.approx(
            nrmse(expected, grid.dataobj[0], grid.brainmask), rel=1e-4
        )

    rows = sens.sensitivity(ExpectationModel, dataset, plan, evaluate, stat="median")
    for row in rows:
        if row["count"] < (n_volumes - 1) / 2:
            assert row["nrmse"] == 0.0
        else:
            assert row["nrmse"] > 0.0


def test_change_is_normalized_by_the_actual_volume():
    """The NRMSE divides the change by the actual volume's spread, not the prediction's."""
    dataset = _dwi(np.random.default_rng(1234))
    evaluate = [0, 5]
    plan = sens.perturbation_plan(
        len(dataset), magnitudes=(2.0,), counts=(8,), seeds=1, center=sens.mask_centroid(dataset)
    )

    rows = sens.sensitivity(ExpectationModel, dataset, plan, evaluate, stat="mean")

    grid = sens.shift_grid(dataset)
    mask = grid.brainmask
    aligned = sens.held_out_predictions(ExpectationModel, grid, evaluate, stat="mean")
    moved = sens.perturb(dataset, plan[0], grid=grid)
    perturbed = sens.held_out_predictions(ExpectationModel, moved, evaluate, stat="mean")
    for row in rows:
        index = row["index"]
        change = np.sqrt(np.mean((perturbed[index] - aligned[index])[mask] ** 2))
        assert row["nrmse"] == pytest.approx(change / grid.dataobj[index][mask].std(), rel=1e-6)
        # The mean of the others is smoother than any actual volume
        assert row["nrmse"] < nrmse(perturbed[index], aligned[index], mask)


@pytest.mark.parametrize(
    ("model", "kwargs"),
    [
        (GPLOOModel, {"hyperparameters": _gploo_record()}),
        (GQIKRRModel, {"krr_lambda": 0.1}),
    ],
)
def test_linear_smoother_change_is_exact(model, kwargs):
    """Moving training volumes changes each prediction by ``S @ ΔY``."""
    dataset = _dwi(np.random.default_rng(1234))
    evaluate = [0, 7, 15]
    perturbation = sens.perturbation_plan(
        len(dataset),
        magnitudes=(2.0,),
        counts=(4,),
        seeds=1,
        center=sens.mask_centroid(dataset),
        fixed=evaluate,
    )[0]
    moved = sens.perturb(dataset, perturbation)

    before = sens.held_out_predictions(
        model, sens.perturb(dataset, sens.NO_PERTURBATION), evaluate, **kwargs
    )
    after = sens.held_out_predictions(model, moved, evaluate, **kwargs)

    instance = model(dataset, **kwargs)
    mask = instance._data_mask
    change = np.asarray(moved.dataobj, dtype=float)[:, mask] - np.asarray(dataset.dataobj)[:, mask]
    expected = instance.smoother @ change
    for row, index in enumerate(evaluate):
        delta = after[index][mask] - before[index][mask]
        # GQI-KRR clips negative signals; it is linear where both predictions are positive
        linear = (after[index][mask] > 0) & (before[index][mask] > 0)
        assert linear.mean() > 0.9
        assert np.allclose(delta[linear], expected[index][linear], rtol=1e-4, atol=1e-2)
        assert np.abs(expected[index]).max() > 1.0, f"row {row} saw no change"


def test_influence_is_the_norm_of_the_smoother_column():
    """Moving volume j alone changes all predictions by ``‖S[:, j]‖ ‖Δy_j‖`` in total."""
    dataset = _dwi(np.random.default_rng(1234))
    record = _gploo_record()
    instance = GPLOOModel(dataset, hyperparameters=record)
    mask = instance._data_mask
    weights = sens.influence(instance)
    j = int(np.argmax(weights))
    evaluate = [i for i in range(len(dataset)) if i != j]

    perturbation = sens.one_at_a_time([j], 2.0, center=sens.mask_centroid(dataset))[0]
    moved = sens.perturb(dataset, perturbation)
    before = sens.held_out_predictions(GPLOOModel, dataset, evaluate, hyperparameters=record)
    after = sens.held_out_predictions(GPLOOModel, moved, evaluate, hyperparameters=record)

    total = np.sqrt(sum(np.sum((after[i][mask] - before[i][mask]) ** 2.0) for i in evaluate))
    delta_j = np.linalg.norm(moved.dataobj[j][mask] - np.asarray(dataset.dataobj[j])[mask])
    assert total == pytest.approx(weights[j] * delta_j, rel=1e-3)


def test_smoother_is_read_only_and_does_not_lock_the_fit():
    instance = GPLOOModel(_dwi(np.random.default_rng(1234)), hyperparameters=_gploo_record())

    smoother = instance.smoother

    assert instance._locked_fit is None
    assert np.allclose(np.diag(smoother), 0.0)
    with pytest.raises(ValueError, match="read-only"):
        smoother[0, 1] = 1.0
