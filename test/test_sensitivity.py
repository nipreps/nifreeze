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
"""Sensitivity of model predictions to motion of their training volumes, and its test helpers."""

import attrs
import nibabel as nb
import numpy as np
import pytest
from nitransforms.linear import Affine
from nitransforms.resampling import apply
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
from nifreeze.testing import perturbations as pert
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


def _cells(dataset, plan):
    """Each perturbation of a plan, with the copy of ``dataset`` it makes."""
    return [(perturbation, pert.perturb(dataset, perturbation)) for perturbation in plan]


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


def test_nrmse_rejects_a_non_positive_scale():
    with pytest.raises(ValueError, match="positive"):
        nrmse(np.arange(3.0), np.arange(3.0), scale=0.0)


def test_metrics_reject_volumes_of_different_shapes():
    with pytest.raises(ValueError, match="Shapes differ"):
        masked_cc(np.ones(3), np.ones(4))


# Perturbation helpers (nifreeze.testing.perturbations)


def test_perturb_moves_the_head_and_nothing_else():
    """A 3 mm shift along x moves the head's center of mass by 3 mm along x."""
    dataset, bzero = _copies(np.random.default_rng(1234))
    shift = np.eye(4)
    shift[0, 3] = 3.0
    original = np.array(dataset.dataobj)

    moved = pert.perturb(dataset, sens.Perturbation(3.0, None, (2,), (shift,)))

    displacement = np.subtract(center_of_mass(moved.dataobj[2]), center_of_mass(bzero)) * ZOOMS
    assert displacement == pytest.approx([3.0, 0.0, 0.0], abs=0.05)
    others = np.arange(len(dataset)) != 2
    assert np.array_equal(moved.dataobj[others], original[others])
    assert np.array_equal(dataset.dataobj, original)


def test_plan_never_moves_fixed_volumes_and_pairs_its_cells():
    fixed = [0, 5]
    plan = pert.perturbation_plan(12, magnitudes=(0.5, 2.0), counts=(1, 3), seeds=3, fixed=fixed)

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
        pert.perturbation_plan(12, magnitudes=(1.0,), counts=(11,), fixed=fixed)


def test_a_volume_motion_never_reaches_its_own_prediction():
    """Moving only volume j changes every prediction but that of volume j."""
    dataset, _ = _copies(np.random.default_rng(1234))
    plan = pert.one_at_a_time([3], 2.0, center=pert.mask_centroid(dataset))

    rows = sens.sensitivity(ExpectationModel, dataset, _cells(dataset, plan), stat="mean")

    assert len(rows) == len(dataset)
    for row in rows:
        assert row["moved"] == (row["index"] == 3)
        assert (row["nrmse"] == 0.0) == row["moved"]


def test_models_can_be_named():
    """A model name goes through the factory and predicts as the model class does."""
    dataset = _dwi(np.random.default_rng(1234))
    mask = dataset.brainmask

    by_name = sens.held_out_predictions("mean", dataset, [0, 3], stat="mean")
    by_class = sens.held_out_predictions(ExpectationModel, dataset, [0, 3], stat="mean")

    for index in (0, 3):
        others = np.delete(np.asarray(dataset.dataobj), index, axis=0).mean(axis=0)
        assert np.allclose(by_name[index][mask], others[mask], rtol=1e-5)
        assert np.array_equal(by_name[index], by_class[index])


def test_metrics_cover_the_given_mask_else_the_brain_mask_else_everything():
    """A prediction wrong only outside the brain is perfect within the brain mask alone."""
    dataset, bzero = _copies(np.random.default_rng(1234))
    corrupted = bzero.copy()
    corrupted[~dataset.brainmask] += 100.0
    unmasked: BaseDataset = BaseDataset(dataobj=dataset.dataobj, affine=dataset.affine)

    def error(data, **kwargs):
        return sens.fidelity(TrivialModel, data, [0], predicted=corrupted, **kwargs)[0]["nrmse"]

    everything = np.ones(dataset.shape3d, dtype=bool)
    assert error(dataset) == 0.0
    assert error(dataset, mask=everything) > 0.0
    with pytest.warns(UserWarning, match="No mask provided"):
        assert error(unmasked) == error(dataset, mask=everything)


def _rotvec(motion):
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(motion[:3, :3]).as_rotvec()


# Controls


def test_trivial_on_copies_is_perfect_under_any_plan():
    """The trivial model predicts the b=0 its training copies came from, wherever they move."""
    dataset, bzero = _copies(np.random.default_rng(1234))
    plan = pert.perturbation_plan(
        len(dataset),
        magnitudes=(0.5, 2.0),
        counts=(1, 4),
        seeds=2,
        center=pert.mask_centroid(dataset),
    )
    aligned = sens.fidelity(TrivialModel, dataset, predicted=bzero)
    rows = sens.sensitivity(TrivialModel, dataset, _cells(dataset, plan), predicted=bzero)

    entropy = mutual_information(bzero, bzero, dataset.brainmask)
    assert len(aligned) == len(dataset)
    for row in aligned:
        assert row["cc"] == pytest.approx(1.0)
        assert row["nrmse"] == 0.0
        assert row["mi"] == pytest.approx(entropy)
    assert len(rows) == len(plan) * len(dataset)
    assert all(row["nrmse"] == 0.0 for row in rows)

    # The control can fail: a 1 mm motion of the held-out volume is visible
    motion = pert.rigid_motion(1.0, (0, 0, 1), (1, 0, 0), pert.mask_centroid(dataset))
    moved = pert.perturb(dataset, sens.Perturbation(1.0, None, (0,), (motion,)))
    assert masked_cc(bzero, moved.dataobj[0], dataset.brainmask) < 0.99
    assert nrmse(bzero, moved.dataobj[0], dataset.brainmask) > 0.1


def test_mean_of_copies_matches_its_closed_form():
    """The mean of copies has a closed form; the median ignores a moved minority."""
    dataset, _ = _copies(np.random.default_rng(1234))
    n_volumes, evaluate = len(dataset), [0]
    plan = pert.perturbation_plan(
        n_volumes,
        magnitudes=(1.0,),
        counts=(1, 3, 6),
        seeds=2,
        center=pert.mask_centroid(dataset),
        fixed=evaluate,
    )

    cells = _cells(dataset, plan)
    rows = sens.sensitivity(ExpectationModel, dataset, cells, evaluate, stat="mean")
    for (perturbation, moved), row in zip(cells, rows, strict=True):
        expected = (
            (n_volumes - 1 - perturbation.count) * dataset.dataobj[0]
            + moved.dataobj[list(perturbation.moved)].sum(axis=0)
        ) / (n_volumes - 1)
        predicted = sens.held_out_predictions(ExpectationModel, moved, evaluate, stat="mean")
        assert np.allclose(predicted[0], expected, rtol=1e-5, atol=1e-3)
        assert row["nrmse"] == pytest.approx(
            nrmse(expected, dataset.dataobj[0], dataset.brainmask), rel=1e-4
        )

    rows = sens.sensitivity(ExpectationModel, dataset, cells, evaluate, stat="median")
    for row in rows:
        if row["count"] < (n_volumes - 1) / 2:
            assert row["nrmse"] == 0.0
        else:
            assert row["nrmse"] > 0.0


def test_change_is_normalized_by_the_actual_volume():
    """The NRMSE divides the change by the actual volume's spread, not the prediction's."""
    dataset = _dwi(np.random.default_rng(1234))
    evaluate = [0, 5]
    plan = pert.perturbation_plan(
        len(dataset), magnitudes=(2.0,), counts=(8,), seeds=1, center=pert.mask_centroid(dataset)
    )

    cells = _cells(dataset, plan)
    rows = sens.sensitivity(ExpectationModel, dataset, cells, evaluate, stat="mean")

    mask = dataset.brainmask
    aligned = sens.held_out_predictions(ExpectationModel, dataset, evaluate, stat="mean")
    perturbed = sens.held_out_predictions(ExpectationModel, cells[0][1], evaluate, stat="mean")
    for row in rows:
        index = row["index"]
        change = np.sqrt(np.mean((perturbed[index] - aligned[index])[mask] ** 2))
        assert row["nrmse"] == pytest.approx(change / dataset.dataobj[index][mask].std(), rel=1e-6)
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
    perturbation = pert.perturbation_plan(
        len(dataset),
        magnitudes=(2.0,),
        counts=(4,),
        seeds=1,
        center=pert.mask_centroid(dataset),
        fixed=evaluate,
    )[0]
    moved = pert.perturb(dataset, perturbation)

    before = sens.held_out_predictions(
        model, pert.perturb(dataset, sens.NO_PERTURBATION), evaluate, **kwargs
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

    perturbation = pert.one_at_a_time([j], 2.0, center=pert.mask_centroid(dataset))[0]
    moved = pert.perturb(dataset, perturbation)
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


# Eddy-current distortions


def _table(rng, n_volumes, bval=1000.0):
    """Random unit gradients at one b-value."""
    vectors = rng.normal(size=(n_volumes, 3))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.column_stack((vectors, np.full(n_volumes, bval)))


def _epi(distortion, n_lines=SHAPE[1], spacing=5e-4):
    """The distortion formed by an EPI readout with evenly spaced lines."""
    times = (n_lines // 2 - np.arange(n_lines)) * spacing
    return attrs.evolve(distortion, readout=times, readout_time=n_lines * spacing)


def _only(distortion, index, coefficients):
    """The distortion of volume ``index`` replaced by ``coefficients``; others undistorted."""
    table = np.zeros_like(distortion.coefficients)
    table[index] = coefficients
    return attrs.evolve(distortion, coefficients=table)


def test_rigid_resampling_matches_nitransforms():
    dataset, bzero = _copies(np.random.default_rng(1234))
    motion = pert.rigid_motion(3.0, (1, 2, 0), (0, 1, 1), pert.mask_centroid(dataset))
    image = nb.Nifti1Image(bzero, dataset.affine)

    moved = pert.perturb(dataset, sens.Perturbation(3.0, None, (0,), (motion,)))

    expected = apply(Affine(np.linalg.inv(motion)), image, reference=image)
    assert np.allclose(moved.dataobj[0], np.asanyarray(expected.dataobj), atol=1e-2)


def test_eddy_distortion_follows_the_gradient():
    """Opposite gradients give opposite distortions; amplitude grows as √b; b=0 is untouched."""
    dataset, _ = _copies(np.random.default_rng(1234), n_volumes=5)
    vector = np.array([0.6, 0.0, 0.8])
    table = np.array(
        [[*vector, 1000.0], [*-vector, 1000.0], [*vector, 4000.0], [0, 0, 0, 0], [0, 1, 0, 4000]]
    )

    distortion = pert.eddy_distortion(dataset, 2.0, axis=1, degree=2, gradients=table)

    coefficients = distortion.coefficients
    assert np.allclose(coefficients[1], -coefficients[0])
    assert np.allclose(coefficients[2], 2.0 * coefficients[0])
    assert not coefficients[3].any()
    points = nb.affines.apply_affine(dataset.affine, np.argwhere(dataset.brainmask)).T
    values, _ = distortion.basis(points)
    assert np.abs(coefficients @ values).max() == pytest.approx(2.0)


def test_distortions_need_diffusion_gradients():
    dataset, _ = _copies(np.random.default_rng(1234), n_volumes=3)
    with pytest.raises(ValueError, match="diffusion gradient"):
        pert.eddy_distortion(dataset, 1.0, axis=1, gradients=np.zeros((3, 4)))


def test_constant_distortion_is_a_translation_along_phase_encoding():
    dataset, _ = _copies(np.random.default_rng(1234))
    template = pert.eddy_distortion(
        dataset, 1.0, axis=1, gradients=_table(np.random.default_rng(0), 9)
    )
    distortion = _only(template, 0, [1.5, 0.0, 0.0, 0.0])

    distorted = pert.perturb(dataset, sens.Perturbation(0.0, None, distortion=distortion))

    # The scanner shows what lies 1.5 mm further along y: the head appears 1.5 mm back
    shift = np.eye(4)
    shift[1, 3] = -1.5
    moved = pert.perturb(dataset, sens.Perturbation(1.5, None, (0,), (shift,)))
    assert np.allclose(distorted.dataobj[0], moved.dataobj[0], atol=1e-3)
    assert np.array_equal(distorted.dataobj[1:], dataset.dataobj[1:])


def test_distortion_follows_the_motion_in_the_scanner_frame():
    """The head moves, then the scanner distorts: a constant distortion acts after the motion."""
    dataset, _ = _copies(np.random.default_rng(1234))
    template = pert.eddy_distortion(
        dataset, 1.0, axis=1, gradients=_table(np.random.default_rng(0), 9)
    )
    motion = pert.rigid_motion(10.0, (1, 0, 1), (0, 0, 1), pert.mask_centroid(dataset))

    both = pert.perturb(
        dataset,
        sens.Perturbation(
            10.0, None, (0,), (motion,), distortion=_only(template, 0, [1.5, 0, 0, 0])
        ),
    )

    shift = np.eye(4)
    shift[1, 3] = -1.5
    after = pert.perturb(dataset, sens.Perturbation(10.0, None, (0,), (shift @ motion,)))
    before = pert.perturb(dataset, sens.Perturbation(10.0, None, (0,), (motion @ shift,)))
    assert np.allclose(both.dataobj[0], after.dataobj[0], atol=1e-3)
    assert not np.allclose(both.dataobj[0], before.dataobj[0], atol=1.0)


def test_distortion_moves_signal_along_phase_encoding_and_conserves_it():
    """A shear leaves the profiles across phase encoding intact; a stretch keeps the total."""
    dataset, bzero = _copies(np.random.default_rng(1234))
    template = pert.eddy_distortion(
        dataset, 1.0, axis=1, gradients=_table(np.random.default_rng(0), 9)
    )
    shear = _only(template, 0, [0.0, 2.0, 0.0, 0.0])
    stretch = _only(template, 0, [0.0, 0.0, 2.0, 0.0])

    sheared = pert.perturb(dataset, sens.Perturbation(0.0, None, distortion=shear)).dataobj[0]
    stretched = pert.perturb(dataset, sens.Perturbation(0.0, None, distortion=stretch)).dataobj[0]

    assert np.allclose(sheared.sum(axis=(1, 2)), bzero.sum(axis=(1, 2)), rtol=1e-3, atol=1.0)
    assert np.allclose(sheared.sum(axis=(0, 1)), bzero.sum(axis=(0, 1)), rtol=1e-3, atol=1.0)
    assert not np.allclose(sheared.sum(axis=(0, 2)), bzero.sum(axis=(0, 2)), rtol=1e-2)
    assert stretched.sum() == pytest.approx(bzero.sum(), rel=1e-3)
    y = np.arange(bzero.shape[1]) - (bzero.shape[1] - 1) / 2

    def spread(volume):
        profile = volume.sum(axis=(0, 2))
        return np.sqrt((profile * y**2).sum() / profile.sum())

    assert spread(stretched) < 0.95 * spread(bzero)


def test_an_epi_readout_with_a_constant_field_shifts_whole_voxels():
    """With evenly spaced lines, a field worth two voxels shows what lies two voxels further."""
    dataset, bzero = _copies(np.random.default_rng(1234))
    epi = _epi(
        pert.eddy_distortion(dataset, 1.0, axis=1, gradients=_table(np.random.default_rng(0), 9))
    )

    formed = epi.distort(bzero, dataset.affine, [2 * ZOOMS, 0.0, 0.0, 0.0])

    assert np.allclose(formed, np.roll(bzero, -2, axis=1), atol=1e-6 * bzero.max())


def test_an_epi_readout_and_resampling_agree_on_small_distortions():
    """Both move the signal the same way along every term; they differ in how they interpolate.

    At 0.1 mm they agree within 10% of the change (4% measured), and a term of
    the wrong sign would differ by twice the change.
    """
    dataset, bzero = _copies(np.random.default_rng(1234))
    distortion = pert.eddy_distortion(
        dataset, 0.1, axis=1, gradients=_table(np.random.default_rng(0), 9)
    )

    for coefficients in distortion.coefficients:
        resampled = distortion.distort(bzero, dataset.affine, coefficients)
        formed = _epi(distortion).distort(bzero, dataset.affine, coefficients)
        change = np.abs(resampled - bzero).max()
        assert np.abs(formed - resampled).max() < 0.1 * change


@pytest.mark.parametrize("readout", [False, True])
def test_no_distortion_leaves_a_volume_exactly_as_it_is(readout):
    """Zero coefficients skip the model, which would add its interpolation error."""
    dataset, bzero = _copies(np.random.default_rng(1234))
    distortion = pert.eddy_distortion(
        dataset, 1.0, axis=1, gradients=_table(np.random.default_rng(0), 9)
    )
    if readout:
        distortion = _epi(distortion)

    assert np.array_equal(distortion.distort(bzero, dataset.affine, np.zeros(4)), bzero)


def test_an_epi_readout_needs_its_line_times():
    dataset, bzero = _copies(np.random.default_rng(1234))
    distortion = pert.eddy_distortion(
        dataset, 1.0, axis=1, gradients=_table(np.random.default_rng(0), 9)
    )
    coefficients = distortion.coefficients[0]

    with pytest.raises(ValueError, match="readout has 19 lines, the grid 20"):
        _epi(distortion, n_lines=19).distort(bzero, dataset.affine, coefficients)
    with pytest.raises(ValueError, match="total readout time"):
        attrs.evolve(_epi(distortion), readout_time=None).distort(
            bzero, dataset.affine, coefficients
        )


def test_a_distortion_reads_from_a_phantom_sidecar():
    """The sidecar lists some volumes; they land at their indices, the others stay undistorted."""
    eddy = {
        "Magnitude": 2.0,
        "MagnitudeMm": 3.6,
        "TotalReadoutTime": 0.04,
        "ReadoutTimes": [0.01, 0.0, -0.01],
        "Distortion": {
            "Axis": 1,
            "Degree": 1,
            "Center": [1.0, 2.0, 3.0],
            "Radius": [1.8, 1.8, 1.8],
            "Frame": np.eye(3).tolist(),
            "Coefficients": [[0.0, 1.0, 2.0, 3.0], [0.0, -1.0, 0.0, 1.0]],
        },
    }

    distortion = sens.Distortion.from_sidecar(eddy, [3, 0], n_volumes=5)

    assert distortion.coefficients.tolist() == [
        [0.0, -1.0, 0.0, 1.0],
        [0.0] * 4,
        [0.0] * 4,
        [0.0, 1.0, 2.0, 3.0],
        [0.0] * 4,
    ]
    assert (distortion.magnitude, distortion.axis, distortion.degree) == (3.6, 1, 1)
    assert np.asarray(distortion.readout).tolist() == [0.01, 0.0, -0.01]
    assert distortion.readout_time == 0.04
    with pytest.raises(ValueError, match="lists 2 volumes, but 1 indices"):
        sens.Distortion.from_sidecar(eddy, [3], n_volumes=5)


def test_copies_load_when_needed_and_rows_give_the_shell():
    """A copy given as a loader is loaded once, and analyzed as if it had been given."""
    dataset = _dwi(np.random.default_rng(1234))
    perturbation = sens.Perturbation(
        0.0, None, distortion=pert.eddy_distortion(dataset, 0.5, axis=1)
    )
    copy = pert.perturb(dataset, perturbation)
    loads: list[None] = []

    def load():
        loads.append(None)
        return copy

    lazy = sens.sensitivity(ExpectationModel, dataset, [(perturbation, load)], [0, 5], stat="mean")
    given = sens.sensitivity(
        ExpectationModel, dataset, [(perturbation, copy)], [0, 5], stat="mean"
    )

    assert len(loads) == 1
    assert lazy == given
    assert [row["shell"] for row in lazy] == [1000.0, 1000.0]


@pytest.mark.parametrize("readout", [False, True])
def test_distortions_move_the_mean_of_the_others_but_not_the_trivial_model(readout):
    """The mean of the others drifts with their distortions; the trivial model does not."""
    dataset, _ = _copies(np.random.default_rng(1234))
    distortion = pert.eddy_distortion(
        dataset, 0.1, axis=1, gradients=_table(np.random.default_rng(0), len(dataset))
    )
    if readout:
        distortion = _epi(distortion)
    cells = _cells(dataset, [sens.Perturbation(0.0, None, distortion=distortion)])

    rows = sens.sensitivity(ExpectationModel, dataset, cells, stat="mean")

    assert all(row["distortion"] == 0.1 and row["nrmse"] > 0.0 for row in rows)
    # Copies of one b=0 volume have no gradients
    assert all(np.isnan(row["shell"]) for row in rows)

    rows = sens.sensitivity(TrivialModel, dataset, cells, predicted=dataset.dataobj[0])
    assert all(row["nrmse"] == 0.0 for row in rows)


def test_a_distortion_on_top_of_motions_is_analyzed_alone():
    """A moved and distorted copy needs no other cell: one model for it, one for the reference."""
    dataset, _ = _copies(np.random.default_rng(1234))
    distortion = pert.eddy_distortion(
        dataset, 0.1, axis=1, gradients=_table(np.random.default_rng(0), len(dataset))
    )
    still = pert.one_at_a_time([2], 1.0, center=pert.mask_centroid(dataset))[0]
    both = attrs.evolve(still, distortion=distortion)
    built = []

    def model(data, **kwargs):
        built.append(data)
        return ExpectationModel(data, **kwargs)

    rows = sens.sensitivity(model, dataset, _cells(dataset, [both]), stat="mean")

    assert len(built) == 2
    assert len(rows) == len(dataset)
    assert all(row["distortion"] == 0.1 for row in rows)
