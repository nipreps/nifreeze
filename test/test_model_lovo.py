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
"""Unit tests for closed-form leave-one-volume-out prediction."""

import warnings

import numpy as np
import pytest
from dipy.core.gradients import gradient_table_from_bvals_bvecs
from sklearn.gaussian_process.kernels import WhiteKernel

from nifreeze import model
from nifreeze.data.dmri import DWI
from nifreeze.model import lovo
from nifreeze.model._dipy import GP_JITTER
from nifreeze.model.gpr import DiffusionGPR, SphericalKriging, uses_bvalue
from nifreeze.model.gqi import GeneralizedQSamplingModel
from nifreeze.testing import simulations as _sim


def _spd(n, rng):
    """A random, well-conditioned symmetric positive-definite matrix."""
    factor = rng.standard_normal((n, n + 5))
    return factor @ factor.T / n + 0.1 * np.eye(n)


def _crossing_signals(n_dirs=40, n_voxels=60, seed=1234):
    """Simulated crossing-fiber signals (volumes by voxels) and their gradients."""
    signal, gtab = _sim.simulate_multifiber_voxels(
        S0=100, hsph_dirs=n_dirs, snr=30, n_voxels=n_voxels, seed=seed
    )
    # Drop the initial b=0
    signals = np.asarray(signal)[:n_voxels, 1:].T
    gradients = np.column_stack([gtab.bvecs, gtab.bvals])[1:]
    return signals, gradients


def _crossing_dwi(n_dirs=40, side=6, seed=1234):
    """A small DWI dataset of simulated crossing fibers."""
    signal, gtab = _sim.simulate_multifiber_voxels(
        S0=100, hsph_dirs=n_dirs, snr=30, n_voxels=side**2, seed=seed
    )
    dataobj = np.asarray(signal)[: side**2].reshape(side, side, 1, -1).astype(np.float32)
    return DWI(
        dataobj=dataobj,
        affine=np.eye(4),
        brainmask=np.ones(dataobj.shape[:3], dtype=bool),
        gradients=np.column_stack([gtab.bvecs, gtab.bvals]),
    )


def _random_dwi(setup_random_dwi_data):
    dwi_dataobj, affine, brainmask_dataobj, gradients, _ = setup_random_dwi_data
    return DWI(
        dataobj=dwi_dataobj,
        affine=affine,
        brainmask=brainmask_dataobj,
        gradients=gradients,
    )


def _fixed_gp_kernel():
    return SphericalKriging(beta_a=1.6, beta_l=0.5) + WhiteKernel(noise_level=0.05)


def _hyperparameter_record(kernel, kernel_model):
    """The in-memory equivalent of a file written by ``save_hyperparameters``."""
    names = [hp.name for hp in kernel.hyperparameters if not hp.fixed]
    return {
        "schema": lovo.HYPERPARAMETERS_SCHEMA,
        "kernel_model": kernel_model,
        "hyperparameters": dict(zip(names, np.exp(kernel.theta).tolist(), strict=True)),
    }


@pytest.mark.parametrize("center", [False, True])
def test_lovo_smoother_matches_conditioning(request, center):
    """Each row equals Gaussian conditioning on the other volumes."""
    rng = request.node.rng
    n_volumes = 12
    covariance = _spd(n_volumes, rng)
    signals = rng.standard_normal((n_volumes, 7)) + 5.0

    smoother = lovo.lovo_smoother(covariance, center=center)
    assert np.allclose(np.diag(smoother), 0.0)
    if center:
        assert np.allclose(smoother.sum(axis=1), 1.0)

    for index in range(n_volumes):
        train = np.ones(n_volumes, dtype=bool)
        train[index] = False
        mean = signals[train].mean(axis=0) if center else 0.0
        weights = np.linalg.solve(covariance[np.ix_(train, train)], covariance[train, index])
        expected = mean + weights @ (signals[train] - mean)
        assert np.allclose(smoother[index] @ signals, expected)


@pytest.mark.parametrize("normalize_y", [False, True])
def test_lovo_smoother_matches_gp_refits(normalize_y):
    """The smoother reproduces one fixed-hyperparameter GP refit per held-out volume."""
    signals, gradients = _crossing_signals()
    X = gradients[:, :3]
    kernel = _fixed_gp_kernel()

    covariance = kernel(X)
    covariance[np.diag_indices_from(covariance)] += GP_JITTER
    smoother = lovo.lovo_smoother(covariance, center=normalize_y)

    for index in (0, 13, 27):
        train = np.ones(len(X), dtype=bool)
        train[index] = False
        gpr = DiffusionGPR(
            kernel=kernel,
            optimizer=None,
            alpha=GP_JITTER,
            normalize_y=normalize_y,
            n_targets=signals.shape[1],
        ).fit(X[train], signals[train])
        expected = gpr.predict(X[index : index + 1])[0]
        assert np.allclose(smoother[index] @ signals, expected, rtol=1e-8, atol=1e-8)


def test_lovo_smoother_rejects_singular_and_warns_ill_conditioned():
    """Indefinite covariances raise, and near-singular ones warn."""
    with pytest.raises(ValueError, match="positive definite"):
        lovo.lovo_smoother(np.array([[1.0, 2.0], [2.0, 1.0]]))

    with pytest.warns(RuntimeWarning, match="ill-conditioned"):
        lovo.lovo_smoother(np.diag([1.0, 1.0, 1e-14]))


def test_lovo_smoother_rotation_invariance():
    """A global rotation of the gradient table leaves the smoother unchanged."""
    _, gradients = _crossing_signals()
    X = gradients[:, :3]
    angle = np.deg2rad(37.0)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0, 0, 1]]
    )
    kernel = _fixed_gp_kernel()
    smoother = lovo.lovo_smoother(kernel(X), center=True)
    rotated = lovo.lovo_smoother(kernel(X @ rotation.T), center=True)
    assert np.allclose(smoother, rotated, atol=1e-10)


def test_select_krr_lambda_minimizes_lovo_error(request):
    """The chosen ridge strength minimizes the summed LOVO error over datasets."""
    rng = request.node.rng
    grams = [_spd(10, rng), _spd(10, rng)]
    signals = [rng.standard_normal((10, 5)), rng.standard_normal((10, 8))]
    grid = (1e-3, 1e-2, 1e-1, 1.0)

    chosen = lovo.select_krr_lambda(grams, signals, grid)
    errors = [
        sum(
            lovo.lovo_error(lovo.lovo_smoother(g + lam * np.eye(10)), y)
            for g, y in zip(grams, signals, strict=True)
        )
        for lam in grid
    ]
    assert chosen == grid[int(np.argmin(errors))]


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_fit_pooled_hyperparameters_identical_tables_equals_appending():
    """With identical tables, pooling subjects equals fitting the appended voxels."""
    signals, gradients = _crossing_signals(n_voxels=40)
    X = gradients[:, :3]
    kernel = SphericalKriging() + WhiteKernel()

    pooled = lovo.fit_pooled_hyperparameters(kernel, [X, X], [signals[:, :20], signals[:, 20:]])
    appended = lovo.fit_pooled_hyperparameters(kernel, [X], [signals])
    assert np.allclose(pooled.theta, appended.theta, atol=1e-6)

    # The optimizer moved away from the start and stayed within bounds
    assert not np.allclose(pooled.theta, kernel.theta)
    assert np.all(pooled.theta >= kernel.bounds[:, 0] - 1e-9)
    assert np.all(pooled.theta <= kernel.bounds[:, 1] + 1e-9)


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_fit_pooled_hyperparameters_different_tables():
    """Datasets with their own gradient tables can be pooled."""
    signals_a, gradients_a = _crossing_signals(n_dirs=30, seed=1)
    signals_b, gradients_b = _crossing_signals(n_dirs=40, seed=2)
    kernel = SphericalKriging() + WhiteKernel()

    pooled = lovo.fit_pooled_hyperparameters(
        kernel, [gradients_a[:, :3], gradients_b[:, :3]], [signals_a, signals_b]
    )
    assert np.all(np.isfinite(pooled.theta))


def test_hyperparameters_roundtrip(tmp_path):
    """Hyperparameter files round-trip and are validated on load."""
    kernel = _fixed_gp_kernel()
    filename = lovo.save_hyperparameters(
        tmp_path / "hp.json", kernel, "spherical", dataset="simulated", n_subjects=1
    )

    record = lovo.load_hyperparameters(filename)
    assert record["kernel_model"] == "spherical"
    assert record["provenance"] == {"dataset": "simulated", "n_subjects": 1}

    restored = lovo.kernel_with_hyperparameters(
        SphericalKriging() + WhiteKernel(), record["hyperparameters"]
    )
    assert np.allclose(restored.theta, kernel.theta)

    # A mapping is accepted as well
    assert lovo.load_hyperparameters(record) == record

    with pytest.raises(ValueError, match="do not match"):
        lovo.kernel_with_hyperparameters(SphericalKriging(), record["hyperparameters"])
    with pytest.raises(ValueError, match="schema"):
        lovo.load_hyperparameters({**record, "schema": "other/1"})
    with pytest.raises(ValueError, match="lacks"):
        lovo.load_hyperparameters({"schema": lovo.HYPERPARAMETERS_SCHEMA})


def test_masked_signal_matrix_memmap(tmp_path, request):
    """Signals are gathered volume by volume, also from memory-mapped data."""
    rng = request.node.rng
    data = np.memmap(tmp_path / "data.dat", dtype=np.float32, mode="w+", shape=(3, 4, 5, 6))
    data[:] = rng.random((3, 4, 5, 6))
    mask = rng.random((3, 4, 5)) > 0.5

    class _Dataset:
        dataobj = data

        def __len__(self):
            return data.shape[-1]

    signals = lovo.masked_signal_matrix(_Dataset(), mask)
    assert signals.shape == (6, mask.sum())
    assert np.allclose(signals, np.asarray(data)[mask].T)


@pytest.mark.parametrize(
    ("name", "cls"),
    [
        ("gp-loo", "GPLOOModel"),
        ("GPLOO", "GPLOOModel"),
        ("gqi-krr", "GQIKRRModel"),
        ("gqikrr", "GQIKRRModel"),
    ],
)
def test_factory_closed_form_models(setup_random_dwi_data, name, cls):
    """The factory builds the closed-form models from their names."""
    instance = model.ModelFactory.init(name, dataset=_random_dwi(setup_random_dwi_data))
    assert type(instance).__name__ == cls
    assert instance.single_fit_is_canary is False


def test_gploo_matches_fixed_hyperparameter_refits(setup_random_dwi_data):
    """``GPLOOModel`` predictions equal GP refits with the same hyperparameters."""
    dwi = _random_dwi(setup_random_dwi_data)
    kernel = _fixed_gp_kernel()
    gp = model.dmri.GPLOOModel(dwi, hyperparameters=_hyperparameter_record(kernel, "spherical"))

    mask = gp._data_mask
    X = np.asarray(dwi.gradients)[:, :3]
    signals = np.moveaxis(np.asarray(dwi.dataobj), -1, 0)[:, mask].astype(float)
    for index in (0, len(dwi) // 2):
        train = np.ones(len(dwi), dtype=bool)
        train[index] = False
        gpr = DiffusionGPR(
            kernel=kernel,
            optimizer=None,
            alpha=GP_JITTER,
            n_targets=signals.shape[1],
        ).fit(X[train], signals[train])
        expected = gpr.predict(X[index : index + 1])[0]

        predicted = gp.fit_predict(index)
        assert predicted is not None
        assert predicted.shape == dwi.dataobj.shape[:-1]
        assert np.allclose(predicted[mask], expected, rtol=1e-4, atol=1e-5)
        assert np.all(predicted[~mask] == 0)


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_gploo_fits_hyperparameters_once(setup_random_dwi_data, monkeypatch):
    """Hyperparameters are fitted once, on the voxel subsample, for all volumes."""
    calls = []
    original = model.dmri.fit_pooled_hyperparameters

    def _counting(kernel, tables, signals, **kwargs):
        calls.append(signals[0].shape)
        return original(kernel, tables, signals, **kwargs)

    monkeypatch.setattr(model.dmri, "fit_pooled_hyperparameters", _counting)

    dwi = _random_dwi(setup_random_dwi_data)
    gp = model.dmri.GPLOOModel(dwi, kernel_model="spherical", n_voxels_hp=1)
    for index in range(4):
        assert gp.fit_predict(index) is not None
    assert calls == [(len(dwi), 1)]


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_gploo_single_fit_and_gallery_protocol(setup_random_dwi_data):
    """Single-fit is not a canary, and every call returns the genuine LOVO prediction."""
    dwi = _random_dwi(setup_random_dwi_data)
    reference = model.dmri.GPLOOModel(dwi, kernel_model="spherical")
    expected = reference.fit_predict(1)
    assert expected is not None

    single = model.dmri.GPLOOModel(dwi, kernel_model="spherical")
    with warnings.catch_warnings():
        warnings.simplefilter("error", model.base.SingleFitCanaryWarning)
        assert single.fit_predict(None) is None
    assert single._locked_fit is True
    predicted = single.fit_predict(1)
    assert predicted is not None
    assert np.allclose(predicted, expected)

    # The gallery fits, forces the lock, then predicts
    gallery = model.dmri.GPLOOModel(dwi, kernel_model="spherical")
    gallery._fit(1)
    gallery._locked_fit = True
    predicted = gallery.fit_predict(1)
    assert predicted is not None
    assert np.allclose(predicted, expected)


def test_gploo_kernel_model(setup_random_dwi_data):
    """``kernel_model`` comes from the hyperparameter record and must not contradict it."""
    dwi = _random_dwi(setup_random_dwi_data)
    record = _hyperparameter_record(_fixed_gp_kernel(), "spherical")

    gp = model.dmri.GPLOOModel(dwi, hyperparameters=record)
    assert gp.fit_predict(0) is not None
    assert gp._kernel is not None
    assert np.allclose(gp._kernel.theta, _fixed_gp_kernel().theta)

    gp = model.dmri.GPLOOModel(dwi, kernel_model="exponential", hyperparameters=record)
    with pytest.raises(ValueError, match="kernel_model"):
        gp.fit_predict(0)


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_gploo_multishell_auto_kernel(request):
    """Multi-shell data select the multi-shell kernel when none is requested."""
    rng = request.node.rng
    gtabs = [_sim.create_single_shell_gradient_table(12, b) for b in (1000, 2000)]
    bvals = np.concatenate([gtabs[0].bvals, gtabs[1].bvals[1:]])
    bvecs = np.concatenate([gtabs[0].bvecs, gtabs[1].bvecs[1:]])
    dataobj = (rng.random((3, 3, 2, len(bvals))) + 0.5).astype(np.float32)
    dwi = DWI(
        dataobj=dataobj,
        affine=np.eye(4),
        brainmask=np.ones(dataobj.shape[:3], dtype=bool),
        gradients=np.column_stack([bvecs, bvals]),
    )
    gp = model.dmri.GPLOOModel(dwi, n_voxels_hp=None)
    assert gp.fit_predict(0) is not None
    assert gp._kernel is not None
    assert uses_bvalue(gp._kernel)


def test_gqikrr_matches_explicit_ridge(setup_random_dwi_data):
    """``GQIKRRModel`` rows equal kernel ridge regression on the other volumes."""
    dwi = _random_dwi(setup_random_dwi_data)
    krr = model.dmri.GQIKRRModel(dwi, krr_lambda=0.1)

    gradients = np.asarray(dwi.gradients, dtype=float)
    gtab = gradient_table_from_bvals_bvecs(gradients[:, -1], gradients[:, :-1])
    basis = GeneralizedQSamplingModel(gtab).kernel
    gram = basis @ basis.T
    gram /= np.trace(gram) / len(gram)

    mask = krr._data_mask
    signals = np.moveaxis(np.asarray(dwi.dataobj), -1, 0)[:, mask].astype(float)
    for index in (0, len(dwi) // 2):
        train = np.ones(len(dwi), dtype=bool)
        train[index] = False
        ridge = gram[np.ix_(train, train)] + 0.1 * np.eye(train.sum())
        expected = gram[index, train] @ np.linalg.solve(ridge, signals[train])
        predicted = krr.fit_predict(index)
        assert predicted is not None
        assert np.allclose(predicted[mask], np.maximum(expected, 0), rtol=1e-4, atol=1e-5)


def test_gqikrr_fixes_gqi_amplitude():
    """Unlike the one-row GQI prediction, GQI-KRR predicts the signal amplitude."""
    dwi = _crossing_dwi()
    mask = np.ones(dwi.dataobj.shape[:3], dtype=bool)
    krr = model.dmri.GQIKRRModel(dwi)
    gqi = model.dmri.GQIModel(dwi)

    def nrmse(_model):
        errors, observed = [], []
        for index in (0, 10, 20, 30):
            errors.append(_model.fit_predict(index)[mask] - dwi.dataobj[..., index][mask])
            observed.append(dwi.dataobj[..., index][mask])
        return np.sqrt(np.mean(np.square(errors))) / np.mean(observed)

    assert krr._krr_lambda is None
    assert nrmse(krr) < 0.2
    assert krr._krr_lambda in lovo.DEFAULT_KRR_LAMBDA_GRID
    assert nrmse(krr) < 0.5 * nrmse(gqi)
