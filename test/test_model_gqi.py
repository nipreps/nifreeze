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
"""Unit and analytical-validity tests for the NiFreeze GQI model.

The kernel-algebra and prediction API tests are ported from the tests written
by @MadeInShineA for DIPY PR dipy/dipy#3553 (which the PR itself lacked) and
adapted to NiFreeze's vendored implementation in :mod:`nifreeze.model.gqi`.

NiFreeze's ``prediction_kernel`` returns the reconstruction kernel transposed
with respect to the DIPY-branch version, i.e. with shape
``(n_gradients, n_vertices)``.  The pseudo-inverse assertions below therefore
carry a ``.T`` on ``K_plus`` relative to the original tests.
"""

import warnings

import numpy as np
import pytest
from dipy.core.gradients import gradient_table
from dipy.data import dsi_voxels, get_fnames, get_sphere
from dipy.reconst.odf import OdfFit, OdfModel

from nifreeze import model
from nifreeze.data.dmri import DWI
from nifreeze.data.dmri.base import DWI_B0_MULTIPLE_VOLUMES_WARN_MSG
from nifreeze.data.dmri.utils import format_gradients
from nifreeze.model.base import MASK_ABSENCE_WARN_MSG
from nifreeze.model.gqi import (
    GeneralizedQSamplingModel,
    gqi_kernel,
    prediction_kernel,
)

SINGLE_VOXEL_CORRELATION_THRESHOLD = 0.8
AVERAGE_CORRELATION_THRESHOLD = 0.8

#: Voxel coordinates exercised by the single-voxel tests (as in the DIPY tests).
TESTED_VOXELS = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0)]

#: Sampling length (lambda) used throughout, matching the DIPY tests.
SAMPLING_LENGTH = 1.2


def _train_gtab(gtab, *, keep_b0):
    """Return a gradient table and the index mask selecting its volumes.

    With ``keep_b0=False`` the b=0/low-b volumes are dropped (the NiFreeze
    pipeline contract, which strips them before any model runs); with
    ``keep_b0=True`` the full table is kept so the b0-fed path is exercised
    directly on :class:`~nifreeze.model.gqi.GeneralizedQSamplingModel`.
    """
    idx = np.arange(len(gtab.bvals)) if keep_b0 else np.where(~gtab.b0s_mask)[0]
    train_gtab = gradient_table(bvals=gtab.bvals[idx], bvecs=gtab.bvecs[idx])
    return train_gtab, idx


#: ``keep_b0`` parametrization shared by the b0-included/excluded tests.
_KEEP_B0 = pytest.mark.parametrize("keep_b0", [False, True], ids=["no_b0", "with_b0"])


# ---------------------------------------------------------------------------
# Forward kernel
# ---------------------------------------------------------------------------
@_KEEP_B0
@pytest.mark.parametrize("method", ["standard", "gqi2"])
def test_gqi_kernel_shape_and_values(method, keep_b0):
    """The forward GQI kernel is real, finite, and ``(n_gradients, n_vertices)``."""
    _, gtab = dsi_voxels()
    train_gtab, _ = _train_gtab(gtab, keep_b0=keep_b0)
    sphere = get_sphere(name="symmetric724")

    K = gqi_kernel(train_gtab, SAMPLING_LENGTH, sphere, method=method)

    assert K.shape == (len(train_gtab.bvals), len(sphere.vertices))
    assert np.isrealobj(K)
    assert np.all(np.isfinite(K))
    assert np.any(K != 0)


@pytest.mark.parametrize(
    "method, constant", [("standard", 1.0), ("gqi2", 1.0 / 3)], ids=["standard", "gqi2"]
)
def test_gqi_kernel_b0_row_is_constant(method, constant):
    """A true b=0 volume yields an angularly flat, collinear kernel row.

    At b=0 the sampling vector is zero, so every vertex sees the same argument:
    ``sinc(0) = 1`` (``standard``) and ``squared_radial_component(0) = 1/3``
    (``gqi2``). Two b=0 volumes therefore produce identical rows carrying no
    angular information -- the rank-deficiency that motivates excluding b=0 from
    the fit (see ``docs/models.rst``, "Reconstruction fidelity and the intercept
    behaviour"). ``dsi_voxels`` only carries a single low-b b0 (b=15), so this
    uses a synthetic pair of true b=0 volumes.
    """
    _, gtab = dsi_voxels()
    sphere = get_sphere(name="symmetric724")
    dw = np.where(~gtab.b0s_mask)[0]
    # Prepend two true b=0 volumes; their (distinct) bvecs are irrelevant at b=0.
    bvals = np.concatenate(([0.0, 0.0], gtab.bvals[dw]))
    bvecs = np.vstack(([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], gtab.bvecs[dw]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g0 = gradient_table(bvals=bvals, bvecs=bvecs)

    K = gqi_kernel(g0, SAMPLING_LENGTH, sphere, method=method)

    b0_rows = K[:2]
    assert np.allclose(b0_rows, constant), f"b0 row is not the constant {constant}"
    assert np.allclose(b0_rows[0], b0_rows[1]), "b0 rows are not identical (collinear)"
    # By contrast, diffusion-weighted rows vary across vertices (carry structure).
    assert K[2:].std(axis=1).min() > 1e-2, "DW rows are unexpectedly flat"


def test_gqi_kernel_unknown_method_warns():
    """An unknown method warns and falls back to the ``"standard"`` kernel."""
    _, gtab = dsi_voxels()
    sphere = get_sphere(name="symmetric724")

    with pytest.warns(UserWarning, match="unknown"):
        K_unknown = gqi_kernel(gtab, SAMPLING_LENGTH, sphere, method="bogus")

    K_standard = gqi_kernel(gtab, SAMPLING_LENGTH, sphere, method="standard")
    assert np.array_equal(K_unknown, K_standard)


# ---------------------------------------------------------------------------
# ODF-family alignment — OdfModel/OdfFit membership and the ``odf`` transform
# ---------------------------------------------------------------------------
@_KEEP_B0
@pytest.mark.parametrize("method", ["standard", "gqi2"])
def test_gqi_odf_family(method, keep_b0):
    """GQI joins DIPY's ``OdfModel``/``OdfFit`` family and exposes ``odf``.

    Guards the base-class alignment (a silent regression to ``ReconstModel``/
    ``ReconstFit`` would break family membership) and checks that ``Fit.odf``
    applies the forward GQI kernel (``data @ gqi_kernel``) for both the model's
    default sphere (reusing the pre-computed kernel) and an explicit sphere,
    over single-voxel and NiFreeze's 2D ``(n_voxels, n_gradients)`` data.
    """
    data, gtab = dsi_voxels()
    train_gtab, idx = _train_gtab(gtab, keep_b0=keep_b0)
    sphere = get_sphere(name="symmetric724")

    gq = GeneralizedQSamplingModel(train_gtab, method=method, sampling_length=SAMPLING_LENGTH)
    assert isinstance(gq, OdfModel)

    n_default = len(gq.sphere.vertices)
    n_sphere = len(sphere.vertices)

    # Single voxel.
    voxel_data = data[(0, 0, 0)][idx]
    voxel_fit = gq.fit(voxel_data)
    assert isinstance(voxel_fit, OdfFit)

    # Default sphere reuses the pre-computed forward kernel (kernel stored as
    # (n_vertices, n_gradients); ``.T`` recovers the forward orientation).
    # ``None`` explicitly requests the model's sphere; it also satisfies the base
    # ``OdfFit.odf(sphere)`` signature that ``voxel_fit`` is narrowed to above.
    default_odf = voxel_fit.odf(None)
    assert default_odf.shape == (n_default,)
    assert np.allclose(default_odf, voxel_data @ gq.kernel)

    # Explicit sphere recomputes the forward kernel.
    explicit_odf = voxel_fit.odf(sphere)
    assert explicit_odf.shape == (n_sphere,)
    assert np.allclose(
        explicit_odf,
        voxel_data @ gqi_kernel(train_gtab, SAMPLING_LENGTH, sphere, method=method),
    )

    # 2D batch (NiFreeze's masked/flattened contract).
    data_2d = data[..., idx].reshape(-1, len(train_gtab.bvals))
    batch_odf = gq.fit(data_2d).odf(sphere)
    assert batch_odf.shape == (data_2d.shape[0], n_sphere)
    assert np.allclose(
        batch_odf,
        data_2d @ gqi_kernel(train_gtab, SAMPLING_LENGTH, sphere, method=method),
    )


# ---------------------------------------------------------------------------
# Reconstruction (prediction) kernel — analytical validity
# ---------------------------------------------------------------------------
@_KEEP_B0
@pytest.mark.parametrize("method", ["standard", "gqi2"])
def test_prediction_kernel_properties(method, keep_b0):
    """The reconstruction kernel is a valid Tikhonov-regularized pseudo-inverse.

    Ported from ``test_prediction_kernel``.  NiFreeze returns
    ``K_plus`` with shape ``(n_gradients, n_vertices)`` (the transpose of the
    DIPY-branch orientation), hence the ``.T`` on ``K_plus`` below.  The
    pseudo-inverse properties hold whether or not b=0/low-b volumes are present.
    """
    _, gtab = dsi_voxels()
    train_gtab, _ = _train_gtab(gtab, keep_b0=keep_b0)
    sphere = get_sphere(name="symmetric724")

    K = gqi_kernel(train_gtab, SAMPLING_LENGTH, sphere, method=method)
    K_plus = prediction_kernel(train_gtab, SAMPLING_LENGTH, sphere, method=method)

    # Shape: (n_gradients, n_vertices) — same shape as the forward kernel here.
    assert K_plus.shape == (len(train_gtab.bvals), len(sphere.vertices))
    assert K_plus.shape == K.shape

    # Finite and non-zero.
    assert np.all(np.isfinite(K_plus)), "K_plus contains non-finite values"
    assert np.any(K_plus != 0), "K_plus is all zeros"

    # Property 1: K @ K_plus.T @ K ≈ K (regularized reconstruction).
    reconstructed_K = K @ K_plus.T @ K
    assert np.allclose(reconstructed_K, K, atol=1e-4, rtol=1e-3), (
        "Regularized reconstruction K K_plus K ≈ K failed"
    )

    # Properties 2 & 3: K @ K_plus.T and K_plus.T @ K are symmetric.  Both equal
    # a function of A = K Kᵀ (A(A+λI)⁻¹ and Kᵀ(A+λI)⁻¹K), symmetric in exact
    # arithmetic.  The near-singular inverse (INVERSE_LAMBDA = 1e-6) leaves float
    # rounding, so symmetry is checked relative to the matrix scale.
    def _assert_symmetric(matrix, name):
        asymmetry = np.abs(matrix - matrix.T).max()
        assert asymmetry <= 1e-3 * np.abs(matrix).max(), f"{name} is not symmetric"

    _assert_symmetric(K @ K_plus.T, "K K_plus")
    _assert_symmetric(K_plus.T @ K, "K_plus K")


# ---------------------------------------------------------------------------
# Prediction API — shape & non-negativity
# ---------------------------------------------------------------------------
@_KEEP_B0
@pytest.mark.parametrize("method", ["standard", "gqi2"])
@pytest.mark.parametrize("voxel_coordinate", TESTED_VOXELS)
def test_predict_single_voxel(method, voxel_coordinate, keep_b0):
    """Single-voxel prediction has the right shape and is non-negative."""
    data, gtab = dsi_voxels()
    train_gtab, idx = _train_gtab(gtab, keep_b0=keep_b0)

    gq = GeneralizedQSamplingModel(train_gtab, method=method, sampling_length=SAMPLING_LENGTH)
    voxel_fit = gq.fit(data[voxel_coordinate][idx])
    voxel_predicted = voxel_fit.predict(train_gtab)

    assert voxel_predicted.shape == (len(train_gtab.bvals),)
    assert np.all(voxel_predicted >= 0), "Predicted signals should be non-negative"

    # Prediction on a subset of gradients.
    subset_gtab = gradient_table(train_gtab.bvals[::2], bvecs=train_gtab.bvecs[::2])
    subset_predicted = voxel_fit.predict(subset_gtab)

    assert subset_predicted.shape == (len(subset_gtab.bvals),)
    assert np.all(subset_predicted >= 0), "Subset predictions should be non-negative"


@_KEEP_B0
@pytest.mark.parametrize("method", ["standard", "gqi2"])
def test_predict_multi_voxel(method, keep_b0):
    """Multi-voxel prediction round-trips shape over NiFreeze's 2D contract.

    NiFreeze always feeds ``predict`` a 2D ``(n_voxels, n_gradients)`` array
    (voxels are masked/flattened upstream), so the DSI grid is reshaped to 2D
    here rather than passed as a 4D volume.
    """
    data, gtab = dsi_voxels()
    train_gtab, idx = _train_gtab(gtab, keep_b0=keep_b0)
    n_gradients = len(train_gtab.bvals)
    data_2d = data[..., idx].reshape(-1, n_gradients)

    gq = GeneralizedQSamplingModel(train_gtab, method=method, sampling_length=SAMPLING_LENGTH)
    multi_predicted = gq.fit(data_2d).predict(train_gtab)

    assert multi_predicted.shape == data_2d.shape
    assert np.all(multi_predicted >= 0), "Predicted signals should be non-negative"

    subset_gtab = gradient_table(train_gtab.bvals[::2], bvecs=train_gtab.bvecs[::2])
    subset_predicted = gq.fit(data_2d).predict(subset_gtab)

    assert subset_predicted.shape == (data_2d.shape[0], len(subset_gtab.bvals))
    assert np.all(subset_predicted >= 0), "Subset predictions should be non-negative"


# ---------------------------------------------------------------------------
# Prediction round-trip — analytical validity on real DSI data
# ---------------------------------------------------------------------------
@_KEEP_B0
@pytest.mark.parametrize("voxel_coordinate", TESTED_VOXELS)
def test_predict_roundtrip_single_voxel(voxel_coordinate, keep_b0):
    """Single-voxel predictions correlate strongly with the original signal.

    Runs with b=0 excluded (the NiFreeze pipeline contract) and with the b=0/
    low-b volume fed directly to the model. Correlation is evaluated on the
    diffusion-weighted subset: including the b0 volume must not corrupt the DW
    reconstruction. (``dsi_voxels`` carries a single low-b b0 at b=15; the DW
    reconstruction is robust to it.)
    """
    data, gtab = dsi_voxels()
    train_gtab, idx = _train_gtab(gtab, keep_b0=keep_b0)
    train_data = data[..., idx]

    gq = GeneralizedQSamplingModel(train_gtab, method="standard", sampling_length=SAMPLING_LENGTH)
    voxel_data = train_data[voxel_coordinate]
    voxel_predicted = gq.fit(voxel_data).predict(train_gtab)

    assert voxel_predicted.shape == (len(train_gtab.bvals),)
    assert np.all(voxel_predicted >= 0), "Predicted signals should be non-negative"

    # Evaluate on the DW subset (all volumes when ``keep_b0=False``) so the b0-fed
    # case measures DW fidelity, not the single high-leverage b0 point.
    dw = ~train_gtab.b0s_mask
    correlation = np.corrcoef(voxel_data[dw], voxel_predicted[dw])[0, 1]
    assert correlation > SINGLE_VOXEL_CORRELATION_THRESHOLD, (
        f"Poor single voxel correlation {correlation:.3f}"
    )

    orig_mean = np.mean(voxel_data)
    pred_mean = np.mean(voxel_predicted)
    ratio = pred_mean / orig_mean if orig_mean > 0 else 1
    assert 0.1 < ratio < 10, f"Signal magnitude unrealistic: ratio={ratio:.3f}"


@_KEEP_B0
def test_predict_roundtrip_multi_voxel(keep_b0):
    """Whole-phantom predictions maintain high per-voxel correlation.

    Runs with b=0 excluded (the NiFreeze pipeline contract) and with the b=0/
    low-b volume fed directly to the model; the DSI grid is reshaped to
    NiFreeze's 2D ``(n_voxels, n_gradients)`` contract. Correlation is evaluated
    on the diffusion-weighted subset so the b0-fed case measures whether the b0
    volume corrupts the DW reconstruction (it does not). When b0 is fed, the b0
    volume itself is additionally characterized: it is reconstructed within a
    loose factor (a characterization bound, not a quality target).
    """
    data, gtab = dsi_voxels()
    train_gtab, idx = _train_gtab(gtab, keep_b0=keep_b0)
    train_data = data[..., idx]

    vol_shape = train_data.shape[:-1]
    n_gradients = train_data.shape[-1]
    train_data_2d = train_data.reshape(-1, n_gradients)

    gq = GeneralizedQSamplingModel(train_gtab, method="standard", sampling_length=SAMPLING_LENGTH)
    predicted_2d = gq.fit(train_data_2d).predict(train_gtab)
    multi_predicted = predicted_2d.reshape(*vol_shape, n_gradients)

    assert np.all(multi_predicted >= 0), "Predicted signals should be non-negative"

    # DW subset (all volumes when ``keep_b0=False``) for the correlation gates.
    dw = ~train_gtab.b0s_mask
    correlations = []
    poor_correlation_voxels = []
    for i in range(vol_shape[0]):
        for j in range(vol_shape[1]):
            for k in range(vol_shape[2]):
                original_voxel = train_data[i, j, k]
                predicted_voxel = multi_predicted[i, j, k]
                correlation = np.corrcoef(original_voxel[dw], predicted_voxel[dw])[0, 1]
                correlations.append(correlation)

                if np.sum(original_voxel) == 0:
                    continue
                if correlation <= SINGLE_VOXEL_CORRELATION_THRESHOLD:
                    poor_correlation_voxels.append((i, j, k, correlation))

    avg_correlation = np.mean(correlations)
    assert avg_correlation > AVERAGE_CORRELATION_THRESHOLD, (
        f"Poor multi-voxel average correlation {avg_correlation:.3f}"
    )
    assert not poor_correlation_voxels, (
        f"Found {len(poor_correlation_voxels)} voxels with poor correlation"
    )

    # When the b0 is fed, characterize its own reconstruction: the large, near-flat
    # b0 volume is recovered within a loose factor (not exactly -- the sinc basis
    # cannot fully represent it), while the DW reconstruction above is unharmed.
    if keep_b0:
        b0 = train_gtab.b0s_mask
        b0_ratio = predicted_2d[:, b0].mean() / train_data_2d[:, b0].mean()
        assert 0.3 < b0_ratio < 3.0, f"b0 reconstruction ratio out of range: {b0_ratio:.3f}"

    orig_mean = np.mean(train_data)
    pred_mean = np.mean(multi_predicted)
    ratio = pred_mean / orig_mean if orig_mean > 0 else 1
    assert 0.1 < ratio < 10, f"Signal magnitude unrealistic: ratio={ratio:.3f}"


# ---------------------------------------------------------------------------
# Regression guard: the fitted ``method`` must drive the reconstruction kernel
# ---------------------------------------------------------------------------
def test_gqi2_method_propagates():
    """``Fit.predict`` uses the model's ``method`` for the reconstruction kernel.

    Before the fix, ``Fit.predict`` built the forward kernel with the model's
    method but always inverted with ``"standard"``, so a ``gqi2`` model produced
    a ``standard``-consistent prediction.  This guards that the two methods now
    genuinely differ and that each is internally consistent.
    """
    data, gtab = dsi_voxels()
    voxel_data = data[(0, 0, 0)]

    standard_fit = GeneralizedQSamplingModel(
        gtab, method="standard", sampling_length=SAMPLING_LENGTH
    ).fit(voxel_data)
    gqi2_fit = GeneralizedQSamplingModel(gtab, method="gqi2", sampling_length=SAMPLING_LENGTH).fit(
        voxel_data
    )

    standard_pred = standard_fit.predict(gtab)
    gqi2_pred = gqi2_fit.predict(gtab)

    # The two methods use different kernels, so predictions must differ.
    assert not np.allclose(standard_pred, gqi2_pred)

    # Each prediction must match the explicit forward/inverse composition for
    # its own method (i.e. no cross-method leakage).
    for fit, method in ((standard_fit, "standard"), (gqi2_fit, "gqi2")):
        expected = np.maximum(
            voxel_data
            @ (
                prediction_kernel(gtab, SAMPLING_LENGTH, fit.model.sphere, method=method)
                @ fit.model.kernel.T
            ).T,
            0,
        )
        assert np.allclose(fit.predict(gtab), expected)


# ---------------------------------------------------------------------------
# Regression: GQIModel forwards model kwargs to the underlying model
# ---------------------------------------------------------------------------
def test_gqimodel_forwards_model_kwargs():
    """``GQIModel`` forwards construction-time model kwargs, without leaking.

    Guards against a regression where ``BaseModel.__init__`` silently dropped
    construction-time kwargs (``method``/``sampling_length``/``recursion_level``
    produced identical output) and where passing a model kwarg to ``fit_predict``
    leaked into ``predict`` and raised ``TypeError``.
    """
    data, gtab = dsi_voxels()
    gradients = format_gradients(np.column_stack((gtab.bvecs, gtab.bvals)))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dataset = DWI(dataobj=data.astype(np.float32), affine=np.eye(4), gradients=gradients)
        index = dataset.dataobj.shape[-1] // 2

        def predict(**kwargs):
            return model.dmri.GQIModel(dataset, **kwargs).fit_predict(index, n_jobs=1)

        # Each model kwarg actually reaches the underlying model (changes output).
        assert not np.allclose(predict(method="standard"), predict(method="gqi2"))
        assert not np.allclose(predict(sampling_length=0.5), predict(sampling_length=3.0))
        assert not np.allclose(predict(recursion_level=3), predict(recursion_level=6))

        # A model kwarg passed at fit time is consumed by ``_fit`` and must not
        # leak into ``predict`` (previously raised an unexpected-keyword error).
        assert model.dmri.GQIModel(dataset).fit_predict(index, n_jobs=1, method="gqi2") is not None


# ---------------------------------------------------------------------------
# Integration: GQIModel LOVO prediction on real Stanford HARDI brain data
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "method",
    [
        "standard",
        pytest.param(
            "gqi2",
            marks=pytest.mark.xfail(
                reason=(
                    "gqi2 is a weaker signal predictor than the default 'standard' kernel; "
                    "its LOVO correlation falls below the 0.6 gate on this single-shell slab "
                    "(see docs/models.rst, 'Reconstruction fidelity')."
                ),
                strict=True,
            ),
        ),
    ],
)
def test_gqi_lovo_prediction_stanford_hardi(method):
    """A held-out DWI volume predicted by ``GQIModel`` correlates with truth.

    Loads a cropped slab of the Stanford HARDI dataset, fits GQI in
    leave-one-volume-out mode through the NiFreeze wrapper, and checks that the
    predicted volume correlates with the actual left-out volume within the mask.

    The ``gqi2`` variant is expected to fail the correlation gate: it is a weaker
    signal predictor (oscillatory, ill-conditioned reconstruction kernel), which
    is why ``"standard"`` is the NiFreeze default.
    """
    from typing import cast

    import nibabel as nb
    from dipy.io.gradients import read_bvals_bvecs
    from nibabel.spatialimages import SpatialImage

    try:
        fdwi, fbval, fbvec = get_fnames(name="stanford_hardi")
    except Exception as exc:  # pragma: no cover - network/cache guard
        pytest.skip(f"Stanford HARDI not available: {exc}")

    img = cast(SpatialImage, nb.load(fdwi))
    # Crop a small central slab to keep the test fast.
    sl = (slice(28, 52), slice(50, 74), slice(35, 40))
    dataobj = np.asarray(img.dataobj[sl].astype(np.float32))
    bvals, bvecs = read_bvals_bvecs(str(fbval), str(fbvec))
    gradients = format_gradients(np.column_stack((bvecs, bvals)))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=MASK_ABSENCE_WARN_MSG, category=UserWarning)
        warnings.filterwarnings(
            "ignore", message=DWI_B0_MULTIPLE_VOLUMES_WARN_MSG, category=UserWarning
        )
        dataset = DWI(dataobj=dataobj, affine=img.affine, gradients=gradients)

    # A simple intensity mask from the b=0 reference confines the comparison.
    assert dataset.bzero is not None
    brainmask = dataset.bzero > np.percentile(dataset.bzero, 60)

    index = dataset.dataobj.shape[-1] // 2
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=MASK_ABSENCE_WARN_MSG, category=UserWarning)
        predicted = model.dmri.GQIModel(
            dataset, method=method, sampling_length=SAMPLING_LENGTH
        ).fit_predict(index, n_jobs=1)

    assert predicted is not None
    assert predicted.shape == dataset.dataobj.shape[:3]
    assert np.all(predicted >= 0)

    truth = dataset.dataobj[..., index]
    corr = np.corrcoef(truth[brainmask], predicted[brainmask])[0, 1]
    assert corr > 0.6, f"LOVO prediction poorly correlated with truth: {corr:.3f}"
