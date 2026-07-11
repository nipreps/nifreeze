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


def _b0_excluded_train_gtab(gtab):
    """Return the gradient table and index mask with b=0 volumes removed."""
    non_b0 = np.where(~gtab.b0s_mask)[0]
    train_gtab = gradient_table(bvals=gtab.bvals[non_b0], bvecs=gtab.bvecs[non_b0])
    return train_gtab, non_b0


# ---------------------------------------------------------------------------
# Forward kernel
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", ["standard", "gqi2"])
def test_gqi_kernel_shape_and_values(method):
    """The forward GQI kernel is real, finite, and ``(n_gradients, n_vertices)``."""
    _, gtab = dsi_voxels()
    sphere = get_sphere(name="symmetric724")

    K = gqi_kernel(gtab, SAMPLING_LENGTH, sphere, method=method)

    assert K.shape == (len(gtab.bvals), len(sphere.vertices))
    assert np.isrealobj(K)
    assert np.all(np.isfinite(K))
    assert np.any(K != 0)


def test_gqi_kernel_unknown_method_warns():
    """An unknown method warns and falls back to the ``"standard"`` kernel."""
    _, gtab = dsi_voxels()
    sphere = get_sphere(name="symmetric724")

    with pytest.warns(UserWarning, match="unknown"):
        K_unknown = gqi_kernel(gtab, SAMPLING_LENGTH, sphere, method="bogus")

    K_standard = gqi_kernel(gtab, SAMPLING_LENGTH, sphere, method="standard")
    assert np.array_equal(K_unknown, K_standard)


# ---------------------------------------------------------------------------
# Reconstruction (prediction) kernel — analytical validity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", ["standard", "gqi2"])
def test_prediction_kernel_properties(method):
    """The reconstruction kernel is a valid Tikhonov-regularized pseudo-inverse.

    Ported from ``test_prediction_kernel``.  NiFreeze returns
    ``K_plus`` with shape ``(n_gradients, n_vertices)`` (the transpose of the
    DIPY-branch orientation), hence the ``.T`` on ``K_plus`` below.
    """
    _, gtab = dsi_voxels()
    sphere = get_sphere(name="symmetric724")

    K = gqi_kernel(gtab, SAMPLING_LENGTH, sphere, method=method)
    K_plus = prediction_kernel(gtab, SAMPLING_LENGTH, sphere, method=method)

    # Shape: (n_gradients, n_vertices) — same shape as the forward kernel here.
    assert K_plus.shape == (len(gtab.bvals), len(sphere.vertices))
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
@pytest.mark.parametrize("method", ["standard", "gqi2"])
@pytest.mark.parametrize("voxel_coordinate", TESTED_VOXELS)
def test_predict_single_voxel(method, voxel_coordinate):
    """Single-voxel prediction has the right shape and is non-negative."""
    data, gtab = dsi_voxels()

    gq = GeneralizedQSamplingModel(gtab, method=method, sampling_length=SAMPLING_LENGTH)
    voxel_fit = gq.fit(data[voxel_coordinate])
    voxel_predicted = voxel_fit.predict(gtab)

    assert voxel_predicted.shape == (len(gtab.bvals),)
    assert np.all(voxel_predicted >= 0), "Predicted signals should be non-negative"

    # Prediction on a subset of gradients.
    subset_gtab = gradient_table(gtab.bvals[::2], bvecs=gtab.bvecs[::2])
    subset_predicted = voxel_fit.predict(subset_gtab)

    assert subset_predicted.shape == (len(subset_gtab.bvals),)
    assert np.all(subset_predicted >= 0), "Subset predictions should be non-negative"


@pytest.mark.parametrize("method", ["standard", "gqi2"])
def test_predict_multi_voxel(method):
    """Multi-voxel prediction round-trips shape over NiFreeze's 2D contract.

    NiFreeze always feeds ``predict`` a 2D ``(n_voxels, n_gradients)`` array
    (voxels are masked/flattened upstream), so the DSI grid is reshaped to 2D
    here rather than passed as a 4D volume.
    """
    data, gtab = dsi_voxels()
    n_gradients = len(gtab.bvals)
    data_2d = data.reshape(-1, n_gradients)

    gq = GeneralizedQSamplingModel(gtab, method=method, sampling_length=SAMPLING_LENGTH)
    multi_predicted = gq.fit(data_2d).predict(gtab)

    assert multi_predicted.shape == data_2d.shape
    assert np.all(multi_predicted >= 0), "Predicted signals should be non-negative"

    subset_gtab = gradient_table(gtab.bvals[::2], bvecs=gtab.bvecs[::2])
    subset_predicted = gq.fit(data_2d).predict(subset_gtab)

    assert subset_predicted.shape == (data_2d.shape[0], len(subset_gtab.bvals))
    assert np.all(subset_predicted >= 0), "Subset predictions should be non-negative"


# ---------------------------------------------------------------------------
# Prediction round-trip — analytical validity on real DSI data
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("voxel_coordinate", TESTED_VOXELS)
def test_predict_roundtrip_single_voxel(voxel_coordinate):
    """Single-voxel predictions correlate strongly with the original signal.

    The b=0 volumes are excluded from both train and test sets.
    """
    data, gtab = dsi_voxels()
    train_gtab, non_b0 = _b0_excluded_train_gtab(gtab)
    train_data = data[..., non_b0]

    gq = GeneralizedQSamplingModel(train_gtab, method="standard", sampling_length=SAMPLING_LENGTH)
    voxel_data = train_data[voxel_coordinate]
    voxel_predicted = gq.fit(voxel_data).predict(train_gtab)

    assert voxel_predicted.shape == (len(train_gtab.bvals),)
    assert np.all(voxel_predicted >= 0), "Predicted signals should be non-negative"

    correlation = np.corrcoef(voxel_data, voxel_predicted)[0, 1]
    assert correlation > SINGLE_VOXEL_CORRELATION_THRESHOLD, (
        f"Poor single voxel correlation {correlation:.3f}"
    )

    orig_mean = np.mean(voxel_data)
    pred_mean = np.mean(voxel_predicted)
    ratio = pred_mean / orig_mean if orig_mean > 0 else 1
    assert 0.1 < ratio < 10, f"Signal magnitude unrealistic: ratio={ratio:.3f}"


def test_predict_roundtrip_multi_voxel():
    """Whole-phantom predictions maintain high per-voxel correlation.

    The b=0 volumes are excluded from both train and test sets, and the DSI
    grid is reshaped to NiFreeze's 2D ``(n_voxels, n_gradients)`` contract.
    """
    data, gtab = dsi_voxels()
    train_gtab, non_b0 = _b0_excluded_train_gtab(gtab)
    train_data = data[..., non_b0]

    vol_shape = train_data.shape[:-1]
    n_gradients = train_data.shape[-1]
    train_data_2d = train_data.reshape(-1, n_gradients)

    gq = GeneralizedQSamplingModel(train_gtab, method="standard", sampling_length=SAMPLING_LENGTH)
    predicted_2d = gq.fit(train_data_2d).predict(train_gtab)
    multi_predicted = predicted_2d.reshape(*vol_shape, n_gradients)

    assert np.all(multi_predicted >= 0), "Predicted signals should be non-negative"

    correlations = []
    poor_correlation_voxels = []
    for i in range(vol_shape[0]):
        for j in range(vol_shape[1]):
            for k in range(vol_shape[2]):
                original_voxel = train_data[i, j, k]
                predicted_voxel = multi_predicted[i, j, k]
                correlation = np.corrcoef(original_voxel, predicted_voxel)[0, 1]
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
                @ fit.model.kernel
            ).T,
            0,
        )
        assert np.allclose(fit.predict(gtab), expected)


# ---------------------------------------------------------------------------
# Integration: GQIModel LOVO prediction on real Stanford HARDI brain data
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", ["standard", "gqi2"])
def test_gqi_lovo_prediction_stanford_hardi(method):
    """A held-out DWI volume predicted by ``GQIModel`` correlates with truth.

    Loads a cropped slab of the Stanford HARDI dataset, fits GQI in
    leave-one-volume-out mode through the NiFreeze wrapper, and checks that the
    predicted volume correlates with the actual left-out volume within the mask.
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
