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

import numpy as np
import pytest
from dipy.io import read_bvals_bvecs

from nifreeze.model import gpr

THETAS = np.linspace(0, np.pi / 2, num=50)
EXPECTED_EXPONENTIAL = [
    1.0,
    0.93789795,
    0.87965256,
    0.82502433,
    0.77378862,
    0.72573476,
    0.68066514,
    0.63839443,
    0.59874883,
    0.5615653,
    0.52669094,
    0.49398235,
    0.46330503,
    0.43453284,
    0.40754745,
    0.38223792,
    0.35850016,
    0.33623656,
    0.31535558,
    0.29577135,
    0.27740334,
    0.26017603,
    0.24401856,
    0.22886451,
    0.21465155,
    0.20132125,
    0.18881879,
    0.17709275,
    0.16609493,
    0.15578009,
    0.14610583,
    0.13703236,
    0.12852236,
    0.12054086,
    0.11305503,
    0.10603408,
    0.09944914,
    0.09327315,
    0.08748069,
    0.08204796,
    0.07695262,
    0.0721737,
    0.06769156,
    0.06348778,
    0.05954506,
    0.05584719,
    0.05237896,
    0.04912612,
    0.04607529,
    0.04321392,
]
EXPECTED_SPHERICAL = [
    1.00000000e00,
    9.60914866e-01,
    9.21882843e-01,
    8.82957040e-01,
    8.44190567e-01,
    8.05636535e-01,
    7.67348053e-01,
    7.29378232e-01,
    6.91780182e-01,
    6.54607013e-01,
    6.17911835e-01,
    5.81747758e-01,
    5.46167893e-01,
    5.11225349e-01,
    4.76973237e-01,
    4.43464666e-01,
    4.10752747e-01,
    3.78890590e-01,
    3.47931306e-01,
    3.17928003e-01,
    2.88933793e-01,
    2.61001786e-01,
    2.34185091e-01,
    2.08536818e-01,
    1.84110079e-01,
    1.60957982e-01,
    1.39133639e-01,
    1.18690159e-01,
    9.96806519e-02,
    8.21582285e-02,
    6.61759987e-02,
    5.17870726e-02,
    3.90445603e-02,
    2.80015720e-02,
    1.87112177e-02,
    1.12266077e-02,
    5.60085192e-03,
    1.88706063e-03,
    1.38343910e-04,
    0.00000000e00,
    0.00000000e00,
    0.00000000e00,
    0.00000000e00,
    0.00000000e00,
    0.00000000e00,
    0.00000000e00,
    0.00000000e00,
    0.00000000e00,
    0.00000000e00,
    0.00000000e00,
]


def _unit_vectors_nd(n_samples: int, n_features: int) -> np.ndarray:
    # Orthogonal unit vectors as rows
    if n_samples < 1 or n_features < 1:
        raise ValueError("n_samples and n_features must be >= 1")
    if n_samples > n_features:
        # In R^d we can have at most d mutually orthogonal non‑zero vectors
        raise ValueError("n_samples must be <= n_features to produce orthogonal vectors")

    return np.eye(n_features, dtype=float)[:n_samples]


# No need to use normalized vectors: compute_pairwise_angles takes care of it.
# The [-1, 0, 1].T vector serves as a case where e.g. the angle between vector
# [1, 0, 0] and the former is 135 unless the closest polarity flag is set to
# True, in which case it yields 45
@pytest.mark.parametrize(
    ("bvecs1", "bvecs2", "closest_polarity", "expected"),
    [
        (
            np.array(
                [
                    [1, 0, 0, 1, 1, 0, -1],
                    [0, 1, 0, 1, 0, 1, 0],
                    [0, 0, 1, 0, 1, 1, 1],
                ]
            ),
            None,
            True,
            np.array(
                [
                    [0.0, np.pi / 2, np.pi / 2, np.pi / 4, np.pi / 4, np.pi / 2, np.pi / 4],
                    [np.pi / 2, 0.0, np.pi / 2, np.pi / 4, np.pi / 2, np.pi / 4, np.pi / 2],
                    [np.pi / 2, np.pi / 2, 0.0, np.pi / 2, np.pi / 4, np.pi / 4, np.pi / 4],
                    [np.pi / 4, np.pi / 4, np.pi / 2, 0.0, np.pi / 3, np.pi / 3, np.pi / 3],
                    [np.pi / 4, np.pi / 2, np.pi / 4, np.pi / 3, 0.0, np.pi / 3, np.pi / 2],
                    [np.pi / 2, np.pi / 4, np.pi / 4, np.pi / 3, np.pi / 3, 0.0, np.pi / 3],
                    [np.pi / 4, np.pi / 2, np.pi / 4, np.pi / 3, np.pi / 2, np.pi / 3, 0.0],
                ]
            ),
        ),
        (
            np.array(
                [
                    [1, 0, 0, 1, 1, 0, -1],
                    [0, 1, 0, 1, 0, 1, 0],
                    [0, 0, 1, 0, 1, 1, 1],
                ]
            ),
            None,
            False,
            np.array(
                [
                    [0.0, np.pi / 2, np.pi / 2, np.pi / 4, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                    [np.pi / 2, 0.0, np.pi / 2, np.pi / 4, np.pi / 2, np.pi / 4, np.pi / 2],
                    [np.pi / 2, np.pi / 2, 0.0, np.pi / 2, np.pi / 4, np.pi / 4, np.pi / 4],
                    [np.pi / 4, np.pi / 4, np.pi / 2, 0.0, np.pi / 3, np.pi / 3, 2 * np.pi / 3],
                    [np.pi / 4, np.pi / 2, np.pi / 4, np.pi / 3, 0.0, np.pi / 3, np.pi / 2],
                    [np.pi / 2, np.pi / 4, np.pi / 4, np.pi / 3, np.pi / 3, 0.0, np.pi / 3],
                    [
                        3 * np.pi / 4,
                        np.pi / 2,
                        np.pi / 4,
                        2 * np.pi / 3,
                        np.pi / 2,
                        np.pi / 3,
                        0.0,
                    ],
                ]
            ),
        ),
        (
            np.array(
                [
                    [1, 0, 0, 1, 1, 0, -1],
                    [0, 1, 0, 1, 0, 1, 0],
                    [0, 0, 1, 0, 1, 1, 1],
                ]
            ),
            np.array(
                [
                    [1, -1],
                    [0, 0],
                    [0, 1],
                ]
            ),
            True,
            np.array(
                [
                    [0.0, np.pi / 4],
                    [np.pi / 2, np.pi / 2],
                    [np.pi / 2, np.pi / 4],
                    [np.pi / 4, np.pi / 3],
                    [np.pi / 4, np.pi / 2],
                    [np.pi / 2, np.pi / 3],
                    [np.pi / 4, 0.0],
                ]
            ),
        ),
        (
            np.array(
                [
                    [1, 0, 0, 1, 1, 0, -1],
                    [0, 1, 0, 1, 0, 1, 0],
                    [0, 0, 1, 0, 1, 1, 1],
                ]
            ),
            np.array(
                [
                    [1, -1],
                    [0, 0],
                    [0, 1],
                ]
            ),
            False,
            np.array(
                [
                    [0.0, 3 * np.pi / 4],
                    [np.pi / 2, np.pi / 2],
                    [np.pi / 2, np.pi / 4],
                    [np.pi / 4, 2 * np.pi / 3],
                    [np.pi / 4, np.pi / 2],
                    [np.pi / 2, np.pi / 3],
                    [3 * np.pi / 4, 0.0],
                ]
            ),
        ),
    ],
)
def test_compute_pairwise_angles(bvecs1, bvecs2, closest_polarity, expected):
    # DIPY requires the vectors to be normalized
    _bvecs1 = (bvecs1 / np.linalg.norm(bvecs1, axis=0)).T
    _bvecs2 = None

    if bvecs2 is not None:
        _bvecs2 = (bvecs2 / np.linalg.norm(bvecs2, axis=0)).T

    obtained = gpr.compute_pairwise_angles(_bvecs1, _bvecs2, closest_polarity)

    if _bvecs2 is not None:
        assert (_bvecs1.shape[0], _bvecs2.shape[0]) == obtained.shape
    assert obtained.shape == expected.shape
    np.testing.assert_array_almost_equal(obtained, expected, decimal=2)


@pytest.mark.parametrize("covariance", ["Spherical", "Exponential"])
def test_kernel(repodata, covariance):
    """Check kernel construction."""

    bvals, bvecs = read_bvals_bvecs(
        str(repodata / "ds000114_singleshell.bval"),
        str(repodata / "ds000114_singleshell.bvec"),
    )

    bvecs = bvecs[bvals > 10]

    KernelType = getattr(gpr, f"{covariance}Kriging")
    kernel = KernelType()
    K = kernel(bvecs)

    assert K.shape == (bvecs.shape[0],) * 2

    assert np.allclose(np.diagonal(K), kernel.diag(bvecs))

    K_predict = kernel(bvecs, [bvecs[10, ...]])

    assert K_predict.shape == (K.shape[0], 1)

    K_predict = kernel(bvecs, bvecs[10:14, ...])
    assert K_predict.shape == (K.shape[0], 4)


def test_unknown_optimizer():
    # Create a GPR with an optimizer string that is not supported
    optimizer = "bad-optimizer"
    gp = gpr.DiffusionGPR(optimizer=optimizer)  # type: ignore

    # A minimal objective function (will not be called by this test path)
    def obj_func(theta, eval_gradient):
        return 0.0

    initial_theta = np.array([0.1])
    bounds = [(0.0, 1.0)]

    # Expect the specific ValueError message including the optimizer name
    with pytest.raises(
        ValueError, match=gpr.UNKNOWN_OPTIMIZER_ERROR_MSG.format(optimizer=optimizer)
    ):
        gp._constrained_optimization(obj_func, initial_theta, bounds)


@pytest.mark.parametrize(
    "beta_a, beta_l, a_bounds, l_bounds, eval_gradient",
    [
        (0.5, 2.0, (0.1, 1.0), (0.1, 10.0), False),
        (1.0, 1.0, (0.1, 2.0), (0.01, 5.0), True),
    ],
)
def test_exponential_kriging_properties(beta_a, beta_l, a_bounds, l_bounds, eval_gradient):
    ek = gpr.ExponentialKriging(beta_a=beta_a, beta_l=beta_l, a_bounds=a_bounds, l_bounds=l_bounds)

    # Hyperparameter properties
    hp_a = ek.hyperparameter_a
    hp_l = ek.hyperparameter_l
    assert getattr(hp_a, "name", None) == "beta_a"
    assert getattr(hp_l, "name", None) == "beta_l"
    # Bounds should match what we passed (fallback to instance attr if absent)
    hp_a_bounds = np.asarray(getattr(hp_a, "bounds", ek.a_bounds), dtype=float)
    expected_a_bounds = np.asarray(ek.a_bounds, dtype=float)
    assert np.allclose(hp_a_bounds, expected_a_bounds)
    hp_l_bounds = np.asarray(getattr(hp_l, "bounds", ek.l_bounds), dtype=float)
    expected_l_bounds = np.asarray(ek.l_bounds, dtype=float)
    assert np.allclose(hp_l_bounds, expected_l_bounds)

    X = _unit_vectors_nd(2, 2)
    K2, _ = ek(X, eval_gradient=eval_gradient)

    # stationarity and repr
    assert ek.is_stationary() is True
    assert "ExponentialKriging" in repr(ek)
    assert f"a={ek.beta_a}" in repr(ek) or "beta_a" in repr(ek)


@pytest.mark.parametrize(
    "beta_a, beta_l, a_bounds, l_bounds, eval_gradient",
    [
        (10.0, 0.5, (0.1, 20.0), (0.1, 5.0), False),  # large a to trigger deriv_a active
        (1.5, 2.0, (0.1, 5.0), (0.1, 10.0), True),
    ],
)
def test_spherical_kriging_properties(beta_a, beta_l, a_bounds, l_bounds, eval_gradient):
    sk = gpr.SphericalKriging(beta_a=beta_a, beta_l=beta_l, a_bounds=a_bounds, l_bounds=l_bounds)

    hp_a = sk.hyperparameter_a
    hp_l = sk.hyperparameter_l
    assert getattr(hp_a, "name", None) == "beta_a"
    assert getattr(hp_l, "name", None) == "beta_l"
    hp_a_bounds = np.asarray(getattr(hp_a, "bounds", sk.a_bounds), dtype=float)
    expected_a_bounds = np.asarray(sk.a_bounds, dtype=float)
    assert np.allclose(hp_a_bounds, expected_a_bounds)
    hp_l_bounds = np.asarray(getattr(hp_l, "bounds", sk.l_bounds), dtype=float)
    expected_l_bounds = np.asarray(sk.l_bounds, dtype=float)
    assert np.allclose(hp_l_bounds, expected_l_bounds)

    X = _unit_vectors_nd(2, 2)
    K, _ = sk(X, eval_gradient=eval_gradient)

    # stationarity and repr
    assert sk.is_stationary() is True
    assert "SphericalKriging" in repr(sk)


@pytest.mark.parametrize(
    "beta_a, beta_l, a_bounds, l_bounds, n_samples, n_features, eval_gradient",
    [
        (0.5, 2.0, (0.1, 1.0), (0.1, 10.0), 2, 3, False),
        (1.0, 1.0, (0.1, 2.0), (0.01, 5.0), 3, 4, True),
    ],
)
def test_exponential_kriging_basic(
    beta_a, beta_l, a_bounds, l_bounds, n_samples, n_features, eval_gradient
):
    ek = gpr.ExponentialKriging(beta_a=beta_a, beta_l=beta_l, a_bounds=a_bounds, l_bounds=l_bounds)

    X = _unit_vectors_nd(n_samples, n_features)
    if eval_gradient:
        K, grad = ek(X, eval_gradient=True)
    else:
        K = ek(X, eval_gradient=False)
        grad = None

    assert K.shape == (n_samples, n_samples)

    thetas = gpr.compute_pairwise_angles(X)
    expected = ek.beta_l * gpr.exponential_covariance(thetas, ek.beta_a)
    assert np.allclose(K, expected)

    if eval_gradient:
        assert grad is not None
        assert grad.shape == (*thetas.shape, 2)  # two params: a and lambda

        C_theta = gpr.exponential_covariance(thetas, ek.beta_a)
        deriv_a = ek.beta_l * C_theta * thetas / (ek.beta_a**2)
        deriv_lambda = C_theta

        expected_grad = np.zeros((*thetas.shape, 2))
        expected_grad[..., 0] = deriv_a
        expected_grad[..., 1] = deriv_lambda

        assert np.allclose(grad, expected_grad)

    # diag
    d = ek.diag(X)
    assert d.shape == (n_samples,)
    assert np.allclose(d, ek.beta_l * np.ones(n_samples))


@pytest.mark.parametrize(
    "beta_a, beta_l, a_bounds, l_bounds, n_samples, n_features, eval_gradient",
    [
        (10.0, 0.5, (0.1, 20.0), (0.1, 5.0), 2, 3, False),  # large a to trigger deriv_a active
        (1.5, 2.0, (0.1, 5.0), (0.1, 10.0), 3, 4, True),
    ],
)
def test_spherical_kriging_basic(
    beta_a, beta_l, a_bounds, l_bounds, n_samples, n_features, eval_gradient
):
    sk = gpr.SphericalKriging(beta_a=beta_a, beta_l=beta_l, a_bounds=a_bounds, l_bounds=l_bounds)

    X = _unit_vectors_nd(n_samples, n_features)
    if eval_gradient:
        K, grad = sk(X, eval_gradient=True)
    else:
        K = sk(X, eval_gradient=False)
        grad = None

    assert K.shape == (n_samples, n_samples)

    thetas = gpr.compute_pairwise_angles(X)
    expected = sk.beta_l * gpr.spherical_covariance(thetas, sk.beta_a)
    assert np.allclose(K, expected)

    if eval_gradient:
        assert grad is not None
        assert grad.shape == (*thetas.shape, 2)  # two params: a and lambda

        # deriv_a as implemented in SphericalKriging
        deriv_a = np.zeros_like(thetas)
        nonzero = thetas <= sk.beta_a
        deriv_a[nonzero] = (
            1.5
            * sk.beta_l
            * (thetas[nonzero] / sk.beta_a**2 - thetas[nonzero] ** 3 / sk.beta_a**4)
        )
        expected_grad = np.dstack((deriv_a, gpr.spherical_covariance(thetas, sk.beta_a)))

        assert np.allclose(grad, expected_grad)

    # diag
    d = sk.diag(X)
    assert d.shape == (n_samples,)
    assert np.allclose(d, sk.beta_l * np.ones(n_samples))
