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
"""DIPY-like models (a sandbox to trial them out before upstreaming to DIPY)."""

from __future__ import annotations

from typing import cast

import numpy as np
from dipy.core.gradients import GradientTable
from dipy.reconst.base import ReconstModel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Kernel, KernelOperator, WhiteKernel

from nifreeze.model.gpr import (
    DiffusionGPR,
    ExponentialKriging,
    MultiShellKernel,
    SphericalKriging,
)

GP_JITTER = 1e-10
"""Small nugget kept on ``alpha`` for numerical stability."""


def _cartesian_components(gtab: GradientTable | np.ndarray, keep_bval: bool = False) -> np.ndarray:
    """Return the gradient directions as Cartesian components.

    When ``keep_bval`` is :obj:`True`, the b-value is appended as a trailing
    column (layout ``[gx, gy, gz, bval]``) so multi-shell kernels can access it;
    otherwise only the 3 orientation components are returned.
    """

    if hasattr(gtab, "bvecs"):
        gtab = cast(GradientTable, gtab)
        components = gtab.bvecs
        if keep_bval:
            components = np.column_stack([components, np.asarray(gtab.bvals)])
    else:
        components = np.asarray(gtab)
        if components.ndim == 1:
            components = components[np.newaxis, :]
        if not keep_bval and components.shape[-1] > 3:
            components = components[:, :-1]
    return components


def _kernel_uses_bval(kernel: Kernel | None) -> bool:
    """Return whether ``kernel`` (possibly compound) contains a multi-shell kernel.

    The b-value column is only needed in the design matrix when a
    :obj:`~nifreeze.model.gpr.MultiShellKernel` is somewhere in the kernel. Because
    the measurement-noise :obj:`~sklearn.gaussian_process.kernels.WhiteKernel` is
    added as a :obj:`~sklearn.gaussian_process.kernels.KernelOperator`, the check
    must recurse into the operands.
    """
    if isinstance(kernel, MultiShellKernel):
        return True
    if isinstance(kernel, KernelOperator):
        return _kernel_uses_bval(kernel.k1) or _kernel_uses_bval(kernel.k2)
    return False


def gp_prediction(
    model: GaussianProcessRegressor,
    gtab: GradientTable | np.ndarray,
    mask: np.ndarray | None = None,
    return_std: bool = False,
) -> np.ndarray:
    """
    Predicts one or more DWI orientations given a model.

    This function checks if the model is fitted and then extracts
    orientations and potentially b-values from the X. It predicts the mean
    and standard deviation of the DWI signal using the model.

    Parameters
    ----------
    model : :obj:`~sklearn.gaussian_process.GaussianProcessRegressor`
        A fitted GaussianProcessRegressor model.
    gtab : :obj:`~dipy.core.gradients.GradientTable` or :obj:`~np.ndarray`
        Gradient table with one or more orientations at which the GP will be evaluated.
    mask : :obj:`~numpy.ndarray`, optional
        A boolean mask indicating which voxels to use (optional).
    return_std : bool, optional
        Whether to return the standard deviation of the predicted signal.

    Returns
    -------
    :obj:`~numpy.ndarray`
        A 3D or 4D array with the simulated gradient(s).

    """

    keep_bval = _kernel_uses_bval(getattr(model, "kernel", None))
    X = _cartesian_components(gtab, keep_bval=keep_bval)

    # Check it's fitted as they do in sklearn internally
    # https://github.com/scikit-learn/scikit-learn/blob/972e17fe1aa12d481b120ad4a3dc076bae736931/\
    # sklearn/gaussian_process/_gpr.py#L410C9-L410C42
    if not hasattr(model, "X_train_"):
        raise RuntimeError("Model is not yet fitted.")

    # X holds the gradient orientations (and the b-value too, for multi-shell).
    orientations = model.predict(X, return_std=return_std)
    assert isinstance(orientations, np.ndarray)
    return orientations


class GaussianProcessModel(ReconstModel):
    """A Gaussian Process (GP) model to simulate single- and multi-shell DWI data."""

    __slots__ = (
        "kernel",
        "_modelfit",
        "sigma_sq",
    )

    def __init__(
        self,
        kernel_model: str = "spherical",
        beta_l: float = 2.0,
        beta_a: float = 0.1,
        sigma_sq: float = 1.0,
        ell: float = 1.0,
        *args,
        **kwargs,
    ) -> None:
        """A GP-based DWI model.

        Implements a GP-based model :footcite:p:`andersson_non-parametric_2015`
        to reconstruct DWI data employing a given DWI fitting model.

        Parameters
        ----------
        kernel_model : :obj:`str`, optional
            Angular covariance model. One of ``"spherical"`` (default),
            ``"exponential"``, or ``"multishell"``. ``"multishell"`` builds a
            :obj:`~nifreeze.model.gpr.MultiShellKernel` (angular × radial product,
            Eqs. 14–15) and expects the b-value to be preserved in the design
            matrix.
        beta_l : :obj:`float`, optional
            Signal scale parameter determining the variability of the signal.
        beta_a : :obj:`float`, optional
            Distance scale parameter determining how fast the covariance
            decreases as one moves along the surface of the sphere. Must have a
            positive value.
        sigma_sq : :obj:`float`, optional
            Initial measurement-noise variance (:math:`\\sigma^2`). It is the
            starting ``noise_level`` of a
            :obj:`~sklearn.gaussian_process.kernels.WhiteKernel` added to the
            covariance kernel, and is *optimized* along with the other
            hyperparameters (rather than held fixed as in a plain ``alpha``).
        ell : :obj:`float`, optional
            Radial (log-b) length scale for the multi-shell kernel (:math:`\\ell`
            in Eq. 15). Only used when ``kernel_model == "multishell"``.

        References
        ----------
        .. footbibliography::

        """

        ReconstModel.__init__(self, None)

        self.sigma_sq = sigma_sq

        if kernel_model == "multishell":
            self.kernel = MultiShellKernel(
                orientation_kernel=SphericalKriging(beta_a=beta_a, beta_l=beta_l),
                radial_kernel=RBF(length_scale=ell),
            )
        else:
            KernelType = SphericalKriging if kernel_model == "spherical" else ExponentialKriging
            # Add the :math:`\sigma^2` term of Andersson et al. (2015) as a WhiteKernel
            self.kernel = KernelType(
                beta_a=beta_a,
                beta_l=beta_l,
            ) + WhiteKernel(noise_level=sigma_sq)

    def fit(
        self,
        data: np.ndarray,
        gtab: GradientTable | np.ndarray,
        mask: np.ndarray | None = None,
        random_state: int = 0,
    ) -> GPFit:
        """Fit method of the DTI model class

        Parameters
        ----------
        gtab : :obj:`~dipy.core.gradients.GradientTable` or :obj:`~np.ndarray`
            The gradient table corresponding to the training data.
        data : :obj:`~numpy.ndarray`
            The measured signal from one voxel.
        mask : :obj:`~numpy.ndarray`
            A boolean array used to mark the coordinates in the data that
            should be analyzed that has the shape data.shape[:-1]
        random_state: :obj:`int`, optional
            Determines random number generation used to initialize the centers
            of the kernel bounds.

        Returns
        -------
        :obj:`~nifreeze.model.dipy.GPFit`
            A model fit container object.

        """

        # Extract b-vecs: scikit-learn wants (n_samples, n_features) where
        # n_samples is the different diffusion-encoding gradient orientations.
        # n_features is 3 (orientation only), or 4 ([gx, gy, gz, bval]) when a
        # multi-shell kernel needs the b-value.
        keep_bval = _kernel_uses_bval(self.kernel)
        X = _cartesian_components(gtab, keep_bval=keep_bval)

        # Data must have shape (n_samples, n_targets) where n_samples is
        # the number of diffusion-encoding gradient orientations, and n_targets
        # is number of voxels.
        y = (
            data[mask[..., None]] if mask is not None else np.reshape(data, (-1, data.shape[-1]))
        ).T

        if (grad_dirs := X.shape[0]) != (signal_dirs := y.shape[0]):
            raise ValueError(
                f"Mismatched gradient directions in data ({signal_dirs}) "
                f"and gradient table ({grad_dirs})."
            )

        gpr = DiffusionGPR(
            kernel=self.kernel,
            random_state=random_state,
            n_targets=y.shape[1],
            alpha=GP_JITTER,
        )
        self._modelfit = GPFit(
            model=gpr.fit(X, y),
            mask=mask,
        )
        return self._modelfit

    def predict(
        self,
        gtab: GradientTable | np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """
        Predict using the Gaussian process model of the DWI signal for one or more gradients.

        Parameters
        ----------
        gtab : :obj:`~dipy.core.gradients.GradientTable` or :obj:`~np.ndarray`
            Gradient table with one or more orientations at which the GP will be evaluated.

        Returns
        -------
        :obj:`~numpy.ndarray`
            A 3D or 4D array with the simulated gradient(s).

        """
        return self._modelfit.predict(gtab)


class GPFit:
    """
    A container class to store the fitted Gaussian process model and mask information.

    This class is typically returned by the `fit` and `multi_fit` methods of the
    `GaussianProcessModel` class. It holds the fitted model and the mask used during fitting.

    Attributes
    ----------
    model : :obj:`~sklearn.gaussian_process.GaussianProcessRegressor`
        The fitted Gaussian process regressor object.
    mask : :obj:`~numpy.ndarray`
        The boolean mask used during fitting (can be :obj:`None`).

    """

    def __init__(
        self,
        model: GaussianProcessRegressor,
        mask: np.ndarray | None = None,
    ) -> None:
        """
        Initialize a Gaussian Process fit container.

        Parameters
        ----------
        model : :obj:`~sklearn.gaussian_process.GaussianProcessRegressor`
            The fitted Gaussian process regressor object.
        mask : :obj:`~numpy.ndarray`, optional
            The boolean mask used during fitting.

        """
        self.model = model
        self.mask = mask

    def predict(
        self,
        gtab: GradientTable | np.ndarray,
    ) -> np.ndarray:
        """
        Generate DWI signal based on a fitted Gaussian Process.

        Parameters
        ----------
        gtab : :obj:`~dipy.core.gradients.GradientTable` or :obj:`~np.ndarray`
            Gradient table with one or more orientations at which the GP will be evaluated.

        Returns
        -------
        :obj:`~numpy.ndarray`
            A 3D or 4D array with the simulated gradient(s).

        """
        return gp_prediction(self.model, gtab, mask=self.mask)
