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
# STATEMENT OF CHANGES:
# This file was created from the original `dipy.reconst.gqi` module
# in DIPY.
# We will be piloting the use of this module in NiFreeze to later
# consider its inclusion in DIPY (https://github.com/dipy/dipy/pull/3553).
# The original module is licensed under the BSD-3-Clause, which is reproduced
# below:
#
# ORIGINAL LICENSE:
# Unless otherwise specified by LICENSE.txt files in individual
# directories, or within individual files or functions, all code is:
#
# Copyright (c) 2008-2025, dipy developers
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#     * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above
#        copyright notice, this list of conditions and the following
#        disclaimer in the documentation and/or other materials provided
#        with the distribution.
#
#     * Neither the name of the dipy developers nor the names of any
#        contributors may be used to endorse or promote products derived
#        from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""Classes and functions for generalized q-sampling"""

import warnings

import numpy as np
from dipy.core.subdivide_octahedron import create_unit_sphere
from dipy.reconst.base import ReconstFit, ReconstModel

INVERSE_LAMBDA = 1e-6
DEFAULT_SPHERE_RECURSION_LEVEL = 5


class GeneralizedQSamplingModel(ReconstModel):
    def __init__(
        self,
        gtab,
        *,
        method="standard",
        sampling_length=1.2,
        sphere=None,
    ):
        r"""Generalized Q-Sampling Imaging.

        Parameters
        ----------
        gtab : :obj:`~dipy.core.gradients.GradientTable`
            Gradient table of the data to fit.
        method : {"standard", "gqi2"}, optional
            GQI reconstruction variant. Defaults to ``"standard"`` (the sinc
            basis of Yeh et al., 2010, Eq. 6/9).

            .. note::

               This default **deliberately deviates from DIPY**, whose GQI
               model defaults to ``"gqi2"``. NiFreeze uses GQI as a *signal
               predictor* rather than for ODF/SDF reconstruction, and the
               ``"standard"`` sinc kernel round-trips the diffusion signal far
               more faithfully than ``"gqi2"`` (which is oscillatory and
               ill-conditioned as a signal-reconstruction operator). Anyone
               cross-referencing DIPY should note the changed default.
        sampling_length : float, optional
            Diffusion sampling length :math:`\sigma` (``lambda`` in Yeh Eq. 9);
            recommended range 1--1.3.
        sphere : :obj:`~dipy.core.sphere.Sphere`, optional
            ODF sampling sphere; defaults to a unit sphere at recursion level 5.
        """
        ReconstModel.__init__(self, gtab)
        self.method = method
        self.Lambda = sampling_length
        self.gtab = gtab
        self.sphere = (
            create_unit_sphere(recursion_level=DEFAULT_SPHERE_RECURSION_LEVEL)
            if sphere is None
            else sphere
        )

        # The gQI vector has shape (n_vertices, n_orientations)
        self.kernel = gqi_kernel(
            self.gtab,
            self.Lambda,
            self.sphere,
            method=self.method,
        ).T

    def fit(self, data, *, mask=None):
        return GeneralizedQSamplingFit(self, data)


class GeneralizedQSamplingFit(ReconstFit):
    def __init__(self, model, data):
        """Store the model and signal data for a fitted voxel (or voxels).

        Parameters
        ----------
        model : :obj:`GeneralizedQSamplingModel`
            The GQI model instance this fit belongs to.
        data : :obj:`~numpy.ndarray`
            Signal values, shaped ``(n_gradients,)`` for a single voxel or
            ``(n_voxels, n_gradients)`` for a masked volume.

        """
        ReconstFit.__init__(self, model, data)

    def predict(self, gtab, *, S0=None):
        r"""Predict the diffusion signal on ``gtab`` from the fitted data.

        .. note::

           This signal-to-signal prediction is a **NiFreeze extension** of GQI
           and is *not* part of Yeh et al. (2010), which defines only the
           forward signal-to-SDF transform (Eq. 6/9). Here the fitted signal is
           mapped to the SDF by the forward GQI kernel and then back to the
           signal domain by a Tikhonov-regularized reconstruction kernel
           (:func:`prediction_kernel`) -- i.e. the regularized least-squares
           signal whose forward GQI transform reproduces the fitted SDF.

        The result is clamped to non-negative values. ``S0`` is accepted for
        API compatibility with the other NiFreeze DWI models but is unused.

        Notes
        -----
        The kernel round-trip approximately preserves signal scale for the
        ``"standard"`` (sinc) kernel, but **not** for ``"gqi2"``: the gqi2
        kernel is more oscillatory/ill-conditioned as a reconstruction
        operator, so re-predicting a gqi2 prediction drifts substantially.
        Prefer ``"standard"`` when the prediction itself is the quantity of
        interest.
        """
        K = (
            prediction_kernel(
                gtab,
                self.model.Lambda,
                self.model.sphere,
                method=self.model.method,
            )
            @ self.model.kernel
        )

        return np.maximum((K @ self.data.T).T, 0)


def gqi_kernel(gtab, param_lambda, sphere, method="standard"):
    # 0.01506 = 6*D where D is the free water diffusion coefficient
    # l_values sqrt(6 D tau) D free water diffusion coefficient and
    # tau included in the b-value
    scaling = np.sqrt(gtab.bvals * 0.01506)
    b_vector = gtab.bvecs * scaling[:, None]

    if method == "gqi2":
        H = squared_radial_component
        return np.real(H(np.dot(b_vector, sphere.vertices.T) * param_lambda))
    elif method != "standard":
        warnings.warn(
            f'GQI model "{method}" unknown, falling back to "standard".',
            stacklevel=1,
        )
    return np.real(np.sinc(np.dot(b_vector, sphere.vertices.T) * param_lambda / np.pi))


def prediction_kernel(gtab, param_lambda, sphere, method="standard"):
    r"""
    Compute the Tikhonov-regularized reconstruction kernel for GQI.

    Parameters
    ----------
    gtab : :obj:`~dipy.core.gradients.GradientTable`
        The gradient table for which the kernel is computed.
    param_lambda : float
        The GQI sampling length (:math:`\lambda`).
    sphere : :obj:`~dipy.core.sphere.Sphere`
        The sphere whose vertices define the ODF sampling directions.
    method : str, optional
        GQI variant, either ``"standard"`` or ``"gqi2"``.

    Returns
    -------
    :obj:`~numpy.ndarray`
        The reconstruction kernel with shape ``(n_gradients, n_vertices)``.

    Notes
    -----
    With the forward GQI kernel :math:`\mathbf{K} \in \mathbb{R}^{n_g \times n_v}`,
    the reconstruction kernel is

    .. math::

        \mathbf{K}^{+} = (\mathbf{K} \mathbf{K}^T + \lambda_0 \mathbf{I})^{-1} \mathbf{K}

    where :math:`\lambda_0 = 10^{-6}` (``INVERSE_LAMBDA``) regularizes the
    inversion and :math:`\mathbf{I}` is the :math:`n_g \times n_g` identity.

    """
    # K.shape = (n_gradients, n_vertices)
    K = gqi_kernel(gtab, param_lambda, sphere, method=method)
    GtG = K @ K.T
    identity = np.eye(GtG.shape[0])
    return np.linalg.inv(GtG + INVERSE_LAMBDA * identity) @ K


def squared_radial_component(x, *, tol=0.01):
    """Part of the GQI2 integral."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = (2 * x * np.cos(x) + (x * x - 2) * np.sin(x)) / (x**3)
    x_near_zero = (x < tol) & (x > -tol)
    return np.where(x_near_zero, 1.0 / 3, result)
