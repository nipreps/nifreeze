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
from dipy.reconst.gqi import squared_radial_component
from dipy.reconst.odf import OdfFit, OdfModel

INVERSE_LAMBDA = 1e-6
r"""
Tikhonov regularization weight :math:`\lambda_0` for the reconstruction kernel
:math:`(\mathbf{K}\mathbf{K}^{\mathsf T} + \lambda_0\mathbf{I})^{-1}\mathbf{K}`
(see :func:`prediction_kernel`).
"""

DEFAULT_SPHERE_RECURSION_LEVEL = 5
"""
Default icosahedral subdivision level of the ODF sampling sphere (1026
vertices); see :ref:`gqi-sphere-density` for the experiment justifying it.
"""

FREE_WATER_DIFFUSIVITY_6D = 0.01506
r"""
:math:`6 D` where :math:`D` is the free-water diffusion coefficient; the GQI
scaling factor :math:`\sqrt{6 D \tau}` with :math:`\tau` folded into the b-value.
"""


class GeneralizedQSamplingModel(OdfModel):
    def __init__(
        self,
        gtab,
        *,
        method="standard",
        sampling_length=1.2,
        sphere=None,
        recursion_level=DEFAULT_SPHERE_RECURSION_LEVEL,
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
               predictor*, and ``"standard"`` round-trips the diffusion signal
               more faithfully than ``"gqi2"`` (see
               :ref:`gqi-reconstruction-fidelity`). Anyone cross-referencing
               DIPY should note the changed default.
        sampling_length : float, optional
            Diffusion sampling length :math:`\sigma` (``lambda`` in Yeh Eq. 9);
            recommended range 1--1.3.
        sphere : :obj:`~dipy.core.sphere.Sphere`, optional
            ODF sampling sphere. When given, ``recursion_level`` is ignored.
        recursion_level : int, optional
            Subdivision level of the icosahedral ODF sampling sphere built when
            ``sphere`` is not provided; higher values give a denser sphere.
            Defaults to :data:`DEFAULT_SPHERE_RECURSION_LEVEL`. See
            :ref:`gqi-sphere-density` for the experiment justifying the default
            (past the fidelity knee for both single-shell and grid acquisitions,
            while denser spheres only keep paying off for grid/multi-shell data).
        """
        OdfModel.__init__(self, gtab)
        self.method = method
        self.Lambda = sampling_length
        self.gtab = gtab
        self.sphere = (
            create_unit_sphere(recursion_level=recursion_level) if sphere is None else sphere
        )

        # Forward GQI kernel, shape (n_gradients, n_vertices). Stored in this
        # (un-transposed) orientation to match DIPY's ``GeneralizedQSamplingFit``
        # so the eventual upstreaming (dipy/dipy#3553) needs no reshuffling.
        self.kernel = gqi_kernel(
            self.gtab,
            self.Lambda,
            self.sphere,
            method=self.method,
        )

    def fit(self, data, *, mask=None):
        return GeneralizedQSamplingFit(self, data)


class GeneralizedQSamplingFit(OdfFit):
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
        OdfFit.__init__(self, model, data)

    def odf(self, sphere=None):
        r"""Compute the discrete orientation distribution function (ODF/SDF).

        Applies the forward GQI kernel to the fitted signal, implementing the
        signal-to-SDF transform of Yeh et al. (2010). This satisfies the
        :obj:`~dipy.reconst.odf.OdfFit` contract shared by DIPY's ODF models and
        is distinct from :meth:`predict`, which maps signal back to signal.

        Parameters
        ----------
        sphere : :obj:`~dipy.core.sphere.Sphere`, optional
            ODF sampling sphere. When :obj:`None` (default), the model's sphere
            is reused and the pre-computed forward kernel is applied directly,
            avoiding recomputation.

        Returns
        -------
        :obj:`~numpy.ndarray`
            The discrete ODF, shaped ``(n_vertices,)`` for a single voxel or
            ``(n_voxels, n_vertices)`` for a masked volume. Unlike
            :meth:`predict`, the result is **not** clamped to non-negative
            values (matching DIPY).
        """
        if sphere is None:
            # ``model.kernel`` is the forward (n_gradients, n_vertices) kernel,
            # applied directly (matching DIPY's ``odf``).
            return self.data @ self.model.kernel

        return self.data @ gqi_kernel(
            self.model.gtab,
            self.model.Lambda,
            sphere,
            method=self.model.method,
        )

    def predict(self, gtab, *, S0=None):
        r"""Predict the diffusion signal on ``gtab`` from the fitted data.

        .. note::

           This signal-to-signal prediction is a **NiFreeze extension** of GQI;
           Yeh et al. (2010) defines only the forward signal-to-SDF transform.
           It maps the fitted signal to the SDF and back via a
           Tikhonov-regularized reconstruction kernel
           (:func:`prediction_kernel`). See :ref:`gqi-models` for the
           construction.

        The result is clamped to non-negative values. ``S0`` is accepted for
        API compatibility with the other NiFreeze DWI models but is unused.

        Notes
        -----
        The kernel round-trip approximately preserves signal scale for the
        ``"standard"`` (sinc) kernel, but not for ``"gqi2"``; prefer
        ``"standard"`` when the predicted amplitude matters. See
        :ref:`gqi-reconstruction-fidelity`.
        """
        # ``model.kernel`` is (n_gradients_in, n_vertices); transpose it to
        # contract the shared vertex axis against the reconstruction kernel,
        # yielding the fused (n_gradients_out, n_gradients_in) operator.
        K = (
            prediction_kernel(
                gtab,
                self.model.Lambda,
                self.model.sphere,
                method=self.model.method,
            )
            @ self.model.kernel.T
        )

        return np.maximum((K @ self.data.T).T, 0)


def gqi_kernel(gtab, param_lambda, sphere, method="standard"):
    r"""
    Forward GQI kernel, shape ``(n_gradients, n_vertices)``.

    Ported from DIPY's :obj:`dipy.reconst.gqi`, modularized as a function.

    Parameters
    ----------
    gtab : :obj:`~dipy.core.gradients.GradientTable`
        The gradient table for which the kernel is computed.
    param_lambda : float
        The GQI sampling length (:math:`\lambda`).
    sphere : :obj:`~dipy.core.sphere.Sphere`
        The sphere whose vertices define the ODF sampling directions.
    method : {"standard", "gqi2"}, optional
        GQI variant. ``"standard"`` implements the sinc reconstruction of Yeh
        et al. (2010), Eq. 6/9 (verified to machine precision against the paper
        and DIPY); ``"gqi2"`` uses the :math:`L^2`-weighted basis of Eq. 8
        (:func:`~dipy.reconst.gqi.squared_radial_component`, imported from DIPY).
        An unknown value falls back to ``"standard"`` with a warning.

    Returns
    -------
    :obj:`~numpy.ndarray`
        The forward GQI kernel with shape ``(n_gradients, n_vertices)``.

    """
    scaling = np.sqrt(gtab.bvals * FREE_WATER_DIFFUSIVITY_6D)
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
