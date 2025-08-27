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
        normalize_peaks=False,
        sphere=None,
    ):
        """Generalized Q-Sampling Imaging."""
        ReconstModel.__init__(self, gtab)
        self.method = method
        self.Lambda = sampling_length
        self.normalize_peaks = normalize_peaks
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

    def fit(self, data, **kwargs):
        return GeneralizedQSamplingFit(self, data)

    def predict(self, odf, *, S0=None):
        """
        Predict a signal for this GeneralizedQSamplingModel instance given parameters.

        Parameters
        ----------
        odf : ndarray
            Map of ODFs.
        gtab : ndarray
            Orientations where signal will be simulated
        sphere : :obj:`~dipy.core.sphere.Sphere`
            A sphere object, must be the same used for calculating the ODFs.

        """

        return prediction_kernel(self.gtab, self.Lambda, self.sphere, method=self.method)


class GeneralizedQSamplingFit(ReconstFit):
    def __init__(self, model, data):
        """Calculates PDF and ODF for a single voxel

        Parameters
        ----------
        model : object,
            DiffusionSpectrumModel
        data : 1d ndarray,
            signal values

        """
        ReconstFit.__init__(self, model, data)
        self._gfa = None
        self.npeaks = 5
        self._peak_values = None
        self._peak_indices = None
        self._qa = None
        self.odf_fit = None

    def fit(self, data, *, mask=None):
        if self.odf_fit is None:
            self.odf_fit = self.odf(data)
        return self

    def odf(self):
        """Calculates the discrete ODF for a given discrete sphere."""
        return self.model.kernel @ self.data

    def predict(self, gtab, *, S0=None):
        """Predict using the fit model."""
        K = (
            prediction_kernel(
                gtab,
                self.model.Lambda,
                self.model.sphere,
            )
            @ self.model.kernel
        )

        return (K @ self.data.T).T


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
    Predict a signal given the ODF.

    Parameters
    ----------
    odf : ndarray
        ODF parameters.

    gtab : GradientTable
        The gradient table for this prediction

    Notes
    -----
    The predicted signal is given by:

    .. math::

        S(\theta, b) = K_{ii}^{-1} \cdot ODF

    where :math:`K_{ii}^{-1}`, is the inverse of the GQI kernels for the
    direction(s) :math:`ii` given by ``gtab``.

    """
    # K.shape = (n_gradients, n_vertices)
    K = gqi_kernel(gtab, param_lambda, sphere, method=method)
    GtG = K @ K.T
    identity = np.eye(GtG.shape[0])
    return np.linalg.inv(GtG + INVERSE_LAMBDA * identity) @ K


def normalize_qa(qa, *, max_qa=None):
    """Normalize quantitative anisotropy.

    Used mostly with GQI rather than GQI2.

    Parameters
    ----------
    qa : array, shape (X, Y, Z, N)
        where N is the maximum number of peaks stored
    max_qa : float,
        maximum qa value. Usually found in the CSF (corticospinal fluid).

    Returns
    -------
    nqa : array, shape (x, Y, Z, N)
        normalized quantitative anisotropy

    Notes
    -----
    Normalized quantitative anisotropy has the very useful property
    to be very small near gray matter and background areas. Therefore,
    it can be used to mask out white matter areas.

    """
    if max_qa is None:
        return qa / qa.max()
    return qa / max_qa


def squared_radial_component(x, *, tol=0.01):
    """Part of the GQI2 integral."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = (2 * x * np.cos(x) + (x * x - 2) * np.sin(x)) / (x**3)
    x_near_zero = (x < tol) & (x > -tol)
    return np.where(x_near_zero, 1.0 / 3, result)


def npa(self, odf, *, width=5):
    """non-parametric anisotropy

    Nimmo-Smith et al.  ISMRM 2011
    """
    # odf = self.odf(s)
    t0, t1, t2 = triple_odf_maxima(self.odf_vertices, odf, width)
    psi0 = t0[1] ** 2
    psi1 = t1[1] ** 2
    psi2 = t2[1] ** 2
    npa = np.sqrt((psi0 - psi1) ** 2 + (psi1 - psi2) ** 2 + (psi2 - psi0) ** 2) / np.sqrt(
        2 * (psi0**2 + psi1**2 + psi2**2)
    )
    # print 'tom >>>> ',t0,t1,t2,npa

    return t0, t1, t2, npa


def equatorial_zone_vertices(vertices, pole, *, width=5):
    """
    finds the 'vertices' in the equatorial zone conjugate
    to 'pole' with width half 'width' degrees
    """
    return [
        i
        for i, v in enumerate(vertices)
        if np.abs(np.dot(v, pole)) < np.abs(np.sin(np.pi * width / 180))
    ]


def polar_zone_vertices(vertices, pole, *, width=5):
    """
    finds the 'vertices' in the equatorial band around
    the 'pole' of radius 'width' degrees
    """
    return [
        i
        for i, v in enumerate(vertices)
        if np.abs(np.dot(v, pole)) > np.abs(np.cos(np.pi * width / 180))
    ]


def upper_hemi_map(v):
    """
    maps a 3-vector into the z-upper hemisphere
    """
    return np.sign(v[2]) * v


def equatorial_maximum(vertices, odf, pole, width):
    eqvert = equatorial_zone_vertices(vertices, pole, width)
    # need to test for whether eqvert is empty or not
    if len(eqvert) == 0:
        print(f"empty equatorial band at {np.array_str(pole)}  pole with width {width:f}")
        return None, None
    eqvals = [odf[i] for i in eqvert]
    eqargmax = np.argmax(eqvals)
    eqvertmax = eqvert[eqargmax]
    eqvalmax = eqvals[eqargmax]

    return eqvertmax, eqvalmax


def patch_vertices(vertices, pole, width):
    """
    find 'vertices' within the cone of 'width' degrees around 'pole'
    """
    return [
        i
        for i, v in enumerate(vertices)
        if np.abs(np.dot(v, pole)) > np.abs(np.cos(np.pi * width / 180))
    ]


def patch_maximum(vertices, odf, pole, width):
    eqvert = patch_vertices(vertices, pole, width)
    # need to test for whether eqvert is empty or not
    if len(eqvert) == 0:
        print(f"empty cone around pole {np.array_str(pole)} with with width {width:f}")
        return np.Null, np.Null
    eqvals = [odf[i] for i in eqvert]
    eqargmax = np.argmax(eqvals)
    eqvertmax = eqvert[eqargmax]
    eqvalmax = eqvals[eqargmax]
    return eqvertmax, eqvalmax


def odf_sum(odf):
    return np.sum(odf)


def patch_sum(vertices, odf, pole, width):
    eqvert = patch_vertices(vertices, pole, width)
    # need to test for whether eqvert is empty or not
    if len(eqvert) == 0:
        print(f"empty cone around pole {np.array_str(pole)} with with width {width:f}")
        return np.Null
    return np.sum([odf[i] for i in eqvert])


def triple_odf_maxima(vertices, odf, width):
    indmax1 = np.argmax([odf[i] for i, v in enumerate(vertices)])
    odfmax1 = odf[indmax1]
    pole = vertices[indmax1]
    eqvert = equatorial_zone_vertices(vertices, pole, width)
    indmax2, odfmax2 = equatorial_maximum(vertices, odf, pole, width)
    indmax3 = eqvert[np.argmin([np.abs(np.dot(vertices[indmax2], vertices[p])) for p in eqvert])]
    odfmax3 = odf[indmax3]
    """
    cross12 = np.cross(vertices[indmax1],vertices[indmax2])
    cross12 = cross12/np.sqrt(np.sum(cross12**2))
    indmax3, odfmax3 = patch_maximum(vertices, odf, cross12, 2*width)
    """
    return [(indmax1, odfmax1), (indmax2, odfmax2), (indmax3, odfmax3)]
