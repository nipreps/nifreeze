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
"""Models for nuclear imaging."""

from os import cpu_count

import numpy as np
from joblib import Parallel, delayed

from nifreeze.model.base import BaseModel

DEFAULT_TIMEFRAME_MIDPOINT_TOL = 1e-2
"""Time frame tolerance in seconds."""


class PETModel(BaseModel):
    """A PET imaging realignment model based on B-Spline approximation."""

    __slots__ = ("_t", "_x", "_xlim", "_order", "_n_ctrl")

    def __init__(self, timepoints=None, xlim=None, n_ctrl=None, order=3, **kwargs):
        """
        Create the B-Spline interpolating matrix.

        Parameters:
        -----------
        timepoints : :obj:`list`
            The timing (in sec) of each PET volume.
            E.g., ``[15.,   45.,   75.,  105.,  135.,  165.,  210.,  270.,  330.,
            420.,  540.,  750., 1050., 1350., 1650., 1950., 2250., 2550.]``

        n_ctrl : :obj:`int`
            Number of B-Spline control points. If `None`, then one control point every
            six timepoints will be used. The less control points, the smoother is the
            model.

        """
        super.__init__(**kwargs)

        if timepoints is None or xlim is None:
            raise TypeError("timepoints must be provided in initialization")

        self._order = order

        self._x = np.array(timepoints, dtype="float32")
        self._xlim = xlim

        if self._x[0] < DEFAULT_TIMEFRAME_MIDPOINT_TOL:
            raise ValueError("First frame midpoint should not be zero or negative")
        if self._x[-1] > (self._xlim - DEFAULT_TIMEFRAME_MIDPOINT_TOL):
            raise ValueError("Last frame midpoint should not be equal or greater than duration")

        # Calculate index coordinates in the B-Spline grid
        self._n_ctrl = n_ctrl or (len(timepoints) // 4) + 1

        # B-Spline knots
        self._t = np.arange(-3, float(self._n_ctrl) + 4, dtype="float32")

    def _fit(self, index: int | None = None, n_jobs=None, **kwargs):
        """Fit the model."""
        from scipy.interpolate import BSpline
        from scipy.sparse.linalg import cg

        if self._locked_fit is not None:
            return n_jobs

        if index is not None:
            raise NotImplementedError("Fitting with held-out data is not supported")

        timepoints = kwargs.get("timepoints", None) or self._x
        x = (np.array(timepoints, dtype="float32") / self._xlim) * self._n_ctrl

        data = self._dataset.dataobj
        brainmask = self._dataset.brainmask

        # Convert data into V (voxels) x T (timepoints)
        data = data.reshape((-1, data.shape[-1])) if brainmask is None else data[brainmask]

        # A.shape = (T, K - 4); T= n. timepoints, K= n. knots (with padding)
        A = BSpline.design_matrix(x, self._t, k=self._order)
        AT = A.T
        ATdotA = AT @ A

        # Parallelize process with joblib
        with Parallel(n_jobs=n_jobs or min(cpu_count() or 1, 8)) as executor:
            results = executor(delayed(cg)(ATdotA, AT @ v) for v in data)

        self._locked_fit = np.array([r[0] for r in results])

    def fit_predict(self, index: int | None = None, **kwargs):
        """Return the corrected volume using B-spline interpolation."""
        from scipy.interpolate import BSpline

        # Fit the BSpline basis on all data
        if self._locked_fit is None:
            self._fit(index, n_jobs=kwargs.pop("n_jobs", None), **kwargs)

        if index is None:  # If no index, just fit the data.
            return None

        # Project sample timing into B-Spline coordinates
        x = (index / self._xlim) * self._n_ctrl
        A = BSpline.design_matrix(x, self._t, k=self._order)

        # A is 1 (num. timepoints) x C (num. coeff)
        # self._coeff is V (num. voxels) x K - 4
        predicted = np.squeeze(A @ self._locked_fit.T)

        brainmask = self._dataset.brainmask
        datashape = self._dataset.dataobj.shape[:3]

        if brainmask is None:
            return predicted.reshape(datashape)

        retval = np.zeros_like(self._dataset.dataobj[..., 0])
        retval[brainmask] = predicted
        return retval
