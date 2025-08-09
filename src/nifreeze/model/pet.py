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

from importlib import import_module
from os import cpu_count
from typing import Any

import nibabel as nb
import numpy as np
from joblib import Parallel, delayed
from nibabel.processing import smooth_image
from scipy.interpolate import BSpline
from scipy.sparse.linalg import cg

from nifreeze.data.pet import PET
from nifreeze.model.base import BaseModel

DEFAULT_TIMEFRAME_MIDPOINT_TOL = 1e-2
"""Time frame tolerance in seconds."""


def _exec_fit(model, data, chunk=None, **kwargs):
    return model.fit(data, **kwargs), chunk


def _exec_predict(model, chunk=None, **kwargs):
    """Propagate model parameters and call predict."""
    return np.squeeze(model.predict(**kwargs)), chunk


class BasePETModel(BaseModel):
    """Interface and default methods for PET models."""

    __slots__ = {
        "_data_mask": "A mask for the voxels that will be fitted and predicted",
        "_x" : "",
        "_xlim" : "",
        "_smooth_fwhm" : "",
        "_thresh_pct" : "",
        "_model_class": "Defining a model class",
        "_modelargs": "Arguments acceptable by the underlying model.",
        "_models": "List with one or more (if parallel execution) model instances",
    }

    def __init__(
        self,
        dataset : PET,
        timepoints=None,
        xlim=None,
        smooth_fwhm=10,
        thresh_pct=20,
        **kwargs,
    ):
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
        super().__init__(dataset, **kwargs)

        # Duck typing, instead of explicitly test for DWI type
        if not hasattr(dataset, "timepoints"):
            raise TypeError("Dataset MUST be a PET object.")
        # ToDO
        # Are the timepoints your "gradients" ??? If so, can they be computed
        # from frame_time or frame_duration
        # Or else frame_time and frame_duration ????

        self._data_mask = (
            dataset.brainmask
            if dataset.brainmask is not None
            else np.ones(dataset.dataobj.shape[:3], dtype=bool)
        )

        if timepoints is None or xlim is None:
            raise TypeError("timepoints must be provided in initialization")

        self._x = np.array(timepoints, dtype="float32")
        self._xlim = xlim
        self._smooth_fwhm = smooth_fwhm
        self._thresh_pct = thresh_pct

        if self._x[0] < DEFAULT_TIMEFRAME_MIDPOINT_TOL:
            raise ValueError("First frame midpoint should not be zero or negative")
        if self._x[-1] > (self._xlim - DEFAULT_TIMEFRAME_MIDPOINT_TOL):
            raise ValueError("Last frame midpoint should not be equal or greater than duration")

        super().__init__(dataset, **kwargs)

    @property
    def is_fitted(self):
        return self._locked_fit is not None

    def _fit(self, index: int | None = None, n_jobs=None, **kwargs):
        """Fit the model."""

        n_jobs = n_jobs or 1

        if self._locked_fit is not None:
            return n_jobs

        if index is not None:
            raise NotImplementedError("Fitting with held-out data is not supported")
        timepoints = kwargs.get("timepoints", None) or self._x

        # data, _, gtab = self._dataset[idxmask]  ### This needs the PET data model to be changed
        data = self._dataset.dataobj
        brainmask = self._dataset.brainmask

        # Preprocess the data
        if self._smooth_fwhm > 0:
            smoothed_img = smooth_image(
                nb.Nifti1Image(data, self._dataset.affine), self._smooth_fwhm
            )
            data = smoothed_img.get_fdata()

        if self._thresh_pct > 0:
            thresh_val = np.percentile(data, self._thresh_pct)
            data[data < thresh_val] = 0

        # Convert data into V (voxels) x T (timepoints)
        data = data.reshape((-1, data.shape[-1])) if brainmask is None else data[brainmask]

        model_str = getattr(self, "_model_class", "")
        module_name, class_name = model_str.rsplit(".", 1)
        model = getattr(
                import_module(module_name),
                class_name,
            )(gtab, **kwargs)  ## What is the gtab of PET ???

        fit_kwargs: dict[str, Any] = {}  # Add here keyword arguments

        # Split data into chunks of group of slices
        data_chunks = np.array_split(data, n_jobs)

        self._models = [None] * n_jobs

        # Parallelize process with joblib
        with Parallel(n_jobs=n_jobs) as executor:
            results = executor(
                delayed(_exec_fit)(model, dchunk, i, **fit_kwargs)
                for i, dchunk in enumerate(data_chunks)
            )
        for submodel, rindex in results:
            self._models[rindex] = submodel

        return n_jobs


    def fit_predict(self, index: int | None = None, **kwargs):
        """Return the corrected volume using B-spline interpolation."""

        n_models = self._fit(
            index,
            n_jobs=kwargs.pop("n_jobs"),
            **kwargs,
        )

        if index is None:  # If no index, just fit the data.
            return None

        ## What is the gtab of a PET dataset "gtab": gradient, "S0": self._S0
        if n_models == 1:
            predicted, _ = _exec_predict(
                self._models[0], **(kwargs | {"gtab": gradient, "S0": self._S0})
            )
        else:
            predicted = [None] * n_models
            S0 = np.array_split(self._S0, n_models)

            # Parallelize process with joblib
            with Parallel(n_jobs=n_models) as executor:
                results = executor(
                    delayed(_exec_predict)(
                        model,
                        chunk=i,
                        **(kwargs | {"gtab": gradient, "S0": S0[i]}),
                    )
                    for i, model in enumerate(self._models)
                )
            for subprediction, index in results:
                predicted[index] = subprediction

            predicted = np.hstack(predicted)

        retval = np.zeros_like(self._data_mask, dtype=self._dataset.dataobj.dtype)
        retval[self._data_mask, ...] = predicted
        return retval


class BSplinePETModel(BasePETModel):
    """A PET imaging realignment model based on B-Spline approximation."""

    __slots__ = (
        "_t",
        "_order",
        "_n_ctrl",
    )

    def __init__(
        self,
        dataset : PET,
        timepoints=None,
        n_ctrl=None,
        order=3,
        **kwargs,
    ):
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
        super().__init__(dataset, **kwargs)

        self._order = order

        # Calculate index coordinates in the B-Spline grid
        self._n_ctrl = n_ctrl or (len(timepoints) // 4) + 1

        # B-Spline knots
        self._t = np.arange(-3, float(self._n_ctrl) + 4, dtype="float32")


    @property
    def is_fitted(self):
        return self._locked_fit is not None

    def _fit(self, index: int | None = None, n_jobs=None, **kwargs):
        """Fit the model."""

        if self._locked_fit is not None:
            return n_jobs

        if index is not None:
            raise NotImplementedError("Fitting with held-out data is not supported")
        timepoints = kwargs.get("timepoints", None) or self._x
        x = (np.array(timepoints, dtype="float32") / self._xlim) * self._n_ctrl

        data = self._dataset.dataobj

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

        if index is None:
            raise RuntimeError(
                f"Model {self.__class__.__name__} does not allow locking.")

        # Project sample timing into B-Spline coordinates
        x = (index / self._xlim) * self._n_ctrl
        A = BSpline.design_matrix(x, self._t, k=self._order)

        # A is 1 (num. timepoints) x C (num. coeff)
        # self._locked_fit is V (num. voxels) x K - 4
        predicted = np.squeeze(A @ self._locked_fit.T)

        brainmask = self._dataset.brainmask
        datashape = self._dataset.dataobj.shape[:3]

        if brainmask is None:
            return predicted.reshape(datashape)

        retval = np.zeros_like(self._dataset.dataobj[..., 0])
        retval[brainmask] = predicted
        return retval
