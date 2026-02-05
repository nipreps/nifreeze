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

from abc import ABC, ABCMeta, abstractmethod
from os import cpu_count
from typing import Union

import numpy as np

# from scipy.sparse import linalg as la
from scipy import linalg as la
from scipy.interpolate import BSpline

from nifreeze.data.pet import PET
from nifreeze.model.base import BaseModel

PET_OBJECT_ERROR_MSG = "Dataset MUST be a PET object."
"""PET object error message."""

PET_MIDFRAME_ERROR_MSG = "Dataset MUST have a 'midframe'."
"""PET midframe error message."""

DEFAULT_TIMEPOINT_TOL = 1e-2
"""Time frame tolerance in seconds."""

MIN_N_TIMEPOINTS = 4
"""Minimum number of timepoints for PET model fitting."""

DEFAULT_BSPLINE_N_CTRL = 3
"""Default number of B-Spline control points."""

DEFAULT_BSPLINE_ORDER = 3
"""Default B-Spline order."""

MIN_TIMEPOINTS_ERROR_MSG = """\
'min_timepoints' must be a valid dataset size."""
"""PET model fitting minimum fitting timepoint allowed values error."""


def _exec_fit(model, data, chunk=None, **kwargs):
    return model.fit(data, **kwargs), chunk


def _exec_predict(model, chunk=None, **kwargs):
    """Propagate model parameters and call predict."""
    return np.squeeze(model.predict(**kwargs)), chunk


class BasePETModel(BaseModel, ABC):
    """Interface and default methods for PET models."""

    __metaclass__ = ABCMeta

    __slots__ = {
        "_model_class": "Defining a model class",
        "_modelargs": "Arguments acceptable by the underlying model",
        "_models": "List with one or more (if parallel execution) model instances",
    }

    def __init__(
        self,
        dataset: PET,
        min_timepoints: int = MIN_N_TIMEPOINTS,
        **kwargs,
    ):
        """Initialization.

        Parameters
        ----------
        dataset : :obj:`~nifreeze.data.pet.base.PET`
            Reference to a PET object.
        min_timepoints : :obj:`int`, optional
            Minimum number of timepoints required to fit the model.

        """

        super().__init__(dataset, **kwargs)

        # Duck typing, instead of explicitly testing for PET type
        if not hasattr(dataset, "total_duration"):
            raise TypeError(PET_OBJECT_ERROR_MSG)

        if not hasattr(dataset, "midframe"):
            raise ValueError(PET_MIDFRAME_ERROR_MSG)

        if not 0 < min_timepoints < len(self._dataset):
            raise ValueError(MIN_TIMEPOINTS_ERROR_MSG)

    @property
    def is_fitted(self) -> bool:
        return self._locked_fit is not None

    @abstractmethod
    def fit_predict(self, index: int | None = None, **kwargs) -> Union[np.ndarray, None]:
        """Predict the corrected volume."""
        return None


class BSplinePETModel(BasePETModel):
    """A PET imaging realignment model based on B-Spline approximation."""

    __slots__ = {
        "_t": "B-Spline knot time-coordinates",
        "_order": "B-Spline order",
        "_n_ctrl": "Number of B-Spline control points",
    }

    def __init__(
        self,
        dataset: PET,
        n_ctrl: int = DEFAULT_BSPLINE_N_CTRL,
        order: int = DEFAULT_BSPLINE_ORDER,
        **kwargs,
    ):
        """Create the B-Spline interpolating matrix.

        Parameters
        ----------
        n_ctrl : :obj:`int`, optional
            Number of B-Spline control points. If :obj:`None`, then one control
            point every six timepoints will be used. The less control points,
            the smoother is the model.
            Please note that this is the number of control points `within the extent`
            of the data, and extra control points are added at either side to
            compensate edge effects when interpolating the initial or end timepoints.
        order : :obj:`int`, optional
            Order of the B-Spline approximation.

        """

        if order < 1:
            raise ValueError("B-Spline order must be at least 1.")

        if n_ctrl is not None and n_ctrl < 1:
            raise ValueError("Number of B-Spline control points must be at least 1.")

        super().__init__(dataset, **kwargs)

        self._order = order

        # Number of control points for the B-Spline basis
        self._n_ctrl = n_ctrl

        # Start and end timepoints
        x0 = float(self._dataset.midframe[0])
        x1 = float(self._dataset.midframe[-1])
        inner_x = np.linspace(x0, x1, self._n_ctrl + 2, dtype="float32")[1:-1]

        # Time-coordinates of the B-Spline knots
        self._t = np.concatenate(
            [
                np.full(self._order + 1, x0, dtype="float32"),
                inner_x,
                np.full(self._order + 1, x1, dtype="float32"),
            ]
        )

    def fit_predict(self, index: int | None = None, **kwargs) -> Union[np.ndarray, None]:
        """Return the corrected volume using B-spline interpolation.

        Predictions for times earlier than the configured start time will return
        the prediction for the start time.
        """

        n_jobs = kwargs.pop("n_jobs", min(cpu_count() or 1, 8))

        # TODO: locked fit handling
        if self._locked_fit is not None:
            return n_jobs

        # Generate a time mask for the frames to fit
        x_mask = np.ones(len(self._dataset), dtype=bool)

        x_mask[index] = False if index is not None else x_mask[index]
        x = self._dataset.midframe[x_mask].tolist()

        # A.shape = (T, K - 4); t= n. timepoints, K= n. knots (with padding)
        A = BSpline.design_matrix(x, t=self._t, k=self._order, extrapolate=False)

        # Get data as 2D array (voxels x timepoints)
        data = (
            self._dataset.dataobj.reshape((-1, len(self._dataset)))
            if self._dataset.brainmask is None
            else self._dataset.dataobj[self._dataset.brainmask, :]
        )

        X = la.lstsq(
            A.toarray().astype("float64"),
            data[:, x_mask].T.astype("float64"),
            cond=None,
            lapack_driver="gelsd",
        )
        coefficients = X[0].T.astype("float32")

        # Generate an interpolation time mask
        interp_mask = np.zeros(len(self._dataset), dtype=bool)
        interp_index = index if index is not None else slice(0, len(interp_mask))
        interp_mask[interp_index] = True

        A = BSpline.design_matrix(
            self._dataset.midframe[interp_mask], self._t, k=self._order, extrapolate=False
        )

        # A is T (num. timepoints) x C (num. coeff)
        # coefficients is V (num. voxels) x C (num. coeff)
        predicted = np.squeeze(A @ coefficients.T).T

        brainmask = self._dataset.brainmask
        datashape = self._dataset.dataobj.shape[:3]

        if brainmask is None:
            return predicted.reshape(datashape)

        retval = np.squeeze(np.zeros_like(self._dataset.dataobj[..., interp_mask]))
        retval[brainmask] = predicted
        return retval
