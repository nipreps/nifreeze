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
from joblib import Parallel, delayed
from scipy.interpolate import BSpline
from scipy.sparse.linalg import cg

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

START_INDEX_RANGE_ERROR_MSG = """\
'start_index' must be a valid dataset index."""
"""PET model fitting start index allowed values error."""

END_INDEX_RANGE_ERROR_MSG = """\
'end_index' must be a valid dataset index later than 'start_index'."""
"""PET model fitting start index allowed values error."""


def _exec_fit(model, data, chunk=None, **kwargs):
    return model.fit(data, **kwargs), chunk


def _exec_predict(model, chunk=None, **kwargs):
    """Propagate model parameters and call predict."""
    return np.squeeze(model.predict(**kwargs)), chunk


class BasePETModel(BaseModel, ABC):
    """Interface and default methods for PET models."""

    __metaclass__ = ABCMeta

    __slots__ = {
        "_start_index": "Start index frame for fitting",
        "_end_index": "End index frame for fitting",
        "_model_class": "Defining a model class",
        "_modelargs": "Arguments acceptable by the underlying model",
        "_models": "List with one or more (if parallel execution) model instances",
    }

    def __init__(
        self,
        dataset: PET,
        start_index: int = 0,
        end_index: int | None = None,
        min_timepoints: int = MIN_N_TIMEPOINTS,
        **kwargs,
    ):
        """Initialization.

        Parameters
        ----------
        start_index : :obj:`int`, optional
            If provided, the model will be fitted using only timepoints starting
            from this index (inclusive). Predictions for timepoints earlier than
            the specified start will reuse the predicted volume for the start
            timepoint. This is useful, for example, to discard a number of
            frames at the beginning of the sequence, which due to their little
            SNR may impact registration negatively.
        end_index : :obj:`int`, optional
            If provided, the model will be fitted using only timepoints up to
            this index (exclusive).
        min_timepoints : :obj:`int`, optional
            Minimum number of timepoints required to fit the model.

        """

        super().__init__(dataset, **kwargs)

        # Duck typing, instead of explicitly testing for PET type
        if not hasattr(dataset, "total_duration"):
            raise TypeError(PET_OBJECT_ERROR_MSG)

        if not hasattr(dataset, "midframe"):
            raise ValueError(PET_MIDFRAME_ERROR_MSG)

        if 0 > start_index >= len(self._dataset) - min_timepoints:
            raise ValueError(START_INDEX_RANGE_ERROR_MSG)

        self._start_index = start_index

        self._end_index = len(self._dataset)
        if end_index is not None:
            if not (start_index + min_timepoints < end_index <= len(self._dataset)):
                raise ValueError(END_INDEX_RANGE_ERROR_MSG)
            self._end_index = end_index

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
        "_edges": "Edge handling strategy",
    }

    def __init__(
        self,
        dataset: PET,
        n_ctrl: int | None = None,
        order: int = 3,
        edges: str = "mirror",
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

        super().__init__(dataset, **kwargs)

        self._order = order
        self._edges = edges

        # Number of control points for the B-Spline basis
        self._n_ctrl = n_ctrl or (len(self._dataset) // 4) + 1
        ctrl_sep = self._dataset.total_duration / (self._n_ctrl + 1)

        # Control point indices (with padding for B-Spline of order k at either side)
        _ctrl_idx = np.arange(-(order + 1), self._n_ctrl + (order + 2), dtype="float32")

        # Time-coordinates of the B-Spline knots
        self._t = self._dataset.midframe[0] + 0.5 * ctrl_sep + _ctrl_idx * ctrl_sep
        # self._t[_ctrl_idx < -1] = self._t[_ctrl_idx < - 1][-1]
        # self._t[_ctrl_idx > self._n_ctrl + 1] = self._t[_ctrl_idx > self._n_ctrl + 1][0]

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
        # x_mask[: self._start_index] = False
        # x_mask[self._end_index :] = False

        x_mask[index] = False if index is not None else x_mask[index]
        x = self._dataset.midframe[x_mask].tolist()

        if self._edges == "mirror":
            # Pad time coordinates to avoid edge effects
            x = [-x[1]] + x + [2 * x[-1] - x[-2]]

        # A.shape = (T, K - 4); t= n. timepoints, K= n. knots (with padding)
        A = BSpline.design_matrix(x, t=self._t, k=self._order, extrapolate=True)
        AT = A.T
        ATdotA = AT @ A

        # Get data as 2D array (voxels x timepoints)
        data = (
            self._dataset.dataobj.reshape((-1, len(self._dataset)))
            if self._dataset.brainmask is None
            else self._dataset.dataobj[self._dataset.brainmask, :]
        )
        # Filter timepoints according to time mask and mirror ends
        data = (
            np.hstack([data[:, 0][:, None], data[:, x_mask], data[:, -1][:, None]])
            if self._edges == "mirror"
            else data[:, x_mask]
        )

        # Parallelize fitting process with joblib
        with Parallel(n_jobs=n_jobs) as executor:
            results = executor(delayed(cg)(ATdotA, AT @ v) for v in data)

        coefficients = np.asarray([r[0] for r in results])

        # Generate an interpolation time mask
        interp_mask = np.zeros(len(self._dataset), dtype=bool)
        interp_index = index if index is not None else slice(self._start_index, self._end_index)
        interp_mask[interp_index] = True

        A = BSpline.design_matrix(
            self._dataset.midframe[interp_mask], self._t, k=self._order, extrapolate=True
        )

        # A is T (num. timepoints) x C (num. coeff)
        # coefficients is V (num. voxels) x C (num. coeff)
        predicted = np.squeeze(A @ coefficients.T)

        brainmask = self._dataset.brainmask
        datashape = self._dataset.dataobj.shape[:3]

        if brainmask is None:
            return predicted.reshape(datashape)

        retval = np.squeeze(np.zeros_like(self._dataset.dataobj[..., interp_mask]))
        retval[brainmask] = predicted
        return retval
