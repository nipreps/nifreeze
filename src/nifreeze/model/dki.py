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
"""A diffusion kurtosis imaging (DKI) model with a normalized ``fit`` interface."""

import numpy as np
from dipy.reconst import multi_voxel
from dipy.reconst.dki import DiffusionKurtosisModel as _DipyDKIModel

_ORCHESTRATION_KWARGS = ("engine", "n_jobs", "vox_per_chunk", "verbose", "inflight_cap")
"""Parallelization kwargs consumed by DIPY's ``multi_voxel_fit``."""

_DIPY_STRIPS_ORCHESTRATION = hasattr(multi_voxel, "ORCHESTRATION_KWARGS")
"""Whether DIPY keeps orchestration kwargs out of the per-voxel fit (dipy/dipy#4054)."""


class _DropOrchestrationKwargs:
    """Call a DKI ``fit_method`` without the orchestration kwargs.

    Backport of dipy/dipy#4054 for DIPY releases whose parallel ``multi_voxel_fit``
    forwards them through ``multi_fit`` into the fit method (dipy/dipy#4053).
    A class rather than a closure, so the model stays picklable for joblib.
    Remove once the DIPY floor includes the fix.
    """

    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        for key in _ORCHESTRATION_KWARGS:
            kwargs.pop(key, None)
        return self.func(*args, **kwargs)


class DiffusionKurtosisModel(_DipyDKIModel):
    """A :obj:`~dipy.reconst.dki.DiffusionKurtosisModel` with uniform API."""

    def __init__(self, gtab, *args, **kwargs):
        super().__init__(gtab, *args, **kwargs)
        if not _DIPY_STRIPS_ORCHESTRATION and self.common_fit_method:
            self.fit_method = _DropOrchestrationKwargs(self.fit_method)

    def fit(self, data, *, mask=None, **kwargs):
        """Fit the model to data."""
        # No orchestration kwargs, or a path multi_fit does not cover: delegate.
        if not kwargs or not self.is_multi_method or self.is_iter_method:
            return super().fit(data, mask=mask)

        data_thres = np.maximum(data, self.min_signal)
        return self.multi_fit(
            data_thres, mask=mask, weights=self.weights, **{**self.kwargs, **kwargs}
        )[0]
