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
from dipy.reconst.dki import DiffusionKurtosisModel as _DipyDKIModel


class DiffusionKurtosisModel(_DipyDKIModel):
    """A :obj:`~dipy.reconst.dki.DiffusionKurtosisModel` with uniform API."""

    def fit(self, data, *, mask=None, **kwargs):
        """Fit the model to data."""
        # No orchestration kwargs, or a path multi_fit does not cover: delegate.
        if not kwargs or not self.is_multi_method or self.is_iter_method:
            return super().fit(data, mask=mask)

        data_thres = np.maximum(data, self.min_signal)
        return self.multi_fit(
            data_thres, mask=mask, weights=self.weights, **{**self.kwargs, **kwargs}
        )[0]
