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
    """A :obj:`~dipy.reconst.dki.DiffusionKurtosisModel` accepting engine kwargs at fit time.

    DIPY decorates DKI's ``multi_fit`` (not ``fit``) with ``multi_voxel_fit``, so
    stock DKI only accepts the parallelization arguments (``engine``, ``n_jobs``,
    ...) through the *constructor*, unlike every other multi-voxel model, which
    accepts them at *fit* time. This subclass overrides ``fit`` to forward
    call-time orchestration kwargs into the already-decorated ``multi_fit``,
    giving DKI the same decorated-``fit`` interface and letting NiFreeze
    parallelize it uniformly with the other models.
    """

    def fit(self, data, *, mask=None, **kwargs):
        """Fit the DKI model, forwarding orchestration kwargs to ``multi_fit``.

        The serial/delegate path (no orchestration kwargs) preserves stock
        behavior, including populating ``self.extra`` for iterative/robust fits.
        The parallel path is only taken for the standard (WLS/OLS) multi-voxel
        methods, whose ``multi_fit`` carries no ``extra`` diagnostics.
        """
        # No orchestration kwargs, or a path multi_fit does not cover: delegate.
        if not kwargs or not self.is_multi_method or self.is_iter_method:
            return super().fit(data, mask=mask)

        data_thres = np.maximum(data, self.min_signal)
        return self.multi_fit(
            data_thres, mask=mask, weights=self.weights, **{**self.kwargs, **kwargs}
        )[0]
