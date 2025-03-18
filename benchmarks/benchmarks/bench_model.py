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
"""Benchmarking for nifreeze's models."""

from abc import ABC

import dipy.data as dpd
import nibabel as nb
import numpy as np
from dipy.core.gradients import get_bval_indices
from dipy.io import read_bvals_bvecs
from dipy.segment.mask import median_otsu
from scipy.ndimage import binary_dilation
from skimage.morphology import ball

from nifreeze.model.gpr import DiffusionGPR, SphericalKriging


class DiffusionGPRBenchmark(ABC):
    def __init__(self):
        self._estimator = None
        self._X_train = None
        self._y_train = None
        self._X_test = None
        self._y_test = None

    def setup(self, *args, **kwargs):
        beta_a = 1.38
        beta_l = 1 / 2.1
        alpha = 0.1
        disp = True
        optimizer = None
        self.make_estimator((beta_a, beta_l, alpha, disp, optimizer))
        self.make_data()

    def make_estimator(self, params):
        beta_a, beta_l, alpha, disp, optimizer = params
        kernel = SphericalKriging(beta_a=beta_a, beta_l=beta_l)
        self._estimator = DiffusionGPR(
            kernel=kernel,
            alpha=alpha,
            disp=disp,
            optimizer=optimizer,
        )

    def make_data(self):
        name = "sherbrooke_3shell"

        dwi_fname, bval_fname, bvec_fname = dpd.get_fnames(name=name)
        dwi_data = nb.load(dwi_fname).get_fdata()
        bvals, bvecs = read_bvals_bvecs(bval_fname, bvec_fname)

        _, brain_mask = median_otsu(dwi_data, vol_idx=[0])
        brain_mask = binary_dilation(brain_mask, ball(8))

        bval = 1000
        indices = get_bval_indices(bvals, bval, tol=20)

        bvecs_shell = bvecs[indices]
        shell_data = dwi_data[..., indices]
        dwi_vol_idx = len(indices) // 2

        # Prepare a train/test mask (False for all directions except the left-out where it's true)
        train_test_mask = np.zeros(bvecs_shell.shape[0], dtype=bool)
        train_test_mask[dwi_vol_idx] = True

        # Generate train/test bvecs
        self._X_train = bvecs_shell[~train_test_mask, :]
        self._X_test = bvecs_shell[train_test_mask, :]

        # Select voxels within brain mask
        y = shell_data[brain_mask]

        # Generate train/test data
        self._y_train = y[:, ~train_test_mask]
        self._y_test = y[:, train_test_mask]

    def time_fit(self, *args):
        self._estimator = self._estimator.fit(self._X_train, self._y_train.T)

    def time_predict(self):
        self._estimator.predict(self._X_test)
