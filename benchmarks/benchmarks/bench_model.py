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

import time

import dipy.data as dpd
import nibabel as nb
import numpy as np
from dipy.core.gradients import get_bval_indices
from dipy.io import read_bvals_bvecs
from dipy.segment.mask import median_otsu
from joblib import cpu_count
from scipy.ndimage import binary_dilation
from skimage.morphology import ball
from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]

from nifreeze.data.dmri import DWI
from nifreeze.model.dmri import DKIModel
from nifreeze.model.gpr import DiffusionGPR, SphericalKriging
from nifreeze.utils.ndimage import load_api


class _DKIBaseBenchmark:
    """Shared setup/timing utilities for DKI ASV benchmarks.
    Not collected directly by ASV (no time_/track_ methods required by users).
    """

    params = ([1000, 2000, 5000], [1, min(4, cpu_count())])
    param_names = ["n_voxels", "n_jobs"]

    _WARMUP_RUNS = 1
    _MEASURE_RUNS = 3

    def __init__(self):
        self._dataset: DWI | None = None
        self._index: int | None = None
        self._serial_cache = {}  # keyed by n_voxels

    def setup(self, n_voxels, n_jobs):
        name = "sherbrooke_3shell"
        dwi_fname, bval_fname, bvec_fname = dpd.get_fnames(name=name)

        img = load_api(dwi_fname, nb.Nifti1Image)
        dwi_data = img.get_fdata()
        bvals, bvecs = read_bvals_bvecs(bval_fname, bvec_fname)

        _, brain_mask = median_otsu(dwi_data, vol_idx=[0])
        brain_mask = binary_dilation(brain_mask, ball(8))

        flat_mask = np.flatnonzero(brain_mask)
        n_voxels = min(n_voxels, flat_mask.size)

        subset_mask = np.zeros(brain_mask.shape, dtype=bool)
        subset_mask[np.unravel_index(flat_mask[:n_voxels], brain_mask.shape)] = True

        gradients = np.hstack((bvecs, bvals[..., np.newaxis]))
        bzero = dwi_data[..., bvals < 50].mean(axis=-1)

        self._dataset = DWI(
            dataobj=dwi_data,
            affine=img.affine,
            brainmask=subset_mask,
            gradients=gradients,
            bzero=bzero,
        )

        shell_1000 = get_bval_indices(bvals, 1000, tol=20)
        self._index = shell_1000[len(shell_1000) // 2]

    def _fit_predict_once(self, n_jobs):
        assert self._dataset is not None
        assert self._index is not None

        with (
            threadpool_limits(limits=1, user_api="blas"),
            threadpool_limits(limits=1, user_api="openmp"),
        ):
            t0 = time.perf_counter()
            DKIModel(self._dataset).fit_predict(self._index, n_jobs=n_jobs)
            return time.perf_counter() - t0

    def _median_runtime(self, n_jobs):
        for _ in range(self._WARMUP_RUNS):
            self._fit_predict_once(n_jobs=n_jobs)
        ts = [self._fit_predict_once(n_jobs=n_jobs) for _ in range(self._MEASURE_RUNS)]
        return float(np.median(ts))

    def _serial_time_for(self, n_voxels):
        if n_voxels not in self._serial_cache:
            self._serial_cache[n_voxels] = self._median_runtime(n_jobs=1)
        return self._serial_cache[n_voxels]


class DKITimingBenchmark(_DKIBaseBenchmark):
    """Absolute runtime benchmark."""

    unit = "seconds"

    def time_fit_predict(self, n_voxels, n_jobs):
        assert self._dataset is not None
        assert self._index is not None
        return self._median_runtime(n_jobs=n_jobs)


class DKISpeedupBenchmark(_DKIBaseBenchmark):
    """Parallel speedup benchmark relative to n_jobs=1."""

    unit = "ratio"

    def track_parallel_speedup(self, n_voxels, n_jobs):
        assert self._dataset is not None
        assert self._index is not None

        if n_jobs == 1:
            return 1.0

        t_serial = self._serial_time_for(n_voxels)
        t_parallel = self._median_runtime(n_jobs=n_jobs)
        return t_serial / t_parallel if t_parallel > 0 else float("inf")


class DiffusionGPRBenchmark:
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
        dwi_data = load_api(dwi_fname, nb.Nifti1Image).get_fdata()
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
        assert self._estimator is not None
        assert self._y_train is not None
        self._estimator = self._estimator.fit(self._X_train, self._y_train.T)

    def time_predict(self):
        assert self._estimator is not None
        self._estimator.predict(self._X_test)
