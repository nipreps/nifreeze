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
"""CPU time of the models.

``time_cpu_*`` benchmarks time with :func:`~._common.cpu_time`, which counts the
CPU seconds of this process and of joblib's worker processes. CPU time does not
include waiting, so it changes less than wall time when other jobs share the
machine. BLAS runs on one thread, as the workflow sets it.
"""

import time
import warnings

import numpy as np
from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]

from nifreeze.data.dmri.base import DWI
from nifreeze.data.pet.base import PET
from nifreeze.data.pet.io import from_nii as from_nii_pet
from nifreeze.model.base import ModelFactory
from nifreeze.model.pet import BSplinePETModel

from ._common import cpu_time, ellipsoid_mask, write_pet
from .bench_memory import MODEL_BVALS, write_model_dataset

try:
    from nifreeze.model.lovo import lovo_smoother
except ImportError:  # not on commits before nifreeze#499
    lovo_smoother = None  # type: ignore[assignment]

_HELD_OUT = len(MODEL_BVALS) // 2


def _init_model(model, path):
    """Return a model of the 18 MB series in ``path``."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ModelFactory.init(model=model, dataset=DWI.from_filename(path))


class _CPUBenchmark:
    """Sampling shared by the CPU benchmarks: at least 6 samples per commit."""

    version = "1"
    timer = staticmethod(cpu_time)
    number = 1
    rounds = 2
    repeat = (3, 10, 5.0)
    warmup_time = 0.0

    def setup(self, *args):
        warnings.simplefilter("ignore")  # GP fits warn about convergence on noise
        self._threads = threadpool_limits(limits=1)

    def teardown(self, *args):
        if (threads := getattr(self, "_threads", None)) is not None:
            threads.restore_original_limits()


# As in ``bench_memory``, ``setup_cache`` returns the file it wrote and
# ``setup`` takes it: asv runs a ``setup`` without arguments before ``setup_cache``.


class ModelCPU(_CPUBenchmark):
    """CPU time of one leave-one-volume-out fit and prediction."""

    params = ["DTI", "DKI", "GP", "GP-LOO", "GQI-KRR"]
    param_names = ["model"]

    def setup_cache(self):
        return write_model_dataset()

    def setup(self, path, model):
        super().setup()
        # ``setup`` runs again before every sample, so each sample fits a new model
        self.model = _init_model(model, path)

    def time_cpu_fit_predict(self, path, model):
        self.model.fit_predict(_HELD_OUT, n_jobs=1)


class GPPredictCPU(_CPUBenchmark):
    """CPU time of GP predictions of every orientation for every masked voxel."""

    def setup_cache(self):
        return write_model_dataset()

    def setup(self, path):
        super().setup()
        self.model = _init_model("GP", path)
        self.model.fit_predict(None)  # fit once on all volumes

    def time_cpu_predict(self, path):
        for index in range(int((MODEL_BVALS > 0).sum())):
            self.model.fit_predict(index)


class DKIParallel(_CPUBenchmark):
    """CPU and wall time of a DKI fit over two joblib workers."""

    # One call before sampling starts the workers, which then serve every sample
    warmup_time = 0.5

    def setup_cache(self):
        return write_model_dataset()

    def setup(self, path):
        super().setup()
        self.model = _init_model("DKI", path)

    def time_cpu_fit_predict(self, path):
        self.model.fit_predict(_HELD_OUT, n_jobs=2)

    def time_fit_predict(self, path):
        self.model.fit_predict(_HELD_OUT, n_jobs=2)

    # Wall time, so the dashboard shows what the second worker saves
    time_fit_predict.timer = time.perf_counter  # type: ignore[attr-defined]


class PETModelCPU(_CPUBenchmark):
    """CPU time of a B-spline leave-one-frame-out fit and prediction on 9 MB."""

    def setup_cache(self):
        rng = np.random.default_rng(1234)
        shape3d = (48, 48, 32)
        data = rng.random((*shape3d, 32), dtype=np.float32)
        args = write_pet("pet", data, ellipsoid_mask(shape3d, 0.45))
        from_nii_pet(**args).to_filename("pet.h5")
        return "pet.h5"

    def setup(self, path):
        super().setup()
        self.model = BSplinePETModel(PET.from_filename(path))

    def time_cpu_fit_predict(self, path):
        self.model.fit_predict(16)


class ClosedFormLOVOCPU(_CPUBenchmark):
    """CPU time of the closed-form leave-one-volume-out smoother (``nifreeze.model.lovo``)."""

    params = [90, 270]
    param_names = ["n_volumes"]
    repeat = (5, 20, 5.0)
    number = 0  # calibrate, as some calls take under a millisecond

    def setup(self, n_volumes):
        if lovo_smoother is None:
            raise NotImplementedError("nifreeze.model.lovo is not available")
        super().setup()
        rng = np.random.default_rng(1234)
        factor = rng.standard_normal((n_volumes, n_volumes + 10))
        self.covariance = factor @ factor.T / n_volumes + 0.1 * np.eye(n_volumes)
        self.smoother = lovo_smoother(self.covariance, center=True).astype(np.float32)
        self.signals = rng.random((n_volumes, 100_000), dtype=np.float32)

    def time_cpu_smoother(self, n_volumes):
        lovo_smoother(self.covariance, center=True)

    def time_cpu_predict_all_volumes(self, n_volumes):
        self.smoother @ self.signals
