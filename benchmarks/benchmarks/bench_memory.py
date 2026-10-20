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
"""Memory used by the data layer and the models.

Every benchmark reports MB, one of:

- ``track_alloc_*``: the peak of the allocations Python traces while the
  operation runs (:mod:`tracemalloc`), which is the same on every run. A
  memory-mapped read allocates almost nothing; an eager one, the series.
- ``track_rss_*``: the median peak RSS increase of forked children that run the
  operation. It also counts what :mod:`tracemalloc` cannot see (pages read
  through a memory map, the buffers of HDF5), and varies by about 10%.
"""

import os
import warnings
from pathlib import Path

import numpy as np
from asv_runner.benchmarks.mark import skip_benchmark_if  # type: ignore

from nifreeze.data import load
from nifreeze.data.base import BaseDataset
from nifreeze.data.dmri.base import DWI
from nifreeze.data.dmri.io import from_nii as from_nii_dwi
from nifreeze.data.dmri.io import to_nifti as to_nifti_dwi
from nifreeze.data.pet.base import PET
from nifreeze.data.pet.io import from_nii as from_nii_pet
from nifreeze.model.base import ModelFactory
from nifreeze.model.pet import BSplinePETModel

from ._common import (
    dwi_series,
    ellipsoid_mask,
    gradient_table,
    load_one_volume,
    peak_alloc_mb,
    peak_rss_mb,
    write_dwi,
    write_legacy_h5,
    write_nifti,
    write_pet,
)

# A 40 MB float32 series: 64 volumes of 64 x 64 x 40 voxels
_SHAPE3D = (64, 64, 40)
_N_VOLUMES = 64
# The model series: 18 MB, with a small mask so that DKI fits in about a second
# while copies of the whole field of view still show
MODEL_SHAPE3D = (48, 48, 32)
MODEL_MASK_FRACTION = 0.2
MODEL_BVALS = np.concatenate([np.zeros(4), np.repeat([1000.0, 2000.0], 30)])


def write_model_dataset() -> str:
    """Write ``model.h5``, the two-shell series every model benchmark fits."""
    rng = np.random.default_rng(1234)
    args = write_dwi(
        "model",
        dwi_series(MODEL_SHAPE3D, MODEL_BVALS, rng),
        gradient_table(MODEL_BVALS, rng),
        ellipsoid_mask(MODEL_SHAPE3D, MODEL_MASK_FRACTION),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the b=0 volumes are averaged
        from_nii_dwi(**args).to_filename("model.h5")
    return "model.h5"


# Each ``setup_cache`` returns the files it wrote and each ``setup`` takes them:
# asv runs a ``setup`` without arguments before ``setup_cache``.


class DWIMemory:
    """Peak allocations of DWI input and output on a 40 MB series."""

    version = "1"
    unit = "MB"

    def setup_cache(self):
        rng = np.random.default_rng(1234)
        bvals = np.full(_N_VOLUMES, 1000.0)
        data = dwi_series(_SHAPE3D, bvals, rng)
        mask = ellipsoid_mask(_SHAPE3D, 0.45)
        gradients = gradient_table(bvals, rng)
        from_nii_dwi(**write_dwi("dwi", data, gradients, mask)).to_filename("dwi.h5")
        write_legacy_h5(
            "dwi_legacy.h5",
            "dmri",
            dataobj=data,
            affine=np.eye(4),
            brainmask=mask,
            gradients=gradients,
        )

        # The same series with four b=0 volumes, which loading moves to ``bzero``
        b0_bvals = np.where(np.arange(_N_VOLUMES) % 16 == 0, 0.0, 1000.0)
        nii_b0 = write_dwi("dwi_b0", data, gradient_table(b0_bvals, rng), mask)
        return {"h5": "dwi.h5", "legacy": "dwi_legacy.h5", "nii_b0": nii_b0}

    def setup(self, files):
        self.dwi = DWI.from_filename(files["h5"])

    def teardown(self, files):
        for name in ("out_dwi.h5", "out_dwi.nii"):
            Path(name).unlink(missing_ok=True)

    def track_alloc_from_filename(self, files):
        return peak_alloc_mb(DWI.from_filename, files["h5"])

    def track_alloc_from_filename_legacy(self, files):
        return peak_alloc_mb(DWI.from_filename, files["legacy"])

    def track_alloc_from_nii(self, files):
        return peak_alloc_mb(from_nii_dwi, **files["nii_b0"])

    def track_alloc_getitem(self, files):
        return peak_alloc_mb(self.dwi.__getitem__, len(self.dwi) // 2)

    def track_alloc_to_filename(self, files):
        return peak_alloc_mb(self.dwi.to_filename, "out_dwi.h5")

    def track_alloc_to_nifti(self, files):
        return peak_alloc_mb(to_nifti_dwi, self.dwi, "out_dwi.nii")


class PETMemory:
    """Peak allocations of PET input and of its model on a 40 MB series."""

    version = "1"
    unit = "MB"

    def setup_cache(self):
        rng = np.random.default_rng(1234)
        data = rng.random((*_SHAPE3D, _N_VOLUMES), dtype=np.float32)
        from_nii_pet(**write_pet("pet", data, ellipsoid_mask(_SHAPE3D, 0.45))).to_filename(
            "pet.h5"
        )
        return "pet.h5"

    def setup(self, path):
        self.pet = PET.from_filename(path)
        # Keep the copy that ``set_transform`` writes out of the temporary directory
        self.pet._filepath = Path("pet_cache.h5")

    def teardown(self, path):
        Path("pet_cache.h5").unlink(missing_ok=True)

    def track_alloc_from_filename(self, path):
        return peak_alloc_mb(PET.from_filename, path)

    def track_alloc_set_transform(self, path):
        affine = np.eye(4)
        affine[0, 3] = 0.5
        return peak_alloc_mb(self.pet.set_transform, len(self.pet) // 2, affine)

    def track_alloc_fit_predict(self, path):
        model = BSplinePETModel(self.pet)
        return peak_alloc_mb(model.fit_predict, len(self.pet) // 2)


class ModelMemory:
    """Peak allocations of one leave-one-volume-out fit, read from an 18 MB file."""

    version = "1"
    unit = "MB"
    params = ["DTI", "DKI", "GP", "GP-LOO"]
    param_names = ["model"]

    def setup_cache(self):
        return write_model_dataset()

    def setup(self, path, model):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = ModelFactory.init(model=model, dataset=DWI.from_filename(path))

    def track_alloc_fit_predict(self, path, model):
        return peak_alloc_mb(self.model.fit_predict, len(MODEL_BVALS) // 2, n_jobs=1)


class PeakRSS:
    """Peak RSS increase of loading a 132 MB series and reading one volume."""

    version = "1"
    unit = "MB"

    def setup_cache(self):
        rng = np.random.default_rng(1234)
        data = rng.random((96, 96, 56, 64), dtype=np.float32)
        write_nifti("rss.nii", data)
        load("rss.nii").to_filename("rss.h5")
        write_legacy_h5("rss_legacy.h5", "base", dataobj=data, affine=np.eye(4))
        return {"h5": "rss.h5", "legacy": "rss_legacy.h5"}

    @skip_benchmark_if(not hasattr(os, "fork"))
    def track_rss_from_filename(self, files):
        return peak_rss_mb(load_one_volume, BaseDataset, files["h5"])

    @skip_benchmark_if(not hasattr(os, "fork"))
    def track_rss_from_filename_legacy(self, files):
        return peak_rss_mb(load_one_volume, BaseDataset, files["legacy"])
