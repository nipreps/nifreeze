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
"""Wall time of the data layer on a 9 MB series."""

from pathlib import Path

import numpy as np

from nifreeze.data.base import to_nifti as to_nifti_base
from nifreeze.data.dmri.base import DWI
from nifreeze.data.dmri.io import from_nii as from_nii_dwi
from nifreeze.data.dmri.io import to_nifti as to_nifti_dwi
from nifreeze.data.pet.base import PET
from nifreeze.data.pet.io import from_nii as from_nii_pet

from ._common import dwi_series, ellipsoid_mask, gradient_table, write_dwi, write_pet

_SHAPE3D = (48, 48, 32)
_N_VOLUMES = 32


class DWIData:
    """Wall time of DWI input and output."""

    version = "1"
    rounds = 2
    repeat = (5, 20, 5.0)

    def setup_cache(self):
        rng = np.random.default_rng(1234)
        bvals = np.full(_N_VOLUMES, 1000.0)
        args = write_dwi(
            "dwi",
            dwi_series(_SHAPE3D, bvals, rng),
            gradient_table(bvals, rng),
            ellipsoid_mask(_SHAPE3D, 0.45),
        )
        from_nii_dwi(**args).to_filename("dwi.h5")
        return args

    def setup(self, args):
        self.dwi = DWI.from_filename("dwi.h5")

    def teardown(self, args):
        for name in ("out_dwi.h5", "out_dwi.nii"):
            Path(name).unlink(missing_ok=True)

    def time_from_nii(self, args):
        from_nii_dwi(**args)

    def time_from_filename(self, args):
        DWI.from_filename("dwi.h5")

    def time_getitem(self, args):
        self.dwi[len(self.dwi) // 2]

    def time_to_filename(self, args):
        self.dwi.to_filename("out_dwi.h5")

    def time_to_nifti(self, args):
        to_nifti_dwi(self.dwi, "out_dwi.nii")


class PETData:
    """Wall time of PET input and output."""

    version = "1"
    rounds = 2
    repeat = (5, 20, 5.0)

    def setup_cache(self):
        rng = np.random.default_rng(1234)
        data = rng.random((*_SHAPE3D, _N_VOLUMES), dtype=np.float32)
        args = write_pet("pet", data, ellipsoid_mask(_SHAPE3D, 0.45))
        from_nii_pet(**args).to_filename("pet.h5")
        return args

    def setup(self, args):
        self.pet = PET.from_filename("pet.h5")

    def teardown(self, args):
        for name in ("out_pet.h5", "out_pet.nii"):
            Path(name).unlink(missing_ok=True)

    def time_from_nii(self, args):
        from_nii_pet(**args)

    def time_from_filename(self, args):
        PET.from_filename("pet.h5")

    def time_getitem(self, args):
        self.pet[len(self.pet) // 2]

    def time_to_filename(self, args):
        self.pet.to_filename("out_pet.h5")

    def time_to_nifti(self, args):
        to_nifti_base(self.pet, "out_pet.nii")
