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
"""Benchmarking for nifreeze's data classes."""

import json
import os
from pathlib import Path
from typing import Any

import nibabel as nb
import numpy as np
from asv_runner.benchmarks.mark import skip_benchmark_if  # type: ignore

from nifreeze.data.base import to_nifti as to_nifti_base
from nifreeze.data.dmri.base import DWI
from nifreeze.data.dmri.io import from_nii as from_nii_dwi
from nifreeze.data.dmri.io import to_nifti as to_nifti_dwi
from nifreeze.data.pet.base import PET
from nifreeze.data.pet.io import from_nii as from_nii_pet


class DWIBenchmark:
    """
    Benchmarks for DWI data class operations.
    """

    def __init__(self):
        self.dwi: DWI | Any = object()
        self.benchmark_dir = Path(__file__).parent / "bench_data_dwi"
        self.nii_file = self.benchmark_dir / "dwi.nii.gz"
        self.h5_file = self.benchmark_dir / "dwi.h5"
        self.brainmask_file = self.benchmark_dir / "brainmask.nii.gz"
        self.gradients_file = self.benchmark_dir / "gradients.txt"

        self.out_nii_file = self.benchmark_dir / "out_dwi.nii.gz"
        self.out_h5_file = self.benchmark_dir / "out_dwi.h5"
        self.out_gradients_file = self.benchmark_dir / "out_dwi.txt"

    def _generate_data(self):
        rng = np.random.default_rng(42)

        os.makedirs(self.benchmark_dir, exist_ok=True)

        # Create a test NIfTI file for the DWI object
        vol_size = (16, 16, 16)
        n_gradients = 10
        dataobj = rng.random((*vol_size, n_gradients)).astype("float32")
        affine = np.eye(4).astype("float32")

        # Create the brainmask
        brainmask_dataobj = rng.choice([True, False], size=vol_size).astype("uint8")

        # Create b-vals and b-vecs
        shells = (1000,)
        bvals = np.ones(n_gradients).astype("float32") * shells
        bvals[0] = 0
        bvecs = rng.random((3, n_gradients)).astype("float32").T
        bvecs = bvecs / np.linalg.norm(bvecs, axis=1, keepdims=True)
        bvecs[0] = np.zeros_like(bvecs[0]).astype("float32")

        gradients = np.column_stack((bvecs, bvals)).astype("float32")

        return dataobj, affine, brainmask_dataobj, gradients

    def setup(self):
        dataobj, affine, brainmask_dataobj, gradients = self._generate_data()

        os.makedirs(self.benchmark_dir, exist_ok=True)

        dwi_img = nb.Nifti1Image(dataobj, affine)
        nb.save(dwi_img, self.nii_file)

        brainmask_img = nb.Nifti1Image(brainmask_dataobj, affine)
        nb.save(brainmask_img, self.brainmask_file)

        np.savetxt(self.gradients_file, gradients, fmt="%.6f")

        self.dwi = DWI(dataobj, affine, brainmask=brainmask_dataobj, gradients=gradients)
        self.dwi.to_filename(self.h5_file)

    def teardown(self):
        if os.path.exists(self.nii_file):
            os.remove(self.nii_file)
        if os.path.exists(self.brainmask_file):
            os.remove(self.brainmask_file)
        if os.path.exists(self.gradients_file):
            os.remove(self.gradients_file)
        if os.path.exists(self.h5_file):
            os.remove(self.h5_file)
        if os.path.exists(self.out_nii_file):
            os.remove(self.out_nii_file)
        if os.path.exists(self.out_gradients_file):
            os.remove(self.out_gradients_file)
        if os.path.exists(self.out_h5_file):
            os.remove(self.out_h5_file)
        if os.path.exists(self.benchmark_dir):
            os.rmdir(self.benchmark_dir)

    def mem_instantiation(self):
        dataobj, affine, brainmask_dataobj, gradients = self._generate_data()
        return DWI(dataobj, affine, brainmask=brainmask_dataobj, gradients=gradients)

    def time_instantiation(self):
        dataobj, affine, brainmask_dataobj, gradients = self._generate_data()
        _ = DWI(dataobj, affine, brainmask=brainmask_dataobj, gradients=gradients)

    # Skip for now. Possibly related to https://github.com/pympler/pympler/issues/151
    # However, using the version
    # https://github.com/mrJean1/pympler/blob/1f999dbf7b334d0e43ab9329d8cfb7043863fc96/pympler/asizeof.py
    # still results in the error
    @skip_benchmark_if(True)
    def mem_instantiation_from_filename(self):
        return DWI.from_filename(self.h5_file)

    def time_instantiation_from_filename(self):
        _ = DWI.from_filename(self.h5_file)

    def mem_instantiation_from_nii(self):
        return from_nii_dwi(
            self.nii_file,
            brainmask_file=self.brainmask_file,
            gradients_file=self.gradients_file,
        )

    def time_instantiation_from_nii(self):
        _ = from_nii_dwi(
            self.nii_file,
            brainmask_file=self.brainmask_file,
            gradients_file=self.gradients_file,
        )

    def peakmem_to_filename(self):
        self.dwi.to_filename(self.out_h5_file)

    def time_to_filename(self):
        self.dwi.to_filename(self.out_h5_file)

    def mem_to_nifti(self):
        return to_nifti_dwi(self.dwi, self.out_nii_file)

    def time_to_nifti(self):
        _ = to_nifti_dwi(self.dwi, self.out_nii_file)

    # Skip for now. Possibly related to https://github.com/pympler/pympler/issues/151
    # However, using the version
    # https://github.com/mrJean1/pympler/blob/1f999dbf7b334d0e43ab9329d8cfb7043863fc96/pympler/asizeof.py
    # still results in the error
    @skip_benchmark_if(True)
    def mem_getitem(self):
        return self.dwi[len(self.dwi) // 2]

    def time_getitem(self):
        _ = self.dwi[len(self.dwi) // 2]


class PETBenchmark:
    """
    Benchmarks for PET data class operations.
    """

    def __init__(self):
        self.pet: PET | Any = object()
        self.benchmark_dir = Path(__file__).parent / "benchmark_data_pet"
        self.nii_file = self.benchmark_dir / "pet.nii.gz"
        self.h5_file = self.benchmark_dir / "pet.h5"
        self.brainmask_file = self.benchmark_dir / "brainmask.nii.gz"
        self.temporal_file = self.benchmark_dir / "temporal.json"

        self.out_nii_file = self.benchmark_dir / "out_pet.nii.gz"
        self.out_h5_file = self.benchmark_dir / "out_pet.h5"
        self.out_temporal_file = self.benchmark_dir / "out_temporal.json"

    def _generate_data(self):
        rng = np.random.default_rng(42)

        os.makedirs(self.benchmark_dir, exist_ok=True)

        # Create a test NIfTI file for the PET object
        vol_size = (16, 16, 16)
        n_frames = 10
        pet_data = rng.random((*vol_size, n_frames)).astype("float32")
        affine = np.eye(4).astype("float32")

        # Create the brainmask
        brainmask_dataobj = rng.choice([True, False], size=vol_size).astype("uint8")

        # Create the temporal information
        frame_time = np.arange(n_frames, dtype=np.float32) + 1
        frame_time -= frame_time[0]
        frame_duration = np.diff(frame_time)
        if len(frame_duration) == (len(frame_time) - 1):
            frame_duration = np.append(frame_duration, frame_duration[-1])
        midframe = frame_time + frame_duration / 2
        total_duration = float(frame_time[-1] + frame_duration[-1])

        return pet_data, affine, brainmask_dataobj, frame_time, midframe, total_duration

    def setup(self):
        pet_data, affine, brainmask_dataobj, frame_time, midframe, total_duration = (
            self._generate_data()
        )

        os.makedirs(self.benchmark_dir, exist_ok=True)

        pet_img = nb.Nifti1Image(pet_data, affine)
        nb.save(pet_img, self.nii_file)

        brainmask_img = nb.Nifti1Image(brainmask_dataobj, affine)
        nb.save(brainmask_img, self.brainmask_file)

        with self.temporal_file.open("w", encoding="utf-8") as f:
            json.dump(
                {"FrameTimesStart": frame_time.tolist()},
                f,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )

        self.pet = PET(
            pet_data,
            affine,
            brainmask=brainmask_dataobj,
            midframe=midframe,
            total_duration=total_duration,
        )
        self.pet.to_filename(self.h5_file)

    def teardown(self):
        if os.path.exists(self.nii_file):
            os.remove(self.nii_file)
        if os.path.exists(self.brainmask_file):
            os.remove(self.brainmask_file)
        if os.path.exists(self.temporal_file):
            os.remove(self.temporal_file)
        if os.path.exists(self.h5_file):
            os.remove(self.h5_file)
        if os.path.exists(self.out_nii_file):
            os.remove(self.out_nii_file)
        if os.path.exists(self.out_temporal_file):
            os.remove(self.out_temporal_file)
        if os.path.exists(self.out_h5_file):
            os.remove(self.out_h5_file)
        if os.path.exists(self.benchmark_dir):
            os.rmdir(self.benchmark_dir)

    def mem_instantiation(self):
        dataobj, affine, brainmask_dataobj, frame_time, midframe, total_duration = (
            self._generate_data()
        )
        return PET(
            dataobj,
            affine,
            brainmask=brainmask_dataobj,
            midframe=midframe,
            total_duration=total_duration,
        )

    def time_instantiation(self):
        dataobj, affine, brainmask_dataobj, frame_time, midframe, total_duration = (
            self._generate_data()
        )
        _ = PET(
            dataobj,
            affine,
            brainmask=brainmask_dataobj,
            midframe=midframe,
            total_duration=total_duration,
        )

    # Skip for now. Possibly related to https://github.com/pympler/pympler/issues/151
    # However, using the version
    # https://github.com/mrJean1/pympler/blob/1f999dbf7b334d0e43ab9329d8cfb7043863fc96/pympler/asizeof.py
    # still results in the error
    @skip_benchmark_if(True)
    def mem_instantiation_from_filename(self):
        return PET.from_filename(self.h5_file)

    def time_instantiation_from_filename(self):
        _ = PET.from_filename(self.h5_file)

    def mem_instantiation_from_nii(self):
        return from_nii_pet(
            self.nii_file,
            self.temporal_file,
            brainmask_file=self.brainmask_file,
        )

    def time_instantiation_from_nii(self):
        _ = from_nii_pet(
            self.nii_file,
            self.temporal_file,
            brainmask_file=self.brainmask_file,
        )

    def peakmem_to_filename(self):
        self.pet.to_filename(self.out_h5_file)

    def time_to_filename(self):
        self.pet.to_filename(self.out_h5_file)

    def mem_to_nifti(self):
        return to_nifti_base(self.pet, self.out_nii_file)

    def time_to_nifti(self):
        _ = to_nifti_base(self.pet, self.out_nii_file)

    # Skip for now. Possibly related to https://github.com/pympler/pympler/issues/151
    # However, using the version
    # https://github.com/mrJean1/pympler/blob/1f999dbf7b334d0e43ab9329d8cfb7043863fc96/pympler/asizeof.py
    # still results in the error
    @skip_benchmark_if(True)
    def mem_getitem(self):
        return self.pet[len(self.pet) // 2]

    def time_getitem(self):
        _ = self.pet[len(self.pet) // 2]
