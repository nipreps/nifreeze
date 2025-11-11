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
"""Integration tests."""

from os import cpu_count

import nibabel as nb
import nitransforms as nt
import numpy as np

from nifreeze.data.dmri import DWI
from nifreeze.estimator import Estimator
from nifreeze.model.base import TrivialModel
from nifreeze.registration.utils import displacements_within_mask


def test_proximity_estimator_trivial_model(datadir, tmp_path):
    """Check the proximity of transforms estimated by the estimator with a trivial B0 model."""

    dwi_motion = DWI.from_filename(datadir / "dmri_data" / "motion_test_data" / "dwi_motion.h5")
    dwi_motion._filepath = tmp_path / "dwi_motion.h5"  # Prevent accidental overwriting

    ground_truth_affines = (
        np.copy(dwi_motion.motion_affines) if dwi_motion.motion_affines is not None else None
    )
    dwi_motion.motion_affines = None  # Erase ground truth for estimation

    model = TrivialModel(dwi_motion)
    estimator = Estimator(model)
    cpus = cpu_count()
    estimator.run(
        dwi_motion,
        seed=12345,
        num_threads=min(cpus if cpus is not None else 1, 8),
    )

    # # Uncomment to see the realigned dataset
    # nt.linear.LinearTransformsMapping(
    #     dwi_motion.motion_affines,
    #     reference=b0nii,
    # ).apply(moved_nii).to_filename(tmp_path / "realigned.nii.gz")
    dwi_orig = DWI.from_filename(datadir / "dwi.h5")
    masknii = (
        nb.Nifti1Image(dwi_orig.brainmask.astype(np.uint8), dwi_orig.affine, None)
        if dwi_orig.brainmask is not None
        else None
    )

    max_error = np.array(
        [
            displacements_within_mask(
                masknii,  # type: ignore
                nt.linear.Affine(est),
                nt.linear.Affine(truth),
            ).max()
            for est, truth in zip(dwi_motion.motion_affines, ground_truth_affines, strict=False)  # type: ignore
        ]
    )

    assert np.all(max_error < 0.25)


def test_stacked_estimators(datadir, tmp_path, monkeypatch):
    """Check that models can be stacked."""

    from nifreeze.utils import iterators

    # Wrap into dataset object
    dwi_motion = DWI.from_filename(datadir / "dmri_data" / "motion_test_data" / "dwi_motion.h5")
    dwi_motion._filepath = tmp_path / "dwi_motion.h5"  # Prevent accidental overwriting
    dwi_motion.motion_affines = None  # Erase ground truth for estimation

    def mock_iterator(*_, **kwargs):
        return []

    monkeypatch.setattr(iterators, "random_iterator", mock_iterator)  # Avoid iterator issues

    estimator1 = Estimator(
        TrivialModel(dwi_motion),
        ants_config="dwi-to-dwi_level0.json",
        clip=False,
    )
    estimator2 = Estimator(
        TrivialModel(dwi_motion),
        prev=estimator1,
        clip=False,
    )

    estimator2.run(dwi_motion)
