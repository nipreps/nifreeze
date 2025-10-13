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

import nitransforms as nt

from nifreeze.estimator import Estimator
from nifreeze.model.base import TrivialModel
from nifreeze.registration.utils import displacements_within_mask


def test_proximity_estimator_trivial_model(motion_data, tmp_path):
    """Check the proximity of transforms estimated by the estimator with a trivial B0 model."""

    b0nii = motion_data["b0nii"]
    moved_nii = motion_data["moved_nii"]
    xfms = motion_data["xfms"]
    dwi_motion = motion_data["moved_nifreeze"]

    model = TrivialModel(dwi_motion)
    estimator = Estimator(model)
    cpus = cpu_count()
    estimator.run(
        dwi_motion,
        seed=12345,
        num_threads=min(cpus if cpus is not None else 1, 8),
    )

    # Uncomment to see the realigned dataset
    xfm = nt.linear.LinearTransformsMapping(
        dwi_motion.motion_affines,
        reference=b0nii,
    )
    nt.resampling.apply(xfm, moved_nii).to_filename(tmp_path / "realigned.nii.gz")

    # For each moved b0 volume
    for i, est in enumerate(dwi_motion.motion_affines):
        assert (
            displacements_within_mask(
                motion_data["masknii"],
                nt.linear.Affine(est),
                xfms[i],
            ).max()
            < 0.25
        )


def test_stacked_estimators(motion_data):
    """Check that models can be stacked."""

    # Wrap into dataset object
    dmri_dataset = motion_data["moved_nifreeze"]

    estimator1 = Estimator(
        TrivialModel(dmri_dataset),
        ants_config="dwi-to-dwi_level0.json",
        clip=False,
    )
    estimator2 = Estimator(
        TrivialModel(dmri_dataset),
        prev=estimator1,
        clip=False,
    )

    estimator2.run(dmri_dataset)
