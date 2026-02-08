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

import hashlib
from os import cpu_count

import nibabel as nb
import nitransforms as nt
import numpy as np

from nifreeze.data.dmri import DWI
from nifreeze.estimator import Estimator
from nifreeze.model.base import TrivialModel
from nifreeze.registration.utils import displacements_within_mask

EXPECTED_DWI_MOTION_SHA256 = "3ba55cc3acfd584a7738f9701724e284f54bdf72261bf535b5dae062d7c0c30e"


def _sha256sum(path):
    hasher = hashlib.sha256()
    with path.open("rb") as fileobj:
        for chunk in iter(lambda: fileobj.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def test_proximity_estimator_trivial_model(datadir, tmp_path, p_error=20.0):
    """
    Check the proximity of transforms estimated by the estimator with a trivial B0 model.

    Parameters
    ----------
    datadir : pathlib.Path
        Path to the test data directory.
    tmp_path : pathlib.Path
        Path to a temporary directory for test outputs.
    p_error : float, optional
        Acceptable percentage error in the estimated transforms, by default 20.0.

    """

    dwi_motion_path = datadir / "dmri_data" / "motion_test_data" / "dwi_motion.h5"
    assert _sha256sum(dwi_motion_path) == EXPECTED_DWI_MOTION_SHA256, (
        "Unexpected checksum for dwi_motion.h5"
    )

    dwi_motion = DWI.from_filename(dwi_motion_path)

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
    has_mask = dwi_orig.brainmask is not None
    masknii = nb.Nifti1Image(
        dwi_orig.brainmask.astype(np.uint8)  # type: ignore
        if has_mask
        else np.ones(dwi_orig.dataobj.shape[:-1], dtype=np.uint8),
        dwi_orig.affine,
        None,
    )

    # Compute FD within brainmask
    max_error_mask = np.array(
        [
            displacements_within_mask(
                masknii,  # type: ignore
                nt.linear.Affine(est),
                nt.linear.Affine(truth),
            ).max()
            for est, truth in zip(dwi_motion.motion_affines, ground_truth_affines, strict=False)  # type: ignore
        ]
    )

    gt_inverse_errors = np.array(
        [
            displacements_within_mask(
                masknii,  # type: ignore
                ~nt.linear.Affine(truth),
                nt.linear.Affine(np.eye(4)),
            ).max()
            for truth in ground_truth_affines  # type: ignore
        ]
    )

    error_levels = gt_inverse_errors * p_error * 0.01
    masksize = (np.asanyarray(masknii.dataobj) > 0).astype(int).sum()
    assert np.all(max_error_mask < error_levels), (
        "Errors per volume [estimated(ground truth) mm]: "
        + ", ".join(
            f"{e:.2f}({g:.2f})" for e, g in zip(max_error_mask, gt_inverse_errors, strict=False)
        )
        + f" (N={masksize} voxels)."
    )


def test_stacked_estimators(datadir, tmp_path, monkeypatch):
    """Check that models can be stacked."""

    from nifreeze.utils import iterators

    # Wrap into dataset object
    dwi_motion = DWI.from_filename(datadir / "dmri_data" / "motion_test_data" / "dwi_motion.h5")
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
