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
import pytest

from nifreeze.data import dmri
from nifreeze.estimator import Estimator
from nifreeze.model.base import TrivialModel
from nifreeze.registration.utils import displacements_within_mask

EXPECTED_DWI_MOTION_SHA256 = "3ba55cc3acfd584a7738f9701724e284f54bdf72261bf535b5dae062d7c0c30e"

DWI_DATA_CASES = (
    # Fully populated: mask + motion + gradients + b0
    (True, True, True, False, False, True, 50, 2000, False),
    # Fully populated: mask + motion + gradients + b0
    (True, True, True, False, False, True, 50, 2000, False),
    # No brainmask
    (False, True, True, False, False, True, 50, 2000, False),
    # No motion file
    (True, False, True, False, False, True, 50, 2000, False),
    # Gradients provided as bvec/bval pair
    (True, True, False, True, True, True, 50, 2000, False),
    # No explicit b0 image available (b0 detection must fall back to thresholding of bvals)
    (True, True, True, False, False, False, 50, 2000, False),
    # Ignore bzero frames entirely (simulate tests that want to drop b0 frames)
    (True, True, True, False, False, True, 50, 2000, True),
    # Low max_b to exercise trimming of high-b volumes
    (True, True, True, False, False, True, 50, 1000, False),
    (True, True, True, False, False, True, 50, 200, False),
    # No brainmask, gradients provided as bvec/bval pair, ignore bzero
    (False, False, False, True, True, False, 50, None, True),
)
"""Each tuple fields correspond to:
(brainmask_flag, motion_flag, gradients_flag, bvec_flag, bval_flag, b0_flag, b0_thres, max_b, ignore_bzero)

The cases aim to exercise combinations for DWI data integration tests:
- Presence/absence of brainmask
- Presence/absence of motion file
- Gradients provided either as a single gradient file or as a bvec/bval pair
- Presence/absence of explicit b0 image
- Different b0 thresholding / max b and ignore_bzero flags
"""


def _dwi_data_to_nifti(
    dwi_dataobj,
    affine,
    brainmask_dataobj,
    b0_dataobj,
):
    dwi = nb.Nifti1Image(dwi_dataobj, affine)
    brainmask = nb.Nifti1Image(brainmask_dataobj, affine)
    b0 = nb.Nifti1Image(b0_dataobj, affine)

    return dwi, brainmask, b0


def _serialize_dwi_data(
    dwi,
    brainmask,
    b0,
    gradients,
    _tmp_path,
):
    dwi_fname = _tmp_path / "dwi.nii.gz"
    brainmask_fname = _tmp_path / "brainmask.nii.gz"
    b0_fname = _tmp_path / "b0.nii.gz"
    gradients_fname = _tmp_path / "gradients.txt"
    bvec_fname = _tmp_path / "dwi.bvec"
    bval_fname = _tmp_path / "dwi.bval"

    nb.save(dwi, dwi_fname)
    nb.save(brainmask, brainmask_fname)
    nb.save(b0, b0_fname)
    np.savetxt(gradients_fname, gradients)
    np.savetxt(bvec_fname, gradients[:-1, :], fmt="%.3f")
    np.savetxt(bval_fname, gradients[-1, :], fmt="%d")

    return (
        dwi_fname,
        brainmask_fname,
        b0_fname,
        gradients_fname,
        bvec_fname,
        bval_fname,
    )


def _create_dmri_data_files(
    tmp_path,
    setup_random_dwi_data,
    brainmask_flag,
    motion_flag,
    gradients_flag,
    bvec_flag,
    bval_flag,
    b0_flag,
):
    """Create mask, motion, bvec/bval, gradients.json, and b0 volumes next to
    tmp_path based on the flags. Returns tuple of paths or None in the same
    order used by tests.
    """

    (
        dwi_dataobj,
        affine,
        brainmask_dataobj,
        b0_dataobj,
        gradients,
        b0_thres,
    ) = setup_random_dwi_data

    dwi, brainmask, b0 = _dwi_data_to_nifti(
        dwi_dataobj,
        affine,
        brainmask_dataobj.astype(np.uint8),
        b0_dataobj,
    )

    (
        dwi_fname,
        brainmask_fname,
        b0_fname,
        gradients_fname,
        bvec_fname,
        bval_fname,
    ) = _serialize_dwi_data(
        dwi,
        brainmask,
        b0,
        gradients,
        tmp_path,
    )

    return (
        dwi_fname,
        brainmask_fname if brainmask_flag else None,
        None if motion_flag else None,
        gradients_fname if gradients_flag else None,
        bvec_fname if bvec_flag else None,
        bval_fname if bval_flag else None,
        b0_fname if b0_flag else None,
    )


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

    dwi_motion = dmri.DWI.from_filename(dwi_motion_path)
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
    dwi_orig = dmri.DWI.from_filename(datadir / "dwi.h5")
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


@pytest.mark.random_gtab_data(6, (1000,), 1)
@pytest.mark.random_dwi_data(50, (4, 4, 4), True)
@pytest.mark.parametrize(
    (
        "brainmask_flag",
        "motion_flag",
        "gradients_flag",
        "bvec_flag",
        "bval_flag",
        "b0_flag",
        "b0_thres",
        "max_b",
        "ignore_bzero",
    ),
    DWI_DATA_CASES,
)
def test_dmri_data_average_dwi_model_estimation(
    tmp_path,
    monkeypatch,
    setup_random_dwi_data,
    brainmask_flag,
    motion_flag,
    gradients_flag,
    bvec_flag,
    bval_flag,
    b0_flag,
    b0_thres,
    max_b,
    ignore_bzero,
):
    # Map boolean flags to real test file paths (or None)
    dwi_data_file, brainmask_file, motion_file, gradients_file, bvec_file, bval_file, b0_file = (
        _create_dmri_data_files(
            tmp_path,
            setup_random_dwi_data,
            brainmask_flag,
            motion_flag,
            gradients_flag,
            bvec_flag,
            bval_flag,
            b0_flag,
        )
    )

    # Build kwargs depending on what is present
    data_kwargs = {
        k: v
        for k, v in (
            ("brainmask_file", brainmask_file),
            ("motion_file", motion_file),
            ("gradients_file", gradients_file),
            ("bvec_file", bvec_file),
            ("bval_file", bval_file),
            ("b0_file", b0_file),
        )
        if v is not None
    }

    model_kwargs = {"b0_thres": b0_thres, "max_b": max_b, "ignore_bzero": ignore_bzero}

    # ToDo
    # Monkey patch estimator run to avoid long registration running processes or else make sure registration runs and provides something
    # But these are registration tests and issue 223 was raised due to deep calls that need to be tested

    # Create the dataset using the repository test data
    dwi_data = dmri.from_nii(dwi_data_file, **data_kwargs)
    estimator = Estimator("avgdwi", model_kwargs=model_kwargs)
    estimator.run(
        dwi_data,
        omp_nthreads=1,
        n_jobs=1,
        seed=42,
    )

    # ToDo
    # Check whether the written affines have the right dimensionality


@pytest.mark.random_gtab_data(6, (1000,), 1)
@pytest.mark.random_dwi_data(50, (4, 4, 4), True)
@pytest.mark.parametrize(
    (
        "brainmask_flag",
        "motion_flag",
        "gradients_flag",
        "bvec_flag",
        "bval_flag",
        "b0_flag",
        "b0_thres",
        "max_b",
        "ignore_bzero",
    ),
    DWI_DATA_CASES,
)
def test_dmri_data_dti_model_estimation(
    tmp_path,
    monkeypatch,
    setup_random_dwi_data,
    brainmask_flag,
    motion_flag,
    gradients_flag,
    bvec_flag,
    bval_flag,
    b0_flag,
    b0_thres,
    max_b,
    ignore_bzero,
):
    # Map boolean flags to real test file paths (or None)
    dwi_data_file, brainmask_file, motion_file, gradients_file, bvec_file, bval_file, b0_file = (
        _create_dmri_data_files(
            tmp_path,
            setup_random_dwi_data,
            brainmask_flag,
            motion_flag,
            gradients_flag,
            bvec_flag,
            bval_flag,
            b0_flag,
        )
    )

    # Build kwargs depending on what is present
    data_kwargs = {
        k: v
        for k, v in (
            ("brainmask_file", brainmask_file),
            ("motion_file", motion_file),
            ("gradients_file", gradients_file),
            ("bvec_file", bvec_file),
            ("bval_file", bval_file),
            ("b0_file", b0_file),
        )
        if v is not None
    }

    model_kwargs = {"b0_thres": b0_thres, "max_b": max_b, "ignore_bzero": ignore_bzero}

    # ToDo
    # Monkey patch estimator run to avoid long registration running processes or else make sure registration runs and provides something
    # But these are registration tests and issue 223 was raised due to deep calls that need to be tested

    # Create the dataset using the repository test data
    dwi_data = dmri.from_nii(dwi_data_file, **data_kwargs)
    estimator = Estimator("dti", model_kwargs=model_kwargs)
    estimator.run(
        dwi_data,
        omp_nthreads=1,
        n_jobs=1,
        seed=42,
    )

    # ToDo
    # Check whether the written affines have the right dimensionality


@pytest.mark.random_gtab_data(10, (1000, 2000), 2)
@pytest.mark.random_dwi_data(50, (4, 4, 4), True)
@pytest.mark.parametrize(
    (
        "brainmask_flag",
        "motion_flag",
        "gradients_flag",
        "bvec_flag",
        "bval_flag",
        "b0_flag",
        "b0_thres",
        "max_b",
        "ignore_bzero",
    ),
    DWI_DATA_CASES,
)
def test_dmri_data_dki_model_estimation(
    tmp_path,
    monkeypatch,
    setup_random_dwi_data,
    brainmask_flag,
    motion_flag,
    gradients_flag,
    bvec_flag,
    bval_flag,
    b0_flag,
    b0_thres,
    max_b,
    ignore_bzero,
):
    # Skip if max_b is constraining the number of nonzero b-values to less than
    # 2 since the DKI model requires multi-shell data (i.e. at least 2 nonzero
    # b-values).
    from dipy.core.gradients import (
        check_multi_b,
        gradient_table_from_bvals_bvecs,
        unique_bvals_magnitude,
    )

    gradients = setup_random_dwi_data[4]
    bvals = setup_random_dwi_data[4][-1, :]
    # ToDo
    # max_b is not used in the code
    # shellmask = (bvals <= max_b)
    shellmask = np.ones_like(bvals, dtype=bool)
    gtab = gradient_table_from_bvals_bvecs(
        gradients[:, shellmask][-1, :], gradients[:, shellmask][:-1, :].T, b0_threshold=b0_thres
    )
    uniqueb = unique_bvals_magnitude(bvals)
    enough_b = check_multi_b(gtab, 3, non_zero=False)
    if not enough_b:
        pytest.skip(f"DKI requires multi-shell data: found {uniqueb} unique b-values.")

    # Map boolean flags to real test file paths (or None)
    dwi_data_file, brainmask_file, motion_file, gradients_file, bvec_file, bval_file, b0_file = (
        _create_dmri_data_files(
            tmp_path,
            setup_random_dwi_data,
            brainmask_flag,
            motion_flag,
            gradients_flag,
            bvec_flag,
            bval_flag,
            b0_flag,
        )
    )

    # Build kwargs depending on what is present
    data_kwargs = {
        k: v
        for k, v in (
            ("brainmask_file", brainmask_file),
            ("motion_file", motion_file),
            ("gradients_file", gradients_file),
            ("bvec_file", bvec_file),
            ("bval_file", bval_file),
            ("b0_file", b0_file),
        )
        if v is not None
    }

    model_kwargs = {"b0_thres": b0_thres, "max_b": max_b, "ignore_bzero": ignore_bzero}

    # ToDo
    # Monkey patch estimator run to avoid long registration running processes or else make sure registration runs and provides something
    # But these are registration tests and issue 223 was raised due to deep calls that need to be tested

    # Create the dataset using the repository test data
    dwi_data = dmri.from_nii(dwi_data_file, **data_kwargs)
    estimator = Estimator("dki", model_kwargs=model_kwargs)
    estimator.run(
        dwi_data,
        omp_nthreads=1,
        n_jobs=1,
        seed=42,
    )

    # ToDo
    # Check whether the written affines have the right dimensionality


def test_stacked_estimators(datadir, tmp_path, monkeypatch):
    """Check that models can be stacked."""

    from nifreeze.utils import iterators

    # Wrap into dataset object
    dwi_motion = dmri.DWI.from_filename(datadir / "dmri_data" / "motion_test_data" / "dwi_motion.h5")
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
