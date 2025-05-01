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
"""Unit tests exercising the dMRI data structure."""

import nibabel as nb
import numpy as np
import pytest
from dipy.io.gradients import read_bvals_bvecs

from nifreeze.data.dmri import DWI, find_shelling_scheme, load


def _create_random_gtab_dataobj(request, n_gradients=10, b0s=1):
    rng = request.node.rng

    bvals = np.hstack([b0s * [0], n_gradients * [1000]])
    bvecs = np.hstack([np.zeros((3, b0s)), rng.random((3, n_gradients))])

    return bvals, bvecs


def _create_dwi_random_dataobj(request, bvals, bvecs):
    rng = request.node.rng

    n_gradients = np.count_nonzero(bvals)
    b0s = len(bvals) - n_gradients
    volumes = n_gradients + b0s
    b0_thres = 50

    vol_size = (34, 36, 24)

    dwi_dataobj = rng.random((*vol_size, volumes), dtype="float32")
    affine = np.eye(4, dtype="float32")
    brainmask_dataobj = rng.random(vol_size, dtype="float32")
    b0_dataobj = rng.random(vol_size, dtype="float32")
    gradients = np.vstack([bvecs, bvals[np.newaxis, :]], dtype="float32")

    return (
        dwi_dataobj,
        affine,
        brainmask_dataobj,
        b0_dataobj,
        gradients,
        b0_thres,
    )


def _create_dwi_random_data(
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

    nb.save(dwi, dwi_fname)
    nb.save(brainmask, brainmask_fname)
    nb.save(b0, b0_fname)
    np.savetxt(gradients_fname, gradients.T)

    return (
        dwi_fname,
        brainmask_fname,
        b0_fname,
        gradients_fname,
    )


def test_load(datadir, tmp_path):
    """Check that the registration parameters for b=0
    gives a good estimate of known affine"""

    dwi_h5 = load(datadir / "dwi.h5")
    dwi_nifti_path = tmp_path / "dwi.nii.gz"
    gradients_path = tmp_path / "dwi.tsv"
    bvecs_path = tmp_path / "dwi.bvecs"
    bvals_path = tmp_path / "dwi.bvals"

    grad_table = np.hstack((np.zeros((4, 1)), dwi_h5.gradients))

    dwi_h5.to_nifti(dwi_nifti_path, insert_b0=True)
    np.savetxt(str(gradients_path), grad_table.T)
    np.savetxt(str(bvecs_path), grad_table[:3])
    np.savetxt(str(bvals_path), grad_table[-1])

    with pytest.raises(RuntimeError):
        load(dwi_nifti_path)

    # Try loading NIfTI + gradients table
    dwi_from_nifti1 = load(dwi_nifti_path, gradients_file=gradients_path)

    assert np.allclose(dwi_h5.dataobj, dwi_from_nifti1.dataobj)
    assert np.allclose(dwi_h5.bzero, dwi_from_nifti1.bzero)
    assert np.allclose(dwi_h5.gradients, dwi_from_nifti1.gradients)

    # Try loading NIfTI + b-vecs/vals
    dwi_from_nifti2 = load(
        dwi_nifti_path,
        bvec_file=bvecs_path,
        bval_file=bvals_path,
    )

    assert np.allclose(dwi_h5.dataobj, dwi_from_nifti2.dataobj)
    assert np.allclose(dwi_h5.bzero, dwi_from_nifti2.bzero)
    assert np.allclose(dwi_h5.gradients, dwi_from_nifti2.gradients)


def test_equality_operator(tmp_path, request):
    # Create some random data
    bvals, bvecs = _create_random_gtab_dataobj(request)

    (
        dwi_dataobj,
        affine,
        brainmask_dataobj,
        b0_dataobj,
        gradients,
        b0_thres,
    ) = _create_dwi_random_dataobj(request, bvals, bvecs)

    dwi, brainmask, b0 = _create_dwi_random_data(
        dwi_dataobj,
        affine,
        brainmask_dataobj,
        b0_dataobj,
    )

    (
        dwi_fname,
        brainmask_fname,
        b0_fname,
        gradients_fname,
    ) = _serialize_dwi_data(
        dwi,
        brainmask,
        b0,
        gradients,
        tmp_path,
    )

    dwi_obj = load(
        dwi_fname,
        gradients_file=gradients_fname,
        b0_file=b0_fname,
        brainmask_file=brainmask_fname,
        b0_thres=b0_thres,
    )
    hdf5_filename = tmp_path / "test_dwi.h5"
    dwi_obj.to_filename(hdf5_filename)

    round_trip_dwi_obj = load(hdf5_filename)

    # Symmetric equality
    assert dwi_obj == round_trip_dwi_obj
    assert round_trip_dwi_obj == dwi_obj


def test_shells(request, repodata):
    bvals, bvecs = read_bvals_bvecs(
        str(repodata / "hcph_multishell.bval"),
        str(repodata / "hcph_multishell.bvec"),
    )

    (
        dwi_dataobj,
        affine,
        brainmask_dataobj,
        b0_dataobj,
        gradients,
        _,
    ) = _create_dwi_random_dataobj(request, bvals, bvecs.T)

    dwi_obj = DWI(
        dataobj=dwi_dataobj,
        affine=affine,
        brainmask=brainmask_dataobj,
        bzero=b0_dataobj,
        gradients=gradients,
    )

    num_bins = 3
    _, expected_bval_groups, expected_bval_est = find_shelling_scheme(
        dwi_obj.gradients[-1, ...], num_bins=num_bins
    )

    indices = [
        np.hstack(np.where(np.isin(dwi_obj.gradients[-1, ...], bvals)))
        for bvals in expected_bval_groups
    ]
    expected_dwi_data = [dwi_obj.dataobj[..., idx] for idx in indices]
    expected_motion_affines = [
        dwi_obj.motion_affines[idx] if dwi_obj.motion_affines else None for idx in indices
    ]
    expected_gradients = [dwi_obj.gradients[..., idx] for idx in indices]

    shell_data = dwi_obj.shells(num_bins=num_bins)
    obtained_bval_est, obtained_dwi_data, obtained_motion_affines, obtained_gradients = zip(
        *shell_data, strict=True
    )

    assert len(shell_data) == num_bins
    assert list(obtained_bval_est) == expected_bval_est
    assert all(
        np.allclose(arr1, arr2)
        for arr1, arr2 in zip(list(obtained_dwi_data), expected_dwi_data, strict=True)
    )
    assert all(
        (arr1 is None and arr2 is None)
        or (arr1 is not None and arr2 is not None and np.allclose(arr1, arr2))
        for arr1, arr2 in zip(list(obtained_motion_affines), expected_motion_affines, strict=True)
    )
    assert all(
        np.allclose(arr1, arr2)
        for arr1, arr2 in zip(list(obtained_gradients), expected_gradients, strict=True)
    )


@pytest.mark.parametrize(
    ("bvals", "exp_scheme", "exp_bval_groups", "exp_bval_estimated"),
    [
        (
            np.asarray(
                [
                    5,
                    300,
                    300,
                    300,
                    300,
                    300,
                    305,
                    1005,
                    995,
                    1000,
                    1000,
                    1005,
                    1000,
                    1000,
                    1005,
                    995,
                    1000,
                    1005,
                    5,
                    995,
                    1000,
                    1000,
                    995,
                    1005,
                    995,
                    1000,
                    995,
                    995,
                    2005,
                    2000,
                    2005,
                    2005,
                    1995,
                    2000,
                    2005,
                    2000,
                    1995,
                    2005,
                    5,
                    1995,
                    2005,
                    1995,
                    1995,
                    2005,
                    2005,
                    1995,
                    2000,
                    2000,
                    2000,
                    1995,
                    2000,
                    2000,
                    2005,
                    2005,
                    1995,
                    2005,
                    2005,
                    1990,
                    1995,
                    1995,
                    1995,
                    2005,
                    2000,
                    1990,
                    2010,
                    5,
                ]
            ),
            "multi-shell",
            [
                np.asarray([5, 5, 5, 5]),
                np.asarray([300, 300, 300, 300, 300, 305]),
                np.asarray(
                    [
                        1005,
                        995,
                        1000,
                        1000,
                        1005,
                        1000,
                        1000,
                        1005,
                        995,
                        1000,
                        1005,
                        995,
                        1000,
                        1000,
                        995,
                        1005,
                        995,
                        1000,
                        995,
                        995,
                    ]
                ),
                np.asarray(
                    [
                        2005,
                        2000,
                        2005,
                        2005,
                        1995,
                        2000,
                        2005,
                        2000,
                        1995,
                        2005,
                        1995,
                        2005,
                        1995,
                        1995,
                        2005,
                        2005,
                        1995,
                        2000,
                        2000,
                        2000,
                        1995,
                        2000,
                        2000,
                        2005,
                        2005,
                        1995,
                        2005,
                        2005,
                        1990,
                        1995,
                        1995,
                        1995,
                        2005,
                        2000,
                        1990,
                        2010,
                    ]
                ),
            ],
            [5, 300, 1000, 2000],
        ),
    ],
)
def test_find_shelling_scheme_array(bvals, exp_scheme, exp_bval_groups, exp_bval_estimated):
    obt_scheme, obt_bval_groups, obt_bval_estimated = find_shelling_scheme(bvals)
    assert obt_scheme == exp_scheme
    assert all(
        np.allclose(obt_arr, exp_arr)
        for obt_arr, exp_arr in zip(obt_bval_groups, exp_bval_groups, strict=True)
    )
    assert np.allclose(obt_bval_estimated, exp_bval_estimated)


@pytest.mark.parametrize(
    ("dwi_btable", "exp_scheme", "exp_bval_groups", "exp_bval_estimated"),
    [
        (
            "ds000114_singleshell",
            "single-shell",
            [
                np.asarray([0, 0, 0, 0, 0, 0, 0]),
                np.asarray(
                    [
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                    ]
                ),
            ],
            [0.0, 1000.0],
        ),
        (
            "hcph_multishell",
            "multi-shell",
            [
                np.asarray([0, 0, 0, 0, 0, 0]),
                np.asarray([700, 700, 700, 700, 700, 700, 700, 700, 700, 700, 700, 700]),
                np.asarray(
                    [
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                        1000,
                    ]
                ),
                np.asarray(
                    [
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                        2000,
                    ]
                ),
                np.asarray(
                    [
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                        3000,
                    ]
                ),
            ],
            [0.0, 700.0, 1000.0, 2000.0, 3000.0],
        ),
        (
            "ds004737_dsi",
            "DSI",
            [
                np.asarray([5, 5, 5, 5, 5, 5, 5, 5, 5]),
                np.asarray([995, 995, 800, 800, 995, 995, 795, 995]),
                np.asarray([1195, 1195, 1195, 1195, 1000, 1195, 1195, 1000]),
                np.asarray([1595, 1595, 1595, 1600.0]),
                np.asarray(
                    [
                        1800,
                        1795,
                        1795,
                        1790,
                        1995,
                        1800,
                        1795,
                        1990,
                        1990,
                        1795,
                        1990,
                        1795,
                        1795,
                        1995,
                    ]
                ),
                np.asarray([2190, 2195, 2190, 2195, 2000, 2000, 2000, 2195, 2195, 2190]),
                np.asarray([2590, 2595, 2600, 2395, 2595, 2600, 2395]),
                np.array([2795, 2790, 2795, 2795, 2790, 2795, 2795, 2790, 2795]),
                np.array([3590, 3395, 3595, 3595, 3395, 3395, 3400]),
                np.array([3790, 3790]),
                np.array([4195, 4195]),
                np.array([4390, 4395, 4390]),
                np.array(
                    [
                        4790,
                        4990,
                        4990,
                        5000,
                        5000,
                        4990,
                        4795,
                        4985,
                        5000,
                        4795,
                        5000,
                        4990,
                        4990,
                        4790,
                        5000,
                        4990,
                        4795,
                        4795,
                        4990,
                        5000,
                        4990,
                    ]
                ),
            ],
            [
                5.0,
                995.0,
                1195.0,
                1595.0,
                1797.5,
                2190.0,
                2595.0,
                2795.0,
                3400.0,
                3790.0,
                4195.0,
                4390.0,
                4990.0,
            ],
        ),
    ],
)
def test_find_shelling_scheme_files(
    dwi_btable, exp_scheme, exp_bval_groups, exp_bval_estimated, repodata
):
    bvals = np.loadtxt(repodata / f"{dwi_btable}.bval")

    obt_scheme, obt_bval_groups, obt_bval_estimated = find_shelling_scheme(bvals)
    assert obt_scheme == exp_scheme
    assert all(
        np.allclose(obt_arr, exp_arr)
        for obt_arr, exp_arr in zip(obt_bval_groups, exp_bval_groups, strict=True)
    )
    assert np.allclose(obt_bval_estimated, exp_bval_estimated)
