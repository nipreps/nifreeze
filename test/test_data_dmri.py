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

from pathlib import Path

import nibabel as nb
import numpy as np
import pytest

from nifreeze.data import load
from nifreeze.data.dmri import DWI, find_shelling_scheme, from_nii, transform_fsl_bvec
from nifreeze.utils.ndimage import load_api


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


def test_main(datadir):
    input_file = datadir / "dwi.h5"

    assert isinstance(load(input_file), DWI)


@pytest.mark.parametrize("insert_b0", (False, True))
@pytest.mark.parametrize("rotate_bvecs", (False, True))
def test_load(datadir, tmp_path, insert_b0, rotate_bvecs):  # noqa: C901
    dwi_h5 = DWI.from_filename(datadir / "dwi.h5")
    dwi_nifti_path = tmp_path / "dwi.nii.gz"
    gradients_path = tmp_path / "dwi.tsv"

    dwi_h5.motion_affines = (
        np.array([nb.affines.from_matvec(np.eye(3), (10, -5, -20))] * len(dwi_h5))
        if rotate_bvecs
        else None
    )

    dwi_h5.to_nifti(dwi_nifti_path, insert_b0=insert_b0)

    nifti_data = load_api(dwi_nifti_path, nb.Nifti1Image).get_fdata()
    if insert_b0:
        nifti_data = nifti_data[..., 1:]

    with pytest.raises(RuntimeError):
        from_nii(dwi_nifti_path)

    # Try loading NIfTI + b-vecs/vals
    out_root = dwi_nifti_path.parent / dwi_nifti_path.name.replace(
        "".join(dwi_nifti_path.suffixes), ""
    )
    bvecs_path = out_root.with_suffix(".bvec")
    bvals_path = out_root.with_suffix(".bval")
    dwi_from_nifti1 = from_nii(
        dwi_nifti_path,
        bvec_file=bvecs_path,
        bval_file=bvals_path,
    )

    if not rotate_bvecs:  # If we set motion_affines, data WILL change
        nifti_data_diff = (~np.isclose(dwi_h5.dataobj, nifti_data, atol=1e-4)).sum()
        assert np.allclose(dwi_h5.dataobj, nifti_data, atol=1e-4), (
            f"``to_nifti()`` changed data contents ({nifti_data_diff} differences found)."
        )
        nifti_dwi_diff = (~np.isclose(dwi_h5.dataobj, dwi_from_nifti1.dataobj, atol=1e-4)).sum()
        assert np.allclose(dwi_h5.dataobj, dwi_from_nifti1.dataobj, atol=1e-4), (
            f"Data objects do not match ({nifti_dwi_diff} differences found)."
        )

    if insert_b0:
        assert np.allclose(dwi_h5.bzero, dwi_from_nifti1.bzero)

    assert np.allclose(dwi_h5.bvals, dwi_from_nifti1.bvals, atol=1e-3)

    bvec_diffs = np.where(
        (~np.isclose(dwi_h5.bvecs, dwi_from_nifti1.bvecs, atol=1e-3)).sum(0).astype(bool)
    )[0].tolist()

    assert not bvec_diffs, "\n".join(
        [f"{dwi_h5.bvecs[:, i]} vs {dwi_from_nifti1.bvecs[:, i]}" for i in bvec_diffs]
    )
    assert np.allclose(dwi_h5.gradients.T, dwi_from_nifti1.gradients.T, atol=1e-3)

    grad_table = dwi_h5.gradients
    if insert_b0:
        grad_table = np.hstack((np.zeros((4, 1)), dwi_h5.gradients))
    np.savetxt(str(gradients_path), grad_table)

    # Try loading NIfTI + gradients table
    dwi_from_nifti2 = from_nii(dwi_nifti_path, gradients_file=gradients_path)

    if not rotate_bvecs:  # If we set motion_affines, data WILL change
        assert np.allclose(dwi_h5.dataobj, dwi_from_nifti2.dataobj)
    if insert_b0:
        assert np.allclose(dwi_h5.bzero, dwi_from_nifti2.bzero)

    assert np.allclose(dwi_h5.gradients, dwi_from_nifti2.gradients)
    assert np.allclose(dwi_h5.bvals, dwi_from_nifti2.bvals, atol=1e-6)
    assert np.allclose(dwi_h5.bvecs, dwi_from_nifti2.bvecs, atol=1e-6)

    # Get the existing bzero data from the DWI instance, write it as a separate
    # file, and do the round-trip
    bzero = dwi_h5.bzero
    nii = nb.Nifti1Image(bzero, dwi_h5.affine, dwi_h5.datahdr)
    if dwi_h5.datahdr is None:
        nii.header.set_xyzt_units("mm")
    b0_file = Path(str(out_root) + "-b0").with_suffix(".nii.gz")
    nii.to_filename(b0_file)

    dwi_h5.to_nifti(dwi_nifti_path, insert_b0=insert_b0)

    dwi_from_nifti3 = from_nii(
        dwi_nifti_path,
        bvec_file=bvecs_path,
        bval_file=bvals_path,
        b0_file=b0_file,
    )

    if not rotate_bvecs:  # If we set motion_affines, data WILL change
        assert np.allclose(dwi_h5.dataobj, dwi_from_nifti3.dataobj)

    assert np.allclose(dwi_h5.bzero, dwi_from_nifti3.bzero)
    assert np.allclose(dwi_h5.gradients, dwi_from_nifti3.gradients, atol=1e-6)
    assert np.allclose(dwi_h5.bvals, dwi_from_nifti3.bvals, atol=1e-6)
    assert np.allclose(dwi_h5.bvecs, dwi_from_nifti3.bvecs, atol=1e-6)

    # Try loading NIfTI + gradients table
    dwi_from_nifti4 = from_nii(dwi_nifti_path, gradients_file=gradients_path, b0_file=b0_file)

    if not rotate_bvecs:  # If we set motion_affines, data WILL change
        assert np.allclose(dwi_h5.dataobj, dwi_from_nifti4.dataobj)

    assert np.allclose(dwi_h5.bzero, dwi_from_nifti4.bzero)
    assert np.allclose(dwi_h5.gradients, dwi_from_nifti4.gradients)
    assert np.allclose(dwi_h5.bvals, dwi_from_nifti4.bvals, atol=1e-6)
    assert np.allclose(dwi_h5.bvecs, dwi_from_nifti4.bvecs, atol=1e-6)


@pytest.mark.random_gtab_data(10, (1000,), 1)
@pytest.mark.random_dwi_data(50, (34, 36, 24), True)
def test_equality_operator(tmp_path, setup_random_dwi_data):
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
    ) = _serialize_dwi_data(
        dwi,
        brainmask,
        b0,
        gradients,
        tmp_path,
    )

    dwi_obj = from_nii(
        dwi_fname,
        gradients_file=gradients_fname,
        b0_file=b0_fname,
        brainmask_file=brainmask_fname,
        b0_thres=b0_thres,
    )
    hdf5_filename = tmp_path / "test_dwi.h5"
    dwi_obj.to_filename(hdf5_filename)

    round_trip_dwi_obj = DWI.from_filename(hdf5_filename)

    # Symmetric equality
    assert dwi_obj == round_trip_dwi_obj
    assert round_trip_dwi_obj == dwi_obj


@pytest.mark.random_dwi_data(50, (34, 36, 24), False)
def test_shells(setup_random_dwi_data):
    (
        dwi_dataobj,
        affine,
        brainmask_dataobj,
        b0_dataobj,
        gradients,
        _,
    ) = setup_random_dwi_data

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

    obtained_bval_est, obtained_indices = zip(*dwi_obj.get_shells(num_bins=num_bins), strict=True)

    obtained_dwi_data, obtained_motion_affines, obtained_gradients = zip(
        *[dwi_obj[indices] for indices in obtained_indices],
        strict=True,
    )

    assert len(obtained_bval_est) == num_bins
    assert list(obtained_bval_est) == expected_bval_est

    assert [i.tolist() for i in obtained_indices] == [i.tolist() for i in indices]
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


@pytest.mark.parametrize(
    "b_ijk",
    [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ],
)
@pytest.mark.parametrize("zooms", [(1.0, 1.0, 1.0), (2.0, 2.0, 2.0), (1.0, 2.0, 3.0)])
@pytest.mark.parametrize(
    "flips", [(1, 1, 1), (-1, 1, 1), (1, -1, 1), (1, 1, -1), (-1, -1, 1), (-1, -1, -1)]
)
@pytest.mark.parametrize("axis_order", [(0, 1, 2), (2, 0, 1), (1, 2, 0)])
@pytest.mark.parametrize("origin", [(0.0, 0.0, 0.0), (-10.0, -10.0, -10.0)])
@pytest.mark.parametrize(
    "angles", [(0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.0, 0.1, 0.0), (0.0, 0.0, 0.1)]
)
def test_transform_fsl_bvec(b_ijk, zooms, flips, axis_order, origin, angles):
    """Test the rotation of FSL b-vectors."""

    angles = dict(zip(("x", "y", "z"), angles, strict=False))
    affine = nb.affines.from_matvec(np.diag(zooms * np.array(flips)), origin)

    # Reorder first 3 columns of affine according to axis_order
    affine = affine[:, list(axis_order) + [3]]

    rotation_matrix = nb.eulerangles.euler2mat(**angles)

    # Ground truth
    rotated_b_ijk = rotation_matrix @ b_ijk

    # Rotation matrix in voxel space to scanner space
    rotation_ras = (
        affine @ nb.affines.from_matvec(rotation_matrix, (0, 0, 0)) @ np.linalg.inv(affine)
    )
    test_b_ijk = transform_fsl_bvec(b_ijk, rotation_ras, affine)

    assert np.allclose(test_b_ijk, rotated_b_ijk, atol=1e-6), (
        f"Expected {rotated_b_ijk}, got {test_b_ijk} for b_ijk={b_ijk}, "
        f"zooms={zooms}, origin={origin}, angles={angles}"
    )
