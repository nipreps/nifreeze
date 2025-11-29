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

import re
from pathlib import Path

import attrs
import nibabel as nb
import numpy as np
import pytest

from nifreeze.data import load
from nifreeze.data.dmri.base import (
    DWI,
    validate_gradients,
)
from nifreeze.data.dmri.io import (
    GRADIENT_BVAL_BVEC_PRIORITY_WARN_MSG,
    GRADIENT_DATA_MISSING_ERROR,
    from_nii,
)
from nifreeze.data.dmri.utils import (
    DTI_MIN_ORIENTATIONS,
    GRADIENT_ABSENCE_ERROR_MSG,
    GRADIENT_EXPECTED_COLUMNS_ERROR_MSG,
    GRADIENT_NDIM_ERROR_MSG,
    GRADIENT_OBJECT_ERROR_MSG,
    GRADIENT_VOLUME_DIMENSIONALITY_MISMATCH_ERROR,
    find_shelling_scheme,
    format_gradients,
    transform_fsl_bvec,
)
from nifreeze.utils.ndimage import load_api

B_MATRIX = np.array(
    [
        [0.0, 0.0, 0.0, 0],
        [1.0, 0.0, 0.0, 1000],
        [0.0, 1.0, 0.0, 1000],
        [0.0, 0.0, 1.0, 1000],
        [1 / np.sqrt(2), 1 / np.sqrt(2), 0.0, 1000],
        [1 / np.sqrt(2), 0.0, 1 / np.sqrt(2), 1000],
        [0.0, 1 / np.sqrt(2), 1 / np.sqrt(2), 1000],
        [1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3), 2000],
        [-1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3), 2000],
        [1 / np.sqrt(3), -1 / np.sqrt(3), 1 / np.sqrt(3), 2000],
        [1 / np.sqrt(3), 1 / np.sqrt(3), -1 / np.sqrt(3), 2000],
    ],
    dtype=np.float32,
)


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
    np.savetxt(gradients_fname, gradients)

    return (
        dwi_fname,
        brainmask_fname,
        b0_fname,
        gradients_fname,
    )


def test_main(datadir):
    input_file = datadir / "dwi.h5"

    assert isinstance(load(input_file), DWI)


@pytest.mark.parametrize(
    "value, expected_exc, expected_msg",
    [
        (np.array([[1], [2]], dtype=object), ValueError, GRADIENT_EXPECTED_COLUMNS_ERROR_MSG),
        (np.zeros((2, 3)), ValueError, GRADIENT_EXPECTED_COLUMNS_ERROR_MSG),
    ],
)
def test_validate_gradients(monkeypatch, value, expected_exc, expected_msg):
    monkeypatch.setattr(DWI, "__init__", lambda self, *a, **k: None)
    inst = DWI()
    dummy_attr = attrs.fields(DWI).gradients
    with pytest.raises(expected_exc, match=re.escape(str(expected_msg))):
        validate_gradients(inst, dummy_attr, value)


@pytest.mark.parametrize(
    "value, expected_exc, expected_msg",
    [
        (None, ValueError, GRADIENT_ABSENCE_ERROR_MSG),
        (3.14, ValueError, GRADIENT_NDIM_ERROR_MSG),
        ([1, 2, 3, 4], ValueError, GRADIENT_NDIM_ERROR_MSG),
        (np.arange(24).reshape(4, 3, 2), ValueError, GRADIENT_NDIM_ERROR_MSG),
        ([[1, 2], [3, 4, 5]], (TypeError, ValueError), GRADIENT_OBJECT_ERROR_MSG),  # Ragged
    ],
)
def test_format_gradients_errors(value, expected_exc, expected_msg):
    with pytest.raises(expected_exc, match=str(expected_msg)):
        format_gradients(value)


@pytest.mark.parametrize(
    "value, expect_transpose",
    [
        # 2D arrays where first dim == 4 and second dim == 4 -> NO transpose
        (B_MATRIX[:4, :], False),
        # 2D arrays where first dim == 4 and second dim != 4 -> transpose
        (B_MATRIX[:3, :].T, True),
        (B_MATRIX[:5, :].T, True),
        (B_MATRIX.T, True),
        # 2D arrays where first dim != 4 -> NO transpose
        (B_MATRIX[:3, :], False),
        (B_MATRIX[:5, :], False),
        (B_MATRIX, False),
        (np.empty((4, 0)), True),  # zero columns -> still triggers transpose
        # List of lists
        ([[1, 0, 0, 100], [0, 1, 0, 100]], False),
        ([[1, 0, 0], [0, 1, 0], [0, 0, 1], [100, 100, 100]], True),
    ],
)
def test_format_gradients_basic(value, expect_transpose):
    obtained = format_gradients(value)

    assert isinstance(obtained, np.ndarray)
    if expect_transpose:
        assert obtained.shape == np.asarray(value).T.shape
        assert np.allclose(obtained, np.asarray(value).T)
    else:
        assert obtained.shape == np.asarray(value).shape
        assert np.allclose(obtained, np.asarray(value))


@pytest.mark.random_uniform_spatial_data((2, 2, 2, 4), 0.0, 1.0)
def test_dwi_post_init_errors(setup_random_uniform_spatial_data):
    data, affine = setup_random_uniform_spatial_data
    with pytest.raises(ValueError, match=GRADIENT_ABSENCE_ERROR_MSG):
        DWI(dataobj=data, affine=affine)

    with pytest.raises(
        ValueError,
        match=f"DWI datasets must have at least {DTI_MIN_ORIENTATIONS} diffusion-weighted",
    ):
        DWI(dataobj=data, affine=affine, gradients=B_MATRIX[: data.shape[-1], :])


@pytest.mark.random_gtab_data(10, (1000, 2000), 2)
@pytest.mark.random_dwi_data(50, (34, 36, 24), True)
@pytest.mark.parametrize("row_major_gradients", (False, True))
@pytest.mark.parametrize("additional_grad_columns", (-1, -2, 1))
def test_dwi_instantiation_gradients_unexpected_columns_error(
    request, setup_random_dwi_data, row_major_gradients, additional_grad_columns
):
    dwi_dataobj, affine, _, b0_dataobj, gradients, _ = setup_random_dwi_data

    # Remove/prepend columns. At this point, it is irrelevant whether the
    # potential N-dimensional vector is normalized or not
    if additional_grad_columns < 1:
        gradients = gradients[:, : gradients.shape[1] + additional_grad_columns]
    else:
        rng = request.node.rng
        add_gradients = rng.random(size=(gradients.shape[0], additional_grad_columns))
        gradients = np.insert(gradients, 0, add_gradients, axis=1)

    if not row_major_gradients:
        gradients = gradients.T

    with pytest.raises(ValueError, match=re.escape(GRADIENT_EXPECTED_COLUMNS_ERROR_MSG)):
        DWI(
            dataobj=dwi_dataobj,
            affine=affine,
            bzero=b0_dataobj,
            gradients=gradients,
        )


@pytest.mark.random_gtab_data(10, (1000, 2000), 2)
@pytest.mark.random_dwi_data(50, (34, 36, 24), True)
@pytest.mark.parametrize("row_major_gradients", (False, True))
def test_dwi_instantiation_gradients_ndim_error(
    tmp_path, setup_random_dwi_data, row_major_gradients
):
    dwi_dataobj, affine, _, b0_dataobj, gradients, _ = setup_random_dwi_data

    # Store a single column from gradients to try loading a 1D-array. Transpose
    # depending on whether to follow the row-major convention or not
    gradients = gradients[:, 0]
    if not row_major_gradients:
        gradients = gradients.T

    with pytest.raises(ValueError, match=GRADIENT_NDIM_ERROR_MSG):
        DWI(
            dataobj=dwi_dataobj,
            affine=affine,
            bzero=b0_dataobj,
            gradients=gradients,
        )


@pytest.mark.random_gtab_data(10, (1000, 2000), 2)
@pytest.mark.random_dwi_data(50, (34, 36, 24), True)
@pytest.mark.parametrize(
    ("additional_volume_count", "additional_gradient_count"),
    [(1, 0), (2, 0), (2, 1), (0, 1), (0, 2), (1, 2)],
)
def test_gradient_instantiation_dwi_vol_mismatch_error(
    setup_random_dwi_data, additional_volume_count, additional_gradient_count
):
    dwi_dataobj, affine, _, b0_dataobj, gradients, _ = setup_random_dwi_data

    # Add additional volumes: simply concatenate the last volume
    if additional_volume_count:
        additional_dwi_dataobj = np.tile(dwi_dataobj[..., -1:], (1, additional_volume_count))
        dwi_dataobj = np.concatenate((dwi_dataobj, additional_dwi_dataobj), axis=-1)
    # Add additional gradients: simply concatenate the gradient
    if additional_gradient_count:
        additional_gradients = np.tile(gradients[-1:, :], (additional_gradient_count, 1))
        gradients = np.concatenate((gradients, additional_gradients), axis=0)

    # Test with b0s present
    n_volumes = dwi_dataobj.shape[-1]
    with pytest.raises(
        ValueError,
        match=GRADIENT_VOLUME_DIMENSIONALITY_MISMATCH_ERROR.format(
            n_volumes=n_volumes, n_gradients=gradients.shape[0]
        ),
    ):
        DWI(
            dataobj=dwi_dataobj,
            affine=affine,
            bzero=b0_dataobj,
            gradients=gradients,
        )

    # Test without b0s present
    dwi_dataobj = dwi_dataobj[..., 2:]
    gradients = gradients[2:, :]
    n_volumes = dwi_dataobj.shape[-1]
    with pytest.raises(
        ValueError,
        match=GRADIENT_VOLUME_DIMENSIONALITY_MISMATCH_ERROR.format(
            n_volumes=n_volumes, n_gradients=gradients.shape[0]
        ),
    ):
        DWI(
            dataobj=dwi_dataobj,
            affine=affine,
            bzero=b0_dataobj,
            gradients=gradients,
        )


@pytest.mark.random_gtab_data(10, (1000, 2000), 2)
@pytest.mark.random_dwi_data(50, (34, 36, 24), True)
@pytest.mark.parametrize("row_major_gradients", (False, True))
def test_load_gradients_ndim_error(tmp_path, setup_random_dwi_data, row_major_gradients):
    dwi_dataobj, affine, brainmask_dataobj, b0_dataobj, gradients, b0_thres = setup_random_dwi_data

    dwi, _, _ = _dwi_data_to_nifti(
        dwi_dataobj, affine, brainmask_dataobj.astype(np.uint8), b0_dataobj
    )

    dwi_fname = tmp_path / "dwi.nii.gz"
    nb.save(dwi, dwi_fname)

    # Store a single column from gradients to try loading a 1D-array. Store as
    # column or array depending on whether to follow the row-major convention or
    # not
    gradients = gradients[:, 0]
    if not row_major_gradients:
        gradients = gradients[np.newaxis, :]

    grads_fname = tmp_path / "grads.txt"
    np.savetxt(grads_fname, gradients, fmt="%.6f")

    with pytest.raises(ValueError, match=GRADIENT_NDIM_ERROR_MSG):
        from_nii(dwi_fname, gradients_file=grads_fname)


@pytest.mark.random_gtab_data(10, (1000, 2000), 2)
@pytest.mark.random_dwi_data(50, (34, 36, 24), True)
@pytest.mark.parametrize("row_major_gradients", (False, True))
@pytest.mark.parametrize("additional_grad_columns", (-1, -2, 1))
def test_load_gradients_expected_columns_error(
    request, tmp_path, setup_random_dwi_data, row_major_gradients, additional_grad_columns
):
    dwi_dataobj, affine, brainmask_dataobj, b0_dataobj, gradients, b0_thres = setup_random_dwi_data

    dwi, _, _ = _dwi_data_to_nifti(
        dwi_dataobj,
        affine,
        brainmask_dataobj.astype(np.uint8),
        b0_dataobj,
    )

    dwi_fname = tmp_path / "dwi.nii.gz"
    nb.save(dwi, dwi_fname)

    # Remove/prepend columns. At this point, it is irrelevant whether the
    # potential N-dimensional vector is normalized or not
    if additional_grad_columns < 1:
        gradients = gradients[:, : gradients.shape[1] + additional_grad_columns]
    else:
        rng = request.node.rng
        add_gradients = rng.random(size=(gradients.shape[0], additional_grad_columns))
        gradients = np.insert(gradients, 0, add_gradients, axis=1)

    if not row_major_gradients:
        gradients = gradients.T

    grads_fname = tmp_path / "grads.txt"
    np.savetxt(grads_fname, gradients, fmt="%.6f")

    with pytest.raises(ValueError, match=re.escape(GRADIENT_EXPECTED_COLUMNS_ERROR_MSG)):
        from_nii(dwi_fname, gradients_file=grads_fname)


@pytest.mark.random_gtab_data(10, (1000, 2000), 2)
@pytest.mark.random_dwi_data(50, (34, 36, 24), True)
def test_load_gradients_bval_bvec_warn(tmp_path, setup_random_dwi_data):
    dwi_dataobj, affine, brainmask_dataobj, b0_dataobj, gradients, _ = setup_random_dwi_data

    dwi, _, _ = _dwi_data_to_nifti(
        dwi_dataobj,
        affine,
        brainmask_dataobj.astype(np.uint8),
        b0_dataobj,
    )

    dwi_fname = tmp_path / "dwi.nii.gz"
    nb.save(dwi, dwi_fname)

    b0_fname = tmp_path / "b0.nii.gz"
    nb.Nifti1Image(b0_dataobj, np.eye(4), None).to_filename(b0_fname)

    grads_fname = tmp_path / "grads.txt"
    np.savetxt(grads_fname, gradients, fmt="%.6f")

    bvals = gradients[:, -1]
    bvecs = gradients[:, :-1]

    bval_fname = tmp_path / "dwi.bval"
    bvec_fname = tmp_path / "dwi.bvec"
    np.savetxt(bvec_fname, bvecs, fmt="%.6f")
    np.savetxt(bval_fname, bvals, fmt="%.6f")

    with pytest.warns(UserWarning, match=re.escape(GRADIENT_BVAL_BVEC_PRIORITY_WARN_MSG)):
        _ = from_nii(
            dwi_fname,
            gradients_file=grads_fname,
            bvec_file=bvec_fname,
            bval_file=bval_fname,
            b0_file=b0_fname,
        )


@pytest.mark.random_gtab_data(10, (1000, 2000), 2)
@pytest.mark.random_dwi_data(50, (34, 36, 24), True)
@pytest.mark.parametrize("row_major_gradients", (False, True))
def test_load_gradients(tmp_path, setup_random_dwi_data, row_major_gradients):
    (
        dwi_dataobj,
        affine,
        brainmask_dataobj,
        b0_dataobj,
        gradients,
        b0_thres,
    ) = setup_random_dwi_data

    dwi, _, _ = _dwi_data_to_nifti(
        dwi_dataobj,
        affine,
        brainmask_dataobj.astype(np.uint8),
        b0_dataobj,
    )

    dwi_fname = tmp_path / "dwi.nii.gz"
    nb.save(dwi, dwi_fname)

    if not row_major_gradients:
        gradients = gradients.T

    grads_fname = tmp_path / "grads.txt"
    np.savetxt(grads_fname, gradients, fmt="%.6f")

    dwi = from_nii(dwi_fname, gradients_file=grads_fname)
    if not row_major_gradients:
        gradmask = gradients.T[:, -1] > b0_thres
    else:
        gradmask = gradients[:, -1] > b0_thres

    if not row_major_gradients:
        expected_nonzero_grads = gradients.T[gradmask]
    else:
        expected_nonzero_grads = gradients[gradmask]

    assert hasattr(dwi, "gradients")
    assert dwi.gradients.shape == expected_nonzero_grads.shape
    assert np.allclose(dwi.gradients, expected_nonzero_grads)


@pytest.mark.random_gtab_data(10, (1000, 2000), 2)
@pytest.mark.random_dwi_data(50, (34, 36, 24), True)
@pytest.mark.parametrize(
    ("transpose_bvals", "transpose_bvecs"),
    [
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ],
)
def test_load_bvecs_bvals(tmp_path, setup_random_dwi_data, transpose_bvals, transpose_bvecs):
    dwi_dataobj, affine, brainmask_dataobj, b0_dataobj, gradients, b0_thres = setup_random_dwi_data

    bvals = gradients[:, -1]
    bvecs = gradients[:, :-1]

    dwi, _, _ = _dwi_data_to_nifti(
        dwi_dataobj,
        affine,
        brainmask_dataobj.astype(np.uint8),
        b0_dataobj,
    )

    dwi_fname = tmp_path / "dwi.nii.gz"
    nb.save(dwi, dwi_fname)

    if transpose_bvals:
        bvals = bvals.T
    if transpose_bvecs:
        bvecs = bvecs.T

    bval_fname = tmp_path / "dwi.bval"
    bvec_fname = tmp_path / "dwi.bvec"
    np.savetxt(bvec_fname, bvecs, fmt="%.6f")
    np.savetxt(bval_fname, bvals, fmt="%.6f")

    dwi = from_nii(dwi_fname, bvec_file=bvec_fname, bval_file=bval_fname)
    gradmask = gradients[:, -1] > b0_thres

    expected_nonzero_grads = gradients[gradmask]
    assert hasattr(dwi, "gradients")
    assert dwi.gradients.shape == expected_nonzero_grads.shape
    assert np.allclose(dwi.gradients, expected_nonzero_grads)


@pytest.mark.random_gtab_data(10, (1000, 2000), 2)
@pytest.mark.random_dwi_data(50, (34, 36, 24), True)
def test_load_gradients_missing(tmp_path, setup_random_dwi_data):
    dwi_dataobj, affine, brainmask_dataobj, b0_dataobj, _, _ = setup_random_dwi_data

    dwi, _, _ = _dwi_data_to_nifti(
        dwi_dataobj,
        affine,
        brainmask_dataobj.astype(np.uint8),
        b0_dataobj,
    )

    dwi_fname = tmp_path / "dwi.nii.gz"
    nb.save(dwi, dwi_fname)

    with pytest.raises(RuntimeError, match=re.escape(GRADIENT_DATA_MISSING_ERROR)):
        from_nii(dwi_fname)


@pytest.mark.skip(reason="to_nifti takes absurdly long")
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
        assert dwi_h5.bzero is not None
        assert dwi_from_nifti1.bzero is not None
        assert np.allclose(dwi_h5.bzero, dwi_from_nifti1.bzero)

    assert np.allclose(dwi_h5.bvals, dwi_from_nifti1.bvals, atol=1e-3)

    bvec_diffs = np.where(
        (~np.isclose(dwi_h5.bvecs, dwi_from_nifti1.bvecs, atol=1e-3)).any(axis=1)
    )[0].tolist()

    assert not bvec_diffs, "\n".join(
        [f"{dwi_h5.bvecs[i, :]} vs {dwi_from_nifti1.bvecs[i, :]}" for i in bvec_diffs]
    )
    assert np.allclose(dwi_h5.gradients, dwi_from_nifti1.gradients, atol=1e-3)

    grad_table = dwi_h5.gradients
    if insert_b0:
        grad_table = np.vstack((np.zeros((1, grad_table.shape[1])), grad_table))
    np.savetxt(str(gradients_path), grad_table)

    # Try loading NIfTI + gradients table
    dwi_from_nifti2 = from_nii(dwi_nifti_path, gradients_file=gradients_path)

    if not rotate_bvecs:  # If we set motion_affines, data WILL change
        assert np.allclose(dwi_h5.dataobj, dwi_from_nifti2.dataobj)
    if insert_b0:
        assert dwi_h5.bzero is not None
        assert dwi_from_nifti2.bzero is not None
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

    assert dwi_h5.bzero is not None
    assert dwi_from_nifti3.bzero is not None
    assert np.allclose(dwi_h5.bzero, dwi_from_nifti3.bzero)
    assert np.allclose(dwi_h5.gradients, dwi_from_nifti3.gradients, atol=1e-6)
    assert np.allclose(dwi_h5.bvals, dwi_from_nifti3.bvals, atol=1e-6)
    assert np.allclose(dwi_h5.bvecs, dwi_from_nifti3.bvecs, atol=1e-6)

    # Try loading NIfTI + gradients table
    dwi_from_nifti4 = from_nii(dwi_nifti_path, gradients_file=gradients_path, b0_file=b0_file)

    if not rotate_bvecs:  # If we set motion_affines, data WILL change
        assert np.allclose(dwi_h5.dataobj, dwi_from_nifti4.dataobj)

    assert dwi_from_nifti4.bzero is not None
    assert np.allclose(dwi_h5.bzero, dwi_from_nifti4.bzero)
    assert np.allclose(dwi_h5.gradients, dwi_from_nifti4.gradients)
    assert np.allclose(dwi_h5.bvals, dwi_from_nifti4.bvals, atol=1e-6)
    assert np.allclose(dwi_h5.bvecs, dwi_from_nifti4.bvecs, atol=1e-6)


@pytest.mark.random_gtab_data(10, (1000,), 1)
@pytest.mark.random_dwi_data(50, (34, 36, 24), True)
def test_equality_operator(tmp_path, setup_random_dwi_data):
    dwi_dataobj, affine, brainmask_dataobj, b0_dataobj, gradients, b0_thres = setup_random_dwi_data

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
    ) = _serialize_dwi_data(dwi, brainmask, b0, gradients, tmp_path)

    # Read back using public API
    dwi_obj_from_nii = from_nii(
        dwi_fname,
        gradients_file=gradients_fname,
        b0_file=b0_fname,
        brainmask_file=brainmask_fname,
    )

    # Direct instantiation with the same arrays
    dwi_obj_direct = DWI(
        dataobj=dwi_dataobj,
        affine=affine,
        brainmask=brainmask_dataobj,
        gradients=gradients,
        bzero=b0_dataobj,
    )

    # Get all user-defined, named attributes
    attrs_to_check = [
        a.name for a in attrs.fields(DWI) if not a.name.startswith("_") and not a.name.isdigit()
    ]

    # Sanity checks (element-wise)
    for attr_name in attrs_to_check:
        val_direct = getattr(dwi_obj_direct, attr_name)
        val_from_nii = getattr(dwi_obj_from_nii, attr_name)

        if val_direct is None or val_from_nii is None:
            assert val_direct is None and val_from_nii is None, f"{attr_name} mismatch"
        else:
            if isinstance(val_direct, np.ndarray):
                assert val_direct.shape == val_from_nii.shape
                assert np.allclose(val_direct, val_from_nii), f"{attr_name} arrays differ"
            else:
                assert val_direct == val_from_nii, f"{attr_name} values differ"

    # Properties derived from gradients should also match
    assert np.allclose(dwi_obj_direct.bvals, dwi_obj_from_nii.bvals)
    assert np.allclose(dwi_obj_direct.bvecs, dwi_obj_from_nii.bvecs)

    # Test equality operator
    assert dwi_obj_direct == dwi_obj_from_nii

    # Test equality operator against an instance from HDF5
    hdf5_filename = tmp_path / "test_dwi.h5"
    dwi_obj_from_nii.to_filename(hdf5_filename)

    round_trip_dwi_obj = DWI.from_filename(hdf5_filename)

    # Symmetric equality
    assert dwi_obj_from_nii == round_trip_dwi_obj
    assert round_trip_dwi_obj == dwi_obj_from_nii


@pytest.mark.random_dwi_data(50, (34, 36, 24), False)
def test_shells(setup_random_dwi_data):
    dwi_dataobj, affine, brainmask_dataobj, b0_dataobj, gradients, _ = setup_random_dwi_data

    dwi_obj = DWI(
        dataobj=dwi_dataobj,
        affine=affine,
        brainmask=brainmask_dataobj,
        bzero=b0_dataobj,
        gradients=gradients,
    )

    num_bins = 3
    _, expected_bval_groups, expected_bval_est = find_shelling_scheme(
        dwi_obj.bvals, num_bins=num_bins
    )

    indices = [np.where(np.isin(dwi_obj.bvals, bvals))[0] for bvals in expected_bval_groups]
    expected_dwi_data = [dwi_obj.dataobj[..., idx] for idx in indices]
    expected_motion_affines = [
        dwi_obj.motion_affines[idx] if dwi_obj.motion_affines else None for idx in indices
    ]
    expected_gradients = [dwi_obj.gradients[idx, ...] for idx in indices]

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

    with pytest.raises(ValueError, match=re.escape("DWI must have at least one high-b shell")):
        find_shelling_scheme(np.zeros(10), num_bins=num_bins)


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
    "bvals",
    [
        np.asarray(
            [
                5,
                300,
                305,
                5,
                1005,
                995,
                1000,
                1005,
                5,
                995,
                1000,
                995,
                995,
                5,
                2005,
                2000,
                2005,
                2005,
                1995,
            ]
        )
    ],
)
@pytest.mark.parametrize(
    "num_bins, multishell_nonempty_bin_count_thr, bval_cap, exp_scheme, exp_bval_groups, exp_bval_estimated",
    [
        # Low multi-shell bin count threshold value
        (
            4,
            3,
            2500,
            "DSI",
            [
                [5, 300, 305, 5, 5, 5],
                [995, 1000, 995, 1000, 995, 995],
                [1005, 1005],
                [2005, 2000, 2005, 2005, 1995],
            ],
            [5.0, 995.0, 1005.0, 2005.0],
        ),
        (
            5,
            3,
            2500,
            "DSI",
            [
                [5, 300, 305, 5, 5, 5],
                [1005, 995, 1000, 1005, 995, 1000, 995, 995],
                [2005, 2000, 2005, 2005, 1995],
            ],
            [5.0, 997.5, 2005.0],
        ),
        # Fewer bins: ensure function still returns a consistent scheme
        (
            3,
            3,
            2500,
            "DSI",
            [
                [5, 300, 305, 5, 5, 5],
                [1005, 995, 1000, 1005, 995, 1000, 995, 995],
                [2005, 2000, 2005, 2005, 1995],
            ],
            [5.0, 997.5, 2005.0],
        ),
        # Tighter cap: high shells beyond cap should be handled
        (
            5,
            3,
            1500,
            "DSI",
            [[5, 5, 5, 5], [300, 305], [1005, 995, 1000, 1005, 995, 1000, 995, 995]],
            [5.0, 302.5, 997.5],
        ),
        # Increase threshold to determine as multi-shell
        (
            3,
            6,
            2500,
            "multi-shell",
            [
                [5, 300, 305, 5, 5, 5],
                [1005, 995, 1000, 1005, 995, 1000, 995, 995],
                [2005, 2000, 2005, 2005, 1995],
            ],
            [5.0, 997.5, 2005.0],
        ),
        (
            4,
            10,
            2500,
            "multi-shell",
            [
                [5, 300, 305, 5, 5, 5],
                [995, 1000, 995, 1000, 995, 995],
                [1005, 1005],
                [2005, 2000, 2005, 2005, 1995],
            ],
            [5.0, 995.0, 1005.0, 2005.0],
        ),
        # Decrease num bins to determine as single-shell
        (
            2,
            10,
            2500,
            "single-shell",
            [
                [5, 300, 305, 5, 995, 1000, 5, 995, 1000, 995, 995, 5],
                [1005, 1005, 2005, 2000, 2005, 2005, 1995],
            ],
            [650.0, 2000.0],
        ),
        # Limit high-shell cap
        (
            2,
            10,
            1000,
            "single-shell",
            [[5, 300, 305, 5, 5, 5], [995, 1000, 995, 1000, 995, 995]],
            [5.0, 995.0],
        ),
    ],
)
def test_find_shelling_scheme_params(
    bvals,
    num_bins,
    multishell_nonempty_bin_count_thr,
    bval_cap,
    exp_scheme,
    exp_bval_groups,
    exp_bval_estimated,
):
    """Test find_shelling_scheme on the same bvals vector with different
    parameter settings.

    For the baseline parameter set we assert exact equality against the known expected
    scheme, groups and estimated b-values. For other parameter sets we assert structural
    invariants and basic sanity checks (no unexpected shapes, estimated b-values within
    the provided cap when applicable, and that groups contain only b-values from the input).
    """
    obt_scheme, obt_bval_groups, obt_bval_estimated = find_shelling_scheme(
        bvals,
        num_bins=num_bins,
        multishell_nonempty_bin_count_thr=multishell_nonempty_bin_count_thr,
        bval_cap=bval_cap,
    )

    # Basic structural checks
    assert obt_scheme == exp_scheme
    assert isinstance(obt_bval_groups, list)
    assert isinstance(obt_bval_estimated, list)

    # Estimated values length should match number of groups
    assert len(obt_bval_estimated) == len(obt_bval_groups)

    # Compare groups and estimated bvals: same number and same elements (order-preserving)
    assert all(
        np.allclose(obt_arr, exp_arr)
        for obt_arr, exp_arr in zip(obt_bval_groups, exp_bval_groups, strict=True)
    )

    # If a finite bval_cap is given, make sure estimated bvals don't exceed it
    if np.isfinite(bval_cap):
        for est in np.asarray(obt_bval_estimated).ravel():
            assert est <= bval_cap + 1e-8  # Tiny tolerance
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
