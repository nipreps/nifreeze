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
import warnings
from pathlib import Path

import attrs
import nibabel as nb
import numpy as np
import pytest
from dipy.core.geometry import normalized_vector

from nifreeze.data import load
from nifreeze.data.dmri.base import (
    BZERO_SHAPE_MISMATCH_ERROR_MSG,
    DWI,
    DWI_B0_MULTIPLE_VOLUMES_WARN_MSG,
    DWI_REDUNDANT_B0_WARN_MSG,
    validate_gradients,
)
from nifreeze.data.dmri.io import (
    GRADIENT_BVAL_BVEC_PRIORITY_WARN_MSG,
    GRADIENT_DATA_MISSING_ERROR_MSG,
    from_nii,
    to_nifti,
)
from nifreeze.data.dmri.utils import (
    DEFAULT_LOWB_THRESHOLD,
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


@pytest.mark.parametrize(
    "case_mark",
    [
        pytest.param(
            None,
            marks=pytest.mark.random_uniform_spatial_data((2, 2, 2, 6), 0.0, 1.0),
        ),
        pytest.param(
            None,
            marks=pytest.mark.random_uniform_spatial_data((2, 2, 2, 4), 0.0, 1.0),
        ),
    ],
)
def test_dwi_post_init_orientation_count_errors(setup_random_uniform_spatial_data, case_mark):
    data, affine = setup_random_uniform_spatial_data
    with pytest.raises(ValueError, match=GRADIENT_ABSENCE_ERROR_MSG):
        DWI(dataobj=data, affine=affine)

    with pytest.raises(
        ValueError,
        match=f"DWI datasets must have at least {DTI_MIN_ORIENTATIONS} diffusion-weighted",
    ):
        DWI(dataobj=data, affine=affine, gradients=B_MATRIX[: data.shape[-1], :])


@pytest.mark.parametrize(
    "extra_b0_axis, extra_b0_count, extra_b0_slice_axis, extra_b0_slice_count",
    [(1, 0, 0, 0), (1, 1, 0, 0), (1, 2, 0, 0), (0, 0, 0, 1), (0, 0, 1, 2), (0, 0, 2, 1)],
)
def test_dwi_post_init_bzero_errors(
    request,
    setup_random_dwi_data,
    extra_b0_axis,
    extra_b0_count,
    extra_b0_slice_axis,
    extra_b0_slice_count,
):
    """Check bzero attribute shape errors when instantiating the DWI class.

    An error is raised if the provided bzero's shape does not match the first
    three values of the DWI data object, i.e. if
      - the bzero volume has one extra axis;
      - there are more than one bzero volumes;
      - there are more slabs with respect to those in a single 3D volume.
    """
    rng = request.node.rng
    dwi_dataobj, affine, _, b0_dataobj, gradients, _ = setup_random_dwi_data
    _b0_dataobj = b0_dataobj
    # Add a new axis if requested
    if extra_b0_axis > 0:
        _b0_dataobj = _b0_dataobj[..., np.newaxis]
    # Add new b0 volumes if requested
    if extra_b0_count >= 1:
        extra_bvols = rng.random((*b0_dataobj.shape, extra_b0_count))
        _b0_dataobj = np.concatenate((b0_dataobj[..., np.newaxis], extra_bvols), axis=-1)
    # Add new slices along the requested axis
    if extra_b0_slice_count >= 1:
        # Create extra slices with the same spatial shape as b0_dataobj except
        # along the chosen axis
        extra_slab_shape = list(_b0_dataobj.shape[:3])
        extra_slab_shape[extra_b0_slice_axis] = extra_b0_slice_count
        extra_slab = rng.random(tuple(extra_slab_shape))

        # If the b0 data object is 4D, make slices 4D with a singleton fourth
        # axis so they can be concatenated spatially
        if _b0_dataobj.ndim == 4:
            extra_slab = extra_slab[..., np.newaxis]

        # Concatenate along the chosen spatial axis to make spatial dims differ
        # from dataobj
        _b0_dataobj = np.concatenate((_b0_dataobj, extra_slab), axis=extra_b0_slice_axis)

    with pytest.raises(
        ValueError,
        match=re.escape(
            BZERO_SHAPE_MISMATCH_ERROR_MSG.format(
                bzero_shape=_b0_dataobj.shape, data_shape=dwi_dataobj.shape[:3]
            )
        ),
    ):
        DWI(dataobj=dwi_dataobj, affine=affine, gradients=gradients, bzero=_b0_dataobj)


@pytest.mark.parametrize("vol_size", [(11, 11, 7)])
@pytest.mark.parametrize(
    "b0_count, provide_bzero", [(0, False), (0, True), (1, False), (1, True), (2, False)]
)
@pytest.mark.parametrize("bval_min, bval_max", [(800.0, 1200.0)])
def test_dwi_post_init_b0_handling(request, vol_size, b0_count, provide_bzero, bval_min, bval_max):
    """Check b0 handling when instantiating the DWI class.

    For each parameter combination:
      - Build a gradient table whose first `b0_count` volumes have b=0
        and the rest have b-values in the range (bval_min, bval_max);
      - Build a random dataobj of shape (**vol_size, N) where N is the number
        of DWI volumes;
      - If `provide_bzero` is True, pass explicit bzero data that must be
        preserved; else, rely on the bzero computed at instantiation, i.e.
        if a single bzero is provided, set the attribute to that value; if there
        are multiple bzeros, set the attribute to the median value.
    """
    rng = request.node.rng

    # Choose n_vols safely above the minimum DTI orientations
    n_vols = max(10, DTI_MIN_ORIENTATIONS + 2)

    # Build b-values array: first b0_count are zeros
    non_b0_count = n_vols - b0_count
    # Sample non-b0 bvals between min and max values
    rest_bvals = rng.uniform(bval_min, bval_max, size=non_b0_count)
    bvals = np.concatenate((np.zeros(b0_count), rest_bvals)).astype(int)

    # Create bvecs and assemble gradients
    bzeros = np.zeros((b0_count, 3))
    bvecs = normalized_vector(rng.random((3, non_b0_count)), axis=0).T
    bvecs = np.vstack((bzeros, bvecs))
    gradients = np.column_stack((bvecs, bvals))

    # Create random dataobj with shape
    dataobj = rng.standard_normal((*vol_size, n_vols)).astype(float)

    # Optionally supply a bzero
    provided = None
    affine = np.eye(4)
    if provide_bzero:
        # Use a constant map so it's easy to assert equality
        provided = np.full((*vol_size, max(1, b0_count)), 42.0, dtype=float).squeeze()
        with warnings.catch_warnings(record=True) as caught:
            dwi_obj = DWI(dataobj=dataobj, affine=affine, gradients=gradients, bzero=provided)

        # If the DWI gradients contained null values (i.e. b0 volumes) and bzero
        # data was provided, a warning must have been raised
        if b0_count >= 1 and provide_bzero:
            assert str(caught[0].message) == DWI_REDUNDANT_B0_WARN_MSG
    else:
        with warnings.catch_warnings(record=True) as caught:
            dwi_obj = DWI(dataobj=dataobj, affine=affine, gradients=gradients)

        # If the DWI gradients contained more than a single null value (i.e.
        # multiple b0 volumes), a warning must have been raised
        if b0_count > 1:
            assert str(caught[0].message) == DWI_B0_MULTIPLE_VOLUMES_WARN_MSG

    # Count expected b0 frames according to the same threshold used by the code
    b0_mask = bvals <= DEFAULT_LOWB_THRESHOLD
    expected_b0_num = int(np.sum(b0_mask))
    # In all cases where b0 frames existed (whether provided externally or not),
    # they should have been removed from the DWI object's internal gradients and
    # dataobj arrays
    expected_non_b0_count = n_vols - expected_b0_num

    # If no b0 frames expected, bzero should be None (unless user provided one)
    if expected_b0_num == 0 and not provide_bzero:
        assert dwi_obj.bzero is None, (
            "Expected bzero to be None when no low-b frames and no provided bzero"
        )
    else:
        assert dwi_obj.bzero is not None
        # If provided_bzero is True, it must be preserved exactly
        if provide_bzero:
            assert provided is not None
            assert np.allclose(dwi_obj.bzero, provided)
        else:
            # When there are b0 frames and no provided bzero:
            #  - If exactly one b0 frame, the stored bzero should be the 3D volume
            #  - If multiple b0 frames, the stored bzero should be the median along last axis
            b0_vols = dataobj[
                ..., b0_mask
            ].squeeze()  # shape (X,Y,Z,expected_b0_num) or (X,Y,Z) if 1
            expected_bzero = b0_vols if b0_vols.ndim == 3 else np.median(b0_vols, axis=-1)
            assert np.allclose(dwi_obj.bzero, expected_bzero)

    assert dwi_obj.gradients.shape[0] == expected_non_b0_count
    assert dwi_obj.dataobj.shape[-1] == expected_non_b0_count

    assert np.allclose(dwi_obj.gradients, gradients[~b0_mask])
    assert np.allclose(dwi_obj.dataobj, dataobj[..., ~b0_mask])


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

    dwi_nii, _, _ = _dwi_data_to_nifti(
        dwi_dataobj, affine, brainmask_dataobj.astype(np.uint8), b0_dataobj
    )

    dwi_fname = tmp_path / "dwi.nii.gz"
    nb.save(dwi_nii, dwi_fname)

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

    dwi_nii, _, _ = _dwi_data_to_nifti(
        dwi_dataobj,
        affine,
        brainmask_dataobj.astype(np.uint8),
        b0_dataobj,
    )

    dwi_fname = tmp_path / "dwi.nii.gz"
    nb.save(dwi_nii, dwi_fname)

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

    dwi_nii, _, _ = _dwi_data_to_nifti(
        dwi_dataobj,
        affine,
        brainmask_dataobj.astype(np.uint8),
        b0_dataobj,
    )

    dwi_fname = tmp_path / "dwi.nii.gz"
    nb.save(dwi_nii, dwi_fname)

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
        # Ignore warning due to redundant b0 volumes
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=DWI_REDUNDANT_B0_WARN_MSG, category=UserWarning
            )
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

    dwi_nii, _, _ = _dwi_data_to_nifti(
        dwi_dataobj,
        affine,
        brainmask_dataobj.astype(np.uint8),
        b0_dataobj,
    )

    dwi_fname = tmp_path / "dwi.nii.gz"
    nb.save(dwi_nii, dwi_fname)

    if not row_major_gradients:
        gradients = gradients.T

    grads_fname = tmp_path / "grads.txt"
    np.savetxt(grads_fname, gradients, fmt="%.6f")

    # Ignore warning due to multiple b0 volumes
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=DWI_B0_MULTIPLE_VOLUMES_WARN_MSG, category=UserWarning
        )
        dwi_obj = from_nii(dwi_fname, gradients_file=grads_fname)

    if not row_major_gradients:
        gradmask = gradients.T[:, -1] > b0_thres
    else:
        gradmask = gradients[:, -1] > b0_thres

    if not row_major_gradients:
        expected_nonzero_grads = gradients.T[gradmask]
    else:
        expected_nonzero_grads = gradients[gradmask]

    assert hasattr(dwi_obj, "gradients")
    assert dwi_obj.gradients.shape == expected_nonzero_grads.shape
    assert np.allclose(dwi_obj.gradients, expected_nonzero_grads)


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

    dwi_nii, _, _ = _dwi_data_to_nifti(
        dwi_dataobj,
        affine,
        brainmask_dataobj.astype(np.uint8),
        b0_dataobj,
    )

    dwi_fname = tmp_path / "dwi.nii.gz"
    nb.save(dwi_nii, dwi_fname)

    if transpose_bvals:
        bvals = bvals.T
    if transpose_bvecs:
        bvecs = bvecs.T

    bval_fname = tmp_path / "dwi.bval"
    bvec_fname = tmp_path / "dwi.bvec"
    np.savetxt(bvec_fname, bvecs, fmt="%.6f")
    np.savetxt(bval_fname, bvals, fmt="%.6f")

    # Ignore warning due to multiple b0 volumes
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=DWI_B0_MULTIPLE_VOLUMES_WARN_MSG, category=UserWarning
        )
        dwi_obj = from_nii(dwi_fname, bvec_file=bvec_fname, bval_file=bval_fname)

    gradmask = gradients[:, -1] > b0_thres

    expected_nonzero_grads = gradients[gradmask]
    assert hasattr(dwi_obj, "gradients")
    assert dwi_obj.gradients.shape == expected_nonzero_grads.shape
    assert np.allclose(dwi_obj.gradients, expected_nonzero_grads)


@pytest.mark.random_gtab_data(10, (1000, 2000), 2)
@pytest.mark.random_dwi_data(50, (34, 36, 24), True)
def test_load_gradients_missing(tmp_path, setup_random_dwi_data):
    dwi_dataobj, affine, brainmask_dataobj, b0_dataobj, _, _ = setup_random_dwi_data

    dwi_nii, _, _ = _dwi_data_to_nifti(
        dwi_dataobj,
        affine,
        brainmask_dataobj.astype(np.uint8),
        b0_dataobj,
    )

    dwi_fname = tmp_path / "dwi.nii.gz"
    nb.save(dwi_nii, dwi_fname)

    with pytest.raises(RuntimeError, match=re.escape(GRADIENT_DATA_MISSING_ERROR_MSG)):
        from_nii(dwi_fname)


@pytest.mark.parametrize("vol_size", [(4, 4, 5)])
@pytest.mark.parametrize("b0_count", [0, 1])
@pytest.mark.parametrize("bval_min, bval_max", [(800.0, 1200.0)])
@pytest.mark.parametrize("provide_bzero", [False, True])
@pytest.mark.parametrize("insert_b0", [False, True])
@pytest.mark.parametrize("motion_affines", [None, 2 * np.eye(4)])
@pytest.mark.parametrize("bvals_dec_places, bvecs_dec_places", [(2, 6), (1, 4)])
@pytest.mark.parametrize("file_basename", [None, "dwi.nii.gz"])
def test_to_nifti(
    request,
    tmp_path,
    monkeypatch,
    vol_size,
    b0_count,
    bval_min,
    bval_max,
    provide_bzero,
    insert_b0,
    motion_affines,
    bvals_dec_places,
    bvecs_dec_places,
    file_basename,
):
    rng = request.node.rng

    # Choose n_vols safely above the minimum DTI orientations
    n_vols = max(10, DTI_MIN_ORIENTATIONS + 2)

    # Build b-values array: first b0_count are zeros
    non_b0_count = n_vols - b0_count
    # Sample non-b0 bvals between min and max values
    rest_bvals = rng.uniform(bval_min, bval_max, size=non_b0_count)
    bvals = np.concatenate((np.zeros(b0_count), rest_bvals)).astype(int)

    # Create bvecs and assemble gradients
    bzeros = np.zeros((b0_count, 3))
    bvecs = normalized_vector(rng.random((3, non_b0_count)), axis=0).T
    bvecs = np.vstack((bzeros, bvecs))
    gradients = np.column_stack((bvecs, bvals))

    # Create random dataobj with shape
    dataobj = rng.standard_normal((*vol_size, n_vols)).astype(float)

    # Optionally supply a bzero
    provided = None
    affine = np.eye(4)
    _motion_affines = (
        np.stack([motion_affines] * non_b0_count) if motion_affines is not None else None
    )
    if provide_bzero:
        # Use a constant map so it's easy to assert equality
        provided = np.full((*vol_size, max(1, b0_count)), 42.0, dtype=float).squeeze()
        # Ignore warning due to redundant b0 volumes
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=DWI_REDUNDANT_B0_WARN_MSG, category=UserWarning
            )
            dwi_obj = DWI(
                dataobj=dataobj,
                affine=affine,
                motion_affines=_motion_affines,
                gradients=gradients,
                bzero=provided,
            )
    else:
        dwi_obj = DWI(
            dataobj=dataobj, affine=affine, motion_affines=_motion_affines, gradients=gradients
        )

    _filename = tmp_path / file_basename if file_basename is not None else file_basename

    # Monkeypatch the to_nifti alias to only perform essential operations for
    # the purpose of this test
    def simple_to_nifti(_dataset, filename=None, write_hmxfms=None, order=None):
        _ = write_hmxfms
        _ = order
        _nii = nb.Nifti1Image(_dataset.dataobj, _dataset.affine)
        if filename is not None:
            _nii.to_filename(filename)
        return _nii

    monkeypatch.setattr("nifreeze.data.dmri.io._base_to_nifti", simple_to_nifti)

    with warnings.catch_warnings(record=True) as caught:
        nii = to_nifti(
            dwi_obj,
            _filename,
            write_hmxfms=False,
            order=3,
            insert_b0=insert_b0,
            bvals_dec_places=bvals_dec_places,
            bvecs_dec_places=bvecs_dec_places,
        )

    no_bzero = dwi_obj.bzero is None or not insert_b0

    # Check the warning
    if no_bzero:
        if insert_b0:
            assert (
                str(caught[0].message)
                == "Ignoring ``insert_b0`` argument as the data object's bzero field is unset"
            )

    bvecs_dwi = dwi_obj.bvecs
    bvals_dwi = dwi_obj.bvals
    # Transform bvecs if motion affines are present
    if dwi_obj.motion_affines is not None:
        rotated = [
            transform_fsl_bvec(_bvec, _affine, dwi_obj.affine, invert=True)
            for _bvec, _affine in zip(bvecs_dwi, dwi_obj.motion_affines, strict=True)
        ]
        bvecs_dwi = np.asarray(rotated)

    # Check the primary NIfTI output
    _dataobj = dwi_obj.dataobj
    # Concatenate the b0 if the primary data has a b0 volume or if it was
    # requested to do so
    if not no_bzero:
        assert dwi_obj.bzero is not None
        # ToDo
        # The code will concatenate as many zeros as they exist to the data
        _dataobj = np.concatenate((dwi_obj.bzero[..., np.newaxis], dwi_obj.dataobj), axis=-1)
        # But when inserting b0 data to the gradients, it inserts a single b0.
        # Here I will insert as many values as b0 volumes to make the test fail
        dwi_b0_count = dwi_obj.bzero.shape[-1] if dwi_obj.bzero.ndim == 4 else 1
        bvals_dwi = np.concatenate((np.zeros(dwi_b0_count), bvals_dwi))
        bvecs_dwi = np.vstack((np.zeros((dwi_b0_count, bvecs_dwi.shape[1])), bvecs_dwi))

    assert isinstance(nii, nb.Nifti1Image)
    assert np.allclose(nii.get_fdata(), _dataobj)
    assert np.allclose(nii.affine, dwi_obj.affine)

    # Check the written files, if any
    if _filename is None:
        assert not any(tmp_path.iterdir()), "Directory is not empty"
    else:
        # Check the written NIfTI file
        assert _filename.is_file()

        _nii_load = load_api(_filename, nb.Nifti1Image)

        # Build a NIfTI file with the data object that potentially contains
        # concatenated b0 data
        _nii_dataobj = nb.Nifti1Image(_dataobj, nii.affine, nii.header)

        np.allclose(_nii_dataobj.get_fdata(), _nii_load.get_fdata())
        np.allclose(_nii_dataobj.affine, _nii_load.affine)

        # Check gradients
        if motion_affines is not None:
            bvecs_file = _filename.with_suffix("").with_suffix(".bvec")
            bvals_file = _filename.with_suffix("").with_suffix(".bval")
            assert bvals_file.is_file()
            assert bvecs_file.is_file()

            # Read the files
            bvals_from_file = np.loadtxt(bvals_file)
            bvecs_from_file = np.loadtxt(bvecs_file).T

            assert np.allclose(bvals_from_file, bvals_dwi, rtol=0, atol=10**-bvals_dec_places)
            assert np.allclose(bvecs_from_file, bvecs_dwi, rtol=0, atol=10**-bvecs_dec_places)


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

    dwi_nii, brainmask, b0 = _dwi_data_to_nifti(
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
    ) = _serialize_dwi_data(dwi_nii, brainmask, b0, gradients, tmp_path)

    # Read back using public API
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=DWI_REDUNDANT_B0_WARN_MSG, category=UserWarning)
        dwi_obj_from_nii = from_nii(
            dwi_fname,
            gradients_file=gradients_fname,
            b0_file=b0_fname,
            brainmask_file=brainmask_fname,
        )

    # Direct instantiation with the same arrays
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=DWI_REDUNDANT_B0_WARN_MSG, category=UserWarning)
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

    # Ignore warning due to redundant b0 volumes
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=DWI_REDUNDANT_B0_WARN_MSG, category=UserWarning)
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
