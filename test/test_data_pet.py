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

import json
import re
from pathlib import Path

import nibabel as nb
import numpy as np
import pytest
from nitransforms.linear import Affine

from nifreeze.data.pet import (
    ARRAY_ATTRIBUTE_SHAPE_ERROR_MSG,
    ATTRIBUTE_SHAPE_MISMATCH_ERROR_MSG,
    PET,
    SCALAR_ATTRIBUTE_ERROR_MSG,
    _compute_frame_duration,
    _compute_uptake_statistic,
    from_nii,
)
from nifreeze.utils.ndimage import load_api


@pytest.fixture
def random_dataset(setup_random_pet_data) -> PET:
    """Create a PET dataset with random data for testing."""

    (
        pet_dataobj,
        affine,
        brainmask_dataobj,
        midframe,
        total_duration,
    ) = setup_random_pet_data

    return PET(
        dataobj=pet_dataobj,
        affine=affine,
        brainmask=brainmask_dataobj,
        midframe=midframe,
        total_duration=total_duration,
    )


@pytest.fixture
def random_nifti_file(tmp_path, setup_random_uniform_spatial_data) -> Path:
    _data, _affine = setup_random_uniform_spatial_data
    _filename = tmp_path / "random_pet.nii.gz"
    _img = nb.Nifti1Image(_data, _affine)
    _img.to_filename(_filename)
    return _filename


@pytest.mark.random_uniform_spatial_data((2, 2, 2, 4), 0.0, 1.0)
@pytest.mark.parametrize("size", ((2, 1), (1, 2), (3, 1), (3, 2)))
def test_pet_midframe_shape_error(setup_random_uniform_spatial_data, size):
    data, affine = setup_random_uniform_spatial_data
    midframe = np.zeros(size, dtype=np.float32)
    with pytest.raises(
        ValueError, match=ARRAY_ATTRIBUTE_SHAPE_ERROR_MSG.format(attribute="midframe")
    ):
        PET(dataobj=data, affine=affine, midframe=midframe)


@pytest.mark.random_uniform_spatial_data((2, 2, 2, 4), 0.0, 1.0)
@pytest.mark.parametrize("size", ((2, 1), (1, 1)))
def test_pet_total_duration_error(request, setup_random_uniform_spatial_data, size):
    data, affine = setup_random_uniform_spatial_data
    midframe = np.zeros(data.shape[-1], dtype=np.float32)
    total_duration = request.node.rng.uniform(5.0, 20.0, size=size)
    with pytest.raises(
        ValueError, match=SCALAR_ATTRIBUTE_ERROR_MSG.format(attribute="total_duration")
    ):
        PET(dataobj=data, affine=affine, midframe=midframe, total_duration=total_duration)


@pytest.mark.random_uniform_spatial_data((2, 2, 2, 4), 0.0, 1.0)
@pytest.mark.parametrize("size", ((2, 1), (1, 2), (3, 1), (3, 2)))
def test_pet_uptake_shape_error(setup_random_uniform_spatial_data, size):
    data, affine = setup_random_uniform_spatial_data
    midframe = np.zeros(data.shape[-1], dtype=np.float32)
    total_duration = 16.2
    uptake = np.zeros(size, dtype=np.float32)
    with pytest.raises(
        ValueError, match=ARRAY_ATTRIBUTE_SHAPE_ERROR_MSG.format(attribute="uptake")
    ):
        PET(
            dataobj=data,
            affine=affine,
            midframe=midframe,
            total_duration=total_duration,
            uptake=uptake,
        )


@pytest.mark.random_uniform_spatial_data((2, 2, 2, 4), 0.0, 1.0)
def test_pet_midframe_length_mismatch(setup_random_uniform_spatial_data):
    data, affine = setup_random_uniform_spatial_data
    total_duration = 16.2
    data_frames = data.shape[-1]
    attr_len = data_frames + 1
    midframe = np.zeros(attr_len, dtype=np.float32)
    uptake = np.zeros(data_frames, dtype=np.float32)
    with pytest.raises(
        ValueError,
        match=re.escape(
            ATTRIBUTE_SHAPE_MISMATCH_ERROR_MSG.format(
                attribute="midframe", attr_len=attr_len, data_frames=data_frames
            )
        ),
    ):
        PET(
            dataobj=data,
            affine=affine,
            midframe=midframe,
            total_duration=total_duration,
            uptake=uptake,
        )


@pytest.mark.random_uniform_spatial_data((2, 2, 2, 4), 0.0, 1.0)
def test_pet_uptake_length_mismatch(setup_random_uniform_spatial_data):
    data, affine = setup_random_uniform_spatial_data
    total_duration = 16.2
    data_frames = data.shape[-1]
    midframe = np.zeros(data_frames, dtype=np.float32)
    attr_len = data_frames + 1
    uptake = np.zeros(attr_len, dtype=np.float32)
    with pytest.raises(
        ValueError,
        match=re.escape(
            ATTRIBUTE_SHAPE_MISMATCH_ERROR_MSG.format(
                attribute="uptake", attr_len=attr_len, data_frames=data_frames
            )
        ),
    ):
        PET(
            dataobj=data,
            affine=affine,
            midframe=midframe,
            total_duration=total_duration,
            uptake=uptake,
        )


@pytest.mark.parametrize(
    "midframe, expected",
    [
        ([1.0, 4.0], [3.0, 3.0]),
        ([0.0, 5.0, 9.0, 12.0], [5.0, 4.0, 3.0, 3.0]),
    ],
)
def test_compute_frame_duration(midframe, expected):
    midframe = np.array(midframe)
    expected = np.array(expected)
    durations = _compute_frame_duration(midframe)
    np.testing.assert_allclose(durations, expected)


@pytest.mark.parametrize("stat_func", (np.sum, np.mean, np.std))
def test_compute_uptake_statistic(stat_func):
    rng = np.random.default_rng(12345)
    data = rng.random((4, 4, 4, 5), dtype=np.float32)

    expected = stat_func(data.reshape(-1, data.shape[-1]), axis=0)
    obtained = _compute_uptake_statistic(data, stat_func=stat_func)
    np.testing.assert_array_equal(obtained, expected)


@pytest.mark.parametrize(
    ("brainmask_file", "frame_time", "frame_duration"),
    [
        (None, [0.0, 5.0], [5.0, 5.0]),
        (None, [10.0, 15.0], [5.0, 5.0]),
        ("mask.nii.gz", [0.0, 5.0], [5.0, 5.0]),
        ("mask.nii.gz", [0.0, 5.0], None),
    ],
)
def test_from_nii(tmp_path, random_nifti_file, brainmask_file, frame_time, frame_duration):
    filename = random_nifti_file
    img = load_api(filename, nb.Nifti1Image)
    if brainmask_file:
        mask_data = np.ones(img.get_fdata().shape[:-1], dtype=bool)
        mask_img = nb.Nifti1Image(mask_data.astype(np.uint8), img.affine)
        mask_img.to_filename(brainmask_file)

    pet_obj = from_nii(
        filename,
        brainmask_file=brainmask_file,
        frame_time=frame_time,
        frame_duration=frame_duration,
    )
    assert isinstance(pet_obj, PET)
    assert pet_obj.dataobj.shape == img.get_fdata().shape
    np.testing.assert_array_equal(pet_obj.affine, img.affine)

    # Convert to a float32 numpy array and zero out the earliest time
    frame_time_arr = np.array(frame_time, dtype=np.float32)
    frame_time_arr -= frame_time_arr[0]
    if frame_duration is None:
        durations = _compute_frame_duration(frame_time_arr)
    else:
        durations = np.array(frame_duration, dtype=np.float32)

    expected_total_duration = float(frame_time_arr[-1] + durations[-1])
    expected_midframe = frame_time_arr + 0.5 * durations

    np.testing.assert_allclose(pet_obj.midframe, expected_midframe)
    assert pet_obj.total_duration == expected_total_duration

    if brainmask_file:
        assert pet_obj.brainmask is not None
        np.testing.assert_array_equal(pet_obj.brainmask, mask_data)


@pytest.mark.random_pet_data(5, (4, 4, 4), np.asarray([10.0, 20.0, 30.0, 40.0, 50.0]), 60.0)
def test_to_nifti(tmp_path, random_dataset):
    out_filename = tmp_path / "random_pet_out.nii.gz"
    random_dataset.to_nifti(str(out_filename))
    assert out_filename.exists()
    loaded_img = load_api(str(out_filename), nb.Nifti1Image)
    assert np.allclose(loaded_img.get_fdata(), random_dataset.dataobj)
    assert np.allclose(loaded_img.affine, random_dataset.affine)
    units = loaded_img.header.get_xyzt_units()
    assert units[0] == "mm"


@pytest.mark.parametrize(
    ("frame_time", "frame_duration"),
    [
        ([0.0, 5.0], [5.0, 5.0]),
    ],
)
def test_round_trip(tmp_path, random_nifti_file, frame_time, frame_duration):
    filename = random_nifti_file
    img = load_api(filename, nb.Nifti1Image)
    pet_obj = from_nii(filename, frame_time=frame_time, frame_duration=frame_duration)
    out_fname = tmp_path / "random_pet_out.nii.gz"
    pet_obj.to_nifti(out_fname)
    assert out_fname.exists()
    loaded_img = load_api(out_fname, nb.Nifti1Image)
    np.testing.assert_array_equal(loaded_img.affine, img.affine)
    np.testing.assert_allclose(loaded_img.get_fdata(), img.get_fdata())
    units = loaded_img.header.get_xyzt_units()
    assert units[0] == "mm"


@pytest.mark.random_pet_data(5, (4, 4, 4), np.asarray([10.0, 20.0, 30.0, 40.0, 50.0]), 60.0)
def test_pet_set_transform_updates_motion_affines(random_dataset):
    idx = 2
    data_before = np.copy(random_dataset.dataobj[..., idx])

    affine = np.eye(4)
    random_dataset.set_transform(idx, affine)

    np.testing.assert_allclose(random_dataset.dataobj[..., idx], data_before)
    assert random_dataset.motion_affines is not None
    assert len(random_dataset.motion_affines) == len(random_dataset)
    assert isinstance(random_dataset.motion_affines[idx], Affine)
    np.testing.assert_array_equal(random_dataset.motion_affines[idx].matrix, affine)

    vol, aff, time = random_dataset[idx]
    assert aff is random_dataset.motion_affines[idx]


@pytest.mark.random_uniform_spatial_data((2, 2, 2, 2), 0.0, 1.0)
def test_pet_load(request, tmp_path, setup_random_uniform_spatial_data):
    data, affine = setup_random_uniform_spatial_data
    img = nb.Nifti1Image(data, affine)
    fname = tmp_path / "pet.nii.gz"
    img.to_filename(fname)

    brainmask_dataobj = request.node.rng.choice([True, False], size=data.shape[:3]).astype(
        np.uint8
    )
    brainmask = nb.Nifti1Image(brainmask_dataobj, affine)
    brainmask_fname = tmp_path / "brainmask.nii.gz"
    brainmask.to_filename(brainmask_fname)

    json_file = tmp_path / "pet.json"
    metadata = {
        "FrameDuration": [1.0, 1.0],
        "FrameTimesStart": [0.0, 1.0],
    }
    json_file.write_text(json.dumps(metadata))

    pet_obj = PET.load(fname, json_file, brainmask_fname)

    assert pet_obj.dataobj.shape == data.shape
    assert np.allclose(pet_obj.midframe, [0.5, 1.5])
    assert pet_obj.total_duration == 2.0
    if pet_obj.brainmask is not None:
        assert pet_obj.brainmask.shape == brainmask_dataobj.shape
