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
from pathlib import Path

import nibabel as nb
import numpy as np
import pytest
from nitransforms.linear import Affine

from nifreeze.data.pet import PET, _compute_frame_duration, _compute_uptake_statistic, from_nii


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


@pytest.mark.random_uniform_spatial_data((2, 2, 2, 2), 0.0, 1.0)
def test_from_nii_requires_frame_time(setup_random_uniform_spatial_data, tmp_path):
    data, affine = setup_random_uniform_spatial_data
    img = nb.Nifti1Image(data, affine)
    fname = tmp_path / "pet.nii.gz"
    img.to_filename(fname)

    with pytest.raises(RuntimeError, match="frame_time must be provided"):
        from_nii(fname)


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
    img = nb.load(filename)
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
        assert hasattr(pet_obj, "brainmask")
        np.testing.assert_array_equal(pet_obj.brainmask, mask_data)


@pytest.mark.random_pet_data(5, (4, 4, 4), np.asarray([10.0, 20.0, 30.0, 40.0, 50.0]), 60.0)
def test_to_nifti(tmp_path, random_dataset):
    out_filename = tmp_path / "random_pet_out.nii.gz"
    random_dataset.to_nifti(str(out_filename))
    assert out_filename.exists()
    loaded_img = nb.load(str(out_filename))
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
    img = nb.load(filename)
    pet_obj = from_nii(filename, frame_time=frame_time, frame_duration=frame_duration)
    out_fname = tmp_path / "random_pet_out.nii.gz"
    pet_obj.to_nifti(out_fname)
    assert out_fname.exists()
    loaded_img = nb.load(out_fname)
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
def test_pet_load(setup_random_uniform_spatial_data, tmp_path):
    data, affine = setup_random_uniform_spatial_data
    img = nb.Nifti1Image(data, affine)
    fname = tmp_path / "pet.nii.gz"
    img.to_filename(fname)

    json_file = tmp_path / "pet.json"
    metadata = {
        "FrameDuration": [1.0, 1.0],
        "FrameTimesStart": [0.0, 1.0],
    }
    json_file.write_text(json.dumps(metadata))

    pet_obj = PET.load(fname, json_file)

    assert pet_obj.dataobj.shape == data.shape
    assert np.allclose(pet_obj.midframe, [0.5, 1.5])
    assert pet_obj.total_duration == 2.0
