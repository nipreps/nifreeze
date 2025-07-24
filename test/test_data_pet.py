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

import nibabel as nb
import numpy as np
import pytest
from nitransforms.linear import Affine

from nifreeze.data.pet import PET, _compute_uptake_statistic, from_nii


@pytest.mark.random_uniform_spatial_data((2, 2, 2, 2), 0.0, 1.0)
def test_from_nii_requires_frame_time(setup_random_uniform_spatial_data, tmp_path):
    data, affine = setup_random_uniform_spatial_data
    img = nb.Nifti1Image(data, affine)
    fname = tmp_path / "pet.nii.gz"
    img.to_filename(fname)

    with pytest.raises(RuntimeError, match="frame_time must be provided"):
        from_nii(fname)


def _create_dataset():
    rng = np.random.default_rng(12345)
    data = rng.random((4, 4, 4, 5), dtype=np.float32)
    affine = np.eye(4, dtype=np.float32)
    mask = np.ones((4, 4, 4), dtype=bool)
    midframe = np.array([10, 20, 30, 40, 50], dtype=np.float32)
    return PET(
        dataobj=data,
        affine=affine,
        brainmask=mask,
        midframe=midframe,
        total_duration=60.0,
    )


@pytest.mark.parametrize("stat_func", (np.sum, np.mean, np.std))
def test_compute_uptake_statistic(stat_func):
    rng = np.random.default_rng(12345)
    data = rng.random((4, 4, 4, 5), dtype=np.float32)

    expected = stat_func(data.reshape(-1, data.shape[-1]), axis=0)
    obtained = _compute_uptake_statistic(data, stat_func=stat_func)
    np.testing.assert_array_equal(obtained, expected)


def test_pet_set_transform_updates_motion_affines():
    dataset = _create_dataset()
    idx = 2
    data_before = np.copy(dataset.dataobj[..., idx])

    affine = np.eye(4)
    dataset.set_transform(idx, affine)

    np.testing.assert_allclose(dataset.dataobj[..., idx], data_before)
    assert dataset.motion_affines is not None
    assert len(dataset.motion_affines) == len(dataset)
    assert isinstance(dataset.motion_affines[idx], Affine)
    np.testing.assert_array_equal(dataset.motion_affines[idx].matrix, affine)

    vol, aff, time = dataset[idx]
    assert aff is dataset.motion_affines[idx]


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
