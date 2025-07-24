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

from nifreeze.data.pet import PET, from_nii


def test_from_nii_requires_frame_time(tmp_path):
    data = np.zeros((2, 2, 2, 2), dtype=np.float32)
    img = nb.Nifti1Image(data, np.eye(4))
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


def test_pet_load(tmp_path):
    data = np.zeros((2, 2, 2, 2), dtype=np.float32)
    affine = np.eye(4)
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
