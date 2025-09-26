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

import types

import numpy as np
import pytest

from nifreeze.data.pet import PET
from nifreeze.estimator import PETMotionEstimator


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


@pytest.mark.random_pet_data(4, (2, 2, 2), np.asarray([1.0, 2.0, 3.0, 4.0]), 5.0)
def test_lofo_split_shapes(random_dataset, tmp_path):
    idx = 2
    (train_data, train_times), (test_data, test_time) = random_dataset.lofo_split(idx)
    assert train_data.shape[-1] == random_dataset.dataobj.shape[-1] - 1
    np.testing.assert_array_equal(test_data, random_dataset.dataobj[..., idx])
    np.testing.assert_array_equal(train_times, np.delete(random_dataset.midframe, idx))
    assert test_time == random_dataset.midframe[idx]


@pytest.mark.random_pet_data(3, (2, 2, 2), np.asarray([1.0, 2.0, 3.0]), 4.0)
def test_to_from_filename_roundtrip(random_dataset, tmp_path):
    out_file = tmp_path / "petdata"
    random_dataset.to_filename(out_file)
    assert (tmp_path / "petdata.h5").exists()
    loaded = PET.from_filename(tmp_path / "petdata.h5")
    np.testing.assert_allclose(loaded.dataobj, random_dataset.dataobj)
    np.testing.assert_allclose(loaded.affine, random_dataset.affine)
    np.testing.assert_allclose(loaded.midframe, random_dataset.midframe)
    assert loaded.total_duration == random_dataset.total_duration


@pytest.mark.random_pet_data(5, (4, 4, 4), np.asarray([10.0, 20.0, 30.0, 40.0, 50.0]), 60.0)
def test_pet_motion_estimator_run(random_dataset, monkeypatch):
    class DummyModel:
        def __init__(self, dataset, timepoints, xlim):
            self.dataset = dataset

        def fit_predict(self, index):
            if index is None:
                return None
            return np.zeros(self.dataset.shape3d, dtype=np.float32)

    monkeypatch.setattr("nifreeze.estimator.PETModel", DummyModel)

    class DummyRegistration:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, cwd=None):
            return types.SimpleNamespace(outputs=types.SimpleNamespace(forward_transforms=[]))

    monkeypatch.setattr("nifreeze.estimator.Registration", DummyRegistration)

    estimator = PETMotionEstimator(None)
    affines = estimator.run(random_dataset)
    assert len(affines) == len(random_dataset)
    for mat in affines:
        np.testing.assert_array_equal(mat, np.eye(4))
