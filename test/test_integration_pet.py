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

import math
import types

import attrs
import numpy as np
import pytest

from nifreeze.data.pet import PET
from nifreeze.estimator import PETMotionEstimator
from nifreeze.model.base import BaseModel


@pytest.mark.random_pet_data(5, (4, 4, 4), np.asarray([10.0, 20.0, 30.0, 40.0, 50.0]))
def test_to_from_filename_roundtrip(tmp_path, setup_random_pet_data):
    pet_dataobj, affine, brainmask_dataobj, _, midframe, total_duration = setup_random_pet_data

    pet_obj = PET(
        dataobj=pet_dataobj,
        affine=affine,
        brainmask=brainmask_dataobj,
        midframe=midframe,
        total_duration=total_duration,
    )

    out_file = tmp_path / "petdata"
    pet_obj.to_filename(out_file)
    assert (tmp_path / "petdata.h5").exists()
    loaded = PET.from_filename(tmp_path / "petdata.h5")

    # Get all user-defined, named attributes
    attrs_to_check = [
        a.name for a in attrs.fields(PET) if not a.name.startswith("_") and not a.name.isdigit()
    ]

    # Sanity checks (element-wise)
    for attr_name in attrs_to_check:
        val_direct = getattr(pet_obj, attr_name)
        val_loaded = getattr(loaded, attr_name)

        if val_direct is None or val_loaded is None:
            assert val_direct is None and val_loaded is None, f"{attr_name} mismatch"
        else:
            if isinstance(val_direct, np.ndarray):
                assert val_direct.shape == val_loaded.shape
                assert np.allclose(val_direct, val_loaded), f"{attr_name} arrays differ"
            else:
                assert math.isclose(val_direct, val_loaded), f"{attr_name} values differ"


@pytest.mark.random_pet_data(5, (4, 4, 4), np.asarray([10.0, 15.0, 20.0, 25.0, 30.0]))
def test_pet_motion_estimator_run(monkeypatch, setup_random_pet_data):
    pet_dataobj, affine, brainmask_dataobj, _, midframe, total_duration = setup_random_pet_data

    pet_obj = PET(
        dataobj=pet_dataobj,
        affine=affine,
        brainmask=brainmask_dataobj,
        midframe=midframe,
        total_duration=total_duration,
    )

    class DummyModel(BaseModel):
        def __init__(self, dataset):
            super().__init__(dataset)
            self.dataset = dataset

        def fit_predict(self, index=None, **kawargs):
            if index is None:
                return None
            return np.zeros(self.dataset.shape3d, dtype=np.float32)

    model = DummyModel(pet_obj)

    class DummyRegistration:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, cwd=None):
            return types.SimpleNamespace(outputs=types.SimpleNamespace(forward_transforms=[]))

    monkeypatch.setattr("nifreeze.estimator.Registration", DummyRegistration)

    estimator = PETMotionEstimator(model)
    affines = estimator.run(pet_obj)
    assert len(affines) == len(pet_obj)
    for mat in affines:
        np.testing.assert_array_equal(mat, np.eye(4))
