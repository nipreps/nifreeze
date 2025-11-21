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

import numpy as np
import pytest

from nifreeze.data.pet import PET
from nifreeze.model.pet import PETModel


@pytest.mark.random_pet_data(
    5,
    (4, 4, 4),
    np.asarray([10.0, 20.0, 30.0, 40.0, 50.0]),
    np.sum,
    np.asarray([3.0, 4.0, 5.0, 6.0, 7.0]),
    np.asarray([2.0, 17.0, 22.0, 32.0, 29.0]),
    47.0,
)
def test_petmodel_fit_predict(setup_random_pet_data):
    (
        pet_dataobj,
        affine,
        brainmask_dataobj,
        frame_time,
        uptake,
        frame_duration,
        midframe,
        total_duration,
    ) = setup_random_pet_data

    pet_obj = PET(
        dataobj=pet_dataobj,
        affine=affine,
        brainmask=brainmask_dataobj,
        frame_time=frame_time,
        uptake=uptake,
        frame_duration=frame_duration,
        midframe=midframe,
        total_duration=total_duration,
    )

    model = PETModel(
        dataset=pet_obj,
        timepoints=pet_obj.midframe,
        xlim=pet_obj.total_duration,
        smooth_fwhm=0,
        thresh_pct=0,
    )

    # Fit on all data
    model.fit_predict(None)
    assert model.is_fitted

    # Predict at a specific timepoint
    vol = model.fit_predict(pet_obj.midframe[2])
    assert vol is not None
    assert vol.shape == pet_obj.shape3d
    assert vol.dtype == pet_obj.dataobj.dtype


@pytest.mark.random_pet_data(
    5,
    (4, 4, 4),
    np.asarray([1.0, 2.0, 3.0, 4.0, 5.0]),
    np.sum,
    np.asarray([1.0, 1.0, 1.0, 1.0, 1.0]),
    np.asarray([0.5, 1.5, 2.5, 3.5, 4.5]),
    5.0,
)
def test_petmodel_invalid_init1(setup_random_pet_data):
    (
        pet_dataobj,
        affine,
        brainmask_dataobj,
        frame_time,
        uptake,
        frame_duration,
        midframe,
        total_duration,
    ) = setup_random_pet_data

    pet_obj = PET(
        dataobj=pet_dataobj,
        affine=affine,
        brainmask=brainmask_dataobj,
        frame_time=frame_time,
        uptake=uptake,
        frame_duration=frame_duration,
        midframe=midframe,
        total_duration=total_duration,
    )

    with pytest.raises(TypeError):
        PETModel(dataset=pet_obj)


@pytest.mark.random_pet_data(
    5,
    (4, 4, 4),
    np.asarray([10.0, 20.0, 30.0, 40.0, 50.0]),
    np.sum,
    np.asarray([5.0, 5.0, 5.0, 5.0, 5.0]),
    np.asarray([2.5, 7.5, 12.5, 17.5, 22.5]),
    25.0,
)
def test_petmodel_time_check(setup_random_pet_data):
    (
        pet_dataobj,
        affine,
        brainmask_dataobj,
        frame_time,
        uptake,
        frame_duration,
        midframe,
        total_duration,
    ) = setup_random_pet_data

    pet_obj = PET(
        dataobj=pet_dataobj,
        affine=affine,
        brainmask=brainmask_dataobj,
        frame_time=frame_time,
        uptake=uptake,
        frame_duration=frame_duration,
        midframe=midframe,
        total_duration=total_duration,
    )

    bad_times = np.array([0, 10, 20, 30, 50], dtype=np.float32)
    with pytest.raises(ValueError):
        PETModel(dataset=pet_obj, timepoints=bad_times, xlim=60.0)
