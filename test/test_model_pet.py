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

import re
import sys

import numpy as np
import pytest

from nifreeze.data.pet import PET
from nifreeze.model.pet import (
    DEFAULT_TIMEPOINT_TOL,
    FIRST_TIMEPOINT_VALUE_ERROR_MSG,
    LAST_TIMEPOINT_CONSISTENCY_ERROR_MSG,
    TIMEPOINT_XLIM_DATA_MISSING_ERROR_MSG,
    PETModel,
)


@pytest.mark.random_pet_data(5, (4, 4, 4), np.asarray([1.0, 2.0, 3.0, 4.0, 5.0]))
@pytest.mark.parametrize(
    "none_params", [("timepoints",), ("xlim",), ("timepoints=timepoints", "xlim")]
)
def test_petmodel_init_parameters_error(request, setup_random_pet_data, none_params):
    rng = request.node.rng
    pet_dataobj, affine, brainmask_dataobj, _, midframe, total_duration = setup_random_pet_data

    pet_obj = PET(
        dataobj=pet_dataobj,
        affine=affine,
        brainmask=brainmask_dataobj,
        midframe=midframe,
        total_duration=total_duration,
    )

    timepoints = rng.random(len(pet_obj)) if "timepoints" in none_params else None
    xlim = rng.random(1).item() if "xlim" in none_params else None

    with pytest.raises(
        ValueError,
        match=re.escape(
            TIMEPOINT_XLIM_DATA_MISSING_ERROR_MSG.format(timepoints=timepoints, xlim=xlim)
        ),
    ):
        PETModel(dataset=pet_obj, timepoints=timepoints, xlim=xlim)  # type: ignore[arg-type]


@pytest.mark.random_pet_data(5, (4, 4, 4), np.asarray([1.0, 2.0, 3.0, 4.0, 5.0]))
def test_petmodel_init_timepoint_value_error(request, setup_random_pet_data):
    rng = request.node.rng
    pet_dataobj, affine, brainmask_dataobj, _, midframe, total_duration = setup_random_pet_data

    pet_obj = PET(
        dataobj=pet_dataobj,
        affine=affine,
        brainmask=brainmask_dataobj,
        midframe=midframe,
        total_duration=total_duration,
    )

    timepoints = rng.random(len(pet_obj))
    xlim = rng.random(1).item()

    timepoints[0] = DEFAULT_TIMEPOINT_TOL - sys.float_info.epsilon

    with pytest.raises(
        ValueError, match=FIRST_TIMEPOINT_VALUE_ERROR_MSG.format(timepoints=timepoints)
    ):
        PETModel(dataset=pet_obj, timepoints=timepoints, xlim=xlim)


@pytest.mark.random_pet_data(5, (4, 4, 4), np.asarray([1.0, 2.0, 3.0, 4.0, 5.0]))
def test_petmodel_parameter_consistency_error(request, setup_random_pet_data):
    rng = request.node.rng
    pet_dataobj, affine, brainmask_dataobj, _, midframe, total_duration = setup_random_pet_data

    pet_obj = PET(
        dataobj=pet_dataobj,
        affine=affine,
        brainmask=brainmask_dataobj,
        midframe=midframe,
        total_duration=total_duration,
    )

    xlim = rng.random(1).item()
    timepoints = np.ones(len(pet_obj)) * DEFAULT_TIMEPOINT_TOL
    timepoints[-1] = xlim - DEFAULT_TIMEPOINT_TOL + sys.float_info.epsilon

    with pytest.raises(
        ValueError,
        match=re.escape(
            LAST_TIMEPOINT_CONSISTENCY_ERROR_MSG.format(timepoints=timepoints, xlim=xlim)
        ),
    ):
        PETModel(dataset=pet_obj, timepoints=timepoints, xlim=xlim)


@pytest.mark.random_pet_data(5, (4, 4, 4), np.asarray([10.0, 20.0, 30.0, 40.0, 50.0]))
def test_petmodel_fit_predict(setup_random_pet_data):
    pet_dataobj, affine, brainmask_dataobj, _, midframe, total_duration = setup_random_pet_data

    pet_obj = PET(
        dataobj=pet_dataobj,
        affine=affine,
        brainmask=brainmask_dataobj,
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
