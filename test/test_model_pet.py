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
from nifreeze.model.pet import (
    DEFAULT_TIMEFRAME_MIDPOINT_TOL,
    FIRST_MIDPOINT_VALUE_ERROR_MSG,
    LAST_MIDPOINT_VALUE_ERROR_MSG,
    TIMEPOINT_XLIM_DATA_MISSING_ERROR_MSG,
    PETModel,
)


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


@pytest.mark.random_pet_data(5, (4, 4, 4), np.asarray([10.0, 20.0, 30.0, 40.0, 50.0]), 60.0)
def test_petmodel_fit_predict(random_dataset):
    model = PETModel(
        dataset=random_dataset,
        timepoints=random_dataset.midframe,
        xlim=random_dataset.total_duration,
        smooth_fwhm=0,
        thresh_pct=0,
    )

    # Fit on all data
    model.fit_predict(None)
    assert model.is_fitted

    # Predict at a specific timepoint
    vol = model.fit_predict(random_dataset.midframe[2])
    assert vol is not None
    assert vol.shape == random_dataset.shape3d
    assert vol.dtype == random_dataset.dataobj.dtype


@pytest.mark.random_pet_data(5, (4, 4, 4), np.asarray([10.0, 20.0, 30.0, 40.0, 50.0]), 60.0)
def test_petmodel_init_mandatory_attr_errors(random_dataset):
    with pytest.raises(TypeError, match=TIMEPOINT_XLIM_DATA_MISSING_ERROR_MSG):
        PETModel(dataset=random_dataset)

    xlim = 55.0
    with pytest.raises(TypeError, match=TIMEPOINT_XLIM_DATA_MISSING_ERROR_MSG):
        PETModel(dataset=random_dataset, xlim=xlim)

    timepoints = np.array([20, 30, 40, 50, 60], dtype=np.float32)
    with pytest.raises(TypeError, match=TIMEPOINT_XLIM_DATA_MISSING_ERROR_MSG):
        PETModel(dataset=random_dataset, timepoints=timepoints)


@pytest.mark.random_pet_data(5, (4, 4, 4), np.asarray([10.0, 20.0, 30.0, 40.0, 50.0]), 60.0)
def test_petmodel_first_midpoint_error(random_dataset):
    timepoints = np.array([0, 10, 20, 30, 50], dtype=np.float32)
    xlim = 60.0
    with pytest.raises(ValueError, match=FIRST_MIDPOINT_VALUE_ERROR_MSG):
        PETModel(dataset=random_dataset, timepoints=timepoints, xlim=xlim)

    timepoints[0] = DEFAULT_TIMEFRAME_MIDPOINT_TOL
    with pytest.raises(ValueError, match=FIRST_MIDPOINT_VALUE_ERROR_MSG):
        PETModel(dataset=random_dataset, timepoints=timepoints, xlim=xlim)


@pytest.mark.random_pet_data(5, (4, 4, 4), np.asarray([10.0, 20.0, 30.0, 40.0, 50.0]), 40.0)
def test_petmodel_last_midpoint_error(random_dataset):
    xlim = 45.0
    timepoints = np.array([5, 10, 20, 30, 50], dtype=np.float32)
    with pytest.raises(ValueError, match=LAST_MIDPOINT_VALUE_ERROR_MSG):
        PETModel(dataset=random_dataset, timepoints=timepoints, xlim=xlim)

    timepoints[-1] = xlim - DEFAULT_TIMEFRAME_MIDPOINT_TOL
    with pytest.raises(ValueError, match=LAST_MIDPOINT_VALUE_ERROR_MSG):
        PETModel(dataset=random_dataset, timepoints=timepoints, xlim=xlim)
