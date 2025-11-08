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
    FIT_INDEX_OUT_OF_RANGE_ERROR_MSG,
    START_INDEX_RANGE_ERROR_MSG,
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
def test_petmodel_invalid_init(random_dataset):
    with pytest.raises(TypeError):
        PETModel(dataset=random_dataset)


@pytest.mark.random_pet_data(5, (4, 4, 4), np.asarray([10.0, 20.0, 30.0, 40.0, 50.0]), 60.0)
def test_petmodel_time_check(random_dataset):
    bad_times = np.array([0, 10, 20, 30, 50], dtype=np.float32)
    with pytest.raises(ValueError):
        PETModel(dataset=random_dataset, timepoints=bad_times, xlim=60.0)


def test_init_start_index_error():
    data = np.ones((1, 1, 1, 3), dtype=float)
    dataset = PET(data)
    timepoints = np.array([15.0, 45.0, 75.0], dtype=float)
    xlim = 100.0

    # Negative start_index raises ValueError
    with pytest.raises(ValueError, match=START_INDEX_RANGE_ERROR_MSG):
        PETModel(dataset, timepoints=timepoints, xlim=xlim, start_index=-1)

    # start_index equal to len(timepoints) is out of range
    with pytest.raises(ValueError, match=START_INDEX_RANGE_ERROR_MSG):
        PETModel(dataset, timepoints=timepoints, xlim=xlim, start_index=len(timepoints))


def test_fit_predict_index_error():
    data = np.ones((1, 1, 1, 3), dtype=float)
    dataset = PET(data)
    timepoints = np.array([15.0, 45.0, 75.0], dtype=float)
    xlim = 100.0

    model = PETModel(
        dataset,
        timepoints=timepoints,
        xlim=xlim,
        smooth_fwhm=0.0,
        thresh_pct=0.0,
    )

    model.fit_predict(None)

    # Requesting an negative index should raise IndexError
    with pytest.raises(IndexError, match=FIT_INDEX_OUT_OF_RANGE_ERROR_MSG):
        model.fit_predict(index=-1)

    # Index equal to len(self._x) should also raise
    with pytest.raises(IndexError, match=FIT_INDEX_OUT_OF_RANGE_ERROR_MSG):
        model.fit_predict(index=len(timepoints))

    # Index greater than to len(self._x) should also raise
    with pytest.raises(IndexError, match=FIT_INDEX_OUT_OF_RANGE_ERROR_MSG):
        model.fit_predict(index=len(timepoints) + 1)


def test_petmodel_start_index_reuses_start_prediction():
    # Create a tiny 1-voxel 5-frame sequence with increasing signal
    data = np.arange(1.0, 6.0, dtype=float).reshape((1, 1, 1, 5))
    dataset = PET(data)

    # Timepoints in seconds (monotonic)
    timepoints = np.array([15.0, 45.0, 75.0, 105.0, 135.0], dtype=float)
    xlim = 150.0

    # Configure the model to start fitting at index=2 (timepoint 75s)
    model = PETModel(
        dataset,
        timepoints=timepoints,
        xlim=xlim,
        smooth_fwhm=0.0,  # disable smoothing for deterministic behaviour
        thresh_pct=0.0,  # disable thresholding
        start_index=2,
    )

    model.fit_predict(None)

    # Prediction for the configured start timepoint
    pred_start = model.fit_predict(index=timepoints[2])

    # Prediction for an earlier timepoint (should reuse start prediction)
    pred_early = model.fit_predict(index=timepoints[1])

    assert np.allclose(pred_start, pred_early), (
        "Earlier frames should reuse start-frame prediction"
    )

    # Prediction for a later timepoint should be allowed and may differ
    pred_late = model.fit_predict(index=timepoints[3])
    assert pred_late is not None

    assert pred_start.shape == data.shape[:3]
    assert pred_early.shape == data.shape[:3]
    assert pred_late.shape == data.shape[:3]
