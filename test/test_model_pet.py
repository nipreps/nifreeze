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

from nifreeze.data.base import BaseDataset
from nifreeze.data.pet import PET
from nifreeze.model.pet import (
    PET_MIDFRAME_ERROR_MSG,
    PET_OBJECT_ERROR_MSG,
    START_INDEX_RANGE_ERROR_MSG,
    BSplinePETModel,
)


def test_pet_base_model():
    from nifreeze.model.pet import BasePETModel

    if sys.version_info >= (3, 12):
        expected_message = re.escape(
            "Can't instantiate abstract class BasePETModel without an implementation "
            "for abstract method 'fit_predict'"
        )
    else:
        expected_message = (
            "Can't instantiate abstract class BasePETModel with abstract method fit_predict"
        )

    with pytest.raises(TypeError, match=expected_message):
        BasePETModel(None)  # type: ignore[abstract, arg-type]


@pytest.mark.random_pet_data(5, (4, 4, 4), np.asarray([1.0, 2.0, 3.0, 4.0, 5.0]))
def test_petmodel_init_dataset_error(request, setup_random_pet_data, monkeypatch):
    rng = request.node.rng
    pet_dataobj, _affine, brainmask_dataobj, _, midframe, total_duration = setup_random_pet_data

    # Create a dummy dataset class without attributes
    class AttributelessPETDataset(BaseDataset[np.ndarray]):
        def __init__(self, dataobj, affine, brainmask):
            self.dataobj = dataobj
            self.affine = affine
            self.brainmask = brainmask

    # Monkeypatch the PET dataset
    monkeypatch.setattr("nifreeze.data.pet.PET", AttributelessPETDataset)

    pet_obj_attless = AttributelessPETDataset(
        dataobj=pet_dataobj, affine=_affine, brainmask=brainmask_dataobj
    )

    with pytest.raises(TypeError, match=PET_OBJECT_ERROR_MSG):
        BSplinePETModel(dataset=pet_obj_attless)  # type:ignore[arg-type]

    # Create a dummy dataset class without total_duration data
    class MidframePETDataset(BaseDataset[np.ndarray]):
        def __init__(self, dataobj, affine, brainmask):
            self.dataobj = dataobj
            self.affine = affine
            self.brainmask = brainmask
            self.midframe = np.ones_like(dataobj.shape[-1])

    # Monkeypatch the PET dataset
    monkeypatch.setattr("nifreeze.data.pet.PET", MidframePETDataset)

    pet_obj_midf = MidframePETDataset(
        dataobj=pet_dataobj, affine=_affine, brainmask=brainmask_dataobj
    )

    with pytest.raises(TypeError, match=PET_OBJECT_ERROR_MSG):
        BSplinePETModel(dataset=pet_obj_midf)  # type:ignore[arg-type]

    # Create a dummy dataset class without midframe data
    class TotalDurationPETDataset(BaseDataset[np.ndarray]):
        def __init__(self, dataobj, affine, brainmask):
            self.dataobj = dataobj
            self.affine = affine
            self.brainmask = brainmask
            self.total_duration = np.ones_like(dataobj.shape[-1])

    # Monkeypatch the PET dataset
    monkeypatch.setattr("nifreeze.data.pet.PET", TotalDurationPETDataset)

    pet_obj_totald = TotalDurationPETDataset(
        dataobj=pet_dataobj, affine=_affine, brainmask=brainmask_dataobj
    )

    with pytest.raises(ValueError, match=PET_MIDFRAME_ERROR_MSG):
        BSplinePETModel(dataset=pet_obj_totald)  # type:ignore[arg-type]


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

    model = BSplinePETModel(dataset=pet_obj, smooth_fwhm=0, thresh_pct=0)

    # Fit on all data
    model.fit_predict(None)
    assert model.is_fitted

    # Predict at a specific timepoint
    index = 2
    vol = model.fit_predict(index)
    assert vol is not None
    assert vol.shape == pet_obj.shape3d
    assert vol.dtype == pet_obj.dataobj.dtype


@pytest.mark.random_pet_data(5, (4, 4, 4), np.asarray([10.0, 20.0, 30.0, 40.0, 50.0]))
def test_init_start_index_error(setup_random_pet_data):
    pet_dataobj, affine, brainmask_dataobj, _, midframe, total_duration = setup_random_pet_data

    pet_obj = PET(
        dataobj=pet_dataobj,
        affine=affine,
        brainmask=brainmask_dataobj,
        midframe=midframe,
        total_duration=total_duration,
    )

    # Negative start_index raises ValueError
    with pytest.raises(ValueError, match=START_INDEX_RANGE_ERROR_MSG):
        BSplinePETModel(pet_obj, start_index=-1)

    # start_index equal to len(timepoints) is out of range
    with pytest.raises(ValueError, match=START_INDEX_RANGE_ERROR_MSG):
        BSplinePETModel(pet_obj, start_index=len(pet_obj.midframe))


@pytest.mark.random_pet_data(5, (4, 4, 4), np.asarray([10.0, 20.0, 30.0, 40.0, 50.0]))
def test_petmodel_start_index_reuses_start_prediction(setup_random_pet_data):
    pet_dataobj, affine, brainmask_dataobj, _, midframe, total_duration = setup_random_pet_data

    pet_obj = PET(
        dataobj=pet_dataobj,
        affine=affine,
        brainmask=brainmask_dataobj,
        midframe=midframe,
        total_duration=total_duration,
    )

    # Configure the model to start fitting at index=2
    model = BSplinePETModel(
        pet_obj,
        smooth_fwhm=0.0,  # disable smoothing for deterministic behaviour
        thresh_pct=0.0,  # disable thresholding
        start_index=2,
    )

    model.fit_predict(None)

    # Prediction for the configured start timepoint
    pred_start = model.fit_predict(index=2)

    # Prediction for an earlier timepoint (should reuse start prediction)
    pred_early = model.fit_predict(index=1)

    assert pred_start is not None
    assert pred_early is not None
    assert np.allclose(pred_start, pred_early), (
        "Earlier frames should reuse start-frame prediction"
    )

    # Prediction for a later timepoint should be allowed and may differ
    pred_late = model.fit_predict(index=3)

    assert pred_late is not None
    assert pred_start.shape == pet_dataobj.shape[:3]
    assert pred_early.shape == pet_dataobj.shape[:3]
    assert pred_late.shape == pet_dataobj.shape[:3]


def test_petmodel_simulated_correlation_motion_free():
    shape = (8, 8, 8)
    n_timepoints = 16
    grid = np.meshgrid(
        np.linspace(0.5, 1.5, shape[0]),
        np.linspace(0.5, 1.5, shape[1]),
        np.linspace(0.5, 1.5, shape[2]),
        indexing="ij",
    )
    spatial_map = np.prod(grid, axis=0)

    t = np.linspace(0, 2 * np.pi, n_timepoints, dtype="float32")
    temporal_basis = 1.0 + 0.3 * np.sin(t) + 0.2 * np.cos(2 * t)

    dataobj = spatial_map[..., np.newaxis] * temporal_basis
    dataobj = dataobj.astype("float32")

    brainmask = spatial_map > 0.9
    midframe = np.arange(n_timepoints, dtype="float32")
    total_duration = float(n_timepoints)

    pet_obj = PET(
        dataobj=dataobj,
        affine=np.eye(4),
        brainmask=brainmask,
        midframe=midframe,
        total_duration=total_duration,
    )

    model = BSplinePETModel(dataset=pet_obj, smooth_fwhm=0, thresh_pct=0)
    model.fit_predict(None)

    predicted_volumes = [model.fit_predict(t_index) for t_index in range(n_timepoints)]
    predicted = np.stack(predicted_volumes, axis=-1)

    original = dataobj[brainmask]
    predicted_masked = predicted[brainmask]

    original_demean = original - original.mean(axis=1, keepdims=True)
    predicted_demean = predicted_masked - predicted_masked.mean(axis=1, keepdims=True)
    numerator = np.sum(original_demean * predicted_demean, axis=1)
    denominator = np.linalg.norm(original_demean, axis=1) * np.linalg.norm(
        predicted_demean, axis=1
    )
    correlations = numerator / denominator

    assert np.all(correlations > 0.95)
