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
from nifreeze.testing.simulations import srtm


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
def test_petmodel_init_dataset_error(setup_random_pet_data, monkeypatch):
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

    model = BSplinePETModel(dataset=pet_obj)

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
    model = BSplinePETModel(pet_obj, start_index=2)

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
    shape = (1, 1, 1)
    n_timepoints = 16

    t = np.linspace(0, 2 * np.pi, n_timepoints, dtype="float32")
    temporal_basis = np.sin(t) + np.cos(2 * t)
    temporal_basis -= temporal_basis.min()

    dataobj = np.ones(shape + (n_timepoints,), dtype="float32")
    dataobj = dataobj * temporal_basis  # broadcasting

    midframe = np.arange(n_timepoints, dtype="float32")
    total_duration = float(n_timepoints)

    pet_obj = PET(
        dataobj=dataobj,
        affine=np.eye(4),
        brainmask=None,
        midframe=midframe,
        total_duration=total_duration,
    )

    model = BSplinePETModel(dataset=pet_obj, n_ctrl=5)

    predicted = np.stack([model.fit_predict(t_index) for t_index in range(n_timepoints)], axis=-1)

    correlations = np.array(
        [
            np.corrcoef(x, y)[0, 1]
            for x, y in zip(
                dataobj.reshape((-1, n_timepoints)),
                predicted.reshape((-1, n_timepoints)),
                strict=False,
            )
        ]
    )

    # original: (N, 16)
    # predicted_masked: (N, 16)

    # # Uncomment to plot prediction
    # import matplotlib

    # matplotlib.use("TkAgg")  # must be set BEFORE importing pyplot
    # import matplotlib.pyplot as plt
    # i = 0  # choose which timeseries/voxel to inspect
    # t = np.arange(n_timepoints)

    # fig, ax = plt.subplots()
    # ax.plot(t, dataobj.reshape((-1, n_timepoints))[i], label="original")
    # ax.plot(t, predicted.reshape((-1, n_timepoints))[i], label="predicted")
    # ax.set_xlabel("timepoint")
    # ax.set_ylabel("signal")
    # ax.set_title(f"series {i}   r={correlations[i]:.3f}")
    # ax.legend()

    # plt.show()
    # import pdb;pdb.set_trace()

    assert np.all(correlations > 0.95)


def _srtm_reference_inputs(n_timepoints: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create (t, dt, cr, cri) for SRTM simulation.
    t is mid-frame time, dt are frame durations, cr is reference TAC, cri is its integral.
    """
    # Simple constant framing (arbitrary units)
    dt = np.ones(n_timepoints, dtype="float32")
    t = np.cumsum(dt) - dt / 2.0

    # A smooth bolus-like reference TAC
    cr = (t**2) * np.exp(-0.35 * t)
    cr = cr / cr.max()

    # Cumulative trapezoidal integral of reference TAC
    cri = np.zeros_like(cr)
    cri[0] = cr[0] * dt[0]
    for i in range(1, n_timepoints):
        cri[i] = cri[i - 1] + 0.5 * (cr[i] + cr[i - 1]) * dt[i]

    return t.astype("float32"), dt, cr.astype("float32"), cri.astype("float32")


def test_petmodel_simulated_correlation_motion_free_srtm():
    # Same structure as the sinusoid-based test, but using SRTM temporal basis
    shape = (1, 1, 1)
    n_timepoints = 30

    t, dt, cr, cri = _srtm_reference_inputs(n_timepoints)

    # SRTM parameters: [R1, k2, BP]
    x = np.array([1.2, 0.15, 2.0], dtype="float32")

    CT, _DT = srtm(x=x, t=t, cr=cr, cri=cri, dt=dt, nr=n_timepoints)

    # Ensure non-negative (PET-like), and avoid a totally flat series
    CT = CT.astype("float32")
    CT -= CT.min()
    assert CT.max() > 0

    # Build synthetic 4D PET: one voxel following CT
    S0 = np.float32(100.0)
    temporal_basis = S0 * (CT / CT.max())

    dataobj = np.ones(shape + (n_timepoints,), dtype="float32") * temporal_basis

    # PET metadata
    midframe = t.astype("float32")
    total_duration = float(np.sum(dt))

    pet_obj = PET(
        dataobj=dataobj,
        affine=np.eye(4),
        brainmask=None,
        midframe=midframe,
        total_duration=total_duration,
    )

    # Fit/predict with spline PET model
    model = BSplinePETModel(dataset=pet_obj, n_ctrl=3)

    predicted = np.stack([model.fit_predict(t_index) for t_index in range(n_timepoints)], axis=-1)

    correlations = np.array(
        [
            np.corrcoef(x, y)[0, 1]
            for x, y in zip(
                dataobj.reshape((-1, n_timepoints)),
                predicted.reshape((-1, n_timepoints)),
                strict=False,
            )
        ]
    )

    assert np.all(correlations > 0.95)
