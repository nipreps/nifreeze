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

import numpy as np
import pytest

from nifreeze.data.base import BaseDataset
from nifreeze.data.pet import PET
from nifreeze.model.pet import (
    PET_MIDFRAME_ERROR_MSG,
    PET_OBJECT_ERROR_MSG,
    BSplinePETModel,
)


def test_pet_base_model():
    from nifreeze.model.pet import BasePETModel

    with pytest.raises(
        TypeError,
        match=re.escape(
            "Can't instantiate abstract class BasePETModel without an implementation "
            "for abstract method 'fit_predict'"
        ),
    ):
        BasePETModel(None, xlim=None)  # type: ignore[abstract, arg-type]


@pytest.mark.random_pet_data(5, (4, 4, 4), np.asarray([1.0, 2.0, 3.0, 4.0, 5.0]))
def test_petmodel_init_dataset_error(request, setup_random_pet_data, monkeypatch):
    rng = request.node.rng
    pet_dataobj, _affine, brainmask_dataobj, _, midframe, total_duration = setup_random_pet_data

    xlim = rng.random(pet_dataobj.shape[-1])

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
        BSplinePETModel(dataset=pet_obj_attless, xlim=xlim)  # type:ignore[arg-type]

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
        BSplinePETModel(dataset=pet_obj_midf, xlim=xlim)  # type:ignore[arg-type]

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
        BSplinePETModel(dataset=pet_obj_totald, xlim=xlim)  # type:ignore[arg-type]


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
