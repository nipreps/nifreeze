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


def test_petmodel_fit_predict():
    dataset = _create_dataset()
    model = PETModel(
        dataset=dataset,
        timepoints=dataset.midframe,
        xlim=dataset.total_duration,
        smooth_fwhm=0,
        thresh_pct=0,
    )

    # Fit on all data
    model.fit_predict(None)
    assert model.is_fitted

    # Predict at a specific timepoint
    vol = model.fit_predict(dataset.midframe[2])
    assert vol.shape == dataset.shape3d
    assert vol.dtype == dataset.dataobj.dtype


def test_petmodel_invalid_init():
    dataset = _create_dataset()
    with pytest.raises(TypeError):
        PETModel(dataset=dataset)


def test_petmodel_time_check():
    dataset = _create_dataset()
    bad_times = np.array([0, 10, 20, 30, 50], dtype=np.float32)
    with pytest.raises(ValueError):
        PETModel(dataset=dataset, timepoints=bad_times, xlim=60.0)
