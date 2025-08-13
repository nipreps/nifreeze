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
"""Unit tests exercising models."""

import contextlib

import numpy as np
import pytest
from dipy.sims.voxel import single_tensor

from nifreeze import model
from nifreeze.data.dmri import DEFAULT_MAX_S0, DEFAULT_MIN_S0, DWI
from nifreeze.model._dipy import GaussianProcessModel
from nifreeze.model.base import mask_absence_warn_msg
from nifreeze.testing import simulations as _sim


@pytest.mark.parametrize("use_mask", (False, True))
def test_trivial_model(request, use_mask):
    """Check the implementation of the trivial B0 model."""

    rng = request.node.rng

    # Should not allow initialization without an oracle
    with pytest.raises(TypeError):
        model.TrivialModel()

    size = (2, 2, 2)
    mask = None
    if use_mask:
        mask = np.ones(size, dtype=bool)
        context = contextlib.nullcontext()
    else:
        context = pytest.warns(UserWarning, match=mask_absence_warn_msg)

    _S0 = rng.normal(size=size)

    _clipped_S0 = np.clip(
        _S0.astype("float32") / _S0.max(),
        a_min=DEFAULT_MIN_S0,
        a_max=DEFAULT_MAX_S0,
    )

    data = DWI(
        dataobj=(*_S0.shape, 10),
        bzero=_clipped_S0,
        brainmask=mask,
    )

    with context:
        tmodel = model.TrivialModel(data)

    predicted = tmodel.fit_predict(4)

    assert np.all(_clipped_S0 == predicted)


def test_average_model():
    """Check the implementation of the average DW model."""

    gtab = np.array(
        [
            [0, 0, 0, 0],
            [-0.31, 0.933, 0.785, 25],
            [0.25, 0.565, 0.21, 500],
            [-0.861, -0.464, 0.564, 1000],
            [0.307, -0.766, 0.677, 1000],
            [0.736, 0.013, 0.774, 1000],
            [-0.31, 0.933, 0.785, 1000],
            [0.25, 0.565, 0.21, 2000],
            [-0.861, -0.464, 0.564, 2000],
            [0.307, -0.766, 0.677, 2000],
        ]
    )

    size = (100, 100, 100, gtab.shape[0])
    data = np.ones(size, dtype=float)
    mask = np.ones(size[:3], dtype=bool)

    data *= gtab[:, -1]
    dataset = DWI(dataobj=data, gradients=gtab, brainmask=mask)

    avgmodel_mean = model.AverageDWIModel(dataset, stat="mean")
    avgmodel_mean_full = model.AverageDWIModel(dataset, stat="mean", atol_low=2000, atol_high=2000)
    avgmodel_median = model.AverageDWIModel(dataset)

    # Verify that average cannot be calculated in shells with one single value
    with pytest.raises(RuntimeError):
        avgmodel_mean.fit_predict(2)

    assert np.allclose(avgmodel_mean.fit_predict(3), 1000)
    assert np.allclose(avgmodel_median.fit_predict(3), 1000)

    grads = list(gtab[:, -1])
    del grads[1]
    assert np.allclose(avgmodel_mean_full.fit_predict(1), np.mean(grads))

    avgmodel_mean_2000 = model.AverageDWIModel(dataset, stat="mean", atol_low=1100)
    avgmodel_median_2000 = model.AverageDWIModel(dataset, atol_low=1100)

    assert np.allclose(avgmodel_mean_2000.fit_predict(9), gtab[3:-1, -1].mean())
    assert np.allclose(avgmodel_median_2000.fit_predict(9), 1000)


@pytest.mark.parametrize(
    (
        "bval_shell",
        "S0",
        "evals",
    ),
    [
        (
            1000,
            100,
            (0.0015, 0.0003, 0.0003),
        )
    ],
)
@pytest.mark.parametrize("snr", (10, 20))
@pytest.mark.parametrize("hsph_dirs", (60, 30))
def test_gp_model(evals, S0, snr, hsph_dirs, bval_shell):
    # Simulate signal for a single tensor
    evecs = _sim.create_single_fiber_evecs()
    gtab = _sim.create_single_shell_gradient_table(hsph_dirs, bval_shell)
    signal = single_tensor(gtab, S0=S0, evals=evals, evecs=evecs, snr=snr)

    # Drop the initial b=0
    gtab = gtab[1:]
    data = signal[1:]

    gp = GaussianProcessModel(kernel_model="spherical")
    assert isinstance(gp, model._dipy.GaussianProcessModel)

    gpfit = gp.fit(data[:-2], gtab[:-2])
    prediction = gpfit.predict(gtab.bvecs[-2:])

    assert prediction.shape == (2,)


def test_factory(datadir):
    """Check that the two different initialisations result in the same models"""

    # Load test data
    dmri_dataset = DWI.from_filename(datadir / "dwi.h5")

    modelargs = {
        "atol_low": 25,
        "atol_high": 25,
        "detrend": True,
        "stat": "mean",
    }
    # Direct initialisation
    model1 = model.AverageDWIModel(dmri_dataset, **modelargs)

    # Initialisation via ModelFactory
    model2 = model.ModelFactory.init(model="avgdwi", dataset=dmri_dataset, **modelargs)

    assert model1._dataset == model2._dataset
    assert model1._detrend == model2._detrend
    assert model1._atol_low == model2._atol_low
    assert model1._atol_high == model2._atol_high
    assert model1._stat == model2._stat
