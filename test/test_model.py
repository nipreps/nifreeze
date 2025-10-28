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
from typing import List

import numpy as np
import pytest
from dipy.sims.voxel import single_tensor

from nifreeze import model
from nifreeze.data.base import BaseDataset
from nifreeze.data.dmri import (
    DEFAULT_LOWB_THRESHOLD,
    DEFAULT_MAX_S0,
    DEFAULT_MIN_S0,
    DTI_MIN_ORIENTATIONS,
    DWI,
)
from nifreeze.model._dipy import GaussianProcessModel
from nifreeze.model.base import UNSUPPORTED_MODEL_ERROR_MSG, mask_absence_warn_msg
from nifreeze.testing import simulations as _sim


# Dummy classes to simulate model factory essential features
class DummyDMRIModel:
    def __init__(self, dataset, **kwargs):
        self._dataset = dataset
        self._kwargs = kwargs


class DummyPETModel:
    def __init__(self, dataset, **kwargs):
        self._dataset = dataset
        self._kwargs = kwargs


class DummyDataset:
    pass


def test_base_model():
    from nifreeze.model.base import BaseModel

    with pytest.raises(
        TypeError,
        match="Can't instantiate abstract class BaseModel without an implementation for abstract method 'fit_predict'",
    ):
        BaseModel(None)  # type: ignore[abstract]


@pytest.mark.parametrize("use_mask", (False, True))
def test_trivial_model(request, use_mask):
    """Check the implementation of the trivial B0 model."""
    from typing import Any

    rng = request.node.rng

    # Should not allow initialization without an oracle
    with pytest.raises(TypeError):
        model.TrivialModel()  # type: ignore[call-arg]

    size = (2, 2, 2)
    mask = None
    context: contextlib.AbstractContextManager[Any]
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
        dataobj=rng.normal(size=(*_S0.shape, 10)),
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


@pytest.mark.random_dwi_data(50, (14, 16, 8), True)
def test_dti_model(setup_random_dwi_data):
    (
        dwi_dataobj,
        affine,
        brainmask_dataobj,
        b0_dataobj,
        gradients,
        _,
    ) = setup_random_dwi_data

    dataset = DWI(
        dataobj=dwi_dataobj,
        affine=affine,
        brainmask=brainmask_dataobj,
        bzero=b0_dataobj,
        gradients=gradients,
    )

    dtimodel = model.DTIModel(dataset)
    predicted = dtimodel.fit_predict(4)
    assert predicted is not None
    assert predicted.shape == dwi_dataobj.shape[:-1]


def test_factory_none_raises(setup_random_base_data):
    dataobj, affine, brainmask, motion_affines, datahdr = setup_random_base_data
    dataset: BaseDataset = BaseDataset(
        dataobj=dataobj,
        affine=affine,
        brainmask=brainmask,
        motion_affines=motion_affines,
        datahdr=datahdr,
    )
    with pytest.raises(RuntimeError, match="No model identifier provided."):
        model.ModelFactory.init(None, dataset=dataset)


def test_model_factory_invalid_model():
    model_name = "not_a_model"
    with pytest.raises(
        NotImplementedError, match=UNSUPPORTED_MODEL_ERROR_MSG.format(model=model_name)
    ):
        model.ModelFactory.init(model_name, dataset=DummyDataset())


@pytest.mark.parametrize(
    "name, expected_cls",
    [
        ("avg", model.ExpectationModel),
        ("average", model.ExpectationModel),
        ("mean", model.ExpectationModel),
    ],
)
def test_factory_variants(name, expected_cls, setup_random_base_data):
    dataobj, affine, brainmask, motion_affines, datahdr = setup_random_base_data
    dataset: BaseDataset = BaseDataset(
        dataobj=dataobj,
        affine=affine,
        brainmask=brainmask,
        motion_affines=motion_affines,
        datahdr=datahdr,
    )
    model_instance = model.ModelFactory.init(name, dataset=dataset)
    assert isinstance(model_instance, expected_cls)


@pytest.mark.parametrize("name", ["avgdwi", "averagedwi", "meandwi"])
def test_factory_avgdwi_variants(monkeypatch, name, setup_random_dwi_data):
    (
        dwi_dataobj,
        affine,
        brainmask_dataobj,
        b0_dataobj,
        gradients,
        _,
    ) = setup_random_dwi_data

    dataset = DWI(
        dataobj=dwi_dataobj,
        affine=affine,
        brainmask=brainmask_dataobj,
        bzero=b0_dataobj,
        gradients=gradients,
    )

    # Dummy class to simulate AverageDWIModel
    class DummyAvgDWI:
        def __init__(self, _dataset, **kwargs):
            self._dataset = _dataset
            self._kwargs = kwargs

    # Patch import for AverageDWIModel
    import sys
    import types as _types

    old_module = sys.modules.get("nifreeze.model.dmri")
    dmri_module = _types.ModuleType("nifreeze.model.dmri")
    dmri_module.AverageDWIModel = DummyAvgDWI  # type: ignore[attr-defined]
    sys.modules["nifreeze.model.dmri"] = dmri_module

    try:
        model_instance = model.ModelFactory.init(name, dataset=dataset)
        assert isinstance(model_instance, DummyAvgDWI)
    finally:
        # Restore previous state
        if old_module is not None:
            sys.modules["nifreeze.model.dmri"] = old_module
        else:
            del sys.modules["nifreeze.model.dmri"]


@pytest.mark.parametrize(
    "model_name, expected_cls",
    [
        ("gqi", DummyDMRIModel),
        ("dti", DummyDMRIModel),
        ("DTI", DummyDMRIModel),
        ("dki", DummyDMRIModel),
        ("pet", DummyPETModel),
        ("PET", DummyPETModel),
    ],
)
def test_model_factory_valid_models(monkeypatch, model_name, expected_cls):
    # Track which module names were requested by the factory
    imported_modules: List[str] = []

    # Monkeypatch import_module to return a dummy module with DTIModel, DKIModel, etc.
    class DummyDMRI:
        DTIModel = DummyDMRIModel
        DKIModel = DummyDMRIModel
        GQIModel = DummyDMRIModel

    class DummyPET:
        # Use a distinct DummyPETModel so we can explicitly verify the factory
        # resolves to nifreeze.model.pet:PETModel (not to a dMRI model).
        PETModel = DummyPETModel

    def dummy_import_module(name):
        imported_modules.append(name)
        if name == "nifreeze.model.dmri":
            return DummyDMRI
        if name == "nifreeze.model.pet":
            return DummyPET
        raise ImportError(f"Unexpected import: {name}")

    monkeypatch.setattr("importlib.import_module", dummy_import_module)
    model_instance = model.ModelFactory.init(model_name, dataset=DummyDataset(), extra="value")
    assert model_instance.__class__ is expected_cls
    assert isinstance(model_instance._dataset, DummyDataset)
    assert model_instance._kwargs.get("extra") == "value"

    # Check the imported modules
    if model_name.lower() == "pet":
        assert "nifreeze.model.pet" in imported_modules, (
            "Factory should import 'nifreeze.model.pet' when model_name is 'pet'"
        )
        assert "nifreeze.model.dmri" not in imported_modules, (
            "Factory should not import 'nifreeze.model.dmri' when resolving PET models"
        )
    else:
        assert "nifreeze.model.dmri" in imported_modules, (
            "Factory should import 'nifreeze.model.dmri' for dMRI model names"
        )
        assert "nifreeze.model.pet" not in imported_modules, (
            "Factory should not import 'nifreeze.model.pet' when resolving dMRI models"
        )


def test_factory_initializations(datadir):
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
    model1 = model.AverageDWIModel(dmri_dataset, **modelargs)  # type: ignore[arg-type]

    # Initialisation via ModelFactory
    model2 = model.ModelFactory.init(model="avgdwi", dataset=dmri_dataset, **modelargs)

    assert model1._dataset == model2._dataset
    assert model1._detrend == model2._detrend
    assert model1._atol_low == model2._atol_low
    assert model1._atol_high == model2._atol_high
    assert model1._stat == model2._stat


def test_dmri_exceptions():
    import re

    class DummyDWI(DWI):
        def __init__(
            self, bzero=True, gradients=True, data_shape=(10, 10, 10, 35), brainmask=None
        ):
            if bzero:
                self.bzero = np.ones(data_shape[:3])
            if gradients:
                self.gradients = np.ones((data_shape[-1], 3))
            self.dataobj = np.ones(data_shape)
            self.brainmask = brainmask

        def __len__(self):
            if hasattr(self, "gradients") and self.gradients is not None:
                return self.gradients.shape[0]
            return 0

    dwi = DummyDWI(bzero=False)
    with pytest.raises(TypeError, match=model.dmri.DWI_OBJECT_ERROR_MSG):
        model.dmri.BaseDWIModel(dwi)

    dwi = DummyDWI(gradients=False)
    with pytest.raises(ValueError, match=model.dmri.DWI_GTAB_ERROR_MSG):
        model.dmri.BaseDWIModel(dwi)

    min_dir = DTI_MIN_ORIENTATIONS - 1
    dwi = DummyDWI(data_shape=(10, 10, 10, min_dir))
    with pytest.raises(
        ValueError, match=re.escape(model.dmri.DWI_SIZE_ERROR_MSG.format(directions=min_dir))
    ):
        model.dmri.BaseDWIModel(dwi)


def test_dmri_max_b_attribute():
    class DummyDWI(DWI):
        def __init__(self, data_shape=(10, 10, 10, 35)):
            self.bzero = np.ones(data_shape[:3])
            self.gradients = np.ones((data_shape[-1], 3))
            self.dataobj = np.ones(data_shape)
            self.brainmask = np.ones(data_shape[:-1]).astype(bool)

        def __len__(self):
            return self.gradients.shape[0]

    dwi = DummyDWI()
    max_b = DEFAULT_LOWB_THRESHOLD + 10
    dwi_base_model = model.dmri.BaseDWIModel(dwi, max_b=max_b)
    assert hasattr(dwi_base_model, "_max_b")
    assert dwi_base_model._max_b == max_b
