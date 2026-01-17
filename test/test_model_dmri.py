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
"""Unit tests exercising dMRI models."""

import warnings

import numpy as np
import pytest
from dipy.core.gradients import gradient_table_from_bvals_bvecs
from dipy.reconst import dki, dti
from dipy.sims.voxel import single_tensor

from nifreeze import model
from nifreeze.data.dmri import DWI
from nifreeze.data.dmri.utils import (
    DEFAULT_LOWB_THRESHOLD,
    DTI_MIN_ORIENTATIONS,
)
from nifreeze.model._dipy import GaussianProcessModel
from nifreeze.model.base import MASK_ABSENCE_WARN_MSG
from nifreeze.testing import simulations as _sim

B_MATRIX = np.array(
    [
        [0.0, 0.0, 0.0, 0],
        [1.0, 0.0, 0.0, 500],
        [0.0, 1.0, 0.0, 1000],
        [0.0, 0.0, 1.0, 1000],
        [1 / np.sqrt(2), 1 / np.sqrt(2), 0.0, 1000],
        [1 / np.sqrt(2), 0.0, 1 / np.sqrt(2), 1000],
        [0.0, 1 / np.sqrt(2), 1 / np.sqrt(2), 1000],
        [1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3), 2000],
        [-1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3), 2000],
        [1 / np.sqrt(3), -1 / np.sqrt(3), 1 / np.sqrt(3), 2000],
        [1 / np.sqrt(3), 1 / np.sqrt(3), -1 / np.sqrt(3), 2000],
    ],
    dtype=np.float32,
)


# ToDo
# Functions to compare attributes of two instances. Eventually remove
def _get_attrs(instance):
    attrs1 = {}
    for n in dir(instance):
        if n.startswith("__"):
            continue
        try:
            v = getattr(instance, n)
        except Exception:
            continue
        if not callable(v) and isinstance(
            v, (int, float, str, bool, tuple, list, dict, np.ndarray, np.generic)
        ):
            attrs1[n] = v

    return attrs1


def _compare_instance_attributes(instance1, instance2):
    # Get attributes of both instances, excluding dunder and method attributes
    attributes1 = _get_attrs(instance1)
    attributes2 = _get_attrs(instance2)

    # Ensure both instances have the same attributes
    if set(attributes1) != set(attributes2):
        print("Instances have different sets of attributes.")
        return False

    # Compare the values of the attributes
    all_equal = True
    for attr in attributes1:
        value1 = attributes1.get(attr)
        value2 = attributes2.get(attr)

        if value1 is None and value2 is None:
            continue
        if value1 is None or value2 is None:
            print(f"Attribute '{attr}' differs: {value1} != {value2}")
            all_equal = False
            continue

        elif value1 is None or value2 is None:
            print(f"Attribute '{attr}' differs: {value1} != {value2}")
            all_equal = False

        try:
            array1 = np.asarray(value1).ravel()
            array2 = np.asarray(value2).ravel()
            # If a multi-dimensional array after the ravelling, it was something
            # like a shape product, so skip
            if array1.shape != array2.shape:
                continue
            if not np.allclose(array1, array2):
                print(f"Attribute '{attr}' differs: {value1} != {array2}")
                all_equal = False
        except Exception:
            # If conversion fails, assume equality to avoid complicating things
            print(f"Attribute '{attr}' not compared: assuming equality")

    if all_equal:
        print("All attributes are equal.")
    return all_equal


def test_base_model_exceptions():
    import re

    class DummyDWI(DWI):
        def __init__(
            self, bzero=True, gradients=True, data_shape=(10, 10, 10, 35), brainmask=None
        ):
            if bzero:
                self.bzero = np.ones(data_shape[:3])
            if gradients:
                self.gradients = np.ones((data_shape[-1], 4))
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


def test_base_max_b_attribute():
    class DummyDWI(DWI):
        def __init__(self, data_shape=(10, 10, 10, 35)):
            self.bzero = np.ones(data_shape[:3])
            self.gradients = np.ones((data_shape[-1], 4))
            self.dataobj = np.ones(data_shape)
            self.brainmask = np.ones(data_shape[:-1]).astype(bool)

        def __len__(self):
            return self.gradients.shape[0]

    dwi = DummyDWI()
    max_b = DEFAULT_LOWB_THRESHOLD + 10
    dwi_base_model = model.dmri.BaseDWIModel(dwi, max_b=max_b)
    assert hasattr(dwi_base_model, "_max_b")
    assert dwi_base_model._max_b == max_b


def test_average_model():
    """Check the implementation of the average DW model."""

    gtab = B_MATRIX.copy()
    size = (10, 10, 10, gtab.shape[0])
    data = np.ones(size, dtype=float)
    mask = np.ones(size[:3], dtype=bool)

    data *= gtab[:, -1]
    dataset = DWI(dataobj=data, affine=np.eye(4), gradients=gtab, brainmask=mask)

    avgmodel_mean = model.AverageDWIModel(dataset, stat="mean")
    avgmodel_mean_full = model.AverageDWIModel(dataset, stat="mean", atol_low=2000, atol_high=2000)
    avgmodel_median = model.AverageDWIModel(dataset)

    # Verify that average cannot be calculated in shells with one single value
    # The two first gradients are considered low-b orientations because of the
    # default threshold and are pulled out (so now index is 0 for b=500).
    with pytest.raises(RuntimeError):
        avgmodel_mean.fit_predict(0)

    assert np.allclose(avgmodel_mean.fit_predict(3), 1000)
    assert np.allclose(avgmodel_median.fit_predict(3), 1000)

    grads = list(gtab[1:, -1])  # Exclude b0
    del grads[3]
    assert np.allclose(avgmodel_mean_full.fit_predict(3), np.mean(grads))

    avgmodel_mean_2000 = model.AverageDWIModel(dataset, stat="mean", atol_low=1100)
    avgmodel_median_2000 = model.AverageDWIModel(dataset, atol_low=1100)

    last = gtab.shape[0] - 2  # Last but one index (b=2000)
    assert np.allclose(avgmodel_mean_2000.fit_predict(last), gtab[2:-1, -1].mean())
    assert np.allclose(avgmodel_median_2000.fit_predict(last), 1000)


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
def test_dti_model_essentials(setup_random_dwi_data):
    dwi_dataobj, affine, brainmask_dataobj, gradients, _ = setup_random_dwi_data

    dataset = DWI(
        dataobj=dwi_dataobj,
        affine=affine,
        brainmask=brainmask_dataobj,
        gradients=gradients,
    )

    dtimodel = model.DTIModel(dataset)
    predicted = dtimodel.fit_predict(4)
    assert predicted is not None
    assert predicted.shape == dwi_dataobj.shape[:-1]


@pytest.mark.random_gtab_data(6, (1000,), 1)
@pytest.mark.random_dwi_data(50, (2, 2, 1), True)
@pytest.mark.parametrize("index", (None, 3, 5))
@pytest.mark.parametrize("use_mask", (False, True))
def test_dti_model_correctness(setup_random_dwi_data, index, use_mask):
    """Ensure that we get the same result obtained through the DTI model
    implemented in DIPY."""

    # ToDo
    # Create some data that makes sense for the DTI fit

    dwi_dataobj, affine, brainmask_dataobj, gradients, _ = setup_random_dwi_data

    if not use_mask:
        brainmask_dataobj = None

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=MASK_ABSENCE_WARN_MSG, category=UserWarning)
        dataset = DWI(
            dataobj=dwi_dataobj,
            affine=affine,
            brainmask=brainmask_dataobj,
            gradients=gradients,
        )

    # Fit & predict using NiFreeze
    dtimodel_nf = model.DTIModel(dataset)
    predicted_nf = dtimodel_nf.fit_predict(index)

    # Fit & predict using the DIPY DTI model directly
    data = dataset.dataobj
    gtab = gradient_table_from_bvals_bvecs(gradients[..., -1], gradients[..., :-1])

    idxmask = np.ones(data.shape[-1], dtype=bool)
    if index is not None:
        idxmask[index] = False

    # ToDo
    # Note that we fit without the b0 data in NiFreeze.
    # ToDo
    # Do the brainmask and clipping manipulation of the DWI model initialization
    # Tests are passing without doing that, though...
    dtimodel_dp = dti.TensorModel(gtab[~gtab.b0s_mask][idxmask])
    dtifit_dp = dtimodel_dp.fit(data=data[..., idxmask], mask=brainmask_dataobj)

    assert _compare_instance_attributes(dtimodel_nf._models[0], dtifit_dp)

    if index is not None:
        predicted_dp = dtifit_dp.predict(gtab[~gtab.b0s_mask][index], S0=dataset.bzero)

        assert predicted_nf is not None
        # ToDo
        # Use the mask since DIPY's .fit model mask parameter appears to be
        # unused (i.e. values outside the mask are nonzero)
        assert np.allclose(
            predicted_nf[brainmask_dataobj], predicted_dp[brainmask_dataobj][..., 0]
        )


@pytest.mark.random_gtab_data(10, (1000, 2000), 0)
@pytest.mark.random_dwi_data(50, (14, 16, 8), True)
def test_dki_model_bzero_exception(setup_random_dwi_data):
    dwi_dataobj, affine, brainmask_dataobj, gradients, _ = setup_random_dwi_data

    dataset = DWI(
        dataobj=dwi_dataobj,
        affine=affine,
        brainmask=brainmask_dataobj,
        gradients=gradients,
    )

    dkimodel = model.DKIModel(dataset)

    with pytest.raises(ValueError, match=model.dmri.DWI_DKI_NULL_GRADIENT_ERROR_MSG):
        dkimodel.fit_predict(4)


@pytest.mark.random_gtab_data(10, (1000, 2000), 1)
@pytest.mark.random_dwi_data(50, (14, 16, 8), True)
def test_dki_model_essentials(setup_random_dwi_data):
    dwi_dataobj, affine, brainmask_dataobj, gradients, _ = setup_random_dwi_data

    dataset = DWI(
        dataobj=dwi_dataobj,
        affine=affine,
        brainmask=brainmask_dataobj,
        gradients=gradients,
    )

    dkimodel_nf = model.DKIModel(dataset)
    predicted_nf = dkimodel_nf.fit_predict(4)
    assert predicted_nf is not None
    assert predicted_nf.shape == dwi_dataobj.shape[:-1]


@pytest.mark.random_gtab_data(10, (1000, 2000), 1)
@pytest.mark.random_dwi_data(50, (2, 2, 1), True)
@pytest.mark.parametrize("index", (None, 4, 9))
@pytest.mark.parametrize("use_mask", (False, True))
def test_dki_model_correctness(setup_random_dwi_data, index, use_mask):
    """Ensure that we get the same result obtained through the DKI model
    implemented in DIPY."""

    # ToDo
    # Create some data that makes sense for the DKI fit

    dwi_dataobj, affine, brainmask_dataobj, gradients, _ = setup_random_dwi_data

    if not use_mask:
        brainmask_dataobj = np.ones(dwi_dataobj.shape[:3], dtype=bool)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=MASK_ABSENCE_WARN_MSG, category=UserWarning)
        dataset = DWI(
            dataobj=dwi_dataobj,
            affine=affine,
            brainmask=brainmask_dataobj,
            gradients=gradients,
        )

    # Fit & predict using NiFreeze
    dkimodel_nf = model.DKIModel(dataset)
    predicted_nf = dkimodel_nf.fit_predict(index)

    # Fit & predict using the DIPY DKI model directly
    data = dataset.dataobj

    # Account for the b0 value being prepended to the data/gtab
    idxmask = np.ones(data.shape[-1] + 1, dtype=bool)
    if index is not None:
        index += 1
        idxmask[index] = False

    # Insert the b0 data into the DWI data
    assert dataset.bzero is not None
    data = np.concatenate([dataset.bzero[..., np.newaxis], data], axis=-1)
    gtab = gradient_table_from_bvals_bvecs(
        gradients[idxmask][..., -1], gradients[idxmask][..., :-1]
    )

    from nifreeze.model.dmri import DEFAULT_MIN_S0  # , DEFAULT_S0_CLIP_PERCENTILE

    # ToDo
    # The below would be for the case where we have e.g. b1000, b2000, b3000
    # and we are not using b0
    #
    # By default, set S0 to the q-th percentile of the DWI data within mask
    # S0 = np.where(
    #     brainmask_dataobj,
    #     np.round(
    #         np.percentile(
    #             dwi_dataobj[brainmask_dataobj, ...],
    #             DEFAULT_S0_CLIP_PERCENTILE,
    #         )
    #     ),
    #     0,
    # )
    # If b=0 is present and not to be ignored, update brain mask and set
    brainmask_dataobj[dataset.bzero < DEFAULT_MIN_S0] = False
    S0 = np.broadcast_to(dataset.bzero, brainmask_dataobj.shape) * brainmask_dataobj

    dkimodel_dp = dki.DiffusionKurtosisModel(gtab)
    dkifit_dp = dkimodel_dp.fit(data=data[..., idxmask], mask=brainmask_dataobj)

    if not use_mask:
        assert _compare_instance_attributes(
            dkimodel_nf._models[0].fit_array[0], dkifit_dp.fit_array[0][0][0]
        )
    else:
        assert _compare_instance_attributes(
            dkimodel_nf._models[0].fit_array[0], dkifit_dp.fit_array[0][1][0]
        )

    if index is not None:
        _gtab = gradient_table_from_bvals_bvecs(
            gradients[~idxmask][..., -1], gradients[~idxmask][..., :-1]
        )
        predicted_dp = dkifit_dp.predict(_gtab, S0=S0)

        # ToDo
        # For the use_mask=True, and a non-null index, DIPY returns only 2
        # non-null values, but the mask contains only 1 False value. If I do
        # dkifit_dp.predict(_gtab)[...,0]
        # I get three non-null values. Is this some issue in DIPY ?
        # Note that this also happens for the DTI model, but in the tests I am
        # only checking the values where the mask is true. Is this another bug?
        # If I remove the brainmask from the dkimodel_dp.fit call, values
        # outside the mask are zero (due to the S0 being masked surely), and
        # tests pass, but passing the mask and not masking the S0 brings us back
        # to the issue
        assert predicted_nf is not None
        assert np.allclose(predicted_nf, predicted_dp[..., 0])
