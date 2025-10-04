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

import nifreeze.estimator
from nifreeze.data.base import BaseDataset
from nifreeze.data.dmri import DEFAULT_LOWB_THRESHOLD
from nifreeze.estimator import Estimator
from nifreeze.model.base import BaseModel
from nifreeze.model.dmri import AverageDWIModel
from nifreeze.utils import iterators

DATAOBJ_SIZE = (5, 5, 5, 4)


class DummyModel(BaseModel):
    def fit_predict(self, idx=None, **kwargs):
        # Return a synthetic 3D image
        return np.ones(DATAOBJ_SIZE[:-1])


class DummyDataset(BaseDataset):
    def __init__(self, rng):
        self.dataobj = rng.uniform(0.0, 1.0, DATAOBJ_SIZE)
        self.brainmask = np.ones(self.dataobj.shape[:-1], dtype=bool)
        self.affine = np.eye(4)

    def __len__(self):
        return len(self.dataobj)

    def __getitem__(self, idx):
        # Return a valid 3D array and a dummy value
        return self.dataobj[idx], 1

    def set_transform(self, idx, matrix):
        pass


def test_estimator_init_model_instance(request):
    rng = request.node.rng
    model = DummyModel(dataset=DummyDataset(rng))
    est = Estimator(model=model)
    assert isinstance(est._model, DummyModel)


@pytest.mark.parametrize(
    "strategy, iterator_func",
    [
        ("linear", iterators.linear_iterator),
        ("random", iterators.random_iterator),
        ("bvalue", iterators.bvalue_iterator),
        ("centralsym", iterators.centralsym_iterator),
    ],
)
def test_estimator_iterator_index_match(setup_random_dwi_data, strategy, iterator_func):
    (
        dwi_dataobj,
        affine,
        brainmask_dataobj,
        b0_dataobj,
        gradients,
        _,
    ) = setup_random_dwi_data

    n_vols = dwi_dataobj.shape[-1]
    vol_shape = b0_dataobj.shape
    bvals = gradients[-1, :][np.where(gradients[-1, :] > DEFAULT_LOWB_THRESHOLD)]

    class DummyDWI:
        def __init__(self):
            self.dataobj = dwi_dataobj
            self.gradients = gradients
            self.bzero = b0_dataobj
            self.brainmask = brainmask_dataobj
            self.affine = affine

        def __len__(self):
            return self.dataobj.shape[-1]

        def __getitem__(self, idx):
            return self.dataobj[..., idx], self.brainmask, self.gradients

    # Patch set_transform to record indices and matrices
    recorded_indices = []
    recorded_matrices = []

    def fake_set_transform(i, xform):
        recorded_indices.append(i)
        recorded_matrices.append(xform)

    dwi_dataset = DummyDWI()
    dwi_dataset.set_transform = fake_set_transform

    # Patch registration to return identity matrix
    class DummyXForm:
        matrix = np.eye(4)

    nifreeze.estimator._run_registration = lambda *a, **k: DummyXForm()

    # Use a dummy model that returns ones
    class DummyModel(AverageDWIModel):
        def fit_predict(self, index, **kwargs):
            return np.ones(vol_shape)

    model = DummyModel(dwi_dataset)
    estimator = Estimator(model, strategy=strategy)
    estimator.run(dwi_dataset)

    # Get expected indices
    if strategy == "linear":
        expected_indices = list(iterator_func(size=n_vols))
    elif strategy == "random":
        expected_indices = sorted(list(iterator_func(size=n_vols, seed=42)))
        recorded_indices_sorted = sorted(recorded_indices)
        assert recorded_indices_sorted == expected_indices
        return
    elif strategy == "bvalue":
        expected_indices = list(iterator_func(bvals=bvals))
    elif strategy == "centralsym":
        expected_indices = list(iterator_func(size=n_vols))
    else:
        raise ValueError(f"Unknown strategy {strategy}")

    # Assert indices and matrices
    assert recorded_indices == expected_indices
    assert all(np.allclose(mat, np.eye(4)) for mat in recorded_matrices)
