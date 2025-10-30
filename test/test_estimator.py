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

from typing import Union

import numpy as np
import pytest

import nifreeze.estimator
from nifreeze.data.base import BaseDataset
from nifreeze.data.dmri import DEFAULT_LOWB_THRESHOLD
from nifreeze.estimator import Estimator
from nifreeze.model.base import BaseModel
from nifreeze.utils import iterators

DATAOBJ_SIZE = (5, 5, 5, 4)


class DummyModel(BaseModel):
    def __init__(self, **kwargs):
        self._rng = np.random.default_rng(1234)
        super().__init__(**kwargs)

    def fit_predict(self, index: int | None = None, **kwargs):
        # Return a synthetic 3D image
        return self._rng.random(DATAOBJ_SIZE[:-1])


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


class DummyDWIDataset(BaseDataset):
    def __init__(self, dwi_dataobj, affine, brainmask_dataobj, b0_dataobj, gradients):
        self.dataobj = dwi_dataobj
        self.affine = affine
        self.brainmask = brainmask_dataobj
        self.bzero = b0_dataobj
        self.gradients = gradients

    def __len__(self):
        return self.dataobj.shape[-1]

    def __getitem__(self, idx):
        return self.dataobj[..., idx], self.brainmask, self.gradients


class DummyPETDataset(BaseDataset):
    def __init__(self, pet_dataobj, affine, brainmask_dataobj, midrame, total_duration):
        self.dataobj = pet_dataobj
        self.affine = affine
        self.brainmask = brainmask_dataobj
        self.midrame = midrame
        self.gradients = total_duration
        self.uptake = np.sum(pet_dataobj.reshape(-1, pet_dataobj.shape[-1]), axis=0)

    def __len__(self):
        return self.dataobj.shape[-1]

    def __getitem__(self, idx):
        return self.dataobj[..., idx], self.brainmask, self.midrame[idx]


def test_estimator_init_model_instance(request):
    rng = request.node.rng
    model = DummyModel(dataset=DummyDataset(rng))
    est = Estimator(model=model)
    assert isinstance(est._model, DummyModel)


def test_estimator_init_model_string(request, monkeypatch):
    rng = request.node.rng
    # Patch ModelFactory.init to return DummyModel
    monkeypatch.setattr(
        "nifreeze.model.base.ModelFactory.init",
        lambda model, dataset, **kwargs: DummyModel(dataset=dataset),
    )
    model_name = "dummy"
    est = Estimator(model=model_name, model_kwargs={})
    _dataset = DummyDataset(rng)
    # Should produce a DummyModel on run
    est.run(_dataset)
    assert isinstance(est._model, str)
    assert est._model == model_name


@pytest.mark.parametrize(
    "strategy, iterator_func, modality",
    [
        ("linear", iterators.linear_iterator, "dwi"),
        ("linear", iterators.linear_iterator, "pet"),
        ("random", iterators.random_iterator, "dwi"),
        ("random", iterators.random_iterator, "pet"),
        ("centralsym", iterators.centralsym_iterator, "dwi"),
        ("centralsym", iterators.centralsym_iterator, "pet"),
        ("bvalue", iterators.bvalue_iterator, "dwi"),
        ("uptake", iterators.uptake_iterator, "pet"),
    ],
)
def test_estimator_iterator_index_match(
    monkeypatch, setup_random_dwi_data, setup_random_pet_data, strategy, iterator_func, modality
):
    dataset: Union["DummyDWIDataset", "DummyPETDataset"]  # Avoids type annotation errors
    if modality == "dwi":
        (
            dwi_dataobj,
            affine,
            brainmask_dataobj,
            b0_dataobj,
            gradients,
            _,
        ) = setup_random_dwi_data

        dataset = DummyDWIDataset(dwi_dataobj, affine, brainmask_dataobj, b0_dataobj, gradients)
        bvals = gradients[-1, :][np.where(gradients[-1, :] > DEFAULT_LOWB_THRESHOLD)]
        kwargs = dict({"bvals": bvals})
    elif modality == "pet":
        (
            pet_dataobj,
            affine,
            brainmask_dataobj,
            midframe,
            total_duration,
        ) = setup_random_pet_data

        dataset = DummyPETDataset(pet_dataobj, affine, brainmask_dataobj, midframe, total_duration)
        uptake = dataset.uptake
        kwargs = dict({"uptake": uptake})
    else:
        raise NotImplementedError(f"{modality} not implemented")

    # Patch set_transform to record indices and matrices
    recorded_indices = []
    recorded_matrices = []

    # Make this accept `self` so it behaves as a proper instance method
    def fake_set_transform(self, i, xform):
        recorded_indices.append(i)
        recorded_matrices.append(xform)

    monkeypatch.setattr(type(dataset), "set_transform", fake_set_transform)

    # Patch registration to return identity matrix
    class DummyXForm:
        matrix = np.eye(4)

    nifreeze.estimator._run_registration = lambda *a, **k: DummyXForm()

    model = DummyModel(dataset=dataset)
    estimator = Estimator(model, strategy=strategy)
    estimator.run(dataset, **kwargs)

    n_vols = len(dataset)

    # Get expected indices
    if strategy == "linear":
        expected_indices = list(iterator_func(size=n_vols))
    elif strategy == "random":
        expected_indices = sorted(list(iterator_func(size=n_vols, seed=42)))
        recorded_indices_sorted = sorted(recorded_indices)
        assert recorded_indices_sorted == expected_indices
        return
    elif strategy == "centralsym":
        expected_indices = list(iterator_func(size=n_vols))
    elif strategy == "bvalue":
        expected_indices = list(iterator_func(bvals=bvals))
    elif strategy == "uptake":
        expected_indices = list(iterator_func(uptake=uptake))
    else:
        raise ValueError(f"Unknown strategy {strategy}")

    # Assert indices and matrices
    assert recorded_indices == expected_indices
    assert all(np.allclose(mat, np.eye(4)) for mat in recorded_matrices)
