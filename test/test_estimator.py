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

from nifreeze.data.base import BaseDataset
from nifreeze.estimator import Estimator
from nifreeze.model.base import BaseModel
from nifreeze.utils import iterators

DATAOBJ_SIZE = (5, 5, 5, 7)


class DummyInsiderModel(BaseModel):
    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, **kwargs)

    def fit_predict(self, index: int | None = None, **kwargs):
        # Return the indexed volume
        return self._dataset.dataobj[..., index]


class DummyDataset(BaseDataset):
    def __init__(self, rng):
        self.dataobj = rng.uniform(0.0, 1.0, DATAOBJ_SIZE)  # np.ones(DATAOBJ_SIZE)
        self.brainmask = rng.choice([True, False], size=self.dataobj.shape[:-1]).astype(bool)
        self.affine = np.eye(4)

    def __len__(self):
        return self.dataobj.shape[-1]

    def __getitem__(self, idx):
        # Return the indexed volume and a dummy value
        return self.dataobj[..., idx], None

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
        return self.dataobj[..., idx], None, self.gradients[idx, ...]


class DummyPETDataset(BaseDataset):
    def __init__(self, pet_dataobj, affine, brainmask_dataobj, midframe, total_duration):
        self.dataobj = pet_dataobj
        self.affine = affine
        self.brainmask = brainmask_dataobj
        self.midframe = midframe
        self.total_duration = total_duration

    def __len__(self):
        return self.dataobj.shape[-1]

    def __getitem__(self, idx):
        return self.dataobj[..., idx], None, self.midframe[idx]


def test_estimator_init_model_instance(request):
    model = DummyInsiderModel(dataset=DummyDataset(rng=request.node.rng))
    est = Estimator(model=model)
    assert isinstance(est._model, DummyInsiderModel)


def test_estimator_init_model_string(request, monkeypatch):
    # Patch ModelFactory.init to return DummyModel
    monkeypatch.setattr(
        "nifreeze.model.base.ModelFactory.init",
        lambda model, dataset, **kwargs: DummyInsiderModel(dataset=dataset),
    )

    def mock_iterator(*_, **kwargs):
        return []

    monkeypatch.setattr(iterators, "random_iterator", mock_iterator)  # Avoid iterator issues

    model_name = "dummy"
    est = Estimator(model=model_name, model_kwargs={})
    _dataset = DummyDataset(rng=request.node.rng)
    # Should produce a DummyModel on run
    est.run(_dataset)
    assert isinstance(est._model, str)
    assert est._model == model_name
