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
