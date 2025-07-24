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

import types

import numpy as np

from nifreeze.data.pet import PET
from nifreeze.estimator import PETMotionEstimator


def _pet_dataset(n_frames=3):
    rng = np.random.default_rng(42)
    data = rng.random((2, 2, 2, n_frames), dtype=np.float32)
    affine = np.eye(4, dtype=np.float32)
    mask = np.ones((2, 2, 2), dtype=bool)
    midframe = np.arange(n_frames, dtype=np.float32) + 1
    return PET(
        dataobj=data,
        affine=affine,
        brainmask=mask,
        midframe=midframe,
        total_duration=float(n_frames + 1),
    )


def test_lofo_split_shapes(tmp_path):
    ds = _pet_dataset(4)
    idx = 2
    (train_data, train_times), (test_data, test_time) = ds.lofo_split(idx)
    assert train_data.shape[-1] == ds.dataobj.shape[-1] - 1
    np.testing.assert_array_equal(test_data, ds.dataobj[..., idx])
    np.testing.assert_array_equal(train_times, np.delete(ds.midframe, idx))
    assert test_time == ds.midframe[idx]


def test_to_from_filename_roundtrip(tmp_path):
    ds = _pet_dataset(3)
    out_file = tmp_path / "petdata"
    ds.to_filename(out_file)
    assert (tmp_path / "petdata.h5").exists()
    loaded = PET.from_filename(tmp_path / "petdata.h5")
    np.testing.assert_allclose(loaded.dataobj, ds.dataobj)
    np.testing.assert_allclose(loaded.affine, ds.affine)
    np.testing.assert_allclose(loaded.midframe, ds.midframe)
    assert loaded.total_duration == ds.total_duration


def test_pet_motion_estimator_run(monkeypatch):
    ds = _pet_dataset(3)

    class DummyModel:
        def __init__(self, dataset, timepoints, xlim):
            self.dataset = dataset

        def fit_predict(self, index):
            if index is None:
                return None
            return np.zeros(ds.shape3d, dtype=np.float32)

    monkeypatch.setattr("nifreeze.estimator.PETModel", DummyModel)

    class DummyRegistration:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, cwd=None):
            return types.SimpleNamespace(outputs=types.SimpleNamespace(forward_transforms=[]))

    monkeypatch.setattr("nifreeze.estimator.Registration", DummyRegistration)

    estimator = PETMotionEstimator(None)
    affines = estimator.run(ds)
    assert len(affines) == len(ds)
    for mat in affines:
        np.testing.assert_array_equal(mat, np.eye(4))
