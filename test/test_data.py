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

import json
import os
from typing import Optional

import h5py
import nibabel as nb
import numpy as np
import pytest

from nifreeze import data
from nifreeze.data import dmri, pet


def _raise_type(*args, **kwargs):
    raise TypeError("Cannot read")


def test_load_hdf5_error(monkeypatch, tmp_path):
    fname = tmp_path / ("dataset" + data.NFDH5_EXT)

    # All three dataclasses raise TypeError: load should raise TypeError
    monkeypatch.setattr(
        data.BaseDataset,
        "from_filename",
        classmethod(lambda _cls, fn: _raise_type()),
        raising=False,
    )
    monkeypatch.setattr(
        data.PET, "from_filename", classmethod(lambda _cls, fn: _raise_type()), raising=False
    )
    monkeypatch.setattr(
        data.DWI, "from_filename", classmethod(lambda _cls, fn: _raise_type()), raising=False
    )

    with pytest.raises(TypeError, match="Could not read data"):
        data.load(fname)


@pytest.mark.parametrize(
    "target, prior_failures",
    [
        (data.BaseDataset, []),
        (data.PET, [data.BaseDataset]),
        (data.DWI, [data.BaseDataset, data.PET]),
    ],
)
def test_load_hdf5_sentinel(monkeypatch, tmp_path, target, prior_failures):
    fname = tmp_path / ("dataset" + data.NFDH5_EXT)

    sentinel = object()

    # Force earlier readers to raise TypeError so load() will try the target
    for cls in prior_failures:
        monkeypatch.setattr(
            cls, "from_filename", classmethod(lambda _cls, fn: _raise_type()), raising=False
        )

    # Make the target reader return our sentinel
    monkeypatch.setattr(
        target, "from_filename", classmethod(lambda _cls, fn: sentinel), raising=False
    )

    assert data.load(fname) is sentinel


@pytest.mark.parametrize(
    "target, prior_failures, vol_size",
    [
        (data.BaseDataset, [], (4, 5, 6, 2)),
        (data.PET, [data.BaseDataset], (3, 4, 3, 5)),
        (data.DWI, [data.BaseDataset, data.PET], (2, 2, 6, 4)),
    ],
)
def test_load_hdf5_data(request, tmp_path, monkeypatch, target, prior_failures, vol_size):
    """
    For each target dataclass, write a tiny HDF5 file with random data, force
    earlier readers to raise TypeError, and monkeypatch the target's
    from_filename to read the HDF5 and return a small object containing the data
    so we can assert it was read.
    """

    rng = request.node.rng

    # Create random arrays to write into the HDF5 file
    _dataobj = rng.random(vol_size).astype(np.float32)
    _affine = np.eye(4).astype(np.float64)
    _brainmask_dataobj = rng.choice([True, False], size=_dataobj.shape[:3]).astype(np.uint8)

    fname = tmp_path / ("dataset" + data.NFDH5_EXT)

    # Write a minimal HDF5 layout that our patched reader will understand
    with h5py.File(fname, "w") as f:
        f.create_dataset("dataobj", data=_dataobj)
        f.create_dataset("affine", data=_affine)
        f.create_dataset("brainmask", data=_brainmask_dataobj)

    # Force earlier readers to raise TypeError so load() will try the target
    for cls in prior_failures:
        monkeypatch.setattr(
            cls, "from_filename", classmethod(lambda _cls, fn: _raise_type()), raising=False
        )

    # Define a reader that reads our HDF5 layout and returns a simple object
    def _from_filename(filename):
        # If called with a path, open the file; otherwise assume it's already
        # an h5py.File/group
        if isinstance(filename, (str, os.PathLike)):
            with h5py.File(filename, "r") as _f:
                _dtobj = np.array(_f["dataobj"])
                _aff = np.array(_f["affine"])
                _bnmsk = np.array(_f["brainmask"]).astype(bool)
        else:
            _f = filename
            _dtobj = np.array(_f["dataobj"])
            _aff = np.array(_f["affine"])
            _bnmsk = np.array(_f["brainmask"]).astype(bool)

        class SimpleBaseDataset:
            def __init__(
                self, dataobj: np.ndarray, affine: np.ndarray, brainmask: Optional[np.ndarray]
            ):
                self.dataobj = dataobj
                self.affine = affine
                self.brainmask = brainmask

        return SimpleBaseDataset(_dtobj, _aff, _bnmsk)

    # Patch the target class's from_filename to use our reader
    monkeypatch.setattr(
        target,
        "from_filename",
        classmethod(lambda _cls, fn: _from_filename(fn)),
        raising=False,
    )

    # Call the top-level loader and verify we got back the object with the data
    retval = data.load(fname)

    # The returned object should have the attributes we set above
    assert hasattr(retval, "dataobj")
    assert hasattr(retval, "affine")
    assert hasattr(retval, "brainmask")

    assert retval.dataobj is not None
    assert retval.dataobj.shape == _dataobj.shape
    assert np.allclose(retval.dataobj, _dataobj)

    assert retval.affine is not None
    assert retval.affine.shape == _affine.shape
    assert np.array_equal(retval.affine, _affine)

    assert retval.brainmask is not None
    assert retval.brainmask.shape == _brainmask_dataobj.shape
    assert np.array_equal(retval.brainmask, _brainmask_dataobj)


@pytest.mark.random_uniform_spatial_data((5, 2, 4), 0.0, 1.0)
@pytest.mark.random_uniform_spatial_data((2, 2, 2, 4), 0.0, 1.0)
@pytest.mark.parametrize(
    "use_brainmask, kwargs",
    [
        (True, {}),
        (False, {"data": 2.0}),
    ],
)
def test_load_base_nifti(
    request, tmp_path, monkeypatch, setup_random_uniform_spatial_data, use_brainmask, kwargs
):
    rng = request.node.rng
    dataobj, affine = setup_random_uniform_spatial_data
    img = nb.Nifti1Image(dataobj, affine)
    img_fname = tmp_path / "data.nii.gz"
    nb.save(img, img_fname)

    brainmask_dataobj = np.ones(dataobj.shape[:3]).astype(bool)
    if use_brainmask:
        brainmask_dataobj = rng.choice([True, False], size=dataobj.shape[:3]).astype(bool)

    brainmask = nb.Nifti1Image(brainmask_dataobj.astype(np.uint8), affine)
    brainmask_fname = tmp_path / "brainmask.nii.gz"
    nb.save(brainmask, brainmask_fname)

    # Monkeypatch BaseDataset to a minimal holder class that mirrors the API
    class SimpleBaseDataset:
        def __init__(self, **kwargs):
            self.dataobj = kwargs["dataobj"]
            self.affine = kwargs["affine"]
            self.brainmask = None

    monkeypatch.setattr(data, "BaseDataset", SimpleBaseDataset)

    retval = data.load(img_fname, brainmask_file=brainmask_fname, **kwargs)

    assert isinstance(retval, data.BaseDataset)

    assert hasattr(retval, "dataobj")
    assert hasattr(retval, "brainmask")
    assert hasattr(retval, "affine")

    assert retval.dataobj is not None
    assert np.allclose(retval.dataobj, dataobj)

    assert retval.affine is not None
    assert np.allclose(retval.affine, affine)

    assert retval.brainmask is not None
    assert np.array_equal(retval.brainmask, brainmask_dataobj)


def test_load_dmri_from_nii(monkeypatch, tmp_path):
    fname = tmp_path / "data.nii.gz"
    mask = tmp_path / "mask.nii.gz"

    # Create minimal valid NIfTI files, so data.load's file-existence check
    # passes
    size = (2, 2, 2, 4)
    affine = np.eye(4)
    img = nb.Nifti1Image(np.zeros(size, dtype=np.float32), affine)
    mask_img = nb.Nifti1Image(np.ones(size[:3], dtype=np.uint8), affine)
    nb.save(img, fname)
    nb.save(mask_img, mask)

    called = {}
    sentinel = object()

    def dummy_from_nii(filename, brainmask_file=None, **kwargs):
        called["filename"] = filename
        called["brainmask_file"] = brainmask_file
        called["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(dmri, "from_nii", dummy_from_nii)

    res = data.load(fname, brainmask_file=mask, gradients_file="grad.txt", bvec_file="bvecs.txt")

    assert res is sentinel
    assert called["filename"] == fname
    assert called["brainmask_file"] == mask
    assert "gradients_file" in called["kwargs"]
    assert (
        "bvec_file" in called["kwargs"]
        or "bvecs_file" in called["kwargs"]
        or "bvecs" in called["kwargs"]
    )


def test_load_pet_from_nii(monkeypatch, tmp_path):
    fname = tmp_path / "data.nii.gz"
    mask = tmp_path / "mask.nii.gz"

    # Create minimal valid NIfTI files, so data.load's file-existence check
    # passes
    size = (2, 2, 2, 4)
    affine = np.eye(4)
    img = nb.Nifti1Image(np.zeros(size, dtype=np.float32), affine)
    mask_img = nb.Nifti1Image(np.ones(size[:3], dtype=np.uint8), affine)
    nb.save(img, fname)
    nb.save(mask_img, mask)

    temporal_fname = tmp_path / "temporal.json"
    temporal_data = {"frame_time": np.ones(4).tolist()}
    with temporal_fname.open("w", encoding="utf-8") as f:
        json.dump(temporal_data, f, ensure_ascii=False, indent=2, sort_keys=True)

    called = {}
    sentinel = object()

    def dummy_from_nii(filename, brainmask_file=None, **kwargs):
        called["filename"] = filename
        called["brainmask_file"] = brainmask_file
        called["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(pet, "from_nii", dummy_from_nii)

    retval = data.load(fname, brainmask_file=mask, temporal_file=temporal_fname)

    assert retval is sentinel
    assert called["filename"] == fname
    assert called["brainmask_file"] == mask
    assert "temporal_file" in called["kwargs"]
