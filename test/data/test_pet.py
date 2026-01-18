# Copyright The NiPreps Developers <nipreps@gmail.com>
"""Additional PET dataset tests."""

from __future__ import annotations

import json
from types import SimpleNamespace

import h5py
import nibabel as nb
import numpy as np
import pytest
from nitransforms.linear import Affine

from nifreeze.data.pet import PET, _compute_frame_duration


def _make_pet(shape=(2, 2, 2, 3)) -> PET:
    data = np.ones(shape, dtype=np.float32)
    midframe = np.arange(shape[-1], dtype=np.float32)
    return PET(dataobj=data, affine=np.eye(4), midframe=midframe, total_duration=float(shape[-1]))


def test_lofo_split_creates_hdf5(tmp_path):
    pet = _make_pet()
    pet._filepath = tmp_path / "lofo_pet.h5"
    assert not pet._filepath.exists()

    (train_data, train_time), (test_data, test_time) = pet.lofo_split(1)

    assert pet._filepath.exists()
    assert train_data.shape[-1] == pet.dataobj.shape[-1] - 1
    np.testing.assert_array_equal(train_time, [0, 2])
    np.testing.assert_array_equal(test_data, pet.dataobj[..., 1])
    assert test_time == pet.midframe[1]


def test_pet_set_transform_updates_and_types(tmp_path, monkeypatch):
    pet = _make_pet()
    pet._filepath = tmp_path / "transform_pet.h5"

    def _fake_apply(xform, img, order):
        return SimpleNamespace(dataobj=np.full(img.shape, 7, dtype=np.float32))

    monkeypatch.setattr("nifreeze.data.pet.apply", _fake_apply)

    pet.set_transform(2, np.eye(4))
    np.testing.assert_array_equal(pet.dataobj[..., 2], 7)
    assert pet.motion_affines is not None
    assert isinstance(pet.motion_affines[2], Affine)


def test_from_filename_and_scalar_cast(tmp_path):
    pet = _make_pet()
    pet.total_duration = np.float32(5.5)
    pet.to_filename(tmp_path / "pet_data.h5")

    loaded = PET.from_filename(tmp_path / "pet_data.h5", keep_file_open=True)
    assert isinstance(loaded.dataobj, h5py.Dataset)
    assert isinstance(loaded.total_duration, float)
    assert loaded._file_handle is not None
    loaded.close()


def test_load_from_nifti_and_hdf5(tmp_path):
    data = np.stack([np.full((2, 2, 2), val, dtype=np.float32) for val in (1, 2, 3)], axis=-1)
    affine = np.eye(4)
    img = nb.Nifti1Image(data, affine)
    nii_path = tmp_path / "pet.nii.gz"
    img.to_filename(nii_path)

    metadata = {"FrameDuration": [1.0, 1.0, 2.0], "FrameTimesStart": [0.0, 1.0, 2.0]}
    json_path = tmp_path / "pet.json"
    json_path.write_text(json.dumps(metadata))

    brainmask = nb.Nifti1Image(np.ones(data.shape[:3], dtype=np.uint8), affine)
    mask_path = tmp_path / "mask.nii.gz"
    brainmask.to_filename(mask_path)

    loaded = PET.load(nii_path, json_path, brainmask_file=mask_path)
    np.testing.assert_array_equal(loaded.dataobj, data)
    np.testing.assert_array_equal(loaded.midframe, np.array([0.5, 1.5, 3.0]))
    assert loaded.total_duration == pytest.approx(4.0)
    np.testing.assert_array_equal(loaded.brainmask, np.ones(data.shape[:3], dtype=np.uint8))

    pet_h5 = tmp_path / "pet_data.h5"
    _make_pet().to_filename(pet_h5)
    loaded_h5 = PET.load(pet_h5, json_path)
    assert isinstance(loaded_h5, PET)

    durations = _compute_frame_duration(np.array([1.0, 3.0, 8.0], dtype=np.float32))
    np.testing.assert_array_equal(durations, [2.0, 5.0, 5.0])


def test_keep_file_open_scalar_conversion(tmp_path):
    pet = _make_pet()
    pet.total_duration = np.float32(3.25)
    out_file = tmp_path / "keep_open.h5"
    pet.to_filename(out_file)

    loaded = PET.from_filename(out_file, keep_file_open=True)
    assert isinstance(loaded.total_duration, float)
    assert loaded._file_handle is not None
    loaded.close()
