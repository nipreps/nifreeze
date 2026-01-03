# Copyright The NiPreps Developers <nipreps@gmail.com>
"""Unit tests for :mod:`nifreeze.data.base`."""

from __future__ import annotations

import re
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from nifreeze.data import NFDH5_EXT, BaseDataset
from nifreeze.data.base import (
    AFFINE_ABSENCE_ERROR_MSG,
    AFFINE_NDIM_ERROR_MSG,
    AFFINE_OBJECT_ERROR_MSG,
    AFFINE_SHAPE_ERROR_MSG,
    BRAINMASK_SHAPE_MISMATCH_ERROR_MSG,
    DATAOBJ_ABSENCE_ERROR_MSG,
    DATAOBJ_NDIM_ERROR_MSG,
    DATAOBJ_OBJECT_ERROR_MSG,
    validate_affine,
    validate_dataobj,
)


class DummyDataset(BaseDataset):
    """Simple subclass used to exercise equality semantics."""


def _build_dataset(shape: tuple[int, int, int, int]) -> BaseDataset:
    data = np.zeros(shape, dtype=np.float32)
    return BaseDataset(dataobj=data, affine=np.eye(4))


def test_validate_dataobj_affine_errors():
    with pytest.raises(ValueError, match=DATAOBJ_ABSENCE_ERROR_MSG):
        validate_dataobj(None, None, None)

    with pytest.raises(TypeError, match=DATAOBJ_OBJECT_ERROR_MSG):
        validate_dataobj(None, None, 5)

    with pytest.raises(ValueError, match=DATAOBJ_NDIM_ERROR_MSG):
        validate_dataobj(None, None, np.zeros((2, 2, 2)))

    with pytest.raises(ValueError, match=AFFINE_ABSENCE_ERROR_MSG):
        validate_affine(None, None, None)

    with pytest.raises(TypeError, match=AFFINE_OBJECT_ERROR_MSG):
        validate_affine(None, None, [[1, 0], [0, 1]])

    with pytest.raises(ValueError, match=AFFINE_NDIM_ERROR_MSG):
        validate_affine(None, None, np.ones((4,)))

    with pytest.raises(ValueError, match=re.escape(AFFINE_SHAPE_ERROR_MSG)):
        validate_affine(None, None, np.ones((3, 3)))


def test_post_init_brainmask_and_handles(tmp_path):
    data = np.zeros((2, 2, 2, 1), dtype=np.float32)
    brainmask = np.ones((2, 2, 3), dtype=bool)
    with pytest.raises(
        ValueError,
        match=re.escape(
            BRAINMASK_SHAPE_MISMATCH_ERROR_MSG.format(
                brainmask_shape=brainmask.shape, data_shape=data.shape[:3]
            )
        ),
    ):
        BaseDataset(dataobj=data, affine=np.eye(4), brainmask=brainmask)

    h5_path = tmp_path / "backing.h5"
    with h5py.File(h5_path, "w") as h5file:
        ds = h5file.create_dataset("data", data=data)
        dataset = BaseDataset(dataobj=ds, affine=np.eye(4))
        assert dataset._file_handle is not None
        dataset.close()
        assert dataset._file_handle is None

    mmap_path = tmp_path / "backing.dat"
    memmap = np.memmap(mmap_path, mode="w+", dtype=np.float32, shape=data.shape)
    dataset = BaseDataset(dataobj=memmap, affine=np.eye(4))
    assert dataset._mmap_path == mmap_path


def test_getitem_and_equality():
    dataset = _build_dataset((2, 2, 2, 2))
    affine_override = np.full((4, 4), 2.0)
    dataset.set_transform(1, affine_override)

    vol, affine = dataset[1]
    np.testing.assert_array_equal(vol, dataset.dataobj[..., 1])
    np.testing.assert_array_equal(affine, affine_override)

    other = DummyDataset(dataobj=dataset.dataobj, affine=dataset.affine)
    assert (dataset == other) is False


def test_hdf5_round_trip(tmp_path):
    dataset = _build_dataset((2, 2, 2, 2))
    out_path = tmp_path / "base_dataset"
    dataset.to_filename(out_path)

    h5_path = out_path.with_suffix(NFDH5_EXT)
    assert h5_path.exists()

    reopened = BaseDataset.from_filename(h5_path, keep_file_open=True)
    assert isinstance(reopened.dataobj, h5py.Dataset)
    assert reopened._file_handle is not None
    reopened.close()
    assert reopened._file_handle is None

    reopened_eager = BaseDataset.from_filename(h5_path, keep_file_open=False)
    assert isinstance(reopened_eager.dataobj, np.ndarray)
    assert reopened_eager._file_handle is None


def test_set_transform_initialization_and_to_nifti(tmp_path, monkeypatch):
    dataset = _build_dataset((2, 2, 2, 2))
    affine_override = np.diag([1, 2, 3, 1])
    dataset.set_transform(0, affine_override)
    assert dataset.motion_affines is not None
    np.testing.assert_array_equal(dataset.motion_affines[0], affine_override)
    np.testing.assert_array_equal(dataset.motion_affines[1], np.eye(4))

    captured = []

    def _fake_apply(xform, img, order):
        captured.append((xform, order))
        fill_value = len(captured)
        return SimpleNamespace(dataobj=np.full(img.shape, fill_value, dtype=np.float32))

    monkeypatch.setattr("nifreeze.data.base.apply", _fake_apply)

    out_file = tmp_path / "resampled.nii.gz"
    nii = BaseDataset.to_nifti(dataset, filename=out_file)
    assert captured and out_file.exists()
    np.testing.assert_array_equal(nii.get_fdata()[..., 0], 1)
    np.testing.assert_array_equal(nii.get_fdata()[..., 1], 2)

    with pytest.warns(UserWarning):
        BaseDataset.to_nifti(_build_dataset((1, 1, 1, 1)), write_hmxfms=True)
