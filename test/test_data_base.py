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
"""Test dataset base class."""

import itertools
import re
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import h5py
import nibabel as nb
import numpy as np
import pytest

from nifreeze.data import NFDH5_EXT, BaseDataset, load
from nifreeze.data.base import (
    AFFINE_ABSENCE_ERROR_MSG,
    AFFINE_NDIM_ERROR_MSG,
    AFFINE_OBJECT_ERROR_MSG,
    AFFINE_SHAPE_ERROR_MSG,
    BRAINMASK_SHAPE_MISMATCH_ERROR_MSG,
    DATAOBJ_ABSENCE_ERROR_MSG,
    DATAOBJ_NDIM_ERROR_MSG,
    DATAOBJ_OBJECT_ERROR_MSG,
    _has_dim_size,
    _has_ndim,
    to_nifti,
)
from nifreeze.utils.ndimage import get_data, load_api

DEFAULT_RANDOM_DATASET_SHAPE = (32, 32, 32, 5)
DEFAULT_RANDOM_DATASET_SIZE = int(np.prod(DEFAULT_RANDOM_DATASET_SHAPE[:3]))


# Dummy transform classes and functions to monkeypatch into the real module
class DummyTransform:
    def __init__(self, idx):
        self.idx = idx

    def to_filename(self, path):
        # Create a tiny marker file so tests can check it was written and its
        # contents
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"transform-{self.idx}")


class DummyLinearTransformsMapping:
    """A class that mimics the iterable mapping of linear transforms.

    Yields DummyTransform instances when iterated. Also supports len()
    and indexing to be interchangeable with sequence-like expectations.
    """

    def __init__(self, transforms, reference=None):
        # Determine number of transforms in an explicit, non-broad-except way.
        if transforms is None:
            n = 0
        elif hasattr(transforms, "__len__"):
            # len() may raise TypeError for objects that do not support it
            try:
                n = len(transforms)
            except TypeError:
                # Treat non-sized objects as zero-length for this test helper
                n = 0
        else:
            # Explicitly raise for unexpected types to fail fast and be explicit
            raise TypeError("transforms must be a sequence or None")

        # Return one DummyTransform per motion_affine
        self._xforms = [DummyTransform(i) for i in range(n)]

    def __iter__(self):
        return iter(self._xforms)

    def __len__(self):
        return len(self._xforms)

    def __getitem__(self, idx):
        return self._xforms[idx]


def dummy_apply(transforms, spatialimage, order=3):
    """A deterministic 'resampling' that modifies the data so tests can verify
    that apply() and the transforms mapping were actually used.

    It returns a new Nifti1Image whose data is the original frame plus
    the transform index (transforms.idx). This makes each frame distinct and
    easily predictable.
    """
    data = np.asanyarray(spatialimage.dataobj).copy()
    # Mutate in a simple, deterministic way that depends on transform index
    data = data + int(getattr(transforms, "idx", 0))
    return nb.Nifti1Image(data, spatialimage.affine, spatialimage.header)


class DummyImageGrid:
    def __init__(self, shape, affine):
        self.shape = shape
        self.affine = affine


class DummyDataset(BaseDataset):
    def __init__(self, shape, datahdr=None, motion_affines=None, dtype=np.int16):
        self.dataobj = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
        self.affine = np.eye(4)
        self.datahdr = datahdr
        self.motion_affines = motion_affines

    def __getitem__(self, idx):
        # to_nifti expects dataset[idx] to return a sequence whose first item is the 3D frame
        return (self.dataobj[..., idx],)


@pytest.mark.parametrize(
    "setup_random_uniform_spatial_data",
    [
        (DEFAULT_RANDOM_DATASET_SHAPE, 0.0, 1.0),
    ],
)
@pytest.fixture
def random_dataset(setup_random_uniform_spatial_data) -> BaseDataset:
    """Create a BaseDataset with random data for testing."""

    data, affine = setup_random_uniform_spatial_data
    return BaseDataset(dataobj=data, affine=affine)


@pytest.mark.parametrize(
    "value, size, expected",
    [
        (np.zeros((2, 4, 5)), 4, True),
        (np.zeros((2, 4, 5)), 6, False),
        # Objects without .shape
        ([1, 2, 3], 3, False),
        # Shape that is not iterable
        (
            type("BadShape", (), {"shape": 5})(),
            5,
            False,
        ),
    ],
)
def test_has_dim_size(value, size, expected):
    assert _has_dim_size(value, size) is expected


@pytest.mark.parametrize(
    "obj_factory, ndim, expected",
    [
        (lambda: type("WithNdim", (), {"ndim": 2})(), 2, True),
        (lambda: type("WithNdim", (), {"ndim": 2})(), 3, False),
        (lambda: type("BadNdim", (), {"ndim": "not-an-int"})(), 2, False),
        (lambda: type("WithShape", (), {"shape": (3, 4)})(), 2, True),
        (lambda: (123), 1, False),  # No ndim or shape
    ],
)
def test_has_ndim(obj_factory, ndim, expected):
    obj = obj_factory()
    assert _has_ndim(obj, ndim) is expected


@pytest.mark.parametrize(
    "value, expected_exc, expected_msg",
    [
        (None, ValueError, DATAOBJ_ABSENCE_ERROR_MSG),
        (1, TypeError, DATAOBJ_OBJECT_ERROR_MSG),
    ],
)
def test_dataobj_basic_errors(value, expected_exc, expected_msg):
    with pytest.raises(expected_exc, match=str(expected_msg)):
        BaseDataset(dataobj=value)  # type: ignore[arg-type]


def test_memmap_dataobj_preserved():
    shape = (64, 64, 32, 4)
    with TemporaryDirectory() as tmpdir:
        memmap_path = Path(tmpdir) / "memmap_data.npy"
        memmap = np.lib.format.open_memmap(memmap_path, mode="w+", dtype=np.float32, shape=shape)
        memmap[0, 0, 0, 0] = 1.0
        memmap.flush()

        dataset: BaseDataset = BaseDataset(dataobj=memmap, affine=np.eye(4))

        assert isinstance(dataset.dataobj, np.memmap)

        frame, _, *_ = dataset[0]
        assert np.shares_memory(frame, dataset.dataobj)


@pytest.mark.random_uniform_spatial_data((2, 2, 2, 4, 6), 0.0, 1.0)
def test_dataobj_ndim_error(setup_random_uniform_spatial_data):
    data, _ = setup_random_uniform_spatial_data
    with pytest.raises(ValueError, match=DATAOBJ_NDIM_ERROR_MSG):
        BaseDataset(dataobj=data)


@pytest.mark.random_uniform_spatial_data((2, 2, 2, 4), 0.0, 1.0)
@pytest.mark.parametrize(
    "affine, expected_exc, expected_msg",
    [
        (None, ValueError, AFFINE_ABSENCE_ERROR_MSG),
        (1, TypeError, AFFINE_OBJECT_ERROR_MSG),
    ],
)
def test_missing_affine_error(
    setup_random_uniform_spatial_data, affine, expected_exc, expected_msg
):
    data, _ = setup_random_uniform_spatial_data
    with pytest.raises(expected_exc, match=str(expected_msg)):
        BaseDataset(dataobj=data, affine=affine)  # type: ignore[arg-type]


@pytest.mark.random_uniform_spatial_data((2, 2, 2, 4), 0.0, 1.0)
@pytest.mark.parametrize("size", ((2,), (3, 4, 2)))
def test_affine_ndim_error(setup_random_uniform_ndim_data, size):
    data = setup_random_uniform_ndim_data
    affine = np.ones(size)
    with pytest.raises(ValueError, match=re.escape(AFFINE_NDIM_ERROR_MSG)):
        BaseDataset(dataobj=data, affine=affine)


@pytest.mark.random_uniform_spatial_data((2, 2, 2, 4), 0.0, 1.0)
@pytest.mark.parametrize("size", ((2, 2), (3, 4), (4, 3), (5, 5)))
def test_affine_shape_error(setup_random_uniform_ndim_data, size):
    data = setup_random_uniform_ndim_data
    affine = np.ones(size)
    with pytest.raises(ValueError, match=re.escape(AFFINE_SHAPE_ERROR_MSG)):
        BaseDataset(dataobj=data, affine=affine)


@pytest.mark.random_uniform_spatial_data((2, 2, 2, 4), 0.0, 1.0)
def test_brainmask_volume_mismatch_error(request, setup_random_uniform_spatial_data):
    data, affine = setup_random_uniform_spatial_data
    data_shape = data.shape[:3]
    brainmask_size = tuple(map(lambda x: x + 1, data_shape))
    brainmask = request.node.rng.choice([True, False], size=brainmask_size)
    with pytest.raises(
        ValueError,
        match=re.escape(
            BRAINMASK_SHAPE_MISMATCH_ERROR_MSG.format(
                brainmask_shape=brainmask.shape, data_shape=data_shape
            )
        ),
    ):
        BaseDataset(dataobj=data, affine=affine, brainmask=brainmask)


def test_base_dataset_init(random_dataset: BaseDataset):
    """Test that the BaseDataset can be initialized with random data."""
    assert random_dataset.dataobj is not None
    assert random_dataset.affine is not None
    assert random_dataset.dataobj.shape == DEFAULT_RANDOM_DATASET_SHAPE
    assert random_dataset.affine.shape == (4, 4)
    assert random_dataset.size3d == DEFAULT_RANDOM_DATASET_SIZE
    assert random_dataset.shape3d == DEFAULT_RANDOM_DATASET_SHAPE[:3]


def test_len(random_dataset: BaseDataset):
    """Test that len(BaseDataset) returns the number of volumes."""
    assert len(random_dataset) == 5  # last dimension is 5 volumes


def test_getitem_volume_index(random_dataset: BaseDataset):
    """
    Test that __getitem__ returns the correct (volume, affine) tuple.

    By default, motion_affines is None, so we expect to get None for the affine.
    """
    # Single volume  # Note that the type ignore can be removed once we can use *Ts
    volume0, aff0 = random_dataset[0]  # type: ignore[misc]  # PY310
    assert volume0.shape == (32, 32, 32)
    # No transforms have been applied yet, so there's no motion_affines array
    assert aff0 is None

    # Slice of volumes
    volume_slice, aff_slice = random_dataset[2:4]  # type: ignore[misc]  # PY310
    assert volume_slice.shape == (32, 32, 32, 2)
    assert aff_slice is None


def test_set_transform(random_dataset: BaseDataset):
    """
    Test that calling set_transform changes the data and motion_affines.
    For simplicity, we'll apply an identity transform and check that motion_affines is updated.
    """
    idx = 0
    data_before = np.copy(random_dataset.dataobj[..., idx])
    # Identity transform
    affine = np.eye(4)
    random_dataset.set_transform(idx, affine)

    # Data shouldn't have changed (since transform is identity).
    volume0, aff0 = random_dataset[idx]  # type: ignore[misc]  # PY310
    assert np.allclose(data_before, volume0)

    # motion_affines should be created and match the transform matrix.
    assert random_dataset.motion_affines is not None
    np.testing.assert_array_equal(random_dataset.motion_affines[idx], affine)
    # The returned affine from __getitem__ should be the same.
    assert aff0 is not None
    np.testing.assert_array_equal(aff0, affine)


def test_to_filename_and_from_filename(random_dataset: BaseDataset):
    """Test writing a dataset to disk and reading it back from file."""
    with TemporaryDirectory() as tmpdir:
        h5_file = Path(tmpdir) / f"test_dataset{NFDH5_EXT}"
        random_dataset.to_filename(h5_file)

        # Check file exists
        assert h5_file.is_file()

        # Read from filename
        ds2: BaseDataset[Any] = BaseDataset.from_filename(h5_file)
        assert ds2.dataobj is not None
        assert ds2.dataobj.shape == (32, 32, 32, 5)
        assert ds2.affine.shape == (4, 4)
        # Ensure the data is the same
        assert np.allclose(random_dataset.dataobj, ds2.dataobj)


def test_from_filename_keep_file_open(random_dataset: BaseDataset, tmp_path: Path):
    h5_file = tmp_path / f"lazy_dataset{NFDH5_EXT}"
    random_dataset.to_filename(h5_file)

    dataset = BaseDataset.from_filename(h5_file, keep_file_open=True)

    try:
        assert isinstance(dataset.dataobj, h5py.Dataset)
        assert dataset._file_handle is not None
        assert dataset._file_handle.id.valid
        assert dataset.get_filename() == h5_file
    finally:
        dataset.close()


def test_hdf5_dataset_pins_file_handle(tmp_path: Path):
    h5_path = tmp_path / "backed.h5"
    h5_file = h5py.File(h5_path, "w")

    try:
        dset = h5_file.create_dataset("dataobj", data=np.zeros((2, 2, 2, 1), dtype=np.float32))
        dataset = BaseDataset(dataobj=dset, affine=np.eye(4))

        assert dataset._file_handle is not None
        assert dataset._file_handle.filename == h5_file.filename

        dataset.close()
        assert dataset._file_handle is None
        assert not h5_file.id.valid
    finally:
        if h5_file.id.valid:
            h5_file.close()


def test_object_to_nifti(random_dataset: BaseDataset):
    """Test writing a dataset to a NIfTI file."""
    with TemporaryDirectory() as tmpdir:
        nifti_file = Path(tmpdir) / "test_dataset.nii.gz"
        random_dataset.to_nifti(nifti_file)

        # Check file exists
        assert nifti_file.is_file()

        # Load the saved file with nibabel
        img = nb.Nifti1Image.from_filename(nifti_file)
        data = img.get_fdata(dtype=np.float32)
        assert data.shape == (32, 32, 32, 5)
        assert np.allclose(data, random_dataset.dataobj)


@pytest.mark.parametrize(
    "filename_is_none, motion_affines_present, write_hmxfms, expected_message",
    [
        # write_hmxfms True but no filename
        (True, True, True, "write_hmxfms is set to True, but no filename was provided."),
        # write_hmxfms True, filename given, but no motion affines
        (
            False,
            False,
            True,
            "write_hmxfms is set to True, but no motion affines were found. Skipping.",
        ),
    ],
)
def test_to_nifti_warnings(
    tmp_path, monkeypatch, filename_is_none, motion_affines_present, write_hmxfms, expected_message
):
    # Monkeypatch the helpers in the module where to_nifti is defined
    import nifreeze.data.base as base_mod

    monkeypatch.setattr(base_mod, "LinearTransformsMapping", DummyLinearTransformsMapping)
    monkeypatch.setattr(base_mod, "apply", dummy_apply)
    monkeypatch.setattr(base_mod, "ImageGrid", DummyImageGrid)

    n_frames = 3
    shape = (4, 4, 2, n_frames)

    motion_affines = [np.eye(4) for _ in range(n_frames)] if motion_affines_present else None

    dataset = DummyDataset(shape, datahdr=None, motion_affines=motion_affines)

    filename = None
    if not filename_is_none:
        filename = tmp_path / "data.nii.gz"

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = to_nifti(dataset, filename=filename, write_hmxfms=write_hmxfms, order=1)

    assert any(expected_message in str(x.message) for x in w)


@pytest.mark.parametrize(
    "filename_is_none, write_hmxfms, motion_affines_present, datahdr_present",
    list(itertools.product([True, False], repeat=4)),
)
def test_to_nifti(
    tmp_path, monkeypatch, filename_is_none, write_hmxfms, motion_affines_present, datahdr_present
):
    # Monkeypatch the helpers in the module where to_nifti is defined
    import nifreeze.data.base as base_mod

    monkeypatch.setattr(base_mod, "LinearTransformsMapping", DummyLinearTransformsMapping)
    monkeypatch.setattr(base_mod, "apply", dummy_apply)
    monkeypatch.setattr(base_mod, "ImageGrid", DummyImageGrid)

    n_frames = 3
    shape = (4, 4, 2, n_frames)
    dtype = np.int16

    datahdr = None
    if datahdr_present:
        hdr = nb.Nifti1Header()
        hdr.set_data_dtype(dtype)
        datahdr = hdr

    motion_affines = None
    if motion_affines_present:
        motion_affines = [np.eye(4) for _ in range(n_frames)]

    dataset = DummyDataset(shape, datahdr=datahdr, motion_affines=motion_affines, dtype=dtype)

    filename = None
    if not filename_is_none:
        filename = tmp_path / "data.nii.gz"

    # Suppress warnings in this test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nii = to_nifti(dataset, filename=filename, write_hmxfms=write_hmxfms, order=1)

    # Check returned data
    assert isinstance(nii, nb.Nifti1Image)
    assert nii.shape == dataset.dataobj.shape

    expected = dataset.dataobj.copy()
    if motion_affines_present:
        # If motion affines are present, fake_apply adds the frame index to each
        # frame
        for i in range(n_frames):
            expected[..., i] = expected[..., i] + i
        assert np.array_equal(nii.dataobj, expected), (
            "Resampled data should reflect fake_apply modifications"
        )
    else:
        # No resampling; data should be identical to original
        assert np.array_equal(nii.dataobj, dataset.dataobj)

    # Header behavior
    if datahdr_present:
        assert datahdr is not None
        assert nii.header.get_data_dtype() == datahdr.get_data_dtype()
    else:
        xyzt = nii.header.get_xyzt_units()
        assert xyzt[0].lower() == "mm"

    # If filename was provided, file should exist and equal to transformed data
    if filename is not None:
        assert filename.is_file()
        nii_load = load_api(filename, nb.Nifti1Image)
        assert np.array_equal(nii_load.get_fdata(), expected)
    else:
        assert not any(tmp_path.iterdir()), "Directory is not empty"

    # When motion_affines present and write_hmxfms True and filename provided,
    # x5 files should be written
    if motion_affines_present and write_hmxfms and filename is not None:
        # The same file is written at every iteration, so earlier transforms are
        # overwritten and only the last transform remains on disk
        found_x5 = list(tmp_path.glob("*.x5"))
        assert len(found_x5) == 1
        x5_path = filename.with_suffix("").with_suffix(".x5")
        assert x5_path.is_file()
        content = x5_path.read_text()
        assert content == f"transform-{n_frames - 1}"
    else:
        found_x5 = list(tmp_path.glob("*.x5"))
        assert len(found_x5) == 0


def test_load_hdf5(random_dataset: BaseDataset):
    """Test the 'load' function with an HDF5 file."""
    with TemporaryDirectory() as tmpdir:
        h5_file = Path(tmpdir) / f"test_dataset{NFDH5_EXT}"
        random_dataset.to_filename(h5_file)

        ds2 = load(h5_file)
        assert ds2.dataobj.shape == (32, 32, 32, 5)
        assert np.allclose(random_dataset.dataobj, ds2.dataobj)


def test_load_nifti(random_dataset: BaseDataset):
    """Test the 'load' function with a NIfTI file."""
    with TemporaryDirectory() as tmpdir:
        nifti_file = Path(tmpdir) / "test_dataset.nii.gz"
        random_dataset.to_nifti(nifti_file)

        ds2 = load(nifti_file)
        assert ds2.dataobj.shape == (32, 32, 32, 5)
        assert np.allclose(random_dataset.dataobj, ds2.dataobj)


def test_get_data(random_dataset: BaseDataset):
    """Test the get_data function."""
    # Get data without specifying dtype

    hdr = nb.Nifti1Header()
    hdr.set_data_dtype(np.int16)
    hdr.set_slope_inter(None, None)  # No scaling
    img = nb.Nifti1Image(random_dataset.dataobj.astype(np.int16), random_dataset.affine, hdr)

    data_int16 = get_data(img)
    assert data_int16.dtype == np.int16

    # Check a warning is issued for non-float dtype
    with pytest.warns(UserWarning, match="Non-float dtypes are ignored"):
        data_non_float = get_data(img, dtype="int32")
        assert data_non_float.dtype == np.int16

    data_float32 = get_data(img, dtype="float32")
    assert data_float32.dtype == np.float32

    data_float64 = get_data(img, dtype="float64")
    assert data_float64.dtype == np.float64

    img.header.set_slope_inter(2.0, 0.5)  # Set scaling
    data_scaled = get_data(img)
    assert data_scaled.dtype == np.float32
