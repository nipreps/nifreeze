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

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import nibabel as nb
import numpy as np
import pytest

from nifreeze.data import NFDH5_EXT, BaseDataset, load
from nifreeze.utils.ndimage import get_data

DEFAULT_RANDOM_DATASET_SHAPE = (32, 32, 32, 5)
DEFAULT_RANDOM_DATASET_SIZE = int(np.prod(DEFAULT_RANDOM_DATASET_SHAPE[:3]))


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


def test_to_nifti(random_dataset: BaseDataset):
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
