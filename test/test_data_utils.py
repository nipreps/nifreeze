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

import gc
import tempfile
import warnings
from pathlib import Path

import h5py
import nibabel as nb
import numpy as np
import numpy.testing as npt
import pytest

from nifreeze.data import utils
from nifreeze.data.dmri.base import DWI
from nifreeze.data.utils import apply_affines, open_dataobj_memmap
from nifreeze.utils.ndimage import load_api


def test_apply_affines(request, tmp_path):
    rng = request.node.rng

    # Create synthetic dataset
    nii_data = rng.random((10, 10, 10, 10))

    # Generate Nifti1Image
    nii = nb.Nifti1Image(nii_data, np.eye(4))

    # Generate synthetic affines
    em_affines = np.expand_dims(np.eye(4), 0).repeat(nii_data.shape[-1], 0)

    nii_t = apply_affines(nii, em_affines)

    npt.assert_allclose(nii.dataobj, nii_t.dataobj)
    npt.assert_array_equal(nii.affine, nii_t.affine)

    # Test with output filename
    out_fname = tmp_path / "affine.nii.gz"
    nii_t_file = apply_affines(nii, em_affines, output_filename=out_fname)
    npt.assert_allclose(nii.dataobj, nii_t_file.dataobj)
    npt.assert_array_equal(nii.affine, nii_t_file.affine)
    assert out_fname.exists()
    # The saved file should load and match the expected result
    nii_loaded = load_api(out_fname, nb.Nifti1Image)
    npt.assert_allclose(nii.dataobj, nii_loaded.dataobj)
    npt.assert_array_equal(nii.affine, nii_loaded.affine)


def _write_legacy(path, data):
    """Write ``data`` (volumes last) as a version-1 NFDH5 ``dataobj``."""
    with h5py.File(path, "w") as out_file:
        out_file.attrs["Format"] = "NFDH5"
        out_file.attrs["Version"] = np.uint16(1)
        out_file.create_group("/0").create_dataset("dataobj", data=data)
    return path


@pytest.mark.parametrize("slab_bytes", [None, 1])
def test_legacy_dataobj_conversion(tmp_path, monkeypatch, slab_bytes):
    """Legacy files convert to the same values with any slab size."""
    if slab_bytes is not None:
        monkeypatch.setattr(utils, "CONVERSION_SLAB_BYTES", slab_bytes)
    data = np.random.default_rng(1234).random((7, 6, 5, 4), dtype=np.float32)
    dataobj = open_dataobj_memmap(_write_legacy(tmp_path / "legacy.h5", data))
    assert isinstance(dataobj, np.memmap)
    npt.assert_array_equal(dataobj, data)


def test_cache_removed_with_last_view(tmp_path):
    """A conversion cache lives exactly as long as some array references it."""
    data = np.random.default_rng(1234).random((7, 6, 5, 4), dtype=np.float32)
    dataobj = open_dataobj_memmap(_write_legacy(tmp_path / "legacy.h5", data))
    cache = Path(dataobj.filename)
    assert cache.is_file()
    assert cache.parent.name.startswith("nifreeze-")

    volume = dataobj[..., 1]
    del dataobj
    gc.collect()
    assert cache.is_file()
    npt.assert_array_equal(volume, data[..., 1])

    del volume
    gc.collect()
    assert not cache.parent.exists()


def test_cache_dir_given_is_kept(tmp_path):
    """Only the cache file is removed when the caller chose the directory."""
    data = np.random.default_rng(1234).random((7, 6, 5, 4), dtype=np.float32)
    cache_dir = tmp_path / "cache"
    dataobj = open_dataobj_memmap(_write_legacy(tmp_path / "legacy.h5", data), cache_dir=cache_dir)
    assert len(list(cache_dir.iterdir())) == 1
    del dataobj
    gc.collect()
    assert cache_dir.is_dir()
    assert not any(cache_dir.iterdir())


def test_in_memory_b0_removal_writes_no_cache(tmp_path, monkeypatch):
    """Dropping b=0 volumes from an in-memory DWI stays in memory."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    rng = np.random.default_rng(1234)
    bvecs = rng.standard_normal((10, 3))
    bvecs /= np.linalg.norm(bvecs, axis=1, keepdims=True)
    bvals = np.full(10, 1000.0)
    bvals[[0, 5]] = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dwi = DWI(
            dataobj=rng.random((4, 4, 4, 10), dtype=np.float32),
            affine=np.eye(4),
            gradients=np.column_stack((bvecs, bvals)),
        )
    assert len(dwi) == 8
    assert not isinstance(dwi.dataobj, np.memmap)
    assert not list(tmp_path.rglob("*.npy"))
