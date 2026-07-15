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
"""Peak-memory tests for the data layer (issue #347).

These are opt-in (``-m memory``): they build a moderately large 4D fixture
(~140 MB) so that the memory cost of an eager full load dwarfs interpreter
noise, and assert that the disk-backed paths keep peak RSS far below the size
of the whole array. They fail on an eager (in-RAM) implementation and pass on
the memory-mapped one.
"""

import h5py
import numpy as np
import pytest

from nifreeze.data.base import BaseDataset
from nifreeze.data.dmri.base import DWI
from nifreeze.data.utils import save_dataobj_volume_major

pytestmark = pytest.mark.memory

# ~140 MB float32 series; a single volume is ~2.4 MB.
_SHAPE = (96, 96, 64, 60)
_FULL_MB = float(np.prod(_SHAPE) * 4) / 1024.0 / 1024.0


def _write_dataset(path):
    """Write a random BaseDataset to ``path`` and return (path, nbytes_mb)."""
    rng = np.random.default_rng(1234)
    dataobj = rng.random(_SHAPE, dtype=np.float32)
    BaseDataset(dataobj=dataobj, affine=np.eye(4)).to_filename(path)
    return path


def test_from_filename_peak_rss(peak_rss, tmp_path):
    """Loading + touching one volume must not materialize the whole series."""
    h5 = _write_dataset(tmp_path / "big.h5")

    def _load_and_index(p):
        ds = BaseDataset.from_filename(p)
        # Touch a single volume only — never the whole array.
        _ = np.asarray(ds[len(ds) // 2][0]).sum()

    peak_mb = peak_rss(_load_and_index, h5)
    assert peak_mb < 0.4 * _FULL_MB, (
        f"from_filename peaked at {peak_mb:.1f} MB for a {_FULL_MB:.1f} MB series "
        "(expected a memory-mapped, lazy load)"
    )


def test_from_filename_returns_memmap(tmp_path):
    """The loaded ``dataobj`` is a memory-mapped ndarray subclass."""
    h5 = _write_dataset(tmp_path / "big.h5")
    ds = BaseDataset.from_filename(h5)
    assert isinstance(ds.dataobj, np.memmap)
    assert isinstance(ds.dataobj, np.ndarray)  # subclass — consumer contract holds


def test_from_filename_roundtrip_values(tmp_path):
    """Lazy load preserves values (checked volume-by-volume to stay bounded)."""
    rng = np.random.default_rng(1234)
    expected = rng.random(_SHAPE, dtype=np.float32)
    h5 = tmp_path / "big.h5"
    BaseDataset(dataobj=expected, affine=np.eye(4)).to_filename(h5)

    ds = BaseDataset.from_filename(h5)
    assert ds.dataobj.shape == _SHAPE
    for i in range(_SHAPE[-1]):
        np.testing.assert_allclose(ds[i][0], expected[..., i])


_B0_INDICES = (0, 20, 40)  # interspersed b=0 volumes


def _write_dwi_with_b0(path, data):
    """Write an NFDH5 DWI that still contains b=0 volumes.

    ``DWI.to_filename`` drops b=0 volumes before writing, so to exercise the
    on-load b0 filter we write the volume-major ``dataobj`` and a b0-bearing
    gradient table directly.
    """
    rng = np.random.default_rng(0)
    n = data.shape[-1]
    bvecs = rng.standard_normal((n, 3)).astype(np.float32)
    bvecs /= np.linalg.norm(bvecs, axis=1, keepdims=True)
    bvals = np.full(n, 1000.0, dtype=np.float32)
    for i in _B0_INDICES:
        bvals[i] = 0.0
        bvecs[i] = 0.0
    with h5py.File(path, "w") as f:
        f.attrs["Format"] = "NFDH5"
        f.attrs["Version"] = np.uint16(2)
        root = f.create_group("/0")
        root.attrs["Type"] = "dmri"
        save_dataobj_volume_major(root, data)
        root.create_dataset("affine", data=np.eye(4))
        root.create_dataset("gradients", data=np.column_stack((bvecs, bvals)))
    return path


@pytest.mark.filterwarnings("ignore:The DWI data contains multiple b0 volumes")
def test_dwi_b0_filtered_dataobj_is_memmap(tmp_path):
    """After dropping b=0 volumes, dataobj stays a disk-backed memmap."""
    rng = np.random.default_rng(1234)
    expected = rng.random(_SHAPE, dtype=np.float32)
    h5 = _write_dwi_with_b0(tmp_path / "dwi.h5", expected)

    dwi = DWI.from_filename(h5)
    keep = [i for i in range(_SHAPE[-1]) if i not in _B0_INDICES]
    assert isinstance(dwi.dataobj, np.memmap)  # not retained as an in-RAM copy
    assert dwi.dataobj.shape[-1] == len(keep)
    for j in range(len(keep)):
        np.testing.assert_allclose(dwi[j][0], expected[..., keep[j]])
