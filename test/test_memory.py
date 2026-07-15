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

import numpy as np
import pytest

from nifreeze.data.base import BaseDataset

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
