# Copyright The NiPreps Developers <nipreps@gmail.com>
"""Tests for :mod:`nifreeze.data.dmri.base`."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from nifreeze.data.dmri.base import DWI, validate_gradients


def _build_dwi(data, gradients):
    return DWI(dataobj=data, affine=np.eye(4), gradients=gradients)


def test_validate_gradients_errors():
    with pytest.raises(ValueError):
        validate_gradients(None, None, np.zeros((2, 3)))

    bad = np.array([[0.0, 0.0, 0.0, np.nan]])
    with pytest.raises(ValueError):
        validate_gradients(None, None, bad)


def test_dwi_post_init_branches():
    data = np.zeros((2, 2, 2, 1), dtype=np.float32)
    gradients = np.zeros((2, 4), dtype=float)
    with pytest.raises(ValueError):
        _build_dwi(data, gradients)

    data = np.zeros((2, 2, 2, 8), dtype=np.float32)
    gradients = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 20.0],
            [1.0, 0.0, 0.0, 1200.0],
            [0.0, 1.0, 0.0, 1500.0],
            [0.0, 0.0, 1.0, 1600.0],
            [1.0, 0.0, 0.0, 1700.0],
            [0.0, 1.0, 1.0, 1800.0],
            [1.0, 1.0, 0.0, 1900.0],
        ]
    )
    dwi = _build_dwi(data, gradients)
    assert dwi.bzero.shape == data.shape[:3]
    assert dwi.dataobj.shape[-1] == 6
    assert dwi.gradients.shape[0] == 6

    few_gradients = np.array([[0.0, 0.0, 1.0, 1000.0]])
    with pytest.raises(ValueError):
        _build_dwi(np.zeros((2, 2, 2, 1), dtype=np.float32), few_gradients)


def test_get_shells_and_io(tmp_path, monkeypatch):
    data = np.zeros((2, 2, 2, 7), dtype=np.float32)
    gradients = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 1000.0],
            [0.0, 1.0, 0.0, 1100.0],
            [1.0, 0.0, 1.0, 1200.0],
            [0.0, 1.0, 1.0, 1300.0],
            [1.0, 1.0, 0.0, 1400.0],
            [1.0, 1.0, 1.0, 1500.0],
        ]
    )
    dwi = _build_dwi(data, gradients)

    monkeypatch.setattr(
        "nifreeze.data.dmri.base.find_shelling_scheme",
        lambda bvals, num_bins, multishell_nonempty_bin_count_thr, bval_cap: (
            "multi-shell",
            [np.array([1000.0, 1100.0, 1200.0]), np.array([1300.0, 1400.0, 1500.0])],
            [1100.0, 1400.0],
        ),
    )
    shells = dwi.get_shells(num_bins=5, multishell_nonempty_bin_count_thr=2, bval_cap=3000)
    assert len(shells) == 2
    assert shells[0][0] == 1100.0
    np.testing.assert_array_equal(shells[0][1], np.array([0, 1, 2]))
    assert shells[1][0] == 1400.0
    np.testing.assert_array_equal(shells[1][1], np.array([3, 4, 5]))

    out_file = tmp_path / "dwi.h5"
    dwi.to_filename(out_file)
    with h5py.File(out_file, "r") as h5file:
        assert h5file.attrs["Type"] == "dmri"

    reloaded = DWI.from_filename(out_file)
    np.testing.assert_array_equal(reloaded.dataobj, dwi.dataobj)
    np.testing.assert_array_equal(reloaded.gradients, dwi.gradients)
