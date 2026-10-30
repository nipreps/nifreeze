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
"""The harness of the registration experiments (``tools/registration/harness.py``)."""

import doctest
import importlib.util
import shutil
from itertools import product
from pathlib import Path

import nibabel as nb
import numpy as np
import pytest
from scipy import ndimage

from nifreeze.registration.ants import _to_nifti, generate_command

AFFINE = np.diag([2.0, 2.0, 2.0, 1.0])
"""Voxels of 2 mm, with the voxel axes along RAS."""
AFFINE[:3, 3] = -32.0


@pytest.fixture(scope="module")
def harness():
    path = Path(__file__).parents[1] / "tools" / "registration" / "harness.py"
    spec = importlib.util.spec_from_file_location("harness", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def volumes():
    """A smooth volume with structure along every axis, its shifted copy, and a mask."""
    rng = np.random.default_rng(1234)
    fixed = ndimage.gaussian_filter(rng.random((32, 32, 32)), 2.0).astype("float32")
    fixed = (fixed - fixed.min()) / (fixed.max() - fixed.min()) * 1000
    mask = np.zeros(fixed.shape, dtype=bool)
    mask[4:-4, 4:-4, 4:-4] = True
    return fixed, mask


def test_the_doctests_pass(harness):
    assert doctest.testmod(harness).failed == 0


def test_settings_override_every_level_or_some(harness):
    settings = harness.settings_from(
        shrink_factors={-1: [1]}, sampling_percentage=[0.5, 0.5], verbose=False
    )
    assert settings["shrink_factors"] == [[3], [1]]
    assert settings["sampling_percentage"] == [0.5, 0.5]
    assert settings["verbose"] is False
    # The template is not modified
    assert harness.settings_from()["shrink_factors"] == [[3], [2]]

    with pytest.raises(KeyError, match="not_a_setting"):
        harness.settings_from(not_a_setting=1)


def test_the_command_is_the_estimators(harness, volumes, tmp_path):
    """With the default settings, the harness runs the estimator's own command line."""
    fixed, mask = volumes
    fixed_path, moving_path = harness.write_inputs(fixed, fixed, AFFINE, mask, tmp_path)
    mask_path = harness.write_mask(mask, AFFINE, tmp_path / "brainmask.nii")

    line = harness.command(
        harness.settings_from(), fixed_path, moving_path, mask_path, "ants-00000"
    )
    expected = generate_command(
        fixed_path,
        moving_path,
        fixedmask_path=mask_path,
        output_transform_prefix="ants-00000",
    ).cmdline
    assert line == expected


def test_the_writer_codes_as_the_estimator(harness, volumes, tmp_path):
    """The harness's own writer gives the estimator's file for the estimator's coding."""
    fixed, mask = volumes
    window = harness.robust_window(fixed, mask)
    _to_nifti(fixed, AFFINE, tmp_path / "estimator.nii", window=window)
    harness._code(fixed, AFFINE, tmp_path / "harness.nii", window, "int12")
    assert (tmp_path / "estimator.nii").read_bytes() == (tmp_path / "harness.nii").read_bytes()


@pytest.mark.parametrize(
    ("clip", "coding"), list(product(("moving", "each", "none"), ("int12", "uint16", "float32")))
)
def test_every_representation_is_written(harness, volumes, tmp_path, clip, coding):
    fixed, mask = volumes
    moving = fixed * 0.5 + 100
    fixed_path, moving_path = harness.write_inputs(
        fixed, moving, AFFINE, mask, tmp_path, clip=clip, coding=coding
    )

    windows = {
        "moving": [harness.robust_window(moving, mask)] * 2,
        "each": [harness.robust_window(fixed, mask), harness.robust_window(moving, mask)],
        "none": [(fixed.min(), fixed.max()), (moving.min(), moving.max())],
    }[clip]
    dtype = {"int12": np.int16, "uint16": np.uint16, "float32": np.float32}[coding]
    levels = {"int12": 4095, "uint16": 65535, "float32": None}[coding]
    for path, data, (a_min, a_max) in zip(
        (fixed_path, moving_path), (fixed, moving), windows, strict=True
    ):
        img = nb.Nifti1Image.from_filename(path)
        assert img.get_data_dtype() == dtype
        assert np.allclose(img.affine, AFFINE)
        written = img.get_fdata()
        clipped = np.clip(data, a_min, a_max)
        tolerance = 1e-3 if levels is None else (a_max - a_min) / levels / 2 + 1e-3
        assert np.abs(written - clipped).max() <= tolerance


def test_the_verdicts(harness):
    rng = np.random.default_rng(0)
    reference = rng.uniform(0.1, 1.0, 200)
    noise = rng.normal(0, 0.001, 200)

    better = harness.paired_comparison(reference - 0.1 + noise, reference, 0.02)
    assert better["verdict"] == "better"
    assert better["median_low"] < better["median"] < better["median_high"] < 0

    assert harness.paired_comparison(reference + noise, reference, 0.02)["verdict"] == (
        "not worse"
    )
    assert harness.paired_comparison(reference + 0.1, reference, 0.02)["verdict"] == "worse"

    loud = rng.normal(0, 1.0, 10)
    assert harness.paired_comparison(reference[:10] + loud, reference[:10], 0.02)["verdict"] == (
        "inconclusive"
    )

    same = harness.paired_comparison(reference, reference, 0.02)
    assert (same["verdict"], same["wilcoxon_p"], same["median"]) == ("not worse", 1.0, 0.0)

    # The bootstrap is reproducible, and resamples within strata
    strata = np.repeat([1000, 2000], 100)
    first = harness.paired_comparison(reference + noise, reference, 0.02, strata=strata)
    again = harness.paired_comparison(reference + noise, reference, 0.02, strata=strata)
    assert first == again

    with pytest.raises(ValueError, match="paired"):
        harness.paired_comparison(reference[:5], reference, 0.02)


def test_the_subsets_do_not_overlap(harness):
    for protocol, n in (("sherbrooke", 192), ("hcpya", 270)):
        dev = harness.subset(protocol, "dev", n)
        confirm = harness.subset(protocol, "confirm", n)
        assert not set(dev) & set(confirm)
        assert len(dev) == len(confirm) == {"sherbrooke": 64, "hcpya": 30}[protocol]


def test_provenance_records_what_ran(harness):
    record = harness.provenance({"metric": ["GC"]})
    assert {
        "AntsPath",
        "AntsVersion",
        "NiFreezeCommit",
        "NiFreezeDirty",
        "NiFreezeVersion",
        "Packages",
        "Threads",
        "Affinity",
        "Settings",
    } <= set(record)
    assert set(record["Threads"]) == set(harness.THREADS)


def test_the_truth_has_no_error(harness, volumes):
    _, mask = volumes
    mask_img = nb.Nifti1Image(mask.astype("uint8"), AFFINE)
    matrix = np.eye(4)
    matrix[:3, 3] = (1.0, -2.0, 0.5)

    measured = harness.errors(mask_img, matrix, matrix)
    assert all(abs(value) < 1e-6 for value in measured.values())
    assert harness.errors(mask_img, np.eye(4), matrix)["mean"] == pytest.approx(np.sqrt(5.25))


@pytest.mark.skipif(shutil.which("antsRegistration") is None, reason="ANTs is not installed")
def test_register_recovers_a_shift(harness, volumes, tmp_path):
    """The estimate maps the fixed image onto the moving one, as the estimator reads it."""
    fixed, mask = volumes
    shift = np.array([2.0, -1.0, 1.5])
    # Content at x moves to x + shift: moving(x + shift) = fixed(x)
    moving = ndimage.shift(fixed, shift / 2.0, order=3, mode="nearest")
    expected = np.eye(4)
    expected[:3, 3] = shift

    mask_path = harness.write_mask(mask, AFFINE, tmp_path / "brainmask.nii")
    estimated, seconds = harness.register(
        fixed, moving, AFFINE, mask, mask_path, tmp_path / "volume"
    )

    mask_img = nb.Nifti1Image(mask.astype("uint8"), AFFINE)
    assert seconds > 0
    assert harness.errors(mask_img, estimated, expected)["mean"] < 0.3
    assert harness.errors(mask_img, np.eye(4), expected)["mean"] > 2.5
