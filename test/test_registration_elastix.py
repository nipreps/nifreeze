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
"""The elastix backend's settings (#671). Its conventions are tested with ANTs' (#619)."""

import numpy as np
import pytest
import yaml
from test_registration_conventions import (
    SKIP_ELASTIX,
    _affine,
    _error,
    _motion,
    _move,
    _volume,
)

from nifreeze.registration.elastix import Elastix, _parameter_maps, get_settings

SPACING = np.array([2.0, 2.0, 2.5])
"""Voxel sizes of the parameter maps' tests."""


def _stage(**fields):
    return {"parameters": {"Transform": ["EulerTransform"]}, **fields}


def test_settings_by_path_or_mapping(tmp_path):
    settings = get_settings()
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump({"elastix": settings}))

    assert get_settings(path) == settings
    copy = get_settings(settings)
    copy["stages"][0]["parameters"]["Transform"] = ["AffineTransform"]
    assert settings["stages"][0]["parameters"]["Transform"] == ["EulerTransform"]


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"stages": [_stage(shrink=2)]}, r"Unknown elastix settings: \['shrink'\]"),
        ({"stages": [_stage()], "winsorize": 0.1}, r"Unknown elastix settings: \['winsorize'\]"),
        ({"stages": []}, "at least one stage"),
    ],
)
def test_unknown_settings_raise(settings, message):
    with pytest.raises(ValueError, match=message):
        get_settings(settings)


def test_the_data_fill_the_samples_the_smoothing_and_the_seed():
    mask = np.ones((10, 10, 10), dtype=bool)
    settings = {"stages": [_stage(sampling_percentage=0.4, shrink_factor=2, smoothing_sigma=1.5)]}

    (pmap,) = _parameter_maps(settings, mask, SPACING, seed=7)

    assert pmap["NumberOfSpatialSamples"] == ["50"]  # 0.4 x 1000 / 2^3
    assert pmap["ImagePyramidSmoothingSchedule"] == ["3.0", "3.0", "3.75"]
    assert pmap["RandomSeed"] == ["7"]


def test_each_resolution_takes_its_own_values():
    mask = np.ones((10, 10, 10), dtype=bool)
    stage = _stage(sampling_percentage=[0.1, 0.2], smoothing_sigma=[2, 0])
    stage["parameters"]["NumberOfResolutions"] = ["2"]

    (pmap,) = _parameter_maps({"stages": [stage]}, mask, SPACING, seed=None)

    assert pmap["NumberOfSpatialSamples"] == ["100", "200"]
    assert pmap["ImagePyramidSmoothingSchedule"] == ["4.0", "4.0", "5.0", "0.0", "0.0", "0.0"]
    assert "RandomSeed" not in pmap

    stage["parameters"]["NumberOfResolutions"] = ["3"]
    with pytest.raises(ValueError, match="2 values for 3 resolutions"):
        _parameter_maps({"stages": [stage]}, mask, SPACING, seed=None)


def test_values_set_in_the_parameters_are_kept():
    mask = np.ones((10, 10, 10), dtype=bool)
    stage = _stage(sampling_percentage=0.4, smoothing_sigma=1.0)
    stage["parameters"] |= {
        "NumberOfSpatialSamples": ["2000"],
        "ImagePyramidSmoothingSchedule": ["0"] * 3,
    }

    (pmap,) = _parameter_maps({"stages": [stage]}, mask, SPACING, seed=None)

    assert pmap["NumberOfSpatialSamples"] == ["2000"]
    assert pmap["ImagePyramidSmoothingSchedule"] == ["0", "0", "0"]


@SKIP_ELASTIX
def test_a_settings_file_with_a_pyramid_registers(tmp_path):
    """Settings given by path, here a downsampling pyramid of three resolutions."""
    settings = get_settings()
    stage = settings["stages"][-1]
    stage["sampling_percentage"] = [0.4, 0.2, 0.1]
    stage["smoothing_sigma"] = [2.0, 1.0, 0.0]
    stage.pop("shrink_factor")
    stage["parameters"] |= {
        "NumberOfResolutions": ["3"],
        "ImagePyramidRescaleSchedule": ["4", "4", "4", "2", "2", "2", "1", "1", "1"],
        "MaximumNumberOfIterations": ["50"],
    }
    settings["stages"] = [stage]
    path = tmp_path / "pyramid.yaml"
    path.write_text(yaml.safe_dump({"elastix": settings}))
    data, mask = _volume()
    affine = _affine("oblique")
    matrix = _motion()

    register = Elastix(path, num_threads=1)
    estimated = register(data, _move(data, affine, matrix), affine, mask, 0).matrix

    assert _error(mask, affine, estimated, matrix) < 0.5
