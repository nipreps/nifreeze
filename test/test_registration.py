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
"""Unit tests exercising the estimator."""

import json
import re
from collections.abc import Sized
from importlib.resources import files
from os import cpu_count
from pathlib import Path
from typing import Callable, Set

import nibabel as nb
import nitransforms as nt
import numpy as np
import pytest
from nibabel.affines import from_matvec
from nibabel.eulerangles import euler2mat
from nipype.interfaces.ants.registration import Registration
from nipype.interfaces.base import Undefined

from nifreeze.registration import ants
from nifreeze.registration.ants import (
    REGISTRATION_MALFORMED_SETTINGS_ERROR_MSG,
    _massage_mask_path,
    _prepare_registration_data,
    generate_command,
)
from nifreeze.registration.utils import displacements_within_mask
from nifreeze.utils.ndimage import load_api


@pytest.fixture
def random_nifti_file(tmp_path, setup_random_uniform_spatial_data) -> Callable:
    _data, _affine = setup_random_uniform_spatial_data

    def _make(filename):
        filename = Path(filename)
        if not filename.is_absolute():
            filename = tmp_path / filename
        _img = nb.Nifti1Image(_data, _affine)
        _img.to_filename(filename)
        return filename

    return _make


@pytest.fixture
def json_file(tmp_path) -> Callable:
    def _make(settings_dict, filename):
        if not filename.is_absolute():
            filename = tmp_path / filename
        filename = Path(filename)
        filename.write_text(json.dumps(settings_dict))
        return filename

    return _make


@pytest.mark.parametrize(
    "init_affine, expect_init_file",
    [
        (
            np.array(
                [
                    [1.0, 0.0, 0.0, 2.0],
                    [0.0, 1.0, 0.0, 3.0],
                    [0.0, 0.0, 1.0, 4.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            ),
            True,
        ),
        (None, False),
    ],
)
def test_prepare_registration_data(tmp_path, init_affine, expect_init_file):
    # Small deterministic volumes
    sample = np.zeros((8, 9, 10), dtype=np.float32)
    sample[2, 3, 4] = 1.0
    predicted = np.zeros_like(sample)
    predicted[2, 3, 5] = 1.0

    affine = np.eye(4)

    predicted_path, sample_path, init_path = _prepare_registration_data(
        sample=sample,
        predicted=predicted,
        affine=affine,
        vol_idx=1,
        dirname=tmp_path,
        clip=None,
        init_affine=init_affine,
    )

    predicted_path = Path(predicted_path)
    sample_path = Path(sample_path)

    # Check NIfTI files exist
    assert predicted_path.exists(), f"Predicted file was not created: {predicted_path}"
    assert sample_path.exists(), f"Sample file was not created: {sample_path}"

    # Check affines preserved
    loaded_sample = load_api(str(sample_path), nb.Nifti1Image)
    loaded_pred = load_api(str(predicted_path), nb.Nifti1Image)
    assert np.allclose(loaded_sample.affine, affine)
    assert np.allclose(loaded_pred.affine, affine)

    init_file_expected_path = tmp_path / "init_00001.mat"
    if expect_init_file:
        # If init_affine is provided, init_<idx>.mat must be returned, exist and
        # be non-empty
        assert init_path is not None, "init_path should not be None when init_affine is provided"
        init_path = Path(init_path)
        assert init_path.exists(), f"Init transform file was not created: {init_path}"
        assert init_path.stat().st_size > 0, "Init transform file is empty"
        # Ensure it's the expected filename in the tmp directory
        assert init_path == init_file_expected_path
    else:
        # When no init_affine provided, function should return None and no init
        # file should exist
        assert init_path is None, "init_path should be None when init_affine is not provided"
        assert not init_file_expected_path.exists(), "Unexpected init file found on disk"


def test_generate_command_malformed_settings(tmp_path, monkeypatch, random_nifti_file, json_file):
    fixed = random_nifti_file(tmp_path / "fixed.nii.gz")
    moving = random_nifti_file(tmp_path / "moving.nii.gz")

    # Build a settings dict with inconsistent list lengths for keys in
    # PARAMETERS_SINGLE_LIST (e.g., 'metric' length 2, 'sampling_percentage' length 1)
    malformed_settings = {
        "metric": ["MattesLevel1", "MattesLevel2"],  # length 2
        "sampling_percentage": [0.2],  # length 1 -> inconsistency triggers the RuntimeError
        # Include a few single-valued params so the JSON looks realistic
        "collapse_output_transforms": 1,
        "dimension": 3,
        "initialize_transforms_per_stage": 0,
        "interpolation": "Linear",
        "output_transform_prefix": "transform",
        "write_composite_transform": 0,
    }

    settings_file = json_file(malformed_settings, tmp_path / "malformed_settings.json")

    # Monkeypatch _get_ants_settings to return our controlled settings file
    monkeypatch.setattr(ants, "_get_ants_settings", lambda settings="": Path(settings_file))

    # Required to avoid mypy errors about objects not having a Size mypy happy
    levels: Set[int] = set()
    for p in ants.PARAMETERS_SINGLE_LIST:
        if p not in malformed_settings:
            continue
        val = malformed_settings[p]
        assert isinstance(val, Sized), (
            f"Settings[{p!r}] must be a sized object (has __len__); got {type(val)!r}"
        )
        levels.add(len(val))
    levels.pop()
    with pytest.raises(
        RuntimeError,
        match=re.escape(REGISTRATION_MALFORMED_SETTINGS_ERROR_MSG.format(levels=levels)),
    ):
        generate_command(fixed_path=fixed, moving_path=moving, default="unused")


def test_generate_command_basic(tmp_path, random_nifti_file):
    """Basic smoke test for generate_command: returns a Registration instance
    and sets fixed/moving image absolute paths in the inputs.
    """

    fixed = random_nifti_file(tmp_path / "fixed.nii.gz")
    moving = random_nifti_file(tmp_path / "moving.nii.gz")

    reg = generate_command(fixed_path=fixed, moving_path=moving)

    assert isinstance(reg, Registration)
    assert reg.inputs.fixed_image[0] == str(Path(fixed).absolute())
    assert reg.inputs.moving_image[0] == str(Path(moving).absolute())


def test_generate_command_with_masks_init_and_threads(tmp_path, random_nifti_file):
    """Ensure generate_command accepts masks, an initial transform path, and
    num_threads, and exposes them on the returned Registration.inputs.
    """
    fixed = random_nifti_file(tmp_path / "fixed.nii.gz")
    moving = random_nifti_file(tmp_path / "moving.nii.gz")

    # Create a dummy mask file and a dummy init transform file
    mask = random_nifti_file(tmp_path / "mask.nii.gz")
    init_mat = tmp_path / "init.mat"
    init_mat.write_text("ITK transform placeholder")

    num_threads = 4
    environ = {"TEST_ENV": "1"}
    terminal_output = "file"

    reg = generate_command(
        fixed_path=fixed,
        moving_path=moving,
        fixedmask_path=mask,
        init_affine=init_mat,
        num_threads=num_threads,
        environ=environ,
        terminal_output=terminal_output,
    )

    # Registration interface sanity checks
    assert isinstance(reg, Registration)
    assert reg.inputs.fixed_image[0] == str(Path(fixed).absolute())
    assert reg.inputs.moving_image[0] == str(Path(moving).absolute())

    # num_threads should be set on inputs when provided
    assert reg.inputs.num_threads == num_threads

    # Initial transform should be propagated to inputs under the expected key
    # generate_command sets settings["initial_moving_transform"] so the
    # Registration input with the same name should exist and hold the path
    # string.
    assert reg.inputs.initial_moving_transform[0] == str(init_mat)

    # The fixed mask should be present on inputs under the key the function
    # uses: "fixed_image_masks" (a list)
    fim = getattr(reg.inputs, "fixed_image_masks", None)
    assert fim is not None, "fixed_image_masks not present on Registration.inputs"
    fim = fim[:]
    # At minimum, the mask path should appear in the list (exact
    # repetition/count depends on settings file)
    assert str(mask) in fim or str(mask.absolute()) in fim


@pytest.mark.parametrize(
    "settings_dict, shrink_override, metric_override, expected_shrink, expected_metric, case_name",
    [
        (
            # Single-level settings (nlevels == 1)
            {
                "shrink_factors": [[3]],
                "metric": ["Mattes"],
                "collapse_output_transforms": True,
                "dimension": 3,
                "initialize_transforms_per_stage": False,
                "interpolation": "Linear",
                "output_transform_prefix": "transform",
                "write_composite_transform": False,
            },
            5,
            ["MI"],
            [[5]],
            [["MI"]],
            "single",
        ),
        (
            # Multi-level settings (nlevels == 2)
            {
                "shrink_factors": [[4], [2]],
                "metric": [["MI"], ["MattesLevel2"]],
                "collapse_output_transforms": True,
                "dimension": 3,
                "initialize_transforms_per_stage": False,
                "interpolation": "Linear",
                "output_transform_prefix": "transform",
                "write_composite_transform": False,
            },
            9,
            ["Demons"],
            [[4], [9]],
            [["MI"], ["Demons"]],
            "multi",
        ),
    ],
)
def test_generate_command_override_nested_list_parameters(
    tmp_path,
    monkeypatch,
    random_nifti_file,
    json_file,
    settings_dict,
    shrink_override,
    metric_override,
    expected_shrink,
    expected_metric,
    case_name,
):
    """Ensure that generate_command applies overrides for:
      - a PARAMETERS_DOUBLE_LIST key (shrink_factors)
      - a PARAMETERS_SINGLE_LIST key (metric)

    The test monkeypatches _get_ants_settings to return a controlled JSON file
    so generate_command's mutation of settings is deterministic.
    """
    fixed = random_nifti_file(tmp_path / "fixed.nii.gz")
    moving = random_nifti_file(tmp_path / "moving.nii.gz")

    settings_file = json_file(settings_dict, tmp_path / f"settings_{case_name}.json")

    # Monkeypatch the resolver
    monkeypatch.setattr(ants, "_get_ants_settings", lambda settings="": Path(settings_file))

    # Call generate_command with overrides that should affect the last level
    reg = generate_command(
        fixed_path=fixed,
        moving_path=moving,
        default="unused",
        shrink_factors=shrink_override,
        metric=metric_override,
    )

    # Assert the resulting Registration.inputs reflect the expected overrides
    assert reg.inputs.shrink_factors[:] == expected_shrink
    assert reg.inputs.metric[:] == expected_metric


@pytest.mark.parametrize(
    "use_masks, use_init_affine",
    [
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ],
)
def test_generate_command_masks_and_init_propagation(
    tmp_path, monkeypatch, use_masks, use_init_affine, random_nifti_file, json_file
):
    """Ensure generate_command:
    - propagates fixed/moving masks into Registration.inputs as
      fixed_image_masks / moving_image_masks
    - propagates an initialization transform path into Registration.inputs as
      initial_moving_transform
    """
    fixed = random_nifti_file(tmp_path / "fixed.nii.gz")
    moving = random_nifti_file(tmp_path / "moving.nii.gz")

    # Minimal settings file (single-level)
    _settings = {
        "shrink_factors": [[2]],
        "metric": ["Mattes"],
        "collapse_output_transforms": True,
        "dimension": 3,
        "initialize_transforms_per_stage": False,
        "interpolation": "Linear",
        "output_transform_prefix": "transform",
        "write_composite_transform": False,
    }
    settings_file = json_file(_settings, tmp_path / "settings.json")

    # Monkeypatch the resolver
    monkeypatch.setattr(ants, "_get_ants_settings", lambda settings="": Path(settings_file))

    fixed_mask = None
    moving_mask = None
    if use_masks:
        fixed_mask = random_nifti_file(tmp_path / "fmask.nii.gz")
        moving_mask = random_nifti_file(tmp_path / "mmask.nii.gz")

    init_path = None
    if use_init_affine:
        init_path = tmp_path / "init.mat"
        init_path.write_text("ITK transform placeholder")

    reg = generate_command(
        fixed_path=fixed,
        moving_path=moving,
        default="unused",
        fixedmask_path=fixed_mask if use_masks else None,
        movingmask_path=moving_mask if use_masks else None,
        init_affine=init_path if use_init_affine else None,
    )

    # Masks: when provided, Registration.inputs should contain
    # fixed_image_masks / moving_image_masks
    fim = getattr(reg.inputs, "fixed_image_masks", Undefined)
    mim = getattr(reg.inputs, "moving_image_masks", Undefined)
    if use_masks:
        assert fim is not Undefined, "fixed_image_masks should be present when mask provided"
        assert mim is not Undefined, "moving_image_masks should be present when mask provided"
        assert fixed_mask is not None
        assert moving_mask is not None
        assert fim is not None and any(
            str(fixed_mask) in s or str(fixed_mask.absolute()) in s for s in fim
        )
        assert mim is not None and any(
            str(moving_mask) in s or str(moving_mask.absolute()) in s for s in mim
        )
    else:
        # If not provided, those inputs may be Undefined (nipype sentinel) or absent/empty.
        assert fim in (Undefined, None, ""), (
            f"expected fixed_image_masks to be undefined/empty, got {fim}"
        )
        assert mim in (Undefined, None, ""), (
            f"expected moving_image_masks to be undefined/empty, got {mim}"
        )

    # Init transform: when provided, initial_moving_transform should be set to
    # its path string
    init_input = getattr(reg.inputs, "initial_moving_transform", None)
    if use_init_affine:
        assert init_input is not Undefined, (
            "initial_moving_transform should be present when init_affine provided"
        )
        assert init_path is not None
        assert init_input is not None
        assert str(init_path) in init_input or str(init_path.absolute()) in init_input
    else:
        # May be Undefined/None/empty if not provided
        assert init_input in (Undefined, None, ""), (
            f"expected initial_moving_transform to be undefined/empty, got {init_input}"
        )


@pytest.mark.parametrize("r_x", [0.0, 0.1, 0.3])
@pytest.mark.parametrize("r_y", [0.0, 0.1, 0.3])
@pytest.mark.parametrize("r_z", [0.0, 0.1, 0.3])
@pytest.mark.parametrize("t_x", [0.0, 1.0])
@pytest.mark.parametrize("t_y", [0.0, 1.0])
@pytest.mark.parametrize("t_z", [0.0, 1.0])
@pytest.mark.parametrize("dataset", ["hcph", "dwi"])
# @pytest.mark.parametrize("dataset", ["dwi"])
def test_ANTs_config_b0(datadir, tmp_path, dataset, r_x, r_y, r_z, t_x, t_y, t_z):
    """Check that the registration parameters for b=0
    gives a good estimate of known affine"""

    fixed = datadir / f"{dataset}-b0_desc-avg.nii.gz"
    fixed_mask = datadir / f"{dataset}-b0_desc-brain.nii.gz"
    moving = tmp_path / "moving.nii.gz"

    b0nii = load_api(fixed, nb.Nifti1Image)
    T = from_matvec(euler2mat(x=r_x, y=r_y, z=r_z), (t_x, t_y, t_z))
    xfm = nt.linear.Affine(T, reference=b0nii)

    nt.resampling.apply(~xfm, b0nii, reference=b0nii).to_filename(moving)

    registration = Registration(
        terminal_output="file",
        from_file=files("nifreeze.registration").joinpath("config/b0-to-b0_level0.json"),
        fixed_image=str(fixed.absolute()),
        moving_image=str(moving.absolute()),
        fixed_image_masks=[str(fixed_mask)],
        random_seed=1234,
        num_threads=cpu_count(),
    )

    result = registration.run(cwd=str(tmp_path)).outputs
    xform = nt.linear.Affine(
        nt.io.itk.ITKLinearTransform.from_filename(result.forward_transforms[0]).to_ras(),
        reference=b0nii,
    )

    masknii = load_api(fixed_mask, nb.Nifti1Image)
    assert displacements_within_mask(masknii, xform, xfm).mean() < (
        0.6 * np.mean(b0nii.header.get_zooms()[:3])
    )


def test_massage_mask_path():
    """Test the case where a warning must be issued."""
    with pytest.warns(UserWarning, match="More mask paths than levels"):
        maskpath = _massage_mask_path(["/some/path"] * 2, 1)

    assert maskpath == ["/some/path"]
