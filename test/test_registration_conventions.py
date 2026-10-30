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
"""The conventions by which data reach ANTs and its estimates come back (#619).

Each test uses small synthetic volumes with known transforms, on three affines: one
aligned with RAS, the TRXScan phantom's flipped frame (diag(-1, -1, 1)), and an
oblique one (the b-vector tests add one aligned with LAS). Tests that fail mark the
bug they found, as strict expected failures, until its fix. A transform maps the
fixed image's points onto the moving image's (fixed to moving, in RAS), as the
estimator reports it.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path

import nibabel as nb
import numpy as np
import pytest
from nitransforms.linear import Affine
from nitransforms.resampling import apply
from scipy import ndimage
from scipy.spatial.transform import Rotation

from nifreeze.data.base import BaseDataset
from nifreeze.data.dmri import DWI
from nifreeze.data.filtering import robust_window
from nifreeze.registration.ants import (
    WINDOW_LEVELS,
    _prepare_registration_data,
    _run_registration,
    _to_nifti,
    generate_command,
)
from nifreeze.registration.utils import displacements_within_mask

SHAPE = (40, 40, 40)
"""The grid of the synthetic volumes."""
NO_ANTS = shutil.which("antsRegistration") is None
"""Whether ANTs is missing, which skips the tests that run it."""


def _affine(kind: str) -> np.ndarray:
    """2 mm voxels, centered on the origin."""
    linear = {
        "ras": np.diag([2.0, 2.0, 2.0]),
        "phantom": np.diag([-2.0, -2.0, 2.0]),
        "las": np.diag([-2.0, 2.0, 2.0]),
        "oblique": Rotation.from_rotvec(np.deg2rad(15) * np.array([1.0, 1.0, 0.0]) / np.sqrt(2))
        .as_matrix()
        .dot(np.diag([2.0, 2.0, 2.0])),
    }[kind]
    affine = np.eye(4)
    affine[:3, :3] = linear
    affine[:3, 3] = -linear @ (np.array(SHAPE) - 1) / 2
    return affine


AFFINES = ("ras", "phantom", "oblique")
"""The three frames every test runs on."""
WRONG_AXIS = pytest.mark.xfail(
    strict=True, reason="nipreps/nifreeze#620: b-vectors rotate about the wrong axis"
)
"""B-vector tables that ``transform_fsl_bvec`` rotates about the wrong axis."""


def _motion(degrees: float = 3.0, shift=(1.5, -1.0, 2.0)) -> np.ndarray:
    """A rigid motion about an oblique axis through the origin (the volume's center)."""
    axis = np.array([1.0, -2.0, 0.5])
    matrix = np.eye(4)
    rotation = Rotation.from_rotvec(np.deg2rad(degrees) * axis / np.linalg.norm(axis))
    matrix[:3, :3] = rotation.as_matrix()
    matrix[:3, 3] = shift
    return matrix


@pytest.fixture
def volume():
    """A smooth volume, with structure along every axis, and a spherical mask."""
    rng = np.random.default_rng(1234)
    data = ndimage.gaussian_filter(rng.random(SHAPE), 2.5)
    data = ((data - data.min()) / (data.max() - data.min()) * 1000).astype("float32")
    grid = np.indices(SHAPE) - (np.array(SHAPE) - 1)[:, None, None, None] / 2
    mask = np.linalg.norm(grid, axis=0) < 14
    return data, mask


def _move(data: np.ndarray, affine: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """The volume after the head moves by ``matrix``: moved(matrix @ x) = data(x)."""
    image = nb.Nifti1Image(data, affine, None)
    moved = apply(Affine(np.linalg.inv(matrix)), image, reference=image, order=3)
    return np.asanyarray(moved.dataobj, dtype="float32")


def _error(mask: np.ndarray, affine: np.ndarray, estimated, expected) -> float:
    mask_img = nb.Nifti1Image(mask.astype("uint8"), affine)
    return float(displacements_within_mask(mask_img, Affine(estimated), Affine(expected)).mean())


def _register(fixed, moving, affine, mask, workdir: Path, init=None) -> np.ndarray:
    """Register as the estimator does (``Estimator.run``), and return the matrix."""
    workdir.mkdir(exist_ok=True)
    mask_path = workdir / "brainmask.nii"
    nb.Nifti1Image(mask.astype(np.uint8), affine, None).to_filename(mask_path)
    fixed_path, moving_path, init_path = _prepare_registration_data(
        moving,
        fixed,
        affine,
        0,
        workdir,
        clip="both",
        init_affine=init,
        window=robust_window(moving, mask),
    )
    xform = _run_registration(
        fixed_path,
        moving_path,
        0,
        workdir,
        fixedmask_path=mask_path,
        init_affine=init_path,
        output_transform_prefix="ants-00000",
        seed=12345,
        num_threads=1,
    )
    return np.asarray(xform.matrix)


def _print_header(path: Path) -> dict[str, np.ndarray]:
    """The origin, spacing and direction ITK reads from a file (ANTs' ``PrintHeader``)."""
    text = subprocess.run(
        ["PrintHeader", str(path)], capture_output=True, text=True, check=True
    ).stdout
    lines = text.splitlines()
    numbers = {}
    for key in ("Spacing", "Origin"):
        found = re.search(rf"^ {key} \[(.*)\]", text, re.M)
        assert found is not None, key
        numbers[key] = np.array(found.group(1).split(", "), float)
    start = lines.index(" Direction ") + 1
    numbers["Direction"] = np.array([line.split() for line in lines[start : start + 3]], float)
    return numbers


# 1. Coding


@pytest.mark.parametrize("kind", AFFINES)
def test_floats_are_coded_within_half_a_step(volume, tmp_path, kind):
    data, mask = volume
    window = (100.0, 900.0)
    _to_nifti(data, _affine(kind), tmp_path / "coded.nii", window=window)

    img = nb.Nifti1Image.from_filename(tmp_path / "coded.nii")
    step = (window[1] - window[0]) / WINDOW_LEVELS
    assert img.get_data_dtype() == np.int16
    assert np.allclose(img.affine, _affine(kind))
    assert np.abs(img.get_fdata() - np.clip(data, *window)).max() <= step / 2 + 1e-4


def test_integers_are_clipped_and_keep_their_type(tmp_path):
    data = np.arange(64, dtype=np.int16).reshape(4, 4, 4)
    _to_nifti(data, _affine("ras"), tmp_path / "integer.nii", window=(10.5, 50.5))

    img = nb.Nifti1Image.from_filename(tmp_path / "integer.nii")
    assert img.get_data_dtype() == np.int16
    assert np.array_equal(np.asanyarray(img.dataobj), np.clip(data, 11, 50))


@pytest.mark.skipif(NO_ANTS, reason="ANTs is not installed")
def test_floats_without_a_window_reach_ants_as_floats(volume, tmp_path):
    data, _ = volume
    _to_nifti(data, _affine("ras"), tmp_path / "float.nii")

    assert nb.Nifti1Image.from_filename(tmp_path / "float.nii").get_data_dtype() == np.float32
    text = subprocess.run(
        ["PrintHeader", str(tmp_path / "float.nii")], capture_output=True, text=True, check=True
    ).stdout
    found = re.search(r"Intensity Range\s*: \[(.*), (.*)\]", text)
    assert found is not None
    low, high = found.groups()
    assert float(low) == pytest.approx(data.min(), abs=1e-3)
    assert float(high) == pytest.approx(data.max(), rel=1e-5)


# 2. Geometry


@pytest.mark.skipif(NO_ANTS, reason="ANTs is not installed")
@pytest.mark.parametrize("kind", AFFINES)
def test_itk_reads_the_mask_on_the_images_grid(volume, tmp_path, kind):
    """The images (sform and qform codes 1) and the mask (as the estimator writes it)."""
    data, mask = volume
    affine = _affine(kind)
    _prepare_registration_data(data, data, affine, 0, tmp_path, clip="both")
    # As Estimator.run writes it: nibabel's default codes (sform 2, qform 0)
    nb.Nifti1Image(mask.astype(np.uint8), affine, None).to_filename(tmp_path / "mask.nii")

    image = _print_header(tmp_path / "sample_00000.nii")
    for path in ("predicted_00000.nii", "mask.nii"):
        other = _print_header(tmp_path / path)
        for key in ("Spacing", "Origin", "Direction"):
            assert np.allclose(other[key], image[key], atol=1e-4), (path, key)

    # And it is the RAS affine, in ITK's LPS
    lps = np.diag([-1.0, -1.0, 1.0])
    assert np.allclose(image["Origin"], lps @ affine[:3, 3], atol=1e-4)
    assert np.allclose(image["Direction"], lps @ affine[:3, :3] / 2.0, atol=1e-4)


# 3. Reading transforms back


@pytest.mark.skipif(NO_ANTS, reason="ANTs is not installed")
@pytest.mark.parametrize("kind", AFFINES)
def test_ants_applies_the_written_transform_as_nitransforms(volume, tmp_path, kind):
    """The initial transform written for ANTs means to ANTs what it means to nitransforms."""
    data, _ = volume
    affine = _affine(kind)
    matrix = _motion()
    _, sample_path, init_path = _prepare_registration_data(
        data, data, affine, 0, tmp_path, init_affine=matrix
    )
    subprocess.run(
        [
            "antsApplyTransforms",
            *("-d", "3", "-i", str(sample_path), "-r", str(sample_path)),
            *("-t", str(init_path), "-o", str(tmp_path / "ants.nii"), "-n", "Linear"),
        ],
        check=True,
        capture_output=True,
    )

    image = nb.Nifti1Image(data, affine, None)
    expected = np.asanyarray(apply(Affine(matrix), image, reference=image, order=1).dataobj)
    resampled = nb.Nifti1Image.from_filename(tmp_path / "ants.nii").get_fdata()
    inside = np.zeros(SHAPE, dtype=bool)
    inside[6:-6, 6:-6, 6:-6] = True
    assert np.abs(resampled - expected)[inside].max() < 1e-2 * data.max()


# 4. Direction


@pytest.mark.skipif(NO_ANTS, reason="ANTs is not installed")
@pytest.mark.parametrize("kind", AFFINES)
def test_the_estimate_maps_fixed_onto_moving(volume, tmp_path, kind):
    data, mask = volume
    affine = _affine(kind)
    matrix = _motion()
    moving = _move(data, affine, matrix)

    estimated = _register(data, moving, affine, mask, tmp_path)

    assert _error(mask, affine, estimated, matrix) < 0.5
    assert _error(mask, affine, estimated, np.linalg.inv(matrix)) > 2.0


# 5. Initialization


@pytest.mark.skipif(NO_ANTS, reason="ANTs is not installed")
@pytest.mark.parametrize("kind", AFFINES)
def test_an_initialized_estimate_is_the_total_transform(volume, tmp_path, kind):
    """Started from part of the motion, as a second pass is, ANTs returns all of it."""
    data, mask = volume
    affine = _affine(kind)
    matrix = _motion()
    moving = _move(data, affine, matrix)
    partial = _motion(degrees=1.5, shift=(0.75, -0.5, 1.0))

    estimated = _register(data, moving, affine, mask, tmp_path, init=partial)

    assert _error(mask, affine, estimated, matrix) < 0.5
    increment = matrix @ np.linalg.inv(partial)
    assert _error(mask, affine, estimated, increment) > 1.0


def test_an_initial_transform_requires_collapsed_outputs(tmp_path):
    with pytest.raises(ValueError, match="collapse_output_transforms"):
        generate_command(
            tmp_path / "fixed.nii",
            tmp_path / "moving.nii",
            init_affine=tmp_path / "init.mat",
            collapse_output_transforms=False,
        )


# 6. Realignment


@pytest.mark.parametrize("kind", AFFINES)
def test_realignment_undoes_the_motion(volume, kind):
    data, mask = volume
    affine = _affine(kind)
    matrix = _motion()
    moved = _move(data, affine, matrix)
    dataset: BaseDataset = BaseDataset(dataobj=np.stack([data, moved]), affine=affine)
    view = dataset.realigned()

    dataset.set_transform(1, matrix)
    dataset.realign_into(view, 1)

    inside = ndimage.binary_erosion(mask, iterations=3)
    realigned = np.asanyarray(view.dataobj[1])
    assert np.abs(realigned - data)[inside].max() < 0.02 * data.max()
    assert np.abs(moved - data)[inside].max() > 0.1 * data.max()


def _dwi(gradients: np.ndarray, affine: np.ndarray) -> DWI:
    rng = np.random.default_rng(1234)
    return DWI(
        dataobj=rng.uniform(1.0, 2.0, (len(gradients), 6, 6, 6)).astype("float32"),
        affine=affine,
        gradients=gradients,
        bzero=np.ones((6, 6, 6), dtype="float32"),
    )


def _directions(n: int = 8) -> np.ndarray:
    rng = np.random.default_rng(1234)
    directions = rng.normal(size=(n, 3))
    return directions / np.linalg.norm(directions, axis=1, keepdims=True)


def _rotated_in_the_head(matrix: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """The scanner's gradient direction, seen from the head after it moved by ``matrix``."""
    return np.linalg.inv(matrix[:3, :3]) @ direction


@pytest.mark.parametrize(
    "kind",
    [
        "ras",
        pytest.param("phantom", marks=WRONG_AXIS),
        pytest.param("oblique", marks=WRONG_AXIS),
        pytest.param("las", marks=WRONG_AXIS),
    ],
)
def test_ras_bvectors_turn_with_the_head(kind):
    """A table in world (RAS) coordinates, as the phantoms' ``.b`` files give it."""
    affine = _affine(kind)
    world = _directions()
    dwi = _dwi(np.column_stack([world, np.full(len(world), 1000.0)]), affine)
    view = dwi.realigned()
    matrix = _motion(degrees=10.0)

    dwi.set_transform(3, matrix)
    dwi.realign_into(view, 3)

    expected = _rotated_in_the_head(matrix, world[3])
    assert np.allclose(view.gradients[3, :3], expected, atol=1e-6)


def _fsl(world: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """World directions in FSL's convention: voxel axes, x flipped if the affine's
    determinant is positive."""
    voxel = np.linalg.solve(affine[:3, :3], world.T).T
    voxel /= np.linalg.norm(voxel, axis=-1, keepdims=True)
    if np.linalg.det(affine[:3, :3]) > 0:
        voxel[..., 0] *= -1
    return voxel


@pytest.mark.parametrize(
    "kind",
    [
        pytest.param("ras", marks=WRONG_AXIS),
        pytest.param("phantom", marks=WRONG_AXIS),
        pytest.param("oblique", marks=WRONG_AXIS),
        # With a negative determinant, FSL's table is on the voxel axes without a flip
        "las",
    ],
)
def test_fsl_bvectors_turn_with_the_head(kind):
    """A table in FSL's convention, as ``.bvec`` files give it."""
    affine = _affine(kind)
    world = _directions()
    dwi = _dwi(np.column_stack([_fsl(world, affine), np.full(len(world), 1000.0)]), affine)
    view = dwi.realigned()
    matrix = _motion(degrees=10.0)

    dwi.set_transform(3, matrix)
    dwi.realign_into(view, 3)

    expected = _fsl(_rotated_in_the_head(matrix, world[3]), affine)
    assert np.allclose(view.gradients[3, :3], expected, atol=1e-6)


# 7. Roles and threads


def _touch(*paths: Path) -> None:
    """nipype requires the files to exist."""
    for path in paths:
        path.touch()


def test_only_the_fixed_image_is_masked(tmp_path):
    _touch(tmp_path / "fixed.nii", tmp_path / "moving.nii", tmp_path / "mask.nii")
    line = generate_command(
        tmp_path / "fixed.nii", tmp_path / "moving.nii", fixedmask_path=tmp_path / "mask.nii"
    ).cmdline
    masks = re.findall(r"--masks \[ ([^,]+), ([^ ]+) \]", line)
    assert len(masks) == 2
    assert all((Path(fixed).name, moving) == ("mask.nii", "NULL") for fixed, moving in masks)


@pytest.mark.xfail(
    strict=True, reason="nipreps/nifreeze#621: nipype exports only NSLOTS, which ITK reads last"
)
def test_the_callers_threads_hold_over_the_environment(tmp_path, monkeypatch):
    """CI sets ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS for every job (``test.yml``)."""
    monkeypatch.setenv("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "4")
    _touch(tmp_path / "fixed.nii", tmp_path / "moving.nii")
    registration = generate_command(tmp_path / "fixed.nii", tmp_path / "moving.nii", num_threads=1)

    # ITK reads ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS first, and NSLOTS only without it
    environ = dict(os.environ) | dict(registration.inputs.environ)
    threads = environ.get("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", environ.get("NSLOTS"))
    assert threads == "1"
