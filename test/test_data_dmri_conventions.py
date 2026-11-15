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
"""The b-vector conventions read and written at the edges of NiFreeze (#642)."""

import re
import shutil
import subprocess

import nibabel as nb
import numpy as np
import pytest
from dipy.core.sphere import Sphere
from dipy.direction import DeterministicMaximumDirectionGetter
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.stopping_criterion import BinaryStoppingCriterion

from nifreeze.data.dmri.io import RASB_HEADER, load_bvec_bval, load_rasb, save_bvec_bval, save_rasb
from nifreeze.data.dmri.utils import (
    BVEC_FRAMES,
    GRADIENT_EXPECTED_COLUMNS_ERROR_MSG,
    bvecs_from_ras,
    bvecs_to_ras,
    realign_bvecs,
)


def _rotation(axis, degrees):
    """Rotation matrix about ``axis`` (Rodrigues' formula)."""
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    theta = np.radians(degrees)
    return np.eye(3) + np.sin(theta) * k + (1 - np.cos(theta)) * k @ k


AFFINES = {
    "ras": np.diag([2.0, 2.0, 2.0, 1.0]),
    "las": np.diag([-2.0, 2.0, 2.0, 1.0]),
    "phantom": np.diag([-2.0, -2.0, 2.0, 1.0]),  # the phantoms' frame (#620)
    "oblique": nb.affines.from_matvec(
        _rotation((1, 2, 3), 25) @ np.diag([2.5, 2.0, 1.5]), (-90.0, 110.0, -40.0)
    ),
}
"""Affines without shear, which every convention must hold on."""

SHEARED = nb.affines.from_matvec(
    AFFINES["oblique"][:3, :3] @ np.array([[1.0, 0.1, 0.0], [0.0, 1.0, 0.2], [0.0, 0.0, 1.0]]),
    (-90.0, 110.0, -40.0),
)


def _gradients(rng, n=12):
    """A RAS+B table of ``n`` unit b-vectors on two shells, after two b=0 rows."""
    bvecs = rng.normal(size=(n, 3))
    bvecs /= np.linalg.norm(bvecs, axis=1, keepdims=True)
    bvals = np.tile([1000.0, 2000.0], n // 2)
    return np.vstack((np.zeros((2, 4)), np.column_stack((bvecs, bvals))))


@pytest.mark.parametrize("frame", BVEC_FRAMES)
@pytest.mark.parametrize("affine", [*AFFINES.values(), SHEARED], ids=[*AFFINES, "sheared"])
def test_round_trip(request, affine, frame):
    """Converting onto the voxel axes and back returns the b-vectors, norms and zeros."""
    bvecs = _gradients(request.node.rng)[:, :3]
    converted = bvecs_from_ras(bvecs, affine, frame)

    assert np.allclose(np.linalg.norm(converted, axis=1), np.linalg.norm(bvecs, axis=1))
    assert np.all(converted[:2] == 0)
    assert np.allclose(bvecs_to_ras(converted, affine, frame), bvecs, atol=1e-12)


@pytest.mark.parametrize(
    ("name", "voxel", "fsl"),
    [
        ("ras", (1, 1, 1), (-1, 1, 1)),
        ("las", (-1, 1, 1), (-1, 1, 1)),
        ("phantom", (-1, -1, 1), (1, -1, 1)),
    ],
)
def test_axis_aligned(request, name, voxel, fsl):
    """On an axis-aligned affine, each component keeps its axis and may change sign."""
    bvecs = _gradients(request.node.rng)[:, :3]

    assert np.allclose(bvecs_from_ras(bvecs, AFFINES[name], "voxel"), bvecs * voxel)
    assert np.allclose(bvecs_from_ras(bvecs, AFFINES[name], "fsl"), bvecs * fsl)


@pytest.mark.parametrize("name", ["oblique", "anisotropic"])
def test_voxel_sizes_do_not_bend_directions(name):
    """The voxel axes are the affine's columns at unit length, whatever the voxel sizes."""
    affine = AFFINES["oblique"] if name == "oblique" else np.diag([2.0, 1.0, 1.0, 1.0])
    axes = affine[:3, :3] / np.linalg.norm(affine[:3, :3], axis=0)
    voxel = np.array([[1.0, 1.0, 0.0]]) / np.sqrt(2.0)

    assert np.allclose(bvecs_to_ras(voxel, affine, "voxel"), voxel @ axes.T)


def test_unknown_frame():
    """Only the frames in ``BVEC_FRAMES`` are converted."""
    with pytest.raises(ValueError, match="Unknown b-vector frame 'lps'"):
        bvecs_to_ras(np.zeros((1, 3)), np.eye(4), "lps")


@pytest.mark.parametrize("name", AFFINES)
def test_voxel_frame_is_what_dipy_tracks(name):
    """DIPY's tractography follows a voxel-frame direction along its world direction."""
    affine = AFFINES[name]
    world = np.array([0.48, 0.6, 0.64])
    voxel = bvecs_from_ras(world[np.newaxis], affine, "voxel")[0]
    shape = (24, 24, 24)

    sphere = Sphere(xyz=np.vstack((voxel, -voxel, np.eye(3), -np.eye(3))))
    pmf = np.zeros((*shape, len(sphere.vertices)))
    pmf[..., 0] = 1.0
    getter = DeterministicMaximumDirectionGetter.from_pmf(pmf, max_angle=30.0, sphere=sphere)
    seed = nb.affines.apply_affine(affine, np.full(3, 12.0))
    streamline = next(
        iter(
            LocalTracking(
                getter,
                BinaryStoppingCriterion(np.ones(shape)),
                seed[np.newaxis],
                affine,
                step_size=0.5,
                maxlen=10,
            )
        )
    )
    tangent = streamline[-1] - streamline[0]

    assert np.allclose(tangent / np.linalg.norm(tangent), world, atol=1e-6)


@pytest.mark.parametrize("name", AFFINES)
def test_bvec_bval_round_trip(request, tmp_path, name):
    """FSL files are written on the voxel axes and read back into RAS+B."""
    affine = AFFINES[name]
    gradients = _gradients(request.node.rng)
    save_bvec_bval(gradients, affine, tmp_path / "dwi.bvec", tmp_path / "dwi.bval")

    written = np.loadtxt(tmp_path / "dwi.bvec")
    assert written.shape == (3, len(gradients))
    assert np.allclose(written.T, bvecs_from_ras(gradients[:, :3], affine, "fsl"), atol=1e-6)
    assert np.allclose(
        load_bvec_bval(tmp_path / "dwi.bvec", tmp_path / "dwi.bval", affine), gradients, atol=1e-6
    )

    # Three columns are read too
    np.savetxt(tmp_path / "columns.bvec", written.T, fmt="%.6f")
    assert np.allclose(
        load_bvec_bval(tmp_path / "columns.bvec", tmp_path / "dwi.bval", affine),
        gradients,
        atol=1e-6,
    )


def test_bvec_bval_shape_mismatch(tmp_path):
    """The b-vector and b-value files must describe the same number of volumes."""
    np.savetxt(tmp_path / "dwi.bvec", np.zeros((3, 5)))
    np.savetxt(tmp_path / "dwi.bval", np.zeros((1, 4)))

    with pytest.raises(ValueError, match="three rows"):
        load_bvec_bval(tmp_path / "dwi.bvec", tmp_path / "dwi.bval", np.eye(4))


@pytest.mark.parametrize("suffix", [".b", ".tsv"])
def test_rasb_round_trip(request, tmp_path, suffix):
    """RAS+B tables are written as they are and read back."""
    gradients = _gradients(request.node.rng)
    save_rasb(gradients, tmp_path / f"dwi{suffix}")

    first = (tmp_path / f"dwi{suffix}").read_text().splitlines()[0]
    assert first == f"# {RASB_HEADER}" if suffix == ".tsv" else len(first.split()) == 4
    assert np.allclose(load_rasb(tmp_path / f"dwi{suffix}"), gradients, atol=1e-6)


def test_rasb_header_without_comment(request, tmp_path):
    """A first line ``R A S B`` is a header, with or without the comment mark."""
    gradients = _gradients(request.node.rng)
    np.savetxt(tmp_path / "dwi.tsv", gradients, delimiter="\t", header=RASB_HEADER, comments="")

    assert np.allclose(load_rasb(tmp_path / "dwi.tsv"), gradients)


def test_rasb_one_column_per_volume(request, tmp_path):
    """A table written with one column per volume (four rows) is transposed."""
    gradients = _gradients(request.node.rng)
    np.savetxt(tmp_path / "dwi.txt", gradients.T)

    assert np.allclose(load_rasb(tmp_path / "dwi.txt"), gradients)


def test_rasb_columns_error(tmp_path):
    """A RAS+B table has four columns."""
    np.savetxt(tmp_path / "dwi.b", np.zeros((5, 3)))

    with pytest.raises(ValueError, match=re.escape(GRADIENT_EXPECTED_COLUMNS_ERROR_MSG)):
        load_rasb(tmp_path / "dwi.b")


@pytest.mark.skipif(shutil.which("mrinfo") is None, reason="MRtrix is not installed")
@pytest.mark.parametrize("name", AFFINES)
def test_bvec_bval_matches_mrtrix(request, tmp_path, name):
    """The FSL reader agrees with MRtrix's import of the same files (``-fslgrad``)."""
    affine = AFFINES[name]
    gradients = _gradients(request.node.rng)
    img = nb.Nifti1Image(np.zeros((2, 2, 2, len(gradients)), dtype="float32"), affine)
    img.set_qform(affine, code=1)
    img.set_sform(affine, code=1)
    img.to_filename(tmp_path / "dwi.nii.gz")
    save_bvec_bval(gradients, affine, tmp_path / "dwi.bvec", tmp_path / "dwi.bval")

    subprocess.run(
        [
            "mrinfo",
            str(tmp_path / "dwi.nii.gz"),
            "-fslgrad",
            str(tmp_path / "dwi.bvec"),
            str(tmp_path / "dwi.bval"),
            "-export_grad_mrtrix",
            str(tmp_path / "dwi.b"),
        ],
        check=True,
        capture_output=True,
    )

    assert np.allclose(load_rasb(tmp_path / "dwi.b"), gradients, atol=1e-5)


@pytest.mark.parametrize("degrees", [0.0, 10.0, 90.0])
def test_realign_bvecs_takes_the_rotation(request, degrees):
    """Only the motion's rotation turns b-vectors: not its translation, scaling or shear."""
    gradients = _gradients(request.node.rng)[2:]
    rotation = _rotation((1, -2, 0.5), degrees)
    rigid = nb.affines.from_matvec(rotation, (5.0, -3.0, 2.0))
    # A symmetric positive-definite stretch, which the polar decomposition separates
    stretched = nb.affines.from_matvec(
        rotation @ np.array([[1.1, 0.05, 0.0], [0.05, 0.95, 0.02], [0.0, 0.02, 1.0]]),
        (5.0, -3.0, 2.0),
    )

    expected = gradients[:, :3] @ np.linalg.inv(rotation).T
    assert np.allclose(realign_bvecs(gradients[:, :3], rigid), expected)
    assert np.allclose(realign_bvecs(gradients[:, :3], stretched), expected)
    assert np.allclose(np.linalg.norm(realign_bvecs(gradients[:, :3], stretched), axis=1), 1.0)
