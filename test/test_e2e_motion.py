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
"""End-to-end motion estimation on DIPY's Sherbrooke dataset with simulated motion.

Every diffusion-weighted volume is moved by a seeded random rigid transform; the
b=0 stays in place and defines the frame of the ground truth. B-vectors are not
rotated. The tests are opt-in (``pytest -m e2e``) and need ANTs.

The model gates record the current single-pass accuracy: the models are fit on
the moving volumes, so their targets mix anatomy at different positions (see
nipreps/nifreeze#497). Tighten them when motion estimates are fed back.
"""

import json
import shutil
from types import SimpleNamespace

import nibabel as nb
import nitransforms as nt
import numpy as np
import pytest
from nitransforms.resampling import apply
from scipy.spatial.transform import Rotation

from nifreeze.cli.run import main
from nifreeze.data import dmri
from nifreeze.data.filtering import robust_window
from nifreeze.registration.ants import _prepare_registration_data, _run_registration
from nifreeze.registration.utils import displacements_within_mask
from nifreeze.utils.ndimage import load_api

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(shutil.which("antsRegistration") is None, reason="ANTs is not installed"),
]

SEED = 20261020
"""Seed of the simulated motion."""
MAX_TRANSLATION_MM = 2.0
"""Largest translation along each axis."""
MAX_ROTATION_DEG = 2.0
"""Largest rotation about each axis (about the centre of the field of view)."""
LOW_B = 50
"""Volumes below this b-value are references and are not moved."""


@pytest.fixture(scope="module")
def sherbrooke_motion(tmp_path_factory):
    """Sherbrooke with a seeded random rigid motion on every diffusion-weighted volume."""
    from dipy.data import get_fnames
    from dipy.segment.mask import median_otsu

    dwi_fname, bval_fname, bvec_fname = get_fnames(name="sherbrooke_3shell")
    work = tmp_path_factory.mktemp("sherbrooke_motion")

    img = load_api(dwi_fname, nb.Nifti1Image)
    data = np.asanyarray(img.dataobj)
    bvals = np.loadtxt(bval_fname)
    moving = np.flatnonzero(bvals >= LOW_B)

    _, mask = median_otsu(data[..., bvals < LOW_B].mean(axis=-1), median_radius=2, numpass=1)
    mask_img = nb.Nifti1Image(mask.astype("uint8"), img.affine)
    mask_path = work / "brainmask.nii"
    mask_img.to_filename(mask_path)

    rng = np.random.default_rng(SEED)
    center = nb.affines.apply_affine(img.affine, (np.array(img.shape[:3]) - 1) / 2)
    moved = data.copy()
    truth = []
    for i in moving:
        angles = rng.uniform(-MAX_ROTATION_DEG, MAX_ROTATION_DEG, 3)
        rotation = Rotation.from_euler("xyz", angles, degrees=True)
        motion = np.eye(4)
        motion[:3, :3] = rotation.as_matrix()
        motion[:3, 3] = (
            center
            - rotation.apply(center)
            + rng.uniform(-MAX_TRANSLATION_MM, MAX_TRANSLATION_MM, 3)
        )
        volume = nb.Nifti1Image(data[..., i].astype("float32"), img.affine)
        resampled = np.asanyarray(
            apply(nt.linear.Affine(motion), volume, reference=volume, order=3).dataobj
        )
        moved[..., i] = np.clip(np.round(resampled), 0, np.iinfo(data.dtype).max)
        # The estimator reports the transform that undoes the motion
        truth.append(np.linalg.inv(motion))

    dwi_path = work / "sherbrooke_motion.nii"
    nb.Nifti1Image(moved, img.affine, img.header).to_filename(dwi_path)

    return SimpleNamespace(
        dwi_path=dwi_path,
        original_path=dwi_fname,
        bvec_path=bvec_fname,
        bval_path=bval_fname,
        mask=mask_img,
        mask_path=mask_path,
        affine=img.affine,
        moving=moving,
        shells=np.round(bvals[moving], -2),
        truth=np.stack(truth),
    )


def _errors(mask: nb.Nifti1Image, estimated: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Largest displacement (mm) within the mask between estimated and true transforms."""
    return np.array(
        [
            displacements_within_mask(mask, nt.linear.Affine(est), nt.linear.Affine(ref)).max()
            for est, ref in zip(estimated, truth, strict=True)
        ]
    )


def _remove_offset(estimated: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Remove the transform shared by all the estimates (the frame of the model's targets)."""
    offsets = np.array(
        [np.linalg.inv(ref) @ est for est, ref in zip(estimated, truth, strict=True)]
    )
    common = np.eye(4)
    common[:3, :3] = Rotation.from_matrix(offsets[:, :3, :3]).mean().as_matrix()
    common[:3, 3] = offsets[:, :3, 3].mean(axis=0)
    return np.array([est @ np.linalg.inv(common) for est in estimated]), common


def _summary(errors: np.ndarray, shells: np.ndarray) -> dict:
    """Median, 95th percentile and maximum, overall and per shell."""

    def stats(values):
        return {
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "max": float(values.max()),
        }

    return {
        "all": stats(errors),
        "shells": {str(int(shell)): stats(errors[shells == shell]) for shell in np.unique(shells)},
    }


def _record(outdir, name: str, result: dict) -> None:
    """Write the result where the CI job collects its summary."""
    print(json.dumps(result))
    if outdir is not None:
        (outdir / "e2e").mkdir(parents=True, exist_ok=True)
        (outdir / "e2e" / f"{name}.json").write_text(json.dumps(result, indent=2))


def test_e2e_registration_oracle(sherbrooke_motion, outdir, tmp_path):
    """Registered to its unmoved original, each volume's motion is recovered."""
    sim = sherbrooke_motion
    original = load_api(sim.original_path, nb.Nifti1Image)
    moved = load_api(sim.dwi_path, nb.Nifti1Image)
    maskdata = np.asanyarray(sim.mask.dataobj) > 0

    # Four volumes per shell
    picks = np.arange(0, len(sim.moving), 16)
    estimated = []
    for k in picks:
        sample = np.asanyarray(moved.dataobj[..., sim.moving[k]])
        target = np.asanyarray(original.dataobj[..., sim.moving[k]]).astype("float32")
        fixed_path, moving_path, _ = _prepare_registration_data(
            sample,
            target,
            sim.affine,
            int(k),
            tmp_path,
            clip="both",
            window=robust_window(sample, maskdata),
        )
        xform = _run_registration(
            fixed_path,
            moving_path,
            int(k),
            tmp_path,
            fixedmask_path=sim.mask_path,
            output_transform_prefix=f"ants-{k:05d}",
            seed=12345,
        )
        estimated.append(xform.matrix)

    errors = _errors(sim.mask, np.stack(estimated), sim.truth[picks])
    result = {"model": "oracle", "errors": _summary(errors, sim.shells[picks])}
    _record(outdir, "oracle", result)

    assert np.median(errors) < 0.2
    assert errors.max() < 1.0


@pytest.mark.parametrize(
    ("model", "relative", "gated_shells", "max_median", "max_p95"),
    [
        # Registering to the b=0 only works for low b-values: higher shells are reported
        ("trivial", False, ("1000",), 2.0, 3.0),
        ("gp-loo", True, None, 2.5, 4.5),
        ("gqi-krr", True, None, 2.5, 5.5),
    ],
    ids=["trivial", "gp-loo", "gqi-krr"],
)
def test_e2e_motion(
    sherbrooke_motion, outdir, tmp_path, model, relative, gated_shells, max_median, max_p95
):
    """The CLI recovers the simulated motion within the current single-pass accuracy."""
    sim = sherbrooke_motion
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    main(
        [
            str(sim.dwi_path),
            "--gradient-file",
            str(sim.bvec_path),
            str(sim.bval_path),
            "--brainmask",
            str(sim.mask_path),
            "--models",
            model,
            "--seed",
            "12345",
            "--output-dir",
            str(output_dir),
            "--write-hdf5",
        ]
    )
    estimated = dmri.DWI.from_filename(next(output_dir.glob("*.h5"))).motion_affines
    assert estimated is not None

    corrected, common = _remove_offset(estimated, sim.truth)
    offset_mm = float(displacements_within_mask(sim.mask, nt.linear.Affine(common)).max())
    absolute = _summary(_errors(sim.mask, estimated, sim.truth), sim.shells)
    relative_errors = _summary(_errors(sim.mask, corrected, sim.truth), sim.shells)
    motion = _summary(
        _errors(sim.mask, np.stack([np.eye(4)] * len(sim.truth)), sim.truth), sim.shells
    )
    _record(
        outdir,
        model,
        {
            "model": model,
            "offset_mm": offset_mm,
            "motion": motion,
            "absolute": absolute,
            "relative": relative_errors,
            "gate": {
                "errors": "relative" if relative else "absolute",
                "shells": list(gated_shells) if gated_shells else "all",
                "max_median": max_median,
                "max_p95": max_p95,
            },
        },
    )

    gated = relative_errors if relative else absolute
    for stats in (
        [gated["shells"][shell] for shell in gated_shells] if gated_shells else [gated["all"]]
    ):
        assert stats["median"] < max_median
        assert stats["p95"] < max_p95
