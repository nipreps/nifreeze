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
"""End-to-end motion estimation on a synthetic phantom with known motion.

The phantom is TRXScan's simulation of the Sherbrooke protocol (``tools/phantoms``,
see ``docs/phantoms.rst``). In its perturbed set at 2° and 2 mm, every
diffusion-weighted volume was simulated again from the head moved by its own
rigid motion (2° about a random axis through the brain's centroid, and 2 mm along
a random direction); the b=0 stays in place and defines the frame of the ground
truth. The protocol's noise is added on load. The gradients are those prescribed,
as a scanner writes them: the moved head turned under them, so even at the right
position a volume differs from the one its model predicts. The truth is the
motion of each volume, which is what the estimator reports. The tests
are opt-in (``pytest -m e2e``), need ANTs, and need ``NIFREEZE_PHANTOMS`` to point
to the phantom dataset (GIN ``tests-nifreeze``, subdataset ``phantoms``).

The model gates record the current single-pass accuracy: the models are fit on
the moving volumes, so their targets mix anatomy at different positions (see
nipreps/nifreeze#497). Tighten them when the instant update is in place.
"""

import json
import shutil
from types import SimpleNamespace

import nibabel as nb
import nitransforms as nt
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from nifreeze.analysis.phantoms import load_phantom
from nifreeze.cli.run import main
from nifreeze.data import dmri
from nifreeze.data.filtering import robust_window
from nifreeze.registration.ants import _prepare_registration_data, _run_registration
from nifreeze.registration.utils import displacements_within_mask

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(shutil.which("antsRegistration") is None, reason="ANTs is not installed"),
]

PROTOCOL = "sherbrooke"
"""The phantom's acquisition protocol: DIPY's Sherbrooke 3-shell."""
MAGNITUDE = 2.0
"""Rotation (degrees) and translation (mm) of every moved volume."""
SEED = 20261020
"""Seed of the noise."""


@pytest.fixture(scope="module")
def phantom_motion(phantoms, tmp_path_factory):
    """The phantom with every diffusion-weighted volume moved, written for the CLI."""
    clean, _ = load_phantom(phantoms, PROTOCOL)
    _, cells = load_phantom(phantoms, PROTOCOL, noise=SEED)
    chosen = [
        (perturbation, load)
        for perturbation, load in cells
        if perturbation.magnitude == MAGNITUDE
        and perturbation.distortion is None
        and perturbation.count == len(clean)
    ]
    if not chosen:
        pytest.skip(f"the phantom moves no copy's every volume by {MAGNITUDE:g}° and mm")
    perturbation, load = chosen[0]
    moved = load()

    work = tmp_path_factory.mktemp("phantom_motion")
    dwi_path = work / "phantom_motion.h5"
    moved.to_filename(dwi_path)
    mask_img = nb.Nifti1Image(np.asarray(clean.brainmask, dtype="uint8"), clean.affine)
    mask_path = work / "brainmask.nii"
    mask_img.to_filename(mask_path)

    # The estimator reports the head's motion, as the sidecars do: the transform from
    # the b=0's frame to each volume's (fixed to moving, in ANTs' terms)
    truth = np.empty((len(clean), 4, 4))
    for index, motion in zip(perturbation.moved, perturbation.motions, strict=True):
        truth[index] = motion

    return SimpleNamespace(
        dwi_path=dwi_path,
        clean=clean,
        moved=moved,
        mask=mask_img,
        mask_path=mask_path,
        affine=clean.affine,
        shells=np.round(clean.gradients[:, 3], -2),
        truth=truth,
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


def test_e2e_registration_oracle(phantom_motion, outdir, tmp_path):
    """Registered to its unmoved original, each volume's motion is recovered.

    The original is noise-free: the noise is keyed on the volume, so a noisy
    original would share the moved volume's noise, voxel for voxel, and pull the
    registration toward no motion.
    """
    sim = phantom_motion
    maskdata = np.asanyarray(sim.mask.dataobj) > 0

    # Four volumes per shell
    picks = np.arange(0, len(sim.clean), 16)
    estimated = []
    for k in picks:
        sample = np.asarray(sim.moved.dataobj[k])
        target = np.asarray(sim.clean.dataobj[k], dtype="float32")
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
    phantom_motion, outdir, tmp_path, model, relative, gated_shells, max_median, max_p95
):
    """The CLI recovers the simulated motion within the current single-pass accuracy."""
    sim = phantom_motion
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    main(
        [
            str(sim.dwi_path),
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
