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
are opt-in (``pytest -m e2e``) and need ``NIFREEZE_PHANTOMS`` to point
to the phantom dataset (GIN ``nipreps-data/nifreeze-trxscan``).

A second copy adds the eddy currents of the phantom's smallest set (``desc-e2``).
After its motion, each volume was formed by an EPI readout under a field linear in
position, which shows at :math:`y - d(y) u` what lies at :math:`y` (:math:`u` is the
phase-encoding direction, and :math:`d` the sidecar's displacement, up to 4 mm within
the mask). Each volume thus shows at :math:`M x - d(M x) u` what the reference shows
at :math:`x`, as the audit of nipreps/nifreeze#626 checked. Since :math:`d` is
linear, that is the affine :math:`E M`, with :math:`M` the head motion and :math:`E`
the eddy currents. A cascade of models and then an eddy-current pass, through the
command line, must recover the two apart (nipreps/nifreeze#712), and so must the
default run: head motion, then three passes of GEM, the global eddy model.

The tests register with the backend that ``NIFREEZE_E2E_REGISTRATION`` names (the CI
job's), or with the default one, and skip if it is not installed. CI registers with
elastix on every run, and with ANTs on its weekly schedule (nipreps/nifreeze#718).

The model gates record the current single-pass accuracy, as set on ANTs: the models
are fit on the moving volumes, so their targets mix anatomy at different positions
(see nipreps/nifreeze#497). Tighten them when the instant update is in place.
The cascade's head motion must pass the gate of its last model on head motion without
eddy currents, and the eddy-current correction must improve on the head motion it starts from.
The gates of the eddy currents and of the whole mapping record their accuracy on both backends.
"""

import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import nibabel as nb
import nitransforms as nt
import numpy as np
import pytest
from nitransforms.linear import LinearTransformsMapping
from nitransforms.manip import TransformChain
from scipy.spatial.transform import Rotation

from nifreeze.analysis.phantoms import SUBJECT, load_phantom
from nifreeze.cli.parser import DEFAULT_GEM_PASSES
from nifreeze.cli.run import main
from nifreeze.data import dmri
from nifreeze.registration.base import RegistrationFactory, default_backend
from nifreeze.registration.utils import displacements_within_mask

REGISTRATION = os.getenv("NIFREEZE_E2E_REGISTRATION") or default_backend()
"""The backend the tests register with."""
MISSING = {
    "ants": shutil.which("antsRegistration") is None,
    "elastix": default_backend() != "elastix",
}
"""Whether each backend is missing: ANTs from ``PATH``, or the ``elastix`` extra."""

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        MISSING.get(REGISTRATION, False), reason=f"{REGISTRATION} is not installed"
    ),
]

PROTOCOL = "sherbrooke"
"""The phantom's acquisition protocol: DIPY's Sherbrooke 3-shell."""
MAGNITUDE = 2.0
"""Rotation (degrees) and translation (mm) of every moved volume."""
SEED = 20261020
"""Seed of the noise."""
ORACLE_GATE = {"max_median": 0.45, "max": 0.6}
"""Largest median and maximum error (mm) of the registrations to the unmoved original."""
EDDY_MM = 4.0
"""Largest displacement (mm) of the eddy currents within the mask (``desc-e2``)."""
CASCADE = ("gqi-krr", "gp-loo")
"""The models of the cascade. The eddy-current pass repeats the last one."""
CASCADE_GATE = {
    # The gate of the cascade's last model on head motion alone
    "head_motion": {"max_median": 1.55, "max_p95": 2.5},
    # Set on both backends (worst: ANTs, 1.47 / 2.38 and 1.34 / 2.14)
    "eddy_currents": {"max_median": 1.85, "max_p95": 3.0},
    "whole": {"max_median": 1.7, "max_p95": 2.7},
}
"""Largest median and 95th percentile (mm) of each score of the cascade, over all shells."""
DEFAULT_MODEL = "gqi-krr"
"""The model of a run that names none, for head motion and GEM's passes."""
DEFAULT_GATE = {
    # The gate of the default model on head motion alone
    "head_motion": {"max_median": 1.35, "max_p95": 2.75},
    # Set on both backends (worst: ANTs, 0.55 / 1.32 and 1.19 / 2.69)
    "eddy_currents": {"max_median": 0.7, "max_p95": 1.65},
    "whole": {"max_median": 1.5, "max_p95": 3.4},
}
"""Largest median and 95th percentile (mm) of each score of the default run, over all shells."""


@pytest.fixture(scope="module")
def phantom_copies(phantoms):
    """The noise-free reference, and the noisy copies (each loads when asked)."""
    clean, _ = load_phantom(phantoms, PROTOCOL)
    _, cells = load_phantom(phantoms, PROTOCOL, noise=SEED)
    return clean, cells


def _choose(clean, cells, eddy_mm: float | None):
    """The copy with every volume moved, and with the eddy currents of ``eddy_mm``."""
    chosen = [
        (perturbation, load)
        for perturbation, load in cells
        if perturbation.magnitude == MAGNITUDE
        and perturbation.count == len(clean)
        and (None if perturbation.distortion is None else perturbation.distortion.magnitude)
        == eddy_mm
    ]
    if not chosen:
        eddy = "" if eddy_mm is None else f", with eddy currents of {eddy_mm:g} mm"
        pytest.skip(f"the phantom moves no copy's every volume by {MAGNITUDE:g}° and mm{eddy}")
    return chosen[0]


def _write(clean, moved, perturbation, work: Path, name: str) -> SimpleNamespace:
    """A copy written for the CLI, with its brain mask and the head motion of each volume."""
    dwi_path = work / f"{name}.h5"
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
        mask=mask_img,
        mask_path=mask_path,
        affine=clean.affine,
        shells=np.round(clean.gradients[:, 3], -2),
        truth=truth,
    )


@pytest.fixture(scope="module")
def phantom_motion(phantom_copies, tmp_path_factory):
    """The phantom with every diffusion-weighted volume moved, written for the CLI."""
    clean, cells = phantom_copies
    perturbation, load = _choose(clean, cells, None)
    moved = load()
    sim = _write(
        clean, moved, perturbation, tmp_path_factory.mktemp("phantom_motion"), "phantom_motion"
    )
    sim.moved = moved
    return sim


@pytest.fixture(scope="module")
def phantom_eddy(phantoms, phantom_copies, tmp_path_factory):
    """The moved phantom with eddy currents too, written for the CLI, and their truth."""
    clean, cells = phantom_copies
    perturbation, load = _choose(clean, cells, EDDY_MM)
    distortion = perturbation.distortion
    assert distortion is not None
    moved = load()
    # As a BIDS sidecar gives it, for GEM: the reference's
    sidecar = phantoms / f"sub-{SUBJECT}" / "dwi" / f"sub-{SUBJECT}_acq-{PROTOCOL}_dwi.json"
    moved.phase_encoding = json.loads(sidecar.read_text())["PhaseEncodingDirection"]
    sim = _write(
        clean, moved, perturbation, tmp_path_factory.mktemp("phantom_eddy"), "phantom_eddy"
    )
    # Exact for the phantom's field, of degree 1 (raises for higher degrees)
    sim.eddy = distortion.matrices()
    sim.whole = sim.eddy @ sim.truth
    return sim


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
    # The sample's window clips both images, as the estimator does
    options = {"workdir": tmp_path} if REGISTRATION == "ants" else {}
    registration = RegistrationFactory.init(REGISTRATION, seed=12345, **options)

    # Four volumes per shell
    picks = np.arange(0, len(sim.clean), 16)
    estimated = []
    for k in picks:
        sample = np.asarray(sim.moved.dataobj[k])
        target = np.asarray(sim.clean.dataobj[k], dtype="float32")
        estimated.append(registration(target, sample, sim.affine, maskdata, int(k)).matrix)

    errors = _errors(sim.mask, np.stack(estimated), sim.truth[picks])
    result = {
        "model": "oracle",
        "registration": REGISTRATION,
        "errors": _summary(errors, sim.shells[picks]),
        "gate": ORACLE_GATE,
    }
    _record(outdir, "oracle", result)

    assert np.median(errors) < ORACLE_GATE["max_median"]
    assert errors.max() < ORACLE_GATE["max"]


@pytest.mark.parametrize(
    ("model", "relative", "gated_shells", "max_median", "max_p95"),
    [
        # Registering to the b=0 only works for low b-values: higher shells are reported
        ("trivial", False, ("1000",), 3.45, 4.6),
        ("gp-loo", True, None, 1.55, 2.5),
        ("gqi-krr", True, None, 1.35, 2.75),
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
            # Head motion alone (the copy has no eddy currents, nor its direction)
            "--gem-passes",
            "0",
            "--backend",
            REGISTRATION,
            "--seed",
            "12345",
            "--output-dir",
            str(output_dir),
            "--write-hdf5",
        ]
    )
    estimated = dmri.DWI.from_filename(next(output_dir.glob("*.h5"))).motion
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
            "registration": REGISTRATION,
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


def _run_eddy(sim: SimpleNamespace, tmp_path: Path, *argv: str) -> tuple[np.ndarray, np.ndarray]:
    """Run the CLI on the eddy copy, and read back its head motion and eddy currents."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    main(
        [
            str(sim.dwi_path),
            "--brainmask",
            str(sim.mask_path),
            *argv,
            "--backend",
            REGISTRATION,
            "--seed",
            "12345",
            "--output-dir",
            str(output_dir),
            "--write-hdf5",
        ]
    )
    dataset = dmri.DWI.from_filename(next(output_dir.glob("*.h5")))
    motion, eddy = dataset.motion, dataset.eddy
    assert motion is not None
    assert eddy is not None

    # The X5 file carries them apart too: head motion first, then the eddy currents
    x5 = next(output_dir.glob("*.x5"))
    assert np.allclose(LinearTransformsMapping.from_filename(x5, fmt="X5").matrix, motion)
    chain = TransformChain.from_filename(x5)
    assert np.allclose(chain.transforms[1].matrix @ chain.transforms[0].matrix, eddy @ motion)
    return motion, eddy


def _check_eddy(
    sim: SimpleNamespace,
    motion: np.ndarray,
    eddy: np.ndarray,
    outdir,
    name: str,
    record: dict,
    gates: dict,
) -> None:
    """Score head motion and eddy currents against their truth, record them, and gate them."""
    whole = eddy @ motion
    # Each estimate against its truth, without the transform its volumes share (the
    # frame of the model's targets). That frame cancels from the eddy currents
    # (whole @ inv(motion)), which are evaluated where the head is.
    head_motion, head_offset = _remove_offset(motion, sim.truth)
    corrected, whole_offset = _remove_offset(whole, sim.whole)
    started, _ = _remove_offset(motion, sim.whole)
    scores = {
        "head_motion": _summary(_errors(sim.mask, head_motion, sim.truth), sim.shells),
        "eddy_currents": _summary(_errors(sim.mask, eddy @ sim.truth, sim.whole), sim.shells),
        "whole": _summary(_errors(sim.mask, corrected, sim.whole), sim.shells),
        # The head motion's mapping alone, as a run without eddy-current correction gives it
        "without_eddy": _summary(_errors(sim.mask, started, sim.whole), sim.shells),
    }
    identity = np.stack([np.eye(4)] * len(sim.truth))
    applied = {
        "whole": _summary(_errors(sim.mask, identity, sim.whole), sim.shells),
        "eddy_currents": _summary(_errors(sim.mask, sim.truth, sim.whole), sim.shells),
    }
    _record(
        outdir,
        name,
        {
            "kind": "cascade",
            **record,
            "registration": REGISTRATION,
            "offset_mm": {
                name: float(displacements_within_mask(sim.mask, nt.linear.Affine(common)).max())
                for name, common in (("head_motion", head_offset), ("whole", whole_offset))
            },
            "applied": applied,
            **scores,
            "gate": gates,
        },
    )

    for score, gate in gates.items():
        assert scores[score]["all"]["median"] < gate["max_median"], score
        assert scores[score]["all"]["p95"] < gate["max_p95"], score
    # The eddy-current correction improves on the head motion it starts from
    for stat in ("median", "p95"):
        assert scores["whole"]["all"][stat] < scores["without_eddy"]["all"][stat], stat


def test_e2e_cascade_eddy(phantom_eddy, outdir, tmp_path):
    """A cascade, then a per-volume eddy-current pass, recover head motion and eddy currents."""
    motion, eddy = _run_eddy(
        phantom_eddy, tmp_path, "--models", *CASCADE, "--gem-passes", "0", "--eddy"
    )
    _check_eddy(
        phantom_eddy,
        motion,
        eddy,
        outdir,
        "cascade-eddy",
        {
            "title": "Cascade, then an eddy-current pass",
            "models": [*CASCADE, f"{CASCADE[-1]} (eddy currents, each volume on its own)"],
        },
        CASCADE_GATE,
    )


def test_e2e_default_eddy(phantom_eddy, outdir, tmp_path):
    """The default run, head motion then GEM's passes, recovers both apart."""
    motion, eddy = _run_eddy(phantom_eddy, tmp_path)
    _check_eddy(
        phantom_eddy,
        motion,
        eddy,
        outdir,
        "default-eddy",
        {
            "title": "The default run: head motion, then GEM",
            "models": [DEFAULT_MODEL, f"GEM × {DEFAULT_GEM_PASSES} ({DEFAULT_MODEL})"],
        },
        DEFAULT_GATE,
    )
