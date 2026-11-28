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
"""The command line's default run on the phantom with the largest motion and eddy currents.

The copy is the Sherbrooke protocol's ``desc-m4s0e8`` (``docs/edd/phantoms.rst``): every
diffusion-weighted volume turned by 4° and shifted by 4 mm, and distorted by eddy
currents that move some voxel of the brain by 8 voxels (16 mm) along phase encoding,
noise-free as simulated. It is written as the command line's input, with the protocol's
phase encoding, and ``nifreeze`` runs on it with its defaults (head motion, then three
passes of GEM). Then measures how far each volume's corrections move the brain, against
the truth, and animates how the run corrects each volume.

Writes, in ``<work>``: the copy (``phantom.h5``) and its brain mask, the run's outputs,
log and time (``default/``, ``default.log``, ``times.json``), and ``sizes.tsv`` (one row
per diffusion-weighted volume); in ``<out>``, ``corrections-phantom.svg`` (made by
:func:`nifreeze.viz.corrections.animate_corrections`; about 19 MB, kept on GIN and
fetched by the documentation's build).
On CPUs 0-5::

    taskset -c 0-5 python tools/gem_phantom.py --dataset <phantoms> --work <work> \\
        --out docs/_static/gem

``--figure-only`` draws the animation again from the run in ``<work>``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import nibabel as nb
import nitransforms as nt
import numpy as np

from nifreeze.analysis.phantoms import SUBJECT, load_phantom
from nifreeze.analysis.sensitivity import Perturbation
from nifreeze.data import dmri
from nifreeze.registration.gem import field_of_view_centre
from nifreeze.registration.utils import displacements_within_mask
from nifreeze.viz.corrections import _split, animate_corrections

PROTOCOL = "sherbrooke"
"""The phantom's protocol."""
MOTION, EDDY = 4.0, 16.0
"""The copy's head motion (degrees and mm) and the largest displacement (mm) of its eddy
currents."""
MAGNIFY = (1.0, 2.5)
"""How many times longer the arrows are than head motion and eddy currents move the
voxels."""
SEED = 12345
"""Seed of the run."""
COLUMNS = ("motion", "motion_truth", "eddy", "eddy_truth", "error")
"""What ``sizes.tsv`` measures, each by its median and its largest value in the brain."""


def copy(dataset: Path, work: Path) -> tuple[Perturbation, Path]:
    """The copy and its truth, with the reference's brain mask (written once)."""
    clean, cells = load_phantom(dataset, PROTOCOL)
    (truth, load), *_ = [
        (p, load)
        for p, load in cells
        if p.magnitude == MOTION
        and p.count == len(clean)
        and p.distortion is not None
        and p.distortion.magnitude == EDDY
    ]
    mask = work / "brainmask.nii.gz"
    if not (work / "phantom.h5").exists():
        moved = load()
        stem = dataset / f"sub-{SUBJECT}" / "dwi" / f"sub-{SUBJECT}_acq-{PROTOCOL}"
        sidecar = json.loads(Path(f"{stem}_dwi.json").read_text())
        moved.phase_encoding = sidecar["PhaseEncodingDirection"]
        moved.to_filename(work / "phantom.h5")
        nb.Nifti1Image(np.asarray(clean.brainmask, dtype="uint8"), clean.affine).to_filename(mask)
    return truth, mask


def run(work: Path, mask: Path) -> None:
    """The command line's default run, with its log and its time."""
    (work / "default").mkdir(parents=True, exist_ok=True)
    command = [
        *(sys.executable, "-m", "nifreeze.cli.run", str(work / "phantom.h5")),
        *("--brainmask", str(mask)),
        *("--seed", str(SEED), "--write-hdf5", "--output-dir", str(work / "default")),
    ]
    start = time.monotonic()
    with (work / "default.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    times = {"default": time.monotonic() - start}
    print(f"default: {times['default'] / 60:.1f} min")
    (work / "times.json").write_text(json.dumps(times, indent=2) + "\n")


def result(work: Path) -> dmri.DWI:
    """The run's output."""
    return dmri.DWI.from_filename(next((work / "default").glob("*.h5")))


def sizes(work: Path, mask_path: Path, truth: Perturbation) -> np.ndarray:
    """
    How far each volume's corrections move the brain (mm), written to ``sizes.tsv``.

    For each diffusion-weighted volume: its b-value, then the median and the largest
    displacement within the brain of the head motion, estimated and true; of the eddy
    currents, estimated and true; and between the estimated and the true whole mappings.
    Both are split about the centre of the field of view, as the animation splits them.
    """
    dataset = result(work)
    assert dataset.motion is not None and dataset.eddy is not None
    assert truth.distortion is not None
    true_motion = np.repeat(np.eye(4)[None], len(dataset), 0)
    true_motion[list(truth.moved)] = truth.motions
    true_field = truth.distortion.matrices()
    mask = nb.Nifti1Image.from_filename(mask_path)
    point = field_of_view_centre(dataset.affine, dataset.shape3d)

    def shift(test: np.ndarray, reference: np.ndarray | None = None) -> np.ndarray:
        reference_xfm = None if reference is None else nt.linear.Affine(reference)
        return displacements_within_mask(mask, nt.linear.Affine(test), reference_xfm)

    rows = []
    for k in range(len(dataset)):
        head, _ = _split(dataset.motion[k], dataset.eddy[k], point)
        true_head, _ = _split(true_motion[k], true_field[k], point)
        whole = dataset.eddy[k] @ dataset.motion[k]
        true_whole = true_field[k] @ true_motion[k]
        parts = (
            shift(head),
            shift(true_head),
            shift(whole, head),
            shift(true_whole, true_head),
            shift(whole, true_whole),
        )
        values = [f(d) for d in parts for f in (np.median, np.max)]
        rows.append([dataset.gradients[k, -1], *values])
    table = np.array(rows)
    header = "\t".join(["b", *(f"{c}_{s}" for c in COLUMNS for s in ("median", "max"))])
    np.savetxt(work / "sizes.tsv", table, fmt="%.3f", delimiter="\t", header=header)
    return table


def animation(work: Path, out: Path) -> Path:
    """How the default run corrects each volume, animated."""
    path = out / "corrections-phantom.svg"
    animate_corrections(result(work), path, magnify=MAGNIFY)
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, required=True, help="phantom dataset root")
    parser.add_argument("--work", type=Path, required=True, help="the run's folder")
    parser.add_argument("--out", type=Path, required=True, help="the animation's folder")
    parser.add_argument("--figure-only", action="store_true", help="reuse the run")
    args = parser.parse_args(argv)

    args.work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    truth, mask = copy(args.dataset, args.work)
    if not args.figure_only:
        run(args.work, mask)
    table = sizes(args.work, mask, truth)
    names = ("Head motion", "Head motion, true", "Eddy currents", "Eddy currents, true")
    for c, name in enumerate((*names, "Error of the whole mapping")):
        print(
            f"{name}: {np.median(table[:, 2 * c + 1]):.2f} mm in the median volume (median "
            f"in the brain), at most {table[:, 2 * c + 2].max():.2f} mm"
        )
    path = animation(args.work, args.out)
    print(f"{path}: {path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
