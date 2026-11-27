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
"""The command line's default run on real data: DIPY's Sherbrooke 3-shell data.

Runs ``nifreeze`` twice on the data, with phase encoding along ``j`` (as
nipreps/nifreeze#752 took it): with ``--gem-passes 0`` (head motion only), and with the
defaults (head motion, then three passes of GEM). Then measures how far each volume's
corrections move the brain, and plays the volumes, original and corrected, in fixed
axial and sagittal cuts under the brain mask's outline: an SVG with a CSS animation, as
NiReports' reportlets do.

Writes, in ``<work>``: the brain mask (``median_otsu`` on the b=0), each run's outputs,
log and time (``motion/``, ``default/``, ``*.log``, ``times.json``), and ``sizes.tsv``
(one row per diffusion-weighted volume); in ``<out>``, ``gem-sherbrooke.svg`` and
``corrections-sherbrooke.svg``, how the default run corrects each volume (made by
:func:`nifreeze.viz.corrections.animate_corrections`; about 26 MB, kept on GIN and
fetched by the documentation's build).
On CPUs 0-5::

    taskset -c 0-5 python tools/gem_sherbrooke.py --work <work> --out docs/_static/gem

``--figure-only`` draws the figure again from the runs in ``<work>``.
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import time
from io import BytesIO
from pathlib import Path

import nibabel as nb
import nitransforms as nt
import numpy as np
from PIL import Image
from skimage.measure import find_contours

from nifreeze.data import dmri
from nifreeze.registration.utils import displacements_within_mask
from nifreeze.viz.corrections import animate_corrections

PHASE_ENCODING = "j"
"""The phase-encoding direction the runs are given (the data have no sidecar)."""
RUNS = {"motion": ["--gem-passes", "0"], "default": []}
"""Each run's options besides the inputs: head motion only, and the defaults."""
LABELS = ("Original", "Head motion", "Head motion and eddy currents")
"""The columns of the figure."""
SEED = 12345
"""Seed of both runs."""
MARGIN, SCALE, GAP, SECONDS, QUALITY = 3, 3, 2, 0.2, 70
"""Voxels around the brain, pixels per voxel, pixels between panels, seconds per frame,
and JPEG quality."""


def inputs() -> tuple[Path, Path, Path]:
    """DIPY's Sherbrooke data: the DWI, its b-vectors and its b-values (fetched once)."""
    import dipy.data as dpd

    dwi, bval, bvec = dpd.get_fnames(name="sherbrooke_3shell")
    return Path(dwi), Path(bvec), Path(bval)


def brain_mask(dwi: Path, bval: Path, path: Path) -> Path:
    """A brain mask from ``median_otsu`` on the b=0 volumes (written once)."""
    from dipy.segment.mask import median_otsu

    if not path.exists():
        img = nb.Nifti1Image.from_filename(dwi)
        b0 = np.flatnonzero(np.loadtxt(bval) < 50).tolist()
        _, mask = median_otsu(np.asanyarray(img.dataobj), vol_idx=b0)
        nb.Nifti1Image(mask.astype("uint8"), img.affine).to_filename(path)
    return path


def run(dwi: Path, bvec: Path, bval: Path, mask: Path, work: Path) -> None:
    """Both runs of the command line, each with its log, and their times."""
    times = {}
    for name, options in RUNS.items():
        (work / name).mkdir(parents=True, exist_ok=True)
        command = [
            *(sys.executable, "-m", "nifreeze.cli.run", str(dwi)),
            *("--gradient-file", str(bvec), str(bval)),
            *("--brainmask", str(mask), "--phase-encoding", PHASE_ENCODING),
            *options,
            *("--seed", str(SEED), "--write-hdf5", "--output-dir", str(work / name)),
        ]
        start = time.monotonic()
        with (work / f"{name}.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        times[name] = time.monotonic() - start
        print(f"{name}: {times[name] / 60:.1f} min")
    (work / "times.json").write_text(json.dumps(times, indent=2) + "\n")


def output(work: Path, name: str, suffix: str) -> Path:
    """The file of a run's outputs with this suffix."""
    return next((work / name).glob(f"*{suffix}"))


def displacements(mask, test: np.ndarray, reference: np.ndarray | None = None) -> np.ndarray:
    """How far a mapping moves each voxel of the brain from another, or from where it is."""
    return displacements_within_mask(
        mask,
        nt.linear.Affine(test),
        None if reference is None else nt.linear.Affine(reference),
    )


def sizes(work: Path, mask_path: Path) -> np.ndarray:
    """
    How far each volume's corrections move the brain (mm), written to ``sizes.tsv``.

    For each diffusion-weighted volume: its b-value, then the median and the largest
    displacement within the brain of the head motion alone (the first run), of the eddy
    currents (the default run), and between the two runs' whole mappings (what the
    third column of the figure changes over the second).
    """
    motion = dmri.DWI.from_filename(output(work, "motion", ".h5"))
    default = dmri.DWI.from_filename(output(work, "default", ".h5"))
    assert motion.motion is not None
    assert default.motion is not None and default.eddy is not None
    mask = nb.load(mask_path)
    rows = []
    for k in range(len(motion)):
        parts = (
            displacements(mask, motion.motion[k]),
            displacements(mask, default.eddy[k]),
            displacements(mask, default.eddy[k] @ default.motion[k], motion.motion[k]),
        )
        rows.append([motion.gradients[k, -1], *(f(d) for d in parts for f in (np.median, np.max))])
    table = np.array(rows)
    header = "b\tmotion_median\tmotion_max\teddy_median\teddy_max\tchange_median\tchange_max"
    np.savetxt(work / "sizes.tsv", table, fmt="%.3f", delimiter="\t", header=header)
    return table


def canonical(path: Path) -> np.ndarray:
    """An image's data, in RAS+ voxel order."""
    return np.asanyarray(nb.as_closest_canonical(nb.load(path)).dataobj)


def cuts(volume: np.ndarray, x: int, z: int) -> tuple[np.ndarray, np.ndarray]:
    """The axial cut (anterior up, right on the right) and the sagittal one (anterior left)."""
    return np.flipud(volume[:, :, z].T), np.flipud(volume[x, :, :].T)[:, ::-1]


def column(axial: np.ndarray, sagittal: np.ndarray, width: int) -> np.ndarray:
    """The axial cut over the sagittal one, each centred in the column's width."""

    def centre(image: np.ndarray) -> np.ndarray:
        left = (width - image.shape[1]) // 2
        return np.pad(image, ((0, 0), (left, width - image.shape[1] - left)))

    return np.vstack([centre(axial), np.zeros((GAP, width)), centre(sagittal)])


def outline(brain: np.ndarray, x: int, z: int, width: int) -> list[str]:
    """The brain's outline on every panel, as SVG polylines in pixels of the frame."""
    axial, sagittal = cuts(brain.astype(float), x, z)
    lines = []
    for c in range(len(LABELS)):
        for top, image in ((0, axial), (axial.shape[0] + GAP, sagittal)):
            left = c * (width + GAP) + (width - image.shape[1]) // 2
            for contour in find_contours(image, 0.5):
                points = " ".join(f"{left + u + 0.5:.1f},{top + v + 0.5:.1f}" for v, u in contour)
                lines.append(f'<polyline points="{points}"/>')
    return lines


def figure(dwi: Path, bval: Path, mask: Path, work: Path, table: np.ndarray, out: Path) -> Path:
    """The volumes, original and corrected, played in acquisition order."""
    bvals = np.loadtxt(bval)
    keep = np.flatnonzero(bvals >= 50)  # the volumes the outputs hold
    original = canonical(dwi)
    corrected = [canonical(output(work, name, ".nii.gz")) for name in RUNS]
    assert all(c.shape[-1] == len(keep) == len(table) for c in corrected)

    brain = canonical(mask) > 0
    corners = np.argwhere(brain)
    low = np.maximum(corners.min(0) - MARGIN, 0)
    high = np.minimum(corners.max(0) + MARGIN + 1, brain.shape)
    crop = tuple(slice(a, b) for a, b in zip(low, high, strict=True))
    brain = brain[crop]
    x, z = brain.shape[0] // 2, brain.shape[2] // 2
    width = max(brain.shape[0], brain.shape[1])
    height = brain.shape[1] + GAP + brain.shape[2]
    frame_width = len(LABELS) * (width + GAP) - GAP

    frames = []
    for k, index in enumerate(keep):
        volumes = [original[..., index][crop], *(c[..., k][crop] for c in corrected)]
        window = float(np.percentile(volumes[0][brain], 99.5))
        columns = []
        for volume in volumes:
            columns += [column(*cuts(np.asarray(volume, "float32"), x, z), width)]
            columns += [np.zeros((height, GAP))]
        frame = np.clip(np.hstack(columns[:-1]) / window, 0, 1)
        buffer = BytesIO()
        Image.fromarray((frame * 255).astype("uint8")).save(buffer, "JPEG", quality=QUALITY)
        _, _, _, eddy, largest, _, _ = table[k]
        title = (
            f"Volume {index} of {len(bvals)}, b={np.round(bvals[index], -2):g} s/mm² · "
            f"median in the brain: head motion {table[k, 1]:.2f} mm, eddy currents "
            f"{eddy:.2f} mm (largest {largest:.2f} mm)"
        )
        frames.append((title, base64.b64encode(buffer.getvalue()).decode()))

    left, top = 80, 52
    view = (left + SCALE * frame_width, top + SCALE * height)
    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'viewBox="0 0 {view[0]} {view[1]}" font-family="sans-serif" font-size="14">',
        "<style>",
        f".frame {{ visibility: hidden; animation: play {SECONDS * len(frames):g}s step-end "
        "infinite; }",
        f"@keyframes play {{ 0% {{ visibility: visible; }} {100 / len(frames):.4f}% "
        "{ visibility: hidden; } 100% { visibility: hidden; } }",
        "image { image-rendering: pixelated; }",
        "</style>",
        f'<rect width="{view[0]}" height="{view[1]}" fill="black"/>',
    ]
    for c, label in enumerate(LABELS):
        middle = left + SCALE * (c * (width + GAP) + width / 2)
        svg.append(
            f'<text x="{middle:.0f}" y="{top - 8}" text-anchor="middle" '
            f'fill="#f0efec">{label}</text>'
        )
    rows = (("Axial", 0, brain.shape[1]), ("Sagittal", brain.shape[1] + GAP, brain.shape[2]))
    for name, start, size in rows:
        middle = top + SCALE * (start + size / 2)
        svg.append(
            f'<text x="{left - 10}" y="{middle + 5:.0f}" text-anchor="end" '
            f'fill="#f0efec">{name}</text>'
        )
    for k, (title, payload) in enumerate(frames):
        svg.append(
            f'<g class="frame" style="animation-delay: {k * SECONDS:g}s">'
            f'<image x="{left}" y="{top}" width="{SCALE * frame_width}" '
            f'height="{SCALE * height}" xlink:href="data:image/jpeg;base64,{payload}"/>'
            f'<text x="{left}" y="20" fill="#f0efec">{title}</text></g>'
        )
    svg.append(
        f'<g transform="translate({left} {top}) scale({SCALE})" fill="none" '
        'stroke="#eb6834" stroke-width="0.4">'
    )
    svg += outline(brain, x, z, width)
    svg += ["</g>", "</svg>"]
    path = out / "gem-sherbrooke.svg"
    path.write_text("\n".join(svg) + "\n")
    return path


def animation(work: Path, out: Path) -> Path:
    """How the default run corrects each volume, animated."""
    path = out / "corrections-sherbrooke.svg"
    animate_corrections(dmri.DWI.from_filename(output(work, "default", ".h5")), path)
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--work", type=Path, required=True, help="the runs' folder")
    parser.add_argument("--out", type=Path, required=True, help="the figure's folder")
    parser.add_argument("--figure-only", action="store_true", help="reuse the runs")
    args = parser.parse_args(argv)

    args.work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    dwi, bvec, bval = inputs()
    mask = brain_mask(dwi, bval, args.work / "brainmask.nii.gz")
    if not args.figure_only:
        run(dwi, bvec, bval, mask, args.work)
    table = sizes(args.work, mask)
    for name, col in (("Head motion", 1), ("Eddy currents", 3), ("Change", 5)):
        print(
            f"{name}: {np.median(table[:, col]):.2f} mm in the median volume (median in "
            f"the brain), at most {table[:, col + 1].max():.2f} mm"
        )
    for path in (
        figure(dwi, bval, mask, args.work, table, args.out),
        animation(args.work, args.out),
    ):
        print(f"{path}: {path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
