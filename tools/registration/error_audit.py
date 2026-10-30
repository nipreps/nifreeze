# /// script
# requires-python = ">=3.12"
# dependencies = ["nifreeze"]
#
# [tool.uv.sources]
# nifreeze = { path = "../..", editable = true }
# ///
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
"""What the registration experiments count as error (#626).

NiFreeze registers each volume (moving) to a prediction (fixed), so its estimate maps
each point x of the reference grid to where that point's signal lies in the volume.
Each check samples a perturbed copy at the points a candidate truth gives, by cubic
B-splines, scales the samples by the Jacobian of that map, and compares them with the
volume the correction should then reproduce, within the reference's brain mask:

A, motion
    each motion-only copy at M x, at M⁻¹ x and at x, against the clean reference;
B, eddy currents
    each eddy-only copy at x - d(x) u (the sidecar's coefficients), at x - d_w(x) u
    (the inverted coefficients, d / (1 - s)), at x + d(x) u (the opposite sign) and at
    x, against the undistorted volume. On combined copies, the eddy currents alone
    (against the motion-only copy), and the whole mapping, at M x - d(M x) u and its
    alternatives (against the clean reference);
C, the domain of E0a's residuals
    E0a's Q1 (the readout) and Q2b (the inverse warp) within the brain mask, within
    the mask grown by the largest displacement plus one voxel, and over the volume.

Residuals are relative to the maximum of the volume compared against. All volumes are
noise-free. Runs in NiFreeze's environment::

    uv run tools/registration/error_audit.py --dataset <dataset> --protocol sherbrooke \\
        --out <dataset>/derivatives/registration/error-audit
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter
from typing import Any

import attrs
import numpy as np
from harness import provenance, write_results
from scipy import ndimage

from nifreeze.analysis.phantoms import load_phantom
from nifreeze.analysis.sensitivity import Distortion

STATS = ("max", "p99")
"""The summaries of each residual."""
CANDIDATES = {
    "motion": ("identity", "matrix", "inverse"),
    "eddy": ("none", "sidecar", "inverted", "opposite"),
    "total": ("motion", "sidecar", "inverted", "opposite"),
    "readout": ("mask", "grown", "volume"),
    "inverse_warp": ("mask", "grown", "volume"),
}
"""The residuals of each check: A (``motion``), B (``eddy``, ``total``) and C."""
COLUMNS = (
    "kind",
    "copy",
    "index",
    "shell",
    "truth_max_vox",
    "inverted_gap_vox",
    *(
        f"{check}_{candidate}_{stat}"
        for check, candidates in CANDIDATES.items()
        for candidate in candidates
        for stat in STATS
    ),
    "seconds",
)
"""The columns of every row; those of the other kinds stay empty."""

# The copy being evaluated, shared with the forked workers
_SHARED: dict[str, Any] = {}


def _name(magnitude: float, seed: int | None, eddy: int | None) -> str:
    motion = f"m{magnitude:g}s{seed}".replace(".", "p") if magnitude else ""
    return motion + (f"e{eddy}" if eddy else "")


def _residual(
    row: dict[str, Any], label: str, estimate: np.ndarray, target: np.ndarray, scale: float
) -> None:
    residual = np.abs(estimate - target)
    row[f"{label}_max"] = float(residual.max() / scale)
    row[f"{label}_p99"] = float(np.percentile(residual, 99) / scale)


def _sampler(volume: np.ndarray, affine: np.ndarray):
    """The volume at world points ``(3, n)``, by cubic B-splines, zero off the grid."""
    spline = ndimage.spline_filter(np.asarray(volume, dtype=float), order=3)
    to_voxels = np.linalg.inv(affine)

    def sample(points: np.ndarray) -> np.ndarray:
        ijk = to_voxels[:3, :3] @ points + to_voxels[:3, 3:]
        return ndimage.map_coordinates(spline, ijk, order=3, prefilter=False)

    return sample


def _rigid(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return matrix[:3, :3] @ points + matrix[:3, 3:]


def _unwarp(
    row: dict[str, Any],
    label: str,
    sample: Any,
    distortion: Distortion,
    coefficients: np.ndarray,
    points: np.ndarray,
    target: np.ndarray,
    scale: float,
) -> None:
    """Sample at the points the readout moved each of ``points`` to, and alternatives."""
    u = distortion.direction[:, np.newaxis]
    values, derivatives = distortion.basis(points)
    displacement, slope = coefficients @ values, coefficients @ derivatives
    inverted = coefficients / (1.0 - coefficients @ _SHARED["slopes"])
    inverted_displacement, inverted_slope = inverted @ values, inverted @ derivatives
    candidates = {
        "sidecar": (points - displacement * u, 1.0 - slope),
        "inverted": (points - inverted_displacement * u, 1.0 - inverted_slope),
        "opposite": (points + displacement * u, 1.0 + slope),
    }
    for candidate, (mapped, jacobian) in candidates.items():
        _residual(row, f"{label}_{candidate}", sample(mapped) * jacobian, target, scale)


def _evaluate(index: int) -> dict[str, Any]:
    start = perf_counter()
    kind, affine, mask = _SHARED["kind"], _SHARED["affine"], _SHARED["mask"]
    points, voxel = _SHARED["points"], _SHARED["voxel"]
    clean = np.asarray(_SHARED["clean"][index], dtype=float)
    copy = np.asarray(_SHARED["copy_data"][index], dtype=float)
    sample = _sampler(copy, affine)
    row: dict[str, Any] = dict.fromkeys(COLUMNS)
    row |= {"kind": kind, "copy": _SHARED["copy"], "index": index}
    row["shell"] = _SHARED["shells"][index]

    if kind == "motion":
        matrix = _SHARED["motions"][index]
        target, scale = clean[mask], clean.max()
        _residual(row, "motion_identity", copy[mask], target, scale)
        _residual(row, "motion_matrix", sample(_rigid(matrix, points)), target, scale)
        inverse = np.linalg.inv(matrix)
        _residual(row, "motion_inverse", sample(_rigid(inverse, points)), target, scale)
        row["seconds"] = perf_counter() - start
        return row

    distortion: Distortion = _SHARED["distortion"]
    coefficients = distortion.coefficients[index]
    values = _SHARED["values"]
    displacement = coefficients @ values
    inverted = coefficients / (1.0 - coefficients @ _SHARED["slopes"])
    row["truth_max_vox"] = float(np.abs(displacement).max() / voxel)
    row["inverted_gap_vox"] = float(np.abs(displacement - inverted @ values).max() / voxel)

    # B: the eddy currents alone, against the volume before them
    undistorted = np.asarray(_SHARED["undistorted"][index], dtype=float)
    target, scale = undistorted[mask], undistorted.max()
    _residual(row, "eddy_none", copy[mask], target, scale)
    _unwarp(row, "eddy", sample, distortion, coefficients, points, target, scale)

    # B: the whole mapping of a combined copy, against the clean reference
    if kind == "combined":
        moved = _rigid(_SHARED["motions"][index], points)
        target, scale = clean[mask], clean.max()
        _residual(row, "total_motion", sample(moved), target, scale)
        _unwarp(row, "total", sample, distortion, coefficients, moved, target, scale)

    # C: E0a's Q1 and Q2b, within the mask, the grown mask and the whole volume
    scale = copy.max()
    readout = distortion.distort(undistorted, affine, coefficients)
    warped = attrs.evolve(distortion, readout=None).distort(undistorted, affine, inverted)
    for label, estimate in (("readout", readout), ("inverse_warp", warped)):
        for domain, where in (("mask", mask), ("grown", _SHARED["grown"]), ("volume", ...)):
            _residual(row, f"{label}_{domain}", estimate[where], copy[where], scale)
    row["seconds"] = perf_counter() - start
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", type=Path, required=True, help="phantom dataset root")
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--out", type=Path, required=True, help="results folder")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--copies", nargs="+", help="only these copies (all if not given)")
    args = parser.parse_args(argv)

    clean, cells = load_phantom(args.dataset, args.protocol)
    n_volumes = len(clean)
    mask = np.asanyarray(clean.brainmask, dtype=bool)
    affine = np.asarray(clean.affine, dtype=float)
    points = affine[:3, :3] @ np.argwhere(mask).T + affine[:3, 3:]
    motion_only = {}
    copies = []
    for perturbation, load in cells:
        if perturbation.magnitude and len(perturbation.moved) != n_volumes:
            continue
        eddy = None
        if (distortion := perturbation.distortion) is not None:
            voxel = float(np.linalg.norm(affine[:3, distortion.axis]))
            eddy = round(distortion.magnitude / voxel)
        else:
            motion_only[(perturbation.magnitude, perturbation.seed)] = load
        name = _name(perturbation.magnitude, perturbation.seed, eddy)
        if args.copies is None or name in args.copies:
            copies.append((name, perturbation, load))

    rows: list[dict[str, Any]] = []
    for name, perturbation, load in sorted(copies, key=lambda c: c[0]):
        distortion = perturbation.distortion
        kind = "eddy" if not perturbation.magnitude else "combined"
        if distortion is None:
            kind = "motion"
        _SHARED.clear()
        _SHARED.update(
            kind=kind,
            copy=name,
            affine=affine,
            mask=mask,
            points=points,
            voxel=None,
            shells=np.round(clean.gradients[:, -1], -2).tolist(),
            clean=clean.dataobj,
            copy_data=load().dataobj,
            motions=dict(zip(perturbation.moved, perturbation.motions, strict=True)),
        )
        if distortion is not None:
            if distortion.degree != 1:
                raise NotImplementedError(
                    "The inverse warp is closed-form only for linear fields."
                )
            voxel = float(np.linalg.norm(affine[:3, distortion.axis]))
            reach = int(np.ceil(distortion.magnitude / voxel)) + 1
            values, derivatives = distortion.basis(points)
            undistorted = clean
            if kind == "combined":
                undistorted = motion_only[(perturbation.magnitude, perturbation.seed)]()
            _SHARED.update(
                distortion=distortion,
                voxel=voxel,
                values=values,
                # The slope of each term along phase encoding, the same at every point
                slopes=derivatives[:, 0],
                undistorted=undistorted.dataobj,
                grown=ndimage.binary_dilation(mask, iterations=reach),
            )
        # Forked workers share the loaded copies without pickling them
        with ProcessPoolExecutor(args.jobs) as pool:
            rows.extend(pool.map(_evaluate, range(n_volumes)))
        print(f"{name}: {n_volumes} volumes", flush=True)

    write_results(
        args.out,
        args.protocol,
        rows,
        {
            "Protocol": args.protocol,
            "Copies": sorted({row["copy"] for row in rows}),
            "Interpolation": "cubic B-splines (scipy.ndimage.map_coordinates, order 3)",
            "Noise": None,
            "Jobs": args.jobs,
            "Provenance": provenance(),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
