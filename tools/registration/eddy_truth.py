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
"""Whether the sidecar's displacement is the eddy-current truth at native resolution (#618).

Each eddy-current copy of a phantom comes with its displacement along phase encoding,
in closed form (:class:`~nifreeze.analysis.sensitivity.Distortion`). TRXScan forms
the distorted volumes through the EPI readout; registration corrects them by warping.
For every volume of every eddy-only (``e*``) and combined (``m*s*e*``) copy, the
undistorted volume (the clean reference, or the motion-only copy with the same
motions) is distorted by the sidecar's coefficients:

Q1, readout
    as TRXScan's readout forms it, which should reproduce TRXScan's volume;
Q2, warp
    by resampling at the displaced points and scaling by the Jacobian, as a
    correction by warping would undo it.

Both are compared with TRXScan's volume within the brain, relative to the volume's
maximum. On the development volumes of a few copies (Q3), the four coefficients that
best map the undistorted volume onto TRXScan's under the warp model are fitted by
least squares, without any registration, and their displacement is compared with
the sidecar's within the brain, in voxels. All volumes are noise-free. Runs in
NiFreeze's environment::

    uv run tools/registration/eddy_truth.py --dataset <dataset> --protocol sherbrooke \\
        --out <dataset>/derivatives/registration/eddy-truth
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter
from typing import Any

import attrs
import numpy as np
from harness import provenance, subset, write_results
from scipy import ndimage
from scipy.optimize import least_squares

from nifreeze.analysis.phantoms import load_phantom
from nifreeze.analysis.sensitivity import Distortion

FITTED = ("e2", "e4", "e8", "m1s0e4", "m4s0e4")
"""The copies whose development volumes are fitted (Q3)."""
FIT_COLUMNS = (
    "fit_mean_vox",
    "fit_p95_vox",
    "fit_max_vox",
    "fit_residual_max",
    "fit_evaluations",
    *(f"fit_c{k}" for k in range(4)),
    *(f"true_c{k}" for k in range(4)),
)
"""The columns of a fitted volume, empty for the others."""

# The copy being evaluated, shared with the forked workers
_SHARED: dict[str, Any] = {}


def _name(magnitude: float, seed: int | None, eddy: int) -> str:
    if magnitude == 0:
        return f"e{eddy}"
    return f"m{magnitude:g}s{seed}e{eddy}".replace(".", "p")


def _points(mask: np.ndarray, affine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The voxel and world coordinates of the mask's voxels, ``(3, n)`` each."""
    ijk = np.argwhere(mask).T.astype(float)
    return ijk, affine[:3, :3] @ ijk + affine[:3, 3:]


def _evaluate(index: int) -> dict[str, Any]:
    start = perf_counter()
    readout: Distortion = _SHARED["readout"]
    warp: Distortion = _SHARED["warp"]
    affine, mask = _SHARED["affine"], _SHARED["mask"]
    undistorted = np.asarray(_SHARED["undistorted"][index], dtype=float)
    distorted = np.asarray(_SHARED["distorted"][index], dtype=float)
    coefficients = readout.coefficients[index]
    scale = distorted.max()
    voxel = float(np.linalg.norm(affine[:3, readout.axis]))

    row: dict[str, Any] = {
        "copy": _SHARED["copy"],
        "index": index,
        "shell": _SHARED["shells"][index],
        "truth_max_vox": float(np.abs(coefficients @ _SHARED["values"]).max() / voxel),
        **dict.fromkeys(FIT_COLUMNS),
    }
    for label, model in (("readout", readout), ("warp", warp)):
        residual = np.abs(model.distort(undistorted, affine, coefficients) - distorted)[mask]
        row[f"{label}_max"] = float(residual.max() / scale)
        row[f"{label}_p99"] = float(np.percentile(residual, 99) / scale)

    if index in _SHARED["fit"]:
        ijk, values, derivatives = _SHARED["fit_points"]
        observed = distorted[tuple(ijk.astype(int))] / scale
        spline = ndimage.spline_filter(undistorted, order=3)
        steps = np.linalg.inv(affine[:3, :3]) @ warp.direction

        def residuals(c: np.ndarray) -> np.ndarray:
            coordinates = ijk + np.outer(steps, c @ values)
            resampled = ndimage.map_coordinates(spline, coordinates, order=3, prefilter=False)
            return resampled * (1.0 + c @ derivatives) / scale - observed

        solution = least_squares(residuals, np.zeros(len(coefficients)), method="lm")
        error = np.abs((solution.x - coefficients) @ _SHARED["values"]) / voxel
        row |= {
            "fit_mean_vox": float(error.mean()),
            "fit_p95_vox": float(np.percentile(error, 95)),
            "fit_max_vox": float(error.max()),
            "fit_residual_max": float(np.abs(solution.fun).max()),
            "fit_evaluations": int(solution.nfev),
            **{f"fit_c{k}": float(v) for k, v in enumerate(solution.x)},
            **{f"true_c{k}": float(v) for k, v in enumerate(coefficients)},
        }
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
    fit = set(subset(args.protocol, "dev", n_volumes))
    motion_only = {
        (p.magnitude, p.seed): load
        for p, load in cells
        if p.distortion is None and len(p.moved) == n_volumes
    }
    eddies = {}
    for perturbation, load in cells:
        if (readout := perturbation.distortion) is not None:
            voxel = float(np.linalg.norm(affine[:3, readout.axis]))
            eddy = round(readout.magnitude / voxel)
            name = _name(perturbation.magnitude, perturbation.seed, eddy)
            if args.copies is None or name in args.copies:
                eddies[(perturbation.magnitude, eddy)] = (name, perturbation, readout, load)
    _, world = _points(mask, affine)

    rows: list[dict[str, Any]] = []
    for key in sorted(eddies):
        name, perturbation, readout, load = eddies[key]
        undistorted = (
            clean
            if perturbation.magnitude == 0
            else motion_only[(perturbation.magnitude, perturbation.seed)]()
        )
        values, _ = readout.basis(world)

        # The warp model is fitted where the displaced signal may land
        voxel = float(np.linalg.norm(affine[:3, readout.axis]))
        reach = int(np.ceil(readout.magnitude / voxel)) + 1
        dilated = ndimage.binary_dilation(mask, iterations=reach)
        fit_ijk, fit_world = _points(dilated, affine)
        fit_values, fit_derivatives = readout.basis(fit_world)

        _SHARED.clear()
        _SHARED.update(
            copy=name,
            readout=readout,
            warp=attrs.evolve(readout, readout=None),
            affine=affine,
            mask=mask,
            values=values,
            shells=np.round(clean.gradients[:, -1], -2).tolist(),
            undistorted=undistorted.dataobj,
            distorted=load().dataobj,
            fit=fit if name in FITTED else set(),
            fit_points=(fit_ijk, fit_values, fit_derivatives),
        )
        # Forked workers share the loaded copy without pickling it
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
            "Fitted": list(FITTED),
            "FitVolumes": sorted(fit),
            "Noise": None,
            "Jobs": args.jobs,
            "Provenance": provenance(),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
