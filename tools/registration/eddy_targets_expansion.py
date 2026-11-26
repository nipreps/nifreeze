# /// script
# requires-python = ">=3.13"
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
"""Why do the affine registrations of the eddy-current pass expand? (#723, Test 2)

Each volume of a copy of Sherbrooke is registered to its noise-free original (the
oracle's target), from its true head motion ``M``, with the eddy-current pass's settings
(:data:`~nifreeze.registration.base.AFFINE`, elastix, one thread, seed 12345) or with a
variant that removes one of their ingredients. Each row gives the volume's residual
``W T^-1`` (``T`` the copy's true mapping): its volume change, ``det(L)^(1/3) - 1`` of
its linear part ``L`` (positive: the estimate expands), and the error of the mapping
(mean and largest displacement within the brain, mm).

The cells (:data:`CELLS`) cross the copy (``unmoved``, ``m2s0`` with head motion only,
``e2`` with eddy currents only, and ``m2s0e2`` with both) with the noise (none, or the
end-to-end test's), with the default settings. The variants (:data:`VARIANTS`) run on
``m2s0e2`` and ``unmoved``, with the noise. With the noise-free ``unmoved`` copy, the
moving image is the target itself.

Writes ``<out>/<protocol>.{tsv,json,npz}``::

    OMP_NUM_THREADS=6 uv run tools/registration/eddy_targets_expansion.py \\
        --dataset <dataset> --out <out> --jobs 6
"""

import argparse
import copy as _copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.metadata import version
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter
from typing import Any

import nibabel as nb
import nitransforms as nt
import numpy as np
from eddy_targets import _dataset_commit, _truth_check, _volumes
from experiments import provenance, write_results

from nifreeze.analysis.phantoms import load_phantom
from nifreeze.registration.base import AFFINE, RegistrationFactory
from nifreeze.registration.elastix import get_settings
from nifreeze.registration.utils import displacements_within_mask

NOISE = 20261020
"""Seed of the noise, as the end-to-end test's."""
SEED = 12345
"""Seed of the registrations."""
COPIES = {
    "unmoved": (0.0, None),
    "m2s0": (2.0, None),
    "e2": (0.0, 4.0),
    "m2s0e2": (2.0, 4.0),
}
"""Each copy's head motion (mm and degrees) and eddy currents (largest mm)."""
VARIANTS = (
    "default",
    "no-histogram-matching",
    "no-winsorizing",
    "no-clipping",
    "no-mask",
    "no-smoothed-stage",
)
"""The settings: the pass's, or with one ingredient removed."""
CELLS = [
    *[(copy, noise, "default") for copy in COPIES for noise in (False, True)],
    *[(copy, True, variant) for copy in ("m2s0e2", "unmoved") for variant in VARIANTS[1:]],
]
"""The registrations: (copy, noisy, variant)."""

_SHARED: dict[str, Any] = {}
"""A cell's data, set before its workers fork (copy on write)."""


def _settings(variant: str) -> tuple[dict, str, bool]:
    """The elastix settings, the clipping and whether the target's mask is used."""
    settings = _copy.deepcopy(get_settings(AFFINE))
    clip, use_mask = "both", True
    if variant == "no-histogram-matching":
        settings["use_histogram_matching"] = False
    elif variant == "no-winsorizing":
        settings.pop("winsorize_lower_quantile")
        settings.pop("winsorize_upper_quantile")
    elif variant == "no-clipping":
        clip = "none"
    elif variant == "no-mask":
        use_mask = False
    elif variant == "no-smoothed-stage":
        settings["stages"] = settings["stages"][1:]
    return settings, clip, use_mask


def _register(index: int) -> tuple[int, np.ndarray, float, bool]:
    """Register volume ``index`` to its noise-free original, from its true head motion."""
    settings, clip, use_mask = _settings(_SHARED["variant"])
    registration = RegistrationFactory.init(
        "elastix", align_config=settings, seed=SEED, num_threads=1, clip=clip
    )
    moving, start = _SHARED["moving"], np.array(_SHARED["motion"][index], dtype=float)
    began = perf_counter()
    try:
        estimate = registration(
            _SHARED["targets"][index],
            moving[index][0],
            moving.affine,
            moving.brainmask if use_mask else None,
            index,
            None if np.allclose(start, np.eye(4)) else start,
        )
    except RuntimeError:  # elastix's, such as samples mapped outside the moving image
        return index, start, perf_counter() - began, True
    return index, np.asarray(estimate.matrix), perf_counter() - began, False


def _copies(clean_cells: list, noisy_cells: list, n_volumes: int) -> dict[tuple, tuple]:
    """Each copy's loader, with and without noise, and its perturbation."""
    found: dict[tuple, tuple] = {}
    for noisy, cells in ((False, clean_cells), (True, noisy_cells)):
        for name, (magnitude, eddy) in COPIES.items():
            if name == "unmoved":
                continue
            chosen = [
                (perturbation, load)
                for perturbation, load in cells
                if perturbation.magnitude == magnitude
                and perturbation.seed in ((0,) if magnitude else (None, 0))
                and perturbation.count == (n_volumes if magnitude else 0)
                and (
                    None if perturbation.distortion is None else perturbation.distortion.magnitude
                )
                == eddy
            ]
            if len(chosen) != 1:
                raise RuntimeError(f"{len(chosen)} copies match {name}, not 1")
            found[name, noisy] = chosen[0]
    return found


def _truth(perturbation: Any, n_volumes: int) -> tuple[np.ndarray, np.ndarray]:
    """The copy's head motion ``M`` and its whole true mapping ``T = E M``."""
    motion = np.repeat(np.eye(4)[np.newaxis], n_volumes, axis=0)
    eddy = motion.copy()
    if perturbation is not None:
        for index, matrix in zip(perturbation.moved, perturbation.motions, strict=True):
            motion[index] = matrix
        if perturbation.distortion is not None:
            eddy = perturbation.distortion.matrices()
    return motion, eddy @ motion


def _rows(
    cell: tuple, results: dict[int, tuple], whole: np.ndarray, mask: nb.Nifti1Image, shells
) -> tuple[list[dict[str, Any]], np.ndarray]:
    copy, noisy, variant = cell
    rows, residuals = [], []
    for index in sorted(results):
        estimate, seconds, failed = results[index]
        residual = estimate @ np.linalg.inv(whole[index])
        errors = displacements_within_mask(
            mask, nt.linear.Affine(estimate), nt.linear.Affine(whole[index])
        )
        rows.append(
            {
                "copy": copy,
                "noise": noisy,
                "variant": variant,
                "index": index,
                "shell": float(shells[index]),
                "volume_change": float(np.cbrt(np.linalg.det(residual[:3, :3])) - 1),
                "error_mean": float(errors.mean()),
                "error_max": float(errors.max()),
                "seconds": seconds,
                "failed": failed,
            }
        )
        residuals.append(residual)
    return rows, np.stack(residuals)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", type=Path, required=True, help="phantom dataset root")
    parser.add_argument("--protocol", default="sherbrooke")
    parser.add_argument("--out", type=Path, required=True, help="results folder")
    parser.add_argument("--limit", type=int, help="volumes per shell (pilot)")
    parser.add_argument("--jobs", type=int, default=1, help="registrations at a time")
    args = parser.parse_args(argv)

    commit = _dataset_commit(args.dataset)
    wall = perf_counter()
    clean, clean_cells = load_phantom(args.dataset, args.protocol)
    noisy_reference, noisy_cells = load_phantom(args.dataset, args.protocol, noise=NOISE)
    n_volumes = len(clean)
    copies = _copies(clean_cells, noisy_cells, n_volumes)
    shells = np.round(clean.gradients[:, -1], -2)
    mask_data = np.asanyarray(clean.brainmask, dtype=bool)
    mask = nb.Nifti1Image(mask_data.astype("uint8"), clean.affine)
    points = nb.affines.apply_affine(clean.affine, np.argwhere(mask_data)).T
    radius = float(np.linalg.norm(points - points.mean(axis=1, keepdims=True), axis=0).mean())
    volumes = _volumes(shells, args.limit, 0)
    targets = {index: np.asarray(clean.dataobj[index], dtype="float32") for index in volumes}

    perturbation = copies["m2s0e2", True][0]
    motion, whole = _truth(perturbation, n_volumes)
    checks = {
        "TruthMm": _truth_check(
            perturbation.distortion, motion, whole @ np.linalg.inv(motion), points
        )
    }
    print(f"loaded in {perf_counter() - wall:.0f} s; truth {checks['TruthMm']:.2e} mm", flush=True)

    rows: list[dict[str, Any]] = []
    estimates: dict[str, Any] = {"volumes": np.asarray(volumes)}
    seconds: dict[str, float] = {}
    for cell in CELLS:
        copy, noisy, variant = cell
        began = perf_counter()
        if copy == "unmoved":
            moving, perturbation = (noisy_reference if noisy else clean), None
        else:
            perturbation, load = copies[copy, noisy]
            moving = load()
        motion, whole = _truth(perturbation, n_volumes)
        _SHARED.update(moving=moving, motion=motion, targets=targets, variant=variant)
        results = {}
        with ProcessPoolExecutor(args.jobs, mp_context=get_context("fork")) as pool:
            futures = [pool.submit(_register, index) for index in volumes]
            for future in as_completed(futures):
                index, estimate, took, failed = future.result()
                results[index] = (estimate, took, failed)
        cell_rows, residuals = _rows(cell, results, whole, mask, shells)
        rows += cell_rows
        key = f"{copy}/{'noisy' if noisy else 'clean'}/{variant}"
        estimates[key] = np.stack([results[i][0] for i in sorted(results)])
        estimates[f"{key}/residual"] = residuals
        seconds[key] = perf_counter() - began
        change = np.median([r["volume_change"] for r in cell_rows])
        print(
            f"{key}: median volume change {100 * change:+.3f}%, {seconds[key]:.0f} s", flush=True
        )

    table, _ = write_results(
        args.out,
        args.protocol,
        rows,
        {
            "Protocol": args.protocol,
            "Copies": {name: {"MotionMm": m, "EddyMm": e} for name, (m, e) in COPIES.items()},
            "Variants": list(VARIANTS),
            "Cells": [list(cell) for cell in CELLS],
            "Noise": NOISE,
            "Seed": SEED,
            "Volumes": volumes,
            "Limit": args.limit,
            "MeanRadiusMm": radius,
            "VoxelSize": np.linalg.norm(clean.affine[:3, :3], axis=0).tolist(),
            "Settings": Path(str(AFFINE)).name,
            "Jobs": args.jobs,
            "Seconds": seconds,
            "WallSeconds": perf_counter() - wall,
            "Checks": checks,
            "DatasetCommit": commit,
            "DatasetCommitAtEnd": _dataset_commit(args.dataset),
            "Provenance": provenance() | {"ItkElastix": version("itk-elastix")},
        },
    )
    np.savez_compressed(args.out / f"{args.protocol}.npz", **estimates)
    print(f"wrote {table}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
