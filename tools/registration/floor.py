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
"""What sets the floor of the oracle's registration error on a native phantom (#609).

In the registration benchmark (``tools/registration/bench.py``, #584), each moved
volume of the native Sherbrooke copies, registered to its noise-free original (the
head at its reference position, before the motion) with the estimator's default ANTs
settings (``b0-to-b0_level0``), missed by about 0.2 mm (mean within the brain) at
every magnitude. These registrations repeat
the benchmark's oracle on every ``--every``-th volume, and vary from the default:

``shrink``
    The finest level's shrink factor: 2, the default, or 1 (full resolution).
``noise``
    The moving volume with the protocol's noise, as in the benchmark, or without.
``optimizer``
    The finest level's optimization: the default (10 iterations of steps up to 1.96,
    convergence at 1e-7 over 2 iterations), or a longer one (100 iterations of steps
    up to 0.25, convergence at 1e-8 over 10 iterations).
``sampling``
    The finest level's metric samples: a random 18% of the voxels, the default, or
    every voxel (``dense``), which a trial on three volumes found biased with noise.
``encoding``
    The images reach ANTs as 12-bit codes, as in the benchmark, or in float32.

The shrink factor and the noise are crossed with the default optimization; the
longer one runs with noise at both shrink factors and without noise at full
resolution, and dense sampling and float32 each run once (see :data:`VARIANTS`).
Besides the benchmark's errors, each row has the residual of the estimate against the
truth, as its displacement at the brain's centroid (mm) and its rotation vector
(degrees), to show any bias. Runs in NiFreeze's environment::

    uv run tools/registration/floor.py --dataset <dataset> --protocol sherbrooke \\
        --out <dataset>/derivatives/registration/floor
"""

import argparse
import csv
import json
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import Any

import nibabel as nb
import nitransforms as nt
import numpy as np
from nipype.interfaces.ants.registration import Registration
from scipy.spatial.transform import Rotation

from nifreeze.analysis.phantoms import load_phantom
from nifreeze.data.filtering import robust_window
from nifreeze.registration.ants import _get_ants_settings, _prepare_registration_data
from nifreeze.registration.utils import displacements_within_mask

NOISE = 20261029
"""The benchmark's seed of the noise added on load."""
SEED = 12345
"""The benchmark's seed of ANTs' random sampling."""
LONGER = {
    "number_of_iterations": [100],
    "transform_parameters": [0.25],
    "convergence_threshold": 1e-8,
    "convergence_window_size": 10,
}
"""The longer optimization of the finest level."""
DENSE = {"sampling_strategy": "None", "sampling_percentage": 1.0}
"""Every voxel as the finest level's metric samples."""
DEFAULT = {"optimizer": "default", "sampling": "random", "encoding": "int12"}
VARIANTS: list[dict[str, Any]] = [
    {"shrink": shrink, "noise": noise, **DEFAULT}
    for shrink, noise in product((2, 1), (True, False))
] + [
    {"shrink": 2, "noise": True, **DEFAULT, "optimizer": "longer"},
    {"shrink": 1, "noise": True, **DEFAULT, "optimizer": "longer"},
    {"shrink": 1, "noise": False, **DEFAULT, "optimizer": "longer"},
    {"shrink": 2, "noise": True, **DEFAULT, "optimizer": "longer", "sampling": "dense"},
    {"shrink": 1, "noise": True, **DEFAULT, "encoding": "float32"},
]
"""The registrations compared."""


def ants_settings(variant: dict) -> dict:
    """The estimator's default settings, with the variant's finest level."""
    settings = json.loads(_get_ants_settings().read_text())
    settings["shrink_factors"][-1] = [variant["shrink"]]
    changes = (LONGER if variant["optimizer"] == "longer" else {}) | (
        DENSE if variant["sampling"] == "dense" else {}
    )
    for key, value in changes.items():
        settings[key][-1] = value
    return settings


def register(
    fixed: np.ndarray,
    moving: np.ndarray,
    affine: np.ndarray,
    mask: np.ndarray,
    mask_path: Path,
    workdir: Path,
    settings: dict,
    encoding: str,
    threads: int,
) -> tuple[np.ndarray, float]:
    """Register as the benchmark does, with any settings; the matrix (RAS) and seconds."""
    start = perf_counter()
    workdir.mkdir(parents=True)
    coded = encoding == "int12"
    fixed_path, moving_path, _ = _prepare_registration_data(
        moving,
        fixed,
        affine,
        0,
        workdir,
        clip="both" if coded else None,
        window=robust_window(moving, mask) if coded else None,
    )
    command = Registration(
        fixed_image=str(fixed_path),
        moving_image=str(moving_path),
        fixed_image_masks=[str(mask_path)] * len(settings["metric"]),
        output_transform_prefix=str(workdir / "ants-"),
        **settings,
    ).cmdline
    environ = os.environ | {
        "ANTS_RANDOM_SEED": str(SEED),
        "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": str(threads),
    }
    subprocess.run(command, shell=True, check=True, cwd=workdir, env=environ, capture_output=True)
    itk = nt.io.itk.ITKLinearTransform.from_filename(workdir / "ants-0GenericAffine.mat")
    matrix = itk.to_ras(reference=fixed_path, moving=moving_path)
    return np.asarray(matrix), perf_counter() - start


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", type=Path, required=True, help="phantom dataset root")
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--out", type=Path, required=True, help="results folder")
    parser.add_argument("--magnitudes", nargs="+", type=float, default=[1.0, 4.0])
    parser.add_argument("--every", type=int, default=3, help="evaluate every n-th volume")
    parser.add_argument("--jobs", type=int, default=4, help="registrations at once")
    parser.add_argument("--threads", type=int, default=1, help="threads of each registration")
    args = parser.parse_args(argv)

    clean, _ = load_phantom(args.dataset, args.protocol, counts=[])
    evaluate = list(range(0, len(clean), args.every))
    mask = np.asanyarray(clean.brainmask, dtype=bool)
    mask_img = nb.Nifti1Image(mask.astype("uint8"), clean.affine)
    centroid = nb.affines.apply_affine(clean.affine, np.argwhere(mask).mean(axis=0))

    # The moving volumes evaluated, with and without noise, and their truth
    moving: dict[tuple[float, bool], dict[int, np.ndarray]] = {}
    truth: dict[float, dict[int, np.ndarray]] = {}
    for seed in (NOISE, None):
        reference, cells = load_phantom(args.dataset, args.protocol, noise=seed)
        for perturbation, load in cells:
            if perturbation.distortion is not None or perturbation.count != len(reference):
                continue
            if perturbation.magnitude not in args.magnitudes:
                continue
            copy = load()
            moving[perturbation.magnitude, seed is not None] = {
                i: np.asarray(copy.dataobj[i], dtype="float32") for i in evaluate
            }
            motions = dict(zip(perturbation.moved, perturbation.motions, strict=True))
            truth[perturbation.magnitude] = {i: np.asarray(motions[i]) for i in evaluate}
            del copy

    tasks = [
        (magnitude, k, index)
        for magnitude in sorted(truth)
        for k in range(len(VARIANTS))
        for index in evaluate
    ]
    settings = [ants_settings(variant) for variant in VARIANTS]
    with tempfile.TemporaryDirectory(prefix="nifreeze-floor-") as tmp:
        root = Path(tmp)
        mask_path = root / "brainmask.nii"
        mask_img.to_filename(mask_path)

        def run(task: tuple[float, int, int]) -> dict:
            magnitude, k, index = task
            variant = VARIANTS[k]
            estimated, seconds = register(
                np.asarray(clean.dataobj[index], dtype="float32"),
                moving[magnitude, variant["noise"]][index],
                clean.affine,
                mask,
                mask_path,
                root / f"m{magnitude:g}-v{k}-{index:05d}",
                settings[k],
                variant["encoding"],
                args.threads,
            )
            expected = truth[magnitude][index]
            errors = displacements_within_mask(
                mask_img, nt.linear.Affine(estimated), nt.linear.Affine(expected)
            )
            shift = estimated[:3, :3] @ centroid + estimated[:3, 3]
            shift -= expected[:3, :3] @ centroid + expected[:3, 3]
            rotation = Rotation.from_matrix(estimated[:3, :3] @ expected[:3, :3].T)
            return {
                "magnitude": magnitude,
                **variant,
                "index": index,
                "shell": float(np.round(clean.gradients[index, -1], -2)),
                "max": float(errors.max()),
                "mean": float(errors.mean()),
                "seconds": seconds,
                **dict(zip(("tx", "ty", "tz"), shift.tolist(), strict=True)),
                **dict(zip(("rx", "ry", "rz"), rotation.as_rotvec(degrees=True), strict=True)),
            }

        rows = []
        with ThreadPoolExecutor(args.jobs) as pool:
            for done, row in enumerate(pool.map(run, tasks), start=1):
                rows.append(row)
                if done % 50 == 0 or done == len(tasks):
                    print(f"{done}/{len(tasks)} registrations", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / f"{args.protocol}.tsv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    ants = subprocess.run(["antsRegistration", "--version"], capture_output=True, text=True)
    (args.out / f"{args.protocol}.json").write_text(
        json.dumps(
            {
                "Protocol": args.protocol,
                "Noise": NOISE,
                "AntsSeed": SEED,
                "Magnitudes": sorted(truth),
                "Evaluate": evaluate,
                "Jobs": args.jobs,
                "Threads": args.threads,
                "Variants": [
                    {**variant, "Settings": s}
                    for variant, s in zip(VARIANTS, settings, strict=True)
                ],
                "AntsVersion": ants.stdout.strip(),
                "NiFreezeCommit": subprocess.run(
                    ["git", "-C", str(Path(__file__).parent / "nifreeze"), "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                ).stdout.strip(),
            },
            indent=2,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
