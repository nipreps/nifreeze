# /// script
# requires-python = ">=3.13"
# dependencies = ["nifreeze[elastix]"]
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
"""Do elastix's settings hold up against ANTs on the suite's real b=0 images? (#675)

The real-data designs of the test suite, with each arm called as the harness calls it
(12-bit window, fixed-image mask, one thread, seed 12345):

``hcph``, ``dwi``
    ``test_registration.py::test_ANTs_config_b0``: the b=0 average registered to itself
    resampled (cubic) under 216 known rigid motions (rotations of 0, 0.1 or 0.3 rad about
    each axis; shifts of 0 or 1 mm along each); mask: ``<dataset>-b0_desc-brain``.
``dwi_motion``
    ``test_integration.py``'s data: ``dwi.h5``'s b=0 moved by 7 known motions
    (``dwi_motion.h5``), registered to the b=0, as the estimator does with the trivial model.

Errors: ``displacements_within_mask`` between the estimate and the known motion (mm).
Arms: ``ants`` (today's path) and configurations of ``elastix_tuning.py`` (``t-x1`` is the
direct mapping). Writes ``<out>/<name>.tsv``::

    uv run tools/registration/elastix_real_b0.py --data $TEST_DATA_HOME \\
        --arms ants t-x1 p2-x1 p3-x1 --out <results>/real-b0
"""

import argparse
import csv
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter

import nibabel as nb
import nitransforms as nt
import numpy as np
from elastix_tuning import SEED, _warm, configuration, configurations
from nibabel.affines import from_matvec
from nibabel.eulerangles import euler2mat

import nifreeze.registration.ants  # noqa: F401 (nipype, imported before the workers fork)
from nifreeze.analysis.registration import AntsRegistration, Register
from nifreeze.data import dmri
from nifreeze.registration.elastix import Elastix
from nifreeze.registration.utils import displacements_within_mask
from nifreeze.utils.ndimage import load_api

ANGLES = (0.0, 0.1, 0.3)
"""Rotations about each axis (rad), as ``test_ANTs_config_b0``."""
SHIFTS = (0.0, 1.0)
"""Shifts along each axis (mm), as ``test_ANTs_config_b0``."""
_SHARED: dict = {}
"""The data, loaded before the workers fork (copy on write)."""


def _register(task: tuple[str, int, np.ndarray]) -> list[dict]:
    dataset, case, truth = task
    fixed, mask, affine, mask_img = _SHARED[dataset]
    if dataset == "dwi_motion":
        moving = _SHARED["moved"][case]
    else:
        b0nii = nb.Nifti1Image(fixed, affine)
        xfm = nt.linear.Affine(truth, reference=b0nii)
        moving = np.asanyarray(
            nt.resampling.apply(~xfm, b0nii, reference=b0nii).dataobj, dtype="float32"
        )
    motion = displacements_within_mask(
        mask_img, nt.linear.Affine(np.eye(4)), nt.linear.Affine(truth)
    ).mean()
    rows = []
    with tempfile.TemporaryDirectory(dir=_SHARED["work"]) as workdir:
        for arm in _SHARED["arms"]:
            register: Register
            if arm == "ants":
                register = AntsRegistration(workdir, seed=SEED, num_threads=1)
            else:
                settings = configuration(*configurations()[arm])
                register = Elastix(settings, seed=SEED, num_threads=1)
            start = perf_counter()
            estimate = register(fixed, moving, affine, mask, case)
            seconds = perf_counter() - start
            errors = displacements_within_mask(
                mask_img, nt.linear.Affine(estimate), nt.linear.Affine(truth)
            )
            rows.append(
                {
                    "dataset": dataset,
                    "case": case,
                    "arm": arm,
                    "motion": float(motion),
                    "mean": float(errors.mean()),
                    "max": float(errors.max()),
                    "seconds": seconds,
                }
            )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--data", type=Path, default=Path(os.environ.get("TEST_DATA_HOME", "~/.nifreeze-tests"))
    )
    parser.add_argument("--arms", nargs="+", default=["ants", "t-x1", "p2-x1", "p3-x1"])
    parser.add_argument("--name", default="real-data", help="name of the results table")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if unknown := set(args.arms) - {"ants", *configurations()}:
        parser.error(f"unknown arms: {sorted(unknown)}")
    data = args.data.expanduser()
    args.out.mkdir(parents=True, exist_ok=True)

    for name in ("hcph", "dwi"):
        b0 = load_api(data / f"{name}-b0_desc-avg.nii.gz", nb.Nifti1Image)
        brain = load_api(data / f"{name}-b0_desc-brain.nii.gz", nb.Nifti1Image)
        mask = np.asanyarray(brain.dataobj) > 0
        _SHARED[name] = (
            np.asanyarray(b0.dataobj, dtype="float32"),
            mask,
            b0.affine,
            nb.Nifti1Image(mask.astype("uint8"), b0.affine),
        )
    moved = dmri.DWI.from_filename(data / "dmri_data" / "motion_test_data" / "dwi_motion.h5")
    if moved.motion_affines is None:
        raise ValueError("dwi_motion.h5 does not record its motions")
    original = dmri.DWI.from_filename(data / "dwi.h5")
    mask = np.asanyarray(original.brainmask, dtype=bool)
    _SHARED["dwi_motion"] = (
        np.asanyarray(moved.bzero, dtype="float32"),
        mask,
        moved.affine,
        nb.Nifti1Image(mask.astype("uint8"), moved.affine),
    )
    _SHARED["moved"] = [
        np.asanyarray(moved.dataobj[i], dtype="float32") for i in range(len(moved))
    ]
    _SHARED["arms"] = args.arms

    motions = [
        from_matvec(euler2mat(x=rx, y=ry, z=rz), (tx, ty, tz))
        for rx, ry, rz, tx, ty, tz in product(ANGLES, ANGLES, ANGLES, SHIFTS, SHIFTS, SHIFTS)
    ]
    tasks = [(name, i, m) for name in ("hcph", "dwi") for i, m in enumerate(motions)]
    tasks += [("dwi_motion", i, np.asarray(m)) for i, m in enumerate(moved.motion_affines)]

    rows = []
    with tempfile.TemporaryDirectory(dir=args.out) as work:
        _SHARED["work"] = work
        start = perf_counter()
        with ProcessPoolExecutor(
            args.jobs, mp_context=get_context("fork"), initializer=_warm
        ) as pool:
            futures = [pool.submit(_register, task) for task in tasks]
            for done, future in enumerate(as_completed(futures), start=1):
                rows += future.result()
                if done % 50 == 0 or done == len(futures):
                    print(f"{done}/{len(futures)}, {perf_counter() - start:.0f} s", flush=True)
    rows.sort(key=lambda r: (r["dataset"], r["case"], args.arms.index(r["arm"])))
    with (args.out / f"{args.name}.tsv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print("wrote", args.out / f"{args.name}.tsv", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
