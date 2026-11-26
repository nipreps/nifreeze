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
"""Do elastix's settings hold up against ANTs on real diffusion-weighted volumes? (#675)

DIPY's Stanford HARDI (150 volumes at b=2000, 10 b=0; already preprocessed), taken as
aligned. Copies move every volume by ``perturbation_plan`` (random axis and direction per
volume, seed 0, about the brain's centroid) at 1, 2, 4 and 8 mm and degrees, through
``perturb`` (cubic), with an unmoved copy (identities) as the control. Every fifth volume
is registered by the accuracy harness (``registration_accuracy``) to the original volume
(``oracle``: the same noise), and to gqi-krr's held-out predictions from the original
(``unmoved``) and from the copy (``moved``). Arms: ``ants`` (today's path) and
configurations of ``elastix_tuning.py`` (``t-x1`` is the direct mapping); one thread, seed
12345. The brain mask is DIPY's ``median_otsu`` of the b=0 volumes. Writes
``<out>/stanford-hardi.tsv``::

    uv run tools/registration/elastix_real_dwi.py --dipy ~/.dipy \\
        --arms ants t-x1 p3-x1 --jobs 6 --out <results>/stanford-hardi
"""

import argparse
import csv
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter

import nibabel as nb
import numpy as np
from dipy.segment.mask import median_otsu
from elastix_tuning import MODEL, SEED, _Failures, _warm, configuration, configurations

import nifreeze.registration.ants  # noqa: F401 (nipype, imported before the workers fork)
from nifreeze.analysis.registration import AntsRegistration, Register, registration_accuracy
from nifreeze.analysis.sensitivity import Perturbation
from nifreeze.data.dmri.io import from_nii
from nifreeze.registration.elastix import Elastix
from nifreeze.testing.perturbations import mask_centroid, perturb, perturbation_plan
from nifreeze.utils.ndimage import load_api

MAGNITUDES = (0.0, 1.0, 2.0, 4.0, 8.0)
"""Motions (mm and degrees); 0 is the control."""
EVERY = 5
"""Every how many volumes are registered."""
CHUNK = 5
"""Volumes per task."""
_SHARED: dict = {}
"""The data, loaded before the workers fork (copy on write)."""


def _register(task: tuple[float, str, list[int]]) -> list[dict]:
    magnitude, arm, indices = task
    moved, motions = _SHARED["copies"][magnitude]
    truth = {i: motions[i] for i in indices}
    with tempfile.TemporaryDirectory(dir=_SHARED["work"]) as workdir:
        backend: Register
        if arm == "ants":
            backend = AntsRegistration(workdir, seed=SEED, num_threads=1)
        else:
            backend = Elastix(configuration(*configurations()[arm]), seed=SEED, num_threads=1)
        register = _Failures(backend)
        dataset = _SHARED["dataset"]
        rows = registration_accuracy(register, dataset, dataset, moved, truth, model=MODEL)
    return [
        {"magnitude": magnitude, "arm": arm} | row | {"failed": failed}
        for row, failed in zip(rows, register.failed, strict=True)
    ]


def _plan(dataset, center: np.ndarray) -> list[Perturbation]:
    """The copies' motions, with the control built from identities.

    ``rigid_motion`` rescales the plan's directions in place (#679), so a zero magnitude
    in the plan would zero them for every later one.
    """
    plan = perturbation_plan(
        len(dataset), [m for m in MAGNITUDES if m], counts=[len(dataset)], seeds=[0], center=center
    )
    everyone = tuple(range(len(dataset)))
    plan.insert(0, Perturbation(0.0, 0, everyone, tuple(np.eye(4) for _ in everyone)))
    for perturbation in plan:
        if not np.isfinite(np.asarray(perturbation.motions)).all():
            raise RuntimeError(f"non-finite motions at {perturbation.magnitude}")
    return plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dipy", type=Path, default=Path(os.environ.get("DIPY_HOME", "~/.dipy")))
    parser.add_argument("--arms", nargs="+", default=["ants", "t-x1", "p3-x1"])
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if unknown := set(args.arms) - {"ants", *configurations()}:
        parser.error(f"unknown arms: {sorted(unknown)}")
    folder = args.dipy.expanduser() / "stanford_hardi"
    args.out.mkdir(parents=True, exist_ok=True)

    image = load_api(folder / "HARDI150.nii.gz", nb.Nifti1Image)
    bvals = np.loadtxt(folder / "HARDI150.bval")
    data = np.asanyarray(image.dataobj, dtype="float32")
    _, mask = median_otsu(data, vol_idx=np.flatnonzero(bvals < 50), median_radius=4, numpass=4)
    mask_path = args.out / "stanford_hardi_desc-brain_mask.nii.gz"
    nb.Nifti1Image(mask.astype("uint8"), image.affine).to_filename(mask_path)
    dataset = from_nii(
        folder / "HARDI150.nii.gz",
        brainmask_file=mask_path,
        bvec_file=folder / "HARDI150.bvec",
        bval_file=folder / "HARDI150.bval",
    )
    copies = {
        p.magnitude: (perturb(dataset, p), dict(zip(p.moved, p.motions, strict=True)))
        for p in _plan(dataset, mask_centroid(dataset))
    }
    _SHARED.update(dataset=dataset, copies=copies)
    volumes = list(range(0, len(dataset), EVERY))
    chunks = [volumes[i : i + CHUNK] for i in range(0, len(volumes), CHUNK)]
    tasks = [(m, arm, c) for m in MAGNITUDES for arm in args.arms for c in chunks]

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
                if done % 10 == 0 or done == len(futures):
                    print(f"{done}/{len(futures)}, {perf_counter() - start:.0f} s", flush=True)
    rows.sort(key=lambda r: (r["magnitude"], args.arms.index(r["arm"]), r["fixed"], r["index"]))
    with (args.out / "stanford-hardi.tsv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print("wrote", args.out / "stanford-hardi.tsv", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
