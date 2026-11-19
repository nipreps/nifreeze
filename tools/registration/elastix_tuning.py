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
"""Do a multi-resolution pyramid or more iterations widen elastix's lead over ANTs? (#675)

Each moved volume of the rigid-motion copies (no eddy currents) is registered with the
rigid default to the harness's three fixed images (:mod:`nifreeze.analysis.registration`),
by today's ANTs path and by elastix (:class:`~nifreeze.registration.elastix.Elastix`) under
each combination of two crossed factors:

schedule
    ``t``, the direct mapping of ANTs' two levels (#674): full-resolution images, whose shrink
    factors only thin the samples. ``p2`` and ``p3``, a pyramid of two (rescaled by 2 and 1)
    or three (4, 2 and 1) resolutions, smoothed by half the rescale factor (voxels) except
    at full resolution. Each level samples a fraction of its own grid's mask voxels and steps
    at most a length: the direct mapping's last level's (0.18, 1.96 mm) at the last level, its
    first level's (0.4, 12 mm) at the others.
iterations
    ``x1``, ``x4`` and ``x16`` times the direct mapping's: 20 at every level but the last,
    10 there.

The optimizer is the direct mapping's (adaptive stochastic gradient descent): the
preconditioned one diverged in the pilot, with these steps and with elastix's default.
``t-x1`` is the direct mapping itself. A registration that raises (such as elastix's
samples mapped outside the moving image) is recorded as ``failed``, with the error of no
correction. Each configuration is written to
``<out>/settings/<name>.yaml`` and registered from that file. One thread per registration,
seed 12345, ``--jobs`` registrations at a time. Writes ``<out>/<protocol>.{tsv,json}``::

    uv run tools/registration/elastix_tuning.py --dataset <dataset> --protocol sherbrooke \\
        --out <dataset>/derivatives/registration/elastix-tuning --jobs 12
"""

import argparse
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import yaml
from experiments import provenance, subset, write_results

import nifreeze.registration.ants  # noqa: F401 (nipype, imported before the workers fork)
from nifreeze.analysis.phantoms import load_phantom
from nifreeze.analysis.registration import (
    FIXED,
    AntsRegistration,
    Register,
    registration_accuracy,
)
from nifreeze.registration.base import RIGID
from nifreeze.registration.elastix import Elastix, get_settings

NOISE = 20261029
"""Seed of the noise of the reference and its copies."""
SEED = 12345
"""Seed of the registrations' random sampling."""
MODEL = "gqi-krr"
"""The model of the ``unmoved`` and ``moved`` targets."""
SETTINGS = RIGID
"""The rigid default, ANTs' and its direct mapping."""
SCHEDULES = {"t": None, "p2": (2, 1), "p3": (4, 2, 1)}
"""The rescale factors of each schedule's pyramid (none for the direct mapping's)."""
ITERATIONS = (1, 4, 16)
"""Multiples of the direct mapping's iterations."""
REFERENCE = "t-x1"
"""The direct mapping (#674), against which the configurations are compared."""
CHUNK = 8
"""Volumes per task."""

_SHARED: dict[str, Any] = {}
"""The data, loaded before the workers fork (copy on write)."""


def configuration(schedule: str, iterations: int) -> dict[str, Any]:
    """
    The settings of one configuration, from the direct mapping.

    >>> configuration("t", 1) == get_settings(SETTINGS)
    True
    >>> stage, = configuration("p3", 4)["stages"]
    >>> stage["shrink_factor"], stage["smoothing_sigma"], stage["sampling_percentage"]
    ([4, 2, 1], [2.0, 1.0, 0.0], [0.4, 0.4, 0.18])
    >>> stage["parameters"]["NumberOfResolutions"]
    ['3']
    >>> [stage["parameters"][key] for key in ("MaximumNumberOfIterations", "MaximumStepLength")]
    [['80', '80', '40'], ['12.0', '12.0', '1.96']]
    >>> stage["parameters"]["ImagePyramidRescaleSchedule"][::3]
    ['4', '2', '1']

    """
    settings = get_settings(SETTINGS)
    first, last = settings["stages"]

    def each(key: str, stage: dict[str, Any]) -> str:
        return stage["parameters"][key][0]

    if (factors := SCHEDULES[schedule]) is None:
        for stage in settings["stages"]:
            count = int(each("MaximumNumberOfIterations", stage))
            stage["parameters"]["MaximumNumberOfIterations"] = [str(count * iterations)]
    else:
        coarse = len(factors) - 1
        parameters = dict(first["parameters"])
        parameters |= {
            "NumberOfResolutions": [str(len(factors))],
            "ImagePyramidRescaleSchedule": [str(f) for f in factors for _ in range(3)],
            "MaximumStepLength": [each("MaximumStepLength", first)] * coarse
            + [each("MaximumStepLength", last)],
            "MaximumNumberOfIterations": [
                str(int(each("MaximumNumberOfIterations", stage)) * iterations)
                for stage in [first] * coarse + [last]
            ],
        }
        settings["stages"] = [
            {
                "sampling_percentage": [first["sampling_percentage"]] * coarse
                + [last["sampling_percentage"]],
                "shrink_factor": list(factors),
                "smoothing_sigma": [f / 2 for f in factors[:-1]] + [0.0],
                "parameters": parameters,
            }
        ]
    return settings


def configurations() -> dict[str, tuple[str, int]]:
    """
    Every configuration's factors, by name.

    >>> names = list(configurations())
    >>> len(names), names[0], names[-1]
    (9, 't-x1', 'p3-x16')

    """
    return {
        f"{schedule}-x{iterations}": (schedule, iterations)
        for schedule, iterations in product(SCHEDULES, ITERATIONS)
    }


def _warm() -> None:
    """Load itk-elastix's modules in each worker, before anything is timed."""
    from scipy import ndimage

    rng = np.random.default_rng(0)
    data = (ndimage.gaussian_filter(rng.random((24, 24, 24)), 2) * 1000).astype("float32")
    mask = np.zeros(data.shape, dtype=bool)
    mask[4:20, 4:20, 4:20] = True
    Elastix(SETTINGS, num_threads=1)(data, data, np.diag([2.0, 2.0, 2.0, 1.0]), mask, 0)


class _Failures:
    """A registration that, when it raises, records a failure and corrects nothing."""

    def __init__(self, register: Any):
        self.register = register
        self.failed: list[bool] = []

    def __call__(self, fixed, moving, affine, mask, index) -> np.ndarray:
        try:
            estimate = self.register(fixed, moving, affine, mask, index)
        except RuntimeError:  # elastix's, such as samples mapped outside the moving image
            self.failed.append(True)
            return np.eye(4)
        self.failed.append(False)
        return estimate


def _register(task: tuple[float, str, list[int]]) -> list[dict[str, Any]]:
    magnitude, arm, indices = task
    moved, motions = _SHARED["copies"][magnitude]
    truth = {index: motions[index] for index in indices}
    data = (_SHARED["clean"], _SHARED["reference"], moved, truth)
    with tempfile.TemporaryDirectory(prefix=f"{arm}-", dir=_SHARED["work"]) as workdir:
        backend: Register
        if arm == "ants":
            backend = AntsRegistration(workdir, seed=SEED, num_threads=1, align_config=SETTINGS)
        else:
            backend = Elastix(_SHARED["settings"] / f"{arm}.yaml", seed=SEED, num_threads=1)
        register = _Failures(backend)
        rows = registration_accuracy(register, *data, model=MODEL, fixed=_SHARED["fixed"])
    # The harness registers in the order of its rows
    return [
        {"magnitude": magnitude, "arm": arm} | row | {"failed": failed}
        for row, failed in zip(rows, register.failed, strict=True)
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", type=Path, required=True, help="phantom dataset root")
    parser.add_argument("--protocol", default="sherbrooke")
    parser.add_argument("--out", type=Path, required=True, help="results folder")
    parser.add_argument("--magnitudes", nargs="+", type=float, default=[0.5, 1.0, 2.0, 4.0])
    parser.add_argument("--fixed", nargs="+", default=list(FIXED), choices=FIXED)
    parser.add_argument("--arms", nargs="+", help="ants and configurations (all if not given)")
    parser.add_argument("--limit", type=int, help="only the first development volumes (pilot)")
    parser.add_argument("--jobs", type=int, default=1, help="registrations at a time")
    parser.add_argument("--work", type=Path, help="ANTs' working folder (a temporary one)")
    args = parser.parse_args(argv)

    names = configurations()
    arms = args.arms or ["ants", *names]
    if unknown := set(arms) - {"ants", *names}:
        parser.error(f"unknown arms: {sorted(unknown)}")
    settings = args.out / "settings"
    settings.mkdir(parents=True, exist_ok=True)
    for name, factors in names.items():
        path = settings / f"{name}.yaml"
        path.write_text(yaml.safe_dump({"elastix": configuration(*factors)}, sort_keys=False))

    start = perf_counter()
    clean, _ = load_phantom(args.dataset, args.protocol, counts=[])
    reference, cells = load_phantom(args.dataset, args.protocol, noise=NOISE)
    volumes = subset(args.protocol, "dev", len(reference))[: args.limit]
    copies = {}
    for magnitude in args.magnitudes:
        (perturbation, load), *others = [
            (p, load)
            for p, load in cells
            if p.distortion is None
            and p.count == len(reference)
            and p.seed == 0
            and p.magnitude == magnitude
        ]
        if others:
            raise RuntimeError(f"more than one rigid copy at {magnitude} mm")
        motions = dict(zip(perturbation.moved, perturbation.motions, strict=True))
        copies[magnitude] = (load(), {i: np.asarray(motions[i]) for i in volumes})
    print(f"loaded {len(copies)} copies in {perf_counter() - start:.0f} s", flush=True)

    chunks = [volumes[i : i + CHUNK] for i in range(0, len(volumes), CHUNK)]
    # The longest first, so that the last tasks are short
    cost = {"ants": 4, **{name: iterations for name, (_, iterations) in names.items()}}
    tasks = sorted(product(args.magnitudes, arms, chunks), key=lambda task: -cost[task[1]])
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="elastix-tuning-", dir=args.work) as work:
        _SHARED.update(
            clean=clean,
            reference=reference,
            copies=copies,
            fixed=args.fixed,
            settings=settings,
            work=work,
        )
        start = perf_counter()
        with ProcessPoolExecutor(
            args.jobs, mp_context=get_context("fork"), initializer=_warm
        ) as pool:
            futures = [pool.submit(_register, task) for task in tasks]
            for done, future in enumerate(as_completed(futures), start=1):
                rows += future.result()
                if done % 10 == 0 or done == len(futures):
                    elapsed = perf_counter() - start
                    print(f"{done}/{len(futures)} tasks, {elapsed:.0f} s", flush=True)

    order = {arm: i for i, arm in enumerate(arms)}
    rows.sort(key=lambda r: (r["magnitude"], order[r["arm"]], FIXED.index(r["fixed"]), r["index"]))
    write_results(
        args.out,
        args.protocol,
        rows,
        {
            "Protocol": args.protocol,
            "Magnitudes": args.magnitudes,
            "Volumes": volumes,
            "VoxelSize": np.linalg.norm(reference.affine[:3, :3], axis=0).tolist(),
            "Fixed": args.fixed,
            "Model": MODEL,
            "Noise": NOISE,
            "Seed": SEED,
            "Settings": SETTINGS,
            "Reference": REFERENCE,
            "Arms": arms,
            "Configurations": {
                name: dict(zip(("Schedule", "Iterations"), factors, strict=True))
                for name, factors in names.items()
            },
            "ThreadsPerRegistration": 1,
            "Jobs": args.jobs,
            "WallSeconds": perf_counter() - start,
            "Provenance": provenance() | {"ItkElastix": _itk_elastix()},
        },
    )
    print(f"wrote {args.out / args.protocol}.tsv", flush=True)
    return 0


def _itk_elastix() -> str:
    from importlib.metadata import version

    return version("itk-elastix")


if __name__ == "__main__":
    raise SystemExit(main())
