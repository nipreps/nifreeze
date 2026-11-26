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
"""Does elastix, with ANTs' settings mapped directly, register as accurately as ANTs? (#672)

Each held-out volume of the rigid-motion copies (no eddy currents) is registered to the
harness's three fixed images (:mod:`nifreeze.analysis.registration`) by today's ANTs path
(:class:`~nifreeze.analysis.registration.AntsRegistration`) and by elastix's direct
mapping (:class:`~nifreeze.registration.elastix.Elastix`, #674), in two stages:

rigid
    The estimator's default, ``b0-to-b0_level0``.
affine
    The second pass, ``dwi-to-dwi_level1``, started from the same arm's rigid estimate.

The held-out volumes are the confirmation subset (:func:`experiments.subset`), which
#674 and #675 never registered. A registration that raises is recorded as ``failed``,
with the error of no correction, and the affine stage starts from no correction.
At the ``--repeat`` magnitudes, elastix registers every pair again (``elastix-again``):
the two runs' estimates are compared bit for bit. The arm ``none`` is the error of no
correction. One thread per registration, seed 12345, ``--jobs`` volumes at a time.
Writes ``<out>/<protocol>.{tsv,json}`` and the estimates, ``<out>/<protocol>.npz``::

    uv run tools/registration/elastix_accuracy.py --dataset <dataset> --protocol sherbrooke \\
        --out <out> --jobs 6
"""

import argparse
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.metadata import version
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from experiments import provenance, subset, write_results

import nifreeze.registration.ants  # noqa: F401 (nipype, imported before the workers fork)
from nifreeze.analysis.phantoms import load_phantom
from nifreeze.analysis.registration import FIXED, AntsRegistration, registration_accuracy
from nifreeze.model.base import ModelFactory
from nifreeze.registration.base import AFFINE, RIGID
from nifreeze.registration.elastix import Elastix

NOISE = 20261029
"""Seed of the noise of the reference and its copies."""
SEED = 12345
"""Seed of the registrations' random sampling."""
MODEL = "gqi-krr"
"""The model of the ``unmoved`` and ``moved`` targets."""
STAGES = {"rigid": RIGID, "affine": AFFINE}
"""Each stage's settings, ANTs' and their direct mapping (a section of the same file)."""
ARMS = ("ants", "elastix")
"""The two registrations compared."""
REPEAT = "elastix-again"
"""Elastix again, at the ``--repeat`` magnitudes."""

_SHARED: dict[str, Any] = {}
"""The data, loaded before the workers fork (copy on write)."""
_PREDICTIONS: dict[tuple[int, int], np.ndarray] = {}
"""A task's held-out predictions, by dataset and volume, shared by its arms and stages."""


class _Predicted:
    """gqi-krr, whose prediction of each volume is computed once per task."""

    def __init__(self, dataset: Any):
        self.dataset = dataset
        self.model: Any = None

    def fit_predict(self, index: int) -> np.ndarray:
        key = (id(self.dataset), int(index))
        if key not in _PREDICTIONS:
            if self.model is None:
                self.model = ModelFactory.init(MODEL, dataset=self.dataset)
            _PREDICTIONS[key] = np.asarray(self.model.fit_predict(index))
        return _PREDICTIONS[key]


class _Stage:
    """One arm's stage: started from ``init`` (by volume), it records every estimate."""

    def __init__(self, register: Any, init: dict[int, np.ndarray] | None = None):
        self.register = register
        self.init = init
        self.estimates: dict[int, np.ndarray] = {}
        self.failed: dict[int, bool] = {}

    def __call__(self, fixed, moving, affine, mask, index) -> np.ndarray:
        init = None if self.init is None else self.init[index]
        try:
            estimate = self.register(fixed, moving, affine, mask, index, init=init)
        except RuntimeError:  # elastix's, such as samples mapped outside the moving image
            estimate, self.failed[index] = np.eye(4), True
        else:
            self.failed[index] = False
        # A matrix, or a transform once the backends share one interface (#667)
        self.estimates[index] = np.asarray(getattr(estimate, "matrix", estimate))
        return self.estimates[index]


def _backend(arm: str, stage: str, workdir: Path) -> Any:
    if arm == "ants":
        settings = STAGES[stage]
        return AntsRegistration(workdir / stage, seed=SEED, num_threads=1, align_config=settings)
    return Elastix(STAGES[stage], seed=SEED, num_threads=1)


def _warm() -> None:
    """Load itk-elastix's modules in each worker, before anything is timed."""
    from scipy import ndimage

    rng = np.random.default_rng(0)
    data = (ndimage.gaussian_filter(rng.random((24, 24, 24)), 2) * 1000).astype("float32")
    mask = np.zeros(data.shape, dtype=bool)
    mask[4:20, 4:20, 4:20] = True
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    for settings in STAGES.values():
        Elastix(settings, num_threads=1)(data, data, affine, mask, 0)


def _register(index: int) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """Register one volume to each fixed image, by every arm, in both stages."""
    _PREDICTIONS.clear()
    magnitude = _SHARED["magnitude"]
    data = (
        _SHARED["clean"],
        _SHARED["reference"],
        _SHARED["moved"],
        {index: _SHARED["truth"][index]},
    )
    base = {"magnitude": magnitude}
    rows = [
        base | {"arm": "none", "stage": "none"} | row | {"failed": False}
        for row in registration_accuracy(lambda *_: np.eye(4), *data, fixed=["oracle"])
    ]
    estimates = {}
    with tempfile.TemporaryDirectory(prefix=f"{index:05d}-", dir=_SHARED["work"]) as workdir:
        for kind in _SHARED["fixed"]:
            for arm in _SHARED["arms"]:
                init = None
                for stage in STAGES:
                    register = _Stage(_backend(arm, stage, Path(workdir) / arm), init=init)
                    for row in registration_accuracy(
                        register, *data, model=_Predicted, fixed=[kind]
                    ):
                        failed = register.failed[row["index"]]
                        rows.append(base | {"arm": arm, "stage": stage} | row | {"failed": failed})
                    key = f"{magnitude:g}/{arm}/{stage}/{kind}/{index}"
                    estimates[key] = register.estimates[index]
                    init = register.estimates
    return rows, estimates


def _dataset_commit(dataset: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(dataset), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", type=Path, required=True, help="phantom dataset root")
    parser.add_argument("--protocol", default="sherbrooke")
    parser.add_argument("--out", type=Path, required=True, help="results folder")
    parser.add_argument("--magnitudes", nargs="+", type=float, default=[0.5, 1.0, 2.0, 4.0])
    parser.add_argument("--fixed", nargs="+", default=list(FIXED), choices=FIXED)
    parser.add_argument("--repeat", nargs="*", type=float, default=[1.0], help="magnitudes")
    parser.add_argument("--volumes", default="confirm", choices=("dev", "confirm"))
    parser.add_argument("--limit", type=int, help="only the first volumes (pilot)")
    parser.add_argument("--jobs", type=int, default=1, help="volumes at a time")
    parser.add_argument("--work", type=Path, help="ANTs' working folder (a temporary one)")
    args = parser.parse_args(argv)

    commit = _dataset_commit(args.dataset)
    start = perf_counter()
    clean, _ = load_phantom(args.dataset, args.protocol, counts=[])
    reference, cells = load_phantom(args.dataset, args.protocol, noise=NOISE)
    volumes = subset(args.protocol, args.volumes, len(reference))[: args.limit]
    print(f"loaded the reference in {perf_counter() - start:.0f} s", flush=True)

    rows: list[dict[str, Any]] = []
    estimates: dict[str, Any] = {}  # Any: numpy's stubs type np.savez's keywords as options
    wall = perf_counter()
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
        arms = [*ARMS, REPEAT] if magnitude in args.repeat else list(ARMS)
        with tempfile.TemporaryDirectory(prefix="elastix-accuracy-", dir=args.work) as work:
            _SHARED.update(
                clean=clean,
                reference=reference,
                moved=load(),
                truth={i: np.asarray(motions[i]) for i in volumes},
                magnitude=magnitude,
                fixed=args.fixed,
                arms=arms,
                work=work,
            )
            for index in volumes:
                estimates[f"{magnitude:g}/truth/{index}"] = _SHARED["truth"][index]
            start = perf_counter()
            with ProcessPoolExecutor(
                args.jobs, mp_context=get_context("fork"), initializer=_warm
            ) as pool:
                futures = [pool.submit(_register, index) for index in volumes]
                for done, future in enumerate(as_completed(futures), start=1):
                    volume_rows, volume_estimates = future.result()
                    rows += volume_rows
                    estimates |= volume_estimates
                    if done % 6 == 0 or done == len(futures):
                        took = perf_counter() - start
                        print(f"m{magnitude:g}: {done}/{len(futures)}, {took:.0f} s", flush=True)
            _SHARED.pop("moved")

    arms = ["none", *ARMS, REPEAT]
    stages = ["none", *STAGES]
    rows.sort(
        key=lambda r: (
            r["magnitude"],
            arms.index(r["arm"]),
            stages.index(r["stage"]),
            FIXED.index(r["fixed"]),
            r["index"],
        )
    )
    table, _ = write_results(
        args.out,
        args.protocol,
        rows,
        {
            "Protocol": args.protocol,
            "Magnitudes": args.magnitudes,
            "Volumes": volumes,
            "Subset": args.volumes,
            "VoxelSize": np.linalg.norm(reference.affine[:3, :3], axis=0).tolist(),
            "Fixed": args.fixed,
            "Model": MODEL,
            "Noise": NOISE,
            "Seed": SEED,
            "Stages": {stage: path.name for stage, path in STAGES.items()},
            "Arms": list(ARMS),
            "Repeat": args.repeat,
            "ThreadsPerRegistration": 1,
            "Jobs": args.jobs,
            "WallSeconds": perf_counter() - wall,
            # Another checkout of the dataset could change it during the run
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
