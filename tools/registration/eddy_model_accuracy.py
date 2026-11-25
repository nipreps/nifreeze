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
"""
How accurately does the global eddy model (GEM) recover the phantom's eddy currents?

GEM (:class:`~nifreeze.registration.eddy_field.EddyFieldFit`, one matrix ``C`` and one
exponent ``p``) runs in the estimator's eddy-current pass on the copies of the Sherbrooke
phantom without head motion (:data:`SIZES`: eddy currents whose largest displacement in
the brain is 4, 8 and 16 mm). Head motion is held at the identity: there is no head-motion
step. The fields' zero point is the phantom's (#744). Each copy runs in two conditions
(:data:`NOISE_LEVELS`), with its moving volumes noise-free or with the protocol's noise,
and GEM is fitted to each of three targets (:data:`TARGETS`):

``original``
    each volume's noise-free original, without eddy currents, which never moves;
``gqi-krr``, ``gp-loo``
    each volume's prediction by that model, from the copy.

Each fit is scored against the truth, whose exponent is 1:

- the fitted exponent ``p``;
- the share of the true field recovered: the gain of the estimated slopes on the true
  ones, ``sum(s * t) / sum(t * t)`` over the volumes and the slopes' components, for all
  the volumes and for each shell;
- each volume's field error: the mean, over the brain, of the distance between the points
  to which the estimated and the true mappings move each voxel's centre.

Writes ``<out>/runs/<name>.{json,npz,tsv}`` and ``<out>/logs/<name>.log`` for each run
(:data:`NAMES`) and, gathered, ``<out>/accuracy.{tsv,json}``, from which
``eddy_model_accuracy_figures.py`` draws the figure of ``docs/gem.rst``. On CPUs 0-5,
for instance::

    taskset -c 0-5 python tools/registration/eddy_model_accuracy.py \\
        --dataset <phantoms> --out <results>
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter
from typing import Any

import nibabel as nb
import nitransforms as nt
import numpy as np
from eddy_targets import _dataset_commit
from eddy_targets_expansion import NOISE
from experiments import provenance, write_results

from nifreeze.analysis.phantoms import load_phantom
from nifreeze.estimator import Estimator
from nifreeze.model.base import BaseModel
from nifreeze.registration.eddy_field import EddyField, EddyFieldFit, field_slopes
from nifreeze.registration.utils import displacements_within_mask, phase_encoding_vector

PROTOCOL = "sherbrooke"
"""The phantom's protocol."""
SIZES = (4.0, 8.0, 16.0)
"""The copies' eddy currents: their largest displacement in the brain (mm)."""
NOISE_LEVELS = {"noise-free": None, "noisy": NOISE}
"""The seed of the noise on the moving volumes in each condition (none without it)."""
TARGETS = ("original", "gqi-krr", "gp-loo")
"""What each volume is registered to: its noise-free original, or a model's prediction."""
NAMES = tuple(f"{noise}-e{size:g}" for noise in NOISE_LEVELS for size in SIZES)
"""The runs, one for each condition and copy."""
SEED = 12345
"""Seed of the estimators."""
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
"""The libraries' thread counts, set to each process's."""


class Original(BaseModel):
    """Each volume's noise-free original: the phantom's reference, as it is."""

    def fit_predict(self, index: int | None = None, **kwargs: Any) -> np.ndarray | None:
        """Return the original of volume ``index``."""
        if index is None:
            return None
        return np.asarray(self._dataset.dataobj[index], dtype="float32")


def _parse(name: str) -> tuple[str, float]:
    """A run's condition and its copy's eddy currents (mm)."""
    noise, size = name.rsplit("-", 1)
    return noise, float(size[1:])


def _copy(root: Path, name: str) -> tuple[Any, Any, dict[str, np.ndarray], np.ndarray]:
    """The noise-free original, the copy, its truth and its fields' zero point."""
    noise, size = _parse(name)
    reference, cells = load_phantom(root, PROTOCOL, noise=None)
    if NOISE_LEVELS[noise] is not None:  # the copies with noise; the original stays noise-free
        _, cells = load_phantom(root, PROTOCOL, noise=NOISE_LEVELS[noise])
    chosen = [
        (perturbation, load)
        for perturbation, load in cells
        if perturbation.magnitude == 0
        and perturbation.distortion is not None
        and perturbation.distortion.magnitude == size
        and perturbation.count == 0
        and perturbation.seed in (None, 0)
    ]
    if len(chosen) != 1:
        raise RuntimeError(f"{len(chosen)} copies match {name}, not 1")
    perturbation, load = chosen[0]
    distortion = perturbation.distortion
    assert distortion is not None
    dwi = load()
    if len(dwi) != len(reference):
        raise RuntimeError(f"{name} has {len(dwi)} volumes, its original {len(reference)}")
    sidecar = next((root / "sub-60501" / "dwi").glob(f"*_acq-{PROTOCOL}_dwi.json"))
    dwi.phase_encoding = json.loads(sidecar.read_text())["PhaseEncodingDirection"]
    identity = np.repeat(np.eye(4)[np.newaxis], len(dwi), axis=0)
    truth = {"motion": identity, "eddy": distortion.matrices()}
    return reference, dwi, truth, np.asarray(distortion.center, dtype=float)


def run(args: argparse.Namespace, name: str) -> None:
    """GEM fitted to each target, on one copy; the raw estimates are saved, then scored."""
    runs = args.out / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    began = perf_counter()
    reference, dwi, truth, zero = _copy(args.dataset, name)
    seconds = {"load": perf_counter() - began}
    options = {"omp_nthreads": args.threads, "n_jobs": args.threads, "seed": SEED}
    arrays: dict[str, Any] = {
        "affine": dwi.affine,
        "mask": np.asanyarray(dwi.brainmask, dtype=bool),
        "gradients": dwi.gradients,
        **{f"truth/{key}": value for key, value in truth.items()},
    }
    record: dict[str, Any] = {
        "Name": name,
        "Noise": NOISE_LEVELS[_parse(name)[0]],
        "ZeroPoint": zero.tolist(),
        "PhaseEncoding": dwi.phase_encoding,
    }

    for targets in TARGETS:
        dwi.motion, dwi.eddy = truth["motion"].copy(), None
        model = Original(reference) if targets == "original" else targets
        fit = EddyFieldFit(num_threads=args.threads, zero_point=zero)
        began = perf_counter()
        Estimator(model, estimates="eddy", registration=fit).run(dwi, **options)
        seconds[targets] = perf_counter() - began
        assert dwi.motion is not None and dwi.eddy is not None
        assert isinstance(fit.field, EddyField)
        arrays[f"{targets}/motion"] = dwi.motion.copy()
        arrays[f"{targets}/eddy"] = dwi.eddy.copy()
        record[targets] = {
            "Matrix": fit.field.matrix.tolist(),
            "Exponent": fit.field.exponent,
            "BRef": fit.field.b_ref,
            "Evaluations": fit.evaluations,
            "Messages": fit.messages,
        }
        print(f"{name}: {targets}, {seconds[targets]:.0f} s", flush=True)

    record |= {"Seconds": seconds, "Threads": args.threads, "Provenance": provenance()}
    np.savez_compressed(runs / f"{name}.npz", **arrays)
    (runs / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n")
    score(args.out, name)


def _share(slopes: np.ndarray, truth: np.ndarray) -> float:
    """The gain of the estimated slopes on the true ones."""
    return float(np.sum(slopes * truth) / np.sum(truth * truth))


def score(out: Path, name: str) -> None:
    """Score a run's saved estimates: ``<out>/runs/<name>.tsv``, and its record's summary."""
    runs = out / "runs"
    saved = np.load(runs / f"{name}.npz")
    record = json.loads((runs / f"{name}.json").read_text())
    affine, mask, gradients = saved["affine"], saved["mask"], saved["gradients"]
    u = phase_encoding_vector(record["PhaseEncoding"], affine)
    mask_img = nb.Nifti1Image(mask.astype("uint8"), affine)
    shells = np.round(gradients[:, 3], -2)
    truth = saved["truth/eddy"] @ saved["truth/motion"]
    truth_slopes = field_slopes(saved["truth/eddy"], u)
    found = EddyField.from_slopes(truth_slopes, gradients)
    record["Truth"] = {
        "Exponent": found.exponent,
        "ModelSlopeResidual": float(np.abs(found.slopes(gradients) - truth_slopes).max()),
    }

    columns: dict[str, np.ndarray] = {}
    summary: dict[str, Any] = {}
    for targets in TARGETS:
        slopes = field_slopes(saved[f"{targets}/eddy"], u)
        mappings = saved[f"{targets}/eddy"] @ saved[f"{targets}/motion"]
        errors = np.array(
            [
                np.mean(
                    displacements_within_mask(mask_img, nt.linear.Affine(e), nt.linear.Affine(t))
                )
                for e, t in zip(mappings, truth, strict=True)
            ]
        )
        columns[f"field_error_{targets}"] = errors
        summary[targets] = {
            "Exponent": record[targets]["Exponent"],
            "Share": _share(slopes, truth_slopes),
            "ShareByShell": {
                f"{shell:g}": _share(slopes[shells == shell], truth_slopes[shells == shell])
                for shell in np.unique(shells)
            },
            "FieldErrorMm": float(np.median(errors)),
        }
    record["Summary"] = summary
    (runs / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n")

    noise, size = _parse(name)
    with (runs / f"{name}.tsv").open("w") as stream:
        header = ["name", "noise", "eddy", "index", "shell", *columns]
        stream.write("\t".join(header) + "\n")
        for index in range(len(shells)):
            values = [name, noise, f"{size:g}", str(index), f"{shells[index]:g}"]
            values += [f"{column[index]:.6g}" for column in columns.values()]
            stream.write("\t".join(values) + "\n")
    print(f"{name}: scored", flush=True)


def _launch(args: argparse.Namespace, name: str) -> int:
    """Run one copy in a process of its own, its output to ``<out>/logs``."""
    logs = args.out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = os.environ | {variable: str(args.threads) for variable in THREAD_VARIABLES}
    command = [sys.executable, str(Path(__file__).resolve()), "--dataset", str(args.dataset)]
    command += ["--out", str(args.out), "--threads", str(args.threads), "--run", name]
    with (logs / f"{name}.log").open("w") as log:
        status = subprocess.run(
            command, env=env, cwd=Path(__file__).parent, stdout=log, stderr=subprocess.STDOUT
        ).returncode
    print(f"{name}: {'done' if status == 0 else f'failed ({status})'}", flush=True)
    return status


def gather(args: argparse.Namespace) -> None:
    """Every run's rows in ``<out>/accuracy.tsv``, and the records in its JSON."""
    rows: list[dict[str, Any]] = []
    records = {}
    for name in NAMES:
        table = args.out / "runs" / f"{name}.tsv"
        if not table.exists():
            continue
        lines = table.read_text().splitlines()
        header = lines[0].split("\t")
        rows += [dict(zip(header, line.split("\t"), strict=True)) for line in lines[1:]]
        records[name] = json.loads((args.out / "runs" / f"{name}.json").read_text())
    write_results(
        args.out,
        "accuracy",
        rows,
        {
            "Runs": records,
            "Protocol": PROTOCOL,
            "NoiseSeed": NOISE,
            "Seed": SEED,
            "DatasetCommit": _dataset_commit(args.dataset),
            "Provenance": provenance(),
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", type=Path, required=True, help="phantom dataset root")
    parser.add_argument("--out", type=Path, required=True, help="results folder")
    parser.add_argument("--jobs", type=int, default=3, help="runs at a time")
    parser.add_argument("--threads", type=int, default=3, help="threads per run")
    parser.add_argument("--only", nargs="+", choices=NAMES, help="these runs only")
    parser.add_argument("--gather", action="store_true", help="only gather the scored runs")
    parser.add_argument("--run", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.dataset, args.out = args.dataset.resolve(), args.out.resolve()
    if args.run is not None:
        run(args, args.run)
        return 0
    if not args.gather:
        chosen = args.only or NAMES
        todo = [name for name in chosen if not (args.out / "runs" / f"{name}.tsv").exists()]
        with ThreadPoolExecutor(args.jobs) as pool:
            statuses = list(pool.map(lambda name: _launch(args, name), todo))
        failed = [name for name, status in zip(todo, statuses, strict=True) if status]
        if failed:
            print(f"failed: {failed}", flush=True)
    gather(args)
    print(f"wrote {args.out / 'accuracy.tsv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
