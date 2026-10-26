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
"""Rungs 1 to 3 of the sensitivity ladder on one protocol of a phantom (``docs/sensitivity.rst``).

Each rung measures how far the models' held-out predictions drift from those on
the aligned reference when TRXScan's perturbed copies of it are given instead
(``tools/phantoms/phantom.py cells``), with the protocol's noise added on load:

1. rigid motion (nipreps/nifreeze#540): each model's fidelity on the reference;
   the drift when the first K volumes of each moved set move; and the drift
   that one volume causes when it moves alone, against the closed-form
   influence of the linear smoothers;
2. eddy currents (#543): the drift, and the share of each volume's own
   distortion that its prediction inherits;
3. both (#545): the first K volumes moved and every volume distorted.

It runs in NiFreeze's environment, never TRXScan's::

    uv run tools/phantoms/ladder.py run --dataset <dataset> --protocol sherbrooke \\
        --out <dataset>/derivatives/sensitivity

and writes, under ``<out>/<protocol>``, one row per model, copy and evaluated
volume (``rung-<n>.tsv``, ``fidelity.tsv``, ``alone.tsv``), with the settings
(``settings.json``).
"""

import argparse
import csv
import json
import multiprocessing
import os
import sys
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

MODELS = ("avgdwi", "gp-loo", "gqi-krr", "gqi")
"""The shell median, the two closed-form smoothers, and generalized q-sampling."""
SCALED = ("gqi",)
"""Models whose predictions are overscaled (nipreps/nifreeze#495), see :class:`Scaled`."""
FRACTIONS = (1 / 12, 1 / 3, 1 / 2, 1)
"""Moved volumes, as fractions of the diffusion-weighted volumes."""
NOISE = 0
"""Seed of the noise added on load, at each protocol's signal-to-noise ratio."""


class Scaled:
    """A model whose prediction of volume ``i`` is multiplied by ``scales[i]``.

    GQI's held-out predictions are off by a factor that the NRMSE against the
    volume's spread does not absorb (nipreps/nifreeze#495). Each volume's factor
    is fixed once, on the reference (:func:`scales`), so that it scales the
    predictions on every copy alike: the drift scales with it, and the inherited
    share does not change.
    """

    def __init__(self, dataset: Any, name: str, factors: dict[int, float], **kwargs: Any):
        from nifreeze.model.base import ModelFactory

        self._model = ModelFactory.init(name, dataset=dataset, **kwargs)
        self._factors = factors

    def fit_predict(self, index: int, **kwargs: Any) -> np.ndarray:
        return self._factors[int(index)] * np.asarray(self._model.fit_predict(index, **kwargs))


def scales(name: str, reference: Any, evaluate: Iterable[int]) -> dict[int, float]:
    """The median of each volume over the median of its prediction, within the mask."""
    from nifreeze.analysis.sensitivity import held_out_predictions

    mask = np.asarray(reference.brainmask, dtype=bool)
    predicted = held_out_predictions(name, reference, evaluate)
    return {
        index: float(
            np.median(np.asarray(reference.dataobj[index])[mask]) / np.median(prediction[mask])
        )
        for index, prediction in predicted.items()
    }


def counts(n_volumes: int) -> list[int]:
    """The numbers of moved volumes: N/12, N/3, N/2 and N, rounded."""
    return sorted({max(1, round(n_volumes * f)) for f in FRACTIONS})


def run_model(
    name: str,
    dataset: Path,
    protocol: str,
    out: Path,
    evaluate: list[int],
    alone: list[int],
) -> str:
    """All three rungs of one model, each written as soon as it is measured."""
    from nifreeze.analysis.phantoms import load_phantom
    from nifreeze.analysis.sensitivity import fidelity, influence, sensitivity
    from nifreeze.model.base import ModelFactory

    n_volumes = len_dw(dataset, protocol)
    reference, cells = load_phantom(dataset, protocol, noise=NOISE, counts=counts(n_volumes))
    model: str | Callable[..., Any] = name
    if name in SCALED:
        model = partial(Scaled, name=name, factors=scales(name, reference, evaluate))

    def tagged(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{"model": name, **row} for row in rows]

    _write(out / f"fidelity_model-{name}.tsv", tagged(fidelity(model, reference, evaluate)))

    still = [(p, load) for p, load in cells if p.distortion is None]
    rows = sensitivity(model, reference, still, evaluate=evaluate)
    _write(out / f"rung-1_model-{name}.tsv", tagged(rows))

    eddy = [(p, load) for p, load in cells if p.distortion is not None and not p.motions]
    rows = sensitivity(model, reference, eddy, evaluate=evaluate)
    _write(out / f"rung-2_model-{name}.tsv", tagged(rows))

    # One count at a time, so that only the predictions of its moved copies are kept
    both: list[dict[str, Any]] = []
    for count in counts(n_volumes):
        group = [(p, load) for p, load in cells if p.motions and p.count == count]
        rows = sensitivity(model, reference, group, evaluate=evaluate)
        both += [row for row in rows if row["distortion"] > 0]
    _write(out / f"rung-3_model-{name}.tsv", tagged(both))

    # Each volume moved alone, by the largest motion; its change against the reference
    # comes from the copy that moves every volume
    largest = max(p.magnitude for p, _ in still)
    _, lonely = load_phantom(dataset, protocol, noise=NOISE, counts=[], alone=alone)
    lonely = [(p, load) for p, load in lonely if p.distortion is None and p.magnitude == largest]
    everyone = next(load() for p, load in still if p.count == n_volumes and p.magnitude == largest)
    weights = None
    if name in ("gp-loo", "gqi-krr"):
        weights = influence(ModelFactory.init(name, dataset=reference))
    rows = sensitivity(model, reference, lonely, evaluate=evaluate)
    measured = []
    for k, (perturbation, _) in enumerate(lonely):
        (volume,) = perturbation.moved
        drift = [
            row["nrmse"]
            for row in rows[k * len(evaluate) : (k + 1) * len(evaluate)]
            if row["index"] != volume
        ]
        change = np.asarray(everyone.dataobj[volume]) - np.asarray(reference.dataobj[volume])
        measured.append(
            {
                "model": name,
                "magnitude": perturbation.magnitude,
                "volume": volume,
                "shell": float(np.round(reference.gradients[volume, -1], -2)),
                "drift": float(np.mean(drift)),
                "change": float(np.linalg.norm(change[reference.brainmask])),
                "influence": float("nan") if weights is None else float(weights[volume]),
            }
        )
    _write(out / f"alone_model-{name}.tsv", measured)
    return name


def len_dw(dataset: Path, protocol: str) -> int:
    """Number of diffusion-weighted volumes of a protocol's reference."""
    from nifreeze.analysis.phantoms import SUBJECT
    from nifreeze.data.dmri.utils import DEFAULT_LOWB_THRESHOLD

    table = Path(dataset) / f"sub-{SUBJECT}" / "dwi" / f"sub-{SUBJECT}_acq-{protocol}_dwi.b"
    return int((np.loadtxt(table, ndmin=2)[:, 3] > DEFAULT_LOWB_THRESHOLD).sum())


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        if not rows:
            return
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=["run"])
    parser.add_argument("--dataset", type=Path, required=True, help="phantom dataset root")
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--out", type=Path, required=True, help="results root")
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    parser.add_argument(
        "--every", type=int, default=1, help="evaluate every n-th volume (bounds memory)"
    )
    parser.add_argument(
        "--alone-every", type=int, default=4, help="move every n-th evaluated volume alone"
    )
    parser.add_argument("--jobs", type=int, default=len(MODELS), help="models run at once")
    args = parser.parse_args(argv)

    n_volumes = len_dw(args.dataset, args.protocol)
    evaluate = list(range(0, n_volumes, args.every))
    alone = evaluate[:: args.alone_every]
    out = args.out / args.protocol
    out.mkdir(parents=True, exist_ok=True)
    settings = {
        "Protocol": args.protocol,
        "Models": args.models,
        "Noise": NOISE,
        "Counts": counts(n_volumes),
        "Evaluate": evaluate,
        "Alone": alone,
        "Scaled": [m for m in args.models if m in SCALED],
        "NiFreezeVersion": version("nifreeze"),
    }
    (out / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")

    jobs = max(1, min(args.jobs, len(args.models)))
    # Each worker gets its share of the BLAS threads: spawned workers read them on import
    threads = str(max(1, len(os.sched_getaffinity(0)) // jobs))
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = threads
    task = partial(
        run_model,
        dataset=args.dataset,
        protocol=args.protocol,
        out=out,
        evaluate=evaluate,
        alone=alone,
    )
    with ProcessPoolExecutor(jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
        for name in pool.map(task, args.models):
            print(f"{args.protocol}: {name} done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
