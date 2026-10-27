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

and writes, under ``<out>/<protocol>``, one row per copy and evaluated volume
(``rung-<n>_model-<model>.tsv``, ``fidelity_model-<model>.tsv``), one per volume
moved alone (``alone_model-<model>.tsv``), and the settings (``settings.json``).
The tables of ``docs/sensitivity.rst`` summarize every protocol of ``<out>``::

    uv run tools/phantoms/ladder.py tables --results <dataset>/derivatives/sensitivity \\
        --out <dataset>/derivatives/sensitivity/tables
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
LABELS = ("N/12", "N/3", "N/2", "N")
"""How the tables name the counts of :data:`FRACTIONS`."""
TABLES = ("rung-1", "alone", "rung-2", "rung-3", "rung-3-compound", "rung-3-share")
"""The tables of :func:`tables`, each written as ``<table>.csv``."""
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


def tables(results: Path, out: Path) -> list[Path]:
    """Summaries of every protocol under ``results``, one CSV per table, as the docs show them.

    A drift is the mean NRMSE over a shell's evaluated volumes. Motions are the largest
    magnitude, and eddy currents the largest distortion, wherever a table does not vary them.
    """
    written: dict[str, list[dict[str, Any]]] = {name: [] for name in TABLES}
    for folder in sorted(p.parent for p in results.glob("*/settings.json")):
        settings = json.loads((folder / "settings.json").read_text())
        for name in settings["Models"]:
            for table, rows in _summaries(folder, settings, name).items():
                written[table] += rows

    out.mkdir(parents=True, exist_ok=True)
    paths = [out / f"{table}.csv" for table in written]
    for path, rows in zip(paths, written.values(), strict=True):
        _write(path, [{k: _round(v) for k, v in row.items()} for row in rows], ",")
    return paths


def _summaries(
    folder: Path, settings: dict[str, Any], name: str
) -> dict[str, list[dict[str, Any]]]:
    """The rows that one model of one protocol adds to each table."""
    from scipy.stats import spearmanr

    n_volumes = settings["Counts"][-1]
    sizes = {
        label: max(1, round(n_volumes * f)) for label, f in zip(LABELS, FRACTIONS, strict=True)
    }
    fidelity = _read(folder / f"fidelity_model-{name}.tsv")
    still = _read(folder / f"rung-1_model-{name}.tsv")
    eddy = _read(folder / f"rung-2_model-{name}.tsv")
    both = _read(folder / f"rung-3_model-{name}.tsv")
    alone = _read(folder / f"alone_model-{name}.tsv")
    shell_of = {row["index"]: row["shell"] for row in still}
    largest = max(row["magnitude"] for row in still)
    strongest = max(row["distortion"] for row in eddy)
    tag = {"protocol": settings["Protocol"], "model": name}

    summaries: dict[str, list[dict[str, Any]]] = {table: [] for table in TABLES}
    for shell in sorted(set(shell_of.values())):
        at = {**tag, "shell": shell}
        moved = _select(still, "nrmse", shell=shell, magnitude=largest, count=sizes["N/3"])
        cc = [row["cc"] for row in fidelity if shell_of[row["index"]] == shell]
        summaries["rung-1"].append(
            {**at, "fidelity": np.mean(cc), "drift": moved.mean(), "drift_sd": moved.std()}
        )
        share = _select(eddy, "inherited", shell=shell)
        drift = _select(eddy, "nrmse", shell=shell, distortion=strongest)
        summaries["rung-2"].append(
            {
                **at,
                "inherited": np.nanmean(share),
                "inherited_q75": np.nanquantile(share, 0.75),
                "drift": drift.mean(),
            }
        )
        largest_both = {"shell": shell, "magnitude": largest, "distortion": strongest}
        summaries["rung-3"].append(
            {
                **at,
                **{
                    f"K={label}": _select(both, "nrmse", count=size, **largest_both).mean()
                    for label, size in sizes.items()
                },
            }
        )
        summaries["rung-3-share"].append(
            {
                **at,
                "no motion": np.nanmean(share),
                **{
                    f"K={label}": np.nanmean(
                        _select(both, "inherited", shell=shell, magnitude=largest, count=size)
                    )
                    for label, size in sizes.items()
                    if label in ("N/3", "N")
                },
            }
        )

    # The drift with both, over the quadrature sum of the drifts with each alone
    ratios = []
    for cell in _cells(both, ("shell", "magnitude", "count", "distortion")):
        motion = {k: v for k, v in cell.items() if k != "distortion"}
        distortion = {"shell": cell["shell"], "distortion": cell["distortion"]}
        separate = np.hypot(
            _select(still, "nrmse", **motion).mean(), _select(eddy, "nrmse", **distortion).mean()
        )
        ratios.append(_select(both, "nrmse", **cell).mean() / separate)
    summaries["rung-3-compound"].append(
        {**tag, "ratio_min": min(ratios), "ratio_max": max(ratios)}
    )

    change = _select(alone, "change")
    spread = _select(alone, "drift")
    weights = _select(alone, "influence")
    summaries["alone"].append(
        {
            **tag,
            "volumes": len(alone),
            "rho_change": spearmanr(spread, change).statistic,
            "rho_influence": (
                float("nan")
                if np.isnan(weights).all()
                else spearmanr(spread, weights * change).statistic
            ),
        }
    )
    return summaries


def _read(path: Path) -> list[dict[str, Any]]:
    """The rows of a TSV, with numbers and booleans parsed."""

    def parse(value: str) -> Any:
        if value in ("True", "False"):
            return value == "True"
        try:
            number = float(value)
        except ValueError:
            return value
        return int(number) if number.is_integer() and "." not in value else number

    with path.open() as stream:
        return [
            {key: parse(value) for key, value in row.items()}
            for row in csv.DictReader(stream, delimiter="\t")
        ]


def _select(rows: list[dict[str, Any]], column: str, **where: Any) -> np.ndarray:
    """A column over the rows that match every condition."""
    return np.array(
        [row[column] for row in rows if all(row[k] == v for k, v in where.items())], dtype=float
    )


def _cells(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    """The distinct combinations of some columns, in order of appearance."""
    distinct = dict.fromkeys(tuple(row[k] for k in keys) for row in rows)
    return [dict(zip(keys, values, strict=True)) for values in distinct]


def _round(value: Any) -> Any:
    """Three significant decimals for floats, as the docs show them."""
    return f"{value:.3f}" if isinstance(value, float | np.floating) else value


def _write(path: Path, rows: list[dict[str, Any]], delimiter: str = "\t") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        if not rows:
            return
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="measure the rungs on one protocol")
    run.add_argument("--dataset", type=Path, required=True, help="phantom dataset root")
    run.add_argument("--protocol", required=True)
    run.add_argument("--out", type=Path, required=True, help="results root")
    run.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    run.add_argument(
        "--every", type=int, default=1, help="evaluate every n-th volume (bounds memory)"
    )
    run.add_argument(
        "--alone-every", type=int, default=4, help="move every n-th evaluated volume alone"
    )
    run.add_argument("--jobs", type=int, default=len(MODELS), help="models run at once")
    summary = commands.add_parser("tables", help="summarize every protocol of a results root")
    summary.add_argument("--results", type=Path, required=True, help="results root")
    summary.add_argument("--out", type=Path, required=True, help="folder of the tables")
    args = parser.parse_args(argv)
    if args.command == "tables":
        for path in tables(args.results, args.out):
            print(path, flush=True)
        return 0

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
