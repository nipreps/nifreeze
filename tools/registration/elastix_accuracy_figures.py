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
"""The decision and tables of the elastix accuracy test (#672).

Reads ``<protocol>.{tsv,json,npz}`` of each protocol, which
``tools/registration/elastix_accuracy.py`` wrote, and applies the decision rules of the
working hypothesis. Elastix is compared with ANTs in every cell (protocol, stage and
fixed image) by :func:`~experiments.paired_comparison` of the mean error per
registration, paired by volume and motion, resampled within motions, with a margin of
0.01 voxel; Wilcoxon's p values are Holm-adjusted across the cells. The rules:

accurate
    In every cell, elastix is "better" or "not worse"; none of its registrations
    fails; and, in each protocol and stage, it lands further from the truth than no
    correction no more often than ANTs.
repeatable
    Registered again, elastix gives every estimate bit for bit.
faster
    In each protocol and stage, its median time is below ANTs'.

The outcome is the first that applies: ``less accurate`` (a cell is "worse", a
registration fails, or elastix lands beyond no correction more often), ``not
repeatable``, ``not shown accurate`` (a cell is "inconclusive"), ``slower`` (accurate
and repeatable, but not faster), or ``default`` (all three hold)::

    uv run tools/registration/elastix_accuracy_figures.py \\
        --results <results> --out <folder>
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from experiments import holm, paired_comparison

CANDIDATE = "elastix"
"""Elastix's direct mapping (#674)."""
REFERENCE = "ants"
"""Today's ANTs path."""
REPEAT = "elastix-again"
"""Elastix, registered again."""
NONE = "none"
"""No correction."""
PROTOCOLS = ("sherbrooke", "hcpya")
STAGES = ("rigid", "affine")
FIXED = ("oracle", "unmoved", "moved")
MARGIN = 0.01
"""The margin of the comparisons, in voxels."""


def read(path: Path) -> list[dict[str, Any]]:
    """The rows of a results table, with numbers as numbers."""
    with path.open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    for row in rows:
        for key in ("magnitude", "shell", "max", "mean", "seconds"):
            row[key] = float(row[key])
        row["index"] = int(row["index"])
        row["failed"] = row["failed"] == "True"
    return rows


def _errors(rows: list[dict[str, Any]], arm: str, stage: str, fixed: str) -> dict[tuple, float]:
    return {
        (row["magnitude"], row["index"]): row["mean"]
        for row in rows
        if row["arm"] == arm and row["stage"] == stage and row["fixed"] == fixed
    }


def compare(rows: list[dict[str, Any]], stage: str, fixed: str, margin: float) -> dict[str, Any]:
    """Elastix against ANTs on one stage and fixed image."""
    ours = _errors(rows, CANDIDATE, stage, fixed)
    theirs = _errors(rows, REFERENCE, stage, fixed)
    pairs = sorted(ours)
    if set(pairs) != set(theirs):
        raise ValueError(f"elastix and ANTs are not paired on {stage}, {fixed}")
    return paired_comparison(
        [ours[p] for p in pairs],
        [theirs[p] for p in pairs],
        margin,
        strata=[p[0] for p in pairs],
    )


def beyond(rows: list[dict[str, Any]], arm: str, stage: str) -> int:
    """How many of an arm's registrations end further from the truth than no correction."""
    none = _errors(rows, NONE, NONE, "oracle")
    return sum(
        error > none[pair]
        for fixed in FIXED
        for pair, error in _errors(rows, arm, stage, fixed).items()
    )


def repeated(estimates: Any) -> tuple[int, int]:
    """How many estimates were registered again, and how many came back bit for bit."""
    again = [key for key in estimates.files if f"/{REPEAT}/" in key]
    same = sum(
        np.array_equal(estimates[key], estimates[key.replace(f"/{REPEAT}/", f"/{CANDIDATE}/")])
        for key in again
    )
    return len(again), int(same)


def seconds(rows: list[dict[str, Any]], arm: str, stage: str) -> float:
    """An arm's median seconds per registration in one stage."""
    return float(
        np.median([row["seconds"] for row in rows if row["arm"] == arm and row["stage"] == stage])
    )


def _stage(rows: list[dict[str, Any]], protocol: str, stage: str) -> dict[str, Any]:
    """Failures, registrations beyond no correction and times of both arms in one stage."""
    arms = (CANDIDATE, REFERENCE)
    times = {arm: seconds(rows, arm, stage) for arm in arms}
    return {
        "protocol": protocol,
        "stage": stage,
        "failures": {
            arm: sum(r["failed"] for r in rows if r["arm"] == arm and r["stage"] == stage)
            for arm in arms
        },
        "beyond_none": {arm: beyond(rows, arm, stage) for arm in arms},
        "seconds": times,
        "ratio": times[CANDIDATE] / times[REFERENCE],
    }


def decide(results: dict[str, tuple[list[dict[str, Any]], float, Any]]) -> dict[str, Any]:
    """Apply the decision rules to each protocol's rows, voxel size (mm) and estimates."""
    cells = [
        {"protocol": protocol, "stage": stage, "fixed": fixed}
        | compare(rows, stage, fixed, MARGIN * voxel)
        | {"margin_mm": MARGIN * voxel}
        for protocol, (rows, voxel, _) in results.items()
        for stage in STAGES
        for fixed in FIXED
    ]
    for cell, adjusted in zip(cells, holm([c["wilcoxon_p"] for c in cells]), strict=True):
        cell["holm_p"] = float(adjusted)
    stages = [
        _stage(rows, protocol, stage)
        for protocol, (rows, _, _) in results.items()
        for stage in STAGES
    ]

    counts = [repeated(estimates) for _, _, estimates in results.values()]
    again, same = sum(n for n, _ in counts), sum(s for _, s in counts)

    verdicts = [cell["verdict"] for cell in cells]
    rules = {
        "worse": [c for c in cells if c["verdict"] == "worse"],
        "inconclusive": [c for c in cells if c["verdict"] == "inconclusive"],
        "failures": sum(s["failures"][CANDIDATE] for s in stages),
        "beyond_more": [
            s for s in stages if s["beyond_none"][CANDIDATE] > s["beyond_none"][REFERENCE]
        ],
        "repeated": again,
        "repeated_same": same,
        "slower": [s for s in stages if s["ratio"] >= 1],
    }
    if rules["worse"] or rules["failures"] or rules["beyond_more"]:
        outcome = "less accurate"
    elif not again or same < again:
        outcome = "not repeatable"
    elif rules["inconclusive"]:
        outcome = "not shown accurate"
    elif rules["slower"]:
        outcome = "slower"
    else:
        outcome = "default"
    return {
        "cells": cells,
        "stages": stages,
        "verdicts": {v: verdicts.count(v) for v in dict.fromkeys(verdicts)},
        "rules": {
            "worse": len(rules["worse"]),
            "inconclusive": len(rules["inconclusive"]),
            "failures": rules["failures"],
            "beyond_more": len(rules["beyond_more"]),
            "repeated": again,
            "repeated_same": same,
            "slower": len(rules["slower"]),
        },
        "outcome": outcome,
    }


def tables(decision: dict[str, Any], out: Path) -> None:
    """The comparisons and the times, as CSV tables for the report."""
    with (out / "elastix-accuracy-comparisons.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Protocol", "Stage", "Fixed", "Median (mm)", "95th pct. (mm)", "Verdict"])
        for c in decision["cells"]:
            writer.writerow(
                [
                    c["protocol"],
                    c["stage"],
                    c["fixed"],
                    f"{c['median']:+.3f} [{c['median_low']:+.3f}, {c['median_high']:+.3f}]",
                    f"{c['p95']:+.3f} [{c['p95_low']:+.3f}, {c['p95_high']:+.3f}]",
                    c["verdict"],
                ]
            )
    with (out / "elastix-accuracy-time.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Protocol", "Stage", "ANTs (s)", "Elastix (s)", "Ratio"])
        for s in decision["stages"]:
            writer.writerow(
                [
                    s["protocol"],
                    s["stage"],
                    f"{s['seconds'][REFERENCE]:.2f}",
                    f"{s['seconds'][CANDIDATE]:.2f}",
                    f"{s['ratio']:.2f}",
                ]
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", type=Path, required=True, help="the results folder")
    parser.add_argument("--protocols", nargs="+", default=list(PROTOCOLS))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    results = {}
    for protocol in args.protocols:
        sidecar = json.loads((args.results / f"{protocol}.json").read_text())
        results[protocol] = (
            read(args.results / f"{protocol}.tsv"),
            float(np.mean(sidecar["VoxelSize"])),
            np.load(args.results / f"{protocol}.npz"),
        )
    decision = decide(results)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "elastix-accuracy-decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    tables(decision, args.out)
    print(decision["outcome"], decision["verdicts"], decision["rules"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
