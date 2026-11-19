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
"""The decision and tables of the elastix tuning experiment (#675).

Reads ``<protocol>.{tsv,json}``, which ``tools/registration/elastix_tuning.py`` wrote, and
applies the decision rules of the working hypothesis. Each configuration is compared
with the translation (``t-x1``) on each fixed image, by
:func:`~experiments.paired_comparison` of the mean error per registration, paired by
volume and motion, resampled within motions, with a margin of 0.01 voxel. A
configuration passes if no registration failed, if, on every fixed image, it is not
worse in the median or at the 95th percentile (the upper bound of the difference below
the margin), if it is better in the median on at least one, and if its median time is
below ANTs'. Of those
that pass, the one with the lowest median difference (averaged over the fixed images)
is chosen, or, within the margin of it, the fastest::

    uv run tools/registration/elastix_tuning_figures.py \\
        --results <dataset>/derivatives/registration/elastix-tuning --out <folder>
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from experiments import holm, paired_comparison

REFERENCE = "t-x1"
"""The translation (#674)."""
BASELINE = "ants"
"""Today's ANTs path."""
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


def _by_pair(rows: list[dict[str, Any]], arm: str, fixed: str) -> dict[tuple, dict[str, Any]]:
    return {
        (row["magnitude"], row["index"]): row
        for row in rows
        if row["arm"] == arm and row["fixed"] == fixed
    }


def compare(
    rows: list[dict[str, Any]], candidate: str, reference: str, fixed: str, margin: float
) -> dict[str, Any]:
    """:func:`~experiments.paired_comparison` of two arms' mean errors on one fixed image."""
    ours, theirs = _by_pair(rows, candidate, fixed), _by_pair(rows, reference, fixed)
    pairs = sorted(ours.keys() & theirs.keys())
    if len(pairs) != len(ours) or len(pairs) != len(theirs):
        raise ValueError(f"{candidate} and {reference} are not paired on {fixed}")
    result = paired_comparison(
        [ours[p]["mean"] for p in pairs],
        [theirs[p]["mean"] for p in pairs],
        margin,
        strata=[p[0] for p in pairs],
    )
    return {"candidate": candidate, "reference": reference, "fixed": fixed} | result


def decide(rows: list[dict[str, Any]], voxel: float) -> dict[str, Any]:
    """Apply the decision rules: the comparisons, each configuration's verdict, the choice."""
    margin = MARGIN * voxel
    arms = list(dict.fromkeys(row["arm"] for row in rows))
    fixed = list(dict.fromkeys(row["fixed"] for row in rows))
    candidates = [arm for arm in arms if arm not in (REFERENCE, BASELINE)]
    seconds = {
        arm: float(np.median([row["seconds"] for row in rows if row["arm"] == arm]))
        for arm in arms
    }

    comparisons = [
        compare(rows, arm, reference, kind, margin)
        for reference in (REFERENCE, BASELINE)
        for arm in arms
        if arm != reference and not (reference == REFERENCE and arm == BASELINE)
        for kind in fixed
    ]
    for reference in (REFERENCE, BASELINE):
        chosen = [c for c in comparisons if c["reference"] == reference]
        for comparison, adjusted in zip(
            chosen, holm([c["wilcoxon_p"] for c in chosen]), strict=True
        ):
            comparison["holm_p"] = float(adjusted)

    verdicts: dict[str, dict[str, Any]] = defaultdict(dict)
    for comparison in comparisons:
        if comparison["reference"] == REFERENCE:
            verdicts[comparison["candidate"]][comparison["fixed"]] = comparison
    configurations = {}
    for arm in candidates:
        each = verdicts[arm]
        median = all(each[k]["verdict"] in ("better", "not worse") for k in fixed)
        tail = all(each[k]["p95_high"] < margin for k in fixed)
        better = any(each[k]["verdict"] == "better" for k in fixed)
        faster = seconds[arm] < seconds[BASELINE] if BASELINE in seconds else None
        failures = sum(row["failed"] for row in rows if row["arm"] == arm)
        accurate = bool(median and tail and better and not failures)
        configurations[arm] = {
            "failures": failures,
            "not_worse_median": median,
            "not_worse_tail": tail,
            "better_somewhere": better,
            "faster_than_ants": faster,
            "passes": bool(accurate and faster),
            "accurate": accurate,
            "median_difference": float(np.mean([each[k]["median"] for k in fixed])),
            "seconds": seconds[arm],
        }

    passing = [arm for arm, c in configurations.items() if c["passes"]]
    if passing:
        best = min(configurations[arm]["median_difference"] for arm in passing)
        close = [
            arm for arm in passing if configurations[arm]["median_difference"] < best + margin
        ]
        choice = min(close, key=lambda arm: configurations[arm]["seconds"])
        outcome = f"adopt {choice}"
    elif slower := [arm for arm, c in configurations.items() if c["accurate"]]:
        choice, outcome = None, "trade-off: " + ", ".join(slower)
    else:
        choice, outcome = None, "keep the translation"
    return {
        "margin_mm": margin,
        "seconds": seconds,
        "comparisons": comparisons,
        "configurations": configurations,
        "choice": choice,
        "outcome": outcome,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", type=Path, required=True, help="the results folder")
    parser.add_argument("--protocol", default="sherbrooke")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    sidecar = json.loads((args.results / f"{args.protocol}.json").read_text())
    voxel = float(np.mean(sidecar["VoxelSize"]))
    decision = decide(read(args.results / f"{args.protocol}.tsv"), voxel)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.protocol}-decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    print(decision["outcome"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
