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
"""The decision, tables and figures of the elastix accuracy test (#672).

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
and repeatable, but not faster), or ``default`` (all three hold).

Beside the decision, it writes what the report gives but the rules don't use (the
comparisons by motion, the median errors, the error of no correction) and, with
``--figures``, the report's two figures::

    uv run tools/registration/elastix_accuracy_figures.py --results <results> \\
        --out docs/_static/registration/elastix-accuracy --figures
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


def context(results: dict[str, tuple[list[dict[str, Any]], float, Any]]) -> dict[str, Any]:
    """What the report gives beside the decision: by motion, the errors, no correction."""
    motion, errors, none = [], [], []
    for protocol, (rows, voxel, _) in results.items():
        uncorrected = _errors(rows, NONE, NONE, "oracle")
        magnitudes = sorted({m for m, _ in uncorrected})
        none += [
            {
                "protocol": protocol,
                "magnitude": m,
                "median": float(np.median([e for (mm, _), e in uncorrected.items() if mm == m])),
            }
            for m in magnitudes
        ]
        for stage in STAGES:
            for fixed in FIXED:
                ours = _errors(rows, CANDIDATE, stage, fixed)
                theirs = _errors(rows, REFERENCE, stage, fixed)
                for m in magnitudes:
                    pairs = sorted(p for p in ours if p[0] == m)
                    motion.append(
                        {"protocol": protocol, "stage": stage, "fixed": fixed, "magnitude": m}
                        | paired_comparison(
                            [ours[p] for p in pairs], [theirs[p] for p in pairs], MARGIN * voxel
                        )
                    )
                errors.append(
                    {"protocol": protocol, "stage": stage, "fixed": fixed}
                    | {
                        f"{arm}_{key}": float(
                            np.median(
                                [
                                    r[key]
                                    for r in rows
                                    if r["arm"] == arm
                                    and r["stage"] == stage
                                    and r["fixed"] == fixed
                                ]
                            )
                        )
                        for arm in (REFERENCE, CANDIDATE)
                        for key in ("mean", "max")
                    }
                )
    return {"motion": motion, "errors": errors, "none": none}


PROTOCOL_NAMES = {"sherbrooke": "Sherbrooke (2 mm)", "hcpya": "HCP-YA (1.25 mm)"}
FIXED_NAMES = {
    "oracle": "Oracle",
    "unmoved": "Prediction, unmoved data",
    "moved": "Prediction, moved data",
}


def _interval(value: float, low: float, high: float) -> str:
    return f"{value:+.3f} [{low:+.3f}, {high:+.3f}]"


def _write(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def tables(decision: dict[str, Any], reported: dict[str, Any], out: Path) -> None:
    """The comparisons, times, errors and comparisons by motion, as the report's tables."""
    _write(
        out / "elastix-accuracy-comparisons.csv",
        [
            "Protocol",
            "Stage",
            "Fixed image",
            "Elastix - ANTs, median (mm)",
            "95th percentile (mm)",
            "Holm p",
            "Verdict",
        ],
        [
            [
                PROTOCOL_NAMES[c["protocol"]],
                c["stage"],
                FIXED_NAMES[c["fixed"]],
                _interval(c["median"], c["median_low"], c["median_high"]),
                _interval(c["p95"], c["p95_low"], c["p95_high"]),
                f"{c['holm_p']:.1g}",
                c["verdict"],
            ]
            for c in decision["cells"]
        ],
    )
    _write(
        out / "elastix-accuracy-time.csv",
        ["Protocol", "Stage", "ANTs (s)", "Elastix (s)", "Ratio"],
        [
            [
                PROTOCOL_NAMES[s["protocol"]],
                s["stage"],
                f"{s['seconds'][REFERENCE]:.2f}",
                f"{s['seconds'][CANDIDATE]:.2f}",
                f"{s['ratio']:.2f}",
            ]
            for s in decision["stages"]
        ],
    )
    _write(
        out / "elastix-accuracy-errors.csv",
        [
            "Protocol",
            "Stage",
            "Fixed image",
            "Mean, ANTs (mm)",
            "Mean, elastix (mm)",
            "Maximum, ANTs (mm)",
            "Maximum, elastix (mm)",
        ],
        [
            [
                PROTOCOL_NAMES[e["protocol"]],
                e["stage"],
                FIXED_NAMES[e["fixed"]],
                *(
                    f"{e[f'{arm}_{key}']:.3f}"
                    for key in ("mean", "max")
                    for arm in ("ants", "elastix")
                ),
            ]
            for e in reported["errors"]
        ],
    )
    magnitudes = sorted({m["magnitude"] for m in reported["motion"]})
    by_cell: dict[tuple, dict[float, dict[str, Any]]] = {}
    for m in reported["motion"]:
        by_cell.setdefault((m["protocol"], m["stage"], m["fixed"]), {})[m["magnitude"]] = m
    _write(
        out / "elastix-accuracy-motion.csv",
        ["Protocol", "Stage", "Fixed image", *(f"{m:g} mm" for m in magnitudes)],
        [
            [
                PROTOCOL_NAMES[protocol],
                stage,
                FIXED_NAMES[fixed],
                *(f"{cells[m]['median']:+.3f}, {cells[m]['verdict']}" for m in magnitudes),
            ]
            for (protocol, stage, fixed), cells in by_cell.items()
        ],
    )
    _write(
        out / "elastix-accuracy-none.csv",
        ["Protocol", *(f"{m:g} mm" for m in magnitudes)],
        [
            [
                PROTOCOL_NAMES[protocol],
                *(f"{n['median']:.2f}" for n in reported["none"] if n["protocol"] == protocol),
            ]
            for protocol in dict.fromkeys(n["protocol"] for n in reported["none"])
        ],
    )


INK, MUTED, RULE = "#0b0b0b", "#52514e", "#d9d8d4"
COLORS = {"oracle": "#2a78d6", "unmoved": "#eb6834", "moved": "#1baf7a"}
"""Three slots validated for color-vision deficiencies (the figures of #609)."""
MARKERS = {"oracle": "o", "unmoved": "s", "moved": "^"}


def _style() -> Any:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
        }
    )
    return plt


def _margins(ax: Any, margin: float, vertical: bool) -> None:
    """The zero line and the margin on either side of it."""
    line = ax.axvline if vertical else ax.axhline
    line(0, color=RULE, linewidth=0.8, zorder=0)
    for value in (-margin, margin):
        line(value, color=MUTED, linewidth=0.7, linestyle=":", zorder=0)


def forest(decision: dict[str, Any], out: Path) -> None:
    """Each cell's median difference with its interval, against the margin."""
    from matplotlib.ticker import MaxNLocator

    plt = _style()
    protocols = list(dict.fromkeys(c["protocol"] for c in decision["cells"]))
    fig, axes = plt.subplots(
        1, len(protocols), figsize=(10, 3.6), layout="constrained", sharey=True, squeeze=False
    )
    for ax, protocol in zip(axes[0], protocols, strict=True):
        cells = [c for c in decision["cells"] if c["protocol"] == protocol]
        margin = cells[0]["margin_mm"]
        _margins(ax, margin, vertical=True)
        ys = [i + (i // len(FIXED)) * 0.8 for i in range(len(cells))][::-1]
        for y, c in zip(ys, cells, strict=True):
            color = COLORS[c["fixed"]]
            ax.plot([c["median_low"], c["median_high"]], [y, y], color=color, linewidth=1.6)
            ax.scatter([c["median"]], [y], color=color, marker=MARKERS[c["fixed"]], s=28, zorder=3)
            ax.text(
                1.01,
                y,
                c["verdict"],
                transform=ax.get_yaxis_transform(),
                va="center",
                color=INK,
                fontsize=8,
            )
        for value, name in ((-margin, "-margin"), (margin, "+margin")):
            ax.text(
                value,
                1.0,
                name,
                transform=ax.get_xaxis_transform(),
                va="bottom",
                ha="center",
                color=MUTED,
                fontsize=7.5,
            )
        ax.set_yticks(ys, [FIXED_NAMES[c["fixed"]] for c in cells])
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.set_title(PROTOCOL_NAMES[protocol], color=INK, pad=14)
        ax.set_xlabel("Elastix - ANTs, median of the mean error (mm)")
        ax.xaxis.set_major_locator(MaxNLocator(5))
    first = axes[0, 0]
    ys = [i + (i // len(FIXED)) * 0.8 for i in range(len(STAGES) * len(FIXED))][::-1]
    for g, stage in enumerate(STAGES):
        first.text(
            -0.02,
            ys[g * len(FIXED)] + 0.75,
            f"{stage.capitalize()} stage",
            transform=first.get_yaxis_transform(),
            ha="right",
            va="center",
            color=INK,
            fontweight="bold",
        )
    first.set_ylim(min(ys) - 0.6, max(ys) + 1.1)
    fig.savefig(out / "elastix-accuracy-cells.png", dpi=200, facecolor="white")
    plt.close(fig)


def by_motion(decision: dict[str, Any], reported: dict[str, Any], out: Path) -> None:
    """The median difference at each motion, by protocol, stage and fixed image."""
    plt = _style()
    protocols = list(dict.fromkeys(c["protocol"] for c in decision["cells"]))
    margin = {c["protocol"]: c["margin_mm"] for c in decision["cells"]}
    fig, axes = plt.subplots(
        len(protocols),
        len(STAGES),
        figsize=(10, 6),
        layout="constrained",
        sharey="row",
        squeeze=False,
    )
    offsets = {"oracle": 0.92, "unmoved": 1.0, "moved": 1.08}
    for r_, protocol in enumerate(protocols):
        for c_, stage in enumerate(STAGES):
            ax = axes[r_, c_]
            _margins(ax, margin[protocol], vertical=False)
            for fixed in FIXED:
                points = sorted(
                    (
                        m
                        for m in reported["motion"]
                        if (m["protocol"], m["stage"], m["fixed"]) == (protocol, stage, fixed)
                    ),
                    key=lambda m: m["magnitude"],
                )
                xs = np.array([m["magnitude"] for m in points]) * offsets[fixed]
                color = COLORS[fixed]
                ax.vlines(
                    xs,
                    [m["median_low"] for m in points],
                    [m["median_high"] for m in points],
                    color=color,
                    linewidth=1.0,
                    alpha=0.7,
                )
                ax.plot(
                    xs,
                    [m["median"] for m in points],
                    color=color,
                    linewidth=1.4,
                    marker=MARKERS[fixed],
                    markersize=5,
                    label=FIXED_NAMES[fixed],
                )
            ax.set_xscale("log", base=2)
            ax.set_xticks([0.5, 1, 2, 4], ["0.5", "1", "2", "4"])
            ax.minorticks_off()
            if r_ == 0:
                ax.set_title(f"{stage.capitalize()} stage", color=INK)
            if c_ == 0:
                ax.set_ylabel(f"{PROTOCOL_NAMES[protocol]}\nelastix - ANTs (mm)")
            if r_ == len(protocols) - 1:
                ax.set_xlabel("Motion (mm and degrees)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=len(FIXED), frameon=False)
    fig.savefig(out / "elastix-accuracy-motion.png", dpi=200, facecolor="white")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", type=Path, required=True, help="the results folder")
    parser.add_argument("--protocols", nargs="+", default=list(PROTOCOLS))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--figures", action="store_true", help="also the report's figures")
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
    reported = context(results)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "elastix-accuracy-decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    tables(decision, reported, args.out)
    if args.figures:
        forest(decision, args.out)
        by_motion(decision, reported, args.out)
    print(decision["outcome"], decision["verdicts"], decision["rules"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
