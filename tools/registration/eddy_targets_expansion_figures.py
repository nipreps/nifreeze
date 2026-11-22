# /// script
# requires-python = ">=3.13"
# dependencies = ["nifreeze", "matplotlib"]
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
"""The decision, tables and figure of Test 2 on #723: why the registrations expand.

Reads ``<protocol>.{tsv,json,npz}``, which ``tools/registration/eddy_targets_expansion.py``
wrote. The measure is each volume's volume change (``volume_change``); the margin is the
volume change that moves the brain's mean radius by 0.01 voxel. The rules:

R0, the registration of an image onto itself
    In the noise-free ``unmoved`` copy with the default settings, the moving image is the
    target. The 95% interval of its median volume change must lie within the margin.
R1, the data
    With the default settings, paired by volume (:func:`~experiments.paired_comparison`
    on the volume change, stratified by shell): the noise (``m2s0e2`` with against
    without), the noise alone (``unmoved`` with against without), the eddy currents
    (``m2s0e2`` against ``m2s0``, with the noise) and the head motion (``m2s0e2`` against
    ``e2``, with the noise). A factor drives the expansion if it is "worse" and, without
    it, the median volume change has its interval within the margin; a factor that is
    "worse" while the expansion stays without it modulates the expansion.
R2, the settings
    On ``m2s0e2`` with the noise, each variant against the default: a setting causes the
    expansion if removing it is "better" and the variant's median volume change has its
    interval within the margin. The same comparisons on ``unmoved`` are reported.

With ``--test1`` (Test 1's ``sherbrooke.npz``), it checks that ``m2s0e2`` with the noise
and the default settings repeats Test 1's oracle bit for bit. With ``--figures``, the
report's figure::

    uv run tools/registration/eddy_targets_expansion_figures.py --results <results> \\
        --test1 <test1>/sherbrooke.npz --out docs/_static/registration/eddy-targets --figures
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from eddy_targets_figures import INK, MUTED, RULE, SHELL_COLORS, SHELL_MARKERS, _style, _write
from experiments import paired_comparison

MARGIN_VOXELS = 0.01
"""The margin, in voxels of displacement at the brain's mean radius."""
N_RESAMPLES = 10_000
DATA = {
    "noise": (("m2s0e2", "True", "default"), ("m2s0e2", "False", "default")),
    "noise, unmoved": (("unmoved", "True", "default"), ("unmoved", "False", "default")),
    "eddy currents": (("m2s0e2", "True", "default"), ("m2s0", "True", "default")),
    "head motion": (("m2s0e2", "True", "default"), ("e2", "True", "default")),
}
"""R1's comparisons: the cell with the factor against the cell without it."""
OUTCOMES = {
    "core": "the registration expands an image onto itself: a core fix comes first",
    "setting": "a setting causes it: Test 3 changes it for the pass and reruns Test 1's "
    "pass and targets",
    "noise": "registering a noisy volume to a noise-free target expands it: Test 3 holds "
    "the eddy estimates' shared transform at the identity",
    "phantom": "the copies carry a scale their truth lacks: the phantom generator is "
    "checked before any remedy",
    "none": "no rule applies: the report describes the cells",
}


def read(path: Path) -> list[dict[str, Any]]:
    with path.open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    for row in rows:
        for key in ("index", "shell", "volume_change", "error_mean", "error_max"):
            row[key] = float(row[key])
    return rows


def _cell(rows: list[dict[str, Any]], key: tuple[str, str, str]) -> dict[int, dict]:
    return {int(r["index"]): r for r in rows if (r["copy"], r["noise"], r["variant"]) == key}


def _median(values: np.ndarray, shells: np.ndarray, seed: int = 0) -> dict[str, float]:
    """The median and its 95% interval, resampled within shells."""
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(shells == s) for s in np.unique(shells)]
    resampled = np.concatenate(
        [rng.choice(g, size=(N_RESAMPLES, len(g)), replace=True) for g in groups], axis=1
    )
    low, high = np.percentile(np.median(values[resampled], axis=1), [2.5, 97.5])
    return {"median": float(np.median(values)), "low": float(low), "high": float(high)}


def cells(rows: list[dict[str, Any]], margin: float) -> dict[str, Any]:
    """Each cell's median volume change and error, overall and by shell."""
    result = {}
    for key in dict.fromkeys((r["copy"], r["noise"], r["variant"]) for r in rows):
        cell = _cell(rows, key)
        change = np.array([v["volume_change"] for v in cell.values()])
        shells = np.array([v["shell"] for v in cell.values()])
        error = np.array([v["error_mean"] for v in cell.values()])
        result["/".join(key)] = {
            "all": _median(change, shells),
            **{
                str(int(s)): {"median": float(np.median(change[shells == s]))}
                for s in np.unique(shells)
            },
            "expanded": float(np.mean(change > margin)),
            "error_mean": float(np.median(error)),
            "failed": int(sum(v["failed"] == "True" for v in cell.values())),
        }
    return result


def compare(rows: list[dict[str, Any]], first: tuple, second: tuple, margin: float) -> dict:
    a, b = _cell(rows, first), _cell(rows, second)
    common = sorted(set(a) & set(b))
    return paired_comparison(
        np.array([a[i]["volume_change"] for i in common]),
        np.array([b[i]["volume_change"] for i in common]),
        margin,
        strata=np.array([a[i]["shell"] for i in common]),
        n_boot=N_RESAMPLES,
        seed=0,
    )


def decide(rows: list[dict[str, Any]], margin: float) -> dict[str, Any]:
    summary = cells(rows, margin)
    within = lambda c: -margin < c["low"] and c["high"] < margin  # noqa: E731
    r0 = within(summary["unmoved/False/default"]["all"])
    r1 = {name: compare(rows, *pair, margin) for name, pair in DATA.items()}
    variants = sorted({r["variant"] for r in rows} - {"default"})
    r2 = {
        copy: {
            variant: compare(rows, (copy, "True", variant), (copy, "True", "default"), margin)
            | {"within": within(summary[f"{copy}/True/{variant}"]["all"])}
            for variant in variants
        }
        for copy in ("m2s0e2", "unmoved")
    }
    causes = [v for v, c in r2["m2s0e2"].items() if c["verdict"] == "better" and c["within"]]
    # A factor drives the expansion if it is "worse" and, without it, there is none
    gone = {name: within(summary["/".join(pair[1])]["all"]) for name, pair in DATA.items()}
    drivers = [name for name, c in r1.items() if c["verdict"] == "worse" and gone[name]]
    modulators = [name for name, c in r1.items() if c["verdict"] == "worse" and not gone[name]]
    if not r0:
        outcome = "core"
    elif causes:
        outcome = "setting"
    elif {"eddy currents", "head motion"} & set(drivers):
        outcome = "phantom"
    elif "noise" in drivers:
        outcome = "noise"
    else:
        outcome = "none"
    return {
        "margin": margin,
        "r0_holds": bool(r0),
        "r1": r1,
        "drivers": drivers,
        "modulators": modulators,
        "r2": r2,
        "causes": causes,
        "outcome": outcome,
        "means": OUTCOMES[outcome],
        "cells": summary,
    }


def _pct(value: float) -> str:
    return f"{round(100 * value, 3) + 0.0:+.3f}%"


def tables(decision: dict[str, Any], out: Path) -> None:
    shells = [k for k in next(iter(decision["cells"].values())) if k.isdigit()]
    _write(
        out / "eddy-expansion-cells.csv",
        [
            "Copy",
            "Noise",
            "Settings",
            "Median volume change [95% interval]",
            *[f"b={s}" for s in shells],
            "Volumes expanded beyond the margin",
            "Median error (mm)",
        ],
        [
            [
                *key.split("/")[:1],
                "yes" if key.split("/")[1] == "True" else "no",
                key.split("/")[2],
                f"{_pct(c['all']['median'])} [{_pct(c['all']['low'])}, {_pct(c['all']['high'])}]",
                *[_pct(c[s]["median"]) for s in shells],
                f"{100 * c['expanded']:.0f}%",
                f"{c['error_mean']:.3f}",
            ]
            for key, c in decision["cells"].items()
        ],
    )

    def row(name: str, c: dict) -> list[str]:
        return [
            name,
            f"{_pct(c['median'])} [{_pct(c['median_low'])}, {_pct(c['median_high'])}]",
            c["verdict"],
        ]

    _write(
        out / "eddy-expansion-data.csv",
        ["Factor (with minus without)", "Median difference [95% interval]", "Verdict"],
        [row(name, c) for name, c in decision["r1"].items()],
    )
    _write(
        out / "eddy-expansion-settings.csv",
        ["Copy", "Removed", "Median difference from the default [95% interval]", "Verdict"],
        [
            [copy, *row(variant, c)]
            for copy, variants in decision["r2"].items()
            for variant, c in variants.items()
        ],
    )


def figure(decision: dict[str, Any], out: Path) -> None:
    """Each cell's median volume change, with its interval, and by shell."""
    plt = _style()
    keys = list(decision["cells"])
    shells = [k for k in decision["cells"][keys[0]] if k.isdigit()]
    fig, ax = plt.subplots(figsize=(7.5, 0.32 * len(keys) + 1.2), layout="constrained")
    margin = 100 * decision["margin"]
    ax.axvline(0, color=RULE, lw=0.8, zorder=0)
    for value in (-margin, margin):
        ax.axvline(value, color=MUTED, lw=0.7, linestyle=":", zorder=0)
    ys = np.arange(len(keys))[::-1]
    for y, key in zip(ys, keys, strict=True):
        c = decision["cells"][key]
        ax.plot(
            [100 * c["all"]["low"], 100 * c["all"]["high"]], [y + 0.27, y + 0.27], color=INK, lw=2
        )
        ax.scatter([100 * c["all"]["median"]], [y + 0.27], color=INK, marker="D", s=26, zorder=3)
        for k, s in enumerate(shells):
            ax.scatter(
                [100 * c[s]["median"]],
                [y + 0.09 - 0.18 * k],
                color=SHELL_COLORS[k],
                marker=SHELL_MARKERS[k],
                s=16,
                zorder=2,
            )
    labels = [
        f"{k.split('/')[0]}, {'noisy' if k.split('/')[1] == 'True' else 'noise-free'}, "
        f"{k.split('/')[2]}"
        for k in keys
    ]
    ax.set_yticks(ys, labels)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Median volume change of the registration's residual (%)")
    handles = [
        plt.Line2D([], [], color=INK, marker="D", lw=2, label="All shells"),
        *[
            plt.Line2D(
                [], [], color=SHELL_COLORS[k], marker=SHELL_MARKERS[k], lw=0, label=f"b={s}"
            )
            for k, s in enumerate(shells)
        ],
    ]
    fig.legend(
        handles=handles, loc="outside upper center", ncols=len(handles), frameon=False, fontsize=8
    )
    fig.savefig(out / "eddy-expansion-cells.png", dpi=200, facecolor="white")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--protocol", default="sherbrooke")
    parser.add_argument("--test1", type=Path, help="Test 1's sherbrooke.npz")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--figures", action="store_true")
    args = parser.parse_args(argv)

    meta = json.loads((args.results / f"{args.protocol}.json").read_text())
    rows = read(args.results / f"{args.protocol}.tsv")
    margin = MARGIN_VOXELS * float(np.mean(meta["VoxelSize"])) / meta["MeanRadiusMm"]
    decision = decide(rows, margin)
    if args.test1 is not None:
        here, there = np.load(args.results / f"{args.protocol}.npz"), np.load(args.test1)
        volumes = here["volumes"].tolist()
        indices = there["oracle/none/indices"].tolist()
        oracle = np.stack([there["oracle/none"][indices.index(i)] for i in volumes])
        decision["repeats_test1"] = bool(np.array_equal(here["m2s0e2/noisy/default"], oracle))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "eddy-expansion-decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    tables(decision, args.out)
    if args.figures:
        figure(decision, args.out)
    print(
        json.dumps(
            {k: decision[k] for k in ("r0_holds", "drivers", "modulators", "causes", "outcome")},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
