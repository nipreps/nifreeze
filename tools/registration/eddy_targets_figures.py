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
"""The decision, tables and figures of the test on the eddy-current pass's targets (#723).

Reads ``<protocol>.{tsv,json}`` and ``<protocol>-cascade.tsv``, which
``tools/registration/eddy_targets.py`` wrote, and applies the rules of Test 1's working
hypothesis to the deciding model (the first of the run's models), on the error per
volume of the whole mapping, in the frame of its targets (:data:`PRIMARY`):

steps
    Each step compares consecutive arms by :func:`~experiments.paired_comparison`,
    paired by volume, stratified by shell, with a margin of 0.01 voxel, overall and
    within each shell. A step makes the floor if it is "worse" overall and within every
    shell, and its median difference is at least half the total's (``pass`` against
    ``oracle``). If more than one does, the largest decides.
order
    The pass's gain on each volume, ``pass`` minus ``cascade-corrected``, against the
    volume's position in the order of registration. Hypothesis 2 holds if Spearman's
    rho is below zero with a one-sided permutation p below 0.05, and the median gain
    of the last third of the order is below the first third's, with the bootstrap
    interval of the difference entirely below minus the margin.

Beside the decision, it writes what the report gives but the rules don't use: the other
models' steps, the fraction of the field recovered by shell, the error of the pass
against the order by shell, the gain by how many volumes along the same direction were
registered before, and the cascade's head motion against the order. With ``--record``
(#721's ``cascade-eddy.json``), it compares the pass with #712's run on the end-to-end
test's scores. With ``--figures``, the report's figures::

    uv run tools/registration/eddy_targets_figures.py --results <results> \\
        --record <cascade-eddy.json> --out docs/_static/registration/eddy-targets --figures
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from experiments import holm, paired_comparison
from scipy import stats

MARGIN = 0.01
"""The margin of the comparisons, in voxels."""
PRIMARY = "whole_frame_mean"
"""The error the rules use: the whole mapping, in the frame of its targets (mean, mm)."""
ERRORS = (
    "whole_frame_mean",
    "whole_frame_max",
    "whole_offset_mean",
    "whole_offset_max",
    "eddy_mean",
    "eddy_max",
)
"""The errors whose medians the report gives."""
NONE = "none"
"""The model of the oracle's arms."""
STEPS = (
    ("S0", "the start", "oracle-cascade", "oracle"),
    ("S1", "the model", "reference", "oracle"),
    ("S2", "resampling", "corrected", "reference"),
    ("S3", "eddy currents in the training data", "motion-corrected", "corrected"),
    ("S4", "the cascade's motion", "cascade-corrected", "motion-corrected"),
    ("S5", "the instant update", "pass", "cascade-corrected"),
    ("total", "the whole floor", "pass", "oracle"),
)
"""Each step: its key, what it measures, the candidate arm and the reference arm."""
OUTCOMES = {
    "S0": "the start: the report explains it before anything else",
    "S1": "the model's prediction: Test 2 compares models for the pass",
    "S2": "resampling: the report explains it before anything else",
    "S3": "the targets carry the field: Test 2 tries targets free of it, "
    "starting with the pass repeated; #591 waits for Test 2",
    "S4": "the cascade's motion: the coupling of the passes (E3b) comes first",
    "S5": "the instant update: the report explains it before anything else",
}
"""What each step means when it makes the floor."""
ARMS = (
    "oracle",
    "oracle-cascade",
    "reference",
    "corrected",
    "motion-corrected",
    "cascade-corrected",
    "pass",
)
N_RESAMPLES = 10_000
"""Resamples and permutations of every interval and test."""
TOLERANCE = 0.05
"""Largest difference (mm) of a median from #712's run."""
RECORDED = {
    "eddy_currents": ("pass", "eddy_max"),
    "whole": ("pass", "whole_offset_max"),
}
"""#712's scores, by the arm and column that repeat them."""


def read(path: Path) -> list[dict[str, Any]]:
    """The rows of a results table, with numbers as numbers."""
    with path.open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    for row in rows:
        for key, value in row.items():
            if key in ("arm", "model", "start"):
                continue
            row[key] = value == "True" if value in ("True", "False") else float(value)
    return rows


def _column(rows: list[dict[str, Any]], arm: str, model: str, key: str) -> dict[int, float]:
    """One arm's value of ``key`` by volume."""
    model = NONE if arm.startswith("oracle") else model
    return {
        int(row["index"]): row[key] for row in rows if row["arm"] == arm and row["model"] == model
    }


def _paired(
    rows: list[dict[str, Any]], model: str, candidate: str, reference: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The two arms' errors on the volumes both registered, and the volumes' shells."""
    first = _column(rows, candidate, model, PRIMARY)
    second = _column(rows, reference, model, PRIMARY)
    shell = _column(rows, candidate, model, "shell")
    common = sorted(set(first) & set(second))
    return (
        np.array([first[i] for i in common]),
        np.array([second[i] for i in common]),
        np.array([shell[i] for i in common]),
    )


def steps(rows: list[dict[str, Any]], model: str, margin: float) -> list[dict[str, Any]]:
    """Each step's comparison, overall and within each shell, and its share of the total."""
    result = []
    for key, name, candidate, reference in STEPS:
        first, second, shells = _paired(rows, model, candidate, reference)
        overall = paired_comparison(
            first, second, margin, strata=shells, n_boot=N_RESAMPLES, seed=0
        )
        by_shell = {
            str(int(shell)): paired_comparison(
                first[shells == shell],
                second[shells == shell],
                margin,
                n_boot=N_RESAMPLES,
                seed=0,
            )
            for shell in np.unique(shells)
        }
        result.append(
            {
                "step": key,
                "name": name,
                "candidate": candidate,
                "reference": reference,
                **overall,
                "shells": by_shell,
            }
        )
    total = result[-1]["median"]
    adjusted = holm([step["wilcoxon_p"] for step in result])
    for step, p in zip(result, adjusted, strict=True):
        step["share"] = step["median"] / total if total else float("nan")
        step["holm_p"] = float(p)
        step["makes_floor"] = bool(
            step["step"] != "total"
            and step["verdict"] == "worse"
            and all(s["verdict"] == "worse" for s in step["shells"].values())
            and step["share"] >= 0.5
        )
    return result


def outcome(result: list[dict[str, Any]]) -> dict[str, Any]:
    """The step that makes the floor, or the largest step that is worse."""
    qualified = [step for step in result if step["makes_floor"]]
    if qualified:
        chosen = max(qualified, key=lambda step: step["median"])
        return {"step": chosen["step"], "means": OUTCOMES[chosen["step"]], "reaches_half": True}
    worse = [s for s in result if s["step"] != "total" and s["verdict"] == "worse"]
    largest = max(worse, key=lambda step: step["median"]) if worse else None
    return {
        "step": None if largest is None else largest["step"],
        "means": "no step reaches half: Test 2 takes the largest step that is worse",
        "reaches_half": False,
    }


def _spearman_permutation(x: np.ndarray, y: np.ndarray, seed: int = 0) -> tuple[float, float]:
    """Spearman's rho, and its one-sided permutation p (rho below zero)."""
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    rho = float(rx @ ry / np.sqrt((rx @ rx) * (ry @ ry)))
    rng = np.random.default_rng(seed)
    shuffled = np.array([rng.permutation(ry) for _ in range(N_RESAMPLES)])
    null = shuffled @ rx / np.sqrt((rx @ rx) * (ry @ ry))
    return rho, float((np.sum(null <= rho) + 1) / (N_RESAMPLES + 1))


def order(rows: list[dict[str, Any]], model: str, margin: float) -> dict[str, Any]:
    """Hypothesis 2: the pass's gain against the position in the order of registration."""
    passed = _column(rows, "pass", model, PRIMARY)
    start = _column(rows, "cascade-corrected", model, PRIMARY)
    rank = _column(rows, "pass", model, "rank")
    common = sorted(set(passed) & set(start), key=lambda index: rank[index])
    gains = np.array([passed[i] - start[i] for i in common])
    ranks = np.array([rank[i] for i in common])
    rho, p = _spearman_permutation(ranks, gains)
    first, _, last = np.array_split(gains, 3)
    rng = np.random.default_rng(0)
    differences = np.median(rng.choice(last, size=(N_RESAMPLES, len(last))), axis=1) - np.median(
        rng.choice(first, size=(N_RESAMPLES, len(first))), axis=1
    )
    low, high = np.percentile(differences, [2.5, 97.5])
    return {
        "n": len(common),
        "rho": rho,
        "p_two_sided": float(stats.spearmanr(ranks, gains).pvalue),
        "p_permutation": p,
        "first_third_median": float(np.median(first)),
        "last_third_median": float(np.median(last)),
        "difference": float(np.median(last) - np.median(first)),
        "difference_low": float(low),
        "difference_high": float(high),
        "holds": bool(rho < 0 and p < 0.05 and high < -margin),
    }


def _siblings_before(rows: list[dict[str, Any]], model: str) -> dict[int, int]:
    """How many volumes along the same direction were registered before each volume."""
    rank = _column(rows, "pass", model, "rank")
    direction = _column(rows, "pass", model, "direction")
    return {
        index: sum(
            1
            for other in rank
            if other != index
            and direction[other] == direction[index]
            and rank[other] < rank[index]
        )
        for index in rank
    }


def reported(
    rows: list[dict[str, Any]], cascade: list[dict[str, Any]], models: list[str], margin: float
) -> dict[str, Any]:
    """What the report gives beside the decision."""
    result: dict[str, Any] = {"medians": {}, "kappa": {}, "pass_order": {}, "siblings": {}}
    for model in models:
        for arm in ARMS:
            key = f"{arm}/{NONE if arm.startswith('oracle') else model}"
            if key in result["medians"]:
                continue
            values = {
                column: np.array(list(_column(rows, arm, model, column).values()))
                for column in (*ERRORS, "kappa", "shell")
            }
            shells = values["shell"]
            medians, kappas = {}, {}
            for shell in ["all", *map(str, map(int, np.unique(shells)))]:
                keep = np.ones(len(shells), bool) if shell == "all" else shells == float(shell)
                medians[shell] = {c: float(np.median(values[c][keep])) for c in ERRORS}
                kappas[shell] = float(np.median(values["kappa"][keep]))
            result["medians"][key] = medians
            result["kappa"][key] = kappas
        passed = _column(rows, "pass", model, PRIMARY)
        rank = _column(rows, "pass", model, "rank")
        shell_of = _column(rows, "pass", model, "shell")
        result["pass_order"][model] = {}
        for value in sorted(set(shell_of.values())):
            members = [i for i in passed if shell_of[i] == value]
            rho, p = stats.spearmanr([rank[i] for i in members], [passed[i] for i in members])
            result["pass_order"][model][str(int(value))] = {"rho": float(rho), "p": float(p)}
        start = _column(rows, "cascade-corrected", model, PRIMARY)
        before = _siblings_before(rows, model)
        result["siblings"][model] = {
            str(count): {
                "n": sum(1 for i in start if before[i] == count),
                "median_gain": float(
                    np.median([passed[i] - start[i] for i in start if before[i] == count])
                ),
            }
            for count in sorted({before[i] for i in start})
        }
        result.setdefault("steps", {})[model] = steps(rows, model, margin)
        result.setdefault("order", {})[model] = order(rows, model, margin)
    result["cascade_order"] = {}
    for model in dict.fromkeys(row["model"] for row in cascade):
        level = [row for row in cascade if row["model"] == model]
        rho, p = stats.spearmanr(
            [row["rank"] for row in level], [row["motion_offset_mean"] for row in level]
        )
        result["cascade_order"][model] = {"rho": float(rho), "p": float(p)}
    return result


def reproduction(
    rows: list[dict[str, Any]], cascade: list[dict[str, Any]], model: str, record: dict
) -> dict[str, Any]:
    """The pass and the cascade against #712's run, on the end-to-end test's scores."""
    shell = _column(rows, "pass", model, "shell")
    columns = {name: _column(rows, arm, model, key) for name, (arm, key) in RECORDED.items()}
    last = [row for row in cascade if row["level"] == max(r["level"] for r in cascade)]
    columns["head_motion"] = {int(row["index"]): row["motion_offset_max"] for row in last}
    result: dict[str, Any] = {"within_tolerance": True}
    for name, values in columns.items():
        result[name] = {}
        for which, summary in [("all", record[name]["all"]), *record[name]["shells"].items()]:
            members = [v for i, v in values.items() if which == "all" or shell[i] == float(which)]
            here, there = float(np.median(members)), summary["median"]
            result[name][which] = {"here": here, "record": there, "difference": here - there}
            result["within_tolerance"] &= abs(here - there) <= TOLERANCE
    return result


def _write(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def _interval(value: float, low: float, high: float) -> str:
    def signed(number: float) -> str:
        # Adding zero turns a rounded -0.0 into 0.0
        return f"{round(number, 3) + 0.0:+.3f}"

    return f"{signed(value)} [{signed(low)}, {signed(high)}]"


def tables(decision: dict[str, Any], extra: dict[str, Any], out: Path) -> None:
    """The report's tables."""
    shells = list(decision["steps"][0]["shells"])
    for model, steps_ in extra["steps"].items():
        # The deciding model's table, and one per other model
        suffix = "" if model == decision["model"] else f"-{model}"
        _write(
            out / f"eddy-targets-steps{suffix}.csv",
            [
                "Step",
                "Measures",
                "All volumes (mm)",
                *[f"b={s} (mm)" for s in shells],
                "Share",
                "Holm p",
            ],
            [
                [
                    step["step"],
                    step["name"],
                    *[
                        _interval(v["median"], v["median_low"], v["median_high"])
                        + f" {v['verdict']}"
                        for v in (step, *step["shells"].values())
                    ],
                    f"{round(step['share'], 2) + 0.0:.2f}",
                    f"{step['holm_p']:.1e}",
                ]
                for step in steps_
            ],
        )
    medians = extra["medians"]
    for name, mean, largest in (
        ("whole", "whole_frame_mean", "whole_frame_max"),
        ("offset", "whole_offset_mean", "whole_offset_max"),
        ("eddy", "eddy_mean", "eddy_max"),
    ):
        _write(
            out / f"eddy-targets-errors-{name}.csv",
            ["Arm", "Model", "All", *[f"b={s}" for s in shells]],
            [
                [
                    key.split("/")[0],
                    key.split("/")[1],
                    *[f"{m[mean]:.2f} / {m[largest]:.2f}" for m in values.values()],
                ]
                for key, values in medians.items()
            ],
        )
    _write(
        out / "eddy-targets-kappa.csv",
        ["Arm", "Model", "All", *[f"b={s}" for s in shells]],
        [
            [key.split("/")[0], key.split("/")[1], *[f"{k:.2f}" for k in values.values()]]
            for key, values in extra["kappa"].items()
        ],
    )
    _write(
        out / "eddy-targets-order.csv",
        [
            "Model",
            "Spearman's rho",
            "One-sided p (rho < 0)",
            "Two-sided p",
            "First third (mm)",
            "Last third (mm)",
            "Difference (mm)",
            "Holds",
        ],
        [
            [
                model,
                f"{o['rho']:+.2f}",
                f"{o['p_permutation']:.2f}",
                f"{o['p_two_sided']:.1e}",
                f"{o['first_third_median']:+.3f}",
                f"{o['last_third_median']:+.3f}",
                _interval(o["difference"], o["difference_low"], o["difference_high"]),
                "yes" if o["holds"] else "no",
            ]
            for model, o in extra["order"].items()
        ],
    )
    _write(
        out / "eddy-targets-order-shells.csv",
        ["Model", *[f"b={s}" for s in shells]],
        [
            [model, *[f"{v['rho']:+.2f} (p {v['p']:.1e})" for v in values.values()]]
            for model, values in extra["pass_order"].items()
        ],
    )
    _write(
        out / "eddy-targets-order-cascade.csv",
        ["Cascade level", "Spearman's rho", "p"],
        [
            [f"{level + 1}: {model}", f"{v['rho']:+.2f}", f"{v['p']:.1e}"]
            for level, (model, v) in enumerate(extra["cascade_order"].items())
        ],
    )
    _write(
        out / "eddy-targets-siblings.csv",
        ["Model", "Same-direction volumes registered before", "Volumes", "Median change (mm)"],
        [
            [model, count, v["n"], f"{v['median_gain']:+.3f}"]
            for model, counts in extra["siblings"].items()
            for count, v in counts.items()
        ],
    )
    if "reproduction" in decision:
        _write(
            out / "eddy-targets-reproduction.csv",
            ["Score", "Shells", "This run (mm)", "#721 (mm)", "Difference (mm)"],
            [
                [
                    name,
                    which if which == "all" else f"b={which}",
                    f"{v['here']:.3f}",
                    f"{v['record']:.3f}",
                    f"{v['difference']:+.3f}",
                ]
                for name, shells_ in decision["reproduction"].items()
                if name != "within_tolerance"
                for which, v in shells_.items()
            ],
        )


INK, MUTED, RULE = "#0b0b0b", "#52514e", "#d9d8d4"
SHELL_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")
"""The first three categorical slots, in fixed order by b-value (validated all-pairs)."""
SHELL_MARKERS = ("o", "s", "^")


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


def _margins(ax: Any, margin: float) -> None:
    ax.axvline(0, color=RULE, linewidth=0.8, zorder=0)
    for value in (-margin, margin):
        ax.axvline(value, color=MUTED, linewidth=0.7, linestyle=":", zorder=0)


def forest(decision: dict[str, Any], out: Path) -> None:
    """Each step's median difference with its interval, overall and by shell."""
    plt = _style()
    result = decision["steps"]
    shells = list(result[0]["shells"])
    fig, ax = plt.subplots(figsize=(7.5, 4.2), layout="constrained")
    _margins(ax, decision["margin_mm"])
    offsets = np.linspace(-0.27, 0.27, len(shells) + 1)
    ys = np.arange(len(result))[::-1].astype(float)
    for y, step in zip(ys, result, strict=True):
        ax.plot([step["median_low"], step["median_high"]], [y + offsets[0]] * 2, color=INK, lw=2)
        ax.scatter([step["median"]], [y + offsets[0]], color=INK, s=30, marker="D", zorder=3)
        for k, shell in enumerate(shells):
            value = step["shells"][shell]
            color, marker = SHELL_COLORS[k], SHELL_MARKERS[k]
            ax.plot(
                [value["median_low"], value["median_high"]],
                [y + offsets[k + 1]] * 2,
                color=color,
                lw=1.6,
            )
            ax.scatter(
                [value["median"]], [y + offsets[k + 1]], color=color, marker=marker, s=24, zorder=3
            )
        ax.text(
            1.01,
            y,
            f"{step['verdict']}, share {round(step['share'], 2) + 0.0:.2f}",
            transform=ax.get_yaxis_transform(),
            va="center",
            color=INK,
            fontsize=8,
        )
    ax.set_yticks(ys, [f"{s['step']}: {s['name']}" for s in result])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    handles = [
        plt.Line2D([], [], color=INK, marker="D", lw=2, label="All shells"),
        *[
            plt.Line2D(
                [], [], color=SHELL_COLORS[k], marker=SHELL_MARKERS[k], lw=1.6, label=f"b={s}"
            )
            for k, s in enumerate(shells)
        ],
    ]
    fig.legend(
        handles=handles, loc="outside upper center", ncols=len(handles), frameon=False, fontsize=8
    )
    ax.set_xlabel("Candidate minus reference arm, median difference of the error (mm)")
    fig.savefig(out / "eddy-targets-steps.png", dpi=200, facecolor="white")
    plt.close(fig)


def order_figure(
    rows: list[dict[str, Any]], model: str, decision: dict[str, Any], out: Path
) -> None:
    """The pass's gain against the order of registration, by shell, with the thirds."""
    plt = _style()
    passed = _column(rows, "pass", model, PRIMARY)
    start = _column(rows, "cascade-corrected", model, PRIMARY)
    rank = _column(rows, "pass", model, "rank")
    shell = _column(rows, "pass", model, "shell")
    common = sorted(set(passed) & set(start), key=lambda index: rank[index])
    fig, ax = plt.subplots(figsize=(7.5, 3.6), layout="constrained")
    ax.axhline(0, color=RULE, linewidth=0.8, zorder=0)
    for k, value in enumerate(sorted(set(shell.values()))):
        members = [i for i in common if shell[i] == value]
        ax.scatter(
            [rank[i] for i in members],
            [passed[i] - start[i] for i in members],
            color=SHELL_COLORS[k],
            marker=SHELL_MARKERS[k],
            s=16,
            label=f"b={int(value)}",
            edgecolors="white",
            linewidths=0.4,
        )
    thirds = np.array_split(np.array(common), 3)
    for third in thirds:
        ranks = [rank[i] for i in third]
        median = np.median([passed[i] - start[i] for i in third])
        ax.plot([min(ranks), max(ranks)], [median, median], color=INK, lw=2)
    result = decision["order"]
    ax.set_title(
        f"Spearman's rho {result['rho']:+.2f}"
        f" (one-sided permutation p {result['p_permutation']:.3f})",
        color=INK,
        fontsize=9,
        loc="left",
    )
    ax.set_xlabel("Position in the order of registration")
    ax.set_ylabel("Change in error with the instant update (mm)")
    ax.legend(loc="lower left", frameon=False, fontsize=8)
    fig.savefig(out / "eddy-targets-order.png", dpi=200, facecolor="white")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", type=Path, required=True, help="the results folder")
    parser.add_argument("--protocol", default="sherbrooke")
    parser.add_argument("--record", type=Path, help="#721's cascade-eddy.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--figures", action="store_true", help="also the report's figures")
    args = parser.parse_args(argv)

    meta = json.loads((args.results / f"{args.protocol}.json").read_text())
    rows = read(args.results / f"{args.protocol}.tsv")
    cascade = read(args.results / f"{args.protocol}-cascade.tsv")
    models = meta["Models"]
    margin = MARGIN * float(np.mean(meta["VoxelSize"]))
    extra = reported(rows, cascade, models, margin)
    decision = {
        "model": models[0],
        "margin_mm": margin,
        "steps": extra["steps"][models[0]],
        "outcome": outcome(extra["steps"][models[0]]),
        "order": order(rows, models[0], margin),
        "checks": meta["Checks"],
    }
    if args.record is not None:
        record = json.loads(args.record.read_text())
        decision["reproduction"] = reproduction(rows, cascade, models[0], record)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "eddy-targets-decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    (args.out / "eddy-targets-reported.json").write_text(json.dumps(extra, indent=2) + "\n")
    tables(decision, extra, args.out)
    if args.figures:
        forest(decision, args.out)
        order_figure(rows, models[0], decision, args.out)
    print(json.dumps({k: decision[k] for k in ("outcome", "order")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
