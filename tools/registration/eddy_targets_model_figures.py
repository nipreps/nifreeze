# /// script
# requires-python = ">=3.13"
# dependencies = ["nifreeze[elastix]", "matplotlib"]
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
"""The decision and tables of Test 3 on #723: the eddy-current pass with 9 parameters.

Reads ``<protocol>.{tsv,json}``, which ``tools/registration/eddy_targets_model.py`` wrote.
The error is Test 1's (``whole_frame_mean``). Comparisons are paired by volume
(:func:`~experiments.paired_comparison`, candidate minus reference, stratified by shell,
10,000 resamples, seed 0) with a margin of 0.01 voxel on the error and Test 2's margin on
the volume change (the change that moves the brain's mean radius by 0.01 voxel). They use
the decision's volumes: Sherbrooke's confirmation subset (the pilot tuned the ITK stages
on development volumes), and every volume of HCP-YA. The rules:

H1, the model
    (a) Every estimate of the 9-parameter arms is head motion followed by a field along
    phase encoding: ``family`` below :data:`FAMILY_MM`. (b) On the noise-free copy,
    ``oracle-i9-clean`` is "better" or "not worse" than ``oracle-e12-clean``, and the
    medians of its head motion and field errors are no larger than ``oracle-e12-clean``'s
    median error.
H2, accuracy, expansion and time
    For each setting of the instant update, ``pass-i9`` against ``pass-e12``: "better" or
    "not worse" on the error, and "better" on the volume change, with the 95% interval of
    its median volume change above minus the margin (it does not contract instead). And
    ``oracle-i9``'s median time per registration is at most :data:`TIME_RATIO` times
    ``oracle-e12``'s (every volume).
H3, the instant update
    ``pass-i9-u`` against ``pass-i9`` is "better" or "not worse", and ``pass-i9-u`` does
    not drift. It drifts if its gain ``g = e(pass-i9-u) - e(pass-i9)`` grows with the
    position in the order of registration ``r``: Spearman's ``rho(r, g)`` above zero with a
    one-sided permutation p below 0.05, and the last third's median gain above the first
    third's by more than the margin, the bootstrap interval of the difference entirely
    above it (Test 1's statistics, in the direction of a drift). If H3 holds, the pass
    keeps the instant update; otherwise it takes the delayed update.
The confirmation, on HCP-YA (``--chosen``, H3's choice on Sherbrooke)
    The chosen pass against ``pass-e12-u``: "better" or "not worse" on the error; and
    ``oracle-i9``'s median time at most :data:`TIME_RATIO` times ``oracle-e12``'s.

Reported beside: the backend (``i12`` against ``e12``) and the model (``i9`` against
``i12``), Test 1's drift on ``e12``, and each arm's medians. Writes
``<out>/eddy-model-<protocol>-decision.json`` and its tables, and with ``--figures`` the
report's figures::

    uv run tools/registration/eddy_targets_model_figures.py --results <results> \\
        --out <out>
    uv run tools/registration/eddy_targets_model_figures.py --results <results> \\
        --protocol hcpya --chosen pass-i9-u --out <out>
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from eddy_targets_figures import _spearman_permutation
from experiments import paired_comparison

MARGIN_VOXELS = 0.01
"""The margins, in voxels: of the error, and of the brain's mean radius."""
PRIMARY = "whole_frame_mean"
"""The error: the whole mapping, in the frame of its targets (mean, mm)."""
FAMILY_MM = 1e-6
"""Largest distance (mm) of a 9-parameter estimate to its family."""
TIME_RATIO = 1.1
"""Largest ratio of the median times per registration, ITK's 9 to elastix's 12."""
N_RESAMPLES = 10_000
"""Resamples of the intervals."""
DECISION_SUBSET = {"sherbrooke": "confirm", "hcpya": None}
"""The volumes of the decision, by protocol (``None``: every volume)."""
TEXT = ("arm", "kind", "registration", "subset", "instant_update")
"""The columns that are not numbers."""


def read(path: Path) -> list[dict[str, Any]]:
    """The rows of a results table, with numbers as numbers."""
    with path.open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    for row in rows:
        for key, value in row.items():
            if key in TEXT:
                continue
            row[key] = value == "True" if value in ("True", "False") else float(value)
    return rows


def _arm(rows: list[dict[str, Any]], arm: str, subset: str | None) -> dict[int, dict]:
    """An arm's rows by volume, on the decision's volumes."""
    return {
        int(row["index"]): row
        for row in rows
        if row["arm"] == arm and (subset is None or row["subset"] == subset)
    }


def compare(
    rows: list[dict[str, Any]],
    candidate: str,
    reference: str,
    key: str,
    margin: float,
    subset: str | None,
) -> dict[str, Any]:
    """Candidate minus reference on ``key``, paired by volume, overall and by shell."""
    first, second = _arm(rows, candidate, subset), _arm(rows, reference, subset)
    common = sorted(set(first) & set(second))
    a = np.array([first[i][key] for i in common])
    b = np.array([second[i][key] for i in common])
    shells = np.array([first[i]["shell"] for i in common])
    result = paired_comparison(a, b, margin, strata=shells, n_boot=N_RESAMPLES, seed=0)
    result["shells"] = {
        str(int(s)): paired_comparison(
            a[shells == s], b[shells == s], margin, n_boot=N_RESAMPLES, seed=0
        )
        for s in np.unique(shells)
    }
    return {"candidate": candidate, "reference": reference, "key": key} | result


def _median(rows: list[dict[str, Any]], arm: str, key: str, subset: str | None) -> dict:
    """An arm's median of ``key`` and its 95% interval, resampled within shells."""
    chosen = _arm(rows, arm, subset)
    values = np.array([row[key] for row in chosen.values()])
    shells = np.array([row["shell"] for row in chosen.values()])
    rng = np.random.default_rng(0)
    groups = [np.flatnonzero(shells == s) for s in np.unique(shells)]
    resampled = np.concatenate(
        [rng.choice(g, size=(N_RESAMPLES, len(g)), replace=True) for g in groups], axis=1
    )
    low, high = np.percentile(np.median(values[resampled], axis=1), [2.5, 97.5])
    return {"median": float(np.median(values)), "low": float(low), "high": float(high)}


def drift(
    rows: list[dict[str, Any]], updated: str, held: str, margin: float, subset: str | None
) -> dict[str, Any]:
    """The instant update's gain against the order of registration."""
    first, second = _arm(rows, updated, subset), _arm(rows, held, subset)
    common = sorted(set(first) & set(second), key=lambda index: first[index]["rank"])
    gains = np.array([first[i][PRIMARY] - second[i][PRIMARY] for i in common])
    ranks = np.array([first[i]["rank"] for i in common])
    rho, p_below = _spearman_permutation(ranks, gains)
    # With the gains negated, rho changes sign and the lower tail's p is the upper's
    _, p_above = _spearman_permutation(ranks, -gains)
    early, _, late = np.array_split(gains, 3)
    rng = np.random.default_rng(0)
    differences = np.median(rng.choice(late, size=(N_RESAMPLES, len(late))), axis=1) - np.median(
        rng.choice(early, size=(N_RESAMPLES, len(early))), axis=1
    )
    low, high = np.percentile(differences, [2.5, 97.5])
    return {
        "updated": updated,
        "held": held,
        "n": len(common),
        "rho": rho,
        "p_above": p_above,
        "p_below": p_below,
        "first_third_median": float(np.median(early)),
        "last_third_median": float(np.median(late)),
        "difference": float(np.median(late) - np.median(early)),
        "difference_low": float(low),
        "difference_high": float(high),
        "drifts": bool(rho > 0 and p_above < 0.05 and low > margin),
    }


def _ok(comparison: dict[str, Any]) -> bool:
    return comparison["verdict"] in ("better", "not worse")


def _time_ratio(rows: list[dict[str, Any]]) -> float:
    """``oracle-i9``'s median time per registration over ``oracle-e12``'s (every volume)."""
    times = {
        arm: np.median([row["seconds"] for row in _arm(rows, arm, None).values()])
        for arm in ("oracle-i9", "oracle-e12")
    }
    return float(times["oracle-i9"] / times["oracle-e12"])


def summary(rows: list[dict[str, Any]], subset: str | None) -> dict[str, Any]:
    """Each arm's medians, overall and by shell."""
    keys = (
        PRIMARY,
        "whole_frame_max",
        "volume_change",
        "strain_pe",
        "strain_across",
        "motion_mean",
        "field_mean",
        "family",
        "seconds",
    )
    result = {}
    for arm in dict.fromkeys(row["arm"] for row in rows):
        chosen = list(_arm(rows, arm, subset).values())
        shells = np.array([row["shell"] for row in chosen])
        errors = np.array([row[PRIMARY] for row in chosen])
        result[arm] = {key: float(np.median([row[key] for row in chosen])) for key in keys}
        for shell in np.unique(shells):
            result[arm][f"{PRIMARY}/{int(shell)}"] = float(np.median(errors[shells == shell]))
        result[arm]["family_max"] = float(max(row["family"] for row in chosen))
        result[arm]["failed"] = int(sum(row["failed"] for row in chosen))
    return result


def decide(rows: list[dict[str, Any]], meta: dict[str, Any]) -> dict[str, Any]:
    """Sherbrooke's decision: H1, H2 and H3."""
    subset = DECISION_SUBSET[meta["Protocol"]]
    voxel = float(np.mean(meta["VoxelSize"]))
    margin = MARGIN_VOXELS * voxel
    change_margin = MARGIN_VOXELS * voxel / meta["MeanRadiusMm"]
    nine = [arm for arm in meta["Arms"] if "-i9" in arm]
    family = max(row["family"] for row in rows if row["arm"] in nine)
    clean = compare(rows, "oracle-i9-clean", "oracle-e12-clean", PRIMARY, margin, subset)
    floor = _median(rows, "oracle-e12-clean", PRIMARY, subset)["median"]
    parts = {
        key: _median(rows, "oracle-i9-clean", key, subset)["median"]
        for key in ("motion_mean", "field_mean")
    }
    h1 = {
        "family_max_mm": family,
        "in_family": family < FAMILY_MM,
        "clean": clean,
        "clean_floor_mm": floor,
        "clean_parts_mm": parts,
        "holds": bool(
            family < FAMILY_MM and _ok(clean) and all(v <= floor for v in parts.values())
        ),
    }
    pairs = {}
    for suffix in ("-u", ""):
        candidate, reference = f"pass-i9{suffix}", f"pass-e12{suffix}"
        error = compare(rows, candidate, reference, PRIMARY, margin, subset)
        change = compare(rows, candidate, reference, "volume_change", change_margin, subset)
        own = _median(rows, candidate, "volume_change", subset)
        pairs[candidate] = {
            "error": error,
            "volume_change": change,
            "own_volume_change": own,
            "holds": bool(
                _ok(error) and change["verdict"] == "better" and own["low"] > -change_margin
            ),
        }
    ratio = _time_ratio(rows)
    h2 = {
        "pairs": pairs,
        "time_ratio": ratio,
        "time_holds": ratio <= TIME_RATIO,
        "holds": bool(all(p["holds"] for p in pairs.values()) and ratio <= TIME_RATIO),
    }
    update = compare(rows, "pass-i9-u", "pass-i9", PRIMARY, margin, subset)
    moving = drift(rows, "pass-i9-u", "pass-i9", margin, subset)
    h3: dict[str, Any] = {
        "update": update,
        "drift": moving,
        "holds": bool(_ok(update) and not moving["drifts"]),
    }
    h3["chosen"] = "pass-i9-u" if h3["holds"] else "pass-i9"
    reported = {
        "backend/oracle": compare(rows, "oracle-i12", "oracle-e12", PRIMARY, margin, subset),
        "backend/pass": compare(rows, "pass-i12-u", "pass-e12-u", PRIMARY, margin, subset),
        "model/oracle": compare(rows, "oracle-i9", "oracle-i12", PRIMARY, margin, subset),
        "model/pass": compare(rows, "pass-i9-u", "pass-i12-u", PRIMARY, margin, subset),
        "chosen/today": compare(rows, h3["chosen"], "pass-e12-u", PRIMARY, margin, subset),
        "drift/e12": drift(rows, "pass-e12-u", "pass-e12", margin, subset),
    }
    return {
        "protocol": meta["Protocol"],
        "subset": subset,
        "margin_mm": margin,
        "volume_change_margin": change_margin,
        "h1": h1,
        "h2": h2,
        "h3": h3,
        "reported": reported,
        "summary": summary(rows, subset),
        "checks": meta["Checks"],
    }


def confirm(rows: list[dict[str, Any]], meta: dict[str, Any], chosen: str) -> dict[str, Any]:
    """HCP-YA's confirmation of the configuration chosen on Sherbrooke."""
    subset = DECISION_SUBSET[meta["Protocol"]]
    margin = MARGIN_VOXELS * float(np.mean(meta["VoxelSize"]))
    error = compare(rows, chosen, "pass-e12-u", PRIMARY, margin, subset)
    ratio = _time_ratio(rows)
    change_margin = margin / meta["MeanRadiusMm"]
    reported = {
        "model/oracle": compare(rows, "oracle-i9", "oracle-e12", PRIMARY, margin, subset),
        "volume_change/pass": compare(
            rows, chosen, "pass-e12-u", "volume_change", change_margin, subset
        ),
        "volume_change/oracle": compare(
            rows, "oracle-i9", "oracle-e12", "volume_change", change_margin, subset
        ),
    }
    return {
        "protocol": meta["Protocol"],
        "subset": subset,
        "margin_mm": margin,
        "volume_change_margin": change_margin,
        "chosen": chosen,
        "error": error,
        "time_ratio": ratio,
        "holds": bool(_ok(error) and ratio <= TIME_RATIO),
        "reported": reported,
        "summary": summary(rows, subset),
        "checks": meta["Checks"],
    }


def _write(path: Path, header: list[str], table: list[list[Any]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(table)


def _interval(value: float, low: float, high: float, scale: float = 1.0, unit: str = "") -> str:
    return f"{value * scale:+.3f}{unit} [{low * scale:+.3f}, {high * scale:+.3f}]"


def comparisons(decision: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """The comparisons on the error, labelled: the rules' first, then those reported."""
    if "chosen" in decision and "h1" not in decision:
        return [
            (f"Confirmation: {decision['chosen']} against pass-e12-u", decision["error"]),
            ("Oracle: i9 against e12", decision["reported"]["model/oracle"]),
        ]
    pairs = decision["h2"]["pairs"]
    reported = decision["reported"]
    return [
        ("H1: oracle-i9-clean against oracle-e12-clean", decision["h1"]["clean"]),
        ("H2: pass-i9-u against pass-e12-u", pairs["pass-i9-u"]["error"]),
        ("H2: pass-i9 against pass-e12", pairs["pass-i9"]["error"]),
        ("H3: pass-i9-u against pass-i9", decision["h3"]["update"]),
        ("Backend, oracle: i12 against e12", reported["backend/oracle"]),
        ("Backend, pass: i12-u against e12-u", reported["backend/pass"]),
        ("Model, oracle: i9 against i12", reported["model/oracle"]),
        ("Model, pass: i9-u against i12-u", reported["model/pass"]),
        (f"Chosen ({decision['h3']['chosen']}) against today's pass", reported["chosen/today"]),
    ]


def tables(decision: dict[str, Any], out: Path) -> None:
    """Each arm's medians, the comparisons, and (Sherbrooke) the volume change and drift."""
    protocol = decision["protocol"]
    arms = decision["summary"]
    shells = [key for key in next(iter(arms.values())) if key.startswith(f"{PRIMARY}/")]
    _write(
        out / f"eddy-model-{protocol}-arms.csv",
        [
            "Arm",
            "Error (mm)",
            *(f"b={key.split('/')[1]}" for key in shells),
            "Volume change",
            "Strain along PE",
            "Strain across PE",
            "Head motion (mm)",
            "Field (mm)",
            "Time (s)",
        ],
        [
            [
                arm,
                f"{values[PRIMARY]:.3f}",
                *(f"{values[key]:.3f}" for key in shells),
                *(
                    f"{100 * values[key] + 0.0:+.3f}%"
                    for key in ("volume_change", "strain_pe", "strain_across")
                ),
                f"{values['motion_mean']:.3f}",
                f"{values['field_mean']:.3f}",
                "" if np.isnan(values["seconds"]) else f"{values['seconds']:.2f}",
            ]
            for arm, values in arms.items()
        ],
    )
    first = comparisons(decision)[0][1]
    shells = list(first["shells"])
    _write(
        out / f"eddy-model-{protocol}-comparisons.csv",
        ["Comparison", "Median difference (mm) [95% interval]", "Verdict"]
        + [f"b={s}" for s in shells],
        [
            [
                label,
                _interval(c["median"], c["median_low"], c["median_high"]),
                c["verdict"],
                *(
                    f"{c['shells'][s]['median']:+.3f} ({c['shells'][s]['verdict']})"
                    for s in shells
                ),
            ]
            for label, c in comparisons(decision)
        ],
    )
    if "h2" not in decision:
        _write(
            out / f"eddy-model-{protocol}-expansion.csv",
            ["Comparison", "Median difference of the volume change [95% interval]", "Verdict"],
            [
                [
                    label,
                    _interval(c["median"], c["median_low"], c["median_high"], 100, "%"),
                    c["verdict"],
                ]
                for label, c in (
                    (
                        f"{decision['chosen']} against pass-e12-u",
                        decision["reported"]["volume_change/pass"],
                    ),
                    ("oracle-i9 against oracle-e12", decision["reported"]["volume_change/oracle"]),
                )
            ],
        )
        return
    table = []
    for arm, pair in decision["h2"]["pairs"].items():
        change, own = pair["volume_change"], pair["own_volume_change"]
        table.append(
            [
                arm,
                _interval(own["median"], own["low"], own["high"], 100, "%"),
                _interval(change["median"], change["median_low"], change["median_high"], 100, "%"),
                change["verdict"],
            ]
        )
    _write(
        out / f"eddy-model-{protocol}-expansion.csv",
        ["Pass", "Its median volume change [95% interval]", "Against elastix 12", "Verdict"],
        table,
    )
    _write(
        out / f"eddy-model-{protocol}-drift.csv",
        [
            "Instant update",
            "Spearman's rho",
            "p (rho above zero)",
            "First third (mm)",
            "Last third (mm)",
            "Last minus first [95% interval]",
            "Drifts",
        ],
        [
            [
                f"{d['updated']} against {d['held']}",
                f"{d['rho']:+.3f}",
                f"{d['p_above']:.4f}",
                f"{d['first_third_median']:+.3f}",
                f"{d['last_third_median']:+.3f}",
                _interval(d["difference"], d["difference_low"], d["difference_high"]),
                "yes" if d["drifts"] else "no",
            ]
            for d in (decision["h3"]["drift"], decision["reported"]["drift/e12"])
        ],
    )


def forest(decision: dict[str, Any], out: Path) -> None:
    """Each comparison's median difference of the error, overall and by shell."""
    from eddy_targets_figures import INK, SHELL_COLORS, SHELL_MARKERS, _margins, _style

    plt = _style()
    labelled = comparisons(decision)
    shells = list(labelled[0][1]["shells"])
    fig, ax = plt.subplots(figsize=(7.5, 0.5 * len(labelled) + 1.2), layout="constrained")
    _margins(ax, decision["margin_mm"])
    offsets = np.linspace(-0.27, 0.27, len(shells) + 1)
    ys = np.arange(len(labelled))[::-1].astype(float)
    for y, (_, c) in zip(ys, labelled, strict=True):
        ax.plot([c["median_low"], c["median_high"]], [y + offsets[0]] * 2, color=INK, lw=2)
        ax.scatter([c["median"]], [y + offsets[0]], color=INK, s=30, marker="D", zorder=3)
        for k, shell in enumerate(shells):
            value = c["shells"][shell]
            ax.plot(
                [value["median_low"], value["median_high"]],
                [y + offsets[k + 1]] * 2,
                color=SHELL_COLORS[k],
                lw=1.6,
            )
            ax.scatter(
                [value["median"]],
                [y + offsets[k + 1]],
                color=SHELL_COLORS[k],
                marker=SHELL_MARKERS[k],
                s=24,
                zorder=3,
            )
        ax.text(
            1.01,
            y,
            c["verdict"],
            transform=ax.get_yaxis_transform(),
            va="center",
            color=INK,
            fontsize=8,
        )
    ax.set_yticks(ys, [label for label, _ in labelled])
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
    ax.set_xlabel("Candidate minus reference, median difference of the error (mm)")
    fig.savefig(out / f"eddy-model-{decision['protocol']}-comparisons.png", dpi=200)
    plt.close(fig)


def expansion_figure(rows: list[dict[str, Any]], decision: dict[str, Any], out: Path) -> None:
    """Each arm's median volume change and strains, with its interval."""
    from eddy_targets_figures import INK, MUTED, SHELL_COLORS, SHELL_MARKERS, _style

    plt = _style()
    subset, margin = decision["subset"], decision["volume_change_margin"]
    arms = list(decision["summary"])
    keys = (
        ("volume_change", "Volume change"),
        ("strain_pe", "Strain along phase encoding"),
        ("strain_across", "Strain across phase encoding"),
    )
    fig, axes = plt.subplots(
        1, len(keys), figsize=(7.5, 0.32 * len(arms) + 1.4), layout="constrained", sharey=True
    )
    ys = np.arange(len(arms))[::-1].astype(float)
    for ax, (key, title) in zip(axes, keys, strict=True):
        ax.axvline(0, color=MUTED, linewidth=0.8, zorder=0)
        for value in (-margin, margin):
            ax.axvline(100 * value, color=MUTED, linewidth=0.7, linestyle=":", zorder=0)
        for y, arm in zip(ys, arms, strict=True):
            median = _median(rows, arm, key, subset)
            ax.plot([100 * median["low"], 100 * median["high"]], [y, y], color=INK, lw=2)
            ax.scatter([100 * median["median"]], [y], color=INK, marker="D", s=24, zorder=3)
            chosen = _arm(rows, arm, subset).values()
            for k, shell in enumerate(sorted({row["shell"] for row in chosen})):
                values = [row[key] for row in chosen if row["shell"] == shell]
                ax.scatter(
                    [100 * np.median(values)],
                    [y - 0.25],
                    color=SHELL_COLORS[k],
                    marker=SHELL_MARKERS[k],
                    s=14,
                    zorder=3,
                    label=f"b={int(shell)}" if (ax is axes[0] and y == ys[0]) else None,
                )
        ax.set_title(title, color=INK, fontsize=9, loc="left")
        ax.set_xlabel("Median (%)")
    axes[0].set_yticks(ys, arms)
    axes[0].tick_params(axis="y", length=0)
    fig.legend(loc="outside upper center", ncols=3, frameon=False, fontsize=8)
    fig.savefig(out / f"eddy-model-{decision['protocol']}-expansion.png", dpi=200)
    plt.close(fig)


def order_figure(rows: list[dict[str, Any]], decision: dict[str, Any], out: Path) -> None:
    """The instant update's gain against the order, for ITK's 9 and elastix's 12."""
    from eddy_targets_figures import INK, RULE, SHELL_COLORS, SHELL_MARKERS, _style

    plt = _style()
    subset = decision["subset"]
    panels = (decision["h3"]["drift"], decision["reported"]["drift/e12"])
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.4), layout="constrained", sharey=True)
    for ax, result in zip(axes, panels, strict=True):
        updated = _arm(rows, result["updated"], subset)
        held = _arm(rows, result["held"], subset)
        common = sorted(set(updated) & set(held), key=lambda index: updated[index]["rank"])
        ax.axhline(0, color=RULE, linewidth=0.8, zorder=0)
        for k, shell in enumerate(sorted({updated[i]["shell"] for i in common})):
            members = [i for i in common if updated[i]["shell"] == shell]
            ax.scatter(
                [updated[i]["rank"] for i in members],
                [updated[i][PRIMARY] - held[i][PRIMARY] for i in members],
                color=SHELL_COLORS[k],
                marker=SHELL_MARKERS[k],
                s=16,
                edgecolors="white",
                linewidths=0.4,
                label=f"b={int(shell)}",
            )
        for third in np.array_split(np.array(common), 3):
            ranks = [updated[i]["rank"] for i in third]
            gain = np.median([updated[i][PRIMARY] - held[i][PRIMARY] for i in third])
            ax.plot([min(ranks), max(ranks)], [gain, gain], color=INK, lw=2)
        ax.set_title(
            f"{result['updated']} against {result['held']}: rho {result['rho']:+.2f}",
            color=INK,
            fontsize=9,
            loc="left",
        )
        ax.set_xlabel("Position in the order of registration")
    axes[0].set_ylabel("Change in error with the instant update (mm)")
    axes[0].legend(loc="upper left", frameon=False, fontsize=8)
    fig.savefig(out / f"eddy-model-{decision['protocol']}-order.png", dpi=200)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--protocol", default="sherbrooke", choices=sorted(DECISION_SUBSET))
    parser.add_argument("--chosen", help="H3's choice on Sherbrooke (the confirmation)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--figures", action="store_true", help="and the report's figures")
    args = parser.parse_args(argv)

    meta = json.loads((args.results / f"{args.protocol}.json").read_text())
    rows = read(args.results / f"{args.protocol}.tsv")
    if args.chosen is None:
        decision = decide(rows, meta)
        brief = {key: decision[key]["holds"] for key in ("h1", "h2", "h3")}
        brief["chosen"] = decision["h3"]["chosen"]
    else:
        decision = confirm(rows, meta, args.chosen)
        brief = {"confirmed": decision["holds"]}
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"eddy-model-{args.protocol}-decision.json"
    path.write_text(json.dumps(decision, indent=2) + "\n")
    tables(decision, args.out)
    if args.figures:
        forest(decision, args.out)
        if args.chosen is None:
            expansion_figure(rows, decision, args.out)
            order_figure(rows, decision, args.out)
    print(json.dumps(brief, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
