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
    keeps the instant update; otherwise it holds its targets.
The confirmation, on HCP-YA (``--chosen``, H3's choice on Sherbrooke)
    The chosen pass against ``pass-e12-u``: "better" or "not worse" on the error; and
    ``oracle-i9``'s median time at most :data:`TIME_RATIO` times ``oracle-e12``'s.

Reported beside: the backend (``i12`` against ``e12``) and the model (``i9`` against
``i12``), Test 1's drift on ``e12``, and each arm's medians. Writes
``<out>/eddy-model-<protocol>-decision.json`` and its tables::

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
    h3 = {
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
    return {
        "protocol": meta["Protocol"],
        "subset": subset,
        "margin_mm": margin,
        "chosen": chosen,
        "error": error,
        "time_ratio": ratio,
        "holds": bool(_ok(error) and ratio <= TIME_RATIO),
        "summary": summary(rows, subset),
        "checks": meta["Checks"],
    }


def _write(path: Path, header: list[str], table: list[list[Any]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(table)


def tables(decision: dict[str, Any], out: Path) -> None:
    """Each arm's medians, as a table."""
    arms = decision["summary"]
    keys = list(next(iter(arms.values())))
    _write(
        out / f"eddy-model-{decision['protocol']}-arms.csv",
        ["arm", *keys],
        [[arm, *(f"{values[k]:.6g}" for k in keys)] for arm, values in arms.items()],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--protocol", default="sherbrooke", choices=sorted(DECISION_SUBSET))
    parser.add_argument("--chosen", help="H3's choice on Sherbrooke (the confirmation)")
    parser.add_argument("--out", type=Path, required=True)
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
    print(json.dumps(brief, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
