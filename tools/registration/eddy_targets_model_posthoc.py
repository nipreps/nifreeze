# /// script
# requires-python = ">=3.13"
# dependencies = ["nifreeze[elastix]"]
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
"""Post hoc, after Test 3's results on #723: the instant update against the order, by shell.

Not part of the registered rules. For each registration (elastix's 12 parameters and
the 9), each shell, and every volume of Sherbrooke (64 per shell): the change in error
with the instant update (the pass with it minus the pass with the delayed update), its
median in each third of the volumes' order of registration within the shell, and
Spearman's rho between that change and the position in the order. The delayed update's
own error against the order is the control: its targets do not depend on the order.
Writes ``<out>/eddy-model-sherbrooke-posthoc-order.csv``::

    uv run tools/registration/eddy_targets_model_posthoc.py --results <results> \\
        --out docs/_static/registration/eddy-targets
"""

import argparse
from pathlib import Path

import numpy as np
from eddy_targets_model_figures import PRIMARY, _arm, _write, read
from scipy import stats


def _p(value: float) -> str:
    return "p < 0.001" if value < 0.001 else f"p {value:.3f}"


REGISTRATIONS = {"i9": "9 parameters (ITK)", "e12": "12 parameters (elastix)"}
"""The registrations, as the table names them."""


def order_by_shell(rows: list[dict]) -> list[list[str]]:
    """Each registration's and shell's change with the instant update against the order."""
    table = []
    for registration, name in REGISTRATIONS.items():
        instant = _arm(rows, f"pass-{registration}-u", None)
        delayed = _arm(rows, f"pass-{registration}", None)
        for shell in sorted({row["shell"] for row in instant.values()}):
            members = sorted(
                (i for i in instant if instant[i]["shell"] == shell),
                key=lambda index: instant[index]["rank"],
            )
            ranks = np.array([instant[i]["rank"] for i in members])
            change = np.array([instant[i][PRIMARY] - delayed[i][PRIMARY] for i in members])
            control = np.array([delayed[i][PRIMARY] for i in members])
            thirds = [float(np.median(part)) for part in np.array_split(change, 3)]
            rho = stats.spearmanr(ranks, change)
            control_rho = stats.spearmanr(ranks, control)
            table.append(
                [
                    name,
                    f"b={int(shell)}",
                    str(len(members)),
                    *(f"{value:+.3f}" for value in thirds),
                    f"{rho.statistic:+.2f} ({_p(rho.pvalue)})",
                    f"{control_rho.statistic:+.2f} ({_p(control_rho.pvalue)})",
                ]
            )
    return table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    table = order_by_shell(read(args.results / "sherbrooke.tsv"))
    args.out.mkdir(parents=True, exist_ok=True)
    _write(
        args.out / "eddy-model-sherbrooke-posthoc-order.csv",
        [
            "Registration",
            "Shell",
            "Volumes",
            "First third (mm)",
            "Middle third (mm)",
            "Last third (mm)",
            "Spearman's rho with the order",
            "Control: the delayed update's error",
        ],
        table,
    )
    for line in table:
        print(", ".join(line))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
