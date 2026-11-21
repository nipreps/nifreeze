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
"""Render the results of ``test/test_e2e_motion.py`` as Markdown tables.

Usage: ``python tools/e2e_summary.py <TEST_OUTPUT_DIR>/e2e >> "$GITHUB_STEP_SUMMARY"``.
Errors are the largest displacement (mm) within the brain mask between the
estimated and the true transforms of each volume.
"""

import json
import sys
from pathlib import Path

ORDER = ("oracle", "trivial", "gp-loo", "gqi-krr")
SCORES = {
    "head_motion": "Head motion",
    "eddy_currents": "Eddy currents",
    "whole": "Whole mapping",
    "without_eddy": "Without the eddy-current pass",
}
"""The scores of a cascade, as its table names them."""


def _cell(stats: dict) -> str:
    return f"{stats['median']:.2f} / {stats['p95']:.2f}"


def main(results_dir: Path) -> str:
    results = {p.stem: json.loads(p.read_text()) for p in sorted(results_dir.glob("*.json"))}
    if not results:
        return "No end-to-end results were found.\n"

    motion = {name: r for name, r in results.items() if r.get("kind") != "cascade"}
    backends = sorted({r["registration"] for r in motion.values() if "registration" in r})
    lines = _motion(motion, backends) if motion else []
    for result in results.values():
        if result.get("kind") == "cascade":
            lines += _cascade(result)
    return "\n".join(lines) + "\n"


def _motion(results: dict, backends: list) -> list:
    """The table of the motion results."""
    names = sorted(results, key=lambda name: (ORDER.index(name) if name in ORDER else 99, name))
    shells = sorted(
        {shell for result in results.values() for shell in _errors(result)["shells"]}, key=int
    )
    lines = [
        "## End-to-end motion estimation (Sherbrooke, simulated motion"
        + "".join(f", {backend}" for backend in backends)
        + ")",
        "",
        "Median / 95th percentile of the per-volume error (mm). Model-based targets are"
        " compared after removing their common offset.",
        "",
        "| Target | Offset | All | " + " | ".join(f"b={shell}" for shell in shells) + " | Gate |",
        "|---|---|---|" + "---|" * len(shells) + "---|",
    ]
    for name in names:
        result = results[name]
        errors = _errors(result)
        gate = result.get("gate") or {}
        if "max_p95" in gate:
            shells_text = gate["shells"]
            if not isinstance(shells_text, str):
                shells_text = "b=" + ",".join(shells_text)
            gate_text = (
                f"{gate['errors']}, {shells_text}: < {gate['max_median']} / {gate['max_p95']}"
            )
        elif gate:
            gate_text = f"< {gate['max_median']} median, < {gate['max']} max"
        else:
            gate_text = ""
        offset = f"{result['offset_mm']:.2f}" if "offset_mm" in result else ""
        cells = [
            _cell(errors["shells"][shell]) if shell in errors["shells"] else "" for shell in shells
        ]
        lines.append(
            f"| {name} | {offset} | {_cell(errors['all'])} | "
            + " | ".join(cells)
            + f" | {gate_text} |"
        )

    applied = next((r["motion"] for r in results.values() if "motion" in r), None)
    if applied is not None:
        lines += ["", f"Applied motion: {_cell(applied['all'])} mm (median / 95th percentile)."]
    return lines


def _cascade(result: dict) -> list:
    """The table of a cascade of models, then an eddy-current pass."""
    shells = sorted(result["whole"]["shells"], key=int)
    offsets = result.get("offset_mm", {})
    gates = result.get("gate", {})
    lines = [
        "",
        "## Cascade, then an eddy-current pass (Sherbrooke, simulated motion and eddy"
        f" currents, {result['registration']})",
        "",
        f"Models: {', '.join(result['models'])}. Median / 95th percentile of the"
        " per-volume error (mm). Head motion and the whole mapping are compared after"
        " removing their common offset. *Without the eddy-current pass* is the cascade's"
        " head motion against the truth of the whole mapping.",
        "",
        "| Score | Offset | All | " + " | ".join(f"b={shell}" for shell in shells) + " | Gate |",
        "|---|---|---|" + "---|" * len(shells) + "---|",
    ]
    for key, label in SCORES.items():
        errors = result[key]
        gate = gates.get(key)
        if gate is not None:
            gate_text = f"< {gate['max_median']} / {gate['max_p95']}"
        elif key == "without_eddy":
            gate_text = "above the whole mapping"
        else:
            gate_text = ""
        offset = f"{offsets[key]:.2f}" if key in offsets else ""
        cells = [_cell(errors["shells"][shell]) for shell in shells]
        lines.append(
            f"| {label} | {offset} | {_cell(errors['all'])} | "
            + " | ".join(cells)
            + f" | {gate_text} |"
        )

    applied = result["applied"]
    lines += [
        "",
        f"Applied: {_cell(applied['whole']['all'])} mm in all, and"
        f" {_cell(applied['eddy_currents']['all'])} mm of eddy currents alone (median /"
        " 95th percentile).",
    ]
    return lines


def _errors(result: dict) -> dict:
    """The errors a result is gated on."""
    if "errors" in result:
        return result["errors"]
    return result[result["gate"]["errors"]]


if __name__ == "__main__":
    sys.stdout.write(main(Path(sys.argv[1])))
