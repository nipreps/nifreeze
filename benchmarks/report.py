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
"""Report ASV results for one commit, or compare two and gate on regressions.

Each benchmark is judged by the kind of resource it measures, which its name
gives. See ``GATES`` for the thresholds.

Usage::

    python benchmarks/report.py --target SHA [--base SHA] [--summary FILE]

With ``--base``, the exit status is 1 if a benchmark regressed past its gate or
failed on the target commit.
"""

import argparse
import itertools
import json
import math
import os
import platform
import sys
from pathlib import Path
from typing import TypeGuard

import attrs


@attrs.define(frozen=True)
class Gate:
    """What fails the job for one kind of benchmark.

    Memory values are the same on every run, so a small change is a real one:
    they fail past ``factor`` when they also change by ``min_change`` MB or
    more. Values under ``floor`` are reported as ``floor`` by the benchmarks
    themselves. Times fail past ``factor`` only if significant (one-sided
    Mann-Whitney U test, ``p < alpha``, at least ``min_samples`` per commit),
    and not when ``main`` takes under ``min_time`` seconds: an extra copy in
    such a call shows in its allocations instead. Quality values are also the
    same on every run, but they are dimensionless and can be zero or negative,
    so they have no ``factor``: they fail when they change by ``min_change`` or
    more in the worse direction, which ``higher_is_worse`` gives.
    """

    title: str
    measure: str
    factor: float | None
    min_change: float | None = None
    floor: float | None = None
    alpha: float = 0.01
    min_samples: int = 6
    min_time: float = 0.01
    higher_is_worse: bool = True

    @property
    def absolute(self) -> bool:
        """Whether values are judged by their change rather than by their ratio."""
        return self.factor is None

    def describe(self) -> str:
        if self.absolute:
            change = "rises" if self.higher_is_worse else "falls"
            return f"fails when it {change} by {self.min_change:g} or more"
        if self.min_change is not None:
            return f"fails at {self.factor}x and {self.min_change:g} MB or more"
        return (
            f"fails at {self.factor}x if significant (p < {self.alpha}) "
            f"and main takes {self.min_time * 1e3:g} ms or more"
        )


GATES = {
    "alloc": Gate("Allocations", "peak traced MB", 1.25, min_change=2.0, floor=2.0),
    "rss": Gate("Peak RSS", "MB, median of 3 processes", 1.5, min_change=16.0, floor=16.0),
    "cpu": Gate("CPU time", "seconds", 1.5),
    "wall": Gate("Wall time", "seconds", 1.5),
    "drift": Gate(
        "Prediction drift", "NRMSE against the aligned prediction", None, min_change=0.01
    ),
    "ratio": Gate(
        "Compounding",
        "combined drift over the quadrature sum of the separate drifts",
        None,
        min_change=0.05,
    ),
}
"""The gate of each kind of benchmark."""

KIND_PREFIXES = (
    ("track_quality_drift_", "drift"),
    ("track_quality_ratio_", "ratio"),
    ("track_alloc_", "alloc"),
    ("track_rss_", "rss"),
    ("time_cpu_", "cpu"),
    ("time_", "wall"),
)

WORSE, BETTER, FAILED = "worse", "better", "failed"


@attrs.define
class Row:
    """One benchmark with one combination of parameters."""

    name: str
    params: str
    kind: str
    unit: str
    target: float | None = None
    target_samples: list[float] | None = None
    base: float | None = None
    base_samples: list[float] | None = None
    verdict: str = ""
    note: str = ""
    ran: bool = True
    """Whether the target's results include this benchmark at all."""

    @property
    def label(self) -> str:
        return f"{self.name}({self.params})" if self.params else self.name

    @property
    def ratio(self) -> float | None:
        target, base = self.target, self.base
        if _is_value(target) and _is_value(base) and base:
            return target / base
        return None

    @property
    def change(self) -> float | None:
        target, base = self.target, self.base
        if _is_value(target) and _is_value(base):
            return target - base
        return None


def _is_value(value: float | None) -> TypeGuard[float]:
    """Whether ``value`` is a result: not ``None`` (failed) nor NaN (skipped)."""
    return value is not None and not math.isnan(value)


def kind_of(name: str) -> str | None:
    """Return the kind of a benchmark from its method name, or ``None``."""
    method = name.rsplit(".", 1)[-1]
    for prefix, kind in KIND_PREFIXES:
        if method.startswith(prefix):
            return kind
    return None


def find_result_file(results_dir: Path, machine: str, commit: str) -> Path:
    """Return the result file of ``commit`` on ``machine``."""
    files = sorted((results_dir / machine).glob(f"{commit[:8]}-*.json"))
    if len(files) != 1:
        raise SystemExit(
            f"Expected one result file for {commit[:8]} in {results_dir / machine}, "
            f"found {len(files)}."
        )
    return files[0]


def load_values(path: Path) -> dict[str, list[tuple[float | None, list[float] | None]]]:
    """Return ``{benchmark: [(value, samples), ...]}``, one pair per parameter combination."""
    data = json.loads(path.read_text())
    columns = data["result_columns"]
    values = {}
    for name, row in data["results"].items():
        row = dict(zip(columns, row, strict=False))
        results = row.get("result")
        if not isinstance(results, list):
            results = [results]
        samples = row.get("samples") or [None] * len(results)
        values[name] = list(zip(results, samples, strict=False))
    return values


def build_rows(benchmarks: dict, target: dict, base: dict | None) -> list[Row]:
    """Return one row per benchmark and parameter combination."""
    rows = []
    for name in sorted(benchmarks):
        info = benchmarks[name]
        kind = kind_of(name)
        if kind is None:
            continue
        combos = list(itertools.product(*(info.get("params") or [])))
        for i, combo in enumerate(combos):
            row = Row(
                name=name.split(".", 1)[-1],
                params=", ".join(str(p).strip("'") for p in combo),
                kind=kind,
                unit=info.get("unit", ""),
            )
            if name in target and i < len(target[name]):
                row.target, row.target_samples = target[name][i]
            else:
                row.ran = False
            if base is not None:
                # Missing on main reads as skipped: the benchmark is new
                row.base, row.base_samples = math.nan, None
                if name in base and i < len(base[name]):
                    row.base, row.base_samples = base[name][i]
            rows.append(row)
    return rows


def _one_sided_p(larger, smaller) -> float:
    """Return the p-value that ``larger`` samples are greater than ``smaller`` ones."""
    from scipy.stats import mannwhitneyu

    return float(mannwhitneyu(larger, smaller, alternative="greater").pvalue)


def _availability(row: Row, compare: bool) -> str | None:
    """Return why ``row`` cannot be compared, or ``None`` if it can.

    Values are ``None`` when a benchmark failed and NaN when it was skipped or
    did not exist on that commit.
    """
    if not row.ran:
        return "not run"
    if row.target is None:
        return FAILED
    if not _is_value(row.target):
        return "skipped"
    if not compare:
        return ""
    if row.base is None:
        return "fails on main"
    if not _is_value(row.base):
        return "new"
    return None


def judge(row: Row, compare: bool = True) -> None:
    """Set ``row.verdict`` and ``row.note`` by the gate of its kind."""
    if (reason := _availability(row, compare)) is not None:
        if reason == FAILED:
            row.verdict = FAILED
        else:
            row.note = reason
        return

    target, base = row.target, row.base
    assert _is_value(target) and _is_value(base)  # ensured by _availability
    gate = GATES[row.kind]
    if gate.absolute:
        _judge_change(row, gate, target - base)
    else:
        _judge_ratio(row, gate, target, base)


def _judge_ratio(row: Row, gate: Gate, target: float, base: float) -> None:
    """Set ``row.verdict`` and ``row.note`` by the ratio of a time or memory value."""
    assert gate.factor is not None
    ratio = target / base if base else math.inf
    if 1 / gate.factor < ratio < gate.factor:
        return
    direction = WORSE if ratio >= gate.factor else BETTER

    if gate.min_change is not None:
        if abs(target - base) >= gate.min_change:
            row.verdict = direction
        return

    if base < gate.min_time:
        row.note = "too short to gate"
        return
    target_samples, base_samples = row.target_samples or [], row.base_samples or []
    if min(len(target_samples), len(base_samples)) < gate.min_samples:
        row.note = "too few samples"
        return
    if direction == WORSE:
        p = _one_sided_p(target_samples, base_samples)
    else:
        p = _one_sided_p(base_samples, target_samples)
    if p < gate.alpha:
        row.verdict = direction
    else:
        row.note = f"not significant (p={p:.2g})"


def _judge_change(row: Row, gate: Gate, change: float) -> None:
    """Set ``row.verdict`` by the absolute ``change`` of a quality value."""
    assert gate.min_change is not None
    worse = change if gate.higher_is_worse else -change
    # A change of exactly ``min_change`` fails, despite its rounding in floating point
    if abs(worse) >= gate.min_change or math.isclose(abs(worse), gate.min_change):
        row.verdict = WORSE if worse > 0 else BETTER


def format_value(
    value: float | None, unit: str, floor: float | None = None, decimals: int | None = None
) -> str:
    """Format a benchmark value for a table."""
    if value is None:
        return "failed"
    if not _is_value(value):
        return "n/a"
    if decimals is not None:
        return f"{value:.{decimals}f}"
    if floor is not None and value <= floor:
        return f"<{floor:g} {unit}"
    if unit == "seconds":
        for scale, suffix in ((1, "s"), (1e-3, "ms"), (1e-6, "us")):
            if value >= scale:
                return f"{value / scale:.3g} {suffix}"
        return f"{value * 1e9:.3g} ns"
    if unit == "MB":
        return f"{value:.2f} MB"
    return f"{value:.4g}"


def _runner() -> str:
    """Describe the processor this script runs on, which ran the benchmarks in CI."""
    model = platform.processor() or platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return f"{model}, {os.cpu_count()} CPUs"


def _outcome(rows: list[Row], compare: bool) -> tuple[str, list[Row]]:
    """Return the headline and the rows that fail the job."""
    failing = [r for r in rows if r.verdict in (WORSE, FAILED)]
    if not compare:
        return ("FAIL" if failing else "OK", failing)
    better = sum(r.verdict == BETTER for r in rows)
    worse = sum(r.verdict == WORSE for r in rows)
    failed = sum(r.verdict == FAILED for r in rows)
    parts = [f"{worse} worse", f"{better} better"]
    if failed:
        parts.append(f"{failed} failed")
    return (f"{'FAIL' if failing else 'PASS'} ({', '.join(parts)})", failing)


def _cells(row: Row, compare: bool) -> list[str]:
    gate = GATES[row.kind]
    if gate.absolute:
        values = [format_value(v, "", decimals=3) for v in (row.base, row.target)]
        change = row.change
        delta = f"{change:+.3f}" if change is not None else ""
    else:
        values = [format_value(v, row.unit, gate.floor) for v in (row.base, row.target)]
        ratio = row.ratio
        delta = f"{ratio:.2f}x" if ratio is not None else ""
    cells = [row.label]
    if compare:
        cells += [*values, delta]
    else:
        cells.append(values[1])
    cells.append(" ".join(filter(None, (row.verdict, row.note))))
    return cells


def _header(compare: bool, change: str = "Ratio") -> list[str]:
    if compare:
        return ["Benchmark", "main", "PR", change, "Verdict"]
    return ["Benchmark", "Value", ""]


def _change_header(gates: list[Gate]) -> str:
    """Name the change column: a ratio, an absolute change (Δ), or both."""
    kinds = {gate.absolute for gate in gates}
    if kinds == {True}:
        return "Δ"
    return "Ratio" if kinds == {False} else "Change"


def render_text(rows: list[Row], target: str, base: str | None) -> str:
    """Return the report as fixed-width plain text."""
    compare = base is not None
    headline, _ = _outcome(rows, compare)
    lines = [f"Benchmarks: {headline}"]
    if base is not None:
        lines.append(f"main {base[:8]} vs pull request {target[:8]}, measured in one job")
    else:
        lines.append(f"Commit {target[:8]}")
    lines.append(f"Runner: {_runner()}")

    for kind, gate in GATES.items():
        kind_rows = [r for r in rows if r.kind == kind]
        if not kind_rows:
            continue
        table = [_header(compare, _change_header([gate]))]
        table += [_cells(r, compare) for r in kind_rows]
        widths = [max(len(row[i]) for row in table) for i in range(len(table[0]))]
        lines += ["", f"{gate.title} ({gate.measure}; {gate.describe()})"]
        for cells in table:
            text = "  ".join(
                cell.ljust(width) if i == 0 or i == len(cells) - 1 else cell.rjust(width)
                for i, (cell, width) in enumerate(zip(cells, widths, strict=True))
            )
            lines.append(f"  {text}".rstrip())
    return "\n".join(lines) + "\n"


def _md_table(header: list[str], body: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for cells in body:
        lines.append("| " + " | ".join(cell.replace("|", "\\|") for cell in cells) + " |")
    return lines


def render_markdown(rows: list[Row], target: str, base: str | None) -> str:
    """Return the report as Markdown, for the job summary."""
    compare = base is not None
    headline, failing = _outcome(rows, compare)
    lines = [f"## Benchmarks: {headline}", ""]
    if base is not None:
        lines.append(
            f"`main` ({base[:8]}) and this pull request merged into it ({target[:8]}), "
            f"measured in one job on {_runner()}."
        )
    else:
        lines.append(f"Commit {target[:8]}, measured on {_runner()}.")

    if failing:
        lines += ["", "### Fails the job", ""]
        lines += _md_table(
            ["Kind"] + _header(compare, _change_header([GATES[r.kind] for r in failing])),
            [[GATES[r.kind].title] + _cells(r, compare) for r in failing],
        )

    for kind, gate in GATES.items():
        kind_rows = [r for r in rows if r.kind == kind]
        if not kind_rows:
            continue
        counts = ", ".join(
            f"{sum(r.verdict == v for r in kind_rows)} {v}"
            for v in (WORSE, BETTER)
            if any(r.verdict == v for r in kind_rows)
        )
        summary = f"{gate.title}: {len(kind_rows)} benchmarks"
        if counts:
            summary += f", {counts}"
        lines += ["", f"<details><summary>{summary}</summary>", ""]
        lines.append(f"{gate.measure[0].upper()}{gate.measure[1:]}; {gate.describe()}.")
        lines.append("")
        lines += _md_table(
            _header(compare, _change_header([gate])), [_cells(r, compare) for r in kind_rows]
        )
        lines += ["", "</details>"]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--target", required=True, help="commit to report (the pull request)")
    parser.add_argument("--base", help="commit to compare against (main)")
    parser.add_argument("--machine", default="gha", help="ASV machine name")
    parser.add_argument(
        "--results",
        type=Path,
        default=Path(__file__).parent / "results",
        help="ASV results directory",
    )
    parser.add_argument("--summary", type=Path, help="append a Markdown report to this file")
    parser.add_argument("--output", type=Path, help="also write the plain-text report here")
    args = parser.parse_args(argv)

    benchmarks = json.loads((args.results / "benchmarks.json").read_text())
    benchmarks = {k: v for k, v in benchmarks.items() if isinstance(v, dict)}
    target = load_values(find_result_file(args.results, args.machine, args.target))
    base = None
    if args.base:
        base = load_values(find_result_file(args.results, args.machine, args.base))

    rows = build_rows(benchmarks, target, base)
    for row in rows:
        judge(row, compare=base is not None)

    text = render_text(rows, args.target, args.base)
    print(text)
    if args.output:
        args.output.write_text(text)
    if args.summary:
        with args.summary.open("a") as summary:
            summary.write(render_markdown(rows, args.target, args.base))

    _, failing = _outcome(rows, base is not None)
    return 1 if failing else 0


if __name__ == "__main__":
    sys.exit(main())
