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
"""Tests of the benchmark report and its gates, on result files written as ASV writes them."""

import importlib.util
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "report", Path(__file__).parent.parent / "report.py"
)
assert _spec is not None and _spec.loader is not None
report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(report)

BASE = "a" * 40
TARGET = "b" * 40
COLUMNS = ["result", "params", "version", "started_at", "duration"] + [
    "stats_ci_99_a",
    "stats_ci_99_b",
    "stats_q_25",
    "stats_q_75",
    "stats_number",
    "stats_repeat",
    "samples",
]


def _samples(median, n=10, spread=0.01, seed=0):
    rng = np.random.default_rng(seed)
    return (median * (1 + spread * rng.standard_normal(n))).tolist()


def _row(values, samples=None):
    """Return a result row; ``values`` and ``samples`` have one entry per parameter combination."""
    nones = [None] * len(values)
    row = [values, [], "1", 0, 1.0, nones, nones, nones, nones, nones, nones]
    if samples is not None:
        row.append(samples)
    return row


def write_results(tmp_path, benchmarks, base, target):
    """Write ``benchmarks.json`` and one result file per commit, as ASV does."""
    results = tmp_path / "results"
    (results / "gha").mkdir(parents=True)
    info: dict[str, Any] = {
        name: {"name": name, "params": params, "unit": unit, "version": "1"}
        for name, (params, unit) in benchmarks.items()
    }
    info["version"] = 2
    (results / "benchmarks.json").write_text(json.dumps(info))
    for commit, rows in ((BASE, base), (TARGET, target)):
        if rows is None:
            continue
        data = {
            "commit_hash": commit,
            "env_name": "virtualenv-py3.12",
            "result_columns": COLUMNS,
            "results": rows,
            "version": 2,
        }
        (results / "gha" / f"{commit[:8]}-virtualenv-py3.12.json").write_text(json.dumps(data))
    return results


def run(tmp_path, benchmarks, base, target, compare=True):
    results = write_results(tmp_path, benchmarks, base, target)
    args = ["--results", str(results), "--target", TARGET, "--output", str(tmp_path / "out.txt")]
    args += ["--summary", str(tmp_path / "summary.md")]
    if compare:
        args += ["--base", BASE]
    status = report.main(args)
    return status, (tmp_path / "out.txt").read_text()


def verdict_of(text, label):
    """Return the last column of the row of ``label`` in the plain-text report."""
    for line in text.splitlines():
        if line.strip().startswith(label + " ") or line.strip() == label:
            # Columns are padded with two spaces or more: label, main, PR, change, verdict
            cells = re.split(r" {2,}", line.strip())
            return cells[4] if len(cells) > 4 else ""
    raise AssertionError(f"{label} not in report:\n{text}")


ALLOC = "bench_memory.M.track_alloc_load"


@pytest.mark.parametrize(
    ("base", "target", "expected", "status"),
    [
        (10.0, 13.0, "worse", 1),  # 1.3x and +3 MB
        (10.0, 12.0, "", 0),  # 1.2x, under the factor
        (2.0, 3.0, "", 0),  # 1.5x, but only +1 MB
        (40.0, 2.0, "better", 0),
    ],
)
def test_allocations(tmp_path, base, target, expected, status):
    got_status, text = run(
        tmp_path, {ALLOC: ([], "MB")}, {ALLOC: _row([base])}, {ALLOC: _row([target])}
    )
    assert got_status == status
    assert verdict_of(text, "M.track_alloc_load") == expected


CPU = "bench_cpu.M.time_cpu_fit"
WALL = "bench_data.M.time_load"


@pytest.mark.parametrize("name", [CPU, WALL])
@pytest.mark.parametrize(
    ("base", "target", "expected", "status"),
    [
        (_samples(1.0), _samples(1.6, seed=1), "worse", 1),
        (_samples(1.0), _samples(1.3, seed=1), "", 0),
        (_samples(1.0), _samples(0.5, seed=1), "better", 0),
        # Past the factor, but too few samples to test
        (_samples(1.0, n=3), _samples(2.0, n=3, seed=1), "too few samples", 0),
        # Past the factor, but too short to gate
        (_samples(1e-3), _samples(5e-3, seed=1), "too short to gate", 0),
    ],
)
def test_times(tmp_path, name, base, target, expected, status):
    benchmarks: dict[str, tuple[list, str]] = {name: ([], "seconds")}
    got_status, text = run(
        tmp_path,
        benchmarks,
        {name: _row([float(np.median(base))], [base])},
        {name: _row([float(np.median(target))], [target])},
    )
    label = name.split(".", 1)[1]
    verdict = verdict_of(text, label)
    assert got_status == status
    if expected:
        assert verdict == expected
    else:
        assert verdict in ("", "too short to gate") or verdict.startswith("not significant")


def test_noise_with_outlier_not_significant(tmp_path):
    """A single slow sample moves the median past the factor but is not significant."""
    base = [1.0] * 6
    target = [1.0] * 5 + [3.0] + [1.6] * 1
    status, text = run(
        tmp_path,
        {CPU: ([], "seconds")},
        {CPU: _row([1.0], [base])},
        {CPU: _row([1.6], [target])},
    )
    assert status == 0
    assert verdict_of(text, "M.time_cpu_fit").startswith("not significant")


DRIFT = "bench_quality.M.track_quality_drift_gploo"
RATIO = "bench_quality.M.track_quality_ratio_gploo"


@pytest.mark.parametrize(
    ("name", "base", "target", "expected", "status"),
    [
        (DRIFT, 0.100, 0.115, "worse", 1),
        (DRIFT, 0.100, 0.110, "worse", 1),  # a change of exactly min_change fails
        (DRIFT, 0.100, 0.105, "", 0),
        (DRIFT, 0.300, 0.200, "better", 0),
        (DRIFT, 0.0, 0.020, "worse", 1),  # a zero base, where a ratio is undefined
        (DRIFT, -0.050, -0.040, "worse", 1),  # negative values: a rise of 0.01
        (RATIO, 1.120, 1.085, "", 0),
    ],
)
def test_quality_gates_on_absolute_change(tmp_path, name, base, target, expected, status):
    """
    Check the verdict on a quality benchmark, which comes from its absolute change.

    A quality gate (``factor=None``) compares the change ``PR - main`` with its
    ``min_change`` (0.01 for the drift, 0.05 for the compounding ratio) instead of
    taking a ratio, which would be undefined for a zero base and misleading for
    negative values. A rise of at least ``min_change`` is "worse" and fails the
    job, a fall of at least ``min_change`` is "better", and a smaller change gets
    no verdict. The report prints both values and the change with three decimals.
    """
    got_status, text = run(
        tmp_path, {name: ([], "unitless")}, {name: _row([base])}, {name: _row([target])}
    )
    label = name.split(".", 1)[1]
    assert got_status == status
    assert verdict_of(text, label) == expected
    assert f"{target - base:+.3f}" in text
    assert f"{base:.3f}" in text and f"{target:.3f}" in text


def test_quality_direction(tmp_path, monkeypatch):
    """
    Check that ``judge`` takes the direction of a regression from the gate.

    The quality gates in ``GATES`` all have ``higher_is_worse=True``. This test
    sets it to ``False`` on the drift gate, as for a metric where higher is better
    (such as a correlation). A drop from 0.5 to 0.4 must then be reported as worse
    and fail the job, and a rise from 0.4 to 0.5 as better.
    """
    monkeypatch.setitem(
        report.GATES, "drift", report.attrs.evolve(report.GATES["drift"], higher_is_worse=False)
    )
    benchmarks: dict[str, tuple[list, str]] = {DRIFT: ([], "unitless")}
    fall, rise = tmp_path / "fall", tmp_path / "rise"
    status, text = run(fall, benchmarks, {DRIFT: _row([0.5])}, {DRIFT: _row([0.4])})
    assert status == 1 and verdict_of(text, "M.track_quality_drift_gploo") == "worse"
    assert "fails when it falls by 0.01 or more" in text
    status, text = run(rise, benchmarks, {DRIFT: _row([0.4])}, {DRIFT: _row([0.5])})
    assert status == 0 and verdict_of(text, "M.track_quality_drift_gploo") == "better"


def test_quality_report(tmp_path):
    """
    Check how quality benchmarks appear in the plain-text and Markdown reports.

    Each quality kind gets its own table, whose title states its gate and whose
    change column (Δ) holds ``PR - main``; ratio kinds keep their "Ratio" column.
    A quality value that is NaN on the pull request is reported as skipped, and
    one missing on ``main`` as new, as for any other benchmark. In the Markdown
    summary, one table lists the failing benchmarks of every kind, so its change
    column is named "Change" and holds ratios and differences alike.
    """
    skipped = "bench_quality.M.track_quality_drift_gqi"
    benchmarks: dict[str, tuple[list, str]] = {
        DRIFT: ([], "unitless"),
        skipped: ([], "unitless"),
        RATIO: ([], "unitless"),
        ALLOC: ([], "MB"),
    }
    base = {DRIFT: _row([0.100]), skipped: _row([0.100]), ALLOC: _row([10.0])}
    target = {
        ALLOC: _row([13.0]),
        DRIFT: _row([0.200]),
        skipped: _row([float("nan")]),
        RATIO: _row([0.950]),
    }
    status, text = run(tmp_path, benchmarks, base, target)
    assert status == 1
    assert "Prediction drift (" in text and "fails when it rises by 0.01 or more" in text
    assert "skipped" in text.split("M.track_quality_drift_gqi", 1)[1].splitlines()[0]
    assert "new" in text.split("M.track_quality_ratio_gploo", 1)[1].splitlines()[0]
    assert re.search(r"main +PR +Δ +Verdict", text)
    assert re.search(r"main +PR +Ratio +Verdict", text)
    summary = (tmp_path / "summary.md").read_text()
    # One table lists the failures of every kind, ratios and changes alike
    assert "| Kind | Benchmark | main | PR | Change | Verdict |" in summary
    assert "| Allocations | M.track_alloc_load | 10.00 MB | 13.00 MB | 1.30x | worse |" in summary
    row = "| Prediction drift | M.track_quality_drift_gploo | 0.100 | 0.200 | +0.100 | worse |"
    assert row in summary


def test_new_failed_and_parameters(tmp_path):
    benchmarks: dict[str, tuple[list, str]] = {
        ALLOC: ([["'DTI'", "'DKI'"]], "MB"),
        "bench_memory.M.track_alloc_new": ([], "MB"),
        "bench_memory.M.track_rss_broken_on_main": ([], "MB"),
        "bench_memory.M.track_alloc_broken": ([], "MB"),
        "bench_memory.M.not_a_benchmark_kind": ([], "MB"),
    }
    base = {
        ALLOC: _row([10.0, 10.0]),
        "bench_memory.M.track_rss_broken_on_main": _row([None]),
        "bench_memory.M.track_alloc_broken": _row([5.0]),
    }
    target = {
        ALLOC: _row([10.0, 30.0]),
        "bench_memory.M.track_alloc_new": _row([3.0]),
        "bench_memory.M.track_rss_broken_on_main": _row([20.0]),
        "bench_memory.M.track_alloc_broken": _row([None]),
    }
    status, text = run(tmp_path, benchmarks, base, target)
    assert status == 1
    assert verdict_of(text, "M.track_alloc_load(DTI)") == ""
    assert verdict_of(text, "M.track_alloc_load(DKI)") == "worse"
    assert "M.track_alloc_new" in text and "new" in text
    assert "fails on main" in text
    assert "failed" in text.split("M.track_alloc_broken", 1)[1].splitlines()[0]
    assert "not_a_benchmark_kind" not in text
    assert text.startswith("Benchmarks: FAIL (1 worse, 0 better, 1 failed)")
    summary = (tmp_path / "summary.md").read_text()
    assert "### Fails the job" in summary
    assert "| Allocations | M.track_alloc_load(DKI) |" in summary


def test_not_run(tmp_path):
    """A benchmark missing from the target's results (e.g., ``--bench``) does not fail."""
    benchmarks: dict[str, tuple[list, str]] = {
        ALLOC: ([], "MB"),
        "bench_memory.M.track_alloc_other": ([], "MB"),
    }
    status, text = run(tmp_path, benchmarks, {ALLOC: _row([1.0])}, {ALLOC: _row([1.0])})
    assert status == 0
    assert "not run" in text.split("M.track_alloc_other", 1)[1].splitlines()[0]


def test_single_commit(tmp_path):
    status, text = run(tmp_path, {ALLOC: ([], "MB")}, None, {ALLOC: _row([40.0])}, compare=False)
    assert status == 0
    assert text.startswith("Benchmarks: OK")
    assert "40.00 MB" in text


def test_missing_results(tmp_path):
    with pytest.raises(SystemExit, match="Expected one result file"):
        run(tmp_path, {ALLOC: ([], "MB")}, None, {ALLOC: _row([1.0])})
