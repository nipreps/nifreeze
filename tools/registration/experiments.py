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
"""The bookkeeping shared by the registration experiments on the phantoms (#615).

Each experiment of nipreps/nifreeze#614 is a script beside this module, which it
imports (``from experiments import subset``). Experiments register through *NiFreeze*
itself (:class:`~nifreeze.analysis.registration.AntsRegistration`) and measure the
error with :func:`~nifreeze.registration.utils.displacements_within_mask`; this
module only designs and records them.

Volumes
    :func:`subset` gives the development and confirmation volumes of a protocol,
    so that settings tuned on some volumes are confirmed on others.
Comparison
    :func:`paired_comparison` compares two variants on the same volumes, and
    :func:`holm` adjusts the p values of several comparisons.
Records
    :func:`provenance` describes what ran, and :func:`write_results` writes it
    with each table; :func:`run_tasks` runs the tasks of an experiment.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
from scipy import stats

import nifreeze
from nifreeze.cli.run import ants_version

SUBSETS = {
    "sherbrooke": {"dev": (3, 0), "confirm": (3, 1)},
    "hcpya": {"dev": (9, 0), "confirm": (9, 4)},
}
"""Every how many volumes, and from which, the development and confirmation subsets."""
THREADS = (
    "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "RAYON_NUM_THREADS",
)
"""The environment variables that set the number of threads."""
VERDICTS = ("better", "worse", "not worse", "inconclusive")
"""The outcomes of :func:`paired_comparison`."""

T = TypeVar("T")
R = TypeVar("R")


def subset(protocol: str, which: str, n: int) -> list[int]:
    """
    The development (``"dev"``) or confirmation (``"confirm"``) volumes of a protocol.

    >>> subset("sherbrooke", "dev", 192)[:4], len(subset("sherbrooke", "dev", 192))
    ([0, 3, 6, 9], 64)
    >>> subset("hcpya", "confirm", 270)[:3], len(subset("hcpya", "confirm", 270))
    ([4, 13, 22], 30)

    """
    every, start = SUBSETS[protocol][which]
    return list(range(start, n, every))


def paired_comparison(
    candidate: Sequence[float] | np.ndarray,
    reference: Sequence[float] | np.ndarray,
    margin: float,
    *,
    strata: Sequence[Any] | np.ndarray | None = None,
    n_boot: int = 10_000,
    seed: int = 0,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """
    Compare two variants on the same volumes.

    The differences ``candidate - reference`` are paired by volume. Volumes are
    resampled with replacement (within each stratum, if given), and the percentile
    intervals of the median difference and of the difference of the 95th
    percentiles are reported, with the Wilcoxon signed-rank p value.

    The verdict, from the interval of the median difference:

    ``better``
        Below zero, with a median difference below ``-margin``.
    ``worse``
        Above ``margin``.
    ``not worse``
        Its upper bound below ``margin``.
    ``inconclusive``
        Otherwise.

    Returns
    -------
    :obj:`dict`
        ``n``, ``median``, ``median_low``, ``median_high``, ``p95``, ``p95_low``,
        ``p95_high``, ``wilcoxon_p`` and ``verdict``.

    """
    candidate = np.asarray(candidate, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if candidate.shape != reference.shape or candidate.ndim != 1:
        raise ValueError("the variants must be paired, one value per volume")
    differences = candidate - reference
    groups = (
        [np.arange(len(differences))]
        if strata is None
        else [np.flatnonzero(np.asarray(strata) == s) for s in np.unique(np.asarray(strata))]
    )

    rng = np.random.default_rng(seed)
    resampled = np.concatenate(
        [rng.choice(group, size=(n_boot, len(group)), replace=True) for group in groups], axis=1
    )
    medians = np.median(differences[resampled], axis=1)
    p95s = np.percentile(candidate[resampled], 95, axis=1) - np.percentile(
        reference[resampled], 95, axis=1
    )
    tails = (100 * (1 - confidence) / 2, 100 * (1 + confidence) / 2)
    median_low, median_high = np.percentile(medians, tails)
    p95_low, p95_high = np.percentile(p95s, tails)

    median = float(np.median(differences))
    if np.allclose(differences, 0):
        wilcoxon_p = 1.0
    else:
        wilcoxon_p = float(stats.wilcoxon(differences).pvalue)

    if median_high < 0 and median < -margin:
        verdict = "better"
    elif median_low > margin:
        verdict = "worse"
    elif median_high < margin:
        verdict = "not worse"
    else:
        verdict = "inconclusive"
    return {
        "n": len(differences),
        "median": median,
        "median_low": float(median_low),
        "median_high": float(median_high),
        "p95": float(np.percentile(candidate, 95) - np.percentile(reference, 95)),
        "p95_low": float(p95_low),
        "p95_high": float(p95_high),
        "wilcoxon_p": wilcoxon_p,
        "verdict": verdict,
    }


def holm(pvalues: Sequence[float] | np.ndarray) -> np.ndarray:
    """
    Holm's step-down adjustment of p values for several comparisons.

    >>> holm([0.01, 0.04, 0.03]).round(2).tolist()
    [0.03, 0.06, 0.06]

    """
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    steps = (len(pvalues) - np.arange(len(pvalues))) * pvalues[order]
    adjusted = np.empty_like(pvalues)
    adjusted[order] = np.minimum(np.maximum.accumulate(steps), 1.0)
    return adjusted


def _git(*args: str) -> str | None:
    directory = Path(nifreeze.__file__).parent
    try:
        result = subprocess.run(
            ["git", "-C", str(directory), *args], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def provenance(settings: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    What ran: the ANTs version, the *NiFreeze* commit, the libraries and the threads.

    The commit is read from the package's own checkout (in the phantom dataset, the
    ``code/nifreeze`` subdataset), and ``NiFreezeDirty`` says whether its tracked
    files had changes.
    """
    status = _git("status", "--porcelain", "--untracked-files=no")
    cpu = None
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    record: dict[str, Any] = {
        "AntsVersion": ants_version(),
        "NiFreezeCommit": _git("rev-parse", "HEAD"),
        "NiFreezeDirty": None if status is None else bool(status),
        "NiFreezeVersion": nifreeze.__version__,
        "Python": platform.python_version(),
        "Packages": {
            package: _version(package)
            for package in ("numpy", "scipy", "nibabel", "nitransforms", "nipype", "dipy")
        },
        "Threads": {name: os.environ.get(name) for name in THREADS},
        "Affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "Cpu": cpu,
    }
    if settings is not None:
        record["Settings"] = settings
    return record


def run_tasks(
    tasks: Iterable[T], function: Callable[[T], R], jobs: int = 1, every: int = 50
) -> list[R]:
    """Run ``function`` on each task, ``jobs`` at a time, in order, with progress."""
    tasks = list(tasks)
    results = []
    with ThreadPoolExecutor(jobs) as pool:
        for done, result in enumerate(pool.map(function, tasks), start=1):
            results.append(result)
            if done % every == 0 or done == len(tasks):
                print(f"{done}/{len(tasks)}", flush=True)
    return results


def write_results(
    out: Path, name: str, rows: list[dict[str, Any]], meta: dict[str, Any]
) -> tuple[Path, Path]:
    """
    Write ``<out>/<name>.tsv``, one row per registration, and ``<out>/<name>.json``.

    The JSON holds ``meta``, and the :func:`provenance` unless ``meta`` has it.
    """
    out.mkdir(parents=True, exist_ok=True)
    table = out / f"{name}.tsv"
    with table.open("w", newline="") as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
    sidecar = out / f"{name}.json"
    record = meta if "Provenance" in meta else meta | {"Provenance": provenance()}
    sidecar.write_text(json.dumps(record, indent=2) + "\n")
    return table, sidecar
