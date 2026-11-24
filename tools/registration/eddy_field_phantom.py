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
"""
Fit the global eddy-current model to the eddy-only phantom, and time it (#745).

The phantom's eddy-only copy (no head motion, a largest displacement of 4 mm, with
noise) is fitted by :class:`~nifreeze.registration.eddy_field.EddyFieldFit`, each volume
compared with its noise-free original and its head motion held at the identity:

``all-q``
    One matrix :math:`C` and the exponent :math:`p`, fitted.
``q-wise``
    One :math:`C` for each shell (the edges :data:`EDGES`), the exponent held.

The fields' zero point is the phantom's, half a voxel off the centre of the field of view
(#744). Each fit runs in its own process, once for each number of ITK threads, so that
its time and peak memory are its own; the error is that of the run with one thread.

Outputs, in ``--out``: ``<protocol>.tsv`` (one row per variant and volume),
``<protocol>.json`` (the fits, the truth, the runs and the provenance), ``runs/`` (each
run's record) and four figures (``slopes``, ``errors``, ``parameters``, ``runtime``).
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import nibabel as nb
import nitransforms as nt
import numpy as np
from eddy_targets import _dataset_commit
from eddy_targets_expansion import COPIES, NOISE
from experiments import provenance, write_results

from nifreeze.analysis.phantoms import load_phantom
from nifreeze.registration.eddy_field import (
    EddyField,
    EddyFieldFit,
    SegmentedEddyField,
    field_of_view_centre,
    field_slopes,
)
from nifreeze.registration.utils import displacements_within_mask, phase_encoding_vector

VARIANTS = ("all-q", "q-wise")
"""The two forms of the model."""
EDGES = (1500.0, 2750.0)
"""The b-values that put each of Sherbrooke's shells (1000, 2000, 3500) in its own segment."""
COPY = "e2"
"""The eddy-only copy (no head motion, a largest displacement of 4 mm)."""
PHASE_ENCODING = "j-"
"""The phantom's phase-encoding direction (its sidecar's ``PhaseEncodingDirection``)."""
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
"""The libraries' thread counts, set to each run's."""


def _rss_mb() -> float:
    """The process's resident memory now (MB)."""
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return float(line.split()[1]) / 1024
    raise RuntimeError("VmRSS is not in /proc/self/status")


def _eddy_only(cells: list, n_volumes: int) -> tuple[Any, Any]:
    """The eddy-only copy's perturbation and loader."""
    motion, eddy = COPIES[COPY]
    chosen = [
        (perturbation, load)
        for perturbation, load in cells
        if perturbation.magnitude == motion
        and perturbation.count == 0
        and perturbation.distortion is not None
        and perturbation.distortion.magnitude == eddy
    ]
    if len(chosen) != 1:
        raise RuntimeError(f"{len(chosen)} copies match {COPY}, not 1")
    return chosen[0]


def _segments(bvals: np.ndarray) -> np.ndarray:
    return np.searchsorted(np.asarray(EDGES), bvals, "right")


def _truth_fields(field: EddyField | SegmentedEddyField, slopes: np.ndarray, table: np.ndarray):
    """The phantom's fields as the fitted model writes them, at the same reference b-values."""
    if isinstance(field, EddyField):
        truth = EddyField.from_slopes(slopes, table, b_ref=field.b_ref)
        residual = np.abs(truth.slopes(table) - slopes).max()
        return {
            "Matrix": truth.matrix.tolist(),
            "Exponent": truth.exponent,
            "BRef": truth.b_ref,
            "ResidualSlope": float(residual),
        }
    parts = []
    for k, part in enumerate(field.parts):
        chosen = _segments(table[:, 3]) == k
        truth = EddyField.from_slopes(slopes[chosen], table[chosen], exponent=part.exponent)
        residual = np.abs(truth.slopes(table[chosen]) - slopes[chosen]).max()
        parts.append(
            {
                "Matrix": truth.matrix.tolist(),
                "Exponent": truth.exponent,
                "BRef": truth.b_ref,
                "ResidualSlope": float(residual),
            }
        )
    return {"Edges": list(field.edges), "Parts": parts}


def _fitted(field: EddyField | SegmentedEddyField) -> dict[str, Any]:
    if isinstance(field, EddyField):
        return {"Matrix": field.matrix.tolist(), "Exponent": field.exponent, "BRef": field.b_ref}
    return {
        "Edges": list(field.edges),
        "Parts": [
            {"Matrix": part.matrix.tolist(), "Exponent": part.exponent, "BRef": part.b_ref}
            for part in field.parts
        ],
    }


def run(args: argparse.Namespace, variant: str, threads: int) -> None:
    """One fit, in this process: its record goes to ``<out>/runs``."""
    began = perf_counter()
    clean, _ = load_phantom(args.dataset, args.protocol)
    _, noisy_cells = load_phantom(args.dataset, args.protocol, noise=NOISE)
    n_volumes = len(clean)
    perturbation, load = _eddy_only(noisy_cells, n_volumes)
    moving = load()
    distortion = perturbation.distortion
    mask = np.asanyarray(clean.brainmask, dtype=bool)
    table = np.asarray(clean.gradients, dtype=float)
    affine = np.asarray(clean.affine, dtype=float)
    loaded, rss_loaded = perf_counter() - began, _rss_mb()

    fit = EddyFieldFit(
        phase_encoding=PHASE_ENCODING,
        num_threads=threads,
        edges=EDGES if variant == "q-wise" else None,
        zero_point=distortion.center,
    )
    began = perf_counter()
    for index in range(n_volumes):
        fit.add(
            index,
            np.asarray(clean.dataobj[index], dtype="float32"),
            np.asarray(moving.dataobj[index], dtype="float32"),
            affine,
            mask,
        )
    added = perf_counter() - began
    del clean, moving
    began = perf_counter()
    field = fit.fit(table)
    fitted = perf_counter() - began
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    truth = distortion.matrices()
    u = phase_encoding_vector(PHASE_ENCODING, affine)
    mask_img = nb.Nifti1Image(mask.astype("uint8"), affine)
    estimated = np.stack([eddy @ motion for _, (motion, eddy) in sorted(fit.mappings().items())])
    errors = [
        displacements_within_mask(
            mask_img, nt.linear.Affine(estimated[index]), nt.linear.Affine(truth[index])
        )
        for index in range(n_volumes)
    ]
    true_slopes = field_slopes(truth, u)
    record = {
        "Variant": variant,
        "Threads": threads,
        "LoadSeconds": loaded,
        "AddSeconds": added,
        "FitSeconds": fitted,
        "Evaluations": fit.evaluations,
        "Messages": fit.messages,
        "RssLoadedMb": rss_loaded,
        "PeakRssMb": peak,
        "ZeroPoint": np.asarray(fit.zero_point).tolist(),
        "Fitted": _fitted(field),
        "Truth": _truth_fields(field, true_slopes, table),
        "Provenance": provenance(),
    }
    runs = args.out / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    stem = runs / f"{variant}-t{threads}"
    stem.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    np.savez_compressed(
        stem.with_suffix(".npz"),
        estimated=estimated,
        truth=truth,
        slopes=field.slopes(table),
        true_slopes=true_slopes,
        error_median=np.array([np.median(e) for e in errors]),
        error_max=np.array([e.max() for e in errors]),
    )
    print(f"{variant}, {threads} threads: fit in {fitted:.0f} s, {fit.evaluations}", flush=True)


def _launch(args: argparse.Namespace, variant: str, threads: int) -> None:
    """Run one fit in a process of its own, with the libraries' threads set to its own."""
    env = os.environ | {name: str(threads) for name in THREAD_VARIABLES}
    command = [sys.executable, str(Path(__file__).resolve()), "--dataset", str(args.dataset)]
    command += ["--protocol", args.protocol, "--out", str(args.out)]
    command += ["--run", variant, str(threads)]
    subprocess.run(command, check=True, env=env, cwd=Path(__file__).parent)


def gather(args: argparse.Namespace) -> None:
    """The table and the record of all runs; the error is the one-thread run's."""
    runs = {
        (variant, threads): json.loads(
            (args.out / "runs" / f"{variant}-t{threads}.json").read_text()
        )
        for variant in VARIANTS
        for threads in args.threads
    }
    arrays = {key: np.load(args.out / "runs" / f"{key[0]}-t{key[1]}.npz") for key in runs}
    clean, _ = load_phantom(args.dataset, args.protocol)
    table = np.asarray(clean.gradients, dtype=float)
    shells = np.round(table[:, 3], -2)
    first = min(args.threads)
    rows = []
    for variant in VARIANTS:
        data = arrays[variant, first]
        for index in range(len(table)):
            rows.append(
                {
                    "variant": variant,
                    "index": index,
                    "shell": float(shells[index]),
                    "error_median": float(data["error_median"][index]),
                    "error_max": float(data["error_max"][index]),
                    **{f"slope_{a}": float(data["slopes"][index, i]) for i, a in enumerate("xyz")},
                    **{
                        f"true_slope_{a}": float(data["true_slopes"][index, i])
                        for i, a in enumerate("xyz")
                    },
                }
            )
    agree = {
        variant: float(
            max(
                np.abs(
                    arrays[variant, threads]["estimated"] - arrays[variant, first]["estimated"]
                ).max()
                for threads in args.threads
            )
        )
        for variant in VARIANTS
    }
    mask = np.asanyarray(clean.brainmask, dtype=bool)
    write_results(
        args.out,
        args.protocol,
        rows,
        {
            "Protocol": args.protocol,
            "Copy": {COPY: COPIES[COPY]},
            "Noise": NOISE,
            "Edges": list(EDGES),
            "PhaseEncoding": PHASE_ENCODING,
            "FieldOfViewCentre": field_of_view_centre(clean.affine, mask.shape).tolist(),
            "Runs": {f"{v}-t{t}": record for (v, t), record in runs.items()},
            "ThreadsAgreeTo": agree,
            "DatasetCommit": _dataset_commit(args.dataset),
            "Provenance": provenance(),
        },
    )
    figures(args.out, rows, runs, args.threads)


INK, MUTED, RULE, NEUTRAL = "#0b0b0b", "#52514e", "#d9d8d4", "#a9a8a3"
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
            "legend.frameon": False,
        }
    )
    return plt


def _zeros(values: np.ndarray) -> int:
    """How many elements are zero (below 1e-12)."""
    return int(np.sum(np.abs(values) < 1e-12))


def _identity(ax: Any, values: np.ndarray) -> None:
    low, high = float(np.min(values)), float(np.max(values))
    pad = 0.05 * (high - low)
    ax.plot([low - pad, high + pad], [low - pad, high + pad], color=RULE, lw=0.8, zorder=0)
    ax.set_aspect("equal")


def figures(out: Path, rows: list[dict], runs: dict, threads: list[int]) -> None:
    """The four figures: slopes, errors, parameters and runtime."""
    plt = _style()
    shells = sorted({row["shell"] for row in rows})
    by = {variant: [row for row in rows if row["variant"] == variant] for variant in VARIANTS}

    # 1. Slopes against the truth
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharex=True, sharey=True)
    for ax, variant in zip(axes, VARIANTS, strict=True):
        every = []
        for shell, color, marker in zip(shells, SHELL_COLORS, SHELL_MARKERS, strict=False):
            chosen = [row for row in by[variant] if row["shell"] == shell]
            true = np.array([[row[f"true_slope_{a}"] for a in "xyz"] for row in chosen])
            found = np.array([[row[f"slope_{a}"] for a in "xyz"] for row in chosen])
            ax.scatter(
                true.ravel(),
                found.ravel(),
                s=9,
                color=color,
                marker=marker,
                lw=0,
                label=f"b={shell:g}",
            )
            every.append(np.r_[true.ravel(), found.ravel()])
        _identity(ax, np.concatenate(every))
        residual = np.array(
            [[row[f"slope_{a}"] - row[f"true_slope_{a}"] for a in "xyz"] for row in by[variant]]
        )
        ax.text(
            0.04,
            0.96,
            f"RMS error {np.sqrt(np.mean(residual**2)):.5f} mm/mm",
            transform=ax.transAxes,
            va="top",
            color=INK,
            fontsize=8,
        )
        ax.set_title(variant, color=INK)
        ax.set_xlabel("true slope (mm/mm)")
    axes[0].set_ylabel("fitted slope (mm/mm)")
    axes[1].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "slopes.png", dpi=200)
    plt.close(fig)

    # 2. The field error per volume, by shell
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    rng = np.random.default_rng(0)
    for ax, variant in zip(axes, VARIANTS, strict=True):
        for position, (shell, color) in enumerate(zip(shells, SHELL_COLORS, strict=False)):
            values = np.array([r["error_median"] for r in by[variant] if r["shell"] == shell])
            ax.scatter(
                position + rng.uniform(-0.18, 0.18, len(values)), values, s=8, color=color, lw=0
            )
            median = float(np.median(values))
            ax.plot([position - 0.28, position + 0.28], [median, median], color=INK, lw=1.2)
            ax.annotate(
                f"{median:.3f}", (position + 0.3, median), color=INK, fontsize=8, va="center"
            )
        ax.set_xticks(range(len(shells)), [f"b={s:g}" for s in shells])
        ax.set_title(variant, color=INK)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("field error per volume (mm, median in the mask)")
    fig.tight_layout()
    fig.savefig(out / "errors.png", dpi=200)
    plt.close(fig)

    # 3. C and p against the truth
    first = min(threads)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
    allq = runs["all-q", first]
    true, found = np.ravel(allq["Truth"]["Matrix"]), np.ravel(allq["Fitted"]["Matrix"])
    axes[0].scatter(true, found, s=14, color=INK, lw=0)
    _identity(axes[0], np.r_[true, found])
    axes[0].set_title("all-q: one C", color=INK)
    axes[0].text(
        0.04,
        0.96,
        f"p = {allq['Fitted']['Exponent']:.3f} (truth {allq['Truth']['Exponent']:.3f})\n"
        f"{_zeros(true)} elements are 0 in the truth",
        transform=axes[0].transAxes,
        va="top",
        color=INK,
        fontsize=8,
    )
    qwise = runs["q-wise", first]
    every = []
    for k, (part, truth) in enumerate(
        zip(qwise["Fitted"]["Parts"], qwise["Truth"]["Parts"], strict=True)
    ):
        true, found = np.ravel(truth["Matrix"]), np.ravel(part["Matrix"])
        axes[1].scatter(
            true,
            found,
            s=14,
            color=SHELL_COLORS[k],
            marker=SHELL_MARKERS[k],
            lw=0,
            label=f"b={shells[k]:g}",
        )
        every.append(np.r_[true, found])
    _identity(axes[1], np.concatenate(every))
    zeros = sum(_zeros(np.ravel(truth["Matrix"])) for truth in qwise["Truth"]["Parts"])
    axes[1].text(
        0.04,
        0.96,
        f"{zeros} elements are 0 in the truth",
        transform=axes[1].transAxes,
        va="top",
        color=INK,
        fontsize=8,
    )
    axes[1].set_title("q-wise: one C per shell", color=INK)
    axes[1].legend(loc="lower right", fontsize=8)
    for ax in axes:
        ax.set_xlabel("true element of C")
    axes[0].set_ylabel("fitted element of C")
    fig.tight_layout()
    fig.savefig(out / "parameters.png", dpi=200)
    plt.close(fig)

    # 4. Runtime and memory
    labels = [f"{v}, {t} thread{'s' if t > 1 else ''}" for v in VARIANTS for t in threads]
    keys = [(v, t) for v in VARIANTS for t in threads]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    y = np.arange(len(keys))[::-1]
    add = np.array([runs[k]["AddSeconds"] for k in keys]) / 60
    fit = np.array([runs[k]["FitSeconds"] for k in keys]) / 60
    axes[0].barh(y, add, height=0.5, color=NEUTRAL, label="preparing the images")
    axes[0].barh(y, fit, left=add, height=0.5, color=SHELL_COLORS[0], label="fitting")
    for yy, total in zip(y, add + fit, strict=True):
        axes[0].annotate(
            f"{total:.1f}",
            (total, yy),
            xytext=(3, 0),
            textcoords="offset points",
            va="center",
            color=INK,
            fontsize=8,
        )
    axes[0].set_yticks(y, labels)
    axes[0].set_xlabel("wall time (min)")
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=8)
    peak = np.array([runs[k]["PeakRssMb"] for k in keys]) / 1024
    loaded = np.array([runs[k]["RssLoadedMb"] for k in keys]) / 1024
    axes[1].barh(y, peak, height=0.5, color=SHELL_COLORS[0], label="peak")
    axes[1].scatter(loaded, y, color=INK, s=14, zorder=3, label="after loading the data")
    axes[1].set_xlabel("resident memory (GB)")
    axes[1].legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "runtime.png", dpi=200)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", type=Path, required=True, help="phantom dataset root")
    parser.add_argument("--protocol", default="sherbrooke")
    parser.add_argument("--out", type=Path, required=True, help="results folder")
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 6], help="ITK threads")
    parser.add_argument("--run", nargs=2, metavar=("VARIANT", "THREADS"), help=argparse.SUPPRESS)
    parser.add_argument("--gather", action="store_true", help="only gather the runs written")
    args = parser.parse_args(argv)
    args.dataset, args.out = args.dataset.resolve(), args.out.resolve()
    if args.run is not None:
        variant, threads = args.run
        if variant not in VARIANTS:
            parser.error(f"unknown variant {variant!r}")
        run(args, variant, int(threads))
        return 0
    if not args.gather:
        for variant in VARIANTS:
            for threads in args.threads:
                _launch(args, variant, threads)
    gather(args)
    print(f"wrote {args.out / (args.protocol + '.tsv')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
