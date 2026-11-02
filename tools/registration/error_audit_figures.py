# /// script
# requires-python = ">=3.12"
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
"""The figures and tables of the audit of what the experiments count as error (#626).

Reads ``derivatives/registration/error-audit`` of the phantom dataset, which
``tools/registration/error_audit.py`` wrote, and writes into ``--out``::

    uv run tools/registration/error_audit_figures.py --dataset <dataset> \\
        --out docs/_static/registration/error-audit

Images are shown in RAS orientation, with anterior up.
"""

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nb
import numpy as np
from error_audit import _rigid, _sampler
from matplotlib.colors import LinearSegmentedColormap

from nifreeze.analysis.phantoms import load_phantom

PROTOCOLS = {"sherbrooke": "Sherbrooke (2 mm)", "hcpya": "HCP-YA (1.25 mm)"}
"""The protocols, as the report names them."""
# One meaning per color in every figure: blue is the truth as the sidecar records it,
# aqua its inverse, orange the opposite sign, yellow no correction
EDDY = {
    "none": ("No correction", "#eda100"),
    "sidecar": ("x - d(x) u, the sidecar's coefficients", "#2a78d6"),
    "inverted": ("x - d_w(x) u, the inverted coefficients", "#1baf7a"),
    "opposite": ("x + d(x) u, the opposite sign", "#eb6834"),
}
"""The candidate maps of check B, by the column prefix of their residuals."""
TOTAL = {
    "motion": ("M x, the motion only", "#eda100"),
    "sidecar": ("M x - d(M x) u", "#2a78d6"),
    "inverted": ("M x - d_w(M x) u", "#1baf7a"),
    "opposite": ("M x + d(M x) u", "#eb6834"),
}
"""The candidate maps of a combined copy's whole mapping (check B)."""
MOTION = {
    "identity": ("x, no correction", "#eda100"),
    "matrix": ("M x, the sidecar's matrix", "#2a78d6"),
    "inverse": ("M⁻¹ x, its inverse", "#1baf7a"),
}
"""The candidate maps of check A."""
DOMAINS = {"mask": "Brain mask", "grown": "Grown mask", "volume": "Whole volume"}
"""The domains of check C."""
GAP = 0.1
"""Where the sidecar's and the inverted displacements differ by more (voxels), check B
also compares them."""

INK, MUTED, RULE = "#0b0b0b", "#52514e", "#d9d8d4"
SEQUENTIAL = LinearSegmentedColormap.from_list("sequential", ["#f0efec", "#e34948", "#5c1414"])

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def read(dataset: Path) -> dict[str, list[dict[str, Any]]]:
    """The rows of each protocol that was run, with numbers as numbers."""
    tables = {}
    for protocol in PROTOCOLS:
        path = dataset / "derivatives" / "registration" / "error-audit" / f"{protocol}.tsv"
        if not path.exists():
            continue
        with path.open() as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
        for row in rows:
            for key, value in row.items():
                if key not in ("kind", "copy"):
                    row[key] = float(value) if value != "" else None
        tables[protocol] = rows
    return tables


def motion(name: str) -> float:
    found = re.match(r"m([0-9p]+)s", name)
    return float(found[1].replace("p", ".")) if found else 0.0


def eddy(name: str) -> int:
    found = re.search(r"e(\d+)$", name)
    return int(found[1]) if found else 0


def order(copies) -> list[str]:
    """By motion, then by eddy currents."""
    return sorted(set(copies), key=lambda name: (motion(name), eddy(name)))


def axial(volume: np.ndarray, affine: np.ndarray, z: int) -> np.ndarray:
    """An axial slice in RAS orientation, anterior up."""
    canonical = nb.as_closest_canonical(nb.Nifti1Image(volume, affine))
    return np.rot90(np.asarray(canonical.dataobj)[:, :, z])


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fraction(passed: np.ndarray) -> str:
    return f"{int(passed.sum())} of {len(passed)}"


def _strips(ax, groups: list[str], series: dict[str, tuple[str, str]], values) -> None:
    """Per group, one strip of points per series, with its median."""
    rng = np.random.default_rng(0)
    width = 0.8 / len(series)
    for position, group in enumerate(groups):
        for k, (key, (_, color)) in enumerate(series.items()):
            offset = (k - (len(series) - 1) / 2) * width
            errors = np.asarray(values(group, key))
            jitter = position + offset + rng.uniform(-0.25, 0.25, len(errors)) * width
            ax.scatter(jitter, errors, s=3, color=color, alpha=0.5, linewidths=0)
            ax.hlines(
                np.median(errors),
                position + offset - 0.4 * width,
                position + offset + 0.4 * width,
                color=INK,
                linewidth=1,
            )
    ax.set_yscale("log")
    low, high = ax.get_ylim()
    ticks = [t for t in (0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0) if low <= t <= high]
    ax.set_yticks(ticks, [f"{t:g}" for t in ticks])
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_xticks(range(len(groups)), groups, rotation=45, ha="right")
    ax.grid(axis="y", color=RULE, linewidth=0.5)


def plot_checks(tables: dict[str, list[dict[str, Any]]], out: Path) -> None:
    """The p99 of every volume's residual, per copy and candidate map, for A and B."""
    fig, axes = plt.subplots(
        len(tables),
        2,
        figsize=(11, 2.6 * len(tables)),
        layout="constrained",
        squeeze=False,
        gridspec_kw={"width_ratios": [1, 3.2]},
    )
    for row, (protocol, rows) in zip(axes, tables.items(), strict=True):
        moved = order(r["copy"] for r in rows if r["kind"] == "motion")
        distorted = order(r["copy"] for r in rows if r["kind"] != "motion")
        _strips(
            row[0],
            moved,
            MOTION,
            lambda copy, key, rows=rows: [
                r[f"motion_{key}_p99"] for r in rows if r["copy"] == copy
            ],
        )
        _strips(
            row[1],
            distorted,
            EDDY,
            lambda copy, key, rows=rows: [r[f"eddy_{key}_p99"] for r in rows if r["copy"] == copy],
        )
        row[0].set_ylabel("p99 of |corrected - expected|\nwithin the brain (/ max)")
        row[0].set_title(f"{PROTOCOLS[protocol]}: A, motion", loc="left", color=INK)
        row[1].set_title(f"{PROTOCOLS[protocol]}: B, eddy currents", loc="left", color=INK)
    # One legend under each column of panels
    for ax, series in ((axes[-1, 0], MOTION), (axes[-1, 1], EDDY)):
        handles = [
            mpl.lines.Line2D([], [], marker="o", linestyle="", color=color, label=label)
            for label, color in series.values()
        ]
        ax.legend(
            handles=handles,
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(0, -0.45),
            ncols=1 if series is MOTION else 2,
        )
    fig.savefig(out / "error-audit-checks.png", dpi=200, facecolor="white")
    plt.close(fig)


def plot_example(dataset: Path, rows: list[dict[str, Any]], out: Path) -> dict[str, Any]:
    """One Sherbrooke volume of ``e8`` and of ``m4s0``, corrected by each candidate."""
    chosen = max((r for r in rows if r["copy"] == "e8"), key=lambda r: r["truth_max_vox"])
    index = int(chosen["index"])
    clean, cells = load_phantom(dataset, "sherbrooke")
    affine = np.asarray(clean.affine, dtype=float)
    mask = np.asanyarray(clean.brainmask, dtype=bool)
    shape = mask.shape
    grid = np.indices(shape, dtype=float).reshape(3, -1)
    points = affine[:3, :3] @ grid + affine[:3, 3:]
    target = np.asarray(clean.dataobj[index], dtype=float)
    scale = target.max()

    distortion, load = next(
        (p.distortion, load)
        for p, load in cells
        if p.distortion is not None
        and p.magnitude == 0
        and round(p.distortion.magnitude / np.linalg.norm(affine[:3, p.distortion.axis])) == 8
    )
    assert distortion is not None
    distorted = np.asarray(load().dataobj[index], dtype=float)
    sample = _sampler(distorted, affine)
    coefficients = distortion.coefficients[index]
    values, derivatives = distortion.basis(points)
    slope = float(coefficients @ derivatives[:, 0])
    inverted = coefficients / (1.0 - slope)
    u = distortion.direction[:, np.newaxis]
    corrected = {
        "none": distorted.ravel(),
        "sidecar": sample(points - (coefficients @ values) * u) * (1.0 - slope),
        "inverted": sample(points - (inverted @ values) * u) * (1.0 - inverted @ derivatives),
        "opposite": sample(points + (coefficients @ values) * u) * (1.0 + slope),
    }

    perturbation, load = next(
        (p, load) for p, load in cells if p.distortion is None and p.magnitude == 4
    )
    matrix = dict(zip(perturbation.moved, perturbation.motions, strict=True))[index]
    moved = np.asarray(load().dataobj[index], dtype=float)
    sample = _sampler(moved, affine)
    realigned = {
        "identity": moved.ravel(),
        "matrix": sample(_rigid(matrix, points)),
        "inverse": sample(_rigid(np.linalg.inv(matrix), points)),
    }

    z = int(np.argwhere(mask)[:, 2].mean())
    # The copy itself is shown, so its residual without correction is only summarized
    lines = [
        ("e8", distorted, EDDY, corrected, ("sidecar", "inverted", "opposite")),
        ("m4s0", moved, MOTION, realigned, ("matrix", "inverse")),
    ]
    fig, axes = plt.subplots(2, 5, figsize=(12.5, 5.8), layout="constrained")
    summary: dict[str, Any] = {}
    for ax_row, (title, copy, series, images, drawn_keys) in zip(axes, lines, strict=True):
        shown: list[tuple[str, np.ndarray, Any, tuple[float, float] | None]] = [
            ("Expected\n(clean reference)", axial(target, affine, z), "gray", (0, scale)),
            (f"TRXScan's {title}\n", axial(copy, affine, z), "gray", (0, scale)),
        ]
        for key, image in images.items():
            residual = np.abs(image.reshape(shape) - target) / scale
            summary[f"{title}, {series[key][0]}"] = {
                "largest": float(residual[mask].max()),
                "p99": float(np.percentile(residual[mask], 99)),
            }
            if key in drawn_keys:
                label = series[key][0].split(",")[0]
                image = axial(residual, affine, z)
                shown.append(
                    (f"Sampled at {label}:\n|corrected - expected|", image, SEQUENTIAL, None)
                )
        for ax, (label, image, cmap, window) in zip(ax_row, shown, strict=False):
            low, high = window if window is not None else (0, 0.1)
            drawn = ax.imshow(image, cmap=cmap, vmin=low, vmax=high, interpolation="none")
            ax.set_title(label, color=INK)
            bar = fig.colorbar(drawn, ax=ax, shrink=0.8, location="bottom")
            bar.ax.tick_params(labelsize=7, colors=MUTED)
            bar.outline.set_edgecolor(RULE)
        for ax in ax_row:
            ax.set_axis_off()
    fig.savefig(out / "error-audit-example.png", dpi=200, facecolor="white")
    plt.close(fig)
    return {
        "index": index,
        "shell": chosen["shell"],
        "slice": z,
        "truth_max_vox": chosen["truth_max_vox"],
        "slope": slope,
        "motion": np.asarray(matrix).tolist(),
        "residuals_in_the_brain": summary,
        "residuals_shown_up_to": 0.1,
    }


Rows = list[dict[str, Any]]
"""Rows of a table."""
STATISTICS = (("p99", "p99"), ("max", "largest"))
"""The summaries of each residual, and how the tables name them."""


def _column(chosen: Rows, key: str) -> np.ndarray:
    return np.array([r[key] for r in chosen], dtype=float)


def _decision(
    check: str, protocol: str, residual: str, candidate: np.ndarray, others: list[np.ndarray]
) -> dict[str, Any]:
    """Whether ``candidate`` is below every one of ``others`` on every volume, and by how
    much it is above the lowest of them where it is not."""
    lowest = np.min(others, axis=0)
    passed = candidate < lowest
    margins = (candidate - lowest)[~passed]
    return {
        "Check": check,
        "Protocol": PROTOCOLS[protocol],
        "Residual": residual,
        "Volumes that pass": _fraction(passed),
        "Rule met": "yes" if passed.all() else "no",
        "Failures: median excess": f"{np.median(margins):.3f}" if len(margins) else "",
        "Failures: largest excess": f"{margins.max():.3f}" if len(margins) else "",
    }


def motion_tables(protocol: str, rows: Rows) -> tuple[Rows, Rows]:
    """A: the residuals per motion copy, and the decision."""
    moved = [r for r in rows if r["kind"] == "motion"]
    summary: Rows = []
    for copy in order(r["copy"] for r in moved):
        chosen = [r for r in moved if r["copy"] == copy]
        entry: dict[str, Any] = {"Protocol": PROTOCOLS[protocol], "Copy": copy}
        entry["Volumes"] = len(chosen)
        for key, (label, _) in MOTION.items():
            median = np.median(_column(chosen, f"motion_{key}_p99"))
            entry[f"{label.split(',')[0]}: median p99"] = f"{median:.3f}"
        entry["M x: worst p99"] = f"{_column(chosen, 'motion_matrix_p99').max():.3f}"
        for key, (label, _) in MOTION.items():
            median = np.median(_column(chosen, f"motion_{key}_max"))
            entry[f"{label.split(',')[0]}: median largest"] = f"{median:.2f}"
        entry["M x: worst largest"] = f"{_column(chosen, 'motion_matrix_max').max():.2f}"
        summary.append(entry)
    decisions = []
    for stat, label in STATISTICS:
        others = [_column(moved, f"motion_{key}_{stat}") for key in ("inverse", "identity")]
        matrix = _column(moved, f"motion_matrix_{stat}")
        decisions.append(_decision("A: M x below M⁻¹ x and x", protocol, label, matrix, others))
    return summary, decisions


def eddy_tables(protocol: str, rows: Rows) -> tuple[Rows, Rows]:
    """B: the residuals of the eddy currents alone, every copy with the same eddy
    currents together, and the decision."""
    distorted = [r for r in rows if r["kind"] != "motion"]
    summary: Rows = []
    for magnitude in sorted({eddy(r["copy"]) for r in distorted}):
        chosen = [r for r in distorted if eddy(r["copy"]) == magnitude]
        entry: dict[str, Any] = {
            "Protocol": PROTOCOLS[protocol],
            "Eddy currents (voxels)": magnitude,
        }
        entry["Volumes"] = len(chosen)
        for key, (label, _) in EDDY.items():
            median = np.median(_column(chosen, f"eddy_{key}_p99"))
            entry[f"{label.split(',')[0]}: median p99"] = f"{median:.3f}"
        for key in ("sidecar", "inverted"):
            label = EDDY[key][0].split(",")[0]
            entry[f"{label}: worst p99"] = f"{_column(chosen, f'eddy_{key}_p99').max():.3f}"
            entry[f"{label}: worst largest"] = f"{_column(chosen, f'eddy_{key}_max').max():.3f}"
        differ = int((_column(chosen, "inverted_gap_vox") > GAP).sum())
        entry[f"Volumes where the two differ by > {GAP} voxel"] = differ
        summary.append(entry)
    gap = _column(distorted, "inverted_gap_vox") > GAP
    decisions = []
    for stat, label in STATISTICS:
        sidecar = _column(distorted, f"eddy_sidecar_{stat}")
        opposite = _column(distorted, f"eddy_opposite_{stat}")
        inverted = _column(distorted, f"eddy_inverted_{stat}")
        decisions.append(
            _decision("B: sidecar's below the opposite sign", protocol, label, sidecar, [opposite])
        )
        decisions.append(
            _decision(
                f"B: sidecar's below the inverted, where they differ by > {GAP} voxel",
                protocol,
                label,
                sidecar[gap],
                [inverted[gap]],
            )
        )
        decisions.append(
            _decision(
                f"(Not in the rule) sidecar's below the inverted, where they differ by"
                f" {GAP} voxel or less",
                protocol,
                label,
                sidecar[~gap],
                [inverted[~gap]],
            )
        )
    return summary, decisions


def total_table(protocol: str, rows: Rows) -> Rows:
    """B: the residuals of the whole mapping of the combined copies, by motion."""
    combined = [r for r in rows if r["kind"] == "combined"]
    summary: Rows = []
    for magnitude in sorted({motion(r["copy"]) for r in combined}):
        chosen = [r for r in combined if motion(r["copy"]) == magnitude]
        entry: dict[str, Any] = {"Protocol": PROTOCOLS[protocol], "Motion": f"{magnitude:g}"}
        entry["Volumes"] = len(chosen)
        for key, (label, _) in TOTAL.items():
            median = np.median(_column(chosen, f"total_{key}_p99"))
            entry[f"{label.split(',')[0]}: median p99"] = f"{median:.3f}"
        summary.append(entry)
    return summary


def domain_table(protocol: str, rows: Rows) -> Rows:
    """C: E0a's Q1 and Q2b, by domain."""
    distorted = [r for r in rows if r["kind"] != "motion"]
    summary: Rows = []
    for model, label in (("readout", "Q1, readout"), ("inverse_warp", "Q2b, inverted warp")):
        for domain, where in DOMAINS.items():
            largest = _column(distorted, f"{model}_{domain}_max")
            p99 = _column(distorted, f"{model}_{domain}_p99")
            summary.append(
                {
                    "Protocol": PROTOCOLS[protocol],
                    "Model": label,
                    "Domain": where,
                    "Largest, median": f"{np.median(largest):.2g}",
                    "Largest, worst": f"{largest.max():.2g}",
                    "p99, median": f"{np.median(p99):.2g}",
                    "p99, worst": f"{p99.max():.2g}",
                }
            )
    return summary


def write_tables(tables: dict[str, Rows], out: Path) -> None:
    """The summaries the report quotes, and the decisions."""
    written: dict[str, Rows] = {k: [] for k in ("motion", "eddy", "total", "domain", "decision")}
    for protocol, rows in tables.items():
        for name, (summary, decisions) in (
            ("motion", motion_tables(protocol, rows)),
            ("eddy", eddy_tables(protocol, rows)),
        ):
            written[name] += summary
            written["decision"] += decisions
        written["total"] += total_table(protocol, rows)
        written["domain"] += domain_table(protocol, rows)
    for name, table in written.items():
        _write(out / f"error-audit-{name}.csv", table)


def write_runs(tables: dict[str, list[dict[str, Any]]], dataset: Path, out: Path) -> None:
    """Each protocol's run: its command, from its DataLad run record, and what it took,
    from the run's ``/usr/bin/time`` log."""
    commands, runs = [], []
    for protocol in tables:
        record = subprocess.run(
            [
                "git",
                "-C",
                str(dataset),
                "log",
                "-1",
                "--format=%h%n%B",
                "--",
                f"derivatives/registration/error-audit/{protocol}.tsv",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        found = re.search(r"^\{.*^\}", record, re.M | re.S)
        commands.append(json.loads(found.group(0))["cmd"] if found else f"(not a run: {protocol})")
        results = dataset / "derivatives" / "registration" / "error-audit"
        meta = json.loads((results / f"{protocol}.json").read_text())
        timing = (dataset / "logs" / f"error-audit-{protocol}.time").read_text()
        wall = re.search(r"Elapsed \(wall clock\) time .*: (\S+)", timing)
        memory = re.search(r"Maximum resident set size \(kbytes\): (\d+)", timing)
        assert wall is not None and memory is not None
        runs.append(
            {
                "Protocol": PROTOCOLS[protocol],
                "Run record": record.split()[0],
                "NiFreeze commit": meta["Provenance"]["NiFreezeCommit"][:8],
                "Jobs": meta["Jobs"],
                "Wall time (h:mm:ss)": wall.group(1),
                "Largest memory of one process (GB)": f"{int(memory.group(1)) * 1024 / 1e9:.1f}",
            }
        )
    (out / "error-audit-command.txt").write_text("\n\n".join(commands) + "\n")
    _write(out / "error-audit-runs.csv", runs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", type=Path, required=True, help="phantom dataset root")
    parser.add_argument("--out", type=Path, required=True, help="figures and tables folder")
    args = parser.parse_args(argv)

    tables = read(args.dataset)
    args.out.mkdir(parents=True, exist_ok=True)
    shown = plot_example(args.dataset, tables["sherbrooke"], args.out)
    (args.out / "error-audit-example.json").write_text(json.dumps(shown, indent=2) + "\n")
    plot_checks(tables, args.out)
    write_tables(tables, args.out)
    write_runs(tables, args.dataset, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
