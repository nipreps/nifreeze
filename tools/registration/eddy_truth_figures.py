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
"""The figures and tables of the eddy-current truth experiment (#618).

Reads ``derivatives/registration/eddy-truth`` of the phantom dataset, which
``tools/registration/eddy_truth.py`` wrote, and writes into ``--out``::

    uv run tools/registration/eddy_truth_figures.py --dataset <dataset> \\
        --out docs/_static/registration/eddy-truth

Images are shown in RAS orientation, with anterior up.
"""

import argparse
import csv
import json
import re
import subprocess
from itertools import product
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import attrs
import matplotlib.pyplot as plt
import nibabel as nb
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from nifreeze.analysis.phantoms import load_phantom

PROTOCOLS = {"sherbrooke": "Sherbrooke (2 mm)", "hcpya": "HCP-YA (1.25 mm)"}
"""The protocols, as the figures name them."""
MODELS = {
    "readout": ("Readout (as TRXScan forms the volume)", "#2a78d6"),
    "warp": ("Warp by the sidecar's coefficients", "#eb6834"),
    "inverse": ("Warp by the inverted coefficients", "#1baf7a"),
}
"""The label and color of each model of the distortion (three slots validated for
color-vision deficiencies, as in the figures of #609)."""
SHELLS = {1000.0: "#2a78d6", 2000.0: "#eb6834", 3000.0: "#1baf7a", 3500.0: "#1baf7a"}
"""The color of each shell; the highest shell of each protocol shares a color."""
THRESHOLD = 0.05
"""The decision threshold (voxels) on the 95th percentile of the fit's error."""
INK, MUTED, RULE = "#0b0b0b", "#52514e", "#d9d8d4"
DIVERGING = LinearSegmentedColormap.from_list("diverging", ["#2a78d6", "#f0efec", "#e34948"])
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
        path = dataset / "derivatives" / "registration" / "eddy-truth" / f"{protocol}.tsv"
        if not path.exists():
            continue
        with path.open() as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
        for row in rows:
            for key, value in row.items():
                if key != "copy":
                    row[key] = float(value) if value != "" else None
        tables[protocol] = rows
    return tables


def order(copies) -> list[str]:
    """Eddy-only copies first, then the combined ones by motion and eddy currents."""

    def key(name: str) -> tuple[float, int]:
        found = re.fullmatch(r"(?:m(?P<m>[0-9p]+)s\d+)?e(?P<e>\d+)", name)
        assert found is not None, name
        motion = float(found["m"].replace("p", ".")) if found["m"] else 0.0
        return motion, int(found["e"])

    return sorted(set(copies), key=key)


def axial(volume: np.ndarray, affine: np.ndarray, z: int) -> np.ndarray:
    """An axial slice in RAS orientation, anterior up."""
    canonical = nb.as_closest_canonical(nb.Nifti1Image(volume, affine))
    return np.rot90(np.asarray(canonical.dataobj)[:, :, z])


def plot_inputs(dataset: Path, rows: list[dict[str, Any]], out: Path) -> dict[str, Any]:
    """One Sherbrooke volume of ``e8``: the volumes, each model's residual, the field."""
    chosen = max((r for r in rows if r["copy"] == "e8"), key=lambda r: r["truth_max_vox"])
    index = int(chosen["index"])
    clean, cells = load_phantom(dataset, "sherbrooke")
    affine = np.asarray(clean.affine, dtype=float)
    readout, load = next(
        (p.distortion, load)
        for p, load in cells
        if p.distortion is not None
        and p.magnitude == 0
        and round(p.distortion.magnitude / np.linalg.norm(affine[:3, p.distortion.axis])) == 8
    )
    assert readout is not None
    undistorted = np.asarray(clean.dataobj[index], dtype=float)
    distorted = np.asarray(load().dataobj[index], dtype=float)
    coefficients = readout.coefficients[index]
    inverse = np.array([chosen[f"inverse_c{k}"] for k in range(len(coefficients))])
    warp = attrs.evolve(readout, readout=None)
    formed = readout.distort(undistorted, affine, coefficients)
    warped = warp.distort(undistorted, affine, coefficients)
    inverted = warp.distort(undistorted, affine, inverse)
    ijk = np.indices(undistorted.shape, dtype=float).reshape(3, -1)
    values, derivatives = readout.basis(affine[:3, :3] @ ijk + affine[:3, 3:])
    # The slope along phase encoding (mm per mm), the same at every point of a linear field
    slope = float(coefficients @ derivatives[:, 0])
    voxel = float(np.linalg.norm(affine[:3, readout.axis]))
    field = (coefficients @ values).reshape(undistorted.shape) / voxel

    mask = np.asanyarray(clean.brainmask, dtype=bool)
    z = int(np.argwhere(mask)[:, 2].mean())
    scale = distorted.max()
    residuals = {
        title: np.abs(model - distorted) / scale
        for title, model in (
            ("|Readout\n- TRXScan|", formed),
            ("|Warp, sidecar's\n- TRXScan|", warped),
            ("|Warp, inverted\n- TRXScan|", inverted),
        )
    }
    panels = [
        ("Undistorted\n(clean reference)", axial(undistorted, affine, z), "gray", (0, scale)),
        ("TRXScan's copy\n", axial(distorted, affine, z), "gray", (0, scale)),
        *(
            (title, axial(residual, affine, z), SEQUENTIAL, (0, 0.1))
            for title, residual in residuals.items()
        ),
        ("Sidecar's displacement\n(voxels)", axial(field, affine, z), DIVERGING, None),
    ]
    limit = float(np.abs(field[mask]).max())
    outline = axial(mask.astype(float), affine, z)
    fig, axes = plt.subplots(1, len(panels), figsize=(12.5, 3.0), layout="constrained")
    for ax, (title, image, cmap, window) in zip(axes, panels, strict=True):
        low, high = window if window is not None else (-limit, limit)
        shown = ax.imshow(image, cmap=cmap, vmin=low, vmax=high, interpolation="none")
        ax.set_title(title, color=INK)
        ax.set_axis_off()
        bar = fig.colorbar(shown, ax=ax, shrink=0.8, location="bottom")
        bar.ax.tick_params(labelsize=7, colors=MUTED)
        bar.outline.set_edgecolor(RULE)
    for ax, residual in zip(axes[2:5], residuals.values(), strict=True):
        ax.text(
            0.03,
            0.97,
            f"largest in the brain\n{residual[mask].max():.2g}",
            transform=ax.transAxes,
            va="top",
            fontsize=7,
            color=INK,
        )
    axes[-1].contour(outline, levels=[0.5], colors=INK, linewidths=0.6)
    fig.savefig(out / "eddy-truth-inputs.png", dpi=200, facecolor="white")
    plt.close(fig)
    return {
        "copy": "e8",
        "index": index,
        "shell": chosen["shell"],
        "slice": z,
        "truth_max_vox": chosen["truth_max_vox"],
        "sidecar_coefficients": coefficients.tolist(),
        "slope": slope,
        "gain": 1.0 / (1.0 - slope),
        "inverted_coefficients": inverse.tolist(),
        "fitted_coefficients": (
            [chosen[f"fit_c{k}"] for k in range(len(coefficients))]
            if chosen["fit_p95_vox"] is not None
            else None
        ),
        "largest_residuals_in_the_brain": {
            title.replace("\n", " "): float(residual[mask].max())
            for title, residual in residuals.items()
        },
        "residuals_shown_up_to": 0.1,
    }


def plot_models(tables: dict[str, list[dict[str, Any]]], out: Path) -> None:
    """Per copy, how far each model of the distortion is from TRXScan's volumes."""
    fig, axes = plt.subplots(
        len(tables), 1, figsize=(7.5, 2.4 * len(tables)), layout="constrained", squeeze=False
    )
    rng = np.random.default_rng(0)
    for ax, (protocol, rows) in zip(axes[:, 0], tables.items(), strict=True):
        copies = order(r["copy"] for r in rows)
        for position, name in enumerate(copies):
            for offset, (model, (_, color)) in zip((-0.27, 0, 0.27), MODELS.items(), strict=True):
                errors = np.array([r[f"{model}_max"] for r in rows if r["copy"] == name])
                jitter = position + offset + rng.uniform(-0.07, 0.07, len(errors))
                ax.scatter(jitter, errors, s=3, color=color, alpha=0.5, linewidths=0)
                ax.hlines(
                    np.median(errors),
                    position + offset - 0.11,
                    position + offset + 0.11,
                    color=INK,
                    linewidth=1,
                )
        ax.set_yscale("log")
        ax.set_xticks(range(len(copies)), copies, rotation=45, ha="right")
        ax.set_ylabel("Largest |model - TRXScan|\nwithin the brain (/ max)")
        ax.set_title(PROTOCOLS[protocol], loc="left", color=INK)
        ax.grid(axis="y", color=RULE, linewidth=0.5)
    handles = [
        mpl.lines.Line2D([], [], marker="o", linestyle="", color=color, label=label)
        for label, color in MODELS.values()
    ]
    fig.legend(handles=handles, frameon=False, loc="outside upper center", ncols=3)
    fig.savefig(out / "eddy-truth-models.png", dpi=200, facecolor="white")
    plt.close(fig)


def plot_fit(tables: dict[str, list[dict[str, Any]]], out: Path) -> None:
    """Per fitted volume, the error of the best warp against the sidecar's displacement
    (as pre-registered) and against the inverted one (post hoc)."""
    against = {
        "fit_p95_vox": "Q3, sidecar's coefficients",
        "fit_inverse_p95_vox": "Q3b, inverted coefficients",
    }
    fig, axes = plt.subplots(
        len(tables),
        len(against),
        figsize=(7.5, 2.7 * len(tables)),
        layout="constrained",
        sharey=True,
        squeeze=False,
    )
    for row_axes, (protocol, rows) in zip(axes, tables.items(), strict=True):
        fitted = [r for r in rows if r["fit_p95_vox"] is not None]
        for ax, (key, title) in zip(row_axes, against.items(), strict=True):
            for shell in sorted({r["shell"] for r in fitted}):
                chosen = [r for r in fitted if r["shell"] == shell]
                ax.scatter(
                    [r["truth_max_vox"] for r in chosen],
                    [r[key] for r in chosen],
                    s=6,
                    color=SHELLS[shell],
                    linewidths=0,
                    label=f"b = {shell:g}",
                )
            ax.axhline(
                THRESHOLD,
                color=MUTED,
                linewidth=0.8,
                linestyle="--",
                label=f"threshold, {THRESHOLD} voxel",
            )
            ax.set_yscale("log")
            ax.set_xlabel("Largest true displacement in the brain (voxels)")
            ax.set_title(f"{PROTOCOLS[protocol]}: {title}", loc="left", color=INK)
            ax.grid(color=RULE, linewidth=0.5)
        row_axes[0].set_ylabel("p95 of the displacement's\nerror in the brain (voxels)")
        row_axes[-1].legend(frameon=False, fontsize=7, loc="upper right", markerscale=2)
    fig.savefig(out / "eddy-truth-fit.png", dpi=200, facecolor="white")
    plt.close(fig)


def write_tables(tables: dict[str, list[dict[str, Any]]], out: Path) -> None:
    """The summaries the report quotes, and the decision."""

    def eddy(name: str) -> int:
        found = re.search(r"e(\d+)$", name)
        assert found is not None, name
        return int(found[1])

    models: list[dict[str, Any]] = []
    fits: list[dict[str, Any]] = []
    for protocol, rows in tables.items():
        # Every copy with the same eddy currents, with and without motion, together
        for magnitude in sorted({eddy(r["copy"]) for r in rows}):
            chosen = [r for r in rows if eddy(r["copy"]) == magnitude]
            for model, (label, _) in MODELS.items():
                largest = np.array([r[f"{model}_max"] for r in chosen])
                p99 = np.array([r[f"{model}_p99"] for r in chosen])
                models.append(
                    {
                        "Protocol": PROTOCOLS[protocol],
                        "Eddy currents (voxels)": magnitude,
                        "Model": label,
                        "Volumes": len(chosen),
                        "Largest, median": f"{np.median(largest):.2g}",
                        "Largest, worst": f"{largest.max():.2g}",
                        "p99, median": f"{np.median(p99):.2g}",
                        "p99, worst": f"{p99.max():.2g}",
                    }
                )
        for name in order(r["copy"] for r in rows if r["fit_p95_vox"] is not None):
            fitted = [r for r in rows if r["copy"] == name and r["fit_p95_vox"] is not None]
            fit = np.array([[r[f"fit_c{k}"] for k in range(4)] for r in fitted])
            for against, prefix, coefficients in (
                ("sidecar's", "fit", "true_c"),
                ("inverted", "fit_inverse", "inverse_c"),
            ):
                p95 = np.array([r[f"{prefix}_p95_vox"] for r in fitted])
                truth = np.array([[r[f"{coefficients}{k}"] for k in range(4)] for r in fitted])
                fits.append(
                    {
                        "Protocol": PROTOCOLS[protocol],
                        "Copy": name,
                        "Against the": against,
                        "Volumes": len(fitted),
                        "p95, median": f"{np.median(p95):.2g}",
                        "p95, worst": f"{p95.max():.2g}",
                        f"p95 above {THRESHOLD}": int((p95 >= THRESHOLD).sum()),
                        "Largest difference, linear terms (mm)": (
                            f"{np.abs(fit[:, 1:] - truth[:, 1:]).max():.2g}"
                        ),
                        "Largest difference, constant term (mm)": (
                            f"{np.abs(fit[:, 0] - truth[:, 0]).max():.2g}"
                        ),
                    }
                )
    for name, rows in (("eddy-truth-models.csv", models), ("eddy-truth-fit.csv", fits)):
        with (out / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    decisions = []
    for (rule, against), protocol in product(
        (
            ("Q3, as pre-registered: against the sidecar's coefficients", "sidecar's"),
            ("Q3b, post hoc: against the inverted coefficients", "inverted"),
        ),
        tables,
    ):
        chosen = [
            row
            for row in fits
            if row["Against the"] == against and row["Protocol"] == PROTOCOLS[protocol]
        ]
        above = sum(row[f"p95 above {THRESHOLD}"] for row in chosen)
        decisions.append(
            {
                "Comparison": rule,
                "Protocol": PROTOCOLS[protocol],
                "Fitted volumes": sum(row["Volumes"] for row in chosen),
                f"p95 above {THRESHOLD} voxel": above,
                "Rule holds": "yes" if above == 0 else "no",
            }
        )
    with (out / "eddy-truth-decision.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(decisions[0]))
        writer.writeheader()
        writer.writerows(decisions)


def write_subset(tables: dict[str, list[dict[str, Any]]], out: Path) -> None:
    """The diffusion-weighted volumes of each copy, and those fitted in Q3, per shell."""
    subset = []
    for protocol, rows in tables.items():
        first = [r for r in rows if r["copy"] == rows[0]["copy"]]
        for shell in sorted({r["shell"] for r in first}):
            chosen = [r for r in first if r["shell"] == shell]
            subset.append(
                {
                    "Protocol": PROTOCOLS[protocol],
                    "b-value": f"{shell:g}",
                    "Volumes per copy": len(chosen),
                    "Fitted volumes per copy": sum(r["fit_p95_vox"] is not None for r in chosen),
                }
            )
    with (out / "eddy-truth-subset.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(subset[0]))
        writer.writeheader()
        writer.writerows(subset)


def write_reproduced(tables: dict[str, list[dict[str, Any]]], dataset: Path, out: Path) -> None:
    """How far the last run is from the pre-registered one, the first to write each table."""
    reproduced = []
    for protocol, rows in tables.items():
        path = f"derivatives/registration/eddy-truth/{protocol}.tsv"
        log = ["git", "-C", str(dataset), "log", "--format=%h", "--", path]
        commits = subprocess.run(log, capture_output=True, text=True, check=True).stdout.split()
        shown = ["git", "-C", str(dataset), "show", f"{commits[-1]}:{path}"]
        text = subprocess.run(shown, capture_output=True, text=True, check=True).stdout
        first = list(csv.DictReader(text.splitlines(), delimiter="\t"))
        assert [(r["copy"], float(r["index"])) for r in first] == [
            (r["copy"], r["index"]) for r in rows
        ]
        # Every column the first run wrote, but its own timing
        columns = [key for key in first[0] if key not in ("copy", "index", "seconds")]
        differences = [
            abs(float(a[key]) - b[key])
            for a, b in zip(first, rows, strict=True)
            for key in columns
            if a[key] != ""
        ]
        reproduced.append(
            {
                "Protocol": PROTOCOLS[protocol],
                "First run": commits[-1],
                "Last run": commits[0],
                "Rows": len(rows),
                "Columns compared": len(columns),
                "Largest difference": f"{max(differences):.1e}",
            }
        )
    with (out / "eddy-truth-reproduced.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(reproduced[0]))
        writer.writeheader()
        writer.writerows(reproduced)


def write_runs(tables: dict[str, list[dict[str, Any]]], dataset: Path, out: Path) -> None:
    """The last run of each protocol: its command, from its DataLad run record, and what
    it took, from the run's ``/usr/bin/time`` log."""
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
                f"derivatives/registration/eddy-truth/{protocol}.tsv",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        found = re.search(r"^\{.*^\}", record, re.M | re.S)
        commands.append(json.loads(found.group(0))["cmd"] if found else f"(not a run: {protocol})")
        results = dataset / "derivatives" / "registration" / "eddy-truth"
        meta = json.loads((results / f"{protocol}.json").read_text())
        timing = (dataset / "logs" / f"eddy-truth-{protocol}.time").read_text()
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
    (out / "eddy-truth-command.txt").write_text("\n\n".join(commands) + "\n")
    with (out / "eddy-truth-runs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(runs[0]))
        writer.writeheader()
        writer.writerows(runs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", type=Path, required=True, help="phantom dataset root")
    parser.add_argument("--out", type=Path, required=True, help="figures and tables folder")
    args = parser.parse_args(argv)

    tables = read(args.dataset)
    args.out.mkdir(parents=True, exist_ok=True)
    shown = plot_inputs(args.dataset, tables["sherbrooke"], args.out)
    (args.out / "eddy-truth-inputs.json").write_text(json.dumps(shown, indent=2) + "\n")
    plot_models(tables, args.out)
    plot_fit(tables, args.out)
    write_tables(tables, args.out)
    write_subset(tables, args.out)
    write_reproduced(tables, args.dataset, args.out)
    write_runs(tables, args.dataset, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
