# /// script
# requires-python = ">=3.13"
# dependencies = ["nifreeze[plotting]"]
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
"""The figures and tables of elastix's settings report (#670, #675).

Reads the results of the runs behind ``docs/edd/elastix-settings.rst``, and writes
into ``--out``:

``--mapping``
    The direct mapping of ANTs' settings against ANTs (#670):
    ``<protocol>-m<magnitude>.{tsv,npz,json}``, rigid and affine stages.
``--tuning``
    The pyramids and iterations against the direct mapping (#675): ``sherbrooke.tsv``.
``--real``
    The real-data checks: ``real-b0/real-data*.tsv`` (the suite's b=0 images) and
    ``stanford-hardi/stanford-hardi.tsv`` (DIPY's Stanford HARDI with known motion).
``--dataset``
    The phantom dataset, for the video of one series::

    uv run tools/registration/elastix_settings_figures.py --dataset <dataset> \\
        --mapping <results>/elastix-mapping --tuning <results>/elastix-tuning \\
        --real <results> --out docs/_static/registration/elastix-settings
"""

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nb
import numpy as np
from experiments import paired_comparison
from matplotlib import animation
from scipy import ndimage

INK, MUTED, RULE = "#0b0b0b", "#52514e", "#d9d8d4"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
"""Three slots validated for color-vision deficiencies (the figures of #609)."""
EDGE = "#e34948"
plt.rcParams.update(
    {
        "font.size": 9,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
    }
)

NOISE = 20261029
VOXEL = {"sherbrooke": 2.0, "hcpya": 1.25, "hcph": 1.6, "dwi": 1.8, "dwi_motion": 1.8}
"""Voxel sizes (mm): the margin of every comparison is 0.01 voxel."""
PROTOCOLS = {"sherbrooke": "Sherbrooke (2 mm)", "hcpya": "HCP-YA (1.25 mm)"}
FIXED = {
    "oracle": "Oracle",
    "unmoved": "Prediction, unmoved data",
    "moved": "Prediction, moved data",
}
OPTIMIZERS = {"asgd": "Adaptive (kept)", "cg": "Conjugate gradient (dropped)"}
SCHEDULES = {"t": "Direct mapping", "p2": "Two-level pyramid", "p3": "Three-level pyramid"}


def label(arm: str) -> str:
    """A configuration's name (``t-x4``) in words: ``Direct mapping, 4x iterations``."""
    if arm == "ants":
        return "ANTs"
    schedule, iterations = arm.split("-x")
    return f"{SCHEDULES[schedule]}, {iterations}x iterations"


def read(path: Path) -> list[dict[str, Any]]:
    """The rows of a results table, with numbers as numbers."""
    with path.open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    for row in rows:
        for key, value in row.items():
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                if value in ("True", "False"):
                    row[key] = value == "True"
    return rows


def write(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def cell(c: dict[str, Any]) -> str:
    return f"{c['median']:+.3f} [{c['median_low']:+.3f}, {c['median_high']:+.3f}], {c['verdict']}"


def compare(
    rows: list[dict[str, Any]], select: dict[str, Any], candidate: str, reference: str, key: str
) -> dict[str, Any]:
    """:func:`~experiments.paired_comparison` of two arms' mean errors, paired by volume."""

    def pairs(arm: str) -> dict[tuple, float]:
        chosen = [r for r in rows if r[key] == arm and all(r[k] == v for k, v in select.items())]
        return {(r["magnitude"], r["index"]): r["mean"] for r in chosen}

    ours, theirs = pairs(candidate), pairs(reference)
    keys = sorted(ours)
    if keys != sorted(theirs):
        raise ValueError(f"{candidate} and {reference} are not paired on {select}")
    protocol = select.get("protocol", "sherbrooke")
    return paired_comparison(
        [ours[k] for k in keys],
        [theirs[k] for k in keys],
        0.01 * VOXEL[protocol],
        strata=[k[0] for k in keys],
    )


# ---------------------------------------------------------------- the direct mapping (#670)
def mapping_tables(mapping: Path, out: Path) -> list[dict[str, Any]]:
    rows = [
        row
        for p in PROTOCOLS
        for path in sorted(mapping.glob(f"{p}-m*.tsv"))
        for row in read(path)
    ]
    comparisons = []
    for protocol in PROTOCOLS:
        for fixed in FIXED:
            for stage in ("rigid", "affine"):
                select = {"protocol": protocol, "fixed": fixed, "stage": stage}
                for arm in OPTIMIZERS:
                    result = compare(rows, select, arm, "ants", "arm")
                    comparisons.append(select | {"arm": arm} | result)
    by = {(c["protocol"], c["fixed"], c["stage"], c["arm"]): c for c in comparisons}
    write(
        out / "elastix-settings-mapping.csv",
        ["Protocol", "Fixed image", "Stage", *(f"{v} - ANTs (mm)" for v in OPTIMIZERS.values())],
        [
            [PROTOCOLS[p], FIXED[f], s, *(cell(by[(p, f, s, a)]) for a in OPTIMIZERS)]
            for p in PROTOCOLS
            for f in FIXED
            for s in ("rigid", "affine")
        ],
    )

    def seconds(protocol: str, stage: str, arm: str) -> str:
        chosen = [
            r["seconds"]
            for r in rows
            if r["protocol"] == protocol and r["stage"] == stage and r["arm"] == arm
        ]
        return f"{np.median(chosen):.2f}"

    write(
        out / "elastix-settings-mapping-time.csv",
        ["Protocol", "Stage", "ANTs (s)", *(f"{v} (s)" for v in OPTIMIZERS.values())],
        [
            [PROTOCOLS[p], s, *(seconds(p, s, a) for a in ("ants", *OPTIMIZERS))]
            for p in PROTOCOLS
            for s in ("rigid", "affine")
        ],
    )
    count = sum(1 for r in rows if r["arm"] == "ants" and r["stage"] == "rigid")
    print(f"mapping: {count} pairs", flush=True)
    return rows


def bland_altman(rows: list[dict[str, Any]], out: Path) -> None:
    """Adaptive minus ANTs against their mean, rigid stage, by protocol and fixed image."""
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.6), layout="constrained", sharey="row")
    for r_, protocol in enumerate(PROTOCOLS):
        for c_, fixed in enumerate(FIXED):
            ax = axes[r_, c_]
            pick = {
                arm: {
                    (r["magnitude"], r["index"]): r["mean"]
                    for r in rows
                    if r["protocol"] == protocol
                    and r["fixed"] == fixed
                    and r["stage"] == "rigid"
                    and r["arm"] == arm
                }
                for arm in ("ants", "asgd")
            }
            keys = sorted(pick["ants"])
            a = np.array([pick["ants"][k] for k in keys])
            e = np.array([pick["asgd"][k] for k in keys])
            diff, mean = e - a, (e + a) / 2
            ax.axhline(0, color=RULE, linewidth=0.8)
            ax.scatter(mean, diff, s=6, color=BLUE, alpha=0.45, linewidths=0)
            low, mid, high = np.percentile(diff, [2.5, 50, 97.5])
            for value, style in ((low, ":"), (mid, "-"), (high, ":")):
                ax.axhline(value, color=INK, linewidth=0.8, linestyle=style)
            ax.text(
                0.02,
                0.97,
                f"median {mid:+.3f} mm\n95% of pairs in [{low:+.3f}, {high:+.3f}]",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7.5,
                color=INK,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.5},
            )
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(mpl.ticker.FixedLocator([0.05, 0.1, 0.2, 0.5, 1, 2, 5]))
            ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
            ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
            if r_ == 0:
                ax.set_title(FIXED[fixed], color=INK)
            if c_ == 0:
                ax.set_ylabel(f"{PROTOCOLS[protocol]}\nelastix - ANTs (mm)")
            if r_ == 1:
                ax.set_xlabel("Mean of the two errors (mm)")
    fig.savefig(out / "elastix-settings-bland-altman.png", dpi=200, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- the video of one series
ORIENTS = ("axial", "coronal", "sagittal")


def _planes(shape: list[int], center: list[int]) -> dict[str, np.ndarray]:
    """Cropped-grid voxels of each cut, rows top to bottom (subject's left on the left)."""
    nx, ny, nz = shape
    ci, cj, ck = center
    u, v = np.meshgrid(np.arange(nx), np.arange(ny))
    axial = np.stack([u, ny - 1 - v, np.full_like(u, ck)])
    u, v = np.meshgrid(np.arange(nx), np.arange(nz))
    coronal = np.stack([u, np.full_like(u, cj), nz - 1 - v])
    u, v = np.meshgrid(np.arange(ny), np.arange(nz))
    sagittal = np.stack([np.full_like(u, ci), u, nz - 1 - v])
    return dict(zip(ORIENTS, (axial, coronal, sagittal), strict=True))


def _sample(volume, affine, matrix, crop, plane, order=1) -> np.ndarray:
    """``volume`` at ``matrix`` applied to the plane's points: the sample where A sends them."""
    c = np.vstack([plane.reshape(3, -1), np.ones((1, plane[0].size))])
    vox = (np.linalg.inv(affine) @ matrix @ crop @ c)[:3]
    out = ndimage.map_coordinates(volume, vox, order=order, mode="constant", cval=0.0)
    return out.reshape(plane.shape[1:])


def _edges(image: np.ndarray, inside: np.ndarray) -> np.ndarray:
    """The target's strongest edges within the brain (Sobel, above the 93rd percentile)."""
    magnitude = np.hypot(ndimage.sobel(image, 0), ndimage.sobel(image, 1))
    if not inside.any():
        return np.zeros_like(inside)
    return inside & (magnitude > np.percentile(magnitude[inside], 93))


KINDS = ("target", "before", "ants", "asgd")


def _series_frames(
    dataset: Path,
    mapping: Path,
    rows: list[dict[str, Any]],
    protocol: str,
    magnitude: float,
    fixed: str,
) -> dict[str, Any]:
    """The cuts of every development volume: the target, and the sample before and after."""
    from experiments import subset

    from nifreeze.analysis.phantoms import load_phantom
    from nifreeze.analysis.sensitivity import held_out_predictions

    reference, cells = load_phantom(dataset, protocol, noise=NOISE)
    indices = subset(protocol, "dev", len(reference))
    _, load = next(
        (p, load)
        for p, load in cells
        if p.distortion is None
        and p.count == len(reference)
        and p.magnitude == magnitude
        and p.seed == 0
    )
    moved = load()
    affine = np.asarray(reference.affine, dtype=float)
    mask = np.asanyarray(reference.brainmask, dtype=bool)
    if fixed == "oracle":
        clean, _ = load_phantom(dataset, protocol, counts=[])
        targets = {i: np.asarray(clean.dataobj[i], dtype="float32") for i in indices}
    else:
        targets = held_out_predictions(
            "gqi-krr", reference if fixed == "unmoved" else moved, indices
        )

    canonical = nb.as_closest_canonical(nb.Nifti1Image(mask.astype("uint8"), affine))
    inside = np.asanyarray(canonical.dataobj) > 0
    lo = np.maximum(np.array(np.nonzero(inside)).min(axis=1) - 6, 0)
    hi = np.minimum(np.array(np.nonzero(inside)).max(axis=1) + 7, inside.shape)
    shift = np.eye(4)
    shift[:3, 3] = lo
    crop = canonical.affine @ shift
    cropped = inside[tuple(slice(a, b) for a, b in zip(lo, hi, strict=True))]
    shape = [int(s) for s in hi - lo]
    center = [int(round(c)) for c in np.array(np.nonzero(cropped)).mean(axis=1)]
    cuts = _planes(shape, center)
    brain = {
        o: _sample(mask.astype("float32"), affine, np.eye(4), crop, cuts[o], 0) > 0.5
        for o in ORIENTS
    }

    estimates = np.load(mapping / f"{protocol}-m{magnitude:g}.npz")
    errors = {
        (r["arm"], int(r["index"])): r["mean"]
        for r in rows
        if r["protocol"] == protocol
        and r["magnitude"] == magnitude
        and r["fixed"] == fixed
        and r["stage"] in ("rigid", "none")
    }
    shells = {int(r["index"]): r["shell"] for r in rows if r["protocol"] == protocol}
    frames = []
    for i in indices:
        target = np.asarray(targets[i], dtype="float32")
        sample = np.asarray(moved.dataobj[i], dtype="float32")
        window = max(np.percentile(target[mask], 99.5), np.percentile(sample[mask], 99.5))
        matrices = {
            "target": np.eye(4),
            "before": np.eye(4),
            "ants": estimates[f"{fixed}/{i}/ants/rigid"],
            "asgd": estimates[f"{fixed}/{i}/asgd/rigid"],
        }
        images = {}
        for kind in KINDS:
            volume = target if kind == "target" else sample
            for o in ORIENTS:
                images[(kind, o)] = np.clip(
                    _sample(volume, affine, matrices[kind], crop, cuts[o]) / window, 0, 1
                )
        frames.append((i, images))
    return {
        "indices": indices,
        "frames": frames,
        "brain": brain,
        "shape": shape,
        "errors": errors,
        "shells": shells,
    }


def _panels(shape: list[int], frames: list, fixed: str) -> tuple:
    """The figure: a row per cut and a column per image, above room for the timeline."""
    titles = {
        "target": f"Target: {FIXED[fixed].lower()}",
        "before": "Before registration",
        "ants": "ANTs",
        "asgd": "elastix, direct mapping",
    }
    fig = plt.figure(figsize=(10.0, 8.4), dpi=100, facecolor="white")
    grid = fig.add_gridspec(
        4,
        4,
        height_ratios=[shape[1], shape[2], shape[2], 0.62 * shape[1]],
        left=0.07,
        right=0.99,
        top=0.9,
        bottom=0.07,
        hspace=0.12,
        wspace=0.03,
    )
    shown, overlays, subtitles = {}, {}, {}
    for c_, kind in enumerate(KINDS):
        for r_, o in enumerate(ORIENTS):
            ax = fig.add_subplot(grid[r_, c_])
            ax.set_axis_off()
            first = frames[0][1][(kind, o)]
            shown[(kind, o)] = ax.imshow(
                first, cmap="gray", vmin=0, vmax=1, aspect="equal", interpolation="bilinear"
            )
            if kind != "target":
                overlays[(kind, o)] = ax.imshow(
                    np.zeros((*first.shape, 4)), aspect="equal", interpolation="nearest"
                )
            if c_ == 0:
                ax.text(
                    -0.04,
                    0.5,
                    o.capitalize(),
                    transform=ax.transAxes,
                    rotation=90,
                    ha="right",
                    va="center",
                    color=INK,
                )
            if o != "sagittal":
                for x, text in ((0.02, "L"), (0.98, "R")):
                    ax.text(
                        x,
                        0.5,
                        text,
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=7,
                    )
            if r_ == 0:
                ax.set_title(titles[kind], color=INK, fontsize=9, pad=14)
                subtitles[kind] = ax.text(
                    0.5,
                    1.02,
                    "",
                    transform=ax.transAxes,
                    ha="center",
                    va="bottom",
                    color=MUTED,
                    fontsize=8,
                )
    header = fig.text(0.5, 0.965, "", ha="center", va="center", color=INK, fontsize=10)
    return fig, grid, shown, overlays, subtitles, header


def series_video(
    dataset: Path,
    mapping: Path,
    rows: list[dict[str, Any]],
    out: Path,
    protocol: str,
    magnitude: float,
    fixed: str,
) -> None:
    """Every development volume of one condition, in acquisition order, ANTs beside elastix."""
    data = _series_frames(dataset, mapping, rows, protocol, magnitude, fixed)
    indices, frames, errors = data["indices"], data["frames"], data["errors"]
    fig, grid, shown, overlays, subtitles, header = _panels(data["shape"], frames, fixed)

    timeline = fig.add_subplot(grid[3, :])
    order = np.arange(1, len(indices) + 1)
    for arm, color, name in (
        ("none", MUTED, "Before registration"),
        ("ants", ORANGE, "ANTs"),
        ("asgd", BLUE, "elastix, direct mapping"),
    ):
        timeline.plot(
            order,
            [errors[(arm, i)] for i in indices],
            color=color,
            linewidth=1.4,
            label=name,
            marker="o",
            markersize=2.5,
        )
    timeline.set_yscale("log")
    # Room above the lines for the legend
    values = [errors[(arm, i)] for arm in ("none", "ants", "asgd") for i in indices]
    timeline.set_ylim(min(values) / 1.3, max(values) * 3)
    timeline.yaxis.set_major_locator(mpl.ticker.FixedLocator([0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]))
    timeline.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    timeline.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    timeline.set_xlim(0.5, len(indices) + 0.5)
    timeline.set_xlabel("Development volume, in acquisition order")
    timeline.set_ylabel("Mean error (mm)")
    timeline.legend(loc="upper left", frameon=False, ncols=3, fontsize=8)
    cursor = timeline.axvline(1, color=INK, linewidth=0.8)

    writer = animation.FFMpegWriter(
        fps=3, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "30"]
    )
    path = out / f"elastix-settings-series-{protocol}-m{magnitude:g}-{fixed}.mp4"
    with writer.saving(fig, str(path), dpi=100):
        for k, (i, images) in enumerate(frames):
            for (kind, o), artist in shown.items():
                artist.set_data(images[(kind, o)])
            for (kind, o), artist in overlays.items():
                rgba = np.zeros((*images[(kind, o)].shape, 4))
                rgba[_edges(images[("target", o)], data["brain"][o])] = mpl.colors.to_rgba(EDGE)
                artist.set_data(rgba)
            subtitles["target"].set_text("")
            subtitles["before"].set_text(f"mean error {errors[('none', i)]:.3f} mm")
            subtitles["ants"].set_text(f"mean error {errors[('ants', i)]:.3f} mm")
            subtitles["asgd"].set_text(f"mean error {errors[('asgd', i)]:.3f} mm")
            header.set_text(
                f"{PROTOCOLS[protocol]}, {magnitude:g} mm and degrees: volume {i} "
                f"(b={data['shells'][i]:g}), {k + 1} of {len(indices)}"
            )
            cursor.set_xdata([k + 1, k + 1])
            writer.grab_frame(facecolor="white")
    plt.close(fig)
    print(path, len(frames), "frames", flush=True)


# ---------------------------------------------------------------- the pyramids (#675)
def tuning_tables(tuning: Path, out: Path) -> None:
    rows = read(tuning / "sherbrooke.tsv")
    arms = list(dict.fromkeys(r["arm"] for r in rows))
    configurations = [a for a in arms if a not in ("ants", "t-x1")]
    seconds = {a: float(np.median([r["seconds"] for r in rows if r["arm"] == a])) for a in arms}
    against = {}
    for reference, candidates in (("t-x1", configurations), ("ants", arms[1:])):
        for arm in candidates:
            for fixed in FIXED:
                against[(reference, arm, fixed)] = compare(
                    rows, {"fixed": fixed}, arm, reference, "arm"
                )

    for reference, name, candidates in (
        ("t-x1", "direct", configurations),
        ("ants", "ants", arms[1:]),
    ):
        write(
            out / f"elastix-settings-pyramids-{name}.csv",
            ["Configuration", *FIXED.values(), "Median seconds"],
            [
                [label(a), *(cell(against[(reference, a, f)]) for f in FIXED), f"{seconds[a]:.2f}"]
                for a in candidates
            ],
        )

    def p95(arm: str, fixed: str) -> str:
        errors = [r["mean"] for r in rows if r["arm"] == arm and r["fixed"] == fixed]
        return f"{np.percentile(errors, 95):.3f}"

    write(
        out / "elastix-settings-pyramids-tail.csv",
        ["Configuration", *(f"{v}: p95 (mm)" for v in FIXED.values()), "Failures"],
        [
            [
                label(a),
                *(p95(a, f) for f in FIXED),
                str(int(sum(r["failed"] for r in rows if r["arm"] == a))),
            ]
            for a in arms
        ],
    )

    # Median error against median time, by fixed image
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), layout="constrained")
    styles = {"t": (BLUE, "o", "-"), "p2": (GREEN, "^", "-"), "p3": (GREEN, "s", "--")}
    for ax, fixed in zip(axes, FIXED, strict=True):
        median = {
            a: float(np.median([r["mean"] for r in rows if r["arm"] == a and r["fixed"] == fixed]))
            for a in arms
        }
        ax.scatter([seconds["ants"]], [median["ants"]], color=ORANGE, s=36, zorder=3)
        ax.annotate(
            "ANTs",
            (seconds["ants"], median["ants"]),
            xytext=(5, 4),
            textcoords="offset points",
            color=INK,
            fontsize=8,
        )
        for schedule, (color, marker, line) in styles.items():
            names = [a for a in arms if a.startswith(f"{schedule}-x")]
            xs, ys = [seconds[a] for a in names], [median[a] for a in names]
            ax.plot(xs, ys, color=color, linestyle=line, linewidth=1.2, label=SCHEDULES[schedule])
            # The marker's size gives the iterations: 1x, 4x and 16x
            ax.scatter(xs, ys, color=color, marker=marker, s=[16, 40, 90], zorder=3)
        ax.set_title(FIXED[fixed], color=INK)
        ax.set_xlabel("Median seconds per registration")
        if fixed == "oracle":
            ax.set_ylabel("Median error (mm)")
            ax.legend(frameon=False, fontsize=7.5, loc="center right")
    fig.savefig(out / "elastix-settings-pyramids.png", dpi=200, facecolor="white")
    plt.close(fig)
    print(f"tuning: {len(rows)} registrations", flush=True)


# ---------------------------------------------------------------- the real-data checks
def real_tables(real: Path, out: Path) -> None:
    rows = [
        row for path in sorted((real / "real-b0").glob("real-data*.tsv")) for row in read(path)
    ]
    arms = ["ants", "t-x1", "p2-x1", "p3-x1"]
    datasets = {
        "hcph": "hcph (216 motions)",
        "dwi": "dwi (216 motions)",
        "dwi_motion": "dwi_motion (7 motions)",
    }
    table = []
    for dataset, name in datasets.items():
        for arm in arms:
            errors = np.array(
                [r["mean"] for r in rows if r["dataset"] == dataset and r["arm"] == arm]
            )
            seconds = [r["seconds"] for r in rows if r["dataset"] == dataset and r["arm"] == arm]
            table.append(
                [
                    name,
                    label(arm),
                    f"{np.median(errors):.3f}",
                    f"{np.percentile(errors, 95):.3f}",
                    f"{errors.max():.3f}",
                    str(int((errors >= 0.6 * VOXEL[dataset]).sum())),
                    f"{np.median(seconds):.2f}",
                ]
            )
    write(
        out / "elastix-settings-b0.csv",
        [
            "Image",
            "Settings",
            "Median (mm)",
            "p95 (mm)",
            "Worst (mm)",
            "Mean error of 0.6 voxel or more",
            "Median seconds",
        ],
        table,
    )

    rows = read(real / "stanford-hardi" / "stanford-hardi.tsv")
    arms = ["ants", "t-x1", "p3-x1"]
    magnitudes = sorted({r["magnitude"] for r in rows})
    table = []
    for fixed in ("unmoved", "moved"):
        for magnitude in magnitudes:
            medians = [
                np.median(
                    [
                        r["mean"]
                        for r in rows
                        if r["arm"] == a and r["fixed"] == fixed and r["magnitude"] == magnitude
                    ]
                )
                for a in arms
            ]
            table.append([FIXED[fixed], f"{magnitude:g}", *(f"{m:.3f}" for m in medians)])
    table.append(
        [
            "Median seconds",
            "",
            *(f"{np.median([r['seconds'] for r in rows if r['arm'] == a]):.2f}" for a in arms),
        ]
    )
    write(
        out / "elastix-settings-stanford.csv",
        ["Fixed image", "Motion (mm and degrees)", *(label(a) for a in arms)],
        table,
    )
    comparisons = []
    for fixed in ("unmoved", "moved"):
        for magnitude in magnitudes[1:]:
            select = {"fixed": fixed, "magnitude": magnitude}
            comparisons.append(
                [
                    FIXED[fixed],
                    f"{magnitude:g}",
                    *(
                        cell(compare(rows, select, c, r, "arm"))
                        for c, r in (("t-x1", "ants"), ("p3-x1", "ants"), ("p3-x1", "t-x1"))
                    ),
                ]
            )
    write(
        out / "elastix-settings-stanford-comparisons.csv",
        [
            "Fixed image",
            "Motion (mm and degrees)",
            "Direct mapping - ANTs (mm)",
            "Three-level pyramid - ANTs (mm)",
            "Three-level pyramid - direct mapping (mm)",
        ],
        comparisons,
    )
    print(f"real data: {len(rows)} Stanford HARDI registrations", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", type=Path, required=True, help="phantom dataset root")
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--tuning", type=Path, required=True)
    parser.add_argument("--real", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-video", action="store_true", help="skip the video (slow)")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    rows = mapping_tables(args.mapping, args.out)
    bland_altman(rows, args.out)
    tuning_tables(args.tuning, args.out)
    real_tables(args.real, args.out)
    if not args.no_video:
        series_video(args.dataset, args.mapping, rows, args.out, "sherbrooke", 4.0, "moved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
