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
"""Draw the figures and tables of ``docs/registration.rst`` from the phantom dataset.

Usage::

    python tools/registration/floor_figures.py --dataset <dataset> \
        --out docs/_static/registration

``<dataset>`` is the phantom dataset of ``docs/phantoms.rst``, with the results that
``tools/registration/floor.py`` wrote into ``derivatives/registration/floor``. The
variants are named A to I in the order the script ran them. Images are shown in RAS
orientation, with anterior up.
"""

import argparse
import base64
import csv
import json
import re
import tempfile
import textwrap
from io import BytesIO
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nb
import nitransforms as nt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from nipype.interfaces.ants.registration import Registration
from nireports.reportlets.mosaic import plot_slice
from PIL import Image

from nifreeze.analysis.phantoms import load_phantom
from nifreeze.registration.utils import displacements_within_mask

PROTOCOL = "sherbrooke"
SHELLS = {1000.0: "#2a78d6", 2000.0: "#eb6834", 3500.0: "#1baf7a"}
"""The color of each shell (three slots validated for color-vision deficiencies)."""
MAGNITUDES = {1.0: "#2a78d6", 4.0: "#eb6834"}
"""The color of each motion magnitude."""
INK, MUTED, RULE = "#0b0b0b", "#52514e", "#d9d8d4"
DIVERGING = LinearSegmentedColormap.from_list("diverging", ["#2a78d6", "#f0efec", "#e34948"])

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


def describe(variant: dict, short: bool = False) -> dict[str, str]:
    """The settings of a variant, in words (``short`` for the labels of a figure)."""
    dense = variant["sampling"] == "dense"
    return {
        "Shrink factor": str(variant["shrink"]),
        "Noise": "yes" if variant["noise"] else "no",
        "Optimization": variant["optimizer"],
        "Sampled voxels": ("all" if dense else "18%")
        if short
        else ("every voxel" if dense else "random 18%"),
        "Inputs": variant["encoding"].replace("int12", "12-bit"),
    }


def changes(variant: dict, default: dict) -> str:
    """What a variant changes from the default (variant A), in words."""
    words = {
        "shrink": "shrink factor 1",
        "noise": "no noise",
        "optimizer": "longer optimization",
        "sampling": "every voxel sampled",
        "encoding": "float32 inputs",
    }
    changed = [words[key] for key in words if variant[key] != default[key]]
    return ", ".join(changed) or "none: the estimator's default, with noise"


def command(settings: dict) -> str:
    """The ``antsRegistration`` command of these settings, with placeholder file names."""
    with tempfile.TemporaryDirectory() as tmp:
        names = [Path(tmp) / name for name in ("fixed.nii", "moving.nii", "mask.nii")]
        for name in names:
            name.touch()
        line = Registration(
            fixed_image=str(names[0]),
            moving_image=str(names[1]),
            fixed_image_masks=[str(names[2])] * len(settings["metric"]),
            output_transform_prefix="ants-",
            **settings,
        ).cmdline
    return line.replace(f"{tmp}/", "")


def options(line: str) -> list[str]:
    """The options of a command, in order (values never start with a dash)."""
    return re.split(r"\s+(?=-)", line)[1:]


def second_stage(line: str) -> list[str]:
    """The options of the second stage, from its ``--transform`` to the next stage's."""
    opts = options(line)
    starts = [i for i, opt in enumerate(opts) if opt.startswith("--transform")]
    return opts[starts[1] : starts[1] + 6]


def axial(volume: np.ndarray, affine: np.ndarray, z: int) -> np.ndarray:
    """An axial slice in RAS orientation, anterior up."""
    canonical = nb.as_closest_canonical(nb.Nifti1Image(volume, affine))
    return np.rot90(np.asarray(canonical.dataobj)[:, :, z])


def inputs_figure(clean, noisy, moved, mask, picks, out: Path) -> Path:
    """The fixed and moving images of one volume per shell, at a magnitude of 1."""
    canonical_mask = np.asarray(
        nb.as_closest_canonical(nb.Nifti1Image(mask.astype("uint8"), clean.affine)).dataobj
    )
    z = int(round(np.argwhere(canonical_mask).mean(axis=0)[2]))
    titles = [
        "Fixed image (oracle):\nnoise-free, before the motion",
        "Moving image:\nnoise-free, after the motion",
        "Moving minus fixed\n(both noise-free)",
        "Moving image with noise\n(as in the benchmark)",
    ]
    fig, axes = plt.subplots(len(picks), 4, figsize=(8.0, 2.1 * len(picks)), squeeze=False)
    for row, (shell, index) in enumerate(picks.items()):
        fixed = np.asarray(clean.dataobj[index], dtype="float32")
        slices = [
            axial(fixed, clean.affine, z),
            axial(moved[index], clean.affine, z),
            axial(moved[index] - fixed, clean.affine, z),
            axial(noisy[index], clean.affine, z),
        ]
        top = np.percentile(fixed[mask], 99.5)
        spread = np.percentile(np.abs(moved[index] - fixed)[mask], 99.5)
        for col, (ax, image) in enumerate(zip(axes[row], slices, strict=True)):
            if col == 2:
                ax.imshow(image, cmap=DIVERGING, vmin=-spread, vmax=spread)
                ax.text(
                    0.03, 0.03, f"±{spread:.0f}", transform=ax.transAxes, color=INK, fontsize=7
                )
            else:
                ax.imshow(image, cmap="gray", vmin=0, vmax=top)
            if col == 0:
                ax.contour(
                    axial(mask.astype("float32"), clean.affine, z),
                    [0.5],
                    colors="#eb6834",
                    linewidths=0.6,
                )
                ax.set_ylabel(f"b={shell:g}\n(volume {index})", color=INK)
            if row == 0:
                ax.set_title(titles[col], color=INK, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
    fig.tight_layout()
    path = out / "floor-inputs.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def outline_polylines(brain: np.ndarray, zs: np.ndarray, rows: int) -> list[str]:
    """The outline of the brain on each panel of a frame, as SVG polylines in pixels."""
    from skimage.measure import find_contours

    nx, ny = brain.shape[:2]
    polylines = []
    for r in range(rows):
        for c, z in enumerate(zs):
            # plot_slice draws the slice transposed, with its origin at the bottom
            for contour in find_contours(brain[:, :, z].T.astype(float), 0.5):
                points = " ".join(
                    f"{c * nx + x + 0.5:.1f},{r * ny + ny - 0.5 - y:.1f}" for y, x in contour
                )
                polylines.append(f'<polyline points="{points}"/>')
    return polylines


def orientations_animation(
    clean, copies: dict, mask: np.ndarray, volumes: list[int], out: Path, encoding: str = "JPEG"
) -> Path:
    """The registered volumes, noise-free and in order: the oracle and each moved copy.

    Each frame draws axial slices with the panel renderer of NiReports' mosaic
    reportlet (:func:`~nireports.reportlets.mosaic.plot_slice`), one pixel per voxel,
    into a single grayscale image; the frames play in an SVG with a CSS animation, as
    NiReports' reportlets do, under one outline of the brain at the reference
    position. (``plot_mosaic`` writes one image per slice into an SVG, which for 64
    frames of three rows would weigh about 6 MB.)
    """
    slices, scale, seconds = 3, 2, 0.25
    reference = nb.as_closest_canonical(nb.Nifti1Image(mask.astype("uint8"), clean.affine))
    zooms = reference.header.get_zooms()[:2]
    brain = np.asarray(reference.dataobj) > 0
    corners = np.argwhere(brain)
    crop = tuple(slice(a, b + 1) for a, b in zip(corners.min(0), corners.max(0), strict=True))
    brain = brain[crop]
    nx, ny, nz = brain.shape
    zs = np.linspace(0, nz - 1, slices + 2).round().astype(int)[1:-1]

    def canonical(data: np.ndarray) -> np.ndarray:
        image = nb.as_closest_canonical(nb.Nifti1Image(data, clean.affine))
        return np.asarray(image.dataobj)[crop]

    labels = ["No motion", "(the oracle)"], *[[f"{m:g}° and {m:g} mm"] for m in sorted(copies)]
    rows = len(labels)
    frames = []
    for index in volumes:
        oracle = np.asarray(clean.dataobj[index], dtype="float32")
        window = float(np.percentile(oracle[mask], 99.5))
        images = [oracle, *[copies[m][index] for m in sorted(copies)]]
        # One pixel per voxel: each panel is nx by ny pixels at 100 dpi
        fig, axes = plt.subplots(rows, slices, figsize=(slices * nx / 100, rows * ny / 100))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
        for row, image in zip(axes, images, strict=True):
            data = canonical(image)
            for ax, z in zip(row, zs, strict=True):
                plot_slice(data[:, :, z], spacing=zooms, vmin=1e-6, vmax=window, ax=ax)
        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=100, facecolor="black")
        plt.close(fig)
        frame = Image.open(buffer).convert("L")
        encoded = BytesIO()
        frame.save(encoded, format=encoding, **({"quality": 80} if encoding == "JPEG" else {}))
        shell = float(np.round(clean.gradients[index, -1], -2))
        frames.append((index, shell, base64.b64encode(encoded.getvalue()).decode()))

    outline = outline_polylines(brain, zs, rows)

    width, height = frame.size
    left, top = 120, 26
    mime = f"image/{encoding.lower()}"
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'viewBox="0 0 {left + scale * width} {top + scale * height}" font-family="sans-serif" '
        'font-size="13">',
        "<style>",
        f".frame {{ visibility: hidden; animation: play {seconds * len(frames):g}s step-end "
        "infinite; }",
        f"@keyframes play {{ 0% {{ visibility: visible; }} {100 / len(frames):.4f}% "
        "{ visibility: hidden; } 100% { visibility: hidden; } }",
        "image { image-rendering: pixelated; }",
        "</style>",
        f'<rect width="{left + scale * width}" height="{top + scale * height}" fill="black"/>',
    ]
    for r, lines in enumerate(labels):
        middle = top + (r + 0.5) * scale * height / rows
        for k, line in enumerate(lines):
            dy = (k - (len(lines) - 1) / 2) * 16
            svg.append(
                f'<text x="{left - 10}" y="{middle + dy + 5:.0f}" text-anchor="end" '
                f'fill="#f0efec">{line}</text>'
            )
    for k, (index, shell, payload) in enumerate(frames):
        svg.append(
            f'<g class="frame" style="animation-delay: {k * seconds:g}s">'
            f'<image x="{left}" y="{top}" width="{scale * width}" height="{scale * height}" '
            f'xlink:href="data:{mime};base64,{payload}"/>'
            f'<text x="{left}" y="18" fill="#f0efec">Volume {index} of {len(clean)}, '
            f"b={shell:g} s/mm²</text></g>"
        )
    svg.append(
        f'<g transform="translate({left} {top}) scale({scale})" fill="none" stroke="#eb6834" '
        f'stroke-width="0.6">'
    )
    svg.extend(outline)
    svg.append("</g>")
    svg.append("</svg>")
    path = out / "floor-orientations.svg"
    path.write_text("\n".join(svg) + "\n")
    return path


def errors_figure(rows, variants, letters, motion, out: Path) -> Path:
    """The error of every registration, by variant, magnitude and shell."""
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 5.0), sharey=True)
    labels = [describe(v, short=True) for v in variants]
    for ax, magnitude in zip(axes, sorted(motion), strict=True):
        for k, letter in enumerate(letters):
            chosen = [r for r in rows if r["variant"] == letter and r["magnitude"] == magnitude]
            for shell, color in SHELLS.items():
                values = [r["mean"] for r in chosen if r["shell"] == shell]
                x = k + rng.uniform(-0.28, 0.28, len(values))
                ax.scatter(
                    x,
                    values,
                    s=12,
                    color=color,
                    edgecolors="white",
                    linewidths=0.4,
                    label=f"b={shell:g}" if k == 0 else None,
                    zorder=2,
                )
            median = np.median([r["mean"] for r in chosen])
            ax.plot([k - 0.36, k + 0.36], [median, median], color=INK, lw=1.5, zorder=3)
            # The median's value, above the points
            ax.text(
                k,
                max(r["mean"] for r in chosen) * 1.3,
                f"{median:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
                color=INK,
                zorder=4,
            )
        ax.axhline(motion[magnitude], color=MUTED, lw=1, ls="--", zorder=1)
        ax.text(
            len(letters) - 0.5,
            motion[magnitude] * 1.08,
            f"no registration (median): {motion[magnitude]:.2f} mm",
            ha="right",
            va="bottom",
            fontsize=7,
            color=MUTED,
        )
        ax.set_yscale("log")
        ax.set_ylim(0.005, 8)
        ax.set_xlim(-0.6, len(letters) - 0.4)
        ax.set_xticks(range(len(letters)), letters, color=INK, fontsize=9)
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="y", color=RULE, lw=0.5, zorder=0)
        ax.set_title(f"Motion of {magnitude:g}° and {magnitude:g} mm", color=INK, fontsize=9)
        # The settings of each variant, below its letter; bold where it differs from A
        for line, key in enumerate(labels[0]):
            y = -0.08 - 0.045 * line
            if ax is axes[0]:
                ax.text(
                    -0.75,
                    y,
                    key,
                    transform=ax.get_xaxis_transform(),
                    ha="right",
                    va="center",
                    fontsize=7,
                    color=MUTED,
                )
            for k, label in enumerate(labels):
                bold = label[key] != labels[0][key]
                ax.text(
                    k,
                    y,
                    label[key],
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color=INK if bold else MUTED,
                    fontweight="bold" if bold else "normal",
                )
    axes[0].set_ylabel("Mean displacement error within the brain (mm)")
    axes[0].legend(loc="upper left", frameon=False, fontsize=7, title="Shell", title_fontsize=7)
    fig.subplots_adjust(left=0.13, right=0.99, top=0.93, bottom=0.27, wspace=0.06)
    path = out / "floor-errors.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def residuals_figure(rows, variants, letters, out: Path) -> Path:
    """The residual displacement of the brain's centroid, along x and y, per variant."""
    fig, axes = plt.subplots(3, 3, figsize=(7.5, 7.8), sharex=True, sharey=True)
    for ax, letter, variant in zip(axes.flat, letters, variants, strict=True):
        chosen = [r for r in rows if r["variant"] == letter]
        ax.axhline(0, color=RULE, lw=0.8, zorder=0)
        ax.axvline(0, color=RULE, lw=0.8, zorder=0)
        for magnitude, color in MAGNITUDES.items():
            points = np.array([(r["tx"], r["ty"]) for r in chosen if r["magnitude"] == magnitude])
            ax.scatter(
                points[:, 0],
                points[:, 1],
                s=9,
                color=color,
                edgecolors="white",
                linewidths=0.3,
                label=f"{magnitude:g}° and {magnitude:g} mm",
                zorder=2,
            )
        shifts = np.array([(r["tx"], r["ty"], r["tz"]) for r in chosen])
        ax.text(
            0.03,
            0.97,
            "mean (x, y, z): "
            + ", ".join(f"{round(v, 2) + 0.0:+.2f}" for v in shifts.mean(axis=0))
            + "\nSD (x, y, z): "
            + ", ".join(f"{v:.2f}" for v in shifts.std(axis=0)),
            transform=ax.transAxes,
            va="top",
            fontsize=6.5,
            color=INK,
        )
        title = changes(variant, variants[0]).replace(
            "none: the estimator's default", "the default"
        )
        ax.set_title(textwrap.fill(f"{letter}: {title}", 34), fontsize=7.5, color=INK)
        ax.set_aspect("equal")
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
    for ax in axes[-1]:
        ax.set_xlabel("x residual, left to right (mm)")
    for ax in axes[:, 0]:
        ax.set_ylabel("y residual, posterior to\nanterior (mm)")
    axes[0, 0].legend(
        loc="lower left", frameon=False, fontsize=6.5, title="Motion", title_fontsize=6.5
    )
    fig.tight_layout()
    path = out / "floor-residuals.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_csv(path: Path, header: list[str], rows: list[list]) -> Path:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def write_tables(
    rows: list[dict],
    variants: list[dict],
    letters: list[str],
    settings: list[dict],
    motion: dict[float, list[float]],
    out: Path,
) -> None:
    """The default command, the variants, the results, and the mean residuals."""
    # The default command, one option per line, and what each variant changes
    lines = [command(s) for s in settings]
    default = options(lines[0])
    (out / "floor-command.txt").write_text(
        "antsRegistration \\\n" + " \\\n".join(f"    {opt}" for opt in default) + "\n"
    )
    base = second_stage(lines[0])
    write_csv(
        out / "floor-variants.csv",
        ["Variant", *describe(variants[0]), "Second-stage options that differ from A"],
        [
            [
                letter,
                *describe(variant).values(),
                " ".join(
                    f"``{o}``" for o, b in zip(second_stage(line), base, strict=True) if o != b
                )
                or "(none)",
            ]
            for letter, variant, line in zip(letters, variants, lines, strict=True)
        ],
    )

    def stats(values: list[float]) -> list[str]:
        return [f"{np.median(values):.3f}", f"{np.quantile(values, 0.95):.3f}"]

    magnitudes = sorted(motion)
    table = [
        [
            "(none)",
            "no registration: the true motion",
            *sum((stats(motion[m]) for m in magnitudes), []),
            "",
        ]
    ]
    for letter, variant in zip(letters, variants, strict=True):
        per = [
            [r["mean"] for r in rows if r["variant"] == letter and r["magnitude"] == m]
            for m in magnitudes
        ]
        seconds = [r["seconds"] for r in rows if r["variant"] == letter]
        table.append(
            [
                letter,
                changes(variant, variants[0]),
                *sum((stats(p) for p in per), []),
                f"{np.median(seconds):.1f}",
            ]
        )
    write_csv(
        out / "floor-results.csv",
        [
            "Variant",
            "Change from A",
            *[f"{m:g}° and {m:g} mm: {s}" for m in magnitudes for s in ("median", "p95")],
            "Seconds (median)",
        ],
        table,
    )

    def shell_median(letter: str, shell: float) -> str:
        values = [r["mean"] for r in rows if r["variant"] == letter and r["shell"] == shell]
        return f"{np.median(values):.3f}"

    write_csv(
        out / "floor-shells.csv",
        ["Variant", *[f"b={s:g}" for s in SHELLS]],
        [[letter, *[shell_median(letter, s) for s in SHELLS]] for letter in letters],
    )

    # The mean residual of each variant: a bias would show as a mean away from zero
    columns = ("tx", "ty", "tz", "rx", "ry", "rz")
    write_csv(
        out / "floor-bias.csv",
        [
            "Variant",
            *[f"Mean shift along {a} (mm)" for a in "xyz"],
            *[f"Mean rotation about {a} (°)" for a in "xyz"],
        ],
        [
            [letter]
            + [
                f"{round(np.mean([r[c] for r in rows if r['variant'] == letter]), 3) + 0.0:+.3f}"
                for c in columns
            ]
            for letter in letters
        ],
    )


def load_inputs(dataset: Path, settings: dict):
    """The data, loaded as the experiment loaded them, and the true motion's error."""
    clean, _ = load_phantom(dataset, PROTOCOL, counts=[])
    mask = np.asanyarray(clean.brainmask, dtype=bool)
    mask_img = nb.Nifti1Image(mask.astype("uint8"), clean.affine)
    evaluate = settings["Evaluate"]
    shells = np.round(clean.gradients[:, -1], -2)
    picks = {float(s): next(i for i in evaluate if shells[i] == s) for s in np.unique(shells)}
    motion, noisy, copies = {}, {}, {}
    for seed in (settings["Noise"], None):
        _, cells = load_phantom(dataset, PROTOCOL, noise=seed)
        for perturbation, load in cells:
            if perturbation.distortion is not None or perturbation.count != len(clean):
                continue
            if perturbation.magnitude not in settings["Magnitudes"]:
                continue
            truth = dict(zip(perturbation.moved, perturbation.motions, strict=True))
            motion[perturbation.magnitude] = [
                float(displacements_within_mask(mask_img, nt.linear.Affine(truth[i])).mean())
                for i in evaluate
            ]
            copy = load()
            if seed is None:
                copies[perturbation.magnitude] = {
                    i: np.asarray(copy.dataobj[i], dtype="float32") for i in evaluate
                }
            elif perturbation.magnitude == 1.0:
                noisy = {i: np.asarray(copy.dataobj[i], dtype="float32") for i in picks.values()}
    return clean, mask, picks, motion, noisy, copies


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", type=Path, required=True, help="phantom dataset root")
    parser.add_argument("--out", type=Path, required=True, help="folder of the figures")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    results = args.dataset / "derivatives" / "registration" / "floor"
    settings = json.loads((results / f"{PROTOCOL}.json").read_text())
    variants = [
        {k: v for k, v in item.items() if k != "Settings"} for item in settings["Variants"]
    ]
    letters = [chr(ord("A") + k) for k in range(len(variants))]
    key = {
        json.dumps(v, sort_keys=True): letter for v, letter in zip(variants, letters, strict=True)
    }
    rows = []
    with (results / f"{PROTOCOL}.tsv").open() as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            variant = {
                "shrink": int(row["shrink"]),
                "noise": row["noise"] == "True",
                "optimizer": row["optimizer"],
                "sampling": row["sampling"],
                "encoding": row["encoding"],
            }
            numbers = {k: float(row[k]) for k in row if k not in variant}
            rows.append({"variant": key[json.dumps(variant, sort_keys=True)], **numbers})

    clean, mask, picks, motion, noisy, copies = load_inputs(args.dataset, settings)
    inputs_figure(clean, noisy, copies[1.0], mask, picks, args.out)
    orientations_animation(clean, copies, mask, settings["Evaluate"], args.out)
    errors_figure(
        rows, variants, letters, {m: float(np.median(v)) for m, v in motion.items()}, args.out
    )
    residuals_figure(rows, variants, letters, args.out)

    write_tables(
        rows,
        variants,
        letters,
        [item["Settings"] for item in settings["Variants"]],
        motion,
        args.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
