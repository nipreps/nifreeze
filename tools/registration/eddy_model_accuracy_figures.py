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
"""The figure and the table of GEM's accuracy on the phantom, in ``docs/gem.rst``.

Reads ``<results>/accuracy.json``, which ``tools/registration/eddy_model_accuracy.py``
wrote. Two panels share the targets along their x-axis (the noise-free originals, then
the predictions of gqi-krr, then those of gp-loo): GEM's fitted exponent ``p``, against
the truth's, and the share of the true field it recovers. Each copy's eddy currents have
a line, solid with the moving volumes noise-free and dashed with noise. The table holds
the same values::

    python tools/registration/eddy_model_accuracy_figures.py --results <results> \\
        --out docs/_static/gem
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from eddy_model_accuracy import NOISE_LEVELS, SIZES, TARGETS

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
COLORS = ("#6da7ec", "#256abf", "#0d366b")
"""One step of a blue ramp for each size of the eddy currents, light to dark."""
LINES: dict[str, tuple[Any, float]] = {"noise-free": ("-", -0.11), "noisy": ((0, (4, 2.5)), 0.11)}
"""Each condition's line style, and its offset along the x-axis."""
DODGE = (-0.045, 0.0, 0.045)
"""Each size's offset along the x-axis, within its condition."""
LIMIT = 2.0
"""The upper limit of the fitted exponent (``EXPONENT_BOUNDS`` of ``eddy_field.py``)."""
LABELS = {
    "original": "noise-free\noriginals",
    "gqi-krr": "gqi-krr\npredictions",
    "gp-loo": "gp-loo\npredictions",
}


def values(summary: dict[str, Any], noise: str, size: float, key: str) -> list[float]:
    """One line's values along the targets: the exponent, or the share recovered."""
    runs = summary["Runs"][f"{noise}-e{size:g}"]["Summary"]
    return [runs[targets][key] for targets in TARGETS]


def figure(summary: dict[str, Any], out: Path) -> None:
    """The two panels, side by side, with the legend below them."""
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK2})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), facecolor=SURFACE)
    panels = {
        "Exponent": ("GEM's fitted exponent p", (0, 2.2), "truth, p = 1", 1.3),
        "Share": ("Share of the true field GEM recovers", (0, 1.12), "the whole field", 0.3),
    }
    for ax, (key, (title, ylim, reference, at)) in zip(axes, panels.items(), strict=True):
        ax.set_facecolor(SURFACE)
        ax.set_xlim(-0.45, len(TARGETS) - 0.55)
        ax.set_ylim(*ylim)
        ax.set_xticks(range(len(TARGETS)), [LABELS[targets] for targets in TARGETS])
        ax.tick_params(colors=INK2, length=0)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.set_title(title, loc="left", color=INK, fontsize=11, pad=10)
        ax.axhline(1.0, color=INK2, linewidth=1, linestyle=(0, (1, 2)), zorder=1)
        ax.text(at, 1.0 + 0.012 * ylim[1], reference, color=INK2, fontsize=8.5, va="bottom")
        limited = []
        for noise, (style, offset) in LINES.items():
            for size, color, dodge in zip(SIZES, COLORS, DODGE, strict=True):
                xs = [x + offset + dodge for x in range(len(TARGETS))]
                ys = values(summary, noise, size, key)
                ax.plot(xs, ys, color=color, linestyle=style, linewidth=2, zorder=2)
                hollow = noise == "noisy"
                for x, y in zip(xs, ys, strict=True):
                    at_limit = key == "Exponent" and y >= LIMIT - 1e-3
                    limited += [x] if at_limit else []
                    ax.plot(
                        x,
                        y,
                        marker="^" if at_limit else "o",
                        markersize=9 if at_limit else 8,
                        markerfacecolor=SURFACE if hollow else color,
                        markeredgecolor=color if hollow else SURFACE,
                        markeredgewidth=2 if hollow else 1.5,
                        linestyle="none",
                        zorder=3,
                    )
        if key == "Exponent":
            ax.axhline(LIMIT, color=AXIS, linewidth=1, zorder=1)
            ax.text(-0.43, LIMIT + 0.03, "upper limit of the fit", color=MUTED, fontsize=8.5)
            if limited:
                ax.text(
                    max(limited) + 0.05,
                    LIMIT - 0.14,
                    "at the limit:\nthe fit says\nnothing about p",
                    color=INK2,
                    fontsize=8.5,
                    va="top",
                )
        else:
            # Each size named at the end of its noise-free line, spread apart
            ends = sorted((values(summary, "noise-free", size, key)[-1], size) for size in SIZES)
            last = len(TARGETS) - 1 + LINES["noise-free"][1]
            for slot, (y, size) in enumerate(ends):
                ax.annotate(
                    f"{size:g} mm",
                    xy=(last + DODGE[SIZES.index(size)], y),
                    xytext=(last + 0.11, 0.06 + 0.1 * slot),
                    va="center",
                    color=INK2,
                    fontsize=8.5,
                    arrowprops={
                        "arrowstyle": "-",
                        "color": MUTED,
                        "linewidth": 0.6,
                        "shrinkA": 0,
                        "shrinkB": 5,
                    },
                )

    handles = [
        Line2D([], [], color=color, linewidth=2, label=f"{size:g} mm")
        for size, color in zip(SIZES, COLORS, strict=True)
    ]
    handles += [
        Line2D(
            [],
            [],
            color=INK2,
            linewidth=2,
            linestyle=LINES["noise-free"][0],
            marker="o",
            markersize=7,
            markerfacecolor=INK2,
            markeredgecolor=SURFACE,
            label="noise-free volumes",
        ),
        Line2D(
            [],
            [],
            color=INK2,
            linewidth=2,
            linestyle=LINES["noisy"][0],
            marker="o",
            markersize=7,
            markerfacecolor=SURFACE,
            markeredgecolor=INK2,
            markeredgewidth=2,
            label="noisy volumes",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=9,
        handlelength=3.5,
        title="Largest eddy-current displacement, and the volumes' noise",
        title_fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(out / "gem-accuracy.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def table(summary: dict[str, Any], out: Path) -> None:
    """The figure's values: a row for each target and condition, a column for each size."""
    sizes = [f"{size:g} mm" for size in SIZES]
    rows = [
        ["Targets", "Moving volumes", *(f"p, {s}" for s in sizes), *(f"share, {s}" for s in sizes)]
    ]
    for k, targets in enumerate(TARGETS):
        for noise in NOISE_LEVELS:
            row = [LABELS[targets].replace("\n", " "), noise]
            for key in ("Exponent", "Share"):
                row += [f"{values(summary, noise, size, key)[k]:.2f}" for size in SIZES]
            rows.append(row)
    with (out / "gem-accuracy.csv").open("w", newline="") as stream:
        csv.writer(stream).writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", type=Path, required=True, help="the experiment's folder")
    parser.add_argument("--out", type=Path, required=True, help="where the figure goes")
    args = parser.parse_args(argv)
    summary = json.loads((args.results / "accuracy.json").read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    figure(summary, args.out)
    table(summary, args.out)
    print(f"wrote {args.out / 'gem-accuracy.png'} and its table", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
