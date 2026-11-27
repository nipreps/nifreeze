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
"""The figures and tables of GEM's accuracy on the phantom, in ``docs/edd/gem-accuracy.rst``.

Reads ``<results>/accuracy.json``, which ``tools/registration/eddy_model_accuracy.py``
wrote. Each figure has a row for each pass of GEM, and each copy's eddy currents a line,
solid with the moving volumes noise-free and dashed with noise:

``gem-accuracy.png``
    along the targets (the noise-free originals, then the predictions of gqi-krr, then
    those of gp-loo): GEM's fitted exponent ``p``, against the truth's, and the share of
    the true field it recovers;
``gem-fits.png``
    along the four fits to gqi-krr's predictions (``p`` fitted; ``p`` held; ``p`` held,
    one ``C`` per shell; ``p`` held, ``C`` fitted at b=1000): the share of the true field
    recovered, and the median field error.

Their tables hold the same values, and ``gem-shells.csv`` each shell's share with one
``C`` per shell::

    python tools/registration/eddy_model_accuracy_figures.py --results <results> \\
        --out docs/_static/gem
"""

import argparse
import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from eddy_model_accuracy import NOISE_LEVELS, PASSES, SIZES, TARGETS

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
COLORS = ("#6da7ec", "#256abf", "#0d366b")
"""One step of a blue ramp for each size of the eddy currents, light to dark."""
LINES: dict[str, tuple[Any, float]] = {"noise-free": ("-", -0.11), "noisy": ((0, (4, 2.5)), 0.11)}
"""Each condition's line style, and its offset along the x-axis."""
DODGE = (-0.045, 0.0, 0.045)
"""Each size's offset along the x-axis, within its condition."""
LIMIT = 2.0
"""The upper limit of the fitted exponent (``EXPONENT_BOUNDS`` of ``gem.py``)."""
LABELS = {
    "original": "noise-free\noriginals",
    "gqi-krr": "gqi-krr\npredictions",
    "gp-loo": "gp-loo\npredictions",
}
FITS = ("gqi-krr", "held", "shells", "b1000")
"""The fits to gqi-krr's predictions, in the order of the second figure."""
FIT_LABELS = {
    "gqi-krr": "p fitted,\nall shells",
    "held": "p held at 1,\nall shells",
    "shells": "p held at 1,\none C per shell",
    "b1000": "p held at 1,\nC from b=1000",
}
ROWS = {1: "One pass", 2: "Two passes", 3: "Three passes"}


def values(
    summary: dict[str, Any], noise: str, size: float, key: str, fits: Sequence[str], number: int
) -> list[float]:
    """One line's values along ``fits``, in pass ``number``."""
    runs = summary["Runs"][f"{noise}-e{size:g}"]["Summary"]
    return [runs[fit][number - 1][key] for fit in fits]


def _panel(ax, summary, fits, labels, key, spec, number, named) -> None:
    """One panel: a line per copy and condition along ``fits``."""
    title, ylim, line, reference, at = spec
    ax.set_facecolor(SURFACE)
    ax.set_xlim(-0.45, len(fits) - 0.55)
    ax.set_ylim(*ylim)
    ax.set_xticks(range(len(fits)), [labels[fit] for fit in fits])
    ax.tick_params(colors=INK2, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_title(f"{ROWS[number]}: {title}", loc="left", color=INK, fontsize=11, pad=10)
    if line is not None:
        ax.axhline(line, color=INK2, linewidth=1, linestyle=(0, (1, 2)), zorder=1)
    if reference is not None:
        ax.text(at, 1.0 + 0.012 * ylim[1], reference, color=INK2, fontsize=8.5, va="bottom")
    limited = []
    for noise, (style, offset) in LINES.items():
        for size, color, dodge in zip(SIZES, COLORS, DODGE, strict=True):
            xs = [x + offset + dodge for x in range(len(fits))]
            ys = values(summary, noise, size, key, fits, number)
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
        _limit(ax, limited)
    if named:
        _name_sizes(ax, summary, fits, key, number)


def _limit(ax, limited: list[float]) -> None:
    """The upper limit of the fitted exponent, and a note on the fits that stopped there."""
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


def _name_sizes(ax, summary, fits, key, number) -> None:
    """Each size named at the end of its noise-free line, spread apart."""
    ends = sorted(
        (values(summary, "noise-free", size, key, fits, number)[-1], size) for size in SIZES
    )
    last = len(fits) - 1 + LINES["noise-free"][1]
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


def _legend(fig) -> None:
    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], color=color, linewidth=2, label=f"{size:g} mm")
        for size, color in zip(SIZES, COLORS, strict=True)
    ]
    for noise, label in (("noise-free", "noise-free volumes"), ("noisy", "noisy volumes")):
        hollow = noise == "noisy"
        handles.append(
            Line2D(
                [],
                [],
                color=INK2,
                linewidth=2,
                linestyle=LINES[noise][0],
                marker="o",
                markersize=7,
                markerfacecolor=SURFACE if hollow else INK2,
                markeredgecolor=INK2 if hollow else SURFACE,
                markeredgewidth=2 if hollow else 1,
                label=label,
            )
        )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=9,
        handlelength=3.5,
        title="Largest eddy-current displacement, and the volumes' noise",
        title_fontsize=9,
        bbox_to_anchor=(0.5, 0.005),
    )


def figure(
    summary: dict[str, Any], out: Path, name: str, fits, labels, panels, named: str | None
) -> None:
    """A row of panels for each pass, with the legend below them."""
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK2})
    height = 4.2 * PASSES
    fig, axes = plt.subplots(PASSES, len(panels), figsize=(10.5, height), facecolor=SURFACE)
    for row, number in zip(axes, range(1, PASSES + 1), strict=True):
        for ax, (key, spec) in zip(row, panels.items(), strict=True):
            _panel(ax, summary, fits, labels, key, spec, number, named=key == named)
    _legend(fig)
    fig.tight_layout(rect=(0, 0.8 / height, 1, 1), h_pad=2.5)  # 0.8 in for the legend
    fig.savefig(out / name, dpi=200, facecolor=SURFACE)
    plt.close(fig)


ACCURACY = {
    "Exponent": ("GEM's fitted exponent p (truth: 1)", (0, 2.2), 1.0, None, 0.0),
    "Share": ("share of the true field recovered", (0, 1.12), 1.0, "the whole field", 0.3),
}
"""Each panel: its title, y-limits, reference line, the line's label and its x."""
FIT_PANELS = {
    "Share": ("share of the true field recovered", (0, 1.12), 1.0, "the whole field", 0.3),
    "FieldErrorMm": ("median field error (mm)", (0, 1.6), None, None, 0.0),
}


def _write(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", newline="") as stream:
        csv.writer(stream).writerows(rows)


def _row(summary, fits, k, noise, number, keys) -> list[str]:
    """One row of a table: a fit's values in a condition and pass, for each size."""
    row = []
    for key, digits in keys:
        row += [f"{values(summary, noise, s, key, fits, number)[k]:.{digits}f}" for s in SIZES]
    return row


def tables(summary: dict[str, Any], out: Path) -> None:
    """The figures' values, and each shell's share with one ``C`` per shell."""
    sizes = [f"{size:g} mm" for size in SIZES]
    passes = range(1, PASSES + 1)
    header = ["Passes", "Targets", "Moving volumes"]
    header += [f"p, {s}" for s in sizes] + [f"share, {s}" for s in sizes]
    rows = [
        [str(number), LABELS[targets].replace("\n", " "), noise]
        + _row(summary, TARGETS, k, noise, number, (("Exponent", 2), ("Share", 2)))
        for number in passes
        for k, targets in enumerate(TARGETS)
        for noise in NOISE_LEVELS
    ]
    _write(out / "gem-accuracy.csv", [header, *rows])

    header = ["Passes", "Fit", "Moving volumes"]
    header += [f"share, {s}" for s in sizes] + [f"error (mm), {s}" for s in sizes]
    rows = [
        [str(number), FIT_LABELS[fit].replace("\n", " "), noise]
        + _row(summary, FITS, k, noise, number, (("Share", 2), ("FieldErrorMm", 3)))
        for number in passes
        for k, fit in enumerate(FITS)
        for noise in NOISE_LEVELS
    ]
    _write(out / "gem-fits.csv", [header, *rows])

    by_shell = [
        (number, noise, size, summary["Runs"][f"{noise}-e{size:g}"]["Summary"]["shells"])
        for number in passes
        for noise in NOISE_LEVELS
        for size in SIZES
    ]
    shells = list(by_shell[0][3][0]["ShareByShell"])
    header = ["Passes", "Moving volumes", "Eddy currents", *(f"b={b}" for b in shells)]
    rows = [
        [str(number), noise, f"{size:g} mm"]
        + [f"{fitted[number - 1]['ShareByShell'][b]:.2f}" for b in shells]
        for number, noise, size, fitted in by_shell
    ]
    _write(out / "gem-shells.csv", [header, *rows])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", type=Path, required=True, help="the experiment's folder")
    parser.add_argument("--out", type=Path, required=True, help="where the figures go")
    args = parser.parse_args(argv)
    summary = json.loads((args.results / "accuracy.json").read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    figure(summary, args.out, "gem-accuracy.png", TARGETS, LABELS, ACCURACY, "Share")
    figure(summary, args.out, "gem-fits.png", FITS, FIT_LABELS, FIT_PANELS, None)
    tables(summary, args.out)
    print(f"wrote the figures and tables in {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
