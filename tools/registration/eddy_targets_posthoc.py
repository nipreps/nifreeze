# /// script
# requires-python = ">=3.13"
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
"""Post hoc probes of the test on the eddy-current pass's targets (#723), not registered.

Run after Test 1's results, they ask why the instant update hurts the pass:

weights
    The smoother weights of each model on the pass's starting view (the copy realigned
    by the cascade's head motion): for each volume, the share of the absolute weight on
    its same-direction volumes in the other shells, and whether they are its two
    largest weights. Beside it, Spearman's rho of the pass's change in error (over
    ``cascade-corrected``, positive where the instant update made it worse) against the error
    the pass left on those registered before the volume (weighted by their absolute
    weights), and, as a control, on those registered after it.
drift
    Each volume's residual ``W (E M)^-1`` (scanner frame), averaged over each third of
    the order of registration, by shell: the mean residual's strain (the Frobenius norm
    of its symmetric polar factor minus the identity), its eigenvalues and largest axis,
    its component along phase encoding, its rigid part (the shift of the brain's centre
    and the rotation), the spread of each volume's residual around it, and its mean
    displacement within the brain.

Reads ``<results>/<protocol>.{tsv,npz}``, which ``tools/registration/eddy_targets.py``
wrote, and the phantom dataset. Writes ``eddy-targets-posthoc.json``, the report's
tables and, with ``--figures``, its figure::

    uv run tools/registration/eddy_targets_posthoc.py --dataset <dataset> \\
        --results <results> --out docs/_static/registration/eddy-targets --figures
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import nibabel as nb
import numpy as np
from eddy_targets import NOISE, _choose, _directions
from eddy_targets_figures import INK, MUTED, RULE, SHELL_COLORS, SHELL_MARKERS, _style, _write
from scipy import stats
from scipy.linalg import polar

from nifreeze.analysis.phantoms import load_phantom
from nifreeze.model.base import ModelFactory

MODELS = ("gp-loo", "gqi-krr")
ARMS = ("cascade-corrected", "pass")
"""The pass, and its targets at the start without the instant update."""
THIRDS = 3


def _column(rows: list[dict[str, str]], arm: str, model: str, key: str) -> dict[int, float]:
    return {
        int(row["index"]): float(row[key])
        for row in rows
        if row["arm"] == arm and row["model"] == model
    }


def weights(
    view: Any, rows: list[dict[str, str]], shells: np.ndarray, directions: np.ndarray
) -> dict[str, Any]:
    """The share of the smoother's weight on the same direction, and the loss it explains."""
    n = len(shells)
    siblings = [
        np.flatnonzero((directions == directions[i]) & (np.arange(n) != i)) for i in range(n)
    ]
    result: dict[str, Any] = {}
    for model in MODELS:
        magnitude = np.abs(
            np.asarray(ModelFactory.init(model, dataset=view).smoother, dtype=float)
        )
        np.fill_diagonal(magnitude, 0.0)
        passed = _column(rows, "pass", model, "whole_frame_mean")
        start = _column(rows, "cascade-corrected", model, "whole_frame_mean")
        rank = _column(rows, "pass", model, "rank")
        gain = np.array([passed[i] - start[i] for i in range(n)])
        error = np.array([passed[i] for i in range(n)])
        ranks = np.array([rank[i] for i in range(n)])
        share = np.array([magnitude[i, siblings[i]].sum() / magnitude[i].sum() for i in range(n)])
        largest = np.array(
            [set(np.argsort(magnitude[i])[-2:]) == set(siblings[i].tolist()) for i in range(n)]
        )
        inherited = {
            side: np.array(
                [
                    sum(
                        magnitude[i, j] * error[j]
                        for j in siblings[i]
                        if (ranks[j] < ranks[i]) == (side == "before")
                    )
                    for i in range(n)
                ]
            )
            for side in ("before", "after")
        }
        result[model] = {
            str(int(shell)): {
                "share": float(np.median(share[shells == shell])),
                "largest": float(largest[shells == shell].mean()),
                **{
                    f"rho_{side}": float(
                        stats.spearmanr(values[shells == shell], gain[shells == shell]).statistic
                    )
                    for side, values in inherited.items()
                },
                "rho_order": float(
                    stats.spearmanr(ranks[shells == shell], gain[shells == shell]).statistic
                ),
            }
            for shell in np.unique(shells)
        }
    return result


def _strain(matrix: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """The strain's size, its eigenvalues and eigenvectors (ascending)."""
    _, symmetric = polar(matrix[:3, :3])
    strain = symmetric - np.eye(3)
    values, vectors = np.linalg.eigh(strain)
    return float(np.linalg.norm(strain)), values, vectors


def _displacement(matrix: np.ndarray, points: np.ndarray) -> float:
    """The mean displacement (mm) of the points under a matrix."""
    moved = points @ matrix[:3, :3].T + matrix[:3, 3]
    return float(np.linalg.norm(moved - points, axis=1).mean())


def drift(
    saved: Any, points: np.ndarray, shells: np.ndarray, direction: np.ndarray
) -> dict[str, Any]:
    """The mean residual of each third of the order, by model, arm and shell."""
    whole = saved["truth/whole"]
    order = saved["order"]
    rank = np.empty(len(order), int)
    rank[order] = np.arange(len(order))
    result: dict[str, Any] = {}
    for model in MODELS:
        for arm in ARMS:
            indices = saved[f"{arm}/{model}/indices"]
            residual = saved[f"{arm}/{model}"] @ np.linalg.inv(whole[indices])
            third = rank[indices] * THIRDS // len(order)
            for shell in ("all", *map(str, map(int, np.unique(shells)))):
                keep = (
                    np.ones(len(indices), bool)
                    if shell == "all"
                    else shells[indices] == float(shell)
                )
                cells = []
                for k in range(THIRDS):
                    members = residual[keep & (third == k)]
                    mean = members.mean(axis=0)
                    size, values, vectors = _strain(mean)
                    axis = vectors[:, np.argmax(np.abs(values))]
                    rotation, symmetric = polar(mean[:3, :3])
                    # The rigid part: the shift of the brain's centre, and the rotation
                    centre = points.mean(axis=0)
                    shift = mean[:3, :3] @ centre + mean[:3, 3] - centre
                    angle = np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1.0, 1.0))
                    # The spread: how far each volume's residual lies from the third's mean
                    spread = np.median(
                        [_displacement(r @ np.linalg.inv(mean), points) for r in members]
                    )
                    cells.append(
                        {
                            "strain": size,
                            "eigenvalues": values.tolist(),
                            "axis": axis.tolist(),
                            "axis_cos_pe": float(abs(axis @ direction)),
                            "along_pe": float(direction @ (symmetric - np.eye(3)) @ direction),
                            "displacement": _displacement(mean, points),
                            "shift": float(np.linalg.norm(shift)),
                            "rotation": float(np.degrees(angle)),
                            "spread": float(spread),
                        }
                    )
                result[f"{model}/{arm}/{shell}"] = cells
    return result


def tables(result: dict[str, Any], out: Path) -> None:
    """The report's tables of the probes."""
    _write(
        out / "eddy-targets-weights.csv",
        [
            "Model",
            "Shell",
            "Weight share",
            "Largest two",
            "Rho, before",
            "Rho, after",
            "Rho, order",
        ],
        [
            [
                model,
                f"b={shell}",
                f"{v['share']:.3f}",
                f"{100 * v['largest']:.0f}%",
                f"{v['rho_before']:+.2f}",
                f"{v['rho_after']:+.2f}",
                f"{v['rho_order']:+.2f}",
            ]
            for model, shells in result["weights"].items()
            for shell, v in shells.items()
        ],
    )
    _write(
        out / "eddy-targets-drift.csv",
        [
            "Model",
            "Arm",
            "Shells",
            "Strain",
            "Shift (mm)",
            "Rotation (degrees)",
            "Spread (mm)",
            "Mean displacement (mm)",
        ],
        [
            [
                *key.split("/")[:2],
                key.split("/")[2] if key.endswith("all") else f"b={key.split('/')[2]}",
                *[
                    " / ".join(f"{c[field]:{spec}}" for c in cells)
                    for field, spec in (
                        ("strain", ".4f"),
                        ("shift", ".2f"),
                        ("rotation", ".2f"),
                        ("spread", ".3f"),
                        ("displacement", ".3f"),
                    )
                ],
            ]
            for key, cells in result["drift"].items()
        ],
    )
    _write(
        out / "eddy-targets-strain-axes.csv",
        [
            "Model",
            "Arm",
            "Third",
            "Eigenvalues",
            "Largest axis",
            "Alignment with PE (absolute cosine)",
            "Along PE",
        ],
        [
            [
                *key.split("/")[:2],
                k + 1,
                ", ".join(f"{v:+.4f}" for v in cell["eigenvalues"]),
                "(" + ", ".join(f"{round(v, 2) + 0.0:+.2f}" for v in cell["axis"]) + ")",
                f"{cell['axis_cos_pe']:.2f}",
                f"{cell['along_pe']:+.4f}",
            ]
            for key, cells in result["drift"].items()
            if key.endswith("/all")
            for k, cell in enumerate(cells)
        ],
    )


def figure(result: dict[str, Any], out: Path) -> None:
    """The strain of the mean residual by third of the order, for each arm, model and shell."""
    plt = _style()
    shells = sorted({key.split("/")[2] for key in result["drift"]} - {"all"}, key=int)
    fig, axes = plt.subplots(1, len(MODELS), figsize=(7.5, 3.4), layout="constrained", sharey=True)
    thirds = np.arange(1, THIRDS + 1)
    for ax, model in zip(axes, MODELS, strict=True):
        for k, shell in enumerate(shells):
            for arm, style in zip(ARMS, (":", "-"), strict=True):
                cells = result["drift"][f"{model}/{arm}/{shell}"]
                ax.plot(
                    thirds,
                    [c["strain"] for c in cells],
                    linestyle=style,
                    color=SHELL_COLORS[k],
                    marker=SHELL_MARKERS[k],
                    markersize=6,
                    lw=2,
                )
        ax.set_xticks(thirds, ["First", "Second", "Last"])
        ax.set_xlabel("Third of the order of registration")
        ax.set_title(model, color=INK, fontsize=9, loc="left")
        ax.grid(axis="y", color=RULE, linewidth=0.6)
    axes[0].set_ylabel("Strain of the mean residual")
    handles = [
        *[
            plt.Line2D(
                [], [], color=SHELL_COLORS[k], marker=SHELL_MARKERS[k], lw=2, label=f"b={s}"
            )
            for k, s in enumerate(shells)
        ],
        plt.Line2D([], [], color=MUTED, lw=2, linestyle="-", label="pass"),
        plt.Line2D([], [], color=MUTED, lw=2, linestyle=":", label="without the update"),
    ]
    fig.legend(
        handles=handles, loc="outside upper center", ncols=len(handles), frameon=False, fontsize=8
    )
    fig.savefig(out / "eddy-targets-drift.png", dpi=200, facecolor="white")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", type=Path, required=True, help="phantom dataset root")
    parser.add_argument("--results", type=Path, required=True, help="the results folder")
    parser.add_argument("--protocol", default="sherbrooke")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--figures", action="store_true", help="also the report's figure")
    args = parser.parse_args(argv)

    reference, cells = load_phantom(args.dataset, args.protocol, noise=NOISE)
    perturbation, load = _choose(cells, len(reference))
    saved = np.load(args.results / f"{args.protocol}.npz")
    with (args.results / f"{args.protocol}.tsv").open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    shells = np.round(reference.gradients[:, -1], -2)
    mask = np.asanyarray(reference.brainmask, dtype=bool)
    points = nb.affines.apply_affine(reference.affine, np.argwhere(mask))

    data = load()
    data.motion = saved["cascade/gp-loo"].copy()
    view = data.realigned()
    result = {
        "weights": weights(view, rows, shells, _directions(reference.gradients)),
        "drift": drift(saved, points, shells, perturbation.distortion.direction),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "eddy-targets-posthoc.json").write_text(json.dumps(result, indent=2) + "\n")
    tables(result, args.out)
    if args.figures:
        figure(result, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
