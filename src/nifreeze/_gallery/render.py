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
"""Figure composition for the prediction gallery.

Renders a compact *observed vs. predicted vs. difference* panel for a held-out
(or held-in, in single-fit mode) volume. Kept dependency-light (matplotlib
``Agg``) so it runs headless in CI; the docs notebooks may compose richer
visuals on top (e.g. ``nireports.plot_dwi``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _mid_slice(volume: np.ndarray, axis: int = 2) -> np.ndarray:
    """Return the middle slice of a 3D volume along ``axis`` (as a 2D array)."""
    idx = volume.shape[axis] // 2
    return np.take(volume, idx, axis=axis)


def save_slice_panel(
    observed: np.ndarray,
    predicted: np.ndarray,
    path: str | Path,
    *,
    title: str | None = None,
    axis: int = 2,
) -> Path:
    """Render observed/predicted/difference middle slices to an image file.

    Parameters
    ----------
    observed : :obj:`~numpy.ndarray`
        The reference (held-out) 3D volume.
    predicted : :obj:`~numpy.ndarray`
        The model-predicted 3D volume (same shape as ``observed``).
    path : :obj:`str` or :obj:`~pathlib.Path`
        Output image path (extension sets the format, e.g. ``.png``).
    title : :obj:`str`, optional
        Figure title.
    axis : :obj:`int`, optional
        Slicing axis for the displayed 2D section.

    Returns
    -------
    :obj:`~pathlib.Path`
        The written image path.

    """
    import matplotlib as mpl

    mpl.use("Agg", force=True)
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    obs = np.rot90(_mid_slice(np.asarray(observed, dtype=float), axis))
    pred = np.rot90(_mid_slice(np.asarray(predicted, dtype=float), axis))
    diff = pred - obs

    # Shared intensity window for observed/predicted (robust to outliers).
    finite = np.concatenate([obs[np.isfinite(obs)], pred[np.isfinite(pred)]])
    vmax = float(np.percentile(finite, 99)) if finite.size else 1.0
    vmin = 0.0
    dmax = float(np.percentile(np.abs(diff[np.isfinite(diff)]), 99)) if diff.size else 1.0
    dmax = dmax or 1.0

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    for ax, data, label, kw in (
        (axes[0], obs, "observed", {"cmap": "gray", "vmin": vmin, "vmax": vmax}),
        (axes[1], pred, "predicted", {"cmap": "gray", "vmin": vmin, "vmax": vmax}),
        (axes[2], diff, "difference", {"cmap": "RdBu_r", "vmin": -dmax, "vmax": dmax}),
    ):
        ax.imshow(data, origin="lower", **kw)
        ax.set_title(label, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=90, bbox_inches="tight")
    plt.close(fig)
    return path
