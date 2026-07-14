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

Renders an affine-aware montage of axial cuts (via :mod:`nilearn.plotting`, which
handles image orientation) with four rows for a held-out (or held-in) volume:

1. **observed** — the reference volume,
2. **predicted** — the model prediction,
3. **checkerboard** — observed/predicted interleaved in blocks (the predicted
   volume is intensity-matched to the observed one by linear regression so the
   two are on a comparable range), which makes mismatches pop as block seams,
4. **difference** — observed minus the matched prediction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

N_CUTS = 6
"""Number of axial (z) cuts per row."""
CHECKER_BLOCK = 12
"""Checkerboard block size in voxels."""
DPI = 150
"""Output resolution for rendered figures."""


def _match_intensity(
    predicted: np.ndarray, observed: np.ndarray, mask: np.ndarray | None
) -> np.ndarray:
    """Linearly regress ``predicted`` onto ``observed`` (regression to the mean).

    Returns ``a * predicted + b`` with ``(a, b)`` the least-squares fit within
    ``mask``, so the two volumes share a comparable intensity range.
    """
    sel = mask if mask is not None else np.ones(observed.shape, dtype=bool)
    x = predicted[sel].astype(float).ravel()
    y = observed[sel].astype(float).ravel()
    if x.size < 2 or np.allclose(x, x[0]):
        return predicted.astype(float)
    a, b = np.linalg.lstsq(np.vstack([x, np.ones_like(x)]).T, y, rcond=None)[0]
    return a * predicted.astype(float) + b


def _checkerboard(a: np.ndarray, b: np.ndarray, block: int = CHECKER_BLOCK) -> np.ndarray:
    """Interleave volumes ``a`` and ``b`` in a 3D checkerboard of ``block`` voxels."""
    idx = np.indices(a.shape)
    pattern = ((idx[0] // block) + (idx[1] // block) + (idx[2] // block)) % 2
    return np.where(pattern.astype(bool), a, b)


def select_cut_coords(
    mask: np.ndarray | None, affine: np.ndarray, n_cuts: int, *, min_frac: float = 0.5
) -> list[float] | None:
    """Choose fixed axial cut positions over the high-mass part of the brain mask.

    Deriving the cuts from the **mask** (not each volume's signal) keeps the
    slices identical across every panel on a page. Thresholding on the per-slice
    voxel count keeps the cuts on the substantive cerebrum and drops thin
    cerebellum/edge slices that carry little information. Returns world-space z
    coordinates, or ``None`` if no mask is available.
    """
    if mask is None:
        return None
    mask = np.asarray(mask, dtype=bool)
    counts = mask.sum(axis=(0, 1))  # voxel count per axial (k) slice
    if counts.max() == 0:
        return None
    valid = np.where(counts >= min_frac * counts.max())[0]
    if valid.size == 0:
        valid = np.where(counts > 0)[0]
    lo, hi = int(valid.min()), int(valid.max())
    xs, ys, _ = np.where(mask)
    cx, cy = int(round(xs.mean())), int(round(ys.mean()))
    ks = np.linspace(lo, hi, n_cuts + 2)[1:-1]  # interior, evenly spaced
    coords = [float((affine @ np.array([cx, cy, k, 1.0]))[2]) for k in ks]
    return sorted(coords)


def save_slice_panel(
    observed: np.ndarray,
    predicted: np.ndarray,
    path: str | Path,
    *,
    affine: np.ndarray,
    mask: np.ndarray | None = None,
    title: str | None = None,
    n_cuts: int = N_CUTS,
) -> Path:
    """Render the four-row observed/predicted/checkerboard/difference montage.

    Parameters
    ----------
    observed : :obj:`~numpy.ndarray`
        The reference (held-out) 3D volume.
    predicted : :obj:`~numpy.ndarray`
        The model-predicted 3D volume (same shape as ``observed``).
    path : :obj:`str` or :obj:`~pathlib.Path`
        Output image path.
    affine : :obj:`~numpy.ndarray`
        The voxel-to-world affine (so cuts are correctly oriented).
    mask : :obj:`~numpy.ndarray`, optional
        Brain mask used for intensity matching.
    title : :obj:`str`, optional
        Figure title.
    n_cuts : :obj:`int`, optional
        Number of axial cuts per row.

    Returns
    -------
    :obj:`~pathlib.Path`
        The written image path.

    """
    import matplotlib as mpl

    mpl.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import nibabel as nb
    from nilearn import plotting

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    matched = _match_intensity(predicted, observed, mask)
    checker = _checkerboard(observed, matched)
    difference = observed - matched

    finite = observed[np.isfinite(observed)]
    vmax = float(np.percentile(finite, 99)) if finite.size else 1.0
    dvals = np.abs(difference[np.isfinite(difference)])
    dmax = float(np.percentile(dvals, 99)) if dvals.size else 1.0
    dmax = dmax or 1.0

    rows = [
        ("observed", observed, {"cmap": "gray", "vmin": 0.0, "vmax": vmax}),
        ("predicted", matched, {"cmap": "gray", "vmin": 0.0, "vmax": vmax}),
        ("checkerboard", checker, {"cmap": "gray", "vmin": 0.0, "vmax": vmax}),
        ("difference", difference, {"cmap": "RdBu_r", "vmin": -dmax, "vmax": dmax}),
    ]

    # Choose axial cut positions from the brain MASK (not each volume's signal),
    # so the exact same world-space slices are used for every row *and* across
    # every panel on the page, focused on the high-mass cerebrum.
    cut_coords: list | int | None = select_cut_coords(mask, affine, n_cuts)
    if cut_coords is None:
        observed_img = nb.Nifti1Image(observed.astype(np.float32), affine)
        try:
            cut_coords = list(plotting.find_cut_slices(observed_img, direction="z", n_cuts=n_cuts))
        except Exception:  # pragma: no cover - degenerate volumes
            cut_coords = n_cuts

    fig, axes = plt.subplots(len(rows), 1, figsize=(2.2 * n_cuts, 2.4 * len(rows)))
    for ax, (label, data, kw) in zip(axes, rows, strict=True):
        img = nb.Nifti1Image(np.asarray(data, dtype=np.float32), affine)
        plotting.plot_img(
            img,
            display_mode="z",
            cut_coords=cut_coords,
            axes=ax,
            black_bg=True,
            annotate=False,
            colorbar=False,
            **kw,
        )
        ax.set_title(label, color="black", fontsize=11)

    if title:
        fig.suptitle(title, fontsize=12)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _find_kriging(kernel):
    """Walk a (possibly composite) kernel to find the angular kriging component."""
    from nifreeze.model.gpr import ExponentialKriging, SphericalKriging

    if isinstance(kernel, (SphericalKriging, ExponentialKriging)):
        return kernel
    for attr in ("orientation_kernel", "radial_kernel", "k1", "k2"):
        sub = getattr(kernel, attr, None)
        if sub is not None and (found := _find_kriging(sub)) is not None:
            return found
    return None


def save_covariance_plot(gpr, path: str | Path, *, title: str | None = None) -> Path:
    """Plot the fitted **angular covariance** function (Andersson 2015, Fig. 3).

    Extracts the (spherical or exponential) kriging component of a fitted GP and
    plots its covariance against angular distance — the covariance structure the
    GP imposes across diffusion orientations.

    Parameters
    ----------
    gpr : fitted GP regressor
        A fitted :obj:`~sklearn.gaussian_process.GaussianProcessRegressor`
        (exposing ``kernel_``) or any object exposing a ``kernel``/``kernel_``.
    path : :obj:`str` or :obj:`~pathlib.Path`
        Output image path.
    title : :obj:`str`, optional
        Figure title.

    """
    import matplotlib as mpl

    mpl.use("Agg", force=True)
    import matplotlib.pyplot as plt

    from nifreeze.model.gpr import (
        ExponentialKriging,
        exponential_covariance,
        spherical_covariance,
    )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    kernel = getattr(gpr, "kernel_", None)
    if kernel is None:
        kernel = getattr(gpr, "kernel", None)
    kriging = _find_kriging(kernel) if kernel is not None else None

    theta = np.linspace(0.0, np.pi / 2, 400)
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    if kriging is None:
        ax.text(0.5, 0.5, "no angular kernel found", ha="center", va="center")
        ax.set_axis_off()
    else:
        a = float(kriging.beta_a)
        lam = float(kriging.beta_l)
        if isinstance(kriging, ExponentialKriging):
            cov = lam * exponential_covariance(theta, a)
            name = f"Exponential (a={a:.2f}, λ={lam:.2f})"
        else:
            cov = lam * spherical_covariance(theta, a)
            name = f"Spherical (a={a:.2f}, λ={lam:.2f})"
        ax.plot(theta, cov, "k")
        ax.set_xlabel("Angular distance θ (rad)")
        ax.set_ylabel("Covariance")
        ax.set_title(name, fontsize=10)
        ax.set_xticks([0.0, np.pi / 8, np.pi / 4, 3 * np.pi / 8, np.pi / 2])
        ax.set_xticklabels(["0", "π/8", "π/4", "3π/8", "π/2"])

    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, facecolor="white")
    plt.close(fig)
    return path
