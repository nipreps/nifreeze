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
"""Draw the figures of ``docs/phantoms.rst`` from a synthetic dataset.

Usage::

    python tools/phantoms/figures.py --dataset <dataset> --protocol hbn \
        --real <real_dwi.nii.gz> --out docs/_static/phantoms

``<dataset>`` holds the reference (``reference``) and the perturbed copies
(``cells``) that ``phantom.py`` writes; ``--real`` is a real dataset acquired with
the same protocol (its ``.bval`` next to it), drawn alongside for comparison.
All images are shown in RAS orientation, with anterior up in axial views.
"""

import argparse
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nb
import numpy as np

from nifreeze.utils.ndimage import load_api

SUBJECT = "60501"


def load(path: str | Path) -> nb.Nifti1Image:
    return load_api(path, nb.Nifti1Image)


def canonical(image: str | Path | nb.Nifti1Image) -> np.ndarray:
    """The image's data in RAS orientation."""
    if not isinstance(image, nb.Nifti1Image):
        image = load(image)
    return np.asarray(nb.as_closest_canonical(image).dataobj, dtype=np.float32)


def views(volume: np.ndarray, center: np.ndarray) -> list[np.ndarray]:
    """Axial, coronal and sagittal slices through ``center``, anterior or superior up."""
    x, y, z = center
    return [np.rot90(volume[:, :, z]), np.rot90(volume[:, y, :]), np.rot90(volume[x, :, :])]


def real_versus_synthetic(folder: Path, protocol: str, real: Path | None, out: Path) -> Path:
    """The real and the synthetic datasets, at b=0 and at the first volume of each shell."""
    stem = folder / f"sub-{SUBJECT}_acq-{protocol}"
    clean = canonical(f"{stem}_rec-clean_dwi.nii.gz")
    noisy = canonical(f"{stem}_rec-noisy_dwi.nii.gz")
    mask = canonical(f"{stem}_desc-brain_mask.nii.gz") > 0
    bvals = np.loadtxt(f"{stem}_dwi.b")[:, 3]
    picks = {"b=0": 0} | {
        f"b={b:g}": int(np.flatnonzero(bvals == b)[0]) for b in np.unique(bvals) if b > 0
    }
    sidecar = json.loads(Path(f"{stem}_dwi.json").read_text())
    rows = [
        ("Synthetic, noise-free", clean, mask),
        (f"Synthetic, noisy (SNR {sidecar['Noise']['SNR']:g})", noisy, mask),
    ]
    if real is not None:
        from dipy.segment.mask import median_otsu

        data = canonical(real)
        _, brain = median_otsu(data, vol_idx=[0], median_radius=3, numpass=2)
        rows.insert(0, ("Real data", data, brain))
    fig, axes = plt.subplots(
        len(rows), 3 * len(picks), figsize=(2 * 3 * len(picks), 2.3 * len(rows))
    )
    for r, (title, data, brain) in enumerate(rows):
        center = np.round(np.argwhere(brain).mean(axis=0)).astype(int)
        for p, (label, index) in enumerate(picks.items()):
            top = np.percentile(data[..., index][brain], 99.5)
            for q, image in enumerate(views(data[..., index], center)):
                ax = axes[r, 3 * p + q]
                ax.imshow(image, cmap="gray", vmin=0, vmax=top, interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                if r == 0:
                    ax.set_title(f"{label}, " + ("axial", "coronal", "sagittal")[q], fontsize=9)
        axes[r, 0].set_ylabel(title, fontsize=9)
    fig.tight_layout()
    target = out / f"{protocol}_real_vs_synthetic.png"
    fig.savefig(target, dpi=100)
    plt.close(fig)
    return target


def truth_maps(folder: Path, protocol: str, out: Path) -> Path:
    """The ground truth written with the reference, on an axial slice."""
    stem = folder / f"sub-{SUBJECT}_acq-{protocol}"
    maps = {
        key: canonical(f"{stem}_desc-{key}_truth.nii.gz")
        for key in ("fa", "md", "mk", "icvf", "odi", "isovf")
    }
    peaks = canonical(f"{stem}_desc-peaks_truth.nii.gz")
    first = peaks[..., :3] if peaks.ndim == 4 else peaks[..., 0, :]
    norms = np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1e-6)
    color = np.clip(np.abs(first) / norms * np.clip(maps["fa"], 0, 1)[..., None], 0, 1)
    mask = canonical(f"{stem}_desc-brain_mask.nii.gz") > 0
    z = int(np.round(np.argwhere(mask)[:, 2].mean()))
    items = [
        ("Fractional anisotropy", maps["fa"], "gray", (0, 1)),
        ("Direction of the first fiber, by FA", color, None, None),
        ("Mean diffusivity (mm²/s)", maps["md"], "magma", (0, 3e-3)),
        ("Mean kurtosis", maps["mk"], "viridis", (0, 1.5)),
        ("Intra-axonal fraction", maps["icvf"], "viridis", (0, 1)),
        ("Orientation dispersion", maps["odi"], "viridis", (0, 1)),
        ("Free-water fraction", maps["isovf"], "Blues", (0, 1)),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for ax, (title, volume, cmap, limits) in zip(axes.ravel(), items, strict=False):
        image = np.rot90(volume[:, :, z])
        if cmap is None:
            ax.imshow(image, interpolation="nearest")
        else:
            assert limits is not None
            shown = ax.imshow(
                image, cmap=cmap, vmin=limits[0], vmax=limits[1], interpolation="nearest"
            )
            fig.colorbar(shown, ax=ax, fraction=0.046)
        ax.set_title(title, fontsize=9)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    axes.ravel()[-1].set_visible(False)
    fig.tight_layout()
    target = out / f"{protocol}_truth.png"
    fig.savefig(target, dpi=100)
    plt.close(fig)
    return target


def displacement(image: nb.Nifti1Image, coefficients: np.ndarray) -> np.ndarray:
    """The eddy-current displacement (mm, along phase encoding) of one volume, in RAS."""
    shape = tuple(int(n) for n in image.shape[:3])
    offsets = np.indices(shape).reshape(3, -1).T - np.asarray(shape) / 2
    field = (offsets @ coefficients[1:] + coefficients[0]).reshape(shape)
    return canonical(nb.Nifti1Image(field.astype(np.float32), image.affine))


def eddy_currents(folder: Path, cells: Path, protocol: str, out: Path) -> Path:
    """Two volumes, one stretched and one sheared along phase encoding, at each magnitude."""
    stem = folder / f"sub-{SUBJECT}_acq-{protocol}"
    clean_image = load(f"{stem}_rec-clean_dwi.nii.gz")
    clean = canonical(clean_image)
    mask = canonical(f"{stem}_desc-brain_mask.nii.gz") > 0
    sets = sorted(
        cells.glob(f"sub-{SUBJECT}_acq-{protocol}_desc-e*_dwi.json"),
        key=lambda p: float(json.loads(p.read_text())["Eddy"]["Magnitude"]),
    )
    sidecars = [json.loads(p.read_text()) for p in sets]
    strongest = sidecars[-1]
    coefficients = np.asarray(strongest["Eddy"]["Distortion"]["Coefficients"])
    # The volume most stretched (phase-encoding term) and the most sheared (readout term)
    chosen = {
        "stretch": int(np.argmax(np.abs(coefficients[:, 2]))),
        "shear": int(np.argmax(np.abs(coefficients[:, 1]))),
    }
    z = int(np.round(np.argwhere(mask)[:, 2].mean()))
    fig, axes = plt.subplots(2, len(sets) + 2, figsize=(3 * (len(sets) + 2), 6.4))
    for r, (kind, position) in enumerate(chosen.items()):
        volume = strongest["SourceVolumes"][position]
        reference = np.rot90(clean[..., volume][:, :, z])
        outline = np.rot90(mask[:, :, z])
        top = np.percentile(clean[..., volume][mask], 99.5)
        panels = [("Reference", reference)]
        for path, sidecar in zip(sets, sidecars, strict=True):
            data = canonical(path.with_suffix("").with_suffix(".nii.gz"))
            index = sidecar["SourceVolumes"].index(volume)
            magnitude = sidecar["Eddy"]["Magnitude"]
            panels.append(
                (
                    f"{magnitude:g} voxels ({sidecar['Eddy']['MagnitudeMm']:.1f} mm)",
                    np.rot90(data[..., index][:, :, z]),
                )
            )
        for c, (title, image) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(image, cmap="gray", vmin=0, vmax=top, interpolation="nearest")
            # The outline of the undistorted brain shows where the signal moved
            ax.contour(outline, levels=[0.5], colors="gold", linewidths=0.6)
            ax.set_title(title, fontsize=9)
        field = displacement(clean_image, coefficients[position])
        limit = float(np.abs(field[mask]).max())
        shown = axes[r, -1].imshow(
            np.rot90(np.where(mask, field, np.nan)[:, :, z]),
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            interpolation="nearest",
        )
        fig.colorbar(shown, ax=axes[r, -1], fraction=0.046)
        axes[r, -1].set_title(
            f"Displacement at {sidecars[-1]['Eddy']['Magnitude']:g} voxels (mm)", fontsize=9
        )
        table = np.loadtxt(f"{stem}_dwi.b")[volume]
        axes[r, 0].set_ylabel(
            f"Volume {volume}, mostly {kind}\n"
            f"b={table[3]:g}, g=({table[0]:.2f}, {table[1]:.2f}, {table[2]:.2f})",
            fontsize=9,
        )
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    target = out / f"{protocol}_eddy_currents.png"
    fig.savefig(target, dpi=100)
    plt.close(fig)
    return target


def motion(folder: Path, cells: Path, protocol: str, out: Path) -> Path:
    """The first moved volumes against the reference, with the outline of the reference brain."""
    stem = folder / f"sub-{SUBJECT}_acq-{protocol}"
    clean = canonical(f"{stem}_rec-clean_dwi.nii.gz")
    mask = canonical(f"{stem}_desc-brain_mask.nii.gz") > 0
    paths = sorted(cells.glob(f"sub-{SUBJECT}_acq-{protocol}_desc-m*s*_dwi.json"))
    path = next(p for p in paths if json.loads(p.read_text())["Kind"] == "motion")
    sidecar = json.loads(path.read_text())
    moved = canonical(path.with_suffix("").with_suffix(".nii.gz"))
    center = np.round(np.argwhere(mask).mean(axis=0)).astype(int)
    count = min(2, moved.shape[-1])
    fig, axes = plt.subplots(
        count, 6, figsize=(16, 3.2 * count), squeeze=False, layout="constrained"
    )
    for r in range(count):
        volume = sidecar["SourceVolumes"][r]
        top = np.percentile(clean[..., volume][mask], 99.5)
        for q, (before, after, outline) in enumerate(
            zip(
                views(clean[..., volume], center),
                views(moved[..., r], center),
                views(mask.astype(float), center),
                strict=True,
            )
        ):
            for k, image in enumerate((before, after)):
                ax = axes[r, 2 * q + k]
                ax.imshow(image, cmap="gray", vmin=0, vmax=top, interpolation="nearest")
                ax.contour(outline, levels=[0.5], colors="gold", linewidths=0.6)
                ax.set_title(
                    f"{('Reference', 'Moved')[k]}, {('axial', 'coronal', 'sagittal')[q]}",
                    fontsize=9,
                )
                ax.set_xticks([])
                ax.set_yticks([])
        axes[r, 0].set_ylabel(f"Volume {volume}", fontsize=9)
    fig.suptitle(
        f"Rigid motion of {sidecar['Motion']['Magnitude']:g}° and "
        f"{sidecar['Motion']['Magnitude']:g} mm, re-simulated in the scanner",
        fontsize=10,
    )
    target = out / f"{protocol}_motion.png"
    fig.savefig(target, dpi=100)
    plt.close(fig)
    return target


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--real", type=Path, help="a real dataset of the same protocol")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    folder = args.dataset / f"sub-{SUBJECT}" / "dwi"
    cells = args.dataset / "derivatives" / "perturbations" / f"sub-{SUBJECT}" / "dwi"
    args.out.mkdir(parents=True, exist_ok=True)
    for target in (
        real_versus_synthetic(folder, args.protocol, args.real, args.out),
        truth_maps(folder, args.protocol, args.out),
        eddy_currents(folder, cells, args.protocol, args.out),
        motion(folder, cells, args.protocol, args.out),
    ):
        print(target)


if __name__ == "__main__":
    main()
