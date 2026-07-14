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
"""Dataset descriptors for the prediction gallery.

Each :class:`DatasetSpec` names an acquisition scheme and a ``loader`` that
returns a fully-constructed :class:`~nifreeze.data.dmri.DWI` (with ``b=0``
already extracted). Real OpenNeuro loaders (ds000206, ds000114, ds003138,
ds004737) are wired in a later phase; this module provides the descriptor type,
the scheme vocabulary, a scheme-verification helper, and a synthetic builder
used for fast, network-free testing.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nifreeze.data.dmri import DWI
from nifreeze.data.dmri.utils import find_shelling_scheme, format_gradients

SINGLE_SHELL = "single-shell"
MULTI_SHELL = "multi-shell"
DSI = "DSI"
SCHEMES = (SINGLE_SHELL, MULTI_SHELL, DSI)
"""Acquisition schemes, matching :func:`~nifreeze.data.dmri.utils.find_shelling_scheme`."""


def default_lovo_indices(dwi: DWI, count: int = 2) -> list[int]:
    """Pick a few, well-spread held-out volume indices for display.

    Two by default keeps the gallery tractable — some models (e.g. the
    multi-shell GP) take minutes per fit, and each rendered volume is already a
    multi-cut montage.
    """
    n = len(dwi)
    if n <= count:
        return list(range(n))
    # Evenly spaced interior indices (avoid the very first/last volume).
    return [int(round(k)) for k in np.linspace(0, n - 1, count + 2)[1:-1]]


@dataclass(frozen=True)
class DatasetSpec:
    """A gallery dataset: how to load it and what scheme it is."""

    name: str
    """Short identifier (e.g. ``"ds000206"``)."""
    title: str
    """Human-readable label for the gallery."""
    scheme: str
    """Declared acquisition scheme; verified against the data at load time."""
    loader: Callable[[], DWI]
    """Callable returning a constructed :class:`~nifreeze.data.dmri.DWI`."""
    lovo_indices: Callable[[DWI], list[int]] = default_lovo_indices
    """Callable choosing which held-out indices to render."""
    notes: str = ""
    """Free-form caveats surfaced on the gallery page (e.g. ``b=1000`` HARDI)."""

    def load(self) -> DWI:
        """Load the dataset and assert its scheme matches the declaration."""
        dwi = self.loader()
        verify_scheme(dwi, self.scheme, name=self.name)
        return dwi


def verify_scheme(dwi: DWI, expected: str, *, name: str = "dataset") -> str:
    """Classify ``dwi`` and raise if it does not match ``expected``.

    Guards the registry against silent drift between a dataset's declared scheme
    and its actual b-values (critical because model applicability keys on it).
    """
    if expected not in SCHEMES:
        raise ValueError(f"Unknown scheme {expected!r}; expected one of {SCHEMES}.")
    # ``DWI`` strips b=0 volumes into ``bzero`` at construction, but the shelling
    # classifier needs the low-b bin to distinguish single-shell (b0 + one shell)
    # from a lone high-b shell. Restore a single b=0 for classification.
    bvals = np.concatenate(([0.0], np.asarray(dwi.bvals, dtype=float)))
    observed, _, _ = find_shelling_scheme(bvals)
    if observed != expected:
        raise ValueError(
            f"Scheme mismatch for {name!r}: declared {expected!r}, found {observed!r}."
        )
    return observed


def synthetic_dwi(
    scheme: str = SINGLE_SHELL,
    *,
    n_directions: int = 24,
    vol_shape: Sequence[int] = (6, 6, 6),
    seed: int = 1234,
) -> DWI:
    """Build a tiny in-memory :class:`~nifreeze.data.dmri.DWI` for tests.

    Produces a dataset whose b-values classify as ``scheme`` (per
    :func:`~nifreeze.data.dmri.utils.find_shelling_scheme`). No physics — just a
    valid, cheap dataset to exercise the runner without any network access.

    Parameters
    ----------
    scheme : :obj:`str`
        One of :data:`SINGLE_SHELL`, :data:`MULTI_SHELL`, :data:`DSI`.
    n_directions : :obj:`int`
        Number of diffusion-weighted directions per shell.
    vol_shape : :obj:`Sequence`
        Spatial shape of the volume.
    seed : :obj:`int`
        Seed for the random number generator (reproducible).

    """
    rng = np.random.default_rng(seed)

    if scheme == SINGLE_SHELL:
        shells: tuple[float, ...] = (1000.0,)
    elif scheme == MULTI_SHELL:
        shells = (1000.0, 2000.0, 3000.0)
    elif scheme == DSI:
        # Many distinct b-values so ``find_shelling_scheme`` returns "DSI".
        shells = tuple(float(b) for b in range(500, 4001, 250))
    else:
        raise ValueError(f"Unknown scheme {scheme!r}; expected one of {SCHEMES}.")

    bvecs_list = []
    bvals_list = []
    for bval in shells:
        v = rng.normal(size=(n_directions, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        bvecs_list.append(v)
        bvals_list.append(np.full(n_directions, bval))

    # Prepend a single b=0 volume.
    bvecs = np.vstack([np.zeros((1, 3)), *bvecs_list])
    bvals = np.concatenate([np.zeros(1), *bvals_list])
    gradients = np.column_stack([bvecs, bvals])

    n_vols = gradients.shape[0]
    dataobj = rng.uniform(50.0, 1000.0, size=(*vol_shape, n_vols)).astype(np.float32)
    brainmask = np.ones(tuple(vol_shape), dtype=bool)

    return DWI(
        dataobj=dataobj,
        affine=np.eye(4),
        brainmask=brainmask,
        gradients=gradients,
    )


def synthetic_spec(scheme: str = SINGLE_SHELL, **kwargs) -> DatasetSpec:
    """A :class:`DatasetSpec` backed by :func:`synthetic_dwi` (for testing)."""
    return DatasetSpec(
        name=f"synthetic-{scheme}",
        title=f"Synthetic {scheme}",
        scheme=scheme,
        loader=lambda: synthetic_dwi(scheme, **kwargs),
        notes="Synthetic data for testing (no physiological meaning).",
    )


# ---------------------------------------------------------------------------
# OpenNeuro data provisioning (DataLad) + minimal preprocessing
# ---------------------------------------------------------------------------

DEFAULT_CACHE = Path.home() / ".cache" / "nifreeze-gallery"
"""Default location for fetched OpenNeuro datasets."""

DEFAULT_LOWB_THRESHOLD = 50
"""b-values at or below this are treated as ``b=0`` for masking."""


def _cache_root(cache_root: str | Path | None = None) -> Path:
    """Resolve the dataset cache directory (env ``NIFREEZE_GALLERY_DATA``)."""
    return Path(cache_root or os.environ.get("NIFREEZE_GALLERY_DATA") or DEFAULT_CACHE)


def _ensure_clone(accession: str, cache_root: str | Path | None = None) -> Path:
    """Ensure the OpenNeuro dataset is cloned (metadata only) and return its path.

    Cloning fetches the file tree (git-annex symlinks) but not the data; the
    actual NIfTIs are pulled on demand by :func:`_get`. Requires ``datalad`` on
    ``PATH`` only if the clone is not already present.
    """
    ds_path = _cache_root(cache_root) / accession
    if not (ds_path / ".datalad").exists():
        datalad = shutil.which("datalad")
        if not datalad:
            raise RuntimeError(
                f"{accession} is not present at {ds_path} and 'datalad' is not on "
                "PATH. Install nifreeze[gallery] or pre-clone the dataset."
            )
        ds_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [datalad, "clone", f"https://github.com/OpenNeuroDatasets/{accession}", str(ds_path)],
            check=True,
        )
    return ds_path


def _get(ds_path: Path, files: Sequence[Path]) -> None:
    """Fetch the given annexed files via ``datalad get`` (skips ones present)."""
    missing = [f for f in files if not Path(f).exists()]
    if not missing:
        return
    datalad = shutil.which("datalad")
    if not datalad:
        raise RuntimeError(
            f"Missing data files and 'datalad' is not on PATH: {missing}. "
            "Install nifreeze[gallery] or pre-fetch the files."
        )
    subprocess.run([datalad, "-C", str(ds_path), "get", *[str(f) for f in missing]], check=True)


def _brain_mask(data: np.ndarray, bvals: np.ndarray, *, median_radius: int, numpass: int):
    """Compute a brain mask from the ``b=0`` volumes via ``median_otsu``."""
    from dipy.segment.mask import median_otsu

    b0_idx = np.where(np.asarray(bvals) <= DEFAULT_LOWB_THRESHOLD)[0]
    vol_idx = b0_idx if b0_idx.size else np.arange(min(3, data.shape[-1]))
    _, mask = median_otsu(data, vol_idx=vol_idx, median_radius=median_radius, numpass=numpass)
    return mask


def _crop_to_mask(
    data: np.ndarray, mask: np.ndarray, affine: np.ndarray, *, margin: int = 2
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Crop ``data``/``mask`` to the mask bounding box (+margin), fixing the affine."""
    xs, ys, zs = np.where(mask)
    lo = np.maximum([xs.min(), ys.min(), zs.min()] - np.array(margin), 0)
    hi = np.minimum([xs.max(), ys.max(), zs.max()] + np.array(margin) + 1, mask.shape)
    sl = tuple(slice(int(lo[k]), int(hi[k])) for k in range(3))
    new_affine = affine.copy()
    new_affine[:3, 3] = affine[:3, :3] @ lo + affine[:3, 3]
    return data[sl], mask[sl], new_affine


def _load_dwi(
    triples: Sequence[tuple[Path, Path, Path]],
    ds_path: Path,
    *,
    crop: bool = True,
    median_radius: int = 2,
    numpass: int = 1,
) -> DWI:
    """Fetch, load, (optionally merge shells,) mask, and crop into a :class:`DWI`.

    ``triples`` is a list of ``(dwi_nii, bval, bvec)``; multiple entries are
    concatenated along the volume axis (used for datasets that store shells as
    separate files, e.g. ds003138).
    """
    from typing import cast

    import nibabel as nb
    from dipy.io import read_bvals_bvecs
    from nibabel.spatialimages import SpatialImage

    _get(ds_path, [f for triple in triples for f in triple])

    data_list: list[np.ndarray] = []
    bvals_list: list[np.ndarray] = []
    bvecs_list: list[np.ndarray] = []
    affine: np.ndarray | None = None
    for dwi_file, bval_file, bvec_file in triples:
        img = cast(SpatialImage, nb.load(str(dwi_file)))
        affine = img.affine
        data_list.append(np.asarray(img.dataobj, dtype=np.float32))
        bvals, bvecs = read_bvals_bvecs(str(bval_file), str(bvec_file))
        bvals_list.append(np.asarray(bvals))
        bvecs_list.append(np.asarray(bvecs))

    assert affine is not None  # triples is always non-empty

    data = np.concatenate(data_list, axis=-1) if len(data_list) > 1 else data_list[0]
    bvals = np.concatenate(bvals_list)
    bvecs = np.vstack(bvecs_list)
    gradients = format_gradients(np.column_stack([bvecs, bvals]))

    mask = _brain_mask(data, bvals, median_radius=median_radius, numpass=numpass)
    if crop:
        data, mask, affine = _crop_to_mask(data, mask, affine)

    return DWI(dataobj=data, affine=affine, brainmask=mask, gradients=gradients)


def _first(ds_path: Path, pattern: str) -> Path:
    """Return the first (sorted) path matching ``pattern`` under ``ds_path``."""
    matches = sorted(ds_path.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching {pattern!r} under {ds_path}.")
    return matches[0]


def _sidecars(nii: Path) -> tuple[Path, Path, Path]:
    """Return ``(nii, bval, bvec)`` for a ``*_dwi.nii.gz`` file."""
    base = str(nii)[: -len(".nii.gz")]
    return nii, Path(base + ".bval"), Path(base + ".bvec")


def load_ds000206(cache_root: str | Path | None = None) -> DWI:
    """Legacy DTI: ds000206 traveling phantom, 30 dir @ b=1000 (GD31)."""
    ds = _ensure_clone("ds000206", cache_root)
    nii = _first(ds, "sub-THP0001/ses-*/dwi/*acq-GD31*_dwi.nii.gz")
    return _load_dwi([_sidecars(nii)], ds)


def load_ds000114(cache_root: str | Path | None = None) -> DWI:
    """Single-shell HARDI: ds000114, 64 dir @ b=1000 (bval/bvec at the root)."""
    ds = _ensure_clone("ds000114", cache_root)
    nii = ds / "sub-01" / "ses-test" / "dwi" / "sub-01_ses-test_dwi.nii.gz"
    return _load_dwi([(nii, ds / "dwi.bval", ds / "dwi.bvec")], ds)


def load_ds003138(cache_root: str | Path | None = None) -> DWI:
    """Multi-shell: ds003138, b=1000/2000/3000 stored as three separate files."""
    ds = _ensure_clone("ds003138", cache_root)
    shell1 = _first(ds, "sub-*/ses-*/dwi/*acq-shell1*_dwi.nii.gz")
    dwidir = shell1.parent
    triples = [_sidecars(_first(dwidir, f"*acq-shell{k}*_dwi.nii.gz")) for k in (1, 2, 3)]
    return _load_dwi(triples, ds)


def load_ds004737(cache_root: str | Path | None = None) -> DWI:
    """DSI (compressed-sensing q-space): ds004737, HASC92 acquisition."""
    ds = _ensure_clone("ds004737", cache_root)
    nii = _first(ds, "sub-001/ses-*/dwi/*acq-HASC92*_dwi.nii.gz")
    return _load_dwi([_sidecars(nii)], ds)


#: The gallery's OpenNeuro datasets, one per acquisition scheme (issue #458).
DATASETS: list[DatasetSpec] = [
    DatasetSpec(
        name="ds000206",
        title="Legacy DTI (ds000206)",
        scheme=SINGLE_SHELL,
        loader=load_ds000206,
        notes="Traveling human phantom; 30 directions at b=1000 s/mm² (GD31).",
    ),
    DatasetSpec(
        name="ds000114",
        title="Single-shell HARDI (ds000114)",
        scheme=SINGLE_SHELL,
        loader=load_ds000114,
        notes="64 directions at b=1000 s/mm² (milder than textbook high-b HARDI).",
    ),
    DatasetSpec(
        name="ds003138",
        title="Multi-shell (ds003138)",
        scheme=MULTI_SHELL,
        loader=load_ds003138,
        notes="Three shells (b=1000/2000/3000 s/mm²) stored as separate files, merged on load.",
    ),
    DatasetSpec(
        name="ds004737",
        title="DSI (ds004737)",
        scheme=DSI,
        loader=load_ds004737,
        notes="Compressed-sensing DSI (q-space grid, HASC92), not a full 258-point grid.",
    ),
]
