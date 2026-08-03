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
already extracted) or :class:`~nifreeze.data.pet.PET`. Real OpenNeuro loaders (ds000206, ds000114, ds003138,
ds004737, ds00PET) are wired in a later phase; this module provides the descriptor type,
the scheme vocabulary, a scheme-verification helper, and a synthetic builder
used for fast, network-free testing.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np

from nifreeze.data.base import BaseDataset
from nifreeze.data.dmri import DWI
from nifreeze.data.dmri.utils import find_shelling_scheme, format_gradients
from nifreeze.data.pet import PET

SINGLE_SHELL = "single-shell"
MULTI_SHELL = "multi-shell"
DSI = "DSI"
DMRI_SCHEMES = (SINGLE_SHELL, MULTI_SHELL, DSI)
"""Acquisition schemes, matching :func:`~nifreeze.data.dmri.utils.find_shelling_scheme`."""

PET_SCHEME = "PET"
SCHEMES = DMRI_SCHEMES + (PET_SCHEME,)
"""All supported schemes types."""

DWITriplet: TypeAlias = tuple[Path, Path, Path]  # DWI data, bvals, bvecs
PETPair: TypeAlias = tuple[Path, Path]  # PET data, frametime
SourceFiles: TypeAlias = list[DWITriplet] | list[PETPair]


def default_lovo_indices(dataset: BaseDataset, count: int = 2) -> list[int]:
    """Pick a few, well-spread held-out volume indices for display.

    Two by default keeps the gallery tractable — some models (e.g. the
    multi-shell GP) take minutes per fit, and each rendered volume is already a
    multi-cut montage.
    """
    n = len(dataset)
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
    loader: Callable[[], DWI | PET]
    """Callable returning a constructed :class:`~nifreeze.data.base.BaseDataset` subclass."""
    lovo_indices: Callable[[BaseDataset], list[int]] = default_lovo_indices
    """Callable choosing which held-out indices to render."""
    notes: str = ""
    """Free-form caveats surfaced on the gallery page (e.g. ``b=1000`` HARDI)."""
    url: str = ""
    """Link to the source dataset (e.g. its OpenNeuro page)."""

    def load(self) -> DWI | PET:
        """Load the dataset and assert its scheme matches the declaration.

        If ``NIFREEZE_GALLERY_H5DIR`` points at a directory holding
        ``<name>.h5`` (written once by the fetch stage), that pre-cropped
        dataset is read directly, so the parallel fit jobs never touch datalad
        or the network.
        """
        h5dir = os.environ.get("NIFREEZE_GALLERY_H5DIR")
        if h5dir:
            cached = Path(h5dir) / f"{self.name}.h5"
            if cached.is_file():
                ds = (
                    DWI.from_filename(cached)
                    if "dwi" in self.name.lower()
                    else PET.from_filename(cached)
                )
                verify_scheme(ds, self.scheme, name=self.name)
                return ds
        ds = self.loader()
        verify_scheme(ds, self.scheme, name=self.name)
        return ds


def verify_scheme(dataset: DWI | PET, expected: str, *, name: str = "dataset") -> str:
    """Classify ``dataset`` and raise if it does not match ``expected``.

    Guards the registry against silent drift between a dataset's declared scheme
    and its actual scheme (critical because model applicability keys on it).
    For PET, scheme verification is a no-op since PET has a single scheme.
    """
    if expected not in SCHEMES:
        raise ValueError(f"Unknown scheme {expected!r}; expected one of {SCHEMES}.")

    # Check for PET data
    if expected == PET_SCHEME:
        if not isinstance(dataset, PET):
            raise ValueError(
                f"Scheme mismatch for {name!r}: declared {expected!r}, found non-PET dataset."
            )
        return expected

    # From here on, expected a DWI scheme
    if not isinstance(dataset, DWI):
        raise ValueError(
            f"Scheme mismatch for {name!r}: declared {expected!r}, found PET dataset."
        )

    # ``DWI`` strips b=0 volumes into ``bzero`` at construction, but the shelling
    # classifier needs the low-b bin to distinguish single-shell (b0 + one shell)
    # from a lone high-b shell. Restore a single b=0 for classification.
    bvals = np.concatenate(([0.0], np.asarray(dataset.bvals, dtype=float)))
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


def synthetic_pet(
    n_frames: int = 10,
    vol_shape: Sequence[int] = (6, 6, 6),
    seed: int = 1234,
) -> PET:
    """Build a tiny in-memory :class:`~nifreeze.data.pet.PET` for tests.

    Parameters
    ----------
    n_frames : :obj:`int`
        Number of temporal frames.
    vol_shape : :obj:`Sequence`
        Spatial shape of the volume.
    seed : :obj:`int`
        Seed for the random number generator (reproducible).

    Returns
    -------
    :class:`~nifreeze.data.pet.PET`
        A synthetic PET dataset.
    """
    rng = np.random.default_rng(seed)

    dataobj = rng.uniform(50.0, 1000.0, size=(*vol_shape, n_frames)).astype(np.float32)
    brainmask = np.ones(tuple(vol_shape), dtype=bool)
    affine = np.eye(4)

    # Generate temporal markers (midframe times in seconds)
    frame_start = np.arange(n_frames, dtype=np.float32) * 2.0  # 2 sec per frame
    frame_duration = np.full(n_frames, 2.0, dtype=np.float32)
    midframe = frame_start + frame_duration / 2.0
    total_duration = frame_start[-1] + frame_duration[-1]

    return PET(
        dataobj=dataobj,
        affine=affine,
        brainmask=brainmask,
        midframe=midframe,
        total_duration=total_duration,
    )


def synthetic_spec(scheme: str, modality: str, **kwargs) -> DatasetSpec:
    """A :class:`DatasetSpec` backed by synthetic data (for testing).

    Parameters
    ----------
    scheme : :obj:`str`
        Acquisition scheme (e.g., "single-shell", "multi-shell", "DSI", "PET").
    modality : :obj:`str`
        "dwi" or "pet".
    **kwargs
        Extra keyword arguments passed to the synthetic builder.

    Returns
    -------
    :class:`DatasetSpec`
        A dataset spec for testing.
    """

    if modality.lower() == "dwi":
        return DatasetSpec(
            name=f"synthetic-{scheme}",
            title=f"Synthetic {scheme}",
            scheme=scheme,
            loader=lambda: synthetic_dwi(scheme, **kwargs),
            notes="Synthetic data for testing (no physiological meaning).",
        )
    elif modality.lower() == "pet":
        return DatasetSpec(
            name="synthetic-pet",
            title="Synthetic PET",
            scheme=PET_SCHEME,
            loader=lambda: synthetic_pet(**kwargs),
            notes="Synthetic data for testing (no physiological meaning).",
        )
    else:
        raise ValueError(f"Unknown modality {modality!r}; expected 'dwi' or 'pet'.")


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


def _brain_mask(
    data: np.ndarray,
    bvals: np.ndarray | None = None,
    *,
    median_radius: int,
    numpass: int,
    n_vols: int = 3,
):
    """Compute a brain mask from the data via ``median_otsu``.

    For DWI data, it uses ``b=0`` volumes if detected (b-values lower
    than :data:`DEFAULT_LOWB_THRESHOLD`), otherwise uses the first few
    volumes. For PET data, it uses the first few volumes (``bvals`` is
    :obj:`None`).

    Parameters
    ----------
    data : :obj:`~numpy.ndarray`
        4D array with shape (X, Y, Z, n_volumes).
    bvals : :obj:`~numpy.ndarray`, optional
        1D array of b-values. If provided, ``b=0`` volumes are
        preferred for masking.
    median_radius : :obj:`int`
        Median filter radius.
    numpass : :obj:`int`
        Number of median filtering passes.
    n_vols : :obj:ìnt`, optional
        Number of volumes to use for mask computation if ``b=0``
        volumes are not available.

    Returns
    -------
    :obj:`~numpy.ndarray`
        Binary brain mask with shape (X, Y, Z).
    """

    from dipy.segment.mask import median_otsu

    if bvals is not None:
        # DWI data
        b0_idx = np.where(np.asarray(bvals) <= DEFAULT_LOWB_THRESHOLD)[0]
        vol_idx = b0_idx if b0_idx.size else np.arange(min(n_vols, data.shape[-1]))
    else:
        # If no b-value data provided (e.g. PET), use the first few volumes
        vol_idx = np.arange(min(n_vols, data.shape[-1]))

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

    from dipy.io import read_bvals_bvecs
    from nibabel.spatialimages import SpatialImage

    from nifreeze.utils.ndimage import load_api

    _get(ds_path, [f for triple in triples for f in triple])

    data_list: list[np.ndarray] = []
    bvals_list: list[np.ndarray] = []
    bvecs_list: list[np.ndarray] = []
    affine: np.ndarray | None = None
    for dwi_file, bval_file, bvec_file in triples:
        img = load_api(dwi_file, SpatialImage)
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


def _load_pet(
    pairs: Sequence[tuple[Path, Path]],
    ds_path: Path,
    *,
    crop: bool = True,
    median_radius: int = 2,
    numpass: int = 1,
) -> PET:
    """Fetch, load, mask, and crop into a :class:`PET`.

    ``pairs`` is a list of ``(pet_nii, temporal_json)``; typically a single pair
    for a given PET series.

    Parameters
    ----------
    pairs : :obj:`Sequence`
        List of ``(pet_nii, temporal_json)`` file paths.
    ds_path : :obj:`~pathlib.Path`
        Root of the OpenNeuro dataset clone.
    crop : :obj:`bool`
        Whether to crop to the brain mask with margin.
    median_radius : :obj:`int`
        Median filter radius for mask computation.
    numpass : :obj:`int`
        Number of median filtering passes.

    Returns
    -------
    :class:`~nifreeze.data.pet.PET`
        A loaded and (optionally cropped) PET dataset.
    """
    from nibabel.spatialimages import SpatialImage

    from nifreeze.data.pet.io import FRAME_TIME_START_KEY
    from nifreeze.data.pet.utils import compute_temporal_markers
    from nifreeze.utils.ndimage import load_api

    _get(ds_path, [f for pair in pairs for f in pair])

    data_list: list[np.ndarray] = []
    midframe_list: list[np.ndarray] = []
    affine: np.ndarray | None = None
    total_duration: float | None = None

    for pet_file, temporal_file in pairs:
        # Load PET NIfTI
        img = load_api(pet_file, SpatialImage)
        affine = img.affine
        data_list.append(np.asarray(img.dataobj, dtype=np.float32))

        # Load temporal metadata and compute midframe times
        with open(temporal_file, "r") as f:
            temporal_attrs = json.load(f)

        frame_time = temporal_attrs.get(FRAME_TIME_START_KEY, None)
        midframe, _total_duration = compute_temporal_markers(frame_time)
        midframe_list.append(midframe)
        total_duration = _total_duration

    assert affine is not None  # pairs is always non-empty
    assert total_duration is not None

    # Concatenate data and midframe times if multiple PET files
    data = np.concatenate(data_list, axis=-1) if len(data_list) > 1 else data_list[0]
    midframe = np.concatenate(midframe_list) if len(midframe_list) > 1 else midframe_list[0]

    # Compute brain mask (PET-specific: no b-values)
    mask = _brain_mask(data, bvals=None, median_radius=median_radius, numpass=numpass)
    if crop:
        data, mask, affine = _crop_to_mask(data, mask, affine)

    return PET(
        dataobj=data,
        affine=affine,
        brainmask=mask,
        midframe=midframe,
        total_duration=total_duration,
    )


def _first(ds_path: Path, pattern: str) -> Path:
    """Return the first (sorted) path matching ``pattern`` under ``ds_path``."""
    matches = sorted(ds_path.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching {pattern!r} under {ds_path}.")
    return matches[0]


def _dwi_sidecars(nii: Path) -> tuple[Path, Path, Path]:
    """Return ``(nii, bval, bvec)`` for a ``*_dwi.nii.gz`` file."""
    base = str(nii)[: -len(".nii.gz")]
    return nii, Path(base + ".bval"), Path(base + ".bvec")


def _pet_sidecars(nii: Path) -> tuple[Path, Path]:
    """Return ``(nii, temporal_json)`` for a ``*_pet.nii.gz`` file.

    Looks for a JSON sidecar with FrameTimesStart.
    """
    base = str(nii)[: -len(".nii.gz")]
    return nii, Path(base + ".json")


# Each ``_resolve_*`` returns ``(dataset_path, [(dwi, bval, bvec), ...])`` so the
# exact subject/run used can be both loaded and reported on the gallery page.
def _resolve_ds000206(cache_root=None) -> tuple[Path, list[tuple[Path, Path, Path]]]:
    ds = _ensure_clone("ds000206", cache_root)
    return ds, [_dwi_sidecars(_first(ds, "sub-THP0001/ses-*/dwi/*acq-GD31*_dwi.nii.gz"))]


def _resolve_ds000114(cache_root=None) -> tuple[Path, list[tuple[Path, Path, Path]]]:
    ds = _ensure_clone("ds000114", cache_root)
    nii = ds / "sub-01" / "ses-test" / "dwi" / "sub-01_ses-test_dwi.nii.gz"
    return ds, [(nii, ds / "dwi.bval", ds / "dwi.bvec")]


def _resolve_ds003138(cache_root=None) -> tuple[Path, list[tuple[Path, Path, Path]]]:
    ds = _ensure_clone("ds003138", cache_root)
    dwidir = _first(ds, "sub-*/ses-*/dwi/*acq-shell1*_dwi.nii.gz").parent
    return ds, [_dwi_sidecars(_first(dwidir, f"*acq-shell{k}*_dwi.nii.gz")) for k in (1, 2, 3)]


def _resolve_ds004737(cache_root=None) -> tuple[Path, list[tuple[Path, Path, Path]]]:
    ds = _ensure_clone("ds004737", cache_root)
    return ds, [_dwi_sidecars(_first(ds, "sub-001/ses-*/dwi/*acq-HASC92*_dwi.nii.gz"))]


def _resolve_ds00PET(cache_root=None) -> tuple[Path, list[tuple[Path, Path]]]:
    ds = _ensure_clone("ds00PET", cache_root)
    return ds, [_pet_sidecars(_first(ds, "sub-001/ses-*/pet/*XXX*_pet.nii.gz"))]


#: Dataset name -> resolver returning the concrete files to load.
RESOLVERS: dict[str, Callable[[str | Path | None], tuple[Path, SourceFiles]]] = {
    "ds000206": _resolve_ds000206,
    "ds000114": _resolve_ds000114,
    "ds003138": _resolve_ds003138,
    "ds004737": _resolve_ds004737,
    "ds00PET": _resolve_ds00PET,
}


def sources_sidecar(name: str, h5dir: str | Path | None = None) -> Path | None:
    """Path of the source-provenance sidecar beside the cached ``<name>.h5``."""
    root = h5dir if h5dir is not None else os.environ.get("NIFREEZE_GALLERY_H5DIR")
    return Path(root) / f"{name}.sources.json" if root else None


def source_relpaths(name: str, cache_root: str | Path | None = None) -> list[str]:
    """Return the file path(s) (relative to the dataset) actually loaded.

    Resolving these from the datalad clone is only possible where the clone
    exists (the fetch stage), so that stage records them in a sidecar next to the
    cached ``<name>.h5``. This prefers that sidecar, which keeps the parallel fit
    jobs off the network entirely.
    """
    sidecar = sources_sidecar(name)
    if sidecar is not None and sidecar.is_file():
        return list(json.loads(sidecar.read_text()))
    ds, files = RESOLVERS[name](cache_root)
    # Flatten file tuples to a list of relative paths
    return [str(f.relative_to(ds)) for file_tuple in files for f in file_tuple]


def load_ds000206(cache_root: str | Path | None = None) -> DWI:
    """Legacy DTI: ds000206 traveling phantom, 30 dir @ b=1000 (GD31)."""
    ds, triples = _resolve_ds000206(cache_root)
    return _load_dwi(triples, ds)


def load_ds000114(cache_root: str | Path | None = None) -> DWI:
    """Single-shell HARDI: ds000114, 64 dir @ b=1000 (bval/bvec at the root)."""
    ds, triples = _resolve_ds000114(cache_root)
    return _load_dwi(triples, ds)


def load_ds003138(cache_root: str | Path | None = None) -> DWI:
    """Multi-shell: ds003138, b=1000/2000/3000 stored as three separate files."""
    ds, triples = _resolve_ds003138(cache_root)
    return _load_dwi(triples, ds)


def load_ds004737(cache_root: str | Path | None = None) -> DWI:
    """DSI (compressed-sensing q-space): ds004737, HASC92 acquisition."""
    ds, triples = _resolve_ds004737(cache_root)
    return _load_dwi(triples, ds)


def load_ds00PET(cache_root: str | Path | None = None) -> PET:
    """Load a PET dataset from OpenNeuro (example dataset ID TBD)."""
    # This will be implemented once an OpenNeuro PET dataset is selected
    ds, triples = _resolve_ds00PET(cache_root)
    raise NotImplementedError("PET dataset loader not yet configured")
    return _load_pet(triples, ds)


#: The gallery's OpenNeuro datasets, one per acquisition scheme (issue #458).
DATASETS: list[DatasetSpec] = [
    DatasetSpec(
        name="ds000206",
        title="Legacy DTI (ds000206)",
        scheme=SINGLE_SHELL,
        loader=load_ds000206,
        notes="Traveling human phantom; 30 directions at b=1000 s/mm² (GD31).",
        url="https://openneuro.org/datasets/ds000206",
    ),
    DatasetSpec(
        name="ds000114",
        title="Single-shell HARDI (ds000114)",
        scheme=SINGLE_SHELL,
        loader=load_ds000114,
        notes="64 directions at b=1000 s/mm² (milder than textbook high-b HARDI).",
        url="https://openneuro.org/datasets/ds000114",
    ),
    DatasetSpec(
        name="ds003138",
        title="Multi-shell (ds003138)",
        scheme=MULTI_SHELL,
        loader=load_ds003138,
        notes="Three shells (b=1000/2000/3000 s/mm²) stored as separate files, merged on load.",
        url="https://openneuro.org/datasets/ds003138",
    ),
    DatasetSpec(
        name="ds004737",
        title="DSI (ds004737)",
        scheme=DSI,
        loader=load_ds004737,
        notes="Compressed-sensing DSI (q-space grid, HASC92), not a full 258-point grid.",
        url="https://openneuro.org/datasets/ds004737",
    ),
    DatasetSpec(
        name="ds00PET",
        title="PET (ds00PET)",
        scheme=PET_SCHEME,
        loader=load_ds00PET,
        notes="PET.",
        url="https://openneuro.org/datasets/ds00PET",
    ),
]
