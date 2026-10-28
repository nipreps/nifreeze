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
"""
The synthetic phantoms of ``tools/phantoms/phantom.py``, for the sensitivity analysis.

A phantom dataset holds, for each acquisition protocol, a noise-free reference
and only the changed volumes of each perturbed copy of it, with their ground
truth in the sidecars (see ``docs/phantoms.rst``). :func:`load_phantom` reads
one protocol as :func:`~nifreeze.analysis.sensitivity.sensitivity` takes it: the
reference, and each copy with its :class:`~nifreeze.analysis.sensitivity.Perturbation`.

"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from copy import copy
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from nibabel.spatialimages import SpatialImage

from nifreeze.analysis.sensitivity import Distortion, Perturbation
from nifreeze.data.dmri import DWI
from nifreeze.data.dmri.utils import DEFAULT_LOWB_THRESHOLD
from nifreeze.data.utils import to_time_first
from nifreeze.testing.phantoms import add_noise
from nifreeze.utils.ndimage import load_api

SUBJECT = "60501"
"""The phantom's subject: TRXScan's adult (NIBS sub-60501)."""

Cell = tuple[Perturbation, Callable[[], DWI]]
"""The ground truth of a perturbed copy, and a function that loads the copy."""

Part = tuple[Path, list[int], int]
"""A file of changed volumes, the reference's volumes that some of them replace, and the
first of those in the file (they follow one another)."""


def load_phantom(
    root: Path | str,
    protocol: str,
    noise: int | None = None,
    counts: Iterable[int] | None = None,
    alone: Iterable[int] | None = None,
) -> tuple[DWI, list[Cell]]:
    """
    Read a protocol's reference and its perturbed copies.

    A copy is assembled from the reference: its eddy-current volumes replace their
    counterparts, then its first K moved volumes replace theirs (or one of them,
    see ``alone``). The copies load one at a time, when
    :func:`~nifreeze.analysis.sensitivity.sensitivity` needs them, since each is as
    large as the reference.

    Parameters
    ----------
    root : :obj:`os.pathlike`
        Root of the phantom dataset.
    protocol : :obj:`str`
        Name of the protocol (``acq-<protocol>``).
    noise : :obj:`int`, optional
        Seed of the noise added to every volume on load
        (:func:`~nifreeze.testing.phantoms.add_noise`, at the protocol's
        signal-to-noise ratio); noise-free if :obj:`None`. The noise is keyed on the
        volume, so a copy has the reference's noise wherever it is unchanged.
    counts : :obj:`~collections.abc.Iterable`, optional
        Numbers of moved volumes: each copy with motions gives one copy per count,
        moving the first K of its volumes. All of them if :obj:`None`.
    alone : :obj:`~collections.abc.Iterable`, optional
        Volumes (numbered as the reference's) that each copy with motions also moves
        alone, with the motion it gives them: one copy per volume, to rank the
        volumes by their influence. None if :obj:`None`.

    Returns
    -------
    :obj:`tuple`
        The reference, as a :class:`~nifreeze.data.dmri.DWI` (b=0 volumes are
        averaged into its ``bzero``, and the diffusion-weighted ones are numbered
        from 0), and the copies, each with its ground truth.

    """
    root = Path(root)
    stem = root / f"sub-{SUBJECT}" / "dwi" / f"sub-{SUBJECT}_acq-{protocol}"
    sidecar = json.loads(Path(f"{stem}_dwi.json").read_text())
    gradients = np.loadtxt(f"{stem}_dwi.b", ndmin=2)
    weighted = np.flatnonzero(gradients[:, 3] > DEFAULT_LOWB_THRESHOLD)
    position = {int(volume): index for index, volume in enumerate(weighted)}
    noisy = partial(_noisy, sigma=float(sidecar["Noise"]["Sigma"]), seed=noise)
    reference = _reference(stem, gradients, noisy)
    splice = partial(_splice, reference, position, noisy)

    eddies: dict[float, tuple[Path, list[int], Distortion]] = {}
    motions: list[tuple[Path, dict[str, Any]]] = []
    for volumes, cell in _changed(root, protocol, sidecar, position):
        if cell["Kind"] != "eddy":
            motions.append((volumes, cell))
            continue
        sources = cell["SourceVolumes"]
        indices = [position[v] for v in sources]
        distortion = Distortion.from_sidecar(cell["Eddy"], indices, len(weighted))
        eddies[cell["Eddy"]["Magnitude"]] = (volumes, sources, distortion)

    cells: list[Cell] = [
        (Perturbation(0.0, None, distortion=distortion), splice([(volumes, sources, 0)]))
        for volumes, sources, distortion in eddies.values()
    ]
    counts = None if counts is None else [int(k) for k in counts]
    alone = [] if alone is None else [int(j) for j in alone]
    for volumes, cell in motions:
        parts: list[Part] = []
        eddy: Distortion | None = None
        if cell["Eddy"] is not None:
            if cell["Eddy"]["Magnitude"] not in eddies:
                raise ValueError(f"{volumes.name} needs the copy with its eddy currents alone.")
            eddy_volumes, eddy_sources, eddy = eddies[cell["Eddy"]["Magnitude"]]
            parts.append((eddy_volumes, eddy_sources, 0))
        for perturbation, part in _moving(volumes, cell, position, eddy, counts, alone):
            cells.append((perturbation, splice([*parts, part])))
    return reference, cells


def _moving(
    volumes: Path,
    cell: dict[str, Any],
    position: dict[int, int],
    eddy: Distortion | None,
    counts: list[int] | None,
    alone: list[int],
) -> list[tuple[Perturbation, Part]]:
    """The copies that a file of moved volumes gives, each with the volumes it splices."""
    sources, motion = cell["SourceVolumes"], cell["Motion"]
    moved = [position[v] for v in sources]
    matrices = [np.asarray(m, dtype=float) for m in motion["Matrices"]]

    def copy(start: int, stop: int) -> tuple[Perturbation, Part]:
        perturbation = Perturbation(
            magnitude=float(motion["Magnitude"]),
            seed=int(motion["Seed"]),
            moved=tuple(moved[start:stop]),
            motions=tuple(matrices[start:stop]),
            distortion=eddy,
        )
        return perturbation, (volumes, sources[start:stop], start)

    copies = []
    for count in [len(sources)] if counts is None else counts:
        if count > len(sources):
            raise ValueError(f"{volumes.name} moves {len(sources)} volumes, not {count}.")
        copies.append(copy(0, count))
    for volume in alone:
        if volume not in moved:
            raise ValueError(f"{volumes.name} does not move volume {volume}.")
        row = moved.index(volume)
        copies.append(copy(row, row + 1))
    return copies


def _noisy(
    volumes: np.ndarray, indices: Iterable[int], sigma: float, seed: int | None
) -> np.ndarray:
    """The volumes with the noise of a seed, keyed on their index in the file."""
    return volumes if seed is None else add_noise(volumes, sigma, seed, indices)


def _reference(stem: Path, gradients: np.ndarray, noisy: Callable[..., np.ndarray]) -> DWI:
    """The reference, with its b=0 volumes averaged as ``from_nii`` does."""
    image = load_api(f"{stem}_rec-clean_dwi.nii.gz", SpatialImage)
    data = np.asarray(image.dataobj, dtype=np.float32)
    weighted = np.flatnonzero(gradients[:, 3] > DEFAULT_LOWB_THRESHOLD)
    bzeros = np.flatnonzero(gradients[:, 3] <= DEFAULT_LOWB_THRESHOLD)
    bzero = None
    if len(bzeros):
        bzero = np.median(noisy(to_time_first(data, keep=bzeros), bzeros), axis=0)
    return DWI(
        dataobj=noisy(to_time_first(data, keep=weighted), weighted),
        affine=image.affine,
        gradients=gradients[weighted],
        bzero=bzero,
        brainmask=np.asarray(load_api(f"{stem}_desc-brain_mask.nii.gz", SpatialImage).dataobj) > 0,
    )


def _changed(
    root: Path, protocol: str, sidecar: dict[str, Any], position: dict[int, int]
) -> list[tuple[Path, dict[str, Any]]]:
    """The files of changed volumes that perturb this reference, with their sidecars."""
    folder = root / "derivatives" / "perturbations" / f"sub-{SUBJECT}" / "dwi"
    found = []
    for path in sorted(folder.glob(f"sub-{SUBJECT}_acq-{protocol}_desc-*_dwi.json")):
        cell = json.loads(path.read_text())
        mine = (sidecar["ProtocolSha256"], sidecar.get("ProtocolVolumes"))
        if (cell["ProtocolSha256"], cell.get("ProtocolVolumes")) != mine:
            raise ValueError(f"{path.name} perturbs another reference.")
        if any(v not in position for v in cell["SourceVolumes"]):
            raise ValueError(f"{path.name} changes volumes that are not diffusion-weighted.")
        found.append((path.with_name(path.name.removesuffix(".json") + ".nii.gz"), cell))
    return found


def _splice(
    reference: DWI,
    position: dict[int, int],
    noisy: Callable[..., np.ndarray],
    parts: list[Part],
) -> Callable[[], DWI]:
    """A function that loads a copy: the reference, with volumes of each file in turn."""

    def load() -> DWI:
        dataobj = np.array(reference.dataobj, dtype=np.float32)
        for path, sources, start in parts:
            changed = load_api(path, SpatialImage)
            grid = changed.shape[:3] == reference.shape3d
            if not grid or not np.allclose(changed.affine, reference.affine):
                raise ValueError(f"{path.name} is not on the reference's grid.")
            rows = changed.dataobj[..., start : start + len(sources)]
            volumes = to_time_first(np.asarray(rows, dtype=np.float32))
            dataobj[[position[v] for v in sources]] = noisy(volumes, sources)
        perturbed = copy(reference)
        perturbed.dataobj = dataobj
        return perturbed

    return load
