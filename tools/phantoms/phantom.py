# /// script
# requires-python = ">=3.12"
# dependencies = ["trxscan[dipy,trx]", "nifreeze"]
#
# [tool.uv.sources]
# nifreeze = { path = "../..", editable = true }
#
# [tool.uv.sources.trxscan]
# git = "https://github.com/PennLINC/TRXScan"
# rev = "680fc4f4559c5dd9fa759c2031f07f83384da2df"
# subdirectory = "python"
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
"""Simulate the synthetic reference datasets with TRXScan (see ``docs/phantoms.rst``).

The only code that imports TRXScan, pinned to one commit above (its motion fixes
are not released yet). Run it with ``uv run``, which builds that environment
from the versions locked in ``phantom.py.lock``::

    uv run tools/phantoms/phantom.py reference --protocol sherbrooke --out <dataset>
    uv run tools/phantoms/phantom.py cells --protocol sherbrooke --out <dataset>
    uv run tools/phantoms/phantom.py verify --protocol sherbrooke
    uv run tools/phantoms/phantom.py fixture --out <dataset>

``reference`` simulates the phantom under one protocol of ``protocols.toml``,
clean: no motion, eddy currents, susceptibility distortion, readout relaxation
or noise; T2 contrast comes from per-tissue S0 at the protocol's echo time. It
writes the noise-free magnitude, one noisy realization at the protocol's SNR,
the brain mask, the gradients, the truth maps and a JSON sidecar, under
``<dataset>/sub-60501/dwi``.

``cells`` simulates the perturbed copies of that reference: diffusion-weighted
volumes moved by rigid motions (re-simulated from the moved phantom), distorted
by eddy currents (TRXScan's linear model, whose displacement is known in closed
form), or both. It writes only the changed volumes, with their ground truth,
under ``<dataset>/derivatives/perturbations/sub-60501/dwi``. Each moved volume
spends most of its time in serial code, so ``--jobs`` simulates them in that many
processes, which share the threads of ``RAYON_NUM_THREADS``.

``verify`` checks, at a coarse resolution, what the perturbed copies rely on:
repeated runs are identical, a subset of the gradient table reproduces the same
volumes of the full run, the closed-form displacement reproduces TRXScan's
eddy-current images, and a moved volume moves as its matrix says.

``fixture`` writes the small dataset that CI simulates for the phantom tests
and the quality benchmarks: the Sherbrooke protocol at 4 mm, one b=0 and 8
directions per shell, its reference and the perturbed copies they use.
"""

import argparse
import json
import math
import multiprocessing
import os
import sys
import tomllib
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from functools import cache
from importlib.metadata import version
from itertools import islice, repeat
from pathlib import Path
from typing import Any

import nibabel as nb
import numpy as np
import trxscan as ts
from dipy.core.gradients import gradient_table

from nifreeze.testing.perturbations import perturbation_plan
from nifreeze.testing.phantoms import add_noise, noise_sigma

HERE = Path(__file__).parent
CATALOG = tomllib.loads((HERE / "protocols.toml").read_text())
TRXSCAN_COMMIT = "680fc4f4559c5dd9fa759c2031f07f83384da2df"
SUBJECT = "60501"
"""TRXScan's phantom: an adult (NIBS sub-60501) with ACT+SIFT2 streamlines, CC0."""
PHANTOM_SEED = 0
"""Seed of the streamlines kept from the phantom's million, the same for every dataset."""
ASSUMED_ECHO_TIME = 0.09
"""Echo time (s) for the T2 contrast of a protocol whose echo time is not documented."""
ROUNDOFF = 1e-4
"""Magnitudes below this (a millionth of TRXScan's signal scale) are FFT round-off."""
EDDY_TAU_MS = 1e12
"""Decay time of the eddy currents: none during the readout, so they only displace."""
CLEAN = ts.Artifacts(distortion=False, relaxation=False)
"""No susceptibility distortion and no T2 weighting along the readout."""
FIXTURE: dict[str, Any] = {
    "name": "sherbrooke",
    "voxel": 4.0,
    "streamlines": 50_000,
    "directions": 8,
    # Rigid motion of all volumes (the first quarter of them is a copy too), at 1 mm and 1°
    "magnitudes": [1.0],
    "eddies": [2.0],
    "seeds": [0],
}
"""The dataset that CI simulates, small enough for every pull request."""


def acquisition(name: str, voxel: float | None = None) -> Any:
    """The protocol of a catalog entry as a clean TRXScan acquisition.

    With ``voxel``, the voxels are isotropic of that size and the matrix keeps
    the field of view.
    """
    entry = CATALOG[name]
    voxels = np.asarray(entry["voxel_mm"], dtype=float)
    matrix = np.asarray(entry["matrix"], dtype=float)
    if voxel is not None:
        matrix = np.round(matrix * voxels / voxel)
        voxels = np.full(3, voxel)
    te_ms = 1000.0 * echo_time(name)[0]
    settings: dict[str, Any] = {}
    if not math.isnan(entry["echo_spacing"]):
        settings["echo_spacing_ms"] = 1000.0 * entry["echo_spacing"]
    return ts.Protocol.DEFAULT.replace(
        voxel_mm=tuple(float(v) for v in voxels),
        matrix=tuple(int(m) for m in matrix),
        # TRXScan encodes phase along j: turn the grid so that j runs along the world's LR
        oblique_deg=(0.0, 0.0, 90.0) if entry["phase_encoding"] == "LR" else None,
        pe="j-",
        te_ms=te_ms,
        # T2 contrast without the readout's T2 weighting, which blurs along phase encoding
        tissue_s0=tuple(math.exp(-te_ms / t2) for t2 in ts.Tissue.ADULT.t2_ms),
        oversample=1,
        phase_model="none",
        partial_fourier=1.0,
        coils=1,
        accel=1,
        **settings,
    )


def echo_time(name: str) -> tuple[float, bool]:
    """The echo time (s) of a catalog entry, and whether it is assumed."""
    documented = CATALOG[name]["echo_time"]
    return (ASSUMED_ECHO_TIME, True) if math.isnan(documented) else (documented, False)


def gradients(name: str, volumes: list[int] | None = None) -> tuple[np.ndarray, Any]:
    """The world-RAS gradient table of a catalog entry, and its DIPY form for TRXScan."""
    table = np.loadtxt(HERE / CATALOG[name]["gradients"])
    if volumes is not None:
        table = table[volumes]
    return table, _dipy(table)


def _dipy(table: np.ndarray) -> Any:
    return gradient_table(table[:, 3], bvecs=table[:, :3], b0_threshold=50)


def spread(name: str, directions: int) -> list[int]:
    """The first b=0 volume and ``directions`` volumes of each shell, spread over the sphere.

    Each shell starts from its first direction and adds, one at a time, the one
    farthest from those already taken (directions are axes: a direction and its
    opposite are the same). The volumes keep the table's order.
    """
    table = gradients(name)[0]
    picks = [int(np.flatnonzero(table[:, 3] < 50)[0])]
    for b in np.unique(table[table[:, 3] >= 50, 3]):
        shell = np.flatnonzero(table[:, 3] == b)
        cosines = np.abs(table[shell, :3] @ table[shell, :3].T)
        taken = [0]
        while len(taken) < directions:
            taken.append(int(np.argmin(cosines[taken].max(axis=0))))
        picks += [int(shell[i]) for i in taken]
    return sorted(picks)


def _table_text(name: str, volumes: list[int] | None) -> str:
    """The gradient file of a catalog entry, with only the rows of ``volumes``."""
    text = (HERE / CATALOG[name]["gradients"]).read_text()
    if volumes is None:
        return text
    lines = text.splitlines(keepends=True)
    rows = [line for line in lines if not line.startswith("#")]
    comments = [line for line in lines if line.startswith("#")]
    comments.append(f"# only volumes {','.join(str(v) for v in volumes)} of the protocol\n")
    return "".join(comments + [rows[v] for v in volumes])


@cache
def phantom(streamlines: int) -> Any:
    return ts.Phantom.load(f"sub-{SUBJECT}").subsample(streamlines, seed=PHANTOM_SEED)


def simulate(
    name: str,
    streamlines: int = 400_000,
    voxel: float | None = None,
    volumes: list[int] | None = None,
    chunk: int | None = 8,
    truth: bool = True,
) -> dict[str, Any]:
    """Simulate the clean reference of one protocol."""
    protocol = acquisition(name, voxel)
    table, gtab = gradients(name, volumes)
    source = phantom(streamlines)
    grid = source.grid(protocol)
    simulation = grid.simulate(
        gtab, protocol, CLEAN, kspace=False, clean_b0=False, chunk=chunk, truth_peaks=truth
    )
    affine = simulation.magnitude.affine
    dwi = _magnitudes(simulation)
    return {
        "protocol": protocol,
        "table": table,
        "affine": affine,
        "dwi": dwi,
        "mask": np.asarray(grid.image("mask").dataobj) > 0.5,
        "white_matter": np.asarray(grid.image("wm").dataobj) >= 0.9,
        "peaks": simulation.truth_peaks if truth else None,
        "maps": grid.microstructure(protocol, gtab) if truth else {},
        "readout": protocol.sidecar(int(simulation.magnitude.shape[1])),
    }


def grid_moved(source: Any, matrix: np.ndarray, protocol: Any, grid: Any) -> Any:
    """The phantom moved by ``matrix`` in the scanner, gridded exactly as ``grid``.

    ``grid`` already carries the protocol's oblique rotation, which TRXScan would
    apply again to the affine of ``like=``.
    """
    return source.moved(matrix).grid(protocol.replace(oblique_deg=None), like=grid)


def _magnitudes(simulation: Any) -> np.ndarray:
    """The simulated magnitudes, with the reconstruction's round-off outside the head zeroed
    (so that the files compress)."""
    dwi = np.asarray(simulation.magnitude.dataobj, dtype=np.float32)
    dwi[dwi < ROUNDOFF] = 0.0
    return dwi


def _label(value: float) -> str:
    """A magnitude as a BIDS label: 0.5 is ``0p5``."""
    return f"{value:g}".replace(".", "p")


def eddy_terms(grid: Any, protocol: Any, table: np.ndarray) -> dict[str, Any]:
    """The closed-form displacement of TRXScan's linear eddy currents, per unit strength.

    With no decay over the readout, TRXScan shifts the signal of volume ``v``
    along the phase-encoding axis (j) by ``-TRT · strength · g_v · c`` voxels,
    where ``g_v`` is the prescribed b-vector times the b-value (its world
    components, applied to the grid axes as given) and ``c`` is the voxel's
    position relative to voxel ``(nx // 2, ny // 2, nz // 2)``, the integer halves
    as TRXScan's k-space indexing takes them. As a
    :class:`~nifreeze.analysis.sensitivity.Distortion` (which pulls, in mm, with
    one term per voxel axis scaled by the voxel size), the coefficients of volume
    ``v`` are ``TRT · strength · voxel_j · (0, g_x, g_y, g_z)``. The image itself
    is formed line by line: the ``n``-th phase-encoding line, ``ky = ny // 2 - n``, is
    acquired at ``times[n]`` (s, from the echo), when the field has added its phase.
    """
    affine = np.asarray(grid.affine, dtype=float)
    shape = np.asarray(grid.dims)
    voxels = np.linalg.norm(affine[:3, :3], axis=0)
    readout = protocol.sidecar(int(shape[1]))["TotalReadoutTime"]
    times = np.asarray(protocol.readout(int(shape[0]), int(shape[1])).t_ms) / 1000.0
    drive = table[:, :3] * table[:, 3:4]
    unit = readout * voxels[1] * np.column_stack((np.zeros(len(table)), drive))
    mask = np.asarray(grid.image("mask").dataobj) > 0.5
    middle = shape // 2
    offsets = np.argwhere(mask) - middle
    return {
        "readout": readout,
        "times": times,
        "center": affine[:3, :3] @ middle + affine[:3, 3],
        "radius": voxels,
        "frame": affine[:3, :3] / voxels,
        "unit": unit,
        # Largest displacement (mm) in the brain, over all volumes, at unit strength
        "reach": float(np.abs(unit[:, 1:] @ offsets.T).max()),
    }


def write_reference(
    name: str,
    out: Path,
    streamlines: int = 400_000,
    seed: int = 0,
    voxel: float | None = None,
    chunk: int | None = 8,
    volumes: list[int] | None = None,
) -> Path:
    """Write the clean and noisy datasets of one protocol, with their truth.

    With ``volumes``, the datasets keep only those volumes of the protocol.
    """
    entry = CATALOG[name]
    result = simulate(name, streamlines, voxel=voxel, volumes=volumes, chunk=chunk)
    dwi, affine, table = result["dwi"], result["affine"], result["table"]
    bzero = dwi[..., table[:, 3] < 50].mean(axis=-1)
    white = result["white_matter"] & result["mask"]
    sigma = noise_sigma(bzero, white, entry["snr_b0"])
    noisy = np.moveaxis(add_noise(np.moveaxis(dwi, -1, 0), sigma, seed), 0, -1)

    folder = out / f"sub-{SUBJECT}" / "dwi"
    folder.mkdir(parents=True, exist_ok=True)
    stem = folder / f"sub-{SUBJECT}_acq-{name}"
    nb.Nifti1Image(dwi, affine).to_filename(f"{stem}_rec-clean_dwi.nii.gz")
    # Stored as scanners store magnitudes; the quantization is far below the noise
    noisy_image = nb.Nifti1Image(noisy, affine)
    noisy_image.set_data_dtype(np.int16)
    noisy_image.to_filename(f"{stem}_rec-noisy_dwi.nii.gz")
    nb.Nifti1Image(result["mask"].astype(np.uint8), affine).to_filename(
        f"{stem}_desc-brain_mask.nii.gz"
    )
    (folder / f"sub-{SUBJECT}_acq-{name}_dwi.b").write_text(_table_text(name, volumes))
    if result["peaks"] is not None:
        result["peaks"].to_filename(f"{stem}_desc-peaks_truth.nii.gz")
    for key, image in result["maps"].items():
        image.to_filename(f"{stem}_desc-{key}_truth.nii.gz")

    sidecar = {
        "PhaseEncodingDirection": "j-",
        "TotalReadoutTime": result["readout"].get("TotalReadoutTime"),
        "EchoTime": echo_time(name)[0],
        "EchoTimeAssumed": echo_time(name)[1],
        "Protocol": name,
        "ProtocolSha256": entry["sha256"],
        "ProtocolVolumes": volumes,
        "Noise": {
            "SNR": entry["snr_b0"],
            "Sigma": sigma,
            "Seed": seed,
            "Keying": "numpy default_rng([seed, volume index])",
            "WhiteMatterVoxels": int(white.sum()),
        },
        "GeneratedBy": _provenance(streamlines),
    }
    Path(f"{stem}_dwi.json").write_text(json.dumps(sidecar, indent=2) + "\n")
    return folder


def simulate_moved(
    source: Any,
    protocol: Any,
    grid: Any,
    table: np.ndarray,
    volume: int,
    matrix: np.ndarray,
    strengths: list[float],
    chunk: int | None,
) -> list[np.ndarray]:
    """One volume of the phantom moved by ``matrix``: without eddy currents, then with each
    strength."""
    # The head moves in the scanner: the phantom is moved, then gridded as the reference
    moved_grid = grid_moved(source, matrix, protocol, grid)
    # The table's largest b-value comes along, as in the reference's simulation
    subset = _dipy(table[[volume, int(np.argmax(table[:, 3]))]])
    images = []
    for strength in [None, *strengths]:
        artifacts = CLEAN
        if strength is not None:
            artifacts = CLEAN.replace(eddy=strength, eddy_tau_ms=EDDY_TAU_MS)
        simulation = moved_grid.simulate(
            subset, protocol, artifacts, kspace=False, clean_b0=False, chunk=chunk
        )
        images.append(_magnitudes(simulation)[..., 0])
    return images


_WORKER: dict[str, Any] = {}
"""What a worker process of ``cells --jobs`` moves, set up once per process."""


def _start_worker(
    name: str, streamlines: int, voxel: float | None, volumes: list[int] | None, threads: int
) -> None:
    ts.set_threads(threads)
    protocol = acquisition(name, voxel)
    source = phantom(streamlines)
    _WORKER.update(
        source=source,
        protocol=protocol,
        grid=source.grid(protocol),
        table=gradients(name, volumes)[0],
    )


def _simulate_in_worker(
    volume: int, matrix: np.ndarray, strengths: list[float], chunk: int | None
) -> list[np.ndarray]:
    return simulate_moved(
        **_WORKER, volume=volume, matrix=matrix, strengths=strengths, chunk=chunk
    )


@contextmanager
def _workers(
    jobs: int, name: str, streamlines: int, voxel: float | None, volumes: list[int] | None
) -> Iterator[ProcessPoolExecutor | None]:
    """Processes that simulate moved volumes, or none for ``jobs=1``.

    They share the threads of ``RAYON_NUM_THREADS`` (all CPUs by default). They are
    spawned, not forked: TRXScan's thread pool does not survive a fork.
    """
    if jobs <= 1:
        yield None
        return
    budget = int(os.environ.get("RAYON_NUM_THREADS") or os.cpu_count() or 1)
    with ProcessPoolExecutor(
        jobs,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_start_worker,
        initargs=(name, streamlines, voxel, volumes, max(1, budget // jobs)),
    ) as pool:
        yield pool


def write_cells(
    name: str,
    out: Path,
    magnitudes: list[float],
    eddies: list[float],
    seeds: list[int],
    count: int | None = None,
    streamlines: int = 400_000,
    voxel: float | None = None,
    chunk: int | None = 8,
    volumes: list[int] | None = None,
    jobs: int = 1,
) -> Path:
    """Write the perturbed copies of one protocol's reference, with their ground truth.

    Only diffusion-weighted volumes are perturbed. For each seed and motion
    magnitude, the first ``count`` volumes of the plan's ordering (all of them by
    default) are re-simulated from the moved phantom, in the order of the plan, so
    that the copy with K moved volumes takes the first K of them. For each
    eddy-current magnitude, every diffusion-weighted volume is simulated with the
    eddy currents, and so is every moved volume. An eddy-current magnitude is the
    largest displacement of any brain voxel over all volumes, in voxels along phase
    encoding; TRXScan's single strength then distorts each volume in proportion
    to its b-value. With ``volumes``, the reference keeps only those volumes of
    the protocol, and the copies are numbered as the reference's volumes. With
    ``jobs``, the moved volumes are simulated in that many processes; the files are
    the same.
    """
    entry = CATALOG[name]
    protocol = acquisition(name, voxel)
    table, gtab = gradients(name, volumes)
    source = phantom(streamlines)
    grid = source.grid(protocol)
    weighted = np.flatnonzero(table[:, 3] >= 50)
    terms = eddy_terms(grid, protocol, table)
    # Magnitudes are in voxels along phase encoding, so they mean the same at any resolution
    strengths = {e: e * terms["radius"][1] / terms["reach"] for e in eddies}
    mask = np.asarray(grid.image("mask").dataobj) > 0.5
    affine = np.asarray(grid.affine, dtype=float)
    center = affine[:3, :3] @ np.argwhere(mask).mean(axis=0) + affine[:3, 3]

    folder = out / "derivatives" / "perturbations" / f"sub-{SUBJECT}" / "dwi"
    folder.mkdir(parents=True, exist_ok=True)
    reference = f"sub-{SUBJECT}/dwi/sub-{SUBJECT}_acq-{name}_rec-clean_dwi.nii.gz"

    def eddy(e: float, volumes: np.ndarray) -> dict[str, Any]:
        return {
            "Magnitude": e,
            "MagnitudeMm": e * float(terms["radius"][1]),
            "Strength": strengths[e],
            "TauMs": EDDY_TAU_MS,
            "TotalReadoutTime": terms["readout"],
            # Gradient ticks make the lines' spacing alternate slightly
            "ReadoutTimes": terms["times"].tolist(),
            "Distortion": {
                "Axis": 1,
                "Degree": 1,
                "Center": terms["center"].tolist(),
                "Radius": terms["radius"].tolist(),
                "Frame": terms["frame"].tolist(),
                "Coefficients": (strengths[e] * terms["unit"][volumes]).tolist(),
            },
        }

    def write(
        desc: str, data: list[np.ndarray], sources: np.ndarray, sidecar: dict[str, Any]
    ) -> None:
        stem = folder / f"sub-{SUBJECT}_acq-{name}_desc-{desc}_dwi"
        nb.Nifti1Image(np.stack(data, axis=-1), affine).to_filename(f"{stem}.nii.gz")
        sidecar = {
            "Protocol": name,
            "ProtocolSha256": entry["sha256"],
            "ProtocolVolumes": volumes,
            "Reference": reference,
            "SourceVolumes": [int(v) for v in sources],
            **sidecar,
            "GeneratedBy": _provenance(streamlines),
        }
        Path(f"{stem}.json").write_text(json.dumps(sidecar, indent=2) + "\n")

    plan = perturbation_plan(
        len(weighted),
        magnitudes,
        (len(weighted) if count is None else count,),
        seeds=seeds,
        center=center,
    )
    units = [
        (int(weighted[j]), np.asarray(matrix))
        for perturbation in plan
        for j, matrix in zip(perturbation.moved, perturbation.motions, strict=True)
    ]
    levels = [strengths[e] for e in eddies]
    with _workers(jobs, name, streamlines, voxel, volumes) as pool:
        # Every moved volume, in the order of the plan; the workers start on them at once
        results: Iterator[list[np.ndarray]] = (
            (simulate_moved(source, protocol, grid, table, v, m, levels, chunk) for v, m in units)
            if pool is None
            else pool.map(
                _simulate_in_worker,
                [v for v, _ in units],
                [m for _, m in units],
                repeat(levels),
                repeat(chunk),
            )
        )

        for e in eddies:
            artifacts = CLEAN.replace(eddy=strengths[e], eddy_tau_ms=EDDY_TAU_MS)
            simulation = grid.simulate(
                gtab, protocol, artifacts, kspace=False, clean_b0=False, chunk=chunk
            )
            distorted = _magnitudes(simulation)
            write(
                f"e{_label(e)}",
                [distorted[..., v] for v in weighted],
                weighted,
                {"Kind": "eddy", "Motion": None, "Eddy": eddy(e, weighted)},
            )

        for perturbation in plan:
            moved = weighted[list(perturbation.moved)]
            motion = {
                "Magnitude": perturbation.magnitude,
                "Seed": perturbation.seed,
                "Center": center.tolist(),
                "Matrices": [np.asarray(m).tolist() for m in perturbation.motions],
            }
            # Each moved volume without eddy currents, then with each magnitude
            images = list(islice(results, perturbation.count))
            label = f"m{_label(perturbation.magnitude)}s{perturbation.seed}"
            still = [i[0] for i in images]
            write(label, still, moved, {"Kind": "motion", "Motion": motion, "Eddy": None})
            for k, e in enumerate(eddies, start=1):
                write(
                    f"{label}e{_label(e)}",
                    [i[k] for i in images],
                    moved,
                    {"Kind": "motion+eddy", "Motion": motion, "Eddy": eddy(e, moved)},
                )
    return folder


def write_fixture(out: Path) -> Path:
    """Write the dataset that CI simulates: a reference and its perturbed copies."""
    settings = {k: FIXTURE[k] for k in ("streamlines", "voxel")}
    volumes = spread(FIXTURE["name"], FIXTURE["directions"])
    write_reference(FIXTURE["name"], out, **settings, volumes=volumes)
    perturbations = {k: FIXTURE[k] for k in ("magnitudes", "eddies", "seeds")}
    write_cells(FIXTURE["name"], out, **perturbations, **settings, volumes=volumes)
    return out


def _provenance(streamlines: int) -> dict[str, Any]:
    return {
        "Name": "nifreeze tools/phantoms/phantom.py",
        "NiFreezeVersion": version("nifreeze"),
        "TRXScanVersion": version("trxscan"),
        "TRXScanCommit": TRXSCAN_COMMIT,
        "Phantom": f"sub-{SUBJECT}",
        "Streamlines": streamlines,
        "StreamlineSeed": PHANTOM_SEED,
    }


def verify(name: str, voxel: float = 4.0, streamlines: int = 50_000) -> list[str]:
    """Check, at a coarse resolution, what the perturbed copies rely on."""
    table = np.loadtxt(HERE / CATALOG[name]["gradients"])
    # A few volumes of each shell, the largest b included
    picks = sorted(
        {int(i) for b in np.unique(table[:, 3]) for i in np.flatnonzero(table[:, 3] == b)[:2]}
    )
    first = simulate(name, streamlines, voxel=voxel, volumes=picks, truth=False)["dwi"]
    again = simulate(name, streamlines, voxel=voxel, volumes=picks, truth=False)["dwi"]
    subset = [picks[0], picks[-1], picks[len(picks) // 2]]
    alone = simulate(name, streamlines, voxel=voxel, volumes=subset, truth=False)["dwi"]
    positions = [picks.index(i) for i in subset]
    failures = []
    if not np.array_equal(first, again):
        failures.append("repeated runs differ")
    if not np.array_equal(alone, first[..., positions]):
        failures.append("a subset of the gradient table does not reproduce the full run")
    failures += _verify_eddy(name, voxel, streamlines, picks)
    failures += _verify_motion(name, voxel, streamlines, picks)
    failures += _verify_workers(name, voxel, streamlines, picks)
    return failures


def _verify_eddy(name: str, voxel: float, streamlines: int, picks: list[int]) -> list[str]:
    """The closed-form displacement reproduces TRXScan's eddy-current images.

    Each phase-encoding column of the clean volume is displaced as TRXScan's
    k-space model does: the signal at ``y`` accrues the phase of its eddy-current
    field ``strength · g · c`` at the time each line is acquired.
    """
    protocol = acquisition(name, voxel)
    table, gtab = gradients(name, picks)
    grid = phantom(streamlines).grid(protocol)
    terms = eddy_terms(grid, protocol, table)
    strength = 2.0 / terms["reach"]
    clean = _magnitudes(grid.simulate(gtab, protocol, CLEAN, kspace=False, clean_b0=False))
    artifacts = CLEAN.replace(eddy=strength, eddy_tau_ms=EDDY_TAU_MS)
    eddy = _magnitudes(grid.simulate(gtab, protocol, artifacts, kspace=False, clean_b0=False))
    nx, ny, nz = clean.shape[:3]
    times = terms["times"]
    lines = -(np.arange(ny) - ny // 2)
    forward = np.exp(-2j * np.pi * np.outer(lines, np.arange(ny)) / ny)
    inverse = np.exp(2j * np.pi * np.outer(np.arange(ny), lines) / ny) / ny
    worst = 0.0
    for v in np.flatnonzero(table[:, 3] >= 50):
        drive = strength * table[v, :3] * table[v, 3]
        expected = np.empty((nx, ny, nz))
        for i in range(nx):
            for k in range(nz):
                field = drive @ np.stack(
                    np.broadcast_arrays(i - nx // 2, np.arange(ny) - ny // 2, k - nz // 2)
                )
                kspace = (forward * np.exp(2j * np.pi * np.outer(times, field))) @ clean[
                    i, :, k, v
                ]
                expected[i, :, k] = np.abs(inverse @ kspace)
        worst = max(worst, float(np.abs(expected - eddy[..., v]).max() / eddy[..., v].max()))
    return [] if worst < 1e-4 else [f"eddy currents differ from their closed form ({worst:.2g})"]


def _verify_motion(name: str, voxel: float, streamlines: int, picks: list[int]) -> list[str]:
    """A moved volume's center of mass moves as its matrix says, within a fifth of a voxel."""
    protocol = acquisition(name, voxel)
    volume = [i for i in picks if gradients(name)[0][i, 3] >= 50][0]
    gtab = gradients(name, [volume])[1]
    source = phantom(streamlines)
    grid = source.grid(protocol)
    affine = np.asarray(grid.affine, dtype=float)

    def center_of_mass(image: np.ndarray) -> np.ndarray:
        ijk = np.argwhere(image > 0)
        return affine[:3, :3] @ np.average(ijk, axis=0, weights=image[image > 0]) + affine[:3, 3]

    still = _magnitudes(grid.simulate(gtab, protocol, CLEAN, kspace=False, clean_b0=False))
    origin = center_of_mass(still[..., 0])
    plan = perturbation_plan(1, (3.0,), (1,), seeds=(0,), center=origin)
    matrix = np.asarray(plan[0].motions[0])
    moved_grid = grid_moved(source, matrix, protocol, grid)
    moved = _magnitudes(moved_grid.simulate(gtab, protocol, CLEAN, kspace=False, clean_b0=False))
    error = np.linalg.norm(
        center_of_mass(moved[..., 0]) - (matrix[:3, :3] @ origin + matrix[:3, 3])
    )
    return [] if error < 0.2 * voxel else [f"a moved volume is {error:.2f} mm off its motion"]


def _verify_workers(name: str, voxel: float, streamlines: int, picks: list[int]) -> list[str]:
    """Moved volumes simulated by worker processes equal those simulated in this one."""
    protocol = acquisition(name, voxel)
    table = gradients(name, picks)[0]
    source = phantom(streamlines)
    grid = source.grid(protocol)
    # The last two, whose index in the subset is not their index in the protocol
    volumes = [int(i) for i in np.flatnonzero(table[:, 3] >= 50)[-2:]]
    matrices = [np.asarray(m) for m in perturbation_plan(2, (2.0,), (2,), seeds=(0,))[0].motions]
    levels = [2.0 / eddy_terms(grid, protocol, table)["reach"]]
    here = [
        simulate_moved(source, protocol, grid, table, v, m, levels, 8)
        for v, m in zip(volumes, matrices, strict=True)
    ]
    with _workers(2, name, streamlines, voxel, picks) as pool:
        assert pool is not None
        there = list(pool.map(_simulate_in_worker, volumes, matrices, repeat(levels), repeat(8)))
    same = all(
        np.array_equal(a, b)
        for x, y in zip(here, there, strict=True)
        for a, b in zip(x, y, strict=True)
    )
    return [] if same else ["moved volumes simulated by worker processes differ"]


def _floats(text: str) -> list[float]:
    return [float(value) for value in text.split(",")]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    reference = commands.add_parser("reference", help="simulate the clean and noisy datasets")
    reference.add_argument("--protocol", required=True, choices=sorted(CATALOG))
    reference.add_argument("--out", type=Path, required=True, help="dataset root")
    reference.add_argument("--streamlines", type=int, default=400_000)
    reference.add_argument("--seed", type=int, default=0, help="seed of the noise")
    reference.add_argument("--voxel", type=float, help="isotropic voxel size (mm), same FOV")
    reference.add_argument("--chunk", type=int, default=8, help="slices per simulated slab")
    cells = commands.add_parser("cells", help="simulate the perturbed copies")
    cells.add_argument("--protocol", required=True, choices=sorted(CATALOG))
    cells.add_argument("--out", type=Path, required=True, help="dataset root")
    cells.add_argument("--magnitudes", type=_floats, default=[0.5, 1.0, 2.0, 4.0])
    cells.add_argument(
        "--eddies",
        type=_floats,
        default=[2.0, 4.0, 8.0],
        help="largest eddy-current displacements, in voxels along phase encoding",
    )
    cells.add_argument("--seeds", type=lambda text: [int(v) for v in text.split(",")], default=[0])
    cells.add_argument("--count", type=int, help="moved volumes (all by default)")
    cells.add_argument("--streamlines", type=int, default=400_000)
    cells.add_argument("--voxel", type=float, help="isotropic voxel size (mm), same FOV")
    cells.add_argument("--chunk", type=int, default=8, help="slices per simulated slab")
    cells.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="processes simulating moved volumes, sharing the threads of RAYON_NUM_THREADS",
    )
    check = commands.add_parser("verify", help="check what the perturbed copies rely on")
    check.add_argument("--protocol", choices=sorted(CATALOG), help="one protocol (all by default)")
    fixture = commands.add_parser("fixture", help="simulate the small dataset of CI")
    fixture.add_argument("--out", type=Path, required=True, help="dataset root")
    args = parser.parse_args(argv)

    if args.command == "reference":
        folder = write_reference(
            args.protocol, args.out, args.streamlines, args.seed, args.voxel, args.chunk
        )
        print(f"wrote {folder}")
        return 0
    if args.command == "cells":
        folder = write_cells(
            args.protocol,
            args.out,
            args.magnitudes,
            args.eddies,
            args.seeds,
            args.count,
            args.streamlines,
            args.voxel,
            args.chunk,
            jobs=args.jobs,
        )
        print(f"wrote {folder}")
        return 0
    if args.command == "fixture":
        print(f"wrote {write_fixture(args.out)}")
        return 0
    names = sorted(CATALOG) if args.protocol is None else [args.protocol]
    failures = [f"{name}: {failure}" for name in names for failure in verify(name)]
    for failure in failures:
        print(f"FAIL: {failure}")
    print("OK" if not failures else f"{len(failures)} check(s) failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
