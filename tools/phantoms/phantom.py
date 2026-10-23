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
are not released yet). Run it with ``uv run``, which builds that environment::

    uv run tools/phantoms/phantom.py reference --protocol sherbrooke --out <dataset>
    uv run tools/phantoms/phantom.py verify --protocol sherbrooke

``reference`` simulates the phantom under one protocol of ``protocols.toml``,
clean: no motion, eddy currents, susceptibility distortion, readout relaxation
or noise; T2 contrast comes from per-tissue S0 at the protocol's echo time. It
writes the noise-free magnitude, one noisy realization at the protocol's SNR,
the brain mask, the gradients, the truth maps and a JSON sidecar, under
``<dataset>/sub-60501/dwi``. ``verify`` checks, at a coarse resolution, what the
perturbed datasets rely on: repeated runs are identical, and a subset of the
gradient table reproduces the same volumes of the full run.
"""

import argparse
import json
import math
import sys
import tomllib
from importlib.metadata import version
from pathlib import Path
from typing import Any

import nibabel as nb
import numpy as np
import trxscan as ts
from dipy.core.gradients import gradient_table

from nifreeze.testing.phantoms import add_noise, noise_sigma

HERE = Path(__file__).parent
CATALOG = tomllib.loads((HERE / "protocols.toml").read_text())
TRXSCAN_COMMIT = "680fc4f4559c5dd9fa759c2031f07f83384da2df"
SUBJECT = "60501"
"""TRXScan's phantom: an adult (NIBS sub-60501) with ACT+SIFT2 streamlines, CC0."""
ASSUMED_ECHO_TIME = 0.09
"""Echo time (s) for the T2 contrast of a protocol whose echo time is not documented."""
ROUNDOFF = 1e-4
"""Magnitudes below this (a millionth of TRXScan's signal scale) are FFT round-off."""


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
    return table, gradient_table(table[:, 3], bvecs=table[:, :3], b0_threshold=50)


def phantom(streamlines: int, seed: int) -> Any:
    return ts.Phantom.load(f"sub-{SUBJECT}").subsample(streamlines, seed=seed)


def simulate(
    name: str,
    streamlines: int = 400_000,
    seed: int = 0,
    voxel: float | None = None,
    volumes: list[int] | None = None,
    chunk: int | None = 8,
    truth: bool = True,
) -> dict[str, Any]:
    """Simulate the clean reference of one protocol."""
    protocol = acquisition(name, voxel)
    table, gtab = gradients(name, volumes)
    source = phantom(streamlines, seed)
    grid = source.grid(protocol)
    clean = ts.Artifacts(distortion=False, relaxation=False)
    simulation = grid.simulate(
        gtab, protocol, clean, kspace=False, clean_b0=False, chunk=chunk, truth_peaks=truth
    )
    affine = simulation.magnitude.affine
    dwi = np.asarray(simulation.magnitude.dataobj, dtype=np.float32)
    # Zero the reconstruction's round-off outside the head, so that the files compress
    dwi[dwi < ROUNDOFF] = 0.0
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


def write_reference(name: str, out: Path, streamlines: int, seed: int, chunk: int) -> Path:
    """Write the clean and noisy datasets of one protocol, with their truth."""
    entry = CATALOG[name]
    result = simulate(name, streamlines, seed, chunk=chunk)
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
    (folder / f"sub-{SUBJECT}_acq-{name}_dwi.b").write_text(
        (HERE / entry["gradients"]).read_text()
    )
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
        "Noise": {
            "SNR": entry["snr_b0"],
            "Sigma": sigma,
            "Seed": seed,
            "Keying": "numpy default_rng([seed, volume index])",
            "WhiteMatterVoxels": int(white.sum()),
        },
        "Phantom": {"Subject": SUBJECT, "Streamlines": streamlines, "SubsampleSeed": seed},
        "GeneratedBy": {
            "Name": "nifreeze tools/phantoms/phantom.py",
            "NiFreezeVersion": version("nifreeze"),
            "TRXScanVersion": version("trxscan"),
            "TRXScanCommit": TRXSCAN_COMMIT,
        },
    }
    Path(f"{stem}_dwi.json").write_text(json.dumps(sidecar, indent=2) + "\n")
    return folder


def verify(name: str, voxel: float = 4.0, streamlines: int = 50_000) -> list[str]:
    """Check, at a coarse resolution, what the perturbed datasets rely on."""
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
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    reference = commands.add_parser("reference", help="simulate the clean and noisy datasets")
    reference.add_argument("--protocol", required=True, choices=sorted(CATALOG))
    reference.add_argument("--out", type=Path, required=True, help="dataset root")
    reference.add_argument("--streamlines", type=int, default=400_000)
    reference.add_argument("--seed", type=int, default=0)
    reference.add_argument("--chunk", type=int, default=8, help="slices per simulated slab")
    check = commands.add_parser("verify", help="check determinism and subsets")
    check.add_argument("--protocol", default="sherbrooke", choices=sorted(CATALOG))
    args = parser.parse_args(argv)

    if args.command == "reference":
        folder = write_reference(args.protocol, args.out, args.streamlines, args.seed, args.chunk)
        print(f"wrote {folder}")
        return 0
    failures = verify(args.protocol)
    for failure in failures:
        print(f"FAIL: {failure}")
    print("OK" if not failures else f"{len(failures)} check(s) failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
