# /// script
# requires-python = ">=3.12"
# dependencies = ["nifreeze"]
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
"""The fixed model targets of the registration experiments (#617).

Each model's held-out prediction of each volume of the development and confirmation
subsets (``experiments.subset``) is written once, so that every experiment registers
to the same fixed images
(:func:`~nifreeze.analysis.sensitivity.held_out_predictions`). The copies are the
noisy reference (``unmoved``) and the perturbed copies of :data:`COPIES`, with the
noise of R1 added on load. Runs in NiFreeze's environment::

    uv run tools/registration/targets.py --dataset <dataset> --protocol sherbrooke \\
        --out <dataset>/derivatives/registration/targets

and writes, under ``<out>/<protocol>``, one HDF5 file per copy and model
(``desc-<copy>_model-<model>_pred.h5``) with the predictions within the brain mask,
and ``targets.{tsv,json}`` with the seconds of each and the provenance.
"""

import argparse
import resource
from pathlib import Path
from time import perf_counter
from typing import Any

import h5py
import numpy as np
from experiments import provenance, subset, write_results

from nifreeze.analysis.phantoms import load_phantom
from nifreeze.analysis.sensitivity import held_out_predictions

COPIES = ("unmoved", "m0p5s0", "m1s0", "m2s0", "m4s0", "e4", "m1s0e4", "m4s0e4")
"""The copies, by their ``desc`` in the phantom dataset (``unmoved`` is the reference)."""
MODELS = {"gp-loo": COPIES, "gqi-krr": COPIES[:5]}
"""The copies each model predicts."""
NOISE = 20261029
"""Seed of the noise added on load, as in R1 (#584) and #609."""


def _name(magnitude: float, seed: int | None, eddy: int | None) -> str:
    motion = f"m{magnitude:g}s{seed}".replace(".", "p") if magnitude else ""
    return motion + (f"e{eddy}" if eddy else "")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", type=Path, required=True, help="phantom dataset root")
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--out", type=Path, required=True, help="results root")
    parser.add_argument("--copies", nargs="+", default=list(COPIES), choices=COPIES)
    args = parser.parse_args(argv)

    reference, cells = load_phantom(args.dataset, args.protocol, noise=NOISE)
    n_volumes = len(reference)
    indices = sorted(
        subset(args.protocol, "dev", n_volumes) + subset(args.protocol, "confirm", n_volumes)
    )
    mask = np.asanyarray(reference.brainmask, dtype=bool)
    affine = np.asarray(reference.affine, dtype=float)

    loaders = {"unmoved": lambda: reference}
    for perturbation, load in cells:
        eddy = None
        if (distortion := perturbation.distortion) is not None:
            voxel = float(np.linalg.norm(affine[:3, distortion.axis]))
            eddy = round(distortion.magnitude / voxel)
        if perturbation.magnitude and len(perturbation.moved) != n_volumes:
            continue
        loaders[_name(perturbation.magnitude, perturbation.seed, eddy)] = load
    if missing := set(args.copies) - set(loaders):
        raise ValueError(f"{args.protocol} has no copies {sorted(missing)}")

    out = args.out / args.protocol
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for name in args.copies:
        dataset = loaders[name]()
        for model, copies in MODELS.items():
            if name not in copies:
                continue
            start = perf_counter()
            predictions = held_out_predictions(model, dataset, indices)
            seconds = perf_counter() - start
            stacked = np.stack([predictions[i] for i in indices])
            # The models predict within the brain mask only, so storing it loses nothing
            if np.any(stacked[:, ~mask]):
                raise RuntimeError(f"{model} predicts {name} outside the brain mask")
            path = out / f"desc-{name}_model-{model}_pred.h5"
            with h5py.File(path, "w") as h5:
                h5.create_dataset(
                    "predictions",
                    data=stacked[:, mask].astype("float32"),
                    chunks=(1, int(mask.sum())),
                    compression="gzip",
                    shuffle=True,
                )
                h5.create_dataset("indices", data=np.asarray(indices, dtype="int32"))
                h5.create_dataset("mask", data=mask.astype("uint8"), compression="gzip")
                h5.create_dataset("affine", data=affine)
                h5.attrs.update(protocol=args.protocol, copy=name, model=model, noise=NOISE)
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
            rows.append(
                {
                    "copy": name,
                    "model": model,
                    "file": path.name,
                    "volumes": len(indices),
                    "seconds": round(seconds, 1),
                    "peak_gb": round(peak, 2),
                }
            )
            print(f"{path.name}: {seconds:.0f} s, peak {peak:.1f} GB", flush=True)
        del dataset

    write_results(
        out,
        "targets",
        rows,
        {
            "Protocol": args.protocol,
            "Copies": args.copies,
            "Models": {
                model: [c for c in copies if c in args.copies] for model, copies in MODELS.items()
            },
            "ModelArguments": {},
            "Volumes": indices,
            "Noise": NOISE,
            "Provenance": provenance(),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
