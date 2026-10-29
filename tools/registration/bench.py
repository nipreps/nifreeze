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
"""Registration accuracy and time on one protocol of a phantom (nipreps/nifreeze#584).

For each moved set in which every diffusion-weighted volume moves (TRXScan's
``tools/phantoms/phantom.py cells``, without eddy currents), each evaluated volume
is registered to the noise-free reference volume (``oracle``) and to a model's
held-out predictions from the reference (``unmoved``) and from the moved set
(``moved``), with the protocol's noise added on load
(:func:`nifreeze.analysis.registration.registration_accuracy`). It runs in
NiFreeze's environment, never TRXScan's::

    uv run tools/registration/bench.py --dataset <dataset> --protocol sherbrooke \\
        --out <dataset>/derivatives/registration

and writes, under ``<out>/<protocol>``, one row per volume and fixed image: for each
magnitude, the oracle's (``desc-m<magnitude>_oracle.tsv``) and each model's
(``desc-m<magnitude>_model-<model>.tsv``), and the settings (``settings.json``).
"""

import argparse
import csv
import json
import tempfile
from importlib.metadata import version
from pathlib import Path
from typing import Any

from nifreeze.analysis.phantoms import load_phantom
from nifreeze.analysis.registration import FIXED, AntsRegistration, registration_accuracy

MODELS = ("avgdwi", "gp-loo", "gqi-krr")
"""The shell median and the two closed-form smoothers."""
NOISE = 20261029
"""Seed of the noise added on load."""


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        if not rows:
            return
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", type=Path, required=True, help="phantom dataset root")
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--out", type=Path, required=True, help="results root")
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    parser.add_argument("--magnitudes", nargs="+", type=float, help="all if not given")
    parser.add_argument("--every", type=int, default=1, help="evaluate every n-th volume")
    parser.add_argument("--threads", type=int, default=8, help="threads of each registration")
    args = parser.parse_args(argv)

    clean, _ = load_phantom(args.dataset, args.protocol, counts=[])
    reference, cells = load_phantom(args.dataset, args.protocol, noise=NOISE)
    chosen = [
        (perturbation, load)
        for perturbation, load in cells
        if perturbation.distortion is None
        and perturbation.count == len(reference)
        and (args.magnitudes is None or perturbation.magnitude in args.magnitudes)
    ]
    evaluate = list(range(0, len(reference), args.every))
    out = args.out / args.protocol
    out.mkdir(parents=True, exist_ok=True)
    (out / "settings.json").write_text(
        json.dumps(
            {
                "Protocol": args.protocol,
                "Models": args.models,
                "Fixed": list(FIXED),
                "Noise": NOISE,
                "Magnitudes": sorted(p.magnitude for p, _ in chosen),
                "Evaluate": evaluate,
                "Registration": "ants",
                "Threads": args.threads,
                "NiFreezeVersion": version("nifreeze"),
            },
            indent=2,
        )
        + "\n"
    )

    for perturbation, load in chosen:
        moved = load()
        motions = dict(zip(perturbation.moved, perturbation.motions, strict=True))
        truth = {index: motions[index] for index in evaluate}
        desc = f"desc-m{perturbation.magnitude:g}".replace(".", "p")
        # The oracle does not depend on the model
        for model in [None, *args.models]:
            with tempfile.TemporaryDirectory(prefix="nifreeze-bench-") as workdir:
                rows = registration_accuracy(
                    AntsRegistration(workdir, num_threads=args.threads),
                    clean,
                    reference,
                    moved,
                    truth,
                    model=model,
                    fixed=["oracle"] if model is None else FIXED[1:],
                )
            path = out / (f"{desc}_oracle.tsv" if model is None else f"{desc}_model-{model}.tsv")
            _write(path, rows)
            print(path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
