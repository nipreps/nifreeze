# /// script
# requires-python = ">=3.13"
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
"""Does an eddy-current pass with 9 parameters do better than the one with 12? (#723, Test 3)

The copy of Test 1 (``m2s0e2``, with the end-to-end test's noise) is registered by each
arm (:data:`ARMS`), with one of three registrations:

``e12``
    elastix, :data:`~nifreeze.registration.base.AFFINE` (today's pass).
``i12``
    ITK's registration framework, the ``itk`` section of ``AFFINE`` without its
    restriction: a full affine, the control that separates the backend from the model.
``i9``
    ITK, the ``itk`` section as packaged: head motion followed by a field along the
    phase-encoding direction (:class:`~nifreeze.registration.itk_backend.ITK`).

The arms:

``oracle-<registration>``, ``oracle-<registration>-clean``
    Each volume registered to its noise-free original, from the true head motion ``M``,
    as Test 1's oracle (``-clean``: the noise-free copy is registered).
``pass-<registration>-u``
    The eddy-current pass (:class:`~nifreeze.estimator.Estimator`, gp-loo,
    ``estimates="eddy"``, one thread per registration) from the cascade's head motion
    ``m̂``, with the instant update: as Test 1's ``pass``.
``pass-<registration>``
    The same pass with the delayed update: its model is built on the copy realigned by
    ``m̂``, which the pass leaves as it is (a model built on another dataset keeps reading
    it), so its estimates apply only after the pass. Its targets are Test 1's
    ``cascade-corrected``.

The phase-encoding direction is the copy's (its distortion's axis, as BIDS writes it).
``m̂`` is Test 1's (``--cascade``, Test 1's ``.npz``) or this script's (the cascade of
gqi-krr then gp-loo, rigid, elastix, as Test 1 runs it). Each arm, and the cascade, runs
in a process of its own: once the ITK backend has run in a process, elastix's estimates
and times change there (#731).

Each row has Test 1's scores (:func:`eddy_targets._scores`, with the frame of the arm's
targets: the cascade's common offset for the passes, none for the oracle), and:

``volume_change``, ``strain_pe``, ``strain_across``
    The residual ``R = W F^-1 T^-1`` (``F`` the frame, ``T = E M`` the truth), as Test 2
    measures it: ``det(L)^(1/3) - 1`` of its linear part ``L``, and the normal strain of
    ``L``'s symmetric polar factor along the phase-encoding direction and (the mean of the
    two) across it. Positive: the estimate stretches the volume.
``family``
    The largest displacement within the brain (mm) between ``W F^-1`` and its split
    (:func:`~nifreeze.registration.utils.split_phase_encoding`) composed back: zero when
    the estimate is head motion followed by a field along phase encoding.
``motion_mean``, ``field_mean``
    ``W F^-1``'s split, ``Ê M̂``, against the truth's, ``E' M'`` (in which the field's
    shift along phase encoding is head motion): the head motion, ``M̂`` against ``M'``,
    and the field at the true head position, ``Ê M'`` against ``T`` (mean, mm).
``seconds``
    The registration's time, one thread (the oracle; the passes' times are the JSON's).
``subset``
    ``dev`` or ``confirm`` (:func:`experiments.subset`).

Writes ``<out>/<protocol>.{tsv,json,npz}``, and each arm's estimates to
``<out>/arms/<protocol>-<arm>.npz`` (an arm already written is not run again). With 6
cores::

    OMP_NUM_THREADS=6 uv run tools/registration/eddy_targets_model.py --dataset <dataset> \\
        --out <out> --cascade <Test 1's sherbrooke.npz> --jobs 6 --threads 6
"""

import argparse
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.metadata import version
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter
from typing import Any

import nibabel as nb
import numpy as np
from eddy_targets import (
    CASCADE,
    MAGNITUDE,
    NOISE,
    SEED,
    _cascade,
    _common,
    _dataset_commit,
    _directions,
    _distances,
    _order,
    _scores,
    _truth,
    _truth_check,
)
from experiments import provenance, subset, write_results
from scipy.linalg import polar

from nifreeze.analysis.phantoms import load_phantom
from nifreeze.estimator import Estimator
from nifreeze.model.base import ModelFactory
from nifreeze.registration.base import AFFINE, RegistrationFactory
from nifreeze.registration.itk_backend import get_settings
from nifreeze.registration.utils import phase_encoding_vector, split_phase_encoding

EDDY_MM = {"sherbrooke": 4.0, "hcpya": 2.5}
"""Largest displacement (mm) of each protocol's eddy currents (``desc-e2``: 2 voxels)."""
MODEL = "gp-loo"
"""The pass's model, as Test 1's decision."""
BACKENDS = {"e12": "elastix", "i12": "itk", "i9": "itk"}
"""The backend of each registration."""
ARMS = (
    "oracle-e12",
    "oracle-e12-clean",
    "oracle-i12",
    "oracle-i9",
    "oracle-i9-clean",
    "pass-e12-u",
    "pass-e12",
    "pass-i12-u",
    "pass-i9-u",
    "pass-i9",
)
"""Every arm, in the order they run."""
TRUTH_TOLERANCE = 1e-9
"""Largest difference (mm) between the matrices and the sidecar's field."""

_SHARED: dict[str, Any] = {}
"""An oracle arm's data, set before its workers fork (copy on write)."""


def _arm(name: str) -> tuple[str, str, bool]:
    """An arm's kind (``oracle`` or ``pass``), its registration, and its flag."""
    kind, registration, *flag = name.split("-")
    return kind, registration, bool(flag)


def _settings(registration: str) -> Any:
    """A registration's settings."""
    if registration == "i12":
        settings = get_settings()
        settings.pop("restrict")
        return settings
    return AFFINE


def _backend(registration: str, phase_encoding: str) -> Any:
    options = {"phase_encoding": phase_encoding} if BACKENDS[registration] == "itk" else {}
    return RegistrationFactory.init(
        BACKENDS[registration],
        align_config=_settings(registration),
        seed=SEED,
        num_threads=1,
        **options,
    )


def _register(index: int) -> tuple[int, np.ndarray, float, bool]:
    """Register volume ``index`` to its noise-free original, from its true head motion."""
    copy, start = _SHARED["copy"], np.array(_SHARED["motion"][index], dtype=float)
    registration = _backend(_SHARED["registration"], copy.phase_encoding)
    began = perf_counter()
    try:
        estimate = registration(
            _SHARED["targets"][index], copy[index][0], copy.affine, copy.brainmask, index, start
        )
    except RuntimeError:  # elastix's, such as samples mapped outside the moving image
        return index, start, perf_counter() - began, True
    return index, np.asarray(estimate.matrix), perf_counter() - began, False


def _warm(index: int) -> None:
    """Load the backend's modules in a worker, after the fork, outside the timings."""
    _register(index)


def _choose(cells: list, n_volumes: int, eddy_mm: float) -> tuple[Any, Any]:
    """The copy with every volume moved, and with eddy currents of ``eddy_mm``."""
    chosen = [
        (perturbation, load)
        for perturbation, load in cells
        if perturbation.magnitude == MAGNITUDE
        and perturbation.seed == 0
        and perturbation.count == n_volumes
        and perturbation.distortion is not None
        and perturbation.distortion.magnitude == eddy_mm
    ]
    if len(chosen) != 1:
        raise RuntimeError(f"{len(chosen)} copies with head motion and eddy currents, not 1")
    return chosen[0]


def _load(args: argparse.Namespace, noisy: bool = True) -> tuple[Any, Any, Any, str]:
    """The noisy reference, the copy's perturbation and loader, and its PE direction."""
    reference, cells = load_phantom(args.dataset, args.protocol, noise=NOISE if noisy else None)
    perturbation, load = _choose(cells, len(reference), EDDY_MM[args.protocol])
    return reference, perturbation, load, "ijk"[perturbation.distortion.axis]


def _path(args: argparse.Namespace, arm: str) -> Path:
    return args.out / "arms" / f"{args.protocol}-{arm}.npz"


def run_cascade(args: argparse.Namespace) -> None:
    """The cascade, as Test 1 runs it; its head motion by level."""
    _, _, load, _ = _load(args)
    seconds: dict[str, float] = {}
    run = {"omp_nthreads": args.threads, "n_jobs": args.threads, "seed": SEED}
    levels = _cascade(load(), run, seconds)
    arrays: dict[str, Any] = {f"cascade/{model}": value for model, value in levels.items()}
    arrays["seconds"] = np.array([seconds[f"cascade/{model}"] for model in CASCADE])
    np.savez_compressed(_path(args, "cascade"), **arrays)


def run_arm(args: argparse.Namespace, arm: str, cascade: np.ndarray | None) -> None:
    """Run one arm and write its estimates."""
    kind, registration, flag = _arm(arm)
    began = perf_counter()
    if kind == "oracle":
        reference, perturbation, load, pe = _load(args, noisy=not flag)
        copy = load()
        copy.phase_encoding = pe
        clean, _ = load_phantom(args.dataset, args.protocol, counts=[])
        volumes = list(range(len(reference)))
        motion = _truth(perturbation, len(reference))["motion"]
        targets = {i: np.asarray(clean.dataobj[i], dtype="float32") for i in volumes}
        _SHARED.update(copy=copy, motion=motion, targets=targets, registration=registration)
        estimates = np.repeat(np.eye(4)[np.newaxis], len(volumes), axis=0)
        seconds, failed = np.zeros(len(volumes)), np.zeros(len(volumes), dtype=bool)
        with ProcessPoolExecutor(
            args.jobs, mp_context=get_context("fork"), initializer=_warm, initargs=(volumes[0],)
        ) as pool:
            for future in as_completed([pool.submit(_register, i) for i in volumes]):
                index, estimate, took, broke = future.result()
                estimates[index], seconds[index], failed[index] = estimate, took, broke
        extra = {}
    else:
        assert cascade is not None
        _, _, load, pe = _load(args)
        data = load()
        data.phase_encoding = pe
        data.motion = cascade.copy()
        # Delayed update: a model built on the copy realigned by m̂ keeps reading that view
        model = MODEL if flag else ModelFactory.init(model=MODEL, dataset=data.realigned())
        Estimator(
            model,
            backend=BACKENDS[registration],
            estimates="eddy",
            align_config=_settings(registration),
        ).run(data, omp_nthreads=args.threads, n_jobs=args.threads, seed=SEED, num_threads=1)
        estimates = np.stack([np.asarray(data.get_transform(i)) for i in range(len(data))])
        seconds = np.full(len(data), np.nan)
        failed = np.zeros(len(data), dtype=bool)
        extra = {"motion": np.asarray(data.motion), "eddy": np.asarray(data.eddy)}
    arrays: dict[str, Any] = {
        "estimates": estimates,
        "seconds": seconds,
        "failed": failed,
        "wall": perf_counter() - began,
        **extra,
    }
    np.savez_compressed(_path(args, arm), **arrays)
    print(f"{arm}: {perf_counter() - began:.0f} s", flush=True)


def _residual(estimate: np.ndarray, truth: np.ndarray, u: np.ndarray) -> dict[str, float]:
    """The residual's volume change, and its strains along and across phase encoding."""
    linear = (estimate @ np.linalg.inv(truth))[:3, :3]
    _, symmetric = polar(linear)
    along = float(u @ symmetric @ u) - 1
    return {
        "volume_change": float(np.cbrt(np.linalg.det(linear)) - 1),
        "strain_pe": along,
        "strain_across": float((np.trace(symmetric) - 3 - along) / 2),
    }


def _split_scores(
    estimate: np.ndarray,
    truth: np.ndarray,
    truth_motion: np.ndarray,
    u: np.ndarray,
    mask: nb.Nifti1Image,
) -> dict[str, float]:
    """The estimate's distance to its family, and its split against the truth's."""
    motion, field = split_phase_encoding(estimate, u)
    return {
        "family": float(_distances(mask, field @ motion, estimate).max()),
        "motion_mean": float(_distances(mask, motion, truth_motion).mean()),
        "field_mean": float(_distances(mask, field @ truth_motion, truth).mean()),
    }


def _arm_rows(
    arm: str, stored: Any, context: dict[str, Any]
) -> tuple[list[dict[str, Any]], float | None]:
    """An arm's rows, and, for a pass with 9 parameters, how far its stored field is from
    its estimates' (mm): the pass must store the split."""
    kind, registration, flag = _arm(arm)
    truth, u, mask = context["truth"], context["u"], context["mask"]
    frame = context["frames"][kind]
    scores = _scores(
        dict(enumerate(stored["estimates"])),
        context["starts"][kind],
        truth,
        mask,
        context["points"],
        frame,
    )
    rows = []
    for index, estimate in enumerate(stored["estimates"]):
        in_frame = estimate @ np.linalg.inv(frame)
        whole = truth["whole"][index]
        rows.append(
            {
                "arm": arm,
                "kind": kind,
                "registration": registration,
                "instant_update": bool(flag) if kind == "pass" else None,
                "noisy": not flag if kind == "oracle" else True,
                "index": index,
                "shell": float(context["shells"][index]),
                "rank": context["rank"][index],
                "direction": int(context["directions"][index]),
                "subset": context["which"].get(index, "other"),
                **scores[index],
                **_residual(in_frame, whole, u),
                **_split_scores(in_frame, whole, context["splits"][index][0], u, mask),
                "seconds": float(stored["seconds"][index]),
                "failed": bool(stored["failed"][index]),
            }
        )
    if kind != "pass" or registration != "i9":
        return rows, None
    gaps = [
        float(_distances(mask, field, split_phase_encoding(estimate, u)[1]).max())
        for field, estimate in zip(stored["eddy"], stored["estimates"], strict=True)
    ]
    return rows, max(gaps)


def _checks(estimates: dict[str, Any], test1: Path | None, mask: nb.Nifti1Image, first: int):
    """Reproductions of Test 1, and the pairing of each pass on its first volume."""
    checks: dict[str, Any] = {}
    if test1 is not None:
        reference = np.load(test1)
        for arm, recorded in (
            ("oracle-e12", "oracle/none"),
            ("pass-e12-u", "pass/gp-loo"),
            ("pass-e12", "cascade-corrected/gp-loo"),
        ):
            if arm not in estimates:
                continue
            indices = reference[f"{recorded}/indices"]
            gaps = [
                float(_distances(mask, estimates[arm][i], reference[recorded][k]).max())
                for k, i in enumerate(indices)
            ]
            checks[f"Test1/{arm}"] = {
                "identical": bool(np.array_equal(estimates[arm][indices], reference[recorded])),
                "median_mm": float(np.median(gaps)),
                "max_mm": float(np.max(gaps)),
            }
    for registration in BACKENDS:
        pair = [f"pass-{registration}-u", f"pass-{registration}"]
        if all(arm in estimates for arm in pair):
            checks[f"FirstVolumeMm/{registration}"] = float(
                _distances(mask, estimates[pair[0]][first], estimates[pair[1]][first]).max()
            )
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", type=Path, required=True, help="phantom dataset root")
    parser.add_argument("--protocol", default="sherbrooke", choices=sorted(EDDY_MM))
    parser.add_argument("--out", type=Path, required=True, help="results folder")
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    parser.add_argument("--cascade", type=Path, help="Test 1's .npz: its cascade and checks")
    parser.add_argument("--jobs", type=int, default=1, help="registrations at a time (oracle)")
    parser.add_argument("--threads", type=int, default=1, help="of the cascade and passes")
    parser.add_argument("--run-arm", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    (args.out / "arms").mkdir(parents=True, exist_ok=True)

    cascade_path = args.cascade or _path(args, "cascade")
    if args.run_arm == "cascade":
        run_cascade(args)
        return 0
    if args.run_arm is not None:
        cascade = None
        if args.run_arm.startswith("pass"):
            cascade = np.load(cascade_path)[f"cascade/{CASCADE[-1]}"]
        run_arm(args, args.run_arm, cascade)
        return 0

    commit = _dataset_commit(args.dataset)
    wall = perf_counter()
    reference, perturbation, _, pe = _load(args)
    n_volumes = len(reference)
    shells = np.round(reference.gradients[:, -1], -2)
    mask_data = np.asanyarray(reference.brainmask, dtype=bool)
    mask = nb.Nifti1Image(mask_data.astype("uint8"), reference.affine)
    points = nb.affines.apply_affine(reference.affine, np.argwhere(mask_data)).T
    truth = _truth(perturbation, n_volumes)
    u = phase_encoding_vector(pe, reference.affine)
    splits = [split_phase_encoding(whole, u) for whole in truth["whole"]]
    checks: dict[str, Any] = {
        "TruthMm": _truth_check(perturbation.distortion, truth["motion"], truth["eddy"], points),
        "TruthInFamily": max(
            float(np.abs(field @ motion - whole).max())
            for (motion, field), whole in zip(splits, truth["whole"], strict=True)
        ),
        "PhaseEncoding": pe,
        "PhaseEncodingDirection": u.tolist(),
    }
    print(
        f"loaded; truth {checks['TruthMm']:.1e} mm, in the family to "
        f"{checks['TruthInFamily']:.1e}; PE {pe} ({u.round(3).tolist()})",
        flush=True,
    )
    if checks["TruthMm"] > TRUTH_TOLERANCE or checks["TruthInFamily"] > TRUTH_TOLERANCE:
        raise RuntimeError("the matrices miss the sidecar's field, or its family")

    # Each arm, and the cascade, in a process of its own
    command = [
        sys.executable,
        __file__,
        "--dataset",
        str(args.dataset),
        "--protocol",
        args.protocol,
        "--out",
        str(args.out),
        "--jobs",
        str(args.jobs),
        "--threads",
        str(args.threads),
    ]
    if args.cascade is not None:
        command += ["--cascade", str(args.cascade)]
    for arm in (["cascade"] if args.cascade is None else []) + args.arms:
        if not _path(args, arm).exists():
            subprocess.run([*command, "--run-arm", arm], check=True)
    cascade = np.load(cascade_path)[f"cascade/{CASCADE[-1]}"]

    order = _order(n_volumes)
    rank = {index: position for position, index in enumerate(order)}
    directions = _directions(reference.gradients)
    which = dict.fromkeys(subset(args.protocol, "dev", n_volumes), "dev")
    which |= dict.fromkeys(subset(args.protocol, "confirm", n_volumes), "confirm")
    context = {
        "truth": truth,
        "u": u,
        "splits": splits,
        "mask": mask,
        "points": points,
        "shells": shells,
        "rank": rank,
        "directions": directions,
        "which": which,
        "frames": {"oracle": np.eye(4), "pass": _common(cascade, truth["motion"])},
        "starts": {"oracle": truth["motion"], "pass": cascade},
    }
    rows, estimates, walls = [], {}, {}
    for arm in args.arms:
        stored = np.load(_path(args, arm))
        estimates[arm], walls[arm] = stored["estimates"], float(stored["wall"])
        arm_rows, stored_split = _arm_rows(arm, stored, context)
        rows += arm_rows
        if stored_split is not None:
            checks[f"StoredSplitMm/{arm}"] = stored_split
        median = np.median([row["whole_frame_mean"] for row in arm_rows])
        print(f"{arm}: median error {median:.3f} mm ({walls[arm]:.0f} s)", flush=True)

    checks |= _checks(estimates, args.cascade, mask, order[0])
    table, _ = write_results(
        args.out,
        args.protocol,
        rows,
        {
            "Protocol": args.protocol,
            "Copy": {"Magnitude": MAGNITUDE, "Seed": 0, "EddyMm": EDDY_MM[args.protocol]},
            "VoxelSize": np.linalg.norm(reference.affine[:3, :3], axis=0).tolist(),
            # The brain's mean distance to its centroid, as Test 2's
            "MeanRadiusMm": float(
                np.linalg.norm(points - points.mean(axis=1, keepdims=True), axis=0).mean()
            ),
            "Noise": NOISE,
            "Seed": SEED,
            "Model": MODEL,
            "Cascade": str(args.cascade) if args.cascade else list(CASCADE),
            "Arms": args.arms,
            "Order": order,
            "Jobs": args.jobs,
            "Threads": args.threads,
            "ArmSeconds": walls,
            "WallSeconds": perf_counter() - wall,
            "Checks": checks,
            "DatasetCommit": commit,
            "DatasetCommitAtEnd": _dataset_commit(args.dataset),
            "Provenance": provenance() | {"ItkElastix": version("itk-elastix")},
        },
    )
    arrays: dict[str, Any] = {**estimates, "cascade": cascade, "order": np.asarray(order)}
    arrays |= {f"truth/{name}": value for name, value in truth.items()}
    np.savez_compressed(args.out / f"{args.protocol}.npz", **arrays)
    print(f"wrote {table}; checks {checks}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
