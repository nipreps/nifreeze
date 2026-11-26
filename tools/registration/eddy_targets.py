# /// script
# requires-python = ">=3.13"
# dependencies = ["nifreeze[elastix]"]
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
"""Do the targets of the eddy-current pass carry the eddy currents it should correct? (#723)

The copy with head motion and eddy currents of #712's end-to-end test (``m2s0e2``,
with that test's noise) runs once through the test's cascade (gqi-krr, then gp-loo;
rigid), whose head motion is ``m̂``. Each volume is then registered with the
eddy-current pass's settings (:data:`~nifreeze.registration.base.AFFINE`, elastix,
one thread, seed 12345) to the target of each arm, from the true head motion ``M``
(``truth``) or from ``m̂`` (``cascade``):

``oracle``, ``oracle-cascade``
    The noise-free reference volume, from ``M`` and from ``m̂``.
``reference``
    The model's held-out prediction from the noisy reference (no motion, no eddy
    currents), from ``M``.
``corrected``
    The prediction from the copy realigned by its true mapping ``E M``, from ``M``.
``motion-corrected``
    The prediction from the copy realigned by ``M`` (its eddy currents left), from ``M``.
``cascade-corrected``
    The prediction from the copy realigned by ``m̂``, from ``m̂``: the pass's targets at
    its start.
``pass``
    The eddy-current pass itself (:class:`~nifreeze.estimator.Estimator`, with
    ``estimates="eddy"`` and the instant update), from ``m̂``, with one thread per
    registration.

The predictions are held out (:func:`~nifreeze.analysis.sensitivity.held_out_predictions`)
and the training data are realigned views (:meth:`~nifreeze.data.base.BaseDataset.realigned`)
with the truth or ``m̂`` set as the copy's estimates. The truth of the eddy currents is
:meth:`~nifreeze.analysis.sensitivity.Distortion.matrices` (#724).

Each row scores one registration against the truth ``E M`` (#626), as #712 scores it:

``eddy_mean``, ``eddy_max``
    The eddy currents, ``eddy @ M`` against ``E @ M``, with ``eddy`` the estimate with
    its start factored out: the mean and the largest displacement within the brain (mm).
``whole_mean``, ``whole_max``, ``whole_offset_mean``, ``whole_offset_max``
    The whole mapping against ``E M``, as it is and with the arm's common offset removed,
    as the e2e test scores it.
``whole_frame_mean``, ``whole_frame_max``
    The whole mapping against ``E M``, in the frame of its targets: the arms whose
    targets come from data realigned by ``m̂`` (:data:`CASCADE_FRAME`) have the
    cascade's common offset (``m̂`` against ``M``) removed, the others nothing.
    ``whole_frame_mean`` is the test's primary error. It measures the correction the
    outputs apply, however it splits between head motion and eddy currents, and the
    pass and its targets at the start differ only where the instant update changed them.
``kappa``
    The fraction of the volume's field that the whole mapping recovers, in the frame of
    its targets, by least squares over the brain's points: 1 if the error has no part
    along the field, 0 if the mapping corrects head motion alone.
``rank``
    The volume's position in the order the estimators visit (each gets the run's seed).
``direction``
    The volume's gradient direction, numbered: volumes of the other shells along the same
    direction share it.

The head motion of each level of the cascade against ``M`` goes to
``<out>/<protocol>-cascade.tsv``. Writes ``<out>/<protocol>.{tsv,json,npz}``, and exits
with 1 if a check fails (after writing). With 6 cores::

    OMP_NUM_THREADS=6 uv run tools/registration/eddy_targets.py --dataset <dataset> \\
        --out <out> --jobs 6 --threads 6
"""

import argparse
import gc
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.metadata import version
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter
from typing import Any

import nibabel as nb
import nitransforms as nt
import numpy as np
from experiments import provenance, write_results
from scipy.spatial.transform import Rotation

from nifreeze.analysis.phantoms import load_phantom
from nifreeze.analysis.sensitivity import held_out_predictions
from nifreeze.estimator import Estimator
from nifreeze.registration.base import AFFINE, RegistrationFactory
from nifreeze.registration.utils import displacements_within_mask
from nifreeze.utils import iterators

PROTOCOL = "sherbrooke"
"""The protocol of #712's end-to-end test."""
MAGNITUDE = 2.0
"""Rotation (degrees) and translation (mm) of every moved volume."""
EDDY_MM = 4.0
"""Largest displacement (mm) of the eddy currents within the mask (``desc-e2``)."""
NOISE = 20261020
"""Seed of the noise, as the end-to-end test's."""
SEED = 12345
"""Seed of the registrations and of the estimators' order, as the end-to-end test's."""
CASCADE = ("gqi-krr", "gp-loo")
"""The cascade of #712's end-to-end test."""
MODELS = ("gp-loo", "gqi-krr")
"""The models of the targets; the first decides."""
TRAINING = ("reference", "corrected", "motion-corrected", "cascade-corrected")
"""The arms whose targets are predictions, by the data the model trains on."""
CASCADE_FRAME = ("cascade-corrected", "pass")
"""The arms whose targets come from data realigned by the cascade, so in its frame."""
TRUTH_TOLERANCE = 1e-9
"""Largest difference (mm) between the matrices and the sidecar's field."""
FIRST_TOLERANCE = 0.01
"""Largest difference (mm) between the pass and ``cascade-corrected`` on the first volume."""

_SHARED: dict[str, Any] = {}
"""The data of an arm, set before its workers fork (copy on write)."""


def _register(index: int) -> tuple[int, np.ndarray, float, bool]:
    """Register volume ``index`` to its target, from its start, as the estimator does."""
    copy, start = _SHARED["copy"], np.array(_SHARED["starts"][index], dtype=float)
    registration = RegistrationFactory.init(
        "elastix", align_config=AFFINE, seed=SEED, num_threads=1
    )
    began = perf_counter()
    try:
        estimate = registration(
            _SHARED["targets"][index], copy[index][0], copy.affine, copy.brainmask, index, start
        )
    except RuntimeError:  # elastix's, such as samples mapped outside the moving image
        return index, start, perf_counter() - began, True
    return index, np.asarray(estimate.matrix), perf_counter() - began, False


def _arm(
    targets: dict[int, np.ndarray], starts: np.ndarray, volumes: list[int], jobs: int
) -> dict[int, tuple[np.ndarray, float, bool]]:
    """Register every volume to its target, ``jobs`` at a time."""
    _SHARED.update(targets=targets, starts=starts)
    results = {}
    with ProcessPoolExecutor(jobs, mp_context=get_context("fork")) as pool:
        futures = [pool.submit(_register, index) for index in volumes]
        for future in as_completed(futures):
            index, estimate, seconds, failed = future.result()
            results[index] = (estimate, seconds, failed)
    _SHARED.pop("targets")
    return results


def _common(estimated: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """The transform all the estimates share (the frame of their targets), as the e2e test."""
    offsets = np.linalg.inv(truth) @ estimated
    common = np.eye(4)
    common[:3, :3] = Rotation.from_matrix(offsets[:, :3, :3]).mean().as_matrix()
    common[:3, 3] = offsets[:, :3, 3].mean(axis=0)
    return common


def _remove_offset(estimated: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Remove the transform all the estimates share."""
    return estimated @ np.linalg.inv(_common(estimated, truth))


def _distances(mask: nb.Nifti1Image, estimated: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """The displacements (mm) within the mask between two matrices."""
    return displacements_within_mask(mask, nt.linear.Affine(estimated), nt.linear.Affine(truth))


def _scores(
    estimates: dict[int, np.ndarray],
    starts: np.ndarray,
    truth: dict[str, np.ndarray],
    mask: nb.Nifti1Image,
    points: np.ndarray,
    frame: np.ndarray,
) -> dict[int, dict[str, float]]:
    """Score each volume's estimate against the truth (see the module's docstring)."""
    indices = sorted(estimates)
    estimated = np.stack([estimates[i] for i in indices])
    whole = truth["whole"][indices]
    without_offset = _remove_offset(estimated, whole)
    in_frame = estimated @ np.linalg.inv(frame)
    homogeneous = np.vstack([points, np.ones(points.shape[1])])
    scores = {}
    for k, index in enumerate(indices):
        motion = truth["motion"][index]
        # The eddy currents with the start factored out, where the head is
        eddy = estimated[k] @ np.linalg.inv(starts[index]) @ motion
        # The field, and the part of the whole mapping's error along it
        head = (motion @ homogeneous)[:3]
        field = (whole[k] @ homogeneous)[:3] - head
        recovered = (in_frame[k] @ homogeneous)[:3] - head
        energy = float(np.sum(field * field))
        scores[index] = {"kappa": float(np.sum(recovered * field) / energy) if energy else np.nan}
        for name, matrix in (
            ("eddy", eddy),
            ("whole", estimated[k]),
            ("whole_offset", without_offset[k]),
            ("whole_frame", in_frame[k]),
        ):
            errors = _distances(mask, matrix, whole[k])
            scores[index] |= {
                f"{name}_mean": float(errors.mean()),
                f"{name}_max": float(errors.max()),
            }
    return scores


def _truth_check(distortion: Any, motion: np.ndarray, eddy: np.ndarray, points: np.ndarray):
    """The largest difference (mm) between ``E M x`` and ``M x - d(M x) u`` in the brain."""
    homogeneous = np.vstack([points, np.ones(points.shape[1])])
    u = distortion.direction[:, np.newaxis]
    largest = 0.0
    for index in range(len(motion)):
        head = (motion[index] @ homogeneous)[:3]
        values, _ = distortion.basis(head)
        expected = head - u * (distortion.coefficients[index] @ values)
        mapped = (eddy[index] @ motion[index] @ homogeneous)[:3]
        largest = max(largest, float(np.abs(mapped - expected).max()))
    return largest


def _dataset_commit(dataset: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(dataset), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _volumes(shells: np.ndarray, limit: int | None, first: int) -> list[int]:
    """Every volume, or ``limit`` per shell evenly spread, and the first one visited."""
    if limit is None:
        return list(range(len(shells)))
    picked = {first}
    for shell in np.unique(shells):
        members = np.flatnonzero(shells == shell)
        picked |= set(members[np.linspace(0, len(members) - 1, limit).astype(int)].tolist())
    return sorted(picked)


def _choose(cells: list, n_volumes: int) -> tuple[Any, Any]:
    """The copy with every volume moved, and with the eddy currents of :data:`EDDY_MM`."""
    chosen = [
        (perturbation, load)
        for perturbation, load in cells
        if perturbation.magnitude == MAGNITUDE
        and perturbation.seed == 0
        and perturbation.count == n_volumes
        and perturbation.distortion is not None
        and perturbation.distortion.magnitude == EDDY_MM
    ]
    if len(chosen) != 1:
        raise RuntimeError(f"{len(chosen)} copies with head motion and eddy currents, not 1")
    return chosen[0]


def _truth(perturbation: Any, n_volumes: int) -> dict[str, np.ndarray]:
    """Each volume's head motion ``M``, its eddy currents ``E``, and ``E M``."""
    motion = np.repeat(np.eye(4)[np.newaxis], n_volumes, axis=0)
    for index, matrix in zip(perturbation.moved, perturbation.motions, strict=True):
        motion[index] = matrix
    eddy = perturbation.distortion.matrices()
    return {"motion": motion, "eddy": eddy, "whole": eddy @ motion}


def _directions(gradients: np.ndarray) -> np.ndarray:
    """Number each volume's gradient direction: the same direction in any shell shares it."""
    vectors = gradients[:, :3] / np.linalg.norm(gradients[:, :3], axis=1, keepdims=True)
    _, numbers = np.unique(np.round(vectors, 4), axis=0, return_inverse=True)
    return numbers.ravel()


def _order(n_volumes: int) -> list[int]:
    """The order every estimator visits the volumes in, drawn as the estimator draws it."""
    return list(
        iterators.random_iterator(
            size=n_volumes,
            bvals=None,
            uptake=None,
            seed=SEED,
            round_decimals=iterators.DEFAULT_ROUND_DECIMALS,
            start_index=0,
            stop_index=None,
        )
    )


def _cascade(copy: Any, run: dict[str, Any], seconds: dict[str, float]) -> dict[str, np.ndarray]:
    """
    Run the cascade one level at a time, and keep each level's head motion.

    Each level starts from the one before, on a view realigned by its estimates, as the
    levels of a chain do.
    """
    levels = {}
    for model in CASCADE:
        began = perf_counter()
        Estimator(model, backend="elastix").run(copy, **run)
        seconds[f"cascade/{model}"] = perf_counter() - began
        assert copy.motion is not None
        levels[model] = copy.motion.copy()
        print(f"cascade {model}: {seconds[f'cascade/{model}']:.0f} s", flush=True)
    copy.motion = None
    return levels


def _cascade_rows(
    levels: dict[str, np.ndarray],
    motion: np.ndarray,
    mask: nb.Nifti1Image,
    shells: np.ndarray,
    rank: dict[int, int],
) -> list[dict[str, Any]]:
    """The head motion of each level of the cascade against ``M``."""
    rows = []
    for level, (model, estimated) in enumerate(levels.items()):
        relative = _remove_offset(estimated, motion)
        for index in range(len(motion)):
            errors = _distances(mask, estimated[index], motion[index])
            offset = _distances(mask, relative[index], motion[index])
            rows.append(
                {
                    "level": level,
                    "model": model,
                    "index": index,
                    "shell": float(shells[index]),
                    "rank": rank[index],
                    "motion_mean": float(errors.mean()),
                    "motion_max": float(errors.max()),
                    "motion_offset_mean": float(offset.mean()),
                    "motion_offset_max": float(offset.max()),
                }
            )
    return rows


def _training(
    arm: str, reference: Any, load: Any, truth: dict[str, np.ndarray], cascade: np.ndarray
) -> Any:
    """The data an arm's model trains on: the reference, or the copy realigned."""
    if arm == "reference":
        return reference
    data = load()
    data.motion = (cascade if arm == "cascade-corrected" else truth["motion"]).copy()
    if arm == "corrected":
        data.eddy = truth["eddy"].copy()
    return data.realigned()


def _pass(
    model: str, load: Any, cascade: np.ndarray, run: dict[str, Any]
) -> dict[int, tuple[np.ndarray, float, bool]]:
    """The eddy-current pass, as the command line runs it after the cascade."""
    data = load()
    data.motion = cascade.copy()
    Estimator(model, backend="elastix", estimates="eddy").run(data, **run, num_threads=1)
    return {
        index: (np.asarray(data.get_transform(index)), float("nan"), False)
        for index in range(len(data))
    }


class _Record:
    """The rows and the estimates of every arm, scored against the truth."""

    def __init__(
        self,
        truth: dict[str, np.ndarray],
        starts: dict[str, np.ndarray],
        mask: nb.Nifti1Image,
        points: np.ndarray,
        shells: np.ndarray,
        rank: dict[int, int],
        directions: np.ndarray,
    ):
        self.truth, self.starts, self.mask, self.points = truth, starts, mask, points
        self.shells, self.rank, self.directions = shells, rank, directions
        self.rows: list[dict[str, Any]] = []
        self.estimates: dict[str, Any] = {}

    def __call__(self, arm: str, model: str, start: str, results: dict[int, tuple]) -> None:
        matrices = {index: result[0] for index, result in results.items()}
        # Targets from data realigned by the cascade are in its frame; the others, in truth's
        frame = (
            _common(self.starts["cascade"], self.truth["motion"])
            if arm in CASCADE_FRAME
            else np.eye(4)
        )
        scores = _scores(matrices, self.starts[start], self.truth, self.mask, self.points, frame)
        for index in sorted(results):
            _, took, failed = results[index]
            self.rows.append(
                {
                    "arm": arm,
                    "model": model,
                    "start": start,
                    "index": index,
                    "shell": float(self.shells[index]),
                    "rank": self.rank[index],
                    "direction": int(self.directions[index]),
                    **scores[index],
                    "seconds": took,
                    "failed": failed,
                }
            )
        self.estimates[f"{arm}/{model}"] = np.stack([matrices[i] for i in sorted(matrices)])
        self.estimates[f"{arm}/{model}/indices"] = np.asarray(sorted(matrices))
        whole = np.median([score["whole_offset_mean"] for score in scores.values()])
        eddy = np.median([score["eddy_mean"] for score in scores.values()])
        print(f"{arm} ({model}): median errors {whole:.3f} (whole), {eddy:.3f} (eddy)")


def _first_volume(
    estimates: dict[str, Any], models: list[str], first: int, mask: nb.Nifti1Image
) -> dict[str, float]:
    """The largest distance (mm) between the pass's and its targets' first estimates."""
    gaps = {}
    for model in models:
        pair = []
        for arm in ("pass", "cascade-corrected"):
            indices = estimates[f"{arm}/{model}/indices"].tolist()
            pair.append(estimates[f"{arm}/{model}"][indices.index(first)])
        gaps[model] = float(_distances(mask, pair[0], pair[1]).max())
    return gaps


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", type=Path, required=True, help="phantom dataset root")
    parser.add_argument("--protocol", default=PROTOCOL)
    parser.add_argument("--out", type=Path, required=True, help="results folder")
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    parser.add_argument("--limit", type=int, help="volumes per shell (pilot)")
    parser.add_argument("--jobs", type=int, default=1, help="registrations at a time")
    parser.add_argument("--threads", type=int, default=1, help="of the cascade and passes")
    args = parser.parse_args(argv)

    commit = _dataset_commit(args.dataset)
    wall = perf_counter()
    clean, _ = load_phantom(args.dataset, args.protocol, counts=[])
    reference, cells = load_phantom(args.dataset, args.protocol, noise=NOISE)
    n_volumes = len(reference)
    perturbation, load = _choose(cells, n_volumes)
    shells = np.round(reference.gradients[:, -1], -2)
    mask_data = np.asanyarray(reference.brainmask, dtype=bool)
    mask = nb.Nifti1Image(mask_data.astype("uint8"), reference.affine)
    points = nb.affines.apply_affine(reference.affine, np.argwhere(mask_data)).T

    truth = _truth(perturbation, n_volumes)
    checks: dict[str, Any] = {
        "TruthMm": _truth_check(perturbation.distortion, truth["motion"], truth["eddy"], points)
    }
    print(f"loaded in {perf_counter() - wall:.0f} s; truth {checks['TruthMm']:.2e} mm")
    if checks["TruthMm"] > TRUTH_TOLERANCE:
        raise RuntimeError(f"the matrices miss the sidecar's field by {checks['TruthMm']} mm")

    order = _order(n_volumes)
    rank = {index: position for position, index in enumerate(order)}
    run = {"omp_nthreads": args.threads, "n_jobs": args.threads, "seed": SEED}
    seconds: dict[str, float] = {}
    copy = load()
    levels = _cascade(copy, run, seconds)
    _SHARED["copy"] = copy
    cascade = levels[CASCADE[-1]]
    starts = {"truth": truth["motion"], "cascade": cascade}
    volumes = _volumes(shells, args.limit, order[0])
    directions = _directions(reference.gradients)
    record = _Record(truth, starts, mask, points, shells, rank, directions)
    record.estimates |= {"order": np.asarray(order), "volumes": np.asarray(volumes)}
    record.estimates |= {f"truth/{name}": value for name, value in truth.items()}
    record.estimates |= {f"cascade/{model}": value for model, value in levels.items()}

    # The oracle: the noise-free reference volume, from the truth and from the cascade
    oracle = {index: np.asarray(clean.dataobj[index], dtype="float32") for index in volumes}
    for arm, start in (("oracle", "truth"), ("oracle-cascade", "cascade")):
        began = perf_counter()
        record(arm, "none", start, _arm(oracle, starts[start], volumes, args.jobs))
        seconds[arm] = perf_counter() - began
    del oracle

    for model in args.models:
        for arm in TRAINING:
            began = perf_counter()
            training = _training(arm, reference, load, truth, cascade)
            targets = held_out_predictions(model, training, volumes)
            del training
            gc.collect()
            start = "cascade" if arm == "cascade-corrected" else "truth"
            record(arm, model, start, _arm(targets, starts[start], volumes, args.jobs))
            del targets
            seconds[f"{arm}/{model}"] = perf_counter() - began
        began = perf_counter()
        record("pass", model, "cascade", _pass(model, load, cascade, run))
        seconds[f"pass/{model}"] = perf_counter() - began
        gc.collect()

    checks["FirstVolumeMm"] = _first_volume(record.estimates, args.models, order[0], mask)
    rows = _cascade_rows(levels, truth["motion"], mask, shells, rank)
    write_results(args.out, f"{args.protocol}-cascade", rows, {"Protocol": args.protocol})
    table, _ = write_results(
        args.out,
        args.protocol,
        record.rows,
        {
            "Protocol": args.protocol,
            "Copy": {"Magnitude": MAGNITUDE, "Seed": 0, "EddyMm": EDDY_MM},
            "VoxelSize": np.linalg.norm(reference.affine[:3, :3], axis=0).tolist(),
            "Noise": NOISE,
            "Seed": SEED,
            "Cascade": list(CASCADE),
            "Models": args.models,
            "Volumes": volumes,
            "Limit": args.limit,
            "Order": order,
            "Settings": Path(str(AFFINE)).name,
            "Jobs": args.jobs,
            "Threads": args.threads,
            "Seconds": seconds,
            "WallSeconds": perf_counter() - wall,
            "Checks": checks,
            # Another checkout of the dataset could change it during the run
            "DatasetCommit": commit,
            "DatasetCommitAtEnd": _dataset_commit(args.dataset),
            "Provenance": provenance() | {"ItkElastix": version("itk-elastix")},
        },
    )
    np.savez_compressed(args.out / f"{args.protocol}.npz", **record.estimates)
    print(f"wrote {table}; checks {checks}", flush=True)
    failed = [model for model, gap in checks["FirstVolumeMm"].items() if gap > FIRST_TOLERANCE]
    if failed:
        print(f"the pass and its targets differ on the first volume visited: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
