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
"""The shared harness of the registration experiments on the phantoms (#615).

Each experiment of nipreps/nifreeze#614 is a script beside this module, which it
imports (``from harness import register``). The harness changes nothing in
*NiFreeze*: the estimator's path runs through its own functions, and the variants an
experiment compares are built from the estimator's default settings.

Registration
    :func:`settings_from` derives settings from a configuration of
    ``nifreeze/registration/config``, and :func:`register` runs one
    ``antsRegistration`` with them, on inputs written by :func:`write_inputs`.
    With the default settings and :data:`TODAY`'s representation, the command is
    the estimator's.
Volumes
    :func:`subset` gives the development and confirmation volumes of a protocol,
    so that settings tuned on some volumes are confirmed on others.
Comparison
    :func:`paired_comparison` compares two variants on the same volumes, and
    :func:`holm` adjusts the p values of several comparisons.
Records
    :func:`provenance` describes what ran, and :func:`write_results` writes it
    with each table.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import shutil
import subprocess
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter
from typing import Any, TypeVar

import nibabel as nb
import nitransforms as nt
import numpy as np
from nipype.interfaces.ants.registration import Registration
from scipy import stats
from scipy.spatial.transform import Rotation

import nifreeze
from nifreeze.data.filtering import robust_window
from nifreeze.registration.ants import _get_ants_settings, _prepare_registration_data
from nifreeze.registration.utils import displacements_within_mask

CLIPS = ("moving", "each", "none")
"""Intensity windows: the moving volume's for both images, one per image, or none."""
CODINGS = {"int12": 4095, "uint16": 65535, "float32": None}
"""How intensities are stored, with the highest code of the integer codings."""
TODAY = ("moving", "int12")
"""The estimator's representation of the registration inputs."""
SUBSETS = {
    "sherbrooke": {"dev": (3, 0), "confirm": (3, 1)},
    "hcpya": {"dev": (9, 0), "confirm": (9, 4)},
}
"""Every how many volumes, and from which, the development and confirmation subsets."""
THREADS = (
    "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "RAYON_NUM_THREADS",
)
"""The environment variables that set the number of threads."""
VERDICTS = ("better", "worse", "not worse", "inconclusive")
"""The outcomes of :func:`paired_comparison`."""

T = TypeVar("T")
R = TypeVar("R")


def settings_from(name: str = "b0-to-b0_level0", **overrides: Any) -> dict[str, Any]:
    """
    Settings of ``antsRegistration``, derived from one of the estimator's configurations.

    Parameters
    ----------
    name : :obj:`str`
        The configuration (a file of ``nifreeze/registration/config``).
    **overrides
        New values. A :obj:`dict` maps levels (indices, negative from the last) to
        their new values; any other value replaces the setting at every level.

    Returns
    -------
    :obj:`dict`
        The settings, as the estimator passes them to nipype's ``Registration``.

    Examples
    --------
    >>> settings_from(shrink_factors={-1: [1]})["shrink_factors"]
    [[3], [1]]
    >>> settings_from(metric=["MI", "MI"])["metric"]
    ['MI', 'MI']

    """
    settings = json.loads(_get_ants_settings(name).read_text())
    if unknown := sorted(set(overrides) - set(settings)):
        raise KeyError(f"settings not in {name}: {', '.join(unknown)}")
    for key, value in overrides.items():
        if isinstance(value, dict):
            for level, level_value in value.items():
                settings[key][level] = deepcopy(level_value)
        else:
            settings[key] = deepcopy(value)
    return settings


def _code(
    data: np.ndarray,
    affine: np.ndarray,
    path: Path,
    window: tuple[float, float] | None,
    coding: str,
) -> None:
    """Write ``data`` as ``nifreeze.registration.ants._to_nifti`` does, with any coding."""
    data = np.squeeze(data)
    levels = CODINGS[coding]
    slope_inter = None
    if levels is None:
        if window is not None:
            data = np.clip(data, *window)
        data = data.astype("float32")
    else:
        a_min, a_max = window if window is not None else (float(data.min()), float(data.max()))
        step = (a_max - a_min) / levels
        dtype = np.int16 if levels < 2**15 else np.uint16
        data = np.round((np.clip(data, a_min, a_max) - a_min) / step).astype(dtype)
        slope_inter = (step, a_min)

    nii = nb.Nifti1Image(data, affine, None)
    if slope_inter is not None:
        nii.header.set_slope_inter(*slope_inter)
    nii.header.set_sform(affine, code=1)
    nii.header.set_qform(affine, code=1)
    nii.to_filename(path)


def write_inputs(
    fixed: np.ndarray,
    moving: np.ndarray,
    affine: np.ndarray,
    mask: np.ndarray,
    workdir: Path,
    index: int = 0,
    clip: str = TODAY[0],
    coding: str = TODAY[1],
) -> tuple[Path, Path]:
    """
    Write the fixed and moving images as ``antsRegistration`` reads them.

    The estimator's representation (:data:`TODAY`), and float32 without clipping, are
    written by the estimator's own ``_prepare_registration_data``.

    Parameters
    ----------
    fixed, moving : :obj:`~numpy.ndarray`
        The images.
    affine : :obj:`~numpy.ndarray`
        Their shared affine.
    mask : :obj:`~numpy.ndarray`
        The brain mask, within which the intensity windows are computed.
    workdir : :obj:`~pathlib.Path`
        Where the images are written.
    index : :obj:`int`
        The volume's index, in the file names.
    clip : :obj:`str`
        One of :data:`CLIPS`.
    coding : :obj:`str`
        One of :data:`CODINGS`.

    Returns
    -------
    fixed_path, moving_path : :obj:`~pathlib.Path`
        The images written.

    """
    if clip not in CLIPS or coding not in CODINGS:
        raise ValueError(f"unknown representation: {clip}, {coding}")

    if (clip, coding) == TODAY:
        # The estimator's window: the moving volume's, within the mask (estimator.py)
        fixed_path, moving_path, _ = _prepare_registration_data(
            moving, fixed, affine, index, workdir, clip="both", window=robust_window(moving, mask)
        )
        return fixed_path, moving_path
    if (clip, coding) == ("none", "float32"):
        fixed_path, moving_path, _ = _prepare_registration_data(
            np.asarray(moving, dtype="float32"),
            np.asarray(fixed, dtype="float32"),
            affine,
            index,
            workdir,
        )
        return fixed_path, moving_path

    windows: dict[str, tuple[float, float] | None] = {"fixed": None, "moving": None}
    if clip == "moving":
        windows["fixed"] = windows["moving"] = robust_window(moving, mask)
    elif clip == "each":
        windows = {"fixed": robust_window(fixed, mask), "moving": robust_window(moving, mask)}

    # The estimator's file names, so that commands differ only in their settings
    fixed_path = workdir / f"predicted_{index:05d}.nii"
    moving_path = workdir / f"sample_{index:05d}.nii"
    _code(fixed, affine, fixed_path, windows["fixed"], coding)
    _code(moving, affine, moving_path, windows["moving"], coding)
    return fixed_path, moving_path


def write_mask(mask: np.ndarray, affine: np.ndarray, path: Path) -> Path:
    """Write the brain mask as the estimator does (``Estimator.run``)."""
    nb.Nifti1Image(np.asanyarray(mask).astype(np.uint8), affine, None).to_filename(path)
    return path


def command(
    settings: dict[str, Any],
    fixed_path: Path,
    moving_path: Path,
    mask_path: Path | None,
    prefix: str,
) -> str:
    """
    The ``antsRegistration`` command line, as the estimator builds it.

    The estimator's ``generate_command`` gives the same line for its default settings,
    with these paths and ``output_transform_prefix=prefix``.
    """
    masks = {}
    if mask_path is not None:
        masks["fixed_image_masks"] = [str(mask_path)] * len(settings["metric"])
    return Registration(
        fixed_image=str(Path(fixed_path).absolute()),
        moving_image=str(Path(moving_path).absolute()),
        output_transform_prefix=prefix,
        **masks,
        **settings,
    ).cmdline


def register(
    fixed: np.ndarray,
    moving: np.ndarray,
    affine: np.ndarray,
    mask: np.ndarray,
    mask_path: Path | None,
    workdir: Path,
    settings: dict[str, Any] | None = None,
    *,
    index: int = 0,
    clip: str = TODAY[0],
    coding: str = TODAY[1],
    seed: int | None = 12345,
    threads: int = 1,
) -> tuple[np.ndarray, float]:
    """
    Register ``moving`` to ``fixed`` with one ``antsRegistration`` process.

    Parameters
    ----------
    fixed, moving, affine, mask
        As in :func:`write_inputs`.
    mask_path : :obj:`~pathlib.Path` or :obj:`None`
        The brain mask written by :func:`write_mask`, for the fixed image.
    workdir : :obj:`~pathlib.Path`
        A new directory, for this registration alone.
    settings : :obj:`dict`, optional
        The settings (see :func:`settings_from`); the estimator's default if not given.
    index : :obj:`int`
        The volume's index, in the file names.
    clip, coding : :obj:`str`
        The representation of the inputs (see :func:`write_inputs`).
    seed : :obj:`int` or :obj:`None`
        Seed of ANTs' random sampling (``ANTS_RANDOM_SEED``).
    threads : :obj:`int`
        ANTs' threads (``ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS``).

    Returns
    -------
    matrix : :obj:`~numpy.ndarray`
        The estimated transform, fixed to moving, in RAS (as the estimator reads it).
    seconds : :obj:`float`
        The time, including writing the inputs and reading the output.

    """
    start = perf_counter()
    workdir.mkdir(parents=True)
    fixed_path, moving_path = write_inputs(
        fixed, moving, affine, mask, workdir, index=index, clip=clip, coding=coding
    )
    prefix = f"ants-{index:05d}"
    line = command(settings or settings_from(), fixed_path, moving_path, mask_path, prefix)
    environ = os.environ | {"ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": str(threads)}
    if seed is not None:
        environ["ANTS_RANDOM_SEED"] = str(seed)
    subprocess.run(line, shell=True, check=True, cwd=workdir, env=environ, capture_output=True)
    itk = nt.io.itk.ITKLinearTransform.from_filename(workdir / f"{prefix}0GenericAffine.mat")
    matrix = np.asarray(itk.to_ras(reference=fixed_path, moving=moving_path))
    return matrix, perf_counter() - start


def errors(mask_img: nb.Nifti1Image, estimated: np.ndarray, expected: np.ndarray) -> dict:
    """
    How far the estimate is from the truth, within the mask and at its centroid.

    Returns the ``max`` and ``mean`` displacement (mm) between the two transforms
    within the mask, the residual displacement of the mask's centroid (``tx``,
    ``ty``, ``tz``, mm) and the residual rotation vector (``rx``, ``ry``, ``rz``,
    degrees), to show any bias.
    """
    displacements = displacements_within_mask(
        mask_img, nt.linear.Affine(estimated), nt.linear.Affine(expected)
    )
    mask = np.asanyarray(mask_img.dataobj) > 0
    centroid = nb.affines.apply_affine(mask_img.affine, np.argwhere(mask).mean(axis=0))
    shift = estimated[:3, :3] @ centroid + estimated[:3, 3]
    shift -= expected[:3, :3] @ centroid + expected[:3, 3]
    rotation = Rotation.from_matrix(estimated[:3, :3] @ expected[:3, :3].T)
    return {
        "max": float(displacements.max()),
        "mean": float(displacements.mean()),
        **dict(zip(("tx", "ty", "tz"), shift.tolist(), strict=True)),
        **dict(zip(("rx", "ry", "rz"), rotation.as_rotvec(degrees=True).tolist(), strict=True)),
    }


def subset(protocol: str, which: str, n: int) -> list[int]:
    """
    The development (``"dev"``) or confirmation (``"confirm"``) volumes of a protocol.

    >>> subset("sherbrooke", "dev", 192)[:4], len(subset("sherbrooke", "dev", 192))
    ([0, 3, 6, 9], 64)
    >>> subset("hcpya", "confirm", 270)[:3], len(subset("hcpya", "confirm", 270))
    ([4, 13, 22], 30)

    """
    every, start = SUBSETS[protocol][which]
    return list(range(start, n, every))


def paired_comparison(
    candidate: Sequence[float] | np.ndarray,
    reference: Sequence[float] | np.ndarray,
    margin: float,
    *,
    strata: Sequence[Any] | np.ndarray | None = None,
    n_boot: int = 10_000,
    seed: int = 0,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """
    Compare two variants on the same volumes.

    The differences ``candidate - reference`` are paired by volume. Volumes are
    resampled with replacement (within each stratum, if given), and the percentile
    intervals of the median difference and of the difference of the 95th
    percentiles are reported, with the Wilcoxon signed-rank p value.

    The verdict, from the interval of the median difference:

    ``better``
        Below zero, with a median difference below ``-margin``.
    ``worse``
        Above ``margin``.
    ``not worse``
        Its upper bound below ``margin``.
    ``inconclusive``
        Otherwise.

    Returns
    -------
    :obj:`dict`
        ``n``, ``median``, ``median_low``, ``median_high``, ``p95``, ``p95_low``,
        ``p95_high``, ``wilcoxon_p`` and ``verdict``.

    """
    candidate = np.asarray(candidate, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if candidate.shape != reference.shape or candidate.ndim != 1:
        raise ValueError("the variants must be paired, one value per volume")
    differences = candidate - reference
    groups = (
        [np.arange(len(differences))]
        if strata is None
        else [np.flatnonzero(np.asarray(strata) == s) for s in np.unique(np.asarray(strata))]
    )

    rng = np.random.default_rng(seed)
    resampled = np.concatenate(
        [rng.choice(group, size=(n_boot, len(group)), replace=True) for group in groups], axis=1
    )
    medians = np.median(differences[resampled], axis=1)
    p95s = np.percentile(candidate[resampled], 95, axis=1) - np.percentile(
        reference[resampled], 95, axis=1
    )
    tails = (100 * (1 - confidence) / 2, 100 * (1 + confidence) / 2)
    median_low, median_high = np.percentile(medians, tails)
    p95_low, p95_high = np.percentile(p95s, tails)

    median = float(np.median(differences))
    if np.allclose(differences, 0):
        wilcoxon_p = 1.0
    else:
        wilcoxon_p = float(stats.wilcoxon(differences).pvalue)

    if median_high < 0 and median < -margin:
        verdict = "better"
    elif median_low > margin:
        verdict = "worse"
    elif median_high < margin:
        verdict = "not worse"
    else:
        verdict = "inconclusive"
    return {
        "n": len(differences),
        "median": median,
        "median_low": float(median_low),
        "median_high": float(median_high),
        "p95": float(np.percentile(candidate, 95) - np.percentile(reference, 95)),
        "p95_low": float(p95_low),
        "p95_high": float(p95_high),
        "wilcoxon_p": wilcoxon_p,
        "verdict": verdict,
    }


def holm(pvalues: Sequence[float] | np.ndarray) -> np.ndarray:
    """
    Holm's step-down adjustment of p values for several comparisons.

    >>> holm([0.01, 0.04, 0.03]).round(2).tolist()
    [0.03, 0.06, 0.06]

    """
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    steps = (len(pvalues) - np.arange(len(pvalues))) * pvalues[order]
    adjusted = np.empty_like(pvalues)
    adjusted[order] = np.minimum(np.maximum.accumulate(steps), 1.0)
    return adjusted


def _git(*args: str) -> str | None:
    directory = Path(nifreeze.__file__).parent
    try:
        result = subprocess.run(
            ["git", "-C", str(directory), *args], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def provenance(settings: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    What ran: the ANTs binary, the *NiFreeze* commit, the libraries and the threads.

    The commit is read from the package's own checkout (in the phantom dataset, the
    ``code/nifreeze`` subdataset), and ``NiFreezeDirty`` says whether its tracked
    files had changes.
    """
    ants = shutil.which("antsRegistration")
    ants_version = None
    if ants is not None:
        ants_version = subprocess.run(
            [ants, "--version"], capture_output=True, text=True
        ).stdout.strip()
    status = _git("status", "--porcelain", "--untracked-files=no")
    cpu = None
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    record: dict[str, Any] = {
        "AntsPath": ants,
        "AntsVersion": ants_version,
        "NiFreezeCommit": _git("rev-parse", "HEAD"),
        "NiFreezeDirty": None if status is None else bool(status),
        "NiFreezeVersion": nifreeze.__version__,
        "Python": platform.python_version(),
        "Packages": {
            package: _version(package)
            for package in ("numpy", "scipy", "nibabel", "nitransforms", "nipype", "dipy")
        },
        "Threads": {name: os.environ.get(name) for name in THREADS},
        "Affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "Cpu": cpu,
    }
    if settings is not None:
        record["Settings"] = settings
    return record


def run_tasks(
    tasks: Iterable[T], function: Callable[[T], R], jobs: int = 1, every: int = 50
) -> list[R]:
    """Run ``function`` on each task, ``jobs`` at a time, in order, with progress."""
    tasks = list(tasks)
    results = []
    with ThreadPoolExecutor(jobs) as pool:
        for done, result in enumerate(pool.map(function, tasks), start=1):
            results.append(result)
            if done % every == 0 or done == len(tasks):
                print(f"{done}/{len(tasks)}", flush=True)
    return results


def write_results(
    out: Path, name: str, rows: list[dict[str, Any]], meta: dict[str, Any]
) -> tuple[Path, Path]:
    """
    Write ``<out>/<name>.tsv``, one row per registration, and ``<out>/<name>.json``.

    The JSON holds ``meta``, and the :func:`provenance` unless ``meta`` has it.
    """
    out.mkdir(parents=True, exist_ok=True)
    table = out / f"{name}.tsv"
    with table.open("w", newline="") as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
    sidecar = out / f"{name}.json"
    record = meta if "Provenance" in meta else meta | {"Provenance": provenance()}
    sidecar.write_text(json.dumps(record, indent=2) + "\n")
    return table, sidecar
