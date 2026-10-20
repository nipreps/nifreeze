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
"""Synthetic data and measurement helpers shared by the benchmarks.

The benchmarks run the same suite against two commits (``main`` and a pull
request), so they must not depend on how a commit lays out its arrays.
Datasets are therefore written as NIfTI files, which have one fixed layout, and
each commit loads them with its own ``from_nii`` and writes its own HDF5 file.
"""

import json
import os
import resource
import sys
import time
import tracemalloc
import warnings
from pathlib import Path

import h5py
import nibabel as nb
import numpy as np

try:
    import psutil
except ImportError:  # pragma: no cover - only CPU benchmarks of parallel fits need it
    psutil = None

ALLOC_FLOOR_MB = 2.0
"""Peak allocations under this value are reported as this value."""
RSS_FLOOR_MB = 16.0
"""Peak RSS increases under this value are reported as this value."""
RSS_REPEATS = 3
"""Forked children per peak-RSS measurement; the median is reported."""


def peak_alloc_mb(fn, *args, **kwargs) -> float:
    """Return the peak memory (MB) Python allocates while ``fn`` runs.

    :mod:`tracemalloc` sees every NumPy and h5py array, so the value is the same
    on every run. It does not see pages read through a memory map, buffers that
    C libraries allocate themselves, or other processes.
    """
    tracemalloc.start()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fn(*args, **kwargs)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    return max(peak / 2**20, ALLOC_FLOOR_MB)


def _child_peak_rss(fn, *args) -> float:
    """Run ``fn`` in a forked child; return its peak RSS increase (MB).

    The child measures its own ``ru_maxrss`` from just after the fork, so the
    footprint of the benchmark process (imports, ``setup``) does not count.
    ``ru_maxrss`` is in KiB on Linux and in bytes on macOS.
    """
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:  # pragma: no cover - child process
        code = 0
        try:
            os.close(read_fd)
            start = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            fn(*args)
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            os.write(write_fd, str(peak - start).encode())
        except BaseException:
            code = 1
        finally:
            os.close(write_fd)
            os._exit(code)

    os.close(write_fd)
    payload = b""
    while chunk := os.read(read_fd, 4096):
        payload += chunk
    os.close(read_fd)
    _, status = os.waitpid(pid, 0)
    if os.waitstatus_to_exitcode(status) != 0:
        raise RuntimeError("the peak-RSS child process failed")

    units = 2**20 if sys.platform == "darwin" else 2**10
    return int(payload) / units


def peak_rss_mb(fn, *args) -> float:
    """Return the median peak RSS increase (MB) of ``fn`` over forked children.

    Unlike :func:`peak_alloc_mb`, this counts pages read through memory maps and
    the buffers of C libraries such as HDF5, but it changes by about 10% from run
    to run.
    """
    peaks = [_child_peak_rss(fn, *args) for _ in range(RSS_REPEATS)]
    return max(float(np.median(peaks)), RSS_FLOOR_MB)


def cpu_time() -> float:
    """Return the CPU seconds used by this process and its children.

    Children count whether they are still running (joblib keeps its workers
    alive between calls) or have exited and been reaped. Use it as the
    ``timer`` of a benchmark so it measures CPU time instead of wall time.
    """
    own = time.process_time()
    reaped = resource.getrusage(resource.RUSAGE_CHILDREN)
    total = own + reaped.ru_utime + reaped.ru_stime
    if psutil is not None:
        for child in psutil.Process().children(recursive=True):
            try:
                times = child.cpu_times()
            except psutil.Error:
                continue
            total += times.user + times.system
    return total


def ellipsoid_mask(shape: tuple[int, ...], fraction: float) -> np.ndarray:
    """Return a centered ellipsoid with semi-axes ``fraction * shape``."""
    grid = np.indices(shape, dtype=float)
    center = (np.array(shape) - 1)[:, None, None, None] / 2
    radii = (fraction * np.array(shape))[:, None, None, None]
    return (((grid - center) / radii) ** 2).sum(0) < 1


def gradient_table(bvals: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return an ``(N, 4)`` gradient table with random unit directions."""
    bvecs = rng.standard_normal((bvals.size, 3))
    bvecs /= np.linalg.norm(bvecs, axis=1, keepdims=True)
    bvecs[bvals == 0] = 0
    return np.column_stack((bvecs, bvals)).astype("float32")


def dwi_series(shape3d: tuple[int, int, int], bvals: np.ndarray, rng) -> np.ndarray:
    """Return a mono-exponential decay plus noise, volumes last (as NIfTI stores it)."""
    data = rng.normal(0.0, 2.0, (*shape3d, bvals.size)).astype("float32")
    data += (100 * np.exp(-bvals * 1e-3)).astype("float32")
    return data


def write_nifti(path: Path | str, data: np.ndarray) -> None:
    """Write ``data`` as an uncompressed NIfTI file with an identity affine.

    Uncompressed, so loading costs array copies and not decompression.
    """
    nb.Nifti1Image(data, np.eye(4)).to_filename(path)


def write_dwi(stem: str, data: np.ndarray, gradients: np.ndarray, mask: np.ndarray) -> dict:
    """Write a DWI series, its gradient table and its mask; return ``from_nii`` arguments."""
    write_nifti(f"{stem}.nii", data)
    write_nifti(f"{stem}_mask.nii", mask.astype("uint8"))
    np.savetxt(f"{stem}_gradients.txt", gradients, fmt="%.6f")
    return {
        "filename": f"{stem}.nii",
        "brainmask_file": f"{stem}_mask.nii",
        "gradients_file": f"{stem}_gradients.txt",
    }


def write_pet(stem: str, data: np.ndarray, mask: np.ndarray) -> dict:
    """Write a PET series with one-second frames and its mask; return ``from_nii`` arguments."""
    write_nifti(f"{stem}.nii", data)
    write_nifti(f"{stem}_mask.nii", mask.astype("uint8"))
    frame_times = np.arange(data.shape[-1], dtype=float)
    Path(f"{stem}.json").write_text(json.dumps({"FrameTimesStart": frame_times.tolist()}))
    return {
        "filename": f"{stem}.nii",
        "temporal_file": f"{stem}.json",
        "brainmask_file": f"{stem}_mask.nii",
    }


def write_legacy_h5(path: Path | str, dataset_type: str, **fields) -> None:
    """Write a version-1 HDF5 file, the layout of every file before format version 2.

    ``fields`` are written as datasets of group ``/0``; ``dataobj`` must have
    the volumes last.
    """
    with h5py.File(path, "w") as out_file:
        out_file.attrs["Format"] = "NFDH5"
        out_file.attrs["Version"] = np.uint16(1)
        if dataset_type == "dmri":
            out_file.attrs["Type"] = "dmri"
        root = out_file.create_group("/0")
        root.attrs["Type"] = "base dataset"
        for name, value in fields.items():
            if value is not None:
                root.create_dataset(name, data=value)


def load_one_volume(cls, path: Path | str) -> None:
    """Load ``path`` and read one volume, as the estimator does."""
    dataset = cls.from_filename(path)
    np.asarray(dataset[len(dataset) // 2][0]).sum()
