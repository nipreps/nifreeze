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
"""Profiles DKI fit_predict scaling across voxel counts and n_jobs to identify parallel
crossover points."""

import time
from multiprocessing import cpu_count

import numpy as np
from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]

from nifreeze.model.dmri import DKIModel


def _median_runtime_seconds(dataset, index, n_jobs, repeats=5, warmup=1):
    # Warmup
    for _ in range(warmup):
        with (
            threadpool_limits(limits=1, user_api="blas"),
            threadpool_limits(limits=1, user_api="openmp"),
        ):
            DKIModel(dataset).fit_predict(index, n_jobs=n_jobs)

    # Timed runs
    ts = []
    for _ in range(repeats):
        with (
            threadpool_limits(limits=1, user_api="blas"),
            threadpool_limits(limits=1, user_api="openmp"),
        ):
            t0 = time.perf_counter()
            DKIModel(dataset).fit_predict(index, n_jobs=n_jobs)
            ts.append(time.perf_counter() - t0)

    return float(np.median(ts))


def profile_dki_parallel_crossover(
    build_dataset_fn,
    voxel_grid=(1000, 5000, 10000, 20000, 40000),
    jobs_grid=None,
    repeats=5,
    warmup=1,
):
    """
    build_dataset_fn(n_voxels) -> (dataset, index)
    """
    if jobs_grid is None:
        max_jobs = min(8, cpu_count())
        jobs_grid = tuple(sorted(set([1, 2, 4, max_jobs])))

    print("\nDKI parallel crossover profile")
    print(f"repeats={repeats}, warmup={warmup}, jobs={jobs_grid}, voxels={voxel_grid}\n")

    header = (
        f"{'n_voxels':>10} | {'n_jobs':>6} | {'median_s':>10} | "
        f"{'speedup_vs_1':>12} | {'efficiency':>10}"
    )
    print(header)
    print("-" * len(header))

    for n_voxels in voxel_grid:
        dataset, index = build_dataset_fn(n_voxels)

        times = {}
        for n_jobs in jobs_grid:
            t = _median_runtime_seconds(
                dataset=dataset,
                index=index,
                n_jobs=n_jobs,
                repeats=repeats,
                warmup=warmup,
            )
            times[n_jobs] = t

        t1 = times[1]
        for n_jobs in jobs_grid:
            t = times[n_jobs]
            speedup = (t1 / t) if t > 0 else float("inf")
            efficiency = speedup / n_jobs
            print(
                f"{n_voxels:10d} | {n_jobs:6d} | {t:10.4f} | {speedup:12.3f} | {efficiency:10.3f}"
            )

        best_jobs = min(times, key=lambda k: times[k])
        best_t = times[best_jobs]
        best_speedup = (t1 / best_t) if best_t > 0 else float("inf")
        print(
            f"{'':10} | {'BEST':>6} | {best_t:10.4f} | "
            f"{best_speedup:12.3f} | {'-':>10}   (n_jobs={best_jobs})"
        )
        print()
