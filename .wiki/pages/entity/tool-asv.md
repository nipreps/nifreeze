---
title: "Airspeed Velocity (asv, asv_runner): the behavior nifreeze's benchmark CI relies on"
entity_type: tool
last_verified: 2026-10-20
confidence_score: 0.9
refresh_needed_if: "the asv pin in pyproject.toml's benchmark extra moves past 0.6.6, or asv_runner (unpinned) changes its minor version"
source_urls:
  - https://github.com/airspeed-velocity/asv/tree/v0.6.6/asv
  - https://github.com/airspeed-velocity/asv_runner/tree/main/asv_runner
  - https://asv.readthedocs.io/en/v0.6.6/
---

# Airspeed Velocity (`asv` 0.6.6, `asv_runner`)

ASV is the benchmark runner of nifreeze's *Benchmark* workflow. `asv` (pinned at
0.6.6 in the `benchmark` extra) builds environments, schedules benchmarks and
stores results; `asv_runner` (installed by `asv` into each benchmark environment,
unpinned) runs them. This page records the behavior the workflow and the suite
depend on, read from the source of `asv` 0.6.6 and `asv_runner` 0.2.5 to 0.3.2.
Several defaults do not match what their names or the help text suggest.

## Which code runs

- **The suite always comes from the working tree**, for every commit benchmarked;
  only the project is built and installed per commit. A suite must therefore
  import and run against every commit it is pointed at.
- **Discovery imports the suite against the first commit given** (then the
  branch tips, then the others), so with `HASHFILE:` the newest commit goes
  first.
- A benchmark whose `setup` raises `NotImplementedError` is **skipped** (stored
  as NaN). Any other failure is stored as `None`, and `asv run` then exits 2.
- `asv run "A^! B^!"` does not run both commits when B is A's parent (git
  excludes B); use `HASHFILE:file` or a range.
- With `-E existing:...`, `PYTHONPATH` is discarded; `ASV_PYTHONPATH` is honored.

## Processes and samples

- **Each benchmark, parameter combination and round runs in a new process**
  (spawned, not forked), started in a temporary directory, or in the
  `setup_cache` directory when the benchmark has one.
- `setup` runs again before every sample after the first.
- **A `setup` that can be called without arguments runs before `setup_cache`.**
  A class whose `setup_cache` returns nothing therefore has its `setup` run
  before its files exist; returning a value makes `setup` take it as its first
  argument and defers it.
- `setup_cache` runs once per commit and environment, and **once per round per
  commit** with `--interleave-rounds`. Its cache key is the function's
  `file:line`.
- `--interleave-rounds` alternates commits ABBA: rounds count down from the
  maximum, and even rounds reverse the commit order.
- `time_` benchmarks pool the samples of all rounds; the stored value is their
  median. `rounds` defaults to 2, `repeat` to (1, 10, 20 s), or (1, 5, 10 s)
  with more than one round; `number` 0 means calibrate each sample to
  `sample_time` (10 ms).
- **`timer`** replaces the wall clock of a `time_` benchmark with any
  zero-argument callable returning seconds (e.g., `time.process_time`).
  Attributes are looked up on the function first, then the class, and the
  first one found wins even if it is `None`.
- `track_` benchmarks run one round and store one value, with no samples or
  statistics.
- **`--quick`** sets `number=1, repeat=1, warmup_time=0, rounds=1`. Results are
  still saved.

## Comparing

- `asv compare` treats every benchmark as lower-is-better, `track_` ratios
  included.
- With `repeat == 1` on either side (e.g., `--quick`), only the factor is
  applied. Otherwise a Mann-Whitney U test (p < 0.002) is used when there are
  enough samples (about 6 per side), and overlap of the 99% confidence
  intervals when there are not. `track_`, `mem_` and `peakmem_` get the factor
  only.
- One `--factor` applies to every benchmark; `regression_factor` in
  `asv.conf.json` is not an ASV setting.
- `x` ("not comparable") marks a benchmark whose `version` differs between the
  two results; by default the version is a hash of the benchmark's source,
  `setup` and `setup_cache`. `asv continuous` never shows `x`, since both sides
  use the current suite.
- `asv continuous` exits 1 on a regression and 2 if any benchmark failed on
  either commit (before printing the table); `asv compare` always exits 0.

## Memory benchmarks

- **`mem_`** returns the size of a *shallow copy* of the returned object
  (`asizeof([obj, copy(obj)]) - asizeof([obj, obj])`). Only an array returned
  directly is measured; an object holding arrays measures a few hundred bytes.
- **`peakmem_`** returns `ru_maxrss` of the benchmark process after the call:
  the lifetime peak, including interpreter start-up, imports, unpickling the
  cache and `setup`. It does not include child processes.
- The usable alternatives, as `track_` benchmarks:
  - `tracemalloc`'s traced peak around the call. NumPy and h5py allocations are
    traced, so the value is the same on every run; pages read through a memory
    map, buffers C libraries allocate themselves, and other processes are not.
  - the `ru_maxrss` increase of a forked child from just after the fork. It sees
    memory maps and C buffers, but varies by about 10% between runs.

## CPU time with joblib

nifreeze's parallel fits run in loky worker processes (DIPY's `paramap` and
nifreeze's own `Parallel` both default to loky). `time.process_time()` in the
parent misses them, and `RUSAGE_CHILDREN` counts a worker only after it has
exited and been reaped, which reusable loky workers do not. Summing
`process_time()`, `RUSAGE_CHILDREN` and the CPU times of the live children
(`psutil.Process().children(recursive=True)`) tracked the true total (2.5 to
2.9 s measured against 2.6 s expected). OpenBLAS threads inflate CPU time about
7x when not pinned to one thread.

## Machines and publishing

- Results go to `results/<machine>/<hash[:8]>-<env>.json`, the units to
  `results/benchmarks.json`. `asv machine --yes --machine NAME` stores only the
  name, so the CPU and RAM fields stay empty.
- **`asv publish` empties `html_dir`** before writing, and **drops results whose
  benchmark `version` differs from the current suite's**. An explicit `version`
  attribute keeps a benchmark's history across edits that do not change what it
  measures.

## In nifreeze

The workflow design these facts lead to (one job, both commits interleaved,
per-kind gates, data written as NIfTI) is described in `benchmarks/README.rst`
and nifreeze#517.
