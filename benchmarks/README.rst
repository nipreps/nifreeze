..  -*- rst -*-

===================
NiFreeze benchmarks
===================
Benchmarks of *NiFreeze* with `Airspeed Velocity`_ (ASV): how much memory,
CPU time and wall time the data layer and the models take.
The results of every commit to ``main`` are published at
https://www.nipreps.org/nifreeze/benchmarks/.

What is measured
----------------
The kind of each benchmark is given by its name, and each kind has its own
threshold for a regression (see ``GATES`` in ``report.py``):

===================  ================================================  ====================================
Name prefix          Measure                                           A pull request fails when
===================  ================================================  ====================================
``track_alloc_``     Peak memory Python allocates during the call      1.25 times ``main`` and 2 MB more
                     (``tracemalloc``), the same on every run
``track_rss_``       Peak RSS increase of forked processes running     1.5 times ``main`` and 16 MB more
                     the call (median of three); also counts memory
                     maps and the buffers of C libraries
``time_cpu_``        CPU seconds, joblib's workers included            1.5 times ``main``, significant
                                                                       (one-sided Mann-Whitney U,
                                                                       *p* < 0.01), and ``main`` takes
                                                                       10 ms or more
``time_``            Wall time                                         as for CPU time
``track_quality_``   Dimensionless, deterministic measures of model    it rises by 0.01 (``drift_``),
                     quality: prediction drift (``drift_``),           0.02 (``share_``) or 0.05
                     inherited share (``share_``), compounding         (``ratio_``) or more, whatever
                     ratio (``ratio_``); lower is better               ``main``'s value
===================  ================================================  ====================================

- ``bench_data.py``: wall time of loading and writing DWI and PET data.
- ``bench_memory.py``: allocations of the same operations on 40 MB series and of
  one leave-one-volume-out fit per model; peak RSS of loading a 132 MB file.
- ``bench_cpu.py``: CPU time of model fits and predictions, and the wall time of
  a DKI fit over two workers.
- ``bench_quality.py``: how far each model's predictions drift on the CI phantom
  (``docs/phantoms.rst``) when volumes move or are distorted by eddy currents,
  the share of a volume's own distortion that its prediction inherits, and
  whether motion and eddy currents compound.
  It needs the phantom at ``NIFREEZE_PHANTOMS``, which the workflow restores
  with ``.github/actions/phantom-fixtures``; without it, it is skipped.

How pull requests are checked
-----------------------------
The *Benchmark* workflow runs the suite on two commits in the same job:
the tip of ``main``, and the pull request merged into it.
Rounds alternate between the two commits, so both see the same machine
and the same dependencies, and a pull request that is behind ``main``
is still compared with what merging it would change.
``report.py`` then prints a plain-text report to the log, writes it to the
job summary, and fails the job if a benchmark regressed past its threshold
or fails on the pull request.

Pushes to ``main`` record their results and publish the dashboard to
``gh-pages`` (``benchmarks/``, raw results in ``benchmarks/results/``).
Running the workflow by hand with a range of commits (e.g., ``HEAD~20..HEAD``)
records those missing from the dashboard.

Running the benchmarks locally
------------------------------
Install the tools with ``pip install -e .[benchmark]``.
From the repository's root, compare your branch with ``main``::

    asv machine --yes --config benchmarks/asv.conf.json --machine local
    git rev-parse HEAD main > commits.txt
    asv run --config benchmarks/asv.conf.json --machine local \
        --interleave-rounds --record-samples HASHFILE:$PWD/commits.txt
    python benchmarks/report.py --machine local \
        --base $(git rev-parse main) --target $(git rev-parse HEAD)

ASV builds each commit in its own virtual environment, and reads a relative
``HASHFILE:`` path from the ``benchmarks/`` directory.
To check that one benchmark runs against the code you have installed::

    asv run --config benchmarks/asv.conf.json --python=same --quick \
        --show-stderr --bench "ModelCPU"

``--quick`` takes a single sample: it shows that a benchmark runs, not how
fast.
The quality benchmarks need the CI phantom: simulate it once with
``uv run tools/phantoms/phantom.py fixture --out <dataset>`` and set
``NIFREEZE_PHANTOMS=<dataset>``.

Writing benchmarks
------------------
- The suite of the working tree runs against both commits, so it must work
  with both.
  Write the data as NIfTI files in ``setup_cache`` and load them with
  ``from_nii``, rather than calling the dataset constructors, whose layout may
  differ between the two commits.
- Guard the import of a nifreeze module that ``main`` may lack
  (``try``/``except ImportError``) and raise ``NotImplementedError`` in
  ``setup`` when it is missing: the benchmark is then reported as new on
  ``main`` instead of failing there.
- Return the files ``setup_cache`` writes and take them as the first argument of
  ``setup``: ASV runs a ``setup`` that takes no arguments before
  ``setup_cache``.
  ``setup_cache`` must neither raise (the whole class fails) nor return
  ``None`` (ASV then passes no cache to ``setup``): to skip, return an empty
  value and raise ``NotImplementedError`` in ``setup``.
- Keep each call above 10 ms of CPU or wall time, or rely on its allocations:
  shorter timings are reported but do not fail a pull request.
- Each class sets ``version``.
  Bump it when what a benchmark measures changes (the helpers in ``_common.py``
  included); the dashboard keeps only results of the current version.

.. _Airspeed Velocity: https://asv.readthedocs.io/
