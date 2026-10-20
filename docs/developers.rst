.. include:: links.rst

For developers
==============
Contributing
------------
*NiFreeze* is a project of the *NiPreps Community*, `which specifies the contributing guidelines <https://www.nipreps.org/community/>`__.
Before delving into the code, please make sure you have read all the guidelines offered online.

Documentation
-------------
Documentation sources are found under the ``docs/`` folder, and builds are archived in the `gh-pages <https://github.com/nipreps/nifreeze/tree/gh-pages>`__ branch of the repository.
With GitHub Pages, the documentation is posted under https://www.nipreps.org/nifreeze.
We maintain versioned documentation, by storing git tags under ``<major>.<minor>/`` folders, i.e., we do not archive every patch release, but only every minor release.
In other words, folder ``0.1/`` of the documentation tree contains the documents for the latest release within the *0.1.x* series.
With every commit (or merge commit) to ``main``, the *development* version of the documentation under the folder ``main/`` is updated too.
The ``gh-pages`` branch is automatically maintained with `a GitHub Action <https://github.com/nipreps/nifreeze/blob/main/.github/workflows/docs-build-update.yml>`__.
Please, do not commit manually to ``gh-pages``.

To build the documentation locally, you first need to make sure that ``setuptools_scm[toml] >= 6.2`` is installed in your environment and then::

  cd <nifreeze-repository>/
  python -m setuptools_scm  # This will generate ``src/nifreeze/_version.py``
  make -C docs/ html

Benchmarking
------------

*NiFreeze* uses asv_ for performance benchmarking. Benchmarking code is found under the
``benchmarks/`` folder. Maintaining reasonable runtime and memory consumption of core
*NiFreeze* components (e.g., models, data I/O, estimator) is critical for the
efficiency of pipelines using it.

The benchmarks measure the memory that operations allocate, their peak resident memory,
their CPU time and their wall time.
Every pull request is benchmarked together with the tip of ``main`` in the same job,
and fails if it makes a benchmark regress past the threshold of its kind;
the job summary lists every result.
The results of each commit to ``main`` are published in the `nifreeze dashboard`_.
``benchmarks/README.rst`` describes the thresholds, how to run the benchmarks locally,
and how to write new ones.

Library API (application program interface)
-------------------------------------------
Information on specific functions, classes, and methods.

.. toctree::
   :glob:

   api/nifreeze.analysis
   api/nifreeze.cli
   api/nifreeze.data
   api/nifreeze.estimator
   api/nifreeze.exceptions
   api/nifreeze.model
   api/nifreeze.registration
   api/nifreeze.testing
   api/nifreeze.utils
   api/nifreeze.viz
