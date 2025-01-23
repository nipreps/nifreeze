..  -*- rst -*-

===================
NiFreeze benchmarks
===================
Benchmarking NiFreeze with Airspeed Velocity.

Usage
-----
Airspeed Velocity manages building and Python environments by itself,
unless told otherwise.
To run the benchmarks, you do not need to install
a development version of *NiFreeze* on your current
*Python* environment.

To run all benchmarks for the latest commit, navigate to *NiFreeze*'s root
``benchmarks`` directory and execute::

    asv run

For testing benchmarks locally, it may be better to run these without
replications::

    export REGEXP="bench.*Ufunc"
    asv run --dry-run --show-stderr --python=same --quick -b $REGEXP

All of the commands above display the results in plain text in the console,
and the results are not saved for comparison with future commits.
For greater control, a graphical view, and to have results saved for future
comparisons, you can run ASV as follows to record results and generate
the HTML reports::

    asv run --skip-existing-commits --steps 10 ALL
    asv publish
    asv preview

More on how to use ``asv`` can be found in the `ASV documentation`_.
Command-line help is available as usual via ``asv --help`` and
``asv run --help``.

.. _ASV documentation: https://asv.readthedocs.io/
