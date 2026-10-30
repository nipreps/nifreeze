.. include:: ../links.rst

Experiment-driven development
=============================
*NiFreeze* changes how it registers volumes only on the evidence of an experiment.
Each experiment starts as a question registered in an issue before anything runs.
The issue states the hypotheses, the data, the metric, the decision rule, and the
code or default that each outcome would change.
The tracking issue `#614 <https://github.com/nipreps/nifreeze/issues/614>`__ lists
every experiment, its status and its decision.

The experiments run on the synthetic phantoms (:doc:`../phantoms`), where the truth
of every volume is known exactly.
Each runs as a ``datalad run`` record in the phantom dataset, from a script under
``tools/registration/``, so every result traces back to the command and the
*NiFreeze* commit that produced it, and can be run again.
A second script generates every figure, table and quoted command of the report, so
no number in a report is typed by hand.
A report names the comparison behind each claim, and says so when an explanation
has not been tested.
When a result calls for a change to the registered plan, the change is posted on the
issue before the run it affects, and the report keeps the registered result next to
the one added after the fact (*post hoc*).

The reports on registration accuracy (:doc:`../registration`) follow the same
approach.

.. toctree::
   :maxdepth: 2

   eddy-truth
