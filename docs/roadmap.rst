.. include:: links.rst

Roadmap
=======
This page records where *NiFreeze* is going, and why.
The work is tracked in a pinned issue
(`#596 <https://github.com/nipreps/nifreeze/issues/596>`__),
with one milestone per track.

Where *NiFreeze* stands
-----------------------
The pieces of an open alternative to FSL's *eddy* are in place:
leave-one-volume-out models (:doc:`models`),
the instant update and its realigned view,
realigned NIfTI outputs with their X5 transforms,
and cascades of models on the command line.
Testing has moved ahead of the correction itself.
The TRXScan phantoms (:doc:`phantoms`) give the exact ground truth,
and the sensitivity analysis and the benchmarks measure how far predictions drift
under motion and eddy currents.
So far they have shown three things:

- Motion is not yet accurate.
  On DIPY's Sherbrooke data, the GP model ends with a median error of 1.38 mm after two passes.
- Eddy currents are not corrected at all.
  The affine registration configurations are never used, and the command line cannot select them.
- The model that fits best is the one that drifts most.
  Choosing a model by its error on clean data alone can therefore make the registration target worse.

Reading the code added four more:

- **Registration goes through the filesystem.**
  Each volume runs its own ``antsRegistration`` process, through nipype,
  on NIfTI files quantized to 12 bits, one volume after another.
- **One registration configuration serves every level, model and modality.**
  It was tuned to register b=0 to b=0 under large rotations, and it never works at full resolution.
- **Only HDF5 inputs are memory-mapped.**
  NIfTI inputs stay in memory, every model fit copies its training data again, and nothing is checkpointed.
- **GQI's held-out predictions are 30 to 80 times too large**
  (`#495 <https://github.com/nipreps/nifreeze/issues/495>`__).

Principle
---------
Measure before changing.
Registration, models and data come first, and each change is judged on the phantom.
The estimator builds on all three, and eddy currents enter the loop last.

Tracks
------
Registration harness (R1)
~~~~~~~~~~~~~~~~~~~~~~~~~
Registration is scored against each phantom volume's exact motion
(`#584 <https://github.com/nipreps/nifreeze/issues/584>`__).
Each moved volume is registered to three fixed images, which together separate registration error from model error:

- the noise-free reference volume of the same direction, at the true position (registration alone);
- a model's prediction from the unmoved data;
- a model's prediction from the moved data, as in a real run.

On the eddy-only sets, the harness scores the displacement along phase encoding.
It also records the time each registration takes.

In-memory registration (R2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~
A backend-independent interface exchanges arrays instead of files
(`#590 <https://github.com/nipreps/nifreeze/issues/590>`__).
ANTsPy comes first: it is the same algorithm, so the harness can show that nothing but the time changed.
Two more backends follow as optional extras, compared on the harness
(`#592 <https://github.com/nipreps/nifreeze/issues/592>`__):

- itk-elastix, which also builds for WebAssembly;
- FireANTs, which runs on GPU.

The default must run on CPU everywhere.

Every backend returns one of two outputs:

- a rigid transform for motion;
- for eddy currents, a very smooth B-spline field along the phase-encoding axis, with a Jacobian intensity correction
  (`#591 <https://github.com/nipreps/nifreeze/issues/591>`__).

This is the field model SDCFlows uses for fieldmaps.

The registration settings are then tuned on the phantom, per shell
(`#593 <https://github.com/nipreps/nifreeze/issues/593>`__).
Tuning used to happen on b=0 images under rotations far larger than head motion.

Models (M)
~~~~~~~~~~
A test checks the amplitude of every model's held-out prediction
(`#586 <https://github.com/nipreps/nifreeze/issues/586>`__),
because the existing tests check only correlation, which ignores scale.

GQI is fixed in place, so that the fix can go to DIPY.
Its prediction becomes kernel ridge regression on the training Gram matrix, with an unpenalized mean per shell.
A weak ridge reproduces GQI-KRR, and a strong one returns the shell mean.
The ridge path therefore trades fit against drift, and it may replace switching models in a cascade.

Data and memory (D)
~~~~~~~~~~~~~~~~~~~
Benchmarks come first
(`#595 <https://github.com/nipreps/nifreeze/issues/595>`__).
Then, step by step:

- one cache directory per dataset
  (`#589 <https://github.com/nipreps/nifreeze/issues/589>`__);
- NIfTI inputs converted volume by volume into a memory-mapped HDF5 file
  (`#594 <https://github.com/nipreps/nifreeze/issues/594>`__);
- a masked, time-first cache that models read from
  (`#513 <https://github.com/nipreps/nifreeze/issues/513>`__);
- outputs streamed from the realigned view
  (`#341 <https://github.com/nipreps/nifreeze/issues/341>`__);
- checkpoints
  (`#171 <https://github.com/nipreps/nifreeze/issues/171>`__);
- parallel workers that read from the file
  (`#158 <https://github.com/nipreps/nifreeze/issues/158>`__).

Estimator (E)
~~~~~~~~~~~~~
Once the three foundations are in place, the estimator changes in four ways:

- fitting overlaps with registration;
- the passes stop on convergence rather than after a fixed number;
- the models go from robust to flexible, either with a ridge schedule or by switching models;
- the cost of the instant update is measured.

The target is a median error below 0.5 mm and a 95th percentile below 1 mm at 2° and 2 mm,
on the phantom, compared with *eddy*.

Eddy currents in the loop (P)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Each volume is registered for rigid motion first, then for the B-spline field along phase encoding.
The realigned view undoes both, so the models no longer train on distorted volumes.
An end-to-end test on the phantom's eddy-current sets gates the result
(`#8 <https://github.com/nipreps/nifreeze/issues/8>`__).

Later
-----
- The full phantom grids and the sensitivity ladder on every protocol
  (`#568 <https://github.com/nipreps/nifreeze/issues/568>`__,
  `#575 <https://github.com/nipreps/nifreeze/issues/575>`__), and a paper.
- Integration into a preprocessing pipeline.
- Documentation notebooks
  (`#519 <https://github.com/nipreps/nifreeze/issues/519>`__,
  `#493 <https://github.com/nipreps/nifreeze/issues/493>`__).
- Fieldmaps
  (`#7 <https://github.com/nipreps/nifreeze/issues/7>`__).
- PET
  (`#396 <https://github.com/nipreps/nifreeze/issues/396>`__).

Open questions
--------------
- Whether the estimator is compared with *eddy* alone, or with SHORELine too.
- Which pipeline adopts *NiFreeze* first: QSIPrep or dMRIPrep.
- Where the full phantom grids are computed.
- Whether fMRI is in scope.
