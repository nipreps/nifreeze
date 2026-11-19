26.0.0rc1 (November 19, 2026)
=============================
The first release candidate of *NiFreeze* 26.0.0, the first release with a calendar version.
*NiFreeze* now holds b-vectors in world RAS+ coordinates, predicts GQI by ridge regression,
and requires Python 3.13 or newer.
The candidate is for testing on real data, and its known limitations are listed below.

.. warning::

   **What breaks.**

   - **B-vectors in world RAS+** (`#643 <https://github.com/nipreps/nifreeze/pull/643>`__, `#654 <https://github.com/nipreps/nifreeze/pull/654>`__).
     ``DWI.gradients`` holds unit b-vectors in world RAS+ coordinates, with the b-value.
     ``from_nii`` converts FSL's ``.bvec`` files with the image's affine, where it took them
     as written before, and reads RAS+B tables (MRtrix's ``.b`` or NiPreps' ``.tsv``) as
     ``gradients_file``; the CLI's ``--gradient-file`` does the same.
     ``to_nifti`` writes FSL's ``.bvec`` and ``.bval`` for the output's affine, and a RAS+B
     ``.tsv``.
     *nfHDF5* files are now written as version 3.
     Versions 1 and 2 are read as RAS+B, with a warning: a file whose table came from FSL
     files holds vectors along the voxel axes, so write it again from its NIfTI and FSL files.
     ``transform_fsl_bvec`` is removed:
     use ``realign_bvecs(bvecs_to_ras(bvecs, affine, "fsl"), matrix)``.
   - **GQI predictions** (`#605 <https://github.com/nipreps/nifreeze/pull/605>`__).
     ``gqi`` and ``gqi-krr`` predict by ridge regression around a mean per shell, and ``gqi``
     no longer overscales the signal.
     The ridge of ``gqi-krr`` is in new units, and its default ``krr_lambda`` is 0.3
     (``DEFAULT_RIDGE``) instead of chosen on the data: pass ``krr_lambda=None`` to choose it
     on the data again.
     ``prediction_kernel`` and ``INVERSE_LAMBDA`` are removed from ``nifreeze.model.gqi``.
   - **Registration settings** (`#603 <https://github.com/nipreps/nifreeze/pull/603>`__).
     The settings of ``--align-config`` and of ``Estimator.run(align_kwargs=...)`` now apply,
     at every level of a cascade, and a caller's ``num_threads`` is no longer replaced by
     ``omp_nthreads``.
     ``generate_command`` refuses unknown settings with a ``TypeError``.
   - **Python 3.13 or newer** (`#646 <https://github.com/nipreps/nifreeze/pull/646>`__), with the oldest versions of the dependencies set by
     `SPEC 0 <https://scientific-python.org/specs/spec-0000/>`__ and no cap on numpy (`#647 <https://github.com/nipreps/nifreeze/pull/647>`__).
     ``nest-asyncio`` moved to the ``notebooks`` extra.

.. important::

   **Known limitations.**
   The roadmap (`#596 <https://github.com/nipreps/nifreeze/issues/596>`__) tracks each of them.

   - **Accuracy.** The work on the estimator has not started, so motion estimates do not
     yet reach the roadmap's target: a median error below 0.5 mm and a 95th percentile below
     1 mm, at 2° and 2 mm of motion.
   - **Eddy currents** are approximated only by an affine registration, with the
     ``dwi-to-dwi_level1`` settings (``Estimator(..., ants_config="dwi-to-dwi_level1")``, or
     ``ants_config`` in the file of ``--align-config``).
     The smooth field along the phase-encoding axis is not implemented
     (`#591 <https://github.com/nipreps/nifreeze/issues/591>`__).
   - **Memory.** Only *nfHDF5* inputs are memory-mapped: NIfTI inputs are held in memory,
     each model fit copies its training data, and nothing is checkpointed
     (`#594 <https://github.com/nipreps/nifreeze/issues/594>`__, `#513 <https://github.com/nipreps/nifreeze/issues/513>`__, `#171 <https://github.com/nipreps/nifreeze/issues/171>`__).
     MRIQC's draft head-motion correction peaked at 8.0 GB on a 0.41 GB series
     (`nipreps/mriqc#1518 <https://github.com/nipreps/mriqc/pull/1518>`__).
   - **Registration** runs ANTs, which is installed apart from *NiFreeze*.
     Volumes are registered one after another, and ANTs gets the images through files
     (`#590 <https://github.com/nipreps/nifreeze/issues/590>`__).
     The estimator cannot use the elastix backend yet
     (`#672 <https://github.com/nipreps/nifreeze/issues/672>`__).

**Data**

- ``to_nifti`` writes the gradients with every NIfTI it writes (`#608 <https://github.com/nipreps/nifreeze/pull/608>`__), and with
  ``insert_b0=True`` it writes the b=0 followed by the resampled volumes, not the original
  ones (`#601 <https://github.com/nipreps/nifreeze/pull/601>`__).
- Readers, writers and conversions of gradient tables: ``load_bvec_bval``,
  ``save_bvec_bval``, ``load_rasb``, ``save_rasb``, ``bvecs_to_ras`` and
  ``bvecs_from_ras`` (`#643 <https://github.com/nipreps/nifreeze/pull/643>`__).

**Registration**

- Registration settings are read from a JSON file, as well as from a packaged name (`#632 <https://github.com/nipreps/nifreeze/pull/632>`__).
- ``generate_command(num_threads=...)`` also sets ITK's thread count, over a global
  setting (`#633 <https://github.com/nipreps/nifreeze/pull/633>`__).
- The CLI writes the version of ``antsRegistration`` it ran into a JSON sidecar (`#634 <https://github.com/nipreps/nifreeze/pull/634>`__).
- ``nifreeze.analysis.registration`` scores a registration against a phantom's known
  motion (`#598 <https://github.com/nipreps/nifreeze/pull/598>`__), and its ``AntsRegistration`` starts from an initial transform
  (``init``) as the estimator does (`#690 <https://github.com/nipreps/nifreeze/pull/690>`__).
- An elastix backend, ``nifreeze.registration.elastix.Elastix``, is given the images in
  memory (`#676 <https://github.com/nipreps/nifreeze/pull/676>`__), with the settings of ``b0-to-b0_level0`` and ``dwi-to-dwi_level1``
  mapped onto elastix's components (`#674 <https://github.com/nipreps/nifreeze/pull/674>`__).
  It needs the new ``elastix`` extra (``itk-elastix``, `#673 <https://github.com/nipreps/nifreeze/pull/673>`__).
  Only the accuracy harness calls it: the estimator still registers with ANTs.

**Validation**

- Reports on what limits the registration error with a perfect target (`#610 <https://github.com/nipreps/nifreeze/pull/610>`__), on the
  eddy-current truth of the experiments (`#623 <https://github.com/nipreps/nifreeze/pull/623>`__), and on elastix's settings against ANTs
  (`#674 <https://github.com/nipreps/nifreeze/pull/674>`__, `#688 <https://github.com/nipreps/nifreeze/pull/688>`__).
- Tests of the conventions by which data reach ANTs and estimates come back (`#622 <https://github.com/nipreps/nifreeze/pull/622>`__), of the
  amplitude of every model's prediction (`#602 <https://github.com/nipreps/nifreeze/pull/602>`__), and of the estimator with a mocked
  registration (`#600 <https://github.com/nipreps/nifreeze/pull/600>`__).
- The phantom generator (``tools/phantoms/``, not shipped) oversamples the simulation and
  simulates a few directions per shell (`#636 <https://github.com/nipreps/nifreeze/pull/636>`__), bins the streamlines at the voxel centres
  (`#637 <https://github.com/nipreps/nifreeze/pull/637>`__), simulates a slice profile (`#659 <https://github.com/nipreps/nifreeze/pull/659>`__), and has its locks checked in CI (`#656 <https://github.com/nipreps/nifreeze/pull/656>`__).
- The registration experiments (``tools/registration/``, not shipped) share their
  bookkeeping (`#616 <https://github.com/nipreps/nifreeze/pull/616>`__) and their model targets (`#627 <https://github.com/nipreps/nifreeze/pull/627>`__), and test elastix's settings (`#683 <https://github.com/nipreps/nifreeze/pull/683>`__).
- ``rigid_motion`` leaves its caller's axis and direction unchanged, so a motion plan that
  starts at zero no longer turns the later motions into NaN (`#684 <https://github.com/nipreps/nifreeze/pull/684>`__).
- Docstrings of ``Perturbation.motions`` (`#641 <https://github.com/nipreps/nifreeze/pull/641>`__) and of the readout of ``Distortion`` (`#625 <https://github.com/nipreps/nifreeze/pull/625>`__).

**Documentation**

- A page on citing *NiFreeze* and writing papers about it, with its authors credited as the
  NiPreps guidelines say (`#663 <https://github.com/nipreps/nifreeze/pull/663>`__, `#666 <https://github.com/nipreps/nifreeze/pull/666>`__).

**Maintenance**

- CI tests Python 3.13 and 3.14 on every pull request (`#646 <https://github.com/nipreps/nifreeze/pull/646>`__), and the oldest versions of the
  dependencies (`#647 <https://github.com/nipreps/nifreeze/pull/647>`__), which the ``nipreps-spec0`` app raises every quarter (`#648 <https://github.com/nipreps/nifreeze/pull/648>`__, `#652 <https://github.com/nipreps/nifreeze/pull/652>`__, `#653 <https://github.com/nipreps/nifreeze/pull/653>`__).
- The CI jobs that install ANTs run on Ubuntu 24.04 (`#639 <https://github.com/nipreps/nifreeze/pull/639>`__), and documentation pull requests
  skip the long jobs (`#613 <https://github.com/nipreps/nifreeze/pull/613>`__).
- CI caches ANTs and no longer installs TeX Live for the tests (`#678 <https://github.com/nipreps/nifreeze/pull/678>`__).
  Pull requests and pushes register 16 corners of the ANTs b=0 grid, and the weekly run all
  432 cases (`#682 <https://github.com/nipreps/nifreeze/pull/682>`__); the benchmarks compare only pull requests that change what
  they measure (`#686 <https://github.com/nipreps/nifreeze/pull/686>`__).
- ``AGENTS.md`` holds the instructions for coding agents, and ``CLAUDE.md`` links to it
  (`#661 <https://github.com/nipreps/nifreeze/pull/661>`__).

0.2.0 (October 29, 2026)
========================

.. warning::

   **Volumes are now stored on the first axis** (`#461 <https://github.com/nipreps/nifreeze/pull/461>`__).
   ``dataset.dataobj`` has shape ``(N, X, Y, Z)``, and ``dataset[i]`` returns ``dataobj[i]``.
   Code that builds datasets directly (``BaseDataset``, ``DWI`` or ``PET`` with ``dataobj=``)
   must pass the volumes first, for example ``np.moveaxis(data, -1, 0)``.
   ``from_nii`` and ``to_nifti`` convert to and from the NIfTI order (volumes last).
   HDF5 files are now written as *nfHDF5* version 2 (``Format="nfHDF5"``, for PET too)
   and memory-mapped when loaded.
   Version 1 files (``Format="NFDH5"`` or ``"EMC/PET"``) still load, after a conversion
   into a temporary file; save them again with ``to_filename`` to load them lazily.
   ``nifreeze.data.NFDH5_EXT`` is renamed ``NFHDF5_EXT``.

**Motion estimation**

- The *instant update*: models train on the data realigned with the motion estimated so far
  (`#538 <https://github.com/nipreps/nifreeze/pull/538>`__), and each registration starts at the current estimate (`#536 <https://github.com/nipreps/nifreeze/pull/536>`__).
- Realigned volumes are written through their NIfTI q/s-form (`#537 <https://github.com/nipreps/nifreeze/pull/537>`__), and the X5 file holds
  every volume's transform (`#531 <https://github.com/nipreps/nifreeze/pull/531>`__).
- Less per-volume overhead preparing registration inputs (`#529 <https://github.com/nipreps/nifreeze/pull/529>`__); start and end indices for
  the estimator (`#423 <https://github.com/nipreps/nifreeze/pull/423>`__).

**Models**

- Closed-form leave-one-volume-out models: ``gp-loo`` and ``gqi-krr`` (`#499 <https://github.com/nipreps/nifreeze/pull/499>`__).
- GP: correct log-space kernel gradients and a tunable noise term (`#457 <https://github.com/nipreps/nifreeze/pull/457>`__), better starting
  values and a multi-shell noise term (`#498 <https://github.com/nipreps/nifreeze/pull/498>`__), robust and scalable hyperparameter
  optimization (`#507 <https://github.com/nipreps/nifreeze/pull/507>`__), and a multi-shell kernel (`#175 <https://github.com/nipreps/nifreeze/pull/175>`__).
- GQI synced with DIPY (`#453 <https://github.com/nipreps/nifreeze/pull/453>`__); DKI fitted through DIPY's ``multi_voxel_fit`` (`#450 <https://github.com/nipreps/nifreeze/pull/450>`__); DTI and
  DKI fitted on the model's mask (`#541 <https://github.com/nipreps/nifreeze/pull/541>`__).
- PET: the B-spline knots span the frames' domain (`#424 <https://github.com/nipreps/nifreeze/pull/424>`__); the separate PET estimator is gone
  (`#426 <https://github.com/nipreps/nifreeze/pull/426>`__).

**Validation**

- An end-to-end motion test with known truth (`#533 <https://github.com/nipreps/nifreeze/pull/533>`__). It runs on volumes that TRXScan
  simulated again from the moved head (`#582 <https://github.com/nipreps/nifreeze/pull/582>`__), on the native Sherbrooke phantom in CI (`#606 <https://github.com/nipreps/nifreeze/pull/606>`__).
- Synthetic phantoms simulated with TRXScan, with their perturbations and ground truth
  (`#555 <https://github.com/nipreps/nifreeze/pull/555>`__, `#557 <https://github.com/nipreps/nifreeze/pull/557>`__, `#559 <https://github.com/nipreps/nifreeze/pull/559>`__, `#563 <https://github.com/nipreps/nifreeze/pull/563>`__, `#570 <https://github.com/nipreps/nifreeze/pull/570>`__, `#573 <https://github.com/nipreps/nifreeze/pull/573>`__, `#576 <https://github.com/nipreps/nifreeze/pull/576>`__).
- A sensitivity analysis of how model predictions drift under motion and eddy currents
  (`#542 <https://github.com/nipreps/nifreeze/pull/542>`__, `#544 <https://github.com/nipreps/nifreeze/pull/544>`__, `#565 <https://github.com/nipreps/nifreeze/pull/565>`__), tracked in CI by quality benchmarks (`#553 <https://github.com/nipreps/nifreeze/pull/553>`__, `#566 <https://github.com/nipreps/nifreeze/pull/566>`__). Rungs 1 to 3 of its
  ladder run on a phantom protocol (`#581 <https://github.com/nipreps/nifreeze/pull/581>`__), and it no longer includes the inherited-distortion
  share, which was a registration (`#580 <https://github.com/nipreps/nifreeze/pull/580>`__).
- A prediction gallery of DWI schemes and models (`#460 <https://github.com/nipreps/nifreeze/pull/460>`__, `#510 <https://github.com/nipreps/nifreeze/pull/510>`__).

**Fixes.** The estimator and models no longer modify the dataset's arrays (`#527 <https://github.com/nipreps/nifreeze/pull/527>`__);
``DWI.to_filename`` reopens the file it wrote (`#516 <https://github.com/nipreps/nifreeze/pull/516>`__); DKI fits data with two non-zero shells
(`#415 <https://github.com/nipreps/nifreeze/pull/415>`__).

The full list of changes is in the
`GitHub release <https://github.com/nipreps/nifreeze/releases/tag/0.2.0>`__.

0.1.0 (February 6, 2026)
========================
The first release of *NiFreeze*.
Its changes since the *eddymotion* fork are listed in the
`GitHub release <https://github.com/nipreps/nifreeze/releases/tag/0.1.0>`__.

*eddymotion* fork
=================
The *NiFreeze* codebase has been ported from *eddymotion* on commit
`68f199d <https://github.com/nipreps/eddymotion/commit/68f199dabc0aff8e904ad9f832ed276122c0845c>`_.
