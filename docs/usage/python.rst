.. include:: ../links.rst

Using *NiFreeze* from Python
============================
To utilize *NiFreeze* functionalities within your Python module or script, follow these steps:

1. **Import NiFreeze Components**: Start by importing necessary components from the *NiFreeze* package:

   To use *NiFreeze* you will typically need to use three main components:

     - Some **data**: allows to build a construct over your data that allows
       the **model** and the **estimator** objects to deal with the data in
       a standardized manner.
     - A **model**: establishes the predictive model used to estimate the
       generate the registration target. It will typically implement the
       ``fit`` and ``predict`` methods. It is specified when instantiating the
       **estimator**.
     - An **estimator**: orchestrates the registration target prediction.

   The same approach applies to any 4D modality data. For illustrative
   purposes, we will use some dMRI data:

   .. code-block:: python

      # Import required components from the nifreeze package
      from nifreeze.data import dmri
      from nifreeze.estimator import Estimator

2. **Load 4D Data**: Load your 4D data into a dataset object using the
   :func:`~nifreeze.data.load` function.

   Use the appropriate parameters for the particular imaging modality (e.g.
   dMRI, fMRI, or PET) that you are using.

   For example, for dMRI data, ensure the gradient table is provided. NiFreeze
   expects the table to have one row per diffusion-weighted image, with the
   first three columns storing the gradient direction in world RAS+ coordinates
   and the last column indicating the timing and strength of the gradients in
   units of s/mm² ``[ R A S+ b ]``.
   FSL's ``.bvec`` and ``.bval`` files are converted with the image's affine.

   .. code-block:: python

      # Load dMRI data into a DWI object
      dataset = dmri.load("/path/to/your/dwi_data.nii.gz", gradients_file="/path/to/your/gradient_file")

      # Or from FSL's files
      dataset = dmri.load("/path/to/your/dwi_data.nii.gz", bvec_file="/path/to/your/dwi.bvec", bval_file="/path/to/your/dwi.bval")

   .. note::

      To run the examples and tests from this page,
      find `sample data <https://osf.io/download/6at98/>`__ on OSF.
      To load from an :ref:`nfHDF5 <nfhdf5>` file, use:

      .. code-block:: python

         dataset = dmri.DWI.from_filename("/path/to/downloaded/dwi_full.h5")

   .. note::

      Datasets store the volumes on the *first* axis: ``dataset.dataobj`` has
      shape ``(N, X, Y, Z)`` and ``dataset[i]`` returns volume ``i``.
      NIfTI files store the volumes last; :func:`~nifreeze.data.dmri.io.from_nii`
      and :meth:`~nifreeze.data.base.BaseDataset.to_nifti` convert between the two.
      *NiFreeze* saves datasets as :ref:`nfHDF5 <nfhdf5>` files, which are
      memory-mapped when loaded, so volumes are read from disk as they are
      accessed.
      Files written by earlier versions of *NiFreeze* (format version 1) still
      load, but they are converted into a temporary file first; save them again
      with ``to_filename`` to skip that step.


3. **Instantiate an NiFreeze Estimator Object**: Create an instance of the
   :class:`~nifreeze.estimator.Estimator` class, which encapsulates tools
   for estimating rigid-body head motion and distortions due to eddy currents.

   .. code-block:: python

      # Create an instance of the Estimator class
      estimator = Estimator(
          model,
          strategy="random",
      )

   The estimator takes a model that determines how the target volume will be
   estimated.

   - ``model``: specifies the model used to generate the registration target
     for each gradient map (dMRI) or frame (dMRI, PET). For a list of
     available models, see :doc:`/api/nifreeze.model`.
   - ``strategy``: strategy used to traverse the 4D sequence. The list of
     supported strategies can be found at :doc:`/api/nifreeze.utils.iterators`.
   - ``prev``: estimators can be stacked and be run sequentially.
   - ``single_fit``: when ``True``, fit the model once on all volumes instead of
     refitting per held-out volume. Use carefully.
   - ``backend``: the registration backend, ``"elastix"`` (the default) or ``"ants"``,
     which register each volume on its own. ``"itk"`` estimates eddy currents as head
     motion followed by a field along the phase-encoding direction (9 parameters,
     :class:`~nifreeze.registration.itk_backend.ITK`), from the ``itk`` section of
     :data:`~nifreeze.registration.base.AFFINE`; the dataset must know its
     phase-encoding direction, and the eddy-current pass then refines the head
     motion and stores the field. ``"gem"`` estimates the eddy currents of all the
     volumes at once (:doc:`/methods/gem`), and needs the direction too.
   - ``align_config``: the registration settings, a mapping or a YAML file with a
     section per backend. NiFreeze ships two:
     :data:`~nifreeze.registration.base.RIGID` (the default, for head motion) and
     :data:`~nifreeze.registration.base.AFFINE` (for eddy currents). To change a
     setting, copy one and edit it.

4. **Fit the Models to Estimate the Affine Transformation**:

   Use the :meth:`~nifreeze.estimator.Estimator.run` method of the
   :class:`~nifreeze.estimator.Estimator` object to estimate the affine
   transformation parameters:

   .. code-block:: python

      # Estimate affine transformation parameters
      _ = estimator.run(
          dataset,
          align_kwargs=align_kwargs,
          omp_nthreads=omp_nthreads,
          n_jobs=n_jobs,
          seed=seed,
      )

   The ``run`` method employs the Leave-One-Volume-Out (LOVO) splitting technique to iteratively process DWI data volumes for each specified model: each volume is registered to a prediction fit on all the *other* volumes, so the target is independent of the volume being aligned. Affine transformations align the volumes, updating the `DWI` object with the estimated parameters. This method accepts several parameters:

   - ``dataset``: The target dataset, represented by this tool's internal type.
   - ``align_kwargs``: Registration settings for every estimator of a chain that
     was not given its own, e.g. ``{"align_config": "my-settings.yaml"}``.
   - ``omp_nthreads``: Maximum number of threads an individual process may use.
   - ``n_jobs``: Number of parallel jobs.
   - ``seed``: Seed for the random number generator (necessary for deterministic estimation).

   The estimates are stored in the dataset, each as an :math:`N \times 4 \times 4`
   array of affine matrices: the head motion in ``motion``, and the eddy currents of
   a DWI in ``eddy``. A volume's mapping composes them, ``eddy @ motion``
   (:meth:`~nifreeze.data.base.BaseDataset.get_transform`), and the outputs are
   resampled with it, once. In image terms, the eddy-current field is undone first, in
   place in the orientation the volume was acquired in, and the head motion after it.
   The b-vectors turn with the head motion alone: eddy currents displace the image
   without turning the head.

   Example:

   .. code-block:: python

      estimator = Estimator(
          model="b0",
          strategy="random",
      )

      # Example of estimating the motion parameters
      _ = estimator.run(
          dataset,
          omp_nthreads=4,
          n_jobs=4,
          seed=42,
      )

   Eddy currents are estimated by more passes, which start from the head-motion
   estimates stored in the dataset and store what they add in ``eddy``.
   By default, the command line estimates the eddy currents of a DWI with three passes
   of GEM, the global eddy model (:doc:`/methods/gem`), after the cascade of ``--models``
   (gqi-krr by default).
   ``--gem-passes N`` sets their number, and ``--gem-passes 0`` turns GEM off.
   Their model is the cascade's last, unless ``--eddy-model`` names another
   (``--models gp-loo --eddy-model gqi-krr``).
   GEM needs the phase-encoding direction, from the input's metadata or
   ``--phase-encoding``: without it, the default run estimates head motion only, with
   a warning.
   ``--gem-passes 0 --eddy`` registers each volume on its own instead, with the affine
   settings, :data:`~nifreeze.registration.base.AFFINE`, unless given its own file.
   In Python, each pass is an estimator:

   .. code-block:: python

      # One pass of GEM
      Estimator("gqi-krr", estimates="eddy", backend="gem").run(dataset, seed=42)
      # Each volume registered on its own
      Estimator("gqi-krr", estimates="eddy").run(dataset, seed=42)

5. **Save Results**: Once transformations are estimated, save the realigned data in your preferred output format, either :ref:`nfHDF5 <nfhdf5>` or NIfTI:

   .. code-block:: python

      # Save realigned DWI data in HDF5 format
      output_filename = "/path/to/save/your/output.h5"
      dataset.to_filename(output_filename)

   or as a NIfTI file:

   .. code-block:: python

     # Save realigned DWI data in NIfTI format
     output_filename = "/path/to/save/your/output.nii.gz"
     dataset.to_nifti(output_filename, write_estimates=True)

   With ``write_estimates=True``, the estimates are written next to it, into one X5
   file (``output.x5``): the head motion first, which plain X5 readers get, and the
   eddy currents after it, chained in the order they apply (``eddy @ motion``).

6. **Plotting**: Visualize data and results:

   Estimated motion results can be visualized using :doc:`/api/nifreeze.viz.motion_viz`
   functions. This includes visualizing framewise displacement, motion
   overlay, and volume-wise motion.

   Data results can be visualized using the NiReports_ functions, e.g.:

   - Use :func:`nireports.reportlets.mosaic.plot_mosaic` to visualize one
     direction of the dMRI dataset or a frame of an fMRI or PET dataset.
   - Employ :func:`nireports.reportlets.modality.dwi.plot_gradients` to
     visualize diffusion gradients.

   Example Usage:

   .. code-block:: python

      # Visualize 4D data at a specific index
      dataset.plot_mosaic(index=0)

   For dMRI gradients, we can visualize diffusion gradients using:

   .. code-block:: python

      # Visualize gradients
      dwi_data.plot_gradients()
