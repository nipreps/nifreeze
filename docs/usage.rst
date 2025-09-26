.. include:: links.rst

How to Use
==========
Incorporating NiFreeze into a Python module or script
-----------------------------------------------------
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

   For example, for dMRI data, ensure the gradient table is provided. It
   should have one column per diffusion-weighted image. The first three rows
   represent the gradient directions, and the last row indicates the timing
   and strength of the gradients in units of s/mm² ``[ R A S+ b ]``.

   .. code-block:: python

      # Load dMRI data into a DWI object
      dataset = dmri.load("/path/to/your/dwi_data.nii.gz", gradients_file="/path/to/your/gradient_file")

   .. note::

      To run the examples and tests from this page,
      find `sample data <https://osf.io/download/6at98/>`__ on OSF.
      To load from an HDF5 file, use:

      .. code-block:: python

         dataset = dmri.DWI.from_filename("/path/to/downloaded/dwi_full.h5")


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
     available models, see :doc:`api/nifreeze.model`.
   - ``strategy``: strategy used to traverse the 4D sequence. The list of
     supported strategies can be found at :doc:`api/nifreeze.utils.iterators`.
   - ``prev``: estimators can be stacked and be run sequentially.

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

   The ``run`` method employs the Leave-One-Volume-Out (LOVO) splitting technique to iteratively process DWI data volumes for each specified model. Affine transformations align the volumes, updating the `DWI` object with the estimated parameters. This method accepts several parameters:

   - ``dataset``: The target dataset, represented by this tool's internal type.
   - ``align_kwargs``: Parameters to configure the image registration process.
   - ``omp_nthreads``: Maximum number of threads an individual process may use.
   - ``n_jobs``: Number of parallel jobs.
   - ``seed``: Seed for the random number generator (necessary for deterministic estimation).

   The estimated parameters encoding the deformations due to head motion and
   eddy currents are stored as an :math:`N \times 4 \times 4` array of affine
   matrices in the dataset object after the model fitting and prediction (done
   under the hood by the estimator).

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

5. **Save Results**: Once transformations are estimated, save the realigned data in your preferred output format, either HDF5 or NIfTI:

   .. code-block:: python

      # Save realigned DWI data in HDF5 format
      output_filename = "/path/to/save/your/output.h5"
      dataset.to_filename(output_filename)

   or as a NIfTI file:

   .. code-block:: python

     # Save realigned DWI data in NIfTI format
     output_filename = "/path/to/save/your/output.nii.gz"
     dataset.to_nifti(output_filename)

6. **Plotting**: Visualize data and results:

   Estimated motion results can be visualized using :doc:`api/nifreeze.viz.motion_viz`
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
