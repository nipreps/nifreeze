Design Principles
=================

Overview
--------

*NiFreeze*'s architecture is modular, facilitating extensibility and
adaptability. *NiFreeze* is designed around the following core concepts:

- **Data ingestion**: Handles input data, providing generalized 4D data
  management across modalities.
- **Preprocessing pipelines**: Implements modality-specific preprocessing
  steps prior to artifact estimation. Examples of preprocessing operations are
  value clipping and smoothing.
- **Modelling**: Implements statistical models to estimate and correct
  artifacts. Models follow a ``fit & predict`` design, where the ``fit`` step
  fits the model to the data, and the ``predict`` step estimates the volume
  that has been held out.
- **Estimation**: Orchestrates the registration target prediction. Estimates
  the motion and distortion artifacts of the data employing a particular
  model and iterating over the data using a leave-one-volume-out approach. It
  computes the registration transforms on the estimated, artifact-free volumes.
  Estimator instances can be stacked so that the output of an estimator
  instance becomes the input to the subsequent instance.

Main Concepts
-------------

Data Ingestion
^^^^^^^^^^^^^^

*NiFreeze* implements a set of modality-specific 4D data readers that expose a
uniform API, enabling transparent access to individual volumes by index.

dMRI
""""

dRMI modality gradients are handled internally arrays arranged according to
the ``[gx gy gz b]`` layout, where ``[gx gy gz]`` are the components of a
given gradient vector, and ``b`` is the b-value in units of :math:`s/mm^2`.

# ToDo
# Or use [ R A S+ b ] instead of ``[gx gy gz b]``  as in usage.rst

The dMRI data class holds a single ``b0`` volume. The rationale behind this is
that this volume serves as a canonical reference for transforming non-zero
gradient volumes. Users can compute such reference ``b0`` volume using their
preferred method and provide it when instantiating a :class:`api/nifreeze.data.dmri.DWI`
class (e.g. through the :meth:`~nifreeze.data.dmri.io.from_nii`); otherwise,
*NiFreeze* computes a reference ``b0`` as the median volume across
non-diffusion-weighted volumes. Note that following this design choice,
the gradient data hosted by the :class:`api/nifreeze.data.dmri.DWI` instance
only contains (nonzero) diffusion-weighted gradient values.

Preprocessing
-------------

*NiFreeze* provides a set of minimal preprocessing utilities under the form
of :class:`api/nifreeze.estimator.Filter`\s. :class:`api/nifreeze.estimator.Filter`\s
can be specified as inputs to estimators so that the data used by the
estimator is processed through the given filter instance.

Modeling
--------

These modules encapsulate the algorithms used for artifact estimation and
correction. By abstracting the modeling logic, *NiFreeze* enables the
integration of diverse methodologies tailored to specific research
requirements.

Models **do not** preprocess the data, and users are expected to have their
data preprocessed minimally (e.g., computing a unique, meaningful *b0* volume
for DWI data, smoothing and thresholding the PET data, clipping or detrending
the data).

Iterators
---------

Iterators in *NiFreeze* manage the traversal of data volumes, allowing access
operations to particular data volumes. They are designed to be extensible,
allowing developers to implement custom traversal logic as needed.


Estimation
----------

*NiFreeze* estimators rely on predictive models to reconstruct artifact-free
volumes from artifact-affected 4D datasets. Each model maintains access to the
full dataset, while the estimator traverses the data according to a specified
iteration strategy (e.g., forward, reverse, or random ordering). At each step,
the estimator selects a target (held-out) volume as defined by the iterator,
fits the model on the remaining volumes, and predicts the held-out volume. A
registration process estimates the affine transformation between the held-out
and predicted volumes, storing the result at the corresponding index in the
dataset's motion parameter attribute.

.. important::

  The **Update & iterate** procedure follows a strict design principle:
  resampling is intentionally excluded from the update step; including it
  would alter the data distribution, making it impossible to attribute
  observed improvements solely to the predictive model.

Corrected datasets and associated metadata are produced in standardized
formats for downstream analysis.

Extending *NiFreeze*
====================

To extend *NiFreeze*'s functionality,

Creating New Iterators
----------------------

To extend *NiFreeze*'s functionality, such as implementing a custom iterator,
follow these general steps:

1. **Create a new function**: Define a new iterator function implementing the
   desired logic.

   .. code-block:: python

    from typing import Iterator

    def my_iterator(**kwargs) -> Iterator[int]:
        # Size is the number of volumes in the dataset
        size = get_size(**kwargs)

        # Implement custom iteration logic
        index_order = establish_order(size)

        return (x for x in index_order)


1. **Integrate the function**: Incorporate the newly created iterator into the
   *NiFreeze* workflow by updating the configuration or pipeline definitions to
   utilize your new implementation.

   .. code-block:: python

    list(my_iterator(10))


1. **Test the Implementation**: Ensure that your custom class functions as intended by
   running unit tests and validating the outputs against expected results.

Creating New Models
-------------------

To implement a new model:

1. Subclass the :class:`api/nifreeze.model.base.BaseModel` class.
1. Implement the ``fit(data)`` method, which returns a 3D volume.
1. Register the model in your pipeline or CLI wrapper.

Example: Extrapolation model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from nifreeze.data.base import BaseDataset
    from nifreeze.model.base import BaseModel

    class ExtrapolationModel(BaseModel):
        """Model a volume extrapolating across training volumes."""

        __slots__ = {
            "_param1": "Parameter 1",
            "_param2": "Parameter 2",
        }

        def __init__(self, dataset: BaseDataset, param1: float, param2: float, **kwargs):

            super().__init__(dataset, **kwargs)

            self._param1 = param1
            self._param2 = param2

        def _fit(self, index, n_jobs=None, **kwargs):
            # Fit the data
            pass

        def fit_predict(self, index: int, **kwargs):
            # Compute prediction
            return pred

so ``fit_predict`` is suitable for case where the model is fit only once per iteration, and ``predict`` is for cases where
the model is fit across multiple volumes in each iteration.

Usage
^^^^^

.. code-block:: python

    model = ExtrapolationModel()
    result = model.fit(data, index=3)


Fitting and Prediction
----------------------

Models have a ``_locked_fit`` property that allows to fit all available data if no index is provided. This is achieved
by calling::

   .. code-block:: python

   model.fit_predict(None)

In this case, the predicted data will always be the same regardless of the index. The ``single_`` prefix in the fitting
strategy provided to the :class:`api/nifreeze.estimator.Estimator` instances tells the estimator to proceed this way.


Usage
-----

The *NiFreeze* command-line interface supports specifying multiple models, which are applied in a cascade fashion. Users
can indicate which model should be used to address a specific problem, such as head motion or eddy current correction.

The transformations computed at each level are stored in an HDF5 file and are used to initialize the immediately next
level of the registration process.

Refer to the :ref:`usage` document for a self-contained usage example.

Contributing to NiFreeze
========================

Set up for Development
----------------------

Install the appropriate modules::

   .. code-block:: bash
   pip install .[test, doc, style, contributing]

Download data from OSF, OpenNeuro, etc.::

   .. code-block:: bash
   download

Typechecking, spellchecking, style, doc

Developer Resources
-------------------

Refer to the :ref:`developers` document for detailed development guidelines.

For guidance on contributing to *NiFreeze*, refer to the *NiPreps*
Contributing Guidelines:

https://www.nipreps.org/community/CONTRIBUTING/

This resource provides detailed instructions on community principles and
contributor acknowledgements.
