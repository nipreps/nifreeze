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

The dMRI data class holds a single ``b0`` volume. The rationale behind this is
that this volume serves as a canonical reference for transforming non-zero
gradient volumes. Users can compute such reference ``b0`` volume using their
preferred method and provide it when instantiating a :class:`api/nifreeze.data.dmri.DWI`
class (e.g. through the :meth:`~nifreeze.data.dmri.io.from_nii`); otherwise,
*NiFreeze* computes a reference ``b0`` as the median volume across
non-diffusion-weighted volumes. Note that following this design choice,
the gradient data hosted by the :class:`api/nifreeze.data.dmri.DWI` instance
only contains (nonzero) diffusion-weighted gradient values.

The gradient table's convention and the files *NiFreeze* saves datasets in are
described in :doc:`/usage/data`.

Preprocessing
^^^^^^^^^^^^^

*NiFreeze* provides a set of minimal preprocessing utilities under the form
of :class:`api/nifreeze.estimator.Filter`\s. :class:`api/nifreeze.estimator.Filter`\s
can be specified as inputs to estimators so that the data used by the
estimator is processed through the given filter instance.

Modeling
^^^^^^^^

These modules encapsulate the algorithms used for artifact estimation and
correction. By abstracting the modeling logic, *NiFreeze* enables the
integration of diverse methodologies tailored to specific research
requirements.

Models **do not** preprocess the data, and users are expected to have their
data preprocessed minimally (e.g., computing a unique, meaningful *b0* volume
for DWI data, smoothing and thresholding the PET data, clipping or detrending
the data).

Iterators
^^^^^^^^^

Iterators in *NiFreeze* manage the traversal of data volumes, allowing access
operations to particular data volumes. They are designed to be extensible,
allowing developers to implement custom traversal logic as needed.


Estimation
^^^^^^^^^^

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

  The **Update & iterate** procedure treats registration and modeling
  differently:

  - **Registration never sees resampled data.** Each volume is registered in
    its original form. The current estimate (found by an earlier estimator of
    the chain, or stored with the dataset) only initializes the registration,
    so that it starts closer to the target.
  - **Models train on the realigned view** (the *instant update*). As soon as a
    volume is registered, the models see it resampled from the original with
    its estimate and, for DWI, with its b-vector rotated accordingly. The
    predictions of the remaining volumes, and those of later estimators,
    improve as the motion is corrected.

  A realigned volume is never resampled again: a new estimate replaces it,
  resampled from the original. ``--no-instant-update``
  (``Estimator.run(instant_update=False)``) trains the models on the original
  volumes only.

Corrected datasets and associated metadata are produced in standardized
formats for downstream analysis.

Fitting and Prediction
^^^^^^^^^^^^^^^^^^^^^^

In the default Leave-One-Volume-Out (LOVO) mode described above, each call to a
model's ``fit_predict(index)`` fits on every volume *except* ``index`` and
predicts the held-out one. This held-out independence is what makes the
registration target valid: the prediction volume ``k`` is registered against
must not itself depend on volume ``k``.

Models also support a *single-fit mode*, activated by calling

.. code-block:: python

    model.fit_predict(None)

which fits the model once on all available data (that is, no volume is held out) and *locks* that fit (stored in ``_locked_fit``). Subsequent calls reuse the
locked model fit instead of refitting.
Do not confuse the *single-fit mode* with *returning the same volume regardless
of the index*.

Single-fit forfeits *independence*: because the queried volume was part
of the data the locked fit was trained on, its prediction is no longer the
unbiased, out-of-sample estimate LOVO provides. Therefore, single-fit can be
used where this bias is either absent or tolerable. For example, contexts where
the bias is anticipated to be negligible, such as a coarse, low-DOF early
rigid-body realignment that a subsequent LOVO stage refines, or trading accuracy
for speed during development or debugging.

Cascades
^^^^^^^^
The *NiFreeze* command-line interface supports specifying multiple models, which are applied in a cascade fashion. Users
can indicate which model should be used to address a specific problem, such as head motion or eddy current correction.

The transformations estimated at each level initialize the registration of the next level, whose models train on
the data realigned with them.

Refer to the :ref:`usage` document for a self-contained usage example.
