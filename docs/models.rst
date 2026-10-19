.. include:: links.rst

.. _models:

=====================
Model implementation
=====================
This page collects implementation notes that complement the auto-generated API
documentation (see :doc:`api/nifreeze.model`). It documents design decisions and
empirical characterizations that are specific to *NiFreeze*'s **code** — not the
underlying domain theory, which lives in the project's grounding knowledge base.

.. _gqi-models:

Generalized q-Sampling Imaging (GQI)
====================================
:class:`~nifreeze.model.gqi.GeneralizedQSamplingModel` is a vendored, lightly
adapted copy of the GQI reconstruction of
`Yeh et al., 2010 <https://doi.org/10.1109/TMI.2010.2045126>`__ as implemented
in DIPY. The forward kernel
(:func:`~nifreeze.model.gqi.gqi_kernel`, ``method="standard"``) implements
the paper's spin-distribution-function reconstruction (Eq. 6/9) verbatim — it
matches DIPY's ``standard`` kernel and a direct transcription of Eq. 9 to machine
precision.

The model and its fit descend from DIPY's ODF-model family
(:class:`~dipy.reconst.odf.OdfModel` / :class:`~dipy.reconst.odf.OdfFit`), and
:meth:`~nifreeze.model.gqi.GeneralizedQSamplingFit.odf` provides the forward
signal → SDF transform expected of that family. Unlike DIPY's ODF models (and
*NiFreeze*'s DTI/DKI), GQI deliberately does **not** use the ``@multi_voxel_fit``
decorator: its fit is a linear operator that merely stores the data, and
prediction is a single vectorized matmul over all voxels, so per-voxel looping
would only discard that vectorization.

GQI as a signal predictor (a NiFreeze extension)
------------------------------------------------
Yeh (2010) defines only the *forward* map signal → SDF. *NiFreeze* additionally
needs the *inverse* to predict the diffusion signal given a b-vector (and -value).
:meth:`~nifreeze.model.gqi.GeneralizedQSamplingFit.predict` composes the
forward GQI kernel with a Tikhonov-regularized reconstruction kernel
(:func:`~nifreeze.model.gqi.prediction_kernel`):

.. math::

   \mathbf{K}^{+} = (\mathbf{K}\mathbf{K}^{\mathsf T} + \lambda_0 \mathbf{I})^{-1}
   \mathbf{K}, \qquad \lambda_0 = 10^{-6}.

This is the regularized least-squares signal whose forward GQI transform
reproduces the fitted SDF. It is a *NiFreeze* modeling choice and is **not** part
of Yeh (2010).

.. _gqi-reconstruction-fidelity:

Reconstruction fidelity and the intercept behaviour
---------------------------------------------------
Applied to the signal, the round-trip operator is
:math:`\mathbf{P} = \mathbf{A}(\mathbf{A}+\lambda_0\mathbf{I})^{-1}` with
:math:`\mathbf{A} = \mathbf{K}\mathbf{K}^{\mathsf T}`. Empirically, on
diffusion-weighted (``b > 0``) volumes:

- :math:`\mathbf{P}` **preserves the mean/DC exactly**
  (:math:`\mathbf{P}\cdot\mathbf 1 = \mathbf 1`)
  and is **scale-homogeneous** (a per-voxel ``S0`` cancels). So there is *no*
  missing additive baseline/``S0`` term, and normalizing by ``S0`` changes
  nothing.
- What it does lose is **angular signal energy**: :math:`\mathbf{P}` is a
  contraction that projects out the part of the signal lying in the null space of
  the GQI basis. The per-voxel regression slope of *predicted vs. observed* is the
  fraction of angular variance the basis can represent (≈ 0.76 on single-shell
  HARDI, ≈ 0.99 on rich grid sampling).
- The apparent non-zero *intercept* is **not** an independent term because the
  mean is preserved: ``intercept = mean · (1 − slope)`` exactly. There is a single
  effect (angular shrinkage), not two.

Consequently the missing degree of freedom is a **representation gap**, worse for
q-space-poor acquisitions (single-shell) than for grid/multi-shell data, not an
``S0`` intercept. Two practical corollaries:

- **b=0 volumes are excluded** from GQI fitting/prediction in *NiFreeze*. Feeding
  raw b=0 signal into the fit *without* an explicit constant term degrades the
  diffusion-weighted reconstruction (the large b=0 amplitude is not representable
  by the diffusion-weighted sinc basis); excluding b=0 avoids this cleanly.
- ``method="gqi2"`` is a **weaker signal predictor** than the default
  ``"standard"``: its kernel is more oscillatory and ill-conditioned, so its
  round-trip does not preserve signal scale and its leave-one-volume-out
  correlation falls below that of ``"standard"``. This is why *NiFreeze* defaults
  to ``"standard"`` even though DIPY's GQI defaults to ``"gqi2"``.

For motion estimation this amplitude shrinkage is benign (registration keys on
relative contrast, which the high correlation preserves); it would matter for a
downstream *quantitative* use of the predicted signal.

.. _gqi-sphere-density:

Sphere density and the default recursion level
----------------------------------------------
The SDF is sampled on an icosahedral sphere whose subdivision
``recursion_level`` (a parameter of
:class:`~nifreeze.model.gqi.GeneralizedQSamplingModel`) sets its vertex count. The
default, :data:`~nifreeze.model.gqi.DEFAULT_SPHERE_RECURSION_LEVEL` = 5 (1026
vertices), was chosen from the following experiment.

**Settings.** ``method="standard"``, ``sampling_length`` :math:`\sigma = 1.2`,
``INVERSE_LAMBDA`` :math:`\lambda_0 = 10^{-6}`. Metric: per-voxel round-trip
slope (fit the model on the diffusion-weighted set, reconstruct that *same* set,
then take the least-squares slope of predicted vs. observed, averaged over
voxels) and the mean per-voxel correlation. **No direction is held out**: this
is a self-consistency measure that isolates the reconstruction, not a
leave-one-out prediction.

**Data.** (1) *Grid / multi-shell*: DIPY's ``dsi_voxels()`` phantom, b=0 excluded
→ 101 diffusion-weighted directions, 600 voxels. (2) *Single-shell*: the Stanford
HARDI dataset (DIPY ``get_fnames("stanford_hardi")``), central slab
``[20:60, 40:80, 28:45]``, brain = b=0 reference above its 60th percentile → 150
diffusion-weighted directions, 10 880 voxels. b=0 excluded in both.

.. list-table:: Round-trip fidelity vs. sphere subdivision (standard method)
   :header-rows: 1
   :widths: 10 12 16 14 16 14

   * - ``recursion_level``
     - vertices
     - grid slope
     - grid corr
     - single-shell slope
     - single-shell corr
   * - 2
     - 18
     - 0.759
     - 0.839
     - 0.339
     - 0.377
   * - 3
     - 66
     - 0.829
     - 0.952
     - 0.519
     - 0.700
   * - 4
     - 258
     - 0.931
     - 0.974
     - 0.750
     - 0.869
   * - 5 (default)
     - 1026
     - 0.965
     - 0.980
     - 0.765
     - 0.879
   * - 6
     - 4098
     - 0.992
     - 0.983
     - 0.788
     - 0.898

**Reading.** Both curves rise steeply up to a knee at ``recursion_level`` 4–5;
the operator's effective rank saturates there (beyond it, a denser sphere only
refines the quadrature). Past the knee the two data types diverge: the grid
acquisition keeps improving toward a near-perfect round-trip (slope → 0.99) — it
is *sphere-limited* — whereas the single-shell acquisition plateaus near 0.78 — it
is *q-space-limited*, and no sphere density recovers the angular signal a single
shell cannot encode.

**Decision.** The default ``recursion_level = 5`` sits just past the knee for both
regimes, so it is adequate for the single-shell data *NiFreeze* typically sees
while leaving headroom for grid/multi-shell acquisitions, where a higher value
(e.g. 6) keeps paying off. Denser spheres cost compute (``O(vertices)`` per kernel
build) for no single-shell benefit, so they are opt-in rather than the default.

**Reproduction.**

.. code-block:: python

   import numpy as np
   from dipy.core.gradients import gradient_table
   from dipy.core.subdivide_octahedron import create_unit_sphere
   from dipy.data import dsi_voxels
   from nifreeze.model.gqi import gqi_kernel

   L, lam = 1.2, 1e-6
   data, gtab = dsi_voxels()
   dw = np.where(~gtab.b0s_mask)[0]
   g = gradient_table(bvals=gtab.bvals[dw], bvecs=gtab.bvecs[dw])
   d2d = data.reshape(-1, data.shape[-1])[:, dw]
   for r in (2, 3, 4, 5, 6):
       K = gqi_kernel(g, L, create_unit_sphere(recursion_level=r), method="standard")
       A = K @ K.T
       P = np.linalg.inv(A + lam * np.eye(A.shape[0])) @ A
       p = np.maximum(d2d @ P.T, 0)
       dc, pc = d2d - d2d.mean(1, keepdims=True), p - p.mean(1, keepdims=True)
       slope = ((dc * pc).sum(1) / ((dc**2).sum(1) + 1e-12)).mean()
       print(r, len(create_unit_sphere(recursion_level=r).vertices), round(float(slope), 3))

.. _closed-form-lovo:

Closed-form leave-one-volume-out prediction
===========================================
Leave-one-volume-out (LOVO) prediction keeps each registration target
independent of the volume being registered. Refitting a model for every held-out
volume is the direct way to achieve this, but for kernel models it is
unnecessary. With fixed hyperparameters, a kernel predictor is linear in the data
and its weights depend only on the gradient table. If :math:`\mathbf{A}` is the
covariance of the :math:`N` diffusion-weighted volumes (including its noise or
ridge term) and :math:`\mathbf{P} = \mathbf{A}^{-1}`, the prediction of volume
:math:`i` from all the others is Gaussian conditioning
(:footcite:t:`rasmussen_gaussian_2006`, Eq. 5.12):

.. math::

   \hat{y}_i = -\sum_{j \neq i} \frac{P_{ij}}{P_{ii}}\, y_j, \qquad
   \mathbf{S} = \mathbf{I} - \operatorname{diag}(\mathbf{P})^{-1}\mathbf{P}.

All :math:`N` LOVO predictions are then rows of one smoother
:math:`\mathbf{S}` (:func:`~nifreeze.model.lovo.lovo_smoother`), whose diagonal
is zero. One :math:`N \times N` inverse replaces :math:`N` model fits, and each
prediction is a single matrix-vector product over the masked voxels. When the
model estimates a constant mean on the :math:`N - 1` training volumes (as
:class:`~nifreeze.model.gpr.DiffusionGPR` does with ``normalize_y=True``), a
closed-form correction makes every row sum to one; the standard deviation of the
normalization cancels in the posterior mean.

Two models implement this through
:class:`~nifreeze.model.lovo.ClosedFormLOVOMixin`, leaving the refitting models
unchanged:

``gp-loo`` (:class:`~nifreeze.model.dmri.GPLOOModel`)
   The Gaussian process of :footcite:t:`andersson_non-parametric_2015`, with
   :math:`\mathbf{A} = \mathbf{K}_\theta(\mathbf{X}) + \sigma^2\mathbf{I}`.
   Hyperparameters :math:`\theta` are either loaded from a JSON file
   (``hyperparameters``, CLI ``--gp-hyperparameters``) or fitted **once**, on a
   random subsample of voxels across all volumes (``n_voxels_hp``, CLI
   ``--nvoxhp``, 1000 by default, like eddy's ``--nvoxhp``
   :footcite:p:`andersson_integrated_2016`).

``gqi-krr`` (:class:`~nifreeze.model.dmri.GQIKRRModel`)
   GQI :footcite:p:`yeh_generalized_2010` as kernel ridge regression. The GQI
   basis :math:`\mathbf{K}` induces the q-space Gram matrix
   :math:`\mathbf{G} = \mathbf{K}\mathbf{K}^{\mathsf T}` and
   :math:`\mathbf{A} = \mathbf{G} + \lambda\mathbf{I}`, with :math:`\lambda`
   relative to the mean diagonal of :math:`\mathbf{G}`. Unlike the one-row
   prediction of :class:`~nifreeze.model.dmri.GQIModel`, which normalizes by the
   target direction's own norm and overscales the signal, it predicts from the
   training Gram matrix. :math:`\lambda` is given (``krr_lambda``) or chosen by
   closed-form LOVO error (:func:`~nifreeze.model.lovo.select_krr_lambda`).

**Held-out independence.** No prediction uses the held-out volume, so single-fit
mode returns the same genuine LOVO predictions and is not a canary. The
hyperparameters fitted at run time (a few scalars) do see all volumes, as in
eddy. Hyperparameters fitted on other subjects remove even that dependence.

**Pooling across subjects.** The hyperparameters of an acquisition protocol can
be fitted on many subjects with
:func:`~nifreeze.model.lovo.fit_pooled_hyperparameters`, which maximizes the sum
of the per-subject log marginal likelihoods, each subject with its own gradient
table (identical tables make this equal to fitting the concatenated voxels).
:func:`~nifreeze.model.lovo.save_hyperparameters` writes the result for
``--gp-hyperparameters``:

.. code-block:: json

   {
     "schema": "nifreeze-gp-hyperparameters/1",
     "kernel_model": "spherical",
     "hyperparameters": {"k1__beta_a": 1.6, "k1__beta_l": 0.5, "k2__noise_level": 1e-05},
     "provenance": {"n_subjects": 20}
   }

Because :math:`\mathbf{S}` depends on the gradients only through their pairwise
angles (and b-values), a global rotation of the gradient table leaves it
unchanged. Rotating a single volume's b-vector changes only one row and one
column of :math:`\mathbf{A}`.

**Caveats.** :func:`~nifreeze.model.lovo.lovo_smoother` warns when
:math:`\mathbf{A}` is ill-conditioned. The constant mean is shared across shells,
as in :class:`~nifreeze.model.dmri.GPModel`. Related ideas exist for denoising,
where Patch2Self regresses each volume on the others
:footcite:p:`fadnavis_patch2self_2020`.

.. rubric:: References

.. footbibliography::
