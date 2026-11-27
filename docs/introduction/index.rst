.. include:: ../links.rst

Overview
========
Often, neuroimaging modalities acquire a series of three-dimensional volumes:
the diffusion-weighted volumes of a diffusion MRI (dMRI) scan, the volumes of a
functional MRI (fMRI) run, the frames of a dynamic positron emission tomography (PET)
scan.
The head moves while the series is acquired, so each volume images the brain at a
slightly different position.
Unless the volumes are realigned, the analysis mixes signal from different locations
of the brain, and head motion biases its results.
Realignment usually registers each volume to a reference volume, one of the series or
their average.
This works when every volume shows the same contrast.
When contrast changes from one volume to the next, as it does across the diffusion
encodings of dMRI or as the tracer distributes during a PET scan, a volume and its
reference differ in more than their position, and the registration confounds the two.

*NiFreeze* estimates head motion by generating *unmoved* predictions of
the moved volume and finding a mapping through image registration.
The *unmoved* reference volume is predicted from a model of the signal.
The model is fit on all the *other* volumes so as not to bias the prediction,
in effect applying a *leave-one-volume-out* approach.
The data structures, the order in which volumes are visited and the estimator that
registers them are the same for every modality; only the model of the signal changes
(:doc:`/introduction/design`).

Diffusion MRI
-------------
Diffusion MRI generally employs echo-planar imaging (EPI) for fast whole-brain
acquisition.
Despite the rapid collection of volumes, typical repetition times are long enough for
head motion to occur, which has been proven detrimental to diffusion MRI
:footcite:p:`yendiki_spurious_2014`.
In the case of dMRI, additional volume-wise, low-order spatial distortions are caused
by eddy currents (EC), which appear as a result of quickly switching diffusion
gradients.
Unaccounted for EC distortion can result in incorrect local model fitting and poor
downstream tractography results
:footcite:p:`zhuang_correction_2006,andersson_comprehensive_2012`.
*FSL*'s ``eddy`` :footcite:p:`andersson_integrated_2016` is the most popular tool for
EC distortion correction, and implements a leave-one-volume-out approach to estimate
EC distortions :footcite:p:`andersson_non-parametric_2015`.
However, *FSL* has commercial restrictions that hinder application within open-source
initiatives such as *NiPreps* :footcite:p:`esteban_standardized_2025`.
In addition, *FSL*'s development model discourages the implementation of alternative
data-modeling approaches to broaden the scope of application (e.g., modalities beyond
dMRI).
*NiFreeze* is an open-source implementation of ``eddy``'s approach to estimate
artifacts that permits alternative models that apply to, for instance, head motion
estimation in fMRI and PET data.
Like ``eddy``, *NiFreeze* estimates each volume's motion and distortion with a
leave-one-volume-out scheme: a signal model fit on all the *other* volumes synthesizes
an independent, same-contrast registration target for the held-out volume, sidestepping
the ill-posed volume-to-volume registration of direction-varying dMRI contrast.
The diffusion models are described in :doc:`/methods/models`, and the global eddy model,
which estimates the eddy currents of all volumes at once, in :doc:`/methods/gem`.

Positron emission tomography
----------------------------
A dynamic PET scan is reconstructed into a series of frames, and can last an hour or
more.
Head motion over the scan degrades the images' resolution, misaligns them with the
attenuation correction, and disturbs the kinetic analysis
:footcite:p:`keller_methods_2012`.
Frame-by-frame realignment registers each frame to a reference frame, and improves the
reliability of the estimates of tracer binding :footcite:p:`montgomery_correction_2006`.
However, the tracer's distribution changes over the scan, so the frames differ in
contrast and in counts, and a frame and its reference differ in more than their
position.
A pharmacokinetic model has been built into the registration of the frames before
:footcite:p:`jiao_spatio-temporal_2014`.
*NiFreeze*'s B-spline model assumes no kinetic model: for each voxel, it fits a smooth
curve, a cubic B-spline with three interior knots evenly spaced over the scan by default,
to the values of the other frames at their mid-frame times, and evaluates the curve at the
time of the frame left out.
The prediction follows the time course of the tracer, so each frame's target has the
frame's contrast.
The estimator runs the model from Python (:class:`~nifreeze.model.pet.BSplinePETModel`).

Functional MRI
--------------
Functional MRI also employs EPI, and head motion during a run produces spurious but
systematic correlations in functional connectivity :footcite:p:`power_spurious_2012`.
The contrast of fMRI changes little from one volume to the next, so the usual realignment
registers each volume to a reference volume, one of the run or their mean
:footcite:p:`jenkinson_improved_2002`.
*NiFreeze* has, for now, only the average model (``avg``) for fMRI: its prediction is the
median of the other volumes, voxel by voxel, which differs little from the usual
realignment to the mean.
A model of the fMRI signal that predicts each volume better than the average is still to
be found, and is what *NiFreeze* needs to improve on the usual realignment.

*NiFreeze* succeeds *eddymotion*, whose `repository <https://github.com/nipreps/eddymotion>`__
is archived; its contributors are listed in
`.maint/EDDYMOTION.md <https://github.com/nipreps/nifreeze/blob/main/.maint/EDDYMOTION.md>`__.

.. footbibliography::
