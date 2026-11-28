.. include:: ../links.rst

The global eddy model (GEM)
===========================
*NiFreeze* can estimate the eddy currents of a DWI by registering each volume, on its
own, to a prediction of it (``--eddy``).
The global eddy model (GEM) estimates them for all the volumes at once instead, with a
model of how the field of the eddy currents follows the diffusion gradient
(:class:`~nifreeze.registration.gem.GEM`,
`#746 <https://github.com/nipreps/nifreeze/pull/746>`__).
Its accuracy on the phantom is reported in :doc:`/edd/gem-accuracy`.

The model
---------
A field of eddy currents that is linear in position moves points only along the
phase-encoding direction :math:`u`.
Its mapping is :math:`E = I - u s^T`: a point :math:`x` moves by :math:`-(s^T x)\, u`,
where the three *slopes* :math:`s` set how the displacement grows along each axis of
the scanner.
The field comes from the diffusion gradient, so its slopes follow the gradient.
In volume :math:`v`, with the unit b-vector :math:`g_v` and the b-value :math:`b_v`,

.. math::

    s_v = \left(\frac{b_v}{b_\text{ref}}\right)^p C g_v,

where the eddy-current tensor :math:`C` (3 × 3) and the exponent :math:`p` are the
same for every volume (:class:`~nifreeze.registration.gem.EddyTensor`), and
the reference b-value :math:`b_\text{ref}` is the geometric mean of the b-values.
:math:`C` maps the gradient's direction to the field's slopes: with the scanner's
frame rotated by :math:`R`, it becomes :math:`R C R^T`, as a tensor does.
The exponent is 1/2 when the field follows the gradient's strength at a fixed timing,
and 1 when it follows the b-value, as in the phantoms that TRXScan makes
(:doc:`/edd/phantoms`).
GEM estimates :math:`p` between 0 and 2.

The fields are zero at one point, by default the center of the field of view, the
middle of its physical extent
(:func:`~nifreeze.registration.gem.field_of_view_centre`).
A field that is zero at the point :math:`z` also translates the volume along
:math:`u`, by :math:`s^T z`.
That translation is part of the head motion, by the convention of
:func:`~nifreeze.registration.utils.split_phase_encoding`, which splits a volume's
mapping into head motion and a field without translation.
GEM derives each volume's translation along :math:`u` from its slopes and the zero
point, and each volume keeps the rest of the head motion it is given.

GEM comes in two forms:

One tensor for all the volumes
    One :math:`C` and one :math:`p` (the default): 10 parameters, or 9 with the
    exponent held.
One tensor for each shell
    One :math:`C` for each segment of b-values, between the edges that the caller gives
    (:class:`~nifreeze.registration.gem.SegmentedEddyTensor`).
    The exponent is held, at 1/2 by default, and each segment's reference b-value is
    the geometric mean of its b-values.
    On shelled data, every volume of a segment has the same b-value, so the exponent
    makes no difference.

A stretch that every volume shares has no parameter of its own to go into, as long as
the b-vectors average out.
Fitted on the images, where the volumes weigh a little differently, a small part of it
still reaches the fields.

.. _gem-bvectors:

The b-vectors
-------------
A GEM pass reads each volume's b-vector in two frames: the scanner's, for the field,
and the head's, for the target.

The field follows the b-vector as acquired.
The eddy currents come from the gradient the scanner plays, which stays put when the
head moves.
So :math:`g_v` in the slopes above is the b-vector of the acquisition, in world (RAS+)
coordinates, never rotated by the head motion.
The pass gives :meth:`~nifreeze.registration.gem.GEM.fit` the dataset's own gradient
table, which keeps the b-vectors as they were read.

The target follows the b-vector in the head.
Each target predicts its volume as the head shows it at the reference position.
When the head turns by :math:`Q_v`, the gradient :math:`g_v` points along
:math:`Q_v^{-1} g_v` in the head, and the model predicts the volume there
(:func:`~nifreeze.data.dmri.utils.realign_bvecs`).
With the instant update (:doc:`/introduction/design`), the realigned view that the model
trains on holds each volume realigned and its b-vector rotated, together.
Without it (``--no-instant-update``), the model trains on the original volumes, at the
b-vectors as acquired.
GEM derives only each volume's translation along :math:`u`, and keeps the rotation of
the head motion it is given, so the rotated b-vectors stay right through cascaded
passes (:ref:`gem-cascade`).

The outputs' b-vectors turn with the head motion alone, as the view's do: the eddy
currents displace the image without turning the head
(:meth:`~nifreeze.data.dmri.DWI.to_nifti`).

How GEM is estimated
--------------------
Estimating GEM is one registration of all the volumes at once, with one set of
parameters, in which each volume moves by its own field.
The cost is the sum, over the volumes, of ITK's similarity metric between each
volume and its target, run in the stages of the ``itk`` section of
:data:`~nifreeze.registration.base.AFFINE`.
ITK's metric evaluates each volume's similarity and its derivative, and scipy's
L-BFGS-B estimates the model's parameters.
The images are prepared as for ITK's registration of each volume on its own: clipped to
the volume's intensity window, winsorized and histogram-matched as the settings ask.

Running GEM
-----------
GEM runs in the estimator's eddy-current pass, given as its backend:

.. code-block:: python

    from nifreeze.estimator import Estimator
    from nifreeze.registration.gem import GEM

    Estimator("gqi-krr", estimates="eddy", backend=GEM()).run(dwi)

``backend="gem"`` does the same: the estimator builds GEM for each run, with its
defaults and the run's ``align_config``, ``clip`` and ``seed``.
The pass predicts every volume first, all from the same data, and adds each to the fit
with its head motion; it then estimates the model and stores each volume's head motion
and field in the dataset.
The fit takes the dataset's phase-encoding direction, unless it is given its own
(``phase_encoding``).
One tensor for each shell takes the edges between the shells, for instance
``GEM(edges=(1500, 2750))`` for shells at b=1000, 2000 and 3500 s/mm².
``zero_point`` sets where the fields are zero, in world (RAS+) coordinates, in mm, and
``exponent`` holds :math:`p`.

On the command line, GEM is the default eddy-current correction of a DWI: three
cascaded passes (:ref:`gem-cascade`), after the head-motion step, with the model of
its cascade's last level (gqi-krr by default) unless ``--eddy-model`` names another.
``--gem-passes`` sets the number of passes:

.. code-block:: bash

    nifreeze dwi.h5                        # head motion, then three passes of GEM
    nifreeze dwi.h5 --gem-passes 1         # one pass
    nifreeze dwi.h5 --gem-passes 0         # head motion only
    nifreeze dwi.h5 --gem-passes 0 --eddy  # each volume registered on its own instead

GEM needs the phase-encoding direction: the input's BIDS sidecar or *nfHDF5* file give
it, or ``--phase-encoding``.
Without it, the default run estimates head motion only, with a warning.
The command line fits GEM in its default form; its other options are set in Python.

On DIPY's Sherbrooke data, the default run took 17 min on six CPUs, against 5.4 min
for head motion alone.

.. figure:: ../_static/gem/gem-sherbrooke.svg
   :width: 100%

   The default run on DIPY's Sherbrooke 3-shell data (64 volumes at each of b=1000,
   2000 and 3500 s/mm², 2 mm voxels), given phase encoding along :math:`j`, played in
   acquisition order, a fifth of a second per volume.
   Left: the original volumes.
   Middle: corrected for head motion only (``--gem-passes 0``).
   Right: the default run, corrected for head motion and three passes of GEM.
   The orange line outlines the brain at the reference position and is the same in
   every frame.
   The header gives how far each volume's corrections move the brain.
   On these data, the eddy currents are well below the voxel size: 0.13 mm in the
   median volume (the median over the brain), and at most 0.86 mm in any voxel, against
   0.35 mm and 1.80 mm for head motion.
   So the right column differs little from the middle one.

``tools/gem_sherbrooke.py`` makes both runs and the figure, on CPUs 0-5:

.. code-block:: bash

    taskset -c 0-5 python tools/gem_sherbrooke.py --work <work> --out docs/_static/gem

The animations below play how the default run corrects each volume, on three axial,
three coronal and three sagittal cuts.
Each volume goes through five steps:

1. the volume as found (*original*);
2. a quiver of where GEM's field sends its voxels, in the volume as acquired;
3. the data travelling there (*eddy-corrected*);
4. a quiver of where head motion sends them next;
5. the data travelling into the reference (*realigned*).

This is the order in which realignment undoes the two corrections.
For each point of the reference, it reads the volume where head motion takes the point,
moved along phase encoding by the field.
So the field is undone first, in place, where the volume was acquired, and head motion
after it.
The third step's frame is the volume read through the field alone, and the last one is
the volume read once through both, as the command line writes it.

On the phantom (:doc:`/edd/phantoms`), the copy with the largest head motion and eddy
currents shows both corrections at work, against a known truth.
Under the Sherbrooke protocol, every diffusion-weighted volume of this copy is turned
by 4° and shifted by 4 mm, and its eddy currents move some voxel of the brain by 8
voxels (16 mm) along phase encoding (:math:`j-`).
The copy is noise-free, as simulated.
The default run took 15 min on six CPUs.

.. raw:: html

   <object data="../_static/gem/corrections-phantom.svg" type="image/svg+xml"
           style="width: 100%; aspect-ratio: 598 / 498">
     <p><a href="https://gin.g-node.org/nipreps-data/tests-nifreeze/raw/master/docs/corrections-phantom.svg">
     The animation</a> (21 MB) is downloaded when the documentation is built.</p>
   </object>

In the median volume, the estimated head motion moves the brain by 4.63 mm (the median
over the brain), against a true 4.77 mm.
The estimated eddy currents move it by 2.30 mm, against a true 2.56 mm.
The corrected volume ends 1.67 mm from where the truth would put it: 1.30 mm at b=1000,
1.78 mm at b=2000 and 2.46 mm at b=3500.
Four volumes, all at b=3500, end more than 5 mm away.
In the worst, volume 181, head motion is estimated at 13.2 mm against a true 4.6 mm.
Its solid outline does not fit the original volume, and its realigned cuts are off the
reference.
``tools/gem_phantom.py`` makes the run and the animation:

.. code-block:: bash

    taskset -c 0-5 python tools/gem_phantom.py --dataset <phantoms> --work <work> \
        --out docs/_static/gem

On DIPY's Sherbrooke data, both corrections are far smaller:

.. raw:: html

   <object data="../_static/gem/corrections-sherbrooke.svg" type="image/svg+xml"
           style="width: 100%; aspect-ratio: 592 / 492">
     <p><a href="https://gin.g-node.org/nipreps-data/tests-nifreeze/raw/master/docs/corrections-sherbrooke.svg">
     The animation</a> (28 MB) is downloaded when the documentation is built.</p>
   </object>

The dashed line outlines the brain mask at the reference position.
The solid line is the mask moved by head motion and the eddy currents: it should fit the
original volume, and it travels with the data into the reference.
Arrows are drawn longer than the voxels move, by the factors in each legend.
On the phantom, they are 2.5 times longer than the eddy currents move the voxels, and as
long as head motion does.
On the Sherbrooke data, they are 30 and 10 times longer.
The data travel at true scale.
A displacement across the cut is drawn as a circle of that size: ⊙ toward you, ⊗ away.
With phase encoding along :math:`j`, GEM moves voxels only across the coronal cuts.
In each cut's corner, the b-vector is projected onto the cut.
It turns with the head during the realignment, as the b-vectors the command line
writes do.
GEM's field is drawn zero at the centre of the field of view, where GEM fits it, and
the shift it makes there counts as head motion.
Either way, the last frame is the volume the command line writes.
Between the frames of each step, which are exact, the data move along the cut as the
step moves it, while the part of the motion across the cut dissolves.
Every volume plays in turn.
The buttons below the animation, or the keys, change the speed (``-`` and ``+``),
pause (space), go to the previous or next volume (arrow keys), and the bar jumps to any
volume.

The animation is drawn from the default run's ``--write-hdf5`` output:

.. code-block:: python

    from nifreeze.data.dmri import DWI
    from nifreeze.viz.corrections import animate_corrections

    animate_corrections(DWI.from_filename("dwi.h5"), "corrections.svg")

Each variant is one ``GEM``, given as the pass's backend:

.. code-block:: python

    GEM()                                   # one C and one p, both estimated
    GEM(exponent=0.5)                       # one C, with p held at 1/2
    GEM(edges=(1500, 2750))                 # one C for each shell, p held at 1/2
    GEM(fit_range=(0, 1500), exponent=0.5)  # C fitted at b=1000, given to all
    GEM(zero_point=(0.0, -18.0, 12.0))      # the fields zero there (RAS+, mm)
    GEM(phase_encoding="j-")                # its own phase-encoding direction

The API reference of :class:`~nifreeze.registration.gem.GEM` lists the rest of its
options.

.. _gem-cascade:

Cascading passes
~~~~~~~~~~~~~~~~
GEM's passes can be cascaded, as many as needed: each pass is an estimator, and a chain
of them (``prev=``) runs them in turn.
Three passes, for instance:

.. code-block:: python

    first = Estimator("gqi-krr", estimates="eddy", backend=GEM())
    second = Estimator("gqi-krr", estimates="eddy", backend=GEM(), prev=first)
    Estimator("gqi-krr", estimates="eddy", backend=GEM(), prev=second).run(dwi)

Each pass's model trains on the volumes that the previous pass corrected (the default);
with ``run(dwi, instant_update=False)``, every pass repeats the first.
FSL eddy, too, trains its predictions again on the corrected volumes at each of its
iterations, five by default.

Each pass after the first is given the head motion and the field that the pass before
stored (`#758 <https://github.com/nipreps/nifreeze/pull/758>`__).
It factors that field out of each volume's mapping, as a field zero at the zero point,
and fits from the head motion that remains.
Each volume's translation along :math:`u` then comes from the new field alone, however
many passes came before.
On the phantom, with gqi-krr's predictions, each of the second and third passes
recovers 59% to 77% of what the pass before left (:ref:`gem-recovers`).

.. _gem-fit-range:

Fitting on low b-values
~~~~~~~~~~~~~~~~~~~~~~~
``fit_range=(low, high)`` fits the model on the volumes whose b-value is at least
``low`` and below ``high`` only, and gives every volume its field from it
(`#757 <https://github.com/nipreps/nifreeze/pull/757>`__).
For instance, ``GEM(fit_range=(0, 1500), exponent=0.5)`` fits :math:`C` on the
b=1000 shell.
The reference b-value is then, by default, the geometric mean of the range's b-values.
With a single b-value in the range, the exponent cannot be estimated, and is held: at
``exponent``, or at 1/2 with a warning.
A wrong exponent goes straight to the other shells: held at 1/2 when the truth is 1,
a fit on the b=1000 shell would give the b=3500 fields at 0.53 of their size.
``fit_range`` and ``edges`` exclude each other.
On the phantom, at 16 mm, :math:`C` fitted at b=1000 gives the smallest median field
error in every pass (:ref:`gem-b1000`).

The options combine: :math:`C` fitted at b=1000, in three cascaded passes, for instance:

.. code-block:: python

    chain = None
    for _ in range(3):
        chain = Estimator(
            "gqi-krr",
            estimates="eddy",
            # p held at 1/2: give your scanner's, if known (the phantom's is 1)
            backend=GEM(fit_range=(0, 1500), exponent=0.5),
            prev=chain,
        )
    chain.run(dwi)

The antecedent: FSL eddy's second-level model
---------------------------------------------
FSL's ``eddy`` :footcite:p:`andersson_integrated_2016` estimates a field for each
volume.
With ``--slm linear`` (off by default), each of the field's coefficients except its
offset is pooled over the diffusion-weighted volumes of every shell, and replaced by
its least-squares fit on :math:`(b_v / 1000)\, g_v`, without an intercept.
For the slopes, that is GEM with one tensor for all the volumes and :math:`p = 1`.
``eddy`` applies it in each iteration, after the step that updates each volume's
estimates.
FSL recommends it for data with few directions, or whose directions do not cover the
whole sphere.
GEM differs from it in four ways:

1. GEM is fitted on the images, in one registration of all the volumes, while ``eddy``
   fits its model to the volumes' own estimates, each counting the same.
2. GEM estimates the exponent, or holds it at a value the caller gives; ``eddy``
   holds it at 1.
3. GEM's zero point is given.
   By default, ``eddy`` regresses the sum of each volume's field offset and translation
   along phase encoding on the gradients, across the volumes: the fit becomes the
   offset, and the rest the translation.
   In effect, its fields' zero point is estimated.
4. GEM can fit one tensor for each shell.

.. footbibliography::
