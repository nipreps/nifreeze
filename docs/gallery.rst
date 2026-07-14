.. include:: links.rst

.. _gallery:

===========================
Prediction gallery
===========================
This gallery shows *predicted* diffusion volumes produced by *NiFreeze*'s models
on several real datasets spanning the acquisition-scheme spectrum — a simple,
legacy DTI dataset, a single-shell HARDI dataset, a multi-shell dataset, and a
DSI dataset. For each dataset, every applicable model is run in **two modes**:

- **LOVO** (leave-one-volume-out): the model is fit on every *other* volume and
  used to predict the held-out orientation, so the prediction is unbiased with
  respect to the target volume.
- **single-fit**: the model is fit **once** on all volumes and then predicts.

The whole gallery is driven by a single declarative matrix
(:mod:`nifreeze._gallery`) of *(dataset × model × mode)* cells. A model's
**capability contract** (:class:`~nifreeze.model.base.BaseModel`) decides, before
fitting, which cells apply — so each page also reports a **coverage table**
recording what was exercised and why any cell was skipped (e.g. DKI needs
multiple shells; the shell-averaging model has no single-fit mode). This makes
the gallery a living record of what the models are validated to do on real data.

.. note::

   The datasets are fetched from OpenNeuro. Two labels are worth stating plainly:
   ds000114 is single-shell at ``b=1000`` s/mm² (high angular resolution, milder
   than textbook high-b HARDI), and ds004737 is *compressed-sensing* DSI (a
   sub-sampled q-space grid, not a full 258-point acquisition).

.. toctree::
   :maxdepth: 1

   notebooks/gallery/dti
   notebooks/gallery/hardi
   notebooks/gallery/multishell
   notebooks/gallery/dsi
