.. include:: ../links.rst

Data and file formats
=====================

The gradient table
------------------
dMRI gradients are held as a table of one row per volume, ``[R A S b]``: a unit
b-vector in world RAS+ coordinates (the frame the image's affine maps voxels
into), then the b-value in units of :math:`s/mm^2`.
Conversions happen only where a table enters or leaves:
:func:`~nifreeze.data.dmri.io.from_nii` converts FSL's ``.bvec`` and ``.bval``
files with the image's affine, and reads RAS+B tables (MRtrix's ``.b``,
NiPreps' ``.tsv``) as they are;
:meth:`~nifreeze.data.dmri.DWI.to_nifti` writes both conventions.
When a volume moves by :math:`M`, its b-vector in the realigned view is
:math:`R^{-1} \mathbf{g}`, with :math:`R` the rotation of :math:`M`
(:func:`~nifreeze.data.dmri.utils.realign_bvecs`).
The dataset's own table keeps the b-vectors as acquired: the eddy currents follow the
gradient the scanner plays, so GEM fits its fields from them (:ref:`gem-bvectors`).

.. _nfhdf5:

nfHDF5 files
------------
*nfHDF5* is the HDF5 layout *NiFreeze* saves datasets in.
:meth:`~nifreeze.data.base.BaseDataset.to_filename` writes it, and
:func:`~nifreeze.data.load` and
:meth:`~nifreeze.data.base.BaseDataset.from_filename` read it back.

- The file attributes ``Format`` (``"nfHDF5"``) and ``Version`` (``4``)
  identify the layout.
- The group ``/0`` holds one HDF5 dataset per field of the dataset object:
  ``dataobj``, ``affine``, ``brainmask`` and ``motion``, plus the
  modality's own fields, such as ``gradients``, ``bzero`` and ``eddy`` (dMRI) or
  ``midframe`` and ``total_duration`` (PET).
- ``motion`` (and ``eddy``, dMRI) store the estimates, ``(N, 4, 4)``: for each
  volume, a RAS+ matrix that maps the reference into it.
- ``dataobj`` stores the volumes on the first axis, ``(N, X, Y, Z)``, and has
  the attribute ``volume_axis`` set to ``0``.
  Each volume is one contiguous block, so the file is memory-mapped when
  loaded and volumes are read from disk as they are accessed.
  Compressed files (``to_filename(..., compression="gzip")``) are
  decompressed into a temporary file first.
- ``gradients`` (dMRI) holds the table in RAS+B, as
  :class:`~nifreeze.data.dmri.DWI` does.
- ``phase_encoding`` (dMRI, when known) holds the phase-encoding direction as BIDS
  writes ``PhaseEncodingDirection`` (``i``, ``j`` or ``k``, with an optional ``-``).
  Files without it load with the direction unknown.

Files written by earlier versions of *NiFreeze* (version 1) store the volumes
last, and their ``Format`` is ``"NFDH5"`` (``"EMC/PET"`` for PET).
They still load, after a conversion into a temporary file;
save them again with ``to_filename`` to skip that step.

Files of versions 1 to 3 name the head-motion estimates ``motion_affines``,
which loads as ``motion``.

Files of versions 1 and 2 do not record the frame of their gradient table.
:meth:`~nifreeze.data.dmri.DWI.from_filename` reads it as RAS+B, and warns.
A table loaded from FSL's ``.bvec`` files into one of these files is on the
voxel axes instead: load it again from its source.
