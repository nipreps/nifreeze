Upcoming release
================

.. warning::

   **Volumes are now stored on the first axis** (#461).
   ``dataset.dataobj`` has shape ``(N, X, Y, Z)``, and ``dataset[i]`` returns ``dataobj[i]``.
   Code that builds datasets directly (``BaseDataset``, ``DWI`` or ``PET`` with ``dataobj=``)
   must pass the volumes first, for example ``np.moveaxis(data, -1, 0)``.
   ``from_nii`` and ``to_nifti`` convert to and from the NIfTI order (volumes last).
   HDF5 files are now written as *nfHDF5* version 2 (``Format="nfHDF5"``, for PET too)
   and memory-mapped when loaded.
   Version 1 files (``Format="NFDH5"`` or ``"EMC/PET"``) still load, after a conversion
   into a temporary file; save them again with ``to_filename`` to load them lazily.
   ``nifreeze.data.NFDH5_EXT`` is renamed ``NFHDF5_EXT``.

*eddymotion* fork
=================
The *NiFreeze* codebase has been ported from *eddymotion* on commit
`68f199d <https://github.com/nipreps/eddymotion/commit/68f199dabc0aff8e904ad9f832ed276122c0845c>`_.
