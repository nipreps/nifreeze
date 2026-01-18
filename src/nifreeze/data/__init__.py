# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
"""Four-dimensional data representation in hard-disk and memory."""

from pathlib import Path
from typing import cast

from nifreeze.data.base import NFDH5_EXT, BaseDataset, _ArrayLike
from nifreeze.data.dmri import DWI
from nifreeze.data.pet import PET


def load(
    filename: Path | str,
    brainmask_file: Path | str | None = None,
    *,
    keep_file_open: bool = False,
    **kwargs,
) -> BaseDataset | DWI | PET:
    """
    Load 4D data from a filename or an HDF5 file.

    Parameters
    ----------
    filename : :obj:`os.pathlike`
        The NIfTI or HDF5 file.
    brainmask_file : :obj:`os.pathlike`, optional
        A brainmask NIfTI file. If provided, will be loaded and
        stored in the returned dataset.

    Returns
    -------
    :obj:`~nifreeze.data.base.BaseDataset`
        The loaded dataset.

    Raises
    ------
    :exc:`ValueError`
        If the file extension is not supported or the file cannot be loaded.

    """

    from contextlib import suppress

    import numpy as np
    from nibabel.spatialimages import SpatialImage

    from nifreeze.utils.ndimage import load_api

    filename = Path(filename)
    if filename.name.endswith(NFDH5_EXT):
        for dataclass in (BaseDataset, PET, DWI):
            with suppress(TypeError):
                return dataclass.from_filename(filename, keep_file_open=keep_file_open)

        raise TypeError("Could not read data")

    if {"gradients_file", "bvec_file"} & set(kwargs):
        from nifreeze.data.dmri import from_nii as dmri_from_nii

        return dmri_from_nii(filename, brainmask_file=brainmask_file, **kwargs)
    elif {"temporal_file"} & set(kwargs):
        from nifreeze.data.pet import from_nii as pet_from_nii

        return pet_from_nii(filename, brainmask_file=brainmask_file, **kwargs)

    img = load_api(filename, SpatialImage)
    retval: BaseDataset = BaseDataset(
        dataobj=cast(_ArrayLike, img.dataobj), affine=np.asanyarray(img.affine)
    )

    if brainmask_file:
        mask = load_api(brainmask_file, SpatialImage)
        retval.brainmask = np.asanyarray(mask.dataobj, dtype=bool)
    else:
        retval.brainmask = np.ones(img.shape[:3], dtype=bool)

    return retval
