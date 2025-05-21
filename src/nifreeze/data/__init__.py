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

from nifreeze.data.base import NFDH5_EXT, BaseDataset
from nifreeze.data.dmri import DWI
from nifreeze.data.pet import PET


def load(
    filename: Path | str,
    brainmask_file: Path | str | None = None,
    motion_file: Path | str | None = None,
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
    motion_file : :obj:`os.pathlike`
        A file containing head motion affine matrices (linear).

    Returns
    -------
    :obj:`~nifreeze.data.base.BaseDataset`
        The loaded dataset.

    Raises
    ------
    ValueError
        If the file extension is not supported or the file cannot be loaded.

    """

    from contextlib import suppress

    import numpy as np
    from nibabel.spatialimages import SpatialImage

    from nifreeze.utils.ndimage import load_api

    if motion_file:
        raise NotImplementedError

    filename = Path(filename)
    if filename.name.endswith(NFDH5_EXT):
        for dataclass in (BaseDataset, PET, DWI):
            with suppress(TypeError):
                return dataclass.from_filename(filename)

        raise TypeError("Could not read data")

    if "gradients_file" in kwargs or "bvec_file" in kwargs:
        from nifreeze.data.dmri import from_nii as dmri_from_nii

        return dmri_from_nii(
            filename, brainmask_file=brainmask_file, motion_file=motion_file, **kwargs
        )
    elif "frame_time" in kwargs or "frame_duration" in kwargs:
        from nifreeze.data.pet import from_nii as pet_from_nii

        return pet_from_nii(
            filename, brainmask_file=brainmask_file, motion_file=motion_file, **kwargs
        )

    img = load_api(filename, SpatialImage)
    retval: BaseDataset = BaseDataset(dataobj=np.asanyarray(img.dataobj), affine=img.affine)

    if brainmask_file:
        mask = load_api(brainmask_file, SpatialImage)
        retval.brainmask = np.asanyarray(mask.dataobj)
    else:
        retval.brainmask = np.ones(img.shape[:3], dtype=bool)

    return retval
