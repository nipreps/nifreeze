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
"""Input/Output utilities for DWI objects."""

from pathlib import Path
from warnings import warn

import nibabel as nb
import numpy as np
from nibabel.spatialimages import SpatialImage

from nifreeze.data.base import to_nifti as _base_to_nifti
from nifreeze.data.dmri.base import DWI
from nifreeze.data.dmri.utils import transform_fsl_bvec
from nifreeze.utils.ndimage import get_data, load_api

GRADIENT_BVAL_BVEC_PRIORITY_WARN_MSG = """\
Both a gradients table file and b-vec/val files are defined; \
ignoring b-vec/val files in favor of the gradients_file."""
""""dMRI gradient file priority warning message."""

GRADIENT_DATA_MISSING_ERROR = "No gradient data provided."
"""dMRI missing gradient data error message."""


def from_nii(
    filename: Path | str,
    brainmask_file: Path | str | None = None,
    gradients_file: Path | str | None = None,
    bvec_file: Path | str | None = None,
    bval_file: Path | str | None = None,
    b0_file: Path | str | None = None,
) -> DWI:
    """
    Load DWI data from NIfTI and construct a DWI object.

    This function loads data from a NIfTI file, optionally loading a gradient table
    from either a separate gradients file or from .bvec / .bval files.

    Parameters
    ----------
    filename : :obj:`os.pathlike`
        The main DWI data file (NIfTI).
    brainmask_file : :obj:`os.pathlike`, optional
        A brainmask NIfTI file. If provided, will be loaded and
        stored in the returned dataset.
    gradients_file : :obj:`os.pathlike`, optional
        A text file containing the gradients table, shape (N, C) where the last column
        stores the b-values. If provided following the column-major convention(C, N),
        it will be transposed automatically. If provided, it supersedes any .bvec / .bval
        combination.
    bvec_file : :obj:`os.pathlike`, optional
        A text file containing b-vectors, shape (N, 3) or (3, N).
    bval_file : :obj:`os.pathlike`, optional
        A text file containing b-values, shape (N,).
    b0_file : :obj:`os.pathlike`, optional
        A NIfTI file containing a b=0 volume (possibly averaged or reference).
        If not provided, and the data contains at least one b=0 volume, one will be computed.

    Returns
    -------
    dwi : :obj:`~nifreeze.data.dmri.DWI`
        A DWI object containing the loaded data, gradient table, and optional
        b-zero volume, and brainmask.

    Raises
    ------
    :exc:`RuntimeError`
        If no gradient information is provided (neither ``gradients_file`` nor
        ``bvec_file`` + ``bval_file``).

    """
    filename = Path(filename)

    # 1) Load a NIfTI
    img = load_api(filename, SpatialImage)
    fulldata = get_data(img)

    # 2) Determine the gradients array from either gradients_file or bvec/bval
    if gradients_file:
        grad = np.loadtxt(gradients_file, dtype="float32")
        if bvec_file and bval_file:
            warn(GRADIENT_BVAL_BVEC_PRIORITY_WARN_MSG, stacklevel=2)
    elif bvec_file and bval_file:
        bvecs = np.loadtxt(bvec_file, dtype="float32")
        if bvecs.shape[1] != 3 and bvecs.shape[0] == 3:
            bvecs = bvecs.T

        bvals = np.loadtxt(bval_file, dtype="float32")
        grad = np.column_stack((bvecs, bvals))
    else:
        raise RuntimeError(GRADIENT_DATA_MISSING_ERROR)

    # 3) Read b-zero volume if provided
    b0_data = None
    if b0_file:
        b0img = load_api(b0_file, SpatialImage)
        b0_data = np.asanyarray(b0img.dataobj)

    # 4) If a brainmask_file was provided, load it
    brainmask_data = None
    if brainmask_file:
        mask_img = load_api(brainmask_file, SpatialImage)
        brainmask_data = np.asanyarray(mask_img.dataobj, dtype=bool)

    # 5) Create and return the DWI instance.
    return DWI(
        dataobj=fulldata,
        affine=img.affine,
        gradients=grad,
        bzero=b0_data,
        brainmask=brainmask_data,
    )


def to_nifti(
    dwi,
    filename: Path | str | None = None,
    write_hmxfms: bool = False,
    order: int = 3,
    insert_b0: bool = False,
    bvals_dec_places: int = 2,
    bvecs_dec_places: int = 6,
) -> nb.Nifti1Image:
    """
    Export the dMRI object to disk (NIfTI, b-vecs, & b-vals files).

    Parameters
    ----------
    filename : :obj:`os.pathlike`, optional
        The output NIfTI file path.
    write_hmxfms : :obj:`bool`, optional
        If ``True``, the head motion affines will be written out to filesystem
        with BIDS' X5 format.
    order : :obj:`int`, optional
        The interpolation order to use when resampling the data.
        Defaults to 3 (cubic interpolation).
    insert_b0 : :obj:`bool`, optional
        Insert a :math:`b=0` at the front of the output NIfTI and add the corresponding
        null gradient value to the output bval/bvec files.
    bvals_dec_places : :obj:`int`, optional
        Decimal places to use when serializing b-values.
    bvecs_dec_places : :obj:`int`, optional
        Decimal places to use when serializing b-vectors.

    """

    no_bzero = dwi.bzero is None or not insert_b0
    bvals = dwi.bvals

    # Rotate b-vectors if dwi.motion_affines is not None
    if dwi.motion_affines is not None:
        rotated = [
            transform_fsl_bvec(bvec, affine, dwi.affine, invert=True)
            for bvec, affine in zip(dwi.gradients[:, :3], dwi.motion_affines, strict=True)
        ]
        bvecs = np.asarray(rotated)
    else:
        bvecs = dwi.bvecs

    # Parent's to_nifti to handle the primary NIfTI export.
    nii = _base_to_nifti(
        dwi,
        filename=filename if no_bzero else None,
        write_hmxfms=write_hmxfms,
        order=order,
    )

    if no_bzero:
        if insert_b0:
            warn(
                "Ignoring ``insert_b0`` argument as the data object's bzero field is unset",
                stacklevel=2,
            )
    else:
        data = np.concatenate((dwi.bzero[..., np.newaxis], dwi.dataobj), axis=-1)  # type: ignore
        nii = nb.Nifti1Image(data, nii.affine, nii.header)

        if filename is not None:
            nii.to_filename(filename)

        # If inserting a b0 volume is requested, add the corresponding null
        # gradient value to the bval/bvec pair
        bvals = np.concatenate((np.zeros(1), bvals))
        bvecs = np.vstack((np.zeros((1, bvecs.shape[1])), bvecs))

    if filename is not None:
        # Convert filename to a Path object.
        out_root = Path(filename).absolute()

        # Get the base stem for writing .bvec / .bval.
        out_root = out_root.parent / out_root.name.replace("".join(out_root.suffixes), "")

        # Construct sidecar file paths.
        bvecs_file = out_root.with_suffix(".bvec")
        bvals_file = out_root.with_suffix(".bval")

        # Save bvecs and bvals to text files. BIDS expects 3 rows x N columns.
        np.savetxt(bvecs_file, bvecs.T, fmt=f"%.{bvecs_dec_places}f")
        np.savetxt(bvals_file, bvals[np.newaxis, :], fmt=f"%.{bvals_dec_places}f")

    return nii
