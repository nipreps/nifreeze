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
from nifreeze.data.dmri.base import DWI, select_b0
from nifreeze.data.dmri.utils import (
    GRADIENT_EXPECTED_COLUMNS_ERROR_MSG,
    GRADIENT_VOLUME_DIMENSIONALITY_MISMATCH_ERROR,
    bvecs_from_ras,
    bvecs_to_ras,
    format_gradients,
    realign_bvecs,
)
from nifreeze.data.utils import to_time_first
from nifreeze.utils.ndimage import get_data, load_api

GRADIENT_BVAL_BVEC_PRIORITY_WARN_MSG = """\
Both a gradients table file and b-vec/val files are defined; \
ignoring b-vec/val files in favor of the gradients_file."""
"""dMRI gradient file priority warning message."""

GRADIENT_DATA_MISSING_ERROR_MSG = "No gradient data provided."
"""dMRI missing gradient data error message."""

GRADIENT_BVEC_BVAL_SHAPE_ERROR_MSG = """\
The b-vector file must hold three rows (or three columns) of as many values as the \
b-value file: found b-vectors of shape {bvecs} and {bvals} b-values."""
"""FSL gradient files shape mismatch error message."""

RASB_HEADER = "\t".join("RASB")
"""The header of NiPreps' RAS+B tables (``.tsv``), as dMRIPrep writes it."""


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

    This function loads data from a NIfTI file, and its gradient table from either a
    RAS+B file or FSL's ``.bvec`` and ``.bval`` files, which are converted into RAS+B
    with the image's affine.

    Parameters
    ----------
    filename : :obj:`os.pathlike`
        The main DWI data file (NIfTI).
    brainmask_file : :obj:`os.pathlike`, optional
        A brainmask NIfTI file. If provided, will be loaded and
        stored in the returned dataset.
    gradients_file : :obj:`os.pathlike`, optional
        A RAS+B gradient table (see :func:`load_rasb`): MRtrix's ``.b``, NiPreps' ``.tsv``,
        or any text file of ``N`` rows ``R A S B``. A table with one column per volume
        (four rows) is transposed. If provided, it supersedes any .bvec / .bval
        combination.
    bvec_file : :obj:`os.pathlike`, optional
        FSL's b-vectors, shape (3, N) or (N, 3) (see :func:`load_bvec_bval`).
    bval_file : :obj:`os.pathlike`, optional
        FSL's b-values, shape (N,).
    b0_file : :obj:`os.pathlike`, optional
        A NIfTI file containing a b=0 volume (possibly averaged or reference).
        If not provided, and the data contains at least one b=0 volume, one will be computed.

    Returns
    -------
    dwi : :obj:`~nifreeze.data.dmri.base.DWI`
        A DWI object containing the loaded data, gradient table, and optional
        b-zero volume, and brainmask.

    Raises
    ------
    :exc:`RuntimeError`
        If no gradient information is provided (neither ``gradients_file`` nor
        ``bvec_file`` + ``bval_file``).

    """
    filename = Path(filename)

    # 1) Open a NIfTI; its data are read below
    img = load_api(filename, SpatialImage)

    # 2) Read the gradient table into RAS+B, from either gradients_file or bvec/bval
    if gradients_file:
        grad = load_rasb(gradients_file)
        if bvec_file and bval_file:
            warn(GRADIENT_BVAL_BVEC_PRIORITY_WARN_MSG, stacklevel=2)
    elif bvec_file and bval_file:
        grad = load_bvec_bval(bvec_file, bval_file, img.affine)
    else:
        raise RuntimeError(GRADIENT_DATA_MISSING_ERROR_MSG)

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

    # 5) NIfTI stores volumes last; datasets store them first. Leave the b=0
    # volumes out while converting, so the series is copied once: ``DWI`` would
    # copy it again to drop them.
    data = get_data(img)
    gradients = format_gradients(grad)  # oriented and normalized as ``DWI`` does
    assert gradients is not None  # format_gradients raises on None
    if data.shape[-1] != gradients.shape[0]:
        raise ValueError(
            GRADIENT_VOLUME_DIMENSIONALITY_MISMATCH_ERROR.format(
                n_volumes=data.shape[-1], n_gradients=gradients.shape[0]
            )
        )
    b0_mask, b0_data = select_b0(gradients, b0_data, lambda mask: to_time_first(data, keep=mask))
    fulldata = to_time_first(data, keep=~b0_mask)
    img.uncache()  # drop nibabel's cached copy of the volumes-last array

    # 6) Create and return the DWI instance.
    return DWI(
        dataobj=fulldata,
        affine=img.affine,
        gradients=gradients[~b0_mask],
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
) -> nb.nifti1.Nifti1Image:
    """
    Export the dMRI object to disk (NIfTI data, and gradients).

    The gradient table is written next to the NIfTI file, matching its volumes, twice:
    in FSL's convention for the output's affine (``.bvec`` and ``.bval``, see
    :func:`save_bvec_bval`), and in RAS+B (NiPreps' ``.tsv``, see :func:`save_rasb`).
    If the DWI object has non-null
    :attr:`~nifreeze.data.base.BaseDataset.motion` (i.e., following the
    motion estimation), the b-vectors are turned into the reference's frame
    (see :func:`~nifreeze.data.dmri.utils.realign_bvecs`).

    Parameters
    ----------
    dwi : :obj:`~nifreeze.data.dmri.base.DWI`
        A DWI object containing the diffusion data to be written, including the
        gradient table, and optional b-zero volume.
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

    Returns
    -------
    nii : :obj:`~nibabel.nifti1.Nifti1Image`
        The main DWI data object.
    """

    no_bzero = dwi.bzero is None or not insert_b0

    bvals = dwi.bvals
    bvecs = dwi.bvecs

    # Turn the b-vectors into the reference's frame if dwi.motion is not None
    if dwi.motion is not None:
        rotated = [
            realign_bvecs(bvec, matrix) for bvec, matrix in zip(dwi.bvecs, dwi.motion, strict=True)
        ]
        bvecs = np.asarray(rotated)

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
        # Prepend the b=0 to the volumes written, which are resampled if motion was estimated
        volumes = np.asanyarray(nii.dataobj)
        data = np.concatenate((dwi.bzero[..., np.newaxis], volumes), axis=-1)  # type: ignore
        nii = nb.Nifti1Image(data, nii.affine, nii.header)
        # And its null gradient to their gradients, rotated if motion was estimated
        bvals = np.concatenate((np.zeros(1), bvals))
        bvecs = np.vstack((np.zeros((1, 3)), bvecs))

        if filename is not None:
            nii.to_filename(filename)

    if filename is not None:
        # Convert filename to a Path object.
        out_root = Path(filename).absolute()

        # Get the base stem for writing the gradient tables.
        out_root = out_root.parent / out_root.name.replace("".join(out_root.suffixes), "")

        gradients = np.column_stack((bvecs, bvals))
        save_bvec_bval(
            gradients,
            nii.affine,
            out_root.with_suffix(".bvec"),
            out_root.with_suffix(".bval"),
            bvals_dec_places=bvals_dec_places,
            bvecs_dec_places=bvecs_dec_places,
        )
        save_rasb(
            gradients,
            out_root.with_suffix(".tsv"),
            bvals_dec_places=bvals_dec_places,
            bvecs_dec_places=bvecs_dec_places,
        )

    return nii


def load_bvec_bval(bvec_file: Path | str, bval_file: Path | str, affine: np.ndarray) -> np.ndarray:
    """
    Read an FSL gradient table (``.bvec`` and ``.bval``) into world RAS+B.

    Parameters
    ----------
    bvec_file : :obj:`os.pathlike`
        The b-vectors, three rows of ``N`` values (three columns are accepted too).
    bval_file : :obj:`os.pathlike`
        The ``N`` b-values.
    affine : :obj:`~numpy.ndarray`
        The affine of the image the table belongs to.

    Returns
    -------
    :obj:`~numpy.ndarray`
        The gradient table, shape ``(N, 4)``: the b-vectors in world RAS+ coordinates
        (see :func:`~nifreeze.data.dmri.utils.bvecs_to_ras`), then the b-values.

    """
    bvecs = np.loadtxt(bvec_file, ndmin=2)
    if bvecs.shape[0] != 3 and bvecs.shape[1] == 3:
        bvecs = bvecs.T
    bvals = np.loadtxt(bval_file, ndmin=1).ravel()
    if bvecs.shape != (3, bvals.size):
        raise ValueError(
            GRADIENT_BVEC_BVAL_SHAPE_ERROR_MSG.format(bvecs=bvecs.shape, bvals=bvals.size)
        )
    return np.column_stack((bvecs_to_ras(bvecs.T, affine, "fsl"), bvals))


def save_bvec_bval(
    gradients: np.ndarray,
    affine: np.ndarray,
    bvec_file: Path | str,
    bval_file: Path | str,
    bvals_dec_places: int = 2,
    bvecs_dec_places: int = 6,
) -> None:
    """
    Write a RAS+B gradient table as an FSL one (``.bvec`` and ``.bval``).

    Parameters
    ----------
    gradients : :obj:`~numpy.ndarray`
        The gradient table, shape ``(N, 4)``, in world RAS+B.
    affine : :obj:`~numpy.ndarray`
        The affine of the image the table is written for.
    bvec_file : :obj:`os.pathlike`
        The output b-vectors, three rows of ``N`` values.
    bval_file : :obj:`os.pathlike`
        The output b-values, one row of ``N`` values.
    bvals_dec_places : :obj:`int`, optional
        Decimal places to use when serializing b-values.
    bvecs_dec_places : :obj:`int`, optional
        Decimal places to use when serializing b-vectors.

    """
    bvecs = bvecs_from_ras(gradients[:, :3], affine, "fsl")
    np.savetxt(bvec_file, bvecs.T, fmt=f"%.{bvecs_dec_places}f")
    np.savetxt(bval_file, gradients[np.newaxis, :, 3], fmt=f"%.{bvals_dec_places}f")


def load_rasb(filename: Path | str) -> np.ndarray:
    """
    Read a RAS+B gradient table: MRtrix's ``.b``, or NiPreps' ``.tsv``.

    Each row holds a b-vector in world RAS+ coordinates and its b-value. Lines starting
    with ``#`` are comments, and a first line ``R A S B`` is a header. A table written with
    one column per volume (four rows) is transposed.

    Parameters
    ----------
    filename : :obj:`os.pathlike`
        The gradient table file.

    Returns
    -------
    :obj:`~numpy.ndarray`
        The gradient table, shape ``(N, 4)``.

    """
    with open(filename) as fobj:
        lines = [line.strip() for line in fobj]
    rows = [line.split() for line in lines if line and not line.startswith("#")]
    if rows and [token.upper() for token in rows[0]] == list("RASB"):
        rows = rows[1:]  # a header written without the comment mark
    table = np.array(rows, dtype=float)
    if table.ndim == 2 and table.shape[0] == 4 and table.shape[1] != 4:
        table = table.T
    if table.ndim != 2 or table.shape[1] != 4:
        raise ValueError(GRADIENT_EXPECTED_COLUMNS_ERROR_MSG)
    return table


def save_rasb(
    gradients: np.ndarray,
    filename: Path | str,
    bvals_dec_places: int = 2,
    bvecs_dec_places: int = 6,
) -> None:
    """
    Write a RAS+B gradient table.

    A ``.tsv`` file is tab-separated with the header of :data:`RASB_HEADER` (NiPreps);
    any other file is space-separated without a header (MRtrix's ``.b``).

    Parameters
    ----------
    gradients : :obj:`~numpy.ndarray`
        The gradient table, shape ``(N, 4)``, in world RAS+B.
    filename : :obj:`os.pathlike`
        The output file.
    bvals_dec_places : :obj:`int`, optional
        Decimal places to use when serializing b-values.
    bvecs_dec_places : :obj:`int`, optional
        Decimal places to use when serializing b-vectors.

    """
    fmt = [f"%.{bvecs_dec_places}f"] * 3 + [f"%.{bvals_dec_places}f"]
    if Path(filename).suffix == ".tsv":
        np.savetxt(filename, gradients, fmt=fmt, delimiter="\t", header=RASB_HEADER)
    else:
        np.savetxt(filename, gradients, fmt=fmt)
