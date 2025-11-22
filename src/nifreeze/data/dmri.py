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
"""dMRI data representation."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from warnings import warn

import attrs
import h5py
import nibabel as nb
import numpy as np
import numpy.typing as npt
from nibabel.spatialimages import SpatialImage
from typing_extensions import Self

from nifreeze.data.base import BaseDataset, _cmp, _data_repr
from nifreeze.utils.ndimage import get_data, load_api

DEFAULT_CLIP_PERCENTILE = 75
"""Upper percentile threshold for intensity clipping."""

DEFAULT_MIN_S0 = 1e-5
"""Minimum value when considering the :math:`S_{0}` DWI signal."""

DEFAULT_MAX_S0 = 1.0
"""Maximum value when considering the :math:`S_{0}` DWI signal."""

DEFAULT_LOWB_THRESHOLD = 50
"""The lower bound for the b-value so that the orientation is considered a DW volume."""

DEFAULT_HIGHB_THRESHOLD = 8000
"""A b-value cap for DWI data."""

DEFAULT_NUM_BINS = 15
"""Number of bins to classify b-values."""

DEFAULT_MULTISHELL_BIN_COUNT_THR = 7
"""Default bin count to consider a multishell scheme."""

DTI_MIN_ORIENTATIONS = 6
"""Minimum number of nonzero b-values in a DWI dataset."""

GRADIENT_ABSENCE_ERROR_MSG = "Gradient table may not be None."
"""Gradient absence error message."""

GRADIENT_OBJECT_ERROR_MSG = "Gradient table must be a numeric homogeneous array-like object"
"""Gradient object error message."""

GRADIENT_VOLUME_DIMENSIONALITY_MISMATCH_ERROR = """\
Gradient table shape does not match the number of diffusion volumes: \
expected {n_volumes} rows, found {n_gradients}."""
"""dMRI volume count vs. gradient count mismatch error message."""

GRADIENT_BVAL_BVEC_PRIORITY_WARN_MSG = """\
Both a gradients table file and b-vec/val files are defined; \
ignoring b-vec/val files in favor of the gradients_file."""
""""dMRI gradient file priority warning message."""

GRADIENT_NDIM_ERROR_MSG = "Gradient table must be a 2D array"
"""dMRI gradient dimensionality error message."""

GRADIENT_DATA_MISSING_ERROR = "No gradient data provided."
"""dMRI missing gradient data error message."""

GRADIENT_EXPECTED_COLUMNS_ERROR_MSG = (
    "Gradient table must have four columns (3 direction components and one b-value)."
)
"""dMRI gradient expected columns error message."""


def format_gradients(
    value: npt.ArrayLike | None,
) -> np.ndarray | None:
    """
    Validate and orient gradient tables to row-major convention.

    Parameters
    ----------
    value : :obj:`ArrayLike`
        The value to format.

    Returns
    -------
    :obj:`~numpy.ndarray`
        Row-major convention gradient table.

    Raises
    ------
    exc:`ValueError`
        If ``value`` is not a 2D :obj:`~numpy.ndarray` (``value.ndim != 2``).

    Examples
    --------
    Passing an already well-formed table returns the data unchanged::

        >>> np.set_printoptions(formatter={"float_kind": lambda x: f"{int(x):4d}"})
        >>> format_gradients(
        ...     [
        ...         [1, 0, 0, 0],
        ...         [0, 1, 0, 1000],
        ...         [0, 0, 1, 2000],
        ...         [0, 0, 0, 0],
        ...         [0, 0, 0, 1000],
        ...     ]
        ... )
        array([[   1,    0,    0,    0],
               [   0,    1,    0, 1000],
               [   0,    0,    1, 2000],
               [   0,    0,    0,    0],
               [   0,    0,    0, 1000]])

    Column-major inputs are automatically transposed when an expected
    number of diffusion volumes is provided::

        >>> format_gradients(
        ...     [[1, 0], [0, 1], [0, 0], [1000, 2000]],
        ... )
        array([[   1,    0,    0, 1000],
               [   0,    1,    0, 2000]])

    Gradient tables must always have two dimensions::

        >>> format_gradients([0, 1, 0, 1000])
        Traceback (most recent call last):
        ...
        ValueError: Gradient table must be a 2D array

    Gradient tables must have a regular shape::

        >>> format_gradients([[1, 2], [3, 4, 5]])
        Traceback (most recent call last):
        ...
        TypeError: Gradient table must be a numeric homogeneous array-like object

    Gradient tables must always have two dimensions::

        >>> format_gradients([0, 1, 0, 1000])
        Traceback (most recent call last):
        ...
        ValueError: Gradient table must be a 2D array

        >>> np.set_printoptions() # reset to defaults
    """

    if value is None:
        raise ValueError(GRADIENT_ABSENCE_ERROR_MSG)

    try:
        formatted = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        # Conversion failed (e.g. nested ragged objects, non-numeric)
        raise TypeError(GRADIENT_OBJECT_ERROR_MSG) from exc

    if formatted.ndim != 2:
        raise ValueError(GRADIENT_NDIM_ERROR_MSG)

    # Transpose if column-major
    return formatted.T if formatted.shape[0] == 4 and formatted.shape[1] != 4 else formatted


def validate_gradients(
    inst: DWI,
    attr: attrs.Attribute,
    value: npt.NDArray[np.floating],
) -> None:
    """Strict validator for use in attribute validation (e.g. attrs / validators).

    Ensures row-major convention for gradient table.

    This function is intended for use as an attrs-style validator.

    Parameters
    ----------
    inst : :obj:`~nifreeze.data.dmri.DWI`
        The instance being validated (unused; present for validator signature).
    attr : :obj:`~attrs.Attribute`
        The attribute being validated; attr.name is used in the error message.
    value : :obj:`~npt.NDArray`
        The value to validate.
    """
    if value.shape[1] != 4:
        raise ValueError(GRADIENT_EXPECTED_COLUMNS_ERROR_MSG)


@attrs.define(slots=True)
class DWI(BaseDataset[np.ndarray]):
    """Data representation structure for dMRI data."""

    gradients: np.ndarray = attrs.field(
        default=None,
        repr=_data_repr,
        eq=attrs.cmp_using(eq=_cmp),
        converter=format_gradients,
        validator=validate_gradients,
    )
    """A 2D numpy array of the gradient table (``N`` orientations x ``C`` components)."""
    bzero: np.ndarray | None = attrs.field(
        default=None, repr=_data_repr, eq=attrs.cmp_using(eq=_cmp)
    )
    """A *b=0* reference map, computed automatically when low-b frames are present."""
    eddy_xfms: list = attrs.field(default=None)
    """List of transforms to correct for estimated eddy current distortions."""

    def __attrs_post_init__(self) -> None:
        if self.gradients is None:
            raise ValueError(GRADIENT_DATA_MISSING_ERROR)

        if self.dataobj.shape[-1] != self.gradients.shape[0]:
            raise ValueError(
                GRADIENT_VOLUME_DIMENSIONALITY_MISMATCH_ERROR.format(
                    n_volumes=self.dataobj.shape[-1],
                    n_gradients=self.gradients.shape[0],
                )
            )

        b0_mask = self.gradients[:, -1] <= DEFAULT_LOWB_THRESHOLD
        b0_num = np.sum(b0_mask)

        if b0_num > 0 and self.bzero is None:
            bzeros = self.dataobj[..., b0_mask]
            self.bzero = bzeros if bzeros.ndim == 3 else np.median(bzeros, axis=-1)

        if b0_num > 0:
            # Remove b0 volumes from dataobj and gradients
            self.gradients = self.gradients[~b0_mask, :]
            self.dataobj = self.dataobj[..., ~b0_mask]

        if self.gradients.shape[0] < DTI_MIN_ORIENTATIONS:
            raise ValueError(
                f"DWI datasets must have at least {DTI_MIN_ORIENTATIONS} diffusion-weighted "
                f"orientations; found {self.dataobj.shape[-1]}."
            )

    def _getextra(self, idx: int | slice | tuple | np.ndarray) -> tuple[np.ndarray]:
        return (self.gradients[idx, ...],)

    # For the sake of the docstring
    def __getitem__(
        self, idx: int | slice | tuple | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        """
        Returns volume(s) and corresponding affine(s) and gradient(s) through fancy indexing.

        Parameters
        ----------
        idx : :obj:`int` or :obj:`slice` or :obj:`tuple` or :obj:`~numpy.ndarray`
            Indexer for the last dimension (or possibly other dimensions if extended).

        Returns
        -------
        volumes : :obj:`~numpy.ndarray`
            The selected data subset.
            If ``idx`` is a single integer, this will have shape ``(X, Y, Z)``,
            otherwise it may have shape ``(X, Y, Z, k)``.
        motion_affine : :obj:`~numpy.ndarray` or ``None``
            The corresponding per-volume motion affine(s) or ``None`` if identity transform(s).
        gradient : :obj:`~numpy.ndarray`
            The corresponding gradient(s), which may have shape ``(C,)`` if a single volume
            or ``(k, C)`` if multiple volumes, or ``None`` if gradients are not available.

        """

        return super().__getitem__(idx)

    @classmethod
    def from_filename(cls, filename: Path | str) -> Self:
        """
        Read an HDF5 file from disk and create a DWI object.

        Parameters
        ----------
        filename : :obj:`os.pathlike`
            The HDF5 file path to read.

        Returns
        -------
        :obj:`~nifreeze.data.dmri.DWI`
            The constructed dataset with data loaded from the file.

        """
        return super().from_filename(filename)

    @property
    def bvals(self):
        return self.gradients[:, -1]

    @property
    def bvecs(self):
        return self.gradients[:, :-1]

    def get_shells(
        self,
        num_bins: int = DEFAULT_NUM_BINS,
        multishell_nonempty_bin_count_thr: int = DEFAULT_MULTISHELL_BIN_COUNT_THR,
        bval_cap: int = DEFAULT_HIGHB_THRESHOLD,
    ) -> list:
        """Get the shell data according to the b-value groups.

        Bin the shell data according to the b-value groups found by
        :obj:`~nifreeze.data.dmri.find_shelling_scheme`.

        Parameters
        ----------
        num_bins : :obj:`int`, optional
            Number of bins.
        multishell_nonempty_bin_count_thr : :obj:`int`, optional
            Bin count to consider a multi-shell scheme.
        bval_cap : :obj:`int`, optional
            Maximum b-value to be considered in a multi-shell scheme.

        Returns
        -------
        :obj:`list`
            Tuples of binned b-values and corresponding data/gradients indices.

        """

        _, bval_groups, bval_estimated = find_shelling_scheme(
            self.bvals,
            num_bins=num_bins,
            multishell_nonempty_bin_count_thr=multishell_nonempty_bin_count_thr,
            bval_cap=bval_cap,
        )
        indices = [np.where(np.isin(self.bvals, bvals))[0] for bvals in bval_groups]
        return list(zip(bval_estimated, indices, strict=True))

    def to_filename(
        self,
        filename: Path | str,
        compression: str | None = None,
        compression_opts: Any = None,
    ) -> None:
        """
        Write the dMRI dataset to an HDF5 file on disk.

        Parameters
        ----------
        filename : :obj:`os.pathlike`
            The HDF5 file path to write to.
        compression : :obj:`str`, optional
            Compression strategy.
            See :obj:`~h5py.Group.create_dataset` documentation.
        compression_opts : :obj:`typing.Any`, optional
            Parameters for compression
            `filters <https://docs.h5py.org/en/stable/high/dataset.html#dataset-compression>`__.

        """
        super().to_filename(filename, compression=compression, compression_opts=compression_opts)
        # Overriding if you'd like to set a custom attribute, for example:
        with h5py.File(filename, "r+") as out_file:
            out_file.attrs["Type"] = "dmri"

    def to_nifti(
        self,
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

        no_bzero = self.bzero is None or not insert_b0
        bvals = self.bvals

        # Rotate b-vectors if self.motion_affines is not None
        if self.motion_affines is not None:
            rotated = [
                transform_fsl_bvec(bvec, affine, self.affine, invert=True)
                for bvec, affine in zip(self.gradients[:, :3], self.motion_affines, strict=True)
            ]
            bvecs = np.asarray(rotated)
        else:
            bvecs = self.bvecs

        # Parent's to_nifti to handle the primary NIfTI export.
        nii = super().to_nifti(
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
            data = np.concatenate((self.bzero[..., np.newaxis], self.dataobj), axis=-1)  # type: ignore
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
        if bvals.ndim > 1:
            bvals = np.squeeze(bvals)
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


def find_shelling_scheme(
    bvals: np.ndarray,
    num_bins: int = DEFAULT_NUM_BINS,
    multishell_nonempty_bin_count_thr: int = DEFAULT_MULTISHELL_BIN_COUNT_THR,
    bval_cap: float = DEFAULT_HIGHB_THRESHOLD,
) -> tuple[str, list[npt.NDArray[np.floating]], list[np.floating]]:
    """
    Find the shelling scheme on the given b-values.

    Computes the histogram of the b-values according to ``num_bins``
    and depending on the nonempty bin count, classify the shelling scheme
    as single-shell if they are 2 (low-b and a shell); multi-shell if they are
    below the ``multishell_nonempty_bin_count_thr`` value; and DSI otherwise.

    Parameters
    ----------
    bvals : :obj:`list` or :obj:`~numpy.ndarray`
         List or array of b-values.
    num_bins : :obj:`int`, optional
        Number of bins.
    multishell_nonempty_bin_count_thr : :obj:`int`, optional
        Bin count to consider a multi-shell scheme.
    bval_cap : :obj:`float`, optional
        Maximum b-value to be considered in a multi-shell scheme.

    Returns
    -------
    scheme : :obj:`str`
        Shelling scheme.
    bval_groups : :obj:`list`
        List of grouped b-values.
    bval_estimated : :obj:`list`
        List of 'estimated' b-values as the median value of each b-value group.

    """

    # Bin the b-values: use -1 as the lower bound to be able to appropriately
    # include b0 values
    hist, bin_edges = np.histogram(bvals, bins=num_bins, range=(-1, min(max(bvals), bval_cap)))

    # Collect values in each bin
    bval_groups = []
    bval_estimated = []
    for lower, upper in zip(bin_edges[:-1], bin_edges[1:], strict=False):
        # Add only if a nonempty b-values mask
        if (mask := (bvals > lower) & (bvals <= upper)).sum():
            bval_groups.append(bvals[mask])
            bval_estimated.append(np.median(bvals[mask]))

    nonempty_bins = len(bval_groups)

    if nonempty_bins < 2:
        raise ValueError("DWI must have at least one high-b shell")

    if nonempty_bins == 2:
        scheme = "single-shell"
    elif nonempty_bins < multishell_nonempty_bin_count_thr:
        scheme = "multi-shell"
    else:
        scheme = "DSI"

    return scheme, bval_groups, bval_estimated


def transform_fsl_bvec(
    b_ijk: np.ndarray, xfm: np.ndarray, imaffine: np.ndarray, invert: bool = False
) -> np.ndarray:
    """
    Transform a b-vector from the original space to the new space defined by the affine.

    Parameters
    ----------
    b_ijk : :obj:`~numpy.ndarray`
        The b-vector in FSL/DIPY conventions (i.e., voxel coordinates).
    xfm : :obj:`~numpy.ndarray`
        The affine transformation to apply.
        Please note that this is the inverse of the head-motion-correction affine,
        which maps coordinates from the realigned space to the moved (scan) space.
        In this case, we want to move the b-vector from the moved (scan) space into
        the realigned space.
    imaffine : :obj:`~numpy.ndarray`
        The image's affine, to convert.
    invert : :obj:`bool`, optional
        If ``True``, the transformation will be inverted.

    Returns
    -------
    :obj:`~numpy.ndarray`
        The transformed b-vector in voxel coordinates (FSL/DIPY).

    """
    xfm = np.linalg.inv(xfm) if invert else xfm.copy()

    # Go from world coordinates (xfm) to voxel coordinates
    ijk2ijk_xfm = np.linalg.inv(imaffine) @ xfm @ imaffine

    return ijk2ijk_xfm[:3, :3] @ b_ijk[:3]
