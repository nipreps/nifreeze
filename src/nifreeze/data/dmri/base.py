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
"""DWI data representation type."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from warnings import warn

import attrs
import h5py
import nibabel as nb
import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from nifreeze.data.base import BaseDataset, _cmp, _data_repr
from nifreeze.data.dmri.utils import (
    DEFAULT_HIGHB_THRESHOLD,
    DEFAULT_LOWB_THRESHOLD,
    DEFAULT_MULTISHELL_BIN_COUNT_THR,
    DEFAULT_NUM_BINS,
    DTI_MIN_ORIENTATIONS,
    GRADIENT_EXPECTED_COLUMNS_ERROR_MSG,
    GRADIENT_VOLUME_DIMENSIONALITY_MISMATCH_ERROR,
    find_shelling_scheme,
    format_gradients,
)

BZERO_SHAPE_MISMATCH_ERROR_MSG = """\
DWI 'bzero' shape ({bzero_shape}) does not match dataset volumes ({data_shape}). \
If you have multiple b0 volumes, either provide one of them or provide a single, \
representative b0."""
"""DWI bzero shape mismatch error message."""

DWI_B0_MULTIPLE_VOLUMES_WARN_MSG = """\
The DWI data contains multiple b0 volumes; computing median across them."""
"""DWI bzero shape mismatch warning message."""

DWI_REDUNDANT_B0_WARN_MSG = """\
The DWI data contains b0 volumes, but the 'bzero' attribute was set. DWI b0 "
volumes will be discarded, and the corresponding 'dataobj' and 'gradient' data \
removed."""
"""DWI b0 and bzero provided warning message."""


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
    inst : :obj:`~nifreeze.data.dmri.base.DWI`
        The instance being validated (unused; present for validator signature).
    attr : :obj:`~attrs.Attribute`
        The attribute being validated; attr.name is used in the error message.
    value : :obj:`~npt.NDArray`
        The value to validate.

    Raises
    ------
    :exc:`ValueError`
        If the gradient table is invalid.


    Examples
    --------
    Non-row-major inputs are rejected::
        >>> validate_gradients(None, None, [[0.0, 0.0], [0.0, 1000]])
        Traceback (most recent call last):
        ...
        ValueError: Gradient table must have four columns (3 direction components and one b-value).

    Non-finite inputs are rejected::

        >>> validate_gradients(None, None, [[np.inf, 0.0, 0.0, 1000]])
        Traceback (most recent call last):
        ...
        ValueError: Gradient table contains NaN or infinite values.
        >>> validate_gradients(None, None, [[np.nan, 0.0, 0.0, 1000]])
        Traceback (most recent call last):
        ...
        ValueError: Gradient table contains NaN or infinite values.

    """
    if np.shape(value)[1] != 4:
        raise ValueError(GRADIENT_EXPECTED_COLUMNS_ERROR_MSG)

    if not np.all(np.isfinite(value)):
        raise ValueError("Gradient table contains NaN or infinite values.")


@attrs.define(slots=True, eq=False)
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
        if self.dataobj.shape[-1] != self.gradients.shape[0]:
            raise ValueError(
                GRADIENT_VOLUME_DIMENSIONALITY_MISMATCH_ERROR.format(
                    n_volumes=self.dataobj.shape[-1],
                    n_gradients=self.gradients.shape[0],
                )
            )
        # Ensure that if b0 data were provided, it is a 3D array with the same
        # shape as the DWI data object
        if self.bzero is not None:
            if self.bzero.shape != tuple(self.dataobj.shape[:3]):
                raise ValueError(
                    BZERO_SHAPE_MISMATCH_ERROR_MSG.format(
                        bzero_shape=self.bzero.shape, data_shape=self.dataobj.shape[:3]
                    )
                )

        b0_mask = self.gradients[:, -1] <= DEFAULT_LOWB_THRESHOLD
        b0_num = np.sum(b0_mask)

        bzeros = None
        # Warn the user if multiple b0 volumes were found in the DWI data and
        # no bzero attribute was provided; warn the user if both the DWI
        # contained b0 values and the bzero attribute was provided
        if b0_num > 0 and self.bzero is None:
            warn(DWI_B0_MULTIPLE_VOLUMES_WARN_MSG, UserWarning) if b0_num > 1 else None
            bzeros = self.dataobj[..., b0_mask]
            bzeros = bzeros.squeeze(axis=-1) if bzeros.shape[-1] == 1 else bzeros
        elif b0_num > 0 and self.bzero is not None:
            warn(DWI_REDUNDANT_B0_WARN_MSG, UserWarning)
            bzeros = self.bzero

        # Set the bzero attribute to the median if necessary
        if bzeros is not None:
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

    def _eq_extras(self, other: BaseDataset) -> bool:
        if not isinstance(other, DWI):
            return False

        return (
            _cmp(self.gradients, other.gradients)
            and _cmp(self.bzero, other.bzero)
            and (self.eddy_xfms == other.eddy_xfms)
        )

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
    def from_filename(cls, filename: Path | str, *, keep_file_open: bool = True) -> Self:
        """
        Read an HDF5 file from disk and create a DWI object.

        Parameters
        ----------
        filename : :obj:`os.pathlike`
            The HDF5 file path to read.

        Returns
        -------
        :obj:`~nifreeze.data.dmri.base.DWI`
            The constructed dataset with data loaded from the file.

        """
        return super().from_filename(filename, keep_file_open=keep_file_open)

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
        compression_opts : :obj:`~typing.Any`, optional
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
    ) -> nb.nifti1.Nifti1Image:
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

        Returns
        -------
        :obj:`~nibabel.nifti1.Nifti1Image`
            NIfTI image written to disk.
        """
        from nifreeze.data.dmri.io import to_nifti

        return to_nifti(
            self,
            filename=filename,
            write_hmxfms=write_hmxfms,
            order=order,
            insert_b0=insert_b0,
            bvals_dec_places=bvals_dec_places,
            bvecs_dec_places=bvecs_dec_places,
        )
