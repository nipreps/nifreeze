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

from collections import namedtuple
from pathlib import Path
from typing import Any
from warnings import warn

import attr
import h5py
import nibabel as nb
import numpy as np
import numpy.typing as npt
from nibabel.spatialimages import SpatialImage
from nitransforms.linear import Affine

from nifreeze.data.base import BaseDataset, _cmp, _data_repr
from nifreeze.utils.ndimage import load_api

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


@attr.s(slots=True)
class DWI(BaseDataset[np.ndarray | None]):
    """Data representation structure for dMRI data."""

    bzero = attr.ib(default=None, repr=_data_repr, eq=attr.cmp_using(eq=_cmp))
    """A *b=0* reference map, preferably obtained by some smart averaging."""
    gradients = attr.ib(default=None, repr=_data_repr, eq=attr.cmp_using(eq=_cmp))
    """A 2D numpy array of the gradient table (4xN)."""
    eddy_xfms = attr.ib(default=None)
    """List of transforms to correct for estimated eddy current distortions."""

    def _getextra(self, idx: int | slice | tuple | np.ndarray) -> tuple[np.ndarray | None]:
        return (self.gradients[..., idx] if self.gradients is not None else None,)

    # For the sake of the docstring
    def __getitem__(
        self, idx: int | slice | tuple | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
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
            The corresponding gradient(s), which may have shape ``(4,)`` if a single volume
            or ``(4, k)`` if multiple volumes, or ``None`` if gradients are not available.

        """

        return super().__getitem__(idx)

    @classmethod
    def from_filename(cls, filename: Path | str) -> DWI:
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
        # Reuse the parent `from_filename` logic to load all attributes
        # that do not start with '_'. Then simply return DWI(**loaded_data).
        from attr import fields

        data: dict[str, Any] = {}
        with h5py.File(filename, "r") as in_file:
            root = in_file["/0"]
            for f in fields(cls):
                if f.name.startswith("_"):
                    continue
                if f.name in root:
                    data[f.name] = np.asanyarray(root[f.name])
                else:
                    data[f.name] = None

        return cls(**data)

    @property
    def bvals(self):
        return self.gradients[-1, ...]

    @property
    def bvecs(self):
        return self.gradients[:-1, ...]

    def set_transform(self, index: int, affine: np.ndarray, order: int = 3) -> None:
        """
        Set an affine transform for a particular index and update the data object.

        The new affine is set as in :obj:`~nifreeze.data.base.BaseDataset.set_transform`,
        and, in addition, the corresponding gradient vector is rotated.

        Parameters
        ----------
        index : :obj:`int`
            The volume index to transform.
        affine : :obj:`numpy.ndarray`
            The 4x4 affine matrix to be applied.
        order : :obj:`int`, optional
            The order of the spline interpolation.

        """
        if not Path(self._filepath).exists():
            self.to_filename(self._filepath)

        ImageGrid = namedtuple("ImageGrid", ("shape", "affine"))
        reference = ImageGrid(shape=self.dataobj.shape[:3], affine=self.affine)

        xform = Affine(matrix=affine, reference=reference)
        bvec = self.bvecs[:, index]

        # invert transform transform b-vector and origin
        r_bvec = (~xform).map([bvec, (0.0, 0.0, 0.0)])
        # Reset b-vector's origin
        new_bvec = r_bvec[1] - r_bvec[0]
        # Normalize and update
        self.bvecs[:, index] = new_bvec / np.linalg.norm(new_bvec)

        super().set_transform(index, affine, order)

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
        filename: Path | str,
        insert_b0: bool = False,
        bvals_dec_places: int = 2,
        bvecs_dec_places: int = 6,
    ) -> None:
        """
        Write a NIfTI file to disk.

        Parameters
        ----------
        filename : :obj:`os.pathlike`
            The output NIfTI file path.
        insert_b0 : :obj:`bool`, optional
            Insert a :math:`b=0` at the front of the output NIfTI.
        bvals_dec_places : :obj:`int`, optional
            Decimal places to use when serializing b-values.
        bvecs_dec_places : :obj:`int`, optional
            Decimal places to use when serializing b-vectors.

        """
        if not insert_b0:
            # Parent's to_nifti to handle the primary NIfTI export.
            super().to_nifti(filename)
        else:
            data = np.concatenate((self.bzero[..., np.newaxis], self.dataobj), axis=-1)
            nii = nb.Nifti1Image(data, self.affine, self.datahdr)
            if self.datahdr is None:
                nii.header.set_xyzt_units("mm")
            nii.to_filename(filename)

        # Convert filename to a Path object.
        out_root = Path(filename).absolute()

        # Get the base stem for writing .bvec / .bval.
        out_root = out_root.parent / out_root.name.replace("".join(out_root.suffixes), "")

        # Construct sidecar file paths.
        bvecs_file = out_root.with_suffix(".bvec")
        bvals_file = out_root.with_suffix(".bval")

        # Save bvecs and bvals to text files
        # Each row of bvecs is one direction (3 rows, N columns).
        np.savetxt(bvecs_file, self.bvecs, fmt=f"%.{bvecs_dec_places}f")
        np.savetxt(bvals_file, self.bvals[np.newaxis, :], fmt=f"%.{bvals_dec_places}f")


def load(
    filename: Path | str,
    brainmask_file: Path | str | None = None,
    motion_file: Path | str | None = None,
    gradients_file: Path | str | None = None,
    bvec_file: Path | str | None = None,
    bval_file: Path | str | None = None,
    b0_file: Path | str | None = None,
    b0_thres: float = DEFAULT_LOWB_THRESHOLD,
) -> DWI:
    """
    Load DWI data and construct a DWI object.

    This function can load data from either an HDF5 file (if the filename ends with ``.h5``)
    or from a NIfTI file, optionally loading a gradient table from either a separate gradients
    file or from .bvec / .bval files.

    Parameters
    ----------
    filename : :obj:`os.pathlike`
        The main DWI data file (NIfTI or HDF5).
    brainmask_file : :obj:`os.pathlike`, optional
        A brainmask NIfTI file. If provided, will be loaded and
        stored in the returned dataset.
    motion_file : :obj:`os.pathlike`
        A file containing head-motion affine matrices (linear)
    gradients_file : :obj:`os.pathlike`, optional
        A text file containing the gradients table, shape (4, N) or (N, 4).
        If provided, it supersedes any .bvec / .bval combination.
    bvec_file : :obj:`os.pathlike`, optional
        A text file containing b-vectors, shape (3, N).
    bval_file : :obj:`os.pathlike`, optional
        A text file containing b-values, shape (N,).
    b0_file : :obj:`os.pathlike`, optional
        A NIfTI file containing a b=0 volume (possibly averaged or reference).
        If not provided, and the data contains at least one b=0 volume, one will be computed.
    b0_thres : float, optional
        Threshold for determining which volumes are considered DWI vs. b=0
        if you combine them in the same file.

    Returns
    -------
    dwi : :obj:`~nifreeze.data.dmri.DWI`
        A DWI object containing the loaded data, gradient table, and optional
        b-zero volume, and brainmask.

    Raises
    ------
    RuntimeError
        If no gradient information is provided (neither ``gradients_file`` nor
        ``bvec_file`` + ``bval_file``).

    """

    if motion_file:
        raise NotImplementedError

    filename = Path(filename)
    # 1) If this is an HDF5 file, just load via the DWI.from_filename method
    if filename.suffix == ".h5":
        return DWI.from_filename(filename)

    # 2) Otherwise, load a NIfTI
    img = load_api(filename, SpatialImage)
    fulldata = img.get_fdata(dtype=np.float32)
    affine = img.affine

    # 3) Determine the gradients array from either gradients_file or bvec/bval
    if gradients_file:
        grad = np.loadtxt(gradients_file, dtype="float32")
        if bvec_file and bval_file:
            warn(
                "Both a gradients table file and b-vec/val files are defined; "
                "ignoring b-vec/val files in favor of the gradients_file.",
                stacklevel=2,
            )
    elif bvec_file and bval_file:
        bvecs = np.loadtxt(bvec_file, dtype="float32")  # shape (3, N)
        if bvecs.shape[0] != 3 and bvecs.shape[1] == 3:
            bvecs = bvecs.T

        bvals = np.loadtxt(bval_file, dtype="float32")  # shape (N,)
        # Stack to shape (4, N)
        grad = np.vstack((bvecs, bvals))
    else:
        raise RuntimeError(
            "No gradient data provided. "
            "Please specify either a gradients_file or (bvec_file & bval_file)."
        )

    # 4) Create the DWI instance. We'll filter out volumes where b-value > b0_thres
    #    as "DW volumes" if the user wants to store only the high-b volumes here
    gradmsk = grad[-1] > b0_thres if grad.shape[0] == 4 else grad[:, -1] > b0_thres

    # The shape checking is somewhat flexible: (4, N) or (N, 4)
    dwi_obj = DWI(
        dataobj=fulldata[..., gradmsk],
        affine=affine,
        # We'll assign the filtered gradients below.
    )

    dwi_obj.gradients = grad[:, gradmsk] if grad.shape[0] == 4 else grad[gradmsk, :].T

    # 6) b=0 volume (bzero)
    #    If the user provided a b0_file, load it
    if b0_file:
        b0img = load_api(b0_file, SpatialImage)
        b0vol = np.asanyarray(b0img.dataobj)
        # We'll assume your DWI class has a bzero: np.ndarray | None attribute
        dwi_obj.bzero = b0vol
    # Otherwise, if any volumes remain outside gradmsk, compute a median B0:
    elif np.any(~gradmsk):
        # The b=0 volumes are those that did NOT pass b0_thres
        b0_volumes = fulldata[..., ~gradmsk]
        # A simple approach is to take the median across that last dimension
        # Note that axis=3 is valid only if your data is 4D (x, y, z, volumes).
        dwi_obj.bzero = np.median(b0_volumes, axis=3)

    # 7) If a brainmask_file was provided, load it
    if brainmask_file:
        mask_img = load_api(brainmask_file, SpatialImage)
        dwi_obj.brainmask = np.asanyarray(mask_img.dataobj, dtype=bool)

    return dwi_obj


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


def _rasb2dipy(gradient):
    import warnings

    gradient = np.asanyarray(gradient)
    if gradient.ndim == 1:
        if gradient.size != 4:
            raise ValueError("Missing gradient information.")
        gradient = gradient[..., np.newaxis]

    if gradient.shape[0] != 4:
        gradient = gradient.T
    elif gradient.shape == (4, 4):
        print("Warning: make sure gradient information is not transposed!")

    with warnings.catch_warnings():
        from dipy.core.gradients import gradient_table

        warnings.filterwarnings("ignore", category=UserWarning)
        retval = gradient_table(gradient[3, :], bvecs=gradient[:3, :].T)
    return retval
