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
"""Representing dMRI data."""

from __future__ import annotations

from collections import namedtuple
from pathlib import Path
from typing import Any
from warnings import warn

import attr
import h5py
import nibabel as nb
import numpy as np
from nitransforms.linear import Affine

from nifreeze.data.base import BaseDataset, _cmp, _data_repr


@attr.s(slots=True)
class DWI(BaseDataset):
    """Data representation structure for dMRI data."""

    bzero = attr.ib(default=None, repr=_data_repr, eq=attr.cmp_using(eq=_cmp))
    """
    A *b=0* reference map, preferably obtained by some smart averaging.
    If the :math:`B_0` fieldmap is set, this *b=0* reference map should also
    be unwarped.
    """
    gradients = attr.ib(default=None, repr=_data_repr, eq=attr.cmp_using(eq=_cmp))
    """A 2D numpy array of the gradient table in RAS+B format (Nx4)."""
    eddy_xfms = attr.ib(default=None)
    """List of transforms to correct for estimatted eddy current distortions."""

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
        volumes : np.ndarray
            The selected data subset. If `idx` is a single integer, this will have shape
            ``(X, Y, Z)``, otherwise it may have shape ``(X, Y, Z, k)``.
        motion_affine : np.ndarray or None
            The corresponding per-volume motion affine(s) or `None` if identity transform(s).
        gradient : np.ndarray
            The corresponding gradient(s), which may have shape ``(4,)`` if a single volume
            or ``(k, 4)`` if multiple volumes, or None if gradients are not available.

        """

        data, affine = super().__getitem__(idx)
        return data, affine, self.gradients[idx, ...]

    def set_transform(self, index: int, affine: np.ndarray, order: int = 3) -> None:
        """
        Set an affine transform for a particular DWI volume, resample that volume,
        and reorient the corresponding gradient vector.

        Parameters
        ----------
        index : int
            The volume index to transform (0-based).
        affine : np.ndarray
            A 4x4 affine matrix to be applied to the DWI volume.
        order : int, optional
            Order of the spline interpolation (0-5). Default is 3.

        Raises
        ------
        ValueError
            If ``gradients`` is None or doesn't match the data shape.
        """
        # Basic validation
        if self.gradients is None:
            raise ValueError("Cannot set a transform on DWI data without a gradient table.")

        # If the gradient table is Nx4, and dataobj has shape (..., N)
        n_volumes = self.dataobj.shape[-1]
        if self.gradients.shape[0] != n_volumes and self.gradients.shape[1] == n_volumes:
            # Possibly transposed gradient table - handle or raise an error
            raise ValueError("Gradient table shape does not match the data's last dimension.")

        reference = namedtuple("ImageGrid", ("shape", "affine"))(
            shape=self.dataobj.shape[:3], affine=self.affine
        )
        xform = Affine(matrix=affine, reference=reference)

        if not Path(self._filepath).exists():
            self.to_filename(self._filepath)

        # read original DWI data & b-vector
        with h5py.File(self._filepath, "r") as in_file:
            root = in_file["/0"]
            dwi_frame = np.asanyarray(root["dataobj"][..., index])
            bvec = np.asanyarray(root["gradients"][index, :3])

        dwmoving = nb.Nifti1Image(dwi_frame, self.affine, None)

        # resample and update orientation at index
        self.dataobj[..., index] = np.asanyarray(
            xform.apply(dwmoving, order=order).dataobj,
            dtype=self.dataobj.dtype,
        )

        # invert transform transform b-vector and origin
        r_bvec = (~xform).map([bvec, (0.0, 0.0, 0.0)])
        # Reset b-vector's origin
        new_bvec = r_bvec[1] - r_bvec[0]
        # Normalize and update
        self.gradients[index, :3] = new_bvec / np.linalg.norm(new_bvec)

        # update transform
        if self.em_affines is None:
            self.em_affines = np.zeros((self.dataobj.shape[-1], 4, 4))

        self.em_affines[index] = xform.matrix

    @classmethod
    def from_filename(cls, filename: Path | str) -> DWI:
        """
        Read an HDF5 file from disk and create a DWI object.

        Parameters
        ----------
        filename : Path or str
            The HDF5 file path to read.

        Returns
        -------
        DWI
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
        filename : Path or str
            Path to the output HDF5 file.
        compression : str, optional
            Compression filter, e.g. 'gzip'. Default is None (no compression).
        compression_opts : Any, optional
            Compression level or other parameters for the HDF5 dataset.
        """
        super().to_filename(filename, compression=compression, compression_opts=compression_opts)
        # Overriding if you'd like to set a custom attribute, for example:
        with h5py.File(filename, "r+") as out_file:
            out_file.attrs["Type"] = "dmri"

    def to_nifti(self, filename: Path | str) -> None:
        """
        Write a NIfTI 1.0 file to disk, and also write out the gradient table
        to sidecar text files (.bvec, .bval).

        Parameters
        ----------
        filename : Path or str
            The output NIfTI file path.

        """
        # First call the parent's to_nifti to handle the primary NIfTI export.
        super().to_nifti(filename)

        # Convert filename to a Path object.
        out_root = Path(filename).absolute()

        # Remove .gz if present, then remove .nii if present.
        # This yields the base stem for writing .bvec / .bval.
        if out_root.suffix == ".gz":
            out_root = out_root.with_suffix("")  # remove '.gz'
        if out_root.suffix == ".nii":
            out_root = out_root.with_suffix("")  # remove '.nii'

        # Construct sidecar file paths.
        bvecs_file = out_root.with_suffix(".bvec")
        bvals_file = out_root.with_suffix(".bval")

        # Save bvecs and bvals to text files
        # Each row of bvecs is one direction (3 rows, N columns).
        np.savetxt(bvecs_file, self.gradients[..., :3].T, fmt="%.6f")
        np.savetxt(bvals_file, self.gradients[..., -1], fmt="%.6f")


def load(
    filename: str | Path,
    gradients_file: str | Path | None = None,
    b0_file: str | Path | None = None,
    brainmask_file: str | Path | None = None,
    bvec_file: str | Path | None = None,
    bval_file: str | Path | None = None,
    b0_thres: float = 50.0,
) -> DWI:
    """
    Load DWI data and construct a DWI object.

    This function can load data from either an HDF5 file (if the filename ends with ``.h5``)
    or from a NIfTI file, optionally loading a gradient table from either a separate gradients
    file or from .bvec / .bval files.

    Parameters
    ----------
    filename : str or Path
        The main DWI data file (NIfTI or HDF5).
    gradients_file : str or Path, optional
        A text file containing the gradients table, shape (4, N) or (N, 4).
        If provided, it supersedes any .bvec / .bval combination.
    b0_file : str or Path, optional
        A NIfTI file containing a b=0 volume (possibly averaged or reference).
        If not provided, and the data contains at least one b=0 volume, one will be computed.
    brainmask_file : str or Path, optional
        A NIfTI file containing a brain mask. If provided, it will be loaded into
        the resulting DWI object.
    bvec_file : str or Path, optional
        A text file containing b-vectors, shape (3, N).
    bval_file : str or Path, optional
        A text file containing b-values, shape (N,).
    b0_thres : float, optional
        Threshold for determining which volumes are considered DWI vs. b=0
        if you combine them in the same file. Default is 50.0.

    Returns
    -------
    dwi : DWI
        A DWI object containing the loaded data, gradient table, and optional
        b-zero volume, brainmask, or fieldmap.

    Raises
    ------
    RuntimeError
        If no gradient information is provided (neither ``gradients_file`` nor
        ``bvec_file`` + ``bval_file``).

    """
    filename = Path(filename)
    # 1) If this is an HDF5 file, just load via the DWI.from_filename method
    if filename.suffix == ".h5":
        return DWI.from_filename(filename)

    # 2) Otherwise, load a NIfTI
    img = nb.load(str(filename))
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
        bvals = np.loadtxt(bval_file, dtype="float32")  # shape (N,)
        # Stack to shape (4, N)
        grad = np.vstack((bvecs, bvals)).T
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

    dwi_obj.gradients = grad[:, gradmsk] if grad.shape[0] == 4 else grad[gradmsk, :]

    # 6) b=0 volume (bzero)
    #    If the user provided a b0_file, load it
    if b0_file:
        b0img = nb.load(str(b0_file))
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
        mask_img = nb.load(str(brainmask_file))
        dwi_obj.brainmask = np.asanyarray(mask_img.dataobj, dtype=bool)

    return dwi_obj
