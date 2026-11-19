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
"""The interface of every registration backend, and the factory that builds them by name."""

from __future__ import annotations

from abc import ABC, abstractmethod
from importlib import import_module
from io import BytesIO
from pathlib import Path
from typing import Any
from warnings import warn

import nibabel as nb
import nitransforms as nt
import numpy as np

from nifreeze.data.filtering import ClippingValueError, robust_window

WINDOW_LEVELS = 4095
"""Highest code of the 12-bit encoding of floating-point registration inputs."""
CLIPPING = ("sample", "predicted", "both")
"""The values of ``clip`` that clip an image to the window."""
BACKENDS = {
    "ants": "nifreeze.registration.ants:Ants",
    "elastix": "nifreeze.registration.elastix:Elastix",
}
"""The backends by name, as ``module:class`` (imported when first built)."""


class RegistrationFactory:
    """Build a registration backend by name, as models are built by name."""

    @staticmethod
    def init(name: str = "ants", **kwargs: Any) -> Registration:
        """
        Build the backend ``name`` (of :data:`BACKENDS`, or any ``"module:class"``).

        >>> type(RegistrationFactory.init("nifreeze.registration.ants:Ants")).__name__
        'Ants'

        """
        module, _, cls = BACKENDS.get(name.lower(), name).partition(":")
        if not cls:
            raise ValueError(f"Unknown registration backend {name!r} (of {sorted(BACKENDS)}).")
        return getattr(import_module(module), cls)(**kwargs)


class Registration(ABC):
    """
    Estimate the transform that maps each point of a fixed image into a moving one.

    The fixed image is the model's prediction; the moving image is the sample, on the
    same grid. Calls run one at a time (elastix sets ITK's process-wide thread count).

    Parameters
    ----------
    settings : :obj:`str`, :obj:`os.pathlike` or :obj:`dict`, optional
        A configuration of the backend's own family, by name or path (:obj:`None`
        selects the backend's default).
    seed : :obj:`int`, optional
        Seed of the backend's random sampling.
    num_threads : :obj:`int`, optional
        Threads of each registration.
    clip : :obj:`str`, optional
        Clip ``"sample"``, ``"predicted"``, ``"both"`` (default) or ``"none"`` of the
        images to the sample's intensity window, estimated within the mask.
    debug_dir : :obj:`os.pathlike`, optional
        Keep ``predicted_<index>.nii`` and ``sample_<index>.nii``, the values
        registered, and ``aligned_<index>.nii``, the sample with its affine corrected
        by the estimate (a viewer shows it on top of the prediction).
    **overrides
        Settings that override those of the configuration.

    """

    def __init__(
        self,
        settings: Any = None,
        *,
        seed: int | None = None,
        num_threads: int | None = None,
        clip: str | bool | None = "both",
        debug_dir: str | Path | None = None,
        **overrides: Any,
    ) -> None:
        self.settings = settings
        self.seed = seed
        self.num_threads = num_threads
        clip = str(clip).lower()
        self.clip = {"true": "both", "false": "none"}.get(clip, clip)
        # Absolute, because nipype runs ANTs from within its working directory
        self.debug_dir = None if debug_dir is None else Path(debug_dir).absolute()
        if self.debug_dir is not None:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.overrides = overrides

    def __call__(
        self,
        fixed: np.ndarray,
        moving: np.ndarray,
        affine: np.ndarray,
        mask: np.ndarray | None = None,
        index: int = 0,
        init: np.ndarray | None = None,
    ) -> nt.linear.Affine:
        """
        Register one volume.

        Parameters
        ----------
        fixed, moving : :obj:`~numpy.ndarray`
            The fixed image (the prediction) and the moving one (the sample).
        affine : :obj:`~numpy.ndarray`
            The voxel-to-RAS+ affine of their grid.
        mask : :obj:`~numpy.ndarray`, optional
            The mask of the fixed image (the moving one is never masked).
        index : :obj:`int`, optional
            The volume's index, which names its files.
        init : :obj:`~numpy.ndarray`, optional
            The initial estimate, a RAS+ fixed-to-moving matrix.

        Returns
        -------
        :obj:`~nitransforms.linear.Affine`
            The fixed-to-moving transform (RAS+), the initialization included.

        """
        window = None
        if self.clip in CLIPPING:
            try:
                window = robust_window(moving, mask)
            except ClippingValueError as exc:
                warn(f"Registering volume {index} without clipping: {exc}", stacklevel=2)
        matrix = self._register(fixed, moving, affine, mask, index, init, window)
        if self.debug_dir is not None:
            _write_aligned(
                self.debug_dir / f"sample_{index:05d}.nii",
                matrix,
                self.debug_dir / f"aligned_{index:05d}.nii",
            )
        return nt.linear.Affine(matrix)

    @abstractmethod
    def _register(
        self,
        fixed: np.ndarray,
        moving: np.ndarray,
        affine: np.ndarray,
        mask: np.ndarray | None,
        index: int,
        init: np.ndarray | None,
        window: tuple[float, float] | None,
    ) -> np.ndarray:
        """Register, with both images clipped to ``window`` (as ``clip`` selects)."""


def _write_aligned(moving_path: str | Path, matrix: np.ndarray, out_path: str | Path) -> None:
    """
    Write the moving image aligned to the fixed image, without resampling.

    The voxel data are copied byte for byte, and the affine is premultiplied by
    the inverse of the estimated transform, so that a viewer displays the image
    on top of the fixed one.

    Parameters
    ----------
    moving_path : :obj:`os.pathlike`
        The moving image, an uncompressed NIfTI-1 file.
    matrix : :obj:`~numpy.ndarray`
        The estimated transform (fixed-to-moving, in world coordinates).
    out_path : :obj:`os.pathlike`
        The output (uncompressed NIfTI-1) file path.

    """
    raw = Path(moving_path).read_bytes()
    # Parse the header from the file itself: a loaded image's header has its scaling reset
    header = nb.Nifti1Header.from_fileobj(BytesIO(raw))
    aligned = np.linalg.inv(matrix) @ header.get_best_affine()
    header.set_sform(aligned, code=1)
    header.set_qform(aligned, code=1)
    Path(out_path).write_bytes(header.binaryblock + raw[header.sizeof_hdr :])
