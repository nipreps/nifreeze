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
"""The interface every registration backend implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import nitransforms as nt
import numpy as np
from typing_extensions import Self


class Registration(ABC):
    """
    Estimate, volume by volume, the transform from a fixed image to a moving one.

    The fixed image is the model's prediction, on the grid of the reference; the
    moving image is the sample. The estimate maps each point of the fixed image
    into the moving one, in RAS+ world coordinates.

    A backend is a context manager: leaving the context releases what it holds
    (working files, threads), and surfaces any error raised while writing debug
    traces.

    Parameters
    ----------
    settings : :obj:`str` or :obj:`os.pathlike`, optional
        A configuration of the backend's own family, by name or path. :obj:`None`
        selects the backend's default.
    num_threads : :obj:`int`, optional
        Threads of each registration.
    seed : :obj:`int`, optional
        Seed of the backend's random sampling.
    clip : :obj:`str`, optional
        Clip the intensity of ``"sample"``, ``"predicted"``, ``"both"`` (default)
        or ``"none"`` of the images to the window given with each call.
    debug_dir : :obj:`os.pathlike`, optional
        Keep, for each volume, the images registered and an aligned trace in this
        folder.
    **overrides
        Settings that override those of the configuration.

    """

    def __init__(
        self,
        settings: str | Path | None = None,
        *,
        num_threads: int | None = None,
        seed: int | None = None,
        clip: str | bool | None = "both",
        debug_dir: str | Path | None = None,
        **overrides: Any,
    ) -> None:
        self.settings = settings
        self.num_threads = num_threads
        self.seed = seed
        self.clip = clip
        self.debug_dir = None if debug_dir is None else Path(debug_dir).absolute()
        self.overrides = overrides

    @abstractmethod
    def __call__(
        self,
        fixed: np.ndarray,
        moving: np.ndarray,
        affine: np.ndarray,
        mask: np.ndarray | None = None,
        init: np.ndarray | None = None,
        *,
        window: tuple[float, float] | None = None,
        index: int = 0,
    ) -> nt.linear.Affine:
        """
        Register one volume.

        Parameters
        ----------
        fixed : :obj:`~numpy.ndarray`
            The fixed image (the prediction), a 3D array.
        moving : :obj:`~numpy.ndarray`
            The moving image (the sample), a 3D array on the grid of ``fixed``.
        affine : :obj:`~numpy.ndarray`
            The voxel-to-RAS+ affine of that grid.
        mask : :obj:`~numpy.ndarray`, optional
            A boolean mask of the fixed image; the moving image is never masked.
        init : :obj:`~numpy.ndarray`, optional
            The initial estimate, a RAS+ 4x4 matrix from fixed to moving.
        window : :obj:`tuple` of :obj:`float`, optional
            The intensity window ``(a_min, a_max)`` that ``clip`` applies.
            :obj:`None` clips nothing.
        index : :obj:`int`, optional
            The volume's index, which names its debug files.

        Returns
        -------
        :obj:`~nitransforms.linear.Affine`
            The estimate, from fixed to moving, with the initialization included
            (not only the increment over ``init``).

        """

    def close(self) -> None:  # noqa: B027 (a backend that holds nothing keeps this no-op)
        """Release what the backend holds."""

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(settings={self.settings!r})"
