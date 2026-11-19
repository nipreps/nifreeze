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
"""
Accuracy and time of registration on a phantom with known motion.

Each moved volume of a perturbed copy is registered to three fixed images, and the
estimated transform is compared with the truth from the copy's sidecar: the head's
motion, from the b=0's frame to the volume's (fixed to moving, as the estimator
reports it).

``oracle``
    The noise-free reference volume of the same gradient direction, at the true
    position: the error of registration alone.
``unmoved``
    The model's held-out prediction from the reference: registration on a model's
    target that motion has not corrupted.
``moved``
    The model's held-out prediction from the copy itself, as in a run before any
    motion is corrected.

The three together separate the error of registration from that of the model. This
analysis is about registration; the sensitivity analysis
(:mod:`nifreeze.analysis.sensitivity`) involves none.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from time import perf_counter
from typing import Any

import nibabel as nb
import nitransforms as nt
import numpy as np

from nifreeze.analysis.sensitivity import held_out_predictions
from nifreeze.data.dmri import DWI
from nifreeze.registration.ants import Ants
from nifreeze.registration.utils import displacements_within_mask

FIXED = ("oracle", "unmoved", "moved")
"""The fixed images each moved volume is registered to."""

Register = Callable[
    [np.ndarray, np.ndarray, np.ndarray, np.ndarray, int], nt.linear.Affine | np.ndarray
]
"""``register(fixed, moving, affine, mask, index)``: the fixed-to-moving transform (RAS).

Any :class:`~nifreeze.registration.base.Registration`, or a function returning a matrix.
"""


class AntsRegistration(Ants):
    """The estimator's ANTs path, in ``workdir`` and with seed 12345 by default."""

    def __init__(
        self, workdir: Path | str, seed: int | None = 12345, ants_config: Any = None, **kwargs
    ):
        super().__init__(ants_config, seed=seed, workdir=workdir, **kwargs)


def registration_accuracy(
    register: Register,
    clean: DWI,
    reference: DWI,
    moved: DWI,
    truth: dict[int, np.ndarray],
    model: str | Callable[..., Any] | None = None,
    fixed: Iterable[str] = FIXED,
    **model_kwargs: Any,
) -> list[dict]:
    """
    Register each moved volume to each fixed image and measure the error.

    Parameters
    ----------
    register : :obj:`~collections.abc.Callable`
        The registration (see :data:`Register`).
    clean : :obj:`~nifreeze.data.dmri.DWI`
        The noise-free reference, the source of the ``oracle`` targets.
    reference : :obj:`~nifreeze.data.dmri.DWI`
        The reference, with the noise of ``moved``.
    moved : :obj:`~nifreeze.data.dmri.DWI`
        The perturbed copy.
    truth : :obj:`dict`
        The true fixed-to-moving matrix of each volume registered, by index.
    model : :obj:`str` or :obj:`~collections.abc.Callable`, optional
        The model of the ``unmoved`` and ``moved`` targets
        (see :func:`~nifreeze.analysis.sensitivity.held_out_predictions`); required
        unless ``fixed`` is only ``oracle``.
    fixed : :obj:`~collections.abc.Iterable`, optional
        The fixed images (of :data:`FIXED`).
    **model_kwargs
        Arguments for the model.

    Returns
    -------
    :obj:`list` of :obj:`dict`
        One row per volume and fixed image, with the volume's ``index`` and
        ``shell``, the ``fixed`` image, the ``max`` and ``mean`` displacement (mm)
        within the brain between the estimated and true transforms, and the
        ``seconds`` the registration took.

    """
    fixed = list(fixed)
    if unknown := set(fixed) - set(FIXED):
        raise ValueError(f"unknown fixed images: {sorted(unknown)}")
    if model is None and set(fixed) - {"oracle"}:
        raise ValueError("a model is required for the unmoved and moved targets")

    indices = sorted(truth)
    mask = np.asanyarray(reference.brainmask, dtype=bool)
    mask_img = nb.Nifti1Image(mask.astype("uint8"), reference.affine)

    def predictions(dataset: DWI) -> dict[int, np.ndarray]:
        assert model is not None
        return held_out_predictions(model, dataset, indices, **model_kwargs)

    targets = {
        "oracle": lambda: {i: np.asarray(clean.dataobj[i], dtype="float32") for i in indices},
        "unmoved": lambda: predictions(reference),
        "moved": lambda: predictions(moved),
    }

    rows = []
    for kind in fixed:
        for index, target in targets[kind]().items():
            start = perf_counter()
            estimated = register(
                np.asarray(target, dtype="float32"),
                np.asarray(moved.dataobj[index]),
                reference.affine,
                mask,
                index,
            )
            seconds = perf_counter() - start
            matrix = getattr(estimated, "matrix", estimated)
            errors = displacements_within_mask(
                mask_img, nt.linear.Affine(matrix), nt.linear.Affine(truth[index])
            )
            rows.append(
                {
                    "index": index,
                    "shell": float(np.round(reference.gradients[index, -1], -2)),
                    "fixed": kind,
                    "max": float(errors.max()),
                    "mean": float(errors.mean()),
                    "seconds": seconds,
                }
            )
    return rows
