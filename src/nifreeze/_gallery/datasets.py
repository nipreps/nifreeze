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
"""Dataset descriptors for the prediction gallery.

Each :class:`DatasetSpec` names an acquisition scheme and a ``loader`` that
returns a fully-constructed :class:`~nifreeze.data.dmri.DWI` (with ``b=0``
already extracted). Real OpenNeuro loaders (ds000206, ds000114, ds003138,
ds004737) are wired in a later phase; this module provides the descriptor type,
the scheme vocabulary, a scheme-verification helper, and a synthetic builder
used for fast, network-free testing.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from nifreeze.data.dmri import DWI
from nifreeze.data.dmri.utils import find_shelling_scheme

SINGLE_SHELL = "single-shell"
MULTI_SHELL = "multi-shell"
DSI = "DSI"
SCHEMES = (SINGLE_SHELL, MULTI_SHELL, DSI)
"""Acquisition schemes, matching :func:`~nifreeze.data.dmri.utils.find_shelling_scheme`."""


def default_lovo_indices(dwi: DWI, count: int = 3) -> list[int]:
    """Pick a few, well-spread held-out volume indices for display."""
    n = len(dwi)
    if n <= count:
        return list(range(n))
    # Evenly spaced interior indices (avoid the very first/last volume).
    return [int(round(k)) for k in np.linspace(0, n - 1, count + 2)[1:-1]]


@dataclass(frozen=True)
class DatasetSpec:
    """A gallery dataset: how to load it and what scheme it is."""

    name: str
    """Short identifier (e.g. ``"ds000206"``)."""
    title: str
    """Human-readable label for the gallery."""
    scheme: str
    """Declared acquisition scheme; verified against the data at load time."""
    loader: Callable[[], DWI]
    """Callable returning a constructed :class:`~nifreeze.data.dmri.DWI`."""
    lovo_indices: Callable[[DWI], list[int]] = default_lovo_indices
    """Callable choosing which held-out indices to render."""
    notes: str = ""
    """Free-form caveats surfaced on the gallery page (e.g. ``b=1000`` HARDI)."""

    def load(self) -> DWI:
        """Load the dataset and assert its scheme matches the declaration."""
        dwi = self.loader()
        verify_scheme(dwi, self.scheme, name=self.name)
        return dwi


def verify_scheme(dwi: DWI, expected: str, *, name: str = "dataset") -> str:
    """Classify ``dwi`` and raise if it does not match ``expected``.

    Guards the registry against silent drift between a dataset's declared scheme
    and its actual b-values (critical because model applicability keys on it).
    """
    if expected not in SCHEMES:
        raise ValueError(f"Unknown scheme {expected!r}; expected one of {SCHEMES}.")
    # ``DWI`` strips b=0 volumes into ``bzero`` at construction, but the shelling
    # classifier needs the low-b bin to distinguish single-shell (b0 + one shell)
    # from a lone high-b shell. Restore a single b=0 for classification.
    bvals = np.concatenate(([0.0], np.asarray(dwi.bvals, dtype=float)))
    observed, _, _ = find_shelling_scheme(bvals)
    if observed != expected:
        raise ValueError(
            f"Scheme mismatch for {name!r}: declared {expected!r}, found {observed!r}."
        )
    return observed


def synthetic_dwi(
    scheme: str = SINGLE_SHELL,
    *,
    n_directions: int = 24,
    vol_shape: Sequence[int] = (6, 6, 6),
    seed: int = 1234,
) -> DWI:
    """Build a tiny in-memory :class:`~nifreeze.data.dmri.DWI` for tests.

    Produces a dataset whose b-values classify as ``scheme`` (per
    :func:`~nifreeze.data.dmri.utils.find_shelling_scheme`). No physics — just a
    valid, cheap dataset to exercise the runner without any network access.

    Parameters
    ----------
    scheme : :obj:`str`
        One of :data:`SINGLE_SHELL`, :data:`MULTI_SHELL`, :data:`DSI`.
    n_directions : :obj:`int`
        Number of diffusion-weighted directions per shell.
    vol_shape : :obj:`Sequence`
        Spatial shape of the volume.
    seed : :obj:`int`
        Seed for the random number generator (reproducible).

    """
    rng = np.random.default_rng(seed)

    if scheme == SINGLE_SHELL:
        shells: tuple[float, ...] = (1000.0,)
    elif scheme == MULTI_SHELL:
        shells = (1000.0, 2000.0, 3000.0)
    elif scheme == DSI:
        # Many distinct b-values so ``find_shelling_scheme`` returns "DSI".
        shells = tuple(float(b) for b in range(500, 4001, 250))
    else:
        raise ValueError(f"Unknown scheme {scheme!r}; expected one of {SCHEMES}.")

    bvecs_list = []
    bvals_list = []
    for bval in shells:
        v = rng.normal(size=(n_directions, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        bvecs_list.append(v)
        bvals_list.append(np.full(n_directions, bval))

    # Prepend a single b=0 volume.
    bvecs = np.vstack([np.zeros((1, 3)), *bvecs_list])
    bvals = np.concatenate([np.zeros(1), *bvals_list])
    gradients = np.column_stack([bvecs, bvals])

    n_vols = gradients.shape[0]
    dataobj = rng.uniform(50.0, 1000.0, size=(*vol_shape, n_vols)).astype(np.float32)
    brainmask = np.ones(tuple(vol_shape), dtype=bool)

    return DWI(
        dataobj=dataobj,
        affine=np.eye(4),
        brainmask=brainmask,
        gradients=gradients,
    )


def synthetic_spec(scheme: str = SINGLE_SHELL, **kwargs) -> DatasetSpec:
    """A :class:`DatasetSpec` backed by :func:`synthetic_dwi` (for testing)."""
    return DatasetSpec(
        name=f"synthetic-{scheme}",
        title=f"Synthetic {scheme}",
        scheme=scheme,
        loader=lambda: synthetic_dwi(scheme, **kwargs),
        notes="Synthetic data for testing (no physiological meaning).",
    )


#: Real OpenNeuro datasets are registered here in a later phase.
DATASETS: list[DatasetSpec] = []
