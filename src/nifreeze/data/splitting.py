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
"""Data splitting helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from nifreeze.data.base import BaseDataset


def lovo_split(dataset: BaseDataset, index: int) -> tuple[Any, Any]:
    """
    Produce one fold of LOVO (leave-one-volume-out).

    Parameters
    ----------
    dataset : :obj:`nifreeze.data.base.BaseDataset`
        Dataset object.
    index : :obj:`int`
        Index of the volume to be left out in this fold.

    Returns
    -------
    :obj:`tuple` of :obj:`tuple`
        A tuple of two elements, the first element being  the components
        of the *train* data (including the data themselves and other metadata
        such as gradients for dMRI, or frame times for PET), and the second
        element being the *test* data.

    """
    mask = np.zeros(len(dataset), dtype=bool)
    mask[index] = True

    return (dataset[~mask], dataset[mask])
