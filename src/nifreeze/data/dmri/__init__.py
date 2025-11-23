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
dMRI data representation
------------------------
This submodule implements data structures and I/O utilities for diffusion MRI data.
Please, beware that *NiFreeze* is very opinionated about the gradients and diffusion
data representation.

**Gradient Table Representation**.
The :class:`~nifreeze.data.dmri.base.DWI` class represents diffusion MRI data must
be provided a gradient table, which is a :class:`numpy.ndarray` of shape (N, 4), where N
is the number of diffusion-weighted volumes.
The first three columns represent the gradient directions (b-vectors), and the fourth column
represents the b-values in s/mm².
*NiFreeze* expects that the gradient directions are normalized to unit length for non-zero
b-values, and that the b=0 volumes have a gradient direction of (0, 0, 0).
When non-unit b-vectors are detected, the corresponding b-value is automatically adjusted to
reflect the actual diffusion weighting.
If the input gradient table does not conform to these expectations, it will be automatically
corrected upon loading.
This means that, if you need to prepare the gradient table in a specific way (e.g., you don't
want normalization of b-vectors to modify the b-values), you should do so before initializing
the :class:`~nifreeze.data.dmri.base.DWI` object.

**Data Representation**.
The :class:`~nifreeze.data.dmri.base.DWI` class requires a ``dataobj`` that can be an array-like
object.
The final step of the initialization process examines the data object and the gradient table,
and removes b=0 volumes from the data **AND** the gradient table.
If no ``bzero`` parameter is provided, a reference low-b volume is computed as the median of all
the low-b volumes (b < 50 s/mm²) and inserted in the ``DWI.bzero`` attribute.
Therefore, ***NiFreeze* WILL NOT be able to reconstruct the original data organization**.
This design choice simplifies the internal representation and processing of diffusion MRI data.
If you want to calculate a b=0 reference map in a more sophisticated way (e.g., after realignment
of all the low-b volumes), you should handle this separately and feed your own reference through
the ``bzero`` parameter.

"""

from nifreeze.data.dmri.base import DWI
from nifreeze.data.dmri.io import from_nii

__all__ = ["DWI", "from_nii"]
