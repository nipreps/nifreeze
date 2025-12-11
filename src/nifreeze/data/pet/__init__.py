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
PET data representation
------------------------
This submodule implements data structures and I/O utilities for PET data.

**Data Representation**.
The :class:`~nifreeze.data.pet.base.PET` class requires a ``dataobj`` that can
be an array-like object.
The class instantiation requires :attr:`~nifreeze.data.pet.base.PET.midframe`
and :attr:`~nifreeze.data.pet.base.PET.total_duration` value data to be
provided.
Both these values are computed from the frame start time (``FrameTimesStart`` in
BIDS terms).
The :meth:`~nifreeze.data.pet.io.from_nii` function takes a temporal file
parameter, which is expected to be a JSON file containing the frame start time.
Ultimately, the midframe and total duration values are computed by the
:meth:`~nifreeze.data.pet.utils.compute_temporal_markers` method.

"""

from nifreeze.data.pet.base import PET
from nifreeze.data.pet.io import from_nii

__all__ = ["PET", "from_nii"]
