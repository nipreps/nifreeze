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
Prediction gallery: a declarative matrix of *(dataset × model × mode)* cells.

This subpackage drives the documentation "gallery" of predicted diffusion
volumes (see issue #458). A single source of truth — the registry of datasets
and models — is executed by :mod:`~nifreeze._gallery.run` to (a) render figures
for the docs, (b) emit a combinatorial coverage manifest recording what was
exercised, and (c) exercise the model layer on real data under code coverage.
"""

from nifreeze._gallery.manifest import (
    STATUS_ERROR,
    STATUS_RAN,
    STATUS_SKIPPED,
    CellResult,
    GalleryManifest,
)

__all__ = [
    "CellResult",
    "GalleryManifest",
    "STATUS_RAN",
    "STATUS_SKIPPED",
    "STATUS_ERROR",
]
