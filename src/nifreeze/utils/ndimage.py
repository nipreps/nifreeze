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

import os
import typing

import nibabel as nb
import numpy as np

ImgT = typing.TypeVar("ImgT", bound=nb.spatialimages.SpatialImage)


def load_api(path: str | os.PathLike[str], api: type[ImgT]) -> ImgT:
    img = nb.load(path)
    if not isinstance(img, api):
        raise TypeError(f"File {path} does not implement {api} interface")
    return img


def get_data(img: ImgT, dtype: np.dtype | str | None = None) -> np.ndarray:
    """Get the data from a nibabel image."""

    # Check if dtype is set and if it is a float type
    is_float = dtype is not None and np.issubdtype(np.dtype(dtype), np.floating)

    header = img.header

    def _no_slope_inter():
        return (None, None)

    # OE: Typechecking whines about header not having get_slope_inter
    if not is_float and getattr(header, "get_slope_inter", _no_slope_inter)() in (
        (None, None),
        (1.0, 0.0),
    ):
        return np.asanyarray(img.dataobj, dtype=header.get_data_dtype())

    return img.get_fdata(dtype=dtype if is_float else np.float32)
