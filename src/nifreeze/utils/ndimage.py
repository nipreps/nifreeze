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
from warnings import warn

import nibabel as nb
import numpy as np

ImgT = typing.TypeVar("ImgT", bound=nb.spatialimages.SpatialImage)


def load_api(path: str | os.PathLike[str], api: type[ImgT]) -> ImgT:
    img = nb.load(path)
    if not isinstance(img, api):
        raise TypeError(f"File {path} does not implement {api} interface")
    return img


def get_data(img: ImgT, dtype: np.dtype | str | None = None) -> np.ndarray:
    """
    Extracts the data array from a nibabel image, handling data type and scaling.

    This function retrieves the data from a nibabel image object, optionally
    casting it to a specified data type.
    If the requested dtype is a floating point type, the function ensures
    that the data is loaded as floats, applying any scaling factors if present.
    Otherwise, it attempts to return the raw data array, avoiding
    unnecessary type conversion or scaling when possible.

    Parameters
    ----------
    img : :obj:`~nibabel.spatialimages.SpatialImage`
        A nibabel image object from which to extract the data.
    dtype : :obj:`~numpy.dtype` or :obj:`str`, optional
        Desired data type for the output array.

    Returns
    -------
    :obj:`~numpy.ndarray`
        The image data as a NumPy array, with type and scaling as specified.

    """

    is_float = dtype is not None and np.issubdtype(np.dtype(dtype), np.floating)
    # Warning: np.dtype(None) returns np.float64
    if not is_float and dtype is not None:
        warn(
            "Non-float dtypes are ignored and the original data type is preserved."
            " Please cast data explicitly.",
            stacklevel=2,
        )

    header = img.header

    def _no_slope_inter():
        return None, None

    # Typechecking whines about header not having get_slope_inter
    if not is_float and getattr(header, "get_slope_inter", _no_slope_inter)() in (
        (None, None),
        (1.0, 0.0),
    ):
        return np.asanyarray(img.dataobj, dtype=header.get_data_dtype())

    return img.get_fdata(dtype=dtype if is_float else np.float32)
