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
"""Utilities for handling diffusion gradient tables."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

DEFAULT_GRADIENT_ATOL = 1e-2
"""Absolute dissimmilarity tolerance to trigger b-vector normalization."""

DEFAULT_HIGHB_THRESHOLD = 8000
"""A b-value cap for DWI data."""

DEFAULT_MULTISHELL_BIN_COUNT_THR = 7
"""Default bin count to consider a multishell scheme."""

DEFAULT_LOWB_THRESHOLD = 50
"""The lower bound for the b-value so that the orientation is considered a DW volume."""

DEFAULT_MAX_S0 = 1.0
"""Maximum value when considering the :math:`S_{0}` DWI signal."""

DEFAULT_MIN_S0 = 1e-5
"""Minimum value when considering the :math:`S_{0}` DWI signal."""

DEFAULT_NUM_BINS = 15
"""Number of bins to classify b-values."""

DTI_MIN_ORIENTATIONS = 6
"""Minimum number of nonzero b-values in a DWI dataset."""

GRADIENT_ABSENCE_ERROR_MSG = "No gradient table was provided."
"""Gradient absence error message."""

GRADIENT_EXPECTED_COLUMNS_ERROR_MSG = (
    "Gradient table must have four columns (3 direction components and one b-value)."
)
"""dMRI gradient expected columns error message."""

GRADIENT_OBJECT_ERROR_MSG = "Gradient table must be a numeric homogeneous array-like object"
"""Gradient object error message."""

GRADIENT_NDIM_ERROR_MSG = "Gradient table must be a 2D array"
"""dMRI gradient dimensionality error message."""

GRADIENT_VOLUME_DIMENSIONALITY_MISMATCH_ERROR = """\
Gradient table shape does not match the number of diffusion volumes: \
expected {n_volumes} rows, found {n_gradients}."""
"""dMRI volume count vs. gradient count mismatch error message."""


def find_shelling_scheme(
    bvals: np.ndarray,
    num_bins: int = DEFAULT_NUM_BINS,
    multishell_nonempty_bin_count_thr: int = DEFAULT_MULTISHELL_BIN_COUNT_THR,
    bval_cap: float = DEFAULT_HIGHB_THRESHOLD,
) -> tuple[str, list[npt.NDArray[np.floating]], list[np.floating]]:
    """
    Find the shelling scheme on the given b-values.

    Computes the histogram of the b-values according to ``num_bins``
    and depending on the nonempty bin count, classify the shelling scheme
    as single-shell if they are 2 (low-b and a shell); multi-shell if they are
    below the ``multishell_nonempty_bin_count_thr`` value; and DSI otherwise.

    Parameters
    ----------
    bvals : :obj:`list` or :obj:`~numpy.ndarray`
         List or array of b-values.
    num_bins : :obj:`int`, optional
        Number of bins.
    multishell_nonempty_bin_count_thr : :obj:`int`, optional
        Bin count to consider a multi-shell scheme.
    bval_cap : :obj:`float`, optional
        Maximum b-value to be considered in a multi-shell scheme.

    Returns
    -------
    scheme : :obj:`str`
        Shelling scheme.
    bval_groups : :obj:`list`
        List of grouped b-values.
    bval_estimated : :obj:`list`
        List of 'estimated' b-values as the median value of each b-value group.

    """

    # Bin the b-values: use -1 as the lower bound to be able to appropriately
    # include b0 values
    hist, bin_edges = np.histogram(bvals, bins=num_bins, range=(-1, min(max(bvals), bval_cap)))

    # Collect values in each bin
    bval_groups = []
    bval_estimated = []
    for lower, upper in zip(bin_edges[:-1], bin_edges[1:], strict=False):
        # Add only if a nonempty b-values mask
        if (mask := (bvals > lower) & (bvals <= upper)).sum():
            bval_groups.append(bvals[mask])
            bval_estimated.append(np.median(bvals[mask]))

    nonempty_bins = len(bval_groups)

    if nonempty_bins < 2:
        raise ValueError("DWI must have at least one high-b shell")

    if nonempty_bins == 2:
        scheme = "single-shell"
    elif nonempty_bins < multishell_nonempty_bin_count_thr:
        scheme = "multi-shell"
    else:
        scheme = "DSI"

    return scheme, bval_groups, bval_estimated


def format_gradients(
    value: npt.ArrayLike | None,
    norm_atol: float = DEFAULT_GRADIENT_ATOL,
    skip_normalization: bool = False,
) -> np.ndarray | None:
    """
    Validate and orient gradient tables to row-major convention.

    Parameters
    ----------
    value : :obj:`ArrayLike`
        The value to format.
    norm_atol : :obj:`float`, optional
        Absolute tolerance to consider a b-vector as unitary or b=0.
    skip_normalization : :obj:`bool`, optional
        If ``True``, skip b-vector normalization.

    Returns
    -------
    :obj:`~numpy.ndarray`
        Row-major convention gradient table.

    Raises
    ------
    exc:`ValueError`
        If ``value`` is not a 2D :obj:`~numpy.ndarray` (``value.ndim != 2``).

    Examples
    --------
    Passing an already well-formed table returns the data unchanged::

        >>> format_gradients(
        ...     [
        ...         [1, 0, 0, 0],
        ...         [0, 1, 0, 1000],
        ...         [0, 0, 1, 2000],
        ...         [0, 0, 0, 0],
        ...         [0, 0, 0, 1000],
        ...     ]
        ... )
        array([[   1,    0,    0,    0],
               [   0,    1,    0, 1000],
               [   0,    0,    1, 2000],
               [   0,    0,    0,    0],
               [   0,    0,    0,    0]], dtype=int16)

    Column-major inputs are automatically transposed when an expected
    number of diffusion volumes is provided::

        >>> format_gradients(
        ...     [[1, 0], [0, 1], [0, 0], [1000, 2000]],
        ... )
        array([[   1,    0,    0, 1000],
               [   0,    1,    0, 2000]], dtype=int16)

    Oversized b-vectors are normalized and the b-value is scaled accordingly::

        >>> format_gradients([[2.0, 0.0, 0.0, 1000]])
        array([[1.e+00, 0.e+00, 0.e+00, 2.e+03]])

    Near-zero b-vectors are suppressed to treat them as b0 measurements::

        >>> format_gradients([[1e-9, 0.0, 0.0, 1200], [1.0, 0.0, 0.0, 1000]])
        array([[   0,    0,    0,    0],
               [   1,    0,    0, 1000]], dtype=int16)

    Normaliztion can be skipped::

        >>> format_gradients([[2.0, 0.0, 0.0, 1000]], skip_normalization=True)
        array([[   2,    0,    0, 1000]], dtype=int16)

    Integer-like inputs are preserved when no rescaling is needed::

        >>> import numpy as np
        >>> format_gradients(np.array([[0, 0, 0, 0], [0, 0, 1, 1000]], dtype=np.int16)).dtype
        dtype('int16')

    Passing ``None`` raises the absence error::

        >>> format_gradients(None)
        Traceback (most recent call last):
        ...
        ValueError: No gradient table was provided.

    Gradient tables must always have two dimensions::

        >>> format_gradients([0, 1, 0, 1000])
        Traceback (most recent call last):
        ...
        ValueError: Gradient table must be a 2D array

    Gradient tables must have a regular shape::

        >>> format_gradients([[1, 2], [3, 4, 5]])
        Traceback (most recent call last):
        ...
        TypeError: Gradient table must be a numeric homogeneous array-like object

    Gradient tables must always have two dimensions::

        >>> format_gradients([0, 1, 0, 1000])
        Traceback (most recent call last):
        ...
        ValueError: Gradient table must be a 2D array

    """

    if value is None:
        raise ValueError(GRADIENT_ABSENCE_ERROR_MSG)

    try:
        formatted = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        # Conversion failed (e.g. nested ragged objects, non-numeric)
        raise TypeError(GRADIENT_OBJECT_ERROR_MSG) from exc

    if formatted.ndim != 2:
        raise ValueError(GRADIENT_NDIM_ERROR_MSG)

    # If the numeric values are all integers, preserve integer dtype
    if np.allclose(formatted, rounded := np.rint(formatted)):
        formatted = rounded.astype(
            formatted.dtype if np.issubdtype(formatted.dtype, np.integer) else np.int16
        )

    # Transpose if column-major
    formatted = formatted.T if formatted.shape[0] == 4 and formatted.shape[1] != 4 else formatted

    if formatted.shape[1] != 4:
        raise ValueError(GRADIENT_EXPECTED_COLUMNS_ERROR_MSG)

    if skip_normalization:
        return formatted

    # Normalize b-vectors in-place
    bvecs = formatted[:, :3]
    norms = np.linalg.norm(bvecs, axis=1)
    b0mask = np.isclose(norms, 0.0, atol=norm_atol)
    unitmask = np.isclose(norms, 1.0, atol=norm_atol)
    mask = ~unitmask & ~b0mask
    if np.any(mask):
        formatted = formatted.astype(float)  # Ensure float for in-place ops
        formatted[mask, :3] = bvecs[mask] / norms[mask, None]  # Norm b-vectors
        formatted[mask, 3] *= norms[mask]  # Scale b-values by norm

    formatted[b0mask, :] = 0  # Zero-out small b-vectors

    return formatted


def transform_fsl_bvec(
    b_ijk: np.ndarray, xfm: np.ndarray, imaffine: np.ndarray, invert: bool = False
) -> np.ndarray:
    """
    Transform a b-vector from the original space to the new space defined by the affine.

    Parameters
    ----------
    b_ijk : :obj:`~numpy.ndarray`
        The b-vector in FSL/DIPY conventions (i.e., voxel coordinates).
    xfm : :obj:`~numpy.ndarray`
        The affine transformation to apply.
        Please note that this is the inverse of the head-motion-correction affine,
        which maps coordinates from the realigned space to the moved (scan) space.
        In this case, we want to move the b-vector from the moved (scan) space into
        the realigned space.
    imaffine : :obj:`~numpy.ndarray`
        The image's affine, to convert.
    invert : :obj:`bool`, optional
        If ``True``, the transformation will be inverted.

    Returns
    -------
    :obj:`~numpy.ndarray`
        The transformed b-vector in voxel coordinates (FSL/DIPY).

    """
    xfm = np.linalg.inv(xfm) if invert else xfm.copy()

    # Go from world coordinates (xfm) to voxel coordinates
    ijk2ijk_xfm = np.linalg.inv(imaffine) @ xfm @ imaffine

    return ijk2ijk_xfm[:3, :3] @ b_ijk[:3]
