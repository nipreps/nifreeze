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
"""Iterators to traverse the volumes in a 4D image."""

import random
from itertools import chain, zip_longest
from typing import Iterator

SIZE_KEYS = ("size", "bvals", "uptake")
"""Keys that may be used to infer the number of volumes in a dataset. When the
size of the structure to iterate over is not given explicitly, these keys
correspond to properties that distinguish one imaging modality from another, and
are part of the 4th axis (e.g. diffusion gradients in DWI or update in PET)."""

SIZE_KEYS_DOC = """
size : obj:`int`, optional
    Size of the structure to iterate over.
bvals : :obj:`list`, optional
    List of b-values corresponding to all orientations of a DWI dataset.
uptake : :obj:`list`, optional
    List of uptake values corresponding to all volumes of the dataset.
"""

ITERATOR_SIZE_ERROR_MSG = (
    f"None of {SIZE_KEYS} were provided to infer size: cannot build iterator without size."
)
"""Iterator size argument error message."""
KWARG_ERROR_MSG = "Keyword argument {kwarg} is required."
"""Iterator keyword argument error message."""
BVALS_KWARG = "bvals"
"""b-vals keyword argument name."""
UPTAKE_KWARG = "uptake"
"""Uptake keyword argument name."""


def _get_size_from_kwargs(kwargs: dict) -> int:
    """Extract the size from kwargs, ensuring only one key is used.

    Parameters
    ----------
    kwargs : :obj:`dict`
        The keyword arguments passed to the iterator function.

    Returns
    -------
    :obj:`int`
        The inferred size.

    Raises
    ------
    :exc:`ValueError`
        If size could not be extracted.
    """
    candidates = [kwargs[k] for k in SIZE_KEYS if k in kwargs]
    if candidates:
        return candidates[0] if isinstance(candidates[0], int) else len(candidates[0])
    raise ValueError(ITERATOR_SIZE_ERROR_MSG)


def linear_iterator(**kwargs) -> Iterator[int]:
    size = _get_size_from_kwargs(kwargs)
    return (s for s in range(size))


linear_iterator.__doc__ = f"""
Traverse the dataset volumes in ascending order.

Other Parameters
----------------
{SIZE_KEYS_DOC}

Notes
-----
Only one of the above keyword arguments may be provided at a time. If ``size``
is given, all other size-related keyword arguments will be ignored. If ``size``
is not provided, the function will attempt to infer the number of volumes from
the length or value of the provided keyword argument. If more than one such
keyword is provided, a :exc:`ValueError` will be raised.

Yields
------
:obj:`int`
    The next index.

Examples
--------
>>> list(linear_iterator(size=10))
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

"""


def random_iterator(**kwargs) -> Iterator[int]:
    size = _get_size_from_kwargs(kwargs)

    _seed = kwargs.get("seed", None)
    _seed = 20210324 if _seed is True else _seed

    random.seed(None if _seed is False else _seed)

    index_order = list(range(size))
    random.shuffle(index_order)
    return (x for x in index_order)


random_iterator.__doc__ = f"""
Traverse the dataset volumes randomly.

If the ``seed`` key is present in the keyword arguments, initializes the seed
of Python's ``random`` pseudo-random number generator library with the given
value. Specifically, if ``False``, ``None`` is used as the seed; if ``True``, a
default seed value is used.

Other Parameters
----------------
seed : :obj:`int`, :obj:`bool`, :obj:`str`, or ``None``
    If :obj:`int` or :obj:`str` or ``None``, initializes the seed of Python's random generator
    with the given value. If ``False``, the random generator is passed ``None``.
    If ``True``, a default seed value is set.

{SIZE_KEYS_DOC}

Notes
-----
Only one of the above keyword arguments may be provided at a time. If ``size``
is given, all other size-related keyword arguments will be ignored. If ``size``
is not provided, the function will attempt to infer the number of volumes from
the length or value of the provided keyword argument. If more than one such
keyword is provided, a :exc:`ValueError` will be raised.

Yields
------
:obj:`int`
    The next index.

Examples
--------
>>> list(random_iterator(size=15, seed=0))  # seed is 0
[1, 10, 9, 5, 11, 2, 3, 7, 8, 4, 0, 14, 12, 6, 13]
>>>  # seed is True -> the default value 20210324 is set
>>> list(random_iterator(size=15, seed=True))
[1, 12, 14, 5, 0, 11, 10, 9, 7, 8, 3, 13, 2, 6, 4]
>>> list(random_iterator(size=15, seed=20210324))
[1, 12, 14, 5, 0, 11, 10, 9, 7, 8, 3, 13, 2, 6, 4]
>>> list(random_iterator(size=15, seed=42))  # seed is 42
[8, 13, 7, 6, 14, 12, 5, 2, 9, 3, 4, 11, 0, 1, 10]

"""


def _value_iterator(values: list, ascending: bool, round_decimals: int = 2) -> Iterator[int]:
    """
    Traverse the given values in ascending or descenting order.

    Parameters
    ----------
    values : :obj:`list`
        List of values to traverse.
    ascending : :obj:`bool`
        If ``True``, traverse in ascending order; traverse in descending order
        otherwise.
    round_decimals : :obj:`int`, optional
        Number of decimals to round values for sorting.

    Yields
    ------
    :obj:`int`
        The next index.

    Examples
    --------
    >>> list(_value_iterator([0.0, 0.0, 1000.0, 1000.0, 700.0, 700.0, 2000.0, 2000.0, 0.0], True))
    [0, 1, 8, 4, 5, 2, 3, 6, 7]
    >>> list(_value_iterator([0.0, 0.0, 1000.0, 1000.0, 700.0, 700.0, 2000.0, 2000.0, 0.0], False))
    [7, 6, 3, 2, 5, 4, 8, 1, 0]
    >>> list(_value_iterator([-1.23, 1.06, 1.02, 1.38, -1.46, -1.12, -1.19, 1.24, 1.05], True))
    [4, 0, 6, 5, 2, 8, 1, 7, 3]
    >>> list(_value_iterator([-1.23, 1.06, 1.02, 1.38, -1.46, -1.12, -1.19, 1.24, 1.05], False))
    [3, 7, 1, 8, 2, 5, 6, 0, 4]

    """

    indexed_vals = sorted(
        ((round(v, round_decimals), i) for i, v in enumerate(values)), reverse=not ascending
    )
    return (index[1] for index in indexed_vals)


def bvalue_iterator(*_, **kwargs) -> Iterator[int]:
    """
    Traverse the volumes in a DWI dataset by increasing b-value.

    Parameters
    ----------
    bvals : :obj:`list`
        List of b-values corresponding to all orientations of the dataset.
        Please note that ``bvals`` is a keyword argument and MUST be provided
        to generate the volume sequence.

    Yields
    ------
    :obj:`int`
        The next index.

    Examples
    --------
    >>> list(bvalue_iterator(bvals=[0.0, 0.0, 1000.0, 1000.0, 700.0, 700.0, 2000.0, 2000.0, 0.0]))
    [0, 1, 8, 4, 5, 2, 3, 6, 7]

    """
    bvals = kwargs.pop(BVALS_KWARG, None)
    if bvals is None:
        raise TypeError(KWARG_ERROR_MSG.format(kwarg=BVALS_KWARG))
    return _value_iterator(bvals, ascending=True, **kwargs)


def uptake_iterator(*_, **kwargs) -> Iterator[int]:
    """
    Traverse the volumes in a PET dataset by decreasing uptake value.

    This function assumes that each uptake value corresponds to a single volume,
    and that this value summarizes the uptake of the volume in a meaningful way,
    e.g. a mean value across the entire volume.

    Parameters
    ----------
    uptake : :obj:`list`
        List of uptake values corresponding to all volumes of the dataset.
        Please note that ``uptake`` is a keyword argument and MUST be provided
        to generate the volume sequence.

    Yields
    ------
    :obj:`int`
        The next index.

    Examples
    --------
    >>> list(uptake_iterator(uptake=[-1.23, 1.06, 1.02, 1.38, -1.46, -1.12, -1.19, 1.24, 1.05]))
    [3, 7, 1, 8, 2, 5, 6, 0, 4]

    """
    uptake = kwargs.pop(UPTAKE_KWARG, None)
    if uptake is None:
        raise TypeError(KWARG_ERROR_MSG.format(kwarg=UPTAKE_KWARG))
    return _value_iterator(uptake, ascending=False, **kwargs)


def centralsym_iterator(**kwargs) -> Iterator[int]:
    size = _get_size_from_kwargs(kwargs)

    linear = list(range(size))
    return (
        x
        for x in chain.from_iterable(
            zip_longest(
                linear[size // 2 :],
                reversed(linear[: size // 2]),
            )
        )
        if x is not None
    )


centralsym_iterator.__doc__ = f"""
Traverse the dataset starting from the center and alternatingly progressing to the sides.

Other Parameters
----------------
{SIZE_KEYS_DOC}

Notes
-----
Only one of the above keyword arguments may be provided at a time. If ``size``
is given, all other size-related keyword arguments will be ignored. If ``size``
is not provided, the function will attempt to infer the number of volumes from
the length or value of the provided keyword argument. If more than one such
keyword is provided, a :exc:`ValueError` will be raised.

Examples
--------
>>> list(centralsym_iterator(size=10))
[5, 4, 6, 3, 7, 2, 8, 1, 9, 0]
>>> list(centralsym_iterator(size=11))
[5, 4, 6, 3, 7, 2, 8, 1, 9, 0, 10]
"""
