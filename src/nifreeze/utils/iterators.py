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
from itertools import chain
from typing import Any, Iterator, Sequence

DEFAULT_ROUND_DECIMALS = 2
"""Round decimals to use when comparing values to be sorted for iteration purposes."""

SIZE_KEYS = ("size", "bvals", "uptake")
"""Keys that may be used to infer the number of volumes in a dataset. When the
size of the structure to iterate over is not given explicitly, these keys
correspond to properties that distinguish one imaging modality from another, and
are part of the 4th axis (e.g. diffusion gradients in DWI or update in PET)."""

DOMAIN_KEYS_DOC = """
size : :obj:`int`, optional
    Number of indices to generate.
bvals : :obj:`list`, optional
    List of b-values corresponding to all orientations of a DWI dataset. If
    provided and ``stop_index`` is not set, then ``stop_index`` is inferred as
    ``len(bvals)``. If ``stop_index`` is set, then ``stop_index`` must be
    ``<= len(bvals)``.
uptake : :obj:`list`, optional
    List of uptake values corresponding to all volumes of the dataset. If
    provided and ``stop_index`` is not set, then ``stop_index`` is inferred as
    ``len(uptake)``. If ``stop_index`` is set, then ``stop_index`` must be
    ``<= len(uptake)``
"""
START_INDEX_DOC = """
start_index : :obj:`int`, optional
    Starting index (inclusive) for the iteration. If provided, only indices
    ``>= start_index`` will be yielded.
"""
STOP_INDEX_DOC = """
stop_index : :obj:`int`, optional
    Stopping index (exclusive) for the iteration. If provided, only indices
    ``< stop_index`` will be yielded; if not provided and no domain-defining
    sequence (``bvals`` or ``uptake``) is provided, then ``stop_index`` is
    inferred as ``start_index + size``.
"""

ITERATOR_NOTES = """
Iterators operate over an absolute index domain and yield absolute indices. The
domain is always the half-open interval ``[start_index, stop_index)`` (end
exclusive). One of the size-defining inputs must be provided: ``size``,
``bvals``, or ``uptake``.  If ``size`` is provided, all other size-related
parameters will be ignored. When a sequence (``bvals``/``uptake``) is used, it
defines the maximum valid index (i.e., the domain length) and ``stop_index``
defaults to ``len(sequence)``.
"""

ITERATOR_SIZE_ERROR_MSG = """\
None of {features} were provided or had no valid values to infer size: cannot \
build iterator without size."""
"""Iterator size argument error message."""
ITERATOR_MULTIPLICITY_ERROR_MSG = """\
f"Multiple {SIZE_KEYS} were provided; provide exactly one or provide a value for
{SIZE_KWARG} to build the iterator."""
"""Iterator size multiplicity error message."""
START_INDEX_POSITIVITY_ERROR_MSG = "'start_index' must be positive."
"""Start index positivity error message."""
START_INDEX_DATA_LENGTH_ERROR_MSG = """\
'start_index' must be less than the length of {feature}."""
"""Start index data length error message."""
STOP_INDEX_ORDERING_ERROR_MSG = "'stop_index' must be larger than 'start_index'."
"""Stop index value ordering error message."""
STOP_INDEX_DATA_LENGTH_ERROR_MSG = """\
'stop_index' must be less or equal to the length of {feature}."""
"""Stop index data length error message."""
BVALS_KWARG = "bvals"
"""b-vals keyword argument name."""
UPTAKE_KWARG = "uptake"
"""Uptake keyword argument name."""
SIZE_KWARG = "size"
"""Size keyword argument name."""
STOP_INDEX_KWARG = "stop_index"
"""Stop index keyword argument name."""
START_INDEX_KWARG = "start_index"
"""Start index keyword argument name."""


def _resolve_domain(
    *, allowed_features: tuple[str, ...] = SIZE_KEYS, **kwargs: Any
) -> tuple[int, int]:
    """Resolve the iteration domain.

    Computes and validates the ``[start_index, stop_index)`` domain (end
    exclusive) from ``kwargs`` and returns ``(start_index, stop_index, feature)``.

    Parameters
    ----------
    allowed_features : :obj:`tuple` of :obj:`str`, optional
        Keys accepted to define the domain length (e.g., ``("size",)`` or
        ``("bvals", "uptake")``). Exactly one of these keys must be present in
        ``kwargs`` with a non-:obj:`None` value.
    **kwargs
        Iterator keyword arguments. Uses ``start_index`` and ``stop_index`` if
        present, and one of the keys in ``allowed_features`` to determine the
        domain length.

    Returns
    -------
    start_index : :obj:`int`
        Inclusive start index.
    stop_index : :obj:`int`
        Exclusive end index.

    Raises
    ------
    :exc:`ValueError`
        If ``start_index`` is negative, no allowed feature is provided, more
        than one allowed feature is provided, ``start_index`` is outside the
        domain, ``stop_index`` <= ``start_index``, or ``stop_index`` exceeds the
        available data length.
    """

    # Determine which allowed feature is provided (non-None)
    provided = [k for k in allowed_features if kwargs.get(k) is not None]
    if not provided:
        raise ValueError(ITERATOR_SIZE_ERROR_MSG.format(features=allowed_features))

    # If size is provided, it takes precedence and other size-related inputs are
    # ignored.
    if SIZE_KWARG in provided:
        feature = SIZE_KWARG
    else:
        if len(provided) > 1:
            raise ValueError(ITERATOR_MULTIPLICITY_ERROR_MSG)
        feature = provided[0]

    start_index = int(kwargs.get(START_INDEX_KWARG, 0) or 0)
    _stop_index = kwargs.get(STOP_INDEX_KWARG, None)

    if start_index < 0:
        raise ValueError(START_INDEX_POSITIVITY_ERROR_MSG)

    # Infer domain length
    value = kwargs[feature]
    n = int(value) if feature == SIZE_KWARG else len(value)

    if start_index >= n:
        raise ValueError(START_INDEX_DATA_LENGTH_ERROR_MSG.format(feature=feature))

    # Normalize stop_index to an exclusive integer end bound (slice semantics
    # for negatives)
    if _stop_index is None:
        stop_index = n
    else:
        stop_index = int(_stop_index)
        if stop_index < 0:
            stop_index = n + stop_index  # e.g., -1 -> n-1

    if stop_index <= start_index:
        raise ValueError(STOP_INDEX_ORDERING_ERROR_MSG)

    if stop_index > n:
        raise ValueError(STOP_INDEX_DATA_LENGTH_ERROR_MSG.format(feature=feature))

    return start_index, stop_index


_resolve_domain.__doc__ = f"""
Resolve the absolute index domain [start_index, stop_index) and validate.

The value of ``start_index`` defaults to 0; if ``stop_index`` is :obj:`None` and
``size`` is provided`, ``stop_index`` is set to ``start_index + size``; if one
of ``bvals`` or ``uptake`` is provided ``stop_index`` is set to ``len(bvals)`` or
``len(uptake)``, respectively.

Other Parameters
----------------
{DOMAIN_KEYS_DOC}
{START_INDEX_DOC}
{STOP_INDEX_DOC}

Returns
-------
:obj:`tuple`
    Starting (inclusive) and stopping index (exclusive) of the domain.

Raises
------
:exc:`ValueError`
    If ``start_index`` is negative.
:exc:`ValueError`
    If ``stop_index`` is not greater than ``start_index``.
:exc:`ValueError`
    If ``stop_index`` exceeds the length of a provided ``bvals``/``uptake``.
:exc:`ValueError`
    If none or more than one of the size-defining parameters (``size``,
    ``bvals``, ``uptake``) are provided.

Notes
-----
{ITERATOR_NOTES}

Examples
--------
>>> _resolve_domain(size=4)
(0, 4)
>>> _resolve_domain(size=4, start_index=3)
(3, 4)
>>> _resolve_domain(bvals=[0, 1000, 2000, 3000])
(0, 4)
>>> _resolve_domain(uptake=[0.1, 0.2, 0.3, 0.4], start_index=2)
(2, 4)
>>> _resolve_domain(bvals=[0, 1, 2, 3, 4], start_index=1, stop_index=4)
(1, 4)
>>> _resolve_domain(size=3, start_index=1, bvals=[10, 20, 30, 40])
(1, 3)
"""


def linear_iterator(**kwargs: Any) -> Iterator[int]:
    start, stop = _resolve_domain(**kwargs)
    return (s for s in range(start, stop))


linear_iterator.__doc__ = f"""
Traverse the dataset volumes in ascending order.

Other Parameters
----------------
{DOMAIN_KEYS_DOC}
{START_INDEX_DOC}
{STOP_INDEX_DOC}

Notes
-----
{ITERATOR_NOTES}

Yields
------
:obj:`int`
    The next index.

Examples
--------
>>> list(linear_iterator(size=10))
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
>>> list(linear_iterator(size=7, start_index=3))
[3, 4, 5, 6]
>>> list(linear_iterator(size=4, start_index=3))
[3]
>>> list(linear_iterator(bvals=[0, 0, 700, 1000, 1000, 200], start_index=3))
[3, 4, 5]
"""


def random_iterator(**kwargs: Any) -> Iterator[int]:
    start, stop = _resolve_domain(**kwargs)

    _seed = kwargs.get("seed", None)
    _seed = 20210324 if _seed is True else _seed

    random.seed(None if _seed is False else _seed)

    index_order = list(range(start, stop))
    random.shuffle(index_order)
    return (x for x in index_order)


random_iterator.__doc__ = f"""
Traverse the dataset volumes randomly.

If the ``seed`` key is present in the keyword arguments, initializes the seed
of Python's ``random`` pseudo-random number generator library with the given
value. Specifically, if :obj:`False`, :obj:`None` is used as the seed;
if :obj:`True`, a default seed value is used.

Other Parameters
----------------
seed : :obj:`int`, :obj:`bool`, :obj:`str`, or :obj:`None`
    If :obj:`int` or :obj:`str` or :obj:`None`, initializes the seed of Python's
    random generator with the given value. If :obj:`False`, the random generator
    is passed :obj:`None`. If :obj:`True`, a default seed value is set.

{DOMAIN_KEYS_DOC}
{START_INDEX_DOC}
{STOP_INDEX_DOC}

Notes
-----
{ITERATOR_NOTES}

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
>>> list(random_iterator(size=4, start_index=3, seed=0))
[3]
"""


def _value_iterator(
    values: Sequence[float], ascending: bool, round_decimals: int = DEFAULT_ROUND_DECIMALS
) -> Iterator[int]:
    """
    Traverse the given values in ascending or descenting order.

    Parameters
    ----------
    values : :obj:`Sequence`
        List of values to traverse.
    ascending : :obj:`bool`
        If :obj:`True`, traverse in ascending order; traverse in descending order
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


def monotonic_value_iterator(*_, **kwargs: Any) -> Iterator[int]:
    start, stop = _resolve_domain(allowed_features=(BVALS_KWARG, UPTAKE_KWARG), **kwargs)

    # At this point, exactly one feature is guaranteed to be present
    feature = next(k for k in (BVALS_KWARG, UPTAKE_KWARG) if kwargs.get(k) is not None)

    ascending = feature == BVALS_KWARG
    values = kwargs[feature]

    # Return a generator that converts relative indices to absolute indices
    return (
        start + idx
        for idx in _value_iterator(
            values[start:stop],
            ascending=ascending,
            round_decimals=kwargs.get("round_decimals", DEFAULT_ROUND_DECIMALS),
        )
    )


monotonic_value_iterator.__doc__ = f"""
Traverse the volumes by increasing b-value in a DWI dataset or by decreasing
uptake value in a PET dataset.

This function requires ``bvals`` or ``uptake`` be a keyword argument to generate
the volume sequence. The b-values are assumed to all orientations in a DWI
dataset, and uptake uptake values correspond to all volumes in a PET dataset.

It is assumed that each uptake value corresponds to a single volume, and that
this value summarizes the uptake of the volume in a meaningful way, e.g. a mean
value across the entire volume.

Other Parameters
----------------
{DOMAIN_KEYS_DOC}
{START_INDEX_DOC}
{STOP_INDEX_DOC}

Notes
-----
{ITERATOR_NOTES}

Yields
------
:obj:`int`
    The next index.

Examples
--------
>>> list(monotonic_value_iterator(bvals=[0.0, 0.0, 1000.0, 1000.0, 700.0, 700.0, 2000.0, 2000.0, 0.0]))
[0, 1, 8, 4, 5, 2, 3, 6, 7]
>>> list(monotonic_value_iterator(uptake=[-1.23, 1.06, 1.02, 1.38, -1.46, -1.12, -1.19, 1.24, 1.05]))
[3, 7, 1, 8, 2, 5, 6, 0, 4]
>>> list(monotonic_value_iterator(bvals=[0.0, 0.0, 1000.0, 1000.0, 700.0, 700.0, 2000.0, 2000.0, 0.0], start_index=4))
[8, 4, 5, 6, 7]
>>> list(monotonic_value_iterator(uptake=[-1.23, 1.06, 1.02, 1.38, -1.46, -1.12, -1.19, 1.24, 1.05], start_index=2))
[3, 7, 8, 2, 5, 6, 4]
"""


def centralsym_iterator(**kwargs: Any) -> Iterator[int]:
    start, stop = _resolve_domain(**kwargs)

    domain = list(range(start, stop))
    size = len(domain)

    center = size // 2

    return (
        domain[pos]
        for pos in chain.from_iterable(
            (center - k, center + k) if k else (center,)
            for k in range(0, center + 1)
            if 0 <= center - k < size or 0 <= center + k < size
        )
        if 0 <= pos < size
    )


centralsym_iterator.__doc__ = f"""
Traverse the dataset starting from the center and alternatingly progressing to the sides.

Other Parameters
----------------
{DOMAIN_KEYS_DOC}
{START_INDEX_DOC}
{STOP_INDEX_DOC}

Notes
-----
{ITERATOR_NOTES}

Examples
--------
>>> list(centralsym_iterator(size=10))
[5, 4, 6, 3, 7, 2, 8, 1, 9, 0]
>>> list(centralsym_iterator(size=11))
[5, 4, 6, 3, 7, 2, 8, 1, 9, 0, 10]
>>> list(centralsym_iterator(size=7, start_index=3))
[5, 4, 6, 3]
"""
