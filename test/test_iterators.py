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

import re

import pytest

from nifreeze.utils.iterators import (
    BVALS_KWARG,
    ITERATOR_SIZE_ERROR_MSG,
    KWARG_ERROR_MSG,
    UPTAKE_KWARG,
    _value_iterator,
    centralsym_iterator,
    linear_iterator,
    monotonic_value_iterator,
    random_iterator,
)


@pytest.mark.parametrize(
    "values, ascending, round_decimals, expected",
    [
        # Simple integers
        ([1, 2, 3], True, 2, [0, 1, 2]),
        ([1, 2, 3], False, 2, [2, 1, 0]),
        # Repeated values
        ([2, 1, 2, 1], True, 2, [1, 3, 0, 2]),
        ([2, 1, 2, 1], False, 2, [2, 0, 3, 1]),  # Ties are reversed due to reverse=True
        # Floats
        ([1.01, 1.02, 0.99], True, 2, [2, 0, 1]),
        ([1.01, 1.02, 0.99], False, 2, [1, 0, 2]),
        # Floats with rounding
        (
            [1.001, 1.002, 0.999],
            True,
            2,
            [0, 1, 2],
        ),  # All round to 1.00 (round_decimals=2), so original order
        (
            [1.001, 1.002, 0.999],
            True,
            4,
            [2, 0, 1],
        ),
        (
            [1.001, 1.002, 0.999],
            False,
            2,
            [2, 1, 0],
        ),  # All round to 1.00 (round_decimals=2), ties are reversed due to reverse=True
        (
            [1.001, 1.002, 0.999],
            False,
            4,
            [1, 0, 2],
        ),
        # Negative and positive
        ([-1.2, 0.0, 3.4, -1.2], True, 2, [0, 3, 1, 2]),
        ([-1.2, 0.0, 3.4, -1.2], False, 2, [2, 1, 3, 0]),  # Ties are reversed due to reverse=True
    ],
)
def test_value_iterator(values, ascending, round_decimals, expected):
    result = list(_value_iterator(values, ascending=ascending, round_decimals=round_decimals))
    assert result == expected


def test_linear_iterator_error():
    with pytest.raises(ValueError, match=re.escape(ITERATOR_SIZE_ERROR_MSG)):
        list(linear_iterator())


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"size": 4}, [0, 1, 2, 3]),
        ({"bvals": [0, 1000, 2000, 3000]}, [0, 1, 2, 3]),
        ({"uptake": [-1.02, -0.56, 0.43, 1.16]}, [0, 1, 2, 3]),
    ],
)
def test_linear_iterator(kwargs, expected):
    assert list(linear_iterator(**kwargs)) == expected


def test_random_iterator_error():
    with pytest.raises(ValueError, match=re.escape(ITERATOR_SIZE_ERROR_MSG)):
        list(random_iterator())


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"size": 5, "seed": 1234}, [1, 2, 4, 0, 3]),
        ({"bvals": [0, 1000, 2000, 3000], "seed": 42}, [2, 1, 3, 0]),
        ({"uptake": [-1.02, -0.56, 0.43, 1.16], "seed": True}, [3, 0, 1, 2]),
    ],
)
def test_random_iterator(kwargs, expected):
    obtained = list(random_iterator(**kwargs))
    assert obtained == expected
    # Determinism check
    assert obtained == list(random_iterator(**kwargs))


def test_centralsym_iterator_error():
    with pytest.raises(ValueError, match=re.escape(ITERATOR_SIZE_ERROR_MSG)):
        list(random_iterator())


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"size": 6}, [3, 2, 4, 1, 5, 0]),
        ({"bvals": [1000] * 6}, [3, 2, 4, 1, 5, 0]),
        ({"bvals": [0, 700, 1000, 2000, 3000]}, [2, 1, 3, 0, 4]),
        ({"bvals": [0, 1000, 700, 2000, 3000]}, [2, 1, 3, 0, 4]),
        ({"uptake": [0.32, 0.27, -0.12]}, [1, 0, 2]),
        ({"uptake": [-1.02, -0.56, 0.43, 0.89, 1.16]}, [2, 1, 3, 0, 4]),
    ],
)
def test_centralsym_iterator(kwargs, expected):
    # The centralsym_iterator's output order depends only on the length
    assert list(centralsym_iterator(**kwargs)) == expected


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"bvals": None},
        {"uptake": None},
        {"bvals": None, "uptake": None},
    ],
)
def test_monotonic_value_iterator_error(kwargs):
    with pytest.raises(
        TypeError, match=KWARG_ERROR_MSG.format(kwarg=f"{BVALS_KWARG} or {UPTAKE_KWARG}")
    ):
        monotonic_value_iterator(**kwargs)


def test_monotonic_value_iterator_sorting_preference():
    result = list(monotonic_value_iterator(bvals=[700, 1000], uptake=[0.14, 0.23, 0.47]))
    assert result == [0, 1]

    result = list(monotonic_value_iterator(bvals=None, uptake=[0.14, 0.23, 0.47]))
    assert result == [2, 1, 0]


@pytest.mark.parametrize(
    "feature, values, expected",
    [
        ("bvals", [0, 700, 1200], [0, 1, 2]),
        ("bvals", [0, 0, 1000, 700], [0, 1, 3, 2]),
        ("bvals", [0, 1000, 1500, 700, 2000], [0, 3, 1, 2, 4]),
        ("uptake", [0.3, 0.2, 0.1], [0, 1, 2]),
        ("uptake", [0.2, 0.1, 0.3], [2, 1, 0]),
        ("uptake", [-1.02, 1.16, -0.56, 0.43], [1, 3, 2, 0]),
    ],
)
def test_monotonic_value_iterator(feature, values, expected):
    obtained = list(monotonic_value_iterator(**{feature: values}))
    assert set(obtained) == set(range(len(values)))
    # If b-values, should be ordered by increasing value; if uptake values,
    # should be ordered by decreasing uptake
    sorted_vals = [values[i] for i in obtained]
    reverse = True if feature == "uptake" else False
    assert sorted_vals == sorted(values, reverse=reverse)


def test_linear_iterator_with_start_index():
    """Test linear iterator with start_index."""
    size = 7
    start_index = 3
    result = list(linear_iterator(size=size, start_index=start_index))
    assert len(result) == size
    assert result == [3, 4, 5, 6, 7, 8, 9]

    size = 4
    result = list(linear_iterator(size=size, start_index=start_index))
    assert len(result) == size
    expected = [3, 4, 5, 6]
    assert result == expected


def test_random_iterator_with_start_index():
    """Test random iterator with start_index."""
    size = 5
    start_index = 3
    result = list(random_iterator(size=size, seed=0, start_index=3))
    assert len(result) == size
    assert all(start_index <= idx < start_index + size for idx in result)
    assert len(set(result)) == 5  # All unique
    expected = [5, 4, 3, 7, 6]
    assert result == expected


def test_centralsym_iterator_with_start_index():
    """Test centralsym iterator with start_index."""
    size = 5
    start_index = 3
    result = list(centralsym_iterator(size=size, start_index=3))
    assert len(result) == size
    # Should iterate over indices 3-7, starting from center
    assert all(start_index <= idx < start_index + size for idx in result)
    expected = [5, 4, 6, 3, 7]
    assert result == expected


def test_monotonic_value_iterator_bvals_with_start_index():
    """Test monotonic_value iterator with bvals and start_index."""
    bvals = [0.0, 0.0, 1000.0, 1000.0, 700.0, 700.0, 2000.0, 2000.0, 0.0]

    # Full range
    full_result = list(monotonic_value_iterator(bvals=bvals))
    assert full_result == [0, 1, 8, 4, 5, 2, 3, 6, 7]

    # With start_index and size
    start_index = 4
    result = list(monotonic_value_iterator(bvals=bvals, start_index=start_index))
    # Indices 4-8: bvals are [700.0, 700.0, 2000.0, 2000.0, 0.0]
    # Sorted ascending: 0.0 (idx 8), 700.0 (idx 4), 700.0 (idx 5), 2000.0 (idx 6), 2000.0 (idx 7)
    # But with original indices: [8, 4, 5, 6, 7] relative to full array
    # Actually indices 4-8 get remapped: 0(->4), 1(->5), 2(->6), 3(->7), 4(->8)
    assert len(result) == len(bvals) - start_index
    expected = [8, 4, 5, 6, 7]  # Indices in original array
    assert result == expected


def test_monotonic_value_iterator_uptake_with_start_index():
    """Test monotonic_value iterator with uptake and start_index."""
    uptake = [-1.23, 1.06, 1.02, 1.38, -1.46, -1.12, -1.19, 1.24, 1.05]

    # Full range (descending for uptake)
    full_result = list(monotonic_value_iterator(uptake=uptake))
    assert full_result == [3, 7, 1, 8, 2, 5, 6, 0, 4]

    # Subset with start_index
    start_index = 2
    result = list(monotonic_value_iterator(uptake=uptake, start_index=start_index))
    # Indices 2-8: uptake values [1.02, 1.38, -1.46, -1.12, -1.19, 1.24, 1.05]
    # Sorted descending: 1.38 (idx 3), 1.24 (idx 7), 1.05 (idx 8), 1.02 (idx 2), -1.12 (idx 5), -1.19 (idx 6), -1.46 (idx 4)
    assert len(result) == len(uptake) - start_index
    expected = [3, 7, 8, 2, 5, 6, 4]
    assert result == expected


@pytest.mark.parametrize(
    "size,start_index,expected",
    [
        (10, 0, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        (7, 3, [3, 4, 5, 6, 7, 8, 9]),
        (4, 3, [3, 4, 5, 6]),
        (3, 0, [0, 1, 2]),
    ],
)
def test_linear_iterator_parametrize(size, start_index, expected):
    """Parametrized tests for linear iterator."""
    result = list(linear_iterator(size=size, start_index=start_index))
    assert result == expected


def test_iterators_produce_correct_count():
    """Test that all iterators produce the correct number of indices."""
    size = 5
    start_index = 10

    assert len(list(linear_iterator(size=size, start_index=start_index))) == size
    assert len(list(random_iterator(size=size, start_index=start_index, seed=0))) == size
    assert len(list(centralsym_iterator(size=size, start_index=start_index))) == size
