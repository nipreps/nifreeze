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
    ITERATOR_MULTIPLICITY_ERROR_MSG,
    ITERATOR_SIZE_ERROR_MSG,
    SIZE_KEYS,
    SIZE_KWARG,
    START_INDEX_DATA_LENGTH_ERROR_MSG,
    START_INDEX_KWARG,
    START_INDEX_POSITIVITY_ERROR_MSG,
    STOP_INDEX_DATA_LENGTH_ERROR_MSG,
    STOP_INDEX_KWARG,
    STOP_INDEX_ORDERING_ERROR_MSG,
    UPTAKE_KWARG,
    _resolve_domain,
    _value_iterator,
    centralsym_iterator,
    linear_iterator,
    monotonic_value_iterator,
    random_iterator,
)


@pytest.mark.parametrize(
    "kwargs, err_type, err_match",
    [
        # No sizing input at all
        ({}, ValueError, re.escape(ITERATOR_SIZE_ERROR_MSG.format(features=SIZE_KEYS))),
        # Negative start
        ({SIZE_KWARG: 1, START_INDEX_KWARG: -1}, ValueError, START_INDEX_POSITIVITY_ERROR_MSG),
        # start_index beyond domain length (validation)
        (
            {SIZE_KWARG: 3, START_INDEX_KWARG: 3},
            ValueError,
            START_INDEX_DATA_LENGTH_ERROR_MSG.format(feature=SIZE_KWARG),
        ),
        (
            {BVALS_KWARG: [0, 1, 2], START_INDEX_KWARG: 3},
            ValueError,
            START_INDEX_DATA_LENGTH_ERROR_MSG.format(feature=BVALS_KWARG),
        ),
        (
            {UPTAKE_KWARG: [0.1, 0.2], START_INDEX_KWARG: 2},
            ValueError,
            START_INDEX_DATA_LENGTH_ERROR_MSG.format(feature=UPTAKE_KWARG),
        ),
        # stop_index <= start_index
        (
            {SIZE_KWARG: 3, START_INDEX_KWARG: 2, STOP_INDEX_KWARG: 1},
            ValueError,
            STOP_INDEX_ORDERING_ERROR_MSG,
        ),
        (
            {SIZE_KWARG: 3, START_INDEX_KWARG: 2, STOP_INDEX_KWARG: 2},
            ValueError,
            STOP_INDEX_ORDERING_ERROR_MSG,
        ),
        (
            {BVALS_KWARG: [0, 1000, 1000], START_INDEX_KWARG: 2, STOP_INDEX_KWARG: 1},
            ValueError,
            STOP_INDEX_ORDERING_ERROR_MSG,
        ),
        (
            {UPTAKE_KWARG: [0.12, 0.23, 0.56, 0.78], START_INDEX_KWARG: 3, STOP_INDEX_KWARG: 1},
            ValueError,
            STOP_INDEX_ORDERING_ERROR_MSG,
        ),
        # stop_index beyond values length (validation)
        (
            {BVALS_KWARG: [0, 1, 2], STOP_INDEX_KWARG: 4},
            ValueError,
            STOP_INDEX_DATA_LENGTH_ERROR_MSG.format(feature=BVALS_KWARG),
        ),
        (
            {UPTAKE_KWARG: [0.1, 0.2], START_INDEX_KWARG: 1, STOP_INDEX_KWARG: 3},
            ValueError,
            STOP_INDEX_DATA_LENGTH_ERROR_MSG.format(feature=UPTAKE_KWARG),
        ),
        # Multiple size-defining inputs are invalid under Option B
        (
            {BVALS_KWARG: [0, 1], UPTAKE_KWARG: [0.1, 0.2]},
            ValueError,
            re.escape(ITERATOR_MULTIPLICITY_ERROR_MSG),
        ),
        (
            {SIZE_KWARG: 3, BVALS_KWARG: [10, 20, 30, 40]},
            ValueError,
            re.escape(ITERATOR_MULTIPLICITY_ERROR_MSG),
        ),
        (
            {SIZE_KWARG: 2, UPTAKE_KWARG: [1.0, 2.0, 3.0]},
            ValueError,
            re.escape(ITERATOR_MULTIPLICITY_ERROR_MSG),
        ),
        (
            {SIZE_KWARG: 3, START_INDEX_KWARG: 1, BVALS_KWARG: [10, 20, 30, 40]},
            ValueError,
            re.escape(ITERATOR_MULTIPLICITY_ERROR_MSG),
        ),
        (
            {SIZE_KWARG: 2, START_INDEX_KWARG: 0, UPTAKE_KWARG: [1.0, 2.0, 3.0]},
            ValueError,
            re.escape(ITERATOR_MULTIPLICITY_ERROR_MSG),
        ),
    ],
)
def test_resolve_domain_errors(kwargs, err_type, err_match):
    with pytest.raises(err_type, match=err_match):
        _resolve_domain(**kwargs)


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({SIZE_KWARG: 4}, (0, 4)),
        ({SIZE_KWARG: 4, START_INDEX_KWARG: 3}, (3, 4)),
        ({SIZE_KWARG: 4, START_INDEX_KWARG: 0, STOP_INDEX_KWARG: 4}, (0, 4)),
        ({SIZE_KWARG: 10, START_INDEX_KWARG: 0, STOP_INDEX_KWARG: 4}, (0, 4)),
        ({BVALS_KWARG: [0, 1000, 2000, 3000]}, (0, 4)),
        ({BVALS_KWARG: [0, 1000, 2000, 3000], START_INDEX_KWARG: 2}, (2, 4)),
        ({BVALS_KWARG: [0, 1000, 2000, 3000], START_INDEX_KWARG: 1, STOP_INDEX_KWARG: 3}, (1, 3)),
        ({UPTAKE_KWARG: [0.1, 0.2, 0.3]}, (0, 3)),
        ({UPTAKE_KWARG: [0.1, 0.2, 0.3], START_INDEX_KWARG: 1}, (1, 3)),
        ({UPTAKE_KWARG: [0.1, 0.2, 0.3], START_INDEX_KWARG: 1, STOP_INDEX_KWARG: 2}, (1, 2)),
        ({SIZE_KWARG: 7, STOP_INDEX_KWARG: -1}, (0, 6)),
        ({BVALS_KWARG: [0, 1, 2, 3], STOP_INDEX_KWARG: -1}, (0, 3)),
        ({UPTAKE_KWARG: [0.1, 0.2, 0.3], START_INDEX_KWARG: 1, STOP_INDEX_KWARG: -1}, (1, 2)),
    ],
)
def test_resolve_domain(kwargs, expected):
    assert _resolve_domain(**kwargs) == expected


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
    assert len(result) == len(expected)
    assert result == expected


def test_linear_iterator_error():
    with pytest.raises(
        ValueError, match=re.escape(ITERATOR_SIZE_ERROR_MSG.format(features=SIZE_KEYS))
    ):
        list(linear_iterator())


@pytest.mark.parametrize(
    "feature, values, start_index, stop_index, expected",
    [
        (SIZE_KWARG, 4, 0, None, [0, 1, 2, 3]),
        (BVALS_KWARG, [0, 1000, 2000, 3000], 0, None, [0, 1, 2, 3]),
        (UPTAKE_KWARG, [-1.02, -0.56, 0.43, 1.16], 0, None, [0, 1, 2, 3]),
        (SIZE_KWARG, 4, 0, None, [0, 1, 2, 3]),
        (SIZE_KWARG, 4, None, None, [0, 1, 2, 3]),
        (SIZE_KWARG, 4, 3, None, [3]),
        (SIZE_KWARG, 7, 5, None, [5, 6]),
        (SIZE_KWARG, 9, 5, 8, [5, 6, 7]),
        (SIZE_KWARG, 9, 4, 9, [4, 5, 6, 7, 8]),
        (SIZE_KWARG, 5, 0, 4, [0, 1, 2, 3]),
        (BVALS_KWARG, [0, 1000, 2000, 3000], None, None, [0, 1, 2, 3]),
        (BVALS_KWARG, [0, 0, 1000, 1000, 700, 700, 2000, 2000, 0], 4, None, [4, 5, 6, 7, 8]),
        (BVALS_KWARG, [0, 0, 1000, 1000, 700, 700, 2000, 2000, 0], 4, 6, [4, 5]),
        (BVALS_KWARG, [0, 0, 1000, 1000, 700, 700], 0, 3, [0, 1, 2]),
        (UPTAKE_KWARG, [-1.02, -0.56, 0.43, 1.16], None, None, [0, 1, 2, 3]),
        (UPTAKE_KWARG, [-1.02, -0.56, 0.43, 1.16], 2, None, [2, 3]),
        (UPTAKE_KWARG, [-1.02, -0.56, 0.43, 1.16], 1, 3, [1, 2]),
        (UPTAKE_KWARG, [-1.02, -0.56, 0.43, 1.16], 0, 3, [0, 1, 2]),
    ],
)
def test_linear_iterator(feature, values, start_index, stop_index, expected):
    kwargs = {feature: values}
    if start_index is not None:
        kwargs[START_INDEX_KWARG] = start_index
    if stop_index is not None:
        kwargs[STOP_INDEX_KWARG] = stop_index

    result = list(linear_iterator(**kwargs))

    domain_len = values if feature == SIZE_KWARG else len(values)
    effective_start = 0 if start_index is None else start_index
    expected_len = (
        (stop_index - effective_start)
        if stop_index is not None
        else (domain_len - effective_start)
    )
    assert len(result) == expected_len
    assert result == expected


def test_random_iterator_error():
    with pytest.raises(
        ValueError, match=re.escape(ITERATOR_SIZE_ERROR_MSG.format(features=SIZE_KEYS))
    ):
        list(random_iterator())


@pytest.mark.parametrize(
    "feature, values, start_index, stop_index, seed, expected",
    [
        (SIZE_KWARG, 5, 0, None, 1234, [1, 2, 4, 0, 3]),
        (BVALS_KWARG, [0, 1000, 2000, 3000], 0, None, 42, [2, 1, 3, 0]),
        (UPTAKE_KWARG, [-1.02, -0.56, 0.43, 1.16], 0, None, True, [3, 0, 1, 2]),
        (SIZE_KWARG, 5, 0, None, None, [4, 0, 3, 1, 2]),
        (SIZE_KWARG, 5, None, None, 1234, [1, 2, 4, 0, 3]),
        (SIZE_KWARG, 5, 2, None, 0, [2, 4, 3]),
        (SIZE_KWARG, 8, 2, 6, 0, [4, 2, 3, 5]),
        (SIZE_KWARG, 5, 0, 4, 0, [2, 0, 1, 3]),
        (SIZE_KWARG, 5, 0, 5, 0, [2, 1, 0, 4, 3]),
        (BVALS_KWARG, [0, 1000, 2000, 3000], 0, None, None, [1, 3, 2, 0]),
        (BVALS_KWARG, [0, 1000, 2000, 3000], None, None, 42, [2, 1, 3, 0]),
        (BVALS_KWARG, [0, 1000, 2000, 3000, 4000], 1, 3, 42, [2, 1]),
        (BVALS_KWARG, [0, 1000, 2000, 3000, 4000], 0, 3, 42, [1, 0, 2]),
        (UPTAKE_KWARG, [-1.02, -0.56, 0.43, 1.16], 0, None, None, [3, 2, 0, 1]),
        (UPTAKE_KWARG, [-1.02, -0.56, 0.43, 1.16], None, None, True, [3, 0, 1, 2]),
        (UPTAKE_KWARG, [-1.02, -0.56, 0.43, 1.16, 1.52, 1.94], 2, 4, True, [2, 3]),
        (UPTAKE_KWARG, [-1.02, -0.56, 0.43, 1.16, 1.52, 1.94], 0, 4, True, [3, 0, 1, 2]),
    ],
)
def test_random_iterator(feature, values, start_index, stop_index, seed, expected):
    kwargs = {feature: values}
    if start_index is not None:
        kwargs[START_INDEX_KWARG] = start_index
    if stop_index is not None:
        kwargs[STOP_INDEX_KWARG] = stop_index
    if seed is not None:
        kwargs["seed"] = seed

    result = list(random_iterator(**kwargs))

    domain_len = values if feature == SIZE_KWARG else len(values)
    effective_start = 0 if start_index is None else start_index
    expected_len = (
        (stop_index - effective_start)
        if stop_index is not None
        else (domain_len - effective_start)
    )
    assert len(result) == expected_len
    assert len(set(result)) == len(expected)  # All unique
    if seed is not None:
        assert result == expected
        # Determinism check
        assert result == list(random_iterator(**kwargs))


def test_centralsym_iterator_error():
    with pytest.raises(
        ValueError, match=re.escape(ITERATOR_SIZE_ERROR_MSG.format(features=SIZE_KEYS))
    ):
        list(random_iterator())


@pytest.mark.parametrize(
    "feature, values, start_index, stop_index, expected",
    [
        (SIZE_KWARG, 6, 0, None, [3, 2, 4, 1, 5, 0]),
        (SIZE_KWARG, 6, None, None, [3, 2, 4, 1, 5, 0]),
        (SIZE_KWARG, 9, 2, 7, [4, 3, 5, 2, 6]),
        (SIZE_KWARG, 9, 0, 5, [2, 1, 3, 0, 4]),
        (BVALS_KWARG, [1000] * 6, 0, None, [3, 2, 4, 1, 5, 0]),
        (BVALS_KWARG, [1000] * 6, None, None, [3, 2, 4, 1, 5, 0]),
        (BVALS_KWARG, [0, 700, 1000, 2000, 3000], 0, None, [2, 1, 3, 0, 4]),
        (BVALS_KWARG, [0, 1000, 700, 2000, 3000], 0, None, [2, 1, 3, 0, 4]),
        (BVALS_KWARG, [0, 1000, 2000, 3000, 4000], 1, None, [3, 2, 4, 1]),
        (BVALS_KWARG, [0, 1000, 2000, 3000, 4000], 1, 4, [2, 1, 3]),
        (BVALS_KWARG, [0, 1000, 2000, 3000, 4000], 0, 4, [2, 1, 3, 0]),
        (UPTAKE_KWARG, [0.32, 0.27, -0.12], 0, None, [1, 0, 2]),
        (UPTAKE_KWARG, [0.32, 0.27, -0.12], None, None, [1, 0, 2]),
        (UPTAKE_KWARG, [-1.02, -0.56, 0.43, 0.89, 1.16], 0, None, [2, 1, 3, 0, 4]),
        (UPTAKE_KWARG, [-1.02, -0.56, 0.43, 1.16, 2.0], 1, None, [3, 2, 4, 1]),
        (UPTAKE_KWARG, [-1.02, -0.56, 0.43, 1.16, 2.0], 1, 4, [2, 1, 3]),
        (UPTAKE_KWARG, [-1.02, -0.56, 0.43, 1.16, 2.0], 1, None, [3, 2, 4, 1]),
    ],
)
def test_centralsym_iterator(feature, values, start_index, stop_index, expected):
    # The centralsym_iterator's output order depends only on the length
    kwargs = {feature: values}
    if start_index is not None:
        kwargs[START_INDEX_KWARG] = start_index
    if stop_index is not None:
        kwargs[STOP_INDEX_KWARG] = stop_index

    result = list(centralsym_iterator(**kwargs))

    domain_len = values if feature == SIZE_KWARG else len(values)
    effective_start = 0 if start_index is None else start_index
    expected_len = (
        (stop_index - effective_start)
        if stop_index is not None
        else (domain_len - effective_start)
    )
    assert len(result) == expected_len
    assert result == expected


@pytest.mark.parametrize(
    "kwargs, err_match",
    [
        ({}, ITERATOR_SIZE_ERROR_MSG.format(features=(BVALS_KWARG, UPTAKE_KWARG))),
        (
            {BVALS_KWARG: None},
            ITERATOR_SIZE_ERROR_MSG.format(features=(BVALS_KWARG, UPTAKE_KWARG)),
        ),
        (
            {UPTAKE_KWARG: None},
            ITERATOR_SIZE_ERROR_MSG.format(features=(BVALS_KWARG, UPTAKE_KWARG)),
        ),
        ({BVALS_KWARG: [0, 1000], UPTAKE_KWARG: [0.12]}, ITERATOR_MULTIPLICITY_ERROR_MSG),
    ],
)
def test_monotonic_value_iterator_error(kwargs, err_match):
    with pytest.raises(ValueError, match=re.escape(err_match)):
        monotonic_value_iterator(**kwargs)


@pytest.mark.parametrize(
    "feature, values, start_index, stop_index, expected",
    [
        (BVALS_KWARG, [0, 700, 1200], 0, None, [0, 1, 2]),
        (BVALS_KWARG, [0, 0, 1000, 700], 0, None, [0, 1, 3, 2]),
        (BVALS_KWARG, [0, 1000, 1500, 700, 2000], 0, None, [0, 3, 1, 2, 4]),
        (UPTAKE_KWARG, [0.3, 0.2, 0.1], 0, None, [0, 1, 2]),
        (UPTAKE_KWARG, [0.2, 0.1, 0.3], 0, None, [2, 0, 1]),
        (UPTAKE_KWARG, [-1.02, 1.16, -0.56, 0.43], 0, None, [1, 3, 2, 0]),
        (BVALS_KWARG, [0, 0, 1000, 1000, 700, 700, 2000, 2000, 0], 4, None, [8, 4, 5, 6, 7]),
        (BVALS_KWARG, [0, 0, 1000, 700], 1, 3, [1, 2]),
        (
            UPTAKE_KWARG,
            [-1.23, 1.06, 1.02, 1.38, -1.46, -1.12, -1.19, 1.24, 1.05],
            2,
            None,
            [3, 7, 8, 2, 5, 6, 4],
        ),
        (UPTAKE_KWARG, [0.2, 0.1, 0.3, 0.4], 1, 3, [2, 1]),
    ],
)
def test_monotonic_value_iterator(feature, values, start_index, stop_index, expected):
    kwargs = {feature: values}
    if start_index is not None:
        kwargs[START_INDEX_KWARG] = start_index
    if stop_index is not None:
        kwargs[STOP_INDEX_KWARG] = stop_index

    result = list(monotonic_value_iterator(**kwargs))

    expected_len = (
        (stop_index - start_index) if stop_index is not None else (len(values) - start_index)
    )
    assert len(result) == expected_len
    assert result == expected
    # If b-values, should be ordered by increasing value; if uptake values,
    # should be ordered by decreasing uptake
    sorted_vals = [values[i] for i in result]
    reverse = True if feature == UPTAKE_KWARG else False
    domain_vals = values[start_index:stop_index]
    assert sorted_vals == sorted(domain_vals, reverse=reverse)
