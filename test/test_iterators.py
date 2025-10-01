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

import pytest

from nifreeze.utils.iterators import (
    _value_iterator,
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
