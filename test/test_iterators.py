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
    centralsym_iterator,
    linear_iterator,
    random_iterator,
)


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"uptake": [-1.02, -0.56, 0.43, 1.16]}, [0, 1, 2, 3]),
    ],
)
def test_linear_iterator(kwargs, expected):
    assert list(linear_iterator(**kwargs)) == expected


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"uptake": [-1.02, -0.56, 0.43, 1.16], "seed": 0}, [2, 0, 1, 3]),
    ],
)
def test_random_iterator(kwargs, expected):
    obtained = list(random_iterator(**kwargs))
    assert obtained == expected
    # Determinism check
    assert obtained == list(random_iterator(**kwargs))


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"uptake": [-1.02, -0.56, 0.43, 0.89, 1.16]}, [[2, 1, 3, 0, 4]]),
    ],
)
def test_centralsym_iterator(kwargs, expected):
    # The centralsym_iterator's output order depends only on the length
    assert list(centralsym_iterator(**kwargs)) == expected
