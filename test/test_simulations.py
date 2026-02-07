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

import numpy as np
import pytest

from nifreeze.testing.simulations import srtm


@pytest.mark.parametrize(
    "x, t, cr, cri, dt, nr, match",
    [
        # x.size != 3
        (
            np.array([1.0, 2.0]),
            np.ones(5),
            np.ones(5),
            np.ones(5),
            np.ones(5),
            5,
            "x must have length 3",
        ),
        # nr mismatch with cr
        (
            np.array([1.0, 0.1, 2.0]),
            np.ones(5),
            np.ones(3),
            np.ones(5),
            np.ones(5),
            5,
            "nr must match",
        ),
        # nr < 1
        (
            np.array([1.0, 0.1, 2.0]),
            np.ones(0),
            np.ones(0),
            np.ones(0),
            np.ones(0),
            0,
            "nr must be >= 1",
        ),
        # 1 + BP == 0
        (
            np.array([1.0, 0.1, -1.0]),
            np.ones(5),
            np.ones(5),
            np.ones(5),
            np.ones(5),
            5,
            r"Invalid BP",
        ),
        # k2 == 0
        (
            np.array([1.0, 0.0, 2.0]),
            np.ones(5),
            np.ones(5),
            np.ones(5),
            np.ones(5),
            5,
            "k2 must be nonzero",
        ),
    ],
)
def test_srtm_validation_errors(x, t, cr, cri, dt, nr, match):
    with pytest.raises(ValueError, match=match):
        srtm(x=x, t=t, cr=cr, cri=cri, dt=dt, nr=nr)
