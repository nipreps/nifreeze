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
"""The amplitude of every model's held-out prediction, on the CI phantom.

Correlation ignores scale, so a model can predict the right pattern at the wrong
amplitude and pass every correlation test (nipreps/nifreeze#495). Here, each
held-out prediction's mean within the brain must stay within a factor of the
observed volume's. The phantom has only 8 directions per shell, which pull the
predictions of the multi-shell models toward the other shells' amplitude (GP, by
up to 1.7 times at b=3500); with 64 directions per shell (the HBN protocol), GP-LOO
and GQI-KRR stay within 5%. So the gate catches scale errors, not that bias.
Without the dataset (``NIFREEZE_PHANTOMS``), these tests are skipped.
"""

import numpy as np
import pytest

from nifreeze.analysis import sensitivity as sens
from nifreeze.analysis.phantoms import load_phantom

EVALUATE = [0, 8, 16]
"""One volume of each shell (b=1000, 2000 and 3500)."""
SEED = 20261029
"""Seed of the noise."""
FACTOR = 2.0
"""Largest ratio, either way, between the predicted and observed means."""


@pytest.fixture(scope="module")
def reference(phantoms):
    """The phantom's reference, with the protocol's noise."""
    dataset, _ = load_phantom(phantoms, "sherbrooke", noise=SEED, counts=[])
    return dataset


@pytest.mark.parametrize(
    "model",
    [
        "avgdwi",
        pytest.param(
            "dti",
            marks=pytest.mark.xfail(
                reason="DTI predicts with the measured b=0 but fits without it (#599)",
                strict=True,
            ),
        ),
        pytest.param(
            "dki",
            marks=pytest.mark.xfail(
                reason="DKI's b=3500 predictions diverge with 8 directions per shell",
                strict=True,
            ),
        ),
        "gp",
        "gp-loo",
        "gqi-krr",
        pytest.param(
            "gqi",
            marks=pytest.mark.xfail(
                reason="GQI normalizes its prediction by the target's own norm (#495)",
                strict=True,
            ),
        ),
    ],
)
def test_held_out_predictions_keep_the_amplitude(reference, model):
    mask = np.asanyarray(reference.brainmask, dtype=bool)

    predicted = sens.held_out_predictions(model, reference, EVALUATE)

    for index, prediction in predicted.items():
        ratio = prediction[mask].mean() / np.asarray(reference.dataobj[index])[mask].mean()
        assert 1 / FACTOR < ratio < FACTOR, f"volume {index}: {ratio:.3f}"
