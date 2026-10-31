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
"""The harness of the registration experiments (``tools/registration/harness.py``)."""

import doctest
import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def harness():
    path = Path(__file__).parents[1] / "tools" / "registration" / "harness.py"
    spec = importlib.util.spec_from_file_location("harness", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_doctests_pass(harness):
    assert doctest.testmod(harness).failed == 0


def test_the_verdicts(harness):
    rng = np.random.default_rng(0)
    reference = rng.uniform(0.1, 1.0, 200)
    noise = rng.normal(0, 0.001, 200)

    better = harness.paired_comparison(reference - 0.1 + noise, reference, 0.02)
    assert better["verdict"] == "better"
    assert better["median_low"] < better["median"] < better["median_high"] < 0

    assert harness.paired_comparison(reference + noise, reference, 0.02)["verdict"] == (
        "not worse"
    )
    assert harness.paired_comparison(reference + 0.1, reference, 0.02)["verdict"] == "worse"

    loud = rng.normal(0, 1.0, 10)
    assert harness.paired_comparison(reference[:10] + loud, reference[:10], 0.02)["verdict"] == (
        "inconclusive"
    )

    same = harness.paired_comparison(reference, reference, 0.02)
    assert (same["verdict"], same["wilcoxon_p"], same["median"]) == ("not worse", 1.0, 0.0)

    # The bootstrap is reproducible, and resamples within strata
    strata = np.repeat([1000, 2000], 100)
    first = harness.paired_comparison(reference + noise, reference, 0.02, strata=strata)
    again = harness.paired_comparison(reference + noise, reference, 0.02, strata=strata)
    assert first == again

    with pytest.raises(ValueError, match="paired"):
        harness.paired_comparison(reference[:5], reference, 0.02)


def test_the_subsets_do_not_overlap(harness):
    for protocol, n in (("sherbrooke", 192), ("hcpya", 270)):
        dev = harness.subset(protocol, "dev", n)
        confirm = harness.subset(protocol, "confirm", n)
        assert not set(dev) & set(confirm)
        assert len(dev) == len(confirm) == {"sherbrooke": 64, "hcpya": 30}[protocol]


def test_provenance_records_what_ran(harness):
    record = harness.provenance({"metric": ["GC"]})
    assert {
        "AntsPath",
        "AntsVersion",
        "NiFreezeCommit",
        "NiFreezeDirty",
        "NiFreezeVersion",
        "Packages",
        "Threads",
        "Affinity",
        "Settings",
    } <= set(record)
    assert set(record["Threads"]) == set(harness.THREADS)
