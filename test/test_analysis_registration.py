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
"""The registration harness on the CI phantom, whose copy moves all 24 volumes by 1° and 1 mm.

Without the dataset (``NIFREEZE_PHANTOMS``), these tests are skipped.
"""

import csv
import importlib.util
import shutil
from pathlib import Path

import nibabel as nb
import nitransforms as nt
import numpy as np
import pytest

from nifreeze.analysis import registration as reg
from nifreeze.analysis.phantoms import load_phantom
from nifreeze.registration.utils import displacements_within_mask

SEED = 0
"""Seed of the noise."""
VOLUMES = [0, 8, 16]
"""One volume of each shell."""


@pytest.fixture(scope="module")
def phantom(phantoms):
    """The noise-free and noisy references, the copy moving every volume, and its truth."""
    clean, _ = load_phantom(phantoms, "sherbrooke", counts=[])
    reference, cells = load_phantom(phantoms, "sherbrooke", noise=SEED, counts=[24])
    perturbation, load = next(
        (p, load) for p, load in cells if p.count == 24 and p.distortion is None
    )
    truth = dict(zip(perturbation.moved, perturbation.motions, strict=True))
    return clean, reference, load(), {index: truth[index] for index in VOLUMES}


def test_the_truth_has_no_error(phantom):
    clean, reference, moved, truth = phantom

    rows = reg.registration_accuracy(
        lambda fixed, moving, affine, mask, index: truth[index],
        clean,
        reference,
        moved,
        truth,
        fixed=["oracle"],
    )

    assert [row["index"] for row in rows] == VOLUMES
    assert [row["shell"] for row in rows] == [1000.0, 2000.0, 3500.0]
    assert all(row["max"] < 1e-9 for row in rows)


def test_no_registration_has_the_error_of_the_motion(phantom):
    clean, reference, moved, truth = phantom
    mask = nb.Nifti1Image(np.asanyarray(reference.brainmask, dtype="uint8"), reference.affine)

    rows = reg.registration_accuracy(
        lambda fixed, moving, affine, mask, index: np.eye(4),
        clean,
        reference,
        moved,
        truth,
        fixed=["oracle"],
    )

    for row in rows:
        motion = displacements_within_mask(mask, nt.linear.Affine(truth[row["index"]]))
        assert row["max"] == pytest.approx(motion.max())
        assert row["mean"] == pytest.approx(motion.mean())


def test_the_model_targets_need_a_model(phantom):
    clean, reference, moved, truth = phantom

    with pytest.raises(ValueError, match="model is required"):
        reg.registration_accuracy(np.eye, clean, reference, moved, truth)
    with pytest.raises(ValueError, match="unknown fixed"):
        reg.registration_accuracy(np.eye, clean, reference, moved, truth, fixed=["b0"])


@pytest.mark.skipif(shutil.which("antsRegistration") is None, reason="ANTs is not installed")
def test_ants_recovers_the_motion_from_the_oracle(phantom, tmp_path):
    """Registered to the noise-free original, the volumes are closer to the truth than unmoved.

    Only a smoke test: at 4 mm, the default settings (whose finest level is shrunk
    by 2) leave errors of the size of the motion, which the asv suite tracks.
    """
    clean, reference, moved, truth = phantom
    mask = nb.Nifti1Image(np.asanyarray(reference.brainmask, dtype="uint8"), reference.affine)

    rows = reg.registration_accuracy(
        reg.AntsRegistration(tmp_path), clean, reference, moved, truth, fixed=["oracle"]
    )

    motions = [
        displacements_within_mask(mask, nt.linear.Affine(truth[row["index"]])).max()
        for row in rows
    ]
    assert all(row["seconds"] > 0 for row in rows)
    assert np.median([row["max"] for row in rows]) < np.median(motions)


@pytest.mark.skipif(shutil.which("antsRegistration") is None, reason="ANTs is not installed")
def test_the_bench_writes_the_oracle_and_each_model(phantoms, tmp_path):
    path = Path(__file__).parents[1] / "tools" / "registration" / "bench.py"
    spec = importlib.util.spec_from_file_location("bench", path)
    assert spec is not None and spec.loader is not None
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)

    args = ["--dataset", str(phantoms), "--protocol", "sherbrooke", "--out", str(tmp_path)]
    bench.main([*args, "--models", "avgdwi", "--magnitudes", "1", "--every", "8"])

    out = tmp_path / "sherbrooke"
    assert sorted(p.name for p in out.iterdir()) == [
        "desc-m1_model-avgdwi.tsv",
        "desc-m1_oracle.tsv",
        "settings.json",
    ]
    with (out / "desc-m1_model-avgdwi.tsv").open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert [row["fixed"] for row in rows] == ["unmoved"] * 3 + ["moved"] * 3
    assert [int(row["index"]) for row in rows] == VOLUMES * 2
