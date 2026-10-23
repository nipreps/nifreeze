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
"""The catalog of acquisition protocols reproduced by the synthetic dataset."""

import hashlib
import math
import tomllib
from contextlib import nullcontext
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from nifreeze.data.dmri.io import from_nii

PHANTOMS = Path(__file__).parents[1] / "tools" / "phantoms"
CATALOG = tomllib.loads((PHANTOMS / "protocols.toml").read_text())
FIELDS = {
    "title": str,
    "gradients": str,
    "sha256": str,
    "shells": dict,
    "sphere": str,
    "source": str,
    "source_sha256": dict,
    "source_frame": str,
    "license": str,
    "citation": str,
    "retrieved": date,
    "scanner": str,
    "voxel_mm": list,
    "matrix": list,
    "phase_encoding": str,
    "echo_time": float,
    "echo_spacing": float,
    "total_readout_time": float,
    "snr_b0": float,
    "snr_source": str,
    "rationale": str,
}


@pytest.fixture(params=sorted(CATALOG), ids=sorted(CATALOG))
def protocol(request):
    entry = CATALOG[request.param]
    table = np.loadtxt(PHANTOMS / entry["gradients"])
    return request.param, entry, table


def test_entries_are_complete(protocol):
    _, entry, _ = protocol
    assert {key: type(entry[key]) for key in FIELDS} == FIELDS
    assert len(entry["voxel_mm"]) == len(entry["matrix"]) == 3
    assert entry["phase_encoding"] in {"AP", "LR", "SI"}
    assert entry["sphere"] in {"half", "whole"}
    assert "doi:" in entry["citation"]


def test_table_matches_its_checksum(protocol):
    _, entry, _ = protocol
    digest = hashlib.sha256((PHANTOMS / entry["gradients"]).read_bytes()).hexdigest()
    assert digest == entry["sha256"]


def test_table_has_the_protocol_shells_and_unit_vectors(protocol):
    _, entry, table = protocol
    vectors, bvals = table[:, :3], table[:, 3]
    shells, counts = np.unique(bvals, return_counts=True)
    assert {str(int(b)): int(n) for b, n in zip(shells, counts, strict=True)} == entry["shells"]
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms[bvals > 0], 1.0, atol=1e-5)
    assert not vectors[bvals == 0].any()
    # A half-sphere table keeps every direction within 90° of their mean
    unit = vectors[bvals > 0]
    mean = unit.mean(axis=0) / np.linalg.norm(unit.mean(axis=0))
    assert (entry["sphere"] == "half") == bool((unit @ mean > 0).all())


def test_tables_load_as_nifreeze_gradients(protocol, tmp_path):
    """A table is a gradients file for ``from_nii``: rows of world RAS vectors and b."""
    import nibabel as nb

    _, entry, table = protocol
    image = tmp_path / "dwi.nii.gz"
    nb.Nifti1Image(np.ones((2, 2, 2, len(table)), dtype=np.float32), np.eye(4)).to_filename(image)
    several = (table[:, 3] < 50).sum() > 1
    with pytest.warns(UserWarning, match="multiple b0") if several else nullcontext():
        dataset = from_nii(image, gradients_file=PHANTOMS / entry["gradients"])
    weighted = table[table[:, 3] >= 50]
    assert np.allclose(dataset.gradients[:, :3], weighted[:, :3], atol=1e-5)
    assert np.allclose(dataset.gradients[:, 3], weighted[:, 3])


def test_snr_is_documented(protocol):
    _, entry, _ = protocol
    assert not math.isnan(entry["snr_b0"]) and entry["snr_b0"] > 0
    assert entry["snr_source"] != "pending"
