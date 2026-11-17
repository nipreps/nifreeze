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

"""Test the runner."""

import json

from nifreeze.cli import run


class _Dataset:
    def to_nifti(self, filename, write_hmxfms=False):
        pass


class _Estimator:
    def __init__(self, *args, **kwargs):
        pass

    def run(self, dataset, **kwargs):
        return dataset


def test_ants_version():
    """The version of the antsRegistration on PATH."""
    version = run.ants_version()
    assert version is not None
    assert version.startswith("ANTs Version")


def test_ants_version_without_ants(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path))
    assert run.ants_version() is None


def test_main_records_ants_version(monkeypatch, tmp_path):
    """The run writes the ANTs version into a sidecar beside its output."""
    monkeypatch.setattr(run, "load", lambda *args, **kwargs: _Dataset())
    monkeypatch.setattr(run, "Estimator", _Estimator)
    monkeypatch.setattr(run, "ants_version", lambda: "ANTs Version: 2.4.4")

    input_file = tmp_path / "dwi.h5"
    input_file.touch()
    run.main([str(input_file), "--models", "avgdwi", "--output-dir", str(tmp_path)])

    sidecar = json.loads((tmp_path / "dwi.json").read_text())
    assert sidecar == {"ANTsVersion": "ANTs Version: 2.4.4"}
