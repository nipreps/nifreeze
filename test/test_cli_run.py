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
from pathlib import Path

import pytest

from nifreeze.cli import run
from nifreeze.registration.base import AFFINE


class _Dataset:
    def to_nifti(self, filename, write_hmxfms=False):
        pass


class _Estimator:
    built: list["_Estimator"] = []

    def __init__(self, *args, **kwargs):
        self.args, self.kwargs, self.ran = args, kwargs, {}
        _Estimator.built.append(self)

    def run(self, dataset, **kwargs):
        self.ran = kwargs
        return dataset


def test_ants_version():
    """The version of the antsRegistration on PATH."""
    version = run.ants_version()
    assert version is not None
    assert version.startswith("ANTs Version")


def test_ants_version_without_ants(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path))
    assert run.ants_version() is None


@pytest.mark.parametrize(
    ("backend", "expected"),
    [("ants", {"ANTsVersion": "ANTs Version: 2.4.4"}), ("elastix", {"ElastixVersion": "0.25.4"})],
)
def test_main_records_the_backend_version(monkeypatch, tmp_path, backend, expected):
    """The run writes the version of the backend that registered beside its output."""
    monkeypatch.setattr(run, "load", lambda *args, **kwargs: _Dataset())
    monkeypatch.setattr(run, "Estimator", _Estimator)
    monkeypatch.setattr(run, "ants_version", lambda: "ANTs Version: 2.4.4")
    monkeypatch.setattr(run, "version", lambda distribution: "0.25.4")
    monkeypatch.setattr(run, "default_backend", lambda: backend)

    input_file = tmp_path / "dwi.h5"
    input_file.touch()
    run.main([str(input_file), "--models", "avgdwi", "--output-dir", str(tmp_path)])

    sidecar = json.loads((tmp_path / "dwi.json").read_text())
    assert sidecar == expected


def test_main_appends_the_eddy_pass(monkeypatch, tmp_path):
    """``--eddy`` chains one more pass of the last model, with the affine settings."""
    monkeypatch.setattr(run, "load", lambda *args, **kwargs: _Dataset())
    monkeypatch.setattr(run, "Estimator", _Estimator)
    monkeypatch.setattr(_Estimator, "built", [])
    monkeypatch.setattr(run, "ants_version", lambda: None)

    input_file = tmp_path / "dwi.h5"
    input_file.touch()
    argv = [str(input_file), "--models", "avgdwi", "gqi-krr", "--output-dir", str(tmp_path)]
    run.main([*argv, "--registration", "ants", "--align-config", "motion.yaml", "--eddy"])

    first, second, eddy = _Estimator.built
    assert [estimator.args[0] for estimator in _Estimator.built] == [
        "avgdwi",
        "gqi-krr",
        "gqi-krr",
    ]
    assert eddy.kwargs["prev"] is second and second.kwargs["prev"] is first
    assert eddy.kwargs["align_config"] == AFFINE
    assert eddy.kwargs["estimates"] == "eddy"
    assert "estimates" not in first.kwargs and "estimates" not in second.kwargs
    assert all(estimator.kwargs["registration"] == "ants" for estimator in _Estimator.built)
    # The run's settings reach the motion passes; the eddy pass keeps its own
    assert eddy.ran["align_kwargs"] == {"align_config": Path("motion.yaml")}
    assert "align_config" not in first.kwargs
