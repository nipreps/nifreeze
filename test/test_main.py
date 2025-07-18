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

import os
import sys
from pathlib import Path

import pytest

import nifreeze.cli.run as cli_run
from nifreeze.__main__ import main


@pytest.fixture(autouse=True)
def set_command(monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(sys, "argv", ["nifreeze"])
        yield


def test_help(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    captured = capsys.readouterr()
    assert captured.out.startswith("usage: nifreeze [-h]")


@pytest.mark.parametrize(
    "write_hdf5",
    [
        False,
        True,
    ],
)
@pytest.mark.filterwarnings("ignore:write_hmxfms is set to True")
@pytest.mark.filterwarnings("error")
def test_main_call(tmp_path, monkeypatch, write_hdf5):
    """Test the main function of the CLI."""

    os.chdir(tmp_path)
    called = {}

    # Define smoke run method
    def smoke_estimator_run(self, dataset, **kwargs):
        called["dataset"] = dataset
        called["kwargs"] = kwargs

    # Monkeypatch
    monkeypatch.setattr(cli_run.Estimator, "run", smoke_estimator_run)

    input_file = Path(os.getenv("TEST_DATA_HOME")) / "dwi.h5"
    argv = [
        str(input_file),
        "--models",
        "dti",
    ]

    if write_hdf5:
        argv.append("--write-hdf5")
        out_filename = "dwi.h5"
        cli_run.main(argv)
    else:
        out_filename = "dwi.nii.gz"
        with pytest.warns(
            UserWarning,
            match="no motion affines were found",
        ):
            cli_run.main(argv)

    assert Path(tmp_path / out_filename).is_file()
    out_bval_filename = Path(Path(input_file).name).stem + ".bval"
    out_bval_path: Path = Path(tmp_path) / out_bval_filename
    out_bvec_filename = Path(Path(input_file).name).stem + ".bvec"
    out_bvec_path: Path = Path(tmp_path) / out_bvec_filename
    assert out_bval_path.is_file()
    assert out_bvec_path.is_file()
    if write_hdf5:
        out_h5_filename = Path(Path(input_file).name).stem + ".h5"
        out_h5_path: Path = Path(tmp_path) / out_h5_filename
        assert out_h5_path.is_file()
