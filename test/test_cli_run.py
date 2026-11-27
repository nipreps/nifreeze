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

import numpy as np
import pytest

from nifreeze.cli import run
from nifreeze.cli.parser import DEFAULT_GEM_PASSES
from nifreeze.data.dmri import DWI
from nifreeze.estimator import GEM_NAME
from nifreeze.registration.base import AFFINE


class _Dataset:
    def to_nifti(self, filename, write_estimates=False):
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
    run.main(
        [
            *argv,
            "--backend",
            "ants",
            "--align-config",
            "motion.yaml",
            "--gem-passes",
            "0",
            "--eddy",
        ]
    )

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
    assert all(estimator.kwargs["backend"] == "ants" for estimator in _Estimator.built)
    # The run's settings reach the motion passes; the eddy pass keeps its own
    assert eddy.ran["align_kwargs"] == {"align_config": Path("motion.yaml")}
    assert "align_config" not in first.kwargs


def test_main_chooses_the_eddy_model(monkeypatch, tmp_path):
    """``--eddy-model`` runs ``--eddy``'s pass with its own model and fit mode."""
    monkeypatch.setattr(run, "load", lambda *args, **kwargs: _Dataset())
    monkeypatch.setattr(run, "Estimator", _Estimator)
    monkeypatch.setattr(_Estimator, "built", [])
    monkeypatch.setattr(run, "ants_version", lambda: None)

    input_file = tmp_path / "dwi.h5"
    input_file.touch()
    run.main(
        [
            str(input_file),
            "--models",
            "singlegp-loo",
            "--eddy-model",
            "gqi-krr",
            "--gem-passes",
            "0",
            "--eddy",
            "--backend",
            "ants",
            "--output-dir",
            str(tmp_path),
        ]
    )

    motion, eddy = _Estimator.built
    assert (motion.args[0], eddy.args[0]) == ("gp-loo", "gqi-krr")
    assert (motion.kwargs["single_fit"], eddy.kwargs["single_fit"]) == (True, False)
    assert eddy.kwargs["prev"] is motion
    assert eddy.kwargs["estimates"] == "eddy"
    assert eddy.kwargs["align_config"] == AFFINE


def _dwi(phase_encoding):
    rng = np.random.default_rng(1234)
    bvecs = rng.standard_normal((6, 3))
    bvecs /= np.linalg.norm(bvecs, axis=1, keepdims=True)
    return DWI(
        dataobj=rng.random((6, 5, 5, 4), dtype=np.float32),
        affine=np.eye(4),
        gradients=np.column_stack((bvecs, np.full(6, 1000.0))),
        phase_encoding=phase_encoding,
    )


@pytest.mark.parametrize(
    ("metadata", "argument", "expected"),
    [(None, None, None), ("j", None, "j"), ("j", "i-", "i-"), (None, "k", "k")],
    ids=["unknown", "metadata", "argument-over-metadata", "argument"],
)
def test_main_takes_the_phase_encoding(monkeypatch, tmp_path, metadata, argument, expected):
    """``--phase-encoding`` overrides the direction the input's metadata gives."""
    dataset = _dwi(metadata)
    monkeypatch.setattr(run, "load", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(run, "Estimator", _Estimator)
    monkeypatch.setattr(_Estimator, "built", [])
    monkeypatch.setattr(run, "ants_version", lambda: None)
    monkeypatch.setattr(DWI, "to_nifti", lambda self, *args, **kwargs: None)

    input_file = tmp_path / "dwi.h5"
    input_file.touch()
    extra = [] if argument is None else ["--phase-encoding", argument]
    run.main(
        [str(input_file), "--backend", "ants", "--gem-passes", "0", "--output-dir", str(tmp_path)]
        + extra
    )

    assert dataset.phase_encoding == expected


def _run_dwi(monkeypatch, tmp_path, dataset, *argv):
    """Run the CLI on ``dataset``, with the estimators and the export mocked."""
    monkeypatch.setattr(run, "load", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(run, "Estimator", _Estimator)
    monkeypatch.setattr(_Estimator, "built", [])
    monkeypatch.setattr(run, "ants_version", lambda: None)
    monkeypatch.setattr(DWI, "to_nifti", lambda self, *args, **kwargs: None)

    input_file = tmp_path / "dwi.h5"
    input_file.touch()
    run.main([str(input_file), "--output-dir", str(tmp_path), *argv])
    return _Estimator.built


@pytest.mark.parametrize(
    ("argv", "passes", "model", "single_fit"),
    [
        ([], DEFAULT_GEM_PASSES, "gqi-krr", False),
        (["--gem-passes", "1"], 1, "gqi-krr", False),
        (["--models", "dti", "gp-loo", "--gem-passes", "2"], 2, "gp-loo", False),
        (["--eddy-model", "singlegp-loo", "--gem-passes", "4"], 4, "gp-loo", True),
        (["--gem-passes", "0"], 0, None, None),
    ],
    ids=["default", "one", "cascade", "own-model", "off"],
)
def test_main_chains_the_gem_passes(monkeypatch, tmp_path, argv, passes, model, single_fit):
    """After the cascade, each GEM pass starts from the estimates of the one before."""
    built = _run_dwi(monkeypatch, tmp_path, _dwi("j"), "--backend", "ants", *argv)

    gem = built[len(built) - passes :]
    assert len(built) - passes == (2 if "--models" in argv else 1)
    assert all(estimator.kwargs["backend"] == "ants" for estimator in built[: -passes or None])
    for before, estimator in zip(built[-passes - 1 :], gem, strict=False):
        assert estimator.kwargs["prev"] is before
        assert estimator.args[0] == model
        assert estimator.kwargs["single_fit"] is single_fit
        assert estimator.kwargs["backend"] == GEM_NAME
        assert estimator.kwargs["estimates"] == "eddy"
        assert "align_config" not in estimator.kwargs


def test_main_skips_gem_without_the_phase_encoding(monkeypatch, tmp_path):
    """Without a phase-encoding direction, the default run estimates head motion only."""
    with pytest.warns(UserWarning, match="GEM does not run"):
        built = _run_dwi(monkeypatch, tmp_path, _dwi(None))
    assert [estimator.kwargs.get("estimates") for estimator in built] == [None]

    # Given along with the data, the direction lets GEM run
    built = _run_dwi(monkeypatch, tmp_path, _dwi(None), "--phase-encoding", "j-")
    assert len(built) == 1 + DEFAULT_GEM_PASSES

    with pytest.raises(ValueError, match="GEM needs the phase-encoding direction"):
        _run_dwi(monkeypatch, tmp_path, _dwi(None), "--gem-passes", "2")


def test_main_runs_gem_on_dmri_only(monkeypatch, tmp_path):
    """Other data get no GEM pass, unless one is asked for, which is an error."""
    monkeypatch.setattr(run, "load", lambda *args, **kwargs: _Dataset())
    monkeypatch.setattr(run, "Estimator", _Estimator)
    monkeypatch.setattr(_Estimator, "built", [])
    monkeypatch.setattr(run, "ants_version", lambda: None)
    input_file = tmp_path / "pet.h5"
    input_file.touch()

    run.main([str(input_file), "--output-dir", str(tmp_path)])
    assert len(_Estimator.built) == 1

    with pytest.raises(ValueError, match="dMRI inputs only"):
        run.main([str(input_file), "--gem-passes", "1", "--output-dir", str(tmp_path)])


@pytest.mark.parametrize(
    ("argv", "itk"),
    [([], True), (["--gem-passes", "0"], False)],
    ids=["gem", "no-gem"],
)
def test_main_records_the_itk_version(monkeypatch, tmp_path, argv, itk):
    """A run whose GEM passes ran records the version of ITK, whatever the backend."""
    monkeypatch.setattr(run, "version", lambda distribution: f"{distribution} 1.0")
    _run_dwi(monkeypatch, tmp_path, _dwi("j"), "--backend", "ants", *argv)

    sidecar = json.loads((tmp_path / "dwi.json").read_text())
    assert sidecar == {"ANTsVersion": None} | ({"ITKVersion": "itk-core 1.0"} if itk else {})


def test_main_refuses_the_phase_encoding_of_other_data(monkeypatch, tmp_path):
    """Only a dMRI dataset has a phase-encoding direction to set."""
    monkeypatch.setattr(run, "load", lambda *args, **kwargs: _Dataset())
    input_file = tmp_path / "pet.h5"
    input_file.touch()
    with pytest.raises(ValueError, match="dMRI inputs only"):
        run.main([str(input_file), "--phase-encoding", "j", "--output-dir", str(tmp_path)])


def test_parser_rejects_an_unknown_phase_encoding(tmp_path, capsys):
    """The direction must be one of BIDS's six."""
    with pytest.raises(SystemExit):
        run.parse_args([str(tmp_path / "dwi.h5"), "--phase-encoding", "AP"])
    assert "invalid choice: 'AP'" in capsys.readouterr().err


def test_main_runs_pet_model_with_timing_file(monkeypatch, tmp_path):
    """``--models pet`` with ``--timing-file`` loads a PET dataset and builds BSplinePETModel."""
    import nibabel as nb

    from nifreeze.data.pet import PET
    from nifreeze.model.base import ModelFactory
    from nifreeze.model.pet import BSplinePETModel

    rng = np.random.default_rng(1234)
    n_frames = 6
    pet_nii = tmp_path / "sub-01_pet.nii.gz"
    nb.Nifti1Image(rng.random((4, 4, 4, n_frames), dtype=np.float32), np.eye(4)).to_filename(
        pet_nii
    )
    timing_json = tmp_path / "sub-01_pet.json"
    timing_json.write_text(json.dumps({"FrameTimesStart": [0, 10, 20, 30, 40, 50]}))

    seen = {}

    class _PetEstimator(_Estimator):
        def run(self, dataset, **kwargs):
            seen["dataset"] = dataset
            seen["model"] = ModelFactory.init(
                self.args[0], dataset=dataset, **self.kwargs.get("model_kwargs", {})
            )
            return dataset

    monkeypatch.setattr(run, "Estimator", _PetEstimator)
    monkeypatch.setattr(_Estimator, "built", [])
    monkeypatch.setattr(run, "ants_version", lambda: None)
    monkeypatch.setattr(PET, "to_nifti", lambda self, *args, **kwargs: None)

    run.main(
        [
            str(pet_nii),
            "--timing-file",
            str(timing_json),
            "--models",
            "pet",
            "--backend",
            "ants",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert isinstance(seen["dataset"], PET)
    assert isinstance(seen["model"], BSplinePETModel)
