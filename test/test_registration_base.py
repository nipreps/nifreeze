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
"""The registration interface: backends by name, plugged into the estimator."""

import gc
import subprocess
import sys
from importlib.metadata import PackageNotFoundError
from pathlib import Path

import numpy as np
import pytest
import yaml
from test_registration_conventions import (
    SKIP_ANTS,
    SKIP_ELASTIX,
    _affine,
    _error,
    _motion,
    _move,
    _volume,
)

from nifreeze.data.base import BaseDataset
from nifreeze.estimator import Estimator
from nifreeze.model.base import TrivialModel
from nifreeze.registration import AFFINE, RIGID, Registration, RegistrationFactory, base
from nifreeze.registration.ants import Ants, generate_command

EXAMPLES = Path(__file__).parent.parent / "examples" / "registration"
"""ANTs settings kept as examples, outside the package."""


class Recorder(Registration):
    """A backend that returns a fixed shift and records what each call received."""

    def __init__(self, shift=(1.0, 0.0, 0.0), **kwargs):
        super().__init__(**kwargs)
        self.matrix = np.eye(4)
        self.matrix[:3, 3] = shift
        self.calls: list[dict] = []

    def _register(self, fixed, moving, affine, mask, index, init, window):
        self.calls.append({"index": index, "init": init, "window": window})
        return self.matrix


def _series(n=3):
    """A volume and copies moved by known motions, with the unmoved volume as the target."""
    data, mask = _volume()
    affine = _affine("oblique")
    motions = [_motion(degrees=1.0 + k, shift=(0.5 * k, -1.0, 1.5)) for k in range(n)]
    dataobj = np.stack([_move(data, affine, m) for m in motions])
    dataset: BaseDataset = BaseDataset(dataobj=dataobj, affine=affine, brainmask=mask)
    return dataset, TrivialModel(dataset, predicted=data), motions


def test_the_factory_builds_backends_by_name():
    assert type(RegistrationFactory.init("ANTs")) is Ants
    assert type(RegistrationFactory.init("nifreeze.registration.ants:Ants")) is Ants
    with pytest.raises(ValueError, match="Unknown registration backend"):
        RegistrationFactory.init("flirt")


def test_elastix_is_the_default_where_installed(monkeypatch):
    def missing(distribution):
        raise PackageNotFoundError(distribution)

    monkeypatch.setattr(base, "version", lambda distribution: "0.25.4")
    assert base.default_backend() == "elastix"
    monkeypatch.setattr(base, "version", missing)
    assert base.default_backend() == "ants"
    assert type(RegistrationFactory.init()) is Ants


def test_settings_are_a_mapping_or_a_section_of_a_file(tmp_path):
    settings = base.load_config(RIGID, "ants")
    assert base.load_config(None, "ants") == settings
    assert base.load_config(AFFINE, "ants")["transforms"] == ["Affine", "Affine"]

    copy = base.load_config(settings, "elastix")  # a mapping is the backend's own
    copy["transforms"][0] = "Affine"
    assert settings["transforms"] == ["Rigid", "Rigid"]

    path = tmp_path / "ants-only.yaml"
    path.write_text(yaml.safe_dump({"ants": settings}))
    assert base.load_config(path, "ants") == settings
    with pytest.raises(ValueError, match="has no settings for elastix"):
        base.load_config(path, "elastix")


@pytest.mark.parametrize("path", sorted(EXAMPLES.glob("*.yaml")), ids=lambda path: path.stem)
def test_the_examples_build_an_ants_command(repodata, path):
    fixed, moving = repodata / "fileA.nii.gz", repodata / "fileB.nii.gz"
    command = generate_command(fixed, moving, align_config=path)
    assert command.cmdline.startswith("antsRegistration")


def test_ants_config_is_gone():
    """A leftover ``ants_config`` raises, instead of being ignored."""
    dataset, model, _ = _series()
    estimator = Estimator(
        model, strategy="linear", registration="ants", ants_config="dwi-to-dwi_level1"
    )
    with pytest.raises(TypeError, match="ants_config"):
        estimator.run(dataset)


def test_the_interface_imports_no_backend():
    """``import nifreeze.registration`` loads neither nipype nor itk."""
    code = (
        "import sys, nifreeze.registration; print('nipype' in sys.modules, 'itk' in sys.modules)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.stdout.split() == ["False", "False"]


def test_a_custom_backend_plugs_into_the_estimator():
    dataset, model, _ = _series()
    backend = Recorder()

    Estimator(model, strategy="linear", registration=backend).run(dataset)

    assert [call["index"] for call in backend.calls] == [0, 1, 2]
    assert all(call["window"] is not None for call in backend.calls)
    assert np.allclose(dataset.motion, backend.matrix)


def test_a_chain_starts_each_backend_from_the_previous_estimate():
    dataset, model, _ = _series()
    first, second = Recorder(), Recorder(shift=(2.0, 0.0, 0.0))
    chain = Estimator(model, strategy="linear", registration=first)

    Estimator(model, strategy="linear", prev=chain, registration=second).run(dataset)

    assert all(call["init"] is None for call in first.calls)
    assert all(np.array_equal(call["init"], first.matrix) for call in second.calls)


def test_a_window_that_cannot_be_estimated_is_not_applied():
    data, mask = _volume()
    backend = Recorder()

    with pytest.warns(UserWarning, match="without clipping"):
        backend(data, -data, _affine("ras"), mask, 3)

    assert backend.calls[0]["window"] is None
    backend = Recorder(clip="none")
    backend(data, data, _affine("ras"), mask, 3)
    assert backend.calls[0]["window"] is None


def test_ants_removes_its_temporary_folder():
    backend = Ants()
    folder = backend.workdir
    assert folder.is_dir()

    del backend
    gc.collect()
    assert not folder.exists()


@pytest.mark.parametrize(
    "name", [pytest.param("elastix", marks=SKIP_ELASTIX), pytest.param("ants", marks=SKIP_ANTS)]
)
def test_the_estimator_registers_with_a_backend_by_name(tmp_path, name):
    """Through the estimator, each backend recovers the motions and leaves its traces."""
    dataset, model, motions = _series()

    Estimator(model, strategy="linear", registration=name).run(
        dataset, omp_nthreads=1, seed=12345, debug_dir=tmp_path
    )

    for estimated, motion in zip(dataset.motion, motions, strict=True):
        assert _error(dataset.brainmask, dataset.affine, estimated, motion) < 0.5
    folder = tmp_path / "00-trivialmodel"
    for kind in ("predicted", "sample", "aligned"):
        assert len(list(folder.glob(f"{kind}_*.nii"))) == len(motions)
