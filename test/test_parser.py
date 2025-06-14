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

"""Test parser."""

from pathlib import Path

import pytest

from nifreeze.cli.parser import (
    _build_parser,
    _determine_single_fit_mode,
    _normalize_model_name,
    parse_args,
)

MIN_ARGS = ["data/dwi.h5"]


@pytest.mark.parametrize(
    ("model_name", "single_fit"),
    [("singleDTI", True), ("PET", False)],
)
def test_determine_single_fit_mode(model_name, single_fit):
    assert _determine_single_fit_mode(model_name) == single_fit


@pytest.mark.parametrize(
    ("model_name", "normalized_name"),
    [("singleDTI", "dti"), ("PET", "pet")],
)
def test_normalize_model_name(model_name, normalized_name):
    assert _normalize_model_name(model_name) == normalized_name


@pytest.mark.parametrize(
    ("argv", "code"),
    [
        ([], 2),
    ],
)
def test_parser_errors(argv, code):
    """Check behavior of the parser."""
    with pytest.raises(SystemExit) as error:
        _build_parser().parse_args(argv)

    assert error.value.code == code


@pytest.mark.parametrize(
    "argv",
    [
        MIN_ARGS,
    ],
)
def test_parser_valid(tmp_path, argv):
    """Check valid arguments."""
    datapath = tmp_path / "data"
    datapath.mkdir(exist_ok=True)
    argv[0] = str(datapath)

    args = _build_parser().parse_args(argv)

    assert args.input_file == datapath
    assert args.models == ["trivial"]


@pytest.mark.parametrize(
    ("input_filebasename", "models", "nthreads", "n_jobs", "seed"),
    [
        ("dwi.h5", "trivial", 1, 1, 1234),
    ],
)
def test_parser_extended(tmp_path, datadir, input_filebasename, models, nthreads, n_jobs, seed):
    input_file = datadir / input_filebasename

    with pytest.raises(SystemExit):
        _build_parser().parse_args([str(input_file), str(tmp_path)])

    args = _build_parser().parse_args(
        [
            str(input_file),
            "--models",
            models,
            "--nthreads",
            str(nthreads),
            "--n-jobs",
            str(n_jobs),
            "--seed",
            str(seed),
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert args.input_file == input_file
    assert args.models == [models]
    assert args.nthreads == nthreads
    assert args.n_jobs == n_jobs
    assert args.seed == seed
    assert args.output_dir == tmp_path


@pytest.mark.parametrize(
    ("argval", "_models"),
    [
        ("b0", "b0"),
        ("s0", "s0"),
        ("avg", "avg"),
        ("average", "average"),
        ("mean", "mean"),
    ],
)
def test_models_arg(tmp_path, argval, _models):
    """Check the correct parsing of the models argument."""
    datapath = tmp_path / "data"
    datapath.mkdir(exist_ok=True)

    args = [str(datapath)] + ["--models", argval]
    opts = _build_parser().parse_args(args)

    assert opts.models == [_models]


@pytest.mark.parametrize(
    ("models", "single_fit"),
    [
        (["trivial"], [False]),
        (["singleDTI"], [True]),
        (["trivial", "singleDTI"], [False, True]),
    ],
)
def test_parser_models_singlefit_detection(repodata, datadir, models, single_fit):
    input_file = datadir / "dwi.h5"
    argv = [
        str(input_file),
        "--models",
        *models,
    ]

    args, extra_kwargs, estimator_kwargs, model_kwargs = parse_args(argv)

    for idx, (_model, _single_fit) in enumerate(zip(models, single_fit, strict=True)):
        _model_name = _normalize_model_name(_model)
        assert estimator_kwargs[_model_name]["single_fit"] == _single_fit
        assert args.models[idx] == _model_name


@pytest.mark.parametrize(
    ("models", "gradient_filebasename", "ignore_b0"),
    [
        (["trivial"], ["hcph_multishell.txt"], False),
        (["trivial"], ["hcph_multishell.bval", "hcph_multishell.bvec"], False),
        (["trivial"], ["hcph_multishell.bval", "hcph_multishell.bvec", "foo.txt"], False),
        (["singleDTI"], ["hcph_multishell.txt"], True),
    ],
)
def test_parser_dwi_data(repodata, datadir, models, gradient_filebasename, ignore_b0):
    gradient_file = []

    for fname in gradient_filebasename:
        gradient_file.append(str(repodata / fname))

    input_file = datadir / "dwi.h5"
    argv = [
        str(input_file),
        "--models",
        *models,
        "--gradient-file",
        *gradient_file,
    ]

    if ignore_b0:
        argv.append("--ignore-b0")

    nfiles = len(gradient_file)
    if nfiles > 2:
        with pytest.raises(SystemExit) as error:
            parse_args(argv)

        assert error.value.code == 2
        return

    args, extra_kwargs, estimator_kwargs, model_kwargs = parse_args(argv)

    if nfiles == 1:
        assert extra_kwargs["gradients_file"] == Path(gradient_file[0])
    elif nfiles == 2:
        assert extra_kwargs["bvec_file"] == Path(gradient_file[0])
        assert extra_kwargs["bval_file"] == Path(gradient_file[1])

    if ignore_b0:
        assert model_kwargs["ignore_bzero"] == ignore_b0


@pytest.mark.parametrize(
    "models",
    [
        "dti",
        "dki",
    ],
)
def test_parsed_dwimodel_instatiation(setup_random_dwi_data, datadir, models):
    from nifreeze.data.dmri import DWI
    from nifreeze.model import DKIModel, DTIModel

    (
        dwi_dataobj,
        affine,
        brainmask_dataobj,
        b0_dataobj,
        gradients,
        b0_thres,
    ) = setup_random_dwi_data

    input_file = datadir / "dwi.h5"
    argv = [
        str(input_file),
        "--models",
        models,
    ]

    args, extra_kwargs, estimator_kwargs, model_kwargs = parse_args(argv)

    dataset = DWI(
        dataobj=dwi_dataobj, affine=affine, brainmask=brainmask_dataobj, gradients=gradients
    )

    if models == "dti":
        model = DTIModel(dataset, **model_kwargs)
        assert model._model_class == "dipy.reconst.dti.TensorModel"
    elif models == "dki":
        model = DKIModel(dataset, **model_kwargs)
        assert model._model_class == "dipy.reconst.dki.DiffusionKurtosisModel"


@pytest.mark.parametrize(
    "models",
    [
        "dti",
    ],
)
def test_parsed_estimator_instatiation(datadir, models):
    from nifreeze.estimator import Estimator

    input_file = datadir / "dwi.h5"
    argv = [
        str(input_file),
        "--models",
        models,
    ]

    args, extra_kwargs, estimator_kwargs, model_kwargs = parse_args(argv)

    prev_model: Estimator | None = None
    for _model in args.models:
        estimator: Estimator = Estimator(
            _model,
            prev=prev_model,
            model_kwargs=model_kwargs,
            **estimator_kwargs,
        )
        prev_model = estimator
