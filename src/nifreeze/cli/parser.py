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
"""Parser module."""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from pathlib import Path

import yaml


def _parse_yaml_config(file_path: str) -> dict:
    """
    Parse YAML configuration file.

    Parameters
    ----------
    file_path : str
        Path to the YAML configuration file.

    Returns
    -------
    dict
        A dictionary containing the parsed YAML configuration.
    """
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def _build_parser() -> ArgumentParser:
    """
    Build parser object.

    Returns
    -------
    :obj:`~argparse.ArgumentParser`
        The parser object defining the interface for the command-line.
    """
    parser = ArgumentParser(
        description="A model-based algorithm for the realignment of 4D brain images.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "input_file",
        action="store",
        type=Path,
        help="Path to the HDF5 file containing the original 4D data.",
    )

    parser.add_argument(
        "--brainmask", action="store", type=Path, help="Path to a brain mask in NIfTI format."
    )

    parser.add_argument(
        "--align-config",
        action="store",
        type=_parse_yaml_config,
        default=None,
        help=(
            "Path to the yaml file containing the parameters "
            "to configure the image registration process."
        ),
    )
    parser.add_argument(
        "--models",
        action="store",
        nargs="+",
        default=["trivial"],
        help="Select the data model to generate registration targets.",
    )
    parser.add_argument(
        "--nthreads",
        "--omp-nthreads",
        "--ncpus",
        action="store",
        type=int,
        default=None,
        help="Maximum number of threads an individual process may use.",
    )
    parser.add_argument(
        "-J",
        "--n-jobs",
        "--njobs",
        dest="n_jobs",
        action="store",
        type=int,
        default=None,
        help="Number of parallel jobs.",
    )
    parser.add_argument(
        "--seed",
        action="store",
        type=int,
        default=None,
        help="Seed the random number generator for deterministic estimation.",
    )
    parser.add_argument(
        "--output-dir",
        action="store",
        type=Path,
        default=Path.cwd(),
        help=(
            "Path to the output directory. Defaults to the current directory."
            "The output file will have the same name as the input file."
        ),
    )

    parser.add_argument("--write-hdf5", action="store_true", help=("Generate an HDF5 file also."))

    g_dmri = parser.add_argument_group("Options for dMRI inputs")
    g_dmri.add_argument(
        "--gradient-file",
        action="store",
        nargs="+",
        type=Path,
        metavar="FILE",
        help="A gradient file containing b-vectors and b-values",
    )
    g_dmri.add_argument(
        "--b0-file",
        action="store",
        type=Path,
        metavar="FILE",
        help="A NIfTI file containing the b-zero reference",
    )

    g_dmri.add_argument(
        "--ignore-b0",
        action="store_true",
        help="Ignore the low-b reference and use the robust signal maximum",
    )

    g_pet = parser.add_argument_group("Options for PET inputs")
    g_pet.add_argument(
        "--timing-file",
        action="store",
        type=Path,
        metavar="FILE",
        help=(
            "A NIfTI file containing the timing information (onsets and durations) "
            "corresponding to the input file"
        ),
    )

    return parser


def _determine_single_fit_mode(model_name: str) -> bool:
    """Determine if a model is to be run in *single-fit mode*.
    If a model is requested to be run in *single-fit mode*, it will be run only
    once in all the data available.
    Parameters
    ----------
    model_name : :obj:`str
        Model name.
    Returns
    -------
    :obj:`bool`
        ``True`` if the model is to be run in *single-fit mode*, ``False``
        otherwise.
    """

    return model_name.lower().startswith("single")


def _normalize_model_name(model_name: str) -> str:
    """Normalize a model name.
    Normalize a model name by converting it to all lowercase and stripping the
    ``single`` prefix if present.
    Parameters
    ----------
    model_name : :obj:`str`
        Model name.
    Returns
    -------
    :obj:`str`
        Normalized model name.
    """

    return model_name.lower().replace("single", "")


def parse_args(argv: list) -> tuple[Namespace, dict, dict, dict]:
    """Parse the command line arguments and return a curated arguments.

    Performs further checks to ensure that necessary data is provided for the
    estimation process.

    Parameters
    ----------
    argv : list
        Arguments.

    Returns
    -------
    args : :obj:`Namespace`
        Populated namespace.
    extra_kwargs : :obj:`dict`
        Extra keyword arguments passed to the dataset.
    estimator_kwargs : :obj:`dict`
        Extra keyword arguments passed to the estimator.
    model_kwargs : :obj:`dict`
        Extra keyword arguments passed to the model.

    """

    parser = _build_parser()
    args = parser.parse_args(argv)

    extra_kwargs = {}

    if args.gradient_file:
        nfiles = len(args.gradient_file)

        if nfiles == 1:
            extra_kwargs["gradients_file"] = args.gradient_file[0]
        elif nfiles == 2:
            extra_kwargs["bvec_file"] = args.gradient_file[0]
            extra_kwargs["bval_file"] = args.gradient_file[1]
        else:
            parser.error("--gradient-file must be one or two files")

    if args.b0_file:
        extra_kwargs["b0_file"] = args.b0_file

    if args.timing_file:
        raise NotImplementedError("Cannot load PET timing information")

    model_kwargs = {}

    if args.ignore_b0:
        model_kwargs["ignore_bzero"] = True

    estimator_kwargs = {}

    for idx, _model in enumerate(args.models):
        single_fit = _determine_single_fit_mode(_model)
        model_name = _normalize_model_name(_model)
        args.models[idx] = model_name
        estimator_kwargs[model_name] = {"single_fit": single_fit}

    return args, extra_kwargs, estimator_kwargs, model_kwargs
