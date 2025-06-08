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

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
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


def build_parser() -> ArgumentParser:
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
