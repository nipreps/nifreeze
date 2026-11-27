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

from argparse import (
    SUPPRESS,
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
    ArgumentTypeError,
    Namespace,
)
from pathlib import Path

from nifreeze.data.dmri.utils import PHASE_ENCODING_DIRECTIONS
from nifreeze.registration.base import AFFINE, BACKENDS

DEFAULT_GEM_PASSES = 3
"""The passes of GEM that a run on dMRI inputs makes unless ``--gem-passes`` sets them."""
EDDY_GEM_ERROR_MSG = (
    "--eddy registers each volume on its own, in place of GEM: give it with --gem-passes 0."
)
"""Both eddy-current corrections asked for error message."""
EDDY_MODEL_ERROR_MSG = (
    "--eddy-model has no pass to run: --gem-passes is 0, and --eddy is not given."
)
"""A model for eddy-current passes that do not run error message."""


def _passes(value: str) -> int:
    """A number of passes: an integer, zero or more."""
    try:
        number = int(value)
    except ValueError:
        number = -1
    if number < 0:
        raise ArgumentTypeError(f"expected zero or more passes, got {value!r}")
    return number


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
        "--backend",
        action="store",
        choices=sorted(BACKENDS),
        default=None,
        help=(
            "The registration backend, which registers each volume on its own (default: elastix)."
        ),
    )
    # The flag's former name, until a later release removes it
    parser.add_argument(
        "--registration",
        action="store",
        dest="backend",
        choices=sorted(BACKENDS),
        deprecated=True,
        help=SUPPRESS,
    )
    parser.add_argument(
        "--align-config",
        action="store",
        type=Path,
        metavar="FILE",
        default=None,
        help=(
            "A YAML file of registration settings, with a section per backend, for "
            "every level of the --models cascade (default: the packaged "
            "nifreeze/registration/config/rigid.yaml). To change a setting, copy and "
            "edit that file."
        ),
    )
    parser.add_argument(
        "--gem-passes",
        action="store",
        type=_passes,
        metavar="N",
        default=None,
        help=(
            "Estimate the eddy currents of a DWI with N passes of the global eddy model "
            "(GEM), after the --models cascade: each pass predicts every volume, from the "
            "volumes the pass before corrected, and fits one model of the fields to all "
            "of them. 0 turns GEM off. GEM needs the phase-encoding direction: without "
            "it, the default run estimates head motion only, with a warning (default: "
            f"{DEFAULT_GEM_PASSES} on dMRI inputs, none on others)."
        ),
    )
    parser.add_argument(
        "--eddy",
        action="store",
        nargs="?",
        type=Path,
        const=AFFINE,
        metavar="FILE",
        default=None,
        help=(
            "Estimate the eddy currents of a DWI by registering each volume on its own, "
            "in place of GEM (with --gem-passes 0): after the --models cascade, one more "
            "pass, started from the motion estimates, with the settings of FILE (default: "
            "the packaged nifreeze/registration/config/affine.yaml)."
        ),
    )
    parser.add_argument(
        "--eddy-model",
        action="store",
        metavar="MODEL",
        default=None,
        help=(
            "The model of the eddy-current passes, GEM's or --eddy's, apart from the "
            "--models cascade (as named there, the 'single' prefix included). It should "
            "predict the diffusion signal: the trivial model registers every volume to "
            "the b=0 reference (default: the cascade's last model)."
        ),
    )
    parser.add_argument(
        "--models",
        action="store",
        nargs="+",
        default=["gqi-krr"],
        help=(
            "Select the data model(s) used to generate registration targets; "
            "multiple models run as a cascade. By default each model runs in "
            "Leave-One-Volume-Out mode (refit per held-out volume). Prefix a "
            "model name with 'single' (e.g. 'singledti') to run it in "
            "single-fit mode: fit once on all volumes and reuse that locked "
            "fit (still predicting per volume from it). Single-fit is faster but "
            "leaks the queried volume into its own target (biasing motion toward zero); "
            "use it for development, testing, or coarse initialization only, "
            "not for accuracy-critical estimation. The closed-form LOVO models "
            "('gp-loo', 'gqi-krr') predict every held-out volume exactly from one "
            "matrix, so the 'single' prefix makes no difference for them."
        ),
    )
    parser.add_argument(
        "--gp-hyperparameters",
        action="store",
        type=Path,
        metavar="FILE",
        default=None,
        help=(
            "JSON file with Gaussian process hyperparameters for the 'gp-loo' model "
            "(e.g., pooled across the subjects of an acquisition protocol). When "
            "given, no hyperparameter is fitted at run time."
        ),
    )
    parser.add_argument(
        "--nvox-prefit",
        action="store",
        type=int,
        default=None,
        help=(
            "Number of random voxels used to fit the hyperparameters of the "
            "Gaussian process models ('gp', 'gp-loo')."
        ),
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
    parser.add_argument(
        "--debug-dir",
        action="store",
        type=Path,
        default=None,
        help=(
            "Keep the registration working files of each model in a folder of this "
            "directory, with an 'aligned_<index>.nii' image per volume: the registered "
            "volume with its affine corrected by the estimated motion (no resampling), "
            "which a viewer shows on top of its target 'predicted_<index>.nii'."
        ),
    )
    parser.add_argument(
        "--no-instant-update",
        action="store_false",
        dest="instant_update",
        help=(
            "Train the models on the original volumes only. By default, as soon as a "
            "volume is registered, the models see it realigned (resampled once from the "
            "original, with its rotated b-vector) when predicting the remaining volumes."
        ),
    )

    g_dmri = parser.add_argument_group("Options for dMRI inputs")
    g_dmri.add_argument(
        "--gradient-file",
        action="store",
        nargs="+",
        type=Path,
        metavar="FILE",
        help=(
            "The gradient table: one RAS+B file (MRtrix's .b, or NiPreps' .tsv), or FSL's "
            ".bvec and .bval files, in that order"
        ),
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
    g_dmri.add_argument(
        "--phase-encoding",
        action="store",
        choices=PHASE_ENCODING_DIRECTIONS,
        metavar="DIRECTION",
        default=None,
        help=(
            "The phase-encoding direction, as BIDS's PhaseEncodingDirection: an axis of "
            "the data array (i, j or k: its first, second or third), with a trailing '-' "
            "for the reverse sign (e.g., j-). It overrides the input's metadata (default: "
            "PhaseEncodingDirection from the input's BIDS sidecar, or from its nfHDF5 file)."
        ),
    )

    g_pet = parser.add_argument_group("Options for PET inputs")
    g_pet.add_argument(
        "--timing-file",
        action="store",
        type=Path,
        metavar="FILE",
        help=(
            "A JSON file containing the timing information (FrameTimesStart) "
            "corresponding to the input file"
        ),
    )

    return parser


def _determine_single_fit_mode(model_name: str) -> bool:
    """
    Determine if a model is to be run in *single-fit mode*.

    A model is run in *single-fit mode* (fit once on all data, reuse the locked
    fit — still predicting per volume from it) when its name is prefixed with
    ``single``. See
    :class:`~nifreeze.estimator.Estimator` for when this mode is appropriate —
    it leaks the queried volume into its own target, so it is for
    development/testing/initialization, not accuracy-critical estimation.

    Parameters
    ----------
    model_name : :obj:`str`
        Model name.

    Returns
    -------
    :obj:`bool`
        :obj:`True` if the model is to be run in *single-fit mode*, :obj:`False`
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


def parse_args(argv: list[str] | None = None) -> tuple[Namespace, dict, dict, dict]:
    """Parse the command line arguments and return a curated arguments.

    Performs further checks to ensure that necessary data is provided for the
    estimation process.

    Parameters
    ----------
    argv : :obj:`list`, optional
        Arguments.

    Returns
    -------
    args : :obj:`~argparse.Namespace`
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
        extra_kwargs["temporal_file"] = args.timing_file

    model_kwargs = {}

    if args.ignore_b0:
        model_kwargs["ignore_bzero"] = True

    if args.gp_hyperparameters:
        model_kwargs["hyperparameters"] = args.gp_hyperparameters

    if args.nvox_prefit is not None:
        model_kwargs["nvox_prefit"] = args.nvox_prefit

    estimator_kwargs = {}

    for idx, _model in enumerate(args.models):
        single_fit = _determine_single_fit_mode(_model)
        model_name = _normalize_model_name(_model)
        args.models[idx] = model_name
        estimator_kwargs[model_name] = {"single_fit": single_fit}

    _check_eddy(parser, args)
    _eddy_pass(args, estimator_kwargs)

    return args, extra_kwargs, estimator_kwargs, model_kwargs


def _check_eddy(parser: ArgumentParser, args: Namespace) -> None:
    """
    Refuse eddy-current options that exclude each other, with the parser's error.

    ``--eddy`` registers each volume on its own in place of GEM, so it needs
    ``--gem-passes 0``, and ``--eddy-model`` needs a pass to run.

    """
    if args.eddy is not None and args.gem_passes != 0:
        parser.error(EDDY_GEM_ERROR_MSG)
    if args.eddy_model is not None and args.eddy is None and args.gem_passes == 0:
        parser.error(EDDY_MODEL_ERROR_MSG)


def _eddy_pass(args: Namespace, estimator_kwargs: dict) -> None:
    """
    Resolve the model of the eddy-current passes, and its fit mode, in ``args``.

    The model of GEM's passes, or of ``--eddy``'s, is the one ``--eddy-model`` names, or
    else the cascade's last (already normalized), each in its own fit mode.

    Parameters
    ----------
    args : :obj:`~argparse.Namespace`
        The parsed arguments, which get ``eddy_model`` and ``eddy_single_fit``.
    estimator_kwargs : :obj:`dict`
        The fit mode of each model of the cascade.

    """
    if args.eddy_model is not None:
        args.eddy_single_fit = _determine_single_fit_mode(args.eddy_model)
        args.eddy_model = _normalize_model_name(args.eddy_model)
    else:
        args.eddy_model = args.models[-1]
        args.eddy_single_fit = estimator_kwargs[args.eddy_model]["single_fit"]
