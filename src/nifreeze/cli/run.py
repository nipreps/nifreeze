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
"""NiFreeze runner."""

import json
import shutil
import subprocess
from importlib.metadata import version
from pathlib import Path
from warnings import warn

from nifreeze.cli.parser import DEFAULT_GEM_PASSES, parse_args
from nifreeze.data import BaseDataset, load
from nifreeze.data.dmri import DWI
from nifreeze.estimator import GEM_NAME, Estimator
from nifreeze.registration.base import default_backend

GEM_DATA_ERROR_MSG = "GEM estimates the eddy currents of dMRI inputs only: give --gem-passes 0."
"""GEM asked for on data other than a DWI error message."""
GEM_PHASE_ENCODING_ERROR_MSG = (
    "GEM needs the phase-encoding direction, which the input's metadata does not give: "
    "give --phase-encoding, or --gem-passes 0."
)
"""GEM asked for on a DWI without a phase-encoding direction error message."""
GEM_SKIPPED_WARN_MSG = (
    "The input's metadata does not give the phase-encoding direction, so GEM does not run "
    "and the eddy currents are not estimated: give --phase-encoding to estimate them."
)
"""The default run skipping GEM warning message."""


def gem_passes(requested: int | None, dataset: BaseDataset) -> int:
    """
    The number of GEM passes a run makes.

    Parameters
    ----------
    requested : :obj:`int`, optional
        The passes ``--gem-passes`` asks for, or :obj:`None` if it is not given.
    dataset : :obj:`~nifreeze.data.base.BaseDataset`
        The input, with its phase-encoding direction set if it is a DWI.

    Returns
    -------
    :obj:`int`
        The passes asked for, or by default :data:`~nifreeze.cli.parser.DEFAULT_GEM_PASSES`
        on a DWI whose phase-encoding direction is known, and none otherwise (with a
        warning on a DWI).

    Raises
    ------
    :obj:`ValueError`
        When passes are asked for on data other than a DWI, or on a DWI without a
        phase-encoding direction.

    """
    if not isinstance(dataset, DWI):
        if requested:
            raise ValueError(GEM_DATA_ERROR_MSG)
        return 0
    if dataset.phase_encoding is None:
        if requested:
            raise ValueError(GEM_PHASE_ENCODING_ERROR_MSG)
        if requested is None:
            warn(GEM_SKIPPED_WARN_MSG, stacklevel=2)
        return 0
    return DEFAULT_GEM_PASSES if requested is None else requested


def ants_version() -> str | None:
    """
    The version of the ``antsRegistration`` found on ``PATH``.

    Returns
    -------
    version : :obj:`str` or :obj:`None`
        The first line of its ``--version`` output, or :obj:`None` if there is none
        on ``PATH``.

    """
    path = shutil.which("antsRegistration")
    if path is None:
        return None
    output = subprocess.run([path, "--version"], capture_output=True, text=True).stdout
    lines = output.strip().splitlines()
    return lines[0] if lines else None


def main(argv: list[str] | None = None) -> None:
    """
    Entry point.

    Parameters
    ----------
    argv : obj:`list`, optional
        Arguments.

    Returns
    -------
    None

    """

    args, extra_kwargs, estimator_kwargs, model_kwargs = parse_args(argv)

    # Open the data with the given file path
    dataset: BaseDataset = load(
        args.input_file,
        brainmask_file=args.brainmask if args.brainmask else None,
        **extra_kwargs,
    )
    if args.phase_encoding is not None:
        # Over the input's metadata, whether it was read from NIfTI or nfHDF5
        if not isinstance(dataset, DWI):
            raise ValueError("--phase-encoding applies to dMRI inputs only.")
        dataset.phase_encoding = args.phase_encoding

    prev_model: Estimator | None = None
    for _model in args.models:
        single_fit = estimator_kwargs[_model]["single_fit"]
        estimator: Estimator = Estimator(
            _model,
            prev=prev_model,
            single_fit=single_fit,
            model_kwargs=model_kwargs,
            backend=args.backend,
        )
        prev_model = estimator

    passes = gem_passes(args.gem_passes, dataset)
    for _ in range(passes):
        # Eddy currents: passes of GEM, each from the estimates of the one before
        estimator = Estimator(
            args.eddy_model,
            prev=estimator,
            single_fit=args.eddy_single_fit,
            model_kwargs=model_kwargs,
            estimates="eddy",
            backend=GEM_NAME,
        )

    if args.eddy is not None:
        # Eddy currents: one more pass, from the motion estimates
        estimator = Estimator(
            args.eddy_model,
            prev=estimator,
            single_fit=args.eddy_single_fit,
            model_kwargs=model_kwargs,
            backend=args.backend,
            estimates="eddy",
            align_config=args.eddy,
        )

    _ = estimator.run(
        dataset,
        align_kwargs=None if args.align_config is None else {"align_config": args.align_config},
        omp_nthreads=args.nthreads,
        n_jobs=args.n_jobs,
        debug_dir=args.debug_dir,
        instant_update=args.instant_update,
        seed=args.seed,
    )

    # Set the output filename to be the same as the input filename
    output_filename = Path(Path(args.input_file).name).stem + ".nii.gz"
    output_path: Path = Path(args.output_dir) / output_filename

    # Save the DWI dataset to the output path
    if args.write_hdf5:
        output_h5_filename = Path(Path(args.input_file).name).stem + ".h5"
        output_h5_path: Path = Path(args.output_dir) / output_h5_filename
        dataset.to_filename(output_h5_path)

    dataset.to_nifti(output_path, write_estimates=True)

    # Record the version of the backend that registered, and of ITK if GEM ran
    if (args.backend or default_backend()) == "ants":
        provenance = {"ANTsVersion": ants_version()}
    else:
        provenance = {"ElastixVersion": version("itk-elastix")}
    if passes:
        provenance["ITKVersion"] = version("itk-core")
    output_json_filename = Path(Path(args.input_file).name).stem + ".json"
    output_json_path: Path = Path(args.output_dir) / output_json_filename
    output_json_path.write_text(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
