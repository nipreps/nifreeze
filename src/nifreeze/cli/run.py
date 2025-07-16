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

from pathlib import Path

from nifreeze.cli.parser import parse_args
from nifreeze.data import BaseDataset, load
from nifreeze.estimator import Estimator


def main(argv=None) -> None:
    """
    Entry point.

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

    prev_model: Estimator | None = None
    for _model in args.models:
        single_fit = estimator_kwargs[_model]["single_fit"]
        estimator: Estimator = Estimator(
            _model,
            prev=prev_model,
            single_fit=single_fit,
            model_kwargs=model_kwargs,
        )
        prev_model = estimator

    _ = estimator.run(
        dataset,
        align_kwargs=args.align_config,
        omp_nthreads=args.nthreads,
        n_jobs=args.n_jobs,
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

    dataset.to_nifti(output_path, write_hmxfms=True)


if __name__ == "__main__":
    main()
