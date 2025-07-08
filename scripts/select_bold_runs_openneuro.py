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
"""
Select relevant fMRI BOLD runs based on a set of requirements.

Requirements:
  - Criterion 1: no single dataset shall contribute more than a fraction of the
    total number of runs.
  - Criterion 2: each BOLD run shall have between a minimum and maximum
    number of timepoints (inclusive).

Example
-------
  $ select_bold_runs_openneuro.py \
     dataset_features \
     selected_openneuro_datasets.tsv \
     1234 \
     --total-runs 4000 \
     --contr-fraction 0.05 \
     --min-timepoints 300 \
     --max-timepoints 1200

"""

import argparse
import logging
import re
import time
from pathlib import Path

import pandas as pd

CONTR_FRACTION = 0.05
"""Allowed contribution fraction for runs per dataset over the total runs."""

MIN_TIMEPOINTS = 300
"""Minimum number of BOLD timepoints per dataset."""

MAX_TIMEPOINTS = 1200
"""Maximum number BOLD timepoints per dataset."""

TOTAL_RUNS = 4000
"""Number of total runs."""

DATASETID = "datasetid"
FILENAME = "filename"
VOLS = "vols"


def filter_on_timepoint_count(
    df: pd.DataFrame, min_timepoints: int, max_timepoints: int
) -> pd.DataFrame:
    """Filter BOLD runs of datasets that are below or above a given number of
    timepoints.

    Filters BOLD runs whose timepoint count is not within the range
    `[min_timepoints, max_timepoints]`.

    Parameters
    ----------
    df : :obj:`~pd.DataFrame`
        BOLD run information.
    min_timepoints : :obj:`int`
        Minimum number of time points.
    max_timepoints : :obj:`int`
        Maximum number of time points.

    Returns
    -------
    :obj:`~pd.DataFrame`
        Filtered BOLD runs.
    """

    # Ensure the BOLD run has [min, max] timepoints (inclusive)
    timepoint_bounds = range(min_timepoints, max_timepoints + 1)
    return df[df[VOLS].isin(timepoint_bounds)]


def filter_on_run_contribution(df: pd.DataFrame, contrib_thr: int, seed: int) -> pd.DataFrame:
    """Filter BOLD runs of datasets to keep their total contribution under a
    threshold.

    Randomly picks BOLD runs of a dataset if the total number of runs exceeds
    the given threshold.

    Parameters
    ----------
    df : :obj:`~pd.DataFrame`
        BOLD run information.
    contrib_thr : :obj:`int`
        Contribution threshold in terms of number of runs.
    seed : :obj:`int`
        Random seed value.

    Returns
    -------
    :obj:`~pd.DataFrame`
        Filtered BOLD runs.
    """

    # Ensure no dataset contributes with more than a given threshold to the
    # total number of runs
    result = (
        df.groupby(DATASETID, group_keys=False)
        .apply(
            lambda x: (
                x.assign(**{DATASETID: x.name}).sample(n=contrib_thr, random_state=seed)
                if len(x) >= contrib_thr
                else x.assign(**{DATASETID: x.name})
            ),
            include_groups=False,
        )  # type: ignore
        .reset_index(drop=True)
    )

    # Make datasetid column come first
    return result[[DATASETID] + [c for c in result.columns if c != DATASETID]]


def filter_runs(
    df: pd.DataFrame, contrib_thr: int, min_timepoints: int, max_timepoints: int, seed: int
) -> pd.DataFrame:
    """Filter BOLD runs based on run count and timepoint criteria.

    Filters the BOLD runs to include only those that fulfil:
      - Criterion 1: the number of runs for a given dataset is below the
        threshold `contrib_thr`.
      - Criterion 2: the number of timepoints per BOLD run is between
       `[min_timepoints, max_timepoints]`.

    Parameters
    ----------
    df : :obj:`~pd.DataFrame`
        BOLD run information.
    contrib_thr : :obj:`int`
        Contribution threshold in terms of number of runs.
    min_timepoints : :obj:`int`
        Minimum number of time points.
    max_timepoints : :obj:`int``
        Maximum number of time points.
    seed : :obj:`int`
        Random seed value.

    Returns
    -------
    :obj:`~pd.DataFrame`
        Filtered BOLD runs.
    """

    # Criterion 2: the BOLD run has [min, max] timepoints (inclusive)
    df = filter_on_timepoint_count(df, min_timepoints, max_timepoints)

    # Criterion 1: the number of runs for a given dataset is below a threshold
    df = filter_on_run_contribution(df, contrib_thr, seed)

    return df


def identify_relevant_runs(
    df: pd.DataFrame,
    contrib_thr: int,
    min_timepoints: int,
    max_timepoints: int,
    seed: int,
) -> pd.DataFrame:
    """Identify relevant BOLD runs in terms of run and timepoint count constraints.

    Identifies the BOLD runs that fulfill the following criteria:
      - Criterion 1: the number of runs for a given dataset is below the
        threshold `contrib_thr`.
      - Criterion 2: the number of timepoints per BOLD run is between
       `[min_timepoints, max_timepoints]`.

    Runs are shuffled before the filtering process.

    Parameters
    ----------
    df : :obj:`~pd.DataFrame`
        BOLD run information.
    contrib_thr : :obj:`int`
        Contribution threshold in terms of the number of runs a dataset can
        contribute with over the total number of runs.
    min_timepoints : :obj:`int`
        Minimum number of time points.
    max_timepoints : :obj:`int``
        Maximum number of time points.
    seed : :obj:`int`
        Random seed value.

    Returns
    -------
    :obj:`~pd.DataFrame`
        Identified relevant BOLD runs.
    """

    # Shuffle records for randomness
    df = df.sample(frac=1, random_state=seed)

    # Filter runs
    df = filter_runs(df, contrib_thr, min_timepoints, max_timepoints, seed)

    return df


def _configure_logging(out_dirname: Path) -> None:
    # Clear existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(f"{out_dirname}/{Path(__file__).stem}.log"),
            logging.StreamHandler(),
        ],
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("in_dirname", type=Path, help="Input dirname")
    parser.add_argument("out_fname", type=Path, help="Output data list filename (*.TSV)")
    parser.add_argument(
        "seed", type=int, help="Random seed. Use the format 'YYYYMMDD' for a date."
    )
    parser.add_argument("--total-runs", type=int, default=TOTAL_RUNS, help="Number of total runs")
    parser.add_argument(
        "--contr-fraction",
        type=float,
        default=CONTR_FRACTION,
        help="Allowed contribution fraction for runs per dataset over the total runs",
    )
    parser.add_argument(
        "--min-timepoints",
        type=int,
        default=MIN_TIMEPOINTS,
        help="Minimum number of BOLD timepoints per dataset",
    )
    parser.add_argument(
        "--max-timepoints",
        type=int,
        default=MAX_TIMEPOINTS,
        help="Maximum number BOLD timepoints per dataset",
    )

    return parser


def _parse_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    return parser.parse_args()


def main() -> None:
    parser = _build_arg_parser()
    args = _parse_args(parser)

    _configure_logging(args.out_fname.parent)

    logging.info(
        "Script called with arguments:\n" + "\n".join(f"  {k}: {v}" for k, v in vars(args).items())
    )

    sep = "\t"

    start = time.time()

    # Consider only files that have the "ds\d{6}\.tsv" pattern (e.g.
    # ds000006.tsv, ds000021.tsv, etc.)
    datasets = {
        entry.stem: entry
        for entry in args.in_dirname.iterdir()
        if entry.is_file() and re.fullmatch(r"ds\d{6}\.tsv", entry.name)
    }

    # Read all feature data and concatenate them into a dataframe
    df = pd.concat([pd.read_csv(val, sep=sep) for val in datasets.values()], ignore_index=True)

    logging.info(f"Analyzing {len(df)} runs...")

    # Identify runs fulfilling the criteria
    contrib_thr = int(args.contr_fraction * args.total_runs)
    df_rel_runs = identify_relevant_runs(
        df,
        contrib_thr,
        args.min_timepoints,
        args.max_timepoints,
        args.seed,
    )

    end = time.time()
    duration = end - start

    logging.info(
        f"Identified {len(df_rel_runs)}/{len(df)} relevant runs in {duration:.2f} seconds."
    )

    # Keep only the first `total_runs`
    df_sel_runs = df_rel_runs.head(args.total_runs).sort_values(by=[DATASETID, FILENAME])

    logging.info(f"Selected the first {len(df_sel_runs)}/{len(df_rel_runs)} runs.")

    df_sel_runs.fillna("NA", inplace=True)
    df_sel_runs.to_csv(args.out_fname, sep=sep, index=False)


if __name__ == "__main__":
    main()
