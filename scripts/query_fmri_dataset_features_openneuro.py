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
"""Extract fMRI BOLD file features from the BOLD data files contained in the
records of each dataset in the input directory. The features computed include
the number of time points of the volume.
"""

import argparse
import ast
import gzip
import io
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3  # type: ignore
import nibabel as nb
import pandas as pd
import requests
from botocore import UNSIGNED  # type: ignore
from botocore.config import Config  # type: ignore
from tqdm import tqdm

OPENNEURO_GRAPHQL_URL = "https://openneuro.org/crn/graphql"
HEADERS = {"Content-Type": "application/json"}

BUCKET = "openneuro.org"
NBYTES = 512
BYTE_RANGE = f"bytes=0-{NBYTES}"

VOLS = "vols"

s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))


def get_nii_timepoints_s3(filename):
    response = s3.get_object(Bucket=BUCKET, Key=filename, Range=BYTE_RANGE)
    data = response["Body"].read()

    with gzip.GzipFile(fileobj=io.BytesIO(data), mode="rb") as img:
        header = nb.Nifti1Image.from_stream(img).header
        return header["dim"][4]


def get_nii_timepoints(url):
    """Compute the number of timepoints of the file pointed by the given URL.

    Computes the number of timepoints as the size along the last dimension from
    the header of the response bitstream without actually downloading the entire
    contents if the server supports Range requests.

    Parameters
    ----------
    url : :obj:`str`
        URL where the file of interest is located.

    Returns
    -------
    :obj:`int:
        Number of timepoints.
    """

    response = requests.get(url, headers={"Range": BYTE_RANGE})
    if response.status_code not in (200, 206):
        raise RuntimeError(f"Failed to fetch byte range from URL: {response.status_code}")

    data = response.content

    with gzip.GzipFile(fileobj=io.BytesIO(data), mode="rb") as img:
        header = nb.Nifti1Image.from_stream(img).header
        return header["dim"][4]


def compute_bold_features(bold_files: dict, max_workers: int = 8) -> dict:
    """Compute BOLD run features.

    Computes the number of timepoints for all BOLD runs in each dataset.

    Parameters
    ----------
    bold_files : :obj:`dict`
        Dataset records.
    max_workers : :obj:`int`, optional
        Maximum number of parallel threads to use.

    Returns
    -------
    :obj:`~dict`
        Dataset records with BOLD features.
    """

    # for dataset_id, df in bold_files.items():
    #     # n_vols = get_nii_timepoints_s3(str(Path(dataset_id) / Path(df.iloc[0]["fullpath"])))
    #     url_list = ast.literal_eval(df.iloc[0]["urls"])
    #     url = url_list[0]
    #     n_vols = get_nii_timepoints(url)

    results: dict[str, list[pd.Series]] = {dataset_id: [] for dataset_id in bold_files}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for dataset_id, df in bold_files.items():
            for _, rec in df.iterrows():
                url = ast.literal_eval(rec["urls"])
                assert len(url) == 1
                url = url[0]
                futures[executor.submit(get_nii_timepoints, url)] = (dataset_id, rec)

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Computing BOLD timepoint counts"
        ):
            dataset_id, rec = futures[future]
            try:
                n_vols = future.result()
                rec_vols = rec.copy()
                rec_vols[VOLS] = n_vols
                results[dataset_id].append(rec_vols)
            except Exception as e:
                logging.info(f"Error processing {dataset_id}: {e}")

    return results


def filter_nonbold_records(fname: str) -> pd.DataFrame:
    """Keep records where 'filename' matches BOLD naming.

    Keeps records where 'filename' ends with '_bold.nii.gz'.

    Parameters
    ----------
    fname : :obj:`str`
        Filename.

    Returns
    -------
    :obj:`~pd.DataFrame`
        BOLD file records.
    """

    df = pd.read_csv(fname, sep="\t")
    return df[df["filename"].apply(lambda fn: bool(re.search(r"_bold\.nii\.gz$", fn)))]


def identify_bold_files(datasets: dict, max_workers: int = 8) -> dict:
    """Identify dataset BOLD files.

    For each dataset, keeps records where 'filename' ends with '_bold.nii.gz'.

    Parameters
    ----------
    datasets : :obj:`dict`
        Dataset file information.
    max_workers : :obj:`int`, optional
        Maximum number of parallel threads to use.

    Returns
    -------
    results : :obj:`dict`
        Dictionary of dataset BOLD files.
    """

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(filter_nonbold_records, val): key for key, val in datasets.items()
        }

        results = {}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Filtering BOLD files"):
            key = futures[future]
            results[key] = future.result()

    return dict(sorted(results.items()))


def write_dataset_file_lists(file_dict: dict, dirname: Path) -> None:
    """Write each dataset's list of files to a TSV file.

    Writes each file list as a TSV named <dataset_id>.tsv, and uses dict keys as
    columns. Skips entries with empty lists.

    Parameters
    ----------
    file_dict: :obj:`dict`
        A mapping from dataset ID to a list of file metadata dicts.
    dirname : :obj:`Path`
        Directory where TSV files will be written.
    """

    for dataset_id, file_list in file_dict.items():
        if not file_list:
            continue

        df = pd.DataFrame(file_list)
        df.fillna("NA", inplace=True)
        tsv_path = Path.joinpath(dirname, f"{dataset_id}.tsv")
        df.to_csv(tsv_path, sep="\t", index=False)


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
    parser.add_argument("out_dirname", type=Path, help="Output dirname")

    return parser


def _parse_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    return parser.parse_args()


def main() -> None:
    parser = _build_arg_parser()
    args = _parse_args(parser)

    _configure_logging(args.out_dirname)

    start = time.time()

    # Consider only files that have the "ds\d{6}\.tsv" pattern (e.g.
    # ds000006.tsv, ds000021.tsv, etc.)
    datasets = {
        entry.stem: entry
        for entry in args.in_dirname.iterdir()
        if entry.is_file() and re.fullmatch(r"ds\d{6}\.tsv", entry.name)
    }

    logging.info(f"Characterizing {len(datasets)} datasets...")

    # Cap at 32 to prevent overcommitting in high-core systems
    max_workers = min(32, os.cpu_count() or 1)
    bold_files = identify_bold_files(datasets, max_workers=max_workers)

    logging.info(f"Found {sum([len(item) for item in bold_files.values()])} BOLD runs.")

    fmri_bold_vols = compute_bold_features(bold_files)

    end = time.time()
    duration = end - start

    logging.info(
        f"Characterized {sum([len(item) for item in bold_files.values()])} BOLD runs in {duration:.2f} seconds."
    )

    write_dataset_file_lists(fmri_bold_vols, args.out_dirname)


if __name__ == "__main__":
    main()
