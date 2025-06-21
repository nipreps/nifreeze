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
"""Query OpenNeuro human MRI dataset files using the dataset IDs read from the
input file. Only those datasets having 'human` in the species field are kept.
Any dataset having one of {'bold', 'fmri', 'mri'} in the 'modality' field is
considered as an fMRI dataset. For each queried dataset, the list of files is
stored to a TSV file, along with the 'id', 'filename', 'size', 'directory',
'annexed', 'key', 'urls', and 'fullpath' features.
"""

import argparse
import ast
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

OPENNEURO_GRAPHQL_URL = "https://openneuro.org/crn/graphql"
HEADERS = {"Content-Type": "application/json"}

MODALITIES = "modalities"
SPECIES = "species"
TAG = "tag"

HUMAN_SPECIES = {"human"}
FMRI_MODALITIES = {"bold", "fmri", "mri"}


def filter_nonhuman_datasets(df: pd.DataFrame) -> pd.Series:
    """Filter non-human data records.

    Filters datasets whose 'species' field does not contain one of
    `HUMAN_SPECIES`.

    Parameters
    ----------
    df : :obj:`~pd.DataFrame`
        Dataset records.

    Returns
    -------
    `~pd.Series`
        Mask of human datasets.
    """

    return df[SPECIES].str.lower().isin(HUMAN_SPECIES)


def filter_nonmri_datasets(df: pd.DataFrame) -> pd.Series:
    """Filter non-MRI data records.

    Filters datasets whose 'modalities' field does not contain one of
    `FMRI_MODALITIES`.

    Parameters
    ----------
    df : :obj:`~pd.DataFrame`
        Dataset records.

    Returns
    -------
    `~pd.Series`
        Mask of MRI datasets.
    """

    return df[MODALITIES].apply(
        lambda x: any(item.lower() in FMRI_MODALITIES for item in ast.literal_eval(x))
        if isinstance(x, str) and x.startswith("[")
        else False
    )


def filter_nonrelevant_datasets(df: pd.DataFrame) -> pd.DataFrame:
    """Filter non-human and non-MRI data records.

    The 'species' field has to contain 'human' and the 'modalities' field has to
    contain one of :obj:`FMRI_MODALITIES`.

    Parameters
    ----------
    df : :obj:`~pd.DataFrame`
        Dataset records.

    Returns
    -------
    `~pd.DataFrame`
        Human MRI dataset records.
    """

    species_mask = filter_nonhuman_datasets(df)
    modality_mask = filter_nonmri_datasets(df)

    logging.info(f"Found {sum(~species_mask)}/{len(df)} non-human datasets.")
    logging.info(f"Found {sum(~modality_mask)}/{len(df)} non-MRI datasets.")

    return df[species_mask & modality_mask]


def query_snapshot_files(dataset_id: str, snapshot_tag: str, tree: str | None = None) -> list:
    """Query the list of files at a specific level of a dataset snapshot.

    Parameters
    ----------
    dataset_id : :obj:`str`
        The OpenNeuro dataset ID (e.g., 'ds000001').
    snapshot_tag : :obj:`str`
        The tag of the snapshot to query (e.g., '1.0.0').
    tree : :obj:`str`, optional
        ID of a directory within the snapshot tree to query; use ``None`` to
        start at the root.

    Returns
    -------
    :obj:`list`
        Each dict represents a file or directory with fields 'id', 'filename',
        'size', 'directory', 'annexed', 'key', and 'urls'.
    """

    query = """
    query getSnapshotFiles($datasetId: ID!, $tag: String!, $tree: String) {
      snapshot(datasetId: $datasetId, tag: $tag) {
        files(tree: $tree) {
          id
          filename
          size
          directory
          annexed
          key
          urls
        }
      }
    }
    """

    variables = {"datasetId": dataset_id, "tag": snapshot_tag, "tree": tree}
    response = requests.post(
        OPENNEURO_GRAPHQL_URL,
        headers=HEADERS,
        json={"query": query, "variables": variables},
    )
    response.raise_for_status()
    return response.json()["data"]["snapshot"]["files"]


def query_snapshot_tree(
    dataset_id: str, snapshot_tag: str, tree: str | None = None, parent_path=""
) -> list:
    """Recursively query all files in an OpenNeuro dataset snapshot.

    Parameters
    ----------
    dataset_id : :obj:`str`
        The OpenNeuro dataset ID (e.g., 'ds000001').
    snapshot_tag : :obj:`str`
        The tag of the snapshot to query (e.g., '1.0.0').
    tree : :obj:`str`, optional
        ID of a directory within the snapshot tree to query; use ``None`` to
        start at the root.
    parent_path : :obj:`str`, optional
        Relative path used to construct full file paths (used during recursion).

    Returns
    -------
    all_files : :obj:`list`
        List of all file entries (not directories), each including a 'fullpath'
        key that shows the complete path from the root.
    """

    all_files = []

    try:
        files = query_snapshot_files(dataset_id, snapshot_tag, tree)
    except requests.HTTPError as e:
        logging.info(f"Error querying {dataset_id}:{snapshot_tag} at tree {tree}: {e}")
        return []

    for f in files:
        current_path = f"{parent_path}/{f['filename']}".lstrip("/")
        if f["directory"]:
            sub_files = query_snapshot_tree(
                dataset_id, snapshot_tag, f["id"], parent_path=current_path
            )
            all_files.extend(sub_files)
        else:
            f["fullpath"] = current_path
            all_files.append(f)

    return all_files


def query_dataset_files(ds):
    """Retrieve all files for a given OpenNeuro dataset snapshot.

    This function takes a dataset metadata dictionary (typically a row from a
    :obj:`~pd.DataFrame`), extracts the dataset ID and snapshot tag, and
    recursively queries all files in the snapshot. If the snapshot tag is
    missing or the request fails, an empty list is returned.

    Parameters
    ----------
    ds : :obj:`~pd.Series`
        A data series containing at least the keys:
            - 'id': Dataset ID (e.g., "ds000001")
            - 'tag': Snapshot tag (e.g., "1.0.0")

    Returns
    -------
    obj:`tuple`
        A `str` with the dataset ID and a :obj:`list` with the metadata
        dictionaries, each including the fields 'id', 'filename', 'size',
        'directory', 'annexed', 'key', 'urls', and 'fullpath'.

    Notes
    -----
    - If 'tag' is missing or marked as "NA", no files are returned.
    - Errors during querying are caught and logged, returning an empty file list.
    """

    ds_id = ds["id"]
    snapshot_tag = ds["tag"]

    if not snapshot_tag or snapshot_tag == "NA":
        return ds_id, []

    try:
        files = query_snapshot_tree(ds_id, snapshot_tag)
        return ds_id, files
    except Exception as e:
        logging.info(f"Failed to process {ds_id}: {e}")
        return ds_id, []


def query_datasets(df, max_workers=8):
    """Perform file queries over a DataFrame of datasets.

    Parameters
    ----------
    df : :obj:`~pd.DataFrame`
        Dataset records.
    max_workers : :obj:`int`, optional
        Maximum number of parallel threads to use.

    Returns
    -------
    results : :obj:`dict`
        Mapping from dataset ID to list of file metadata dictionaries.
    """

    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(query_dataset_files, row): row["id"] for _, row in df.iterrows()
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing datasets"):
            ds_id, files = future.result()
            results[ds_id] = files

    return results


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
    parser.add_argument("dataset_fname", type=Path, help="Dataset list filename (*.TSV)")
    parser.add_argument("out_dirname", type=Path, help="Output dirname")

    return parser


def _parse_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    return parser.parse_args()


def main() -> None:
    parser = _build_arg_parser()
    args = _parse_args(parser)

    _configure_logging(args.out_dirname)

    logging.info(
        "Script called with arguments:\n" + "\n".join(f"  {k}: {v}" for k, v in vars(args).items())
    )

    logging.info(f"Querying {OPENNEURO_GRAPHQL_URL}...")

    sep = "\t"
    start = time.time()

    # Ensure that the tag column is read as a string to prevent leading zeros
    # from being stripped
    _df = pd.read_csv(args.dataset_fname, sep=sep, dtype={TAG: str})

    logging.info(f"Querying {len(_df)} datasets...")

    # Filter nonrelevant datasets
    df = filter_nonrelevant_datasets(_df)

    logging.info(f"Filtered {len(_df) - len(df)}/{len(_df)} non-human, non-MRI datasets.")

    mri_datasets_fname = Path.joinpath(
        args.out_dirname,
        args.dataset_fname.with_name(args.dataset_fname.stem + "_mri" + args.dataset_fname.suffix),
    )
    df.to_csv(mri_datasets_fname, sep=sep, index=False)

    # Cap at 32 to prevent overcommitting in high-core systems
    max_workers = min(32, os.cpu_count() or 1)
    datasets_files = query_datasets(df, max_workers=max_workers)

    end = time.time()
    duration = end - start

    logging.info(f"Queried {len(datasets_files)} datasets in {duration:.2f} seconds.")

    # Serialize
    write_dataset_file_lists(datasets_files, args.out_dirname)


if __name__ == "__main__":
    main()
