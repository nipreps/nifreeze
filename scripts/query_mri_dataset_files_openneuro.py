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
Query OpenNeuro for human (f)MRI datasets' files using IDs read from the input file.

Only those datasets having 'human' in the species field are kept.
Any dataset having one of {'bold', 'fmri', 'mri'} in the 'modality' field
is considered an fMRI dataset. For each queried dataset, the list of files is
stored in a TSV file, along with the 'id', 'filename', 'size', 'directory',
'annexed', 'key', 'urls', and 'fullpath' features.

Examples
--------
  $ query_mri_dataset_files_openneuro.py \
     openneuro_datasets.tsv \
     dataset_files


"""

import argparse
import ast
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

OPENNEURO_GRAPHQL_URL = "https://openneuro.org/crn/graphql"
HEADERS = {"Content-Type": "application/json"}

DATASETID = "datasetid"
DIRECTORY = "directory"
FILENAME = "filename"
FULLPATH = "fullpath"
ID = "id"
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


def post_with_retry(
    url: str, headers: dict, payload: dict, retries: int = 5, backoff: float = 1.5
) -> requests.Response | None:
    """Post an HTTP request with retrying.

    If the request is unsuccessful, retry ``retries`` times after an exponential
    wait time computed as :math:`backoff^attempt`.

    Parameters
    ----------
    url : :obj:`str`
        URL to post to.
    headers : :obj:`dict`
        HTTP headers.
    payload : :obj:`dict`
        HTTP payload.
    retries : :obj:`int`, optional
        Number of retry attempts.
    backoff : :obj:`float`, optional
        Retry delay.

    Returns
    -------
    :obj:`requests.Response` or None
        Request response. ``None`` if attempts failed.
    """

    for attempt in range(retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            return response
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            if status == 502 and attempt < retries - 1:
                wait = backoff**attempt
                logging.warning(f"502 Bad Gateway, retrying in {wait:.1f}s...")
                time.sleep(wait)
            else:
                logging.warning(f"HTTPError for {url}: {e}")
                return None
        except requests.exceptions.SSLError as e:
            logging.warning(f"SSLError for {url}: {e}")
            return None
        except requests.exceptions.RequestException as e:
            logging.warning(f"RequestException for {url}: {e}")
            return None
        except Exception as e:
            logging.warning(f"Request failed for {url}: {e}")
            return None

    return None


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
    response = post_with_retry(
        OPENNEURO_GRAPHQL_URL, HEADERS, {"query": query, "variables": variables}
    )

    # Ensure that the JSON response object contains all required keys
    if response is None:
        logging.warning(f"Empty response for {dataset_id}:{snapshot_tag}")
        return []

    json_response = response.json()
    snapshot = json_response.get("data", {}).get("snapshot")

    if snapshot is None:
        logging.warning(f"No snapshot returned for {dataset_id}:{snapshot_tag}")
        return []

    return snapshot.get("files", []) or []


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
    except Exception as e:
        logging.warning(f"Failed to query {dataset_id}:{snapshot_tag} at tree {tree}: {e}")
        return []

    for f in files:
        current_path = f"{parent_path}/{f[FILENAME]}".lstrip("/")
        if f[DIRECTORY]:
            sub_files = query_snapshot_tree(
                dataset_id, snapshot_tag, f[ID], parent_path=current_path
            )
            all_files.extend(sub_files)
        else:
            f[FULLPATH] = current_path
            all_files.append(f)

    return all_files


def query_dataset_files(dataset_id: str, snapshot_tag: str) -> list:
    """Retrieve all files for a given OpenNeuro dataset snapshot.

    This function takes a dataset metadata dictionary (typically a row from a
    :obj:`~pd.DataFrame`), extracts the dataset ID and snapshot tag, and
    recursively queries all files in the snapshot. If the snapshot tag is
    missing or the request fails, an empty list is returned.

    Parameters
    ----------
    dataset_id : :obj:`str`
        Dataset ID (e.g., 'ds000001').
    snapshot_tag : :obj:`str`
        Snapshot tag (e.g., '1.0.0').

    Returns
    -------
    :obj:`list`
        List of files containing their the metadata dictionaries, each including
        the fields 'id', 'filename', 'size', 'directory', 'annexed', 'key',
        'urls', and 'fullpath'.

    Notes
    -----
    - If 'tag' is missing or marked as ``NA``, no files are returned.
    - Errors during querying are caught and logged, returning an empty list.
    """

    if not snapshot_tag or snapshot_tag == "NA":
        logging.warning(f"Snapshot empty for {dataset_id}")
        return []

    try:
        files = query_snapshot_tree(dataset_id, snapshot_tag)
    except Exception as e:
        logging.warning(f"Post request error for {dataset_id}:{snapshot_tag}: {e}")
        return []

    return files


def query_datasets(df: pd.DataFrame, max_workers: int = 8) -> tuple:
    """Perform file queries over a DataFrame of datasets.

    Parameters
    ----------
    df : :obj:`~pd.DataFrame`
        Dataset records.
    max_workers : :obj:`int`, optional
        Maximum number of parallel threads to use.

    Returns
    -------
    :obj:`tuple`
        A mapping from dataset ID to list of file metadata dictionaries, and a
        list of failed dataset ID and snapshot tags.
    """

    success_results = {}
    failure_results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(query_dataset_files, row[ID], row[TAG]): (row[ID], row[TAG])
            for _, row in df.iterrows()
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing datasets"):
            dataset_id, snapshot_tag = futures[future]
            try:
                result = future.result(timeout=20)
                if result:
                    success_results[dataset_id] = [
                        {DATASETID: dataset_id, TAG: snapshot_tag} | file for file in result
                    ]
                else:
                    logging.warning(f"Empty result for {dataset_id}:{snapshot_tag}")
                    failure_results.append({DATASETID: dataset_id, TAG: snapshot_tag})
            except TimeoutError:
                logging.info(f"Timeout for {dataset_id}:{snapshot_tag}")
                failure_results.append({DATASETID: dataset_id, TAG: snapshot_tag})
            except Exception as e:
                logging.info(f"Failed to process {dataset_id}:{snapshot_tag}: {e}")
                failure_results.append({DATASETID: dataset_id, TAG: snapshot_tag})

    # Sort results before returning
    return {
        k: sorted(v, key=lambda s: s[FULLPATH]) for k, v in sorted(success_results.items())
    }, sorted(failure_results, key=lambda x: (x[DATASETID], x[TAG]))


def write_dataset_file_lists(file_dict: dict, dirname: Path, sep: str) -> None:
    """Write each dataset's list of files to a TSV file.

    Writes each file list as a TSV named <dataset_id>.tsv, and uses dict keys as
    columns. Skips entries with empty lists.

    Parameters
    ----------
    file_dict : :obj:`dict`
        A mapping from dataset ID to a list of file metadata dicts.
    dirname : :obj:`Path`
        Directory where TSV files will be written.
    sep : :obj:`str`
        Separator.
    """

    for dataset_id, file_list in file_dict.items():
        if not file_list:
            continue

        df = pd.DataFrame(file_list)
        df.fillna("NA", inplace=True)
        tsv_path = Path.joinpath(dirname, f"{dataset_id}.tsv")
        df.to_csv(tsv_path, sep=sep, index=False)


def write_dataset_tags(dataset_tags: list, fname: Path, sep: str) -> None:
    """Write dataset tag dictionaries to a TSV file.

    Parameters
    ----------
    dataset_tags : :obj:`list`
        Dictionaries of dataset ID and snapshot tags.
    fname : :obj:`Path`
        Filename.
    sep : :obj:`str`
        Separator.
    """

    df = pd.DataFrame(dataset_tags)
    df.to_csv(fname, sep=sep, index=False)


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
    # from being stripped. Similarly, keep the "NA" values empty as otherwise
    # pandas loads them as "NaN" which is not considered a string but a number
    # and causes issues downstream when trying to compare values for sorting
    # results.
    _df = pd.read_csv(
        args.dataset_fname, sep=sep, dtype={TAG: str}, keep_default_na=False, na_values=[""]
    )

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
    success_results, failed_results = query_datasets(df, max_workers=max_workers)

    end = time.time()
    duration = end - start

    logging.info(
        f"Queried {len(success_results) + len(failed_results)} datasets in {duration:.2f} seconds."
    )
    logging.info(f"{len(success_results)} queries succeeded.")
    logging.info(f"{len(failed_results)} queries failed.")

    # Serialize
    write_dataset_file_lists(success_results, args.out_dirname, sep)
    failed_datasets_info_fname = Path.joinpath(args.out_dirname, "failed_dataset_tag_queries.tsv")
    write_dataset_tags(failed_results, failed_datasets_info_fname, sep)


if __name__ == "__main__":
    main()
