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
Query existing dataset information from OpenNeuro.

The 'id', 'name', 'species', 'tag', 'dataset_doi', 'modalities', and 'tasks' features of the
available datasets are stored in a TSV file.

Example
-------
  $ query_datasets_openneuro.py \
     openneuro_datasets.tsv
"""

import argparse
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

MAX_QUERY_SIZE = 100
"""Maximum page size."""

DATASET_DOI = "DatasetDOI"
ID = "id"
MODALITIES = "modalities"
NAME = "name"
SPECIES = "species"
TAG = "tag"
TASKS = "tasks"


def fetch_page(after_cursor: str | None = None) -> dict:
    """Fetch a single page of OpenNeuro datasets using GraphQL.

    Parameters
    ----------
    after_cursor: str, optional
        The pagination cursor indicating where to start. If ``None``, fetches
        the first page.

    Returns
    -------
    :obj:`dict`
         Dictionary with keys 'edges' (list of datasets) and 'pageInfo'
         (pagination metadata).
    """

    query = """
    query DatasetsWithLatestSnapshots($after: String, $first: Int!) {
      datasets(first: $first, after: $after, orderBy: { created: ascending }) {
        edges {
          node {
            id
            name
            metadata {
              species
            }
            latestSnapshot {
              tag
              description {
                DatasetDOI
              }
              summary {
                modalities
                tasks
              }
            }
          }
        }
        pageInfo {
          endCursor
          hasNextPage
        }
      }
    }
    """

    variables = {"after": after_cursor, "first": MAX_QUERY_SIZE}
    response = requests.post(
        OPENNEURO_GRAPHQL_URL, headers=HEADERS, json={"query": query, "variables": variables}
    )
    response.raise_for_status()
    return response.json()["data"]["datasets"]


def get_cursors() -> list:
    """Serially walk through the entire OpenNeuro dataset list to collect all pagination cursors.

    This function starts from the beginning and keeps fetching pages until the
    last one, recording the 'endCursor' of each page to enable parallel fetching
    later.

    Returns
    -------
    cursors : :obj:`list`
        List of cursors, where the first cursor is ``None`` (start of list), and
        the rest are page markers returned by GraphQL.
    """

    cursors = [None]
    current_cursor = None
    with tqdm(desc="Discovering cursors", unit="page") as pbar:
        while True:
            data = fetch_page(current_cursor)
            page_info = data["pageInfo"]
            if page_info["hasNextPage"]:
                current_cursor = page_info["endCursor"]
                cursors.append(current_cursor)
                pbar.update(1)
            else:
                break
    return cursors


def fetch_pages(cursors: list, max_workers: int = 8) -> list:
    """Fetch all OpenNeuro dataset pages in parallel using a precomputed list of cursors.

    Parameters
    ----------
    cursors : :obj:`list`
        List of cursors.
    max_workers : :obj:`int`, optional
        Maximum number of parallel threads to use.

    Returns
    -------
    results : :obj:`list`
        List of datasets.
    """

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_page, cursor): cursor for cursor in cursors}
        with tqdm(total=len(futures), desc="Fetching pages", unit="page") as pbar:
            for future in as_completed(futures):
                data = future.result()
                results.extend(data["edges"])
                pbar.update(1)
    return results


def edges_to_dataframe(edges: list) -> pd.DataFrame:
    """Convert a list of dataset edges (GraphQL response) into a pandas DataFrame.

    Returned values are sorted by the dataset 'id'.

    Parameters
    ----------
    edges : :obj:`list`
        GraphQL edges. Each edge contains a 'node' with dataset metadata.

    Returns
    -------
    :obj:`~pd.DataFrame`
        A DataFrame with the relevant dataset information, namely 'id', 'name',
        'species', 'tag', 'dataset_doi', 'modalities', and 'tasks'.
    """

    rows = []
    for item in edges:
        if item is None:
            continue
        node = item["node"]
        snapshot = node.get("latestSnapshot", {})
        row = {
            ID.lower(): node.get(ID),
            NAME.lower(): node.get(NAME, None),
            SPECIES.lower(): node.get("metadata", None).get(SPECIES),
            TAG.lower(): snapshot.get(TAG),
            DATASET_DOI.lower(): snapshot.get("description", {}).get(DATASET_DOI),
            MODALITIES.lower(): snapshot.get("summary", {}).get(MODALITIES)
            if snapshot.get("summary")
            else None,
            TASKS.lower(): snapshot.get("summary", {}).get(TASKS)
            if snapshot.get("summary")
            else None,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    # Replace all empty strings by "NA"
    df.replace("", "NA", inplace=True)
    return df.fillna("NA").sort_values("id")


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
    parser.add_argument("out_fname", type=Path, help="Output data list filename (*.TSV)")

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

    logging.info(f"Querying {OPENNEURO_GRAPHQL_URL}...")

    start = time.time()

    # Precompute all cursors
    cursors = get_cursors()

    # Fetch all pages in parallel
    # Cap at 32 to prevent overcommitting in high-core systems
    max_workers = min(32, os.cpu_count() or 1)
    edges = fetch_pages(cursors, max_workers=max_workers)

    end = time.time()
    duration = end - start

    logging.info(f"Found {len(edges)} datasets in {duration:.2f} seconds.")

    # Serialize
    df = edges_to_dataframe(edges)
    df.to_csv(args.out_fname, sep="\t", index=False)


if __name__ == "__main__":
    main()
