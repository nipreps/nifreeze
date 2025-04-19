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
"""Select fMRI data from OpenNeuro based on a seed value and the following
constraints:
- A total of 4000 runs across datasets
- No single dataset contributes to more than 5% of the total number of runs (200 runs)
- BOLD run has between 300 and 1200 timepoints
"""

import argparse
import logging
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import requests
from datalad import api as dl  # type: ignore

# from concurrent.futures import ThreadPoolExecutor, as_completed

OPENNEURO_DATASETS_URL = "https://github.com/OpenNeuroDataset"
OPENNEURO_GRAPHQL_URL = "https://openneuro.org/crn/graphql"
OPENNEURO_REST_URL = "https://openneuro.org/crn/datasets"
HEADERS = {"Content-Type": "application/json"}

INDIVIDUAL_DS_CONTR = 0.05
"""Allowed contribution threshold for runs per dataset."""

MIN_TIMEPOINTS = 400
"""Minimum number of BOLD timepoints per dataset."""

MAX_TIMEPOINTS = 1200
"""Maximum number BOLD timepoints per dataset."""

TOTAL_RUNS = 4000
"""Number of total runs."""

MAX_QUERY_SIZE = 2
"""Maximum number of datasets to be queried."""

MAX_WORKERS = 8
"""Maximum number of workers."""


logging.basicConfig(
    filename=f"{Path(__file__).stem}",
    level=logging.INFO,
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def query_datasets(seed):
    # ToDo
    # Do something with the seed here or else do it later: get all datasets
    # and select 4000 randomly using the seed

    # ToDo
    # Note that some (many) datasets have an empty string in the species. Do we
    # use only humans? Discard all those that are not humans?
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
    variables = {
        "after": None,
        "first": MAX_QUERY_SIZE,
    }

    response = requests.post(
        OPENNEURO_GRAPHQL_URL, headers=HEADERS, json={"query": query, "variables": variables}
    )
    response.raise_for_status()
    return response.json()["data"]["datasets"]["edges"]


def query_snapshot_files(dataset_id, snapshot_tag, tree=None):
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


def query_snapshot_tree(dataset_id, snapshot_tag, tree=None, parent_path=""):
    all_files = []
    try:
        files = query_snapshot_files(dataset_id, snapshot_tag, tree)
    except requests.HTTPError as e:
        print(f"Error querying {dataset_id}:{snapshot_tag} at tree {tree}: {e}")
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


def filter_nonrelevant_files(files):
    # Keep only files that match BOLD naming
    bold_files = [f for f in files if re.search(r"_bold\.nii\.gz$", f["filename"])]
    return bold_files


def get_json_sidecar_info(files, bold_filename):
    # ToDo
    # Fetch this with datalad
    json_filename = re.sub(r"_bold\.nii\.gz$", "_bold.json", bold_filename)
    json_file = next((f for f in files if f["filename"].endswith(json_filename)), None)
    if json_file:
        try:
            resp = requests.get(json_file["urls"][0])
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
    return {}


def build_dataset_url(ds_id):
    return f"{OPENNEURO_DATASETS_URL}/{ds_id}.git"


def uninstall_dataset(dataset_path):
    # Uninstall the dataset (removes from git-annex and subdatasets if any)
    ds_dl = dl.Dataset(dataset_path)
    ds_dl.uninstall(path=dataset_path, recursive=True)


def clone_dataset(dataset_url, ds_path, snapshot_tag=None):
    ds = dl.install(source=dataset_url, path=ds_path)

    # Checkout a specific snapshot
    if snapshot_tag:
        ds.repo.checkout(snapshot_tag)

    return ds


def install_dataset(ds_id, ds_path, snapshot_tag=None):
    dataset_url = build_dataset_url(ds_id)

    if ds_path.exists():
        logging.info(f"Path already exists, loading existing dataset: {ds_path}")
        try:
            # Attempt to load the existing dataset
            dl_ds = dl.Dataset(ds_path)
            if not dl_ds.is_installed():
                logging.info(f"Dataset at {ds_path} is not installed correctly, reinstalling...")
                return clone_dataset(dataset_url, ds_path, snapshot_tag=snapshot_tag)
            else:
                return dl_ds
        except Exception as e:
            logging.info(f"Error loading dataset at {ds_path}: {e}")
            return None
    else:
        return clone_dataset(dataset_url, ds_path, snapshot_tag=snapshot_tag)


def get_nii_timepoints(filename):
    img = nib.load(filename)
    shape = img.shape
    assert len(shape) == 4
    return shape[-1]


def estimate_runs_from_json(json_data):
    try:
        if "RepetitionTime" in json_data and "AcquisitionDuration" in json_data:
            tr = float(json_data["RepetitionTime"])  # in seconds
            duration = float(json_data["AcquisitionDuration"])  # in seconds
            return int(round(duration / tr))
    except Exception:
        pass
    return None


def filter_nonrelevant_datasets(datasets):
    return [
        ds for ds in datasets if "MRI" in ds["node"]["latestSnapshot"]["summary"]["modalities"]
    ]


def process_dataset(ds, out_path):
    results = []

    contrib_thr = int(INDIVIDUAL_DS_CONTR * TOTAL_RUNS)

    ds_id = ds["node"]["id"]
    snapshot = ds["node"]["latestSnapshot"]
    snapshot_tag = snapshot["tag"]

    # ToDo
    # Check len(tasks) > 1 ??

    logging.info(f"Considering dataset {ds_id}")

    files = query_snapshot_tree(ds_id, snapshot_tag)
    bold_files = filter_nonrelevant_files(files)
    # ToDo
    # If the number of runs exceeds a percentage of the target number of
    # runs, pick only a fraction: First 200, randomly or what?
    total_runs = len(bold_files)
    if total_runs > contrib_thr:
        logging.info(f"Dataset {ds_id} has more than {contrib_thr} runs. Picking Y")

        idx = rng.random(contrib_thr, total_runs)

        logging.info(f"Indices: {idx}")

        bold_files = list(bold_files[idx])

        logging.info(f"Kept: {bold_files}")

    total_runs = len(bold_files)
    logging.info(f"Number of runs {total_runs}")

    ds_path = out_path / ds_id
    dl_ds = install_dataset(ds_id, ds_path, snapshot_tag=snapshot_tag)

    if dl_ds is None:
        return results

    for f in bold_files:
        # File to fetch (relative path in dataset)
        bold_fname = ds_path / f["fullpath"]
        # Get the file (download if needed)
        dl_ds.get(bold_fname)
        n_vols = get_nii_timepoints(bold_fname)  # url)
        if n_vols is None:
            logging.info("Timepoint information could not be found: computing using TR")

            # ToDo
            # Test this
            json_info = get_json_sidecar_info(files, f["filename"])
            n_vols = estimate_runs_from_json(json_info)

        logging.info(f"Number of timepoints: {n_vols}")

        if n_vols and MIN_TIMEPOINTS <= n_vols <= MAX_TIMEPOINTS:
            results.append(
                {
                    "id": ds_id,
                    "name": ds["name"],
                    "dataset_label": ds["label"],
                    "snapshot_tag": snapshot_tag,
                    "bold_file": f["filename"],
                    "n_timepoints": n_vols,
                }
            )
            # ToDo
            # Think if this is OK
            break  # Report only one BOLD file per snapshot
        else:
            logging.info(
                f"Not within required boundaries ({MIN_TIMEPOINTS}, {MAX_TIMEPOINTS}). Discarding run"
            )

    # Remove dataset at this point
    uninstall_dataset(ds_path)

    return results


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("seed", type=int, help="Random seed. Format (YYYYMMDD")
    parser.add_argument("out_fname", type=Path, help="Output data list filename (*.TSV)")
    parser.add_argument("out_ds_dirname", type=Path, help="Output dataset dirname")
    parser.add_argument("--total_runs", type=int, default=TOTAL_RUNS, help="Number of total runs")
    parser.add_argument(
        "--contr_thr",
        type=float,
        default=INDIVIDUAL_DS_CONTR,
        help="Allowed contribution threshold for runs per dataset",
    )
    parser.add_argument(
        "--min_timepoints_thr",
        type=int,
        default=MIN_TIMEPOINTS,
        help="Minimum number of BOLD timepoints per dataset",
    )
    parser.add_argument(
        "--max_timepoints_thr",
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

    rng = np.random.default_rng(args.seed)

    # Get the dataset information
    datasets = query_datasets(rng)

    # Filter datasets: rely on the "modality" property
    mri_datasets = filter_nonrelevant_datasets(datasets)

    all_results: list[dict] = []
    futures = [process_dataset(ds, args.out_ds_dirname) for ds in mri_datasets]

    # with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    #     futures = [executor.submit(process_dataset, ds) for ds in mri_datasets]
    #     for future in as_completed(futures):
    #         all_results.extend(future.result())

    # If the number of total runs is lower than the minimum required, log and exit
    runs = sum(map(lambda x: x["n_timepoints"], all_results))
    if runs < TOTAL_RUNS:
        logging.info(
            f"{runs} total runs found. Please try querying more files to exceed the {TOTAL_RUNS} threshold."
        )
        return

    # Save to CSV
    columns = ["dataset_label", "snapshot_tag", "bold_file", "n_timepoints"]
    df = pd.DataFrame(all_results, columns=columns)
    df.to_csv(args.output_scores, sep="\t", index=False)


if __name__ == "__main__":
    main()
