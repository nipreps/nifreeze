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
"""Fetch the prediction gallery pages published by the ``gallery`` workflow.

The gallery pages are not tracked: the ``gallery`` workflow uploads them (one
``<dataset>.rst`` page plus its ``<dataset>/*.png`` panels) as the
``gallery-pages`` artifact, and the docs build downloads them with this script.
Runs triggered by a push or a manual dispatch publish; scheduled runs are only a
fallback, used once the newer artifacts have expired (after 30 days).

Run as ``python tools/fetch_gallery.py [--dest docs/gallery] [--strict]``. It needs
the GitHub CLI with a token (``GH_TOKEN`` on CI, ``gh auth login`` locally).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SPHINXEXT = Path(__file__).resolve().parent.parent / "docs" / "sphinxext"
sys.path.insert(0, str(SPHINXEXT))

from gallery.datasets import DATASETS  # noqa: E402  (after sys.path plumbing)
from gallery.pages import PROVENANCE_FILE  # noqa: E402

WORKFLOW = "gallery.yml"
ARTIFACT = "gallery-pages"
PUBLISHING_EVENTS = ("push", "workflow_dispatch")
"""Gallery runs that publish; scheduled runs are a weekly check."""

RUN_FIELDS = "databaseId,event,headSha,createdAt"
FIGURE = re.compile(r"^\.\. figure:: (\S+)", re.MULTILINE)


def _gh(args: list[str]) -> str:
    """Run the GitHub CLI and return its standard output."""
    return subprocess.run(["gh", *args], check=True, capture_output=True, text=True).stdout


def list_runs(repo: str, branch: str = "main", limit: int = 30) -> list[dict]:
    """Return the successful ``gallery`` runs of ``branch``, newest first."""
    out = _gh(
        [
            "run",
            "list",
            "--repo",
            repo,
            "--workflow",
            WORKFLOW,
            "--branch",
            branch,
            "--status",
            "success",
            "--limit",
            str(limit),
            "--json",
            RUN_FIELDS,
        ]
    )
    return json.loads(out)


def view_run(repo: str, run_id: int) -> dict:
    """Return one run's metadata."""
    return json.loads(_gh(["run", "view", str(run_id), "--repo", repo, "--json", RUN_FIELDS]))


def select_runs(runs: list[dict], sha: str | None = None) -> list[dict]:
    """Order candidate runs: matching ``sha`` first, then publishing runs, newest first."""
    newest = sorted(runs, key=lambda run: run["createdAt"], reverse=True)
    return sorted(
        newest,
        key=lambda run: (
            sha is not None and run["headSha"] != sha,
            run["event"] not in PUBLISHING_EVENTS,
        ),
    )


def download(repo: str, run_id: int, dest: Path) -> bool:
    """Download the run's ``gallery-pages`` artifact into ``dest``."""
    try:
        _gh(
            [
                "run",
                "download",
                str(run_id),
                "--repo",
                repo,
                "--name",
                ARTIFACT,
                "--dir",
                str(dest),
            ]
        )
    except subprocess.CalledProcessError as exc:
        print(f"Run {run_id}: cannot download {ARTIFACT} ({exc.stderr.strip()}).", flush=True)
        return False
    return True


def validate(pages_dir: Path, datasets: list[str]) -> list[str]:
    """Return the problems that make ``pages_dir`` unfit for publishing."""
    problems = []
    n_figures = 0
    for name in datasets:
        page = pages_dir / f"{name}.rst"
        if not page.is_file():
            problems.append(f"missing page {page.name}")
            continue
        for target in FIGURE.findall(page.read_text()):
            n_figures += 1
            if not (pages_dir / target).is_file():
                problems.append(f"{page.name} embeds a missing figure {target}")
    if not n_figures:
        problems.append("no page embeds a figure")
    return problems


def install(staged: Path, dest: Path, run: dict, repo: str) -> None:
    """Replace ``dest`` with the validated pages and record their provenance."""
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(staged, dest)
    (dest / PROVENANCE_FILE).write_text(
        json.dumps(
            {
                "run_id": run["databaseId"],
                "event": run["event"],
                "head_sha": run["headSha"],
                "created_at": run["createdAt"],
                "url": f"https://github.com/{repo}/actions/runs/{run['databaseId']}",
            },
            indent=2,
        )
        + "\n"
    )


def fetch(
    dest: Path,
    repo: str,
    run_id: int | None = None,
    sha: str | None = None,
) -> dict | None:
    """Install the pages of the first run with valid pages into ``dest``; return it."""
    datasets = [d.name for d in DATASETS]
    try:
        runs = [view_run(repo, run_id)] if run_id else select_runs(list_runs(repo), sha=sha)
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"Cannot list the {WORKFLOW} runs of {repo}: {exc}", flush=True)
        return None

    for run in runs:
        with tempfile.TemporaryDirectory() as tmp:
            staged = Path(tmp) / "pages"
            if not download(repo, run["databaseId"], staged):
                continue
            if problems := validate(staged, datasets):
                print(f"Run {run['databaseId']}: skipped ({'; '.join(problems)}).", flush=True)
                continue
            install(staged, dest, run, repo)
        print(
            f"Fetched the gallery of run {run['databaseId']} ({run['event']}, "
            f"{run['headSha'][:8]}, {run['createdAt']}) into {dest}.",
            flush=True,
        )
        return run
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        type=Path,
        default=SPHINXEXT.parent / "gallery",
        help="Where to write the pages (default: docs/gallery).",
    )
    parser.add_argument(
        "--repo",
        default="nipreps/nifreeze",
        help="Repository whose gallery runs publish the pages.",
    )
    parser.add_argument("--run-id", type=int, help="Fetch this run's pages only.")
    parser.add_argument("--sha", help="Prefer a run of this commit (e.g., a release tag).")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail when no published pages can be fetched (instead of placeholders).",
    )
    args = parser.parse_args(argv)

    if fetch(args.dest, args.repo, run_id=args.run_id, sha=args.sha) is not None:
        return 0

    message = f"No valid {ARTIFACT} artifact found; dispatch {WORKFLOW} on main to publish one."
    if args.strict:
        print(f"ERROR: {message}", file=sys.stderr, flush=True)
        return 1
    print(f"WARNING: {message} The docs build will show placeholder pages.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
