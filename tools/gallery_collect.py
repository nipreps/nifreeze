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
"""Collect the parallel gallery fit-job fragments into one rendered gallery.

Each fit job (one ``(dataset × model × mode)`` cell) uploads an artifact holding
its slice of the output directory: a ``gallery_manifest.json`` fragment and the
rendered ``<dataset>/<panel>.png`` figures. This script folds every downloaded
fragment back into a single output directory — one merged manifest plus the full
panel tree — so the notebooks can embed the figures (``NIFREEZE_GALLERY_OUT``)
without re-fitting anything.

Run as ``python tools/gallery_collect.py --staging <dir> --out <dir>``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

SPHINXEXT = Path(__file__).resolve().parent.parent / "docs" / "sphinxext"
sys.path.insert(0, str(SPHINXEXT))

from gallery.datasets import DATASETS  # noqa: E402
from gallery.manifest import (  # noqa: E402  (after sys.path plumbing)
    STATUS_ERROR,
    CellResult,
    GalleryManifest,
)
from gallery.pages import write_pages  # noqa: E402

NO_OUTPUT_REASON = "no output produced (fit job failed, timed out, or was cancelled)"
"""Reason recorded for an expected cell that never reported back."""


def reconcile(manifest: GalleryManifest, expected: list[dict[str, str]]) -> int:
    """Record every expected cell that produced no fragment as an error.

    A fit job killed by a timeout or OOM uploads nothing, so its cell would
    otherwise vanish from the coverage table — reading as though it was never
    part of the matrix. The manifest exists to say what was exercised, so an
    absent cell is reported as a failure rather than silently dropped.
    """
    seen = {(c.dataset, c.model, c.mode) for c in manifest.cells}
    missing = [c for c in expected if (c["dataset"], c["model"], c["mode"]) not in seen]
    for cell in missing:
        manifest.cells.append(
            CellResult(
                dataset=cell["dataset"],
                scheme=cell.get("scheme", "?"),
                model=cell["model"],
                mode=cell["mode"],
                status=STATUS_ERROR,
                reason=NO_OUTPUT_REASON,
            )
        )
    manifest.cells.sort(key=lambda c: (c.dataset, c.model, c.mode))
    return len(missing)


def collect(
    staging: Path,
    out: Path,
    expected: list[dict[str, str]] | None = None,
    pages_dir: Path | None = None,
) -> GalleryManifest:
    """Merge manifest fragments and panels from ``staging`` into ``out``."""
    out.mkdir(parents=True, exist_ok=True)

    # 1. Merge every manifest fragment (recursively, across per-artifact subdirs).
    manifest = GalleryManifest.from_tree(staging)

    # 2. Account for cells that never reported back at all.
    n_missing = reconcile(manifest, expected or [])
    if n_missing:
        print(f"WARNING: {n_missing} expected cell(s) produced no output.", flush=True)

    manifest.to_json(out / "gallery_manifest.json")
    (out / "coverage.rst").write_text(manifest.coverage_table_rst())

    # 3. Reassemble the panel tree: each PNG lives under ``<dataset>/<file>.png``
    #    inside its artifact; copy it to the same-named dataset dir under ``out``.
    n_panels = 0
    for png in sorted(staging.rglob("*.png")):
        dest = out / png.parent.name / png.name
        # Local runs pass --staging == --out, where the panels are already in place.
        if png.resolve() == dest.resolve():
            n_panels += 1
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(png, dest)
        n_panels += 1

    # 4. Render the documentation pages over that output (plain rST + images).
    #    Every registered dataset gets a page, whether or not it reported cells:
    #    the toctree references them all, so a missing page breaks the docs build.
    if pages_dir is not None:
        pages = write_pages(
            manifest, pages_dir, panels_from=out, datasets=[d.name for d in DATASETS]
        )
        print(f"Wrote {len(pages)} gallery page(s) to {pages_dir}.", flush=True)

    counts = manifest.counts()
    print(
        f"Collected {len(manifest.cells)} cells "
        f"({counts['ran']} ran, {counts['skipped']} skipped, {counts['error']} errored) "
        f"and {n_panels} panels into {out}.",
        flush=True,
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--staging",
        type=Path,
        required=True,
        help="Directory holding the downloaded per-cell artifacts.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for the merged manifest and panel tree.",
    )
    parser.add_argument(
        "--expect",
        type=Path,
        default=None,
        help=(
            "JSON file listing the expected (dataset, model, mode) cells; any that "
            "produced no output are recorded as errors instead of being dropped."
        ),
    )
    parser.add_argument(
        "--pages",
        type=Path,
        default=None,
        help="Write the per-dataset reStructuredText pages (and panels) into this directory.",
    )
    args = parser.parse_args(argv)
    expected = json.loads(args.expect.read_text()) if args.expect else None
    collect(args.staging, args.out, expected, args.pages)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
