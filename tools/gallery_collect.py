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
import shutil
import sys
from pathlib import Path

SPHINXEXT = Path(__file__).resolve().parent.parent / "docs" / "sphinxext"
sys.path.insert(0, str(SPHINXEXT))

from gallery.manifest import GalleryManifest  # noqa: E402  (after sys.path plumbing)


def collect(staging: Path, out: Path) -> GalleryManifest:
    """Merge manifest fragments and panels from ``staging`` into ``out``."""
    out.mkdir(parents=True, exist_ok=True)

    # 1. Merge every manifest fragment (recursively, across per-artifact subdirs).
    manifest = GalleryManifest.from_tree(staging)
    manifest.to_json(out / "gallery_manifest.json")
    (out / "coverage.rst").write_text(manifest.coverage_table_rst())

    # 2. Reassemble the panel tree: each PNG lives under ``<dataset>/<file>.png``
    #    inside its artifact; copy it to the same-named dataset dir under ``out``.
    n_panels = 0
    for png in sorted(staging.rglob("*.png")):
        dest_dir = out / png.parent.name
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(png, dest_dir / png.name)
        n_panels += 1

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
    args = parser.parse_args(argv)
    collect(args.staging, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
