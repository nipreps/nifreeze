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
"""Notebook helpers for the prediction gallery.

Keeps the documentation notebooks thin: a single call runs one dataset's slice
of the matrix and displays the coverage table plus the rendered panels inline,
so the executed notebook (built by the scheduled gallery job) embeds the images
and nbsphinx renders them without re-execution.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

from gallery.datasets import DATASETS, source_relpaths
from gallery.manifest import STATUS_RAN, GalleryManifest


def _precomputed_dir(out_dir: str | Path | None) -> Path | None:
    """Return the directory holding a pre-rendered gallery, or ``None``.

    In CI the parallel fit jobs render panels and the collect job merges their
    manifests into ``NIFREEZE_GALLERY_OUT/gallery_manifest.json``; when that
    exists the notebook only *embeds* the stored figures instead of re-fitting.
    Falls back to live computation (``None``) for local, no-data use.
    """
    candidate = out_dir if out_dir is not None else os.environ.get("NIFREEZE_GALLERY_OUT")
    if candidate is None:
        return None
    root = Path(candidate)
    return root if (root / "gallery_manifest.json").is_file() else None


def show_dataset(
    name: str,
    *,
    out_dir: str | Path | None = None,
    model_keys: Sequence[str] | None = None,
) -> GalleryManifest:
    """Display one dataset's gallery cells inline.

    If a pre-rendered gallery is available (``NIFREEZE_GALLERY_OUT`` or
    ``out_dir`` holds a ``gallery_manifest.json``), the stored panels are
    embedded without re-fitting — this is how the CI-executed notebooks render.
    Otherwise the models are fit live (local use without pre-rendered output).

    Parameters
    ----------
    name : :obj:`str`
        Registered dataset name (e.g. ``"ds000206"``).
    out_dir : :obj:`str` or :obj:`~pathlib.Path`, optional
        Where panels are read from / written to (a temporary directory by
        default when computing live).
    model_keys : sequence of :obj:`str`, optional
        Restrict to these gallery model keys (live computation only).

    Returns
    -------
    :class:`~gallery.manifest.GalleryManifest`
        The coverage manifest for this dataset.

    """
    from IPython.display import Image, Markdown, display

    specs = [d for d in DATASETS if d.name == name]
    if not specs:
        known = ", ".join(d.name for d in DATASETS)
        raise ValueError(f"Unknown dataset {name!r}; registered: {known}.")
    spec = specs[0]

    # Report the exact OpenNeuro subject/run being used.
    try:
        paths = source_relpaths(name)
        files = ", ".join(f"`{p}`" for p in paths)
        src = f"[OpenNeuro {name}]({spec.url})" if spec.url else name
        display(Markdown(f"**Source:** {src} — {files}"))
    except Exception:  # pragma: no cover - display best-effort
        pass
    if spec.notes:
        display(Markdown(f"*{spec.notes}*"))

    precomputed = _precomputed_dir(out_dir)
    if precomputed is not None:
        # Embed mode: read the merged manifest and show only this dataset's cells.
        full = GalleryManifest.from_json(precomputed / "gallery_manifest.json")
        manifest = GalleryManifest(
            cells=[c for c in full.cells if c.dataset == name],
            metadata=full.metadata,
        )
        base = precomputed
    else:
        # Live mode: fit the models now and render into a scratch directory.
        from gallery.run import run_gallery

        base = Path(out_dir) if out_dir is not None else Path(tempfile.mkdtemp())
        manifest = run_gallery([spec], out_dir=base, model_keys=model_keys, render=True)

    display(Markdown(manifest.coverage_table_markdown()))

    for cell in manifest.cells:
        if cell.status != STATUS_RAN or not cell.artifacts:
            continue
        display(Markdown(f"### {cell.model} · {cell.mode}"))
        for rel in cell.artifacts:
            display(Image(filename=str(base / rel)))

    return manifest
