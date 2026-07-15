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
"""Render the gallery documentation pages from a coverage manifest.

The fit jobs already render every panel to disk, so the documentation is just a
view over that output: one reStructuredText page per dataset embedding the
stored PNGs, plus the coverage table. No notebook, kernel, or re-computation is
involved — the pages are plain rST and the docs build only has to copy images.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from gallery.datasets import DATASETS
from gallery.manifest import STATUS_RAN, GalleryManifest


def _heading(text: str, char: str = "=", overline: bool = False) -> str:
    """Return ``text`` as an rST heading underlined (and optionally overlined)."""
    rule = char * len(text)
    return f"{rule}\n{text}\n{rule}\n" if overline else f"{text}\n{rule}\n"


def _cell_title(cell) -> str:
    """Human-readable heading for a cell, flagging the self-consistency canary."""
    title = f"{cell.model} · {cell.mode}"
    return f"{title} (canary)" if cell.canary else title


def placeholder_page_rst(name: str) -> str:
    """Render a page for a dataset whose panels have not been published yet.

    The toctree references every dataset, so a page must exist even before the
    first gallery run lands — and it should say plainly that nothing has been
    published rather than imply an empty result.
    """
    spec = next((d for d in DATASETS if d.name == name), None)
    title = spec.title if spec is not None else name
    parts = [f".. _gallery_{name}:\n", _heading(title, "=", overline=True)]
    if spec is not None and spec.notes:
        parts.append(f".. note::\n\n   {spec.notes}\n")
    parts.append(
        "The panels for this dataset have not been published yet. They are produced\n"
        "by the ``gallery`` workflow, which fits each *(model × mode)* cell in its own\n"
        "job and opens a pull request refreshing these pages.\n"
    )
    return "\n".join(parts) + "\n"


def dataset_page_rst(manifest: GalleryManifest, name: str) -> str:
    """Render one dataset's gallery page.

    Parameters
    ----------
    manifest : :class:`~gallery.manifest.GalleryManifest`
        The merged manifest (cells for other datasets are ignored).
    name : :obj:`str`
        The dataset to render (e.g. ``"ds004737"``).

    Returns
    -------
    :obj:`str`
        The page as reStructuredText.

    """
    spec = next((d for d in DATASETS if d.name == name), None)
    title = spec.title if spec is not None else name
    scoped = GalleryManifest(
        cells=[c for c in manifest.cells if c.dataset == name],
        metadata=manifest.metadata,
    )

    parts = [f".. _gallery_{name}:\n", _heading(title, "=", overline=True)]

    # Provenance: the exact subject/run, recorded by the fit stage.
    sources = manifest.metadata.get("sources", {}).get(name, [])
    if sources:
        files = ", ".join(f"``{p}``" for p in sources)
        src = f"`OpenNeuro {name} <{spec.url}>`__" if spec is not None and spec.url else name
        parts.append(f"**Source:** {src} — {files}\n")
    if spec is not None and spec.notes:
        parts.append(f".. note::\n\n   {spec.notes}\n")

    parts.append(
        f"Predicted diffusion volumes for the **{title}** dataset, across every\n"
        "applicable model in both **LOVO** (leave-one-volume-out, holding out the\n"
        "predicted orientation) and **single-fit** (fit once on all volumes) modes.\n"
    )

    parts.append(_heading("Coverage", "-"))
    parts.append(scoped.coverage_table_rst())

    for cell in scoped.cells:
        if cell.status != STATUS_RAN or not cell.artifacts:
            continue
        parts.append(_heading(_cell_title(cell), "-"))
        for rel in cell.artifacts:
            parts.append(f".. figure:: {rel}\n   :alt: {cell.model} {cell.mode} {cell.dataset}\n")

    return "\n".join(parts) + "\n"


def write_pages(
    manifest: GalleryManifest,
    pages_dir: Path,
    panels_from: Path | None = None,
    datasets: list[str] | None = None,
) -> list[Path]:
    """Write one page per dataset into ``pages_dir``, copying panels alongside.

    Panels are referenced relative to the page (``<dataset>/<panel>.png``), so
    the rendered figures live next to the page that embeds them and the docs
    build needs nothing but Sphinx's ordinary image handling.

    ``datasets`` pins which pages are written (defaults to those the manifest
    knows about). A dataset with no cells gets a placeholder, so the toctree
    never references a missing document.
    """
    pages_dir = Path(pages_dir)
    pages_dir.mkdir(parents=True, exist_ok=True)

    names = datasets if datasets is not None else sorted({c.dataset for c in manifest.cells})
    have_cells = {c.dataset for c in manifest.cells}

    written: list[Path] = []
    for name in names:
        page = pages_dir / f"{name}.rst"
        if name in have_cells:
            page.write_text(dataset_page_rst(manifest, name))
        else:
            page.write_text(placeholder_page_rst(name))
        written.append(page)

        if panels_from is not None:
            src_dir = Path(panels_from) / name
            if src_dir.is_dir():
                dest_dir = pages_dir / name
                dest_dir.mkdir(parents=True, exist_ok=True)
                for png in sorted(src_dir.glob("*.png")):
                    shutil.copy2(png, dest_dir / png.name)
    return written
