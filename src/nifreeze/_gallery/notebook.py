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

import tempfile
from collections.abc import Sequence
from pathlib import Path

from nifreeze._gallery.datasets import DATASETS
from nifreeze._gallery.manifest import STATUS_RAN, GalleryManifest
from nifreeze._gallery.run import run_gallery


def show_dataset(
    name: str,
    *,
    out_dir: str | Path | None = None,
    model_keys: Sequence[str] | None = None,
) -> GalleryManifest:
    """Run one dataset's gallery cells and display the results inline.

    Parameters
    ----------
    name : :obj:`str`
        Registered dataset name (e.g. ``"ds000206"``).
    out_dir : :obj:`str` or :obj:`~pathlib.Path`, optional
        Where panels are written (a temporary directory by default).
    model_keys : sequence of :obj:`str`, optional
        Restrict to these gallery model keys.

    Returns
    -------
    :class:`~nifreeze._gallery.manifest.GalleryManifest`
        The coverage manifest for this dataset.

    """
    from IPython.display import Image, Markdown, display

    specs = [d for d in DATASETS if d.name == name]
    if not specs:
        known = ", ".join(d.name for d in DATASETS)
        raise ValueError(f"Unknown dataset {name!r}; registered: {known}.")
    spec = specs[0]

    out = Path(out_dir) if out_dir is not None else Path(tempfile.mkdtemp())
    manifest = run_gallery([spec], out_dir=out, model_keys=model_keys, render=True)

    if spec.notes:
        display(Markdown(f"*{spec.notes}*"))
    display(Markdown(manifest.coverage_table_markdown()))

    for cell in manifest.cells:
        if cell.status != STATUS_RAN or not cell.artifacts:
            continue
        display(Markdown(f"### {cell.model} · {cell.mode}"))
        for rel in cell.artifacts:
            display(Image(filename=str(out / rel)))

    return manifest
