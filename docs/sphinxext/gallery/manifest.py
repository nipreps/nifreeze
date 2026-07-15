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
"""Coverage manifest for the prediction gallery.

Records every *(dataset × model × mode)* cell as ``ran``, ``skipped`` (with a
reason), or ``error`` (with the exception message). This is the machine-readable
record of *what was exercised*, and it also renders the coverage grid shown on
the gallery documentation page.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

STATUS_RAN = "ran"
"""The cell was fit/predicted and rendered."""
STATUS_SKIPPED = "skipped"
"""The cell is inapplicable (model does not support the dataset/mode)."""
STATUS_ERROR = "error"
"""The cell was applicable but raised while running."""


@dataclass
class CellResult:
    """The outcome of a single *(dataset, model, mode)* gallery cell."""

    dataset: str
    """Dataset identifier (e.g. ``"ds000206"``)."""
    scheme: str
    """Acquisition scheme (``"single-shell"``, ``"multi-shell"``, ``"DSI"``)."""
    model: str
    """Gallery model key (e.g. ``"dti"``, ``"gp-multishell"``)."""
    mode: str
    """``"lovo"`` or ``"single-fit"``."""
    status: str
    """One of :data:`STATUS_RAN`, :data:`STATUS_SKIPPED`, :data:`STATUS_ERROR`."""
    reason: str | None = None
    """Why the cell was skipped or errored (``None`` when it ran)."""
    indices: list[int] = field(default_factory=list)
    """Held-out/predicted volume indices."""
    artifacts: list[str] = field(default_factory=list)
    """Paths (relative to the output directory) of rendered figures."""
    canary: bool = False
    """Whether single-fit here is only a self-consistency canary (warning captured)."""


@dataclass
class GalleryManifest:
    """A collection of :class:`CellResult` plus provenance metadata."""

    cells: list[CellResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- (de)serialization ---------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "metadata": self.metadata,
            "cells": [asdict(cell) for cell in self.cells],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> GalleryManifest:
        """Build a manifest from its dictionary representation."""
        return cls(
            cells=[CellResult(**cell) for cell in payload.get("cells", [])],
            metadata=dict(payload.get("metadata", {})),
        )

    def to_json(self, path: str | Path, *, indent: int = 2) -> Path:
        """Write the manifest to ``path`` as JSON and return the path."""
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=indent) + "\n")
        return path

    @classmethod
    def from_json(cls, path: str | Path) -> GalleryManifest:
        """Read a manifest from a JSON file."""
        return cls.from_dict(json.loads(Path(path).read_text()))

    # -- summaries -----------------------------------------------------------
    def counts(self) -> dict[str, int]:
        """Count cells by status."""
        out: dict[str, int] = {STATUS_RAN: 0, STATUS_SKIPPED: 0, STATUS_ERROR: 0}
        for cell in self.cells:
            out[cell.status] = out.get(cell.status, 0) + 1
        return out

    def coverage_table_rst(self) -> str:
        """Render the coverage matrix as a reStructuredText ``list-table``.

        The table has one row per cell, with a check mark for ``ran`` cells and
        the skip/error reason otherwise, so a reader sees at a glance what the
        gallery exercised and what it deliberately left out.
        """
        symbol = {STATUS_RAN: "✓", STATUS_SKIPPED: "—", STATUS_ERROR: "✗"}
        header = (
            ".. list-table:: Gallery coverage\n"
            "   :header-rows: 1\n"
            "   :widths: 18 14 16 12 8 32\n\n"
            "   * - Dataset\n"
            "     - Scheme\n"
            "     - Model\n"
            "     - Mode\n"
            "     - Ran\n"
            "     - Reason\n"
        )
        rows = []
        for cell in self.cells:
            rows.append(
                f"   * - {cell.dataset}\n"
                f"     - {cell.scheme}\n"
                f"     - {cell.model}\n"
                f"     - {cell.mode}\n"
                f"     - {symbol.get(cell.status, cell.status)}\n"
                f"     - {cell.reason or ''}\n"
            )
        return header + "".join(rows)

    def coverage_table_markdown(self) -> str:
        """Render the coverage matrix as a GitHub-flavored Markdown table."""
        symbol = {STATUS_RAN: "✓", STATUS_SKIPPED: "—", STATUS_ERROR: "✗"}
        lines = [
            "| Dataset | Scheme | Model | Mode | Ran | Reason |",
            "| --- | --- | --- | --- | :-: | --- |",
        ]
        for cell in self.cells:
            mode = f"{cell.mode} (canary)" if cell.canary else cell.mode
            lines.append(
                f"| {cell.dataset} | {cell.scheme} | {cell.model} | {mode} "
                f"| {symbol.get(cell.status, cell.status)} | {cell.reason or ''} |"
            )
        return "\n".join(lines)
