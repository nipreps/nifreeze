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
"""Generate the prediction gallery: coverage manifest + executed notebooks.

Two passes over the real (OpenNeuro) data:

1. Run the full ``(dataset × model × mode)`` matrix under ``coverage`` to emit
   the combinatorial coverage manifest (``gallery_manifest.json``), the rendered
   panels, and line coverage of the model layer (``coverage-gallery.xml``).
2. Execute the documentation notebooks in place so their outputs are stored for
   nbsphinx (which renders them with ``nbsphinx_execute = "never"``).

Intended for the scheduled ``gallery`` CI job, not per-PR runs.
"""

import glob
import os
import subprocess
import sys
from pathlib import Path

OUT = Path(os.environ.get("NIFREEZE_GALLERY_OUT", "gallery-out"))
OUT.mkdir(parents=True, exist_ok=True)


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    rc = subprocess.call(cmd)
    if rc:
        sys.exit(rc)


def main() -> None:
    # 1. Coverage-instrumented full matrix -> manifest + panels + line coverage.
    _run(
        [
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--source=nifreeze",
            "-m",
            "nifreeze._gallery.run",
            "--out",
            str(OUT),
        ]
    )
    _run([sys.executable, "-m", "coverage", "xml", "-o", "coverage-gallery.xml"])

    # 2. Execute the gallery notebooks in place so docs render stored outputs.
    notebooks = sorted(glob.glob("docs/notebooks/gallery/*.ipynb"))
    if notebooks:
        _run(
            [
                sys.executable,
                "-m",
                "jupyter",
                "nbconvert",
                "--to",
                "notebook",
                "--execute",
                "--inplace",
                "--ExecutePreprocessor.timeout=3600",
                *notebooks,
            ]
        )


if __name__ == "__main__":
    main()
