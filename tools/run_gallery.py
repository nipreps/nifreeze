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
"""Generate the whole prediction gallery locally, on one machine.

One pass over the real (OpenNeuro) data:

1. Run the full ``(dataset × model × mode)`` matrix under ``coverage`` to emit
   the combinatorial coverage manifest (``gallery_manifest.json``), the rendered
   panels, and line coverage of the model layer (``coverage-gallery.xml``).
2. Render the documentation pages over those panels (plain rST + images).

Every model is fit exactly once: the pages are a *view* over the rendered
output, never a second computation. CI does the same work, but fans step 1 out
into one job per cell (see ``.github/workflows/gallery.yml``); this script is
the single-machine equivalent for local runs.
"""

import os
import subprocess
import sys
from pathlib import Path

OUT = Path(os.environ.get("NIFREEZE_GALLERY_OUT", "gallery-out"))
OUT.mkdir(parents=True, exist_ok=True)

# The gallery package lives under docs/ (it is docs tooling, not shipped library
# code), so put it on PYTHONPATH for the subprocesses that import it.
SPHINXEXT = Path(__file__).resolve().parent.parent / "docs" / "sphinxext"


def _env() -> dict[str, str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(SPHINXEXT), existing] if existing else [str(SPHINXEXT)]
    )
    env["NIFREEZE_GALLERY_OUT"] = str(OUT)
    return env


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    rc = subprocess.call(cmd, env=_env())
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
            "gallery.run",
            "--out",
            str(OUT),
        ]
    )
    _run([sys.executable, "-m", "coverage", "xml", "-o", "coverage-gallery.xml"])

    # 2. Render the docs pages over the panels step 1 just wrote (no re-fitting).
    _run(
        [
            sys.executable,
            str(Path(__file__).resolve().parent / "gallery_collect.py"),
            "--staging",
            str(OUT),
            "--out",
            str(OUT),
            "--pages",
            "docs/gallery",
        ]
    )


if __name__ == "__main__":
    main()
