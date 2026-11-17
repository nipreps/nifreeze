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
"""How accurately and how fast a volume is registered (nipreps/nifreeze#584).

Each diffusion-weighted volume of the CI phantom's copy that moves all 24 volumes
by 1° and 1 mm (``tools/phantoms/phantom.py fixture``, see ``docs/phantoms.rst``),
with the noise of seed 0, is registered to the noise-free reference volume of the
same direction: the error of registration alone, with no model involved
(:func:`~nifreeze.analysis.registration.registration_accuracy`).

- ``track_registration_oracle_median`` and ``_p95``: the largest displacement (mm)
  within the brain between the estimated and true motions, over the 24 volumes;
- ``time_register``: one registration, with the estimator's ANTs path.

ANTs runs on one thread with a fixed seed, so the errors are deterministic and
``benchmarks/report.py`` judges them by their absolute change. Without the dataset,
without ANTs, or on commits without the harness (before nifreeze#584), the
benchmarks are skipped.
"""

import os
import shutil
import tempfile
import warnings
from pathlib import Path

import numpy as np

try:
    from nifreeze.analysis.phantoms import load_phantom
    from nifreeze.analysis.registration import AntsRegistration, registration_accuracy
except ImportError:  # not on commits before nifreeze#584
    registration_accuracy = None  # type: ignore[assignment]


def _phantom():
    """The noise-free and noisy references, the copy moving every volume, and its truth."""
    root = os.getenv("NIFREEZE_PHANTOMS")
    if (
        registration_accuracy is None
        or shutil.which("antsRegistration") is None
        or not root
        or not Path(root).is_dir()
    ):
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clean, _ = load_phantom(root, "sherbrooke", counts=[])
        reference, cells = load_phantom(root, "sherbrooke", noise=0, counts=[24])
        perturbation, load = next(
            (p, load) for p, load in cells if p.count == 24 and p.distortion is None
        )
        truth = dict(zip(perturbation.moved, perturbation.motions, strict=True))
        return clean, reference, load(), truth


class RegistrationAccuracy:
    """Registration error on the CI phantom, against the oracle."""

    # Bump when the phantom of ``tools/phantoms/phantom.py fixture`` changes, too
    version = "1"
    timeout = 300

    def setup_cache(self):
        # asv fails a class whose ``setup_cache`` raises: an empty mapping tells
        # ``setup`` to skip
        phantom = _phantom()
        if phantom is None:
            return {}
        clean, reference, moved, truth = phantom
        with tempfile.TemporaryDirectory() as workdir:
            rows = registration_accuracy(
                AntsRegistration(workdir, num_threads=1),
                clean,
                reference,
                moved,
                truth,
                fixed=["oracle"],
            )
        errors = [row["max"] for row in rows]
        return {"median": float(np.median(errors)), "p95": float(np.percentile(errors, 95))}

    setup_cache.timeout = 900  # type: ignore[attr-defined]

    def setup(self, values):
        if not values:
            raise NotImplementedError("no phantom (NIFREEZE_PHANTOMS), no ANTs, or no harness")
        self.values = values

    def track_registration_oracle_median(self, values):
        return self.values["median"]

    def track_registration_oracle_p95(self, values):
        return self.values["p95"]


class RegistrationTime:
    """One registration of the CI phantom, with the estimator's ANTs path."""

    timeout = 120

    def setup(self):
        phantom = _phantom()
        if phantom is None:
            raise NotImplementedError("no phantom (NIFREEZE_PHANTOMS), no ANTs, or no harness")
        clean, reference, moved, _ = phantom
        self.workdir = tempfile.TemporaryDirectory()
        self.register = AntsRegistration(self.workdir.name, num_threads=1)
        self.args = (
            np.asarray(clean.dataobj[0], dtype="float32"),
            np.asarray(moved.dataobj[0]),
            reference.affine,
            np.asanyarray(reference.brainmask, dtype=bool),
            0,
        )

    def teardown(self):
        self.workdir.cleanup()

    def time_register(self):
        self.register(*self.args)
