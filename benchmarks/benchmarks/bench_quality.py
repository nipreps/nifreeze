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
"""How far the models' predictions drift under motion and eddy currents.

``track_quality_*`` benchmarks run each commit's
:func:`~nifreeze.analysis.sensitivity.sensitivity` on the synthetic phantom that
``.github/actions/phantom-fixtures`` restores and points ``NIFREEZE_PHANTOMS`` to
(``tools/phantoms/phantom.py fixture``, see ``docs/phantoms.rst``): the Sherbrooke
protocol at 4 mm, 24 diffusion-weighted volumes, with the noise of seed 0. Every
commit reads the same files, and the values are deterministic, so
``benchmarks/report.py`` judges them by their absolute change.

Four perturbed copies are analyzed: 6 (a quarter) and all 24 volumes moved by
1° and 1 mm (``motion_quarter``, ``motion_all``), eddy currents that displace
some brain voxel by 2 voxels (``eddy``), and both, with every volume moved
(``both``). For every third volume:

- ``drift_<copy>``: the mean NRMSE of its held-out prediction on the copy against
  that on the reference, over the volume's standard deviation;
- ``ratio_both``: the drift with both over the quadrature sum of the drifts with
  each, which exceeds 1 when they compound.

Without the dataset, or on commits without the phantom loader (before
nifreeze#563), the benchmarks are skipped.
"""

import os
import warnings
from pathlib import Path

import numpy as np

try:
    from nifreeze.analysis.phantoms import load_phantom
    from nifreeze.analysis.sensitivity import sensitivity
except ImportError:  # not on commits before nifreeze#563
    load_phantom = None  # type: ignore[assignment]

COPIES = {
    (0, True): "eddy",
    (6, False): "motion_quarter",
    (24, False): "motion_all",
    (24, True): "both",
}
"""The copies analyzed, by their number of moved volumes and whether they are distorted."""
EVALUATE = list(range(0, 24, 3))
"""The volumes predicted: every third, so that each shell has some."""
MODELS = ["avgdwi", "gp-loo", "gqi-krr", "gqi"]


def measure(model: str, reference, cells) -> dict[str, float]:
    """The drifts and compounding ratio of one model."""
    rows = sensitivity(model, reference, cells, evaluate=EVALUATE)
    copies: dict[str, list[dict]] = {}
    for row in rows:
        copies.setdefault(COPIES[row["count"], row["distortion"] > 0], []).append(row)
    values = {
        f"drift_{name}": np.mean([r["nrmse"] for r in group]) for name, group in copies.items()
    }
    values["ratio_both"] = values["drift_both"] / np.hypot(
        values["drift_motion_all"], values["drift_eddy"]
    )
    return {name: float(value) for name, value in values.items()}


class SensitivityQuality:
    """Drift and compounding of each model on the CI phantom."""

    # Bump when the phantom of ``tools/phantoms/phantom.py fixture`` changes, too
    version = "1"
    params = MODELS
    param_names = ["model"]
    timeout = 300

    def setup_cache(self):
        # asv fails a class whose ``setup_cache`` raises, and takes ``None`` for no cache at
        # all: without a phantom, an empty mapping tells ``setup`` to skip
        root = os.getenv("NIFREEZE_PHANTOMS")
        if load_phantom is None or not root or not Path(root).is_dir():
            return {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reference, cells = load_phantom(root, "sherbrooke", noise=0, counts=[6, 24])
            chosen = [
                (perturbation, load)
                for perturbation, load in cells
                if (perturbation.count, perturbation.distortion is not None) in COPIES
            ]
            return {model: measure(model, reference, chosen) for model in MODELS}

    setup_cache.timeout = 1800  # type: ignore[attr-defined]

    def setup(self, values, model):
        if not values:
            raise NotImplementedError("no phantom (NIFREEZE_PHANTOMS) or no phantom loader")
        self.values = values[model]

    def track_quality_drift_motion_quarter(self, values, model):
        return self.values["drift_motion_quarter"]

    def track_quality_drift_motion_all(self, values, model):
        return self.values["drift_motion_all"]

    def track_quality_drift_eddy(self, values, model):
        return self.values["drift_eddy"]

    def track_quality_drift_both(self, values, model):
        return self.values["drift_both"]

    def track_quality_ratio_both(self, values, model):
        return self.values["ratio_both"]
