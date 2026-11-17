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
"""The ridge of GQI's prediction that leave-one-volume-out error selects, on DIPY's data.

For each dataset, every diffusion-weighted volume (b=0 excluded, as in NiFreeze)
is predicted in closed form from all the others, at each ridge of the grid of
:class:`~nifreeze.model.dmri.GQIKRRModel`. The table reports the error at each
ridge (normalized by each voxel's signal energy, as
:func:`~nifreeze.model.lovo.lovo_error`, and averaged over voxels), the ridge
that minimizes it, and, at that ridge, the median over volumes of the ratio
between the predicted and observed mean signals. ``docs/models.rst`` reports
its output::

    python tools/gqi_ridge.py
"""

import logging

import numpy as np
from dipy.core.gradients import gradient_table
from dipy.data import dsi_voxels, get_fnames
from dipy.io.gradients import read_bvals_bvecs
from nibabel.spatialimages import SpatialImage

from nifreeze.model.gqi import GeneralizedQSamplingModel, shell_bvalues, sphere_gram
from nifreeze.model.lovo import DEFAULT_KRR_LAMBDA_GRID, lovo_smoother
from nifreeze.utils.ndimage import load_api

VOXELS = 5000
"""Voxels drawn from each dataset's mask."""
SEED = 1234
"""Seed of the draw."""


def _dipy(name, slab):
    """The diffusion-weighted signals (volumes by voxels) of a DIPY dataset's slab."""
    fdwi, fbval, fbvec = get_fnames(name=name)
    bvals, bvecs = read_bvals_bvecs(str(fbval), str(fbvec))
    data = np.asarray(load_api(fdwi, SpatialImage).dataobj[slab], dtype=np.float64)
    weighted = bvals > 50
    reference = data[..., ~weighted].mean(axis=-1)
    mask = reference > np.percentile(reference, 60)
    return data[mask][:, weighted].T, bvals[weighted], bvecs[weighted]


def datasets():
    """Name, signals (volumes by voxels), b-values and b-vectors of each dataset."""
    data, gtab = dsi_voxels()
    weighted = ~gtab.b0s_mask
    signals = data.reshape(-1, data.shape[-1])[:, weighted].T.astype(float)
    yield "grid (dsi_voxels)", signals, gtab.bvals[weighted], gtab.bvecs[weighted]
    yield "single-shell (stanford_hardi)", *_dipy("stanford_hardi", np.s_[20:60, 40:80, 28:45])
    slab = np.s_[32:96, 32:96, 25:35]
    yield "multi-shell (sherbrooke_3shell)", *_dipy("sherbrooke_3shell", slab)


def main():
    # DIPY logs to standard output, which holds the table
    logging.getLogger("dipy").setLevel(logging.WARNING)
    rng = np.random.default_rng(SEED)
    columns, selected, amplitudes = [], [], []
    for name, signals, bvals, bvecs in datasets():
        if signals.shape[1] > VOXELS:
            signals = signals[:, np.sort(rng.choice(signals.shape[1], VOXELS, replace=False))]
        gram = sphere_gram(GeneralizedQSamplingModel(gradient_table(bvals, bvecs=bvecs)).kernel)
        energy = np.sum(signals**2, axis=0)
        errors, ratios = [], []
        for ridge in DEFAULT_KRR_LAMBDA_GRID:
            smoother = lovo_smoother(gram + ridge * np.eye(len(gram)), shells=shell_bvalues(bvals))
            predicted = smoother @ signals
            errors.append(np.mean(np.sum((signals - predicted) ** 2, axis=0) / energy))
            ratios.append(np.median(predicted.mean(axis=1) / signals.mean(axis=1)))
        best = int(np.argmin(errors))
        columns.append((f"{name}, {len(bvals)} volumes", errors))
        selected.append(DEFAULT_KRR_LAMBDA_GRID[best])
        amplitudes.append(ratios[best])

    print(".. list-table:: Leave-one-volume-out error of GQI along the ridge path")
    print("   :header-rows: 1\n")
    print("   * - ridge")
    for name, _ in columns:
        print(f"     - {name}")
    for row, ridge in enumerate(DEFAULT_KRR_LAMBDA_GRID):
        print(f"   * - {ridge:.2g}")
        for _, errors in columns:
            print(f"     - {errors[row]:.4f}")
    print("   * - selected")
    for ridge in selected:
        print(f"     - {ridge:.2g}")
    print("   * - amplitude")
    for ratio in amplitudes:
        print(f"     - {ratio:.3f}")


if __name__ == "__main__":
    main()
