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
"""Measure the b=0 signal-to-noise ratio of the open datasets behind the phantom protocols.

The synthetic dataset adds complex Gaussian noise of standard deviation
``sigma = mean white-matter b=0 / SNR`` to its noise-free images. This script
measures that SNR on real data acquired with each protocol, where the data are
openly downloadable. The signal is the mean b=0 within white matter (fractional
anisotropy above 0.5, from a tensor fit). The noise ``sigma`` is the median,
within white matter, of the local PCA estimate (Manjón et al., 2013); with
several b=0 volumes, it is checked against the spread of their differences.

Usage::

    python tools/phantoms/measure_snr.py --cache ~/.cache/nifreeze-phantoms
"""

import argparse
import urllib.request
from pathlib import Path

import numpy as np
from nibabel.spatialimages import SpatialImage

from nifreeze.utils.ndimage import load_api

DATASETS = {
    "hbn": "https://fcp-indi.s3.amazonaws.com/data/Projects/HBN/MRI/Site-CBIC/"
    "sub-NDARAA396TWZ/dwi/sub-NDARAA396TWZ_acq-64dir_dwi",
    "aomic": "https://s3.amazonaws.com/openneuro.org/ds002790/sub-0001/dwi/sub-0001_dwi",
}
"""Datasets downloaded from a URL stem (``.nii.gz``, ``.bval`` and ``.bvec`` appended)."""

DIPY_DATASETS = {"sherbrooke": "sherbrooke_3shell", "stanford": "stanford_hardi"}
"""Datasets fetched by DIPY."""


def fetch(name: str, cache: Path) -> tuple[Path, Path, Path]:
    """Return the NIfTI, b-value and b-vector files of a dataset, downloading them if needed."""
    if name in DIPY_DATASETS:
        from dipy.data import get_fnames

        image, bval, bvec = get_fnames(name=DIPY_DATASETS[name])
        return Path(image), Path(bval), Path(bvec)
    files = []
    for extension in (".nii.gz", ".bval", ".bvec"):
        target = cache / f"{name}{extension}"
        if not target.exists():
            cache.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(DATASETS[name] + extension, target)
        files.append(target)
    return files[0], files[1], files[2]


def measure(image: Path, bval: Path, bvec: Path) -> dict[str, float]:
    """Return the white-matter b=0 signal, the noise SD and their ratio."""
    from dipy.core.gradients import gradient_table
    from dipy.denoise.pca_noise_estimate import pca_noise_estimate
    from dipy.reconst.dti import TensorModel
    from dipy.segment.mask import median_otsu
    from scipy.ndimage import binary_erosion

    data = np.asarray(load_api(image, SpatialImage).dataobj, dtype=np.float32)
    bvals, bvecs = np.loadtxt(bval), np.loadtxt(bvec)
    bvecs = bvecs.T if bvecs.shape[0] == 3 else bvecs
    b0s = np.flatnonzero(bvals < 50)
    _, brain = median_otsu(data, vol_idx=b0s, median_radius=4, numpass=4)
    # A tensor fit on the lowest shell and the b=0 volumes, which the tensor describes best
    lowest = bvals[bvals >= 50].min()
    keep = (bvals < 50) | (np.abs(bvals - lowest) < 100)
    gtab = gradient_table(bvals[keep], bvecs=bvecs[keep])
    fa = TensorModel(gtab).fit(data[..., keep], mask=brain).fa
    white = binary_erosion(np.nan_to_num(fa) > 0.5) & brain
    bzero = data[..., b0s].mean(axis=-1)
    signal = float(bzero[white].mean())
    local = pca_noise_estimate(data, gradient_table(bvals, bvecs=bvecs), correct_bias=True)
    sigma = float(np.median(local[white]))
    result = {
        "signal": signal,
        "sigma": sigma,
        "snr": signal / sigma,
        "white_matter_voxels": int(white.sum()),
    }
    if len(b0s) > 1:
        # Differences of consecutive b=0 volumes: their SD is sigma times sqrt(2)
        differences = np.diff(data[..., b0s], axis=-1)[white]
        result["sigma_from_b0_differences"] = float(differences.std() / np.sqrt(2.0))
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cache", type=Path, default=Path.home() / ".cache/nifreeze-phantoms")
    parser.add_argument("datasets", nargs="*", default=[*DATASETS, *DIPY_DATASETS])
    args = parser.parse_args(argv)
    for name in args.datasets:
        result = measure(*fetch(name, args.cache))
        print(name, {key: round(value, 2) for key, value in result.items()})


if __name__ == "__main__":
    main()
