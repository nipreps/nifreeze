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
"""Compute the number of fiber orientations (NuFO) on dMRI data using a SSST CSD model."""

import argparse
from pathlib import Path

import nibabel as nb
import numpy as np
from dipy.core.gradients import gradient_table
from dipy.data import default_sphere
from dipy.direction import peaks_from_model
from dipy.io.gradients import read_bvals_bvecs
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel, auto_response_ssst
from dipy.segment.mask import median_otsu
from scipy.ndimage import binary_dilation
from skimage.morphology import ball


def _build_arg_parser() -> argparse.ArgumentParser:
    """
    Build argument parser for command-line interface.

    Returns
    -------
    :obj:`~argparse.ArgumentParser`
        Argument parser for the script.

    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("dwi_fname", help="DWI filename", type=Path)
    parser.add_argument("bval_fname", help="b-vals filename", type=Path)
    parser.add_argument("bvec_fname", help="b-vecs filename", type=Path)
    return parser


def _parse_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """
    Parse command-line arguments.

    Parameters
    ----------
    parser : :obj:`~argparse.ArgumentParser`
        Argument parser for the script.

    Returns
    -------
    :obj:`~argparse.Namespace`
        Parsed arguments.
    """
    return parser.parse_args()


def main() -> None:
    """Main function for running the experiment."""
    parser = _build_arg_parser()
    args = _parse_args(parser)

    # Read the data
    dwi_data = nb.load(args.dwi_fname).get_fdata()
    bvals, bvecs = read_bvals_bvecs(str(args.bval_fname), str(args.bvec_fname))
    gtab = gradient_table(bvals, bvecs=bvecs)

    # ToDo
    # Allow this to be a parameter
    _, brain_mask = median_otsu(dwi_data, vol_idx=[0])
    brain_mask = binary_dilation(brain_mask, ball(8))

    dwi_data_masked = dwi_data.copy()
    dwi_data_masked[~brain_mask, :] = 0

    # Create a CSD model
    roi_radii = 10
    fa_thr = 0.7
    response, ratio = auto_response_ssst(
        gtab, dwi_data_masked, roi_radii=roi_radii, fa_thr=fa_thr)

    # ToDo
    # Ensure there is enough DWI volumes to fit
    sh_order_max = 6
    csd_model = ConstrainedSphericalDeconvModel(
        gtab, response=response, sh_order_max=sh_order_max)

    dwi_data_masked = dwi_data_masked[:, :, 33:36]
    brain_mask = brain_mask[:, :, 33:36]

    # Compute peaks from the model
    # ToDo
    # Make these parameters
    relative_peak_threshold = 0.5
    min_separation_angle = 25
    npeaks = 5
    csd_peaks = peaks_from_model(
        model=csd_model,
        data=dwi_data,
        sphere=default_sphere,
        relative_peak_threshold=relative_peak_threshold,
        min_separation_angle=min_separation_angle,
        mask=brain_mask,
        return_sh=True,
        return_odf=False,
        normalize_peaks=True,
        npeaks=npeaks,
        parallel=True,
        num_processes=None,
    )

    # Get the number of peaks
    peak_counts = csd_peaks.peak_indices
    num_peaks = (peak_counts != -1).sum(axis=3)

    # ToDo
    # Get the NuFO from the peaks

    # ToDo
    # Serialize csd_peaks to HDF5 or num_peaks

    # Create mask for voxels with more than a given number of peaks
    peak_count_threshold = 2
    multi_peak_mask = num_peaks > peak_count_threshold
    multi_peak_indices = np.argwhere(multi_peak_mask)

    # Output: binary mask and indices
    print(f"{multi_peak_mask.sum()} voxels out of {brain_mask.sum()} have more than {peak_count_threshold} peaks.")


if __name__ == "__main__":
    main()
