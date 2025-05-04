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
from dipy.core.gradients import gradient_table, GradientTable
from dipy.data import default_sphere
from dipy.direction import peaks_from_model
from dipy.io.gradients import read_bvals_bvecs
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel, auto_response_ssst
from dipy.segment.mask import median_otsu
from scipy.ndimage import binary_dilation
from skimage.morphology import ball


def check_sh_sufficiency(lmax : int, gtab: GradientTable):
    # Count nonzero b-value volumes
    # Threshold to distinguish b0 vs DWI

    dwi_mask = ~gtab.b0s_mask
    num_dwi_volumes = np.count_nonzero(dwi_mask)

    # lmax: SH order to check
    n_coeff = int((lmax + 1) * (lmax + 2) / 2)

    print(f"Number of DWI volumes: {num_dwi_volumes}")
    print(f"Required for lmax={lmax}: {n_coeff}")

    if num_dwi_volumes < n_coeff:
        raise ValueError(f"Not enough DWI directions ({num_dwi_volumes}) for the desired SH order ({lmax}).")



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
    parser.add_argument("in_dwi_fname", help="Input DWI filename", type=Path)
    parser.add_argument("in_bval_fname", help="Input b-vals filename", type=Path)
    parser.add_argument("in_bvec_fname", help="Input b-vecs filename", type=Path)
    parser.add_argument("out_nufo_fname", help="Output NuFO filename", type=Path)
    parser.add_argument("--in_brain_mask_fname", help="Input brain mask filename", type=Path)
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
    dwi_img = nb.load(args.in_dwi_fname)
    dwi_data = dwi_img.get_fdata()
    bvals, bvecs = read_bvals_bvecs(str(args.in_bval_fname), str(args.in_bvec_fname))
    # ToDo
    # Make these parameter
    b0_threshold = 50
    gtab = gradient_table(bvals, bvecs=bvecs, b0_threshold=b0_threshold)

    if args.in_brain_mask_fname is not None:
        brain_mask = nb.load(args.in_brain_mask_fname).get_fdata()
    else:
        _, brain_mask = median_otsu(dwi_data, vol_idx=[0])
        brain_mask = binary_dilation(brain_mask, ball(8))

    dwi_data_masked = dwi_data.copy()
    dwi_data_masked[~brain_mask, :] = 0

    # Create a CSD model
    roi_radii = 10
    fa_thr = 0.7
    response, ratio = auto_response_ssst(
        gtab, dwi_data_masked, roi_radii=roi_radii, fa_thr=fa_thr)

    # Ensure there is enough DWI volumes to fit
    # ToDo
    # Make these parameters
    sh_order_max = 6
    check_sh_sufficiency(sh_order_max, gtab)
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
        data=dwi_data[:, :, 33:36, :],
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
    peak_idx = csd_peaks.peak_indices
    nufo = (peak_idx != -1).sum(axis=-1)

    nb.Nifti1Image(
        nufo.astype("uint8"), dwi_img.affine, header=dwi_img.header
    ).to_filename(args.out_nufo_fname)


if __name__ == "__main__":
    main()
