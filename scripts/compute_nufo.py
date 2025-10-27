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
from dipy.core.gradients import GradientTable, gradient_table
from dipy.data import default_sphere
from dipy.direction import peaks_from_model
from dipy.io.gradients import read_bvals_bvecs
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel, auto_response_ssst
from dipy.segment.mask import median_otsu

from nifreeze.utils.ndimage import load_api

DEFAULT_LOWB_THRESHOLD = 50
"""Lower bound for the b-value so that the orientation is considered a DW volume."""


def check_sh_sufficiency(lmax: int, gtab: GradientTable) -> None:
    """Check that there is enough gradient volumes (directions) for the provided spherical harmonics order.

    Returns
    -------
    lmax : :obj:`int`
        Spherical harmonics order maximum value.
    gtab : :obj:`~dipy.core.gradientsGradientTable`
        Gradient table.
    """
    # Count nonzero b-value volumes
    # Threshold to distinguish b0 vs DWI

    dwi_mask = ~gtab.b0s_mask
    num_dwi_volumes = np.count_nonzero(dwi_mask)

    # lmax: SH order to check
    n_coeff = int((lmax + 1) * (lmax + 2) / 2)

    print(f"Number of DWI volumes: {num_dwi_volumes}")
    print(f"Required for lmax={lmax}: {n_coeff}")

    if num_dwi_volumes < n_coeff:
        raise ValueError(
            f"Not enough DWI directions ({num_dwi_volumes}) for the desired SH order ({lmax})."
        )


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
    parser.add_argument("in_dwi_fname", type=Path, help="Input DWI filename")
    parser.add_argument("in_bval_fname", type=Path, help="Input b-vals filename")
    parser.add_argument("in_bvec_fname", type=Path, help="Input b-vecs filename")
    parser.add_argument("out_nufo_fname", type=Path, help="Output NuFO filename")

    g_dmri = parser.add_argument_group("Options for dMRI inputs")
    g_dmri.add_argument("--in_brain_mask_fname", type=Path, help="Input brain mask filename")
    g_dmri.add_argument(
        "--b0-threshold",
        type=float,
        default=DEFAULT_LOWB_THRESHOLD,
        help="Lower bound for the b-value so that the orientation is considered a DW volume",
    )

    g_response = parser.add_argument_group("Options for tissue response computation")
    g_response.add_argument(
        "--roi-radii",
        type=int,
        default=10,
        help="Radius of a cuboid where the tissue response is computed",
    )
    g_response.add_argument(
        "--fa-thr",
        type=float,
        default=0.7,
        help="Threshold for the fractional anisotropy value to compute the tissue response",
    )

    g_deconv = parser.add_argument_group("Options for CSD fit")
    g_deconv.add_argument(
        "--sh-order-max", type=int, default=6, help="Maximum spherical harmonics order"
    )

    g_peaks = parser.add_argument_group("Options for peak extraction")
    g_peaks.add_argument(
        "--relative-peak-threshold",
        type=float,
        default=0.5,
        help="Threshold to filter peaks smaller than the fraction with respect to the largest peak",
    )
    g_peaks.add_argument(
        "--min-separation-angle", type=float, default=25, help="Minimum distance between peaks"
    )
    g_peaks.add_argument("--npeaks", type=int, default=5, help="Number of peaks to be detected")

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
    dwi_img = load_api(args.in_dwi_fname, nb.Nifti1Image)
    dwi_data = dwi_img.get_fdata()
    bvals, bvecs = read_bvals_bvecs(str(args.in_bval_fname), str(args.in_bvec_fname))

    gtab = gradient_table(bvals, bvecs=bvecs, b0_threshold=args.b0_threshold)

    if args.in_brain_mask_fname is not None:
        brain_mask = load_api(args.in_brain_mask_fname, nb.Nifti1Image).get_fdata()
    else:
        _, brain_mask = median_otsu(dwi_data, vol_idx=[0])

    dwi_data_masked = dwi_data.copy()
    dwi_data_masked[~brain_mask.astype(bool), :] = 0

    # Create a CSD model
    response, ratio = auto_response_ssst(
        gtab, dwi_data_masked, roi_radii=args.roi_radii, fa_thr=args.fa_thr
    )

    # Ensure there is enough DWI volumes to fit
    check_sh_sufficiency(args.sh_order_max, gtab)
    csd_model = ConstrainedSphericalDeconvModel(
        gtab, response=response, sh_order_max=args.sh_order_max
    )

    # Compute peaks from the model
    csd_peaks = peaks_from_model(
        model=csd_model,
        data=dwi_data,
        sphere=default_sphere,
        relative_peak_threshold=args.relative_peak_threshold,
        min_separation_angle=args.min_separation_angle,
        mask=brain_mask,
        return_sh=True,
        return_odf=False,
        normalize_peaks=True,
        npeaks=args.npeaks,
        parallel=True,
        num_processes=None,
    )

    # Get the number of peaks
    peak_idx = csd_peaks.peak_indices
    nufo = (peak_idx != -1).sum(axis=-1)

    nb.Nifti1Image(nufo.astype("uint8"), dwi_img.affine, header=dwi_img.header).to_filename(
        args.out_nufo_fname
    )


if __name__ == "__main__":
    main()
