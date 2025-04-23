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

import numpy as np
import pytest
from dipy.data import fetch_stanford_hardi, read_stanford_hardi
from dipy.segment.mask import median_otsu
from scipy.ndimage import affine_transform, binary_dilation
from skimage.morphology import ball

from nifreeze.analysis.motion import compute_relative_difference
from nifreeze.viz.motion_viz import (
    _get_axis,
    plot_framewise_displacement,
    plot_motion_overlay,
    plot_volumewise_motion,
)

fetch_stanford_hardi()
img, _ = read_stanford_hardi()
img_data = img.get_fdata()


@pytest.mark.parametrize("orientation", ["axial", "coronal", "sagittal"])
def test_plot_motion_overlay(tmp_path, orientation):
    _, brain_mask = median_otsu(img_data, vol_idx=[0])
    brain_mask = binary_dilation(brain_mask, ball(2))

    # Create an affine transformation (rotation + translation)
    theta = np.deg2rad(5)  # 5 degree rotation
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    # Rotate around Z axis and shift by +5 in x and -3 in y
    rotation_matrix = np.array([[cos_t, -sin_t, 0], [sin_t, cos_t, 0], [0, 0, 1]])
    translation = np.array([5, -3, 0])  # in voxel units

    # Transform matrix
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = translation

    # Inverse affine because scipy applies the inverse
    inv_transform = np.linalg.inv(transform)

    dwi_dir_data = img_data[..., 1]

    # Apply the transformation
    shifted = affine_transform(
        dwi_dir_data, inv_transform[:3, :3], offset=inv_transform[:3, 3], order=1
    )

    # Compute relative difference
    rel_diff = compute_relative_difference(dwi_dir_data, shifted, brain_mask)

    smooth = True
    axis = _get_axis(orientation)
    slice_idx = img_data.shape[axis] // 2

    with pytest.raises(IndexError):
        _ = plot_motion_overlay(
            rel_diff[..., np.newaxis],
            dwi_dir_data,
            brain_mask,
            orientation,
            slice_idx,
            smooth=smooth,
        )

    _slice_idx = img_data.shape[axis]
    with pytest.raises(IndexError):
        _ = plot_motion_overlay(
            rel_diff,
            dwi_dir_data,
            brain_mask,
            orientation,
            _slice_idx,
            smooth=smooth,
        )

    ax = plot_motion_overlay(
        rel_diff, dwi_dir_data, brain_mask, orientation, slice_idx, smooth=smooth
    )
    out_svg = tmp_path / "motion_overlay.svg"
    ax.figure.savefig(out_svg, format="svg")


def test_plot_volumewise_motion(request, tmp_path):
    rng = request.node.rng

    # Simulate motion for a given number of frames
    n_frames = 100
    frames = np.arange(n_frames)

    # Simulated translations (in mm)
    translations = rng.standard_normal((n_frames, 3)).cumsum(axis=0) * 0.2

    # Simulated rotations (in degrees)
    rotations = rng.standard_normal((n_frames, 3)).cumsum(axis=0) * 0.1

    # Combine into one motion matrix: shape (n_frames, 6)
    motion_params = np.hstack([translations, rotations])

    ax = plot_volumewise_motion(frames, motion_params)
    out_svg = tmp_path / "volumewise_motion.svg"
    ax[0].figure.savefig(out_svg, format="svg")


def test_plot_framewise_displacement(request, tmp_path):
    rng = request.node.rng

    n_frames = 100
    low = 0.0
    high = 1.0

    label1 = "AFNI 3dVolreg FD"
    label2 = "nifreeze FD"

    # Test appropriate dimensionality of data
    with pytest.raises(ValueError):
        fd1 = rng.uniform(low=low, high=high, size=(n_frames, 2))
        fd2 = rng.uniform(low=low, high=high, size=n_frames)
        plot_framewise_displacement(fd1, fd2, label1, label2)

        fd1 = rng.uniform(low=low, high=high, size=n_frames)
        fd2 = rng.uniform(low=low, high=high, size=(n_frames, 2))
        plot_framewise_displacement(fd1, fd2, label1, label2)

        fd1 = rng.uniform(low=low, high=high, size=n_frames + 1)
        fd2 = rng.uniform(low=low, high=high, size=n_frames)
        plot_framewise_displacement(fd1, fd2, label1, label2)

        fd1 = rng.uniform(low=low, high=high, size=n_frames)
        fd2 = rng.uniform(low=low, high=high, size=n_frames + 1)
        plot_framewise_displacement(fd1, fd2, label1, label2)

    # Simulate framewise_displacement for a given number of frames
    fd1 = rng.uniform(low=low, high=high, size=n_frames)
    fd2 = rng.uniform(low=low, high=high, size=n_frames)

    ax = plot_framewise_displacement(fd1, fd2, label1, label2)
    out_svg = tmp_path / "framewise_displacement.svg"
    ax.figure.savefig(out_svg, format="svg")
