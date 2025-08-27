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

from typing import Union

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from scipy.ndimage import gaussian_filter

ORIENTATIONS = ["sagittal", "coronal", "axial"]


def _extract_slice(img_data: np.ndarray, orientation: str, slice_idx: int) -> np.ndarray:
    """Extract slice data from the given volume at the given orientation slice index.

    Parameters
    ----------
    img_data : :obj:`~numpy.ndarray`
        Image data to be sliced.
    orientation : :obj:`str`
        Orientation. Can be one of obj:`ORIENTATIONS`.
    slice_idx : :obj:`int`
        Slice index.

    Returns
    -------
    :obj:`~numpy.ndarray`
        Image slice.
    """

    axis = ORIENTATIONS.index(orientation)

    axis_sizw = img_data.shape[axis]

    if not (0 <= slice_idx < axis_sizw):
        raise IndexError(
            f"Slice index {slice_idx} out of bounds for axis {orientation} with size {axis_sizw}"
        )

    slice_obj: list[int | slice] = [slice(None)] * 3
    slice_obj[axis] = slice_idx
    slice_2d = img_data[tuple(slice_obj)]
    return slice_2d if axis == 2 else np.rot90(slice_2d)


def plot_framewise_displacement(
    fd: pd.DataFrame,
    labels: list,
    cmap_name: str = "viridis",
    ax: Union[Axes, None] = None,
) -> Axes:
    """Plot framewise displacement data.

    Plots the framewise displacement data corresponding to different
    realizations.

    Parameters
    ----------
    fd : :obj:`~pd.DataFrame`
        Framewise displacement values.
    labels : :obj:`list`
        Labels for legend.
    cmap_name : :obj:`str`, optional
        Colormap name.
    ax : :obj:`Axes`, optional
        Figure axes.

    Returns
    -------
    ax : :obj:`Axes`
        Figure plot axis.
    """

    n_cols = len(fd.columns)
    n_labels = len(labels)

    if n_cols != n_labels:
        raise ValueError(
            f"The number of realizations and labels does not match: {n_cols}; {n_labels}"
        )

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)

    # Plot the framewise displacement
    n_frames = fd.index.to_numpy()

    cmap = cm.get_cmap(cmap_name, n_cols)
    colors = [mcolors.to_hex(cmap(i)) for i in range(n_cols)]

    for i, col in enumerate(fd.columns):
        ax.plot(n_frames, fd[col], label=labels[i], color=colors[i])

    ax.set_ylabel("FD (mm)")
    ax.legend(loc="upper right")
    ax.set_xticks([])  # Hide x-ticks to keep x-axis clean

    return ax


def plot_volumewise_motion(
    frames: np.ndarray,
    motion_params: np.ndarray,
    ax: np.ndarray | None = None,
) -> np.ndarray:
    """Plot mean volume-wise motion parameters.

    Plots the mean translation and rotation parameters along the ``x``, ``y``,
    and ``z`` axes.

    Parameters
    ----------
    frames : :obj:`~numpy.ndarray`
        Frame indices.
    motion_params : :obj:`~numpy.ndarray`
        Motion parameters.Motion parameters: translation and rotation. Each row
        represents one frame, and columns represent each coordinate axis ``x``,
        ``y``, and ``z``. Translation parameters are followed by rotation
        parameters column-wise.
    ax : :obj:`~numpy.ndarray`, optional
        Figure axes.

    Returns
    -------
    ax : :obj:`~numpy.ndarray`
        Figure plot axes array.
    """

    if ax is None:
        fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)

    # Plot translations
    ax[0].plot(frames, motion_params[:, 0], label="x")
    ax[0].plot(frames, motion_params[:, 1], label="y")
    ax[0].plot(frames, motion_params[:, 2], label="z")
    ax[0].set_ylabel("Translation (mm)")
    ax[0].legend(loc="upper right")
    ax[0].set_title("Translation vs frames")

    # Plot rotations
    ax[1].plot(frames, motion_params[:, 3], label="Rx")
    ax[1].plot(frames, motion_params[:, 4], label="Ry")
    ax[1].plot(frames, motion_params[:, 5], label="Rz")
    ax[1].set_ylabel("Rotation (deg)")
    ax[1].set_xlabel("Time (s)")
    ax[1].legend(loc="upper right")
    ax[1].set_title("Rotation vs frames")

    return ax


def plot_motion_overlay(
    rel_diff: np.ndarray,
    img_data: np.ndarray,
    brain_mask: np.ndarray,
    orientation: str,
    slice_idx: int,
    smooth: bool = True,
    ax: Union[Axes, None] = None,
) -> Axes:
    """Plot motion relative difference as an overlay on a given orientation and slice of the imaging data.

    The values of the relative difference can optionally be smoothed using a
    Gaussian filter for a more appealing visual result.

    Parameters
    ----------
    rel_diff : :obj:`~numpy.ndarray`
        Relative motion difference.
    img_data : :obj:`~numpy.ndarray`
        Imaging data.
    brain_mask : :obj:`~numpy.ndarray`
        Brain mask.
    orientation : :obj:`str`
        Orientation. Can be one of obj:`ORIENTATIONS`.
    slice_idx : :obj:`int`
        Slice index to plot.
    smooth : :obj:`bool`, optional
        ``True`` to smooth the motion relative difference.
    ax : :obj:`Axes`, optional
        Figure axis.

    Returns
    -------
    ax : :obj:`Axes`
        Figure plot axis.
    """

    # Check dimensionality
    if img_data.shape != rel_diff.shape:
        raise IndexError(
            f"Dimension mismatch: imaging data shape {img_data.shape}, overlay shape {rel_diff.shape}"
        )

    # Smooth the relative difference
    smoothed_diff = rel_diff
    if smooth:
        smoothed_diff = gaussian_filter(rel_diff, sigma=1)

    # Mask the background
    masked_img_data = np.where(brain_mask, img_data, np.nan)
    masked_smooth_diff = np.where(brain_mask, smoothed_diff, np.nan)

    masked_img_slice = _extract_slice(masked_img_data, orientation, slice_idx)
    diff_img_slice = _extract_slice(masked_smooth_diff, orientation, slice_idx)

    # Show overlay on a slice
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)

    ax.imshow(masked_img_slice, cmap="gray")
    im = ax.imshow(diff_img_slice, cmap="bwr", alpha=0.5)
    ax.figure.colorbar(im, ax=ax, label="Relative Difference (%)")
    ax.set_title("Smoothed Relative Difference Overlay")
    ax.axis("off")

    return ax
