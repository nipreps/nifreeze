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
"""
Utilities to aid in performing and evaluating image registration.

This module provides functions to compute displacements of image coordinates
under a transformation, useful for assessing the accuracy of image registration
processes.

"""

from __future__ import annotations

from itertools import product
from typing import Tuple

import nibabel as nb
import nitransforms as nt
import numpy as np

RADIUS = 50.0
"""Typical radius (in mm) of a sphere mimicking the size of a typical human brain."""


def displacements_within_mask(
    mask_img: nb.spatialimages.SpatialImage,
    test_xfm: nt.base.TransformBase,
    reference_xfm: nt.base.TransformBase | None = None,
) -> np.ndarray:
    """
    Compute the distance between voxel coordinates mapped through two transforms.

    Parameters
    ----------
    mask_img : :obj:`~nibabel.spatialimages.SpatialImage`
        A mask image that defines the region of interest. Voxel coordinates
        within the mask are transformed.
    test_xfm : :obj:`~nitransforms.base.TransformBase`
        The transformation to test. This transformation is applied to the
        voxel coordinates.
    reference_xfm : :obj:`~nitransforms.base.TransformBase`, optional
        A reference transformation to compare with. If :obj:`None`, the identity
        transformation is assumed (no transformation).

    Returns
    -------
    :obj:`~numpy.ndarray`
        An array of displacements (in mm) for each voxel within the mask.

    """
    # Mask data as boolean (True for voxels inside the mask)
    maskdata = np.asanyarray(mask_img.dataobj) > 0
    # Convert voxel coordinates to world coordinates using affine transform
    xyz = nb.affines.apply_affine(
        mask_img.affine,
        np.argwhere(maskdata),
    )
    # Apply the test transformation
    targets = test_xfm.map(xyz)

    # Compute the difference (displacement) between the test and reference transformations
    diffs = targets - xyz if reference_xfm is None else targets - reference_xfm.map(xyz)
    return np.linalg.norm(diffs, axis=-1)


def compute_fd_from_transform(
    img: nb.spatialimages.SpatialImage,
    test_xfm: nt.base.TransformBase,
    radius: float = RADIUS,
) -> float:
    """
    Compute the framewise displacement (FD) for a given transformation.

    Parameters
    ----------
    img : :obj:`~nibabel.spatialimages.SpatialImage`
        The reference image. Used to extract the center coordinates.
    test_xfm : :obj:`~nitransforms.base.TransformBase`
        The transformation to test. Applied to coordinates around the image center.
    radius : :obj:`float`, optional
        The radius (in mm) of the spherical neighborhood around the center of the image.

    Returns
    -------
    :obj:`float`
        The average framewise displacement (FD) for the test transformation.

    """
    affine = img.affine
    # Compute the center of the image in voxel space
    center_ijk = 0.5 * (np.array(img.shape[:3]) - 1)
    # Convert to world coordinates
    center_xyz = nb.affines.apply_affine(affine, center_ijk)
    # Generate coordinates of points at radius distance from center
    fd_coords = np.array(list(product(*((radius, -radius),) * 3))) + center_xyz
    # Compute the average displacement from the test transformation
    return np.mean(np.linalg.norm(test_xfm.map(fd_coords) - fd_coords, axis=-1))


def compute_fd_from_motion(motion_parameters: np.ndarray, radius: float = RADIUS) -> np.ndarray:
    """Compute framewise displacement (FD) from motion parameters.

    Each row in the motion parameters represents one frame, and columns
    represent each coordinate axis ``x``, `y``, and ``z``. Translation
    parameters are followed by rotation parameters column-wise.

    Parameters
    ----------
    motion_parameters : :obj:`~numpy.ndarray`
        Motion parameters.
    radius : :obj:`float`, optional
        Radius (in mm) of a sphere mimicking the size of a typical human brain.

    Returns
    -------
    :obj:`~numpy.ndarray`
        The framewise displacement (FD) as the sum of absolute differences
        between consecutive frames.
    """

    translations = motion_parameters[:, :3]
    rotations_deg = motion_parameters[:, 3:]
    rotations_rad = np.deg2rad(rotations_deg)

    # Compute differences between consecutive frames
    d_translations = np.vstack([np.zeros((1, 3)), np.diff(translations, axis=0)])
    d_rotations = np.vstack([np.zeros((1, 3)), np.diff(rotations_rad, axis=0)])

    # Convert rotations from radians to displacement on a sphere
    rotation_displacement = d_rotations * radius

    # Compute FD as sum of absolute differences
    return np.sum(np.abs(d_translations) + np.abs(rotation_displacement), axis=1)


def extract_motion_parameters(affine: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Extract translation (mm) and rotation (degrees) parameters from an affine matrix.

    Parameters
    ----------
    affine : :obj:`~numpy.ndarray`
        The affine transformation matrix.

    Returns
    -------
    :obj:`tuple`
        Extracted translation and rotation parameters.
    """

    translation = affine[:3, 3]
    rotation_rad = np.arctan2(
        [affine[2, 1], affine[0, 2], affine[1, 0]], [affine[2, 2], affine[0, 0], affine[1, 1]]
    )
    rotation_deg = np.rad2deg(rotation_rad)
    return *translation, *rotation_deg


def phase_encoding_vector(phase_encoding: str, affine: np.ndarray) -> np.ndarray:
    """
    The phase-encoding direction in world (RAS+) coordinates, as a unit vector.

    Parameters
    ----------
    phase_encoding : :obj:`str`
        The direction as BIDS writes ``PhaseEncodingDirection``: an axis of the data
        array (``i``, ``j`` or ``k``) and, with a trailing ``-``, the reverse sign.
    affine : :obj:`~numpy.ndarray`
        The voxel-to-RAS+ affine of the data.

    Returns
    -------
    :obj:`~numpy.ndarray`
        The direction, shape ``(3,)``.

    Examples
    --------
    >>> affine = np.diag([-2.0, 2.0, 2.0, 1.0])
    >>> phase_encoding_vector("j-", affine).tolist(), phase_encoding_vector("i", affine).tolist()
    ([0.0, -1.0, 0.0], [-1.0, 0.0, 0.0])

    """
    column = np.asarray(affine, dtype=float)[:3, "ijk".index(phase_encoding[0])]
    sign = -1.0 if phase_encoding.endswith("-") else 1.0
    return sign * column / np.linalg.norm(column) + 0.0  # no negative zeros


def split_phase_encoding(
    matrix: np.ndarray, direction: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split a mapping into head motion, then a field along phase encoding.

    A field of eddy currents linear in position moves points only along the
    phase-encoding direction :math:`u`, so a volume's mapping is :math:`W = E M`, with
    :math:`M` rigid and :math:`E = I - u s^T` (a scaling along :math:`u` and two shears;
    a translation along :math:`u` belongs to :math:`M`). The split is unique. Every
    :math:`v \\perp u` has :math:`v^T A = v^T R` (:math:`A`, :math:`R` the linear parts
    of :math:`W`, :math:`M`), so :math:`R`'s rows across :math:`u` are :math:`A`'s, and
    its row along :math:`u` is their cross product; then
    :math:`s^T = (u^T R - u^T A) R^T`, and :math:`M`'s translation is
    :math:`E^{-1} t_W`. A mapping outside this family is projected: its part across
    :math:`u` is taken as rigid.

    Parameters
    ----------
    matrix : :obj:`~numpy.ndarray`
        The mapping :math:`W`, a ``4 x 4`` RAS+ matrix.
    direction : :obj:`~numpy.ndarray`
        The phase-encoding direction :math:`u` in world coordinates (its sign does not
        matter).

    Returns
    -------
    motion, field : :obj:`~numpy.ndarray`
        :math:`M` and :math:`E`, each a ``4 x 4`` RAS+ matrix, with ``field @ motion``
        equal to ``matrix`` when it is in the family.

    Examples
    --------
    >>> motion = np.eye(4)
    >>> motion[:3, :3] = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    >>> motion[:3, 3] = [1.0, 2.0, 3.0]
    >>> field = np.eye(4)
    >>> field[1, :3] = [0.02, 0.97, -0.01]
    >>> found = split_phase_encoding(field @ motion, np.array([0.0, 1.0, 0.0]))
    >>> np.allclose(found[0], motion), np.allclose(found[1], field)
    (True, True)

    """
    matrix = np.asarray(matrix, dtype=float)
    u = np.asarray(direction, dtype=float)
    u = u / np.linalg.norm(u)
    # An orthonormal, right-handed basis {u, v1, v2}
    helper = np.eye(3)[np.argmin(np.abs(u))]
    v1 = np.cross(u, helper)
    v1 /= np.linalg.norm(v1)
    v2 = np.cross(u, v1)

    linear = matrix[:3, :3]
    across = np.stack([v1 @ linear, v2 @ linear])
    # The rows across u of a rotation are orthonormal: project them if they are not
    left, _, right = np.linalg.svd(across, full_matrices=False)
    across = left @ right
    basis = np.stack([u, v1, v2])
    rotation = basis.T @ np.vstack([np.cross(across[0], across[1]), across])

    slope = (u @ rotation - u @ linear) @ rotation.T
    field = np.eye(4)
    field[:3, :3] -= np.outer(u, slope)
    motion = np.eye(4)
    motion[:3, :3] = rotation
    motion[:3, 3] = np.linalg.solve(field[:3, :3], matrix[:3, 3])
    return motion, field
