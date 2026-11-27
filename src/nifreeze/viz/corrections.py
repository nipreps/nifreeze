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
How each volume of a DWI dataset is corrected, as an animated SVG.

For each volume, :func:`animate_corrections` plays, on three axial, three coronal and
three sagittal cuts:

1. the volume as found, labelled *original*;
2. where head motion sends its voxels, as a quiver;
3. the data travelling there, labelled *realigned* on arrival;
4. where the eddy-current field sends them next, as a second quiver;
5. the data travelling to the corrected position, labelled *corrected*.

Each volume's mapping :math:`W = E M` (:meth:`~nifreeze.data.base.BaseDataset.get_transform`)
is split into head motion :math:`M_z` and a field :math:`E_z` that moves nothing at a
point :math:`z` (the centre of the field of view, GEM's default), so that the shift the
field makes there counts as head motion.
:math:`E_z M_z = W`, so the last frame is the volume the command line writes.

Every frame of the first, third and fifth steps is an exact resampling.
In between, the browser moves the data continuously along the cut, with the 2D
mapping each step induces on it, while the part of the motion across the cut
dissolves from one frame into the next.

Drawn throughout:

* the brain mask at the reference position (dashed), and the mask moved by head motion
  (solid), which travels with the data into the reference;
* in each cut's corner, the b-vector projected onto the cut, which turns with the head
  during the realignment, as the b-vectors the command line writes do;
* across a cut, ``⊙`` points toward the viewer and ``⊗`` away from them.

The quivers are drawn ``magnify`` times their size (by default, ×10 for head motion and
×30 for the smaller eddy currents); the data travel at true scale.

The SVG plays on its own (as an ``<img>``, it loops the first volume).
Opened as a document (a browser tab, ``<object>`` or ``<iframe>``), it plays every
volume, with controls for speed (keys ``-`` and ``+``), pause (space), the previous and
next volume (arrow keys), and a progress bar that jumps to a volume.

"""

from __future__ import annotations

import base64
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path
from typing import Any

import attrs
import nibabel as nb
import nitransforms as nt
import numpy as np
from nitransforms.resampling import apply
from PIL import Image
from skimage.measure import approximate_polygon, find_contours

from nifreeze.data.base import ImageGrid
from nifreeze.data.dmri import DWI
from nifreeze.data.dmri.utils import realign_bvecs
from nifreeze.registration.gem import _zero_at, field_of_view_centre
from nifreeze.registration.utils import displacements_within_mask

ORIENTATIONS = ("axial", "coronal", "sagittal")
"""The rows of the panel, top to bottom."""
_VIEWS = {
    # The axis across the cut, then the axes drawn left to right and top to bottom,
    # each with its sign, in the canonical (RAS+) voxel frame
    "axial": (2, (0, 1), (1, -1)),  # right on the right, anterior up
    "coronal": (1, (0, 1), (2, -1)),  # right on the right, superior up
    "sagittal": (0, (1, -1), (2, -1)),  # anterior on the left, superior up
}
MARGIN = 3
"""Voxels kept around the brain."""
GAP = 4.0
"""Space between cuts (mm)."""
SPACING = 20.0
"""Space between the quiver's arrows (mm)."""
COMPASS = 7.0
"""Radius of the b-vector's compass (mm)."""
SMALLEST = 0.8
"""Arrows and marks smaller than this (mm, as drawn) are left out."""
LEFT, TOP = 34.0, 30.0
"""Room for the row labels and the header (mm)."""
COLORS = {
    "text": "#f0efec",
    "muted": "#8f8d88",
    "reference": "#eb6834",
    "moved": "#4fc3f7",
    "motion": "#ffd54f",
    "eddy": "#f06292",
    "bvec": "#81c784",
}
"""Colours of the text and of each overlay."""
SPEEDS = (0.25, 0.5, 1, 2, 4)
"""The playback speeds the controls offer."""


@attrs.frozen
class Timing:
    """
    How long each step of a volume's sequence lasts (s).

    >>> marks = Timing().marks(eddy=True)
    >>> marks["realign"], marks["realigned"], marks["correct"], marks["end"]
    (3.25, 5.25, 7.0, 10.0)
    >>> Timing().marks(eddy=False)["end"]
    6.25

    """

    original: float = 1.0
    """The volume as found."""
    motion: float = 2.0
    """The quiver of head motion."""
    realign: float = 2.0
    """The travel to the realigned position."""
    eddy: float = 1.5
    """The quiver of the eddy-current field."""
    correct: float = 2.0
    """The travel to the corrected position."""
    hold: float = 1.0
    """The corrected volume, before the next one."""
    fade: float = 0.25
    """A quiver fading in or out."""

    def marks(self, eddy: bool = True) -> dict[str, float]:
        """When each step starts (s), and the sequence's length (``"end"``)."""
        marks = {"motion": self.original}
        marks["realign"] = marks["motion"] + self.motion + self.fade
        marks["realigned"] = marks["realign"] + self.realign
        if eddy:
            marks["correct"] = marks["realigned"] + self.eddy + self.fade
            marks["corrected"] = marks["correct"] + self.correct
        else:
            marks["corrected"] = marks["realigned"]
        marks["end"] = marks["corrected"] + self.hold
        return marks


@attrs.frozen(eq=False)
class _View:
    """One cut: where it lies in the volume, and where it is drawn on the panel."""

    orientation: str
    normal: int
    """The canonical voxel axis across the cut."""
    index: int
    """The cut's index along ``normal``."""
    origin: np.ndarray
    """The cell's top-left corner on the panel (mm)."""
    size: np.ndarray
    """The cell's width and height (mm)."""
    shape: tuple[int, int]
    """The cut's voxels, top to bottom and left to right."""
    matrix: np.ndarray
    """``(2, 3)``, from the canonical frame (mm along the voxel axes) to the panel."""
    offset: np.ndarray
    """``(2,)``, the panel's position of the canonical frame's origin."""
    toward: np.ndarray
    """The unit vector of the canonical frame that points toward the viewer."""
    depth: float
    """The cut's position along ``normal`` in the canonical frame (mm)."""

    def points(self, xy: np.ndarray) -> np.ndarray:
        """The canonical frame's points (mm) under panel points ``(N, 2)``."""
        xy = np.atleast_2d(xy)
        points = np.zeros((len(xy), 3))
        for row in range(2):
            axis = int(np.flatnonzero(self.matrix[row])[0])
            points[:, axis] = (xy[:, row] - self.offset[row]) / self.matrix[row, axis]
        points[:, self.normal] = self.depth
        return points

    def project(self, mapping: np.ndarray) -> np.ndarray:
        """
        The 2D affine (``2 x 3``, on the panel) that a mapping of the canonical frame
        induces on the cut: the cut's points are mapped, and their part across the cut
        is dropped.
        """
        corners = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]) + self.origin
        mapped = nb.affines.apply_affine(mapping, self.points(corners))
        image = mapped @ self.matrix.T + self.offset
        linear = np.stack([image[1] - image[0], image[2] - image[0]], axis=1)
        return np.c_[linear, image[0] - linear @ corners[0]]

    def pixel(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The cut's rows and columns under panel points ``(N, 2)``."""
        local = (np.atleast_2d(xy) - self.origin) / self.size * np.array(self.shape[::-1])
        return local[:, 1].astype(int), local[:, 0].astype(int)

    def cut(self, volume: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
        """The cut of a canonical volume, cropped, as drawn (rows top to bottom)."""
        _, (ax, sx), (ay, sy) = _VIEWS[self.orientation]
        rest = [axis for axis in range(3) if axis != self.normal]
        data = np.take(volume, self.index, axis=self.normal)
        data = data[tuple(slice(low[axis], high[axis]) for axis in rest)]
        if rest.index(ay) != 0:
            data = data.T
        return data[:: -1 if sy < 0 else 1, :: -1 if sx < 0 else 1]

    def outline(self, mask: np.ndarray) -> str:
        """The outline of a mask's cut (as drawn), as SVG path data on the panel."""
        scale = self.size / np.array(self.shape[::-1])
        parts = []
        for contour in find_contours(np.pad(mask.astype(float), 1), 0.5):
            contour = approximate_polygon(contour - 1, tolerance=0.1)
            xy = (contour[:, ::-1] + 0.5) * scale + self.origin
            parts.append("M" + "L".join(f"{_n(x)} {_n(y)}" for x, y in xy))
        return "".join(parts)


@attrs.frozen(eq=False)
class _Layout:
    """The canonical frame, the crop, and where each cut sits on the panel."""

    orientation: np.ndarray
    """The reorientation to the closest canonical (RAS+) voxel order."""
    frame: np.ndarray
    """``4 x 4``, from the canonical frame (mm along the voxel axes) to world RAS+."""
    low: np.ndarray
    high: np.ndarray
    """The crop, in canonical voxels."""
    views: list[_View]
    pixel: float
    """The size of the sprite's pixels (mm)."""
    sprite: tuple[int, int]
    """The sprite's rows and columns."""
    corners: list[tuple[int, int]]
    """Each cut's top-left pixel in the sprite."""

    @property
    def width(self) -> float:
        return self.sprite[1] * self.pixel

    @property
    def height(self) -> float:
        return self.sprite[0] * self.pixel

    def canonical(self, data: np.ndarray) -> np.ndarray:
        """A volume in the canonical voxel order."""
        return nb.orientations.apply_orientation(np.asanyarray(data), self.orientation)

    def local(self, mapping: np.ndarray) -> np.ndarray:
        """A world RAS+ mapping, in the canonical frame."""
        return np.linalg.solve(self.frame, mapping @ self.frame)

    def direction(self, vector: np.ndarray) -> np.ndarray:
        """A world RAS+ direction, as a unit vector of the canonical frame."""
        local = np.linalg.solve(self.frame[:3, :3], vector)
        norm = np.linalg.norm(local)
        return local / norm if norm > 0 else local

    def sprite_of(self, cuts: Sequence[np.ndarray]) -> np.ndarray:
        """The panel's raster: each cut, resized to its cell, in place."""
        canvas = np.zeros(self.sprite, dtype="float32")
        for view, cut, (row, col) in zip(self.views, cuts, self.corners, strict=True):
            rows, cols = np.rint(view.size / self.pixel).astype(int)[::-1]
            if cut.shape != (rows, cols):
                cut = np.asarray(
                    Image.fromarray(np.asarray(cut, "float32")).resize(
                        (cols, rows), Image.Resampling.BILINEAR
                    )
                )
            canvas[row : row + rows, col : col + cols] = cut
        return canvas


def _layout(dataset: DWI, cuts: int) -> _Layout:
    orientation = nb.orientations.io_orientation(dataset.affine)
    shape = np.array(dataset.shape3d)
    affine = dataset.affine @ nb.orientations.inv_ornt_aff(orientation, shape)
    shape[orientation[:, 0].astype(int)] = shape.copy()
    zooms = np.sqrt((affine[:3, :3] ** 2).sum(0))
    frame = affine @ np.diag([*(1 / zooms), 1.0])

    if dataset.brainmask is not None:
        brain = nb.orientations.apply_orientation(dataset.brainmask, orientation) > 0
    else:
        brain = np.ones(shape, dtype=bool)
    found = np.argwhere(brain)
    first, last = found.min(0), found.max(0)
    low = np.maximum(first - MARGIN, 0)
    high = np.minimum(last + 1 + MARGIN, shape)

    pixel = float(zooms.min())
    gap = max(1, round(GAP / pixel))
    cells = {}
    for name in ORIENTATIONS:
        _, (ax, _), (ay, _) = _VIEWS[name]
        cells[name] = tuple(
            round((high[axis] - low[axis]) * zooms[axis] / pixel) for axis in (ay, ax)
        )
    column = max(cols for _, cols in cells.values())

    views, corners, top = [], [], 0
    for name in ORIENTATIONS:
        normal, (ax, sx), (ay, sy) = _VIEWS[name]
        rows, cols = cells[name]
        positions = np.linspace(first[normal], last[normal], cuts + 2)[1:-1]
        for c, index in enumerate(np.rint(positions).astype(int)):
            left = c * (column + gap) + (column - cols) // 2
            origin = np.array([LEFT + left * pixel, TOP + top * pixel])
            size = np.array([cols, rows]) * pixel
            matrix = np.zeros((2, 3))
            offset = origin.copy()
            for row, (axis, sign, extent) in enumerate(((ax, sx, cols), (ay, sy, rows))):
                scale = extent * pixel / ((high[axis] - low[axis]) * zooms[axis])
                matrix[row, axis] = sign * scale
                start = low[axis] - 0.5 if sign > 0 else high[axis] - 0.5
                offset[row] -= sign * scale * start * zooms[axis]
            toward = -np.cross(np.sign(matrix[0]), np.sign(matrix[1]))
            views.append(
                _View(
                    orientation=name,
                    normal=normal,
                    index=int(index),
                    origin=origin,
                    size=size,
                    shape=(int(high[ay] - low[ay]), int(high[ax] - low[ax])),
                    matrix=matrix,
                    offset=offset,
                    toward=toward,
                    depth=float(index * zooms[normal]),
                )
            )
            corners.append((top, left))
        top += rows + gap
    sprite = (top - gap, cuts * column + (cuts - 1) * gap)
    return _Layout(
        orientation=orientation,
        frame=frame,
        low=low,
        high=high,
        views=views,
        pixel=pixel,
        sprite=sprite,
        corners=corners,
    )


def _split(
    motion: np.ndarray, eddy: np.ndarray, point: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split a volume's mapping into head motion, then a field that moves nothing at a point.

    The field keeps its linear part, and the shift it made at ``point`` goes to the head
    motion, which stays rigid: ``field @ head == eddy @ motion``.

    >>> eddy = np.eye(4)
    >>> eddy[1, :3] = [0.01, 0.98, 0.0]
    >>> motion = np.eye(4)
    >>> motion[:3, 3] = [1.0, 2.0, 3.0]
    >>> head, field = _split(motion, eddy, np.array([0.0, 50.0, 0.0]))
    >>> np.allclose(field @ head, eddy @ motion), np.allclose(field @ [0, 50, 0, 1], [0, 50, 0, 1])
    (True, True)
    >>> np.allclose(head[:3, :3], np.eye(3)), head[1, 3].round(3)
    (True, np.float64(0.98))

    """
    field = _zero_at(eddy, point)
    return np.linalg.solve(field, eddy @ motion), field


def _n(value: float, digits: int = 1) -> str:
    """A number, short: no trailing zeros, no negative zero."""
    text = f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return "0" if text in ("", "-0") else text


def _matrix(affine: np.ndarray) -> str:
    """A ``2 x 3`` affine as a CSS or SVG ``matrix()``."""
    (a, c, e), (b, d, f) = affine
    return f"matrix({_n(a, 5)},{_n(b, 5)},{_n(c, 5)},{_n(d, 5)},{_n(e, 3)},{_n(f, 3)})"


def _inverse(affine: np.ndarray) -> np.ndarray:
    full = np.linalg.inv(np.vstack([affine, [0.0, 0.0, 1.0]]))
    return full[:2]


def _arrow(base: np.ndarray, vector: np.ndarray) -> str:
    """An arrow as SVG path data: the shaft from ``base``, then the head's two strokes."""
    length = float(np.hypot(*vector))
    if length < SMALLEST:
        return ""
    head = min(0.35 * length, 2.5)
    left, right = (
        head * np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) @ vector / length
        for a in np.radians([152, -152])
    )
    return (
        f"M{_n(base[0])} {_n(base[1])}l{_n(vector[0])} {_n(vector[1])}"
        f"l{_n(left[0])} {_n(left[1])}m{_n(-left[0])} {_n(-left[1])}"
        f"l{_n(right[0])} {_n(right[1])}"
    )


def _across(base: np.ndarray, size: float, cap: float) -> str:
    """A displacement across the cut as SVG path data: ``⊙`` toward the viewer, ``⊗`` away."""
    radius = min(abs(size), cap)
    if radius < SMALLEST:
        return ""
    x, y = base
    circle = (
        f"M{_n(x + radius)} {_n(y)}a{_n(radius)} {_n(radius)} 0 1 0 {_n(-2 * radius)} 0"
        f"a{_n(radius)} {_n(radius)} 0 1 0 {_n(2 * radius)} 0"
    )
    if size > 0:
        return circle + f"M{_n(x)} {_n(y)}h0"
    s, d = _n(x - radius / 2), _n(y - radius / 2)
    return (
        circle + f"M{s} {d}l{_n(radius)} {_n(radius)}m0 {_n(-radius)}l{_n(-radius)} {_n(radius)}"
    )


def _quiver(view: _View, mapping: np.ndarray, mask: np.ndarray, magnify: float) -> str:
    """
    Where a mapping of the canonical frame sends the cut's points inside a mask, as SVG
    path data: an arrow along the cut, a mark across it, both ``magnify`` times their size.
    """
    paths = []
    for axis, extent in enumerate(view.size):
        count = max(1, int(extent // SPACING))
        start = view.origin[axis] + (extent - (count - 1) * SPACING) / 2
        if axis == 0:
            xs = start + SPACING * np.arange(count)
        else:
            ys = start + SPACING * np.arange(count)
    grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    rows, cols = view.pixel(grid)
    inside = mask[np.clip(rows, 0, mask.shape[0] - 1), np.clip(cols, 0, mask.shape[1] - 1)]
    grid = grid[inside > 0.5]
    if not len(grid):
        return ""
    points = view.points(grid)
    moves = nb.affines.apply_affine(mapping, points) - points
    along = magnify * moves @ view.matrix.T
    across = magnify * moves @ view.toward
    for base, vector, size in zip(grid, along, across, strict=True):
        paths.append(_arrow(base, vector) + _across(base, size, 0.45 * SPACING))
    return "".join(paths)


def _turn(centre: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """The rotation and scaling about ``centre`` that takes vector ``start`` to ``end``."""
    if np.hypot(*start) < 1e-3 or np.hypot(*end) < 1e-3:
        return np.c_[np.eye(2), np.zeros(2)]
    ratio = complex(*end) / complex(*start)
    linear = np.array([[ratio.real, -ratio.imag], [ratio.imag, ratio.real]])
    return np.c_[linear, centre - linear @ centre]


def _compass(view: _View, start: np.ndarray, end: np.ndarray) -> str:
    """The b-vector on a cut: its projection as an arrow, and its part across the cut."""
    centre = view.origin + np.array([view.size[0] - COMPASS - 1.5, COMPASS + 1.5])
    tips = [COMPASS * np.sign(view.matrix) @ v for v in (start, end)]
    across = _across(centre, 0.6 * COMPASS * float(start @ view.toward), COMPASS)
    return (
        f'<g class="turn" style="--turn:{_matrix(_turn(centre, *tips))}">'
        f'<path class="b" d="{_arrow(centre, tips[0])}"/></g>'
        + (f'<path class="b" d="{across}"/>' if across else "")
    )


def _encode(image: np.ndarray, window: float, quality: int) -> str:
    """A grayscale JPEG of an image scaled to ``[0, window]``, base64-encoded."""
    scaled = np.clip(np.nan_to_num(image) / max(window, 1e-6), 0, 1)
    buffer = BytesIO()
    Image.fromarray((255 * scaled).astype("uint8")).save(buffer, "JPEG", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode()


def _resample(matrix: np.ndarray, image: nb.Nifti1Image, order: int) -> np.ndarray:
    """An image resampled on its own grid at the points ``matrix`` maps it to."""
    grid = ImageGrid(shape=image.shape[:3], affine=image.affine)
    moved = apply(nt.linear.Affine(matrix, reference=grid), image, order=order)
    return np.asanyarray(moved.dataobj)


def _frames(dataset: DWI, index: int, head: np.ndarray) -> dict[str, np.ndarray]:
    """
    A volume as found (``"o"``), realigned by head motion (``"r"``) and, with eddy
    currents, corrected (``"c"``), resampled as the command line's output is
    (:func:`~nifreeze.data.base.to_nifti`).
    """
    data = np.asanyarray(dataset.dataobj[index])
    image = nb.Nifti1Image(data, dataset.affine, dataset.datahdr)
    dtype = dataset.dataobj.dtype
    frames = {"o": data, "r": np.asanyarray(_resample(head, image, 3), dtype=dtype)}
    mapping = dataset.get_transform(index)
    if dataset.eddy is not None and mapping is not None:
        frames["c"] = np.asanyarray(_resample(mapping, image, 3), dtype=dtype)
    return frames


def _volume(
    dataset: DWI,
    index: int,
    layout: _Layout,
    point: np.ndarray,
    magnify: tuple[float, float],
    quality: int,
) -> str:
    """One volume's slot: its three frames, quivers, outlines, compasses and labels."""
    motion = np.eye(4) if dataset.motion is None else dataset.motion[index]
    eddy = None if dataset.eddy is None else dataset.eddy[index]
    head, field = _split(motion, np.eye(4) if eddy is None else eddy, point)
    mapping = field @ head
    frames = _frames(dataset, index, head)
    data = frames["o"]
    brain = (
        np.ones(dataset.shape3d, dtype="float32")
        if dataset.brainmask is None
        else dataset.brainmask.astype("float32")
    )
    moved = _resample(np.linalg.inv(head), nb.Nifti1Image(brain, dataset.affine), 1)

    window = float(np.percentile(data[brain > 0], 99.5)) if brain.any() else float(data.max())
    low, high = layout.low, layout.high
    images = []
    for key, frame in frames.items():
        volume = layout.canonical(np.asarray(frame, dtype="float32"))
        sprite = layout.sprite_of([view.cut(volume, low, high) for view in layout.views])
        images.append(
            f'<image id="v{index}{key}" x="{_n(LEFT)}" y="{_n(TOP)}" '
            f'width="{_n(layout.width)}" height="{_n(layout.height)}" '
            'preserveAspectRatio="none" '
            f'href="data:image/jpeg;base64,{_encode(sprite, window, quality)}"/>'
        )

    reference = layout.canonical(brain)
    moved = layout.canonical(moved)
    back = layout.local(np.linalg.inv(head))  # where head motion sends the volume's points
    rest = layout.local(np.linalg.inv(np.linalg.solve(head, mapping)))  # and GEM's field
    bvec = dataset.gradients[index, :3]
    start = layout.direction(bvec)
    end = layout.direction(realign_bvecs(bvec, head))

    cells = []
    for v, view in enumerate(layout.views):
        travel = view.project(back)
        stack = (
            f'<g class="go-m" style="--end:{_matrix(travel)}">'
            f'<use class="im-o" href="#v{index}o"/>'
            f'<g transform="{_matrix(_inverse(travel))}"><use class="im-r" href="#v{index}r"/></g>'
            "</g>"
        )
        if eddy is not None:
            correct = view.project(rest)
            stack = (
                f'<g class="go-e" style="--end:{_matrix(correct)}">{stack}'
                f'<g transform="{_matrix(_inverse(correct))}">'
                f'<use class="im-c" href="#v{index}c"/></g></g>'
            )
        inside = view.cut(moved, low, high)
        quivers = f'<path class="q-m" d="{_quiver(view, back, inside, magnify[0])}"/>'
        if eddy is not None:
            inside = view.cut(reference, low, high)
            quivers += f'<path class="q-e" d="{_quiver(view, rest, inside, magnify[1])}"/>'
        cells.append(
            f'<g clip-path="url(#cell{v})">{stack}'
            f'<g class="go-m" style="--end:{_matrix(travel)}">'
            f'<path class="mc" d="{view.outline(view.cut(moved, low, high) > 0.5)}"/></g>'
            f"{quivers}{_compass(view, start, end)}</g>"
        )

    mask_img = nb.Nifti1Image(brain.astype("uint8"), dataset.affine)
    sizes = [
        np.median(displacements_within_mask(mask_img, nt.linear.Affine(head))),
        np.median(
            displacements_within_mask(mask_img, nt.linear.Affine(mapping), nt.linear.Affine(head))
        ),
    ]
    bval = float(dataset.gradients[index, 3])
    title = f"Volume {index} of {len(dataset)} · b = {np.round(bval, -1):g} s/mm²"
    detail = f"Median shift in the brain: head motion {sizes[0]:.2f} mm"
    if eddy is not None:
        detail += f", eddy currents {sizes[1]:.2f} mm"
    right = LEFT + layout.width
    labels = [
        f'<text class="lab-o state" x="{_n(right)}" y="12">original</text>',
        f'<text class="lab-r state" x="{_n(right)}" y="12">realigned</text>'
        if eddy is not None
        else "",
        f'<text class="lab-c state" x="{_n(right)}" y="12">corrected</text>',
    ]
    return (
        f'<g class="slot" data-index="{index}"><defs>{"".join(images)}</defs>'
        f'<text x="{_n(LEFT)}" y="12" class="title">{title}</text>'
        f'<text x="{_n(LEFT)}" y="21" class="detail">{detail}</text>'
        + "".join(labels)
        + "".join(cells)
        + "</g>"
    )


def _keyframes(name: str, end: float, stops: Sequence[tuple[float, str, bool]]) -> str:
    """CSS keyframes from stops ``(time, declarations, ease into the next)``, held to 100%."""
    stops = [(0.0, stops[0][1], False), *stops, (end, stops[-1][1], False)]
    frames = []
    for time, declarations, ease in stops:
        timing = " animation-timing-function: ease-in-out;" if ease else ""
        frames.append(f"{100 * min(time, end) / end:.3f}% {{ {declarations};{timing} }}")
    return f"@keyframes {name} {{ " + " ".join(frames) + " }"


def _style(timing: Timing, eddy: bool) -> str:
    marks = timing.marks(eddy)
    end, fade, quick = marks["end"], timing.fade, 0.15
    hidden, shown = "opacity: 0", "opacity: 1"
    still, moved, turned = "transform: none", "transform: var(--end)", "transform: var(--turn)"
    off_o = marks["realign"] - fade
    animations = {
        "lab-o": [(off_o, shown, False), (off_o + quick, hidden, False)],
        "lab-c": [(marks["corrected"], hidden, False), (marks["corrected"] + quick, shown, False)],
        "q-m": [
            (marks["motion"], hidden, False),
            (marks["motion"] + fade, shown, False),
            (off_o, shown, False),
            (marks["realign"], hidden, False),
        ],
        "go-m": [(marks["realign"], still, True), (marks["realigned"], moved, False)],
        "turn": [(marks["realign"], still, True), (marks["realigned"], turned, False)],
        "im-o": [(marks["realigned"], shown, False), (marks["realigned"] + 0.01, hidden, False)],
        "im-r": [(marks["realign"], hidden, True), (marks["realigned"], shown, False)],
        "mc": [
            (marks["realigned"] - 0.3 * timing.realign, shown, False),
            (marks["realigned"], hidden, False),
        ],
    }
    if eddy:
        off_r = marks["correct"] - fade
        animations |= {
            "lab-r": [
                (marks["realigned"], hidden, False),
                (marks["realigned"] + quick, shown, False),
                (off_r, shown, False),
                (off_r + quick, hidden, False),
            ],
            "q-e": [
                (marks["realigned"], hidden, False),
                (marks["realigned"] + fade, shown, False),
                (off_r, shown, False),
                (marks["correct"], hidden, False),
            ],
            "go-e": [(marks["correct"], still, True), (marks["corrected"], moved, False)],
            "im-c": [(marks["correct"], hidden, True), (marks["corrected"], shown, False)],
        }
    rules = [
        ".slot { display: none; }",
        f".slot.on {{ display: inline; animation: cycle {end:g}s linear both infinite; }}",
        "@keyframes cycle { from { opacity: 1; } to { opacity: 1; } }",
        ".js .on, .js .on * { animation-iteration-count: 1 !important; }",
        "image { image-rendering: optimizeQuality; }",
        f"text {{ fill: {COLORS['text']}; }}",
        ".title { font-size: 7px; } .detail { font-size: 5px; fill: " + COLORS["muted"] + "; }",
        ".state { font-size: 10px; text-anchor: end; font-weight: bold; }",
        f".lab-o {{ fill: {COLORS['text']}; }} .lab-r {{ fill: {COLORS['moved']}; }}"
        f" .lab-c {{ fill: {COLORS['bvec']}; }}",
        "path { fill: none; stroke-linecap: round; stroke-linejoin: round; }",
        f".mc {{ stroke: {COLORS['moved']}; stroke-width: 0.6; }}",
        f".ref {{ stroke: {COLORS['reference']}; stroke-width: 0.6; stroke-dasharray: 2 1.5; }}",
        f".q-m {{ stroke: {COLORS['motion']}; stroke-width: 0.6; }}",
        f".q-e {{ stroke: {COLORS['eddy']}; stroke-width: 0.6; }}",
        f".b {{ stroke: {COLORS['bvec']}; stroke-width: 0.8; }}",
        f".dial {{ stroke: {COLORS['muted']}; stroke-width: 0.3; }}",
        ".controls { display: none; } .js .controls { display: inline; }",
        ".button { cursor: pointer; } .button rect { fill: #2b2a28; stroke: #5d5b57;"
        " stroke-width: 0.3; } .button:hover rect { fill: #3d3b38; }",
        ".button text { font-size: 4.5px; text-anchor: middle; }",
        ".legend { font-size: 4.5px; fill: " + COLORS["muted"] + "; }",
    ]
    for name, stops in animations.items():
        rules.append(_keyframes(name, end, stops))
        rules.append(f".on .{name} {{ animation: {name} {end:g}s linear both infinite; }}")
    return "\n".join(rules)


_SCRIPT = """
(function () {
  const svg = document.documentElement;
  svg.classList.add('js');
  const slots = Array.from(document.querySelectorAll('.slot'));
  const speeds = %(speeds)s;
  const track = document.getElementById('track');
  const done = document.getElementById('done');
  const speed = document.getElementById('speed');
  const pause = document.getElementById('pause');
  const [left, width] = [Number(track.getAttribute('x')), Number(track.getAttribute('width'))];
  let current = 0, rate = speeds.indexOf(1), paused = false;

  function update() {
    for (const a of slots[current].getAnimations({subtree: true})) {
      a.playbackRate = speeds[rate];
      if (paused) { a.pause(); } else if (a.playState === 'paused') { a.play(); }
    }
    speed.textContent = speeds[rate] + '\\u00d7';
    pause.textContent = paused ? '\\u25b6 play' : '\\u275a\\u275a pause';
    done.setAttribute('width', width * (current + 1) / slots.length);
  }
  function show(k) {
    // Animations the controls have touched outlive their element's display: end them
    for (const a of slots[current].getAnimations({subtree: true})) a.cancel();
    slots[current].classList.remove('on');
    current = (k + slots.length) %% slots.length;
    slots[current].classList.add('on');
    update();
  }
  const actions = {
    previous: () => show(current - 1),
    next: () => show(current + 1),
    pause: () => { paused = !paused; update(); },
    slower: () => { rate = Math.max(0, rate - 1); update(); },
    faster: () => { rate = Math.min(speeds.length - 1, rate + 1); update(); },
  };
  for (const button of document.querySelectorAll('.button')) {
    button.addEventListener('click', actions[button.dataset.action]);
  }
  svg.addEventListener('animationend', (event) => {
    if (event.target === slots[current] && event.animationName === 'cycle') show(current + 1);
  });
  document.addEventListener('keydown', (event) => {
    const keys = {' ': 'pause', 'ArrowLeft': 'previous', 'ArrowRight': 'next',
                  '-': 'slower', '_': 'slower', '+': 'faster', '=': 'faster'};
    if (keys[event.key]) { event.preventDefault(); actions[keys[event.key]](); }
  });
  for (const bar of [track, done]) {
    bar.addEventListener('click', (event) => {
      const point = svg.createSVGPoint();
      [point.x, point.y] = [event.clientX, event.clientY];
      const x = point.matrixTransform(svg.getScreenCTM().inverse()).x;
      show(Math.min(slots.length - 1, Math.floor((x - left) / width * slots.length)));
    });
  }
  update();
})();
"""


def _controls(top: float, width: float) -> str:
    buttons = [
        ("previous", "◀ previous", 0),
        ("pause", "❚❚ pause", 1),
        ("next", "next ▶", 2),
        ("slower", "− slower", 4),
        ("faster", "faster +", 5),
    ]
    parts = ['<g class="controls">']
    for action, label, slot in buttons:
        x = LEFT + slot * 24
        text_id = ' id="pause"' if action == "pause" else ""
        parts.append(
            f'<g class="button" data-action="{action}"><rect x="{_n(x)}" y="{_n(top)}" '
            f'width="22" height="8" rx="1.5"/><text{text_id} x="{_n(x + 11)}" '
            f'y="{_n(top + 5.6)}">{label}</text></g>'
        )
    parts.append(
        f'<text id="speed" class="legend" x="{_n(LEFT + 3 * 24 + 11)}" y="{_n(top + 5.6)}" '
        'text-anchor="middle">1×</text>'
    )
    bar = top + 12
    parts.append(
        f'<rect id="track" class="button" x="{_n(LEFT)}" y="{_n(bar)}" width="{_n(width)}" '
        'height="3" fill="#2b2a28"/>'
        f'<rect id="done" class="button" x="{_n(LEFT)}" y="{_n(bar)}" width="0" height="3" '
        f'fill="{COLORS["moved"]}"/>'
    )
    parts.append("</g>")
    return "".join(parts)


def _legend(top: float, magnify: tuple[float, float], eddy: bool) -> str:
    entries = [
        ("ref", "brain mask, reference"),
        ("mc", "brain mask, moved by head motion"),
        ("q-m", f"head motion (arrows ×{magnify[0]:g})"),
    ]
    if eddy:
        entries.append(("q-e", f"eddy currents (arrows ×{magnify[1]:g})"))
    entries.append(("b", "b-vector"))
    parts, x = [], LEFT
    for style, label in entries:
        parts.append(
            f'<path class="{style}" style="opacity: 1; animation: none" '
            f'd="M{_n(x)} {_n(top - 1.5)}h7"/>'
            f'<text class="legend" x="{_n(x + 9)}" y="{_n(top)}">{label}</text>'
        )
        x += 11 + 2.3 * len(label)
    parts.append(
        f'<text class="legend" x="{_n(x)}" y="{_n(top)}">'
        "⊙ toward you, ⊗ away (across the cut)</text>"
    )
    return "".join(parts)


def animate_corrections(
    dataset: DWI,
    out: str | Path | None = None,
    *,
    volumes: Sequence[int] | None = None,
    cuts: int = 3,
    magnify: float | tuple[float, float] = (10.0, 30.0),
    zero_point: Any = None,
    timing: Timing | None = None,
    quality: int = 75,
) -> str:
    """
    Animate how each volume of a DWI dataset is corrected, as an SVG.

    Parameters
    ----------
    dataset : :obj:`~nifreeze.data.dmri.DWI`
        The dataset with its estimates (``motion`` and, if set, ``eddy``), such as the
        command line's ``--write-hdf5`` output.
    out : :obj:`os.PathLike`, optional
        Where to write the SVG.
    volumes : sequence of :obj:`int`, optional
        The volumes to play, in this order (all of them by default).
    cuts : :obj:`int`, optional
        Cuts per orientation, spread evenly across the brain.
    magnify : :obj:`float` or pair of :obj:`float`, optional
        How many times larger the quivers are drawn than the displacements: one factor
        for both, or one for head motion, then one for the eddy currents.
    zero_point : array-like, optional
        The point (RAS+, mm) where the eddy-current field moves nothing; its shift there
        counts as head motion. By default, the centre of the field of view (as GEM's).
    timing : :obj:`Timing`, optional
        How long each step lasts.
    quality : :obj:`int`, optional
        The frames' JPEG quality.

    Returns
    -------
    :obj:`str`
        The SVG document.

    """
    timing = timing or Timing()
    eddy = dataset.eddy is not None
    indices = range(len(dataset)) if volumes is None else [int(v) for v in volumes]
    point = (
        field_of_view_centre(dataset.affine, dataset.shape3d)
        if zero_point is None
        else np.asarray(zero_point, dtype=float)
    )
    layout = _layout(dataset, cuts)
    motion_factor, eddy_factor = np.broadcast_to(np.asarray(magnify, dtype=float), (2,))
    factors = (float(motion_factor), float(eddy_factor))
    slots = [_volume(dataset, k, layout, point, factors, quality) for k in indices]
    slots[0] = slots[0].replace('class="slot"', 'class="slot on"', 1)

    brain = layout.canonical(dataset.brainmask) if dataset.brainmask is not None else None
    static = []
    for v, view in enumerate(layout.views):
        x, y = view.origin
        w, h = view.size
        static.append(
            f'<clipPath id="cell{v}"><rect x="{_n(x)}" y="{_n(y)}" width="{_n(w)}" '
            f'height="{_n(h)}"/></clipPath>'
        )
    overlays = []
    for view in layout.views:
        if brain is not None:
            overlays.append(
                f'<path class="ref" d="{view.outline(view.cut(brain, layout.low, layout.high))}"/>'
            )
        centre = view.origin + np.array([view.size[0] - COMPASS - 1.5, COMPASS + 1.5])
        overlays.append(
            f'<circle class="dial" cx="{_n(centre[0])}" cy="{_n(centre[1])}" '
            f'r="{_n(COMPASS)}" fill="none"/>'
            f'<text class="legend" x="{_n(view.origin[0] + 1.5)}" '
            f'y="{_n(view.origin[1] + view.size[1] - 1.5)}">'
            f"{'xyz'[view.normal]} = {view.index}</text>"
        )
    for name in ORIENTATIONS:
        rows = [view for view in layout.views if view.orientation == name]
        middle = rows[0].origin[1] + rows[0].size[1] / 2
        overlays.append(
            f'<text class="title" x="{_n(LEFT - 3)}" y="{_n(middle + 2)}" '
            f'text-anchor="end">{name.capitalize()}</text>'
        )

    bottom = TOP + layout.height
    width = LEFT + layout.width + 4
    height = bottom + 36
    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {_n(width)} {_n(height)}" font-family="sans-serif">',
        f"<style>\n{_style(timing, eddy)}\n</style>",
        f"<defs>{''.join(static)}</defs>",
        f'<rect width="{_n(width)}" height="{_n(height)}" fill="black"/>',
        *slots,
        *overlays,
        _legend(bottom + 9, factors, eddy),
        _controls(bottom + 14, layout.width),
        f"<script><![CDATA[{_SCRIPT % {'speeds': list(SPEEDS)}}]]></script>",
        "</svg>",
    ]
    document = "\n".join(svg) + "\n"
    if out is not None:
        Path(out).write_text(document, encoding="utf-8")
    return document
