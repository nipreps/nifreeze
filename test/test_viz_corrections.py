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
"""The animation of how each volume is corrected."""

import re
import xml.etree.ElementTree as ET

import nibabel as nb
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from nifreeze.data.base import to_nifti
from nifreeze.data.dmri import DWI
from nifreeze.data.dmri.utils import realign_bvecs
from nifreeze.viz import corrections
from nifreeze.viz.corrections import COMPASS, Timing, animate_corrections

SHAPE = (20, 24, 16)
PE = np.array([0.0, 1.0, 0.0])


def _rigid(angles, shift) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_euler("xyz", angles, degrees=True).as_matrix()
    matrix[:3, 3] = shift
    return matrix


def _field(slopes) -> np.ndarray:
    field = np.eye(4)
    field[:3, :3] -= np.outer(PE, slopes)
    return field


def _dwi(affine=None, eddy=True) -> DWI:
    """A smooth head in an ellipsoidal brain, six volumes with motion and eddy currents."""
    rng = np.random.default_rng(1234)
    affine = np.diag([2.0, 2.0, 2.5, 1.0]) if affine is None else affine
    affine = affine.copy()
    affine[:3, 3] = -affine[:3, :3] @ ((np.array(SHAPE) - 1) / 2)
    grid = np.stack(np.meshgrid(*[np.linspace(-1, 1, n) for n in SHAPE], indexing="ij"))
    radius = np.sqrt((grid**2).sum(0))
    head = np.exp(-4 * radius**2) + 0.1 * np.cos(6 * grid[0])
    bvecs = rng.standard_normal((6, 3))
    bvecs /= np.linalg.norm(bvecs, axis=1, keepdims=True)
    dwi = DWI(
        dataobj=np.stack([head * (1 + 0.1 * k) for k in range(6)]).astype("float32"),
        affine=affine,
        brainmask=radius < 0.8,
        gradients=np.column_stack((bvecs, np.repeat([1000.0, 2000.0, 3000.0], 2))),
        phase_encoding="j",
    )
    dwi.motion = np.stack(
        [_rigid([1, -2, 3], [0.5, -1.0, 0.8]), np.eye(4), _rigid([0, 0, 5], [1.5, 0, 0])] * 2
    )
    if eddy:
        dwi.eddy = np.stack(
            [_field([0.01, -0.02, 0.005]), _field([0, 0, 0]), _field([-0.01, 0.015, 0])] * 2
        )
    return dwi


def test_split_keeps_the_mapping():
    """Head motion stays rigid, the field moves nothing at the point, and they compose."""
    motion = _rigid([4, -3, 2], [1.0, 2.0, -3.0])
    eddy = _field([0.02, -0.03, 0.01])
    point = np.array([5.0, -12.0, 30.0])

    head, field = corrections._split(motion, eddy, point)

    np.testing.assert_allclose(field @ head, eddy @ motion, atol=1e-12)
    np.testing.assert_allclose(head[:3, :3] @ head[:3, :3].T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(field[:3, :3], eddy[:3, :3])
    np.testing.assert_allclose(field @ [*point, 1.0], [*point, 1.0], atol=1e-12)


@pytest.mark.parametrize("eddy", [True, False])
def test_the_frames_are_the_corrections(eddy):
    """
    The middle frame is the volume read through the field alone, and the last one the
    volume the command line writes, each as the command line resamples it.
    """
    dwi = _dwi(eddy=eddy)
    written = np.asanyarray(to_nifti(dwi).dataobj)
    assert dwi.motion is not None
    eddies = np.repeat(np.eye(4)[None], len(dwi), 0) if dwi.eddy is None else dwi.eddy
    point = np.array([3.0, -4.0, 2.0])
    fields = np.stack(
        [corrections._split(m, e, point)[1] for m, e in zip(dwi.motion, eddies, strict=True)]
    )
    # The same volumes, with no head motion and the fields as their eddy currents
    unwarped = _dwi(eddy=eddy)
    unwarped.motion, unwarped.eddy = None, fields
    in_place = np.asanyarray(to_nifti(unwarped).dataobj)

    for index in range(len(dwi)):
        frames = corrections._frames(dwi, index, fields[index])
        assert list(frames) == (["o", "e", "r"] if eddy else ["o", "r"])
        np.testing.assert_array_equal(frames["o"], dwi.dataobj[index])
        if eddy:
            np.testing.assert_array_equal(frames["e"], in_place[..., index])
        np.testing.assert_array_equal(frames["r"], written[..., index])


def test_the_field_acts_where_the_volume_was_acquired():
    """
    The field moves voxels along phase encoding in the volume as acquired, however the
    head turned: across the coronal cuts only, and up and down the axial ones.
    """
    dwi = _dwi()
    assert dwi.motion is not None
    dwi.motion[0] = _rigid([0, 0, 20], [1.0, -2.0, 0.5])  # a large turn about z
    layout = corrections._layout(dwi, cuts=3)
    slot = corrections._volume(dwi, 0, layout, np.zeros(3), (10.0, 30.0), 75)
    paths = re.findall(r'class="q-e" d="([^"]*)"', slot)
    assert len(paths) == len(layout.views)

    for view, path in zip(layout.views, paths, strict=True):
        arrows = _arrows(path)
        if view.orientation == "coronal":
            assert len(arrows) == 0 and "a" in path
        elif view.orientation == "axial":
            assert len(arrows) > 0
            np.testing.assert_allclose(arrows[:, 0], 0.0, atol=0.06)


def test_the_layout_is_canonical():
    """Cuts are drawn from the closest canonical orientation, whatever the input's."""
    flipped = np.array([[0.0, 0, -2, 0], [2, 0, 0, 0], [0, 2.5, 0, 0], [0, 0, 0, 1]])
    dwi = _dwi(affine=flipped)
    layout = corrections._layout(dwi, cuts=3)

    image = nb.as_closest_canonical(nb.Nifti1Image(np.asarray(dwi.dataobj[0]), dwi.affine))
    np.testing.assert_array_equal(layout.canonical(dwi.dataobj[0]), image.get_fdata())
    zooms = np.array(image.header.get_zooms()[:3])
    voxel = np.array([3, 4, 5])
    np.testing.assert_allclose(
        layout.frame @ [*(voxel * zooms), 1.0], image.affine @ [*voxel, 1.0], atol=1e-9
    )
    assert [view.orientation for view in layout.views] == [
        name for name in corrections.ORIENTATIONS for _ in range(3)
    ]


def _view(layout, orientation):
    return next(view for view in layout.views if view.orientation == orientation)


def test_the_projection_of_a_mapping_onto_a_cut():
    """Along the cut, a mapping moves the drawing; across it, nothing."""
    layout = corrections._layout(_dwi(), cuts=3)
    axial = _view(layout, "axial")
    across = np.eye(4)
    across[2, 3] = 3.0
    np.testing.assert_allclose(axial.project(across), np.c_[np.eye(2), np.zeros(2)], atol=1e-9)

    shift = np.eye(4)
    shift[:3, 3] = [1.0, 2.0, 0.0]
    scale = np.abs(axial.matrix).sum(1)  # mm on the panel per mm of the volume
    np.testing.assert_allclose(axial.project(shift)[:, 2], scale * [1.0, -2.0], atol=1e-9)

    turn = _rigid([0, 0, 10], [0, 0, 0])
    linear = axial.project(turn)[:, :2]
    np.testing.assert_allclose(linear @ linear.T, np.diag(scale**2), atol=1e-9)
    assert np.linalg.det(linear) > 0


def _arrows(path: str) -> np.ndarray:
    """The shafts of a quiver's arrows, ``(N, 2)``."""
    found = re.findall(r"M[-\d.]+ [-\d.]+l([-\d.]+) ([-\d.]+)l", path)
    return np.array(found, dtype=float).reshape(-1, 2)


def test_the_quiver_of_a_translation():
    """A shift along x: equal arrows on axial and coronal cuts, marks across sagittal ones."""
    dwi = _dwi()
    layout = corrections._layout(dwi, cuts=3)
    shift = np.eye(4)
    shift[0, 3] = 0.5  # toward the right
    mask = np.ones((100, 100))

    for name in ("axial", "coronal"):
        view = _view(layout, name)
        path = corrections._quiver(view, shift, mask, magnify=10.0)
        arrows = _arrows(path)
        assert len(arrows) > 0 and "a" not in path
        np.testing.assert_allclose(arrows, [[5.0, 0.0]] * len(arrows), atol=0.06)

    sagittal = corrections._quiver(_view(layout, "sagittal"), shift, mask, magnify=10.0)
    assert len(_arrows(sagittal)) == 0
    assert "a5 5 0 1 0" in sagittal  # circles of 5 mm
    assert "h0" not in sagittal  # right is away from the viewer of a sagittal cut: crosses


def test_the_compass_turns_with_the_head():
    """The b-vector turns with the head during the realignment, as the output b-vectors."""
    layout = corrections._layout(_dwi(), cuts=3)
    axial = _view(layout, "axial")
    quarter = _rigid([0, 0, 90], [0, 0, 0])
    start = np.array([1.0, 0.0, 0.0])
    end = realign_bvecs(start, quarter)  # the head's -y

    compass = corrections._compass(axial, start, end)
    tip = re.search(r'd="M[-\d.]+ [-\d.]+l([-\d.]+) ([-\d.]+)l', compass)
    turn = re.search(r"--turn:matrix\(([^)]*)\)", compass)
    assert tip is not None and turn is not None
    np.testing.assert_allclose(np.array(tip.groups(), dtype=float), [COMPASS, 0.0], atol=0.06)
    a, b, c, d, *_ = map(float, turn[1].split(","))
    # The head's -y is drawn downward: a quarter turn clockwise on the screen
    np.testing.assert_allclose([a, b, c, d], [0.0, 1.0, -1.0, 0.0], atol=1e-4)


def _opacity(style: str, name: str, time: float, end: float) -> float:
    """An element's opacity at a time (s), held from the last stop before it."""
    line = next(row for row in style.splitlines() if row.startswith(f"@keyframes {name} "))
    stops = [(float(p), float(o)) for p, o in re.findall(r"([\d.]+)% \{ opacity: ([\d.]+);", line)]
    return [o for p, o in stops if p <= 100 * time / end][-1]


# What shows at times away from any transition: the original, the field's quiver, the
# eddy-corrected volume under head motion's quiver, and the realigned volume
_SHOWN = {
    True: {
        0.5: {"lab-o", "im-o", "mo"},
        2.0: {"lab-o", "im-o", "mo", "q-e"},
        6.0: {"lab-e", "im-e", "mc", "q-m"},
        9.5: {"lab-r", "im-r"},
    },
    False: {
        0.5: {"lab-o", "im-o", "mc"},
        2.0: {"lab-o", "im-o", "mc", "q-m"},
        5.75: {"lab-r", "im-r"},
    },
}


@pytest.mark.parametrize("eddy", [True, False])
def test_the_steps_play_in_order(eddy):
    """The field is undone first, then head motion, and the solid outline fades on arrival."""
    timing = Timing()
    style = corrections._style(timing, eddy)
    end = timing.marks(eddy)["end"]
    names = {"lab-o", "lab-r", "im-o", "im-r", "mc", "q-m"}
    names |= {"lab-e", "im-e", "mo", "q-e"} if eddy else set()

    for time, shown in _SHOWN[eddy].items():
        found = {name for name in names if _opacity(style, name, time, end) == 1}
        assert found == shown, time


@pytest.mark.parametrize("eddy", [True, False])
def test_the_document(tmp_path, eddy):
    """One slot per volume, in the order given, the first one playing, labels and script."""
    dwi = _dwi(eddy=eddy)
    out = tmp_path / "corrections.svg"

    svg = animate_corrections(dwi, out, volumes=[2, 0], timing=Timing(hold=0.5))

    assert out.read_text(encoding="utf-8") == svg
    root = ET.fromstring(svg)
    ns = {"svg": "http://www.w3.org/2000/svg"}
    slots = [g for g in root.iter("{http://www.w3.org/2000/svg}g") if "slot" in g.get("class", "")]
    assert [g.get("data-index") for g in slots] == ["2", "0"]
    assert [g.get("class") for g in slots] == ["slot on", "slot"]
    labels = [t.text for t in root.iterfind(".//svg:text", ns) if "state" in t.get("class", "")]
    expected = ["original", "eddy-corrected", "realigned"] if eddy else ["original", "realigned"]
    assert labels == expected * 2
    assert root.find("svg:script", ns) is not None
    style = root.findtext("svg:style", "", ns)
    assert ("@keyframes q-e" in style) is eddy
    end = Timing(hold=0.5).marks(eddy)["end"]
    assert f"animation: cycle {end:g}s" in style
    legend = [t.text for t in root.iterfind(".//svg:text", ns) if "arrows" in (t.text or "")]
    assert legend == (["eddy currents (arrows ×30)"] if eddy else []) + [
        "head motion (arrows ×10)"
    ]
    images = list(root.iter("{http://www.w3.org/2000/svg}image"))
    assert len(images) == 2 * (3 if eddy else 2)
    # The field is undone first: its travel plays inside the head motion's
    travels = [g for g in root.iter("{http://www.w3.org/2000/svg}g") if g.get("class") == "go-m"]
    assert travels
    for travel in travels:
        assert (travel.find("svg:g[@class='go-e']", ns) is not None) is eddy
    for matrix in re.findall(r"matrix\(([^)]*)\)", svg):
        assert np.isfinite(np.array(matrix.split(","), dtype=float)).all()
