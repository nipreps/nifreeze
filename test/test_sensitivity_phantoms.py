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
"""The sensitivity analysis on TRXScan's own output: the CI phantom.

``tools/phantoms/phantom.py fixture`` simulates the Sherbrooke protocol at 4 mm:
a reference with one b=0 and 8 directions per shell (24 diffusion-weighted
volumes, b=1000, 2000 and 3500), a copy with eddy currents that displace some
brain voxel by 2 voxels (8 mm), a copy with every volume moved by 1° and 1 mm,
and a copy with both. Without the dataset (``NIFREEZE_PHANTOMS``), these tests
are skipped.
"""

import numpy as np
import pytest

from nifreeze.analysis import sensitivity as sens
from nifreeze.analysis.phantoms import load_phantom

SHELLS = {1000.0: range(0, 8), 2000.0: range(8, 16), 3500.0: range(16, 24)}
"""The dataset's volumes of each shell."""


@pytest.fixture(scope="module")
def phantom(phantoms):
    """The noise-free reference, and its copies by number of moved volumes and distortion."""
    reference, cells = load_phantom(phantoms, "sherbrooke", counts=[6, 24])
    return reference, {(p.count, p.distortion is not None): (p, load) for p, load in cells}


def test_the_dataset_has_the_fixture_shells(phantom):
    reference, cells = phantom

    assert len(reference) == 24
    for shell, volumes in SHELLS.items():
        assert np.all(reference.gradients[list(volumes), 3] == shell)
    assert sorted(cells) == [(0, True), (6, False), (6, True), (24, False), (24, True)]


def test_the_epi_model_reproduces_trxscan_eddy_currents(phantom):
    """Each volume, distorted from the clean one by its closed form, matches TRXScan's.

    Within 10⁻⁴ of the volume's maximum (6.7·10⁻⁶ measured), so a TRXScan
    commit that changes its eddy-current model, or how it records it, fails here.
    """
    reference, cells = phantom
    perturbation, load = cells[(0, True)]
    distortion = perturbation.distortion
    assert distortion is not None
    distorted = load()

    for index in range(len(reference)):
        formed = distortion.distort(
            reference.dataobj[index], reference.affine, distortion.coefficients[index]
        )
        actual = distorted.dataobj[index]
        assert np.abs(formed - actual).max() < 1e-4 * actual.max()


def test_a_moved_copy_changes_only_its_moved_volumes(phantom):
    """The first six moved volumes are those of the copy that moves all, with the same motions."""
    reference, cells = phantom
    perturbation, load = cells[(6, False)]
    moved = load()

    still = [index for index in range(len(reference)) if index not in perturbation.moved]
    assert np.array_equal(moved.dataobj[still], reference.dataobj[still])
    for index in perturbation.moved:
        assert not np.allclose(moved.dataobj[index], reference.dataobj[index])
    everyone, _ = cells[(24, False)]
    assert everyone.moved[:6] == perturbation.moved
    assert all(
        np.array_equal(a, b)
        for a, b in zip(everyone.motions[:6], perturbation.motions, strict=True)
    )


def test_the_reference_against_itself_does_not_drift(phantom):
    reference, _ = phantom

    rows = sens.sensitivity(
        "avgdwi", reference, [(sens.NO_PERTURBATION, reference)], evaluate=[0, 8, 16]
    )

    assert [row["nrmse"] for row in rows] == [0.0, 0.0, 0.0]


def test_the_analysis_of_the_phantom_carries_each_copy_truth(phantom):
    """One row per copy and volume, each with its shell, its motion and its distortion."""
    reference, cells = phantom
    evaluate = [volumes[0] for volumes in SHELLS.values()]

    rows = sens.sensitivity("avgdwi", reference, list(cells.values()), evaluate=evaluate)

    assert len(rows) == len(cells) * len(evaluate)
    moved = {(p.count, p.distortion is not None): p.moved for p, _ in cells.values()}
    for row in rows:
        key = (row["count"], row["distortion"] > 0)
        assert row["shell"] == reference.gradients[row["index"], 3]
        assert row["moved"] == (row["index"] in moved[key])
        # Every diffusion-weighted volume is distorted, by 2 voxels of 4 mm at most
        assert row["distortion"] in (0.0, pytest.approx(8.0))
        if key == (0, True):
            assert row["nrmse"] > 0.0


@pytest.fixture(scope="module")
def alone(phantoms):
    """Copies that move two volumes alone: the last of the plan's order, and one inside it."""
    _, cells = load_phantom(phantoms, "sherbrooke", counts=[24])
    everyone = next(p for p, _ in cells if p.count == 24 and p.distortion is None)
    volumes = [everyone.moved[-1], everyone.moved[10]]
    reference, cells = load_phantom(phantoms, "sherbrooke", counts=[24], alone=volumes)
    return reference, volumes, cells


def test_a_volume_moved_alone_is_its_counterpart_in_the_moved_set(alone):
    """Each copy moves one volume, with the motion and the image it has when all move.

    Without eddy currents, every other volume is the reference's; with them, every
    other volume is the copy's with eddy currents alone.
    """
    reference, volumes, cells = alone
    full = {p.distortion is not None: (p, load()) for p, load in cells if p.count == 24}
    eddy = next(load() for p, load in cells if p.count == 0)

    for distorted in (False, True):
        everyone, moved = full[distorted]
        background = eddy if distorted else reference
        for volume in volumes:
            perturbation, load = next(
                (p, load)
                for p, load in cells
                if p.moved == (volume,) and (p.distortion is not None) == distorted
            )
            copy = load()
            row = everyone.moved.index(volume)
            assert np.array_equal(perturbation.motions[0], everyone.motions[row])
            assert np.array_equal(copy.dataobj[volume], moved.dataobj[volume])
            others = [index for index in range(len(reference)) if index != volume]
            assert np.array_equal(copy.dataobj[others], background.dataobj[others])


def test_the_analysis_takes_volumes_moved_alone(alone):
    """A copy with eddy currents and one volume moved finds the copy with that motion alone."""
    reference, volumes, cells = alone
    chosen = [(p, load) for p, load in cells if p.count == 1]

    rows = sens.sensitivity("avgdwi", reference, chosen, evaluate=[0, 8, 16])

    assert len(rows) == 3 * len(chosen) == 12
    for row in rows:
        assert row["moved"] == (row["index"] in volumes)


def test_only_moved_volumes_can_move_alone(phantoms):
    with pytest.raises(ValueError, match="does not move volume 24"):
        load_phantom(phantoms, "sherbrooke", alone=[24])
