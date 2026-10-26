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
"""Reading the synthetic phantoms into the sensitivity analysis.

A toy dataset is written with the layout of ``tools/phantoms/phantom.py``: a
reference with one b=0 and six diffusion-weighted volumes, and three perturbed
copies (eddy currents of every diffusion-weighted volume; motion of volumes 4, 2
and 6 of the file; both). Each file adds its own constant to the reference's
volumes, so a loaded copy shows where each of its volumes came from.
"""

import json

import nibabel as nb
import numpy as np
import pytest

from nifreeze.analysis import sensitivity as sens
from nifreeze.analysis.phantoms import load_phantom
from nifreeze.testing.phantoms import add_noise

SHAPE = (6, 8, 4)
AFFINE = np.diag([2.0, 2.0, 2.0, 1.0])
TABLE = np.array(
    [
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 1000.0],
        [0.0, 1.0, 0.0, 1000.0],
        [0.0, 0.0, 1.0, 1000.0],
        [0.6, 0.8, 0.0, 2000.0],
        [0.0, 0.6, 0.8, 2000.0],
        [0.8, 0.0, 0.6, 2000.0],
    ]
)
MOVED = [4, 2, 6]
"""File volumes of the motion copies, in the order of their plan."""
MARKS = {"e2": 100.0, "m1s0": 1000.0, "m1s0e2": 10000.0}
"""What each file adds to the reference's volumes."""
SIGMA = 2.0


def _eddy(volumes):
    rows = np.array([[0.0, 0.1 * v, -0.2 * v, 0.05 * v] for v in volumes])
    return {
        "Magnitude": 2.0,
        "MagnitudeMm": 4.0,
        "TotalReadoutTime": 0.04,
        "ReadoutTimes": ((SHAPE[1] // 2 - np.arange(SHAPE[1])) * 0.005).tolist(),
        "Distortion": {
            "Axis": 1,
            "Degree": 1,
            "Center": [6.0, 8.0, 4.0],
            "Radius": [2.0, 2.0, 2.0],
            "Frame": np.eye(3).tolist(),
            "Coefficients": rows.tolist(),
        },
    }


def _motion():
    matrices = []
    for k in range(len(MOVED)):
        matrix = np.eye(4)
        matrix[:3, 3] = [0.5 * (k + 1), 0.0, 0.0]
        matrices.append(matrix.tolist())
    return {"Magnitude": 1.0, "Seed": 0, "Center": [5.0, 7.0, 3.0], "Matrices": matrices}


@pytest.fixture
def phantom(tmp_path):
    """The toy dataset's root, and its noise-free reference ``(X, Y, Z, N)``."""
    rng = np.random.default_rng(1234)
    clean = rng.uniform(10.0, 50.0, size=(*SHAPE, len(TABLE))).astype(np.float32)
    folder = tmp_path / "sub-60501" / "dwi"
    folder.mkdir(parents=True)
    stem = folder / "sub-60501_acq-toy"
    nb.Nifti1Image(clean, AFFINE).to_filename(f"{stem}_rec-clean_dwi.nii.gz")
    mask = np.zeros(SHAPE, dtype=np.uint8)
    mask[1:-1, 1:-1, 1:-1] = 1
    nb.Nifti1Image(mask, AFFINE).to_filename(f"{stem}_desc-brain_mask.nii.gz")
    np.savetxt(f"{stem}_dwi.b", TABLE, header="toy: x y z b")
    reference = {"ProtocolSha256": "toy", "ProtocolVolumes": None, "Noise": {"Sigma": SIGMA}}
    (folder / "sub-60501_acq-toy_dwi.json").write_text(json.dumps(reference))

    cells = tmp_path / "derivatives" / "perturbations" / "sub-60501" / "dwi"
    cells.mkdir(parents=True)
    weighted = list(range(1, len(TABLE)))
    sets = {
        "e2": (weighted, {"Kind": "eddy", "Motion": None, "Eddy": _eddy(weighted)}),
        "m1s0": (MOVED, {"Kind": "motion", "Motion": _motion(), "Eddy": None}),
        "m1s0e2": (MOVED, {"Kind": "motion+eddy", "Motion": _motion(), "Eddy": _eddy(MOVED)}),
    }
    for desc, (sources, truth) in sets.items():
        name = f"sub-60501_acq-toy_desc-{desc}_dwi"
        changed = clean[..., sources] + MARKS[desc]
        nb.Nifti1Image(changed, AFFINE).to_filename(cells / f"{name}.nii.gz")
        sidecar = {"ProtocolSha256": "toy", "ProtocolVolumes": None, "SourceVolumes": sources}
        (cells / f"{name}.json").write_text(json.dumps({**sidecar, **truth}))
    return tmp_path, clean


def _edit(root, desc, **changes):
    """Change the sidecar of one perturbed copy."""
    path = root / f"derivatives/perturbations/sub-60501/dwi/sub-60501_acq-toy_desc-{desc}_dwi.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), **changes}))


def _time_first(volumes):
    return np.moveaxis(volumes, -1, 0)


def test_the_reference_keeps_its_diffusion_weighted_volumes_from_zero(phantom):
    root, clean = phantom

    reference, _ = load_phantom(root, "toy")
    assert reference.bzero is not None and reference.brainmask is not None

    assert np.array_equal(reference.dataobj, _time_first(clean[..., 1:]))
    assert np.array_equal(reference.bzero, clean[..., 0])
    assert np.allclose(reference.gradients[:, 3], TABLE[1:, 3])
    assert reference.brainmask.sum() == 4 * 6 * 2


def test_copies_splice_their_changed_volumes_into_the_reference(phantom):
    """Eddy currents change every diffusion-weighted volume; then the moved ones replace theirs."""
    root, clean = phantom
    weighted = _time_first(clean[..., 1:])

    _, cells = load_phantom(root, "toy")
    eddy, motion, both = (load() for _, load in cells)

    assert np.array_equal(eddy.dataobj, weighted + MARKS["e2"])
    # File volumes 4, 2 and 6 are the dataset's volumes 3, 1 and 5
    expected = weighted.copy()
    expected[[3, 1, 5]] += MARKS["m1s0"]
    assert np.array_equal(motion.dataobj, expected)
    expected = weighted + MARKS["e2"]
    expected[[3, 1, 5]] = weighted[[3, 1, 5]] + MARKS["m1s0e2"]
    assert np.array_equal(both.dataobj, expected)


def test_each_copy_comes_with_its_ground_truth(phantom):
    """Volumes are numbered as in the dataset; a combined copy has every volume's eddy currents."""
    root, _ = phantom

    _, cells = load_phantom(root, "toy", counts=[1, 3])
    truths = [perturbation for perturbation, _ in cells]

    eddy = truths[0]
    assert eddy.distortion is not None and eddy.distortion.readout is not None
    assert (eddy.magnitude, eddy.moved, eddy.distortion.magnitude) == (0.0, (), 4.0)
    for index in range(6):
        file_volume = index + 1
        assert np.allclose(
            eddy.distortion.coefficients[index], np.array([0, 0.1, -0.2, 0.05]) * file_volume
        )
    assert eddy.distortion.readout_time == 0.04 and len(eddy.distortion.readout) == SHAPE[1]

    moved = [(t.moved, t.distortion is not None) for t in truths[1:]]
    assert moved == [((3,), False), ((3, 1, 5), False), ((3,), True), ((3, 1, 5), True)]
    assert truths[2].motions[2][0, 3] == 1.5 and (truths[2].magnitude, truths[2].seed) == (1, 0)
    assert truths[3].distortion is eddy.distortion

    with pytest.raises(ValueError, match="moves 3 volumes, not 4"):
        load_phantom(root, "toy", counts=[4])


def test_noise_is_the_reference_noise_wherever_a_copy_is_unchanged(phantom):
    """Noise is keyed on the file's volume, so a copy differs only where its volumes changed."""
    root, clean = phantom
    noisy = add_noise(_time_first(clean), SIGMA, seed=7)

    reference, cells = load_phantom(root, "toy", noise=7)
    motion = cells[1][1]()
    assert reference.bzero is not None

    assert np.array_equal(reference.dataobj, noisy[1:])
    assert np.array_equal(reference.bzero, noisy[0])
    unchanged = [0, 2, 4]
    assert np.array_equal(motion.dataobj[unchanged], reference.dataobj[unchanged])
    expected = add_noise(_time_first(clean[..., MOVED] + MARKS["m1s0"]), SIGMA, 7, MOVED)
    assert np.array_equal(motion.dataobj[[3, 1, 5]], expected)


def test_copies_of_another_reference_are_refused(phantom):
    root, _ = phantom
    _edit(root, "m1s0", ProtocolVolumes=[0, 1, 2])
    with pytest.raises(ValueError, match="m1s0_dwi.json perturbs another reference"):
        load_phantom(root, "toy")


def test_copies_may_not_change_b0_volumes(phantom):
    root, _ = phantom
    _edit(root, "m1s0", SourceVolumes=[0, 2, 6])
    with pytest.raises(ValueError, match="not diffusion-weighted"):
        load_phantom(root, "toy")


def test_a_combined_copy_needs_the_eddy_currents_alone(phantom):
    root, _ = phantom
    (root / "derivatives/perturbations/sub-60501/dwi/sub-60501_acq-toy_desc-e2_dwi.json").unlink()
    with pytest.raises(ValueError, match="m1s0e2_dwi.nii.gz needs the copy with its eddy"):
        load_phantom(root, "toy")


def test_a_copy_on_another_grid_is_refused(phantom):
    root, clean = phantom
    path = root / "derivatives/perturbations/sub-60501/dwi/sub-60501_acq-toy_desc-e2_dwi.nii.gz"
    nb.Nifti1Image(clean[..., 1:], np.diag([2.5, 2.0, 2.0, 1.0])).to_filename(path)

    _, cells = load_phantom(root, "toy")

    with pytest.raises(ValueError, match="not on the reference's grid"):
        cells[0][1]()


def test_the_analysis_takes_a_phantom_as_it_loads(phantom):
    """Every copy is analyzed, each volume with its shell."""
    root, _ = phantom

    rows = sens.sensitivity("avgdwi", *load_phantom(root, "toy", noise=0))

    assert len(rows) == 3 * 6
    assert sorted({row["shell"] for row in rows}) == [1000.0, 2000.0]
    assert all(np.isfinite(row["inherited"]) == (row["distortion"] > 0) for row in rows)
