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
"""Head motion and a field along phase encoding, estimated together (#730)."""

import warnings
from typing import Any

import nibabel as nb
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from test_registration_conventions import SKIP_ELASTIX, _affine, _error, _motion, _move, _volume

from nifreeze.data.dmri import DWI
from nifreeze.estimator import Estimator
from nifreeze.registration.base import Registration, RegistrationFactory
from nifreeze.registration.utils import (
    phase_encoding_vector,
    remove_shared_field,
    split_phase_encoding,
)

SLOPE = np.array([0.015, -0.025, 0.02])
"""A field's slope (mm per mm) along each world axis."""


def _field(direction: np.ndarray, slope: np.ndarray = SLOPE) -> np.ndarray:
    """A field along ``direction``, linear in position: ``I - u s^T``."""
    field = np.eye(4)
    field[:3, :3] -= np.outer(direction, slope)
    return field


def _settings(restrict: bool = True) -> dict:
    """The packaged ITK stages, with or without the restriction to phase encoding."""
    from nifreeze.registration.itk_backend import get_settings

    settings = get_settings()
    if not restrict:
        settings.pop("restrict")
    return settings


def test_split_recovers_motion_and_field():
    """The closed form splits any mapping of the family, whatever the frame and sign."""
    rng = np.random.default_rng(0)
    for seed in range(200):
        motion = np.eye(4)
        motion[:3, :3] = Rotation.from_rotvec(rng.normal(scale=0.05, size=3)).as_matrix()
        motion[:3, 3] = rng.normal(scale=2.0, size=3)
        voxels = Rotation.random(random_state=seed).as_matrix()
        if seed % 2:
            voxels = voxels @ np.diag([1.0, 1.0, -1.0])  # a reflected voxel frame
        direction = voxels[:, seed % 3] * (-1.0 if seed % 5 else 1.0)
        field = _field(direction, rng.normal(scale=0.03, size=3))

        found_motion, found_field = split_phase_encoding(field @ motion, direction)

        assert np.allclose(found_motion, motion, atol=1e-12)
        assert np.allclose(found_field, field, atol=1e-12)


def _shells_on_a_half_sphere(rng: np.random.Generator) -> np.ndarray:
    """Three shells of 16 b-vectors each, all on one half of the sphere (not balanced)."""
    vectors = rng.normal(size=(48, 3))
    vectors[:, 2] = np.abs(vectors[:, 2])
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.column_stack([vectors, np.repeat([995.0, 2005.0, 3000.0], 16)])


def _stretch_about(direction: np.ndarray, shared: np.ndarray, centre: np.ndarray):
    """A field along ``direction`` about ``centre``: ``x - u a^T (x - c)``."""
    field = _field(direction, shared)
    field[:3, 3] = direction * (shared @ centre)
    return field


def test_remove_shared_field_recovers_the_field_that_follows_the_gradient():
    """A shared field after one linear in the b-vector is removed exactly, per shell."""
    rng = np.random.default_rng(1)
    centre = np.array([-1.5, -18.5, 6.5])
    for seed in range(20):
        gradients = _shells_on_a_half_sphere(rng)
        direction = Rotation.random(random_state=seed).as_matrix()[:, seed % 3]
        shells = np.round(gradients[:, 3], -2)
        truth, stored = [], []
        coefficients = {b: rng.normal(scale=0.01, size=(3, 3)) for b in np.unique(shells)}
        shared = {b: rng.normal(scale=0.005, size=3) for b in np.unique(shells)}
        for g, b in zip(gradients[:, :3], shells, strict=True):
            motion = np.eye(4)
            motion[:3, :3] = Rotation.from_rotvec(rng.normal(scale=0.03, size=3)).as_matrix()
            motion[:3, 3] = rng.normal(scale=2.0, size=3)
            field = _field(direction, coefficients[b] @ g)
            truth.append((motion, field))
            biased = _stretch_about(direction, shared[b], centre) @ field @ motion
            stored.append(split_phase_encoding(biased, direction))
        motions, fields = (np.stack(part) for part in zip(*stored, strict=True))

        found_motion, found_field = remove_shared_field(
            motions, fields, gradients, direction, centre
        )

        assert np.allclose(found_motion, [m for m, _ in truth], atol=1e-10)
        assert np.allclose(found_field, [f for _, f in truth], atol=1e-12)


def test_remove_shared_field_keeps_a_field_that_follows_the_gradient():
    """With nothing shared, the fields stay, though the shells' mean slope is not zero."""
    rng = np.random.default_rng(2)
    gradients = _shells_on_a_half_sphere(rng)
    direction = np.array([0.0, -1.0, 0.0])
    coefficients = rng.normal(scale=0.01, size=(3, 3))
    fields = np.stack([_field(direction, coefficients @ g) for g in gradients[:, :3]])
    motion = np.repeat(np.eye(4)[np.newaxis], len(fields), axis=0)
    slopes = (direction @ (np.eye(3) - fields[:, :3, :3])).reshape(-1, 3)
    assert np.abs(slopes.mean(axis=0)).max() > 1e-3  # removing the mean would change them

    found_motion, found_field = remove_shared_field(
        motion, fields, gradients, direction, np.zeros(3), shells=gradients[:, 3] > 1500
    )

    assert np.allclose(found_field, fields, atol=1e-12)
    assert np.allclose(found_motion, motion, atol=1e-12)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("eddy", "both be"),
        ("gradients", "one row per volume"),
        ("shells", "one label per volume"),
        ("plane", "do not span three dimensions"),
    ],
)
def test_remove_shared_field_validates_its_inputs(change, message):
    rng = np.random.default_rng(3)
    gradients = _shells_on_a_half_sphere(rng)
    identity = np.repeat(np.eye(4)[np.newaxis], len(gradients), axis=0)
    arguments: dict[str, Any] = {
        "motion": identity,
        "eddy": identity[:-1] if change == "eddy" else identity,
        "gradients": gradients[:, :3] if change == "gradients" else gradients,
        "direction": np.array([0.0, 1.0, 0.0]),
        "centre": np.zeros(3),
        "shells": np.zeros(3) if change == "shells" else None,
    }
    if change == "plane":
        arguments["gradients"] = gradients.copy()
        arguments["gradients"][:16, 2] = 0.0
    with pytest.raises(ValueError, match=message):
        remove_shared_field(**arguments)


def test_phase_encoding_vector_follows_the_voxel_axis():
    affine = _affine("oblique")
    column = affine[:3, 1] / np.linalg.norm(affine[:3, 1])
    assert np.allclose(phase_encoding_vector("j", affine), column)
    assert np.allclose(phase_encoding_vector("j-", affine), -column)


@SKIP_ELASTIX
@pytest.mark.parametrize("phase_encoding", ["i", "j-", "k"])
def test_itk_recovers_motion_and_field_on_oblique_data(phase_encoding):
    """On an oblique grid, the 9 parameters find the head motion and the field apart."""
    from nifreeze.registration.itk_backend import ITK

    data, mask = _volume()
    affine = _affine("oblique")
    motion = _motion()
    field = _field(phase_encoding_vector(phase_encoding, affine))
    moving = _move(data, affine, field @ motion)
    register = ITK(_settings(), num_threads=1, phase_encoding=phase_encoding, clip="none")

    estimate = register(data, moving, affine, mask, 0, None).matrix

    assert _error(mask, affine, estimate, field @ motion) < 0.05
    parts = register.split(estimate, affine)
    assert parts is not None
    found_motion, found_field = parts
    assert np.allclose(found_field @ found_motion, estimate, atol=1e-9)  # in the family
    assert _error(mask, affine, found_motion, motion) < 0.05
    assert np.abs(found_field - field).max() < 2e-3


@SKIP_ELASTIX
def test_itk_estimates_a_full_affine_without_the_restriction():
    """Without ``restrict``, the backend is a 12-parameter affine, and does not split."""
    from nifreeze.registration.itk_backend import ITK

    data, mask = _volume()
    affine = _affine("oblique")
    truth = _field(np.array([0.3, 0.9, 0.3]) / np.linalg.norm([0.3, 0.9, 0.3])) @ _motion()
    register = ITK(_settings(restrict=False), num_threads=1, phase_encoding="j", clip="none")

    estimate = register(data, _move(data, affine, truth), affine, mask, 0, None).matrix

    assert _error(mask, affine, estimate, truth) < 0.05
    assert register.split(estimate, affine) is None


@SKIP_ELASTIX
def test_itk_needs_the_phase_encoding_to_restrict():
    from nifreeze.registration.itk_backend import ITK, NO_PHASE_ENCODING_ERROR_MSG

    data, mask = _volume()
    with pytest.raises(ValueError, match=NO_PHASE_ENCODING_ERROR_MSG[:40]):
        ITK(_settings(), num_threads=1)(data, data, _affine("ras"), mask, 0, None)


def _moved() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A volume and mask on an oblique grid, a truth (phase encoding ``j``), the moved one."""
    data, mask = _volume()
    affine = _affine("oblique")
    truth = _field(phase_encoding_vector("j", affine)) @ _motion()
    return data, mask, affine, truth, _move(data, affine, truth)


@SKIP_ELASTIX
def test_itk_keeps_the_values_it_registers(tmp_path):
    """With ``debug_dir``, the backend writes the prepared images it registered."""
    from nifreeze.registration.elastix import _winsorize
    from nifreeze.registration.itk_backend import ITK

    data, mask, affine, _, moving = _moved()
    settings = _settings()
    register = ITK(settings, num_threads=1, phase_encoding="j", clip="none", debug_dir=tmp_path)

    register(data, moving, affine, mask, 3, None)

    quantiles = settings["winsorize_lower_quantile"], settings["winsorize_upper_quantile"]
    for name, values in (("predicted", data), ("sample", moving)):
        image = nb.Nifti1Image.from_filename(tmp_path / f"{name}_00003.nii")
        assert np.allclose(image.affine, affine)
        assert np.allclose(image.get_fdata(), _winsorize(values, *quantiles))
    assert (tmp_path / "aligned_00003.nii").exists()


@SKIP_ELASTIX
def test_itk_registers_without_winsorizing_and_with_histogram_matching():
    """The optional preprocessing, with ITK's own seed: the estimate still finds the truth."""
    from nifreeze.registration.itk_backend import ITK

    data, mask, affine, truth, moving = _moved()
    settings = _settings()
    del settings["winsorize_lower_quantile"], settings["winsorize_upper_quantile"]
    settings["use_histogram_matching"] = True
    register = ITK(settings, seed=None, num_threads=1, phase_encoding="j", clip="none")

    estimate = register(data, moving, affine, mask, 0, None).matrix

    assert _error(mask, affine, estimate, truth) < 0.05


@SKIP_ELASTIX
@pytest.mark.parametrize("num_threads", [None, 1])
def test_itk_restores_the_thread_counts(num_threads):
    """ITK's thread counts are process-wide: a registration leaves them as it found them."""
    import itk

    from nifreeze.registration.itk_backend import ITK

    threader = itk.MultiThreaderBase
    data, mask, affine, _, moving = _moved()
    before = (
        threader.GetGlobalMaximumNumberOfThreads(),
        threader.GetGlobalDefaultNumberOfThreads(),
    )

    ITK(_settings(), num_threads=num_threads, phase_encoding="j", clip="none")(
        data, moving, affine, mask, 0, None
    )

    after = (
        threader.GetGlobalMaximumNumberOfThreads(),
        threader.GetGlobalDefaultNumberOfThreads(),
    )
    assert after == before


@SKIP_ELASTIX
def test_itk_starts_from_the_projection_of_a_mapping_outside_the_family():
    """
    A start outside the 9-parameter family (e.g., an earlier 12-parameter estimate) warns,
    and the registration starts from its projection; a start in the family does not warn.
    """
    from nifreeze.registration.itk_backend import ITK, OUTSIDE_FAMILY_WARN_MSG

    data, mask, affine, truth, moving = _moved()
    frame = affine[:3, :3] / np.linalg.norm(affine[:3, :3], axis=0)
    across = np.eye(4)
    across[:3, :3] = frame @ np.diag([1.03, 1.0, 1.0]) @ frame.T  # a scaling along i
    register = ITK(_settings(), num_threads=1, phase_encoding="j", clip="none")

    with pytest.warns(UserWarning, match=OUTSIDE_FAMILY_WARN_MSG.format(index=4)[:40]):
        estimate = register(data, moving, affine, mask, 4, across @ _motion()).matrix
    assert _error(mask, affine, estimate, truth) < 0.05

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        register(data, moving, affine, mask, 4, truth)
    assert not [w for w in caught if "initial mapping" in str(w.message)]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"smoothing": 1.0}, r"Unknown ITK settings: \['smoothing'\]"),
        ({"restrict": "readout"}, "Unknown ITK restrict 'readout'"),
        ({"stages": []}, "at least one stage"),
        (
            {"stages": [{"metric": "correlation", "sampling_percentage": 0.1, "samples": 100}]},
            "sampling_percentage or samples, not both",
        ),
    ],
)
def test_itk_settings_are_validated(change, message):
    from nifreeze.registration.itk_backend import get_settings

    with pytest.raises(ValueError, match=message):
        get_settings(get_settings() | change)


def test_itk_stage_settings_are_validated():
    from nifreeze.registration.itk_backend import get_settings

    settings = get_settings()
    settings["stages"][0]["metric"] = "mse"
    with pytest.raises(ValueError, match="Unknown ITK metric 'mse'"):
        get_settings(settings)


def test_the_other_backends_take_the_phase_encoding_and_do_not_split():
    for name in ("ants", "elastix"):
        try:
            register = RegistrationFactory.init(name, phase_encoding="j-")
        except ImportError:  # elastix's extra is not installed
            continue
        assert register.phase_encoding == "j-"
        assert register.split(np.eye(4), np.eye(4)) is None


def _dwi(phase_encoding: str | None = "j-") -> DWI:
    rng = np.random.default_rng(1234)
    bvecs = rng.standard_normal((8, 3))
    bvecs /= np.linalg.norm(bvecs, axis=1, keepdims=True)
    return DWI(
        dataobj=rng.random((8, 6, 6, 4), dtype=np.float32),
        affine=np.diag([2.0, 2.0, 2.0, 1.0]),
        brainmask=np.ones((6, 6, 4), dtype=bool),
        gradients=np.column_stack((bvecs, np.full(8, 1000.0))),
        phase_encoding=phase_encoding,
    )


class _Splitting(Registration):
    """Returns a known mapping of the family, and splits it."""

    def _register(self, fixed, moving, affine, mask, index, init, window):
        return _field(phase_encoding_vector("j-", affine)) @ _motion()

    def split(self, matrix, affine):
        return split_phase_encoding(matrix, phase_encoding_vector("j-", affine))


def test_set_estimates_stores_each_field():
    dwi = _dwi()
    motion, field = _motion(), _field(np.array([0.0, 1.0, 0.0]))
    dwi.set_estimates(2, motion=motion, eddy=field)
    assert dwi.motion is not None and dwi.eddy is not None

    assert np.allclose(dwi.motion[2], motion) and np.allclose(dwi.eddy[2], field)
    transform = dwi.get_transform(2)
    assert transform is not None and np.allclose(transform, field @ motion)
    assert np.allclose(dwi.motion[0], np.eye(4)) and np.allclose(dwi.eddy[0], np.eye(4))
    with pytest.raises(ValueError, match="has no estimates 'b0'"):
        dwi.set_estimates(0, b0=np.eye(4))


def test_the_eddy_pass_stores_the_split():
    """A backend that splits refines the head motion and stores the field."""
    dwi = _dwi()
    dwi.motion = np.repeat(np.eye(4)[np.newaxis], 8, axis=0)
    Estimator("avg", estimates="eddy", registration=_Splitting()).run(
        dwi, n_jobs=1, omp_nthreads=1, instant_update=False
    )
    assert dwi.eddy is not None

    assert np.allclose(dwi.motion, _motion())
    assert np.allclose(dwi.eddy, _field(phase_encoding_vector("j-", dwi.affine)))


def test_the_estimator_gives_the_backend_the_phase_encoding(monkeypatch):
    captured = {}

    def init(name=None, **options):
        captured.update(options)
        return _Splitting(phase_encoding=options.get("phase_encoding"))

    monkeypatch.setattr(RegistrationFactory, "init", staticmethod(init))
    Estimator("avg", estimates="eddy").run(_dwi("k"), n_jobs=1, omp_nthreads=1)

    assert captured["phase_encoding"] == "k"
