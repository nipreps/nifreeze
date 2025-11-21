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

import json
import math
import warnings
from contextlib import suppress
from pathlib import Path
from typing import Any, Type

import attrs
import nibabel as nb
import numpy as np
import pytest
from nitransforms.linear import Affine

from nifreeze.data import load as nifreeze_load
from nifreeze.data.pet import (
    ARRAY_ATTRIBUTE_NDIM_ERROR_MSG,
    ARRAY_ATTRIBUTE_OBJECT_ERROR_MSG,
    ATTRIBUTE_ABSENCE_ERROR_MSG,
    ATTRIBUTE_VOLUME_DIMENSIONALITY_MISMATCH_ERROR,
    FRAME_DURATION_KEY,
    FRAME_MIDFRAME_KEY,
    FRAME_TIME_START_KEY,
    FRAME_UPTAKE_KEY,
    PET,
    SCALAR_ATTRIBUTE_ERROR_MSG,
    SEQUENCE_TOTAL_DURATION_KEY,
    TEMPORAL_FILE_KEY_ERROR_MSG,
    UPTAKE_FILE_KEY_ERROR_MSG,
    _compute_frame_duration,
    compute_uptake_statistic,
    format_array_like,
    format_scalar_like,
    from_nii,
    validate_1d_array,
)
from nifreeze.utils.ndimage import load_api


def _pet_data_to_nifti(pet_dataobj, affine, brainmask_dataobj):
    pet = nb.Nifti1Image(pet_dataobj, affine)
    brainmask = nb.Nifti1Image(brainmask_dataobj, affine)

    return pet, brainmask


def _serialize_pet_data(
    pet, brainmask, frame_time, frame_duration, midframe, total_duration, uptake, _tmp_path
):
    pet_fname = _tmp_path / "pet.nii.gz"
    brainmask_fname = _tmp_path / "brainmask.nii.gz"
    temporal_fname = _tmp_path / "temporal.json"
    uptake_fname = _tmp_path / "uptake.json"

    nb.save(pet, pet_fname)
    nb.save(brainmask, brainmask_fname)

    temporal_data = {
        FRAME_TIME_START_KEY: frame_time.tolist(),
        FRAME_DURATION_KEY: frame_duration.tolist(),
        FRAME_MIDFRAME_KEY: midframe.tolist(),
        SEQUENCE_TOTAL_DURATION_KEY: total_duration,
    }
    with temporal_fname.open("w", encoding="utf-8") as f:
        json.dump(temporal_data, f, ensure_ascii=False, indent=2, sort_keys=True)

    uptake_data = {FRAME_UPTAKE_KEY: uptake.tolist()}
    with uptake_fname.open("w", encoding="utf-8") as f:
        json.dump(uptake_data, f, ensure_ascii=False, indent=2, sort_keys=True)

    return pet_fname, brainmask_fname, temporal_fname, uptake_fname


@pytest.fixture
def random_nifti_file(tmp_path, setup_random_uniform_spatial_data) -> Path:
    _data, _affine = setup_random_uniform_spatial_data
    _filename = tmp_path / "random_pet.nii.gz"
    _img = nb.Nifti1Image(_data, _affine)
    _img.to_filename(_filename)
    return _filename


@pytest.mark.parametrize(
    "attr_name, value, expected_exc, expected_msg",
    [
        ("any_name", None, ValueError, ATTRIBUTE_ABSENCE_ERROR_MSG),
        (
            "any_name",
            [[10.0], [20.0, 30.0], [40.0], [50.0]],
            TypeError,
            ARRAY_ATTRIBUTE_OBJECT_ERROR_MSG,
        ),  # Ragged
        (
            "any_name",
            np.array([[-0.9], [0.06, 0.12], [0.27], [0.08]], dtype=object),
            TypeError,
            ARRAY_ATTRIBUTE_OBJECT_ERROR_MSG,
        ),  # Ragged
    ],
)
def test_format_array_like_errors(attr_name, value, expected_exc, expected_msg):
    # Produce a valid attrs.Attribute for the test
    dummy_attr_cls: Type[Any] = attrs.make_class("Dummy", {attr_name: attrs.field()})
    dummy_attr = dummy_attr_cls.__attrs_attrs__[0]
    with pytest.raises(expected_exc, match=expected_msg.format(attribute=attr_name)):
        format_array_like(value, dummy_attr)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "value, expected_exc, expected_msg",
    [
        (None, ValueError, ATTRIBUTE_ABSENCE_ERROR_MSG),
        (True, ValueError, SCALAR_ATTRIBUTE_ERROR_MSG),
        ((2, 2), ValueError, SCALAR_ATTRIBUTE_ERROR_MSG),
        (np.asarray([1.0, 2.0]), ValueError, SCALAR_ATTRIBUTE_ERROR_MSG),
    ],
)
def test_format_scalar_errors(value, expected_exc, expected_msg):
    # Produce a valid attrs.Attribute for the test
    dummy_attr_cls: Type[Any] = attrs.make_class("Dummy", {"total_duration": attrs.field()})
    dummy_attr = dummy_attr_cls.__attrs_attrs__[0]

    with pytest.raises(expected_exc, match=expected_msg.format(attribute="total_duration")):
        format_scalar_like(value, dummy_attr)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [[1.0, 2.0, 3.0, 4.0], (1.0, 2.0, 3.0, 4.0)])
@pytest.mark.parametrize("attr_name", ("frame_time", "uptake", "frame_duration", "midframe"))
def test_format_array_like(value, attr_name):
    # Produce a valid attrs.Attribute for the test
    dummy_attr_cls: Type[Any] = attrs.make_class("Dummy", {attr_name: attrs.field()})
    dummy_attr = dummy_attr_cls.__attrs_attrs__[0]

    obtained = format_array_like(value, dummy_attr)
    assert isinstance(obtained, np.ndarray)
    if attr_name == "frame_time":
        # Time-origin shift
        value -= np.asarray(value)[0]
        assert obtained.shape == np.asarray(value).shape
    else:
        assert obtained.shape == np.asarray(value).shape
    assert np.allclose(obtained, np.asarray(value))


@pytest.mark.parametrize("value", [1.0, [2.0], (3.0,), np.array(4.0), np.array([5.0])])
def test_format_scalar_like(value):
    # Produce a valid attrs.Attribute for the test
    dummy_attr_cls: Type[Any] = attrs.make_class("Dummy", {"total_duration": attrs.field()})
    dummy_attr = dummy_attr_cls.__attrs_attrs__[0]

    obtained = format_scalar_like(value, dummy_attr)
    assert isinstance(obtained, float)
    assert np.allclose(obtained, np.asarray(value))


@pytest.mark.parametrize("attr_name, value", [("my_attr", np.asarray([1.0, 2.0, 3.0, 4.0]))])
@pytest.mark.parametrize("extra_dimensions", (1, 2))
@pytest.mark.parametrize("transpose", (True, False))
def test_validate_1d_arr_errors(
    request, monkeypatch, attr_name, value, extra_dimensions, transpose
):
    def _add_extra_dim(_rng, _attr_name, _extra_dimensions, _transpose, _value):
        _arr = np.concatenate(
            [
                _value[:, None],
                rng.random((_value.size, _extra_dimensions)),
            ],
            axis=1,
        )
        _arr = _arr.T if _transpose else _arr
        return _arr

    rng = request.node.rng
    _value = _add_extra_dim(rng, attr_name, extra_dimensions, transpose, value)

    monkeypatch.setattr(PET, "__init__", lambda self, *a, **k: None)

    # Produce a valid attrs.Attribute for the test
    dummy_attr_cls: Type[Any] = attrs.make_class("Dummy", {attr_name: attrs.field()})
    new_attr = dummy_attr_cls.__attrs_attrs__[0]

    # Replace PET's attribute metadata with just the new_attr.
    # attrs.fields() reads PET.__attrs_attrs__ at runtime, so setting this
    # effectively "removes" previous attributes and leaves only our single one.
    monkeypatch.setattr(PET, "__attrs_attrs__", (new_attr,), raising=False)
    # Also set a matching annotation dict
    monkeypatch.setattr(PET, "__annotations__", {attr_name: object()}, raising=False)

    # Instantiate and obtain the attrs.Attribute from PET
    inst = PET()
    dummy_attr = attrs.fields(PET)[0]
    # assert isinstance(dummy_attr, attrs.Attribute)
    # assert dummy_attr.name == attr_name

    with pytest.raises(
        ValueError,
        match=ARRAY_ATTRIBUTE_NDIM_ERROR_MSG.format(attribute=attr_name),
    ):
        validate_1d_array(inst, dummy_attr, _value)


@pytest.mark.random_pet_data(
    4,
    (2, 2, 2),
    np.asarray([1.0, 2.0, 3.0, 4.0]),
    np.sum,
    np.asarray([1.0, 1.0, 1.0, 1.0]),
    np.asarray([0.5, 1.5, 2.5, 3.5]),
    4.0,
)
@pytest.mark.parametrize("attr_name", ("frame_time", "uptake", "frame_duration", "midframe"))
@pytest.mark.parametrize("extra_dimensions", (1, 2))
@pytest.mark.parametrize("transpose", (True, False))
def test_pet_instantiation_attribute_validate_1d_arr_errors(
    request, setup_random_pet_data, attr_name, extra_dimensions, transpose
):
    def _add_extra_dim(_rng, _attr_name, _extra_dimensions, _transpose, **_kwargs):
        _arr = np.concatenate(
            [
                _kwargs[_attr_name][:, None],
                rng.random((_kwargs[_attr_name].size, _extra_dimensions)),
            ],
            axis=1,
        )
        _kwargs[_attr_name] = _arr.T if _transpose else _arr
        return _kwargs

    rng = request.node.rng
    pet_dataobj, affine, _, frame_time, uptake, frame_duration, midframe, total_duration = (
        setup_random_pet_data
    )

    attrs_dict = dict(
        frame_time=frame_time,
        uptake=uptake,
        frame_duration=frame_duration,
        midframe=midframe,
        total_duration=total_duration,
    )
    _attrs_dict = _add_extra_dim(rng, attr_name, extra_dimensions, transpose, **attrs_dict)

    with pytest.raises(
        ValueError,
        match=ARRAY_ATTRIBUTE_NDIM_ERROR_MSG.format(attribute=attr_name),
    ):
        PET(dataobj=pet_dataobj, affine=affine, **_attrs_dict)  # type: ignore[arg-type]


@pytest.mark.random_uniform_spatial_data((2, 2, 2, 4), 0.0, 1.0)
@pytest.mark.parametrize("attr_name", ("frame_time", "uptake"))
def test_pet_instantiation_attribute_convert_absence_errors(
    setup_random_uniform_spatial_data,
    attr_name,
):
    data, affine = setup_random_uniform_spatial_data

    n_frames = data.shape[-1]
    # Create a dict with default valid attribute values
    attrs_dict: dict[str, np.ndarray | float | None] = dict(
        frame_time=np.zeros(n_frames, dtype=np.float32),
        uptake=np.zeros(n_frames, dtype=np.float32),
        frame_duration=np.ones(n_frames, dtype=np.float32),
        midframe=np.ones(n_frames, dtype=np.float32),
        total_duration=1.0,
    )

    # Override only the attribute under test
    attrs_dict[attr_name] = None

    with pytest.raises(ValueError, match=ATTRIBUTE_ABSENCE_ERROR_MSG.format(attribute=attr_name)):
        PET(dataobj=data, affine=affine, **attrs_dict)  # type: ignore[arg-type]


@pytest.mark.random_uniform_spatial_data((2, 2, 2, 4), 0.0, 1.0)
@pytest.mark.parametrize("attr_name", ("frame_time", "uptake", "frame_duration", "midframe"))
@pytest.mark.parametrize(
    "value",
    [
        ([[10.0], [20.0, 30.0], [40.0], [50.0]]),  # Ragged
        (np.array([[-0.9], [0.06, 0.12], [0.27], [0.08]], dtype=object)),  # Ragged
    ],
)
def test_pet_instantiation_attribute_convert_object_errors(
    setup_random_uniform_spatial_data, attr_name, value
):
    data, affine = setup_random_uniform_spatial_data

    n_frames = data.shape[-1]
    # Create a dict with default valid attributes
    attrs_dict = dict(
        frame_time=np.zeros(n_frames, dtype=np.float32),
        uptake=np.zeros(n_frames, dtype=np.float32),
        frame_duration=np.ones(n_frames, dtype=np.float32),
        midframe=np.ones(n_frames, dtype=np.float32),
    )

    # Override only the attribute under test
    attrs_dict[attr_name] = value

    with pytest.raises(
        TypeError, match=ARRAY_ATTRIBUTE_OBJECT_ERROR_MSG.format(attribute=attr_name)
    ):
        PET(dataobj=data, affine=affine, **attrs_dict)  # type: ignore[arg-type]


@pytest.mark.random_pet_data(
    4,
    (2, 2, 2),
    np.asarray([1.0, 2.0, 3.0, 4.0]),
    np.sum,
    np.asarray([1.0, 1.0, 1.0, 1.0]),
    np.asarray([0.5, 1.5, 2.5, 3.5]),
    4.0,
)
@pytest.mark.parametrize("attr_name", ("frame_time", "uptake", "frame_duration", "midframe"))
@pytest.mark.parametrize(
    ("extra_volume_count", "extra_attribute_count"),
    [(1, 0), (2, 0), (2, 1), (0, 1), (0, 2), (1, 2)],
)
def test_pet_instantiation_attribute_vol_mismatch_error(
    setup_random_pet_data, attr_name, extra_volume_count, extra_attribute_count
):
    pet_dataobj, affine, _, frame_time, uptake, frame_duration, midframe, total_duration = (
        setup_random_pet_data
    )

    n_frames = int(pet_dataobj.shape[-1])
    attrs_dict = dict(
        frame_time=frame_time,
        uptake=uptake,
        frame_duration=frame_duration,
        midframe=midframe,
    )

    # Add extra volumes: simply concatenate the last volume
    if extra_volume_count:
        extra_dwi_dataobj = np.tile(pet_dataobj[..., -1:], (1, extra_volume_count))
        pet_dataobj = np.concatenate((pet_dataobj, extra_dwi_dataobj), axis=-1)
        n_frames = int(pet_dataobj.shape[-1])
    # Add extra values to attribute: simply concatenate the attribute
    if extra_attribute_count:
        base = attrs_dict[attr_name]
        extra_vals = np.repeat(base[-1], extra_attribute_count)
        attrs_dict[attr_name] = np.concatenate((base, extra_vals))

    # Determine which attribute to check
    if attr_name == "frame_time" or extra_volume_count != 0:
        attr_to_check = "frame_time"
    else:
        attr_to_check = attr_name

    attr_val = attrs_dict[attr_to_check]

    with pytest.raises(
        ValueError,
        match=ATTRIBUTE_VOLUME_DIMENSIONALITY_MISMATCH_ERROR.format(
            attribute=attr_to_check, n_frames=n_frames, attr_len=len(attr_val)
        ),
    ):
        PET(dataobj=pet_dataobj, affine=affine, **attrs_dict)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "midframe, expected",
    [
        ([1.0, 4.0], [3.0, 3.0]),
        ([0.0, 5.0, 9.0, 12.0], [5.0, 4.0, 3.0, 3.0]),
    ],
)
def test_compute_frame_duration(midframe, expected):
    midframe = np.array(midframe)
    expected = np.array(expected)
    durations = _compute_frame_duration(midframe)
    np.testing.assert_allclose(durations, expected)


@pytest.mark.parametrize("stat_func", (np.sum, np.mean, np.std))
def test_compute_uptake_statistic(stat_func):
    rng = np.random.default_rng(12345)
    data = rng.random((4, 4, 4, 5), dtype=np.float32)

    expected = stat_func(data.reshape(-1, data.shape[-1]), axis=0)
    obtained = compute_uptake_statistic(data, stat_func=stat_func)
    np.testing.assert_array_equal(obtained, expected)


@pytest.mark.random_pet_data(
    4,
    (2, 2, 2),
    np.asarray([1.0, 2.0, 3.0, 4.0]),
    np.sum,
    np.asarray([1.0, 1.0, 1.0, 1.0]),
    np.asarray([0.5, 1.5, 2.5, 3.5]),
    4.0,
)
def test_from_nii_errors(tmp_path, setup_random_pet_data):
    (
        pet_dataobj,
        affine,
        brainmask_dataobj,
        frame_time,
        uptake,
        frame_duration,
        midframe,
        total_duration,
    ) = setup_random_pet_data

    pet, brainmask = _pet_data_to_nifti(pet_dataobj, affine, brainmask_dataobj.astype(np.uint8))

    pet_fname = tmp_path / "pet.nii.gz"
    brainmask_fname = tmp_path / "brainmask.nii.gz"
    temporal_fname = tmp_path / "temporal.json"
    uptake_fname = tmp_path / "uptake.json"

    nb.save(pet, pet_fname)
    nb.save(brainmask, brainmask_fname)

    # Check frame time
    temporal_data = {
        FRAME_DURATION_KEY: frame_duration.tolist(),
        FRAME_MIDFRAME_KEY: midframe.tolist(),
        SEQUENCE_TOTAL_DURATION_KEY: total_duration,
    }
    with temporal_fname.open("w", encoding="utf-8") as f:
        json.dump(temporal_data, f, ensure_ascii=False, indent=2, sort_keys=True)

    with pytest.raises(
        RuntimeError, match=TEMPORAL_FILE_KEY_ERROR_MSG.format(key=FRAME_TIME_START_KEY)
    ):
        from_nii(
            pet_fname,
            temporal_fname,
            brainmask_file=brainmask_fname,
        )

    # Check uptake
    temporal_data = {
        FRAME_TIME_START_KEY: frame_time.tolist(),
        FRAME_DURATION_KEY: frame_duration.tolist(),
        FRAME_MIDFRAME_KEY: midframe.tolist(),
        SEQUENCE_TOTAL_DURATION_KEY: total_duration,
    }
    with temporal_fname.open("w", encoding="utf-8") as f:
        json.dump(temporal_data, f, ensure_ascii=False, indent=2, sort_keys=True)

    uptake_data = {"mykey": 1.0}
    with uptake_fname.open("w", encoding="utf-8") as f:
        json.dump(uptake_data, f, ensure_ascii=False, indent=2, sort_keys=True)

    uptake_stat_func = np.sum
    with warnings.catch_warnings(record=True) as caught:
        # Suppress exception
        with suppress(Exception):
            from_nii(
                pet_fname,
                temporal_fname,
                brainmask_file=brainmask_fname,
                uptake_file=uptake_fname,
                uptake_stat_func=uptake_stat_func,
            )

    assert str(caught[0].message) == UPTAKE_FILE_KEY_ERROR_MSG.format(func=uptake_stat_func)


@pytest.mark.random_uniform_spatial_data((2, 2, 2, 3), 0.0, 1.0)
@pytest.mark.random_uniform_spatial_data((2, 2, 2, 4), 0.0, 1.0)
@pytest.mark.parametrize(
    "attr_name, use_default_val",
    [
        ("uptake", False),
        ("uptake", True),
        ("frame_duration", False),
        ("frame_duration", True),
        ("midframe", False),
        ("midframe", True),
        ("total_duration", False),
        ("total_duration", True),
    ],
)
def test_from_nii(
    request, tmp_path, setup_random_uniform_spatial_data, attr_name, use_default_val
):
    from nifreeze.data.base import BaseDataset

    rng = request.node.rng

    pet_dataobj, affine = setup_random_uniform_spatial_data
    brainmask_dataobj = rng.choice([True, False], size=pet_dataobj.shape[:3]).astype(np.uint8)

    pet, brainmask = _pet_data_to_nifti(pet_dataobj, affine, brainmask_dataobj)

    n_frames = pet_dataobj.shape[-1]
    frame_time = np.arange(n_frames, dtype=np.float32) + 1
    frame_time -= frame_time[0]

    # Create some random data as default attribute values
    frame_duration = rng.random(n_frames)
    uptake = rng.random(n_frames)
    midframe = rng.random(n_frames)
    total_duration = rng.random(1).item()

    frame_duration = (
        _compute_frame_duration(frame_time)
        if attr_name == "frame_duration" and use_default_val
        else frame_duration
    )
    midframe = (
        frame_time + frame_duration / 2
        if attr_name == "midframe" and use_default_val
        else midframe
    )
    total_duration = (
        float(frame_time[-1] + frame_duration[-1])
        if attr_name == "total_duration" and use_default_val
        else total_duration
    )
    uptake = (
        compute_uptake_statistic(pet_dataobj, np.sum)
        if attr_name == "uptake" and use_default_val
        else uptake
    )
    total_duration = (
        float(frame_time[-1] + frame_duration[-1])
        if attr_name == "total_duration" and use_default_val
        else total_duration
    )

    pet_fname, brainmask_fname, temporal_fname, uptake_fname = _serialize_pet_data(
        pet, brainmask, frame_time, frame_duration, midframe, total_duration, uptake, tmp_path
    )

    # Read back using public API
    pet_obj_from_nii = from_nii(
        pet_fname,
        temporal_fname,
        brainmask_file=brainmask_fname,
        uptake_file=uptake_fname,
    )

    assert isinstance(pet_obj_from_nii, PET)

    attrs_dict: dict[str, np.ndarray | float | None] = dict(
        frame_time=frame_time,
        uptake=uptake,
        frame_duration=frame_duration,
        midframe=midframe,
        total_duration=total_duration,
    )

    # Get all user-defined, named attributes
    attrs_to_check = [
        a.name for a in attrs.fields(PET) if not a.name.startswith("_") and not a.name.isdigit()
    ]
    # No need to check base class attributes: remove them
    base_attrs = [
        a.name
        for a in attrs.fields(BaseDataset)
        if not a.name.startswith("_") and not a.name.isdigit()
    ]
    attrs_to_check = [_ for _ in attrs_to_check if _ not in base_attrs]

    for attr_name in attrs_to_check:
        val_direct = attrs_dict[attr_name]
        val_from_nii = getattr(pet_obj_from_nii, attr_name)

        if val_direct is None or val_from_nii is None:
            assert val_direct is None and val_from_nii is None, f"{attr_name} mismatch"
        else:
            if isinstance(val_direct, np.ndarray):
                assert val_direct.shape == val_from_nii.shape
                assert np.allclose(val_direct, val_from_nii), f"{attr_name} arrays differ"
            else:
                assert math.isclose(val_direct, val_from_nii), f"{attr_name} values differ"


@pytest.mark.random_pet_data(
    4,
    (2, 2, 2),
    np.asarray([1.0, 2.0, 3.0, 4.0]),
    np.sum,
    np.asarray([1.0, 1.0, 1.0, 1.0]),
    np.asarray([0.5, 1.5, 2.5, 3.5]),
    4.0,
)
def test_to_nifti(tmp_path, setup_random_pet_data):
    (
        pet_dataobj,
        affine,
        brainmask_dataobj,
        frame_time,
        uptake,
        frame_duration,
        midframe,
        total_duration,
    ) = setup_random_pet_data

    pet_obj = PET(
        dataobj=pet_dataobj,
        affine=affine,
        brainmask=brainmask_dataobj,
        frame_time=frame_time,
        uptake=uptake,
        frame_duration=frame_duration,
        midframe=midframe,
        total_duration=total_duration,
    )

    out_filename = tmp_path / "random_pet_out.nii.gz"
    pet_obj.to_nifti(str(out_filename))
    assert out_filename.exists()
    loaded_img = load_api(str(out_filename), nb.Nifti1Image)
    assert np.allclose(loaded_img.get_fdata(), pet_obj.dataobj)
    assert np.allclose(loaded_img.affine, pet_obj.affine)
    units = loaded_img.header.get_xyzt_units()
    assert units[0] == "mm"


@pytest.mark.random_pet_data(
    2,
    (2, 2, 2),
    np.asarray([0.0, 5.0]),
    np.sum,
    np.asarray([5.0, 5.0]),
    np.asarray([2.5, 7.5]),
    10.0,
)
def test_round_trip(tmp_path, setup_random_pet_data):
    (
        pet_dataobj,
        affine,
        brainmask_dataobj,
        frame_time,
        uptake,
        frame_duration,
        midframe,
        total_duration,
    ) = setup_random_pet_data

    pet, brainmask = _pet_data_to_nifti(pet_dataobj, affine, brainmask_dataobj.astype(np.uint8))

    pet_fname, _, temporal_fname, uptake_fname = _serialize_pet_data(
        pet, brainmask, frame_time, frame_duration, midframe, total_duration, uptake, tmp_path
    )

    img = load_api(pet_fname, nb.Nifti1Image)
    pet_obj = from_nii(
        pet_fname,
        temporal_fname,
        uptake_file=uptake_fname,
    )
    out_fname = tmp_path / "random_pet_out.nii.gz"
    pet_obj.to_nifti(out_fname)
    assert out_fname.exists()
    loaded_img = load_api(out_fname, nb.Nifti1Image)
    assert np.allclose(loaded_img.affine, img.affine)
    np.testing.assert_allclose(loaded_img.get_fdata(), img.get_fdata())
    units = loaded_img.header.get_xyzt_units()
    assert units[0] == "mm"


def test_equality_operator(tmp_path, setup_random_pet_data):
    (
        pet_dataobj,
        affine,
        brainmask_dataobj,
        frame_time,
        uptake,
        frame_duration,
        midframe,
        total_duration,
    ) = setup_random_pet_data

    pet, brainmask = _pet_data_to_nifti(pet_dataobj, affine, brainmask_dataobj.astype(np.uint8))

    pet_fname, brainmask_fname, temporal_fname, uptake_fname = _serialize_pet_data(
        pet, brainmask, frame_time, frame_duration, midframe, total_duration, uptake, tmp_path
    )

    # Read back using public API
    pet_obj_from_nii = from_nii(
        pet_fname,
        temporal_fname,
        brainmask_file=brainmask_fname,
        uptake_file=uptake_fname,
    )

    # Direct instantiation with the same arrays
    pet_obj_direct = PET(
        dataobj=pet_dataobj,
        affine=affine,
        brainmask=brainmask_dataobj,
        frame_time=frame_time,
        uptake=uptake,
        frame_duration=frame_duration,
        midframe=midframe,
        total_duration=total_duration,
    )

    # Get all user-defined, named attributes
    attrs_to_check = [
        a.name for a in attrs.fields(PET) if not a.name.startswith("_") and not a.name.isdigit()
    ]

    # Sanity checks (element-wise)
    for attr_name in attrs_to_check:
        val_direct = getattr(pet_obj_direct, attr_name)
        val_from_nii = getattr(pet_obj_from_nii, attr_name)

        if val_direct is None or val_from_nii is None:
            assert val_direct is None and val_from_nii is None, f"{attr_name} mismatch"
        else:
            if isinstance(val_direct, np.ndarray):
                assert val_direct.shape == val_from_nii.shape
                assert np.allclose(val_direct, val_from_nii), f"{attr_name} arrays differ"
            else:
                assert math.isclose(val_direct, val_from_nii), f"{attr_name} values differ"

    # Test equality operator
    assert pet_obj_direct == pet_obj_from_nii

    # Test equality operator against an instance from HDF5
    hdf5_filename = tmp_path / "test_pet.h5"
    pet_obj_from_nii.to_filename(hdf5_filename)

    round_trip_pet_obj = PET.from_filename(hdf5_filename)

    # Symmetric equality
    assert pet_obj_from_nii == round_trip_pet_obj
    assert round_trip_pet_obj == pet_obj_from_nii


@pytest.mark.random_pet_data(
    4,
    (2, 2, 2),
    np.asarray([1.0, 2.0, 3.0, 4.0]),
    np.sum,
    np.asarray([1.0, 1.0, 1.0, 1.0]),
    np.asarray([0.5, 1.5, 2.5, 3.5]),
    4.0,
)
def test_pet_set_transform_updates_motion_affines(setup_random_pet_data):
    (
        pet_dataobj,
        affine,
        brainmask_dataobj,
        frame_time,
        uptake,
        frame_duration,
        midframe,
        total_duration,
    ) = setup_random_pet_data

    pet_obj = PET(
        dataobj=pet_dataobj,
        affine=affine,
        brainmask=brainmask_dataobj,
        frame_time=frame_time,
        uptake=uptake,
        frame_duration=frame_duration,
        midframe=midframe,
        total_duration=total_duration,
    )

    idx = 2
    data_before = np.copy(pet_obj.dataobj[..., idx])

    affine = np.eye(4)
    pet_obj.set_transform(idx, affine)

    np.testing.assert_allclose(pet_obj.dataobj[..., idx], data_before)
    assert pet_obj.motion_affines is not None
    assert len(pet_obj.motion_affines) == len(pet_obj)
    assert isinstance(pet_obj.motion_affines[idx], Affine)
    assert np.allclose(pet_obj.motion_affines[idx].matrix, affine)

    vol, aff, time = pet_obj[idx]
    assert aff is pet_obj.motion_affines[idx]


@pytest.mark.random_pet_data(
    4,
    (2, 2, 2),
    np.asarray([1.0, 2.0, 3.0, 4.0]),
    np.sum,
    np.asarray([1.0, 1.0, 1.0, 1.0]),
    np.asarray([0.5, 1.5, 2.5, 3.5]),
    4.0,
)
def test_pet_load(tmp_path, setup_random_pet_data):
    (
        pet_dataobj,
        affine,
        brainmask_dataobj,
        frame_time,
        uptake,
        frame_duration,
        midframe,
        total_duration,
    ) = setup_random_pet_data

    pet, brainmask = _pet_data_to_nifti(pet_dataobj, affine, brainmask_dataobj.astype(np.uint8))

    # Direct instantiation with the same arrays
    pet_obj_direct = PET(
        dataobj=pet_dataobj,
        affine=affine,
        brainmask=brainmask_dataobj,
        frame_time=frame_time,
        uptake=uptake,
        frame_duration=frame_duration,
        midframe=midframe,
        total_duration=total_duration,
    )

    pet_fname, brainmask_fname, temporal_fname, uptake_fname = _serialize_pet_data(
        pet, brainmask, frame_time, frame_duration, midframe, total_duration, uptake, tmp_path
    )

    pet_from_nii_kwargs = {"temporal_file": temporal_fname, "uptake_file": uptake_fname}

    pet_obj_load = nifreeze_load(pet_fname, brainmask_fname, **pet_from_nii_kwargs)

    # Get all user-defined, named attributes
    attrs_to_check = [
        a.name for a in attrs.fields(PET) if not a.name.startswith("_") and not a.name.isdigit()
    ]

    # Sanity checks (element-wise)
    for attr_name in attrs_to_check:
        val_direct = getattr(pet_obj_direct, attr_name)
        val_load = getattr(pet_obj_load, attr_name)

        if val_direct is None or val_load is None:
            assert val_direct is None and val_load is None, f"{attr_name} mismatch"
        else:
            if isinstance(val_direct, np.ndarray):
                assert val_direct.shape == val_load.shape
                assert np.allclose(val_direct, val_load), f"{attr_name} arrays differ"
            else:
                assert math.isclose(val_direct, val_load), f"{attr_name} values differ"
