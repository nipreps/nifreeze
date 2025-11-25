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
from pathlib import Path
from typing import Any, Type

import attrs
import h5py
import nibabel as nb
import numpy as np
import pytest
from nitransforms.linear import Affine

from nifreeze.data import load as nifreeze_load
from nifreeze.data.pet.base import (
    ARRAY_ATTRIBUTE_NDIM_ERROR_MSG,
    ARRAY_ATTRIBUTE_OBJECT_ERROR_MSG,
    ATTRIBUTE_ABSENCE_ERROR_MSG,
    ATTRIBUTE_VOLUME_DIMENSIONALITY_MISMATCH_ERROR_MSG,
    PET,
    SCALAR_ATTRIBUTE_ERROR_MSG,
    TEMPORAL_ATTRIBUTE_INCONSISTENCY_ERROR_MSG,
    format_array_like,
    format_scalar_like,
    validate_1d_array,
)
from nifreeze.data.pet.io import FRAME_TIME_START_KEY, TEMPORAL_FILE_KEY_ERROR_MSG, from_nii
from nifreeze.data.pet.utils import compute_temporal_markers, compute_uptake_statistic
from nifreeze.utils.ndimage import load_api


def _pet_data_to_nifti(pet_dataobj, affine, brainmask_dataobj):
    pet = nb.Nifti1Image(pet_dataobj, affine)
    brainmask = nb.Nifti1Image(brainmask_dataobj, affine)

    return pet, brainmask


def _serialize_pet_data(pet, brainmask, frame_time, _tmp_path):
    pet_fname = _tmp_path / "pet.nii.gz"
    brainmask_fname = _tmp_path / "brainmask.nii.gz"
    temporal_fname = _tmp_path / "temporal.json"

    nb.save(pet, pet_fname)
    nb.save(brainmask, brainmask_fname)

    temporal_data = {FRAME_TIME_START_KEY: frame_time.tolist()}
    with temporal_fname.open("w", encoding="utf-8") as f:
        json.dump(temporal_data, f, ensure_ascii=False, indent=2, sort_keys=True)

    return pet_fname, brainmask_fname, temporal_fname


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
    "attr_name, value, expected_exc, expected_msg",
    [
        ("any_name", None, ValueError, ATTRIBUTE_ABSENCE_ERROR_MSG),
        ("any_name", True, ValueError, SCALAR_ATTRIBUTE_ERROR_MSG),
        ("any_name", (2, 2), ValueError, SCALAR_ATTRIBUTE_ERROR_MSG),
        ("any_name", np.asarray([1.0, 2.0]), ValueError, SCALAR_ATTRIBUTE_ERROR_MSG),
    ],
)
def test_format_scalar_errors(attr_name, value, expected_exc, expected_msg):
    # Produce a valid attrs.Attribute for the test
    dummy_attr_cls: Type[Any] = attrs.make_class("Dummy", {attr_name: attrs.field()})
    dummy_attr = dummy_attr_cls.__attrs_attrs__[0]

    with pytest.raises(expected_exc, match=expected_msg.format(attribute=attr_name)):
        format_scalar_like(value, dummy_attr)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [[1.0, 2.0, 3.0, 4.0], (1.0, 2.0, 3.0, 4.0)])
@pytest.mark.parametrize("attr_name", ("midframe",))
def test_format_array_like(value, attr_name):
    # Produce a valid attrs.Attribute for the test
    dummy_attr_cls: Type[Any] = attrs.make_class("Dummy", {attr_name: attrs.field()})
    dummy_attr = dummy_attr_cls.__attrs_attrs__[0]

    obtained = format_array_like(value, dummy_attr)
    assert isinstance(obtained, np.ndarray)
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


@pytest.mark.random_pet_data(4, (2, 2, 2), np.asarray([1.0, 2.0, 3.0, 4.0]))
@pytest.mark.parametrize("attr_name", ("midframe",))
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
    pet_dataobj, affine, _, _, midframe, total_duration = setup_random_pet_data

    attrs_dict = dict(
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
@pytest.mark.parametrize("attr_name", ("midframe", "total_duration"))
def test_pet_instantiation_attribute_convert_absence_errors(
    setup_random_uniform_spatial_data,
    attr_name,
):
    data, affine = setup_random_uniform_spatial_data

    n_frames = data.shape[-1]
    # Create a dict with default valid attribute values
    attrs_dict: dict[str, np.ndarray | float | None] = dict(
        midframe=np.ones(n_frames, dtype=np.float32),
        total_duration=1.0,
    )

    # Override only the attribute under test
    attrs_dict[attr_name] = None

    with pytest.raises(ValueError, match=ATTRIBUTE_ABSENCE_ERROR_MSG.format(attribute=attr_name)):
        PET(dataobj=data, affine=affine, **attrs_dict)  # type: ignore[arg-type]


@pytest.mark.random_uniform_spatial_data((2, 2, 2, 4), 0.0, 1.0)
@pytest.mark.parametrize("attr_name", ("midframe",))
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
    # Create a dict with some valid attributes
    attrs_dict = dict(
        midframe=np.ones(n_frames, dtype=np.float32),
        total_duration=n_frames + 1,
    )

    # Override only the attribute under test
    attrs_dict[attr_name] = value

    with pytest.raises(
        TypeError, match=ARRAY_ATTRIBUTE_OBJECT_ERROR_MSG.format(attribute=attr_name)
    ):
        PET(dataobj=data, affine=affine, **attrs_dict)  # type: ignore[arg-type]


@pytest.mark.random_pet_data(4, (2, 2, 2), np.asarray([1.0, 2.0, 3.0, 4.0]))
@pytest.mark.parametrize("attr_name", ("midframe",))
@pytest.mark.parametrize(
    ("extra_volume_count", "extra_attribute_count"),
    [(1, 0), (2, 0), (2, 1), (0, 1), (0, 2), (1, 2)],
)
def test_pet_instantiation_attribute_vol_mismatch_error(
    setup_random_pet_data, attr_name, extra_volume_count, extra_attribute_count
):
    pet_dataobj, affine, _, _, midframe, total_duration = setup_random_pet_data

    n_frames = int(pet_dataobj.shape[-1])
    attrs_dict = dict(
        midframe=midframe,
        total_duration=total_duration,
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

    attr_val = attrs_dict[attr_name]

    with pytest.raises(
        ValueError,
        match=ATTRIBUTE_VOLUME_DIMENSIONALITY_MISMATCH_ERROR_MSG.format(
            attribute=attr_name, n_frames=n_frames, attr_len=len(attr_val)
        ),
    ):
        PET(dataobj=pet_dataobj, affine=affine, **attrs_dict)  # type: ignore[arg-type]


@pytest.mark.random_pet_data(4, (2, 2, 2), np.asarray([1.0, 2.0, 3.0, 4.0]))
@pytest.mark.random_pet_data(3, (2, 2, 2), np.asarray([1.0, 2.0, 3.0]))
@pytest.mark.parametrize(
    "attr_name, excess_value",
    [
        ("midframe", 1.0),
        ("midframe", 2.0),
        ("total_duration", 0.0),
        ("total_duration", -1.0),
        ("total_duration", -2.0),
    ],
)
def test_pet_instantiation_attribute_inconsistency_error(
    setup_random_pet_data, attr_name, excess_value
):
    pet_dataobj, affine, _, _, midframe, total_duration = setup_random_pet_data

    if attr_name == "midframe":
        midframe[-1] = total_duration + excess_value
    elif attr_name == "total_duration":
        total_duration = midframe[-1] + excess_value

    attrs_dict = dict(
        midframe=midframe,
        total_duration=total_duration,
    )

    with pytest.raises(
        ValueError,
        match=TEMPORAL_ATTRIBUTE_INCONSISTENCY_ERROR_MSG.format(
            total_duration=total_duration, last_midframe=midframe[-1]
        ),
    ):
        PET(dataobj=pet_dataobj, affine=affine, **attrs_dict)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "frame_time, expected_midframe, expected_total_duration",
    [
        ([1.0, 4.0], [1.5, 4.5], 6.0),
        ([0.0, 5.0, 9.0, 12.0], [2.5, 7.0, 10.5, 13.5], 15.0),
    ],
)
def test_compute_temporal_markers(frame_time, expected_midframe, expected_total_duration):
    frame_time = np.array(frame_time)
    expected_midframe = np.array(expected_midframe)
    midframe, total_duration = compute_temporal_markers(frame_time)
    np.testing.assert_allclose(midframe, expected_midframe)
    assert np.isclose(total_duration, expected_total_duration)


@pytest.mark.parametrize("stat_func", (np.sum, np.mean, np.std))
def test_compute_uptake_statistic(stat_func):
    rng = np.random.default_rng(12345)
    data = rng.random((4, 4, 4, 5), dtype=np.float32)

    expected = stat_func(data.reshape(-1, data.shape[-1]), axis=0)
    obtained = compute_uptake_statistic(data, stat_func=stat_func)
    np.testing.assert_array_equal(obtained, expected)


@pytest.mark.random_pet_data(4, (2, 2, 2), np.asarray([1.0, 2.0, 3.0, 4.0]))
def test_from_nii_errors(tmp_path, setup_random_pet_data):
    pet_dataobj, affine, brainmask_dataobj, frame_time, midframe, total_duration = (
        setup_random_pet_data
    )

    pet, brainmask = _pet_data_to_nifti(pet_dataobj, affine, brainmask_dataobj.astype(np.uint8))

    pet_fname = tmp_path / "pet.nii.gz"
    brainmask_fname = tmp_path / "brainmask.nii.gz"
    temporal_fname = tmp_path / "temporal.json"

    nb.save(pet, pet_fname)
    nb.save(brainmask, brainmask_fname)

    # Check frame time
    temporal_data = {"any_key": frame_time.tolist()}
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


@pytest.mark.random_pet_data(3, (2, 2, 2), np.asarray([1.0, 4.0, 6.0]))
@pytest.mark.random_pet_data(4, (2, 2, 2), np.asarray([1.0, 2.0, 3.0, 4.0]))
def test_from_nii(tmp_path, setup_random_pet_data):
    from nifreeze.data.base import BaseDataset

    pet_dataobj, affine, brainmask_dataobj, frame_time, midframe, total_duration = (
        setup_random_pet_data
    )

    pet, brainmask = _pet_data_to_nifti(pet_dataobj, affine, brainmask_dataobj.astype(np.uint8))

    pet_fname, brainmask_fname, temporal_fname = _serialize_pet_data(
        pet, brainmask, frame_time, tmp_path
    )

    # Read back using public API
    pet_obj_from_nii = from_nii(pet_fname, temporal_fname, brainmask_file=brainmask_fname)

    assert isinstance(pet_obj_from_nii, PET)

    attrs_dict: dict[str, np.ndarray | float | None] = dict(
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


@pytest.mark.random_pet_data(4, (2, 2, 2), np.asarray([1.0, 2.0, 3.0, 4.0]))
def test_to_nifti(tmp_path, setup_random_pet_data):
    pet_dataobj, affine, brainmask_dataobj, _, midframe, total_duration = setup_random_pet_data

    pet_obj = PET(
        dataobj=pet_dataobj,
        affine=affine,
        brainmask=brainmask_dataobj,
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


@pytest.mark.random_pet_data(2, (2, 2, 2), np.asarray([0.0, 5.0]))
def test_round_trip(tmp_path, setup_random_pet_data):
    pet_dataobj, affine, brainmask_dataobj, frame_time, midframe, total_duration = (
        setup_random_pet_data
    )

    pet, brainmask = _pet_data_to_nifti(pet_dataobj, affine, brainmask_dataobj.astype(np.uint8))

    pet_fname, _, temporal_fname = _serialize_pet_data(pet, brainmask, frame_time, tmp_path)

    img = load_api(pet_fname, nb.Nifti1Image)
    pet_obj = from_nii(pet_fname, temporal_fname)
    out_fname = tmp_path / "random_pet_out.nii.gz"
    pet_obj.to_nifti(out_fname)
    assert out_fname.exists()
    loaded_img = load_api(out_fname, nb.Nifti1Image)
    assert np.allclose(loaded_img.affine, img.affine)
    np.testing.assert_allclose(loaded_img.get_fdata(), img.get_fdata())
    units = loaded_img.header.get_xyzt_units()
    assert units[0] == "mm"


def test_equality_operator(tmp_path, setup_random_pet_data):
    pet_dataobj, affine, brainmask_dataobj, frame_time, midframe, total_duration = (
        setup_random_pet_data
    )

    pet, brainmask = _pet_data_to_nifti(pet_dataobj, affine, brainmask_dataobj.astype(np.uint8))

    pet_fname, brainmask_fname, temporal_fname = _serialize_pet_data(
        pet, brainmask, frame_time, tmp_path
    )

    # Read back using public API
    pet_obj_from_nii = from_nii(pet_fname, temporal_fname, brainmask_file=brainmask_fname)

    # Direct instantiation with the same arrays
    pet_obj_direct = PET(
        dataobj=pet_dataobj,
        affine=affine,
        brainmask=brainmask_dataobj,
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


@pytest.mark.random_pet_data(4, (2, 2, 2), np.asarray([1.0, 2.0, 3.0, 4.0]))
def test_pet_set_transform_updates_motion_affines(setup_random_pet_data):
    pet_dataobj, affine, brainmask_dataobj, _, midframe, total_duration = setup_random_pet_data

    pet_obj = PET(
        dataobj=pet_dataobj,
        affine=affine,
        brainmask=brainmask_dataobj,
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


@pytest.mark.random_pet_data(4, (2, 2, 2), np.asarray([1.0, 2.0, 3.0, 4.0]))
def test_pet_load(tmp_path, setup_random_pet_data):
    pet_dataobj, affine, brainmask_dataobj, frame_time, midframe, total_duration = (
        setup_random_pet_data
    )

    pet, brainmask = _pet_data_to_nifti(pet_dataobj, affine, brainmask_dataobj.astype(np.uint8))

    # Direct instantiation with the same arrays
    pet_obj_direct = PET(
        dataobj=pet_dataobj,
        affine=affine,
        brainmask=brainmask_dataobj,
        midframe=midframe,
        total_duration=total_duration,
    )

    pet_fname, brainmask_fname, temporal_fname = _serialize_pet_data(
        pet, brainmask, frame_time, tmp_path
    )

    pet_from_nii_kwargs = {"temporal_file": temporal_fname}

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


def test_pet_from_filename_keeps_handle(random_dataset: PET, tmp_path: Path):
    h5_file = tmp_path / "pet_lazy.h5"
    random_dataset.to_filename(h5_file)

    loaded = PET.from_filename(h5_file, keep_file_open=True)

    try:
        assert isinstance(loaded.dataobj, h5py.Dataset)
        assert isinstance(loaded.total_duration, float)
        assert loaded._file_handle is not None
        assert loaded._file_handle.id.valid
    finally:
        loaded.close()
