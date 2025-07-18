import json

import nibabel as nb
import numpy as np
import pytest

from nifreeze.data.pet import PET, from_nii


def test_from_nii_requires_frame_time(tmp_path):
    data = np.zeros((2, 2, 2, 2), dtype=np.float32)
    img = nb.Nifti1Image(data, np.eye(4))
    fname = tmp_path / "pet.nii.gz"
    img.to_filename(fname)

    with pytest.raises(RuntimeError, match="frame_time must be provided"):
        from_nii(fname)


def test_pet_load(tmp_path):
    data = np.zeros((2, 2, 2, 2), dtype=np.float32)
    affine = np.eye(4)
    img = nb.Nifti1Image(data, affine)
    fname = tmp_path / "pet.nii.gz"
    img.to_filename(fname)

    json_file = tmp_path / "pet.json"
    metadata = {
        "FrameDuration": [1.0, 1.0],
        "FrameTimesStart": [0.0, 1.0],
    }
    json_file.write_text(json.dumps(metadata))

    pet_obj = PET.load(fname, json_file)

    assert pet_obj.dataobj.shape == data.shape
    assert np.allclose(pet_obj.midframe, [0.5, 1.5])
    assert pet_obj.total_duration == 2.0
