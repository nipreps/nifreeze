import numpy as np
import nibabel as nb
import pytest

from nifreeze.data.pet import from_nii


def test_from_nii_requires_frame_time(tmp_path):
    data = np.zeros((2, 2, 2, 2), dtype=np.float32)
    img = nb.Nifti1Image(data, np.eye(4))
    fname = tmp_path / "pet.nii.gz"
    img.to_filename(fname)

    with pytest.raises(RuntimeError, match="frame_time must be provided"):
        from_nii(fname)
