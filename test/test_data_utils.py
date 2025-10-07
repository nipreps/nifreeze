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

import nibabel as nb
import numpy as np
import numpy.testing as npt

from nifreeze.data.utils import apply_affines


def test_apply_affines(request, tmp_path):
    rng = request.node.rng

    # Create synthetic dataset
    nii_data = rng.random((10, 10, 10, 10))

    # Generate Nifti1Image
    nii = nb.Nifti1Image(nii_data, np.eye(4))

    # Generate synthetic affines
    em_affines = np.expand_dims(np.eye(4), 0).repeat(nii_data.shape[-1], 0)

    nii_t = apply_affines(nii, em_affines)

    npt.assert_allclose(nii.dataobj, nii_t.dataobj)
    npt.assert_array_equal(nii.affine, nii_t.affine)

    # Test with output filename
    out_fname = tmp_path / "affine.nii.gz"
    nii_t_file = apply_affines(nii, em_affines, output_filename=out_fname)
    npt.assert_allclose(nii.dataobj, nii_t_file.dataobj)
    npt.assert_array_equal(nii.affine, nii_t_file.affine)
    assert out_fname.exists()
    # The saved file should load and match the expected result
    nii_loaded = nb.load(out_fname)
    npt.assert_allclose(nii.dataobj, nii_loaded.dataobj)
    npt.assert_array_equal(nii.affine, nii_loaded.affine)
