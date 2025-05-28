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
"""Unit tests for the PET data structure."""

import numpy as np
from nifreeze.data.pet import PET


def _create_pet_dataset(request, n_frames=6):
    rng = request.node.rng
    data = rng.random((4, 4, 4, n_frames), dtype=np.float32)
    affine = np.eye(4, dtype=np.float32)
    midframe = np.linspace(10.0, 10.0 * n_frames, n_frames, dtype=np.float32)
    total_duration = float(midframe[-1] + 10.0)
    return PET(
        dataobj=data,
        affine=affine,
        midframe_time=midframe,
        total_duration=total_duration,
    )


def test_lofo_split_padding(request):
    dataset = _create_pet_dataset(request, n_frames=7)
    (train, times), _ = dataset.lofo_split(3)
    assert train.shape[-1] == times.shape[0]


def test_lofo_split_pad_mode(request):
    dataset = _create_pet_dataset(request, n_frames=7)
    (train, times), _ = dataset.lofo_split(3, pad_mode="reflect")
    assert train.shape[-1] == times.shape[0]
