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

import re
from typing import Union

import nibabel as nb
import numpy as np
import pytest

import nifreeze.estimator
from nifreeze.data.base import BaseDataset
from nifreeze.data.dmri.utils import DEFAULT_LOWB_THRESHOLD
from nifreeze.data.filtering import ClippingValueError
from nifreeze.data.pet import PET
from nifreeze.data.pet.utils import compute_uptake_statistic
from nifreeze.estimator import Estimator
from nifreeze.model.base import BaseModel, ExpectationModel
from nifreeze.model.pet import BSplinePETModel
from nifreeze.utils import iterators
from nifreeze.utils.ndimage import load_api

DATAOBJ_SIZE = (7, 5, 5, 5)


class DummyInsiderModel(BaseModel):
    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, **kwargs)

    def fit_predict(self, index: int | None = None, **kwargs):
        # Return the indexed volume
        return self._dataset.dataobj[index]


class DummyDataset(BaseDataset):
    def __init__(self, rng):
        dataobj = rng.uniform(0.0, 1.0, DATAOBJ_SIZE)  # np.ones(DATAOBJ_SIZE)
        super().__init__(
            dataobj=dataobj,
            affine=np.eye(4),
            brainmask=rng.choice([True, False], size=dataobj.shape[1:]).astype(bool),
        )

    def __len__(self):
        return self.dataobj.shape[0]

    def __getitem__(self, idx):
        # Return the indexed volume and a dummy value
        return self.dataobj[idx], None

    def set_transform(self, idx, matrix):
        pass


class DummyDWIDataset(BaseDataset):
    def __init__(self, dwi_dataobj, affine, brainmask_dataobj, b0_dataobj, gradients):
        super().__init__(dataobj=dwi_dataobj, affine=affine, brainmask=brainmask_dataobj)
        self.bzero = b0_dataobj
        self.gradients = gradients

    def __len__(self):
        return self.dataobj.shape[0]

    def __getitem__(self, idx):
        return self.dataobj[idx], None, self.gradients[idx, ...]


class DummyPETDataset(BaseDataset):
    def __init__(self, pet_dataobj, affine, brainmask_dataobj, midframe, total_duration):
        super().__init__(dataobj=pet_dataobj, affine=affine, brainmask=brainmask_dataobj)
        self.midframe = midframe
        self.total_duration = total_duration

    def __len__(self):
        return self.dataobj.shape[0]

    def __getitem__(self, idx):
        return self.dataobj[idx], None, self.midframe[idx]


def test_estimator_invalid_start_index(request):
    """Test that negative start_index raises an error."""
    dataset = DummyDataset(rng=request.node.rng)
    model = DummyInsiderModel(dataset=dataset)

    estimator = Estimator(model, start_index=-1)
    with pytest.raises(ValueError, match=re.escape(iterators.START_INDEX_POSITIVITY_ERROR_MSG)):
        estimator.run(dataset)

    estimator = Estimator(model, start_index=len(dataset))
    with pytest.raises(
        ValueError,
        match=re.escape(
            iterators.START_INDEX_DATA_LENGTH_ERROR_MSG.format(feature=iterators.SIZE_KWARG)
        ),
    ):
        estimator.run(dataset)


def test_estimator_invalid_stop_index(request):
    """Test that stop_index <= start_index raises an error."""
    dataset = DummyDataset(rng=request.node.rng)
    model = DummyInsiderModel(dataset=dataset)

    estimator = Estimator(model, start_index=2, stop_index=2)
    with pytest.raises(ValueError, match=re.escape(iterators.STOP_INDEX_ORDERING_ERROR_MSG)):
        estimator.run(dataset)

    estimator = Estimator(model, stop_index=len(dataset) + 1)
    with pytest.raises(
        ValueError,
        match=re.escape(
            iterators.STOP_INDEX_DATA_LENGTH_ERROR_MSG.format(feature=iterators.SIZE_KWARG)
        ),
    ):
        estimator.run(dataset)


def test_estimator_start_index(request, monkeypatch):
    """Test that estimator respects start_index."""
    recorded_indices = []

    dataset = DummyDataset(rng=request.node.rng)

    def fake_set_transform(self, i, xform):
        recorded_indices.append(i)

    monkeypatch.setattr(
        dataset, "set_transform", fake_set_transform.__get__(dataset, type(dataset))
    )

    class DummyXForm:
        matrix = np.eye(4)

    monkeypatch.setattr(
        nifreeze.estimator,
        "_run_registration",
        lambda *a, **k: DummyXForm(),
    )

    model = DummyInsiderModel(dataset=dataset)

    estimator = Estimator(model, strategy="linear", start_index=3)
    estimator.run(dataset)

    assert recorded_indices == [3, 4, 5, 6]


def test_estimator_clip_reaches_every_volume(request, monkeypatch):
    """A ``clip`` setting applies to every volume, not only the first."""
    recorded_clips = []

    dataset = DummyDataset(rng=request.node.rng)

    def fake_prepare(*args, clip=None, **kwargs):
        recorded_clips.append(args[5] if len(args) > 5 else clip)
        return None, None, None

    class DummyXForm:
        matrix = np.eye(4)

    monkeypatch.setattr(nifreeze.estimator, "_prepare_registration_data", fake_prepare)
    monkeypatch.setattr(nifreeze.estimator, "_run_registration", lambda *a, **k: DummyXForm())

    model = DummyInsiderModel(dataset=dataset)
    Estimator(model, strategy="linear", clip="none").run(dataset)

    assert recorded_clips == ["none"] * len(dataset)


def test_estimator_clip_windows_cached(request, monkeypatch):
    """Each window is estimated once and reused by the next estimator of the chain."""
    dataset = DummyDataset(rng=request.node.rng)
    estimated = []
    received = []

    def fake_window(volume, mask=None):
        estimated.append(volume)
        return float(len(estimated)), float(len(estimated)) + 1.0

    def fake_prepare(*args, window=None, **kwargs):
        received.append((args[3], window))
        return None, None, None

    class DummyXForm:
        matrix = np.eye(4)

    monkeypatch.setattr(nifreeze.estimator, "robust_window", fake_window)
    monkeypatch.setattr(nifreeze.estimator, "_prepare_registration_data", fake_prepare)
    monkeypatch.setattr(nifreeze.estimator, "_run_registration", lambda *a, **k: DummyXForm())

    first = Estimator(DummyInsiderModel(dataset=dataset), strategy="linear")
    Estimator(DummyInsiderModel(dataset=dataset), strategy="linear", prev=first).run(dataset)

    n = len(dataset)
    assert len(estimated) == n
    assert [index for index, _ in received] == list(range(n)) * 2
    assert received[:n] == received[n:]


@pytest.mark.parametrize(("num_threads", "expected"), [(None, 5), (3, 3)])
def test_estimator_registration_settings(request, monkeypatch, num_threads, expected):
    """The run's settings reach every estimator of a chain, under each estimator's own."""
    dataset = DummyDataset(rng=request.node.rng)
    received = []

    class DummyXForm:
        matrix = np.eye(4)

    def fake_registration(*args, **kwargs):
        received.append(kwargs)
        return DummyXForm()

    monkeypatch.setattr(
        nifreeze.estimator, "_prepare_registration_data", lambda *a, **k: (None, None, None)
    )
    monkeypatch.setattr(nifreeze.estimator, "_run_registration", fake_registration)

    first = Estimator(DummyInsiderModel(dataset=dataset), strategy="linear")
    second = Estimator(
        DummyInsiderModel(dataset=dataset), strategy="linear", prev=first, metric="GC"
    )
    threads = {} if num_threads is None else {"num_threads": num_threads}
    second.run(
        dataset,
        align_kwargs={"ants_config": "dwi-to-dwi_level1", "metric": "Mattes"},
        omp_nthreads=5,
        **threads,
    )

    n = len(dataset)
    assert len(received) == 2 * n
    assert all("align_kwargs" not in kwargs for kwargs in received)
    assert all(kwargs["ants_config"] == "dwi-to-dwi_level1" for kwargs in received)
    assert all(kwargs["num_threads"] == expected for kwargs in received)
    assert [kwargs["metric"] for kwargs in received] == ["Mattes"] * n + ["GC"] * n


def test_estimator_debug_dir(request, monkeypatch, tmp_path):
    """With ``debug_dir``, each pass keeps its inputs and an aligned trace per volume."""
    dataset = DummyDataset(rng=request.node.rng)
    shift = np.eye(4)
    shift[:3, 3] = [1.0, -2.0, 0.5]

    class DummyXForm:
        matrix = shift

    workdirs = set()

    def fake_registration(fixed_path, moving_path, vol_idx, dirname, **kwargs):
        workdirs.add(dirname)
        return DummyXForm()

    monkeypatch.setattr(nifreeze.estimator, "_run_registration", fake_registration)
    # A relative debug_dir must reach the registration as an absolute path,
    # because nipype runs ANTs from within the working directory
    monkeypatch.chdir(tmp_path)

    first = Estimator(DummyInsiderModel(dataset=dataset), strategy="linear")
    Estimator(DummyInsiderModel(dataset=dataset), strategy="linear", prev=first).run(
        dataset, clip="none", debug_dir="debug"
    )

    pass_dirs = [tmp_path / "debug" / f"{n:02d}-dummyinsidermodel" for n in range(2)]
    assert workdirs == set(pass_dirs)
    for pass_dir in pass_dirs:
        for i in range(len(dataset)):
            sample = load_api(pass_dir / f"sample_{i:05d}.nii", nb.Nifti1Image)
            aligned = load_api(pass_dir / f"aligned_{i:05d}.nii", nb.Nifti1Image)
            assert np.allclose(aligned.affine, np.linalg.inv(shift) @ sample.affine)


def _smooth_dataset() -> BaseDataset:
    """Smooth volumes, so that sub-voxel shifts change them."""
    grid = np.indices(DATAOBJ_SIZE[1:]).astype("float32")
    volumes = [
        100.0
        * np.exp(-((grid[0] - 2 - 0.1 * k) ** 2 + (grid[1] - 2) ** 2 + (grid[2] - 2) ** 2) / 4)
        for k in range(DATAOBJ_SIZE[0])
    ]
    return BaseDataset(
        dataobj=np.stack(volumes).astype("float32"),
        affine=np.eye(4),
        brainmask=np.ones(DATAOBJ_SIZE[1:], dtype=bool),
    )


def _shifting_registration(monkeypatch, n_volumes: int) -> list:
    """Make each registration return a growing shift; collect the predictions."""
    predictions: list = []
    shifts = iter(np.linspace(0.2, 0.8, n_volumes))

    def fake_prepare(sample, predicted, *args, **kwargs):
        predictions.append(np.copy(predicted))
        return None, None, None

    class DummyXForm:
        def __init__(self, shift):
            self.matrix = np.eye(4)
            self.matrix[1, 3] = shift

    monkeypatch.setattr(nifreeze.estimator, "_prepare_registration_data", fake_prepare)
    monkeypatch.setattr(
        nifreeze.estimator, "_run_registration", lambda *a, **k: DummyXForm(next(shifts))
    )
    return predictions


@pytest.mark.parametrize("given", ["name", "instance"])
@pytest.mark.parametrize("instant_update", [True, False])
def test_estimator_instant_update(monkeypatch, instant_update, given):
    """With the instant update, models train on volumes realigned as soon as they are."""
    dataset = _smooth_dataset()
    original = np.copy(dataset.dataobj)
    predictions = _shifting_registration(monkeypatch, len(dataset))
    model = "avg" if given == "name" else ExpectationModel(dataset)

    views: dict = {}
    Estimator(model, strategy="linear").run(
        dataset, clip="none", instant_update=instant_update, realigned=views
    )

    assert np.array_equal(dataset.dataobj, original)
    if not instant_update:
        assert views == {}
        # Every volume is predicted from the raw volumes
        assert np.allclose(predictions[-1], np.median(original[:-1], axis=0))
        return

    (view,) = views.values()
    # The view holds every volume realigned once, from the original
    assert np.allclose(view.dataobj, dataset.realigned().dataobj)
    # The last volume is predicted from all the others, already realigned
    assert np.allclose(predictions[-1], np.median(view.dataobj[:-1], axis=0))
    assert not np.allclose(predictions[-1], np.median(original[:-1], axis=0))
    if given == "instance":
        # The instance now reads the realigned view
        assert isinstance(model, ExpectationModel)
        assert model._dataset is view


def test_estimator_instance_of_another_dataset(monkeypatch):
    """A model instance built on another dataset keeps reading it."""
    dataset = _smooth_dataset()
    other = _smooth_dataset()
    predictions = _shifting_registration(monkeypatch, len(dataset))
    model = ExpectationModel(other)

    views: dict = {}
    Estimator(model, strategy="linear").run(dataset, clip="none", realigned=views)

    assert views == {}
    assert model._dataset is other
    assert np.allclose(predictions[-1], np.median(other.dataobj[:-1], axis=0))


@pytest.mark.random_pet_data(5, (4, 4, 4), np.asarray([10.0, 20.0, 30.0, 40.0, 50.0]))
def test_estimator_instant_update_pet(monkeypatch, setup_random_pet_data):
    """PET models train on realigned frames, while the frames themselves stay original."""
    pet_dataobj, affine, brainmask_dataobj, _, midframe, total_duration = setup_random_pet_data
    dataset = PET(
        dataobj=pet_dataobj,
        affine=affine,
        brainmask=brainmask_dataobj,
        midframe=midframe,
        total_duration=total_duration,
    )
    original = np.copy(dataset.dataobj)
    _shifting_registration(monkeypatch, len(dataset))
    model = BSplinePETModel(dataset=dataset)

    views: dict = {}
    Estimator(model, strategy="linear").run(dataset, clip="none", realigned=views)

    (view,) = views.values()
    assert model._dataset is view
    assert np.array_equal(dataset.dataobj, original)
    assert np.allclose(view.dataobj, dataset.realigned().dataobj)
    assert not np.allclose(view.dataobj, original)


@pytest.mark.parametrize("modality", ["base", "pet"])
def test_estimator_chain_initializes_from_previous_estimate(request, monkeypatch, modality):
    """The second estimator of a chain starts each registration at the first one's estimate."""
    dataobj = request.node.rng.uniform(0.0, 1.0, DATAOBJ_SIZE).astype("float32")
    brainmask = np.ones(DATAOBJ_SIZE[1:], dtype=bool)
    dataset: BaseDataset
    if modality == "pet":
        dataset = PET(
            dataobj=dataobj,
            affine=np.eye(4),
            brainmask=brainmask,
            midframe=10.0 * np.arange(DATAOBJ_SIZE[0]) + 5.0,
            total_duration=10.0 * DATAOBJ_SIZE[0],
        )
    else:
        dataset = BaseDataset(dataobj=dataobj, affine=np.eye(4), brainmask=brainmask)
    inits: list[np.ndarray | None] = []
    shifts = iter(range(1, 2 * DATAOBJ_SIZE[0] + 1))

    def fake_prepare(*args, init_affine=None, **kwargs):
        inits.append(None if init_affine is None else np.array(init_affine))
        return None, None, None

    class DummyXForm:
        def __init__(self, shift):
            self.matrix = np.eye(4)
            self.matrix[0, 3] = shift

    monkeypatch.setattr(nifreeze.estimator, "_prepare_registration_data", fake_prepare)
    monkeypatch.setattr(
        nifreeze.estimator, "_run_registration", lambda *a, **k: DummyXForm(next(shifts))
    )

    first = Estimator(DummyInsiderModel(dataset=dataset), strategy="linear")
    Estimator(DummyInsiderModel(dataset=dataset), strategy="linear", prev=first).run(
        dataset, clip="none"
    )

    n = len(dataset)
    # The first pass starts from the identity
    assert all(init is None for init in inits[:n])
    # The second pass starts from the transform the first pass estimated for each volume
    second = [init for init in inits[n:] if init is not None]
    assert [init[0, 3] for init in second] == list(range(1, n + 1))


def test_estimator_clip_degenerate_window(request, monkeypatch):
    """A volume whose window cannot be estimated is registered without clipping."""
    dataset = DummyDataset(rng=request.node.rng)
    received = []

    def fake_window(volume, mask=None):
        raise ClippingValueError("degenerate")

    def fake_prepare(*args, clip=None, window=None, **kwargs):
        received.append((clip, window))
        return None, None, None

    class DummyXForm:
        matrix = np.eye(4)

    monkeypatch.setattr(nifreeze.estimator, "robust_window", fake_window)
    monkeypatch.setattr(nifreeze.estimator, "_prepare_registration_data", fake_prepare)
    monkeypatch.setattr(nifreeze.estimator, "_run_registration", lambda *a, **k: DummyXForm())

    with pytest.warns(UserWarning, match="without clipping"):
        Estimator(DummyInsiderModel(dataset=dataset), strategy="linear").run(dataset)

    assert received == [("none", None)] * len(dataset)


def test_estimator_start_and_end_index(request, monkeypatch):
    """Test that estimator respects both start_index and end_index."""
    recorded_indices = []

    dataset = DummyDataset(rng=request.node.rng)

    def fake_set_transform(self, i, xform):
        recorded_indices.append(i)

    monkeypatch.setattr(
        dataset, "set_transform", fake_set_transform.__get__(dataset, type(dataset))
    )

    class DummyXForm:
        matrix = np.eye(4)

    monkeypatch.setattr(
        nifreeze.estimator,
        "_run_registration",
        lambda *a, **k: DummyXForm(),
    )

    model = DummyInsiderModel(dataset=dataset)

    estimator = Estimator(model, strategy="linear", start_index=2, stop_index=5)
    estimator.run(dataset)

    # Should only process indices 2-4 (stop_index is exclusive)
    assert recorded_indices == [2, 3, 4]


# The random DWI volumes are 2x2x2, too small to estimate a clipping window
@pytest.mark.filterwarnings("ignore:Registering volume .* without clipping:UserWarning")
@pytest.mark.parametrize(
    "strategy, iterator_func, modality",
    [
        ("linear", iterators.linear_iterator, "dwi"),
        ("linear", iterators.linear_iterator, "pet"),
        ("random", iterators.random_iterator, "dwi"),
        ("random", iterators.random_iterator, "pet"),
        ("centralsym", iterators.centralsym_iterator, "dwi"),
        ("centralsym", iterators.centralsym_iterator, "pet"),
        ("monotonic_value", iterators.monotonic_value_iterator, "dwi"),
        ("monotonic_value", iterators.monotonic_value_iterator, "pet"),
    ],
)
def test_estimator_iterator_index_match(
    monkeypatch, setup_random_dwi_data, setup_random_pet_data, strategy, iterator_func, modality
):
    dataset: Union["DummyDWIDataset", "DummyPETDataset"]  # Avoids type annotation errors
    if modality == "dwi":
        dwi_dataobj, affine, brainmask_dataobj, gradients, b0_thres = setup_random_dwi_data

        b0s_mask = gradients[:, -1] <= b0_thres
        b0_dataobj = dwi_dataobj[b0s_mask].squeeze()

        dataset = DummyDWIDataset(dwi_dataobj, affine, brainmask_dataobj, b0_dataobj, gradients)
        bvals = gradients[:, -1][gradients[:, -1] > DEFAULT_LOWB_THRESHOLD]
        kwargs = {"bvals": bvals}
    elif modality == "pet":
        (
            pet_dataobj,
            affine,
            brainmask_dataobj,
            _,
            midframe,
            total_duration,
        ) = setup_random_pet_data

        dataset = DummyPETDataset(pet_dataobj, affine, brainmask_dataobj, midframe, total_duration)
        uptake = compute_uptake_statistic(pet_dataobj, stat_func=np.sum)
        kwargs = {"uptake": uptake}
    else:
        raise NotImplementedError(f"{modality} not implemented")

    # Patch set_transform to record indices and matrices
    recorded_indices = []
    recorded_matrices = []

    # Make this accept `self` so it behaves as a proper instance method
    def fake_set_transform(self, i, xform):
        recorded_indices.append(i)
        recorded_matrices.append(xform)

    monkeypatch.setattr(
        dataset, "set_transform", fake_set_transform.__get__(dataset, type(dataset))
    )

    # Patch registration to return identity matrix
    class DummyXForm:
        matrix = np.eye(4)

    monkeypatch.setattr(
        nifreeze.estimator,
        "_run_registration",
        lambda *a, **k: DummyXForm(),
    )

    model = DummyInsiderModel(dataset=dataset)
    estimator = Estimator(model, strategy=strategy)
    estimator.run(dataset, **kwargs)

    # Effective iterator size: because we always pass bvals/uptake,
    # iterators may infer the traversal length from those vectors rather than
    # len(dataset).
    if modality == "dwi":
        iter_size = len(bvals)
    else:  # pet
        iter_size = len(uptake)

    # Get expected indices
    if strategy == "linear":
        expected_indices = list(iterator_func(size=iter_size))
    elif strategy == "random":
        expected_indices = sorted(iterator_func(size=iter_size, seed=42))
        recorded_indices_sorted = sorted(recorded_indices)
        assert recorded_indices_sorted == expected_indices
        return
    elif strategy == "centralsym":
        expected_indices = list(iterator_func(size=iter_size))
    elif strategy == "monotonic_value":
        if modality == "dwi":
            it_kwargs = {iterators.BVALS_KWARG: bvals}
        elif modality == "pet":
            it_kwargs = {iterators.UPTAKE_KWARG: uptake}
        else:
            raise NotImplementedError(f"Modality {modality} not implemented")
        expected_indices = list(iterator_func(**it_kwargs))
    else:
        raise ValueError(f"Unknown strategy {strategy}")

    # Assert indices and matrices
    assert recorded_indices == expected_indices
    assert all(np.allclose(mat, np.eye(4)) for mat in recorded_matrices)
