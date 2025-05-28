import numpy as np
import nibabel as nb
import pytest

from nifreeze.data.pet import PET
from nifreeze.model.pet import PETModel


def _make_dataset(rng):
    data = rng.random((4, 4, 4, 5), dtype=np.float32)
    affine = np.eye(4, dtype=np.float32)
    midframe = np.arange(5, 50, 10, dtype=np.float32)
    return PET(dataobj=data, affine=affine, midframe=midframe, total_duration=50.0)


def test_pet_model_fit_predict(request):
    rng = request.node.rng
    dataset = _make_dataset(rng)
    model = PETModel(
        dataset,
        timepoints=dataset.midframe,
        xlim=dataset.total_duration,
        smooth_fwhm=0,
        thresh_pct=0,
    )

    # Fit the model once on the whole dataset
    assert model.fit_predict(None) is None

    # Predict one frame
    predicted = model.fit_predict(2)
    assert predicted.shape == dataset.dataobj.shape[:3]


def test_pet_model_boundaries(request):
    rng = request.node.rng
    data = rng.random((2, 2, 2, 3), dtype=np.float32)
    affine = np.eye(4, dtype=np.float32)

    # First frame too early
    midframe = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    with pytest.raises(ValueError):
        PETModel(
            PET(dataobj=data, affine=affine, midframe=midframe, total_duration=2.0),
            timepoints=midframe,
            xlim=2.0,
            smooth_fwhm=0,
            thresh_pct=0,
        )

    # Last frame beyond duration
    midframe = np.array([0.5, 1.5, 2.0], dtype=np.float32)
    with pytest.raises(ValueError):
        PETModel(
            PET(dataobj=data, affine=affine, midframe=midframe, total_duration=2.0),
            timepoints=midframe,
            xlim=2.0,
            smooth_fwhm=0,
            thresh_pct=0,
        )
        
        
def test_first_last_frames_stable():
    data = np.stack([np.full((2, 2, 2), i, dtype=np.float32) for i in range(5)], axis=-1)
    midframe = np.linspace(1, 5, num=5, dtype=np.float32)
    pet = PET(dataobj=data, affine=np.eye(4), midframe=midframe, total_duration=6.0)
    model = PETModel(pet, timepoints=midframe, xlim=pet.total_duration, n_ctrl=len(midframe))
    first = model.fit_predict(midframe[0])
    last = model.fit_predict(midframe[-1])
    np.testing.assert_allclose(first, data[..., 0])
    np.testing.assert_allclose(last, data[..., -1])
