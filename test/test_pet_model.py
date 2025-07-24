import numpy as np
import pytest

from nifreeze.data.pet import PET
from nifreeze.model.pet import PETModel


@pytest.fixture
def random_dataset(setup_random_pet_data) -> PET:
    """Create a PET dataset with random data for testing."""

    (
        pet_dataobj,
        affine,
        brainmask_dataobj,
        midframe,
        total_duration,
    ) = setup_random_pet_data

    return PET(
        dataobj=pet_dataobj,
        affine=affine,
        brainmask=brainmask_dataobj,
        midframe=midframe,
        total_duration=total_duration,
    )


@pytest.mark.random_pet_data(5, (4, 4, 4), np.asarray([10.0, 20.0, 30.0, 40.0, 50.0]), 60.0)
def test_petmodel_fit_predict(random_dataset):
    model = PETModel(
        dataset=random_dataset,
        timepoints=random_dataset.midframe,
        xlim=random_dataset.total_duration,
        smooth_fwhm=0,
        thresh_pct=0,
    )

    # Fit on all data
    model.fit_predict(None)
    assert model.is_fitted

    # Predict at a specific timepoint
    vol = model.fit_predict(random_dataset.midframe[2])
    assert vol.shape == random_dataset.shape3d
    assert vol.dtype == random_dataset.dataobj.dtype


@pytest.mark.random_pet_data(5, (4, 4, 4), np.asarray([10.0, 20.0, 30.0, 40.0, 50.0]), 60.0)
def test_petmodel_invalid_init(random_dataset):
    with pytest.raises(TypeError):
        PETModel(dataset=random_dataset)


@pytest.mark.random_pet_data(5, (4, 4, 4), np.asarray([10.0, 20.0, 30.0, 40.0, 50.0]), 60.0)
def test_petmodel_time_check(random_dataset):
    bad_times = np.array([0, 10, 20, 30, 50], dtype=np.float32)
    with pytest.raises(ValueError):
        PETModel(dataset=random_dataset, timepoints=bad_times, xlim=60.0)
