import nitransforms as nt
import numpy as np

from nifreeze.data.pet import PET
from nifreeze.estimator import PETMotionEstimator


def _create_dataset():
    rng = np.random.default_rng(12345)
    data = rng.random((4, 4, 4, 5), dtype=np.float32)
    affine = np.eye(4, dtype=np.float32)
    mask = np.ones((4, 4, 4), dtype=bool)
    midframe = np.array([10, 20, 30, 40, 50], dtype=np.float32)
    return PET(
        dataobj=data,
        affine=affine,
        brainmask=mask,
        midframe=midframe,
        total_duration=60.0,
    )


def test_petmotionestimator_run(monkeypatch):
    dataset = _create_dataset()

    def _fake_run_registration(*args, **kwargs):
        return nt.linear.Affine(np.eye(4))

    monkeypatch.setattr("nifreeze.estimator._run_registration", _fake_run_registration)

    estimator = PETMotionEstimator()
    affines = estimator.run(dataset)

    assert len(affines) == len(dataset)
    for mat in affines:
        assert np.allclose(mat, np.eye(4))
