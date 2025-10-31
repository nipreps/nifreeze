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
"""py.test configuration."""

import os
from pathlib import Path

import nibabel as nb
import nitransforms as nt
import numpy as np
import pytest
from dipy.core.geometry import normalized_vector
from dipy.io.gradients import read_bvals_bvecs

from nifreeze.data.dmri import DWI

test_data_env = os.getenv("TEST_DATA_HOME", str(Path.home() / "nifreeze-tests"))
test_output_dir = os.getenv("TEST_OUTPUT_DIR")
test_workdir = os.getenv("TEST_WORK_DIR")

_datadir = (Path(__file__).parent / "data").absolute()


def pytest_report_header(config):
    return f"""\
TEST_DATA_HOME={test_data_env}.
TEST_OUTPUT_DIR={test_output_dir or "<unset> (output files will be discarded)"}.
TEST_WORK_DIR={test_workdir or "<unset> (intermediate files will be discarded)"}.
"""


@pytest.fixture(autouse=True)
def doctest_imports(doctest_namespace):
    """Populates doctests with some conveniency imports."""
    doctest_namespace["np"] = np
    doctest_namespace["nb"] = nb
    doctest_namespace["os"] = os
    doctest_namespace["Path"] = Path
    doctest_namespace["repodata"] = _datadir


@pytest.fixture(scope="session")
def outdir():
    """Determine if test artifacts should be stored somewhere or deleted."""
    return None if test_output_dir is None else Path(test_output_dir)


@pytest.fixture(scope="session")
def datadir():
    """Return a data path outside the package's structure (i.e., large datasets)."""
    return Path(test_data_env)


@pytest.fixture(scope="session")
def repodata():
    """Return the path to this repository's test data folder."""
    return _datadir


def pytest_addoption(parser):
    parser.addoption(
        "--warnings-as-errors",
        action="store_true",
        help="Consider all uncaught warnings as errors.",
    )


@pytest.fixture(scope="session")
def motion_data(tmp_path_factory, datadir):
    # Temporary directory for session-scoped fixtures
    tmp_path = tmp_path_factory.mktemp("motion_test_data")

    dwdata = DWI.from_filename(datadir / "dwi.h5")
    b0nii = nb.Nifti1Image(dwdata.bzero, dwdata.affine, None)
    masknii = (
        nb.Nifti1Image(dwdata.brainmask.astype(np.uint8), dwdata.affine, None)
        if dwdata.brainmask is not None
        else None
    )

    # Generate a list of large-yet-plausible bulk-head motion
    xfms = nt.linear.LinearTransformsMapping(
        [
            nb.affines.from_matvec(nb.eulerangles.euler2mat(x=0.03, z=0.005), (0.8, 0.2, 0.2)),
            nb.affines.from_matvec(nb.eulerangles.euler2mat(x=0.02, z=0.005), (0.8, 0.2, 0.2)),
            nb.affines.from_matvec(nb.eulerangles.euler2mat(x=0.02, z=0.02), (0.4, 0.2, 0.2)),
            nb.affines.from_matvec(nb.eulerangles.euler2mat(x=-0.02, z=0.02), (0.4, 0.2, 0.2)),
            nb.affines.from_matvec(nb.eulerangles.euler2mat(x=-0.02, z=0.002), (0.0, 0.2, 0.2)),
            nb.affines.from_matvec(nb.eulerangles.euler2mat(y=-0.02, z=0.002), (0.0, 0.2, 0.2)),
            nb.affines.from_matvec(nb.eulerangles.euler2mat(y=-0.01, z=0.002), (0.0, 0.4, 0.2)),
        ],
        reference=b0nii,
    )

    # Induce motion into dataset (i.e., apply the inverse transforms)
    moved_nii = nt.resampling.apply(~xfms, b0nii, reference=b0nii)

    # Save the moved dataset for debugging or further processing
    moved_path = tmp_path / "test.nii.gz"
    ground_truth_path = tmp_path / "ground_truth.nii.gz"
    moved_nii.to_filename(moved_path)
    nt.resampling.apply(xfms, moved_nii).to_filename(ground_truth_path)

    # Wrap into dataset object
    dwi_motion = DWI(
        dataobj=np.asanyarray(moved_nii.dataobj),
        affine=b0nii.affine,
        bzero=dwdata.bzero,
        gradients=dwdata.gradients[..., : len(xfms)],
        brainmask=dwdata.brainmask,
    )

    # Return data as a dictionary (or any format that makes sense for your tests)
    return {
        "b0nii": b0nii,
        "masknii": masknii,
        "moved_nii": moved_nii,
        "xfms": xfms,
        "moved_path": moved_path,
        "ground_truth_path": ground_truth_path,
        "moved_nifreeze": dwi_motion,
    }


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    have_werrors = os.getenv("NIFREEZE_WERRORS", False)
    have_werrors = session.config.getoption("--warnings-as-errors", False) or have_werrors
    if have_werrors:
        # Check if there were any warnings during the test session
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        if reporter.stats.get("warnings", None):
            session.exitstatus = 2


@pytest.hookimpl
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    have_werrors = os.getenv("NIFREEZE_WERRORS", False)
    have_werrors = config.getoption("--warnings-as-errors", False) or have_werrors
    have_warnings = terminalreporter.stats.get("warnings", None)
    if have_warnings and have_werrors:
        terminalreporter.ensure_newline()
        terminalreporter.section("Werrors", sep="=", red=True, bold=True)
        terminalreporter.line(
            "Warnings as errors: Activated.\n"
            f"{len(have_warnings)} warnings were raised and treated as errors.\n"
        )


@pytest.fixture(autouse=True)
def random_number_generator(request):
    """Automatically set a fixed-seed random number generator for all tests."""
    request.node.rng = np.random.default_rng(1234)


def _generate_random_uniform_nd_data(request, size, a=0.0, b=1.0):
    rng = request.node.rng
    data = rng.random(size=size).astype(np.float32)
    return (b - a) * data + a


def _generate_random_uniform_spatial_data(request, size, a, b):
    data = _generate_random_uniform_nd_data(request, size, a, b)
    affine = np.eye(4, dtype="float32")
    return data, affine


@pytest.fixture(autouse=True)
def setup_random_uniform_spatial_data(request):
    """Automatically generate random spatial data for tests."""
    marker = request.node.get_closest_marker("random_uniform_spatial_data")

    size = (32, 32, 32, 5)
    a = 0.0
    b = 1.0
    if marker:
        size, a, b = marker.args

    return _generate_random_uniform_spatial_data(request, size, a, b)


@pytest.fixture(autouse=True)
def setup_random_uniform_ndim_data(request):
    """Automatically generate random data for tests."""
    marker = request.node.get_closest_marker("random_uniform_ndim_data")

    size = (32, 32, 32, 5)
    a = 0.0
    b = 1.0
    if marker:
        size, a, b = marker.args

    return _generate_random_uniform_nd_data(request, size, a, b)


def _generate_random_choices(rng, values, count):
    values = set(values)

    num_elements = len(values)

    if count < num_elements:
        raise ValueError(
            f"Count must be at least the number of unique values to guarantee inclusion\nProvided: {count} and {values}."
        )

    # Start by assigning one of each value
    selected_values = list(values)

    # Distribute remaining count: randomly distribute N among the values
    remaining = count - num_elements
    partitions = rng.multinomial(remaining, np.ones(num_elements) / num_elements)

    # Add the remaining values according to the partitions
    for val, extra_count in zip(values, partitions, strict=True):
        selected_values.extend([val] * extra_count)

    return sorted(selected_values)


def _generate_random_bvals(rng, b0s, shells, n_gradients):
    # Generate a random number of elements for each shell
    bvals_shells = _generate_random_choices(rng, shells, n_gradients)

    bvals = np.hstack([b0s * [0], bvals_shells])

    return bvals


def _generate_random_bvecs(rng, b0s, n_gradients):
    return np.hstack([np.zeros((3, b0s)), normalized_vector(rng.random((3, n_gradients)), axis=0)])


@pytest.fixture(autouse=True)
def setup_random_bval_data(request):
    """Automatically generate random b-val data for tests."""
    marker = request.node.get_closest_marker("random_bval_data")

    n_gradients = 10
    shells = (1000, 2000, 3000)
    b0s = 1
    if marker:
        n_gradients, shells, b0s = marker.args

    rng = request.node.rng
    return _generate_random_bvals(rng, b0s, shells, n_gradients)


@pytest.fixture
def setup_random_bvec_data(request, bvals, bval_tolerance):
    """Automatically generate random b-vec data for tests."""
    rng = request.node.rng

    is_b0 = np.abs(bvals) <= bval_tolerance
    b0s = np.sum(is_b0)
    n_gradients = np.sum(~is_b0)

    return _generate_random_bvecs(rng, b0s, n_gradients)


@pytest.fixture(autouse=True)
def setup_random_gtab_data(request):
    """Automatically generate random gtab data for tests."""
    marker = request.node.get_closest_marker("random_gtab_data")

    n_gradients = 10
    shells = (1000, 2000, 3000)
    b0s = 1
    if marker:
        n_gradients, shells, b0s = marker.args

    rng = request.node.rng

    bvals = _generate_random_bvals(rng, b0s, shells, n_gradients)
    bvecs = _generate_random_bvecs(rng, b0s, n_gradients)

    return bvals, bvecs


@pytest.fixture(autouse=True)
def setup_random_base_data(request):
    """Automatically generate random BaseDataset data for tests."""
    marker = request.node.get_closest_marker("random_base_data")

    vol_size = (4, 4, 4)
    volumes = 5
    if marker:
        vol_size, volumes = marker.args

    rng = request.node.rng

    base_dataobj, affine = _generate_random_uniform_spatial_data(
        request, (*vol_size, volumes), 0.0, 1.0
    )
    brainmask_dataobj = rng.choice([True, False], size=vol_size).astype(bool)
    motion_affines = rng.random((volumes, 4, 4))
    datahdr = None

    return (
        base_dataobj,
        affine,
        brainmask_dataobj,
        motion_affines,
        datahdr,
    )


@pytest.fixture(autouse=True)
def setup_random_dwi_data(request, setup_random_gtab_data):
    """Automatically generate random DWI data for tests."""
    marker = request.node.get_closest_marker("random_dwi_data")

    b0_thres = 50
    vol_size = (2, 2, 2)
    use_random_gtab = True
    if marker:
        b0_thres, vol_size, use_random_gtab = marker.args

    rng = request.node.rng

    if use_random_gtab:
        bvals, bvecs = setup_random_gtab_data
    else:
        bvals, bvecs = read_bvals_bvecs(
            str(_datadir / "hcph_multishell.bval"),
            str(_datadir / "hcph_multishell.bvec"),
        )
        bvecs = bvecs.T

    n_gradients = np.count_nonzero(bvals)
    b0s = len(bvals) - n_gradients
    volumes = n_gradients + b0s

    dwi_dataobj, affine = _generate_random_uniform_spatial_data(
        request, (*vol_size, volumes), 0.0, 1.0
    )
    brainmask_dataobj = rng.choice([True, False], size=vol_size).astype(bool)
    b0_dataobj = rng.random(vol_size, dtype="float32")
    gradients = np.vstack([bvecs, bvals[np.newaxis, :]], dtype="float32")

    return (
        dwi_dataobj,
        affine,
        brainmask_dataobj,
        b0_dataobj,
        gradients,
        b0_thres,
    )


@pytest.fixture(autouse=True)
def setup_random_pet_data(request):
    """Automatically generate random PET data for tests."""
    marker = request.node.get_closest_marker("random_pet_data")

    n_frames = 5
    vol_size = (4, 4, 4)
    midframe = np.arange(n_frames, dtype=np.float32) + 1
    total_duration = float(n_frames + 1)
    if marker:
        n_frames, vol_size, midframe, total_duration = marker.args

    rng = request.node.rng

    pet_dataobj, affine = _generate_random_uniform_spatial_data(
        request, (*vol_size, n_frames), 0.0, 1.0
    )
    brainmask_dataobj = rng.choice([True, False], size=vol_size).astype(bool)

    return (
        pet_dataobj,
        affine,
        brainmask_dataobj,
        midframe,
        total_duration,
    )
