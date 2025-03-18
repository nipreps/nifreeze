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
    masknii = nb.Nifti1Image(dwdata.brainmask.astype("uint8"), dwdata.affine, None)

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
    moved_nii = (~xfms).apply(b0nii, reference=b0nii)

    # Save the moved dataset for debugging or further processing
    moved_path = tmp_path / "test.nii.gz"
    ground_truth_path = tmp_path / "ground_truth.nii.gz"
    moved_nii.to_filename(moved_path)
    xfms.apply(moved_nii).to_filename(ground_truth_path)

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
