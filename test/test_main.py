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

import importlib.util
import runpy
import sys
import types

import pytest

from nifreeze.__main__ import main


def _make_dummy_run_module(call_recorder: dict):
    """Create a fake nifreeze.cli.run module with a main() that records it was
    called.
    """
    dummy_cli_run = types.ModuleType("nifreeze.cli.run")

    def _main():
        # Record that main was invoked and with which argv
        call_recorder["called"] = True
        call_recorder["argv"] = list(sys.argv)

    # Use setattr to avoid a static attribute access that mypy flags on
    # ModuleType
    setattr(dummy_cli_run, "main", _main)  # noqa
    return dummy_cli_run


def _make_dummy_package():
    """
    Create a package-like module object for 'nifreeze' so package-relative imports
    inside nifreeze.__main__ resolve to injected sys.modules entries like
    'nifreeze.cli.run'. Use spec_from_loader(loader=None) so the spec has a
    loader argument and linters/runtime checks are satisfied.
    """
    spec = importlib.util.spec_from_loader("nifreeze", loader=None, is_package=True)
    pkg = types.ModuleType("nifreeze")
    pkg.__spec__ = spec
    pkg.__path__ = []  # mark as a package
    return pkg


@pytest.fixture(autouse=True)
def set_command(monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(sys, "argv", ["nifreeze"])
        yield


def test_help(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    captured = capsys.readouterr()
    assert captured.out.startswith("usage: nifreeze [-h]")


@pytest.mark.parametrize(
    "initial_argv0, expect_rewrite",
    [
        ("something/path/__main__.py", True),
        (f"{sys.executable}", False),
    ],
)
def test_nifreeze_call(monkeypatch, initial_argv0, expect_rewrite):
    """Execute the package's __main__ and assert that:
    - nifreeze.cli.run.main() is called
    - sys.argv[0] gets rewritten only when '__main__.py' is in argv[0]
    """
    orig_modules = sys.modules.copy()

    recorder = {"called": False, "argv": None}

    # Remove any pre-existing nifreeze-related modules to avoid runpy's warning:
    # runpy warns when it finds nifreeze.__main__ in sys.modules after the package
    # is imported but before executing __main__. Removing such entries ensures
    # importlib/runpy will load and execute the package/__main__ without the
    # spurious warning and without accidentally reusing a stale module object.
    for key in list(sys.modules.keys()):
        if key == "nifreeze" or key.startswith("nifreeze."):
            # Pop and drop: we'll restore full original snapshot afterwards
            sys.modules.pop(key, None)

    # Insert a dummy run module so "from .cli.run import main" resolves to our
    # dummy
    sys.modules["nifreeze.cli.run"] = _make_dummy_run_module(recorder)

    # Install the dummy run module (monkeypatch target)
    sys.modules["nifreeze.cli.run"] = _make_dummy_run_module(recorder)

    # Set argv[0] to the desired test value
    sys_argv_backup = list(sys.argv)
    sys.argv[0:1] = [initial_argv0]

    try:
        # Execute nifreeze.__main__ as a script (so its if __name__ == "__main__" block runs)
        runpy.run_module("nifreeze.__main__", run_name="__main__")
    finally:
        # Restore sys.argv and sys.modules to avoid side effects on other tests
        sys.argv[:] = sys_argv_backup
        # Restore modules: remove keys we injected and put back original modules
        # First clear any modules added during run_module
        for key in list(sys.modules.keys()):
            if key not in orig_modules:
                del sys.modules[key]
        # Put back original modules mapping
        sys.modules.update(orig_modules)

    # Assert main() was called
    assert recorder["called"] is True

    # Tell the type checker (and document the runtime expectation) that argv is
    # a list
    assert isinstance(recorder["argv"], list)

    if expect_rewrite:
        expected = f"{sys.executable} -m nifreeze"
        # recorder["argv"] captured sys.argv as seen by dummy main()
        assert recorder["argv"][0] == expected
    else:
        # argv[0] should not have been rewritten
        assert recorder["argv"][0] == initial_argv0
