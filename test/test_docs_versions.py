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
"""Tests of the docs deploy (``tools/docs_versions.py``) on small site trees."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, "tools")
import docs_versions  # noqa: E402

NEW_SWITCHER = "// the new menu\n"


def make_build(path: Path, version: str, page: str = "index.html") -> Path:
    """Write a minimal Sphinx HTML build of ``version`` at ``path``."""
    (path / "_static" / "js").mkdir(parents=True)
    (path / "_static" / "documentation_options.js").write_text(
        f"const DOCUMENTATION_OPTIONS = {{\n    VERSION: '{version}',\n    LANGUAGE: 'en',\n}};\n"
    )
    (path / "_static" / "js" / "version-switch.js").write_text("// the old menu\n")
    (path / page).write_text(f"<p>{version}</p>\n")
    return path


@pytest.fixture
def switcher(tmp_path: Path) -> Path:
    path = tmp_path / "version-switch.js"
    path.write_text(NEW_SWITCHER)
    return path


@pytest.fixture
def site(tmp_path: Path) -> Path:
    """Today's site: main, 0.2 and 0.1, a benchmarks folder and eddymotion's leftover."""
    site = tmp_path / "site"
    make_build(site / "main", "26.0.0.dev165")
    make_build(site / "0.2", "0.2.0")
    make_build(site / "0.1", "0.1.0")
    (site / "benchmarks").mkdir()
    (site / "benchmarks" / "index.html").write_text("<p>asv</p>\n")
    (site / "js").mkdir()
    (site / "js" / "rtd-versions.js").write_text("// unused\n")
    (site / "versions.json").write_text('{"tags": ["0.1", "0.2"], "heads": ["main"]}')
    return site


def test_series():
    """Tags go to the folder of their major and minor version."""
    assert docs_versions.series("26.0.0") == "26.0"
    assert docs_versions.series("26.0.0rc1") == "26.0"
    assert docs_versions.series("26.10.3") == "26.10"
    assert docs_versions.series("0.2.0") == "0.2"


def test_catalog_orders_development_then_newest(site: Path, tmp_path: Path):
    """main comes first, then the series by version, and only a final release is latest."""
    docs_versions.place(site, make_build(tmp_path / "rc", "26.0.0rc1"), tag="26.0.0rc1")
    info = docs_versions.catalog(site)
    assert info["latest"] == "0.2"
    assert [(e["path"], e["version"], e["kind"]) for e in info["versions"]] == [
        ("main", "26.0.0.dev165", "dev"),
        ("26.0", "26.0.0rc1", "prerelease"),
        ("0.2", "0.2.0", "release"),
        ("0.1", "0.1.0", "release"),
    ]


def test_catalog_sorts_by_version_not_as_strings(tmp_path: Path):
    """26.10 is newer than 26.9, although it sorts before it as a string."""
    site = tmp_path / "site"
    make_build(site / "26.9", "26.9.0")
    make_build(site / "26.10", "26.10.0")
    info = docs_versions.catalog(site)
    assert info["latest"] == "26.10"
    assert [e["path"] for e in info["versions"]] == ["26.10", "26.9"]


def test_place_main_and_tags(site: Path, tmp_path: Path):
    """Without a tag the build replaces main; a tag goes to its series' folder."""
    assert docs_versions.place(site, make_build(tmp_path / "dev", "26.0.0.dev170")) == "main"
    assert docs_versions.build_version(site / "main") == "26.0.0.dev170"

    rc = make_build(tmp_path / "rc", "26.0.0rc1")
    assert docs_versions.place(site, rc, tag="26.0.0rc1") == "26.0"
    final = make_build(tmp_path / "final", "26.0.0")
    assert docs_versions.place(site, final, tag="26.0.0") == "26.0"
    assert docs_versions.build_version(site / "26.0") == "26.0.0"
    assert docs_versions.catalog(site)["latest"] == "26.0"


def test_place_never_replaces_a_final_release_with_a_prerelease(site: Path, tmp_path: Path):
    """A patch's pre-release, or an older build, leaves the series' release in place."""
    docs_versions.place(site, make_build(tmp_path / "a", "26.0.0"), tag="26.0.0")
    assert (
        docs_versions.place(site, make_build(tmp_path / "b", "26.0.1rc1"), tag="26.0.1rc1") is None
    )
    assert (
        docs_versions.place(site, make_build(tmp_path / "c", "0.2.0rc1"), tag="0.2.0rc1") is None
    )
    assert docs_versions.build_version(site / "26.0") == "26.0.0"
    assert docs_versions.build_version(site / "0.2") == "0.2.0"


def test_place_refuses_a_build_of_another_version(site: Path, tmp_path: Path):
    """The tag and the build's version must agree."""
    with pytest.raises(ValueError, match="not of tag"):
        docs_versions.place(site, make_build(tmp_path / "x", "26.0.0.dev170"), tag="26.0.0rc1")


def test_index_writes_versions_root_and_404(site: Path, switcher: Path):
    """The index lists the folders, the root goes to the latest release, and 404 knows both."""
    docs_versions.index(site, "/nifreeze/", switcher)

    info = json.loads((site / "versions.json").read_text())
    assert info["latest"] == "0.2"
    assert [e["path"] for e in info["versions"]] == ["main", "0.2", "0.1"]

    root = (site / "index.html").read_text()
    assert 'url=0.2/"' in root and 'replace("0.2/"' in root

    not_found = (site / "404.html").read_text()
    assert 'var base = "/nifreeze/";' in not_found
    assert 'var folders = ["main", "0.2", "0.1"];' in not_found
    # The old page redirected to the relative index.html, which loops below the top level
    assert "URL=index.html" not in not_found
    assert (site / ".nojekyll").exists()


def test_index_without_releases_goes_to_main(tmp_path: Path, switcher: Path):
    """Before the first release, the root goes to the development version."""
    site = tmp_path / "site"
    make_build(site / "main", "26.0.0.dev1")
    info = docs_versions.index(site, "/nifreeze/", switcher)
    assert info["latest"] is None
    assert 'url=main/"' in (site / "index.html").read_text()


def test_index_copies_the_menu_into_every_version(site: Path, switcher: Path):
    """Every version gets the new menu script; nothing else in them changes."""
    before = {
        p: p.read_bytes() for p in site.rglob("*") if p.is_file() and p.name != "version-switch.js"
    }
    docs_versions.index(site, "/nifreeze/", switcher)

    for folder in ("main", "0.2", "0.1"):
        assert (site / folder / "_static/js/version-switch.js").read_text() == NEW_SWITCHER
    for path, content in before.items():
        if path.parent == site or not path.exists():
            continue  # the index files are rewritten, and eddymotion's leftover removed
        assert path.read_bytes() == content, path


def test_index_removes_eddymotions_menu(site: Path, switcher: Path):
    """``js/rtd-versions.js`` goes, with the folder it leaves empty."""
    docs_versions.index(site, "/nifreeze/", switcher)
    assert not (site / "js").exists()
    assert (site / "benchmarks" / "index.html").exists()


def test_main_deploys_and_indexes(site: Path, tmp_path: Path, switcher: Path):
    """The command line places a tag's build, then rebuilds the index."""
    build = make_build(tmp_path / "rc", "26.0.0rc1")
    docs_versions.main(
        [
            "deploy",
            "--site",
            str(site),
            "--build",
            str(build),
            "--tag",
            "26.0.0rc1",
            "--switcher",
            str(switcher),
        ]
    )
    info = json.loads((site / "versions.json").read_text())
    assert [e["path"] for e in info["versions"]] == ["main", "26.0", "0.2", "0.1"]
    assert info["latest"] == "0.2"
    assert (site / "26.0" / "_static/js/version-switch.js").read_text() == NEW_SWITCHER
