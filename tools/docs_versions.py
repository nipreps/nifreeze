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
"""Place a documentation build into the published site, and index its versions.

The site (the ``gh-pages`` branch) holds one folder per version: ``main`` for the
development version, and ``X.Y`` for each release series, which holds the series' latest
release or, before it, its latest pre-release. Run as::

    python tools/docs_versions.py deploy --site <gh-pages> --build <html> [--tag <tag>]

Without ``--tag``, the build goes to ``main``. The site's index is then rebuilt from the
folders it holds, whatever deployed them:

- ``versions.json``, which the version menu (``docs/_static/js/version-switch.js``) reads;
- ``index.html``, which sends visitors to the latest release (``main`` before the first);
- ``404.html``, which sends a missing page to the index of its version, or to the site's;
- every version's copy of ``version-switch.js``, so that older builds get the same menu.

"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

from packaging.version import Version

DEV_FOLDER = "main"
OPTIONS = Path("_static") / "documentation_options.js"
SWITCHER = Path("_static") / "js" / "version-switch.js"
DEFAULT_SWITCHER = Path(__file__).parents[1] / "docs" / SWITCHER
# Left by eddymotion's version menu, which nothing loads
STALE = [Path("js") / "rtd-versions.js"]


def build_version(folder: Path) -> str | None:
    """Read the version a Sphinx build was made for, or ``None`` if it is not one."""
    options = folder / OPTIONS
    if not options.is_file():
        return None
    match = re.search(r"VERSION:\s*['\"]([^'\"]+)['\"]", options.read_text())
    return match.group(1) if match else None


def series(tag: str) -> str:
    """The folder of a release series: ``26.0`` for ``26.0.1`` and ``26.0.0rc1``."""
    release = Version(tag).release
    return f"{release[0]}.{release[1] if len(release) > 1 else 0}"


def kind(path: str, version: str) -> str:
    """``dev`` for the development folder, ``prerelease`` or ``release`` for a series."""
    if path == DEV_FOLDER:
        return "dev"
    return "prerelease" if Version(version).is_prerelease else "release"


def catalog(site: Path) -> dict:
    """List the versions the site holds: development first, then newest to oldest."""
    entries = []
    for folder in sorted(p for p in site.iterdir() if p.is_dir()):
        version = build_version(folder)
        if version is None:
            continue
        entries.append(
            {
                "name": folder.name,
                "version": version,
                "path": folder.name,
                "kind": kind(folder.name, version),
            }
        )

    dev = [e for e in entries if e["kind"] == "dev"]
    series_ = sorted(
        (e for e in entries if e["kind"] != "dev"),
        key=lambda e: Version(e["version"]),
        reverse=True,
    )
    releases = [e for e in series_ if e["kind"] == "release"]
    return {
        "latest": releases[0]["path"] if releases else None,
        "versions": dev + series_,
    }


def place(site: Path, build: Path, tag: str | None = None) -> str | None:
    """Copy a build into its folder; return the folder, or ``None`` if it was skipped.

    A pre-release never replaces a final release, and no build replaces a newer one.
    """
    version = build_version(build)
    if version is None:
        raise ValueError(f"{build} is not a Sphinx HTML build")
    if tag is None:
        folder = DEV_FOLDER
    else:
        if Version(tag) != Version(version):
            raise ValueError(f"the build is of version {version}, not of tag {tag}")
        folder = series(tag)
        published = build_version(site / folder)
        if published is not None:
            old, new = Version(published), Version(version)
            if (not old.is_prerelease and new.is_prerelease) or new < old:
                print(f"{folder}/ holds {published}, which {version} does not replace")
                return None

    target = site / folder
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(build, target, symlinks=True)
    return folder


def root_page(target: str) -> str:
    """The site's ``index.html``: an immediate redirect to ``target``."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>NiFreeze documentation</title>
  <meta http-equiv="refresh" content="0; url={target}/">
  <link rel="canonical" href="{target}/">
  <script>window.location.replace("{target}/" + window.location.hash);</script>
</head>
<body>
  <p>The documentation of <em>NiFreeze</em> is at <a href="{target}/">{target}/</a>.</p>
</body>
</html>
"""


def not_found_page(base_url: str, folders: list[str]) -> str:
    """The site's ``404.html``: to the index of the missing page's version, or the site's."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Page not found - NiFreeze documentation</title>
  <script>
    (function () {{
      var base = {json.dumps(base_url)};
      var folders = {json.dumps(folders)};
      var path = window.location.pathname;
      var target = base;
      if (path.indexOf(base) === 0) {{
        var first = decodeURIComponent(path.slice(base.length).split("/")[0]);
        if (folders.indexOf(first) !== -1) target = base + first + "/";
      }}
      // Never send a missing index back to itself
      if (target === path || target + "index.html" === path) target = base;
      window.location.replace(target);
    }})();
  </script>
</head>
<body>
  <p>This page does not exist.
  Go to the <a href="{base_url}">documentation of <em>NiFreeze</em></a>.</p>
</body>
</html>
"""


def index(site: Path, base_url: str, switcher: Path = DEFAULT_SWITCHER) -> dict:
    """Rebuild ``versions.json``, the root and 404 pages, and every version's menu script."""
    info = catalog(site)
    (site / "versions.json").write_text(json.dumps(info, indent=4) + "\n")
    (site / "index.html").write_text(root_page(info["latest"] or DEV_FOLDER))
    folders = [entry["path"] for entry in info["versions"]]
    (site / "404.html").write_text(not_found_page(base_url, folders))
    (site / ".nojekyll").touch()

    for folder in folders:
        copy = site / folder / SWITCHER
        if copy.is_file():
            shutil.copyfile(switcher, copy)

    for stale in STALE:
        path = site / stale
        if path.exists():
            path.unlink()
            if not any(path.parent.iterdir()):
                path.parent.rmdir()
    return info


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    deploy = sub.add_parser("deploy", help="place a build, then rebuild the site's index")
    deploy.add_argument("--site", type=Path, required=True, help="the site (gh-pages) folder")
    deploy.add_argument("--build", type=Path, help="a Sphinx HTML build to place")
    deploy.add_argument("--tag", help="the release tag of the build (default: main)")
    deploy.add_argument("--base-url", default="/nifreeze/", help="the site's URL path")
    deploy.add_argument("--switcher", type=Path, default=DEFAULT_SWITCHER)
    args = parser.parse_args(argv)

    if args.build is not None:
        folder = place(args.site, args.build, args.tag)
        if folder is not None:
            print(f"Placed {args.build} into {folder}/")
    info = index(args.site, args.base_url, args.switcher)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
