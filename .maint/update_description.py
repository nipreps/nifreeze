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

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "click",
#     "toml",
# ]
# ///
"""Update the description of the zenodo record using the pyproject description."""

import json
import re
from pathlib import Path

import click
import toml


@click.group()
def cli():
    """Generate project description boilerplate."""
    pass


@cli.command()
@click.option("-z", "--zenodo-file", type=click.Path(exists=True), default=".zenodo.json")
@click.option("-p", "--pyproject-file", type=click.Path(exists=True), default="pyproject.toml")
def zenodo(
    zenodo_file,
    pyproject_file,
):
    """Generate a new Zenodo payload file."""

    zenodo = json.loads(Path(zenodo_file).read_text())

    pyproject = toml.load(pyproject_file)
    new_desc = pyproject.get("project", {}).get("description", "")

    original_desc = zenodo.get("description", "")

    # Extract existing opening and closing HTML tags (e.g., <p>...</p>)
    match = re.match(r"^(<[^>]+>)(.*?)(</[^>]+>)$", original_desc, re.DOTALL)
    if match:
        opening_tag, _, closing_tag = match.groups()
        zenodo["description"] = f"{opening_tag}{new_desc}{closing_tag}"
    else:
        # If no tags detected, wrap the new description in <p> tags
        zenodo["description"] = f"<p>{new_desc}</p>"

    Path(zenodo_file).write_text("%s\n" % json.dumps(zenodo, indent=2))


if __name__ == "__main__":
    """ Install entry-point """
    cli()
