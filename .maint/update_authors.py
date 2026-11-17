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
# requires-python = ">=3.13"
# dependencies = [
#     "click",
#     "fuzzywuzzy",
#     "python-Levenshtein",
#     "pyyaml",
# ]
# ///
"""Sort the authors of the Zenodo record, ``CITATION.cff`` and the documentation.

Authors are ranked by the lines they wrote that the repository still holds (``git summary
--line``, from git-extras), with the PIs last in papers. Run from the repository's root::

    uv run .maint/update_authors.py zenodo
    uv run .maint/update_authors.py publication --release 26.0.0rc1 --docs docs/citation.rst

"""

import json
import shutil
import subprocess as sp
import sys
from pathlib import Path

import click
import yaml
from fuzzywuzzy import fuzz, process

CFF_BEGIN = "# BEGIN AUTHORS"
CFF_END = "# END AUTHORS"
DOCS_BEGIN = ".. BEGIN AUTHORS"
DOCS_END = ".. END AUTHORS"


def read_md_table(md_text):
    """
    Extract the first table found in a markdown document as a Python dict.

    Examples
    --------
    >>> read_md_table('''
    ... # Some text
    ...
    ... More text
    ...
    ... | **Header1** | **Header2** |
    ... | --- | --- |
    ... | val1 | val2 |
    ... |  | val4 |
    ...
    ... | **Header3** | **Header4** |
    ... | --- | --- |
    ... | val1 | val2 |
    ... |  | val4 |
    ... ''')
    [{'header1': 'val1', 'header2': 'val2'}, {'header2': 'val4'}]

    """
    prev = None
    keys = None
    retval = []
    for line in md_text.splitlines():
        if line.strip().startswith("| --- |"):
            keys = (k.replace("*", "").strip() for k in prev.split("|"))
            keys = [k.lower() for k in keys if k]
            continue
        elif not keys:
            prev = line
            continue

        if not line or not line.strip().startswith("|"):
            break

        values = [v.strip() or None for v in line.split("|")][1:-1]
        retval.append({k: v for k, v in zip(keys, values, strict=False) if v})

    return retval


def sort_contributors(entries, git_lines, exclude=None):
    """Return a list of author dictionaries, ordered by contribution."""
    sorted_authors = sorted(entries, key=lambda i: i["name"])

    first_last = [" ".join(val["name"].split(",")[::-1]).strip() for val in sorted_authors]
    first_last_excl = [" ".join(val["name"].split(",")[::-1]).strip() for val in exclude or []]

    unmatched = []
    author_matches = []
    for ele in git_lines:
        matches = process.extract(ele, first_last, scorer=fuzz.token_sort_ratio, limit=2)
        # matches is a list [('First match', % Match), ('Second match', % Match)]
        if matches[0][1] > 80:
            val = sorted_authors[first_last.index(matches[0][0])]
        else:
            # skip unmatched names
            if ele not in first_last_excl:
                unmatched.append(ele)
            continue

        if val not in author_matches:
            author_matches.append(val)

    names = {" ".join(val["name"].split(",")[::-1]).strip() for val in author_matches}
    for missing_name in first_last:
        if missing_name not in names:
            missing = sorted_authors[first_last.index(missing_name)]
            author_matches.append(missing)

    position_matches = []
    for i, item in enumerate(author_matches):
        pos = item.pop("position", None)
        if pos is not None:
            position_matches.append((i, int(pos)))

    for i, pos in position_matches:
        if pos < 0:
            pos += len(author_matches) + 1
        author_matches.insert(pos, author_matches.pop(i))

    return author_matches, unmatched


def get_git_lines():
    """Authors by the lines they wrote that the repository still holds, most first."""
    if not shutil.which("git-summary"):
        raise RuntimeError("git summary not found: please install git-extras.")

    print("Running git summary --line on the repository", file=sys.stderr)
    lines = sp.check_output(["git", "summary", "--line"]).decode().splitlines()
    return [
        " ".join(line.strip().split()[1:-1])
        for line in lines
        if "%" in line and "Not Committed Yet" not in line
    ]


def _replace_block(path, begin, end, block):
    """Replace the lines between the ``begin`` and ``end`` marker lines of a file."""
    lines = Path(path).read_text().splitlines(keepends=True)
    first = [i for i, line in enumerate(lines) if line.startswith(begin)]
    last = [i for i, line in enumerate(lines) if line.startswith(end)]
    if len(first) != 1 or len(last) != 1 or first[0] > last[0]:
        raise RuntimeError(f'{path} must contain one "{begin}" line followed by one "{end}" line.')
    Path(path).write_text("".join(lines[: first[0] + 1]) + block + "".join(lines[last[0] :]))


def _cff_names(*md_files):
    """Map each full name to its (given, family) pair, from the ``.maint`` tables."""
    return {
        f"{row.get('name', '')} {row.get('lastname', '')}".strip(): (
            row.get("name"),
            row.get("lastname"),
        )
        for md_file in md_files
        for row in read_md_table(Path(md_file).read_text())
    }


def _cff_person(entry, names):
    """Convert a Zenodo creator into a CFF person."""
    given, family = names.get(entry["name"], (None, None))
    person = {}
    if family:
        person["family-names"] = family
    if given:
        person["given-names"] = given
    if not person:
        person["name"] = entry["name"]
    if entry.get("orcid"):
        person["orcid"] = f"https://orcid.org/{entry['orcid']}"
    if entry.get("affiliation") and entry["affiliation"] != "Unknown affiliation":
        person["affiliation"] = entry["affiliation"]
    return person


def _namelast(inlist):
    retval = []
    for i in inlist:
        i["name"] = (f"{i.pop('name', '')} {i.pop('lastname', '')}").strip()
        if not i["name"]:
            i["name"] = i.get("handle", "<Unknown Name>")
        retval.append(i)
    return retval


def _report(misses):
    if misses:
        print(
            "Some people made commits, but are missing in .maint/ "
            f"files: {', '.join(sorted(misses))}",
            file=sys.stderr,
        )


@click.group()
def cli():
    """Generate authorship boilerplates."""
    pass


@cli.command()
@click.option("-z", "--zenodo-file", type=click.Path(exists=True), default=".zenodo.json")
@click.option("-m", "--maintainers", type=click.Path(exists=True), default=".maint/MAINTAINERS.md")
@click.option(
    "-c", "--contributors", type=click.Path(exists=True), default=".maint/CONTRIBUTORS.md"
)
@click.option("--pi", type=click.Path(exists=True), default=".maint/PIs.md")
@click.option("-f", "--former-file", type=click.Path(exists=True), default=".maint/FORMER.md")
@click.option("--cff-file", type=click.Path(exists=True), default="CITATION.cff")
def zenodo(
    zenodo_file,
    maintainers,
    contributors,
    pi,
    former_file,
    cff_file,
):
    """Generate a new Zenodo payload file and the matching CITATION.cff authors."""
    data = get_git_lines()

    zenodo = json.loads(Path(zenodo_file).read_text())

    former = _namelast(read_md_table(Path(former_file).read_text()))
    zen_creators, miss_creators = sort_contributors(
        _namelast(read_md_table(Path(maintainers).read_text())),
        data,
        exclude=former,
    )

    zen_contributors, miss_contributors = sort_contributors(
        _namelast(read_md_table(Path(contributors).read_text())),
        data,
        exclude=former,
    )

    zen_pi = _namelast(reversed(read_md_table(Path(pi).read_text())))

    # PIs who are also contributors keep their place among them
    contributor_names = {c["name"] for c in zen_contributors}
    zenodo["creators"] = zen_creators
    zenodo["contributors"] = zen_contributors + [
        p for p in zen_pi if p["name"] not in contributor_names
    ]
    creator_names = {c["name"] for c in zenodo["creators"] if c["name"] != "<Unknown Name>"}

    zenodo["contributors"] = [c for c in zenodo["contributors"] if c["name"] not in creator_names]

    _report(set(miss_creators).intersection(miss_contributors))

    # Remove position
    for creator in zenodo["creators"]:
        creator.pop("position", None)
        creator.pop("handle", None)
        if "affiliation" not in creator:
            creator["affiliation"] = "Unknown affiliation"
        elif isinstance(creator["affiliation"], list):
            creator["affiliation"] = creator["affiliation"][0]

    for creator in zenodo["contributors"]:
        creator.pop("handle", None)
        creator["type"] = "Researcher"
        creator.pop("position", None)

        if "affiliation" not in creator:
            creator["affiliation"] = "Unknown affiliation"
        elif isinstance(creator["affiliation"], list):
            creator["affiliation"] = creator["affiliation"][0]

    Path(zenodo_file).write_text(f"{json.dumps(zenodo, indent=2, ensure_ascii=False)}\n")

    # CITATION.cff has no contributor role: its authors mirror the Zenodo creators
    names = _cff_names(maintainers, contributors, pi)
    authors = [_cff_person(creator, names) for creator in zenodo["creators"]]
    block = yaml.safe_dump({"authors": authors}, sort_keys=False, allow_unicode=True, width=10**6)
    _replace_block(cff_file, CFF_BEGIN, CFF_END, block)


@cli.command()
@click.option("-m", "--maintainers", type=click.Path(exists=True), default=".maint/MAINTAINERS.md")
@click.option(
    "-c", "--contributors", type=click.Path(exists=True), default=".maint/CONTRIBUTORS.md"
)
@click.option("--pi", type=click.Path(exists=True), default=".maint/PIs.md")
@click.option("-f", "--former-file", type=click.Path(exists=True), default=".maint/FORMER.md")
@click.option("--release", help="The version the list is for (with --docs).")
@click.option(
    "--docs",
    "docs_file",
    type=click.Path(exists=True),
    help="Write the list into this page, between its BEGIN and END AUTHORS comments.",
)
def publication(
    maintainers,
    contributors,
    pi,
    former_file,
    release,
    docs_file,
):
    """Generate the list of authors and affiliations for papers."""
    if docs_file and not release:
        raise click.UsageError("--docs needs --release.")

    members = _namelast(read_md_table(Path(maintainers).read_text())) + _namelast(
        read_md_table(Path(contributors).read_text())
    )
    former_names = _namelast(read_md_table(Path(former_file).read_text()))

    hits, misses = sort_contributors(
        members,
        get_git_lines(),
        exclude=former_names,
    )

    pi_hits = _namelast(reversed(read_md_table(Path(pi).read_text())))
    pi_names = [pi["name"] for pi in pi_hits]
    hits = [hit for hit in hits if hit["name"] not in pi_names] + pi_hits

    def _aslist(value):
        if isinstance(value, (list, tuple)):
            return value
        return [value]

    # Remove position
    affiliations = []
    for item in hits:
        item.pop("position", None)
        for a in _aslist(item.get("affiliation", "Unaffiliated")):
            if a not in affiliations:
                affiliations.append(a)

    aff_indexes = [
        ", ".join(
            [
                "%d" % (affiliations.index(a) + 1)
                for a in _aslist(author.get("affiliation", "Unaffiliated"))
            ]
        )
        for author in hits
    ]

    _report(misses)

    authors = "; ".join(
        f"{i['name']} \\ :sup:`{idx}`\\ " for i, idx in zip(hits, aff_indexes, strict=True)
    )
    numbered = "\n".join(f"{i + 1}. {a}" for i, a in enumerate(affiliations))
    if not docs_file:
        print(f"Authors ({len(hits)}):\n{authors}.\n\n\nAffiliations:\n{numbered}")
        return

    _replace_block(
        docs_file,
        DOCS_BEGIN,
        DOCS_END,
        f"\nThe default author list for *NiFreeze* {release}:\n\n"
        f"{authors}.\n\nAffiliations:\n\n{numbered}\n\n",
    )


if __name__ == "__main__":
    """Install entry-point"""
    cli()
