<!--
Copyright The NiPreps Developers <nipreps@gmail.com>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

We support and encourage derived works from this project, please read
about our expectations at

    https://www.nipreps.org/community/licensing/
-->

# AGENTS instructions

The project's source code lives under `src/nifreeze/` and tests under `tests/`.

## Testing

### Pre-requisites

- Some software needs to be installed prior to testing, for example ANTs
  ```
  conda install -c conda-forge ants=2.5 libiconv
  ```
- Notebooks generate figures with latex commands inside, therefore:
  ```
  sudo apt install texlive texlive-latex-extra texlive-fonts-recommended cm-super dvipng
  ```

Details about testing are found in `.github/workflows/test.yml`

### Unit tests

- Unit tests can be executed with pytest: `pytest tests/`.
- The project includes doctests, which can be run with `pytest --doctest-module src/nifreeze`

### Integration tests and benchmarks

- The full battery of tests can be run through tox (`tox -v`)
- Install tox using `uv tool install --with=tox-uv --with=tox-gh-actions tox`

## Documentation building

Documentation can be built as described in `.github/workflows/docs-build-pr.yml`.

## Linting

Before accepting new PRs, we use the latest version of Ruff to lint the code, as in `.github/workflows/contrib.yml`

## Codex instructions

- Always plan first
- Think harder in the planning phase
- When proposing tasks, highlight potential critical points that could lead to side effects.
