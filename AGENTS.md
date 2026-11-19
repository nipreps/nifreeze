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

# Agent instructions

Guidance for coding agents (Claude Code, Codex and others) working in this repository. `CLAUDE.md` is a symlink to this file.

## Project Overview

NiFreeze is a framework for estimation of volume-to-volume motion and volume-wise artifacts (such as eddy currents in diffusion-weighted volumes of dMRI) in 4D neuroimaging (dMRI, fMRI, PET). Part of the NiPreps ecosystem. Source lives under `src/nifreeze/`, tests under `test/`.

## Common Commands

### Bootstrap (required before first run)
```bash
python -m hatch version  # generates src/nifreeze/_version.py
```

### Running Tests

#### Pre-requisites
- Some software needs to be installed prior to testing, for example ANTs
  ```bash
  conda install -c conda-forge ants=2.4 libitk=5.3 libiconv
  ```
- Notebooks generate figures with latex commands inside, therefore:
  ```bash
  sudo apt install texlive texlive-latex-extra texlive-fonts-recommended cm-super dvipng
  ```
- A number of tests use pre-existing data (stored in the git-annex-enabled GIN G-Node https://gin.g-node.org/nipreps-data/tests-nifreeze) that need be found at location indicated by the environment variable `TEST_DATA_HOME`:
  ```bash
  uv tool install git-annex
  uv tool install --with=datalad-osf --with=datalad-next datalad
  uv tool install --with=datalad-next datalad-osf
  datalad wtf  # check datalad is installed

  # Install the dataset
  if [[ ! -d "${TEST_DATA_HOME}" ]]; then
      datalad install -rg --source=https://gin.g-node.org/nipreps-data/tests-nifreeze ${TEST_DATA_HOME}
  else
      cd ${TEST_DATA_HOME}
      datalad update --merge -r .
      datalad get -r -J4 *
  fi
  ```
  Files in GIN's annex can be retrieved using curl by composing the URL like this one for the [`dmri_data/motion_test_data/dwi_motion.h5` file](https://gin.g-node.org/nipreps-data/tests-nifreeze/raw/master/dmri_data/motion_test_data/dwi_motion.h5)

- Some test data comes from DIPY:
  ```bash
  echo "from dipy.data import fetch_stanford_hardi; fetch_stanford_hardi()" > fetch.py
  uv tool install dipy
  uv add --script fetch.py dipy
  uv run fetch.py
  ```

Details about testing are found in `.github/workflows/test.yml`

### Testing

```bash
pytest test/                              # unit tests only
pytest --doctest-modules src/nifreeze     # doctests only
pytest -sv --doctest-modules src test     # both (what tox runs)
pytest test/test_model_dmri.py            # single test file
pytest test/test_model_dmri.py::test_name # single test
pytest test/ -n auto                      # parallel execution
```

Full test suite via tox: `tox -v` (install with `uv tool install --with=tox-uv --with=tox-gh-actions tox`)

Other tox environments: `tox -e typecheck`, `tox -e spellcheck`, `tox -e docs`

### Linting and Formatting
```bash
ruff check --fix <files>   # lint and auto-fix
ruff format <files>        # reformat
```

CI runs the latest Ruff with `pipx run` (`.github/workflows/contrib.yml`). Ruff config: line-length 99, target Python 3.13, rules: F, E, C, W, B, I, ICN. Import convention: `nibabel` aliased as `nb`.

### Type checking
```bash
tox -e typecheck
```

### Spellcheck
```bash
tox -e spellcheck
```

### Documentation
```bash
python tools/fetch_gallery.py  # gallery pages (needs `gh auth`); placeholders otherwise
make -C docs/ html
```

CI builds the documentation as in `.github/workflows/docs.yml`, which also deploys it to `gh-pages` (`tools/docs_versions.py`).

## Architecture

Three-layer design: **Data → Model → Estimator**

- **Data** (`nifreeze/data/`): `BaseDataset` (attrs-based) with modality subclasses in `dmri/` (DWI) and `pet/`. Handles NIfTI/HDF5 I/O, brain masks, affine transforms, and train/test splitting (`splitting.py`).

- **Model** (`nifreeze/model/`): `BaseModel` ABC with `fit_predict(index, **kwargs) → ndarray`. `ModelFactory.init(model_name)` instantiates models. Implementations: DTI, DKI, AverageDWI, GP (in `dmri.py`), BSpline PET (`pet.py`), Gaussian Process Regression (`gpr.py`). Models operate in Leave-One-Volume-Out (LOVO) mode or single-fit mode.

- **Estimator** (`nifreeze/estimator.py`): Orchestrates model fitting + ANTs registration per volume. Core method: `Estimator.run(dataset, ...)`. Supports chaining via `prev_model`. Uses joblib for parallel execution.

- **Registration** (`nifreeze/registration/`): ANTs wrappers via nipype. Config YAML files in `registration/config/`.

- **CLI**: Entry point `nifreeze` → `nifreeze.cli.run:main`, parser in `cli/parser.py`.

- **Utilities**: Traversal strategies (`utils/iterators.py`), image operations (`utils/ndimage.py`), visualization (`viz/`), analysis/filtering (`analysis/`).

## Testing Details

- External test data from GIN (git-annex): location set by `TEST_DATA_HOME` (default `~/.nifreeze-tests`). Some tests use DIPY's Stanford HARDI dataset (`DIPY_HOME`).
- `TEST_OUTPUT_DIR` and `TEST_WORK_DIR` control artifact persistence (unset = discarded).
- `conftest.py` provides auto-use fixtures: `random_number_generator` (fixed seed 1234), `setup_random_dwi_data`, `setup_random_pet_data`, `setup_random_base_data`, etc. Tests use custom markers like `@pytest.mark.random_dwi_data(b0_thres, vol_size, use_random_gtab)`.
- ANTs must be installed for registration tests (`conda install -c conda-forge ants=2.4 libitk=5.3 libiconv`).
- Pytest is configured with `--doctest-modules` by default and `PYTHONHASHSEED=0`.
- Tests marked `slow` (exhaustive grids of cases) run in a plain `pytest`. `test.yml` deselects them on pull requests and pushes through `PYTEST_ADDOPTS`, and runs them on the weekly schedule and on manual runs.

## Key Conventions

- Calendar-based versioning (nipreps-calver) via hatch-vcs
- Python 3.13+ required; code targets `py313`
- Uses `attrs` for data classes (not dataclasses)
- `nibabel` is always imported as `nb`

## Working as an agent

- Plan first, and think harder while planning.
- When proposing tasks, flag the critical points that could have side effects.

## Issues

Every PR answers an open issue. An issue's opening states what is known and what is asked, in this order:

1. **Context**, one line if possible: what the work is about, linking the tools it names.
2. **The problem**, **Working hypothesis** or **The request**: the known facts; or the experiment's question and what each outcome would mean; or the specific change asked for. Show arithmetic inline (`0.074 = 0.12 - 0.046`).
3. **Important considerations**: a numbered list, each item a known fact and its consequence for the project.
4. **Additional information**, when needed: figures, relevant experience, links, file paths and records.

- An opening carries no result it does not have yet, and builds no suspense: the facts, then the consequence or the question.
- Results go in follow-up comments, in the order they are found. An experiment posts "**Test N, working hypothesis.**" (question, data, metric, decision rules) before it runs, and "**Test N, results.**" after. Write "working hypothesis", never "pre-registration".
- A figure goes in the comment of the result it shows, never in an earlier one, with a few plain lines: what it shows, how to read it, what it means.
- Issue titles use the uppercase types below. An `EDD:` title is the experiment's question.

## Pull requests

- Open a PR only for an open issue; if there is none, offer to open the issue instead. Always open it as a draft, so humans do not start reviewing before it is ready.
- The title has an uppercase type and is in Title Case (`ENH: Simulate A Slice Profile In The Phantom Generator`).
- The issue gives the why, so the description is brief:
  1. one line: what the PR does;
  2. **Changes**: a few bullets, one per concern;
  3. **Checked**: how it was verified, with the commands;
  4. `Resolves: #N`, then `X-Refs:` for related issues and PRs.

## Long reports

Tables, logs, benchmark output and full results go below a one-line summary, folded so that nothing is hidden and nothing overwhelms:

```markdown
One line on what the report found.

<details><summary>The full report</summary>

...

</details>
```

Leave a blank line after `<summary>` and before `</details>`, so the Markdown inside renders.

## Commits and types

- Format: `<type>[(<scope>)]: <message>`, as short as possible. Add a body only when the commit signals something important (something broken, a security fix, a major API change).
- Types are lowercase in commits and uppercase in PR and issue titles: `enh` (enhancements and features), `fix` (bug fixes), `doc` (documentation), `mnt` (maintenance: dependencies, CI, packaging), `sty` (style), `tst` (tests), and `edd`.
- `edd` is experiment-driven development (#614): an experiment's question, the helper code under `tools/registration/` that runs only that experiment, and its report under `docs/edd/`. A core or tool change that an experiment needs is `enh` or `fix`, with its own issue.
- Branches: `<type>/<identifier>[-gh<issue>]`, e.g. `fix/memory-leak-gh127`.
