# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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
  uvx datalad-installer --sudo ok git-annex
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
python -m ruff check --fix <files>   # lint and auto-fix
python -m run ruff format <files>        # reformat
```

Ruff config: line-length 99, target Python 3.10, rules: F, E, C, W, B, I, ICN. Import convention: `nibabel` aliased as `nb`.

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
make -C docs/ html
```

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

## Git etiquette and conventions

- Semantic commit format: `<type>[(<scope>)]: <message>` — types: `fix:`, `enh:`, `sty:`, `doc:`, `mnt:`, etc. Scope parenthetical is optional.
- PR titles use uppercase (e.g., `FIX: <description>`, `ENH: <description>`, description is title-case). PRs link existing issues and other PRs if necessary in the message body, using GitHub's keywords, e.g., `Resolves: #17`, `Fixes: #19`, `X-Refs: #29`.
- Branch names follow the convention `<type>/<identifier>[-gh<issue-number>]`. If the new branch addresses a bug/feature with an existing issue open, the issue is referenced with the optional suffix (e.g., `fix/memory-leak-gh127` for a memory leak reported in issue \#127).

## Key Conventions

- Calendar-based versioning (nipreps-calver) via hatch-vcs
- Python 3.10+ required; code targets `py310`
- Uses `attrs` for data classes (not dataclasses)
- `nibabel` is always imported as `nb`
