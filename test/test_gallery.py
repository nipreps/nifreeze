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
"""Fast, network-free tests for the prediction gallery harness."""

from __future__ import annotations

import pytest
from gallery.datasets import (
    DATASETS,
    DSI,
    MULTI_SHELL,
    SINGLE_SHELL,
    _cache_root,
    default_lovo_indices,
    load_ds000206,
    synthetic_dwi,
    synthetic_spec,
    verify_scheme,
)
from gallery.manifest import (
    STATUS_RAN,
    STATUS_SKIPPED,
    CellResult,
    GalleryManifest,
)
from gallery.registry import (
    GALLERY_MODELS,
    check_applicability,
    check_mode,
)
from gallery.run import run_gallery


def _cell(manifest, model, mode):
    """Return the single cell matching ``model``/``mode``."""
    matches = [c for c in manifest.cells if c.model == model and c.mode == mode]
    assert len(matches) == 1, f"expected one {model}/{mode} cell, got {len(matches)}"
    return matches[0]


@pytest.mark.parametrize(
    "scheme", [SINGLE_SHELL, MULTI_SHELL, DSI], ids=["single", "multi", "dsi"]
)
def test_synthetic_dwi_scheme(scheme):
    """The synthetic builder yields data classified as the requested scheme."""
    dwi = synthetic_dwi(scheme, n_directions=24)
    assert verify_scheme(dwi, scheme) == scheme


def test_verify_scheme_mismatch_raises():
    dwi = synthetic_dwi(SINGLE_SHELL)
    with pytest.raises(ValueError, match="Scheme mismatch"):
        verify_scheme(dwi, MULTI_SHELL, name="synthetic")


def test_default_lovo_indices():
    dwi = synthetic_dwi(SINGLE_SHELL, n_directions=24)
    indices = default_lovo_indices(dwi, count=3)
    assert len(indices) == 3
    assert all(0 <= i < len(dwi) for i in indices)
    assert indices == sorted(set(indices))


def test_manifest_roundtrip(tmp_path):
    manifest = GalleryManifest(
        cells=[
            CellResult("dsX", SINGLE_SHELL, "dti", "lovo", STATUS_RAN, indices=[1, 2]),
            CellResult(
                "dsX", SINGLE_SHELL, "average", "single-fit", STATUS_SKIPPED, "no single-fit"
            ),
        ],
        metadata={"nifreeze_version": "0.0"},
    )
    path = manifest.to_json(tmp_path / "m.json")
    reloaded = GalleryManifest.from_json(path)
    assert reloaded.to_dict() == manifest.to_dict()
    assert reloaded.counts()[STATUS_RAN] == 1
    assert "list-table" in manifest.coverage_table_rst()


def test_capability_filtering_helpers():
    """The registry helpers reflect the model capability contract."""
    specs = {m.key: m for m in GALLERY_MODELS}

    # Average supports both modes (single-fit averages all volumes).
    assert check_mode(specs["average"], "single-fit")[0]
    assert check_mode(specs["average"], "lovo")[0]

    # DKI is multi-shell only.
    assert not check_applicability(specs["dki"], SINGLE_SHELL)[0]
    assert check_applicability(specs["dki"], MULTI_SHELL)[0]
    assert not check_applicability(specs["dki"], DSI)[0]

    # GP kernels are scheme-specific.
    assert check_applicability(specs["gp-spherical"], SINGLE_SHELL)[0]
    assert not check_applicability(specs["gp-spherical"], MULTI_SHELL)[0]
    assert check_applicability(specs["gp-multishell"], MULTI_SHELL)[0]
    assert not check_applicability(specs["gp-multishell"], SINGLE_SHELL)[0]

    # DTI excludes DSI.
    assert not check_applicability(specs["dti"], DSI)[0]


def test_run_gallery_average_single_shell():
    """Average runs in both modes (single-fit averages all volumes)."""
    spec = synthetic_spec(SINGLE_SHELL, n_directions=20)
    manifest = run_gallery([spec], model_keys=["average"], render=False)

    assert _cell(manifest, "average", "lovo").status == STATUS_RAN
    assert _cell(manifest, "average", "single-fit").status == STATUS_RAN


def test_run_gallery_dti_renders(tmp_path):
    """DTI runs both modes and writes figures + a manifest to disk."""
    spec = synthetic_spec(SINGLE_SHELL, n_directions=20)
    manifest = run_gallery([spec], model_keys=["dti"], out_dir=tmp_path, render=True)

    lovo = _cell(manifest, "dti", "lovo")
    single = _cell(manifest, "dti", "single-fit")
    assert lovo.status == STATUS_RAN
    assert single.status == STATUS_RAN
    assert lovo.indices and lovo.artifacts

    # Figures and manifest are on disk.
    for rel in lovo.artifacts:
        assert (tmp_path / rel).is_file()
    assert (tmp_path / "gallery_manifest.json").is_file()
    assert (tmp_path / "coverage.rst").is_file()


def test_datasets_registry():
    """The OpenNeuro registry declares the four expected scheme datasets."""
    by_name = {d.name: d for d in DATASETS}
    assert set(by_name) == {"ds000206", "ds000114", "ds003138", "ds004737"}
    assert by_name["ds000206"].scheme == SINGLE_SHELL
    assert by_name["ds000114"].scheme == SINGLE_SHELL
    assert by_name["ds003138"].scheme == MULTI_SHELL
    assert by_name["ds004737"].scheme == DSI
    assert all(callable(d.loader) for d in DATASETS)


def _ds000206_cached() -> bool:
    """Whether ds000206's GD31 volume has been fetched into the local cache."""
    root = _cache_root() / "ds000206"
    # Mirror the loader's ``_first`` (sorted) so we check the file it will pick.
    matches = sorted(root.glob("sub-THP0001/ses-*/dwi/*acq-GD31*_dwi.nii.gz"))
    # ``exists()`` follows the annex symlink: True only if the data is present.
    return bool(matches) and matches[0].exists()


def test_select_cut_coords():
    """Cut coords come from the mask's high-mass slices, in world-z order."""
    import numpy as np
    from gallery.render import select_cut_coords

    assert select_cut_coords(None, np.eye(4), 4) is None

    mask = np.zeros((10, 10, 20), dtype=bool)
    mask[3:7, 3:7, 5:15] = True  # substantive content only in slices z=5..14
    coords = select_cut_coords(mask, np.eye(4), 4)
    assert coords is not None
    assert len(coords) == 4
    assert coords == sorted(coords)
    # With an identity affine, world-z == voxel-k; all cuts sit in the mass band.
    assert all(5 <= c <= 14 for c in coords)


def test_local_correlation():
    """Sliding-window correlation is full-resolution, ~1 for a linear map."""
    import numpy as np
    from gallery.render import _local_correlation

    rng = np.random.default_rng(0)
    observed = rng.normal(size=(16, 16, 16))
    predicted = 2.0 * observed + 1.0  # perfectly (linearly) correlated

    full = np.ones((16, 16, 16), dtype=bool)
    corr = _local_correlation(observed, predicted, full, window=7)
    # Full resolution: same shape as the inputs (stride 1).
    assert corr.shape == observed.shape
    vals = corr[np.isfinite(corr)]
    assert vals.size > 0
    assert np.all(vals > 0.99)

    # Voxels outside the mask stay NaN; in-mask voxels are computed.
    partial = np.zeros((16, 16, 16), dtype=bool)
    partial[:8, :8, :8] = True
    corr2 = _local_correlation(observed, predicted, partial, window=7)
    assert np.isnan(corr2[12, 12, 12])
    assert np.isfinite(corr2[2, 2, 2])


@pytest.mark.skipif(not _ds000206_cached(), reason="ds000206 data not fetched locally")
def test_load_ds000206_real():
    """The ds000206 loader builds a valid single-shell DWI from real data."""
    from gallery.datasets import source_relpaths

    dwi = load_ds000206()
    assert verify_scheme(dwi, SINGLE_SHELL) == SINGLE_SHELL
    assert len(dwi) >= 6
    assert dwi.brainmask is not None
    assert dwi.brainmask.shape == dwi.dataobj.shape[:3]

    # The exact subject/run is resolvable for the gallery page.
    paths = source_relpaths("ds000206")
    assert paths and paths[0].startswith("sub-THP0001/") and "acq-GD31" in paths[0]


def test_run_gallery_dki_scheme_gating():
    """DKI is skipped on single-shell and runs on multi-shell."""
    ss = synthetic_spec(SINGLE_SHELL, n_directions=12)
    ms = synthetic_spec(MULTI_SHELL, n_directions=12)

    ss_manifest = run_gallery([ss], model_keys=["dki"], render=False)
    assert _cell(ss_manifest, "dki", "lovo").status == STATUS_SKIPPED

    ms_manifest = run_gallery([ms], model_keys=["dki"], render=False)
    assert _cell(ms_manifest, "dki", "lovo").status == STATUS_RAN
