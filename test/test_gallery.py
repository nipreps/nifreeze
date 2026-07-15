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
from gallery.run import applicable_matrix, run_gallery


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


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_run_gallery_single_fit_canary():
    """The gallery captures the self-consistency-canary warning for GQI/GP."""
    spec = synthetic_spec(SINGLE_SHELL, n_directions=14)
    manifest = run_gallery(
        [spec], model_keys=["dti", "gqi", "gp-spherical", "average"], render=False
    )

    assert _cell(manifest, "gqi", "single-fit").canary is True
    assert _cell(manifest, "gp-spherical", "single-fit").canary is True
    assert _cell(manifest, "dti", "single-fit").canary is False
    assert _cell(manifest, "average", "single-fit").canary is False
    # LOVO is never a canary.
    assert _cell(manifest, "gqi", "lovo").canary is False


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


def test_applicable_matrix():
    """The CI matrix lists only capability-applicable (dataset, model, mode) cells."""
    cells = applicable_matrix()

    # Every cell is a runnable combination (no inapplicable pairs leak through).
    specs = {m.key: m for m in GALLERY_MODELS}
    for c in cells:
        assert check_applicability(specs[c["model"]], c["scheme"])[0]
        assert check_mode(specs[c["model"]], c["mode"])[0]

    # DSI (ds004737) supports only average + GQI; GP and DKI never appear there.
    dsi_models = {c["model"] for c in cells if c["dataset"] == "ds004737"}
    assert dsi_models == {"average", "gqi"}

    # DKI is multi-shell only — never on the single-shell datasets.
    for c in cells:
        if c["model"] == "dki":
            assert c["scheme"] == MULTI_SHELL
        if c["model"] == "gp-spherical":
            assert c["scheme"] == SINGLE_SHELL

    # Both modes are enumerated for every applicable (dataset, model) pair.
    pairs = {(c["dataset"], c["model"]) for c in cells}
    assert len(cells) == 2 * len(pairs)


def test_applicable_matrix_filtered():
    """``applicable_matrix`` honors a modes restriction."""
    cells = applicable_matrix(modes=["lovo"])
    assert cells
    assert {c["mode"] for c in cells} == {"lovo"}


def test_manifest_merge_and_from_tree(tmp_path):
    """Manifest fragments merge into one sorted manifest, via objects and on disk."""
    a = GalleryManifest(
        cells=[CellResult("dsB", SINGLE_SHELL, "gqi", "lovo", STATUS_RAN, indices=[1])],
        metadata={"nifreeze_version": "1"},
    )
    b = GalleryManifest(
        cells=[CellResult("dsA", SINGLE_SHELL, "dti", "single-fit", STATUS_RAN, indices=[0])],
        metadata={"dipy_version": "2"},
    )

    merged = GalleryManifest.merge([a, b])
    assert len(merged.cells) == 2
    # Cells are sorted by (dataset, model, mode) regardless of fragment order.
    assert [c.dataset for c in merged.cells] == ["dsA", "dsB"]
    assert merged.metadata == {"nifreeze_version": "1", "dipy_version": "2"}

    # Same result reading fragments scattered across per-artifact subdirectories.
    (tmp_path / "cell-a").mkdir()
    (tmp_path / "cell-b").mkdir()
    a.to_json(tmp_path / "cell-a" / "gallery_manifest.json")
    b.to_json(tmp_path / "cell-b" / "gallery_manifest.json")
    from_tree = GalleryManifest.from_tree(tmp_path)
    assert from_tree.to_dict() == merged.to_dict()


def test_gallery_collect(tmp_path):
    """The collect tool merges manifest fragments and reassembles the panel tree."""
    import sys

    sys.path.insert(0, "tools")
    import gallery_collect

    staging = tmp_path / "staging"
    out = tmp_path / "out"
    # Two artifact subdirs, each a fit-job's slice of the output directory.
    for key, model, mode in [("a", "gqi", "lovo"), ("b", "dti", "single-fit")]:
        cell_dir = staging / f"gallery-cell-{key}"
        (cell_dir / "dsX").mkdir(parents=True)
        png = cell_dir / "dsX" / f"{model}_{mode}_001.png"
        png.write_bytes(b"\x89PNG\r\n\x1a\n")
        GalleryManifest(
            cells=[
                CellResult(
                    "dsX",
                    SINGLE_SHELL,
                    model,
                    mode,
                    STATUS_RAN,
                    indices=[1],
                    artifacts=[f"dsX/{model}_{mode}_001.png"],
                )
            ]
        ).to_json(cell_dir / "gallery_manifest.json")

    manifest = gallery_collect.collect(staging, out)

    assert len(manifest.cells) == 2
    assert (out / "gallery_manifest.json").is_file()
    # Panels are reassembled under <dataset>/ so notebooks can embed them.
    assert (out / "dsX" / "gqi_lovo_001.png").is_file()
    assert (out / "dsX" / "dti_single-fit_001.png").is_file()


def test_gallery_collect_reconciles_missing_cells(tmp_path):
    """A cell that produced no output is reported as an error, never dropped.

    A timed-out or OOM-killed fit job uploads nothing; the coverage table must
    still account for it rather than read as if it was never attempted.
    """
    import sys

    sys.path.insert(0, "tools")
    import gallery_collect

    staging = tmp_path / "staging"
    cell_dir = staging / "gallery-cell-a"
    cell_dir.mkdir(parents=True)
    GalleryManifest(
        cells=[CellResult("dsX", SINGLE_SHELL, "gqi", "lovo", STATUS_RAN, indices=[1])]
    ).to_json(cell_dir / "gallery_manifest.json")

    expected = [
        {"dataset": "dsX", "model": "gqi", "mode": "lovo", "scheme": SINGLE_SHELL},
        {"dataset": "dsX", "model": "dki", "mode": "lovo", "scheme": SINGLE_SHELL},  # vanished
    ]
    manifest = gallery_collect.collect(staging, tmp_path / "out", expected)

    assert len(manifest.cells) == 2
    dki = [c for c in manifest.cells if c.model == "dki"][0]
    assert dki.status == "error"
    assert dki.reason is not None
    assert "no output produced" in dki.reason
    # The cell that did report is untouched.
    assert [c for c in manifest.cells if c.model == "gqi"][0].status == STATUS_RAN


def test_dataset_page_rst():
    """A dataset page embeds the stored panels and reports provenance + coverage."""
    from gallery.pages import dataset_page_rst

    manifest = GalleryManifest(
        cells=[
            CellResult(
                "ds000206",
                SINGLE_SHELL,
                "dti",
                "lovo",
                STATUS_RAN,
                indices=[1],
                artifacts=["ds000206/dti_lovo_001.png"],
            ),
            CellResult(
                "ds000206",
                SINGLE_SHELL,
                "gqi",
                "single-fit",
                STATUS_RAN,
                indices=[1],
                artifacts=["ds000206/gqi_single-fit_001.png"],
                canary=True,
            ),
            # Another dataset's cells must not leak onto this page.
            CellResult("ds004737", DSI, "gqi", "lovo", STATUS_RAN, indices=[0]),
        ],
        metadata={"sources": {"ds000206": ["sub-THP0001/ses-1/dwi/x_dwi.nii.gz"]}},
    )

    page = dataset_page_rst(manifest, "ds000206")

    assert ".. _gallery_ds000206:" in page
    # Figures point at the stored panels, relative to the page.
    assert ".. figure:: ds000206/dti_lovo_001.png" in page
    assert ".. figure:: ds000206/gqi_single-fit_001.png" in page
    # The canary is labelled, and provenance/coverage are present.
    assert "gqi · single-fit (canary)" in page
    assert "sub-THP0001/ses-1/dwi/x_dwi.nii.gz" in page
    assert "list-table" in page
    # Scoped to this dataset only.
    assert "ds004737" not in page


def test_write_pages_figures_resolve(tmp_path):
    """Every figure a page references exists next to it after collection.

    A dangling figure path renders an empty gallery without failing anything,
    so the page and the panels it embeds are checked together.
    """
    import re

    from gallery.pages import write_pages

    panels = tmp_path / "panels"
    (panels / "ds000206").mkdir(parents=True)
    (panels / "ds000206" / "dti_lovo_001.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    manifest = GalleryManifest(
        cells=[
            CellResult(
                "ds000206",
                SINGLE_SHELL,
                "dti",
                "lovo",
                STATUS_RAN,
                indices=[1],
                artifacts=["ds000206/dti_lovo_001.png"],
            )
        ]
    )
    pages_dir = tmp_path / "pages"
    write_pages(manifest, pages_dir, panels_from=panels, datasets=["ds000206"])

    page = (pages_dir / "ds000206.rst").read_text()
    figures = re.findall(r"\.\. figure:: (\S+)", page)
    assert figures
    for rel in figures:
        assert (pages_dir / rel).is_file(), f"dangling figure reference: {rel}"


def test_write_pages_placeholder_for_unpublished(tmp_path):
    """Every requested dataset gets a page, so the toctree never dangles."""
    from gallery.pages import write_pages

    manifest = GalleryManifest(
        cells=[CellResult("ds000206", SINGLE_SHELL, "dti", "lovo", STATUS_RAN, indices=[1])]
    )
    pages = write_pages(manifest, tmp_path, datasets=["ds000206", "ds004737"])

    assert len(pages) == 2
    assert (tmp_path / "ds000206.rst").is_file()
    # The dataset with no cells is honest about not being published, not empty.
    unpublished = (tmp_path / "ds004737.rst").read_text()
    assert "have not been published yet" in unpublished


def test_merge_deep_merges_sources():
    """Per-dataset ``sources`` survive merging (one fragment knows one dataset)."""
    a = GalleryManifest(metadata={"sources": {"dsA": ["a.nii.gz"]}, "nifreeze_version": "1"})
    b = GalleryManifest(metadata={"sources": {"dsB": ["b.nii.gz"]}})

    merged = GalleryManifest.merge([a, b])
    assert merged.metadata["sources"] == {"dsA": ["a.nii.gz"], "dsB": ["b.nii.gz"]}
    assert merged.metadata["nifreeze_version"] == "1"


def test_source_relpaths_prefers_sidecar(tmp_path, monkeypatch):
    """The fetch-stage sidecar keeps the fit jobs off datalad entirely."""
    from gallery import datasets

    monkeypatch.setenv("NIFREEZE_GALLERY_H5DIR", str(tmp_path))
    (tmp_path / "ds000206.sources.json").write_text('["sub-THP0001/ses-1/dwi/x_dwi.nii.gz"]')
    # Resolving would clone; the sidecar must short-circuit that.
    monkeypatch.setitem(
        datasets.RESOLVERS, "ds000206", lambda *a, **k: pytest.fail("should not clone")
    )

    assert datasets.source_relpaths("ds000206") == ["sub-THP0001/ses-1/dwi/x_dwi.nii.gz"]


def test_run_gallery_dki_scheme_gating():
    """DKI is skipped on single-shell and runs on multi-shell."""
    ss = synthetic_spec(SINGLE_SHELL, n_directions=12)
    ms = synthetic_spec(MULTI_SHELL, n_directions=12)

    ss_manifest = run_gallery([ss], model_keys=["dki"], render=False)
    assert _cell(ss_manifest, "dki", "lovo").status == STATUS_SKIPPED

    ms_manifest = run_gallery([ms], model_keys=["dki"], render=False)
    assert _cell(ms_manifest, "dki", "lovo").status == STATUS_RAN
