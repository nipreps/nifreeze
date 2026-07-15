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
"""Runner for the prediction gallery.

Iterates the *(dataset × model × mode)* matrix, loading each dataset once,
running LOVO and single-fit predictions for a few volumes, rendering panels, and
recording every cell in a :class:`~gallery.manifest.GalleryManifest`.
Inapplicable cells are pre-filtered by the capability contract (recorded as
``skipped``); unexpected failures are captured as ``error`` so one bad cell never
aborts the run.

Run as ``python -m gallery.run --out <dir>``.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path

from gallery import datasets as _datasets
from gallery.datasets import DatasetSpec
from gallery.manifest import (
    STATUS_ERROR,
    STATUS_RAN,
    STATUS_SKIPPED,
    CellResult,
    GalleryManifest,
)
from gallery.registry import (
    GALLERY_MODELS,
    GALLERY_MODES,
    ModelSpec,
    build_model,
    check_applicability,
    check_mode,
)


def _provenance() -> dict[str, str]:
    """Collect version provenance for the manifest metadata."""
    meta: dict[str, str] = {}
    try:
        from nifreeze import __version__ as nifreeze_version

        meta["nifreeze_version"] = str(nifreeze_version)
    except Exception:  # pragma: no cover - defensive
        pass
    try:
        import dipy

        meta["dipy_version"] = str(dipy.__version__)
    except Exception:  # pragma: no cover - defensive
        pass
    return meta


def _fitted_gpr(model):
    """Return the fitted sklearn GP regressor behind a GP model, or ``None``.

    Reaches through ``GPModel`` → ``GPFit`` → the underlying
    :obj:`~sklearn.gaussian_process.GaussianProcessRegressor`; returns ``None``
    for non-GP models (which have no such object).
    """
    models = getattr(model, "_models", None)
    if not models:
        return None
    return getattr(models[0], "model", None)


def _fmt_seconds(seconds: float) -> str:
    """Human-readable duration (``ms`` under a second, else ``s``)."""
    return f"{seconds * 1000:.0f} ms" if seconds < 1.0 else f"{seconds:.1f} s"


def _timed_predict(model, idx: int, mode: str, fit_shared: float):
    """Predict volume ``idx`` and return ``(predicted, fit_seconds, predict_seconds)``.

    Separates fit from predict without doing extra work: in single-fit mode the
    fit is shared (measured once), so each call is predict-only; in LOVO mode the
    model is (re)fit without ``idx``, then temporarily locked so the prediction
    reuses that fit — timing the two phases independently.
    """
    if mode == "single-fit" or not callable(getattr(model, "_fit", None)):
        t0 = time.perf_counter()
        predicted = model.fit_predict(idx)
        predict_s = time.perf_counter() - t0
        return predicted, fit_shared, predict_s

    # LOVO: fit on all-but-idx, then lock so ``fit_predict`` predicts without refitting.
    t0 = time.perf_counter()
    model._fit(idx)
    fit_s = time.perf_counter() - t0
    previous = model._locked_fit
    model._locked_fit = True
    try:
        t0 = time.perf_counter()
        predicted = model.fit_predict(idx)
        predict_s = time.perf_counter() - t0
    finally:
        model._locked_fit = previous
    return predicted, fit_s, predict_s


def _run_cell(
    spec: ModelSpec,
    mode: str,
    dwi,
    indices: Sequence[int],
    *,
    dataset_name: str,
    scheme: str,
    out_dir: Path | None,
    render: bool,
) -> CellResult:
    """Build, fit/predict, and render a single applicable cell."""
    import warnings

    from gallery.render import save_covariance_plot, save_slice_panel
    from nifreeze.model.base import SingleFitCanaryWarning

    model = build_model(spec, dwi)
    fit_shared = 0.0
    canary = False
    if mode == "single-fit":
        # Lock the fit once on all volumes, capturing the self-consistency-canary
        # warning (if the model emits it) rather than reading a model attribute.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            t0 = time.perf_counter()
            model.fit_predict(None)
            fit_shared = time.perf_counter() - t0
        canary = any(issubclass(w.category, SingleFitCanaryWarning) for w in caught)

    artifacts: list[str] = []
    used_indices: list[int] = []
    for idx in indices:
        idx = int(idx)
        predicted, fit_s, predict_s = _timed_predict(model, idx, mode, fit_shared)
        if predicted is None:
            continue
        used_indices.append(idx)
        if render and out_dir is not None:
            observed = dwi[idx][0]
            rel = f"{dataset_name}/{spec.key}_{mode}_{idx:03d}.png"
            canary_tag = " · canary" if canary else ""
            title = (
                f"{dataset_name} · {spec.label} · {mode}{canary_tag} · vol {idx}    "
                f"(fit {_fmt_seconds(fit_s)}, predict {_fmt_seconds(predict_s)})"
            )
            save_slice_panel(
                observed,
                predicted,
                out_dir / rel,
                affine=dwi.affine,
                mask=dwi.brainmask,
                title=title,
            )
            artifacts.append(rel)

    # For GP models, also plot the fitted angular covariance (Andersson Fig. 3).
    fitted_gpr = _fitted_gpr(model) if spec.key.startswith("gp") else None
    if render and out_dir is not None and fitted_gpr is not None:
        rel = f"{dataset_name}/{spec.key}_{mode}_covariance.png"
        save_covariance_plot(
            fitted_gpr,
            out_dir / rel,
            title=f"{dataset_name} · {spec.label} · {mode} · covariance",
        )
        artifacts.append(rel)

    return CellResult(
        dataset=dataset_name,
        scheme=scheme,
        model=spec.key,
        mode=mode,
        status=STATUS_RAN,
        indices=used_indices,
        artifacts=artifacts,
        canary=canary,
    )


def _evaluate_cell(
    spec: ModelSpec,
    mode: str,
    ds: DatasetSpec,
    dwi,
    indices: Sequence[int],
    applicable: bool,
    reason: str | None,
    out_path: Path | None,
    render: bool,
) -> CellResult:
    """Resolve a single cell to a skipped/errored/ran :class:`CellResult`."""
    if not applicable:
        return CellResult(ds.name, ds.scheme, spec.key, mode, STATUS_SKIPPED, reason)
    mode_ok, mode_reason = check_mode(spec, mode)
    if not mode_ok:
        return CellResult(ds.name, ds.scheme, spec.key, mode, STATUS_SKIPPED, mode_reason)
    try:
        return _run_cell(
            spec,
            mode,
            dwi,
            indices,
            dataset_name=ds.name,
            scheme=ds.scheme,
            out_dir=out_path,
            render=render,
        )
    except Exception as exc:
        return CellResult(ds.name, ds.scheme, spec.key, mode, STATUS_ERROR, reason=str(exc))


def run_gallery(
    dataset_specs: Sequence[DatasetSpec],
    *,
    models: Sequence[ModelSpec] | None = None,
    model_keys: Sequence[str] | None = None,
    modes: Sequence[str] = GALLERY_MODES,
    out_dir: str | Path | None = None,
    render: bool = True,
) -> GalleryManifest:
    """Execute the gallery matrix and return its manifest.

    Parameters
    ----------
    dataset_specs : sequence of :class:`~gallery.datasets.DatasetSpec`
        Datasets to run.
    models : sequence of :class:`~gallery.registry.ModelSpec`, optional
        Model specs to attempt (defaults to :data:`GALLERY_MODELS`).
    model_keys : sequence of :obj:`str`, optional
        Restrict to these model keys (a convenience filter over ``models``).
    modes : sequence of :obj:`str`, optional
        Prediction modes (defaults to :data:`GALLERY_MODES`).
    out_dir : :obj:`str` or :obj:`~pathlib.Path`, optional
        Where figures and the manifest are written. If ``None``, nothing is
        written to disk (manifest is still returned).
    render : :obj:`bool`, optional
        Whether to render figures (requires ``out_dir``).

    Returns
    -------
    :class:`~gallery.manifest.GalleryManifest`
        The coverage manifest.

    """
    model_specs = list(models if models is not None else GALLERY_MODELS)
    if model_keys is not None:
        wanted = set(model_keys)
        model_specs = [m for m in model_specs if m.key in wanted]

    out_path = Path(out_dir) if out_dir is not None else None
    if out_path is not None:
        out_path.mkdir(parents=True, exist_ok=True)

    manifest = GalleryManifest(metadata=_provenance())

    for ds in dataset_specs:
        try:
            dwi = ds.load()
        except Exception as exc:
            manifest.cells.append(
                CellResult(
                    dataset=ds.name,
                    scheme=ds.scheme,
                    model="<load>",
                    mode="-",
                    status=STATUS_ERROR,
                    reason=f"dataset load failed: {exc}",
                )
            )
            continue

        # Carry the exact subject/run forward in the manifest: the collect job
        # embeds the panels without the dataset (or its datalad clone) present,
        # so provenance has to travel with the manifest rather than be re-resolved.
        try:
            manifest.metadata.setdefault("sources", {})[ds.name] = _datasets.source_relpaths(
                ds.name
            )
        except Exception:  # pragma: no cover - provenance is best-effort
            pass

        indices = ds.lovo_indices(dwi)

        for spec in model_specs:
            applicable, reason = check_applicability(spec, ds.scheme)
            for mode in modes:
                manifest.cells.append(
                    _evaluate_cell(
                        spec, mode, ds, dwi, indices, applicable, reason, out_path, render
                    )
                )

    if out_path is not None:
        manifest.to_json(out_path / "gallery_manifest.json")
        (out_path / "coverage.rst").write_text(manifest.coverage_table_rst())

    return manifest


def applicable_matrix(
    dataset_specs: Sequence[DatasetSpec] | None = None,
    *,
    models: Sequence[ModelSpec] | None = None,
    modes: Sequence[str] = GALLERY_MODES,
) -> list[dict[str, str]]:
    """Enumerate the applicable *(dataset, model, mode)* cells.

    The capability contract is the single source of truth, so inapplicable
    combinations (e.g. GP on DSI, DKI on single-shell) are excluded here rather
    than spawning empty CI jobs. Returns one dict per runnable cell, suitable
    for a GitHub Actions ``strategy.matrix`` via ``fromJSON``.
    """
    specs = list(dataset_specs if dataset_specs is not None else _datasets.DATASETS)
    model_specs = list(models if models is not None else GALLERY_MODELS)
    cells: list[dict[str, str]] = []
    for ds in specs:
        for spec in model_specs:
            if not check_applicability(spec, ds.scheme)[0]:
                continue
            for mode in modes:
                if not check_mode(spec, mode)[0]:
                    continue
                cells.append(
                    {"dataset": ds.name, "model": spec.key, "mode": mode, "scheme": ds.scheme}
                )
    return cells


def _select_datasets(names: Sequence[str] | None) -> list[DatasetSpec]:
    """Resolve dataset specs by name (all registered datasets if ``names`` is None)."""
    specs = list(_datasets.DATASETS)
    if not specs:
        raise SystemExit("No datasets registered.")
    if not names:
        return specs
    by_name = {d.name: d for d in specs}
    unknown = [n for n in names if n not in by_name]
    if unknown:
        raise SystemExit(f"Unknown dataset(s) {unknown}; registered: {sorted(by_name)}.")
    return [by_name[n] for n in names]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory for figures and the coverage manifest.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Restrict to these dataset names (default: all registered).",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Restrict to these gallery model keys (default: all).",
    )
    parser.add_argument(
        "--modes",
        nargs="*",
        default=None,
        help=f"Restrict to these modes (default: {', '.join(GALLERY_MODES)}).",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Skip figure rendering (emit the manifest only).",
    )
    parser.add_argument(
        "--fetch-only",
        action="store_true",
        help="Materialize (download + crop) the selected datasets and exit, without fitting.",
    )
    parser.add_argument(
        "--print-matrix",
        action="store_true",
        help="Print the applicable (dataset, model, mode) cells as JSON and exit.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = _build_arg_parser().parse_args(argv)

    dataset_specs = _select_datasets(args.datasets)

    if args.print_matrix:
        print(json.dumps(applicable_matrix(dataset_specs, modes=args.modes or GALLERY_MODES)))
        return 0

    if args.fetch_only:
        # Materialize each dataset once and, when NIFREEZE_GALLERY_H5DIR is set,
        # persist the cropped DWI as a self-contained ``<name>.h5`` so the fit
        # jobs load it directly and never touch datalad or the network.
        import os

        h5dir = os.environ.get("NIFREEZE_GALLERY_H5DIR")
        for ds in dataset_specs:
            print(f"Fetching {ds.name} ...", flush=True)
            dwi = ds.load()
            if h5dir:
                dest = Path(h5dir) / f"{ds.name}.h5"
                dest.parent.mkdir(parents=True, exist_ok=True)
                dwi.to_filename(dest)
                print(f"  saved {dest}", flush=True)
                # Record which subject/run was used while the datalad clone is
                # still around; the fit jobs read this instead of re-resolving.
                sidecar = _datasets.sources_sidecar(ds.name, h5dir)
                if sidecar is not None:
                    sidecar.write_text(json.dumps(_datasets.source_relpaths(ds.name)))
                    print(f"  saved {sidecar}", flush=True)
        return 0

    if args.out is None:
        raise SystemExit("--out is required unless --print-matrix or --fetch-only is given.")

    manifest = run_gallery(
        dataset_specs,
        model_keys=args.models,
        modes=args.modes or GALLERY_MODES,
        out_dir=args.out,
        render=not args.no_render,
    )
    counts = manifest.counts()
    print(
        f"Gallery complete: {counts[STATUS_RAN]} ran, "
        f"{counts[STATUS_SKIPPED]} skipped, {counts[STATUS_ERROR]} errored."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
