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
"""
Registration with elastix, through ``itk-elastix`` (the ``elastix`` extra).

The images pass to elastix in memory: no files, unlike ANTs' command line.
:class:`Elastix` is a :class:`~nifreeze.registration.base.Registration`:
``register(fixed, moving, affine, mask, index, init)`` returns the transform that maps
each point of the fixed image (the model's prediction) onto the moving image (the
sample), in RAS+.

The images reach elastix with the values ANTs reads: clipped to the sample's window (as
``clip`` selects) and coded to 12 bits (as :func:`~nifreeze.registration.ants._to_nifti`
writes them), then winsorized and histogram-matched with ANTs' parameters, when the
settings ask for it. Only the fixed image is masked.

Settings are the ``elastix`` sections of :data:`~nifreeze.registration.base.RIGID` and
:data:`~nifreeze.registration.base.AFFINE` (or of any such file): one elastix parameter
map per stage, with the values that depend on the data filled at each call.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from os import PathLike
from typing import Any

import nibabel as nb
import numpy as np

from nifreeze.registration.base import WINDOW_LEVELS, Registration, load_config

LPS = np.diag([-1.0, -1.0, 1.0, 1.0])
"""The change between RAS+ and ITK's LPS world (its own inverse)."""
WINSORIZE_BINS = 256
"""Bins of the histogram from which ANTs takes the winsorizing quantiles."""
MATCH_LEVELS = 256
"""Histogram levels of ANTs' histogram matching."""
MATCH_POINTS = 12
"""Match points of ANTs' histogram matching."""
SETTINGS_KEYS = {
    "stages",
    "winsorize_lower_quantile",
    "winsorize_upper_quantile",
    "use_histogram_matching",
}
"""The keys of a settings file."""
STAGE_KEYS = {"parameters", "sampling_percentage", "shrink_factor", "smoothing_sigma"}
"""The keys of a stage."""
UNKNOWN_SETTINGS_ERROR_MSG = "Unknown elastix settings: {keys}"
"""Unknown settings error message."""
NO_STAGES_ERROR_MSG = "Elastix settings need at least one stage"
"""Settings without stages error message."""


def get_settings(align_config: Mapping[str, Any] | str | PathLike | None = None) -> dict:
    """
    Read elastix settings.

    Parameters
    ----------
    align_config : :obj:`~collections.abc.Mapping` or :obj:`os.pathlike`, optional
        Elastix's settings, or a file with an ``elastix`` section
        (default: :data:`~nifreeze.registration.base.RIGID`).

    Returns
    -------
    :obj:`dict`
        A copy of the settings, validated.

    Examples
    --------
    >>> settings = get_settings()
    >>> [stage["parameters"]["Transform"] for stage in settings["stages"]]
    [['EulerTransform'], ['EulerTransform']]

    >>> from nifreeze.registration import AFFINE
    >>> get_settings(AFFINE)["stages"][0]["parameters"]["Transform"]
    ['AffineTransform']

    >>> get_settings({"stages": [{"parameters": {}, "smoothing": 1}]})
    Traceback (most recent call last):
    ValueError: Unknown elastix settings: ['smoothing']

    """
    loaded = load_config(align_config, "elastix")
    if not loaded.get("stages"):
        raise ValueError(NO_STAGES_ERROR_MSG)
    unknown = set(loaded) - SETTINGS_KEYS
    for stage in loaded["stages"]:
        unknown |= set(stage) - STAGE_KEYS
    if unknown:
        raise ValueError(UNKNOWN_SETTINGS_ERROR_MSG.format(keys=sorted(unknown)))
    return loaded


def _coded(data: np.ndarray, window: tuple[float, float]) -> np.ndarray:
    """
    The values ANTs reads from the file :func:`~nifreeze.registration.ants._to_nifti` writes.

    With a window of 4095 units, each code is one unit:

    >>> _coded(np.array([-3.0, 10.4, 10.6, 5000.0]), (0.0, 4095.0)).tolist()
    [0.0, 10.0, 11.0, 4095.0]
    >>> _coded(np.array([0, 3, 9], dtype="int16"), (0.5, 8.5)).tolist()
    [1.0, 3.0, 8.0]

    """
    a_min, a_max = window
    if np.issubdtype(data.dtype, np.integer):
        return np.clip(data, int(np.ceil(a_min)), int(np.floor(a_max))).astype("float32")
    step = (a_max - a_min) / WINDOW_LEVELS
    codes = np.round((np.clip(data, a_min, a_max) - a_min) / step)
    return (codes * step + a_min).astype("float32")


def _quantile(data: np.ndarray, q: float) -> float:
    """
    A quantile as ANTs takes it for winsorizing: ITK's ``Histogram::Quantile``, 256 bins.

    >>> round(_quantile(np.linspace(0, 1, 10001), 0.25), 3)
    0.25

    """
    lo, hi = float(data.min()), float(data.max())
    hi += (hi - lo) / WINSORIZE_BINS / 10  # ITK widens the last bin to hold the maximum
    counts, edges = np.histogram(data, bins=WINSORIZE_BINS, range=(lo, hi))
    cumulated = np.cumsum(counts) / data.size
    n = min(int(np.searchsorted(cumulated, q)), WINSORIZE_BINS - 1)
    before = cumulated[n - 1] if n else 0.0
    share = counts[n] / data.size
    return float(edges[n] + (q - before) / share * (edges[n + 1] - edges[n]))


def _winsorize(data: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """Clip to two quantiles and rescale to [0, 1], in double precision as ANTs does."""
    low, high = _quantile(data, lower), _quantile(data, upper)
    values = np.asarray(data, dtype="float64")
    return np.clip((values - low) / (high - low), 0.0, 1.0).astype("float32")


def _to_itk(data: np.ndarray, affine: np.ndarray, pixel: str = "float32") -> Any:
    """An ITK image (LPS) of an array in nibabel's order, on a RAS+ grid."""
    import itk

    image = itk.image_from_array(np.ascontiguousarray(np.transpose(data, (2, 1, 0)).astype(pixel)))
    lps = LPS @ affine
    spacing = np.linalg.norm(lps[:3, :3], axis=0)
    image.SetSpacing(spacing.tolist())
    image.SetOrigin(lps[:3, 3].tolist())
    image.SetDirection(itk.matrix_from_array(np.ascontiguousarray(lps[:3, :3] / spacing)))
    return image


def _per_level(value: float | Sequence[float], levels: int) -> list[float]:
    """One value per resolution: a number for all of them, or a list of one each."""
    values = [float(v) for v in value] if isinstance(value, Sequence) else [float(value)] * levels
    if len(values) != levels:
        raise ValueError(f"{len(values)} values for {levels} resolutions")
    return values


def _parameter_maps(
    settings: dict, mask: np.ndarray, spacing: np.ndarray, seed: int | None
) -> list[dict[str, list[str]]]:
    """
    The stages' parameter maps, with the values that depend on the data.

    - ``NumberOfSpatialSamples``: the stage's ``sampling_percentage`` of the mask's
      voxels, divided by the cube of its ``shrink_factor`` (ANTs samples a grid shrunk
      by that factor);
    - ``ImagePyramidSmoothingSchedule``: the stage's ``smoothing_sigma`` (voxels) times
      the voxel size along each axis, in mm;
    - ``RandomSeed``: the seed.

    Each may be one number or a list of one per resolution (``NumberOfResolutions``).
    A value set in the stage's ``parameters`` is kept as it is.
    """
    maps = []
    for stage in settings["stages"]:
        pmap = {key: [str(v) for v in value] for key, value in stage["parameters"].items()}
        levels = int(pmap.get("NumberOfResolutions", ["1"])[0])
        if "sampling_percentage" in stage and "NumberOfSpatialSamples" not in pmap:
            fractions = _per_level(stage["sampling_percentage"], levels)
            shrinks = _per_level(stage.get("shrink_factor", 1), levels)
            pmap["NumberOfSpatialSamples"] = [
                str(int(round(p * mask.sum() / k**3)))
                for p, k in zip(fractions, shrinks, strict=True)
            ]
        if "smoothing_sigma" in stage and "ImagePyramidSmoothingSchedule" not in pmap:
            sigmas = _per_level(stage["smoothing_sigma"], levels)
            pmap["ImagePyramidSmoothingSchedule"] = [str(s * d) for s in sigmas for d in spacing]
        if seed is not None:
            pmap["RandomSeed"] = [str(seed)]
        maps.append(pmap)
    return maps


def _lps_matrix(transform: Any) -> np.ndarray:
    """The LPS 4x4 of an affine ITK transform, from the images of four points."""
    points = np.vstack([np.zeros(3), 100 * np.eye(3)])
    images = np.array([transform.TransformPoint(p.tolist()) for p in points])
    matrix = np.eye(4)
    matrix[:3, 3] = images[0]
    matrix[:3, :3] = (images[1:] - images[0]).T / 100
    return matrix


class Elastix(Registration):
    """
    Register with elastix, given the images in memory.

    Parameters
    ----------
    align_config : :obj:`~collections.abc.Mapping` or :obj:`os.pathlike`, optional
        Elastix's settings, or a file with an ``elastix`` section (see
        :func:`get_settings`).
    seed : :obj:`int`, optional
        Seed of elastix's random sampling (``RandomSeed``); :obj:`None` leaves
        elastix's default.
    **kwargs
        See :class:`~nifreeze.registration.base.Registration`. With ``num_threads``
        set, estimates repeat bit for bit at a fixed seed.

    """

    def __init__(
        self,
        align_config: Mapping[str, Any] | str | PathLike | None = None,
        *,
        seed: int | None = 12345,
        **kwargs: Any,
    ) -> None:
        super().__init__(get_settings(align_config), seed=seed, **kwargs)

    def _register(self, fixed, moving, affine, mask, index, init, window) -> np.ndarray:
        import itk

        mask = np.ones(fixed.shape, dtype=bool) if mask is None else np.asanyarray(mask, bool)
        fixed_values, moving_values = (
            _coded(data, window)
            if window is not None and self.clip in (which, "both")
            else np.asarray(data, dtype="float32")
            for data, which in ((fixed, "predicted"), (moving, "sample"))
        )
        settings = self.align_config
        if "winsorize_lower_quantile" in settings:
            lower = settings["winsorize_lower_quantile"]
            upper = settings["winsorize_upper_quantile"]
            fixed_values = _winsorize(fixed_values, lower, upper)
            moving_values = _winsorize(moving_values, lower, upper)
        if self.debug_dir is not None:
            for name, values in (("predicted", fixed_values), ("sample", moving_values)):
                nb.Nifti1Image(values, affine).to_filename(
                    self.debug_dir / f"{name}_{index:05d}.nii"
                )

        # elastix sets the process's maximum thread count; set it here too and restore it
        threader = itk.MultiThreaderBase
        previous = (
            threader.GetGlobalMaximumNumberOfThreads(),
            threader.GetGlobalDefaultNumberOfThreads(),
        )
        if self.num_threads is not None:
            threader.SetGlobalMaximumNumberOfThreads(self.num_threads)
            threader.SetGlobalDefaultNumberOfThreads(self.num_threads)
        try:
            fixed_image = _to_itk(fixed_values, affine)
            moving_image = _to_itk(moving_values, affine)
            if settings.get("use_histogram_matching", False):
                moving_image = itk.histogram_matching_image_filter(
                    moving_image,
                    reference_image=fixed_image,
                    number_of_histogram_levels=MATCH_LEVELS,
                    number_of_match_points=MATCH_POINTS,
                    threshold_at_mean_intensity=True,
                )

            parameters = itk.ParameterObject.New()
            spacing = np.linalg.norm(affine[:3, :3], axis=0)
            for pmap in _parameter_maps(settings, mask, spacing, self.seed):
                parameters.AddParameterMap(pmap)

            image_type = itk.Image[itk.F, 3]
            options: dict[str, Any] = {"log_to_console": False}
            if self.num_threads is not None:
                options["number_of_threads"] = self.num_threads
            method = itk.ElastixRegistrationMethod[image_type, image_type].New(
                fixed_image=fixed_image,
                moving_image=moving_image,
                parameter_object=parameters,
                fixed_mask=_to_itk(mask, affine, "uint8"),
                **options,
            )
            if init is not None:
                initial = LPS @ np.asarray(init, dtype=float) @ LPS
                transform = itk.AffineTransform[itk.D, 3].New()
                transform.SetMatrix(itk.matrix_from_array(np.ascontiguousarray(initial[:3, :3])))
                transform.SetTranslation(initial[:3, 3].tolist())
                method.SetInitialTransform(transform)
            method.UpdateLargestPossibleRegion()
            # The initial transform and every stage, composed (fixed to moving, LPS)
            total = _lps_matrix(method.GetCombinationTransform())
        finally:
            threader.SetGlobalMaximumNumberOfThreads(previous[0])
            threader.SetGlobalDefaultNumberOfThreads(previous[1])
        return LPS @ total @ LPS
