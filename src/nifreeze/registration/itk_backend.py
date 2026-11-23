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
Registration with ITK's registration framework (the ``…v4`` classes of ITK 5).

ITK's framework optimizes several transforms together, which elastix's does not. So
this backend can estimate each volume's mapping as head motion followed by a field along
the phase-encoding direction, :math:`W = E M`, with 9 parameters (6 rigid, and the
phase-encoding row of :math:`E`; see
:func:`~nifreeze.registration.utils.split_phase_encoding`), when the settings say
``restrict: phase-encoding``. Otherwise it estimates a full affine (12 parameters).
With ``motion: fixed`` as well, the head motion stays as the start gives it (in the
eddy-current pass, the head-motion step's estimate), and only the field's 3 parameters
are estimated, about the grid's centre. The estimates are then not split: the
eddy-current pass stores them as it stores elastix's, with the head motion it holds
factored out and kept unchanged.

The images reach ITK in their voxel frame (identity direction, origin zero, the voxel
spacing), so ITK's physical axes are the voxel axes, and the restriction to the
phase-encoding axis, an axis of the data array, is exact on oblique data. The estimate
goes back to RAS+. The images are prepared as the elastix backend prepares them:
clipped to the sample's window and coded to 12 bits (as ``clip`` selects), then
winsorized and histogram-matched when the settings ask. Only the fixed image is masked.

Settings are the ``itk`` sections of :data:`~nifreeze.registration.base.AFFINE` (or of
any such file), with one registration per stage. A stage samples its metric on a share of
the grid (``sampling_percentage``), or at a number of points within the mask
(``samples``), as elastix's random samplers do: its cost then does not grow with the grid.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from os import PathLike
from typing import Any
from warnings import warn

import nibabel as nb
import numpy as np

from nifreeze.registration.base import AFFINE, Registration, load_config
from nifreeze.registration.elastix import MATCH_LEVELS, MATCH_POINTS, _coded, _winsorize
from nifreeze.registration.utils import phase_encoding_vector, split_phase_encoding

RESTRICTIONS = ("phase-encoding",)
"""The fields a setting ``restrict`` can name."""
MOTIONS = ("refine", "fixed")
"""What a restricted registration does with the head motion: refine it, or keep the start's."""
METRICS = ("mattes", "correlation")
"""The metrics of a stage: Mattes' mutual information, or normalized correlation."""
SAMPLINGS = ("none", "regular", "random")
"""The samplings of a stage's metric."""
SETTINGS_KEYS = {
    "stages",
    "restrict",
    "motion",
    "winsorize_lower_quantile",
    "winsorize_upper_quantile",
    "use_histogram_matching",
}
"""The keys of a settings file."""
STAGE_KEYS = {
    "metric",
    "bins",
    "sampling",
    "sampling_percentage",
    "samples",
    "smoothing_sigma",
    "shrink_factor",
    "iterations",
    "learning_rate",
    "relaxation",
    "minimum_step",
    "gradient_tolerance",
}
"""The keys of a stage."""
UNKNOWN_SETTINGS_ERROR_MSG = "Unknown ITK settings: {keys}"
"""Unknown settings error message."""
NO_STAGES_ERROR_MSG = "ITK settings need at least one stage"
"""Settings without stages error message."""
UNKNOWN_VALUE_ERROR_MSG = "Unknown ITK {key} {value!r}: expected one of {values}"
"""Unknown value of a setting error message."""
BOTH_SAMPLINGS_ERROR_MSG = "An ITK stage takes sampling_percentage or samples, not both"
"""A stage with both a share and a count of samples error message."""
FIXED_MOTION_ERROR_MSG = (
    "ITK settings keep the head motion fixed only with restrict: phase-encoding"
)
"""Fixed head motion without the restriction error message."""
NO_PHASE_ENCODING_ERROR_MSG = (
    "The settings restrict the field to phase encoding, but its direction is unknown: "
    "give the backend phase_encoding (the dataset's, as BIDS writes it)."
)
"""Restricted settings without a phase-encoding direction error message."""
OUTSIDE_FAMILY_WARN_MSG = (
    "Volume {index}'s initial mapping is not head motion followed by a field along phase "
    "encoding; the registration starts from its projection."
)
"""An initial mapping outside the 9-parameter family warning message."""


def get_settings(align_config: Mapping[str, Any] | str | PathLike | None = None) -> dict:
    """
    Read and validate ITK settings.

    Parameters
    ----------
    align_config : :obj:`~collections.abc.Mapping` or :obj:`os.pathlike`, optional
        The settings, or a file with an ``itk`` section (default:
        :data:`~nifreeze.registration.base.AFFINE`).

    Returns
    -------
    :obj:`dict`
        A copy of the settings, validated.

    Examples
    --------
    >>> settings = get_settings()
    >>> settings["restrict"], [stage["metric"] for stage in settings["stages"]]
    ('phase-encoding', ['mattes', 'correlation'])

    >>> get_settings({"stages": [{"metric": "mattes", "smoothing": 1}]})
    Traceback (most recent call last):
    ValueError: Unknown ITK settings: ['smoothing']

    >>> get_settings(settings | {"motion": "fixed"})["motion"]
    'fixed'

    """
    loaded = load_config(AFFINE if align_config is None else align_config, "itk")
    if not loaded.get("stages"):
        raise ValueError(NO_STAGES_ERROR_MSG)
    unknown = set(loaded) - SETTINGS_KEYS
    for stage in loaded["stages"]:
        unknown |= set(stage) - STAGE_KEYS
    if unknown:
        raise ValueError(UNKNOWN_SETTINGS_ERROR_MSG.format(keys=sorted(unknown)))
    checks: list[tuple[str, Any, tuple[Any, ...]]] = [
        ("restrict", loaded.get("restrict"), (None, *RESTRICTIONS)),
        ("motion", loaded.get("motion", "refine"), MOTIONS),
    ]
    for stage in loaded["stages"]:
        if {"sampling_percentage", "samples"} <= set(stage):
            raise ValueError(BOTH_SAMPLINGS_ERROR_MSG)
        checks += [
            ("metric", stage.get("metric"), METRICS),
            ("sampling", stage.get("sampling", "none"), SAMPLINGS),
        ]
    for key, value, values in checks:
        if value not in values:
            raise ValueError(UNKNOWN_VALUE_ERROR_MSG.format(key=key, value=value, values=values))
    if loaded.get("motion") == "fixed" and loaded.get("restrict") is None:
        raise ValueError(FIXED_MOTION_ERROR_MSG)
    return loaded


def _prepare(
    fixed: np.ndarray,
    moving: np.ndarray,
    window: tuple[float, float] | None,
    clip: str,
    settings: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """
    The values of both images as the backend registers them.

    Each image is coded to the sample's ``window`` as ``clip`` selects, and both are
    winsorized when the settings ask.
    """
    fixed_values, moving_values = (
        _coded(data, window)
        if window is not None and clip in (which, "both")
        else np.asarray(data, dtype="float32")
        for data, which in ((fixed, "predicted"), (moving, "sample"))
    )
    if "winsorize_lower_quantile" in settings:
        lower = settings["winsorize_lower_quantile"]
        upper = settings["winsorize_upper_quantile"]
        fixed_values = _winsorize(fixed_values, lower, upper)
        moving_values = _winsorize(moving_values, lower, upper)
    return fixed_values, moving_values


def _matched(moving: Any, fixed: Any) -> Any:
    """The moving ITK image, histogram-matched to the fixed one as ANTs does."""
    import itk

    return itk.histogram_matching_image_filter(
        moving,
        reference_image=fixed,
        number_of_histogram_levels=MATCH_LEVELS,
        number_of_match_points=MATCH_POINTS,
        threshold_at_mean_intensity=True,
    )


@contextmanager
def _threads(num_threads: int | None) -> Iterator[None]:
    """
    Set ITK's process-wide thread counts, and restore them on leaving.

    Elastix sets the process's maximum thread count, so each backend sets it too.
    """
    import itk

    threader = itk.MultiThreaderBase
    previous = (
        threader.GetGlobalMaximumNumberOfThreads(),
        threader.GetGlobalDefaultNumberOfThreads(),
    )
    if num_threads is not None:
        threader.SetGlobalMaximumNumberOfThreads(num_threads)
        threader.SetGlobalDefaultNumberOfThreads(num_threads)
    try:
        yield
    finally:
        threader.SetGlobalMaximumNumberOfThreads(previous[0])
        threader.SetGlobalDefaultNumberOfThreads(previous[1])


def _voxel_image(data: np.ndarray, spacing: np.ndarray, pixel: str = "float32") -> Any:
    """An ITK image in the voxel frame: identity direction, origin zero, the spacing."""
    import itk

    image = itk.image_from_array(np.ascontiguousarray(np.transpose(data, (2, 1, 0)).astype(pixel)))
    image.SetSpacing(spacing.tolist())
    return image


def _frame(affine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The map from the voxel frame (voxel axes, in mm) to RAS+, and the spacing."""
    affine = np.asarray(affine, dtype=float)
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    frame = np.eye(4)
    frame[:3, :3] = affine[:3, :3] / spacing
    frame[:3, 3] = affine[:3, 3]
    return frame, spacing


def _matrix(transform: Any) -> np.ndarray:
    """The ``4 x 4`` matrix of an ITK matrix-offset transform."""
    import itk

    matrix = np.eye(4)
    matrix[:3, :3] = itk.array_from_matrix(transform.GetMatrix())
    matrix[:3, 3] = list(transform.GetOffset())
    return matrix


def _set(transform: Any, matrix: np.ndarray, centre: np.ndarray) -> None:
    """Set an ITK matrix-offset transform, about ``centre``, to a ``4 x 4`` matrix."""
    import itk

    linear = np.ascontiguousarray(matrix[:3, :3])
    transform.SetCenter(centre.tolist())
    transform.SetMatrix(itk.matrix_from_array(linear))
    # ITK has T(x) = A (x - c) + c + t, so a matrix's offset o needs t = o - c + A c
    transform.SetTranslation((matrix[:3, 3] - centre + linear @ centre).tolist())


class ITK(Registration):
    """
    Register with ITK's registration framework, given the images in memory.

    Parameters
    ----------
    align_config : :obj:`~collections.abc.Mapping` or :obj:`os.pathlike`, optional
        ITK's settings, or a file with an ``itk`` section (see :func:`get_settings`).
    seed : :obj:`int`, optional
        Seed of the metrics' random sampling; :obj:`None` leaves ITK's.
    **kwargs
        See :class:`~nifreeze.registration.base.Registration`. With settings restricted
        to phase encoding, ``phase_encoding`` is required.

    """

    def __init__(
        self,
        align_config: Mapping[str, Any] | str | PathLike | None = None,
        *,
        seed: int | None = 12345,
        **kwargs: Any,
    ) -> None:
        super().__init__(get_settings(align_config), seed=seed, **kwargs)
        self.stops: list[tuple[int, str]] = []
        """Each stage's iterations and stop condition, in the last registration."""

    @property
    def restricted(self) -> bool:
        """Whether the field is restricted to the phase-encoding direction."""
        return self.align_config.get("restrict") == "phase-encoding"

    @property
    def fixed_motion(self) -> bool:
        """Whether the head motion stays as the start gives it (``motion: fixed``)."""
        return self.align_config.get("motion") == "fixed"

    def split(
        self, matrix: np.ndarray, affine: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """
        See :meth:`~nifreeze.registration.base.Registration.split`.

        With ``motion: fixed``, the head motion was not estimated, so there is nothing to
        split: the field, about the grid's centre, carries its own shift along phase
        encoding, which the split would move into the head motion.
        """
        if not self.restricted or self.phase_encoding is None or self.fixed_motion:
            return None
        return split_phase_encoding(matrix, phase_encoding_vector(self.phase_encoding, affine))

    def _register(self, fixed, moving, affine, mask, index, init, window) -> np.ndarray:
        import itk

        if self.restricted and self.phase_encoding is None:
            raise ValueError(NO_PHASE_ENCODING_ERROR_MSG)
        mask = np.ones(fixed.shape, dtype=bool) if mask is None else np.asanyarray(mask, bool)
        settings = self.align_config
        fixed_values, moving_values = _prepare(fixed, moving, window, self.clip, settings)
        if self.debug_dir is not None:
            for name, values in (("predicted", fixed_values), ("sample", moving_values)):
                nb.Nifti1Image(values, affine).to_filename(
                    self.debug_dir / f"{name}_{index:05d}.nii"
                )

        frame, spacing = _frame(affine)
        start = np.linalg.inv(frame) @ (np.eye(4) if init is None else np.asarray(init)) @ frame
        centre = spacing * (np.array(fixed.shape) - 1) / 2

        with _threads(self.num_threads):
            fixed_image = _voxel_image(fixed_values, spacing)
            moving_image = _voxel_image(moving_values, spacing)
            if settings.get("use_histogram_matching", False):
                moving_image = _matched(moving_image, fixed_image)
            mask_object = itk.ImageMaskSpatialObject[3].New()
            mask_object.SetImage(_voxel_image(mask, spacing, "uint8"))
            mask_object.Update()

            transform, members, weights = self._transform(start, centre, index)
            inside = int(mask.sum())
            self.stops = [
                self._stage(
                    stage, fixed_image, moving_image, (mask_object, inside), transform, weights
                )
                for stage in settings["stages"]
            ]
            total = np.eye(4)
            for member in members:  # applied last first
                total = total @ _matrix(member)
        return frame @ total @ np.linalg.inv(frame)

    def _transform(
        self, start: np.ndarray, centre: np.ndarray, index: int
    ) -> tuple[Any, list[Any], list[float] | None]:
        """
        The transform to optimize, its members (applied last first) and its weights.

        Restricted, it is the composite of a field ``E`` and head motion ``M`` (applied
        first), with every parameter of ``E`` but its phase-encoding row frozen, in the
        voxel frame; with ``motion: fixed``, ``M``'s are frozen too. ``E`` keeps the
        start's shift along phase encoding: none when the start has no field, so that the
        field then turns about the grid's centre. Otherwise it is a full affine.
        """
        import itk

        if not self.restricted:
            whole = itk.AffineTransform[itk.D, 3].New()
            _set(whole, start, centre)
            return whole, [whole], None

        assert self.phase_encoding is not None
        axis = "ijk".index(self.phase_encoding[0])
        motion, field = split_phase_encoding(start, np.eye(3)[axis])
        if not np.allclose(field @ motion, start, atol=1e-6):
            warn(OUTSIDE_FAMILY_WARN_MSG.format(index=index), stacklevel=3)
        rigid = itk.Euler3DTransform[itk.D].New()
        _set(rigid, motion, centre)
        eddy = itk.AffineTransform[itk.D, 3].New()
        _set(eddy, field, centre)
        composite = itk.CompositeTransform[itk.D, 3].New()
        composite.AddTransform(eddy)
        composite.AddTransform(rigid)  # the last added is applied first
        composite.SetAllTransformsToOptimizeOn()
        # The optimized parameters: the rigid ones, then E's matrix (row-major) and offset
        row = np.zeros((3, 3))
        row[axis] = 1.0
        rigid_weight = 0.0 if self.fixed_motion else 1.0
        weights = [rigid_weight] * 6 + row.ravel().tolist() + [0.0, 0.0, 0.0]
        return composite, [eddy, rigid], weights

    def _stage(
        self,
        stage: Mapping[str, Any],
        fixed: Any,
        moving: Any,
        mask: tuple[Any, int],
        transform: Any,
        weights: list[float] | None,
    ) -> tuple[int, str]:
        """
        Optimize ``transform`` in place with one stage's settings; how it stopped.

        ``mask`` is the fixed image's mask, and how many voxels it holds.
        """
        import itk

        mask, inside = mask
        image_type = type(fixed)
        if stage["metric"] == "mattes":
            metric = itk.MattesMutualInformationImageToImageMetricv4[image_type, image_type].New()
            metric.SetNumberOfHistogramBins(int(stage.get("bins", 32)))
        else:
            metric = itk.CorrelationImageToImageMetricv4[image_type, image_type].New()
        metric.SetFixedImageMask(mask)

        optimizer = itk.RegularStepGradientDescentOptimizerv4[itk.D].New()
        optimizer.SetLearningRate(float(stage["learning_rate"]))
        optimizer.SetMinimumStepLength(float(stage["minimum_step"]))
        optimizer.SetNumberOfIterations(int(stage["iterations"]))
        optimizer.SetRelaxationFactor(float(stage["relaxation"]))
        optimizer.SetGradientMagnitudeTolerance(float(stage.get("gradient_tolerance", 1e-8)))
        if weights is not None:
            optimizer.SetWeights(itk.OptimizerParameters[itk.D](weights))
        scales = itk.RegistrationParameterScalesFromPhysicalShift[type(metric)].New()
        scales.SetMetric(metric)
        optimizer.SetScalesEstimator(scales)

        method = itk.ImageRegistrationMethodv4[image_type, image_type].New(
            FixedImage=fixed, MovingImage=moving, Metric=metric, Optimizer=optimizer
        )
        method.SetInitialTransform(transform)
        method.SetInPlace(True)
        method.SetNumberOfLevels(1)
        method.SetShrinkFactorsPerLevel([int(stage.get("shrink_factor", 1))])
        method.SetSmoothingSigmasPerLevel([float(stage.get("smoothing_sigma", 0.0))])
        method.SetSmoothingSigmasAreSpecifiedInPhysicalUnits(False)
        strategies = itk.ImageRegistrationMethodv4Enums
        method.SetMetricSamplingStrategy(
            {
                "none": strategies.MetricSamplingStrategy_NONE,
                "regular": strategies.MetricSamplingStrategy_REGULAR,
                "random": strategies.MetricSamplingStrategy_RANDOM,
            }[stage.get("sampling", "none")]
        )
        percentage = float(stage.get("sampling_percentage", 1.0))
        if "samples" in stage:
            # ITK draws a share of the (shrunk) grid and drops the points outside the mask
            shrink = int(stage.get("shrink_factor", 1))
            percentage = min(1.0, int(stage["samples"]) * shrink**3 / max(inside, 1))
        method.SetMetricSamplingPercentage(percentage)
        if self.seed is not None:
            method.MetricSamplingReinitializeSeed(int(self.seed))
        method.Update()
        return int(optimizer.GetCurrentIteration()), optimizer.GetStopConditionDescription()
