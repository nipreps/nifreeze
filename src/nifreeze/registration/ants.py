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
"""Using ANTs for image registration."""

from __future__ import annotations

from collections import namedtuple
from collections.abc import Mapping
from pathlib import Path
from shutil import rmtree
from tempfile import mkdtemp
from typing import Any
from warnings import warn
from weakref import finalize

import nibabel as nb
import nitransforms as nt
import numpy as np
from nipype.interfaces.ants.registration import Registration
from nitransforms.linear import Affine

from nifreeze.registration import base
from nifreeze.registration.base import WINDOW_LEVELS, _write_aligned  # noqa: F401

PARAMETERS_SINGLE_LIST = {
    "radius_or_number_of_bins",
    "sampling_percentage",
    "metric",
    "sampling_strategy",
}

REGISTRATION_MALFORMED_SETTINGS_ERROR_MSG = "Malformed settings file (levels: {levels})"
"""Registration malformed settings error message."""

REGISTRATION_INIT_WITHOUT_COLLAPSE_ERROR_MSG = (
    "An initial transform requires 'collapse_output_transforms': otherwise the first "
    "output transform is the initialization, not the estimate."
)
"""Initial transform without collapsed outputs error message."""


def _to_nifti(
    data: np.ndarray,
    affine: np.ndarray,
    filename: str | Path,
    window: tuple[float, float] | None = None,
) -> None:
    """
    Save data as a NIfTI file, optionally clipped to an intensity window.

    Clipping works on a copy (``data`` may be a view into the dataset) and keeps
    the original units. Integer data keep their type and values within the
    window. Floating-point data are stored as 12-bit codes (int16), and the
    ``scl_slope`` and ``scl_inter`` header fields map them back to the window.

    Parameters
    ----------
    data : :obj:`~numpy.ndarray`
        The image data to be saved.
    affine : :obj:`~numpy.ndarray`
        The affine transformation matrix.
    filename : :obj:`os.pathlike`
        The file path where the NIfTI file will be saved.
    window : :obj:`tuple` of :obj:`float`, optional
        The intensity window ``(a_min, a_max)`` to clip to. If :obj:`None`,
        the data are saved as they are.

    """
    data = np.squeeze(data)
    slope_inter = None
    if window is not None:
        a_min, a_max = window
        if np.issubdtype(data.dtype, np.integer):
            data = np.clip(data, int(np.ceil(a_min)), int(np.floor(a_max)))
        else:
            step = (a_max - a_min) / WINDOW_LEVELS
            data = np.round((np.clip(data, a_min, a_max) - a_min) / step).astype(np.int16)
            slope_inter = (step, a_min)

    nii = nb.Nifti1Image(
        data,
        affine,
        None,
    )
    if slope_inter is not None:
        nii.header.set_slope_inter(*slope_inter)
    nii.header.set_sform(affine, code=1)
    nii.header.set_qform(affine, code=1)
    nii.to_filename(filename)


def _prepare_registration_data(
    sample: np.ndarray,
    predicted: np.ndarray,
    affine: np.ndarray,
    vol_idx: int,
    dirname: Path | str,
    clip: str | bool | None = None,
    init_affine: np.ndarray | None = None,
    window: tuple[float, float] | None = None,
) -> tuple[Path, Path, Path | None]:
    """
    Prepare the registration data: save the moving and predicted (fixed) images to disk.

    Both images are clipped (as ``clip`` selects) to the same intensity window,
    so that they reach the registration with the same intensity mapping.

    Parameters
    ----------
    sample : :obj:`~numpy.ndarray`
        Current volume for which a transformation is to be estimated.
    predicted : :obj:`~numpy.ndarray`
        Predicted volume's data array (that is, spatial reference).
    affine : :obj:`~numpy.ndarray`
        Orientation affine from the original NIfTI.
    vol_idx : :obj:`int`
        Volume index.
    dirname : :obj:`os.pathlike`
        Directory name where the data is saved.
    clip : :obj:`str`, optional
        Clip intensity of ``"sample"``, ``"predicted"``, ``"both"``,
        or ``"none"`` of the images.
    init_affine : :obj:`~numpy.ndarray`, optional
        Initial affine transform.
    window : :obj:`tuple` of :obj:`float`, optional
        The intensity window ``(a_min, a_max)``. If :obj:`None` and clipping is
        requested, it is estimated from ``sample`` with
        :obj:`~nifreeze.data.filtering.robust_window`.

    Returns
    -------
    predicted_path : :obj:`~pathlib.Path`
        Predicted image filename.
    sample_path : :obj:`~pathlib.Path`
        Current volume's filename.
    init_path : :obj:`~pathlib.Path` or :obj:`None`
        An initialization affine (for second and further estimators).

    """

    predicted_path = Path(dirname) / f"predicted_{vol_idx:05d}.nii"
    sample_path = Path(dirname) / f"sample_{vol_idx:05d}.nii"

    clip = str(clip).lower()
    clip_sample = clip in ("sample", "both", "true")
    clip_predicted = clip in ("predicted", "both", "true")
    if window is None and (clip_sample or clip_predicted):
        from nifreeze.data.filtering import robust_window

        window = robust_window(sample)

    _to_nifti(sample, affine, sample_path, window=window if clip_sample else None)
    _to_nifti(predicted, affine, predicted_path, window=window if clip_predicted else None)

    init_path = None
    if init_affine is not None:
        ImageGrid = namedtuple("ImageGrid", ("shape", "affine"))
        reference = ImageGrid(shape=sample.shape[:3], affine=affine)
        initial_xform = Affine(matrix=init_affine, reference=reference)
        init_path = Path(dirname) / f"init_{vol_idx:05d}.mat"
        initial_xform.to_filename(init_path, fmt="itk")

    return predicted_path, sample_path, init_path


def _massage_mask_path(mask_path: str | Path | list[str], nlevels: int) -> list[str]:
    """
    Generate nipype-compatible masks paths.

    Parameters
    ----------
    mask_path : :obj:`os.pathlike` or :obj:`list`
        Path(s) to the mask file(s).
    nlevels : :obj:`int`
        Number of registration levels.

    Returns
    -------
    :obj:`list`
        A list of mask paths formatted for *Nipype*.

    Examples
    --------
    >>> _massage_mask_path("/some/path", 2)
    ['/some/path', '/some/path']

    >>> _massage_mask_path(["/some/path"] * 2, 2)
    ['/some/path', '/some/path']

    >>> _massage_mask_path(["/some/path"] * 2, 4)
    ['NULL', 'NULL', '/some/path', '/some/path']

    """
    if isinstance(mask_path, (str, Path)):
        return [str(mask_path)] * nlevels
    if len(mask_path) < nlevels:
        return ["NULL"] * (nlevels - len(mask_path)) + mask_path
    if len(mask_path) > nlevels:
        warn("More mask paths than levels", stacklevel=1)
        return mask_path[:nlevels]
    return mask_path


def generate_command(
    fixed_path: str | Path,
    moving_path: str | Path,
    fixedmask_path: str | Path | list[str] | None = None,
    movingmask_path: str | Path | list[str] | None = None,
    init_affine: str | Path | None = None,
    align_config: Mapping | str | Path | None = None,
    terminal_output: str | None = None,
    num_threads: int | None = None,
    environ: dict | None = None,
    output_transform_prefix: str | None = None,
) -> Registration:
    """
    Generate an ANTs' command line.

    Parameters
    ----------
    fixed_path : :obj:`os.pathlike`
        Path to the fixed image.
    moving_path : :obj:`os.pathlike`
        Path to the moving image.
    fixedmask_path : :obj:`os.pathlike` or :obj:`list`, optional
        Path to the fixed image mask.
    movingmask_path : :obj:`os.pathlike` or :obj:`list`, optional
        Path to the moving image mask.
    init_affine : :obj:`os.pathlike`, optional
        Initial affine transformation.
    align_config : :obj:`~collections.abc.Mapping` or :obj:`os.pathlike`, optional
        ANTs' settings, or a file with an ``ants`` section
        (default: :data:`~nifreeze.registration.base.RIGID`).
    terminal_output : :obj:`str`, optional
        Redirect terminal output (Nipype configuration)
    environ : :obj:`dict`, optional
        Add environment variables to the execution.
    num_threads : :obj:`int`, optional
        Set the number of threads for ANTs' execution.
    output_transform_prefix : :obj:`str`, optional
        The prefix of the output transforms (default: nipype's).

    Returns
    -------
    :obj:`~nipype.interfaces.ants.Registration`
        The configured Nipype interface of ANTs registration.

    Examples
    --------
    >>> generate_command(
    ...     fixed_path=repodata / 'fileA.nii.gz',
    ...     moving_path=repodata / 'fileB.nii.gz',
    ... ).cmdline  # doctest: +NORMALIZE_WHITESPACE
    'antsRegistration --collapse-output-transforms 1 --dimensionality 3 \
    --initialize-transforms-per-stage 0 --interpolation Linear --output transform \
    --transform Rigid[ 12.0 ] \
    --metric GC[ .../fileA.nii.gz, \
        .../fileB.nii.gz, \
        1, 3, Random, 0.4 ] \
    --convergence [ 20, 1e-06, 4 ] --smoothing-sigmas 2.71vox --shrink-factors 3 \
    --use-histogram-matching 1 \
    --transform Rigid[ 1.96 ] \
    --metric GC[ .../fileA.nii.gz, \
        .../fileB.nii.gz, \
        1, 4, Random, 0.18 ] \
    --convergence [ 10, 1e-07, 2 ] --smoothing-sigmas 0.0vox --shrink-factors 2 \
    --use-histogram-matching 1 \
    -v --winsorize-image-intensities [ 0.063, 0.991 ] \
    --write-composite-transform 0'

    >>> generate_command(
    ...     fixed_path=repodata / 'fileA.nii.gz',
    ...     moving_path=repodata / 'fileB.nii.gz',
    ...     fixedmask_path=repodata / 'maskA.nii.gz',
    ...     align_config=base.AFFINE,
    ... ).cmdline  # doctest: +NORMALIZE_WHITESPACE
    'antsRegistration --collapse-output-transforms 1 --dimensionality 3 \
    --initialize-transforms-per-stage 0 --interpolation Linear --output transform \
    --transform Affine[ 0.01 ] --metric Mattes[ .../fileA.nii.gz, .../fileB.nii.gz, 1, 32, \
    Regular, 0.2 ] --convergence [ 100, 1e-06, 15 ] --smoothing-sigmas 2.0vox \
    --shrink-factors 1 --use-histogram-matching 1 --masks [ .../maskA.nii.gz, NULL ] \
    --transform Affine[ 0.001 ] --metric GC[ .../fileA.nii.gz, .../fileB.nii.gz, 1, 5, \
    Random, 0.1 ] --convergence [ 50, 1e-07, 5 ] --smoothing-sigmas 0.0vox \
    --shrink-factors 1 --use-histogram-matching 1 --masks [ .../maskA.nii.gz, NULL ] \
    --winsorize-image-intensities [ 0.0001, 0.9998 ]  --write-composite-transform 0'

    >>> generate_command(
    ...     fixed_path=repodata / 'fileA.nii.gz',
    ...     moving_path=repodata / 'fileB.nii.gz',
    ...     fixedmask_path=[repodata / 'maskA.nii.gz'],
    ...     align_config=base.AFFINE,
    ... ).cmdline  # doctest: +NORMALIZE_WHITESPACE
    'antsRegistration --collapse-output-transforms 1 --dimensionality 3 \
    --initialize-transforms-per-stage 0 --interpolation Linear --output transform \
    --transform Affine[ 0.01 ] --metric Mattes[ .../fileA.nii.gz, .../fileB.nii.gz, 1, 32, \
    Regular, 0.2 ] --convergence [ 100, 1e-06, 15 ] --smoothing-sigmas 2.0vox \
    --shrink-factors 1 --use-histogram-matching 1 --masks [ NULL, NULL ] --transform \
    Affine[ 0.001 ] --metric GC[ .../fileA.nii.gz, .../fileB.nii.gz, 1, 5, Random, 0.1 ] \
    --convergence [ 50, 1e-07, 5 ] --smoothing-sigmas 0.0vox --shrink-factors 1 \
    --use-histogram-matching 1 --masks [ .../maskA.nii.gz, NULL ] \
    --winsorize-image-intensities [ 0.0001, 0.9998 ]  --write-composite-transform 0'

    """

    settings = base.load_config(align_config, "ants")
    if output_transform_prefix is not None:
        settings["output_transform_prefix"] = output_transform_prefix

    # Determine number of levels and assert consistency of levels
    levels = {len(settings[p]) for p in PARAMETERS_SINGLE_LIST if p in settings}
    nlevels = levels.pop()
    if levels:
        raise RuntimeError(REGISTRATION_MALFORMED_SETTINGS_ERROR_MSG.format(levels=levels))

    # Set fixed masks if provided
    if fixedmask_path is not None:
        settings["fixed_image_masks"] = [
            str(p) for p in _massage_mask_path(fixedmask_path, nlevels)
        ]

    # Set moving masks if provided
    if movingmask_path is not None:
        settings["moving_image_masks"] = [
            str(p) for p in _massage_mask_path(movingmask_path, nlevels)
        ]

    # Set initializing affine if provided
    if init_affine is not None:
        # The output is read back as the absolute estimate, which holds only
        # when ANTs collapses the initialization into the estimated transform
        if not settings.get("collapse_output_transforms", False):
            raise ValueError(REGISTRATION_INIT_WITHOUT_COLLAPSE_ERROR_MSG)
        settings["initial_moving_transform"] = str(init_affine)

    # Generate command line with nipype and return
    reg_iface = Registration(
        fixed_image=str(Path(fixed_path).absolute()),
        moving_image=str(Path(moving_path).absolute()),
        terminal_output=terminal_output,
        environ=environ or {},
        **settings,
    )
    if num_threads:
        reg_iface.inputs.num_threads = num_threads
        # nipype exports only NSLOTS, which ITK reads only when this is unset
        reg_iface.inputs.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(num_threads)

    return reg_iface


def _run_registration(
    fixed_path: str | Path,
    moving_path: str | Path,
    vol_idx: int,
    dirname: Path,
    **kwargs,
) -> nt.base.BaseTransform:
    """
    Register the moving image to the fixed image.

    Parameters
    ----------
    fixed_path : :obj:`Path`
        Fixed image filename.
    moving_path : :obj:`Path`
        Moving image filename.
    vol_idx : :obj:`int`
        Dataset volume index.
    dirname : :obj:`Path`
        Directory name where the transformation is saved.
    kwargs : :obj:`dict`
        Parameters to configure the image registration process.

    Returns
    -------
    xform : :obj:`~nitransforms.base.BaseTransform`
        Registration transformation.

    """

    align_kwargs = kwargs.copy()
    environ = align_kwargs.pop("environ", None)
    num_threads = align_kwargs.pop("num_threads", None)

    if (seed := align_kwargs.pop("seed", None)) is not None:
        environ = environ or {}
        environ["ANTS_RANDOM_SEED"] = str(seed)

    registration = generate_command(
        fixed_path,
        moving_path,
        environ=environ,
        terminal_output="file_split",
        num_threads=num_threads,
        **align_kwargs,
    )

    (dirname / f"cmd-{vol_idx:05d}.sh").write_text(registration.cmdline)

    # execute ants command line
    result = registration.run(cwd=str(dirname)).outputs

    # read output transform
    xform = nt.linear.Affine(
        nt.io.itk.ITKLinearTransform.from_filename(result.forward_transforms[0]).to_ras(
            reference=fixed_path, moving=moving_path
        ),
    )

    return xform


class Ants(base.Registration):
    """
    Register with ``antsRegistration``, through files and nipype.

    Each call writes both images (:func:`_prepare_registration_data`), runs
    ``antsRegistration`` (:func:`_run_registration`) and reads the transform back.

    Parameters
    ----------
    align_config : :obj:`~collections.abc.Mapping` or :obj:`os.pathlike`, optional
        ANTs' settings, or a file with an ``ants`` section
        (default: :data:`~nifreeze.registration.base.RIGID`).
    workdir : :obj:`os.pathlike`, optional
        Where the files go without ``debug_dir`` (default: a temporary folder,
        removed with the backend).
    **kwargs
        See :class:`~nifreeze.registration.base.Registration`.

    """

    def __init__(
        self,
        align_config: Mapping | str | Path | None = None,
        *,
        workdir: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        # Read once, so that a file without an ``ants`` section fails here
        super().__init__(base.load_config(align_config, "ants"), **kwargs)
        folder = self.debug_dir or workdir
        if folder is None:
            folder = mkdtemp(prefix="nifreeze-ants-")
            finalize(self, rmtree, folder, ignore_errors=True)
        self.workdir = Path(folder).absolute()
        self.workdir.mkdir(parents=True, exist_ok=True)
        self._mask: tuple[np.ndarray, Path] | None = None

    def _register(self, fixed, moving, affine, mask, index, init, window) -> np.ndarray:
        mask_path = None
        if mask is not None:
            if self._mask is None or self._mask[0] is not mask:
                # Written once for as long as the same mask comes, as the estimator did
                path = self.workdir / "brainmask.nii"
                nb.Nifti1Image(mask.astype(np.uint8), affine, None).to_filename(path)
                self._mask = (mask, path)
            mask_path = self._mask[1]

        fixed_path, moving_path, init_path = _prepare_registration_data(
            moving,
            fixed,
            affine,
            index,
            self.workdir,
            clip=self.clip if window is not None else "none",
            window=window,
            init_affine=init,
        )
        xform = _run_registration(
            fixed_path,
            moving_path,
            index,
            self.workdir,
            init_affine=init_path,
            fixedmask_path=mask_path,
            output_transform_prefix=f"ants-{index:05d}",
            num_threads=self.num_threads,
            seed=self.seed,
            align_config=self.align_config,
        )
        return np.asarray(xform.matrix)
