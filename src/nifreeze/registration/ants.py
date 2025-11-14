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
from importlib.resources import files
from json import loads
from pathlib import Path
from warnings import warn

import nibabel as nb
import nitransforms as nt
import numpy as np
from nipype.interfaces.ants.registration import Registration
from nitransforms.linear import Affine
from nitransforms.resampling import apply

PARAMETERS_SINGLE_VALUE = {
    "collapse_output_transforms",
    "dimension",
    "initial_moving_transform",
    "initialize_transforms_per_stage",
    "interpolation",
    "output_transform_prefix",
    "verbose",
    "winsorize_lower_quantile",
    "winsorize_upper_quantile",
    "write_composite_transform",
}

PARAMETERS_SINGLE_LIST = {
    "radius_or_number_of_bins",
    "sampling_percentage",
    "metric",
    "sampling_strategy",
}
PARAMETERS_DOUBLE_LIST = {"shrink_factors", "smoothing_sigmas", "transform_parameters"}


def _to_nifti(
    data: np.ndarray, affine: np.ndarray, filename: str | Path, clip: bool = True
) -> None:
    """
    Save data as a NIfTI file, optionally applying clipping.

    Parameters
    ----------
    data : :obj:`~numpy.ndarray`
        The image data to be saved.
    affine : :obj:`~numpy.ndarray`
        The affine transformation matrix.
    filename : :obj:`os.pathlike`
        The file path where the NIfTI file will be saved.
    clip : :obj:`bool`, optional
        Whether to apply clipping to the data before saving.

    """
    data = np.squeeze(data)
    if clip:
        from nifreeze.data.filtering import advanced_clip

        advanced_clip(data, inplace=True)

    nii = nb.Nifti1Image(
        data,
        affine,
        None,
    )
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
) -> tuple[Path, Path, Path | None]:
    """
    Prepare the registration data: save the moving and predicted (fixed) images to disk.

    Parameters
    ----------
    sample : :obj:`~numpy.ndarray`
        Current volume for which a transformation is to be estimated.
    predicted : :obj:`~numpy.ndarray`
        Predicted volume's data array (that is, spatial reference).
    affine : :obj:`numpy.ndarray`
        Orientation affine from the original NIfTI.
    vol_idx : :obj:`int`
        Volume index.
    dirname : :obj:`os.pathlike`
        Directory name where the data is saved.
    clip : :obj:`str` or ``None``
        Clip intensity of ``"sample"``, ``"predicted"``, ``"both"``,
        or ``"none"`` of the images.

    Returns
    -------
    predicted_path : :obj:`~pathlib.Path`
        Predicted image filename.
    sample_path : :obj:`~pathlib.Path`
        Current volume's filename.
    init_path : :obj:`~pathlib.Path` or ``None``
        An initialization affine (for second and further estimators).

    """

    predicted_path = Path(dirname) / f"predicted_{vol_idx:05d}.nii.gz"
    sample_path = Path(dirname) / f"sample_{vol_idx:05d}.nii.gz"

    _to_nifti(
        sample,
        affine,
        sample_path,
        clip=str(clip).lower() in ("sample", "both", "true"),
    )
    _to_nifti(
        predicted,
        affine,
        predicted_path,
        clip=str(clip).lower() in ("predicted", "both", "true"),
    )

    init_path = None
    if init_affine is not None:
        ImageGrid = namedtuple("ImageGrid", ("shape", "affine"))
        reference = ImageGrid(shape=sample.shape[:3], affine=affine)
        initial_xform = Affine(matrix=init_affine, reference=reference)
        init_path = Path(dirname) / f"init_{vol_idx:05d}.mat"
        initial_xform.to_filename(init_path, fmt="itk")

    return predicted_path, sample_path, init_path


def _get_ants_settings(settings: str = "b0-to-b0_level0") -> Path:
    """
    Retrieve the path to ANTs settings configuration file.

    Parameters
    ----------
    settings : :obj:`str`, optional
        Name of the settings configuration.

    Returns
    -------
    :obj:`~pathlib.Path`
        The path to the configuration file.

    Examples
    --------
    >>> _get_ants_settings()
    PosixPath('.../config/b0-to-b0_level0.json')

    >>> _get_ants_settings("b0-to-b0_level1")
    PosixPath('.../config/b0-to-b0_level1.json')

    """
    return Path(str(files("nifreeze.registration").joinpath(f"config/{settings}.json")))


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
    default: str = "b0-to-b0_level0",
    terminal_output: str | None = None,
    num_threads: int | None = None,
    environ: dict | None = None,
    **kwargs,
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
    default : :obj:`str`, optional
        Default settings configuration.
    terminal_output : :obj:`str`, optional
        Redirect terminal output (Nipype configuration)
    environ : :obj:`dict`, optional
        Add environment variables to the execution.
    num_threads : :obj:`int`, optional
        Set the number of threads for ANTs' execution.
    **kwargs : :obj:`dict`
        Additional parameters for ANTs registration.

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
    ...     default="dwi-to-b0_level0",
    ... ).cmdline  # doctest: +NORMALIZE_WHITESPACE
    'antsRegistration --collapse-output-transforms 1 --dimensionality 3 \
    --initialize-transforms-per-stage 0 --interpolation Linear --output transform \
    --transform Rigid[ 0.01 ] --metric Mattes[ \
        .../fileA.nii.gz, \
        .../fileB.nii.gz, \
        1, 32, Regular, 0.2 \
    ] --convergence [ 100x50, 1e-05, 10 ] --smoothing-sigmas 2.0x0.0vox \
    --shrink-factors 2x1 --use-histogram-matching 1 --transform Rigid[ 0.001 ] \
    --metric Mattes[ \
        .../fileA.nii.gz, \
        .../fileB.nii.gz, \
        1, 32, Random, 0.1 \
    ] --convergence [ 25, 1e-06, 2 ] --smoothing-sigmas 0.0vox --shrink-factors 1 \
    --use-histogram-matching 1 -v --winsorize-image-intensities [ 0.0001, 0.9998 ] \
    --write-composite-transform 0'

    >>> generate_command(
    ...     fixed_path=repodata / 'fileA.nii.gz',
    ...     moving_path=repodata / 'fileB.nii.gz',
    ...     fixedmask_path=repodata / 'maskA.nii.gz',
    ...     default="dwi-to-b0_level0",
    ... ).cmdline  # doctest: +NORMALIZE_WHITESPACE
    'antsRegistration --collapse-output-transforms 1 --dimensionality 3 \
    --initialize-transforms-per-stage 0 --interpolation Linear --output transform \
    --transform Rigid[ 0.01 ] --metric Mattes[ \
        .../fileA.nii.gz, \
        .../fileB.nii.gz, \
        1, 32, Regular, 0.2 ] \
    --convergence [ 100x50, 1e-05, 10 ] --smoothing-sigmas 2.0x0.0vox --shrink-factors 2x1 \
    --use-histogram-matching 1 --masks [ \
        .../maskA.nii.gz, NULL ] \
    --transform Rigid[ 0.001 ] --metric Mattes[ \
        .../fileA.nii.gz, \
        .../fileB.nii.gz, \
        1, 32, Random, 0.1 ] \
    --convergence [ 25, 1e-06, 2 ] --smoothing-sigmas 0.0vox --shrink-factors 1 \
    --use-histogram-matching 1 --masks [ \
        .../maskA.nii.gz, NULL ] \
    -v --winsorize-image-intensities [ 0.0001, 0.9998 ]  --write-composite-transform 0'

    >>> generate_command(
    ...     fixed_path=repodata / 'fileA.nii.gz',
    ...     moving_path=repodata / 'fileB.nii.gz',
    ...     default="dwi-to-b0_level0",
    ... ).cmdline  # doctest: +NORMALIZE_WHITESPACE
    'antsRegistration --collapse-output-transforms 1 --dimensionality 3 \
    --initialize-transforms-per-stage 0 --interpolation Linear --output transform \
    --transform Rigid[ 0.01 ] --metric Mattes[ \
        .../fileA.nii.gz, \
        .../fileB.nii.gz, \
        1, 32, Regular, 0.2 \
    ] --convergence [ 100x50, 1e-05, 10 ] --smoothing-sigmas 2.0x0.0vox \
    --shrink-factors 2x1 --use-histogram-matching 1 --transform Rigid[ 0.001 ] \
    --metric Mattes[ \
        .../fileA.nii.gz, \
        .../fileB.nii.gz, \
        1, 32, Random, 0.1 \
    ] --convergence [ 25, 1e-06, 2 ] --smoothing-sigmas 0.0vox --shrink-factors 1 \
    --use-histogram-matching 1 -v --winsorize-image-intensities [ 0.0001, 0.9998 ] \
    --write-composite-transform 0'

    >>> generate_command(
    ...     fixed_path=repodata / 'fileA.nii.gz',
    ...     moving_path=repodata / 'fileB.nii.gz',
    ...     fixedmask_path=[repodata / 'maskA.nii.gz'],
    ...     default="dwi-to-b0_level0",
    ... ).cmdline  # doctest: +NORMALIZE_WHITESPACE
    'antsRegistration --collapse-output-transforms 1 --dimensionality 3 \
    --initialize-transforms-per-stage 0 --interpolation Linear --output transform \
    --transform Rigid[ 0.01 ] --metric Mattes[ \
        .../fileA.nii.gz, \
        .../fileB.nii.gz, \
        1, 32, Regular, 0.2 ] \
    --convergence [ 100x50, 1e-05, 10 ] --smoothing-sigmas 2.0x0.0vox --shrink-factors 2x1 \
    --use-histogram-matching 1 --masks [ NULL, NULL ] \
    --transform Rigid[ 0.001 ] --metric Mattes[ \
        .../fileA.nii.gz, \
        .../fileB.nii.gz, \
        1, 32, Random, 0.1 ] \
    --convergence [ 25, 1e-06, 2 ] --smoothing-sigmas 0.0vox --shrink-factors 1 \
    --use-histogram-matching 1 --masks [ \
        .../maskA.nii.gz, NULL ] \
    -v --winsorize-image-intensities [ 0.0001, 0.9998 ]  --write-composite-transform 0'

    """

    # Bootstrap settings from defaults file and override with single-valued parameters in args
    settings = loads(_get_ants_settings(default).read_text()) | {
        k: kwargs.pop(k) for k in PARAMETERS_SINGLE_VALUE if k in kwargs
    }

    # Determine number of levels and assert consistency of levels
    levels = {len(settings[p]) for p in PARAMETERS_SINGLE_LIST if p in settings}
    nlevels = levels.pop()
    if levels:
        raise RuntimeError(f"Malformed settings file (levels: {levels})")

    # Override list (and nested-list) parameters
    for key, value in kwargs.items():
        if key in PARAMETERS_DOUBLE_LIST:
            value = [value]
        elif key not in PARAMETERS_SINGLE_LIST:
            continue

        if levels == 1:
            settings[key] = [value]
        else:
            settings[key][-1] = value

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

    if "ants_config" in kwargs:
        align_kwargs["default"] = align_kwargs.pop("ants_config").replace(".json", "")

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
    # debugging: generate aligned file for testing
    apply(xform, moving_path, reference=fixed_path).to_filename(
        dirname / f"dbg_{vol_idx:05d}.nii.gz"
    )

    return xform
