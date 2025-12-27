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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY kIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#

from importlib import import_module
from typing import Any, Union

import numpy as np
from dipy.core.gradients import gradient_table_from_bvals_bvecs
from joblib import Parallel, delayed

from nifreeze.data.dmri import DWI
from nifreeze.data.dmri.utils import DEFAULT_LOWB_THRESHOLD, DEFAULT_MIN_S0, DTI_MIN_ORIENTATIONS
from nifreeze.data.filtering import BVAL_ATOL, dwi_select_shells, grand_mean_normalization
from nifreeze.model.base import BaseModel, ExpectationModel

DEFAULT_S0_CLIP_PERCENTILE = 98
"""Upper percentile threshold for non-diffusion-weighted signal estimation."""
DWI_OBJECT_ERROR_MSG = "Dataset MUST be a DWI object."
"""dMRI object error message."""
DWI_GTAB_ERROR_MSG = "Dataset MUST have a gradient table."
"""dMRI gradient table error message."""
DWI_SIZE_ERROR_MSG = "DWI dataset is too small ({directions} directions)."
"""dMRI dataset size error message."""
DWI_DKI_NULL_GRADIENT_ERROR_MSG = """\
No 'bzero' found on dataset. DIPY's DKI requires a null gradient."""
"""dMRI dataset missing 'bzero' DKI model error message."""


def _exec_fit(model, data, chunk=None, **kwargs):
    return model.fit(data, **kwargs), chunk


def _exec_predict(model, chunk=None, **kwargs):
    """Propagate model parameters and call predict."""
    return np.squeeze(model.predict(**kwargs)), chunk


def _compute_data_mask(
    shape: tuple,
    brainmask: np.ndarray | None = None,
    bzero: np.ndarray | None = None,
    ignore_bzero: bool = False,
    default_min_S0: float = DEFAULT_MIN_S0,
) -> np.ndarray:
    """Compute the data mask for the given volume shape.

    If no ``brainmask`` is provided, a default mask (all :obj:`True`) is
    created. If ``b0`` data is not to be ignored, values where the ``b0`` is
    below ``default_min_S0`` are set to :obj:`False` in the mask.

    Parameters
    ----------
    shape : :obj:`tuple`
        DWI volume shape (3D).
    brainmask : :obj:`~numpy.ndarray`
        Brainmask.
    bzero : :obj:`~numpy.ndarray`
        ``b0`` data.
    ignore_bzero : :obj:`bool`, optional
        :obj:`True` to ignore ``b0``.
    default_min_S0 : :obj:`float`, optional
        Minimum S0 value to consider when ``b0`` data is not ignored.

    Returns
    -------
    :obj:`~numpy.ndarray`
        The computed data mask.
    """
    # Use the brainmask if available, else create a default mask
    data_mask = brainmask if brainmask is not None else np.ones(shape, dtype=bool)

    # Modify the mask if b0 is not ignored
    if not ignore_bzero and bzero is not None:
        data_mask[bzero < default_min_S0] = False

    return data_mask


def _compute_S0(
    dwi_dataobj: np.ndarray,
    data_mask: np.ndarray,
    bzero: np.ndarray | None = None,
    ignore_bzero: bool = False,
    default_percentile: int = DEFAULT_S0_CLIP_PERCENTILE,
) -> np.ndarray:
    """Compute the DWI ``S0`` value (non-diffusion-weighted signal).

    Generates an array of the same size as the number of voxels in the
    ``data_mask`` that are marked as ``True``. All values in the ``S0`` array
    are set to the rounded value of the specified percentile of the
    non-diffusion-weighted image data (``bzero``) within the region defined by
    the ``data_mask``.

    Parameters
    ----------
    dwi_dataobj : :obj:`~numpy.ndarray`
        DWI data.
    data_mask : :obj:`~numpy.ndarray`
        The mask to apply on the data.
    bzero : :obj:`~numpy.ndarray`
        ``b0`` data.
    ignore_bzero : :obj:`bool`, optional
        :obj:`True` to ignore ``b0``.
    default_percentile : :obj:`int`, optional
        Percentile threshold for S0 computation.

    Returns
    -------
    :obj:`~numpy.ndarray`
        The S0 signal array.
    """

    # By default, set S0 to the q-th percentile of the DWI data within mask
    S0 = np.full(
        data_mask.sum(),
        np.round(np.percentile(dwi_dataobj[data_mask, ...], default_percentile)),
    )

    # Modify S0 if b0 is not ignored
    if not ignore_bzero and bzero is not None:
        S0 = bzero[data_mask]

    return S0


class BaseDWIModel(BaseModel):
    """Interface and default methods for DWI models."""

    __slots__ = {
        "_max_b": "The maximum b-value supported by the model",
        "_data_mask": "A mask for the voxels that will be fitted and predicted",
        "_S0": "The S0 (b=0 reference signal) that will be fed into DIPY models",
        "_model_class": "Defining a model class, DIPY models are instantiated automagically",
        "_modelargs": "Arguments acceptable by the underlying DIPY-like model",
        "_models": "List with one or more (if parallel execution) model instances",
    }

    def __init__(self, dataset: DWI, max_b: float | int | None = None, **kwargs):
        """Initialization.

        Parameters
        ----------
        dataset : :obj:`~nifreeze.data.dmri.base.DWI`
            Reference to a DWI object.

        """

        # Duck typing, instead of explicitly testing for DWI type
        if not hasattr(dataset, "bzero"):
            raise TypeError(DWI_OBJECT_ERROR_MSG)

        if not hasattr(dataset, "gradients") or dataset.gradients is None:
            raise ValueError(DWI_GTAB_ERROR_MSG)

        if len(dataset) < DTI_MIN_ORIENTATIONS:
            raise ValueError(DWI_SIZE_ERROR_MSG.format(directions=dataset.gradients.shape[0]))

        if max_b is not None and max_b > DEFAULT_LOWB_THRESHOLD:
            self._max_b = max_b

        ignore_bzero = kwargs.pop("ignore_bzero", False)

        # Compute the data mask (ignores or uses b0 as specified)
        self._data_mask = _compute_data_mask(
            dataset.dataobj.shape[:3],
            dataset.brainmask,
            dataset.bzero,
            ignore_bzero=ignore_bzero,
        )

        # Compute S0 based on the mask and b0 ignore flag
        self._S0 = _compute_S0(
            dataset.dataobj, self._data_mask, dataset.bzero, ignore_bzero=ignore_bzero
        )

        super().__init__(dataset, **kwargs)

    def _fit(self, index: int | None = None, n_jobs: int | None = None, **kwargs) -> int:
        """Fit the model chunk-by-chunk asynchronously"""

        n_jobs = n_jobs or 1

        if self._locked_fit is not None:
            return n_jobs

        model_str = getattr(self, "_model_class", "")
        if "DiffusionKurtosisModel" in model_str:
            if self._dataset.bzero is None:
                raise ValueError(DWI_DKI_NULL_GRADIENT_ERROR_MSG)

        brainmask = self._dataset.brainmask
        idxmask = np.ones(len(self._dataset), dtype=bool)

        if index is not None:
            idxmask[index] = False
        else:
            self._locked_fit = True

        data, _, gtab = self._dataset[idxmask]

        # DIPY models (or one with a fully-compliant interface)
        if "dipy" in model_str or "GeneralizedQSamplingModel" in model_str:
            gtab = gradient_table_from_bvals_bvecs(gtab[:, -1], gtab[:, :-1])

        # Prepend the b0 to the gradients and the data for the kurtosis model
        if "DiffusionKurtosisModel" in model_str:
            # At this point, we are confident that the data contains a non-null
            # b0 attribute
            gtab = gradient_table_from_bvals_bvecs(
                np.concatenate([np.asarray([0]), gtab.bvals]),
                np.concatenate([np.zeros([1, 3]), gtab.bvecs]),
            )
            data = np.concatenate([self._dataset.bzero[..., np.newaxis], data], axis=-1)
            # ToDo
            # The index and idxmask no longer match to the gtab and data
            # lengths, but the index is no longer used

        # Select voxels within mask or just unravel 3D if no mask
        data = data[brainmask, ...] if brainmask is not None else data.reshape(-1, data.shape[-1])

        if model_str:
            module_name, class_name = model_str.rsplit(".", 1)
            model = getattr(
                import_module(module_name),
                class_name,
            )(gtab, **kwargs)
        else:
            raise NotImplementedError(f"{model_str} not implemented.")

        fit_kwargs: dict[str, Any] = {}  # Add here keyword arguments

        is_dki = model_str == "dipy.reconst.dki.DiffusionKurtosisModel"

        # One single CPU - linear execution (full model)
        # DKI model does not allow parallelization as implemented here
        if n_jobs == 1 or is_dki:
            _modelfit, _ = _exec_fit(model, data, **fit_kwargs)
            self._models = [_modelfit]
            return 1
        elif is_dki:
            _modelfit = model.multi_fit(data, **fit_kwargs)
            self._models = [_modelfit]
            return 1

        # Split data into chunks of group of slices
        data_chunks = np.array_split(data, n_jobs)

        self._models = [None] * n_jobs

        # Parallelize process with joblib
        with Parallel(n_jobs=n_jobs) as executor:
            results = executor(
                delayed(_exec_fit)(model, dchunk, i, **fit_kwargs)
                for i, dchunk in enumerate(data_chunks)
            )
        for submodel, rindex in results:
            self._models[rindex] = submodel

        return n_jobs

    def fit_predict(self, index: int | None = None, **kwargs) -> Union[np.ndarray, None]:
        """
        Predict asynchronously chunk-by-chunk the diffusion signal.

        Parameters
        ----------
        index : :obj:`int`
            The volume index that is left-out in fitting, and then predicted.

        """

        kwargs.pop("omp_nthreads", None)  # Drop omp_nthreads
        n_models = self._fit(
            index,
            n_jobs=kwargs.pop("n_jobs", None),
            **kwargs,
        )

        if index is None:
            return None

        gradient = self._dataset.gradients[index, :]

        model_str = getattr(self, "_model_class", "")
        if "dipy" in model_str or "GeneralizedQSamplingModel" in model_str:
            gradient = gradient_table_from_bvals_bvecs(
                gradient[np.newaxis, -1], gradient[np.newaxis, :-1]
            )

        if n_models == 1:
            predicted, _ = _exec_predict(
                self._models[0], **(kwargs | {"gtab": gradient, "S0": self._S0})
            )
        else:
            predicted = [None] * n_models
            S0 = np.array_split(self._S0, n_models)

            # Parallelize process with joblib
            with Parallel(n_jobs=n_models) as executor:
                results = executor(
                    delayed(_exec_predict)(
                        model,
                        chunk=i,
                        **(kwargs | {"gtab": gradient, "S0": S0[i]}),
                    )
                    for i, model in enumerate(self._models)
                )
            for subprediction, rindex in results:
                predicted[rindex] = subprediction

            predicted = np.hstack(predicted)

        retval = np.zeros_like(self._data_mask, dtype=self._dataset.dataobj.dtype)
        retval[self._data_mask, ...] = predicted
        return retval


class AverageDWIModel(ExpectationModel):
    """A trivial model that returns an average DWI volume."""

    __slots__ = ("_atol_low", "_atol_high", "_detrend")

    def __init__(
        self,
        dataset: DWI,
        stat: str = "median",
        atol_low: float = BVAL_ATOL,
        atol_high: float = BVAL_ATOL,
        detrend: bool = False,
        **kwargs,
    ):
        r"""
        Implement object initialization.

        Parameters
        ----------
        dataset : :obj:`~nifreeze.data.dmri.base.DWI`
            Reference to a DWI object.
        stat : :obj:`str`, optional
            Whether the summary statistic to apply is ``"mean"`` or ``"median"``.
        atol_low : :obj:`float`, optional
            A lower bound for the b-value corresponding to the diffusion weighted images
            that will be averaged.
        atol_low : :obj:`float`, optional
            An upper bound for the b-value corresponding to the diffusion weighted images
            that will be averaged.
        detrend : :obj:`bool`, optional
            Whether the overall distribution of each diffusion weighted image will be
            standardized and centered around the
            :data:`src.nifreeze.model.base.DEFAULT_CLIP_PERCENTILE` percentile.

        """
        super().__init__(dataset, stat=stat, **kwargs)

        self._atol_low = atol_low
        self._atol_high = atol_high
        self._detrend = detrend

    def fit_predict(self, index: int | None = None, *_, **kwargs) -> np.ndarray:
        """Return the average map."""

        if index is None:
            raise RuntimeError(f"Model {self.__class__.__name__} does not allow locking.")

        shellmask = dwi_select_shells(
            self._dataset.gradients,
            index,
            atol_low=self._atol_low,
            atol_high=self._atol_high,
        )

        shelldata = self._dataset.dataobj[..., shellmask]

        # Regress out global signal differences
        if self._detrend:
            shelldata = grand_mean_normalization(shelldata, mask=None)

        # Select the summary statistic
        avg_func = np.median if self._stat == "median" else np.mean

        # Calculate the average
        return avg_func(shelldata, axis=-1)


class DTIModel(BaseDWIModel):
    """A wrapper of :obj:`dipy.reconst.dti.TensorModel`."""

    _modelargs = (
        "min_signal",
        "return_S0_hat",
        "fit_method",
        "weighting",
        "sigma",
        "jac",
    )
    _model_class = "dipy.reconst.dti.TensorModel"


class DKIModel(BaseDWIModel):
    """A wrapper of :obj:`dipy.reconst.dki.DiffusionKurtosisModel`."""

    _modelargs = DTIModel._modelargs
    _model_class = "dipy.reconst.dki.DiffusionKurtosisModel"


class GQIModel(BaseDWIModel):
    """A wrapper of :obj:`dipy.reconst.gqi.GeneralizedQSamplingModel`."""

    _modelargs = (
        "method",
        "sampling_length",
        "normalize_peaks",
    )
    _model_class = "nifreeze.model.gqi.GeneralizedQSamplingModel"


class GPModel(BaseDWIModel):
    """A wrapper of :obj:`~nifreeze.model.dipy.GaussianProcessModel`."""

    _modelargs = ("kernel_model",)
    _model_class = "nifreeze.model._dipy.GaussianProcessModel"
