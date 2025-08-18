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

from nifreeze.data.dmri import DTI_MIN_ORIENTATIONS, DWI
from nifreeze.data.filtering import BVAL_ATOL, dwi_select_shells, grand_mean_normalization
from nifreeze.model.base import BaseModel, ExpectationModel

S0_EPSILON = 1e-6
B_MIN = 50


def _exec_fit(model, data, chunk=None, **kwargs):
    return model.fit(data, **kwargs), chunk


def _exec_predict(model, chunk=None, **kwargs):
    """Propagate model parameters and call predict."""
    return np.squeeze(model.predict(**kwargs)), chunk


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
        dataset : :obj:`~nifreeze.data.dmri.DWI`
            Reference to a DWI object.

        """

        # Duck typing, instead of explicitly testing for DWI type
        if not hasattr(dataset, "bzero"):
            raise TypeError("Dataset MUST be a DWI object.")

        if not hasattr(dataset, "gradients") or dataset.gradients is None:
            raise ValueError("Dataset MUST have a gradient table.")

        if len(dataset) < DTI_MIN_ORIENTATIONS:
            raise ValueError(
                f"DWI dataset is too small ({dataset.gradients.shape[0]} directions)."
            )

        if max_b is not None and max_b > B_MIN:
            self._max_b = max_b

        self._data_mask = (
            dataset.brainmask
            if dataset.brainmask is not None
            else np.ones(dataset.dataobj.shape[:3], dtype=bool)
        )

        # By default, set S0 to the 98% percentile of the DWI data within mask
        self._S0 = np.full(
            self._data_mask.sum(),
            np.round(np.percentile(dataset.dataobj[self._data_mask, ...], 98)),
        )

        # If b=0 is present and not to be ignored, update brain mask and set
        if not kwargs.pop("ignore_bzero", False) and dataset.bzero is not None:
            self._data_mask[dataset.bzero < S0_EPSILON] = False
            self._S0 = dataset.bzero[self._data_mask]

        super().__init__(dataset, **kwargs)

    def _fit(self, index: int | None = None, n_jobs: int | None = None, **kwargs) -> int:
        """Fit the model chunk-by-chunk asynchronously"""

        n_jobs = n_jobs or 1

        if self._locked_fit is not None:
            return n_jobs

        brainmask = self._dataset.brainmask
        idxmask = np.ones(len(self._dataset), dtype=bool)

        if index is not None:
            idxmask[index] = False
        else:
            self._locked_fit = True

        data, _, gtab = self._dataset[idxmask]
        # Select voxels within mask or just unravel 3D if no mask
        data = data[brainmask, ...] if brainmask is not None else data.reshape(-1, data.shape[-1])

        # DIPY models (or one with a fully-compliant interface)
        model_str = getattr(self, "_model_class", "")
        if "dipy" in model_str or "GeneralizedQSamplingModel" in model_str:
            gtab = gradient_table_from_bvals_bvecs(gtab[-1, :], gtab[:-1, :].T)

        if model_str:
            module_name, class_name = model_str.rsplit(".", 1)
            model = getattr(
                import_module(module_name),
                class_name,
            )(gtab, **kwargs)

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
            n_jobs=kwargs.pop("n_jobs"),
            **kwargs,
        )

        if index is None:
            return None

        gradient = self._dataset.gradients[:, index]

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
            for subprediction, index in results:
                predicted[index] = subprediction

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
        dataset : :obj:`~nifreeze.data.dmri.DWI`
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
