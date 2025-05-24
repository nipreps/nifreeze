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

import numpy as np
from dipy.core.gradients import gradient_table_from_bvals_bvecs

from nifreeze.data.dmri import (
    DEFAULT_CLIP_PERCENTILE,
    DTI_MIN_ORIENTATIONS,
    DWI,
)
from nifreeze.model.base import BaseModel, ExpectationModel

S0_EPSILON = 1e-6
B_MIN = 50


class BaseDWIModel(BaseModel):
    """Interface and default methods for DWI models."""

    __slots__ = {
        "_max_b": "The maximum b-value supported by the model",
        "_data_mask": "A mask for the voxels that will be fitted and predicted",
        "_S0": "The S0 (b=0 reference signal) that will be fed into DIPY models",
        "_model_class": "Defining a model class, DIPY models are instantiated automagically",
        "_modelargs": "Arguments acceptable by the underlying DIPY-like model.",
        "_model_fit": "Fitted model",
    }

    def __init__(self, dataset: DWI, max_b: float | int | None = None, **kwargs):
        r"""Initialization.

        Parameters
        ----------
        dataset : :obj:`~nifreeze.data.dmri.DWI`
            Reference to a DWI object.

        """

        # Duck typing, instead of explicitly test for DWI type
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

    def _fit(self, index: int | None = None, n_jobs=None, **kwargs):
        """Fit the model chunk-by-chunk asynchronously"""

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
        if "dipy" in model_str:
            gtab = gradient_table_from_bvals_bvecs(gtab[-1, :], gtab[:-1, :].T)

        if model_str:
            module_name, class_name = model_str.rsplit(".", 1)
            model = getattr(
                import_module(module_name),
                class_name,
            )(gtab, **kwargs)

        self._model_fit = model.fit(
            data,
            engine="serial" if n_jobs == 1 else "joblib",
            n_jobs=n_jobs,
        )
        return n_jobs

    def fit_predict(self, index: int | None = None, **kwargs):
        """
        Predict asynchronously chunk-by-chunk the diffusion signal.

        Parameters
        ----------
        index : :obj:`int`
            The volume index that is left-out in fitting, and then predicted.

        """

        self._fit(
            index,
            n_jobs=kwargs.pop("n_jobs"),
            **kwargs,
        )

        if index is None:
            self._locked_fit = True
            return None

        gradient = self._dataset.gradients[:, index]

        if "dipy" in getattr(self, "_model_class", ""):
            gradient = gradient_table_from_bvals_bvecs(
                gradient[np.newaxis, -1], gradient[np.newaxis, :-1]
            )

        predicted = np.squeeze(
            self._model_fit.predict(
                gtab=gradient,
                S0=self._S0,
            )
        )

        retval = np.zeros_like(self._data_mask, dtype=self._dataset.dataobj.dtype)
        retval[self._data_mask, ...] = predicted
        return retval


class AverageDWIModel(ExpectationModel):
    """A trivial model that returns an average DWI volume."""

    __slots__ = ("_th_low", "_th_high", "_detrend")

    def __init__(
        self,
        dataset: DWI,
        stat: str = "median",
        th_low: float = 100.0,
        th_high: float = 100.0,
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
        th_low : :obj:`float`, optional
            A lower bound for the b-value corresponding to the diffusion weighted images
            that will be averaged.
        th_high : :obj:`float`, optional
            An upper bound for the b-value corresponding to the diffusion weighted images
            that will be averaged.
        detrend : :obj:`bool`, optional
            Whether the overall distribution of each diffusion weighted image will be
            standardized and centered around the
            :data:`src.nifreeze.model.base.DEFAULT_CLIP_PERCENTILE` percentile.

        """
        super().__init__(dataset, stat=stat, **kwargs)

        self._th_low = th_low
        self._th_high = th_high
        self._detrend = detrend

    def fit_predict(self, index: int | None = None, *_, **kwargs):
        """Return the average map."""

        if index is None:
            raise RuntimeError(f"Model {self.__class__.__name__} does not allow locking.")

        bvalues = self._dataset.gradients[:, -1]
        bcenter = bvalues[index]

        shellmask = np.ones(len(self._dataset), dtype=bool)

        # Keep only bvalues within the range defined by th_high and th_low
        shellmask[index] = False
        shellmask[bvalues > (bcenter + self._th_high)] = False
        shellmask[bvalues < (bcenter - self._th_low)] = False

        if not shellmask.sum():
            raise RuntimeError(f"Shell corresponding to index {index} (b={bcenter}) is empty.")

        shelldata = self._dataset.dataobj[..., shellmask]

        # Regress out global signal differences
        if self._detrend:
            centers = np.median(shelldata, axis=(0, 1, 2))
            reference = np.percentile(centers[centers >= 1.0], DEFAULT_CLIP_PERCENTILE)
            centers[centers < 1.0] = reference
            drift = reference / centers
            shelldata = shelldata * drift

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


class GPModel(BaseDWIModel):
    """A wrapper of :obj:`~nifreeze.model.dipy.GaussianProcessModel`."""

    _modelargs = ("kernel_model",)
    _model_class = "nifreeze.model._dipy.GaussianProcessModel"
