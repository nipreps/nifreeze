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
from joblib import Parallel, delayed

from nifreeze.data.dmri import (
    DEFAULT_CLIP_PERCENTILE,
    DTI_MIN_ORIENTATIONS,
    DWI,
)
from nifreeze.model.base import BaseModel, ExpectationModel


def _exec_fit(model, data, chunk=None):
    retval = model.fit(data)
    return retval, chunk


def _exec_predict(model, chunk=None, **kwargs):
    """Propagate model parameters and call predict."""
    return np.squeeze(model.predict(**kwargs)), chunk


class BaseDWIModel(BaseModel):
    """Interface and default methods for DWI models."""

    __slots__ = {
        "_model_class": "Defining a model class, DIPY models are instantiated automagically",
        "_modelargs": "Arguments acceptable by the underlying DIPY-like model.",
    }

    def __init__(self, dataset: DWI, **kwargs):
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

        if dataset.gradients.shape[0] < DTI_MIN_ORIENTATIONS:
            raise ValueError(
                f"DWI dataset is too small ({dataset.gradients.shape[0]} directions)."
            )

        super().__init__(dataset, **kwargs)

    def _fit(self, index, n_jobs=None, **kwargs):
        """Fit the model chunk-by-chunk asynchronously"""
        n_jobs = n_jobs or 1

        brainmask = self._dataset.brainmask
        idxmask = np.ones(len(self._dataset), dtype=bool)
        idxmask[index] = False

        data, _, gtab = self._dataset[idxmask]
        # Select voxels within mask or just unravel 3D if no mask
        data = data[brainmask, ...] if brainmask is not None else data.reshape(-1, data.shape[-1])

        # DIPY models (or one with a fully-compliant interface)
        model_str = getattr(self, "_model_class", None)
        if model_str:
            module_name, class_name = model_str.rsplit(".", 1)
            self._model = getattr(
                import_module(module_name),
                class_name,
            )(gtab, **kwargs)

        # One single CPU - linear execution (full model)
        if n_jobs == 1:
            self._model, _ = _exec_fit(self._model, data)
            return

        # Split data into chunks of group of slices
        data_chunks = np.array_split(data, n_jobs)

        self._models = [None] * n_jobs

        # Parallelize process with joblib
        with Parallel(n_jobs=n_jobs) as executor:
            results = executor(
                delayed(_exec_fit)(self._model, dchunk, i) for i, dchunk in enumerate(data_chunks)
            )
        for submodel, rindex in results:
            self._models[rindex] = submodel

        self._model = None  # Preempt further actions on the model
        return n_jobs

    def fit_predict(self, index: int, **kwargs):
        """
        Predict asynchronously chunk-by-chunk the diffusion signal.

        Parameters
        ----------
        index : :obj:`int`
            The volume index that is left-out in fitting, and then predicted.

        """

        n_models = self._fit(index, **kwargs)

        brainmask = self._dataset.brainmask
        gradient = self._dataset.gradients[index]

        S0 = self._dataset.bzero
        if S0 is not None:
            S0 = S0[brainmask, ...] if brainmask is not None else S0.reshape(-1)

        if n_models == 1:
            predicted, _ = _exec_predict(self._model, **(kwargs | {"gtab": gradient, "S0": S0}))
        else:
            S0 = np.array_split(S0, n_models) if S0 is not None else np.full(n_models, None)

            predicted = [None] * n_models

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

        if brainmask is not None:
            retval = np.zeros_like(brainmask, dtype="float32")
            retval[brainmask, ...] = predicted
        else:
            retval = predicted.reshape(self._dataset.shape[:-1])

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

    def fit_predict(self, index, *_, **kwargs):
        """Return the average map."""

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
