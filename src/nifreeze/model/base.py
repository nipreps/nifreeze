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
"""Base infrastructure for nifreeze's models."""

from abc import ABC, ABCMeta, abstractmethod
from warnings import warn

import numpy as np

MASK_ABSENCE_WARN_MSG = (
    "No mask provided; consider using a mask to avoid issues in model optimization."
)
"""Mask warning message."""
PREDICTED_MAP_ERROR_MSG = "This model requires the predicted map at initialization"
"""Oracle requirement error message."""
UNSUPPORTED_MODEL_ERROR_MSG = "Unsupported model <{model}>."
"""Unsupported model error message"""
SINGLE_FIT_CANARY_MSG = (
    "Single-fit for this model is a self-consistency check (canary): the held-out "
    "volume is also in the training set, so it is reproduced near-perfectly. It "
    "validates the implementation rather than being a genuine prediction."
)
"""Single-fit canary warning message."""


class SingleFitCanaryWarning(UserWarning):
    """Single-fit for this model only makes sense as a self-consistency canary."""


class ModelFactory:
    """A factory for instantiating data models."""

    @staticmethod
    def init(model: str | None = None, **kwargs):
        """
        Instantiate a diffusion model.

        Parameters
        ----------
        model : :obj:`str`
            Diffusion model.
            Options: ``"DTI"``, ``"DKI"``, ``"GQI"``, ``"GP"``, ``"S0"``,
            ``"AverageDWI"``. ``"GP"`` (aliases ``"GPR"``,
            ``"GaussianProcess"``) builds a
            :obj:`~nifreeze.model.dmri.GPModel`; pass ``kernel_model`` through
            ``kwargs`` to select the covariance.

        Return
        ------
        model : :obj:`~dipy.reconst.ReconstModel`
            A model object compliant with DIPY's interface.

        """
        if model is None:
            raise RuntimeError("No model identifier provided.")

        if model.lower() == "trivial":
            return TrivialModel(kwargs.pop("dataset"), **kwargs)

        if model.lower() in ("avg", "average", "mean"):
            return ExpectationModel(kwargs.pop("dataset"), **kwargs)

        if model.lower() in ("avgdwi", "averagedwi", "meandwi"):
            from nifreeze.model.dmri import AverageDWIModel

            return AverageDWIModel(kwargs.pop("dataset"), **kwargs)

        if model.lower() in ("gp", "gpr", "gaussianprocess"):
            from nifreeze.model.dmri import GPModel

            return GPModel(kwargs.pop("dataset"), **kwargs)

        if model.lower() in ("gqi", "dti", "dki", "pet"):
            from importlib import import_module

            thismod = import_module(
                f"nifreeze.model.{'pet' if model.lower() == 'pet' else 'dmri'}"
            )
            Model = getattr(thismod, f"{model.upper()}Model")
            return Model(kwargs.pop("dataset"), **kwargs)

        raise NotImplementedError(UNSUPPORTED_MODEL_ERROR_MSG.format(model=model))


class BaseModel(ABC):
    """
    Defines the interface and default methods.

    Implements the interface of :obj:`dipy.reconst.base.ReconstModel`.
    Instead of inheriting from the abstract base, this implementation
    follows type adaptation principles, as it is easier to maintain
    and to read (see https://www.youtube.com/watch?v=3MNVP9-hglc).

    """

    __metaclass__ = ABCMeta

    single_fit_is_canary: bool = False
    """Whether single-fit only makes sense as a self-consistency canary (see
    :meth:`_warn_single_fit_canary`)."""

    __slots__ = {
        "_dataset": "The NiFreeze dataset instance this model operates on",
        "_locked_fit": (
            "If resolves ``True`` as a boolean, indicates that the "
            "single-fit (locked) state is active. "
            "See :obj:`~nifreeze.model.base.BaseModel.fit_predict` "
            "for a description of the single-fit mode."
        ),
    }

    def __init__(self, dataset, **kwargs):
        """Base initialization."""

        self._locked_fit = None
        self._dataset = dataset
        # Warn if mask not present
        if dataset.brainmask is None:
            warn(MASK_ABSENCE_WARN_MSG, stacklevel=2)

    @abstractmethod
    def fit_predict(self, index: int | None = None, **kwargs) -> np.ndarray | None:
        """
        Fit and predict the indicated index of the dataset (abstract signature).

        In the default Leave-One-Volume-Out (LOVO) mode, ``index`` names the
        volume to hold out: the model is fit on every *other* volume and used to
        predict the held-out one. This held-out independence sets a base to
        claim that predictions are *unbiased*.

        If ``index`` is :obj:`None`, the model is executed in *single-fit mode*:
        it is fit **once** on all available data (no volume held out). The fit
        is *locked* (``_locked_fit`` evaluates to true), such that later calls
        reuse the fit object without refitting.

        Parameters
        ----------
        index : :obj:`int`, optional
            The volume index to hold out and predict (LOVO mode).
            If :obj:`None`, single-fit mode is used and no held-out prediction is
            produced.

        """
        return None

    def _warn_single_fit_canary(self) -> None:
        """Warn if single-fit for this model is only a self-consistency canary."""
        if self.single_fit_is_canary:
            warn(SINGLE_FIT_CANARY_MSG, SingleFitCanaryWarning, stacklevel=3)


class TrivialModel(BaseModel):
    """A trivial model that returns a given map always."""

    def __init__(self, dataset, predicted=None, **kwargs):
        """Implement object initialization."""

        super().__init__(dataset, **kwargs)
        self._locked_fit = (
            predicted
            if predicted is not None
            # Infer from dataset if not provided at initialization
            else getattr(dataset, "reference", getattr(dataset, "bzero", None))
        )

        if self._locked_fit is None:
            raise TypeError(PREDICTED_MAP_ERROR_MSG)

    def fit_predict(self, *_, **kwargs) -> np.ndarray | None:
        """Return the reference map."""

        # No need to check fit (if not fitted, has raised already)
        return self._locked_fit


class ExpectationModel(BaseModel):
    """A trivial model that returns an expectation map (for example, average)."""

    __slots__ = ("_stat",)

    def __init__(self, dataset, stat="median", **kwargs):
        """Initialize a new model."""
        super().__init__(dataset, **kwargs)
        self._stat = stat

    def fit_predict(self, index: int | None = None, **kwargs) -> np.ndarray:
        """
        Return the expectation map.

        Parameters
        ----------
        index : :obj:`int`
            The volume index that is left-out in fitting, and then predicted.

        """

        if self._locked_fit is not None:
            return self._locked_fit

        # Select the summary statistic
        avg_func = getattr(np, kwargs.pop("stat", self._stat))

        # Create index mask
        index_mask = np.ones(len(self._dataset), dtype=bool)

        if index is not None:
            index_mask[index] = False
            # Calculate the average
            return avg_func(self._dataset[index_mask][0], axis=-1)

        self._locked_fit = avg_func(self._dataset[index_mask][0], axis=-1)
        return self._locked_fit
