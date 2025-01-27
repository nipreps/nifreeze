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

from abc import abstractmethod
from warnings import warn

import numpy as np

mask_absence_warn_msg = (
    "No mask provided; consider using a mask to avoid issues in model optimization."
)


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
            Options: ``"DTI"``, ``"DKI"``, ``"S0"``, ``"AverageDWI"``

        Return
        ------
        model : :obj:`~dipy.reconst.ReconstModel`
            A model object compliant with DIPY's interface.

        """
        if model is None:
            raise RuntimeError("No model identifier provided.")

        if model.lower() in ("s0", "b0"):
            return TrivialModel(kwargs.pop("dataset"))

        if model.lower() in ("avg", "average", "mean"):
            return ExpectationModel(kwargs.pop("dataset"), **kwargs)

        if model.lower() in ("avgdwi", "averagedwi", "meandwi"):
            from nifreeze.model.dmri import AverageDWIModel

            return AverageDWIModel(kwargs.pop("dataset"), **kwargs)

        if model.lower() in ("dti", "dki", "pet"):
            Model = globals()[f"{model.upper()}Model"]
            return Model(kwargs.pop("dataset"), **kwargs)

        raise NotImplementedError(f"Unsupported model <{model}>.")


class BaseModel:
    """
    Defines the interface and default methods.

    Implements the interface of :obj:`dipy.reconst.base.ReconstModel`.
    Instead of inheriting from the abstract base, this implementation
    follows type adaptation principles, as it is easier to maintain
    and to read (see https://www.youtube.com/watch?v=3MNVP9-hglc).

    """

    __slots__ = ("_dataset",)

    def __init__(self, dataset, **kwargs):
        """Base initialization."""

        self._dataset = dataset
        # Warn if mask not present
        if dataset.brainmask is None:
            warn(mask_absence_warn_msg, stacklevel=2)

    @abstractmethod
    def fit_predict(self, index, **kwargs) -> np.ndarray:
        """Fit and predict the indicate index of the dataset (abstract signature)."""
        raise NotImplementedError("Cannot call fit_predict() on a BaseModel instance.")


class TrivialModel(BaseModel):
    """A trivial model that returns a given map always."""

    __slots__ = ("_predicted",)

    def __init__(self, dataset, predicted=None, **kwargs):
        """Implement object initialization."""

        super().__init__(dataset, **kwargs)
        self._predicted = (
            predicted
            if predicted is not None
            # Infer from dataset if not provided at initialization
            else getattr(dataset, "reference", getattr(dataset, "bzero", None))
        )

        if self._predicted is None:
            raise TypeError("This model requires the predicted map at initialization")

    def fit_predict(self, *_, **kwargs):
        """Return the reference map."""

        # No need to check fit (if not fitted, has raised already)
        return self._predicted


class ExpectationModel(BaseModel):
    """A trivial model that returns an expectation map (for example, average)."""

    __slots__ = ("_stat",)

    def __init__(self, dataset, stat="median", **kwargs):
        """Initialize a new model."""
        super().__init__(dataset, **kwargs)
        self._stat = stat

    def fit_predict(self, index: int, **kwargs):
        """
        Return the expectation map.

        Parameters
        ----------
        index : :obj:`int`
            The volume index that is left-out in fitting, and then predicted.

        """
        # Select the summary statistic
        avg_func = getattr(np, kwargs.pop("stat", self._stat))

        # Create index mask
        mask = np.ones(len(self._dataset), dtype=bool)
        mask[index] = False

        # Calculate the average
        return avg_func(self._dataset.dataobj[mask][0], axis=-1)
