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
"""Orchestrates model and registration in volume-to-volume artifact estimation."""

from __future__ import annotations

from os import cpu_count
from pathlib import Path
from timeit import default_timer as timer
from typing import TypeVar

import numpy as np
from tqdm import tqdm
from typing_extensions import Self

from nifreeze.data.base import ESTIMATES_FIELD_ERROR_MSG, BaseDataset
from nifreeze.model.base import BaseModel, ModelFactory, TrivialModel
from nifreeze.registration.base import Registration, RegistrationFactory
from nifreeze.utils import iterators

DatasetT = TypeVar("DatasetT", bound=BaseDataset)

DEFAULT_CHUNK_SIZE: int = int(1e6)
FIT_MSG = "Fit&predict"
REG_MSG = "Realign"


class Filter:
    """Alters an input data object (e.g., downsampling)."""

    def run(self, dataset: DatasetT, **kwargs) -> DatasetT:
        """
        Trigger execution of the designated filter.

        Parameters
        ----------
        dataset : :obj:`~nifreeze.data.base.BaseDataset`
            The input dataset this estimator operates on.

        Returns
        -------
        dataset : :obj:`~nifreeze.data.base.BaseDataset`
            The dataset, after filtering.

        """
        return dataset


class Estimator:
    """
    Orchestrates components for a single estimation step.

    The estimator traverses the 4D sequence with a Leave-One-Volume-Out (LOVO)
    strategy: for each target volume it fits the model on the *remaining*
    volumes, predicts the held-out volume, and registers the observed volume to
    that prediction. Held-out independence is what makes the estimated transform
    valid (see :meth:`~nifreeze.model.base.BaseModel.fit_predict`).

    Parameters
    ----------
    model : :obj:`~nifreeze.model.base.BaseModel` or :obj:`str`
        The model (or model name) used to generate the registration target.
    strategy : :obj:`str`, optional
        Name of the iterator used to traverse the 4D sequence.
    prev : :obj:`~nifreeze.estimator.Estimator`, :obj:`~nifreeze.estimator.Filter`, optional
        A previous estimator/filter whose output initializes this one (cascade).
    model_kwargs : :obj:`dict`, optional
        Extra keyword arguments passed through to the model at construction.
    single_fit : :obj:`bool`, optional
        Run the model in *single-fit mode*: fit **once** on all volumes up front
        (via ``model.fit_predict(None)``) and lock that *fit* instead of
        refitting per held-out volume. Prediction is no longer unbiased.
    start_index : :obj:`int`, optional
        First volume index to process.
    stop_index : :obj:`int`, optional
        One-past-last volume index to process (``None`` = to the end).
    registration : :obj:`~nifreeze.registration.base.Registration` or :obj:`str`, optional
        The registration backend, or the name of one
        (:data:`~nifreeze.registration.base.BACKENDS`), built for each run from the
        registration settings (default: elastix where ``itk-elastix`` is installed,
        ANTs otherwise; see :func:`~nifreeze.registration.base.default_backend`).
    estimates : :obj:`str`, optional
        The field of the dataset this estimator's estimates go to, one of its
        :attr:`~nifreeze.data.base.BaseDataset.ESTIMATES`: ``"motion"`` (default), or
        ``"eddy"`` for the eddy currents of a DWI (as ``--eddy`` does). Each volume is
        registered from its mapping, every estimate composed
        (:meth:`~nifreeze.data.base.BaseDataset.get_transform`), and the other fields'
        estimates are factored out of the result
        (:meth:`~nifreeze.data.base.BaseDataset.set_transform`).
    **kwargs
        Registration settings of this estimator, which take precedence over the run's
        ``align_kwargs``: ``align_config`` (a mapping, or a file with a section per
        backend, such as :data:`~nifreeze.registration.base.RIGID`, the default, and
        :data:`~nifreeze.registration.base.AFFINE`), and the options of
        :class:`~nifreeze.registration.base.Registration`.

    """

    __slots__ = (
        "_model",
        "_single_fit",
        "_strategy",
        "_prev",
        "_model_kwargs",
        "_align_kwargs",
        "_start_index",
        "_stop_index",
        "_registration",
        "_estimates",
    )

    def __init__(
        self,
        model: BaseModel | str,
        strategy: str = "random",
        prev: Estimator | Filter | None = None,
        model_kwargs: dict | None = None,
        single_fit: bool = False,
        start_index: int = 0,
        stop_index: int | None = None,
        registration: Registration | str | None = None,
        estimates: str = "motion",
        **kwargs,
    ):
        self._model = model
        self._prev = prev
        self._strategy = strategy
        self._single_fit = single_fit
        self._model_kwargs = model_kwargs or {}
        self._align_kwargs = kwargs or {}

        self._start_index = start_index
        self._stop_index = stop_index
        self._registration = registration
        self._estimates = estimates

    def run(self, dataset: DatasetT, **kwargs) -> Self:
        """
        Trigger execution of the workflow this estimator belongs.

        Parameters
        ----------
        dataset : :obj:`~nifreeze.data.base.BaseDataset`
            The input dataset this estimator operates on.
        align_kwargs : :obj:`dict`, optional
            Registration settings for every estimator of the chain (e.g.,
            ``{"align_config": <the file of --align-config>}``). The settings given to an
            estimator at construction take precedence.
        clip : :obj:`str`, optional
            Clip the intensity of ``"sample"``, ``"predicted"``, ``"both"``
            (default) or ``"none"`` of the images to the sample's window before
            registration.
        debug_dir : :obj:`os.pathlike`, optional
            Keep, in a folder of ``debug_dir`` per estimator of the chain, the images
            each volume was registered with and an ``aligned_<index>.nii`` trace: the
            sample with its affine corrected by the estimate, which a viewer shows on
            top of ``predicted_<index>.nii``.
        instant_update : :obj:`bool`, optional
            Whether models train on the realigned view of the dataset (default):
            right after a volume is registered, the model sees it resampled once,
            from the original volume, with the estimate (and, for DWI, its rotated
            b-vector). The dataset itself is never modified; registration and
            export keep using the original volumes. It applies to models given by
            name and to model instances built on this dataset (see
            :meth:`~nifreeze.model.base.BaseModel.set_dataset`), except the
            trivial model, whose reference never moves.
        num_threads : :obj:`int`, optional
            Threads of each registration (default: ``omp_nthreads``).
        realigned : :obj:`dict`, optional
            Realigned views by dataset, shared by the estimators of a chain.

        Returns
        -------
        :obj:`~nifreeze.estimator.Estimator`
            The estimator, after fitting.

        """
        # Before the previous estimators of the chain run
        if self._estimates not in dataset.ESTIMATES:
            raise ValueError(
                ESTIMATES_FIELD_ERROR_MSG.format(
                    name=type(dataset).__name__,
                    field=self._estimates,
                    fields=", ".join(dataset.ESTIMATES),
                )
            )

        # Shared with the previous estimators of the chain
        views = kwargs.setdefault("realigned", {})
        if self._prev is not None:
            result = self._prev.run(dataset, **kwargs)
            if isinstance(self._prev, Filter):
                dataset = result  # type: ignore[assignment]

        n_jobs = kwargs.pop("n_jobs", None) or min(cpu_count() or 1, 8)
        n_threads = kwargs.pop("omp_nthreads", None) or ((cpu_count() or 2) - 1)

        # The realigned view the model trains on (one per dataset in a chain)
        view = None
        if kwargs.pop("instant_update", True) and not (
            # The trivial model's reference never moves
            isinstance(self._model, TrivialModel)
            or (isinstance(self._model, str) and self._model.lower() == "trivial")
            # A model instance built on another dataset keeps reading it
            or (isinstance(self._model, BaseModel) and self._model._dataset is not dataset)
        ):
            view = views.get(id(dataset))
            if view is None:
                view = views[id(dataset)] = dataset.realigned()

        num_voxels = dataset.brainmask.sum() if dataset.brainmask is not None else dataset.size3d
        chunk_size = DEFAULT_CHUNK_SIZE * (n_threads or 1)

        # Calculate the size parameter for the iterator (exclusive upper bound)
        n = len(dataset)
        size = n
        start_index = self._start_index or 0
        stop_index = (
            None
            if self._stop_index is None
            else (n + self._stop_index if self._stop_index < 0 else self._stop_index)
        )

        # Prepare iterator
        iterfunc = getattr(iterators, f"{self._strategy}_iterator")
        index_iter = iterfunc(
            size=size,
            bvals=kwargs.pop("bvals", None),
            uptake=kwargs.pop("uptake", None),
            seed=kwargs.get("seed", None),
            round_decimals=kwargs.pop("round_decimals", iterators.DEFAULT_ROUND_DECIMALS),
            start_index=start_index,
            stop_index=stop_index,
        )

        # Initialize model
        if isinstance(self._model, str):
            if self._model.endswith("dti"):
                self._model_kwargs["step"] = chunk_size

            # Factory creates the appropriate model and pipes arguments
            model = ModelFactory.init(
                model=self._model,
                dataset=dataset if view is None else view,
                **self._model_kwargs,
            )
        else:
            model = self._model
            if view is not None:
                model.set_dataset(view)

        # Prepare fit/predict keyword arguments
        fit_pred_kwargs = {
            "n_jobs": n_jobs,
            "omp_nthreads": n_threads,
        }
        if model.__class__.__name__ == "DTIModel":
            fit_pred_kwargs["step"] = chunk_size

        print(f"Dataset size: {num_voxels}x{len(dataset)}.")
        print(f"Parallel execution: {fit_pred_kwargs}.")
        print(f"Model: {model}.")

        if self._single_fit:
            print("Fitting 'single' model started ...")
            start = timer()
            model.fit_predict(None, **fit_pred_kwargs)
            print(f"Fitting 'single' model finished, elapsed {timer() - start}s.")

        # The backend given, or one built from the registration settings: the run's
        # (shared by the chain), this estimator's, and then the run's other arguments
        registration = self._registration
        if not isinstance(registration, Registration):
            options = (kwargs.pop("align_kwargs", None) or {}) | self._align_kwargs | kwargs
            options.pop("realigned", None)
            debug_dir = options.pop("debug_dir", None)
            options |= {
                "num_threads": options.pop("num_threads", n_threads),
                "debug_dir": None
                if debug_dir is None
                else Path(debug_dir) / self._debug_name(model),
            }
            # Unset options keep each backend's default
            options = {key: value for key, value in options.items() if value is not None}
            registration = RegistrationFactory.init(registration, **options)
        print(f"Registration: {type(registration).__name__}.")

        # Calculate effective dataset length for progress bar
        if self._stop_index is not None:
            dataset_length = self._stop_index - self._start_index
        else:
            dataset_length = len(dataset) - self._start_index

        with tqdm(total=dataset_length, unit="vols.") as pbar:
            # run an original-to-synthetic affine registration
            for i in index_iter:
                pbar.set_description_str(f"{FIT_MSG: <16} vol. <{i}>")

                # fit the model
                predicted = model.fit_predict(  # type: ignore[union-attr]
                    i,
                    **fit_pred_kwargs,
                )

                # Start from the current mapping, every estimate composed (left by a
                # previous estimator of the chain, or stored in the dataset)
                init_affine = None
                current = dataset.get_transform(i)
                if current is not None:
                    # A copy: a backend may keep it, and the dataset's changes later
                    current = np.array(current, dtype=float)
                    if not np.allclose(current, np.eye(4)):
                        init_affine = current

                pbar.set_description_str(f"{REG_MSG: <16} vol. <{i}>")
                xform = registration(
                    predicted, dataset[i][0], dataset.affine, dataset.brainmask, i, init_affine
                )

                # update
                dataset.set_transform(i, xform.matrix, self._estimates)
                if view is not None:
                    # Instant update: the next predictions see this volume realigned
                    dataset.realign_into(view, i)
                    model.update(i)
                pbar.update()

        return self

    def _debug_name(self, model: BaseModel) -> str:
        """Name this estimator's folder of ``debug_dir`` by position in the chain and model."""
        position, prev = 0, self._prev
        while prev is not None:
            position += isinstance(prev, Estimator)
            prev = getattr(prev, "_prev", None)
        name = self._model if isinstance(self._model, str) else type(model).__name__
        return f"{position:02d}-{name.lower()}"
