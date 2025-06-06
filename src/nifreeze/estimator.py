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
from tempfile import TemporaryDirectory
from timeit import default_timer as timer
from typing import TypeVar

from tqdm import tqdm
from typing_extensions import Self

from nifreeze.data.base import BaseDataset
from nifreeze.model.base import BaseModel, ModelFactory
from nifreeze.registration.ants import (
    _prepare_registration_data,
    _run_registration,
)
from nifreeze.utils import iterators

DatasetT = TypeVar("DatasetT", bound=BaseDataset)

DEFAULT_CHUNK_SIZE: int = int(1e6)
FIT_MSG = "Fit&predict"
PRE_MSG = "Predicted"
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
    """Orchestrates components for a single estimation step."""

    __slots__ = ("_model", "_single_fit", "_strategy", "_prev", "_model_kwargs", "_align_kwargs")

    def __init__(
        self,
        model: BaseModel | str,
        strategy: str = "random",
        prev: Estimator | Filter | None = None,
        model_kwargs: dict | None = None,
        single_fit: bool = False,
        **kwargs,
    ):
        self._model = model
        self._prev = prev
        self._strategy = strategy
        self._single_fit = single_fit
        self._model_kwargs = model_kwargs or {}
        self._align_kwargs = kwargs or {}

    def run(self, dataset: DatasetT, **kwargs) -> Self:
        """
        Trigger execution of the workflow this estimator belongs.

        Parameters
        ----------
        dataset : :obj:`~nifreeze.data.base.BaseDataset`
            The input dataset this estimator operates on.

        Returns
        -------
        :obj:`~nifreeze.estimator.Estimator`
            The estimator, after fitting.

        """
        if self._prev is not None:
            result = self._prev.run(dataset, **kwargs)
            if isinstance(self._prev, Filter):
                dataset = result  # type: ignore[assignment]

        n_jobs = kwargs.pop("n_jobs", None) or min(cpu_count() or 1, 8)
        n_threads = kwargs.pop("omp_nthreads", None) or ((cpu_count() or 2) - 1)

        num_voxels = dataset.brainmask.sum() if dataset.brainmask is not None else dataset.size3d
        chunk_size = DEFAULT_CHUNK_SIZE * (n_threads or 1)

        # Prepare iterator
        iterfunc = getattr(iterators, f"{self._strategy}_iterator")
        index_iter = iterfunc(len(dataset), seed=kwargs.get("seed", None))

        # Initialize model
        if isinstance(self._model, str):
            if self._model.endswith("dti"):
                self._model_kwargs["step"] = chunk_size

            # Factory creates the appropriate model and pipes arguments
            model = ModelFactory.init(
                model=self._model,
                dataset=dataset,
                **self._model_kwargs,
            )
        else:
            model = self._model

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

        kwargs["num_threads"] = n_threads
        kwargs = self._align_kwargs | kwargs

        dataset_length = len(dataset)
        with TemporaryDirectory() as tmp_dir:
            print(f"Processing in <{tmp_dir}>")
            ptmp_dir = Path(tmp_dir)

            bmask_path = None
            if dataset.brainmask is not None:
                import nibabel as nb

                bmask_path = ptmp_dir / "brainmask.nii.gz"
                nb.Nifti1Image(
                    dataset.brainmask.astype("uint8"), dataset.affine, None
                ).to_filename(bmask_path)

            with tqdm(total=dataset_length, unit="vols.") as pbar:
                # run a original-to-synthetic affine registration
                for i in index_iter:
                    pbar.set_description_str(f"{FIT_MSG: <16} vol. <{i}>")

                    # fit the model
                    predicted = model.fit_predict(  # type: ignore[union-attr]
                        i,
                        **fit_pred_kwargs,
                    )

                    pbar.set_description_str(f"{PRE_MSG: <16} vol. <{i}>")

                    # prepare data for running ANTs
                    predicted_path, volume_path, init_path = _prepare_registration_data(
                        dataset[i][0],  # Access the target volume
                        predicted,
                        dataset.affine,
                        i,
                        ptmp_dir,
                        kwargs.pop("clip", "both"),
                    )

                    pbar.set_description_str(f"{REG_MSG: <16} vol. <{i}>")

                    xform = _run_registration(
                        predicted_path,
                        volume_path,
                        i,
                        ptmp_dir,
                        init_affine=init_path,
                        fixedmask_path=bmask_path,
                        output_transform_prefix=f"ants-{i:05d}",
                        **kwargs,
                    )

                    # update
                    dataset.set_transform(i, xform.matrix)
                    pbar.update()

        return self
