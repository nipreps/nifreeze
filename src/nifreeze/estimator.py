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

import os
from os import cpu_count
from pathlib import Path

from tempfile import NamedTemporaryFile, TemporaryDirectory
from timeit import default_timer as timer
from typing import TypeVar

import nitransforms as nt
import nibabel as nib
import numpy as np
from nipype.interfaces.ants.registration import Registration
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

            # Example: change model parameters only for DKI
            # if self._model.endswith("dki"):
            #     self._model_kwargs["fit_model"] = "CWLS"

            # Factory creates the appropriate model and pipes arguments
            model = ModelFactory.init(
                model=self._model,
                dataset=dataset,
                **self._model_kwargs,
            )
        else:
            model = self._model

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


class PETMotionEstimator:
    """Estimates motion within PET imaging data."""

    @staticmethod
    def estimate(
        pet_data,
        *,
        align_kwargs=None,
        omp_nthreads=None,
        n_jobs=None,
        seed=None,
        **kwargs,
    ):
        """
        Estimate motion parameters for PET data.

        Parameters
        ----------
        pet_data : :obj:`PET`
            The PET dataset to be processed.
        align_kwargs : :obj:`dict`
            Configuration parameters for the registration algorithm.
        omp_nthreads : :obj:`int`
            Maximum number of OpenMP threads to use.
        n_jobs : :obj:`int`
            Number of parallel jobs.
        seed : :obj:`int` or :obj:`bool`
            Seed for random number generation.
        """
        align_kwargs = align_kwargs or {}
        index_order = np.arange(
            len(pet_data))  # Assuming motion needs to be estimated across all frames

        if "num_threads" not in align_kwargs and omp_nthreads is not None:
            align_kwargs["num_threads"] = omp_nthreads

        affine_matrices = []
        for i in tqdm(index_order, desc="Estimating PET motion"):
            train_data, test_data = pet_data.lofo_split(i)

            # Save Nifti images to temporary files
            with NamedTemporaryFile(delete=False,
                                    suffix='.nii.gz') as fixed_file, \
                    NamedTemporaryFile(delete=False,
                                       suffix='.nii.gz') as moving_file:
                nib.Nifti1Image(train_data[0], pet_data.affine).to_filename(
                    fixed_file.name)
                nib.Nifti1Image(test_data[0], pet_data.affine).to_filename(
                    moving_file.name)

                # Setup registration
                registration = Registration(
                    from_file=Path(
                        '/Users/martinnorgaard/Documents/GitHub/eddymotion/src/eddymotion/config/pet-to-pet_level1.json'),
                    fixed_image=fixed_file.name,
                    moving_image=moving_file.name,
                    **align_kwargs
                )
                try:
                    result = registration.run()
                    if result.outputs.forward_transforms:
                        # Load the affine transformation matrix from the result
                        transform = nt.io.itk.ITKLinearTransform.from_filename(
                            result.outputs.forward_transforms[0])
                        matrix = transform.to_ras(reference=fixed_file.name,
                                                  moving=moving_file.name)
                        affine_matrices.append(matrix)
                    else:
                        print(f"No transforms produced for index {i}")
                except Exception as e:
                    print(f"Failed to process frame {i} due to {e}")

            # Clean up temporary files
            os.unlink(fixed_file.name)
            os.unlink(moving_file.name)

        return affine_matrices
