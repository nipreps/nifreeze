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

from importlib.resources import files
from os import cpu_count
from pathlib import Path
from tempfile import TemporaryDirectory
from timeit import default_timer as timer
from typing import TypeVar

import nibabel as nb
import nitransforms as nt
import numpy as np
from tqdm import tqdm
from typing_extensions import Self

from nifreeze.data.base import BaseDataset
from nifreeze.data.pet import PET
from nifreeze.model.base import BaseModel, ModelFactory
from nifreeze.model.pet import PETModel
from nifreeze.registration.ants import (
    Registration,
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
        index_iter = iterfunc(size=len(dataset), seed=kwargs.get("seed", None))

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
                bmask_path = ptmp_dir / "brainmask.nii.gz"
                nb.Nifti1Image(
                    dataset.brainmask.astype(np.uint8), dataset.affine, None
                ).to_filename(bmask_path)

            with tqdm(total=dataset_length, unit="vols.") as pbar:
                # run an original-to-synthetic affine registration
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
    """Estimates motion within PET imaging data aligned with generic Estimator workflow."""

    def __init__(self, align_kwargs: dict | None = None, strategy: str = "lofo"):
        self.align_kwargs = align_kwargs or {}
        self.strategy = strategy

    def run(self, pet_dataset: PET, omp_nthreads: int | None = None) -> list:
        n_frames = len(pet_dataset)
        frame_indices = np.arange(n_frames)

        if omp_nthreads:
            self.align_kwargs["num_threads"] = omp_nthreads

        affine_matrices = []

        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            for idx in tqdm(frame_indices, desc="Estimating PET motion"):
                (train_data, train_times), (test_data, test_time) = pet_dataset.lofo_split(idx)

                if train_times is None:
                    raise ValueError(
                        f"train_times is None at index {idx}, check midframe initialization."
                    )

                # Build a temporary dataset excluding the test frame
                train_dataset = PET(
                    dataobj=train_data,
                    affine=pet_dataset.affine,
                    brainmask=pet_dataset.brainmask,
                    midframe=train_times,
                    total_duration=pet_dataset.total_duration,
                )

                # Instantiate PETModel explicitly
                model = PETModel(
                    dataset=train_dataset,
                    timepoints=train_times,
                    xlim=pet_dataset.total_duration,
                )

                # Fit the model once on the training dataset
                model.fit_predict(None)

                # Predict the reference volume at the test frame's timepoint
                predicted = model.fit_predict(test_time)

                fixed_image_path = tmp_path / f"fixed_frame_{idx:03d}.nii.gz"
                moving_image_path = tmp_path / f"moving_frame_{idx:03d}.nii.gz"

                fixed_img = nb.Nifti1Image(predicted, pet_dataset.affine)
                moving_img = nb.Nifti1Image(test_data, pet_dataset.affine)

                moving_img = nb.as_closest_canonical(moving_img, enforce_diag=True)

                fixed_img.to_filename(fixed_image_path)
                moving_img.to_filename(moving_image_path)

                registration_config = files("nifreeze.registration.config").joinpath(
                    "pet-to-pet_level1.json"
                )

                registration = Registration(
                    from_file=registration_config,
                    fixed_image=str(fixed_image_path),
                    moving_image=str(moving_image_path),
                    output_warped_image=True,
                    output_transform_prefix=f"ants_{idx:03d}",
                    **self.align_kwargs,
                )

                try:
                    result = registration.run(cwd=str(tmp_path))
                    if result.outputs.forward_transforms:
                        transform = nt.io.itk.ITKLinearTransform.from_filename(
                            result.outputs.forward_transforms[0]
                        )
                        matrix = transform.to_ras(
                            reference=str(fixed_image_path), moving=str(moving_image_path)
                        )
                        affine_matrices.append(matrix)
                    else:
                        affine_matrices.append(np.eye(4))
                        print(f"No transforms produced for index {idx}")
                except Exception as e:
                    affine_matrices.append(np.eye(4))
                    print(f"Failed to process frame {idx} due to {e}")

        return affine_matrices
