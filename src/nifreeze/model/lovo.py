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
r"""Closed-form leave-one-volume-out (LOVO) prediction for kernel models.

A kernel predictor with fixed hyperparameters is linear in the data, and its
weights depend only on the gradient table. Given the covariance
:math:`\mathbf{A}` of the :math:`N` diffusion-weighted volumes, the prediction
of volume :math:`i` from all the others is
(:footcite:t:`rasmussen_gaussian_2006`, Eq. 5.12)

.. math::
    \hat{y}_i = -\sum_{j \neq i} \frac{P_{ij}}{P_{ii}} y_j,
    \quad \mathbf{P} = \mathbf{A}^{-1},

so all :math:`N` LOVO predictions are given by a single smoother matrix
:math:`\mathbf{S} = \mathbf{I} - \operatorname{diag}(\mathbf{P})^{-1}\mathbf{P}`,
with a zero diagonal. One inverse replaces :math:`N` model fits.

References
----------
.. footbibliography::

"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence
from warnings import warn

import numpy as np
from scipy.linalg import LinAlgError, cho_factor, cho_solve
from scipy.optimize import minimize
from sklearn.gaussian_process.kernels import Kernel

from nifreeze.model._dipy import DEFAULT_NVOX_PREFIT, GP_JITTER
from nifreeze.model.gpr import LBFGS_DEFAULT_GTOL, DiffusionGPR

DEFAULT_KRR_LAMBDA_GRID = tuple(np.logspace(-4, 1, 11))
"""Default grid of ridge strengths (relative to the mean Gram diagonal)."""

HYPERPARAMETERS_SCHEMA = "nifreeze-gp-hyperparameters/1"
"""Schema identifier written to (and required from) hyperparameter files."""

ILL_CONDITIONED_THRESHOLD = 1e12
"""Condition number above which the LOVO smoother is considered unreliable."""

ILL_CONDITIONED_WARN_MSG = (
    "The covariance matrix is ill-conditioned (condition number {cond:.2e}); "
    "closed-form LOVO predictions may be inaccurate."
)
"""Warning raised when the covariance is close to singular."""


def lovo_smoother(covariance: np.ndarray, center: bool = False) -> np.ndarray:
    r"""
    Build the closed-form leave-one-volume-out smoother from a covariance matrix.

    Parameters
    ----------
    covariance : :obj:`~numpy.ndarray`
        Symmetric positive-definite covariance :math:`\mathbf{A}` of shape
        ``(N, N)``, including any noise or ridge term on its diagonal.
    center : :obj:`bool`, optional
        Whether the predictor estimates a constant mean on the :math:`N - 1`
        training volumes (as scikit-learn's ``normalize_y=True`` does). The
        standard deviation of the normalization cancels in the posterior mean,
        so only the mean needs correcting.

    Returns
    -------
    :obj:`~numpy.ndarray`
        The smoother :math:`\mathbf{S}` of shape ``(N, N)``. Row :math:`i` holds
        the weights predicting volume :math:`i` from all the others, and its
        diagonal is zero. With ``center=True``, every row sums to one.

    Examples
    --------
    >>> A = np.array([[2.0, 1.0, 0.5], [1.0, 2.0, 1.0], [0.5, 1.0, 2.0]])
    >>> S = lovo_smoother(A)
    >>> bool(np.allclose(np.diag(S), 0.0))
    True
    >>> bool(np.allclose(lovo_smoother(A, center=True).sum(axis=1), 1.0))
    True

    """
    n_volumes = len(covariance)
    eigenvalues = np.linalg.eigvalsh(covariance)
    if eigenvalues[0] > 0 and (cond := eigenvalues[-1] / eigenvalues[0]) > (
        ILL_CONDITIONED_THRESHOLD
    ):
        warn(ILL_CONDITIONED_WARN_MSG.format(cond=cond), RuntimeWarning, stacklevel=2)

    try:
        precision = cho_solve(cho_factor(covariance, lower=True), np.eye(n_volumes))
    except LinAlgError as exc:
        raise ValueError("The covariance matrix is not positive definite.") from exc

    smoother = -precision / np.diag(precision)[:, np.newaxis]
    np.fill_diagonal(smoother, 0.0)
    if center:
        smoother += ((1.0 - smoother.sum(axis=1)) / (n_volumes - 1))[:, np.newaxis]
        np.fill_diagonal(smoother, 0.0)
    return smoother


def lovo_error(smoother: np.ndarray, signals: np.ndarray) -> float:
    r"""
    Compute the closed-form LOVO error, normalized by each voxel's signal energy.

    Parameters
    ----------
    smoother : :obj:`~numpy.ndarray`
        LOVO smoother of shape ``(N, N)`` (see :func:`lovo_smoother`).
    signals : :obj:`~numpy.ndarray`
        Signals of shape ``(N, V)`` (volumes by voxels).

    Returns
    -------
    :obj:`float`
        :math:`\sum_v \lVert \mathbf{y}_v - \mathbf{S}\mathbf{y}_v \rVert^2 /
        \lVert \mathbf{y}_v \rVert^2`.

    """
    residuals = signals - smoother @ signals
    energy = np.sum(signals**2, axis=0)
    valid = energy > 0
    return float(np.sum(np.sum(residuals[:, valid] ** 2, axis=0) / energy[valid]))


def select_krr_lambda(
    grams: Sequence[np.ndarray],
    signals: Sequence[np.ndarray],
    grid: Sequence[float] = DEFAULT_KRR_LAMBDA_GRID,
) -> float:
    """
    Choose a ridge strength by closed-form LOVO error, pooled over datasets.

    Parameters
    ----------
    grams : :obj:`~typing.Sequence` of :obj:`~numpy.ndarray`
        One Gram matrix of shape ``(N_s, N_s)`` per dataset (e.g., subject).
    signals : :obj:`~typing.Sequence` of :obj:`~numpy.ndarray`
        One signal matrix of shape ``(N_s, V_s)`` per dataset.
    grid : :obj:`~typing.Sequence` of :obj:`float`, optional
        Candidate ridge strengths, added to the diagonal of each Gram matrix.

    Returns
    -------
    :obj:`float`
        The ridge strength with the lowest summed LOVO error.

    """
    errors = [
        sum(
            lovo_error(lovo_smoother(gram + lam * np.eye(len(gram))), y)
            for gram, y in zip(grams, signals, strict=True)
        )
        for lam in grid
    ]
    return float(grid[int(np.argmin(errors))])


def fit_pooled_hyperparameters(
    kernel: Kernel,
    tables: Sequence[np.ndarray],
    signals: Sequence[np.ndarray],
    alpha: float = GP_JITTER,
    maxiter: int = 200,
) -> Kernel:
    r"""
    Fit GP hyperparameters shared by several datasets (e.g., subjects).

    Maximizes :math:`\sum_s \log p(\mathbf{Y}_s \mid \mathbf{X}_s, \theta)`,
    where each dataset keeps its own gradient table :math:`\mathbf{X}_s`. With
    identical tables, this equals fitting the concatenated voxels.

    Parameters
    ----------
    kernel : :obj:`~sklearn.gaussian_process.kernels.Kernel`
        Kernel providing the starting values and bounds.
    tables : :obj:`~typing.Sequence` of :obj:`~numpy.ndarray`
        One design matrix per dataset, of shape ``(N_s, n_features)``.
    signals : :obj:`~typing.Sequence` of :obj:`~numpy.ndarray`
        One signal matrix per dataset, of shape ``(N_s, V_s)``.
    alpha : :obj:`float`, optional
        Value added to the diagonal of each covariance (numerical jitter).
    maxiter : :obj:`int`, optional
        Maximum number of L-BFGS-B iterations.

    Returns
    -------
    :obj:`~sklearn.gaussian_process.kernels.Kernel`
        A copy of ``kernel`` with the fitted hyperparameters.

    """
    gprs = [
        DiffusionGPR(
            kernel=kernel,
            optimizer=None,
            alpha=alpha,
            normalize_y=True,
            n_targets=y.shape[1],
        ).fit(X, np.asarray(y, dtype=float))
        for X, y in zip(tables, signals, strict=True)
    ]
    # As in DiffusionGPR, optimize the mean log marginal likelihood per
    # observation, so the first bounded L-BFGS-B step does not grow with the data
    scale = 1.0 / sum(np.size(gpr.y_train_) for gpr in gprs)

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        values, gradients = zip(
            *(
                gpr.log_marginal_likelihood(theta, eval_gradient=True, clone_kernel=False)
                for gpr in gprs
            ),
            strict=True,
        )
        lml = float(np.sum(values))
        if not np.isfinite(lml):
            return 1e25, np.zeros_like(theta)
        return -lml * scale, -np.sum(gradients, axis=0) * scale

    result = minimize(  # type: ignore[call-overload]
        objective,
        kernel.theta,
        jac=True,
        method="L-BFGS-B",
        bounds=kernel.bounds,
        options={"maxiter": maxiter, "gtol": LBFGS_DEFAULT_GTOL * scale},
    )
    return kernel.clone_with_theta(result.x)


def _hyperparameter_names(kernel: Kernel) -> list[str]:
    return [hp.name for hp in kernel.hyperparameters if not hp.fixed]  # type: ignore[attr-defined]


def save_hyperparameters(
    filename: Path | str,
    kernel: Kernel,
    kernel_model: str,
    **provenance: Any,
) -> Path:
    """
    Write fitted GP hyperparameters to a JSON file.

    Parameters
    ----------
    filename : :obj:`os.pathlike`
        Output file.
    kernel : :obj:`~sklearn.gaussian_process.kernels.Kernel`
        Kernel holding the hyperparameters.
    kernel_model : :obj:`str`
        The ``kernel_model`` used to build the kernel (e.g., ``"multishell"``).
    **provenance
        Free-form, JSON-serializable metadata (e.g., dataset, number of subjects).

    Returns
    -------
    :obj:`~pathlib.Path`
        The written file.

    """
    record = {
        "schema": HYPERPARAMETERS_SCHEMA,
        "kernel_model": kernel_model,
        "hyperparameters": dict(
            zip(_hyperparameter_names(kernel), np.exp(kernel.theta).tolist(), strict=True)
        ),
        "provenance": provenance,
    }
    filename = Path(filename)
    filename.write_text(json.dumps(record, indent=2))
    return filename


def load_hyperparameters(source: Path | str | Mapping[str, Any]) -> dict[str, Any]:
    """
    Read and validate a GP hyperparameter record.

    Parameters
    ----------
    source : :obj:`os.pathlike` or :obj:`~typing.Mapping`
        A JSON file written by :func:`save_hyperparameters`, or its parsed content.

    Returns
    -------
    :obj:`dict`
        The record, with keys ``schema``, ``kernel_model``, ``hyperparameters``
        and ``provenance``.

    """
    record = dict(source) if isinstance(source, Mapping) else json.loads(Path(source).read_text())
    if record.get("schema") != HYPERPARAMETERS_SCHEMA:
        raise ValueError(f"Unsupported hyperparameter schema {record.get('schema')!r}.")
    for key in ("kernel_model", "hyperparameters"):
        if key not in record:
            raise ValueError(f"Hyperparameter record lacks {key!r}.")
    record.setdefault("provenance", {})
    return record


def kernel_with_hyperparameters(kernel: Kernel, values: Mapping[str, float]) -> Kernel:
    """
    Return a copy of ``kernel`` with the given hyperparameter values.

    Parameters
    ----------
    kernel : :obj:`~sklearn.gaussian_process.kernels.Kernel`
        Kernel defining the structure (and the expected hyperparameter names).
    values : :obj:`~typing.Mapping`
        Hyperparameter values in natural units, keyed by scikit-learn name.

    Returns
    -------
    :obj:`~sklearn.gaussian_process.kernels.Kernel`
        The kernel with its hyperparameters set.

    """
    names = _hyperparameter_names(kernel)
    if set(names) != set(values):
        raise ValueError(
            f"Hyperparameters {sorted(values)} do not match the kernel's {sorted(names)}."
        )
    return kernel.clone_with_theta(np.log([float(values[name]) for name in names]))


def masked_signal_matrix(dataset: Any, mask: np.ndarray, dtype: Any = np.float32) -> np.ndarray:
    """
    Gather the masked signals of all volumes, one volume at a time.

    Parameters
    ----------
    dataset : :obj:`~nifreeze.data.base.BaseDataset`
        Dataset whose ``dataobj`` has volumes along its first axis.
    mask : :obj:`~numpy.ndarray`
        Boolean mask of the voxels to gather.
    dtype : :obj:`~numpy.dtype`, optional
        Output data type.

    Returns
    -------
    :obj:`~numpy.ndarray`
        Signals of shape ``(N, V)`` (volumes by masked voxels).

    """
    signals = np.empty((len(dataset), int(mask.sum())), dtype=dtype)
    for index in range(len(dataset)):
        signals[index] = np.asarray(dataset.dataobj[index])[mask]
    return signals


class ClosedFormLOVOMixin:
    """
    Exact leave-one-volume-out prediction from a single linear smoother.

    Subclasses implement :meth:`_covariance`, which returns the ``(N, N)``
    covariance of the volumes given the gradient table. The smoother is built
    once and rebuilt only when the gradients change. Each prediction is then one
    row of the smoother times the signal matrix. Because the held-out volume
    never contributes to its own prediction, single-fit mode returns the same
    genuine LOVO predictions and is not a canary.

    Hyperparameters (if any) are estimated once, on a random subsample of voxels
    across all volumes. This lets a handful of scalars see the held-out volume,
    while every prediction remains held-out.

    """

    single_fit_is_canary = False

    _center: bool = False
    """Whether the predictor estimates a constant mean (see :func:`lovo_smoother`)."""

    # Provided by the model classes this mixin is combined with
    _dataset: Any
    _data_mask: np.ndarray
    _locked_fit: bool | None

    def __init__(
        self,
        dataset: Any,
        *,
        nvox_prefit: int | None = DEFAULT_NVOX_PREFIT,
        random_state: int | None = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(dataset, **kwargs)  # type: ignore[call-arg]
        self._nvox_prefit = nvox_prefit
        self._random_state = random_state
        self._signals: np.ndarray | None = None
        self._smoother: np.ndarray | None = None
        self._smoother_gradients: np.ndarray | None = None

    def _covariance(self, gradients: np.ndarray) -> np.ndarray:
        """Return the ``(N, N)`` covariance of the volumes for ``gradients``."""
        raise NotImplementedError

    def _postprocess(self, predicted: np.ndarray) -> np.ndarray:
        """Post-process the predicted signals of the masked voxels."""
        return predicted

    def _lovo_signals(self) -> np.ndarray:
        if self._signals is None:
            self._signals = masked_signal_matrix(self._dataset, self._data_mask)
        return self._signals

    @property
    def smoother(self) -> np.ndarray:
        """The ``(N, N)`` smoother for the current gradient table (read-only).

        Row ``i`` holds the weights that predict volume ``i`` from the others.
        """
        self._fit(0)  # The index never enters the fit; any index avoids locking it
        assert self._smoother is not None
        smoother = self._smoother.view()
        smoother.flags.writeable = False
        return smoother

    def update(self, index: int) -> None:
        """Replace row ``index`` of the cached signals with the dataset's volume.

        The smoother is rebuilt at the next fit if the gradient table changed.
        """
        if self._signals is not None:
            self._signals[index] = np.asarray(self._dataset.dataobj[index])[self._data_mask]

    def _hyperparameter_signals(self) -> np.ndarray:
        """Return a seeded random subsample of voxels, shape ``(N, nvox_prefit)``."""
        signals = self._lovo_signals()
        n_voxels = signals.shape[1]
        if self._nvox_prefit is None or self._nvox_prefit >= n_voxels:
            return signals.astype(float)
        rng = np.random.default_rng(self._random_state)
        columns = np.sort(rng.choice(n_voxels, size=self._nvox_prefit, replace=False))
        return signals[:, columns].astype(float)

    def _fit(self, index: int | None = None, n_jobs: int | None = None, **kwargs: Any) -> int:
        """Build (or reuse) the LOVO smoother; ``index`` never enters the fit."""
        if index is None:
            self._locked_fit = True

        gradients = np.asarray(self._dataset.gradients, dtype=float)
        stale = self._smoother_gradients is None or not np.array_equal(
            gradients, self._smoother_gradients
        )
        if self._smoother is None or stale:
            self._smoother = lovo_smoother(self._covariance(gradients), center=self._center)
            self._smoother_gradients = gradients.copy()
        return 1

    def fit_predict(self, index: int | None = None, **kwargs: Any) -> np.ndarray | None:
        """Predict volume ``index`` from all the other volumes (closed form)."""
        self._fit(index)
        if index is None:
            return None

        assert self._smoother is not None
        signals = self._lovo_signals()
        predicted = self._smoother[index].astype(signals.dtype) @ signals

        retval = np.zeros(self._data_mask.shape, dtype=np.float32)
        retval[self._data_mask] = self._postprocess(predicted)
        return retval
