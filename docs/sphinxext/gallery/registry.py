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
"""Model registry and capability filtering for the prediction gallery.

Each :class:`ModelSpec` names a gallery model and reads its applicability from
the model class's declarative *capability contract*
(:class:`~nifreeze.model.base.BaseModel`), so the constraints are not restated
here. GP is the one exception: its applicable scheme depends on the kernel, so
the two GP entries carry an explicit ``scheme_override``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from gallery.datasets import SCHEMES
from nifreeze.data.dmri import DWI
from nifreeze.model.base import BaseModel, ModelFactory
from nifreeze.model.dmri import (
    AverageDWIModel,
    DKIModel,
    DTIModel,
    GPModel,
    GQIModel,
)

#: Every acquisition scheme — the default for a model without ``applicable_schemes``.
ANY_SCHEME = frozenset(SCHEMES)

#: Gallery prediction modes.
GALLERY_MODES = ("lovo", "single-fit")


@dataclass(frozen=True)
class ModelSpec:
    """A gallery model: how to build it, and (via its class) what it supports."""

    key: str
    """Gallery identifier (e.g. ``"dti"``, ``"gp-multishell"``)."""
    label: str
    """Human-readable label."""
    factory_name: str
    """Name passed to :meth:`~nifreeze.model.base.ModelFactory.init`."""
    model_cls: type[BaseModel]
    """Concrete model class (source of the capability contract)."""
    kwargs: Mapping[str, object] = field(default_factory=dict)
    """Extra construction kwargs (e.g. ``{"kernel_model": "multishell"}``)."""
    scheme_override: frozenset[str] | None = None
    """Overrides ``model_cls.applicable_schemes`` (used for GP kernels)."""

    @property
    def applicable_schemes(self) -> frozenset[str]:
        """Schemes this spec applies to; a model without the attribute means any."""
        if self.scheme_override is not None:
            return self.scheme_override
        return getattr(self.model_cls, "applicable_schemes", ANY_SCHEME)

    @property
    def supports_single_fit(self) -> bool:
        """Whether the underlying model supports single-fit mode."""
        return self.model_cls.supports_single_fit


#: The models the gallery attempts, per issue #458. Applicability is resolved by
#: capability filtering, so this list is the *superset* of what may be shown.
GALLERY_MODELS: list[ModelSpec] = [
    ModelSpec("average", "Average DWI (shell)", "avgdwi", AverageDWIModel),
    ModelSpec("dti", "DTI", "dti", DTIModel),
    ModelSpec("dki", "DKI", "dki", DKIModel),
    ModelSpec("gqi", "GQI", "gqi", GQIModel),
    ModelSpec(
        "gp-spherical",
        "GP (spherical)",
        "gp",
        GPModel,
        {"kernel_model": "spherical"},
        frozenset({"single-shell"}),
    ),
    ModelSpec(
        "gp-multishell",
        "GP (multi-shell)",
        "gp",
        GPModel,
        {"kernel_model": "multishell"},
        frozenset({"multi-shell"}),
    ),
]


def check_applicability(spec: ModelSpec, scheme: str) -> tuple[bool, str | None]:
    """Whether ``spec`` applies to ``scheme``; returns ``(ok, reason)``."""
    if scheme not in spec.applicable_schemes:
        allowed = ", ".join(sorted(spec.applicable_schemes))
        return False, f"{spec.label} not applicable to {scheme} (supports: {allowed})"
    return True, None


def check_mode(spec: ModelSpec, mode: str) -> tuple[bool, str | None]:
    """Whether ``spec`` supports ``mode``; returns ``(ok, reason)``."""
    if mode not in GALLERY_MODES:
        return False, f"Unknown mode {mode!r}"
    if mode == "single-fit" and not spec.supports_single_fit:
        return False, f"{spec.label} does not support single-fit mode"
    return True, None


def build_model(spec: ModelSpec, dwi: DWI) -> BaseModel:
    """Instantiate the model for ``spec`` on ``dwi`` via the factory."""
    return ModelFactory.init(spec.factory_name, dataset=dwi, **dict(spec.kwargs))
