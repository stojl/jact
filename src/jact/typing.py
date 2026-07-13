# pyright: strict, reportMissingImports=false

"""Callable protocols used by jact models and cashflows."""

from __future__ import annotations

from typing import Any, Protocol, TypeAlias

import jax.numpy as jnp
import numpy as np
from jax import Array

ArrayLike: TypeAlias = (
    Array
    | np.ndarray[Any, Any]
    | np.bool_
    | np.number[Any]
    | bool
    | int
    | float
    | complex
)

__all__ = [
    "ArrayLike",
    "Intensity",
    "GroupedIntensity",
    "Payment",
    "When",
    "DurationAt",
    "Weight",
]


class Intensity(Protocol):
    """A single-transition intensity callable.

    The return value may have any shape broadcastable to
    ``(batch, duration)``.
    """

    def __call__(
        self,
        t: jnp.ndarray,
        d: jnp.ndarray,
        /,
        **kwargs: Any,
    ) -> ArrayLike: ...


class GroupedIntensity(Protocol):
    """A multi-transition intensity callable.

    The return value has a leading transition axis. Each selected transition
    output may have any shape broadcastable to ``(batch, duration)``.
    """

    def __call__(
        self,
        t: jnp.ndarray,
        d: jnp.ndarray,
        /,
        **kwargs: Any,
    ) -> ArrayLike: ...


class Payment(Protocol):
    """A state or transition payment callable.

    The return value may have any shape broadcastable to
    ``(batch, duration)``.
    """

    def __call__(
        self,
        t: jnp.ndarray,
        d: jnp.ndarray,
        /,
        **kwargs: Any,
    ) -> ArrayLike: ...


class When(Protocol):
    """A scheduled-event time callable.

    The return value must be scalar or broadcastable to ``(batch,)``.
    """

    def __call__(
        self, **kwargs: Any
    ) -> ArrayLike: ...


class DurationAt(Protocol):
    """A duration-event target callable.

    The return value must be scalar or broadcastable to ``(batch,)``.
    """

    def __call__(
        self, **kwargs: Any
    ) -> ArrayLike: ...


class Weight(Protocol):
    """A cashflow-view weight callable.

    The return value must be scalar or broadcastable to ``(batch,)``.
    """

    def __call__(
        self, t: jnp.ndarray, /, **kwargs: Any
    ) -> ArrayLike: ...
