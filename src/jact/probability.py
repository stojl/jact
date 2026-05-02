"""Probability output types and dispatch.

The public surface is six frozen-dataclass output types
(``StateProbability``, ``DensityProbability``, ``Density``, ``PointMass``,
``MarginalComponents``, ``Full``) plus the ``ProbabilityOutput`` union and
support for arbitrary user-supplied callables.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, NamedTuple, Union

import jax
import jax.numpy as jnp

__all__ = [
    "StateProbability",
    "DensityProbability",
    "Density",
    "PointMass",
    "MarginalComponents",
    "Full",
    "ProbabilityOutput",
    "CallbackFn",
    "resolve_callback",
]


def _validate_shape(
    name: str,
    value: jnp.ndarray,
    expected_shape: tuple[int, ...],
) -> None:
    shape = jnp.shape(value)
    if shape != expected_shape:
        raise ValueError(
            f"{name} must have shape {expected_shape}, got {shape}."
        )


def _validate_non_negative_if_concrete(value: jnp.ndarray) -> None:
    try:
        arr = jnp.asarray(value)
        if bool(jnp.any(arr < 0)):
            raise ValueError("value must be non-negative.")
    except Exception as exc:  # pragma: no cover - tracer path
        if "tracer" not in type(exc).__name__.lower():
            try:
                message = str(exc).lower()
            except Exception:  # pragma: no cover
                message = ""
            if "tracer" not in message and "concret" not in message:
                raise


@jax.tree_util.register_pytree_node_class
class _PointMass:
    """Per-individual point mass carried along a characteristic.

    Internal solver pytree; not part of the public output surface. The
    user-facing reducers expose point-mass data as plain dicts.
    """

    __slots__ = ("value", "d_0", "log_value")

    def __init__(
        self,
        value: jnp.ndarray,
        d_0: jnp.ndarray,
        log_value: jnp.ndarray | None = None,
    ):
        value_shape = jnp.shape(value)
        _validate_shape("d_0", d_0, value_shape)
        if log_value is not None:
            _validate_shape("log_value", log_value, value_shape)
        _validate_non_negative_if_concrete(value)
        self.value = value
        self.d_0 = d_0
        self.log_value = (
            jnp.where(value > 0, jnp.log(value), -jnp.inf)
            if log_value is None
            else log_value
        )

    def tree_flatten(self):
        return (self.value, self.d_0, self.log_value), None

    @classmethod
    def tree_unflatten(cls, _, children):
        value, d_0, log_value = children
        self = cls.__new__(cls)
        self.value = value
        self.d_0 = d_0
        self.log_value = log_value
        return self


class StateCarry(NamedTuple):
    """Per-state solver carry.

    Internal solver pytree; not part of the public output surface.
    """

    density: jnp.ndarray
    point_mass: _PointMass | None


CallbackFn = Callable[[tuple[StateCarry, ...]], Any]


# --------------------------------------------------------------------------- #
# Public output types                                                         #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class StateProbability:
    """Total state occupancy after marginalizing over duration.

    Returns a ``(T, B, S)`` tensor of duration-marginal density plus
    point-mass value per state.
    """


@dataclass(frozen=True)
class DensityProbability:
    """Duration-marginal density per state, stacked into a tensor.

    Returns a ``(T, B, S)`` tensor; excludes point masses.
    """


@dataclass(frozen=True)
class Density:
    """Absolutely continuous duration density per state, stacked.

    Returns a ``(T, B, S, D)`` tensor of raw duration density per state.
    """


@dataclass(frozen=True)
class PointMass:
    """Point-mass component per state, keyed by state name.

    Returns ``{state_name: (T, B)}``, including only states that carry a
    point mass.
    """


@dataclass(frozen=True)
class MarginalComponents:
    """Duration-marginal density per state plus point masses.

    Returns ``{"density": (T, B, S), "point_mass": {state_name: (T, B)}}``.
    """


@dataclass(frozen=True)
class Full:
    """Per-state duration density and point masses, keyed by state name.

    Returns ``{"density": (T, B, S, D), "point_mass": {state_name: (T, B)}}``.
    """


ProbabilityOutput = Union[
    StateProbability,
    DensityProbability,
    Density,
    PointMass,
    MarginalComponents,
    Full,
]


# --------------------------------------------------------------------------- #
# Internal jit'd reducer implementations                                      #
# --------------------------------------------------------------------------- #


@jax.jit
def _none_callback(state: tuple[StateCarry, ...]):
    del state
    return None


@jax.jit
def _state_probability_callback(state: tuple[StateCarry, ...]):
    return jnp.stack(
        tuple(
            jnp.sum(carry.density, axis=-1)
            if carry.point_mass is None
            else jnp.sum(carry.density, axis=-1) + carry.point_mass.value
            for carry in state
        ),
        axis=-1,
    )


@jax.jit
def _density_callback(state: tuple[StateCarry, ...]):
    return jnp.stack(tuple(carry.density for carry in state), axis=-2)


@jax.jit
def _density_probability_callback(state: tuple[StateCarry, ...]):
    return jnp.stack(
        tuple(jnp.sum(carry.density, axis=-1) for carry in state),
        axis=-1,
    )


def _point_mass_dict(
    state: tuple[StateCarry, ...],
    state_names: tuple[str, ...],
) -> dict[str, jnp.ndarray]:
    return {
        state_names[i]: carry.point_mass.value
        for i, carry in enumerate(state)
        if carry.point_mass is not None
    }


@lru_cache(maxsize=None)
def _full_callback(state_names: tuple[str, ...]) -> CallbackFn:
    def fn(state: tuple[StateCarry, ...]):
        return {
            "density": jnp.stack(
                tuple(carry.density for carry in state), axis=-2
            ),
            "point_mass": _point_mass_dict(state, state_names),
        }

    return fn


@lru_cache(maxsize=None)
def _marginal_components_callback(
    state_names: tuple[str, ...],
) -> CallbackFn:
    def fn(state: tuple[StateCarry, ...]):
        return {
            "density": jnp.stack(
                tuple(jnp.sum(carry.density, axis=-1) for carry in state),
                axis=-1,
            ),
            "point_mass": _point_mass_dict(state, state_names),
        }

    return fn


@lru_cache(maxsize=None)
def _point_mass_callback(state_names: tuple[str, ...]) -> CallbackFn:
    def fn(state: tuple[StateCarry, ...]):
        return _point_mass_dict(state, state_names)

    return fn


# --------------------------------------------------------------------------- #
# Dispatch                                                                    #
# --------------------------------------------------------------------------- #


def resolve_callback(
    output: Union[None, ProbabilityOutput, CallbackFn],
    state_names: tuple[str, ...],
) -> CallbackFn:
    """Resolve a probability output specification to a JIT-friendly callable.

    Returns a callable bound to ``state_names`` for the reducers that need
    the names. ``output=None`` resolves to a private no-op so the solver can
    still scan; the result is discarded by the caller.
    """
    if output is None:
        return _none_callback
    if isinstance(output, StateProbability):
        return _state_probability_callback
    if isinstance(output, DensityProbability):
        return _density_probability_callback
    if isinstance(output, Density):
        return _density_callback
    if isinstance(output, PointMass):
        return _point_mass_callback(state_names)
    if isinstance(output, MarginalComponents):
        return _marginal_components_callback(state_names)
    if isinstance(output, Full):
        return _full_callback(state_names)
    if callable(output):
        return output
    raise TypeError(  # pyright: ignore[reportUnreachable]
        "probability must be None, a probability-output instance "
        "(StateProbability, DensityProbability, Density, PointMass, "
        "MarginalComponents, Full), or a callable; "
        f"got {type(output).__name__}."
    )
