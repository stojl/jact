# pyright: strict, reportMissingImports=false, reportUnknownMemberType=false, reportUntypedClassDecorator=false, reportUntypedFunctionDecorator=false
"""Probability output types and dispatch.

The public surface is seven frozen-dataclass output types
(``StateProbability``, ``DensityProbability``, ``Density``, ``PointMass``,
``MarginalComponents``, ``Tail``, ``Full``) plus the ``ProbabilityOutput`` union and
support for arbitrary user-supplied callables.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, NamedTuple, TypedDict, Union

import jax
import jax.numpy as jnp

__all__ = [
    "StateProbability",
    "DensityProbability",
    "Density",
    "PointMass",
    "MarginalComponents",
    "Full",
    "Tail",
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
        if not _is_tracer_or_concretization_error(exc):
            raise


def _is_tracer_or_concretization_error(exc: Exception) -> bool:
    exc_type = type(exc).__name__.lower()
    if "tracer" in exc_type:
        return True
    try:
        message = str(exc).lower()
    except Exception:  # pragma: no cover
        message = ""
    return "tracer" in message or "concret" in message


@jax.tree_util.register_pytree_node_class
class _PointMass:
    """Per-individual point mass carried along a characteristic.

    Internal solver pytree; not part of the public output surface. The
    user-facing reducers expose point-mass data as plain dicts.
    """

    __slots__ = ("value", "d_0", "initial_value", "log_survival")

    def __init__(
        self,
        value: jnp.ndarray,
        d_0: jnp.ndarray,
        initial_value: jnp.ndarray | None = None,
        log_survival: jnp.ndarray | None = None,
    ) -> None:
        value_shape = jnp.shape(value)
        _validate_shape("d_0", d_0, value_shape)
        if initial_value is not None:
            _validate_shape("initial_value", initial_value, value_shape)
        if log_survival is not None:
            _validate_shape("log_survival", log_survival, value_shape)
        _validate_non_negative_if_concrete(value)
        self.value = value
        self.d_0 = d_0
        # Keep mass separate from survival so zero initial mass has a finite,
        # nonzero derivative. Accumulating log survival retains small hazards.
        self.initial_value = value if initial_value is None else initial_value
        self.log_survival = (
            jnp.zeros_like(value) if log_survival is None else log_survival
        )

    def tree_flatten(
        self,
    ) -> tuple[
        tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
        None,
    ]:
        return (self.value, self.d_0, self.initial_value, self.log_survival), None

    @classmethod
    def tree_unflatten(
        cls,
        _aux: None,
        children: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> _PointMass:
        value, d_0, initial_value, log_survival = children
        self = cls.__new__(cls)
        self.value = value
        self.d_0 = d_0
        self.initial_value = initial_value
        self.log_survival = log_survival
        return self


class _TailProbability(NamedTuple):
    """Continuous mass at a fixed representative duration."""

    mass: jnp.ndarray
    duration: jnp.ndarray


class StateCarry(NamedTuple):
    """Per-state solver carry.

    Internal solver pytree; not part of the public output surface.
    """

    density: jnp.ndarray
    point_mass: _PointMass | None
    tail: _TailProbability | None = None


CallbackFn = Callable[[tuple[StateCarry, ...]], Any]
PointMassResult = dict[str, jax.Array]


class TailResult(TypedDict):
    mass: jax.Array
    duration: jax.Array


class _OptionalTailResult(TypedDict, total=False):
    tail: TailResult


class ComponentsResult(_OptionalTailResult):
    density: jax.Array
    point_mass: PointMassResult


ArrayCallback = Callable[[tuple[StateCarry, ...]], jax.Array]
PointMassCallback = Callable[[tuple[StateCarry, ...]], PointMassResult]
ComponentsCallback = Callable[[tuple[StateCarry, ...]], ComponentsResult]


# --------------------------------------------------------------------------- #
# Public output types                                                         #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class StateProbability:
    """Total state occupancy after marginalizing over duration.

    Returns a ``(T, B, S)`` tensor of duration-marginal density plus
    tail mass and point-mass value per state.
    """


@dataclass(frozen=True)
class DensityProbability:
    """Duration-marginal density per state, stacked into a tensor.

    Returns a ``(T, B, S)`` tensor including tail mass; excludes point masses.
    """


@dataclass(frozen=True)
class Density:
    """Absolutely continuous duration density per state, stacked.

    Returns a ``(T, B, S, D)`` tensor of retained regular cells per state.
    Compressed tail mass is available separately through ``Tail()``.
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
class Tail:
    """Continuous tail diagnostics with ``mass`` and ``duration`` (T, B, S).

    Both leaves are zero without compression. Under compression the duration
    is fixed at the cutoff, even when the tail mass is zero.
    """


@dataclass(frozen=True)
class Full:
    """Per-state duration density and point masses, keyed by state name.

    Returns ``{"density": (T, B, S, D), "point_mass": {state_name: (T, B)}}``.
    Under compression adds ``"tail"`` with the same payload as ``Tail()``.
    """


ProbabilityOutput = Union[
    StateProbability,
    DensityProbability,
    Density,
    PointMass,
    MarginalComponents,
    Full,
    Tail,
]


# --------------------------------------------------------------------------- #
# Internal jit'd reducer implementations                                      #
# --------------------------------------------------------------------------- #


@jax.jit
def _none_callback(state: tuple[StateCarry, ...]) -> None:
    del state
    return None


def _continuous_mass(carry: StateCarry) -> jnp.ndarray:
    mass = jnp.sum(carry.density, axis=-1)
    return mass if carry.tail is None else mass + carry.tail.mass


@jax.jit
def _tail_callback(state: tuple[StateCarry, ...]) -> TailResult:
    return {
        "mass": jnp.stack(tuple(
            jnp.zeros_like(carry.density[:, 0])
            if carry.tail is None else carry.tail.mass for carry in state
        ), axis=-1),
        "duration": jnp.stack(tuple(
            jnp.zeros_like(carry.density[:, 0])
            if carry.tail is None else carry.tail.duration for carry in state
        ), axis=-1),
    }


@jax.jit
def _state_probability_callback(state: tuple[StateCarry, ...]) -> jnp.ndarray:
    return jnp.stack(
        tuple(
            _continuous_mass(carry)
            if carry.point_mass is None
            else _continuous_mass(carry) + carry.point_mass.value
            for carry in state
        ),
        axis=-1,
    )


@jax.jit
def _density_callback(state: tuple[StateCarry, ...]) -> jnp.ndarray:
    return jnp.stack(tuple(carry.density for carry in state), axis=-2)


@jax.jit
def _density_probability_callback(
    state: tuple[StateCarry, ...],
) -> jnp.ndarray:
    return jnp.stack(
        tuple(_continuous_mass(carry) for carry in state),
        axis=-1,
    )


def _point_mass_dict(
    state: tuple[StateCarry, ...],
    state_names: tuple[str, ...],
) -> PointMassResult:
    return {
        state_names[i]: carry.point_mass.value
        for i, carry in enumerate(state)
        if carry.point_mass is not None
    }


@lru_cache(maxsize=None)
def _full_callback(state_names: tuple[str, ...]) -> ComponentsCallback:
    def fn(state: tuple[StateCarry, ...]) -> ComponentsResult:
        result: ComponentsResult = {
            "density": jnp.stack(
                tuple(carry.density for carry in state), axis=-2
            ),
            "point_mass": _point_mass_dict(state, state_names),
        }
        if state[0].tail is not None:
            result["tail"] = _tail_callback(state)
        return result

    return fn


@lru_cache(maxsize=None)
def _marginal_components_callback(
    state_names: tuple[str, ...],
) -> ComponentsCallback:
    def fn(state: tuple[StateCarry, ...]) -> ComponentsResult:
        return {
            "density": jnp.stack(
                tuple(_continuous_mass(carry) for carry in state),
                axis=-1,
            ),
            "point_mass": _point_mass_dict(state, state_names),
        }

    return fn


@lru_cache(maxsize=None)
def _point_mass_callback(state_names: tuple[str, ...]) -> PointMassCallback:
    def fn(state: tuple[StateCarry, ...]) -> PointMassResult:
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
    if isinstance(output, Tail):
        return _tail_callback
    if isinstance(output, Full):
        return _full_callback(state_names)
    if callable(output):
        return output
    raise TypeError(  # pyright: ignore[reportUnreachable]
        "probability must be None, a probability-output instance "
        "(StateProbability, DensityProbability, Density, PointMass, "
        "MarginalComponents, Tail, Full), or a callable; "
        f"got {type(output).__name__}."
    )
