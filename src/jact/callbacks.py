"""Probability callbacks for extracting results from the solver state."""

from __future__ import annotations

from typing import Any, Callable, NamedTuple, Union

import jax
import jax.numpy as jnp

__all__ = [
    "PointMass",
    "StateCarry",
    "collapse_point",
    "collapse_point_no_duration",
    "default",
    "no_duration",
    "no_point",
    "no_point_no_duration",
    "none_callback",
    "point_only",
    "point_only_no_duration",
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
class PointMass:
    """Per-individual point mass carried along a characteristic."""

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
    """Per-state solver carry."""

    density: jnp.ndarray
    point_mass: PointMass | None


@jax.jit
def none_callback(state: tuple[StateCarry, ...]):
    """Return nothing. Use when only side effects of the solve matter."""
    return None


@jax.jit
def default(state: tuple[StateCarry, ...]):
    """Return the full pytree state unchanged."""
    return state


@jax.jit
def no_duration(state: tuple[StateCarry, ...]):
    """Marginalize over duration, preserving the per-state pytree."""
    return tuple(
        StateCarry(
            density=jnp.sum(carry.density, axis=-1),
            point_mass=carry.point_mass,
        )
        for carry in state
    )


@jax.jit
def collapse_point(state: tuple[StateCarry, ...]):
    """Collapse point mass into density for each state."""
    return tuple(
        carry.density
        if carry.point_mass is None
        else carry.density.at[..., 0].add(carry.point_mass.value)
        for carry in state
    )


@jax.jit
def collapse_point_no_duration(state: tuple[StateCarry, ...]):
    """Collapse point mass and marginalize over duration."""
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
def point_only(state: tuple[StateCarry, ...]):
    """Return only the point-mass component per state."""
    return tuple(carry.point_mass for carry in state)


@jax.jit
def point_only_no_duration(state: tuple[StateCarry, ...]):
    """Return only the duration-marginal point mass per state."""
    return tuple(
        None if carry.point_mass is None else carry.point_mass.value
        for carry in state
    )


@jax.jit
def no_point(state: tuple[StateCarry, ...]):
    """Return only the absolutely continuous density per state."""
    return tuple(carry.density for carry in state)


@jax.jit
def no_point_no_duration(state: tuple[StateCarry, ...]):
    """Return the duration-marginal density, restacked across states."""
    return jnp.stack(
        tuple(jnp.sum(carry.density, axis=-1) for carry in state),
        axis=-1,
    )


_CALLBACKS = {
    "none": none_callback,
    "default": default,
    "no_duration": no_duration,
    "collapse_point": collapse_point,
    "collapse_point_no_duration": collapse_point_no_duration,
    "point_only": point_only,
    "point_only_no_duration": point_only_no_duration,
    "no_point": no_point,
    "no_point_no_duration": no_point_no_duration,
}


def resolve_callback(
    callback: Union[None, str, Callable[[tuple[StateCarry, ...]], Any]],
) -> Callable[[tuple[StateCarry, ...]], Any]:
    """Resolve a callback specification to a callable."""
    if callback is None:
        return none_callback
    if callable(callback):
        return callback
    if isinstance(callback, str):
        if callback not in _CALLBACKS:
            available = ", ".join(sorted(_CALLBACKS.keys()))
            raise ValueError(
                f"Unknown callback '{callback}'. "
                f"Available callbacks: {available}"
            )
        return _CALLBACKS[callback]
    raise TypeError(
        "callback must be None, a string, or a callable, "
        f"got {type(callback)}"
    )
