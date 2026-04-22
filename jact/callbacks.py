"""Probability callbacks for extracting results from the solver state."""

from __future__ import annotations

from typing import Any, Callable, NamedTuple, Union

import jax
import jax.numpy as jnp


class StateCarry(NamedTuple):
    """Per-state solver carry."""

    density: jnp.ndarray
    point_mass: jnp.ndarray | None


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
            point_mass=(
                None
                if carry.point_mass is None
                else jnp.sum(carry.point_mass, axis=-1)
            ),
        )
        for carry in state
    )


@jax.jit
def collapse_point(state: tuple[StateCarry, ...]):
    """Collapse point mass into density for each state."""
    return tuple(
        carry.density
        if carry.point_mass is None
        else carry.density + carry.point_mass
        for carry in state
    )


@jax.jit
def collapse_point_no_duration(state: tuple[StateCarry, ...]):
    """Collapse point mass and marginalize over duration."""
    return jnp.stack(
        tuple(
            jnp.sum(carry.density, axis=-1)
            if carry.point_mass is None
            else jnp.sum(carry.density + carry.point_mass, axis=-1)
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
        None if carry.point_mass is None else jnp.sum(carry.point_mass, axis=-1)
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
