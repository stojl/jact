"""Probability callbacks for extracting results from the solver state."""

from __future__ import annotations

from typing import Any, Callable, NamedTuple, Union

import jax
import jax.numpy as jnp

__all__ = [
    "density",
    "density_probability",
    "full",
    "marginal_components",
    "none_callback",
    "point_mass",
    "state_probability",
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
    """Per-individual point mass carried along a characteristic.

    Internal solver pytree; not part of the public output surface. The
    user-facing callbacks expose point-mass data as plain dicts.
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
    point_mass: PointMass | None


CallbackFn = Callable[[tuple[StateCarry, ...]], Any]


def _point_mass_dict(
    state: tuple[StateCarry, ...],
    state_names: tuple[str, ...],
) -> dict[str, dict[str, jnp.ndarray]]:
    return {
        state_names[i]: {"value": carry.point_mass.value}
        for i, carry in enumerate(state)
        if carry.point_mass is not None
    }


def none_callback(state_names: tuple[str, ...]) -> CallbackFn:
    """Return nothing. Use when only side effects of the solve matter."""
    del state_names

    def fn(state: tuple[StateCarry, ...]):
        del state
        return None

    return fn


def full(state_names: tuple[str, ...]) -> CallbackFn:
    """Per-state duration density and point masses, keyed by state name."""

    def fn(state: tuple[StateCarry, ...]):
        return {
            "density": jnp.stack(
                tuple(carry.density for carry in state), axis=-2
            ),
            "point_mass": _point_mass_dict(state, state_names),
        }

    return fn


def marginal_components(state_names: tuple[str, ...]) -> CallbackFn:
    """Duration-marginalized density per state plus point masses."""

    def fn(state: tuple[StateCarry, ...]):
        return {
            "density": jnp.stack(
                tuple(jnp.sum(carry.density, axis=-1) for carry in state),
                axis=-1,
            ),
            "point_mass": _point_mass_dict(state, state_names),
        }

    return fn


def state_probability(state_names: tuple[str, ...]) -> CallbackFn:
    """Total state occupancy after marginalizing over duration."""
    del state_names

    def fn(state: tuple[StateCarry, ...]):
        return jnp.stack(
            tuple(
                jnp.sum(carry.density, axis=-1)
                if carry.point_mass is None
                else jnp.sum(carry.density, axis=-1) + carry.point_mass.value
                for carry in state
            ),
            axis=-1,
        )

    return fn


def point_mass(state_names: tuple[str, ...]) -> CallbackFn:
    """Point-mass component per state, keyed by state name."""

    def fn(state: tuple[StateCarry, ...]):
        return _point_mass_dict(state, state_names)

    return fn


def density(state_names: tuple[str, ...]) -> CallbackFn:
    """Absolutely continuous duration density per state, stacked into a tensor."""
    del state_names

    def fn(state: tuple[StateCarry, ...]):
        return jnp.stack(tuple(carry.density for carry in state), axis=-2)

    return fn


def density_probability(state_names: tuple[str, ...]) -> CallbackFn:
    """Duration-marginal density per state, stacked into a tensor."""
    del state_names

    def fn(state: tuple[StateCarry, ...]):
        return jnp.stack(
            tuple(jnp.sum(carry.density, axis=-1) for carry in state),
            axis=-1,
        )

    return fn


_CALLBACKS = {
    "none": none_callback,
    "full": full,
    "marginal_components": marginal_components,
    "state_probability": state_probability,
    "point_mass": point_mass,
    "density": density,
    "density_probability": density_probability,
}


def resolve_callback(
    callback: Union[None, str, CallbackFn],
    state_names: tuple[str, ...],
) -> CallbackFn:
    """Resolve a callback specification to a callable bound to ``state_names``."""
    if callback is None:
        return none_callback(state_names)
    if callable(callback):
        return callback
    if isinstance(callback, str):
        if callback not in _CALLBACKS:
            available = ", ".join(sorted(_CALLBACKS.keys()))
            raise ValueError(
                f"Unknown callback '{callback}'. "
                f"Available callbacks: {available}"
            )
        return _CALLBACKS[callback](state_names)
    raise TypeError(
        "callback must be None, a string, or a callable, "
        f"got {type(callback)}"
    )
