"""Probability callbacks for extracting results from the solver state.

Callbacks control what is recorded at each time step during the solve.
Each callback receives the raw solver state and returns an arbitrary
PyTree that gets stacked across time steps.

The solver tracks two objects internally:

- ``p``: the absolutely continuous part of the duration density,
  shape ``(batch, n_states, D)``.
- ``p_point``: the point mass at duration zero for the initial state,
  shape ``(batch, D)``.

Users who need full access to these can use the ``'default'`` callback.
For most use cases, one of the built-in callbacks that marginalizes
over duration or collapses the point mass is more convenient.
"""

from typing import Callable, Optional, Union

import jax
import jax.numpy as jnp


# -------------------------------------------------------------------- #
# Built-in callbacks                                                    #
# -------------------------------------------------------------------- #


@jax.jit
def none_callback(p: jnp.ndarray, p_point: jnp.ndarray):
    """Return nothing. Use when only the final state matters."""
    return None


@jax.jit
def default(p: jnp.ndarray, p_point: jnp.ndarray):
    """Return the full density and point mass (no reduction)."""
    return p, p_point


@jax.jit
def no_duration(p: jnp.ndarray, p_point: jnp.ndarray):
    """Marginalize over duration, keeping density and point mass separate."""
    return p[..., -1], p_point[..., -1]


@jax.jit
def collapse_point(p: jnp.ndarray, p_point: jnp.ndarray):
    """Collapse point mass into the first state's density."""
    p_with_point = p[..., 0, :] + p_point
    p = p.at[..., 0, :].set(p_with_point)
    return p


@jax.jit
def collapse_point_no_duration(p: jnp.ndarray, p_point: jnp.ndarray):
    """Collapse point mass and marginalize over duration.

    This is the most common callback for actuarial applications:
    returns transition probabilities as shape ``(batch, n_states)``.
    """
    p = collapse_point(p, p_point)
    return p[..., -1]


@jax.jit
def point_only(p: jnp.ndarray, p_point: jnp.ndarray):
    """Return only the point mass."""
    return p_point


@jax.jit
def point_only_no_duration(p: jnp.ndarray, p_point: jnp.ndarray):
    """Return only the point mass, marginalized over duration."""
    return p_point[..., -1]


@jax.jit
def no_point(p: jnp.ndarray, p_point: jnp.ndarray):
    """Return only the absolutely continuous density."""
    return p


@jax.jit
def no_point_no_duration(p: jnp.ndarray, p_point: jnp.ndarray):
    """Return the density marginalized over duration (no point mass)."""
    return p[..., -1]


# -------------------------------------------------------------------- #
# Registry                                                              #
# -------------------------------------------------------------------- #

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
    callback: Union[None, str, Callable],
) -> Callable:
    """Resolve a callback specification to a callable.

    Parameters
    ----------
    callback : None, str, or callable
        If None, uses the ``'none'`` callback.
        If a string, looks up a built-in callback by name.
        If a callable, returns it directly.

    Returns
    -------
    callable
    """
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
        f"callback must be None, a string, or a callable, "
        f"got {type(callback)}"
    )
