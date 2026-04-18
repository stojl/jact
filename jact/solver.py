"""Semi-Markov solver with duration-dependent transition intensities.

This module contains the numerical solver that computes transition
probabilities by stepping through time using a Heun scheme
(second-order predictor-corrector) inside ``jax.lax.scan``.

The solver operates on a J×J matrix of intensity callables, tracks
the duration density for each state, and handles the point mass at
duration zero separately for numerical accuracy.
"""

from __future__ import annotations

from functools import partial, reduce
from typing import Any, Callable, Dict, Optional, Sequence, Union

import jax
import jax.numpy as jnp

from .callbacks import resolve_callback


# -------------------------------------------------------------------- #
# Low-level array operations                                            #
# -------------------------------------------------------------------- #


@jax.jit
def _update_p(
    p: jnp.ndarray,
    delta: jnp.ndarray,
    next_inflow: jnp.ndarray,
    step_size: float,
) -> jnp.ndarray:
    """Shift duration axis and apply derivative update to density."""
    p_next = p.at[..., 1:].set(p[..., :-1] + step_size * delta[..., :-1])
    p_next = p_next.at[..., 0].set(step_size * next_inflow)
    return p_next


@jax.jit
def _update_p_point(
    p_point: jnp.ndarray,
    delta: jnp.ndarray,
    step_size: jnp.ndarray,
) -> jnp.ndarray:
    """Shift duration axis and apply derivative update to point mass."""
    p_next = p_point.at[..., 1:].set(
        p_point[..., :-1] + step_size * delta[..., :-1]
    )
    p_next = p_next.at[..., 0].set(0.0)
    return p_next


# -------------------------------------------------------------------- #
# Core derivative computation                                           #
# -------------------------------------------------------------------- #


def _compute_core(p_single, p_point_single, mu_plus_matrix, mu_minus_matrix):
    """Compute inflows, density derivative, and point mass derivative.

    This function operates on a single individual (no batch dimension)
    and is vmapped over the batch axis by :func:`_compute_derivative`.

    Parameters
    ----------
    p_single : jnp.ndarray
        Density for one individual, shape ``(J, D-1)``.
    p_point_single : jnp.ndarray
        Point mass for one individual, shape ``(D-1,)``.
    mu_plus_matrix : nested list
        Intensity values evaluated at ``grid + perturbation``.
    mu_minus_matrix : nested list
        Intensity values evaluated at ``grid - perturbation``.

    Returns
    -------
    next_inflow : jnp.ndarray
        Inflow to each state at duration zero, shape ``(J,)``.
    delta_p : jnp.ndarray
        Derivative of the density, shape ``(J, D-1)``.
    delta_p_point : jnp.ndarray
        Derivative of the point mass, shape ``(D-1,)``.
    """
    J, D_minus_1 = p_single.shape

    outflow_plus_list = [[] for _ in range(J)]
    outflow_avg_list = [[] for _ in range(J)]
    next_inflow_list = []

    for j in range(J):
        inflow_terms_for_j = []
        for i in range(J):
            m_p = mu_plus_matrix[i][j]
            m_m = mu_minus_matrix[i][j]

            if m_p is not None:
                m_p_slice = m_p[:-1]
                m_avg = 0.5 * (m_p_slice + m_m[1:])

                outflow_plus_list[i].append(m_p_slice)
                outflow_avg_list[i].append(m_avg)

                term_p = jnp.sum(m_avg * p_single[i, :])
                inflow_terms_for_j.append(term_p)

                if i == 0:
                    term_p_point = jnp.sum(m_p_slice * p_point_single)
                    inflow_terms_for_j.append(term_p_point)

        if inflow_terms_for_j:
            next_inflow_list.append(reduce(jax.lax.add, inflow_terms_for_j))
        else:
            next_inflow_list.append(0.0)

    final_outflow_plus = jnp.stack(
        [
            reduce(jax.lax.add, l) if l else jnp.zeros(D_minus_1)
            for l in outflow_plus_list
        ]
    )
    final_outflow_avg = jnp.stack(
        [
            reduce(jax.lax.add, l) if l else jnp.zeros(D_minus_1)
            for l in outflow_avg_list
        ]
    )

    next_inflow = jnp.array(next_inflow_list)

    delta_p = -p_single * final_outflow_avg
    delta_p_point = -final_outflow_plus[0, :] * p_point_single

    return next_inflow, delta_p, delta_p_point


@jax.jit
def _compute_derivative(p, p_point, mu_plus_matrix, mu_minus_matrix):
    """Vmap the core computation over the batch dimension."""
    mu_axes = tuple(
        tuple(0 if entry is not None else None for entry in row)
        for row in mu_plus_matrix
    )
    vmap_func = jax.vmap(
        _compute_core, in_axes=(0, 0, mu_axes, mu_axes)
    )
    return vmap_func(p, p_point, mu_plus_matrix, mu_minus_matrix)


# -------------------------------------------------------------------- #
# Intensity evaluation                                                  #
# -------------------------------------------------------------------- #


def _evaluate_intensities(matrix, *args, **kwargs):
    """Evaluate all intensity callables in the matrix.

    Parameters
    ----------
    matrix : nested list of callables or None
        The J×J intensity matrix.
    *args, **kwargs
        Arguments passed to each callable.

    Returns
    -------
    nested list of arrays or None
    """
    return jax.tree_util.tree_map(
        lambda f: f(*args, **kwargs),
        matrix,
    )


# -------------------------------------------------------------------- #
# Heun scheme solver                                                    #
# -------------------------------------------------------------------- #


@partial(
    jax.jit,
    static_argnames=["step_size", "intensity", "prob_callback", "perturbation"],
)
def _heun_solver(
    p_0: jnp.ndarray,
    p_point_0: jnp.ndarray,
    grid: jnp.ndarray,
    step_size: float,
    intensity: Sequence[Sequence[Optional[Callable[..., jnp.ndarray]]]],
    intensity_kwargs: Dict[str, jnp.ndarray],
    prob_callback: Callable[..., jnp.ndarray],
    perturbation: jnp.ndarray,
):
    """Run the Heun scheme solver.

    Parameters
    ----------
    p_0 : jnp.ndarray
        Initial density, shape ``(batch, J, D)``.
    p_point_0 : jnp.ndarray
        Initial point mass, shape ``(batch, D)``.
    grid : jnp.ndarray
        Duration grid, shape ``(1, D+1)``.
    step_size : float
        Time step size.
    intensity : nested list
        J×J matrix of intensity callables.
    intensity_kwargs : dict
        Covariate arrays passed to intensity callables.
    prob_callback : callable
        Callback for extracting results at each time step.
    perturbation : float
        Small value for finite-difference grid offset.

    Returns
    -------
    dict
        Result dictionary with ``'probability'`` key.
    """
    grid_minus = grid - perturbation
    grid_plus = grid + perturbation

    def heun_scan(carry, t):
        p, p_point = carry

        t_left = t + perturbation

        mu_plus = _evaluate_intensities(
            intensity, t_left, grid_plus, **intensity_kwargs
        )
        mu_minus = _evaluate_intensities(
            intensity, t_left, grid_minus, **intensity_kwargs
        )

        next_inflow, delta_p, delta_p_point = _compute_derivative(
            p, p_point, mu_plus, mu_minus
        )

        t += step_size

        p_2 = _update_p(p, delta_p, next_inflow, step_size)
        p_point_2 = _update_p_point(p_point, delta_p_point, step_size)

        t_right = t - perturbation

        mu_plus = _evaluate_intensities(
            intensity, t_right, grid_plus, **intensity_kwargs
        )
        mu_minus = _evaluate_intensities(
            intensity, t_right, grid_minus, **intensity_kwargs
        )

        next_inflow_2, delta_p_2, delta_p_point_2 = _compute_derivative(
            p_2, p_point_2, mu_plus, mu_minus
        )

        next_inflow_2 = 0.5 * (
            next_inflow + next_inflow_2 + delta_p_2[..., 0]
        )
        delta_p2 = 0.5 * (delta_p_2[..., 1:] + delta_p[..., :-1])
        delta_p_point2 = 0.5 * (
            delta_p_point_2[..., 1:] + delta_p_point[..., :-1]
        )

        delta_p = delta_p.at[..., :-1].set(delta_p2)
        delta_p_point = delta_p_point.at[..., :-1].set(delta_p_point2)

        p = _update_p(p, delta_p, next_inflow_2, step_size)
        p_point = _update_p_point(p_point, delta_p_point, step_size)

        next_carry = (p, p_point)

        history = {
            "probability": prob_callback(p, p_point),
        }

        return next_carry, history

    scan_grid = jnp.swapaxes(grid[..., :-1], 0, -1)

    _, result = jax.lax.scan(heun_scan, (p_0, p_point_0), scan_grid)

    init_prob_callback_value = prob_callback(p_0, p_point_0)
    result["probability"] = jax.tree_util.tree_map(
        lambda arr, init: jnp.concatenate(
            [jnp.expand_dims(init, axis=0), arr]
        ),
        result["probability"],
        init_prob_callback_value,
    )

    return result


# -------------------------------------------------------------------- #
# Result transposition                                                  #
# -------------------------------------------------------------------- #


@jax.jit
def _transpose_probability(x):
    """Move the time axis to a natural position.

    The scan produces time as the leading axis. This moves it so that
    the result is indexed as ``(batch, ..., time)`` or similar.
    """
    N = x.ndim
    if N == 1:
        return x
    if N == 2:
        return jnp.transpose(x, axes=(1, 0))
    return jnp.moveaxis(x, 0, -2)


# -------------------------------------------------------------------- #
# Public solver interface                                               #
# -------------------------------------------------------------------- #


def _get_reference_function(intensity_matrix):
    """Find the first non-None callable in the intensity matrix."""
    for row in intensity_matrix:
        for f in row:
            if f is not None:
                return f
    return None


def solve(
    model,
    initial: str,
    horizon: int,
    steps_per_unit: int,
    callback: Union[None, str, Callable] = "collapse_point_no_duration",
    perturbation: float = 1e-12,
    transpose_result: bool = True,
    **kwargs,
) -> dict:
    """Compute transition probabilities from a given initial state.

    The solver reduces the model to the subgraph reachable from
    ``initial``, so only relevant states are computed. The initial
    state is always at index 0 in the reduced system.

    Parameters
    ----------
    model : Model
        The multi-state model with intensity callables.
    initial : str
        The starting state name.
    horizon : int
        Number of time units to solve over.
    steps_per_unit : int
        Discretization resolution per time unit.
    callback : None, str, or callable, optional
        Probability callback. Default is ``'collapse_point_no_duration'``.
    perturbation : float, optional
        Grid perturbation for finite differences. Default is ``1e-12``.
    transpose_result : bool, optional
        Whether to transpose the time axis in the result. Default is True.
    **kwargs
        Covariate arrays, each of shape ``(batch, ...)``.

    Returns
    -------
    dict
        Result dictionary with:
        - ``'probability'``: transition probabilities at each time step.
        - ``'states'``: tuple of state names in the order they appear
          in the probability arrays (initial state first).
    """
    # Reduce to reachable subgraph
    reduced = model.reduce(initial)
    intensity_matrix = reduced.solver_matrix
    n_states = reduced.n_states

    solver_steps = steps_per_unit * horizon
    grid = jnp.linspace(0, horizon, solver_steps + 1, endpoint=True)
    grid = jnp.expand_dims(grid, 0)
    step_size = 1 / steps_per_unit

    prob_callback = resolve_callback(callback)

    # Determine batch size from a reference callable
    reference_fn = _get_reference_function(intensity_matrix)
    if reference_fn is None:
        raise ValueError(
            "The intensity matrix contains no callables. "
            "Cannot determine batch size."
        )
    dummy = reference_fn(0, grid, **kwargs)
    batch_size = dummy.shape[0]

    # Initial conditions: everyone starts in the initial state (index 0)
    p_point_0 = jnp.zeros((batch_size, solver_steps))
    p_0 = jnp.zeros((batch_size, n_states, solver_steps))
    p_point_0 = p_point_0.at[..., 0].set(1)

    result = _heun_solver(
        p_0,
        p_point_0,
        grid,
        step_size,
        intensity_matrix,
        kwargs,
        prob_callback,
        perturbation,
    )

    if transpose_result:
        result["probability"] = jax.tree_util.tree_map(
            _transpose_probability, result["probability"]
        )

    result["states"] = reduced.reachable_states

    return result
