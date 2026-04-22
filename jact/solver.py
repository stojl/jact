"""Semi-Markov solver with duration-dependent transition intensities."""

from __future__ import annotations

from functools import partial
from typing import Any, Callable, Dict, Optional, Sequence, Union

import jax
import jax.numpy as jnp

from .callbacks import StateCarry, resolve_callback
from .initial_distribution import InitialDistribution


def _zero_batch_like(state: StateCarry) -> jnp.ndarray:
    return jnp.zeros(state.density.shape[:-1], dtype=state.density.dtype)


def _update_density(
    density: jnp.ndarray,
    delta: jnp.ndarray,
    next_inflow: jnp.ndarray,
    step_size: float,
) -> jnp.ndarray:
    if density.shape[-1] == 1:
        return density.at[..., 0].set(
            density[..., 0] + step_size * delta[..., 0] + step_size * next_inflow
        )

    density_next = density.at[..., 1:-1].set(
        density[..., :-2] + step_size * delta[..., :-2]
    )
    density_next = density_next.at[..., -1].set(
        density[..., -2]
        + step_size * delta[..., -2]
        + density[..., -1]
        + step_size * delta[..., -1]
    )
    density_next = density_next.at[..., 0].set(step_size * next_inflow)
    return density_next


def _update_point_mass(
    point_mass: jnp.ndarray | None,
    delta: jnp.ndarray | None,
    step_size: float,
) -> jnp.ndarray | None:
    if point_mass is None or delta is None:
        return None
    if point_mass.shape[-1] == 1:
        return point_mass.at[..., 0].set(
            point_mass[..., 0] + step_size * delta[..., 0]
        )

    point_mass_next = point_mass.at[..., 1:-1].set(
        point_mass[..., :-2] + step_size * delta[..., :-2]
    )
    point_mass_next = point_mass_next.at[..., -1].set(
        point_mass[..., -2]
        + step_size * delta[..., -2]
        + point_mass[..., -1]
        + step_size * delta[..., -1]
    )
    point_mass_next = point_mass_next.at[..., 0].set(0.0)
    return point_mass_next


def _compute_derivative(
    state: tuple[StateCarry, ...],
    mu_plus_matrix,
    mu_minus_matrix,
):
    """Compute per-state inflows and derivatives."""
    outflow_plus = [jnp.zeros_like(carry.density) for carry in state]
    outflow_avg = [jnp.zeros_like(carry.density) for carry in state]
    next_inflow = [_zero_batch_like(carry) for carry in state]

    for i, carry_i in enumerate(state):
        for j, _ in enumerate(state):
            mu_plus = mu_plus_matrix[i][j]
            if mu_plus is None:
                continue

            mu_minus = mu_minus_matrix[i][j]
            mu_plus_slice = mu_plus[..., :-1]
            mu_avg = 0.5 * (mu_plus[..., :-1] + mu_minus[..., 1:])

            outflow_plus[i] = outflow_plus[i] + mu_plus_slice
            outflow_avg[i] = outflow_avg[i] + mu_avg
            next_inflow[j] = next_inflow[j] + jnp.sum(
                mu_avg * carry_i.density,
                axis=-1,
            )

            if carry_i.point_mass is not None:
                next_inflow[j] = next_inflow[j] + jnp.sum(
                    mu_plus_slice * carry_i.point_mass,
                    axis=-1,
                )

    delta_state = []
    for carry, outflow_plus_i, outflow_avg_i in zip(
        state,
        outflow_plus,
        outflow_avg,
    ):
        delta_density = -carry.density * outflow_avg_i
        delta_point_mass = None
        if carry.point_mass is not None:
            delta_point_mass = -carry.point_mass * outflow_plus_i
        delta_state.append(StateCarry(delta_density, delta_point_mass))

    return tuple(next_inflow), tuple(delta_state)


def _evaluate_intensities(matrix, *args, **kwargs):
    """Evaluate every intensity callable in a solver matrix."""
    return jax.tree_util.tree_map(lambda f: f(*args, **kwargs), matrix)


def _heun_step(
    state: tuple[StateCarry, ...],
    t: jnp.ndarray,
    grid: jnp.ndarray,
    step_size: float,
    intensity: Sequence[Sequence[Optional[Callable[..., jnp.ndarray]]]],
    intensity_kwargs: Dict[str, jnp.ndarray],
    perturbation: jnp.ndarray,
) -> tuple[StateCarry, ...]:
    """Advance the full solver state by one time step."""
    grid_minus = grid - perturbation
    grid_plus = grid + perturbation
    t_left = t + perturbation

    mu_plus = _evaluate_intensities(
        intensity, t_left, grid_plus, **intensity_kwargs
    )
    mu_minus = _evaluate_intensities(
        intensity, t_left, grid_minus, **intensity_kwargs
    )

    next_inflow, delta_state = _compute_derivative(state, mu_plus, mu_minus)

    predictor = tuple(
        StateCarry(
            density=_update_density(
                carry.density,
                delta.density,
                inflow,
                step_size,
            ),
            point_mass=_update_point_mass(
                carry.point_mass,
                delta.point_mass,
                step_size,
            ),
        )
        for carry, delta, inflow in zip(state, delta_state, next_inflow)
    )

    t_right = t + step_size - perturbation
    mu_plus_2 = _evaluate_intensities(
        intensity, t_right, grid_plus, **intensity_kwargs
    )
    mu_minus_2 = _evaluate_intensities(
        intensity, t_right, grid_minus, **intensity_kwargs
    )

    next_inflow_2, delta_state_2 = _compute_derivative(
        predictor, mu_plus_2, mu_minus_2
    )

    corrected_state = []
    for inflow_1, inflow_2, delta_1, delta_2, carry in zip(
        next_inflow,
        next_inflow_2,
        delta_state,
        delta_state_2,
        state,
    ):
        corrected_inflow = 0.5 * (
            inflow_1 + inflow_2 + delta_2.density[..., 0]
        )
        corrected_density = delta_1.density.at[..., :-1].set(
            0.5 * (
                delta_2.density[..., 1:] + delta_1.density[..., :-1]
            )
        )

        corrected_point_mass = delta_1.point_mass
        if corrected_point_mass is not None and delta_2.point_mass is not None:
            corrected_point_mass = corrected_point_mass.at[..., :-1].set(
                0.5
                * (
                    delta_2.point_mass[..., 1:]
                    + corrected_point_mass[..., :-1]
                )
            )

        corrected_state.append(
            StateCarry(
                density=_update_density(
                    carry.density,
                    corrected_density,
                    corrected_inflow,
                    step_size,
                ),
                point_mass=_update_point_mass(
                    carry.point_mass,
                    corrected_point_mass,
                    step_size,
                ),
            )
        )

    return tuple(corrected_state)


@partial(
    jax.jit,
    static_argnames=[
        "step_size",
        "intensity",
        "prob_callback",
        "perturbation",
        "record_every",
    ],
)
def _heun_solver(
    state_0: tuple[StateCarry, ...],
    grid: jnp.ndarray,
    step_size: float,
    intensity: Sequence[Sequence[Optional[Callable[..., jnp.ndarray]]]],
    intensity_kwargs: Dict[str, jnp.ndarray],
    prob_callback: Callable[..., Any],
    perturbation: jnp.ndarray,
    record_every: int,
):
    """Run the Heun scheme solver and record callback outputs."""
    n_steps = grid.shape[-1] - 1
    n_records = n_steps // record_every

    def block_scan(carry, block_start):
        offsets = jnp.arange(record_every, dtype=grid.dtype)

        def step_scan(inner_carry, offset):
            current_t = block_start + offset * step_size
            return (
                _heun_step(
                    inner_carry,
                    current_t,
                    grid,
                    step_size,
                    intensity,
                    intensity_kwargs,
                    perturbation,
                ),
                None,
            )

        carry, _ = jax.lax.scan(step_scan, carry, offsets)
        return carry, prob_callback(carry)

    initial_probability = prob_callback(state_0)
    block_starts = jnp.arange(n_records, dtype=grid.dtype) * (
        record_every * step_size
    )
    _, probability = jax.lax.scan(block_scan, state_0, block_starts)

    probability = jax.tree_util.tree_map(
        lambda arr, init: (
            None
            if init is None
            else jnp.concatenate([jnp.expand_dims(init, axis=0), arr], axis=0)
        ),
        probability,
        initial_probability,
    )

    return {"probability": probability}


def _get_reference_function(intensity_matrix):
    """Find the first non-None callable in the intensity matrix."""
    for row in intensity_matrix:
        for fn in row:
            if fn is not None:
                return fn
    return None


def _get_covariate_batch_size(kwargs: Dict[str, Any]) -> int | None:
    batch_size = None
    for name, value in kwargs.items():
        shape = jnp.shape(value)
        if len(shape) == 0:
            raise ValueError(
                f"Covariate '{name}' must have shape (batch, ...), got scalar."
            )
        if batch_size is None:
            batch_size = shape[0]
        elif batch_size != shape[0]:
            raise ValueError("Covariate batch dimensions must match.")
    return batch_size


def _broadcast_batch(value: Any, batch_size: int) -> jnp.ndarray:
    arr = jnp.asarray(value)
    if arr.ndim == 0:
        return jnp.broadcast_to(arr, (batch_size,))
    if arr.ndim == 1 and arr.shape[0] == batch_size:
        return arr
    raise ValueError("Expected a scalar or (batch,) array.")


def _canonicalize_initial(
    initial: Union[str, jnp.ndarray, InitialDistribution],
    initial_duration: Any,
) -> InitialDistribution:
    if isinstance(initial, InitialDistribution):
        try:
            has_nonzero_duration = bool(jnp.any(jnp.asarray(initial_duration) != 0.0))
        except Exception:
            has_nonzero_duration = initial_duration != 0.0
        if has_nonzero_duration:
            raise ValueError(
                "initial_duration is invalid when initial is an "
                "InitialDistribution."
            )
        return initial
    if isinstance(initial, str):
        return InitialDistribution.at(initial, duration=initial_duration)
    return InitialDistribution.per_individual(
        states=initial,
        duration=initial_duration,
        initial_states=None,
    )


def _seed_point_mass(
    mass: Any,
    duration: Any,
    batch_size: int,
    duration_slots: int,
    steps_per_unit: int,
) -> jnp.ndarray:
    mass_arr = _broadcast_batch(mass, batch_size)
    duration_arr = _broadcast_batch(duration, batch_size)
    indices = jnp.rint(duration_arr * steps_per_unit).astype(jnp.int32)
    indices = jnp.clip(indices, 0, max(duration_slots - 1, 0))
    return jax.nn.one_hot(indices, duration_slots, dtype=mass_arr.dtype) * (
        mass_arr[:, None]
    )


def solve(
    model: Any,
    initial: Union[str, jnp.ndarray, InitialDistribution],
    horizon: int,
    steps_per_unit: int,
    initial_duration: Any = 0.0,
    callback: Union[None, str, Callable] = "collapse_point_no_duration",
    record_every: int = 1,
    perturbation: float = 1e-12,
    **kwargs: Any,
) -> dict[str, Any]:
    """Compute transition probabilities from a documented initial condition."""
    reserved = {"initial", "initial_duration"}
    overlap = reserved.intersection(kwargs)
    if overlap:
        names = ", ".join(sorted(overlap))
        raise ValueError(f"Reserved covariate names are not allowed: {names}")

    solver_steps = steps_per_unit * horizon
    if record_every <= 0 or solver_steps % record_every != 0:
        raise ValueError(
            "record_every must be a positive integer dividing "
            "horizon * steps_per_unit."
        )

    initial_distribution = _canonicalize_initial(initial, initial_duration)
    model_states = model.state_space.states
    initial_distribution.validate_for_model(model_states)
    canonical = initial_distribution.canonicalize(model_states)

    reduced = model.reduce(canonical.states)
    intensity_matrix = tuple(
        tuple(row) for row in reduced.solver_matrix
    )
    grid = jnp.linspace(0, horizon, solver_steps + 1, endpoint=True)[None, :]
    step_size = 1 / steps_per_unit
    prob_callback = resolve_callback(callback)

    distribution_batch = canonical.batch_size
    covariate_batch = _get_covariate_batch_size(kwargs)
    if (
        distribution_batch is not None
        and covariate_batch is not None
        and distribution_batch != covariate_batch
    ):
        raise ValueError(
            "InitialDistribution batch size must match covariate batch size."
        )

    reference_fn = _get_reference_function(intensity_matrix)
    if reference_fn is None:
        raise ValueError(
            "The intensity matrix contains no callables. Cannot solve."
        )

    reference_output = reference_fn(0.0, grid, **kwargs)
    reference_batch = reference_output.shape[0]

    batch_size = distribution_batch
    if batch_size is None:
        batch_size = covariate_batch
    if batch_size is None:
        batch_size = reference_batch
    if reference_batch != batch_size:
        raise ValueError("Intensity batch size must match solver batch size.")

    declared_index = {
        state: i for i, state in enumerate(canonical.states)
    }
    state_0 = []
    for state_name in reduced.reachable_states:
        density = jnp.zeros((batch_size, solver_steps), dtype=reference_output.dtype)
        point_mass = None
        if state_name in declared_index:
            idx = declared_index[state_name]
            point_mass = _seed_point_mass(
                canonical.masses[idx],
                canonical.durations[idx],
                batch_size,
                solver_steps,
                steps_per_unit,
            )
        state_0.append(StateCarry(density=density, point_mass=point_mass))

    result = _heun_solver(
        tuple(state_0),
        grid,
        step_size,
        intensity_matrix,
        kwargs,
        prob_callback,
        perturbation,
        record_every,
    )
    result["states"] = reduced.reachable_states
    return result
