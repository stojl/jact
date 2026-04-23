"""Semi-Markov solver with duration-dependent transition intensities."""

from __future__ import annotations

from functools import partial
from typing import Any, Callable, Dict, Optional, Sequence, Union

import jax
import jax.numpy as jnp

from .callbacks import PointMass, StateCarry, resolve_callback
from .initial_distribution import InitialDistribution

def _state_has_point_masses(state: tuple[StateCarry, ...]) -> bool:
    return any(carry.point_mass is not None for carry in state)


def _stack_state_densities(state: tuple[StateCarry, ...]) -> jnp.ndarray:
    return jnp.stack(tuple(carry.density for carry in state), axis=0)


def _stack_point_masses(
    state: tuple[StateCarry, ...],
) -> tuple[jnp.ndarray, jnp.ndarray, tuple[bool, ...]]:
    value_template = state[0].density[:, 0]
    values = []
    d_0 = []
    mask = []
    for carry in state:
        if carry.point_mass is None:
            values.append(jnp.zeros_like(value_template))
            d_0.append(jnp.zeros_like(value_template))
            mask.append(False)
        else:
            values.append(carry.point_mass.value)
            d_0.append(carry.point_mass.d_0)
            mask.append(True)
    return jnp.stack(values, axis=0), jnp.stack(d_0, axis=0), tuple(mask)


def _dense_state_to_tuple(
    densities: jnp.ndarray,
    point_values: jnp.ndarray | None,
    point_d_0: jnp.ndarray | None,
    point_mask: tuple[bool, ...],
) -> tuple[StateCarry, ...]:
    state = []
    for i, has_point_mass in enumerate(point_mask):
        point_mass = None
        if has_point_mass:
            point_mass = PointMass(value=point_values[i], d_0=point_d_0[i])
        state.append(StateCarry(density=densities[i], point_mass=point_mass))
    return tuple(state)


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

    evolved = density + step_size * delta
    density_next = evolved.at[..., 1:-1].set(evolved[..., :-2])
    density_next = density_next.at[..., -1].add(evolved[..., -2])
    density_next = density_next.at[..., 0].set(step_size * next_inflow)
    return density_next
def _evaluate_intensity_at_point(
    fn: Callable[..., jnp.ndarray],
    t: jnp.ndarray,
    d_per_individual: jnp.ndarray,
    intensity_kwargs: Dict[str, jnp.ndarray],
) -> jnp.ndarray:
    """Evaluate a point intensity, preferring a batched call shape."""
    d_batched = d_per_individual[:, None]
    batched_result = fn(t, d_batched, **intensity_kwargs)
    batched_result = jnp.asarray(batched_result)

    if (
        batched_result.ndim >= 2
        and batched_result.shape[0] == d_per_individual.shape[0]
        and batched_result.shape[-1] == 1
    ):
        return jnp.squeeze(batched_result, axis=-1)

    names = tuple(intensity_kwargs.keys())
    values = tuple(intensity_kwargs[name] for name in names)

    def eval_one(d_i, *covariates):
        kwargs = {
            name: jnp.expand_dims(value, axis=0)
            for name, value in zip(names, covariates)
        }
        return jnp.squeeze(fn(t, d_i[None, None], **kwargs), axis=(0, 1))

    return jax.vmap(
        eval_one,
        in_axes=(0,) + (0,) * len(values),
    )(d_per_individual, *values)


def _compute_density_derivative_no_points(
    densities: jnp.ndarray,
    mu_plus_matrix,
    mu_minus_matrix,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute inflows and density derivatives for the no-point-mass case."""
    outflow_avg = jnp.zeros_like(densities)
    next_inflow = jnp.zeros(densities.shape[:-1], dtype=densities.dtype)

    for i, row in enumerate(mu_plus_matrix):
        for j, mu_plus in enumerate(row):
            if mu_plus is None:
                continue

            mu_avg = 0.5 * (mu_plus[..., :-1] + mu_minus_matrix[i][j][..., 1:])
            outflow_avg = outflow_avg.at[i].add(mu_avg)
            next_inflow = next_inflow.at[j].add(
                jnp.sum(mu_avg * densities[i], axis=-1)
            )

    delta_density = -densities * outflow_avg
    return next_inflow, delta_density


def _compute_dense_derivative_with_points(
    densities: jnp.ndarray,
    point_values: jnp.ndarray,
    point_d_0: jnp.ndarray,
    point_mask: tuple[bool, ...],
    intensity: Sequence[Sequence[Optional[Callable[..., jnp.ndarray]]]],
    t: jnp.ndarray,
    mu_plus_matrix,
    mu_minus_matrix,
    intensity_kwargs: Dict[str, jnp.ndarray],
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute inflows and derivatives for a dense state representation."""
    outflow_avg = jnp.zeros_like(densities)
    outflow_plus_point = jnp.zeros_like(point_values)
    next_inflow = jnp.zeros(densities.shape[:-1], dtype=densities.dtype)

    for i, row in enumerate(mu_plus_matrix):
        point_duration = point_d_0[i] + t if point_mask[i] else None
        for j, mu_plus in enumerate(row):
            if mu_plus is None:
                continue

            mu_avg = 0.5 * (mu_plus[..., :-1] + mu_minus_matrix[i][j][..., 1:])
            outflow_avg = outflow_avg.at[i].add(mu_avg)
            next_inflow = next_inflow.at[j].add(
                jnp.sum(mu_avg * densities[i], axis=-1)
            )

            if point_mask[i]:
                mu_at_point = _evaluate_intensity_at_point(
                    intensity[i][j],
                    t,
                    point_duration,
                    intensity_kwargs,
                )
                outflow_plus_point = outflow_plus_point.at[i].add(mu_at_point)
                next_inflow = next_inflow.at[j].add(mu_at_point * point_values[i])

    delta_density = -densities * outflow_avg
    delta_point = -point_values * outflow_plus_point
    return next_inflow, delta_density, delta_point
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
    return _heun_step_dense(
        state,
        t,
        grid,
        step_size,
        intensity,
        intensity_kwargs,
        perturbation,
        include_point_masses=_state_has_point_masses(state),
    )


def _heun_step_dense(
    state: tuple[StateCarry, ...],
    t: jnp.ndarray,
    grid: jnp.ndarray,
    step_size: float,
    intensity: Sequence[Sequence[Optional[Callable[..., jnp.ndarray]]]],
    intensity_kwargs: Dict[str, jnp.ndarray],
    perturbation: jnp.ndarray,
    include_point_masses: bool,
) -> tuple[StateCarry, ...]:
    """Advance a state tuple through a dense stacked representation."""
    densities = _stack_state_densities(state)
    point_values = None
    point_d_0 = None
    point_mask = tuple(False for _ in state)
    if include_point_masses:
        point_values, point_d_0, point_mask = _stack_point_masses(state)

    grid_minus = grid - perturbation
    grid_plus = grid + perturbation
    t_left = t + perturbation

    mu_plus = _evaluate_intensities(
        intensity, t_left, grid_plus, **intensity_kwargs
    )
    mu_minus = _evaluate_intensities(
        intensity, t_left, grid_minus, **intensity_kwargs
    )
    if include_point_masses:
        next_inflow, delta_density, delta_point = (
            _compute_dense_derivative_with_points(
                densities,
                point_values,
                point_d_0,
                point_mask,
                intensity,
                t_left,
                mu_plus,
                mu_minus,
                intensity_kwargs,
            )
        )
    else:
        next_inflow, delta_density = _compute_density_derivative_no_points(
            densities,
            mu_plus,
            mu_minus,
        )
        delta_point = None

    predictor = _update_density(
        densities,
        delta_density,
        next_inflow,
        step_size,
    )
    predictor_point = None
    if include_point_masses:
        predictor_point = point_values + step_size * delta_point

    t_right = t + step_size - perturbation
    mu_plus_2 = _evaluate_intensities(
        intensity, t_right, grid_plus, **intensity_kwargs
    )
    mu_minus_2 = _evaluate_intensities(
        intensity, t_right, grid_minus, **intensity_kwargs
    )
    if include_point_masses:
        next_inflow_2, delta_density_2, delta_point_2 = (
            _compute_dense_derivative_with_points(
                predictor,
                predictor_point,
                point_d_0,
                point_mask,
                intensity,
                t_right,
                mu_plus_2,
                mu_minus_2,
                intensity_kwargs,
            )
        )
    else:
        next_inflow_2, delta_density_2 = _compute_density_derivative_no_points(
            predictor,
            mu_plus_2,
            mu_minus_2,
        )
        delta_point_2 = None

    corrected_inflow = 0.5 * (
        next_inflow + next_inflow_2 + delta_density_2[..., 0]
    )
    corrected_density = delta_density.at[..., :-1].set(
        0.5 * (delta_density_2[..., 1:] + delta_density[..., :-1])
    )
    corrected = _update_density(
        densities,
        corrected_density,
        corrected_inflow,
        step_size,
    )
    corrected_point = None
    if include_point_masses:
        corrected_point = point_values + step_size * 0.5 * (
            delta_point + delta_point_2
        )
    return _dense_state_to_tuple(
        corrected,
        corrected_point,
        point_d_0,
        point_mask,
    )


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
) -> PointMass:
    return PointMass(
        value=_broadcast_batch(mass, batch_size),
        d_0=_broadcast_batch(duration, batch_size),
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
