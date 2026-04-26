"""Semi-Markov solver with midpoint quadrature."""

from __future__ import annotations

from functools import partial
from typing import Any, Callable, Dict, Sequence, Union

import jax
import jax.numpy as jnp

from .callbacks import PointMass, StateCarry, resolve_callback
from .initial_distribution import InitialDistribution


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
    point_values: jnp.ndarray,
    point_d_0: jnp.ndarray,
    point_mask: tuple[bool, ...],
) -> tuple[StateCarry, ...]:
    state = []
    for i, has_point_mass in enumerate(point_mask):
        point_mass = None
        if has_point_mass:
            point_mass = PointMass(value=point_values[i], d_0=point_d_0[i])
        state.append(StateCarry(density=densities[i], point_mass=point_mass))
    return tuple(state)


def _evaluate_intensity_at_point(
    fn: Callable[..., jnp.ndarray],
    t: jnp.ndarray,
    d_per_individual: jnp.ndarray,
    intensity_kwargs: Dict[str, jnp.ndarray],
) -> jnp.ndarray:
    """Evaluate a point intensity, preferring a batched call shape."""
    d_batched = d_per_individual[:, None]
    batched_result = jnp.asarray(fn(t, d_batched, **intensity_kwargs))

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


def _nonnegative(values: jnp.ndarray) -> jnp.ndarray:
    return jnp.maximum(values, jnp.zeros_like(values))


def _integrated_density_hazard(
    fn: Callable[..., jnp.ndarray],
    t: jnp.ndarray,
    duration_mid: jnp.ndarray,
    step_size: float,
    intensity_kwargs: Dict[str, jnp.ndarray],
) -> jnp.ndarray:
    midpoint = jnp.asarray(fn(t + 0.5 * step_size, duration_mid, **intensity_kwargs))
    hazard = step_size * midpoint
    return _nonnegative(hazard)


def _integrated_point_hazard(
    fn: Callable[..., jnp.ndarray],
    t: jnp.ndarray,
    point_d_0: jnp.ndarray,
    step_size: float,
    intensity_kwargs: Dict[str, jnp.ndarray],
) -> jnp.ndarray:
    midpoint = _evaluate_intensity_at_point(
        fn,
        t + 0.5 * step_size,
        point_d_0 + t + 0.5 * step_size,
        intensity_kwargs,
    )
    hazard = step_size * midpoint
    return _nonnegative(hazard)


def _stable_transfer_factor(total_hazard: jnp.ndarray) -> jnp.ndarray:
    safe_total = jnp.where(total_hazard > 0, total_hazard, 1.0)
    factor = -jnp.expm1(-total_hazard) / safe_total
    return jnp.where(total_hazard > 0, factor, 1.0)


def _advance_density(
    density: jnp.ndarray,
    survival: jnp.ndarray,
    inflow: jnp.ndarray,
) -> jnp.ndarray:
    if density.shape[-1] == 1:
        return density.at[..., 0].set(density[..., 0] * survival[..., 0] + inflow)

    next_density = jnp.zeros_like(density)
    next_density = next_density.at[..., 1:-1].set(
        density[..., :-2] * survival[..., :-2]
    )
    next_density = next_density.at[..., -1].set(
        density[..., -1] * survival[..., -1]
        + density[..., -2] * survival[..., -2]
    )
    next_density = next_density.at[..., 0].set(inflow)
    return next_density


def _solver_step(
    state: tuple[StateCarry, ...],
    t: jnp.ndarray,
    duration_mid: jnp.ndarray,
    step_size: float,
    solver_matrix: Sequence[Sequence[Callable[..., jnp.ndarray] | None]],
    intensity_kwargs: Dict[str, jnp.ndarray],
) -> tuple[StateCarry, ...]:
    densities = _stack_state_densities(state)
    point_values, point_d_0, point_mask = _stack_point_masses(state)
    next_inflow = jnp.zeros(densities.shape[:-1], dtype=densities.dtype)
    next_point_values = point_values
    next_densities = []

    for i, row in enumerate(solver_matrix):
        density_total = jnp.zeros_like(densities[i])
        point_total = jnp.zeros_like(point_values[i])
        density_hazards = []
        point_hazards = []

        for j, fn in enumerate(row):
            if fn is None:
                continue

            density_hazard = _integrated_density_hazard(
                fn,
                t,
                duration_mid,
                step_size,
                intensity_kwargs,
            )
            density_total = density_total + density_hazard
            density_hazards.append((j, density_hazard))

            if point_mask[i]:
                point_hazard = _integrated_point_hazard(
                    fn,
                    t,
                    point_d_0[i],
                    step_size,
                    intensity_kwargs,
                )
                point_total = point_total + point_hazard
                point_hazards.append((j, point_hazard))

        density_survival = jnp.exp(-density_total)
        density_transfer_factor = _stable_transfer_factor(density_total)
        for j, density_hazard in density_hazards:
            transferred = density_hazard * density_transfer_factor
            next_inflow = next_inflow.at[j].add(
                jnp.sum(densities[i] * transferred, axis=-1)
            )

        if point_mask[i]:
            point_survival = jnp.exp(-point_total)
            point_transfer_factor = _stable_transfer_factor(point_total)
            next_point_values = next_point_values.at[i].set(
                point_values[i] * point_survival
            )
            for j, point_hazard in point_hazards:
                next_inflow = next_inflow.at[j].add(
                    point_values[i] * point_hazard * point_transfer_factor
                )

        next_densities.append(
            _advance_density(densities[i], density_survival, next_inflow[i])
        )

    return _dense_state_to_tuple(
        jnp.stack(tuple(next_densities), axis=0),
        next_point_values,
        point_d_0,
        point_mask,
    )


@partial(
    jax.jit,
    static_argnames=[
        "step_size",
        "solver_matrix",
        "prob_callback",
        "record_every",
    ],
)
def _midpoint_solver(
    state_0: tuple[StateCarry, ...],
    duration_mid: jnp.ndarray,
    step_size: float,
    solver_matrix: Sequence[Sequence[Callable[..., jnp.ndarray] | None]],
    intensity_kwargs: Dict[str, jnp.ndarray],
    prob_callback: Callable[..., Any],
    record_every: int,
):
    """Run the midpoint solver and record callback outputs."""
    n_steps = duration_mid.shape[-1]
    n_records = n_steps // record_every

    def block_scan(carry, block_start):
        offsets = jnp.arange(record_every, dtype=duration_mid.dtype)

        def step_scan(inner_carry, offset):
            current_t = block_start + offset * step_size
            next_state = _solver_step(
                inner_carry,
                current_t,
                duration_mid,
                step_size,
                solver_matrix,
                intensity_kwargs,
            )
            return next_state, None

        carry, _ = jax.lax.scan(step_scan, carry, offsets)
        return carry, prob_callback(carry)

    initial_probability = prob_callback(state_0)
    block_starts = jnp.arange(n_records, dtype=duration_mid.dtype) * (
        record_every * step_size
    )
    _, probability = jax.lax.scan(
        block_scan,
        state_0,
        block_starts,
    )

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


def _get_reference_function(solver_matrix):
    """Find the first non-None callable in the solver matrix."""
    for row in solver_matrix:
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
    solver_matrix = reduced.solver_matrix
    grid = jnp.linspace(0, horizon, solver_steps + 1, endpoint=True)[None, :]
    duration_left = grid[:, :-1]
    duration_mid = 0.5 * (duration_left + grid[:, 1:])
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

    reference_fn = _get_reference_function(solver_matrix)
    if reference_fn is None:
        raise ValueError(
            "The intensity matrix contains no callables. Cannot solve."
        )

    reference_output = reference_fn(0.0, duration_left, **kwargs)
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
    result = _midpoint_solver(
        tuple(state_0),
        duration_mid,
        step_size,
        solver_matrix,
        kwargs,
        prob_callback,
        record_every,
    )
    result["states"] = reduced.reachable_states
    return result
