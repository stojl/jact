# pyright: strict, reportMissingImports=false, reportUnknownMemberType=false, reportPrivateUsage=false
"""JAX-native event simulation for discretized semi-Markov models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from numbers import Integral
from typing import Any, Literal, NamedTuple, TypeAlias, cast

import jax
import jax.numpy as jnp
import numpy as np

from .initial_distribution import InitialDistribution, _CanonicalDistribution
from .model import Model
from .simulation_result import SimulationResult
from .typing import ArrayLike, GroupedIntensity, Intensity

__all__ = ["simulate"]

_DType: TypeAlias = np.dtype[np.generic]


@dataclass(frozen=True)
class IntensityBlock:
    """One original callable and its canonical simulation-edge outputs."""

    fn: Intensity | GroupedIntensity
    edge_ids: tuple[int, ...]
    output_indices: tuple[int | None, ...]


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SimulationPlan:
    """Immutable sparse topology and grouped intensity evaluation plan."""

    initial_states: tuple[str, ...]
    states: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    edge_sources: jax.Array
    edge_targets: jax.Array
    outgoing_edge_ids: jax.Array
    outgoing_targets: jax.Array
    outgoing_valid: jax.Array
    intensity_blocks: tuple[IntensityBlock, ...]
    n_states: int
    n_edges: int
    max_out_degree: int

    def tree_flatten(
        self,
    ) -> tuple[
        tuple[jax.Array, ...],
        tuple[
            tuple[str, ...],
            tuple[str, ...],
            tuple[tuple[str, str], ...],
            tuple[IntensityBlock, ...],
            int,
            int,
            int,
        ],
    ]:
        children = (
            self.edge_sources,
            self.edge_targets,
            self.outgoing_edge_ids,
            self.outgoing_targets,
            self.outgoing_valid,
        )
        aux = (
            self.initial_states,
            self.states,
            self.edges,
            self.intensity_blocks,
            self.n_states,
            self.n_edges,
            self.max_out_degree,
        )
        return children, aux

    @classmethod
    def tree_unflatten(
        cls,
        aux: tuple[
            tuple[str, ...],
            tuple[str, ...],
            tuple[tuple[str, str], ...],
            tuple[IntensityBlock, ...],
            int,
            int,
            int,
        ],
        children: tuple[jax.Array, ...],
    ) -> SimulationPlan:
        (
            initial_states,
            states,
            edges,
            intensity_blocks,
            n_states,
            n_edges,
            max_out_degree,
        ) = aux
        (
            edge_sources,
            edge_targets,
            outgoing_edge_ids,
            outgoing_targets,
            outgoing_valid,
        ) = children
        return cls(
            initial_states=initial_states,
            states=states,
            edges=edges,
            edge_sources=edge_sources,
            edge_targets=edge_targets,
            outgoing_edge_ids=outgoing_edge_ids,
            outgoing_targets=outgoing_targets,
            outgoing_valid=outgoing_valid,
            intensity_blocks=intensity_blocks,
            n_states=n_states,
            n_edges=n_edges,
            max_out_degree=max_out_degree,
        )


class _SimulationCarry(NamedTuple):
    current_state: jax.Array
    current_time: jax.Array
    current_duration: jax.Array
    remaining_hazard: jax.Array
    jump_count: jax.Array
    overflow: jax.Array
    finished: jax.Array
    random_key: jax.Array
    jump_times: jax.Array
    jump_durations: jax.Array
    state_path: jax.Array
    truncated_at_time: jax.Array
    invalid_negative: jax.Array
    invalid_nonfinite: jax.Array


def _compile_simulation_plan(
    model: Model,
    initial_states: Sequence[str],
) -> SimulationPlan:
    reduced = model.reduce(initial_states)
    states = reduced.reachable_states
    state_index = {state: index for index, state in enumerate(states)}
    assignments = model._simulation_assignments(states)
    edges = tuple((info.source, info.target) for info in assignments)

    edge_sources = jnp.asarray(
        [state_index[source] for source, _target in edges],
        dtype=jnp.int32,
    )
    edge_targets = jnp.asarray(
        [state_index[target] for _source, target in edges],
        dtype=jnp.int32,
    )

    outgoing_lists: list[list[int]] = [[] for _state in states]
    for edge_id, source in enumerate(np.asarray(edge_sources).tolist()):
        outgoing_lists[int(source)].append(edge_id)
    max_out_degree = max(1, *(len(values) for values in outgoing_lists))
    outgoing_edge_ids = np.zeros(
        (len(states), max_out_degree),
        dtype=np.int32,
    )
    outgoing_targets = np.zeros_like(outgoing_edge_ids)
    outgoing_valid = np.zeros_like(outgoing_edge_ids, dtype=np.bool_)
    edge_target_values = np.asarray(edge_targets)
    for source, edge_ids in enumerate(outgoing_lists):
        for slot, edge_id in enumerate(edge_ids):
            outgoing_edge_ids[source, slot] = edge_id
            outgoing_targets[source, slot] = edge_target_values[edge_id]
            outgoing_valid[source, slot] = True

    block_entries: list[
        tuple[Intensity | GroupedIntensity, list[int], list[int | None]]
    ] = []
    block_lookup: dict[tuple[int, bool], int] = {}
    for edge_id, info in enumerate(assignments):
        # Keep scalar and leading-axis interpretations separate even if a user
        # deliberately reuses one callable object in both assignment styles.
        key = (id(info.callable), info.index is None)
        block_index = block_lookup.get(key)
        if block_index is None:
            block_index = len(block_entries)
            block_lookup[key] = block_index
            block_entries.append((info.callable, [], []))
        block_entries[block_index][1].append(edge_id)
        block_entries[block_index][2].append(info.index)

    intensity_blocks = tuple(
        IntensityBlock(
            fn=fn,
            edge_ids=tuple(edge_ids),
            output_indices=tuple(output_indices),
        )
        for fn, edge_ids, output_indices in block_entries
    )
    return SimulationPlan(
        initial_states=reduced.initial_states,
        states=states,
        edges=edges,
        edge_sources=edge_sources,
        edge_targets=edge_targets,
        outgoing_edge_ids=jnp.asarray(outgoing_edge_ids),
        outgoing_targets=jnp.asarray(outgoing_targets),
        outgoing_valid=jnp.asarray(outgoing_valid),
        intensity_blocks=intensity_blocks,
        n_states=len(states),
        n_edges=len(edges),
        max_out_degree=max_out_degree,
    )


def _broadcast_grid_output(
    value: ArrayLike,
    target_shape: tuple[int, int],
    label: str,
) -> jax.Array:
    arr = jnp.asarray(value)
    if arr.ndim == 1 and arr.shape[0] == target_shape[0]:
        arr = arr[:, None]
    try:
        return jnp.broadcast_to(arr, target_shape)
    except ValueError as exc:
        raise ValueError(
            f"{label} with shape {arr.shape} cannot broadcast to "
            f"{target_shape}."
        ) from exc


def _evaluate_intensity_slab(
    plan: SimulationPlan,
    t_mid: jax.Array,
    duration_mid: jax.Array,
    intensity_kwargs: dict[str, jax.Array],
    batch_size: int,
    negative_tolerance: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    target_shape = (batch_size, duration_mid.shape[0])
    rates = jnp.zeros(
        (plan.n_edges, *target_shape),
        dtype=duration_mid.dtype,
    )
    duration_grid = duration_mid[None, :]

    for block in plan.intensity_blocks:
        raw = jnp.asarray(block.fn(t_mid, duration_grid, **intensity_kwargs))
        for edge_id, output_index in zip(
            block.edge_ids,
            block.output_indices,
            strict=True,
        ):
            if output_index is None:
                selected = raw
            else:
                if raw.ndim == 0 or raw.shape[0] <= output_index:
                    available = 0 if raw.ndim == 0 else raw.shape[0]
                    raise ValueError(
                        "Multi-output assignment returned too few transition "
                        f"outputs: expected at least {output_index + 1}, "
                        f"got {available}."
                    )
                selected = raw[output_index]
            value = _broadcast_grid_output(
                selected,
                target_shape,
                "Intensity output",
            )
            rates = rates.at[edge_id].set(value)

    invalid_negative = jnp.any(rates < -negative_tolerance)
    invalid_nonfinite = jnp.any(~jnp.isfinite(rates))
    safe_rates = jnp.where(jnp.isfinite(rates), rates, 0.0)
    return jnp.maximum(safe_rates, 0.0), invalid_negative, invalid_nonfinite


def _split_keys(keys: jax.Array, count: int) -> jax.Array:
    return jax.vmap(lambda key: jax.random.split(key, count))(keys)


@partial(
    jax.jit,
    static_argnames=("batch_size", "max_jumps", "negative_tolerance"),
)
def _event_kernel(
    plan: SimulationPlan,
    trajectory_masses: jax.Array,
    trajectory_durations: jax.Array,
    initial_state_options: jax.Array,
    trajectory_keys: jax.Array,
    individual_indices: jax.Array,
    calendar_left: jax.Array,
    calendar_right: jax.Array,
    duration_mid: jax.Array,
    intensity_kwargs: dict[str, jax.Array],
    *,
    batch_size: int,
    max_jumps: int,
    negative_tolerance: float,
) -> _SimulationCarry:
    trajectory_count = trajectory_masses.shape[0]

    initial_keys = _split_keys(trajectory_keys, 3)
    totals = jnp.sum(trajectory_masses, axis=-1)
    draw = jax.vmap(jax.random.uniform)(initial_keys[:, 0])
    thresholds = draw * totals
    cumulative = jnp.cumsum(trajectory_masses, axis=-1)
    component = jnp.argmax(cumulative > thresholds[:, None], axis=-1)
    component = component.astype(jnp.int32)
    rows = jnp.arange(trajectory_count, dtype=jnp.int32)
    initial_state = initial_state_options[component]
    initial_duration = trajectory_durations[rows, component]
    remaining_hazard = jax.vmap(jax.random.exponential)(initial_keys[:, 1])
    value_dtype = duration_mid.dtype

    jump_times = jnp.full(
        (trajectory_count, max_jumps),
        jnp.nan,
        dtype=value_dtype,
    )
    jump_durations = jnp.full_like(jump_times, jnp.nan)
    state_path = jnp.full(
        (trajectory_count, max_jumps + 1),
        -1,
        dtype=jnp.int32,
    )
    state_path = state_path.at[:, 0].set(initial_state)
    carry = _SimulationCarry(
        current_state=initial_state,
        current_time=jnp.zeros((trajectory_count,), dtype=value_dtype),
        current_duration=initial_duration.astype(value_dtype),
        remaining_hazard=remaining_hazard.astype(value_dtype),
        jump_count=jnp.zeros((trajectory_count,), dtype=jnp.int32),
        overflow=jnp.zeros((trajectory_count,), dtype=jnp.bool_),
        finished=jnp.zeros((trajectory_count,), dtype=jnp.bool_),
        random_key=initial_keys[:, 2],
        jump_times=jump_times,
        jump_durations=jump_durations,
        state_path=state_path,
        truncated_at_time=jnp.full(
            (trajectory_count,),
            jnp.nan,
            dtype=value_dtype,
        ),
        invalid_negative=jnp.asarray(False),
        invalid_nonfinite=jnp.asarray(False),
    )
    step_size = calendar_right[0] - calendar_left[0]
    dtype_epsilon = jnp.finfo(value_dtype).eps

    def simulate_calendar_cell(
        scan_carry: _SimulationCarry,
        cell: tuple[jax.Array, jax.Array],
    ) -> tuple[_SimulationCarry, None]:
        t_left, t_right = cell
        rates, invalid_negative, invalid_nonfinite = _evaluate_intensity_slab(
            plan,
            0.5 * (t_left + t_right),
            duration_mid,
            intensity_kwargs,
            batch_size,
            negative_tolerance,
        )
        scan_carry = scan_carry._replace(
            invalid_negative=scan_carry.invalid_negative | invalid_negative,
            invalid_nonfinite=(
                scan_carry.invalid_nonfinite | invalid_nonfinite
            ),
        )

        def continue_cell(inner_carry: _SimulationCarry) -> jax.Array:
            active = (
                (inner_carry.current_time < t_right)
                & ~inner_carry.finished
            )
            return jnp.any(active)

        def advance_or_jump(inner_carry: _SimulationCarry) -> _SimulationCarry:
            active = (
                (inner_carry.current_time < t_right)
                & ~inner_carry.finished
            )
            duration_units = inner_carry.current_duration / step_size
            index_tolerance = (
                8.0
                * dtype_epsilon
                * jnp.maximum(1.0, jnp.abs(duration_units))
            )
            duration_index = jnp.floor(
                duration_units + index_tolerance
            ).astype(jnp.int32)
            safe_duration_index = jnp.clip(
                duration_index,
                0,
                duration_mid.shape[0] - 1,
            )
            duration_right = (
                duration_index.astype(value_dtype) + 1.0
            ) * step_size
            dt_duration = jnp.maximum(
                duration_right - inner_carry.current_duration,
                0.0,
            )
            dt_calendar = jnp.maximum(
                t_right - inner_carry.current_time,
                0.0,
            )
            dt_boundary = jnp.minimum(dt_duration, dt_calendar)

            adjacency_edges = plan.outgoing_edge_ids[
                inner_carry.current_state
            ]
            adjacency_targets = plan.outgoing_targets[
                inner_carry.current_state
            ]
            adjacency_valid = plan.outgoing_valid[
                inner_carry.current_state
            ]
            if plan.n_edges:
                outgoing_rates = rates[
                    adjacency_edges,
                    individual_indices[:, None],
                    safe_duration_index[:, None],
                ]
                outgoing_rates = jnp.where(
                    adjacency_valid,
                    outgoing_rates,
                    0.0,
                )
            else:
                outgoing_rates = jnp.zeros(
                    (trajectory_count, plan.max_out_degree),
                    dtype=value_dtype,
                )
            total_rate = jnp.sum(outgoing_rates, axis=-1)
            integrated_hazard = total_rate * dt_boundary
            wants_jump = (
                active
                & (total_rate > 0.0)
                & (inner_carry.remaining_hazard <= integrated_hazard)
            )
            has_room = inner_carry.jump_count < max_jumps
            overflows = wants_jump & ~has_room
            fires = wants_jump & has_room
            crosses = active & ~wants_jump

            next_time = inner_carry.current_time + dt_boundary
            crosses_calendar = dt_calendar <= dt_duration
            crossed_time = jnp.where(crosses_calendar, t_right, next_time)
            crossed_duration = jnp.where(
                dt_duration <= dt_calendar,
                duration_right,
                inner_carry.current_duration + dt_boundary,
            )
            safe_rate = jnp.where(total_rate > 0.0, total_rate, 1.0)
            event_delta = inner_carry.remaining_hazard / safe_rate
            event_time = inner_carry.current_time + event_delta
            event_duration = inner_carry.current_duration + event_delta

            event_keys = _split_keys(inner_carry.random_key, 3)
            destination_draw = jax.vmap(jax.random.uniform)(
                event_keys[:, 0]
            )
            destination_threshold = destination_draw * total_rate
            rate_cumulative = jnp.cumsum(outgoing_rates, axis=-1)
            destination_slot = jnp.argmax(
                rate_cumulative > destination_threshold[:, None],
                axis=-1,
            )
            destination = adjacency_targets[rows, destination_slot]
            new_hazard = jax.vmap(jax.random.exponential)(
                event_keys[:, 1]
            ).astype(value_dtype)

            current_time = jnp.where(
                fires,
                event_time,
                jnp.where(crosses, crossed_time, inner_carry.current_time),
            )
            current_duration = jnp.where(
                fires,
                0.0,
                jnp.where(
                    crosses,
                    crossed_duration,
                    inner_carry.current_duration,
                ),
            )
            current_state = jnp.where(
                fires,
                destination,
                inner_carry.current_state,
            )
            remaining = jnp.where(
                crosses & (total_rate > 0.0),
                jnp.maximum(
                    inner_carry.remaining_hazard - integrated_hazard,
                    0.0,
                ),
                inner_carry.remaining_hazard,
            )
            remaining = jnp.where(fires, new_hazard, remaining)
            key_mask = fires.reshape(
                (trajectory_count,)
                + (1,) * (inner_carry.random_key.ndim - 1)
            )
            random_key = jnp.where(
                key_mask,
                event_keys[:, 2],
                inner_carry.random_key,
            )

            next_jump_count = inner_carry.jump_count + fires.astype(jnp.int32)
            updated_jump_times = inner_carry.jump_times
            updated_jump_durations = inner_carry.jump_durations
            updated_state_path = inner_carry.state_path
            if max_jumps:
                slots = jnp.clip(
                    inner_carry.jump_count,
                    0,
                    max_jumps - 1,
                )
                old_times = updated_jump_times[rows, slots]
                old_durations = updated_jump_durations[rows, slots]
                updated_jump_times = updated_jump_times.at[rows, slots].set(
                    jnp.where(fires, event_time, old_times)
                )
                updated_jump_durations = updated_jump_durations.at[
                    rows, slots
                ].set(jnp.where(fires, event_duration, old_durations))
                path_slots = slots + 1
                old_states = updated_state_path[rows, path_slots]
                updated_state_path = updated_state_path.at[
                    rows, path_slots
                ].set(jnp.where(fires, destination, old_states))

            return inner_carry._replace(
                current_state=current_state,
                current_time=current_time,
                current_duration=current_duration,
                remaining_hazard=remaining,
                jump_count=next_jump_count,
                overflow=inner_carry.overflow | overflows,
                finished=inner_carry.finished | overflows,
                random_key=random_key,
                jump_times=updated_jump_times,
                jump_durations=updated_jump_durations,
                state_path=updated_state_path,
                truncated_at_time=jnp.where(
                    overflows,
                    inner_carry.current_time,
                    inner_carry.truncated_at_time,
                ),
            )

        return jax.lax.while_loop(
            continue_cell,
            advance_or_jump,
            scan_carry,
        ), None

    carry, _ = jax.lax.scan(
        simulate_calendar_cell,
        carry,
        (calendar_left, calendar_right),
    )
    return carry


def _get_covariate_batch_size(
    kwargs: Mapping[str, jax.Array],
) -> int | None:
    batch_size: int | None = None
    for value in kwargs.values():
        shape = jnp.shape(value)
        if not shape:
            continue
        if batch_size is None:
            batch_size = shape[0]
        elif batch_size != shape[0]:
            raise ValueError("Covariate batch dimensions must match.")
    return batch_size


def _broadcast_batch(value: ArrayLike, batch_size: int) -> jax.Array:
    arr = jnp.asarray(value)
    if arr.ndim == 0:
        return jnp.broadcast_to(arr, (batch_size,))
    if arr.ndim == 1 and arr.shape[0] == batch_size:
        return arr
    raise ValueError("Expected a scalar or (batch,) array.")


def _canonicalize_initial(
    initial: str | ArrayLike | InitialDistribution,
    initial_duration: ArrayLike,
) -> InitialDistribution:
    if isinstance(initial, InitialDistribution):
        try:
            has_nonzero_duration = bool(
                jnp.any(jnp.asarray(initial_duration) != 0.0)
            )
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


def _simulation_value_dtype(
    canonical: _CanonicalDistribution,
    kwargs: Mapping[str, jax.Array],
) -> _DType:
    leaves = [
        jnp.asarray(value)
        for value in (
            *canonical.masses,
            *canonical.durations,
            *kwargs.values(),
        )
    ]
    float_leaves = [
        leaf for leaf in leaves if jnp.issubdtype(leaf.dtype, jnp.inexact)
    ]
    if not float_leaves:
        return cast(_DType, jnp.asarray(0.0).dtype)
    return cast(_DType, jnp.result_type(*float_leaves))


def _validate_integer(name: str, value: object, *, allow_zero: bool) -> int:
    adjective = "non-negative" if allow_zero else "positive"
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a {adjective} integer.")
    integer = int(value)
    if integer < 0 or (integer == 0 and not allow_zero):
        raise ValueError(f"{name} must be a {adjective} integer.")
    return integer


def _resolve_single_device(
    devices: int | Sequence[Any] | None,
) -> Any | None:
    if devices is None:
        return None
    if isinstance(devices, bool):
        raise ValueError("devices must be an integer or a sequence of jax.Device.")
    if isinstance(devices, int):
        if devices <= 0:
            raise ValueError("devices must select at least one device.")
        if devices > 1:
            raise NotImplementedError(
                "Multi-device simulation is not implemented yet; use devices=1."
            )
        return jax.local_devices()[0]
    selected = tuple(devices)
    if not selected:
        raise ValueError("devices must select at least one device.")
    if len(selected) > 1:
        raise NotImplementedError(
            "Multi-device simulation is not implemented yet; select one device."
        )
    return selected[0]


def simulate(
    model: Model,
    initial: str | ArrayLike | InitialDistribution,
    horizon: int,
    steps_per_unit: int,
    initial_duration: ArrayLike = 0.0,
    *,
    max_jumps: int,
    replicates: int = 1,
    key: jax.Array,
    overflow: Literal["return", "raise"] = "return",
    devices: int | Sequence[Any] | None = None,
    **kwargs: Any,
) -> SimulationResult:
    """Sample event histories from the midpoint-discretized intensity model."""
    reserved = {"initial", "initial_duration"}
    overlap = reserved.intersection(kwargs)
    if overlap:
        names = ", ".join(sorted(overlap))
        raise ValueError(f"Reserved covariate names are not allowed: {names}")
    horizon = _validate_integer("horizon", horizon, allow_zero=False)
    steps_per_unit = _validate_integer(
        "steps_per_unit",
        steps_per_unit,
        allow_zero=False,
    )
    max_jumps = _validate_integer("max_jumps", max_jumps, allow_zero=True)
    replicates = _validate_integer("replicates", replicates, allow_zero=False)
    if overflow not in ("return", "raise"):
        raise ValueError("overflow must be either 'return' or 'raise'.")
    try:
        jax.random.key_data(key)
    except (TypeError, ValueError) as exc:
        raise TypeError("key must be a JAX PRNG key.") from exc

    initial_distribution = _canonicalize_initial(initial, initial_duration)
    model_states = model.state_space.states
    initial_distribution.validate_for_model(model_states)
    canonical = initial_distribution.canonicalize(model_states)
    plan = _compile_simulation_plan(model, canonical.states)

    intensity_kwargs = {
        name: jnp.asarray(value) for name, value in kwargs.items()
    }
    distribution_batch = canonical.batch_size
    covariate_batch = _get_covariate_batch_size(intensity_kwargs)
    if (
        distribution_batch is not None
        and covariate_batch is not None
        and distribution_batch != covariate_batch
    ):
        raise ValueError(
            "InitialDistribution batch size must match covariate batch size."
        )
    batch_size = distribution_batch
    if batch_size is None:
        batch_size = covariate_batch
    if batch_size is None:
        batch_size = 1
    value_dtype = _simulation_value_dtype(canonical, intensity_kwargs)

    masses = jnp.stack(
        tuple(_broadcast_batch(value, batch_size) for value in canonical.masses),
        axis=-1,
    ).astype(value_dtype)
    durations = jnp.stack(
        tuple(
            _broadcast_batch(value, batch_size)
            for value in canonical.durations
        ),
        axis=-1,
    ).astype(value_dtype)
    concrete_masses = np.asarray(masses)
    concrete_durations = np.asarray(durations)
    if not np.all(np.isfinite(concrete_masses)):
        raise ValueError("Initial masses must be finite.")
    if np.any(concrete_masses < 0):
        raise ValueError("Initial masses must be non-negative.")
    mass_totals = concrete_masses.sum(axis=-1)
    if np.any(mass_totals <= 0):
        raise ValueError(
            "Initial component masses must have a positive total per individual."
        )
    if not np.all(np.isfinite(concrete_durations)):
        raise ValueError("Initial durations must be finite.")
    if np.any(concrete_durations < 0):
        raise ValueError("Initial durations must be non-negative.")

    individual_indices = jnp.repeat(
        jnp.arange(batch_size, dtype=jnp.int32),
        replicates,
    )
    replicate_indices = jnp.tile(
        jnp.arange(replicates, dtype=jnp.int32),
        batch_size,
    )
    trajectory_masses = masses[individual_indices]
    trajectory_durations = durations[individual_indices]
    reduced_index = {state: index for index, state in enumerate(plan.states)}
    initial_state_options = jnp.asarray(
        [reduced_index[state] for state in canonical.states],
        dtype=jnp.int32,
    )
    trajectory_keys = jax.vmap(
        lambda individual, replicate: jax.random.fold_in(
            jax.random.fold_in(key, individual),
            replicate,
        )
    )(individual_indices, replicate_indices)

    step_size = 1.0 / steps_per_unit
    solver_steps = horizon * steps_per_unit
    calendar_grid = jnp.linspace(
        0.0,
        float(horizon),
        solver_steps + 1,
        dtype=value_dtype,
    )
    max_initial_duration = float(np.max(concrete_durations))
    duration_cells = max(
        1,
        int(np.ceil((max_initial_duration + horizon) * steps_per_unit)),
    )
    duration_mid = (
        jnp.arange(duration_cells, dtype=value_dtype) + 0.5
    ) * jnp.asarray(step_size, dtype=value_dtype)

    selected_device = _resolve_single_device(devices)
    context = (
        jax.default_device(selected_device)
        if selected_device is not None
        else nullcontext()
    )
    with context:
        carry = _event_kernel(
            plan,
            trajectory_masses,
            trajectory_durations,
            initial_state_options,
            trajectory_keys,
            individual_indices,
            calendar_grid[:-1],
            calendar_grid[1:],
            duration_mid,
            intensity_kwargs,
            batch_size=batch_size,
            max_jumps=max_jumps,
            negative_tolerance=1e-12,
        )

    if bool(np.asarray(carry.invalid_nonfinite)):
        raise ValueError("Intensity values must be finite.")
    if bool(np.asarray(carry.invalid_negative)):
        raise ValueError(
            "Intensity values must be non-negative "
            "(apart from values within 1e-12 of zero)."
        )

    shape = (batch_size, replicates)
    result = SimulationResult(
        states=plan.states,
        jump_times=carry.jump_times.reshape((*shape, max_jumps)),
        jump_durations=carry.jump_durations.reshape((*shape, max_jumps)),
        state_path=carry.state_path.reshape((*shape, max_jumps + 1)),
        jump_count=carry.jump_count.reshape(shape),
        overflow=carry.overflow.reshape(shape),
        truncated_at_time=carry.truncated_at_time.reshape(shape),
        final_state=carry.current_state.reshape(shape),
        final_duration=carry.current_duration.reshape(shape),
    )
    if overflow == "raise":
        overflow_count = int(np.asarray(jnp.sum(result.overflow)))
        if overflow_count:
            trajectory_count = batch_size * replicates
            raise RuntimeError(
                f"{overflow_count} of {trajectory_count} trajectories exceeded "
                f"max_jumps={max_jumps}. Increase max_jumps or inspect the "
                "high-transition trajectories."
            )
    return result
