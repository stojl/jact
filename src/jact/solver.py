"""Semi-Markov solver with midpoint quadrature."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from functools import partial
from numbers import Integral
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from .cashflows import (
    ByKind,
    ByState,
    CashflowDeclaration,
    DurationEvent,
    Group,
    Raw,
    Scalar,
    ScheduledEvent,
    StateRate,
    Total,
    TransitionLump,
    validate_cashflow_views,
)
from .initial_distribution import InitialDistribution
from .probability import (
    ProbabilityOutput,
    StateCarry,
    StateProbability,
    _PointMass,
    resolve_callback,
)
from .result import ModelResult

__all__ = ["solve"]

_KIND_STATE_RATE = 0
_KIND_TRANSITION_LUMP = 1
_KIND_SCHEDULED_EVENT = 2
_KIND_DURATION_EVENT = 3

_SOURCE_COMPONENT = 0
_SOURCE_COMPONENT_SUM = 1
_SOURCE_STATE = 2
_SOURCE_KIND = 3
_SOURCE_TOTAL = 4


class _RowHazards(NamedTuple):
    """Per-source-state hazards shared between the advance and cashflow steps."""

    density_hazards: tuple[tuple[int, jnp.ndarray], ...]
    point_hazards: tuple[tuple[int, jnp.ndarray], ...]
    density_transfer_factor: jnp.ndarray
    point_transfer_factor: jnp.ndarray
    density_midpoint_factor: jnp.ndarray
    density_total: jnp.ndarray
    point_total: jnp.ndarray


def _stack_state_densities(state: tuple[StateCarry, ...]) -> jnp.ndarray:
    return jnp.stack(tuple(carry.density for carry in state), axis=0)


def _stack_point_masses(
    state: tuple[StateCarry, ...],
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, tuple[bool, ...]]:
    value_template = state[0].density[:, 0]
    values = []
    d_0 = []
    log_values = []
    mask = []
    for carry in state:
        if carry.point_mass is None:
            values.append(jnp.zeros_like(value_template))
            d_0.append(jnp.zeros_like(value_template))
            log_values.append(jnp.full_like(value_template, -jnp.inf))
            mask.append(False)
        else:
            values.append(carry.point_mass.value)
            d_0.append(carry.point_mass.d_0)
            log_values.append(carry.point_mass.log_value)
            mask.append(True)
    return (
        jnp.stack(values, axis=0),
        jnp.stack(d_0, axis=0),
        jnp.stack(log_values, axis=0),
        tuple(mask),
    )


def _dense_state_to_tuple(
    densities: jnp.ndarray,
    point_values: jnp.ndarray,
    point_d_0: jnp.ndarray,
    point_log_values: jnp.ndarray,
    point_mask: tuple[bool, ...],
) -> tuple[StateCarry, ...]:
    state = []
    for i, has_point_mass in enumerate(point_mask):
        point_mass = None
        if has_point_mass:
            point_mass = _PointMass(
                value=point_values[i],
                d_0=point_d_0[i],
                log_value=point_log_values[i],
            )
        state.append(StateCarry(density=densities[i], point_mass=point_mass))
    return tuple(state)


def _evaluate_intensity_at_point(
    fn: Callable[..., jnp.ndarray],
    t: jnp.ndarray,
    d_per_individual: jnp.ndarray,
    intensity_kwargs: dict[str, jnp.ndarray],
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


def _integrated_density_hazard(
    fn: Callable[..., jnp.ndarray],
    t: jnp.ndarray,
    duration_mid: jnp.ndarray,
    step_size: float,
    intensity_kwargs: dict[str, jnp.ndarray],
) -> jnp.ndarray:
    midpoint = jnp.asarray(fn(t + 0.5 * step_size, duration_mid, **intensity_kwargs))
    return jnp.maximum(step_size * midpoint, 0.0)


def _integrated_point_hazard(
    fn: Callable[..., jnp.ndarray],
    t: jnp.ndarray,
    point_d_0: jnp.ndarray,
    step_size: float,
    intensity_kwargs: dict[str, jnp.ndarray],
) -> jnp.ndarray:
    midpoint = _evaluate_intensity_at_point(
        fn,
        t + 0.5 * step_size,
        point_d_0 + t + 0.5 * step_size,
        intensity_kwargs,
    )
    return jnp.maximum(step_size * midpoint, 0.0)


def _transfer_factor(total_hazard: jnp.ndarray) -> jnp.ndarray:
    transfer_mass = -jnp.expm1(-total_hazard)
    safe_total = jnp.where(total_hazard > 0, total_hazard, 1.0)
    transfer_factor = transfer_mass / safe_total
    return jnp.where(total_hazard > 0, transfer_factor, 1.0)


def _advance_density(
    density: jnp.ndarray,
    total_hazard: jnp.ndarray,
    inflow: jnp.ndarray,
) -> jnp.ndarray:
    def survive(values: jnp.ndarray, hazard: jnp.ndarray) -> jnp.ndarray:
        survived = values + values * jnp.expm1(-hazard)
        return jnp.maximum(survived, jnp.zeros_like(values))

    if density.shape[-1] == 1:
        survived = survive(density[..., 0], total_hazard[..., 0])
        return density.at[..., 0].set(survived + inflow)

    next_density = jnp.zeros_like(density)
    next_density = next_density.at[..., 1:-1].set(
        survive(density[..., :-2], total_hazard[..., :-2])
    )
    next_density = next_density.at[..., -1].set(
        survive(density[..., -1], total_hazard[..., -1])
        + survive(density[..., -2], total_hazard[..., -2])
    )
    next_density = next_density.at[..., 0].set(inflow)
    return next_density


def _zero_leaves(count: int, template: jnp.ndarray) -> tuple[jnp.ndarray, ...]:
    return tuple(jnp.zeros_like(template) for _ in range(count))


def _add_leaf(
    leaves: tuple[jnp.ndarray, ...],
    index: int,
    value: jnp.ndarray,
) -> tuple[jnp.ndarray, ...]:
    return tuple(leaf + value if i == index else leaf for i, leaf in enumerate(leaves))


def _add_cashflow_contribution(
    component_total: jnp.ndarray,
    by_state: tuple[jnp.ndarray, ...],
    by_kind: tuple[jnp.ndarray, ...],
    *,
    state_index: int,
    kind: int,
    contribution: jnp.ndarray,
) -> tuple[jnp.ndarray, tuple[jnp.ndarray, ...], tuple[jnp.ndarray, ...]]:
    return (
        component_total + contribution,
        _add_leaf(by_state, state_index, contribution),
        _add_leaf(by_kind, kind, contribution),
    )


def _sum_leaves(values: tuple[jnp.ndarray, ...]) -> jnp.ndarray:
    total = jnp.zeros_like(values[0])
    for value in values:
        total = total + value
    return total


def _call_payment(
    fn: Callable[..., jnp.ndarray],
    t: jnp.ndarray,
    d: jnp.ndarray,
    intensity_kwargs: dict[str, jnp.ndarray],
) -> jnp.ndarray:
    return jnp.asarray(fn(t, d, **intensity_kwargs))


def _scheduled_event_index(
    event_time: jnp.ndarray,
    step_size: float,
) -> jnp.ndarray:
    dtype = jnp.result_type(event_time, 1.0)
    x = jnp.asarray(event_time, dtype=dtype) / jnp.asarray(step_size, dtype=dtype)
    nearest = jnp.round(x)
    tol = jnp.sqrt(jnp.asarray(jnp.finfo(x.dtype).eps, dtype=x.dtype))
    snapped = jnp.where(jnp.abs(x - nearest) <= tol, nearest, jnp.floor(x))
    return snapped.astype(jnp.int32)


def _duration_event_index(
    delay: jnp.ndarray,
    step_size: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    dtype = jnp.result_type(delay, 1.0)
    step = jnp.asarray(step_size, dtype=dtype)
    delay_index = _scheduled_event_index(delay, step_size)
    effective_delay = delay_index.astype(dtype) * step
    return delay_index, effective_delay


def _is_near_grid_zero(
    value: jnp.ndarray,
    step_size: float,
) -> jnp.ndarray:
    dtype = jnp.result_type(value, 1.0)
    x = jnp.asarray(value, dtype=dtype) / jnp.asarray(step_size, dtype=dtype)
    tol = jnp.sqrt(jnp.asarray(jnp.finfo(x.dtype).eps, dtype=x.dtype))
    return jnp.abs(x) <= tol


def _solver_step_dynamics(
    state: tuple[StateCarry, ...],
    t: jnp.ndarray,
    duration_mid: jnp.ndarray,
    step_size: float,
    solver_matrix: Sequence[Sequence[Callable[..., jnp.ndarray] | None]],
    intensity_kwargs: dict[str, jnp.ndarray],
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    tuple[bool, ...],
    tuple[_RowHazards, ...],
]:
    densities = _stack_state_densities(state)
    point_values, point_d_0, point_log_values, point_mask = _stack_point_masses(state)
    row_hazards = []

    for source_index, row in enumerate(solver_matrix):
        density_total = jnp.zeros_like(densities[source_index])
        point_total = jnp.zeros_like(point_values[source_index])
        density_hazards = []
        point_hazards = []

        for target_index, fn in enumerate(row):
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
            density_hazards.append((target_index, density_hazard))

            if point_mask[source_index]:
                point_hazard = _integrated_point_hazard(
                    fn,
                    t,
                    point_d_0[source_index],
                    step_size,
                    intensity_kwargs,
                )
                point_total = point_total + point_hazard
                point_hazards.append((target_index, point_hazard))

        row_hazards.append(
            _RowHazards(
                density_hazards=tuple(density_hazards),
                point_hazards=tuple(point_hazards),
                density_transfer_factor=_transfer_factor(density_total),
                point_transfer_factor=_transfer_factor(point_total),
                density_midpoint_factor=jnp.exp(-0.5 * density_total),
                density_total=density_total,
                point_total=point_total,
            )
        )

    return (
        densities,
        point_values,
        point_d_0,
        point_log_values,
        point_mask,
        tuple(row_hazards),
    )


def _advance_solver_step_from_dynamics(
    densities: jnp.ndarray,
    point_values: jnp.ndarray,
    point_d_0: jnp.ndarray,
    point_log_values: jnp.ndarray,
    point_mask: tuple[bool, ...],
    row_hazards: tuple[_RowHazards, ...],
) -> tuple[StateCarry, ...]:
    next_inflow = jnp.zeros(densities.shape[:-1], dtype=densities.dtype)
    next_point_values = point_values
    next_point_log_values = point_log_values

    for i, hz in enumerate(row_hazards):
        for j, density_hazard in hz.density_hazards:
            transferred = density_hazard * hz.density_transfer_factor
            next_inflow = next_inflow.at[j].add(
                jnp.sum(densities[i] * transferred, axis=-1)
            )

        if point_mask[i]:
            next_log_value = point_log_values[i] - hz.point_total
            next_point_log_values = next_point_log_values.at[i].set(next_log_value)
            next_point_values = next_point_values.at[i].set(
                jnp.exp(next_log_value)
            )
            for j, point_hazard in hz.point_hazards:
                next_inflow = next_inflow.at[j].add(
                    point_values[i] * point_hazard * hz.point_transfer_factor
                )

    next_densities = []
    for i, hz in enumerate(row_hazards):
        next_densities.append(
            _advance_density(densities[i], hz.density_total, next_inflow[i])
        )

    return _dense_state_to_tuple(
        jnp.stack(tuple(next_densities), axis=0),
        next_point_values,
        point_d_0,
        next_point_log_values,
        point_mask,
    )


def _compute_cashflow_step(
    densities: jnp.ndarray,
    point_values: jnp.ndarray,
    point_d_0: jnp.ndarray,
    point_log_values: jnp.ndarray,
    point_mask: tuple[bool, ...],
    row_hazards: tuple[_RowHazards, ...],
    t: jnp.ndarray,
    duration_mid: jnp.ndarray,
    duration_left: jnp.ndarray,
    step_size: float,
    intensity_kwargs: dict[str, jnp.ndarray],
    cashflow_components: tuple[Any, ...],
    scheduled_events: tuple[Any, ...],
) -> tuple[
    tuple[jnp.ndarray, ...],
    tuple[jnp.ndarray, ...],
    tuple[jnp.ndarray, ...],
    tuple[jnp.ndarray, ...],
    tuple[jnp.ndarray, ...],
    tuple[jnp.ndarray, ...],
]:
    template = densities[0, :, 0]
    by_component = _zero_leaves(len(cashflow_components), template)
    by_state = _zero_leaves(densities.shape[0], template)
    by_kind = _zero_leaves(4, template)
    event_by_component = _zero_leaves(len(cashflow_components), template)
    event_by_state = _zero_leaves(densities.shape[0], template)
    event_by_kind = _zero_leaves(4, template)
    t_mid = t + 0.5 * step_size
    n_steps = duration_mid.shape[-1]

    for component_index, component in enumerate(cashflow_components):
        kind = component[0]
        component_total = jnp.zeros_like(template)
        event_component_total = jnp.zeros_like(template)

        if kind == _KIND_STATE_RATE:
            for state_index, payment_fn in component[1]:
                hz = row_hazards[state_index]
                density_midpoint = densities[state_index] * hz.density_midpoint_factor
                payment = _call_payment(
                    payment_fn,
                    t_mid,
                    duration_mid,
                    intensity_kwargs,
                )
                contribution = step_size * jnp.sum(
                    density_midpoint * payment,
                    axis=-1,
                )
                if point_mask[state_index]:
                    point_payment = _evaluate_intensity_at_point(
                        payment_fn,
                        t_mid,
                        point_d_0[state_index] + t_mid,
                        intensity_kwargs,
                    )
                    point_midpoint = jnp.exp(
                        point_log_values[state_index] - 0.5 * hz.point_total
                    )
                    contribution = contribution + (
                        step_size * point_midpoint * point_payment
                    )
                component_total, by_state, by_kind = _add_cashflow_contribution(
                    component_total,
                    by_state,
                    by_kind,
                    state_index=state_index,
                    kind=_KIND_STATE_RATE,
                    contribution=contribution,
                )

        elif kind == _KIND_TRANSITION_LUMP:
            for source_index, hazard_slot, payment_fn in component[1]:
                hz = row_hazards[source_index]
                _, density_hazard = hz.density_hazards[hazard_slot]
                payment = _call_payment(
                    payment_fn,
                    t_mid,
                    duration_mid,
                    intensity_kwargs,
                )
                contribution = jnp.sum(
                    densities[source_index]
                    * density_hazard
                    * hz.density_transfer_factor
                    * payment,
                    axis=-1,
                )
                if point_mask[source_index]:
                    _, point_hazard = hz.point_hazards[hazard_slot]
                    point_payment = _evaluate_intensity_at_point(
                        payment_fn,
                        t_mid,
                        point_d_0[source_index] + t_mid,
                        intensity_kwargs,
                    )
                    contribution = contribution + (
                        point_values[source_index]
                        * point_hazard
                        * hz.point_transfer_factor
                        * point_payment
                    )
                component_total, by_state, by_kind = _add_cashflow_contribution(
                    component_total,
                    by_state,
                    by_kind,
                    state_index=source_index,
                    kind=_KIND_TRANSITION_LUMP,
                    contribution=contribution,
                )

        elif kind == _KIND_SCHEDULED_EVENT:
            event_time, event_index = scheduled_events[component_index]
            current_index = jnp.round(t / step_size).astype(jnp.int32)
            active = (
                (event_index == current_index)
                & (event_time >= 0)
                & (event_index < n_steps)
            )
            active = active.astype(template.dtype)
            for state_index, payment_fn in component[2]:
                payment = _call_payment(
                    payment_fn,
                    t,
                    duration_left,
                    intensity_kwargs,
                )
                contribution = active * jnp.sum(
                    densities[state_index] * payment,
                    axis=-1,
                )
                if point_mask[state_index]:
                    point_payment = _evaluate_intensity_at_point(
                        payment_fn,
                        t,
                        point_d_0[state_index] + t,
                        intensity_kwargs,
                    )
                    contribution = contribution + (
                        active * point_values[state_index] * point_payment
                    )
                component_total, by_state, by_kind = _add_cashflow_contribution(
                    component_total,
                    by_state,
                    by_kind,
                    state_index=state_index,
                    kind=_KIND_SCHEDULED_EVENT,
                    contribution=contribution,
                )
                (
                    event_component_total,
                    event_by_state,
                    event_by_kind,
                ) = _add_cashflow_contribution(
                    event_component_total,
                    event_by_state,
                    event_by_kind,
                    state_index=state_index,
                    kind=_KIND_SCHEDULED_EVENT,
                    contribution=contribution,
                )

        elif kind == _KIND_DURATION_EVENT:
            for (
                state_index,
                delay,
                delay_index,
                effective_delay,
                payment_fn,
            ) in component[1]:
                delay = _broadcast_batch(delay, template.shape[0])
                delay_index = _broadcast_batch(delay_index, template.shape[0])
                effective_delay = _broadcast_batch(
                    effective_delay,
                    template.shape[0],
                )
                in_horizon = (delay >= 0) & (delay_index < n_steps)
                safe_index = jnp.clip(delay_index, 0, n_steps - 1)
                density_at_delay = jnp.take_along_axis(
                    densities[state_index],
                    safe_index[:, None],
                    axis=-1,
                )[:, 0]
                payment = _evaluate_intensity_at_point(
                    payment_fn,
                    t,
                    effective_delay,
                    intensity_kwargs,
                )
                contribution = in_horizon.astype(template.dtype) * (
                    density_at_delay * payment
                )

                if point_mask[state_index]:
                    remaining = effective_delay - point_d_0[state_index]
                    trigger_index = _scheduled_event_index(remaining, step_size)
                    current_index = jnp.round(t / step_size).astype(jnp.int32)
                    not_past_target = (remaining >= 0) | _is_near_grid_zero(
                        remaining,
                        step_size,
                    )
                    active_point = (
                        in_horizon
                        & not_past_target
                        & (trigger_index == current_index)
                        & (trigger_index < n_steps)
                    )
                    point_payment = _evaluate_intensity_at_point(
                        payment_fn,
                        t,
                        effective_delay,
                        intensity_kwargs,
                    )
                    contribution = contribution + (
                        active_point.astype(template.dtype)
                        * point_values[state_index]
                        * point_payment
                    )
                component_total, by_state, by_kind = _add_cashflow_contribution(
                    component_total,
                    by_state,
                    by_kind,
                    state_index=state_index,
                    kind=_KIND_DURATION_EVENT,
                    contribution=contribution,
                )
                (
                    event_component_total,
                    event_by_state,
                    event_by_kind,
                ) = _add_cashflow_contribution(
                    event_component_total,
                    event_by_state,
                    event_by_kind,
                    state_index=state_index,
                    kind=_KIND_DURATION_EVENT,
                    contribution=contribution,
                )

        by_component = _add_leaf(by_component, component_index, component_total)
        event_by_component = _add_leaf(
            event_by_component,
            component_index,
            event_component_total,
        )

    return (
        by_component,
        by_state,
        by_kind,
        event_by_component,
        event_by_state,
        event_by_kind,
    )


def _compute_scheduled_events(
    cashflow_components: tuple[Any, ...],
    step_size: float,
    intensity_kwargs: dict[str, jnp.ndarray],
) -> tuple[Any, ...]:
    scheduled_events = []
    for component in cashflow_components:
        if component[0] == _KIND_SCHEDULED_EVENT:
            event_time = jnp.asarray(component[1](**intensity_kwargs))
            scheduled_events.append(
                (event_time, _scheduled_event_index(event_time, step_size))
            )
        else:
            scheduled_events.append(None)
    return tuple(scheduled_events)


def _compute_duration_events(
    cashflow_components: tuple[Any, ...],
    step_size: float,
    intensity_kwargs: dict[str, jnp.ndarray],
) -> tuple[Any, ...]:
    duration_events = []
    for component in cashflow_components:
        if component[0] == _KIND_DURATION_EVENT:
            attachments = []
            for state_index, delay_source, payment_fn in component[1]:
                delay = (
                    jnp.asarray(delay_source(**intensity_kwargs))
                    if callable(delay_source)
                    else jnp.asarray(delay_source)
                )
                delay_index, effective_delay = _duration_event_index(
                    delay,
                    step_size,
                )
                attachments.append(
                    (state_index, delay, delay_index, effective_delay, payment_fn)
                )
            duration_events.append((_KIND_DURATION_EVENT, tuple(attachments)))
        else:
            duration_events.append(component)
    return tuple(duration_events)


def _source_value(
    source: tuple[Any, ...],
    by_component: tuple[jnp.ndarray, ...],
    by_state: tuple[jnp.ndarray, ...],
    by_kind: tuple[jnp.ndarray, ...],
) -> jnp.ndarray:
    source_kind = source[0]
    if source_kind == _SOURCE_COMPONENT:
        return by_component[source[1]]
    if source_kind == _SOURCE_COMPONENT_SUM:
        return _sum_leaves(tuple(by_component[index] for index in source[1]))
    if source_kind == _SOURCE_STATE:
        return by_state[source[1]]
    if source_kind == _SOURCE_KIND:
        return by_kind[source[1]]
    return _sum_leaves(by_component)


def _evaluate_weight(
    weight: Callable[..., jnp.ndarray] | Scalar | None,
    t: jnp.ndarray,
    intensity_kwargs: dict[str, jnp.ndarray],
    template: jnp.ndarray,
) -> jnp.ndarray:
    if weight is None:
        return jnp.ones_like(template)
    value = weight(t, **intensity_kwargs) if callable(weight) else weight
    arr = jnp.asarray(value, dtype=template.dtype)
    if arr.ndim == 0:
        return jnp.broadcast_to(arr, template.shape)
    return arr


def _compute_cashflow_views(
    by_component: tuple[jnp.ndarray, ...],
    by_state: tuple[jnp.ndarray, ...],
    by_kind: tuple[jnp.ndarray, ...],
    event_by_component: tuple[jnp.ndarray, ...],
    event_by_state: tuple[jnp.ndarray, ...],
    event_by_kind: tuple[jnp.ndarray, ...],
    t: jnp.ndarray,
    step_size: float,
    intensity_kwargs: dict[str, jnp.ndarray],
    cashflow_views: tuple[Any, ...],
    template: jnp.ndarray,
) -> tuple[tuple[jnp.ndarray, ...], ...]:
    midpoint_by_component = tuple(
        total - event for total, event in zip(by_component, event_by_component)
    )
    midpoint_by_state = tuple(
        total - event for total, event in zip(by_state, event_by_state)
    )
    midpoint_by_kind = tuple(
        total - event for total, event in zip(by_kind, event_by_kind)
    )
    view_values = []
    for (
        _view_name,
        _terminal,
        weight,
        leaf_sources,
        _leaf_names,
        _view_kind,
    ) in cashflow_views:
        factor = _evaluate_weight(
            weight,
            t + 0.5 * step_size,
            intensity_kwargs,
            template,
        )
        event_factor = _evaluate_weight(
            weight,
            t,
            intensity_kwargs,
            template,
        )
        view_values.append(
            tuple(
                _source_value(
                    source,
                    midpoint_by_component,
                    midpoint_by_state,
                    midpoint_by_kind,
                )
                * factor
                + _source_value(
                    source,
                    event_by_component,
                    event_by_state,
                    event_by_kind,
                )
                * event_factor
                for source in leaf_sources
            )
        )
    return tuple(view_values)


def _zero_view_values(
    cashflow_views: tuple[Any, ...],
    template: jnp.ndarray,
) -> tuple[tuple[jnp.ndarray, ...], ...]:
    return tuple(
        tuple(jnp.zeros_like(template) for _source in leaf_sources)
        for (
            _view_name,
            _terminal,
            _weight,
            leaf_sources,
            _leaf_names,
            _view_kind,
        ) in cashflow_views
    )


def _shard_batch_array(
    value: jnp.ndarray,
    device_count: int,
) -> tuple[jnp.ndarray, int]:
    batch_size = value.shape[0]
    remainder = batch_size % device_count
    padded_size = (
        batch_size if remainder == 0 else batch_size + device_count - remainder
    )
    if padded_size != batch_size:
        pad_width = [(0, 0)] * value.ndim
        pad_width[0] = (0, padded_size - batch_size)
        value = jnp.pad(value, tuple(pad_width))
    per_device = padded_size // device_count
    return value.reshape((device_count, per_device) + value.shape[1:]), batch_size


def _shard_batch_tree(tree: Any, device_count: int) -> tuple[Any, int]:
    """Pad and shard a batch-major PyTree over a leading device axis."""
    batch_sizes: list[int] = []

    def shard(value):
        if value is None:
            return None
        arr = jnp.asarray(value)
        if arr.ndim == 0:
            raise ValueError("Cannot shard scalar values over devices.")
        sharded, batch_size = _shard_batch_array(arr, device_count)
        batch_sizes.append(batch_size)
        return sharded

    sharded = jax.tree_util.tree_map(shard, tree)
    if not batch_sizes:
        raise ValueError("Cannot shard an empty tree.")
    batch_size = batch_sizes[0]
    if any(size != batch_size for size in batch_sizes):
        raise ValueError("All sharded leaves must have the same batch size.")
    return sharded, batch_size


def _unshard_batch_array(value: jnp.ndarray, original_batch_size: int) -> jnp.ndarray:
    if value.ndim < 2:
        return value
    if value.ndim == 2:
        merged = value.reshape((value.shape[0] * value.shape[1],) + value.shape[2:])
        return merged[:original_batch_size]

    time_major = jnp.moveaxis(value, 0, 1)
    merged = time_major.reshape(
        (time_major.shape[0], time_major.shape[1] * time_major.shape[2])
        + time_major.shape[3:]
    )
    return merged[:, :original_batch_size, ...]


def _unshard_batch_tree(tree: Any, original_batch_size: int) -> Any:
    """Merge a pmapped solver output tree back onto the public batch axis."""

    def unshard(value):
        if value is None:
            return None
        return _unshard_batch_array(jnp.asarray(value), original_batch_size)

    return jax.tree_util.tree_map(unshard, tree)


def _add_selected_view_values(
    left: tuple[tuple[jnp.ndarray, ...], ...],
    right: tuple[tuple[jnp.ndarray, ...], ...],
    cashflow_views: tuple[Any, ...],
    *,
    terminal: bool,
) -> tuple[tuple[jnp.ndarray, ...], ...]:
    return tuple(
        tuple(
            left_leaf + right_leaf if view_terminal is terminal else left_leaf
            for left_leaf, right_leaf in zip(left_values, right_values)
        )
        for (
            (_view_name, view_terminal, *_),
            left_values,
            right_values,
        ) in zip(cashflow_views, left, right)
    )


_PMAP_IN_AXES = (0, None, None, None, None, 0, None, None, None, None)
_PMAP_STATIC_ARGNUMS = (3, 4, 6, 7, 8, 9)


@partial(
    jax.pmap,
    in_axes=_PMAP_IN_AXES,
    static_broadcasted_argnums=_PMAP_STATIC_ARGNUMS,
)
def _midpoint_solver_pmapped_all_devices(
    state_0: tuple[StateCarry, ...],
    duration_mid: jnp.ndarray,
    duration_left: jnp.ndarray,
    step_size: float,
    solver_matrix: Sequence[Sequence[Callable[..., jnp.ndarray] | None]],
    intensity_kwargs: dict[str, jnp.ndarray],
    prob_callback: Callable[..., Any],
    record_every: int,
    cashflow_components: tuple[Any, ...] = (),
    cashflow_views: tuple[Any, ...] = (),
):
    return _midpoint_solver(
        state_0,
        duration_mid,
        duration_left,
        step_size,
        solver_matrix,
        intensity_kwargs,
        prob_callback,
        record_every,
        cashflow_components,
        cashflow_views,
    )


def _midpoint_solver_pmapped_on_devices(devices: tuple[jax.Device, ...]):
    return jax.pmap(
        _midpoint_solver,
        in_axes=_PMAP_IN_AXES,
        static_broadcasted_argnums=_PMAP_STATIC_ARGNUMS,
        devices=devices,
    )


@partial(
    jax.jit,
    static_argnames=[
        "step_size",
        "solver_matrix",
        "prob_callback",
        "record_every",
        "cashflow_components",
        "cashflow_views",
    ],
)
def _midpoint_solver(
    state_0: tuple[StateCarry, ...],
    duration_mid: jnp.ndarray,
    duration_left: jnp.ndarray,
    step_size: float,
    solver_matrix: Sequence[Sequence[Callable[..., jnp.ndarray] | None]],
    intensity_kwargs: dict[str, jnp.ndarray],
    prob_callback: Callable[..., Any],
    record_every: int,
    cashflow_components: tuple[Any, ...] = (),
    cashflow_views: tuple[Any, ...] = (),
):
    """Run the midpoint solver and record probability outputs."""
    n_steps = duration_mid.shape[-1]
    n_records = n_steps // record_every
    has_cashflows = bool(cashflow_components)
    value_template = state_0[0].density[:, 0]
    block_0 = _zero_view_values(cashflow_views, value_template)
    terminal_0 = _zero_view_values(cashflow_views, value_template)
    scheduled_events = _compute_scheduled_events(
        cashflow_components,
        step_size,
        intensity_kwargs,
    )
    duration_components = _compute_duration_events(
        cashflow_components,
        step_size,
        intensity_kwargs,
    )

    def block_scan(carry, block_start):
        state_carry, terminal_carry = carry
        offsets = jnp.arange(record_every, dtype=duration_mid.dtype)

        def step_scan(inner_carry, offset):
            inner_state, block_cashflows, terminal_cashflows = inner_carry
            current_t = block_start + offset * step_size

            dynamics = _solver_step_dynamics(
                inner_state,
                current_t,
                duration_mid,
                step_size,
                solver_matrix,
                intensity_kwargs,
            )
            (
                raw_component,
                raw_state,
                raw_kind,
                event_component,
                event_state,
                event_kind,
            ) = _compute_cashflow_step(
                *dynamics,
                current_t,
                duration_mid,
                duration_left,
                step_size,
                intensity_kwargs,
                duration_components,
                scheduled_events,
            )
            step_cashflows = _compute_cashflow_views(
                raw_component,
                raw_state,
                raw_kind,
                event_component,
                event_state,
                event_kind,
                current_t,
                step_size,
                intensity_kwargs,
                cashflow_views,
                value_template,
            )
            block_cashflows = _add_selected_view_values(
                block_cashflows,
                step_cashflows,
                cashflow_views,
                terminal=False,
            )
            terminal_cashflows = _add_selected_view_values(
                terminal_cashflows,
                step_cashflows,
                cashflow_views,
                terminal=True,
            )
            next_state = _advance_solver_step_from_dynamics(*dynamics)
            return (next_state, block_cashflows, terminal_cashflows), None

        (state_carry, block_cashflows, terminal_carry), _ = jax.lax.scan(
            step_scan,
            (state_carry, block_0, terminal_carry),
            offsets,
        )
        stream_output = tuple(
            None if terminal else values
            for (_view_name, terminal, *_), values in zip(
                cashflow_views,
                block_cashflows,
            )
        )
        return (state_carry, terminal_carry), (
            prob_callback(state_carry),
            stream_output,
        )

    # Rematerialize one recorded block at a time during reverse-mode so the
    # transpose does not need to retain every inner solver step.
    block_scan = jax.checkpoint(block_scan)
    initial_probability = prob_callback(state_0)
    block_starts = jnp.arange(n_records, dtype=duration_mid.dtype) * (
        record_every * step_size
    )
    (final_state, final_terminal), scan_output = jax.lax.scan(
        block_scan,
        (state_0, terminal_0),
        block_starts,
    )
    probability, cashflow_streams = scan_output

    probability = jax.tree_util.tree_map(
        lambda arr, init: (
            None
            if init is None
            else jnp.concatenate([jnp.expand_dims(init, axis=0), arr], axis=0)
        ),
        probability,
        initial_probability,
    )

    result = {"probability": probability}
    if has_cashflows:
        result["cashflow_streams"] = cashflow_streams
        result["cashflow_terminal"] = final_terminal
    return result


def _get_reference_function(solver_matrix):
    """Find the first non-None callable in the solver matrix."""
    for row in solver_matrix:
        for fn in row:
            if fn is not None:
                return fn
    return None


def _get_covariate_batch_size(kwargs: dict[str, Any]) -> int | None:
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


def _validate_positive_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer.")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _canonicalize_initial(
    initial: str | jnp.ndarray | InitialDistribution,
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
) -> _PointMass:
    return _PointMass(
        value=_broadcast_batch(mass, batch_size),
        d_0=_broadcast_batch(duration, batch_size),
    )


def _prepare_cashflow_components(
    declaration: CashflowDeclaration | None,
    reachable_states: tuple[str, ...],
    solver_matrix: Sequence[Sequence[Callable[..., jnp.ndarray] | None]],
) -> tuple[tuple[Any, ...], ...]:
    if declaration is None:
        return ()

    state_index = {state: i for i, state in enumerate(reachable_states)}
    transition_slot = {}
    for source_index, row in enumerate(solver_matrix):
        slot = 0
        for target_index, fn in enumerate(row):
            if fn is not None:
                transition_slot[(source_index, target_index)] = slot
                slot += 1

    prepared = []
    for _name, component in declaration.components:
        if isinstance(component, StateRate):
            attachments = tuple(
                (state_index[state], fn)
                for state, fn in component.payments.items()
                if state in state_index
            )
            prepared.append((_KIND_STATE_RATE, attachments))
        elif isinstance(component, TransitionLump):
            attachments = tuple(
                (
                    state_index[source],
                    transition_slot[(state_index[source], state_index[target])],
                    fn,
                )
                for (source, target), fn in component.payments.items()
                if source in state_index and target in state_index
            )
            prepared.append((_KIND_TRANSITION_LUMP, attachments))
        elif isinstance(component, ScheduledEvent):
            attachments = tuple(
                (state_index[state], fn)
                for state, fn in component.payments.items()
                if state in state_index
            )
            prepared.append((_KIND_SCHEDULED_EVENT, component.when, attachments))
        elif isinstance(component, DurationEvent):
            attachments = tuple(
                (state_index[state], component.delays[state], fn)
                for state, fn in component.payments.items()
                if state in state_index
            )
            prepared.append((_KIND_DURATION_EVENT, attachments))
    return tuple(prepared)


def _prepare_cashflow_views(
    declaration: CashflowDeclaration,
    views: Mapping[str, Raw | Group | Total | ByState | ByKind] | None,
    reachable_states: tuple[str, ...],
) -> tuple[tuple[Any, ...], ...]:
    frozen_views = validate_cashflow_views(declaration, views)
    component_index = {
        name: i for i, (name, _component) in enumerate(declaration.components)
    }
    prepared = []
    for view_name, view in frozen_views:
        if isinstance(view, Raw):
            if view.name is None:
                leaf_names = declaration.names
                sources = tuple(
                    (_SOURCE_COMPONENT, component_index[name]) for name in leaf_names
                )
                view_kind = "mapping"
            else:
                leaf_names = (view.name,)
                sources = ((_SOURCE_COMPONENT, component_index[view.name]),)
                view_kind = "single"
        elif isinstance(view, Group):
            leaf_names = (view_name,)
            sources = (
                (
                    _SOURCE_COMPONENT_SUM,
                    tuple(component_index[member] for member in view.members),
                ),
            )
            view_kind = "single"
        elif isinstance(view, Total):
            leaf_names = (view_name,)
            sources = ((_SOURCE_TOTAL,),)
            view_kind = "single"
        elif isinstance(view, ByState):
            leaf_names = reachable_states
            sources = tuple(
                (_SOURCE_STATE, index) for index, _state in enumerate(reachable_states)
            )
            view_kind = "mapping"
        elif isinstance(view, ByKind):
            leaf_names = (
                "state_rate",
                "transition_lump",
                "scheduled_event",
                "duration_event",
            )
            sources = tuple((_SOURCE_KIND, index) for index in range(4))
            view_kind = "mapping"
        prepared.append(
            (
                view_name,
                view.terminal,
                view.weight,
                sources,
                tuple(leaf_names),
                view_kind,
            )
        )
    return tuple(prepared)


def _format_cashflow_view_values(
    raw_result: dict[str, Any],
    prepared_views: tuple[tuple[Any, ...], ...],
) -> dict[str, Any]:
    streams = raw_result["cashflow_streams"]
    terminals = raw_result["cashflow_terminal"]
    formatted = {}
    for index, (
        view_name,
        terminal,
        _weight,
        _sources,
        leaf_names,
        view_kind,
    ) in enumerate(prepared_views):
        view_values = terminals[index] if terminal else streams[index]
        if view_kind == "single":
            formatted[view_name] = view_values[0]
        else:
            formatted[view_name] = {
                leaf_name: value for leaf_name, value in zip(leaf_names, view_values)
            }
    return formatted


def _cashflow_reference_function(
    declaration: CashflowDeclaration | None,
) -> Callable[..., jnp.ndarray] | None:
    if declaration is None:
        return None
    for _name, component in declaration.components:
        for fn in component.payments.values():
            return fn
    return None


def _resolve_devices(
    devices: int | Sequence[jax.Device] | None,
) -> tuple[jax.Device, ...]:
    if devices is None:
        return ()
    if isinstance(devices, bool):
        raise ValueError("devices must be an integer or a sequence of jax.Device.")
    local_devices = tuple(jax.local_devices())
    if isinstance(devices, int):
        device_count = int(devices)
        if device_count <= 0:
            raise ValueError("devices must select at least one device.")
        if device_count > len(local_devices):
            raise ValueError(
                f"devices={device_count} requested, but only "
                f"{len(local_devices)} local devices are available."
            )
        return local_devices[:device_count]
    selected = tuple(devices)
    if not selected:
        raise ValueError("devices must select at least one device.")
    return selected


def _run_midpoint_solver(
    state_0: tuple[StateCarry, ...],
    duration_mid: jnp.ndarray,
    duration_left: jnp.ndarray,
    step_size: float,
    solver_matrix: Sequence[Sequence[Callable[..., jnp.ndarray] | None]],
    intensity_kwargs: dict[str, jnp.ndarray],
    prob_callback: Callable[..., Any],
    record_every: int,
    cashflow_components: tuple[Any, ...],
    cashflow_views: tuple[Any, ...],
    devices: tuple[jax.Device, ...],
) -> dict[str, Any]:
    if len(devices) <= 1:
        return _midpoint_solver(
            state_0,
            duration_mid,
            duration_left,
            step_size,
            solver_matrix,
            intensity_kwargs,
            prob_callback,
            record_every,
            cashflow_components,
            cashflow_views,
        )

    sharded_state_0, batch_size = _shard_batch_tree(state_0, len(devices))
    if jax.tree_util.tree_leaves(intensity_kwargs):
        sharded_kwargs, kwargs_batch_size = _shard_batch_tree(
            intensity_kwargs,
            len(devices),
        )
        if kwargs_batch_size != batch_size:
            raise ValueError("Covariate batch dimensions must match solver batch size.")
    else:
        sharded_kwargs = intensity_kwargs

    if devices == tuple(jax.local_devices()):
        sharded_result = _midpoint_solver_pmapped_all_devices(
            sharded_state_0,
            duration_mid,
            duration_left,
            step_size,
            solver_matrix,
            sharded_kwargs,
            prob_callback,
            record_every,
            cashflow_components,
            cashflow_views,
        )
    else:
        sharded_result = _midpoint_solver_pmapped_on_devices(devices)(
            sharded_state_0,
            duration_mid,
            duration_left,
            step_size,
            solver_matrix,
            sharded_kwargs,
            prob_callback,
            record_every,
            cashflow_components,
            cashflow_views,
        )
    return _unshard_batch_tree(sharded_result, batch_size)


def solve(
    model: Any,
    initial: str | jnp.ndarray | InitialDistribution,
    horizon: int,
    steps_per_unit: int,
    initial_duration: Any = 0.0,
    probability: None | ProbabilityOutput | Callable = StateProbability(),
    cashflows: CashflowDeclaration | None = None,
    cashflow_views: Mapping[str, Raw | Group | Total | ByState | ByKind] | None = None,
    record_every: int = 1,
    devices: int | Sequence[jax.Device] | None = None,
    **kwargs: Any,
) -> ModelResult:
    """Compute transition probabilities from a documented initial condition."""
    if "freeze_initial" in kwargs:
        raise TypeError(
            "solve() got an unexpected keyword argument 'freeze_initial'"
        )
    if "callback" in kwargs:
        raise TypeError("solve() got an unexpected keyword argument 'callback'")

    reserved = {"initial", "initial_duration"}
    overlap = reserved.intersection(kwargs)
    if overlap:
        names = ", ".join(sorted(overlap))
        raise ValueError(f"Reserved covariate names are not allowed: {names}")

    horizon = _validate_positive_integer("horizon", horizon)
    steps_per_unit = _validate_positive_integer("steps_per_unit", steps_per_unit)
    solver_steps = steps_per_unit * horizon
    if record_every <= 0 or solver_steps % record_every != 0:
        raise ValueError(
            "record_every must be a positive integer dividing "
            "horizon * steps_per_unit."
        )

    probability_disabled = probability is None
    if cashflow_views is not None and cashflows is None:
        raise ValueError("cashflow_views requires cashflows.")
    if cashflows is not None and not isinstance(cashflows, CashflowDeclaration):
        raise TypeError("cashflows must be a CashflowDeclaration or None.")
    if cashflows is not None and cashflows.state_space is not model.state_space:
        raise ValueError("cashflows must be declared from model.state_space.")
    if cashflows is not None:
        _prepare_cashflow_views(cashflows, cashflow_views, model.state_space.states)

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
    prob_callback = resolve_callback(probability, reduced.reachable_states)
    prepared_cashflow_components = _prepare_cashflow_components(
        cashflows,
        reduced.reachable_states,
        solver_matrix,
    )
    prepared_cashflow_views = (
        _prepare_cashflow_views(cashflows, cashflow_views, reduced.reachable_states)
        if cashflows is not None
        else ()
    )

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
        reference_fn = _cashflow_reference_function(cashflows)
    if reference_fn is None:
        raise ValueError(
            "The intensity matrix contains no callables. Cannot solve."
        )

    reference_output = jnp.asarray(reference_fn(0.0, duration_left, **kwargs))
    if reference_output.ndim != 2 or reference_output.shape[1] != solver_steps:
        raise ValueError(
            "Reference intensity output must have shape "
            f"(batch, {solver_steps})."
        )
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
    selected_devices = _resolve_devices(devices)
    result = _run_midpoint_solver(
        tuple(state_0),
        duration_mid,
        duration_left,
        step_size,
        solver_matrix,
        kwargs,
        prob_callback,
        record_every,
        prepared_cashflow_components,
        prepared_cashflow_views,
        selected_devices,
    )
    probability_out = None if probability_disabled else result["probability"]
    cashflows_out = None
    if cashflows is not None:
        cashflows_out = _format_cashflow_view_values(
            result,
            prepared_cashflow_views,
        )

    return ModelResult(
        states=reduced.reachable_states,
        probability=probability_out,
        cashflows=cashflows_out,
    )
