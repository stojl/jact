"""Public callable protocol tests."""

from __future__ import annotations

from collections.abc import Mapping
from importlib.resources import files
from typing import Any, get_type_hints

import jax.numpy as jnp

import jact
from jact.initial_distribution import _CanonicalDistribution, _component_payload
from jact.probability import (
    ComponentsResult,
    PointMassResult,
    StateCarry,
    _density_callback,
    _full_callback,
    _point_mass_callback,
    _state_probability_callback,
)
from jact.solver import _SolverResult


def _grid_callable(
    t: jnp.ndarray,
    d: jnp.ndarray,
    **kwargs: Any,
) -> jact.typing.ArrayLike:
    return t + d


def _source_callable(**kwargs: Any) -> jact.typing.ArrayLike:
    return kwargs.get("value", 0.0)


def _weight_callable(
    t: jnp.ndarray,
    **kwargs: Any,
) -> jact.typing.ArrayLike:
    return t


intensity: jact.typing.Intensity = _grid_callable
grouped_intensity: jact.typing.GroupedIntensity = _grid_callable
payment: jact.typing.Payment = _grid_callable
when: jact.typing.When = _source_callable
duration_at: jact.typing.DurationAt = _source_callable
weight: jact.typing.Weight = _weight_callable


def _internal_type_check(
    initial: jact.InitialDistribution,
    state: tuple[StateCarry, ...],
) -> None:
    """Static assertions for the internal types tightened in this pass."""
    canonical: _CanonicalDistribution = initial.canonicalize(("healthy",))
    mass, duration = _component_payload({"mass": 1.0, "duration": 0.0})
    mass_value: jact.typing.ArrayLike = mass
    duration_value: jact.typing.ArrayLike = duration
    state_probability: jnp.ndarray = _state_probability_callback(state)
    density: jnp.ndarray = _density_callback(state)
    point_mass: PointMassResult = _point_mass_callback(("healthy",))(state)
    full: ComponentsResult = _full_callback(("healthy",))(state)
    solver_result = _SolverResult(
        probability=state_probability,
        cashflow_streams=None,
        cashflow_terminal=None,
    )
    probability: Any = solver_result.probability
    streams = solver_result.cashflow_streams
    terminal = solver_result.cashflow_terminal
    _ = (
        canonical,
        mass_value,
        duration_value,
        density,
        point_mass,
        full,
        probability,
        streams,
        terminal,
    )


def _consumer_type_check(
    state_space: jact.StateSpace,
    model: jact.Model,
) -> None:
    components: Mapping[str, jact.cashflows.CashflowComponent] = {
        "premium": jact.cashflows.StateRate({"healthy": payment})
    }
    views: Mapping[str, jact.cashflows.CashflowView] = {
        "total": jact.cashflows.Total(terminal=True)
    }
    declaration: jact.cashflows.CashflowDeclaration = state_space.cashflows(
        components
    )
    initial: jact.InitialDistribution = state_space.initial_at(
        "healthy",
        duration=jnp.asarray(0.0),
    )
    method_result: jact.ModelResult = model.solve(
        initial=initial,
        horizon=1,
        steps_per_unit=1,
        probability=None,
        cashflows=declaration,
        cashflow_views=views,
    )
    function_result: jact.ModelResult = jact.solve(
        model=model,
        initial=initial,
        horizon=1,
        steps_per_unit=1,
        probability=None,
    )
    _ = method_result, function_result


def test_typing_is_a_public_submodule():
    assert jact.typing.__all__ == [
        "ArrayLike",
        "Intensity",
        "GroupedIntensity",
        "Payment",
        "When",
        "DurationAt",
        "Weight",
    ]
    assert "typing" in jact.__all__


def test_distribution_declares_inline_typing_support():
    assert files("jact").joinpath("py.typed").is_file()


def test_callable_protocols_are_not_flat_top_level_aliases():
    for name in jact.typing.__all__[1:]:
        assert not hasattr(jact, name)


def test_intensity_wrappers_return_public_protocols():
    assert (
        get_type_hints(jact.wrappers.bind_intensity)["return"]
        is jact.typing.Intensity
    )
    for wrapper in (
        jact.wrappers.bind_grouped_intensity,
        jact.wrappers.bind_exit_intensity,
    ):
        assert get_type_hints(wrapper)["return"] is jact.typing.GroupedIntensity
