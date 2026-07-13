"""Public callable protocol tests."""

from __future__ import annotations

from collections.abc import Mapping
from importlib.resources import files
from typing import Any, get_type_hints

import jax.numpy as jnp

import jact
from jact._cashflow_ir import PreparedCashflowView
from jact.cashflows import Scalar, Total, _normalised_view
from jact.initial_distribution import _CanonicalDistribution, _component_payload
from jact.model import ReducedModel, SolverMatrix, _make_slice_wrapper
from jact.probability import (
    ComponentsResult,
    PointMassResult,
    StateCarry,
    _density_callback,
    _full_callback,
    _point_mass_callback,
    _state_probability_callback,
)
from jact.solver import (
    _canonicalize_initial,
    _format_cashflow_view_values,
    _shard_batch_tree,
    _SolverResult,
    _unshard_batch_tree,
)


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
    initial_value: jact.typing.ArrayLike,
    model: jact.Model,
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
    reduced: ReducedModel = model.reduce("healthy")
    solver_matrix: SolverMatrix = reduced.solver_matrix
    sliced_intensity: jact.typing.Intensity = _make_slice_wrapper(
        grouped_intensity, 0
    )
    sliced_output: jact.typing.ArrayLike = sliced_intensity(
        jnp.asarray(0.0),
        jnp.zeros((1, 1)),
    )
    shortcut: jact.InitialDistribution = _canonicalize_initial(
        initial_value, 0.0
    )
    tree = {"value": jnp.ones((1, 2))}
    sharded_tree: dict[str, jnp.ndarray]
    sharded_tree, original_size = _shard_batch_tree(tree, 1)
    unsharded_tree: dict[str, jnp.ndarray] = _unshard_batch_tree(
        sharded_tree, original_size
    )
    normalised_view = _normalised_view(Total(weight=jnp.asarray(0.5)))
    normalised_weight: jact.typing.Weight | Scalar | None = (
        normalised_view.weight
    )
    prepared_view = PreparedCashflowView(
        name="total",
        terminal=False,
        weight=normalised_weight,
        sources=(),
        leaf_names=("total",),
        output="single",
    )
    rebuilt: jact.ModelResult = jact.ModelResult.tree_unflatten(
        ("healthy",),
        (state_probability, {"total": state_probability}),
    )
    formatted_cashflows: dict[
        str, jnp.ndarray | dict[str, jnp.ndarray]
    ] = _format_cashflow_view_values(
        _SolverResult(
            probability=None,
            cashflow_streams=((state_probability,),),
            cashflow_terminal=((state_probability,),),
        ),
        (prepared_view,),
    )
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
        solver_matrix,
        sliced_output,
        shortcut,
        prepared_view,
        unsharded_tree,
        rebuilt,
        formatted_cashflows,
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
