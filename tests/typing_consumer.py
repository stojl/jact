"""Public-only static consumer checks for solve result inference."""

from __future__ import annotations

from typing import Any

import jax
from typing_extensions import assert_type

import jact


def _custom_probability(
    state: Any,
) -> dict[str, tuple[jax.Array, list[jax.Array]]]:
    density = state[0].density
    return {"nested": (density, [density])}


def _check_method_solve(model: jact.Model) -> None:
    default = model.solve("healthy", 1, 1)
    assert_type(default, jact.ModelResult[jax.Array])
    assert_type(default.probability, jax.Array)

    for reducer in (
        jact.probability.StateProbability(),
        jact.probability.DensityProbability(),
        jact.probability.Density(),
    ):
        array_result = model.solve(
            "healthy", 1, 1, probability=reducer
        )
        assert_type(array_result, jact.ModelResult[jax.Array])

    point_mass = model.solve(
        "healthy", 1, 1, probability=jact.probability.PointMass()
    )
    assert_type(
        point_mass,
        jact.ModelResult[jact.probability.PointMassResult],
    )

    for reducer in (
        jact.probability.MarginalComponents(),
        jact.probability.Full(),
    ):
        components = model.solve(
            "healthy", 1, 1, probability=reducer
        )
        assert_type(
            components,
            jact.ModelResult[jact.probability.ComponentsResult],
        )

    disabled = model.solve("healthy", 1, 1, probability=None)
    assert_type(disabled, jact.ModelResult[None])
    assert_type(disabled.probability, None)

    custom = model.solve(
        "healthy", 1, 1, probability=_custom_probability
    )
    assert_type(
        custom,
        jact.ModelResult[
            dict[str, tuple[jax.Array, list[jax.Array]]]
        ],
    )

    positional = model.solve(
        "healthy", 1, 1, 0.0, jact.probability.PointMass()
    )
    assert_type(
        positional,
        jact.ModelResult[jact.probability.PointMassResult],
    )

    if custom.cashflows is not None:
        assert_type(
            custom.cashflows,
            dict[str, jax.Array | dict[str, jax.Array]],
        )
        value = custom.cashflows["total"]
        if isinstance(value, dict):
            assert_type(value["premium"], jax.Array)
        else:
            assert_type(value, jax.Array)


def _check_function_solve(model: jact.Model) -> None:
    default = jact.solve(model, "healthy", 1, 1)
    assert_type(default, jact.ModelResult[jax.Array])

    for reducer in (
        jact.probability.StateProbability(),
        jact.probability.DensityProbability(),
        jact.probability.Density(),
    ):
        array_result = jact.solve(
            model, "healthy", 1, 1, probability=reducer
        )
        assert_type(array_result, jact.ModelResult[jax.Array])

    point_mass = jact.solve(
        model,
        "healthy",
        1,
        1,
        probability=jact.probability.PointMass(),
    )
    assert_type(
        point_mass.probability,
        jact.probability.PointMassResult,
    )

    for reducer in (
        jact.probability.MarginalComponents(),
        jact.probability.Full(),
    ):
        components = jact.solve(
            model, "healthy", 1, 1, probability=reducer
        )
        assert_type(
            components.probability,
            jact.probability.ComponentsResult,
        )

    disabled = jact.solve(model, "healthy", 1, 1, probability=None)
    assert_type(disabled.probability, None)

    custom = jact.solve(
        model,
        "healthy",
        1,
        1,
        probability=_custom_probability,
    )
    assert_type(
        custom.probability,
        dict[str, tuple[jax.Array, list[jax.Array]]],
    )

    positional = jact.solve(
        model, "healthy", 1, 1, 0.0, jact.probability.PointMass()
    )
    assert_type(
        positional,
        jact.ModelResult[jact.probability.PointMassResult],
    )


def _check_backward_compatibility(array: jax.Array) -> None:
    empty: jact.ModelResult = jact.ModelResult(states=("healthy",))
    bare: jact.ModelResult = jact.ModelResult(
        states=("healthy",),
        probability=array,
    )
    constructed = jact.ModelResult(
        states=("healthy",),
        probability={"nested": (array, [array])},
    )
    assert_type(
        constructed,
        jact.ModelResult[dict[str, tuple[jax.Array, list[jax.Array]]]],
    )
    _ = empty, bare
