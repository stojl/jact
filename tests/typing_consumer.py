"""Public-only static consumer checks for solve result inference."""

from __future__ import annotations

from typing import Any

import jax
from typing_extensions import assert_type

import jact


def _check_scaled_declaration(
    space: jact.StateSpace,
    payment: jact.typing.Payment,
    weight: jact.typing.Weight,
    when: jact.typing.When,
) -> None:
    scaled = jact.cashflows.Scaled("core", weight=weight)
    declaration = space.cashflows(
        {
            "rate": jact.cashflows.StateRate({"healthy": scaled}),
            "lump": jact.cashflows.TransitionLump({("healthy", "dead"): scaled}),
            "event": jact.cashflows.ScheduledEvent(when, {"healthy": scaled}),
            "duration": jact.cashflows.DurationEvent(
                {"healthy": 0.5}, {"healthy": scaled},
            ),
        },
        cores={"core": payment},
    )
    assert_type(declaration, jact.cashflows.CashflowDeclaration)
    assert_type(declaration.cores["core"], jact.typing.Payment)


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

    tail = model.solve(
        "healthy", 1, 1, probability=jact.probability.Tail(),
        intensity_duration_limit=0.5, payment_duration_limit=0.25,
        probability_duration_limit=0.75,
    )
    assert_type(tail, jact.ModelResult[jact.probability.TailResult])
    assert_type(tail.probability["mass"], jax.Array)

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

    tail = jact.solve(
        model, "healthy", 1, 1, 0.0, jact.probability.Tail(),
        intensity_duration_limit=0.5, payment_duration_limit=0.25,
        probability_duration_limit=0.75,
    )
    assert_type(tail.probability, jact.probability.TailResult)
    assert_type(tail.probability["duration"], jax.Array)

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
