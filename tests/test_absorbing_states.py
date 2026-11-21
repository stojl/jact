"""Probability, cashflow, and autodiff coverage for absorbing-state marginals."""

from __future__ import annotations

from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jact
from jact import cashflows as cf
from jact import probability as pr
from jact.solver import _accumulate_marginal_mass


def _model():
    transitions = (("active", "disabled"), ("active", "dead"), ("disabled", "dead"))
    space = jact.StateSpace(("active", "disabled", "dead"), transitions)

    def hazard(t, d, **kwargs):
        return kwargs.get("rate", 0.25) * (1 + 0.2 * d + 0.1 * t)

    model = space.build(transitions=dict.fromkeys(transitions, hazard))
    return space, model


def _payment(t, d, **kwargs):
    return 1 + d * d


@pytest.mark.parametrize("limit", [None, 0.0, 1.0, 8.0])
def test_absorbing_marginals_match_full_grid(limit):
    space, model = _model()
    initial = space.initial_distribution(
        {
            "active": {"mass": jnp.array([0.8, 0.6]), "duration": 0},
            "dead": {"mass": jnp.array([0.2, 0.4]), "duration": 2.5},
        }
    )
    cashflows = space.cashflows(
        {"death": cf.TransitionLump({("disabled", "dead"): _payment})}
    )
    options: dict[str, Any] = dict(
        initial=initial,
        horizon=4,
        steps_per_unit=4,
        record_every=2,
        probability_duration_limit=limit,
        cashflows=cashflows,
    )
    full = model.solve(probability=pr.Full(), **options)
    continuous = full.probability["density"].sum(axis=-1)
    dead_index = full.states.index("dead")
    if limit is not None:
        assert "tail" in full.probability
        tail = full.probability["tail"]["mass"]
        if limit < options["horizon"]:
            assert bool(jnp.all(tail[-1, :, dead_index] > 0))
        continuous = continuous + tail
    occupancy = continuous
    for name, initial_mass in full.probability["point_mass"].items():
        occupancy = occupancy.at[..., full.states.index(name)].add(initial_mass)

    total = model.solve(probability=pr.StateProbability(), **options)
    density = model.solve(probability=pr.DensityProbability(), **options)
    components = model.solve(probability=pr.MarginalComponents(), **options)

    for actual in (total, density, components):
        assert actual.states == full.states
        assert actual.cashflows is not None
        assert full.cashflows is not None
        np.testing.assert_allclose(
            actual.cashflows["raw"]["death"],
            full.cashflows["raw"]["death"],
            atol=2e-6,
            rtol=2e-6,
        )
    np.testing.assert_allclose(total.probability, occupancy, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(density.probability, continuous, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(
        components.probability["density"], continuous, atol=2e-6, rtol=2e-6
    )
    np.testing.assert_array_equal(
        components.probability["point_mass"]["dead"],
        full.probability["point_mass"]["dead"],
    )


@pytest.mark.parametrize(
    "payment",
    [
        cf.StateRate({"dead": _payment}),
        cf.ScheduledEvent(when=lambda **kwargs: 2.0, payments={"dead": _payment}),
        cf.DurationEvent(at_durations={"dead": 0.75}, payments={"dead": _payment}),
    ],
)
def test_absorbing_state_payments_keep_the_grid(payment):
    space, model = _model()
    cashflows = space.cashflows({"payment": payment})
    options: dict[str, Any] = dict(
        cashflows=cashflows,
        probability_duration_limit=1.0,
        cashflow_views={"pv": cf.Total(terminal=True)},
    )
    actual = model.solve("active", 4, 4, probability=pr.StateProbability(), **options)
    full = model.solve("active", 4, 4, probability=pr.Full(), **options)
    assert actual.cashflows is not None
    assert full.cashflows is not None
    actual_pv = cast(jax.Array, actual.cashflows["pv"])
    expected_pv = cast(jax.Array, full.cashflows["pv"])
    np.testing.assert_allclose(actual_pv, expected_pv, rtol=2e-6)
    assert bool(jnp.all(expected_pv > 0))


def test_absorbing_occupancy_agrees_with_full_grid_under_reverse_mode():
    _, model = _model()

    def occupancy(rate, *, full_grid):
        options: dict[str, Any] = dict(
            rate=rate, probability_duration_limit=1.0, record_every=4
        )
        if full_grid:
            result = model.solve("active", 4, 4, probability=pr.Full(), **options)
            dead = result.states.index("dead")
            regular = jnp.sum(result.probability["density"][-1, :, dead], axis=-1)
            assert "tail" in result.probability
            return jnp.sum(regular + result.probability["tail"]["mass"][-1, :, dead])
        result = model.solve("active", 4, 4, **options)
        return jnp.sum(result.probability[-1, :, result.states.index("dead")])

    def evaluate(full_grid):
        def value(rate):
            return occupancy(rate, full_grid=full_grid)

        return jax.jit(jax.value_and_grad(value))(jnp.array(0.25, dtype=jnp.float32))

    np.testing.assert_allclose(evaluate(False), evaluate(True), rtol=3e-6, atol=1e-7)


def test_absorbing_accumulator_retains_small_inflows_over_many_steps():
    initial = jnp.array([[0.75, 0.0]], dtype=jnp.float32)
    inflow = jnp.array([1e-8], dtype=jnp.float32)
    steps = 4096

    @jax.jit
    def accumulate(mass):
        def step(carry, _):
            return _accumulate_marginal_mass(carry, inflow), None

        final, _ = jax.lax.scan(step, mass, None, length=steps)
        return jnp.sum(final, axis=-1)

    np.testing.assert_allclose(
        accumulate(initial), [0.75 + steps * 1e-8], rtol=0.0, atol=3e-8
    )
