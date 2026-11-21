"""Duration-uniform states agree with solves that retain the full grid."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jact
import jact.solver as solver
from jact import cashflows as cf
from jact import probability as pr
from jact.solver import _accumulate_marginal_mass, _survive_uniform_mass


def _model():
    edges = (("active", "disabled"), ("disabled", "active"), ("disabled", "dead"))
    space = jact.StateSpace(("active", "disabled", "dead"), edges)

    def hazard(t, d, *, rate, age, **kwargs):
        # Each output is uniform on its given grid, but changing that grid
        # or reusing its value for every point evaluation would be incorrect.
        return rate * (1 + 0.1 * t + d.mean() + 0.01 * age[:, None])

    model = space.build(transitions=dict.fromkeys(edges, hazard))
    return space, model


def _continuous(probability: pr.ComponentsResult) -> jax.Array:
    mass = probability["density"].sum(axis=-1)
    if "tail" in probability:
        mass = mass + probability["tail"]["mass"]
    return mass


@pytest.mark.parametrize(
    "limit, observed", [(None, None), (1.0, None), (8.0, "payment"), (1.0, "event")]
)
def test_uniform_states_agree_with_full_grid(limit, observed, monkeypatch):
    space, model = _model()
    selected: list[frozenset[int]] = []
    select = solver._plan_uniform_densities

    def record_selection(*args, **kwargs):
        states, evaluate = select(*args, **kwargs)
        selected.append(states)
        return states, evaluate

    monkeypatch.setattr(solver, "_plan_uniform_densities", record_selection)

    def uniform_payment(t, d, *, age, **kwargs):
        return 1 + t + d.mean() + 0.02 * age[:, None]

    def duration_payment(t, d, **kwargs):
        return 1 + d * d

    components: dict[str, cf.CashflowComponent] = {
        "rate": cf.StateRate({"active": uniform_payment, "disabled": uniform_payment}),
        "jump": cf.TransitionLump(
            {
                ("active", "disabled"): uniform_payment,
                ("disabled", "dead"): uniform_payment,
            }
        ),
        "scheduled": cf.ScheduledEvent(
            when=lambda **kwargs: 2.0, payments={"active": uniform_payment}
        ),
    }
    if observed == "payment":
        components["rate"] = cf.StateRate(
            {"active": duration_payment, "disabled": uniform_payment}
        )
    if observed == "event":
        components["duration"] = cf.DurationEvent(
            at_durations={"active": 0.75}, payments={"active": uniform_payment}
        )
    cashflows = space.cashflows(components)
    initial = space.initial_distribution(
        {
            "active": {"mass": 0.7, "duration": jnp.array([0.0, 1.0])},
            "disabled": {"mass": 0.3, "duration": jnp.array([2.0, 0.25])},
        }
    )
    options: dict[str, Any] = dict(
        cashflows=cashflows,
        cashflow_views={"raw": cf.Raw(), "pv": cf.Total(terminal=True)},
        rate=0.15,
        age=jnp.array([34.0, 76.0]),
        record_every=4,
        intensity_duration_limit=0.8,
        payment_duration_limit=1.0,
        probability_duration_limit=limit,
    )
    actual = jax.jit(
        lambda: model.solve(
            initial, 4, 8, probability=pr.MarginalComponents(), **options
        )
    )()
    full = model.solve(initial, 4, 8, probability=pr.Full(), **options)
    assert actual.states == full.states
    expected_uniform = {"active", "disabled"} if observed is None else {"disabled"}
    assert {frozenset(actual.states[i] for i in indices) for indices in selected} == {
        frozenset(expected_uniform)
    }
    np.testing.assert_allclose(
        actual.probability["density"],
        _continuous(full.probability),
        rtol=4e-6,
        atol=2e-6,
    )
    for left, right in zip(
        jax.tree.leaves(actual.probability["point_mass"]),
        jax.tree.leaves(full.probability["point_mass"]),
        strict=True,
    ):
        np.testing.assert_allclose(left, right, rtol=4e-6, atol=2e-6)
    assert jax.tree.structure(actual.cashflows) == jax.tree.structure(full.cashflows)
    for left, right in zip(
        jax.tree.leaves(actual.cashflows), jax.tree.leaves(full.cashflows), strict=True
    ):
        np.testing.assert_allclose(left, right, rtol=4e-6, atol=2e-6)


def test_uniform_states_support_reverse_mode():
    _, model = _model()

    def objective(rate, *, full_grid):
        options: dict[str, Any] = dict(
            rate=rate,
            age=jnp.array([28.0, 58.0]),
            record_every=4,
        )
        if full_grid:
            result = model.solve("active", 4, 4, probability=pr.Full(), **options)
            occupancy = _continuous(result.probability)
        else:
            result = model.solve(
                "active", 4, 4, probability=pr.DensityProbability(), **options
            )
            occupancy = result.probability
        return occupancy[-1, :, result.states.index("disabled")].sum()

    def evaluate(full_grid):
        def loss(rate):
            return objective(rate, full_grid=full_grid)

        return jax.jit(jax.value_and_grad(loss))(jnp.asarray(0.2))

    np.testing.assert_allclose(evaluate(False), evaluate(True), rtol=4e-6, atol=2e-6)


def test_uniform_mass_retains_small_losses_over_many_steps():
    initial = jnp.array([[0.75, 0.0]], dtype=jnp.float32)
    hazard = jnp.array([[1e-8]], dtype=jnp.float32)
    inflow = jnp.array([2e-8], dtype=jnp.float32)
    steps = 4096

    @jax.jit
    def evolve(mass):
        def step(carry, _):
            surviving = _survive_uniform_mass(carry, hazard)
            return _accumulate_marginal_mass(surviving, inflow), None

        final, _ = jax.lax.scan(step, mass, None, length=steps)
        return final.sum(axis=-1)

    expected = 0.75 * np.exp(-steps * 1e-8) + 2e-8 * np.expm1(-steps * 1e-8) / np.expm1(
        -1e-8
    )
    np.testing.assert_allclose(evolve(initial), [expected], rtol=0.0, atol=3e-8)
