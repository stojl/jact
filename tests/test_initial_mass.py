"""Initial mass dtypes and derivatives at the boundary of a mixture."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jact


@pytest.fixture
def model():
    space = jact.StateSpace(["active", "dead"], [("active", "dead")])
    return space.build(transitions={("active", "dead"): lambda t, d, **kw: kw["rate"]})


@pytest.mark.parametrize("kind", ["indices", "components", "normalised"])
def test_integer_initial_masses_preserve_probabilities(model, kind):
    if kind == "indices":
        initial = jact.InitialDistribution.per_individual(
            jnp.array([0, 1]), duration=jnp.array([0, 1])
        )
    else:
        initial = jact.InitialDistribution(
            {
                "active": {"mass": jnp.array([1, 0]), "duration": 0},
                "dead": {"mass": jnp.array([0, 1]), "duration": 1},
            },
            normalise=kind == "normalised",
        )
    result = model.solve(initial, horizon=1, steps_per_unit=4, rate=0.2)
    survival = jnp.exp(-0.2 * jnp.arange(5) / 4)
    expected = jnp.stack(
        (
            jnp.stack((survival, 1 - survival), axis=-1),
            jnp.broadcast_to(jnp.array([0.0, 1.0]), (5, 2)),
        ),
        axis=1,
    )
    np.testing.assert_allclose(result.probability, expected, atol=1e-6)
    np.testing.assert_allclose(result.probability.sum(axis=-1), 1, atol=1e-6)


@pytest.mark.parametrize("mass", [0.0, 0.3, 1.0])
@pytest.mark.parametrize("normalise", [False, True])
def test_initial_mass_derivatives_match_survival_and_cashflows(model, mass, normalise):
    def unit(t, d, **kw):
        return 1.0

    cashflows = model.state_space.cashflows(
        {
            "rate": jact.cashflows.StateRate({"active": unit}),
            "jump": jact.cashflows.TransitionLump({("active", "dead"): unit}),
            "scheduled": jact.cashflows.ScheduledEvent(
                when=lambda **kw: 0.5, payments={"active": unit}
            ),
            "duration": jact.cashflows.DurationEvent(
                at_durations={"active": 0.5}, payments={"active": unit}
            ),
        }
    )

    def solve(x):
        initial = jact.InitialDistribution(
            {
                "active": {"mass": x, "duration": 0.0},
                "dead": {"mass": 1 - x, "duration": 0.0},
            },
            normalise=normalise,
        )
        result = model.solve(
            initial,
            horizon=2,
            steps_per_unit=4,
            record_every=4,
            rate=0.2,
            cashflows=cashflows,
            cashflow_views={"raw": jact.cashflows.Raw(terminal=True)},
        )
        assert result.cashflows is not None
        raw = result.cashflows["raw"]
        assert isinstance(raw, dict)
        return jnp.concatenate((result.probability[-1, 0], *raw.values()))

    survival = jnp.exp(-0.4)
    rate = 0.25 * jnp.exp(-0.2 * (jnp.arange(8) + 0.5) / 4).sum()
    event = jnp.exp(-0.1)
    expected = jnp.array([survival, -survival, rate, 1 - survival, event, event])
    x = jnp.array(mass)
    np.testing.assert_allclose(jax.jit(jax.jacrev(solve))(x), expected, atol=1e-6)
    np.testing.assert_allclose(jax.jit(jax.jacfwd(solve))(x), expected, atol=1e-6)


@pytest.mark.parametrize("horizon,rate", [(1000, 1e-8), (20, 0.8)])
def test_point_survival_retains_small_hazards_over_long_horizons(model, horizon, rate):
    def survival(rate_value):
        return model.solve(
            initial=jact.InitialDistribution(
                {"active": {"mass": 0.3, "duration": 0.0}}, normalise=False
            ),
            horizon=horizon,
            steps_per_unit=1,
            record_every=horizon,
            rate=rate_value,
        ).probability[-1, 0, 0]

    value, derivative = jax.jit(jax.value_and_grad(survival))(jnp.array(rate))
    expected = 0.3 * np.exp(-rate * horizon)
    np.testing.assert_allclose(value, expected, rtol=2e-6)
    np.testing.assert_allclose(derivative, -horizon * expected, rtol=2e-5)
