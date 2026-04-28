from __future__ import annotations

import jax.numpy as jnp
import pytest

import jact


def _constant_intensity(rate: float, batch: int = 1):
    def fn(t, d, **kwargs):
        size = kwargs["age"].shape[0] if "age" in kwargs else batch
        return jnp.full((size, d.shape[-1]), rate)

    return fn


def _constant_payment(amount: float, batch: int = 1):
    def fn(t, d, **kwargs):
        size = kwargs["age"].shape[0] if "age" in kwargs else batch
        return jnp.full((size, d.shape[-1]), amount)

    return fn


def test_cashflow_declaration_validation():
    ss = jact.StateSpace(["healthy", "dead"], [("healthy", "dead")])

    with pytest.raises(ValueError, match="not a declared state"):
        ss.cashflows({"bad": jact.StateRate({"disabled": _constant_payment(1.0)})})

    with pytest.raises(ValueError, match="unknown transition"):
        ss.cashflows({
            "bad": jact.TransitionLump({
                ("dead", "healthy"): _constant_payment(1.0)
            })
        })

    with pytest.raises(TypeError, match="cashflow components"):
        ss.cashflows({"bad": object()})

    cashflows = ss.cashflows({"premium": jact.StateRate({
        "healthy": _constant_payment(1.0)
    })})

    with pytest.raises(ValueError, match="unknown component"):
        jact.solve(
            ss.build(transitions={("healthy", "dead"): _constant_intensity(0.1)}),
            initial="healthy",
            horizon=1,
            steps_per_unit=1,
            cashflows=cashflows,
            cashflow_views={"bad": jact.Raw("missing")},
        )

    with pytest.raises(ValueError, match="unknown component"):
        jact.solve(
            ss.build(transitions={("healthy", "dead"): _constant_intensity(0.1)}),
            initial="healthy",
            horizon=1,
            steps_per_unit=1,
            cashflows=cashflows,
            cashflow_views={"bad": jact.Group(["missing"])},
        )

    with pytest.raises(TypeError, match="terminal"):
        jact.solve(
            ss.build(transitions={("healthy", "dead"): _constant_intensity(0.1)}),
            initial="healthy",
            horizon=1,
            steps_per_unit=1,
            cashflows=cashflows,
            cashflow_views={"bad": jact.Total(terminal="yes")},  # type: ignore[arg-type]
        )

    with pytest.raises(TypeError, match="weight"):
        jact.solve(
            ss.build(transitions={("healthy", "dead"): _constant_intensity(0.1)}),
            initial="healthy",
            horizon=1,
            steps_per_unit=1,
            cashflows=cashflows,
            cashflow_views={"bad": jact.Total(weight=object())},  # type: ignore[arg-type]
        )


def test_state_rate_no_transition_interval_and_terminal_outputs():
    ss = jact.StateSpace(["active"], [])
    model = ss.build(transitions={})
    cashflows = ss.cashflows({"premium": jact.StateRate({
        "active": _constant_payment(2.0)
    })})

    result = model.solve(
        initial="active",
        horizon=2,
        steps_per_unit=4,
        probability=None,
        cashflows=cashflows,
        cashflow_views={
            "premium": jact.Raw("premium"),
            "pv": jact.Total(weight=0.5, terminal=True),
        },
    )

    assert "probability" not in result
    assert result["cashflows"]["premium"].shape == (8, 1)
    assert jnp.allclose(result["cashflows"]["premium"], 0.5)
    assert jnp.allclose(result["cashflows"]["pv"], jnp.array([2.0]))


def test_transition_lump_matches_integrated_transition_probability():
    rate = 0.2
    benefit = 10.0
    ss = jact.StateSpace(["alive", "dead"], [("alive", "dead")])
    model = ss.build(transitions={("alive", "dead"): _constant_intensity(rate)})
    cashflows = ss.cashflows({"death": jact.TransitionLump({
        ("alive", "dead"): _constant_payment(benefit)
    })})

    result = model.solve(
        initial="alive",
        horizon=3,
        steps_per_unit=64,
        cashflows=cashflows,
        cashflow_views={"death": jact.Raw("death", terminal=True)},
    )

    expected = benefit * (1.0 - jnp.exp(-rate * 3.0))
    assert jnp.allclose(result["cashflows"]["death"], expected, atol=2e-4)


def test_state_rate_in_survival_model_uses_midpoint_occupancy():
    rate = 0.2
    ss = jact.StateSpace(["alive", "dead"], [("alive", "dead")])
    model = ss.build(transitions={("alive", "dead"): _constant_intensity(rate)})
    cashflows = ss.cashflows({"annuity": jact.StateRate({
        "alive": _constant_payment(1.0)
    })})

    result = model.solve(
        initial="alive",
        horizon=3,
        steps_per_unit=128,
        probability=None,
        cashflows=cashflows,
        cashflow_views={"annuity": jact.Raw("annuity", terminal=True)},
    )

    expected = (1.0 - jnp.exp(-rate * 3.0)) / rate
    assert jnp.allclose(result["cashflows"]["annuity"], expected, atol=2e-5)


def test_mixed_views_by_state_by_kind_and_weighted_total():
    ss = jact.StateSpace(["healthy", "dead"], [("healthy", "dead")])
    model = ss.build(transitions={("healthy", "dead"): _constant_intensity(0.1)})
    cashflows = ss.cashflows({
        "premium": jact.StateRate({"healthy": _constant_payment(1.0)}),
        "death": jact.TransitionLump({
            ("healthy", "dead"): _constant_payment(5.0)
        }),
    })

    result = model.solve(
        initial="healthy",
        horizon=1,
        steps_per_unit=8,
        cashflows=cashflows,
        cashflow_views={
            "raw": jact.Raw(),
            "benefits": jact.Group(["death"]),
            "total": jact.Total(),
            "half": jact.Total(weight=0.5, terminal=True),
            "state": jact.ByState(terminal=True),
            "kind": jact.ByKind(terminal=True),
        },
    )["cashflows"]

    raw_sum = result["raw"]["premium"] + result["raw"]["death"]
    assert jnp.allclose(result["total"], raw_sum)
    assert jnp.allclose(result["benefits"], result["raw"]["death"])
    assert jnp.allclose(result["half"], jnp.sum(raw_sum, axis=0) * 0.5)
    assert set(result["state"]) == {"healthy", "dead"}
    assert jnp.allclose(result["state"]["dead"], 0.0)
    assert set(result["kind"]) == {
        "state_rate",
        "transition_lump",
        "scheduled_event",
    }
    assert jnp.allclose(result["kind"]["scheduled_event"], 0.0)


def test_state_rate_includes_initial_point_mass_duration():
    ss = jact.StateSpace(["active"], [])
    model = ss.build(transitions={})

    def duration_payment(t, d, **kwargs):
        return jnp.broadcast_to(d, (2, d.shape[-1]))

    cashflows = ss.cashflows({"duration": jact.StateRate({
        "active": duration_payment
    })})

    result = model.solve(
        initial="active",
        initial_duration=jnp.array([2.0, 5.0]),
        horizon=1,
        steps_per_unit=32,
        probability=None,
        cashflows=cashflows,
        cashflow_views={"duration": jact.Raw("duration", terminal=True)},
    )

    expected = jnp.array([2.5, 5.5])
    assert jnp.allclose(result["cashflows"]["duration"], expected, atol=1e-6)


def test_scheduled_event_snapping_individual_times_outside_horizon_and_pre_step():
    ss = jact.StateSpace(["active", "dead"], [("active", "dead")])
    model = ss.build(transitions={("active", "dead"): _constant_intensity(10.0, 4)})

    def when(**kwargs):
        return kwargs["event_time"]

    cashflows = ss.cashflows({"bonus": jact.ScheduledEvent(
        when=when,
        payments={"active": _constant_payment(7.0, 4)},
    )})

    result = model.solve(
        initial="active",
        horizon=1,
        steps_per_unit=4,
        probability=None,
        cashflows=cashflows,
        cashflow_views={
            "bonus": jact.Raw("bonus"),
            "state": jact.ByState(terminal=True),
        },
        event_time=jnp.array([0.0, 0.25, 0.49, 2.0]),
        age=jnp.arange(4.0),
    )["cashflows"]

    assert result["bonus"].shape == (4, 4)
    assert jnp.allclose(result["bonus"][0, 0], 7.0)
    assert jnp.allclose(result["bonus"][1, 1], 7.0 * jnp.exp(-2.5))
    assert jnp.allclose(result["bonus"][1, 2], 7.0 * jnp.exp(-2.5))
    assert jnp.allclose(result["bonus"][:, 3], 0.0)
    assert jnp.allclose(result["state"]["dead"], 0.0)


def test_scheduled_event_snaps_near_grid_before_flooring():
    ss = jact.StateSpace(["active"], [])
    model = ss.build(transitions={})

    def when(**kwargs):
        return kwargs["event_time"]

    cashflows = ss.cashflows({"bonus": jact.ScheduledEvent(
        when=when,
        payments={"active": _constant_payment(7.0, 5)},
    )})

    result = model.solve(
        initial="active",
        horizon=3,
        steps_per_unit=1,
        probability=None,
        cashflows=cashflows,
        cashflow_views={"bonus": jact.Raw("bonus")},
        event_time=jnp.array([
            1.0 - 1e-4,
            1.0 + 1e-4,
            1.0 - 1e-3,
            -1e-4,
            3.0 - 1e-4,
        ], dtype=jnp.float32),
    )["cashflows"]["bonus"]

    expected = jnp.array([
        [0.0, 0.0, 7.0, 0.0, 0.0],
        [7.0, 7.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    assert jnp.allclose(result, expected)


def test_probability_alias_and_callback_conflict():
    ss = jact.StateSpace(["active"], [])
    model = ss.build(transitions={})
    cashflows = ss.cashflows({"premium": jact.StateRate({
        "active": _constant_payment(1.0)
    })})

    result = model.solve(
        initial="active",
        horizon=1,
        steps_per_unit=1,
        probability=None,
        cashflows=cashflows,
    )
    assert "probability" not in result

    with pytest.raises(ValueError, match="conflicting"):
        model.solve(
            initial="active",
            horizon=1,
            steps_per_unit=1,
            callback="default",
            probability="collapse_point_no_duration",
            cashflows=cashflows,
        )
