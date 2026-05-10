from __future__ import annotations

import jax.numpy as jnp
import pytest

import jact
from jact.cashflows import validate_cashflow_views


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


def _time_intensity(base: float, time_coef: float, batch: int = 1):
    def fn(t, d, **kwargs):
        size = kwargs["age"].shape[0] if "age" in kwargs else batch
        level = base + time_coef * t
        return jnp.full((size, d.shape[-1]), level)

    return fn


def _duration_intensity(base: float, duration_coef: float, batch: int = 1):
    def fn(t, d, **kwargs):
        size = kwargs["age"].shape[0] if "age" in kwargs else batch
        level = base + duration_coef * d
        return jnp.broadcast_to(level, (size, d.shape[-1]))

    return fn


def _time_duration_intensity(
    base: float,
    time_coef: float,
    duration_coef: float,
    batch: int = 1,
):
    def fn(t, d, **kwargs):
        size = kwargs["age"].shape[0] if "age" in kwargs else batch
        level = base + time_coef * t + duration_coef * d
        return jnp.broadcast_to(level, (size, d.shape[-1]))

    return fn


def _duration_payment(base: float = 0.0, duration_coef: float = 1.0, batch: int = 1):
    def fn(t, d, **kwargs):
        size = kwargs["age"].shape[0] if "age" in kwargs else batch
        level = base + duration_coef * d
        return jnp.broadcast_to(level, (size, d.shape[-1]))

    return fn


def _time_duration_payment(
    base: float,
    time_coef: float,
    duration_coef: float,
    batch: int = 1,
):
    def fn(t, d, **kwargs):
        size = kwargs["age"].shape[0] if "age" in kwargs else batch
        level = base + time_coef * t + duration_coef * d
        return jnp.broadcast_to(level, (size, d.shape[-1]))

    return fn


def _linear_duration_hazard_integral(
    horizon: float,
    base: float,
    duration_coef: float,
    initial_duration,
):
    d_0 = jnp.asarray(initial_duration)
    return (base + duration_coef * d_0) * horizon + 0.5 * duration_coef * horizon**2


def _linear_time_duration_hazard_integral(
    horizon: float,
    base: float,
    time_coef: float,
    duration_coef: float,
    initial_duration,
):
    d_0 = jnp.asarray(initial_duration)
    return (
        (base + duration_coef * d_0) * horizon
        + 0.5 * (time_coef + duration_coef) * horizon**2
    )


def test_cashflow_declaration_validation():
    ss = jact.StateSpace(["healthy", "dead"], [("healthy", "dead")])

    with pytest.raises(ValueError, match="not a declared state"):
        ss.cashflows(
            {"bad": jact.cashflows.StateRate({"disabled": _constant_payment(1.0)})}
        )

    with pytest.raises(ValueError, match="unknown transition"):
        ss.cashflows({
            "bad": jact.cashflows.TransitionLump({
                ("dead", "healthy"): _constant_payment(1.0)
            })
        })

    with pytest.raises(TypeError, match="cashflow components"):
        ss.cashflows({"bad": object()})

    cashflows = ss.cashflows({"premium": jact.cashflows.StateRate({
        "healthy": _constant_payment(1.0)
    })})

    with pytest.raises(ValueError, match="unknown component"):
        jact.solve(
            ss.build(transitions={("healthy", "dead"): _constant_intensity(0.1)}),
            initial="healthy",
            horizon=1,
            steps_per_unit=1,
            cashflows=cashflows,
            cashflow_views={"bad": jact.cashflows.Raw("missing")},
        )

    with pytest.raises(ValueError, match="unknown component"):
        jact.solve(
            ss.build(transitions={("healthy", "dead"): _constant_intensity(0.1)}),
            initial="healthy",
            horizon=1,
            steps_per_unit=1,
            cashflows=cashflows,
            cashflow_views={"bad": jact.cashflows.Group(["missing"])},
        )

    with pytest.raises(TypeError, match="terminal"):
        jact.solve(
            ss.build(transitions={("healthy", "dead"): _constant_intensity(0.1)}),
            initial="healthy",
            horizon=1,
            steps_per_unit=1,
            cashflows=cashflows,
            cashflow_views={"bad": jact.cashflows.Total(terminal="yes")},  # type: ignore[arg-type]
        )

    with pytest.raises(TypeError, match="weight"):
        jact.solve(
            ss.build(transitions={("healthy", "dead"): _constant_intensity(0.1)}),
            initial="healthy",
            horizon=1,
            steps_per_unit=1,
            cashflows=cashflows,
            cashflow_views={"bad": jact.cashflows.Total(weight=object())},  # type: ignore[arg-type]
        )


def test_cashflow_declaration_copies_payment_mappings():
    ss = jact.StateSpace(["healthy", "dead"], [("healthy", "dead")])
    payments = {"healthy": _constant_payment(1.0)}

    declaration = ss.cashflows({"premium": jact.cashflows.StateRate(payments)})
    payments["dead"] = _constant_payment(2.0)

    component = declaration.component("premium")
    assert isinstance(component, jact.cashflows.StateRate)
    assert set(component.payments) == {"healthy"}


def test_discount_factor_removed_from_public_api():
    with pytest.raises(ImportError):
        exec("from jact import discount_factor", {})

    assert "discount_factor" not in jact.__all__
    assert not hasattr(jact, "discount_factor")


def test_cashflow_views_accept_rank_zero_array_weight():
    ss = jact.StateSpace(["active"], [])
    model = ss.build(transitions={})
    cashflows = ss.cashflows({"premium": jact.cashflows.StateRate({
        "active": _constant_payment(2.0)
    })})

    result = model.solve(
        initial="active",
        horizon=1,
        steps_per_unit=4,
        probability=None,
        cashflows=cashflows,
        cashflow_views={
            "pv": jact.cashflows.Total(weight=jnp.array(0.5), terminal=True),
        },
    )

    assert jnp.allclose(result.cashflows["pv"], jnp.array([1.0]))


def test_cashflow_views_reject_non_scalar_array_weight():
    ss = jact.StateSpace(["active"], [])
    model = ss.build(transitions={})
    cashflows = ss.cashflows({"premium": jact.cashflows.StateRate({
        "active": _constant_payment(2.0)
    })})

    with pytest.raises(TypeError, match="weight"):
        model.solve(
            initial="active",
            horizon=1,
            steps_per_unit=4,
            probability=None,
            cashflows=cashflows,
            cashflow_views={
                "pv": jact.cashflows.Total(weight=jnp.array([0.5, 1.0]), terminal=True),
            },
        )


def test_empty_cashflow_view_mapping_returns_empty_outputs():
    ss = jact.StateSpace(["active"], [])
    model = ss.build(transitions={})
    cashflows = ss.cashflows({"premium": jact.cashflows.StateRate({
        "active": _constant_payment(2.0)
    })})

    result = model.solve(
        initial="active",
        horizon=1,
        steps_per_unit=4,
        probability=None,
        cashflows=cashflows,
        cashflow_views={},
    )

    assert result.cashflows == {}
    assert result.probability is None


def test_group_view_members_are_frozen_during_validation():
    ss = jact.StateSpace(["healthy", "dead"], [("healthy", "dead")])
    cashflows = ss.cashflows({
        "premium": jact.cashflows.StateRate({"healthy": _constant_payment(1.0)}),
        "death": jact.cashflows.TransitionLump({
            ("healthy", "dead"): _constant_payment(5.0)
        }),
    })
    members = ["death"]
    frozen_views = validate_cashflow_views(
        cashflows,
        {"benefits": jact.cashflows.Group(members)},
    )
    members.append("premium")

    name, view = frozen_views[0]
    assert name == "benefits"
    assert isinstance(view, jact.cashflows.Group)
    assert view.members == ("death",)


def test_state_rate_no_transition_interval_and_terminal_outputs():
    ss = jact.StateSpace(["active"], [])
    model = ss.build(transitions={})
    cashflows = ss.cashflows({"premium": jact.cashflows.StateRate({
        "active": _constant_payment(2.0)
    })})

    result = model.solve(
        initial="active",
        horizon=2,
        steps_per_unit=4,
        probability=None,
        cashflows=cashflows,
        cashflow_views={
            "premium": jact.cashflows.Raw("premium"),
            "pv": jact.cashflows.Total(weight=0.5, terminal=True),
        },
    )

    assert result.probability is None
    assert result.cashflows["premium"].shape == (8, 1)
    assert jnp.allclose(result.cashflows["premium"], 0.5)
    assert jnp.allclose(result.cashflows["pv"], jnp.array([2.0]))


def test_transition_lump_matches_integrated_transition_probability():
    rate = 0.2
    benefit = 10.0
    ss = jact.StateSpace(["alive", "dead"], [("alive", "dead")])
    model = ss.build(transitions={("alive", "dead"): _constant_intensity(rate)})
    cashflows = ss.cashflows({"death": jact.cashflows.TransitionLump({
        ("alive", "dead"): _constant_payment(benefit)
    })})

    result = model.solve(
        initial="alive",
        horizon=3,
        steps_per_unit=64,
        cashflows=cashflows,
        cashflow_views={"death": jact.cashflows.Raw("death", terminal=True)},
    )

    expected = benefit * (1.0 - jnp.exp(-rate * 3.0))
    assert jnp.allclose(result.cashflows["death"], expected, atol=2e-4)


def test_state_rate_in_survival_model_uses_midpoint_occupancy():
    rate = 0.2
    ss = jact.StateSpace(["alive", "dead"], [("alive", "dead")])
    model = ss.build(transitions={("alive", "dead"): _constant_intensity(rate)})
    cashflows = ss.cashflows({"annuity": jact.cashflows.StateRate({
        "alive": _constant_payment(1.0)
    })})

    result = model.solve(
        initial="alive",
        horizon=3,
        steps_per_unit=128,
        probability=None,
        cashflows=cashflows,
        cashflow_views={"annuity": jact.cashflows.Raw("annuity", terminal=True)},
    )

    expected = (1.0 - jnp.exp(-rate * 3.0)) / rate
    assert jnp.allclose(result.cashflows["annuity"], expected, atol=2e-5)


def test_constant_intensity_time_duration_state_rate_matches_closed_form():
    rate = 0.25
    horizon = 2
    initial_duration = jnp.array([0.0, 0.75], dtype=jnp.float32)
    base = 1.5
    time_coef = 0.4
    duration_coef = 0.2
    ss = jact.StateSpace(["alive", "dead"], [("alive", "dead")])
    model = ss.build(transitions={("alive", "dead"): _constant_intensity(rate, 2)})
    cashflows = ss.cashflows({"annuity": jact.cashflows.StateRate({
        "alive": _time_duration_payment(base, time_coef, duration_coef, 2)
    })})

    result = model.solve(
        initial="alive",
        initial_duration=initial_duration,
        horizon=horizon,
        steps_per_unit=256,
        probability=None,
        cashflows=cashflows,
        cashflow_views={"annuity": jact.cashflows.Raw("annuity", terminal=True)},
        age=jnp.arange(2.0),
    )

    survival = jnp.exp(-rate * horizon)
    integral_0 = (1.0 - survival) / rate
    integral_1 = (1.0 - survival * (1.0 + rate * horizon)) / rate**2
    expected = (
        (base + duration_coef * initial_duration) * integral_0
        + (time_coef + duration_coef) * integral_1
    )
    assert jnp.allclose(result.cashflows["annuity"], expected, atol=2e-5)


def test_point_mass_state_rate_remains_accurate_at_high_resolution_float32():
    rate = 0.25
    horizon = 2
    initial_duration = jnp.array([0.0, 0.75], dtype=jnp.float32)
    base = 1.5
    time_coef = 0.4
    duration_coef = 0.2
    ss = jact.StateSpace(["alive", "dead"], [("alive", "dead")])
    model = ss.build(transitions={("alive", "dead"): _constant_intensity(rate, 2)})
    cashflows = ss.cashflows({"annuity": jact.cashflows.StateRate({
        "alive": _time_duration_payment(base, time_coef, duration_coef, 2)
    })})

    result = model.solve(
        initial="alive",
        initial_duration=initial_duration,
        horizon=horizon,
        steps_per_unit=1024,
        probability=None,
        cashflows=cashflows,
        cashflow_views={"annuity": jact.cashflows.Raw("annuity", terminal=True)},
        age=jnp.arange(2.0, dtype=jnp.float32),
    )

    survival = jnp.exp(-rate * horizon)
    integral_0 = (1.0 - survival) / rate
    integral_1 = (1.0 - survival * (1.0 + rate * horizon)) / rate**2
    expected = (
        (base + duration_coef * initial_duration) * integral_0
        + (time_coef + duration_coef) * integral_1
    )
    assert jnp.allclose(result.cashflows["annuity"], expected, atol=1e-5)


def test_time_dependent_intensity_transition_lump_matches_closed_form():
    base = 0.08
    time_coef = 0.05
    benefit = 7.0
    horizon = 3
    ss = jact.StateSpace(["alive", "dead"], [("alive", "dead")])
    model = ss.build(
        transitions={("alive", "dead"): _time_intensity(base, time_coef)}
    )
    cashflows = ss.cashflows({"death": jact.cashflows.TransitionLump({
        ("alive", "dead"): _constant_payment(benefit)
    })})

    result = model.solve(
        initial="alive",
        horizon=horizon,
        steps_per_unit=512,
        probability=None,
        cashflows=cashflows,
        cashflow_views={"death": jact.cashflows.Raw("death", terminal=True)},
    )

    cumulative_hazard = base * horizon + 0.5 * time_coef * horizon**2
    expected = benefit * (1.0 - jnp.exp(-cumulative_hazard))
    assert jnp.allclose(result.cashflows["death"], expected, atol=2e-5)


def test_duration_dependent_intensity_cashflows_match_closed_form():
    base = 0.06
    duration_coef = 0.09
    benefit = 3.0
    horizon = 2
    initial_duration = jnp.array([0.0, 0.5, 1.25], dtype=jnp.float32)
    ss = jact.StateSpace(["alive", "dead"], [("alive", "dead")])
    model = ss.build(
        transitions={("alive", "dead"): _duration_intensity(base, duration_coef, 3)}
    )
    cashflows = ss.cashflows({
        "hazard_rate": jact.cashflows.StateRate({
            "alive": _duration_payment(base, duration_coef, 3)
        }),
        "death": jact.cashflows.TransitionLump({
            ("alive", "dead"): _constant_payment(benefit, 3)
        }),
    })

    result = model.solve(
        initial="alive",
        initial_duration=initial_duration,
        horizon=horizon,
        steps_per_unit=1024,
        probability=None,
        cashflows=cashflows,
        cashflow_views={
            "hazard_rate": jact.cashflows.Raw("hazard_rate", terminal=True),
            "death": jact.cashflows.Raw("death", terminal=True),
        },
        age=jnp.arange(3.0),
    ).cashflows

    cumulative_hazard = _linear_duration_hazard_integral(
        horizon,
        base,
        duration_coef,
        initial_duration,
    )
    transition_probability = 1.0 - jnp.exp(-cumulative_hazard)
    assert jnp.allclose(result["hazard_rate"], transition_probability, atol=5e-5)
    assert jnp.allclose(
        result["death"],
        benefit * transition_probability,
        atol=2e-5,
    )


def test_time_duration_dependent_intensity_cashflows_match_closed_form():
    base = 0.04
    time_coef = 0.03
    duration_coef = 0.07
    horizon = 2
    initial_duration = jnp.array([0.25, 1.0], dtype=jnp.float32)
    ss = jact.StateSpace(["alive", "dead"], [("alive", "dead")])
    model = ss.build(
        transitions={
            ("alive", "dead"): _time_duration_intensity(
                base,
                time_coef,
                duration_coef,
                2,
            )
        }
    )
    cashflows = ss.cashflows({
        "hazard_rate": jact.cashflows.StateRate({
            "alive": _time_duration_payment(base, time_coef, duration_coef, 2)
        }),
        "death": jact.cashflows.TransitionLump({
            ("alive", "dead"): _constant_payment(1.0, 2)
        }),
    })

    result = model.solve(
        initial="alive",
        initial_duration=initial_duration,
        horizon=horizon,
        steps_per_unit=1024,
        probability=None,
        cashflows=cashflows,
        cashflow_views={
            "hazard_rate": jact.cashflows.Raw("hazard_rate", terminal=True),
            "death": jact.cashflows.Raw("death", terminal=True),
        },
        age=jnp.arange(2.0),
    ).cashflows

    cumulative_hazard = _linear_time_duration_hazard_integral(
        horizon,
        base,
        time_coef,
        duration_coef,
        initial_duration,
    )
    transition_probability = 1.0 - jnp.exp(-cumulative_hazard)
    assert jnp.allclose(result["hazard_rate"], transition_probability, atol=5e-5)
    assert jnp.allclose(result["death"], transition_probability, atol=2e-5)


def test_discounted_constant_state_rate_matches_closed_form_present_value():
    rate = 0.2
    discount_rate = 0.03
    payment = 4.0
    horizon = 5
    ss = jact.StateSpace(["alive", "dead"], [("alive", "dead")])
    model = ss.build(transitions={("alive", "dead"): _constant_intensity(rate)})
    cashflows = ss.cashflows({"annuity": jact.cashflows.StateRate({
        "alive": _constant_payment(payment)
    })})

    def flat_discount_weight(t, **kwargs):
        return jnp.exp(-discount_rate * t)

    result = model.solve(
        initial="alive",
        horizon=horizon,
        steps_per_unit=256,
        probability=None,
        cashflows=cashflows,
        cashflow_views={
            "pv": jact.cashflows.Total(
                weight=flat_discount_weight,
                terminal=True,
            )
        },
    )

    combined_rate = rate + discount_rate
    expected = payment * (1.0 - jnp.exp(-combined_rate * horizon)) / combined_rate
    assert jnp.allclose(result.cashflows["pv"], expected, atol=3e-5)


def test_mixed_views_by_state_by_kind_and_weighted_total():
    ss = jact.StateSpace(["healthy", "dead"], [("healthy", "dead")])
    model = ss.build(transitions={("healthy", "dead"): _constant_intensity(0.1)})
    cashflows = ss.cashflows({
        "premium": jact.cashflows.StateRate({"healthy": _constant_payment(1.0)}),
        "death": jact.cashflows.TransitionLump({
            ("healthy", "dead"): _constant_payment(5.0)
        }),
    })

    result = model.solve(
        initial="healthy",
        horizon=1,
        steps_per_unit=8,
        cashflows=cashflows,
        cashflow_views={
            "raw": jact.cashflows.Raw(),
            "benefits": jact.cashflows.Group(["death"]),
            "total": jact.cashflows.Total(),
            "half": jact.cashflows.Total(weight=0.5, terminal=True),
            "state": jact.cashflows.ByState(terminal=True),
            "kind": jact.cashflows.ByKind(terminal=True),
        },
    ).cashflows

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

    cashflows = ss.cashflows({"duration": jact.cashflows.StateRate({
        "active": duration_payment
    })})

    result = model.solve(
        initial="active",
        initial_duration=jnp.array([2.0, 5.0]),
        horizon=1,
        steps_per_unit=32,
        probability=None,
        cashflows=cashflows,
        cashflow_views={"duration": jact.cashflows.Raw("duration", terminal=True)},
    )

    expected = jnp.array([2.5, 5.5])
    assert jnp.allclose(result.cashflows["duration"], expected, atol=1e-6)


def test_scheduled_event_snapping_individual_times_outside_horizon_and_pre_step():
    ss = jact.StateSpace(["active", "dead"], [("active", "dead")])
    model = ss.build(transitions={("active", "dead"): _constant_intensity(10.0, 4)})

    def when(**kwargs):
        return kwargs["event_time"]

    cashflows = ss.cashflows({"bonus": jact.cashflows.ScheduledEvent(
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
            "bonus": jact.cashflows.Raw("bonus"),
            "state": jact.cashflows.ByState(terminal=True),
        },
        event_time=jnp.array([0.0, 0.25, 0.49, 2.0]),
        age=jnp.arange(4.0),
    ).cashflows

    assert result["bonus"].shape == (4, 4)
    assert jnp.allclose(result["bonus"][0, 0], 7.0)
    assert jnp.allclose(result["bonus"][1, 1], 7.0 * jnp.exp(-2.5))
    assert jnp.allclose(result["bonus"][1, 2], 7.0 * jnp.exp(-2.5))
    assert jnp.allclose(result["bonus"][:, 3], 0.0)
    assert jnp.allclose(result["state"]["dead"], 0.0)


def test_scheduled_event_matches_closed_form_survival_probability():
    rate = 0.2
    benefit = 10.0
    event_time = 2.0
    ss = jact.StateSpace(["alive", "dead"], [("alive", "dead")])
    model = ss.build(transitions={("alive", "dead"): _constant_intensity(rate)})

    def when(**kwargs):
        return jnp.asarray(event_time)

    cashflows = ss.cashflows({"bonus": jact.cashflows.ScheduledEvent(
        when=when,
        payments={"alive": _constant_payment(benefit)},
    )})

    result = model.solve(
        initial="alive",
        horizon=4,
        steps_per_unit=4,
        probability=None,
        cashflows=cashflows,
        cashflow_views={"bonus": jact.cashflows.Raw("bonus", terminal=True)},
    )

    expected = benefit * jnp.exp(-rate * event_time)
    assert jnp.allclose(result.cashflows["bonus"], expected, atol=1e-6)


def test_discounted_scheduled_event_matches_closed_form_present_value():
    rate = 0.2
    discount_rate = 0.03
    benefit = 10.0
    event_time = 2.0
    ss = jact.StateSpace(["alive", "dead"], [("alive", "dead")])
    model = ss.build(transitions={("alive", "dead"): _constant_intensity(rate)})

    def when(**kwargs):
        return jnp.asarray(event_time)

    def discount_weight(t, **kwargs):
        return jnp.exp(-discount_rate * t)

    cashflows = ss.cashflows({"bonus": jact.cashflows.ScheduledEvent(
        when=when,
        payments={"alive": _constant_payment(benefit)},
    )})

    result = model.solve(
        initial="alive",
        horizon=4,
        steps_per_unit=4,
        probability=None,
        cashflows=cashflows,
        cashflow_views={
            "pv": jact.cashflows.Total(weight=discount_weight, terminal=True),
        },
    )

    expected = benefit * jnp.exp(-(rate + discount_rate) * event_time)
    assert jnp.allclose(result.cashflows["pv"], expected, atol=1e-6)


def test_duration_dependent_scheduled_event_payment_matches_closed_form():
    rate = 0.2
    event_time = 2.0
    initial_duration = jnp.array([0.0, 0.75, 1.5], dtype=jnp.float32)
    base = 5.0
    duration_coef = 1.25
    ss = jact.StateSpace(["alive", "dead"], [("alive", "dead")])
    model = ss.build(transitions={("alive", "dead"): _constant_intensity(rate, 3)})

    def when(**kwargs):
        return jnp.asarray(event_time)

    cashflows = ss.cashflows({"bonus": jact.cashflows.ScheduledEvent(
        when=when,
        payments={"alive": _duration_payment(base, duration_coef, 3)},
    )})

    result = model.solve(
        initial="alive",
        initial_duration=initial_duration,
        horizon=4,
        steps_per_unit=4,
        probability=None,
        cashflows=cashflows,
        cashflow_views={"bonus": jact.cashflows.Raw("bonus", terminal=True)},
        age=jnp.arange(3.0),
    )

    payment = base + duration_coef * (initial_duration + event_time)
    expected = payment * jnp.exp(-rate * event_time)
    assert jnp.allclose(result.cashflows["bonus"], expected, atol=1e-6)


def test_scheduled_event_by_state_and_by_kind_match_closed_form_multi_state():
    mu = 0.07
    nu = 0.03
    active_benefit = 8.0
    disabled_benefit = 3.0
    event_time = 2.0
    total_rate = mu + nu
    ss = jact.StateSpace(
        ["active", "disabled", "dead"],
        [("active", "disabled"), ("active", "dead")],
    )
    model = ss.build(
        transitions={
            ("active", "disabled"): _constant_intensity(mu),
            ("active", "dead"): _constant_intensity(nu),
        }
    )

    def when(**kwargs):
        return jnp.asarray(event_time)

    cashflows = ss.cashflows({"event": jact.cashflows.ScheduledEvent(
        when=when,
        payments={
            "active": _constant_payment(active_benefit),
            "disabled": _constant_payment(disabled_benefit),
        },
    )})

    result = model.solve(
        initial="active",
        horizon=4,
        steps_per_unit=4,
        probability=None,
        cashflows=cashflows,
        cashflow_views={
            "state": jact.cashflows.ByState(terminal=True),
            "kind": jact.cashflows.ByKind(terminal=True),
        },
    ).cashflows

    expected_active = active_benefit * jnp.exp(-total_rate * event_time)
    expected_disabled = (
        disabled_benefit
        * mu
        / total_rate
        * (1.0 - jnp.exp(-total_rate * event_time))
    )
    assert jnp.allclose(result["state"]["active"], expected_active, atol=1e-6)
    assert jnp.allclose(result["state"]["disabled"], expected_disabled, atol=1e-6)
    assert jnp.allclose(result["state"]["dead"], 0.0, atol=1e-6)
    assert jnp.allclose(
        result["kind"]["scheduled_event"],
        expected_active + expected_disabled,
        atol=1e-6,
    )


def test_scheduled_event_snaps_near_grid_before_flooring():
    ss = jact.StateSpace(["active"], [])
    model = ss.build(transitions={})

    def when(**kwargs):
        return kwargs["event_time"]

    cashflows = ss.cashflows({"bonus": jact.cashflows.ScheduledEvent(
        when=when,
        payments={"active": _constant_payment(7.0, 5)},
    )})

    result = model.solve(
        initial="active",
        horizon=3,
        steps_per_unit=1,
        probability=None,
        cashflows=cashflows,
        cashflow_views={"bonus": jact.cashflows.Raw("bonus")},
        event_time=jnp.array([
            1.0 - 1e-4,
            1.0 + 1e-4,
            1.0 - 1e-3,
            -1e-4,
            3.0 - 1e-4,
        ], dtype=jnp.float32),
    ).cashflows["bonus"]

    expected = jnp.array([
        [0.0, 0.0, 7.0, 0.0, 0.0],
        [7.0, 7.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    assert jnp.allclose(result, expected)


def test_probability_none_omits_probability_and_callback_is_rejected():
    ss = jact.StateSpace(["active"], [])
    model = ss.build(transitions={})
    cashflows = ss.cashflows({"premium": jact.cashflows.StateRate({
        "active": _constant_payment(1.0)
    })})

    result = model.solve(
        initial="active",
        horizon=1,
        steps_per_unit=1,
        probability=None,
        cashflows=cashflows,
    )
    assert result.probability is None

    with pytest.raises(TypeError, match="unexpected keyword argument 'callback'"):
        model.solve(
            initial="active",
            horizon=1,
            steps_per_unit=1,
            callback="default",
            cashflows=cashflows,
        )

    with pytest.raises(TypeError, match="unexpected keyword argument 'callback'"):
        jact.solve(
            model,
            initial="active",
            horizon=1,
            steps_per_unit=1,
            callback="default",
            cashflows=cashflows,
        )
