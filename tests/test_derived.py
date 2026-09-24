"""Shared derived fields across model, cashflow, and solve declarations."""

import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jact


def _space():
    return jact.StateSpace(
        ["active", "disabled", "dead"],
        [("active", "disabled"), ("active", "dead"), ("disabled", "dead")],
    )


def _assert_tree_close(left, right):
    jax.tree.map(lambda x, y: np.testing.assert_allclose(x, y, rtol=2e-5), left, right)


def test_cross_scope_fields_match_inline_features_with_cashflows_and_limits():
    ss = _space()

    def risk(t, d, **kw):
        return kw["risk"]

    def benefit(t, d, **kw):
        return kw["benefit"][:, None] + kw["basis"]

    derived_model = ss.build(
        transitions={
            ("active", "disabled"): risk,
            ("active", "dead"): lambda t, d, **kw: 0.5 * kw["risk"],
            ("disabled", "dead"): lambda t, d, **kw: 0.8 * kw["risk"],
        },
        derived={
            "scale": lambda rate: 0.1 * rate,
            "age": lambda t, age0: age0 + t,
        },
    )
    components = {
        "income": jact.cashflows.StateRate({"active": benefit}),
        "death": jact.cashflows.TransitionLump({("active", "dead"): benefit}),
        "fixed": jact.cashflows.ScheduledEvent(
            when=lambda **kw: 0.5 + 0 * kw["scale"],
            payments={"active": benefit},
        ),
        "duration": jact.cashflows.DurationEvent(
            at_durations={"active": lambda **kw: 0.5 + 0 * kw["scale"]},
            payments={"active": benefit},
        ),
    }
    derived_cashflows = ss.cashflows(
        components,
        derived={"benefit": lambda age, scale: 0.01 * age + scale},
    )
    views = {
        "raw": jact.cashflows.Raw(),
        "pv": jact.cashflows.Total(
            weight=lambda t, **kw: kw["discount"], terminal=True
        ),
    }
    solve_fields = {
        "basis": lambda d: 0.02 * d,
        "risk": lambda age, basis, scale, benefit: (
            scale[:, None] + 0.001 * age[:, None] + 0.001 * benefit[:, None] + basis
        ),
        "discount": lambda t, interest: jnp.exp(-interest * t),
    }
    common: dict[str, Any] = dict(
        initial="active",
        horizon=2,
        steps_per_unit=4,
        initial_duration=jnp.array([0.0, 0.25]),
        probability=jact.probability.Full(),
        cashflow_views=views,
        intensity_duration_limit=0.6,
        payment_duration_limit=0.8,
        probability_duration_limit=1.0,
        age0=jnp.array([30.0, 45.0]),
        rate=jnp.array([0.5, 0.8]),
        interest=0.03,
    )
    actual = jact.solve(
        derived_model, cashflows=derived_cashflows, derived=solve_fields, **common
    )

    def inline_risk(t, d, kw):
        age = kw["age0"] + t
        benefit = 0.01 * age + 0.1 * kw["rate"]
        return (
            0.1 * kw["rate"][:, None]
            + 0.001 * age[:, None]
            + 0.001 * benefit[:, None]
            + 0.02 * d
        )

    def inline_benefit(t, d, kw):
        age = kw["age0"] + t
        return (0.01 * age + 0.1 * kw["rate"])[:, None] + 0.02 * d

    inline_model = ss.build(
        transitions={
            ("active", "disabled"): lambda t, d, **kw: inline_risk(t, d, kw),
            ("active", "dead"): lambda t, d, **kw: 0.5 * inline_risk(t, d, kw),
            ("disabled", "dead"): lambda t, d, **kw: 0.8 * inline_risk(t, d, kw),
        }
    )
    inline_components = {
        "income": jact.cashflows.StateRate(
            {"active": lambda t, d, **kw: inline_benefit(t, d, kw)}
        ),
        "death": jact.cashflows.TransitionLump(
            {("active", "dead"): lambda t, d, **kw: inline_benefit(t, d, kw)}
        ),
        "fixed": jact.cashflows.ScheduledEvent(
            when=lambda **kw: 0.5 + 0 * kw["rate"],
            payments={"active": lambda t, d, **kw: inline_benefit(t, d, kw)},
        ),
        "duration": jact.cashflows.DurationEvent(
            at_durations={"active": lambda **kw: 0.5 + 0 * kw["rate"]},
            payments={"active": lambda t, d, **kw: inline_benefit(t, d, kw)},
        ),
    }
    inline_views = {
        "raw": jact.cashflows.Raw(),
        "pv": jact.cashflows.Total(
            weight=lambda t, **kw: jnp.exp(-kw["interest"] * t), terminal=True
        ),
    }
    expected = inline_model.solve(
        cashflows=ss.cashflows(inline_components),
        **{**common, "cashflow_views": inline_views},
    )
    _assert_tree_close(actual.probability, expected.probability)
    _assert_tree_close(actual.cashflows, expected.cashflows)


@pytest.mark.parametrize(
    "model_fields,cashflow_fields,solve_fields,inputs,match",
    [
        ({"x": lambda: 1}, {"x": lambda: 2}, None, {}, "Duplicate"),
        ({"x": lambda: 1}, None, {"x": lambda: 2}, {}, "Duplicate"),
        ({"x": lambda: 1}, None, None, {"x": 3}, "collides"),
        ({"t": lambda: 1}, None, None, {}, "reserved"),
        ({"d": lambda: 1}, None, None, {}, "reserved"),
        ({"initial": lambda: 1}, None, None, {}, "reserved"),
        ({"initial_duration": lambda: 1}, None, None, {}, "reserved"),
        ({"x": lambda y: y, "y": lambda x: x}, None, None, {}, "Cycle"),
        ({"x": lambda y: y}, {"y": lambda x: x}, None, {}, "Cycle"),
        ({"x": lambda missing: missing}, None, None, {}, "missing dependency"),
    ],
)
def test_graph_validation(model_fields, cashflow_fields, solve_fields, inputs, match):
    ss = jact.StateSpace(["active", "dead"], [("active", "dead")])
    model = ss.build(
        transitions={("active", "dead"): lambda t, d, **kw: 0.1},
        derived=model_fields,
    )
    cashflows = (
        ss.cashflows(
            {"income": jact.cashflows.StateRate({"active": lambda t, d, **kw: 1.0})},
            derived=cashflow_fields,
        )
        if cashflow_fields
        else None
    )
    with pytest.raises((ValueError, TypeError), match=match):
        model.solve(
            initial="active",
            horizon=1,
            steps_per_unit=2,
            cashflows=cashflows,
            derived=solve_fields,
            **inputs,
        )


def test_fields_are_resolved_once_per_context_and_input_fields_once_per_solve():
    counts = {"input": 0, "time": 0, "duration": 0}

    def input_field(rate):
        counts["input"] += 1
        return rate

    def time_field(t, base):
        counts["time"] += 1
        return base + 0 * t

    def duration_field(d, timed):
        counts["duration"] += 1
        return timed + 0 * d

    ss = jact.StateSpace(
        ["active", "dead", "lapsed"],
        [("active", "dead"), ("active", "lapsed")],
    )
    model = ss.build(
        transitions={
            ("active", "dead"): lambda t, d, **kw: kw["surface"],
            ("active", "lapsed"): lambda t, d, **kw: 0.5 * kw["surface"],
        },
        derived={"base": input_field, "timed": time_field, "surface": duration_field},
    )
    cashflows = ss.cashflows(
        {
            "income": jact.cashflows.StateRate(
                {"active": lambda t, d, **kw: kw["surface"]}
            )
        }
    )
    model.solve(
        initial="active",
        horizon=1,
        steps_per_unit=2,
        cashflows=cashflows,
        rate=0.1,
    )
    assert counts == {"input": 1, "time": 1, "duration": 2}


def test_hoisted_grid_fields_match_inline_point_and_payment_contexts():
    ss = jact.StateSpace(["active", "dead"], [("active", "dead")])

    def derived_value(t, d, **kw):
        return kw["surface"]

    def inline_value(t, d, **kw):
        return jnp.sin(d * kw["scale"]) + 0.01 * t

    derived_model = ss.build(
        transitions={("active", "dead"): derived_value},
        derived={
            "basis": lambda d, scale: jnp.sin(d * scale),
            "surface": lambda t, basis: basis + 0.01 * t,
        },
    )
    inline_model = ss.build(transitions={("active", "dead"): inline_value})
    derived_cashflows = ss.cashflows(
        {
            "rate": jact.cashflows.StateRate({"active": derived_value}),
            "event": jact.cashflows.ScheduledEvent(
                when=lambda **kw: 0.5, payments={"active": derived_value}
            ),
        }
    )
    inline_cashflows = ss.cashflows(
        {
            "rate": jact.cashflows.StateRate({"active": inline_value}),
            "event": jact.cashflows.ScheduledEvent(
                when=lambda **kw: 0.5, payments={"active": inline_value}
            ),
        }
    )
    common: dict[str, Any] = dict(
        initial="active",
        initial_duration=jnp.array([0.0, 0.25]),
        horizon=1,
        steps_per_unit=4,
        intensity_duration_limit=0.6,
        payment_duration_limit=0.8,
        probability=jact.probability.Full(),
        scale=jnp.array([0.7, 1.2])[:, None],
    )
    actual = derived_model.solve(cashflows=derived_cashflows, **common)
    expected = inline_model.solve(cashflows=inline_cashflows, **common)
    _assert_tree_close(actual.probability, expected.probability)
    _assert_tree_close(actual.cashflows, expected.cashflows)


def test_compiled_duration_grid_transform_is_outside_scan():
    ss = jact.StateSpace(["active", "dead"], [("active", "dead")])
    model = ss.build(
        transitions={("active", "dead"): lambda t, d, **kw: kw["basis"]},
        derived={"basis": lambda d, scale: jnp.sin(d * scale)},
    )
    steps = 16

    def solve(scale):
        return model.solve(
            initial="active", horizon=1, steps_per_unit=steps, scale=scale
        ).probability

    scale = jax.device_put(jnp.array(0.7), jax.local_devices(backend="cpu")[0])
    compiled = jax.jit(solve).lower(scale).compile()
    executable = compiled.runtime_executable()
    assert executable is not None
    hlo = executable.hlo_modules()[0].to_string()
    grid_sines = [
        line
        for line in hlo.splitlines()
        if " sine(" in line and f"[1,{steps}]" in line
    ]
    assert grid_sines
    assert all("while/body" not in line for line in grid_sines)
    assert any(
        " sine(" in line and "while/body" in line and f"[1,{steps}]" not in line
        for line in hlo.splitlines()
    )
    assert jnp.isfinite(jax.grad(lambda x: solve(x)[-1, 0, 0])(jnp.array(0.7)))


def test_grouped_intensity_consumes_shared_field():
    ss = jact.StateSpace(
        ["active", "dead", "lapsed"],
        [("active", "dead"), ("active", "lapsed")],
    )

    def grouped(t, d, **kw):
        return jnp.stack((kw["surface"], 0.5 * kw["surface"]))

    model = ss.build(
        exits={"active": grouped},
        derived={"surface": lambda t, d, rate: rate + 0.01 * t + 0 * d},
    )
    result = model.solve(initial="active", horizon=1, steps_per_unit=4, rate=0.1)
    assert result.probability.shape == (5, 1, 3)
    assert jnp.all(result.probability[-1, 0] > 0)


def test_derived_fields_support_jit_and_grad():
    ss = jact.StateSpace(["active", "dead"], [("active", "dead")])
    model = ss.build(
        transitions={("active", "dead"): lambda t, d, **kw: kw["hazard"] + 0 * d},
        derived={"hazard": lambda rate, t: rate + 0.1 * t},
    )

    @jax.jit
    def survival(rate):
        return model.solve(
            initial="active",
            horizon=1,
            steps_per_unit=4,
            rate=rate,
        ).probability[-1, 0, 0]

    value = survival(jnp.asarray(0.2))
    assert jnp.isfinite(value)
    assert jnp.isfinite(jax.grad(survival)(jnp.asarray(0.2)))


def test_callable_instance_without_hash_can_define_field():
    @dataclass
    class RateField:
        def __call__(self, rate):
            return rate * 2

    ss = jact.StateSpace(["active", "dead"], [("active", "dead")])
    model = ss.build(
        transitions={("active", "dead"): lambda t, d, **kw: kw["hazard"] + 0 * d},
        derived={"hazard": RateField()},
    )
    result = model.solve(initial="active", horizon=1, steps_per_unit=2, rate=0.1)
    assert jnp.allclose(result.probability[-1, 0, 0], jnp.exp(-0.2))


def test_derived_fields_on_two_cpu_devices():
    script = """
import jact
import jax.numpy as jnp
ss = jact.StateSpace(['active', 'dead'], [('active', 'dead')])
model = ss.build(
    transitions={('active', 'dead'): lambda t, d, **kw: kw['rate'][:, None] + 0*d},
    derived={'rate': lambda input_rate: input_rate * 2},
)
r = model.solve(
    initial='active', horizon=1, steps_per_unit=2, devices=2,
    input_rate=jnp.array([0.1, 0.2, 0.3, 0.4]),
)
assert r.probability.shape == (3, 4, 2)
expected = jnp.exp(-2*jnp.array([0.1, 0.2, 0.3, 0.4]))
assert jnp.allclose(r.probability[-1, :, 0], expected)
grid_model = ss.build(
    transitions={('active', 'dead'): lambda t, d, **kw: 0.1 + kw['basis']},
    derived={'basis': lambda d, input_rate: jnp.sin(d * input_rate[:, None])},
)
inputs = dict(
    initial='active', horizon=1, steps_per_unit=4,
    input_rate=jnp.array([0.1, 0.2, 0.3, 0.4]),
)
parallel = grid_model.solve(devices=2, **inputs)
serial = grid_model.solve(**inputs)
assert jnp.allclose(parallel.probability, serial.probability)
"""
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    subprocess.run([sys.executable, "-c", script], env=env, check=True)
