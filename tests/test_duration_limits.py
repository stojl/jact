"""Duration approximations: numerical references, transport, and public outputs."""

from __future__ import annotations

import inspect
import os
import subprocess
import sys
import textwrap
from itertools import product
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jact
from jact import cashflows as cf
from jact import probability as pr


def _hazard(t, d, **kwargs):
    return kwargs.get("rate", 0.3) * (1 + 0.2 * d + 0.1 * t)


def _payment(t, d, **kwargs):
    return 1 + t + d * d


def _constant(t, d, **kwargs):
    return kwargs.get("rate", 0.3) + 0 * d


def _unit(t, d, **kwargs):
    return jnp.ones_like(d)


def _clamp(fn, limit):
    def call(t, d, **kwargs):
        return fn(t, d if limit is None else jnp.minimum(d, limit), **kwargs)

    return call


def _space():
    return jact.StateSpace(states=("a", "b"), transitions=(("a", "b"), ("b", "a")))


def _model(space, fn=_hazard):
    return space.build(transitions={("a", "b"): fn, ("b", "a"): fn})


def _cashflows(space, fn=_payment):
    return space.cashflows(
        {
            "rate": cf.StateRate({"a": fn, "b": fn}),
            "jump": cf.TransitionLump({("a", "b"): fn, ("b", "a"): fn}),
            "scheduled": cf.ScheduledEvent(
                when=lambda **kw: 2.5, payments={"a": fn, "b": fn}
            ),
            "duration": cf.DurationEvent(
                at_durations={"a": 1.5, "b": 1.5}, payments={"a": fn, "b": fn}
            ),
        }
    )


def _assert_tree_close(left, right, atol=2e-6):
    assert jax.tree_util.tree_structure(left) == jax.tree_util.tree_structure(right)
    for a, b in zip(jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right)):
        np.testing.assert_allclose(a, b, atol=atol, rtol=2e-6)


@pytest.mark.parametrize(
    "name",
    [
        "intensity_duration_limit",
        "payment_duration_limit",
        "probability_duration_limit",
    ],
)
def test_limits_are_keyword_only_static_python_scalars(name):
    model = _model(_space())
    for entry in (jact.solve, jact.Model.solve):
        parameter = inspect.signature(entry).parameters[name]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is None
    for invalid in (
        True,
        False,
        -0.1,
        float("nan"),
        float("inf"),
        1j,
        "1",
        jnp.asarray(1.0),
        np.array([1.0]),
    ):
        with pytest.raises(ValueError, match=name):
            model.solve("a", 1, 2, **cast(dict[str, Any], {name: invalid}))


@pytest.mark.parametrize(
    "intensity,payment", list(product((None, 0.0, 0.6, 1.0), repeat=2))
)
def test_function_limits_match_clamped_reference(intensity, payment):
    space = _space()
    model = _model(space)
    reference = _model(space, _clamp(_hazard, intensity))
    args: dict[str, Any] = dict(
        initial="a",
        horizon=3,
        steps_per_unit=4,
        initial_duration=jnp.array([0.0, 4.0]),
        probability=pr.Full(),
    )
    actual = model.solve(
        **args,
        cashflows=_cashflows(space),
        intensity_duration_limit=intensity,
        payment_duration_limit=payment,
    )
    expected = reference.solve(
        **args, cashflows=_cashflows(space, _clamp(_payment, payment))
    )
    _assert_tree_close(actual, expected)
    if intensity is None:
        _assert_tree_close(actual.probability, model.solve(**args).probability)


def test_none_and_beyond_sample_limits_preserve_default():
    space = _space()
    model = _model(space)
    args: dict[str, Any] = dict(
        initial="a",
        horizon=3,
        steps_per_unit=4,
        initial_duration=6.0,
        probability=pr.Full(),
        cashflows=_cashflows(space),
    )
    exact = model.solve(**args)
    assert set(exact.probability) == {"density", "point_mass"}
    _assert_tree_close(
        exact,
        model.solve(**args, intensity_duration_limit=20, payment_duration_limit=20),
    )
    disabled = model.solve("a", 1, 2, probability=pr.Tail()).probability
    assert set(disabled) == {"mass", "duration"}
    for value in disabled.values():
        np.testing.assert_array_equal(value, np.zeros((3, 1, 2)))


@pytest.mark.parametrize(
    "limit,width", [(0.0, 1), (0.6, 3), (1.0, 5), (0.3 - 1e-16, 2), (20.0, 12)]
)
def test_compression_conserves_mass_and_keeps_fixed_tail(limit, width):
    # The snapping case uses a different resolution.
    spu = 10 if limit == 0.3 - 1e-16 else 4
    if spu == 10:
        width = 4
    space = _space()
    model = _model(space, _constant)
    args: dict[str, Any] = dict(
        initial="a", horizon=3, steps_per_unit=spu, initial_duration=8.0
    )
    full = model.solve(
        **args, probability=pr.Full(), probability_duration_limit=limit
    ).probability
    assert full["density"].shape == (3 * spu + 1, 1, 2, width)
    assert "tail" in full
    tail = full["tail"]
    np.testing.assert_array_equal(
        tail["duration"], np.full((3 * spu + 1, 1, 2), limit, dtype=np.float32)
    )
    np.testing.assert_array_equal(tail["mass"][0], 0)
    continuous = full["density"].sum(axis=-1) + tail["mass"]
    total = continuous.at[:, :, 0].add(full["point_mass"]["a"])
    np.testing.assert_allclose(total.sum(axis=-1), 1.0, atol=2e-6)
    assert np.all(full["density"] >= 0) and np.all(tail["mass"] >= 0)
    np.testing.assert_allclose(total, model.solve(**args).probability, atol=2e-6)
    components = model.solve(
        **args, probability=pr.MarginalComponents(), probability_duration_limit=limit
    ).probability
    np.testing.assert_allclose(components["density"], continuous, atol=2e-6)
    _assert_tree_close(components["point_mass"], full["point_mass"])
    for reducer, expected in (
        (pr.Density(), full["density"]),
        (pr.DensityProbability(), continuous),
        (pr.Tail(), tail),
    ):
        result = model.solve(
            **args, probability=cast(Any, reducer), probability_duration_limit=limit
        )
        _assert_tree_close(result.probability, expected)


@pytest.mark.parametrize(
    "intensity,payment,probability",
    [
        (None, None, 0.0),
        (0.2, None, 0.8),
        (None, 0.2, 0.8),
        (0.2, 0.4, 0.8),
        (0.8, 0.4, 0.2),
        (0.4, 0.8, 0.2),
        (0.0, 0.0, 0.0),
    ],
)
def test_composition_reference_on_same_compressed_grid(intensity, payment, probability):
    space = _space()
    args: dict[str, Any] = dict(
        initial="a",
        horizon=3,
        steps_per_unit=4,
        initial_duration=2.0,
        probability=pr.Full(),
        probability_duration_limit=probability,
    )
    actual = _model(space).solve(
        **args,
        cashflows=_cashflows(space),
        intensity_duration_limit=intensity,
        payment_duration_limit=payment,
    )
    expected = _model(space, _clamp(_hazard, intensity)).solve(
        **args, cashflows=_cashflows(space, _clamp(_payment, payment))
    )
    _assert_tree_close(actual, expected)


def test_one_cell_inflow_tail_survival_and_fixed_duration_payments():
    space = jact.StateSpace(states=("a", "b"), transitions=(("a", "b"),))
    model = space.build(transitions={("a", "b"): _constant})
    declaration = space.cashflows(
        {
            "rate": cf.StateRate({"b": lambda t, d, **kw: d}),
            "scheduled": cf.ScheduledEvent(
                when=lambda **kw: 2.0, payments={"b": lambda t, d, **kw: d}
            ),
        }
    )
    result = model.solve(
        "a",
        3,
        1,
        probability=pr.Full(),
        cashflows=declaration,
        probability_duration_limit=0.2,
    )
    full = result.probability
    assert "tail" in full
    b = result.states.index("b")
    transfer = -np.expm1(-0.3)
    np.testing.assert_allclose(full["density"][1, 0, b, 0], 0.5 * transfer, atol=1e-7)
    np.testing.assert_allclose(full["tail"]["mass"][1, 0, b], 0.5 * transfer, atol=1e-7)
    np.testing.assert_allclose(
        full["tail"]["mass"][2, 0, b],
        transfer + 0.5 * np.exp(-0.3) * transfer,
        atol=1e-7,
    )
    raw = cast(Any, result.cashflows)["raw"]
    np.testing.assert_allclose(
        raw["scheduled"][2, 0], 0.2 * full["tail"]["mass"][2, 0, b], atol=1e-7
    )
    expected_rate = (
        0.5 * full["density"][2, 0, b, 0]
        + 0.2 * full["tail"]["mass"][2, 0, b]
        + 0.25 * np.exp(-0.6) * transfer
    )
    np.testing.assert_allclose(raw["rate"][2, 0], expected_rate, atol=1e-7)


def test_tail_exits_use_cutoff_hazard_and_competing_exit_factors():
    space = jact.StateSpace(
        states=("a", "b", "c", "d"), transitions=(("a", "b"), ("b", "d"), ("b", "c"))
    )
    model = space.build(
        transitions={
            ("a", "b"): _constant,
            ("b", "d"): lambda t, d, **kw: 0.2 + d,
            ("b", "c"): lambda t, d, **kw: 0.4 + 2 * d,
        }
    )
    result = model.solve(
        "a", 3, 2, probability=pr.Full(), probability_duration_limit=0.2
    )
    full = result.probability
    assert "tail" in full
    b = result.states.index("b")
    # K=1; all surviving regular mass plus half the new inflow enters the tail.
    # New regular cell is precisely that other half (b has no chained incoming).
    survival = np.exp(-0.5 * (0.6 + 3 * 0.2))
    regular_survival = np.exp(-0.5 * (0.6 + 3 * 0.25))
    expected = (
        full["tail"]["mass"][:-1, 0, b] * survival
        + full["density"][:-1, 0, b, 0] * regular_survival
        + full["density"][1:, 0, b, 0]
    )
    np.testing.assert_allclose(full["tail"]["mass"][1:, 0, b], expected, atol=1e-7)


@pytest.mark.parametrize("target", [0.0, 0.5, 0.75, 1.5, 3.0])
def test_duration_events_only_retained_cells_and_true_initial_points(target):
    space = jact.StateSpace(states=("a", "b"), transitions=(("a", "b"),))
    model = space.build(transitions={("a", "b"): _constant})
    declaration = space.cashflows(
        {
            "a_event": cf.DurationEvent(
                at_durations={"a": target}, payments={"a": _unit}
            ),
            "b_event": cf.DurationEvent(
                at_durations={"b": target}, payments={"b": _unit}
            ),
            "late": cf.ScheduledEvent(
                when=lambda **kw: 2.5, payments={"a": _unit, "b": _unit}
            ),
        }
    )
    args: dict[str, Any] = dict(
        initial="a",
        horizon=3,
        steps_per_unit=4,
        initial_duration=0.25,
        cashflows=declaration,
        probability=pr.Full(),
    )
    exact = model.solve(**args)
    compressed = model.solve(**args, probability_duration_limit=0.5)
    raw = cast(Any, compressed.cashflows)["raw"]
    exact_raw = cast(Any, exact.cashflows)["raw"]
    np.testing.assert_allclose(raw["a_event"], exact_raw["a_event"], atol=1e-7)
    if target <= 0.5:
        np.testing.assert_allclose(raw["b_event"], exact_raw["b_event"], atol=1e-7)
    else:
        np.testing.assert_array_equal(raw["b_event"], 0)
    np.testing.assert_allclose(raw["late"].sum(), 1.0, atol=1e-6)


def test_traced_duration_target_outer_jit_gradients_and_recording():
    space = _space()
    model = _model(space)
    declaration = space.cashflows(
        {
            "event": cf.DurationEvent(
                at_durations={"a": lambda **kw: kw["target"]}, payments={"a": _payment}
            ),
            "rate": cf.StateRate({"a": _payment, "b": _payment}),
        }
    )
    views = {"stream": cf.Total(), "terminal": cf.Total(terminal=True)}

    def solve(rate, target, record_every=1):
        return model.solve(
            "a",
            3,
            4,
            initial_duration=0.5,
            rate=rate,
            target=target,
            cashflows=declaration,
            cashflow_views=views,
            probability=pr.Tail(),
            record_every=record_every,
            intensity_duration_limit=0.4,
            payment_duration_limit=0.3,
            probability_duration_limit=0.2,
        )

    compiled = jax.jit(solve)
    result = compiled(jnp.asarray(0.3), jnp.asarray(1.5))
    blocked = solve(jnp.asarray(0.3), jnp.asarray(1.5), record_every=3)
    _assert_tree_close(
        jax.tree_util.tree_map(lambda x: x[::3], result.probability),
        blocked.probability,
    )
    raw = cast(Any, result.cashflows)
    np.testing.assert_allclose(raw["stream"].sum(axis=0), raw["terminal"], atol=2e-6)
    _assert_tree_close(raw["terminal"], cast(Any, blocked.cashflows)["terminal"])

    def value(rate):
        return cast(Any, solve(rate, jnp.asarray(1.5)).cashflows)["terminal"].sum()

    gradient = jax.jit(jax.grad(value))(jnp.asarray(0.3))
    finite_difference = (value(0.301) - value(0.299)) / 0.002
    np.testing.assert_allclose(gradient, finite_difference, rtol=0.01, atol=0.005)


def test_callables_receive_compact_grids_and_no_redundant_threshold():
    intensity_shapes = []
    midpoint_shapes = []
    left_shapes = []

    def intensity(t, d, **kw):
        intensity_shapes.append(d.shape)
        return 0.1 + 0.1 * d

    def mid_payment(t, d, **kw):
        midpoint_shapes.append(d.shape)
        return 1 + d

    def left_payment(t, d, **kw):
        left_shapes.append(d.shape)
        return 1 + d

    space = _space()
    model = _model(space, intensity)
    declaration = space.cashflows(
        {
            "rate": cf.StateRate({"b": mid_payment}),
            "scheduled": cf.ScheduledEvent(
                when=lambda **kw: 2.0, payments={"b": left_payment}
            ),
        }
    )
    model.solve(
        "a",
        3,
        4,
        cashflows=declaration,
        intensity_duration_limit=0.5,
        payment_duration_limit=0.5,
    )
    assert (1, 3) in intensity_shapes and max(s[-1] for s in intensity_shapes) == 3
    assert midpoint_shapes == [(1, 3)]
    assert left_shapes == [(1, 3)]  # 0, .25, .5; cutoff already sampled
    intensity_shapes.clear()
    midpoint_shapes.clear()
    left_shapes.clear()
    model.solve(
        "a",
        3,
        4,
        cashflows=declaration,
        intensity_duration_limit=20.0,
        payment_duration_limit=20.0,
    )
    assert max(s[-1] for s in intensity_shapes) == 12
    assert max(s[-1] for s in midpoint_shapes) == 12
    assert left_shapes == [(1, 12)]
    left_shapes.clear()
    midpoint_shapes.clear()
    model.solve(
        "a",
        3,
        4,
        cashflows=declaration,
        intensity_duration_limit=0.5,
        payment_duration_limit=0.5,
        probability_duration_limit=0.5,
    )
    # The cutoff is both the last left node and the tail duration.
    assert left_shapes == [(1, 3)]
    assert midpoint_shapes == [(1, 3)]


def test_custom_callback_and_fitted_wrapper():
    def features(t, d, **kwargs):
        return jnp.stack((jnp.broadcast_to(t, d.shape), d), axis=-1)

    intensity = jact.wrappers.bind_intensity(
        lambda params, x: 0.2 + params["slope"] * x[..., 1],
        {"slope": 0.1},
        features,
    )
    model = _model(_space(), intensity)

    def callback(state):
        tail = state[0].tail
        assert tail is not None
        return {"regular": state[0].density, "tail": (tail.mass, tail.duration)}

    result = model.solve(
        "a",
        2,
        4,
        probability=callback,
        intensity_duration_limit=0.3,
        probability_duration_limit=0.5,
    )
    assert result.probability["regular"].shape == (9, 1, 3)
    np.testing.assert_array_equal(result.probability["tail"][1], 0.5)


def test_padded_multi_device_batch_matches_single_device():
    script = textwrap.dedent("""
        import jax
        import jax.numpy as jnp
        from typing import Any
        from tests.test_duration_limits import (
            _space, _model, _cashflows, _assert_tree_close,
        )
        from jact import probability as pr, cashflows as cf
        space = _space()
        model = _model(space)
        args: dict[str, Any] = dict(initial="a", horizon=3, steps_per_unit=4,
                    initial_duration=jnp.arange(5.) / 2,
                    rate=jnp.arange(5.)[:, None] / 10 + .1,
                    probability=pr.Full(), cashflows=_cashflows(space),
                    cashflow_views={"stream": cf.Total(),
                                    "terminal": cf.Total(terminal=True)},
                    record_every=3, intensity_duration_limit=.3,
                    payment_duration_limit=.6, probability_duration_limit=.2)
        reference = model.solve(**args)
        for devices in (2, tuple(reversed(jax.local_devices()))):
            actual = model.solve(**args, devices=devices)
            _assert_tree_close(actual, reference)
            assert actual.probability["tail"]["mass"].shape == (5, 5, 2)
    """)
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        XLA_FLAGS="--xla_force_host_platform_device_count=2",
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("function_limit", [None, 0.2])
def test_compression_preserves_constant_tail_cashflows(function_limit):
    space = _space()
    fn = _constant if function_limit is None else _hazard
    payment = _unit if function_limit is None else _payment
    model = _model(space, fn)
    declaration = space.cashflows(
        {
            "rate": cf.StateRate({"a": payment, "b": payment}),
            "jump": cf.TransitionLump({("a", "b"): payment, ("b", "a"): payment}),
            "scheduled": cf.ScheduledEvent(
                when=lambda **kw: 2.5, payments={"a": payment, "b": payment}
            ),
        }
    )
    args: dict[str, Any] = dict(
        initial="a",
        initial_duration=4.0,
        horizon=3,
        steps_per_unit=4,
        cashflows=declaration,
        record_every=2,
        cashflow_views={"total": cf.Total(weight=lambda t, **kw: jnp.exp(-0.05 * t))},
        intensity_duration_limit=function_limit,
        payment_duration_limit=function_limit,
    )
    exact = model.solve(**args)
    compressed = model.solve(**args, probability_duration_limit=0.5)
    _assert_tree_close(exact, compressed)
