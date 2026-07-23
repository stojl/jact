"""Tests for the JAX-native semi-Markov event simulator."""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jact
from jact.simulation import (
    _compile_simulation_plan,
    _evaluate_intensity_slab,
)


def _two_state_model(rate: Any) -> jact.Model:
    state_space = jact.StateSpace(
        states=("active", "absorbed"),
        transitions=(("active", "absorbed"),),
    )
    return state_space.build(
        transitions={("active", "absorbed"): rate},
    )


def test_constant_hazard_has_exponential_event_probability():
    rate = 0.4
    model = _two_state_model(lambda t, d, **kwargs: rate)

    result = model.simulate(
        initial="active",
        horizon=2,
        steps_per_unit=4,
        max_jumps=1,
        replicates=20_000,
        key=jax.random.key(123),
    )

    expected = 1.0 - math.exp(-rate * 2.0)
    observed = float(jnp.mean(result.final_state == 1))
    assert observed == pytest.approx(expected, abs=0.012)
    assert not bool(jnp.any(result.overflow))
    valid_times = np.asarray(result.jump_times)[np.asarray(result.valid_mask())]
    assert np.all(valid_times > 0.0)
    assert np.all(valid_times <= 2.0)
    assert np.any(np.mod(valid_times, 0.25) != 0.0)


def test_empirical_final_state_probability_agrees_with_solve():
    model = _two_state_model(lambda t, d, **kwargs: 0.7)
    solved = model.solve(
        initial="active",
        horizon=2,
        steps_per_unit=8,
    )
    simulated = model.simulate(
        initial="active",
        horizon=2,
        steps_per_unit=8,
        max_jumps=1,
        replicates=30_000,
        key=jax.random.key(321),
    )

    expected = float(solved.probability[-1, 0, 1])
    observed = float(jnp.mean(simulated.final_state == 1))
    assert observed == pytest.approx(expected, abs=0.01)


def test_competing_grouped_hazards_select_expected_destination():
    state_space = jact.StateSpace(
        states=("active", "cause_a", "cause_b"),
        transitions=(
            ("active", "cause_a"),
            ("active", "cause_b"),
        ),
    )

    def grouped(t, d, **kwargs):
        return jnp.asarray([0.25, 0.75])

    model = state_space.build(exits={"active": grouped})
    result = model.simulate(
        initial="active",
        horizon=8,
        steps_per_unit=2,
        max_jumps=1,
        replicates=20_000,
        key=jax.random.key(7),
    )

    event = np.asarray(result.jump_count[0]) == 1
    destination = np.asarray(result.final_state[0])[event]
    assert event.mean() > 0.99
    assert np.mean(destination == 1) == pytest.approx(0.25, abs=0.012)


@pytest.mark.parametrize(
    ("rate", "expected"),
    [
        (lambda t, d, **kwargs: 2.0, np.full((2, 3), 2.0)),
        (
            lambda t, d, **kwargs: jnp.arange(3.0),
            np.broadcast_to(np.arange(3.0), (2, 3)),
        ),
        (
            lambda t, d, **kwargs: kwargs["value"],
            np.asarray([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]),
        ),
        (
            lambda t, d, **kwargs: kwargs["value"][:, None] + d,
            np.asarray([[1.5, 2.5, 3.5], [2.5, 3.5, 4.5]]),
        ),
    ],
)
def test_intensity_slab_broadcasting(rate, expected):
    model = _two_state_model(rate)
    plan = _compile_simulation_plan(model, ("active",))
    rates, invalid_negative, invalid_nonfinite = _evaluate_intensity_slab(
        plan,
        jnp.asarray(0.5),
        jnp.asarray([0.5, 1.5, 2.5]),
        {"value": jnp.asarray([1.0, 2.0])},
        batch_size=2,
        negative_tolerance=1e-12,
    )

    assert np.allclose(np.asarray(rates[0]), expected)
    assert not bool(invalid_negative)
    assert not bool(invalid_nonfinite)


def test_duration_and_calendar_discontinuities_are_respected():
    duration_model = _two_state_model(
        lambda t, d, **kwargs: jnp.where(d < 1.0, 0.0, 2.0)
    )
    duration_result = duration_model.simulate(
        initial="active",
        horizon=2,
        steps_per_unit=4,
        max_jumps=1,
        replicates=8_000,
        key=jax.random.key(1),
    )
    duration_times = np.asarray(duration_result.jump_times)[
        np.asarray(duration_result.valid_mask())
    ]
    assert np.all(duration_times >= 1.0)
    assert duration_times.size > 4_000

    calendar_model = _two_state_model(
        lambda t, d, **kwargs: jnp.where(t < 1.0, 0.0, 2.0)
    )
    calendar_result = calendar_model.simulate(
        initial="active",
        horizon=2,
        steps_per_unit=4,
        max_jumps=1,
        replicates=8_000,
        key=jax.random.key(2),
    )
    calendar_times = np.asarray(calendar_result.jump_times)[
        np.asarray(calendar_result.valid_mask())
    ]
    assert np.all(calendar_times >= 1.0)
    assert calendar_times.size > 4_000


def test_multiple_jumps_reset_duration_and_follow_declared_edges():
    state_space = jact.StateSpace(
        states=("a", "b"),
        transitions=(("a", "b"), ("b", "a")),
    )
    model = state_space.build(
        transitions={
            ("a", "b"): lambda t, d, **kwargs: 3.0,
            ("b", "a"): lambda t, d, **kwargs: 3.0,
        }
    )
    result = model.simulate(
        initial="a",
        horizon=2,
        steps_per_unit=4,
        max_jumps=32,
        replicates=2_000,
        key=jax.random.key(5),
    )

    times = np.asarray(result.jump_times)
    durations = np.asarray(result.jump_durations)
    paths = np.asarray(result.state_path)
    counts = np.asarray(result.jump_count)
    assert not bool(jnp.any(result.overflow))
    for replicate, count in enumerate(counts[0]):
        if count == 0:
            continue
        trajectory_times = times[0, replicate, :count]
        trajectory_durations = durations[0, replicate, :count]
        trajectory_states = paths[0, replicate, : count + 1]
        assert np.all(np.diff(trajectory_times) > 0.0)
        assert trajectory_times[0] == pytest.approx(trajectory_durations[0])
        if count > 1:
            assert np.allclose(
                np.diff(trajectory_times),
                trajectory_durations[1:],
                atol=1e-6,
            )
        assert np.all(trajectory_states[1:] != trajectory_states[:-1])


def test_per_individual_covariates_initial_states_and_replicates():
    state_space = jact.StateSpace(
        states=("a", "b", "done"),
        transitions=(("a", "done"), ("b", "done")),
    )

    def rate(t, d, **kwargs):
        del t, d
        return kwargs["hazard"]

    model = state_space.build(
        transitions={
            ("a", "done"): rate,
            ("b", "done"): rate,
        }
    )
    initial = state_space.initial_per_individual(
        state_names=("a", "b"),
        duration=jnp.asarray([0.0, 0.5]),
    )
    result = model.simulate(
        initial=initial,
        horizon=1,
        steps_per_unit=2,
        max_jumps=1,
        replicates=3,
        key=jax.random.key(9),
        hazard=jnp.asarray([[0.0], [100.0]]),
    )

    assert result.state_path.shape == (2, 3, 2)
    assert np.all(np.asarray(result.state_path[0, :, 0]) == 0)
    assert np.all(np.asarray(result.state_path[1, :, 0]) == 1)
    assert np.all(np.asarray(result.jump_count[0]) == 0)
    assert np.all(np.asarray(result.jump_count[1]) == 1)


def test_component_mixture_sampling_and_structural_reduction():
    state_space = jact.StateSpace(
        states=("unreachable", "a", "b", "done"),
        transitions=(("a", "done"), ("b", "done")),
    )
    model = state_space.build(
        transitions={
            ("a", "done"): lambda t, d, **kwargs: 0.0,
            ("b", "done"): lambda t, d, **kwargs: 0.0,
        }
    )
    initial = jact.InitialDistribution(
        components={
            "a": {"mass": 0.2, "duration": 0.0},
            "b": {"mass": 0.8, "duration": 1.0},
        }
    )
    result = model.simulate(
        initial=initial,
        horizon=1,
        steps_per_unit=2,
        max_jumps=1,
        replicates=10_000,
        key=jax.random.key(11),
    )

    assert result.states == ("a", "b", "done")
    initial_states = np.asarray(result.state_path[0, :, 0])
    assert np.mean(initial_states == 0) == pytest.approx(0.2, abs=0.012)
    assert np.all(
        np.asarray(result.final_duration[0])[initial_states == 0] == 1.0
    )
    assert np.all(
        np.asarray(result.final_duration[0])[initial_states == 1] == 2.0
    )


def test_overflow_zero_buffer_and_raise_policy():
    model = _two_state_model(lambda t, d, **kwargs: 100.0)
    returned = model.simulate(
        initial="active",
        horizon=1,
        steps_per_unit=1,
        max_jumps=0,
        replicates=8,
        key=jax.random.key(3),
    )
    assert returned.jump_times.shape == (1, 8, 0)
    assert returned.state_path.shape == (1, 8, 1)
    assert np.all(np.asarray(returned.overflow))
    assert np.all(np.asarray(returned.jump_count) == 0)

    with pytest.raises(RuntimeError, match="exceeded max_jumps=0"):
        model.simulate(
            initial="active",
            horizon=1,
            steps_per_unit=1,
            max_jumps=0,
            replicates=8,
            key=jax.random.key(3),
            overflow="raise",
        )


def test_reproducibility_result_helpers_and_pytree():
    model = _two_state_model(lambda t, d, **kwargs: 1.0)
    def run():
        return model.simulate(
            initial="active",
            horizon=1,
            steps_per_unit=4,
            max_jumps=1,
            replicates=20,
            key=jax.random.key(42),
        )

    first = run()
    second = run()

    assert jnp.array_equal(first.jump_count, second.jump_count)
    assert jnp.array_equal(
        first.jump_times,
        second.jump_times,
        equal_nan=True,
    )
    assert jax.tree_util.tree_structure(first) == jax.tree_util.tree_structure(
        second
    )
    assert jnp.array_equal(
        first.valid_mask(),
        jnp.arange(1) < first.jump_count[..., None],
    )
    path = first.path(0, 0)
    assert path["state_names"][0] == "active"
    different = model.simulate(
        initial="active",
        horizon=1,
        steps_per_unit=4,
        max_jumps=1,
        replicates=20,
        key=jax.random.key(43),
    )
    assert not jnp.array_equal(
        first.jump_times,
        different.jump_times,
        equal_nan=True,
    )


def test_module_level_simulate_and_single_device_option():
    model = _two_state_model(lambda t, d, **kwargs: 0.0)
    result = jact.simulate(
        model=model,
        initial="active",
        horizon=1,
        steps_per_unit=1,
        max_jumps=1,
        key=jax.random.key(0),
        devices=1,
    )
    assert isinstance(result, jact.SimulationResult)
    assert result.jump_times.shape == (1, 1, 1)


@pytest.mark.parametrize(
    ("rate", "message"),
    [
        (lambda t, d, **kwargs: -0.1, "non-negative"),
        (lambda t, d, **kwargs: jnp.nan, "finite"),
    ],
)
def test_invalid_intensities_are_rejected(rate, message):
    model = _two_state_model(rate)
    with pytest.raises(ValueError, match=message):
        model.simulate(
            initial="active",
            horizon=1,
            steps_per_unit=1,
            max_jumps=1,
            key=jax.random.key(0),
        )


def test_sparse_plan_edge_order_adjacency_and_group_preservation():
    state_space = jact.StateSpace(
        states=("z", "a", "b", "c"),
        transitions=(
            ("a", "c"),
            ("a", "b"),
            ("b", "c"),
        ),
    )

    def exits(t, d, **kwargs):
        return jnp.asarray([1.0, 2.0])

    model = state_space.build(
        exits={"a": exits},
        transitions={("b", "c"): lambda t, d, **kwargs: 3.0},
    )
    plan = _compile_simulation_plan(model, ("a",))

    assert plan.states == ("a", "b", "c")
    assert plan.edges == (("a", "b"), ("a", "c"), ("b", "c"))
    assert np.array_equal(
        np.asarray(plan.outgoing_valid).sum(axis=1),
        np.asarray([2, 1, 0]),
    )
    assert plan.intensity_blocks[0].fn is exits
    assert plan.intensity_blocks[0].edge_ids == (0, 1)
    assert plan.intensity_blocks[0].output_indices == (0, 1)


def test_absorbing_only_reduced_model_advances_duration():
    state_space = jact.StateSpace(
        states=("unused", "absorbed"),
        transitions=(("unused", "absorbed"),),
    )
    model = state_space.build(
        transitions={("unused", "absorbed"): lambda t, d, **kwargs: 1.0}
    )
    result = model.simulate(
        initial="absorbed",
        initial_duration=2.0,
        horizon=3,
        steps_per_unit=2,
        max_jumps=1,
        key=jax.random.key(0),
    )
    assert result.states == ("absorbed",)
    assert int(result.final_state[0, 0]) == 0
    assert float(result.final_duration[0, 0]) == pytest.approx(5.0)
