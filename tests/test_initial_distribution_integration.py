"""Integration tests for InitialDistribution-related public behavior."""

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

import jact


def _constant_intensity(t, d, **kwargs):
    age = kwargs["age"]
    return jnp.ones((age.shape[0], d.shape[-1]))


@pytest.fixture
def illness_death():
    state_space = jact.StateSpace(
        states=["healthy", "disabled", "dead"],
        transitions=[
            ("healthy", "disabled"),
            ("healthy", "dead"),
            ("disabled", "dead"),
        ],
    )
    model = state_space.build(
        transitions={
            ("healthy", "disabled"): _constant_intensity,
            ("healthy", "dead"): _constant_intensity,
            ("disabled", "dead"): _constant_intensity,
        }
    )
    return state_space, model


class TestStateSpaceHelpers:
    def test_initial_distribution_validates_component_names(self, illness_death):
        state_space, _ = illness_death
        with pytest.raises(ValueError):
            state_space.initial_distribution(
                components={"unknown": {"mass": 1.0, "duration": 0.0}}
            )

    def test_initial_per_individual_requires_exactly_one_input(self, illness_death):
        state_space, _ = illness_death
        with pytest.raises(ValueError):
            state_space.initial_per_individual(
                state_names=["healthy"],
                state_indices=jnp.array([0], dtype=jnp.int32),
            )

    def test_initial_per_individual_name_and_index_paths_match(self, illness_death):
        state_space, _ = illness_death
        by_name = state_space.initial_per_individual(
            state_names=["healthy", "disabled"],
            duration=jnp.array([0.0, 1.0]),
            initial_states=("healthy", "disabled"),
        )
        by_index = state_space.initial_per_individual(
            state_indices=jnp.array([0, 1], dtype=jnp.int32),
            duration=jnp.array([0.0, 1.0]),
            initial_states=("healthy", "disabled"),
        )

        canonical_name = by_name.canonicalize(state_space.states)
        canonical_index = by_index.canonicalize(state_space.states)

        assert canonical_name.states == canonical_index.states
        for left, right in zip(canonical_name.masses, canonical_index.masses):
            assert jnp.array_equal(left, right)


class TestModelReduction:
    def test_reduce_accepts_multiple_initial_states(self, illness_death):
        _, model = illness_death
        reduced = model.reduce(("healthy", "disabled"))

        assert reduced.initial_states == ("healthy", "disabled")
        assert reduced.reachable_states == ("healthy", "disabled", "dead")
        assert reduced.n_states == 3

