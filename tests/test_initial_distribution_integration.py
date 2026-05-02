"""Integration tests for InitialDistribution-related public behavior."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

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

    def test_initial_distribution_rejects_non_mapping_component_payload(
        self, illness_death
    ):
        state_space, _ = illness_death
        with pytest.raises(TypeError, match="payload must be a mapping"):
            state_space.initial_distribution(
                components={"healthy": 1.0}  # type: ignore[arg-type]
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

    def test_initial_per_individual_rejects_name_outside_restricted_initial_states(
        self, illness_death
    ):
        state_space, _ = illness_death
        with pytest.raises(ValueError, match="is not a valid initial state"):
            state_space.initial_per_individual(
                state_names=["dead"],
                initial_states=("healthy", "disabled"),
            )

    def test_initial_per_individual_rejects_float_state_indices(self, illness_death):
        state_space, _ = illness_death
        with pytest.raises(TypeError, match="integer dtype"):
            state_space.initial_per_individual(
                state_indices=jnp.array([0.0, 1.0], dtype=jnp.float32),
            )


class TestModelReduction:
    def test_reduce_accepts_multiple_initial_states(self, illness_death):
        _, model = illness_death
        reduced = model.reduce(("healthy", "disabled"))

        assert reduced.initial_states == ("healthy", "disabled")
        assert reduced.reachable_states == ("healthy", "disabled", "dead")
        assert reduced.n_states == 3


class TestInitialDistributionSolveIntegration:
    def test_at_flows_through_solver_with_initial_duration(self, illness_death):
        _, model = illness_death

        result = model.solve(
            initial=jact.InitialDistribution.at(
                "healthy", duration=jnp.array([0.0, 1.0])
            ),
            horizon=1,
            steps_per_unit=4,
            probability="point_mass",
            age=jnp.arange(2, dtype=jnp.float32),
        )

        point_result = result["probability"]

        assert result["states"] == ("healthy", "disabled", "dead")
        assert "healthy" in point_result
        assert "disabled" not in point_result
        assert "dead" not in point_result
        healthy_point = point_result["healthy"]
        assert healthy_point["value"].shape == (5, 2)
        assert jnp.allclose(healthy_point["value"][0], jnp.ones((2,)))
        assert jnp.allclose(healthy_point["d_0"][0], jnp.array([0.0, 1.0]))

    def test_per_individual_none_initial_states_uses_full_model_for_solver(
        self, illness_death
    ):
        state_space, model = illness_death
        dist = jact.InitialDistribution.per_individual(
            states=jnp.array(
                [
                    state_space.state_index("healthy"),
                    state_space.state_index("disabled"),
                ],
                dtype=jnp.int32,
            ),
            duration=jnp.array([0.0, 0.0]),
            initial_states=None,
        )

        result = model.solve(
            initial=dist,
            horizon=1,
            steps_per_unit=4,
            probability="point_mass",
            age=jnp.arange(2, dtype=jnp.float32),
        )

        point_result = result["probability"]

        assert result["states"] == state_space.states
        assert set(point_result.keys()) == {"healthy", "disabled", "dead"}
        assert point_result["healthy"]["value"].shape == (5, 2)
        assert point_result["disabled"]["value"].shape == (5, 2)
        assert point_result["dead"]["value"].shape == (5, 2)
        assert jnp.allclose(point_result["healthy"]["value"][0], jnp.array([1.0, 0.0]))
        assert jnp.allclose(point_result["disabled"]["value"][0], jnp.array([0.0, 1.0]))
        assert jnp.allclose(point_result["dead"]["value"][0], jnp.zeros((2,)))

    def test_component_mixture_seeds_only_declared_states(self, illness_death):
        _, model = illness_death
        dist = jact.InitialDistribution(
            components={
                "healthy": {
                    "mass": jnp.array([0.25, 0.75]),
                    "duration": jnp.array([0.0, 0.0]),
                },
                "disabled": {
                    "mass": jnp.array([0.75, 0.25]),
                    "duration": jnp.array([1.0, 0.0]),
                },
            }
        )

        result = model.solve(
            initial=dist,
            horizon=1,
            steps_per_unit=4,
            probability="point_mass",
            age=jnp.arange(2, dtype=jnp.float32),
        )

        point_result = result["probability"]

        assert result["states"] == ("healthy", "disabled", "dead")
        assert set(point_result.keys()) == {"healthy", "disabled"}
        assert point_result["healthy"]["value"].shape == (5, 2)
        assert point_result["disabled"]["value"].shape == (5, 2)
        healthy_value = point_result["healthy"]["value"]
        disabled_value = point_result["disabled"]["value"]
        disabled_d_0 = point_result["disabled"]["d_0"]
        assert jnp.allclose(healthy_value[0], jnp.array([0.25, 0.75]))
        assert jnp.allclose(disabled_value[0], jnp.array([0.75, 0.25]))
        assert jnp.allclose(disabled_d_0[0], jnp.array([1.0, 0.0]))

    def test_component_mixture_all_zero_normalized_mass_stays_zero(self, illness_death):
        state_space, _ = illness_death
        dist = state_space.initial_distribution(
            components={
                "healthy": {"mass": 0.0, "duration": 0.0},
                "disabled": {"mass": 0.0, "duration": 1.0},
            },
            normalise=True,
        )

        canonical = dist.canonicalize(state_space.states)

        assert canonical.states == ("healthy", "disabled")
        assert jnp.array_equal(canonical.masses[0], jnp.asarray(0.0))
        assert jnp.array_equal(canonical.masses[1], jnp.asarray(0.0))

    def test_initial_distribution_batch_mismatch_fails_at_solver_entry(
        self, illness_death
    ):
        _, model = illness_death
        dist = jact.InitialDistribution(
            components={
                "healthy": {
                    "mass": jnp.ones((3,)),
                    "duration": jnp.zeros((3,)),
                }
            }
        )

        with pytest.raises(ValueError, match="batch size"):
            model.solve(
                initial=dist,
                horizon=1,
                steps_per_unit=4,
                age=jnp.arange(2, dtype=jnp.float32),
            )
