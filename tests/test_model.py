"""Unit tests for jact.Model / ReducedModel, per docs/api_spec.md."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

import jact
from jact.model import ReducedModel, TransitionInfo


def _single_transition_fn(t, grid, **kwargs):
    batch = kwargs["age"].shape[0]
    width = grid.shape[-1]
    return jnp.full((batch, width), 3.0)


def _healthy_mortality_group(t, grid, **kwargs):
    batch = kwargs["age"].shape[0]
    width = grid.shape[-1]
    return jnp.stack(
        [
            jnp.full((batch, width), 11.0),
            jnp.full((batch, width), 22.0),
        ],
        axis=0,
    )


def _disabled_exit_group(t, grid, **kwargs):
    batch = kwargs["age"].shape[0]
    width = grid.shape[-1]
    return jnp.stack(
        [
            jnp.full((batch, width), 101.0),
            jnp.full((batch, width), 202.0),
        ],
        axis=0,
    )


@pytest.fixture
def illness_death_state_space():
    return jact.StateSpace(
        states=["healthy", "disabled", "dead"],
        transitions=[
            ("healthy", "disabled"),
            ("healthy", "dead"),
            ("disabled", "dead"),
        ],
    )


@pytest.fixture
def mixed_assignment_state_space():
    return jact.StateSpace(
        states=["healthy", "disabled", "recovered", "dead", "archived"],
        transitions=[
            ("healthy", "disabled"),
            ("healthy", "dead"),
            ("disabled", "recovered"),
            ("disabled", "dead"),
            ("recovered", "dead"),
        ],
    )


@pytest.fixture
def mixed_assignment_model(mixed_assignment_state_space):
    return mixed_assignment_state_space.build(
        transitions={
            ("healthy", "disabled"): _single_transition_fn,
        },
        exits={
            "disabled": _disabled_exit_group,
        },
        groups={
            _healthy_mortality_group: [
                ("healthy", "dead"),
                ("recovered", "dead"),
            ],
        },
    )


class TestModelBuildValidation:
    def test_build_accepts_transition_assignments(self, illness_death_state_space):
        model = illness_death_state_space.build(
            transitions={
                ("healthy", "disabled"): _single_transition_fn,
                ("healthy", "dead"): _single_transition_fn,
                ("disabled", "dead"): _single_transition_fn,
            }
        )

        assert isinstance(model, jact.Model)
        assert model.state_space is illness_death_state_space

    def test_build_accepts_mixed_assignment_modes(self, mixed_assignment_model):
        assert isinstance(mixed_assignment_model, jact.Model)

    def test_build_rejects_missing_transition_coverage(
        self, illness_death_state_space
    ):
        with pytest.raises(ValueError, match="not covered by any model"):
            illness_death_state_space.build(
                transitions={
                    ("healthy", "disabled"): _single_transition_fn,
                    ("healthy", "dead"): _single_transition_fn,
                }
            )

    def test_build_rejects_overlapping_transition_assignment(
        self, illness_death_state_space
    ):
        with pytest.raises(ValueError, match="assigned multiple times"):
            illness_death_state_space.build(
                transitions={
                    ("healthy", "dead"): _single_transition_fn,
                    ("disabled", "dead"): _single_transition_fn,
                },
                exits={
                    "healthy": _disabled_exit_group,
                },
            )

    def test_build_rejects_exits_for_absorbing_state(self, illness_death_state_space):
        with pytest.raises(ValueError, match="has no outgoing transitions"):
            illness_death_state_space.build(
                exits={"dead": _disabled_exit_group},
                transitions={
                    ("healthy", "disabled"): _single_transition_fn,
                    ("healthy", "dead"): _single_transition_fn,
                    ("disabled", "dead"): _single_transition_fn,
                },
            )

    def test_build_rejects_unknown_transition(self, illness_death_state_space):
        with pytest.raises(ValueError, match="not declared in the StateSpace"):
            illness_death_state_space.build(
                transitions={
                    ("healthy", "disabled"): _single_transition_fn,
                    ("healthy", "dead"): _single_transition_fn,
                    ("dead", "healthy"): _single_transition_fn,
                    ("disabled", "dead"): _single_transition_fn,
                }
            )


class TestModelInfo:
    def test_info_returns_transition_metadata(self, mixed_assignment_model):
        info = mixed_assignment_model.info("disabled", "dead")

        assert isinstance(info, TransitionInfo)
        assert info.source == "disabled"
        assert info.target == "dead"
        assert info.assignment == "exits"
        assert info.callable is _disabled_exit_group
        assert info.index == 1

    def test_info_for_group_assignment_uses_list_order(self, mixed_assignment_model):
        info = mixed_assignment_model.info("recovered", "dead")

        assert info.assignment == "groups"
        assert info.callable is _healthy_mortality_group
        assert info.index == 1

    def test_info_for_single_transition_has_no_index(self, mixed_assignment_model):
        info = mixed_assignment_model.info("healthy", "disabled")

        assert info.assignment == "transitions"
        assert info.callable is _single_transition_fn
        assert info.index is None

    def test_info_rejects_unknown_transition(self, mixed_assignment_model):
        with pytest.raises(ValueError, match="No transition 'dead' -> 'healthy'"):
            mixed_assignment_model.info("dead", "healthy")


class TestReducedSolverMatrix:
    def test_transition_assignment_preserves_original_callable(
        self, mixed_assignment_model
    ):
        reduced = mixed_assignment_model.reduce("healthy")

        assert reduced.solver_matrix[0][1] is _single_transition_fn

    def test_exit_assignment_slices_targets_in_state_space_order(
        self, mixed_assignment_model
    ):
        reduced = mixed_assignment_model.reduce("disabled")
        batch = 2
        grid = jnp.zeros((batch, 4))
        age = jnp.arange(batch)

        recovered_fn = reduced.solver_matrix[0][1]
        dead_fn = reduced.solver_matrix[0][2]

        assert recovered_fn is not _disabled_exit_group
        assert dead_fn is not _disabled_exit_group
        assert jnp.array_equal(
            recovered_fn(0.0, grid, age=age),
            jnp.full((batch, 4), 101.0),
        )
        assert jnp.array_equal(
            dead_fn(0.0, grid, age=age),
            jnp.full((batch, 4), 202.0),
        )

    def test_group_assignment_slices_transitions_in_declared_group_order(
        self, mixed_assignment_model
    ):
        reduced = mixed_assignment_model.reduce("healthy")
        batch = 3
        grid = jnp.zeros((batch, 2))
        age = jnp.arange(batch)

        healthy_dead_fn = reduced.solver_matrix[0][3]
        recovered_dead_fn = reduced.solver_matrix[2][3]

        assert healthy_dead_fn is not _healthy_mortality_group
        assert recovered_dead_fn is not _healthy_mortality_group
        assert jnp.array_equal(
            healthy_dead_fn(0.0, grid, age=age),
            jnp.full((batch, 2), 11.0),
        )
        assert jnp.array_equal(
            recovered_dead_fn(0.0, grid, age=age),
            jnp.full((batch, 2), 22.0),
        )


class TestModelReduction:
    def test_reduce_returns_reduced_model_dataclass(self, mixed_assignment_model):
        reduced = mixed_assignment_model.reduce("healthy")

        assert isinstance(reduced, ReducedModel)
        assert reduced.initial_states == ("healthy",)
        assert reduced.reachable_states == (
            "healthy",
            "disabled",
            "recovered",
            "dead",
        )
        assert reduced.n_states == 4
        assert len(reduced.solver_matrix) == 4
        assert all(len(row) == 4 for row in reduced.solver_matrix)

    def test_reduce_reorders_initial_states_to_state_space_order(
        self, mixed_assignment_model
    ):
        reduced = mixed_assignment_model.reduce(("disabled", "healthy"))

        assert reduced.initial_states == ("healthy", "disabled")
        assert reduced.reachable_states == (
            "healthy",
            "disabled",
            "recovered",
            "dead",
        )

    def test_reduce_uses_union_of_reachability_and_excludes_unreachable_states(
        self, mixed_assignment_model
    ):
        reduced = mixed_assignment_model.reduce(("healthy", "disabled"))

        assert reduced.reachable_states == (
            "healthy",
            "disabled",
            "recovered",
            "dead",
        )
        assert "archived" not in reduced.reachable_states

    def test_reduce_from_absorbing_state_returns_single_none_cell(
        self, mixed_assignment_model
    ):
        reduced = mixed_assignment_model.reduce("archived")

        assert reduced.initial_states == ("archived",)
        assert reduced.reachable_states == ("archived",)
        assert reduced.n_states == 1
        assert reduced.solver_matrix == [[None]]

    def test_reduce_rejects_empty_initial_iterable(self, mixed_assignment_model):
        with pytest.raises(ValueError, match="at least one state"):
            mixed_assignment_model.reduce(())

    def test_reduce_rejects_duplicate_initial_states(self, mixed_assignment_model):
        with pytest.raises(ValueError, match="must not contain duplicates"):
            mixed_assignment_model.reduce(("healthy", "healthy"))

    def test_reduce_rejects_unknown_state(self, mixed_assignment_model):
        with pytest.raises(ValueError, match="not a declared state"):
            mixed_assignment_model.reduce(("healthy", "missing"))
