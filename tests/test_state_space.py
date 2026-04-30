"""Unit tests for jact.StateSpace, per docs/api_spec.md."""

from __future__ import annotations

import json

import pytest

import jact

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def illness_death():
    return jact.StateSpace(
        states=["healthy", "disabled", "dead"],
        transitions=[
            ("healthy", "disabled"),
            ("healthy", "dead"),
            ("disabled", "dead"),
        ],
    )


@pytest.fixture
def recovery_model():
    return jact.StateSpace(
        states=["healthy", "disabled", "recovered", "dead"],
        transitions=[
            ("healthy", "disabled"),
            ("healthy", "dead"),
            ("disabled", "recovered"),
            ("disabled", "dead"),
            ("recovered", "dead"),
        ],
    )


# ---------------------------------------------------------------------------
# Construction & validation
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_basic_construction(self, illness_death):
        assert illness_death is not None

    def test_construction_accepts_iterables(self):
        ss = jact.StateSpace(
            states=("a", "b"),
            transitions=(("a", "b"),),
        )
        assert ss.n_states == 2

    def test_construction_accepts_generators(self):
        ss = jact.StateSpace(
            states=(state for state in ["a", "b", "c"]),
            transitions=(transition for transition in [("a", "b"), ("b", "c")]),
        )
        assert ss.states == ("a", "b", "c")
        assert ss.transitions == frozenset({("a", "b"), ("b", "c")})

    def test_non_string_state_raises(self):
        with pytest.raises(TypeError, match="State names must be strings"):
            jact.StateSpace(
                states=["a", 1],
                transitions=[("a", "a")],  # type: ignore[list-item]
            )

    def test_duplicate_states_raise(self):
        with pytest.raises((ValueError, Exception)):
            jact.StateSpace(
                states=["a", "a", "b"],
                transitions=[("a", "b")],
            )

    def test_unknown_source_raises(self):
        with pytest.raises((ValueError, KeyError, Exception)):
            jact.StateSpace(
                states=["a", "b"],
                transitions=[("c", "b")],
            )

    def test_unknown_target_raises(self):
        with pytest.raises((ValueError, KeyError, Exception)):
            jact.StateSpace(
                states=["a", "b"],
                transitions=[("a", "c")],
            )

    def test_self_transition_raises(self):
        with pytest.raises((ValueError, Exception)):
            jact.StateSpace(
                states=["a", "b"],
                transitions=[("a", "a"), ("a", "b")],
            )

    def test_duplicate_transition_raises(self):
        with pytest.raises((ValueError, Exception)):
            jact.StateSpace(
                states=["a", "b"],
                transitions=[("a", "b"), ("a", "b")],
            )

    def test_empty_transitions_allowed(self):
        ss = jact.StateSpace(states=["a", "b"], transitions=[])
        assert ss.n_states == 2
        assert ss.transitions == frozenset()


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestProperties:
    def test_states_is_tuple_preserving_order(self, illness_death):
        assert illness_death.states == ("healthy", "disabled", "dead")
        assert isinstance(illness_death.states, tuple)

    def test_n_states(self, illness_death):
        assert illness_death.n_states == 3

    def test_transitions_is_frozenset(self, illness_death):
        assert isinstance(illness_death.transitions, frozenset)
        assert illness_death.transitions == frozenset({
            ("healthy", "disabled"),
            ("healthy", "dead"),
            ("disabled", "dead"),
        })

    def test_absorbing(self, illness_death):
        assert illness_death.absorbing == ("dead",)
        assert isinstance(illness_death.absorbing, tuple)

    def test_transient(self, illness_death):
        assert illness_death.transient == ("healthy", "disabled")
        assert isinstance(illness_death.transient, tuple)

    def test_absorbing_order_matches_states(self, recovery_model):
        # absorbing states appear in the same relative order as `states`
        assert recovery_model.absorbing == ("dead",)
        assert recovery_model.transient == ("healthy", "disabled", "recovered")

    def test_no_absorbing_states(self):
        ss = jact.StateSpace(
            states=["a", "b"],
            transitions=[("a", "b"), ("b", "a")],
        )
        assert ss.absorbing == ()
        assert ss.transient == ("a", "b")

    def test_all_absorbing_states(self):
        ss = jact.StateSpace(states=["a", "b"], transitions=[])
        assert ss.absorbing == ("a", "b")
        assert ss.transient == ()


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------


class TestQueries:
    def test_exits(self, illness_death):
        assert illness_death.exits("healthy") == (
            ("healthy", "disabled"),
            ("healthy", "dead"),
        )

    def test_exits_absorbing_returns_empty(self, illness_death):
        assert illness_death.exits("dead") == ()

    def test_targets(self, illness_death):
        assert illness_death.targets("healthy") == ("disabled", "dead")

    def test_targets_absorbing(self, illness_death):
        assert illness_death.targets("dead") == ()

    def test_sources(self, illness_death):
        assert illness_death.sources("dead") == ("healthy", "disabled")

    def test_sources_of_initial(self, illness_death):
        assert illness_death.sources("healthy") == ()

    def test_has_transition_true(self, illness_death):
        assert illness_death.has_transition("healthy", "dead") is True

    def test_has_transition_false(self, illness_death):
        assert illness_death.has_transition("dead", "healthy") is False

    def test_has_transition_no_self_transition(self, illness_death):
        assert illness_death.has_transition("healthy", "healthy") is False

    def test_state_index(self, illness_death):
        assert illness_death.state_index("healthy") == 0
        assert illness_death.state_index("disabled") == 1
        assert illness_death.state_index("dead") == 2

    def test_query_unknown_state_raises(self, illness_death):
        with pytest.raises((ValueError, KeyError, Exception)):
            illness_death.state_index("nonexistent")


# ---------------------------------------------------------------------------
# Reachability
# ---------------------------------------------------------------------------


class TestReachability:
    def test_reachable_from_initial(self, illness_death):
        # starting state is first, other reachable states in original ordering
        assert illness_death.reachable_from("healthy") == (
            "healthy",
            "disabled",
            "dead",
        )

    def test_reachable_from_intermediate(self, illness_death):
        assert illness_death.reachable_from("disabled") == ("disabled", "dead")

    def test_reachable_from_absorbing(self, illness_death):
        assert illness_death.reachable_from("dead") == ("dead",)

    def test_reachable_starting_state_is_first(self, recovery_model):
        result = recovery_model.reachable_from("disabled")
        assert result[0] == "disabled"

    def test_reachable_preserves_original_ordering(self, recovery_model):
        # from "healthy" all states reachable; order: start first, then others
        # as declared in `states`
        result = recovery_model.reachable_from("healthy")
        assert result[0] == "healthy"
        remainder = result[1:]
        declared_order = ("disabled", "recovered", "dead")
        assert remainder == declared_order

    def test_reachable_excludes_unreachable(self):
        ss = jact.StateSpace(
            states=["a", "b", "c", "d"],
            transitions=[("a", "b"), ("c", "d")],
        )
        result = ss.reachable_from("a")
        assert "c" not in result
        assert "d" not in result
        assert set(result) == {"a", "b"}

    def test_reachable_transitively(self):
        ss = jact.StateSpace(
            states=["a", "b", "c", "d"],
            transitions=[("a", "b"), ("b", "c"), ("c", "d")],
        )
        assert ss.reachable_from("a") == ("a", "b", "c", "d")

    def test_reachable_returns_tuple(self, illness_death):
        assert isinstance(illness_death.reachable_from("healthy"), tuple)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_roundtrip_preserves_states_and_transitions(
        self, illness_death, tmp_path
    ):
        path = tmp_path / "ss.json"
        illness_death.to_json(str(path))
        loaded = jact.StateSpace.from_json(str(path))
        assert loaded.states == illness_death.states
        assert loaded.transitions == illness_death.transitions

    def test_roundtrip_preserves_queries(self, recovery_model, tmp_path):
        path = tmp_path / "ss.json"
        recovery_model.to_json(str(path))
        loaded = jact.StateSpace.from_json(str(path))
        assert loaded.exits("disabled") == recovery_model.exits("disabled")
        assert loaded.absorbing == recovery_model.absorbing
        assert loaded.transient == recovery_model.transient
        assert loaded.reachable_from("healthy") == recovery_model.reachable_from(
            "healthy"
        )

    def test_to_json_creates_file(self, illness_death, tmp_path):
        path = tmp_path / "ss.json"
        illness_death.to_json(str(path))
        assert path.exists()

    def test_to_json_orders_transitions_by_state_space_order(self, tmp_path):
        state_space = jact.StateSpace(
            states=["b", "a", "c"],
            transitions=[
                ("c", "a"),
                ("b", "c"),
                ("b", "a"),
                ("a", "c"),
            ],
        )
        path = tmp_path / "ss.json"

        state_space.to_json(str(path))

        with open(path) as f:
            payload = json.load(f)
        assert payload["transitions"] == [
            ["b", "a"],
            ["b", "c"],
            ["a", "c"],
            ["c", "a"],
        ]
