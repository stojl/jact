"""State space definition for multi-state models."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from .model import Model


class StateSpace:
    """Defines the topology of a multi-state model.

    A StateSpace specifies which states exist and which transitions
    between them are possible. It carries no intensity models and no data.

    Parameters
    ----------
    states : Sequence[str]
        Ordered list of state names.
    transitions : Sequence[tuple[str, str]]
        List of allowed transitions as (source, target) pairs.

    Examples
    --------
    >>> ss = StateSpace(
    ...     states=["healthy", "disabled", "dead"],
    ...     transitions=[
    ...         ("healthy", "disabled"),
    ...         ("healthy", "dead"),
    ...         ("disabled", "dead"),
    ...     ],
    ... )
    >>> ss.n_states
    3
    >>> ss.absorbing
    ('dead',)
    """

    def __init__(
        self,
        states: Sequence[str],
        transitions: Sequence[tuple[str, str]],
    ):
        self._validate_inputs(states, transitions)

        self._states = tuple(states)
        self._transitions = frozenset(transitions)
        self._state_to_index = {s: i for i, s in enumerate(self._states)}

    # ------------------------------------------------------------------ #
    # Validation                                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _validate_inputs(
        states: Sequence[str],
        transitions: Sequence[tuple[str, str]],
    ) -> None:
        # No duplicate state names
        if len(states) != len(set(states)):
            dupes = [s for s in states if states.count(s) > 1]
            raise ValueError(f"Duplicate state names: {set(dupes)}")

        state_set = set(states)

        for src, tgt in transitions:
            if src not in state_set:
                raise ValueError(
                    f"Transition source '{src}' is not a declared state."
                )
            if tgt not in state_set:
                raise ValueError(
                    f"Transition target '{tgt}' is not a declared state."
                )
            if src == tgt:
                raise ValueError(
                    f"Self-transition '{src}' -> '{src}' is not allowed."
                )

        # No duplicate transitions
        if len(transitions) != len(set(transitions)):
            dupes = [t for t in transitions if transitions.count(t) > 1]
            raise ValueError(f"Duplicate transitions: {set(dupes)}")

    # ------------------------------------------------------------------ #
    # Properties                                                          #
    # ------------------------------------------------------------------ #

    @property
    def states(self) -> tuple[str, ...]:
        """Ordered tuple of state names."""
        return self._states

    @property
    def n_states(self) -> int:
        """Number of states."""
        return len(self._states)

    @property
    def transitions(self) -> frozenset[tuple[str, str]]:
        """Set of allowed transitions."""
        return self._transitions

    @property
    def absorbing(self) -> tuple[str, ...]:
        """States with no outgoing transitions."""
        sources = {src for src, _ in self._transitions}
        return tuple(s for s in self._states if s not in sources)

    @property
    def transient(self) -> tuple[str, ...]:
        """States with at least one outgoing transition."""
        sources = {src for src, _ in self._transitions}
        return tuple(s for s in self._states if s in sources)

    # ------------------------------------------------------------------ #
    # Queries                                                             #
    # ------------------------------------------------------------------ #

    def exits(self, state: str) -> tuple[tuple[str, str], ...]:
        """All transitions out of a given state, ordered by target index.

        Parameters
        ----------
        state : str
            Source state name.

        Returns
        -------
        tuple[tuple[str, str], ...]
            Transitions from *state*, sorted by target state order.
        """
        self._check_state(state)
        result = [(s, t) for s, t in self._transitions if s == state]
        result.sort(key=lambda pair: self._state_to_index[pair[1]])
        return tuple(result)

    def targets(self, state: str) -> tuple[str, ...]:
        """Target states reachable from a given state, ordered by index.

        Parameters
        ----------
        state : str
            Source state name.

        Returns
        -------
        tuple[str, ...]
            Target state names, sorted by state order.
        """
        return tuple(t for _, t in self.exits(state))

    def sources(self, state: str) -> tuple[str, ...]:
        """States that have a transition into the given state.

        Parameters
        ----------
        state : str
            Target state name.

        Returns
        -------
        tuple[str, ...]
            Source state names, sorted by state order.
        """
        self._check_state(state)
        result = [s for s, t in self._transitions if t == state]
        result.sort(key=lambda s: self._state_to_index[s])
        return tuple(result)

    def state_index(self, state: str) -> int:
        """Return the integer index of a state.

        Parameters
        ----------
        state : str
            State name.

        Returns
        -------
        int
            Zero-based index in the state ordering.
        """
        self._check_state(state)
        return self._state_to_index[state]

    def has_transition(self, source: str, target: str) -> bool:
        """Check if a transition is declared.

        Parameters
        ----------
        source : str
            Source state name.
        target : str
            Target state name.

        Returns
        -------
        bool
        """
        return (source, target) in self._transitions

    def reachable_from(self, state: str) -> tuple[str, ...]:
        """States reachable from a given state (including itself).

        Performs a breadth-first traversal of the transition graph
        starting from *state*. The result is ordered: the starting
        state comes first, followed by reachable states in the order
        they appear in the original state list.

        Parameters
        ----------
        state : str
            Starting state name.

        Returns
        -------
        tuple[str, ...]
            Reachable states, starting state first, then by state order.
        """
        self._check_state(state)
        visited = {state}
        queue = [state]
        while queue:
            current = queue.pop(0)
            for tgt in self.targets(current):
                if tgt not in visited:
                    visited.add(tgt)
                    queue.append(tgt)

        # Return in original state order, but with starting state first
        others = [s for s in self._states if s in visited and s != state]
        return (state, *others)

    def _check_state(self, state: str) -> None:
        if state not in self._state_to_index:
            raise ValueError(
                f"'{state}' is not a declared state. "
                f"Available states: {self._states}"
            )

    # ------------------------------------------------------------------ #
    # Model building                                                      #
    # ------------------------------------------------------------------ #

    def build(
        self,
        transitions: Optional[Dict[Tuple[str, str], Callable[..., Any]]] = None,
        exits: Optional[Dict[str, Callable[..., Any]]] = None,
        groups: Optional[Dict[Callable[..., Any], List[Tuple[str, str]]]] = None,
    ) -> Model:
        """Create a Model by assigning intensity callables to transitions.

        Every transition in this StateSpace must be covered exactly once
        across the three arguments. See :class:`Model` for details.

        Parameters
        ----------
        transitions : dict[(str, str), callable], optional
            One callable per transition.
        exits : dict[str, callable], optional
            One callable covering all exits from a state.
        groups : dict[callable, list[tuple[str, str]]], optional
            One callable covering an arbitrary set of transitions.

        Returns
        -------
        Model
        """
        from .model import Model

        return Model(
            state_space=self,
            transitions=transitions,
            exits=exits,
            groups=groups,
        )

    # ------------------------------------------------------------------ #
    # Serialization                                                       #
    # ------------------------------------------------------------------ #

    def to_json(self, path: str) -> None:
        """Save the state space to a JSON file.

        Parameters
        ----------
        path : str
            File path.
        """
        data = {
            "states": list(self._states),
            "transitions": [list(t) for t in sorted(self._transitions)],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> StateSpace:
        """Load a state space from a JSON file.

        Parameters
        ----------
        path : str
            File path.

        Returns
        -------
        StateSpace
        """
        with open(path) as f:
            data = json.load(f)
        return cls(
            states=data["states"],
            transitions=[tuple(t) for t in data["transitions"]],
        )

    # ------------------------------------------------------------------ #
    # Display                                                             #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        trans_str = ", ".join(f"{s}->{t}" for s, t in sorted(self._transitions))
        return (
            f"StateSpace(states={list(self._states)}, "
            f"transitions=[{trans_str}])"
        )
