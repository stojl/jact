"""Model definition: a StateSpace bound to intensity callables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from .state_space import StateSpace


@dataclass(frozen=True)
class TransitionInfo:
    """Metadata about how a transition's intensity is provided."""

    source: str
    target: str
    assignment: str  # "transitions", "exits", or "groups"
    callable: Any
    index: Optional[int]  # index into multi-output callable, None for single


@dataclass(frozen=True)
class ReducedModel:
    """A solver-ready representation reduced to reachable states.

    Created internally by :meth:`Model.reduce` when preparing a solve
    from a specific initial state. Users do not create this directly.

    Attributes
    ----------
    initial_states : tuple[str, ...]
        The declared initial-state set.
    reachable_states : tuple[str, ...]
        States reachable from ``initial``, with ``initial`` first.
    solver_matrix : list[list[callable or None]]
        Reduced intensity matrix over reachable states only.
    n_states : int
        Number of reachable states.
    """

    initial_states: tuple[str, ...]
    reachable_states: tuple[str, ...]
    solver_matrix: list[list[Any]]
    n_states: int


class Model:
    """A multi-state model with intensity callables assigned to transitions.

    Created via :meth:`StateSpace.build`. Not instantiated directly.

    Parameters
    ----------
    state_space : StateSpace
        The underlying state space.
    transitions : dict, optional
        ``{(source, target): callable}`` for single-transition assignments.
    exits : dict, optional
        ``{source_state: callable}`` for all-exits-from-a-state assignments.
    groups : dict, optional
        ``{callable: [(src, tgt), ...]}`` for arbitrary transition groups.
    """

    def __init__(
        self,
        state_space: StateSpace,
        transitions: Optional[Dict[Tuple[str, str], Callable]] = None,
        exits: Optional[Dict[str, Callable]] = None,
        groups: Optional[Dict[Callable, List[Tuple[str, str]]]] = None,
    ):
        self._state_space = state_space
        self._transitions_map = transitions or {}
        self._exits_map = exits or {}
        self._groups_map = groups or {}
        self._transition_info: Dict[Tuple[str, str], TransitionInfo] = {}

        self._validate_and_register()
        self._build_full_solver_matrix()

    # ------------------------------------------------------------------ #
    # Validation                                                          #
    # ------------------------------------------------------------------ #

    def _validate_and_register(self) -> None:
        """Check that every transition is covered exactly once."""
        covered: Dict[Tuple[str, str], str] = {}

        # Single transitions
        for (src, tgt), fn in self._transitions_map.items():
            self._register_transition(
                src, tgt, "transitions", fn, index=None, covered=covered
            )

        # Exits
        for src, fn in self._exits_map.items():
            self._state_space._check_state(src)
            exit_transitions = self._state_space.exits(src)
            if not exit_transitions:
                raise ValueError(
                    f"State '{src}' has no outgoing transitions — "
                    f"cannot assign exits."
                )
            for i, (s, t) in enumerate(exit_transitions):
                self._register_transition(
                    s, t, "exits", fn, index=i, covered=covered
                )

        # Groups
        for fn, trans_list in self._groups_map.items():
            for i, (src, tgt) in enumerate(trans_list):
                self._register_transition(
                    src, tgt, "groups", fn, index=i, covered=covered
                )

        # Check all transitions are covered
        uncovered = self._state_space.transitions - set(covered.keys())
        if uncovered:
            uncovered_str = ", ".join(
                f"'{s}' -> '{t}'" for s, t in sorted(uncovered)
            )
            raise ValueError(
                f"The following transitions are not covered by any model: "
                f"{uncovered_str}"
            )

    def _register_transition(
        self,
        src: str,
        tgt: str,
        assignment: str,
        fn: Callable,
        index: Optional[int],
        covered: Dict[Tuple[str, str], str],
    ) -> None:
        """Register a single transition, checking for conflicts."""
        if not self._state_space.has_transition(src, tgt):
            raise ValueError(
                f"Transition '{src}' -> '{tgt}' is not declared in the "
                f"StateSpace."
            )
        if (src, tgt) in covered:
            raise ValueError(
                f"Transition '{src}' -> '{tgt}' is assigned multiple times "
                f"(via '{covered[(src, tgt)]}' and '{assignment}')."
            )
        covered[(src, tgt)] = assignment
        self._transition_info[(src, tgt)] = TransitionInfo(
            source=src,
            target=tgt,
            assignment=assignment,
            callable=fn,
            index=index,
        )

    # ------------------------------------------------------------------ #
    # Full solver matrix (all states)                                     #
    # ------------------------------------------------------------------ #

    def _build_full_solver_matrix(self) -> None:
        """Build the J×J matrix of callables over all states.

        Used as the basis for building reduced matrices per initial state.
        """
        J = self._state_space.n_states

        self._full_solver_matrix: list[list[Any]] = [
            [None for _ in range(J)] for _ in range(J)
        ]

        for (src, tgt), info in self._transition_info.items():
            i = self._state_space.state_index(src)
            j = self._state_space.state_index(tgt)

            if info.index is None:
                self._full_solver_matrix[i][j] = info.callable
            else:
                fn = info.callable
                idx = info.index
                self._full_solver_matrix[i][j] = _make_slice_wrapper(fn, idx)

    # ------------------------------------------------------------------ #
    # Reduction to reachable states                                       #
    # ------------------------------------------------------------------ #

    def reduce(
        self,
        initial: Union[str, Iterable[str]],
    ) -> ReducedModel:
        """Build a reduced solver matrix for a declared initial-state set.

        Computes the union of the reachable subgraphs from the declared
        initial states and extracts only the relevant rows and columns
        from the full solver matrix. Initial states appear first in
        state-space ordering, followed by the remaining reachable states
        in the same ordering.

        Parameters
        ----------
        initial : str or iterable[str]
            The starting state name or declared initial-state set.

        Returns
        -------
        ReducedModel
            A solver-ready object with the reduced intensity matrix
            and reachable state metadata.
        """
        if isinstance(initial, str):
            declared_initial = (initial,)
        else:
            declared_initial = tuple(initial)
            if not declared_initial:
                raise ValueError("initial must contain at least one state.")
        for state in declared_initial:
            self._state_space._check_state(state)
        if len(declared_initial) != len(set(declared_initial)):
            raise ValueError("initial state set must not contain duplicates.")

        reachable_set = set()
        for state in declared_initial:
            reachable_set.update(self._state_space.reachable_from(state))

        initial_ordered = tuple(
            state for state in self._state_space.states if state in declared_initial
        )
        reachable_tail = tuple(
            state
            for state in self._state_space.states
            if state in reachable_set and state not in declared_initial
        )
        reachable = initial_ordered + reachable_tail
        n_reachable = len(reachable)

        # Map reachable state names to their indices in the full matrix
        full_indices = [
            self._state_space.state_index(s) for s in reachable
        ]

        # Extract the submatrix
        reduced_matrix = [
            [
                self._full_solver_matrix[full_indices[i]][full_indices[j]]
                for j in range(n_reachable)
            ]
            for i in range(n_reachable)
        ]

        return ReducedModel(
            initial_states=initial_ordered,
            reachable_states=reachable,
            solver_matrix=reduced_matrix,
            n_states=n_reachable,
        )

    # ------------------------------------------------------------------ #
    # Properties                                                          #
    # ------------------------------------------------------------------ #

    @property
    def state_space(self) -> StateSpace:
        """The underlying StateSpace."""
        return self._state_space

    # ------------------------------------------------------------------ #
    # Queries                                                             #
    # ------------------------------------------------------------------ #

    def info(self, source: str, target: str) -> TransitionInfo:
        """Get metadata about a transition's intensity assignment.

        Parameters
        ----------
        source : str
            Source state name.
        target : str
            Target state name.

        Returns
        -------
        TransitionInfo
        """
        key = (source, target)
        if key not in self._transition_info:
            raise ValueError(
                f"No transition '{source}' -> '{target}' in this model."
            )
        return self._transition_info[key]

    # ------------------------------------------------------------------ #
    # Solver entry point                                                  #
    # ------------------------------------------------------------------ #

    def solve(
        self,
        initial: str,
        horizon: int,
        steps_per_unit: int,
        callback: Union[None, str, Callable] = "collapse_point_no_duration",
        **kwargs,
    ):
        """Compute transition probabilities from a given initial state.

        Parameters
        ----------
        initial : str
            The starting state. Only states reachable from this state
            are included in the computation.
        horizon : int
            Number of time units to solve over.
        steps_per_unit : int
            Discretization resolution per time unit.
        callback : str or callable, optional
            Probability callback. See :mod:`jact.callbacks`.
        **kwargs
            Covariate arrays passed to intensity callables.

        Returns
        -------
        dict
            Result dictionary with ``'probability'`` key.
            State ordering in the result corresponds to
            ``model.reduce(initial).reachable_states``.
        """
        from .solver import solve

        return solve(
            model=self,
            initial=initial,
            horizon=horizon,
            steps_per_unit=steps_per_unit,
            callback=callback,
            **kwargs,
        )

    # ------------------------------------------------------------------ #
    # Display                                                             #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        assignments = []
        for (src, tgt), info in sorted(self._transition_info.items()):
            assignments.append(f"  {src}->{tgt}: {info.assignment}")
        body = "\n".join(assignments)
        return f"Model(\n{body}\n)"


def _make_slice_wrapper(fn: Callable, index: int) -> Callable:
    """Create a callable that evaluates fn and returns output[index].

    Parameters
    ----------
    fn : callable
        Multi-output intensity function returning shape
        ``(n_transitions, batch, D)``.
    index : int
        Which transition to slice.

    Returns
    -------
    callable
        Function with signature ``(t, grid, **kwargs) -> (batch, D)``.
    """

    def wrapper(t, grid, **kwargs):
        full_output = fn(t, grid, **kwargs)
        return full_output[index]

    return wrapper
