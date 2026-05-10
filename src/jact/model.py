"""Model definition: a StateSpace bound to intensity callables."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .initial_distribution import InitialDistribution
from .probability import ProbabilityOutput, StateProbability
from .result import ModelResult
from .state_space import StateSpace

__all__ = ["Model", "ReducedModel", "TransitionInfo"]


@dataclass(frozen=True)
class TransitionInfo:
    """Metadata about how a transition's intensity is provided."""

    source: str
    target: str
    assignment: str  # "transitions", "exits", or "groups"
    callable: Any
    index: int | None  # index into multi-output callable, None for single


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
    solver_matrix : tuple[tuple[callable or None, ...], ...]
        Reduced intensity matrix over reachable states only.
    n_states : int
        Number of reachable states.
    """

    initial_states: tuple[str, ...]
    reachable_states: tuple[str, ...]
    solver_matrix: tuple[tuple[Any, ...], ...]
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
        transitions: Mapping[tuple[str, str], Any] | None = None,
        exits: Mapping[str, Any] | None = None,
        groups: Mapping[Any, Sequence[tuple[str, str]]] | None = None,
    ):
        self._state_space = state_space
        self._transitions_map = transitions or {}
        self._exits_map = exits or {}
        self._groups_map = groups or {}
        self._transition_info: dict[tuple[str, str], TransitionInfo] = {}

        self._validate_and_register()
        self._build_full_solver_matrix()

    # ------------------------------------------------------------------ #
    # Validation                                                          #
    # ------------------------------------------------------------------ #

    def _validate_and_register(self) -> None:
        """Check that every transition is covered exactly once."""
        covered: dict[tuple[str, str], str] = {}

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
            if not trans_list:
                raise ValueError(
                    "Group assignments must cover at least one transition."
                )
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
        fn: Any,
        index: int | None,
        covered: dict[tuple[str, str], str],
    ) -> None:
        """Register a single transition, checking for conflicts."""
        if not callable(fn):
            raise TypeError(
                f"{assignment} assignment for '{src}' -> '{tgt}' must be callable."
            )
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
                self._full_solver_matrix[i][j] = _make_slice_wrapper(
                    info.callable, info.index
                )

    # ------------------------------------------------------------------ #
    # Reduction to reachable states                                       #
    # ------------------------------------------------------------------ #

    def reduce(
        self,
        initial: str | Iterable[str],
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
        declared_initial = _normalise_initial_states(initial)
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
        full_indices = [self._state_space.state_index(s) for s in reachable]

        # Extract the submatrix
        reduced_matrix = tuple(
            tuple(
                self._full_solver_matrix[full_indices[i]][full_indices[j]]
                for j in range(n_reachable)
            )
            for i in range(n_reachable)
        )
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
        initial: str | Any | InitialDistribution,
        horizon: int,
        steps_per_unit: int,
        initial_duration: Any = 0.0,
        probability: None | ProbabilityOutput | Callable = StateProbability(),
        cashflows: Any = None,
        cashflow_views: Any = None,
        record_every: int = 1,
        **kwargs,
    ) -> ModelResult:
        """Compute transition probabilities from a documented initial condition.

        Parameters
        ----------
        initial : str, (batch,) int array, or InitialDistribution
            Initial condition. ``str`` is shorthand for one declared
            structural initial state with all mass there and duration
            ``initial_duration``. A ``(batch,)`` integer array is
            shorthand for per-individual initial-state indices into the
            full model state list. An ``InitialDistribution`` separates
            the structural initial-state declaration from the runtime
            mass and duration values within it; model reduction follows
            the declared structural set, not runtime mass support.
        horizon : int
            Number of time units to solve over.
        steps_per_unit : int
            Discretization resolution per time unit.
        initial_duration : float or (batch,) array, optional
            Per-individual ``d_0`` for the ``str`` and ``(batch,)``
            shorthand forms of ``initial``. Default is ``0.0``.
        probability : ProbabilityOutput, callable, or None, optional
            Probability output reducer. Default is
            ``jact.probability.StateProbability()``, which returns a
            ``(T, batch, S)`` tensor of per-state occupancy with state-name
            order given by ``result.states``. Other built-in choices are
            ``jact.probability.Density()``, ``DensityProbability()``,
            ``PointMass()``, ``MarginalComponents()``, and ``Full()``; see
            ``docs/api_spec.md`` for the full output-shape table. Custom
            callables receive ``tuple[StateCarry, ...]`` and may return any
            PyTree, which is stacked along the leading time axis. ``None``
            disables probability output entirely.
        cashflows : CashflowDeclaration, optional
            Cashflow declaration to evaluate.
        cashflow_views : dict, optional
            Solve-time cashflow aggregation views.
        record_every : int, optional
            Record probability output every ``record_every``-th solver
            step. Must divide ``horizon * steps_per_unit``. Default is
            ``1``.
        **kwargs
            Covariate arrays of shape ``(batch, ...)`` passed to
            intensity callables.

        Returns
        -------
        ModelResult
            Dataclass with ``.states`` (always set), ``.probability``
            (``None`` when ``probability=None``), and ``.cashflows``
            (``None`` when ``cashflows=None``). Time is the leading axis
            of every probability leaf and every streamed cashflow leaf.
        """
        from .solver import solve

        return solve(
            model=self,
            initial=initial,
            horizon=horizon,
            steps_per_unit=steps_per_unit,
            initial_duration=initial_duration,
            probability=probability,
            cashflows=cashflows,
            cashflow_views=cashflow_views,
            record_every=record_every,
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
        try:
            size = full_output.shape[0]
        except Exception:
            size = None
        if size is None:
            try:
                size = len(full_output)
            except Exception:
                size = None
        if size is not None and index >= size:
            raise ValueError(
                "Multi-output assignment returned too few transition "
                f"outputs: expected at least {index + 1}, got {size}."
            )
        return full_output[index]

    return wrapper


def _normalise_initial_states(initial: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(initial, str):
        return (initial,)
    states = tuple(initial)
    if not states:
        raise ValueError("initial must contain at least one state.")
    return states
