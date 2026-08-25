"""Deterministic topology metadata for intensity fitting."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

import jax.numpy as jnp
from jax import Array

from ..state_space import StateSpace

__all__ = ["TopologySpec"]


@dataclass(frozen=True)
class TopologySpec:
    """Static integer representation of a :class:`jact.StateSpace`.

    Edge identifiers always follow ``StateSpace.ordered_transitions()``.
    Instances contain only immutable Python metadata, making them suitable as
    static arguments around jitted likelihood functions.
    """

    states: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    edge_sources: tuple[int, ...]
    edge_targets: tuple[int, ...]
    outgoing_edges: tuple[tuple[int, ...], ...]
    fingerprint: str

    @classmethod
    def from_state_space(cls, state_space: StateSpace) -> TopologySpec:
        """Derive a topology specification using JACT's canonical ordering."""
        if not isinstance(state_space, StateSpace):
            raise TypeError("state_space must be a jact.StateSpace.")
        states = state_space.states
        edges = state_space.ordered_transitions()
        edge_sources = tuple(state_space.state_index(source) for source, _ in edges)
        edge_targets = tuple(state_space.state_index(target) for _, target in edges)
        edge_ids = {edge: index for index, edge in enumerate(edges)}
        outgoing = tuple(
            tuple(edge_ids[edge] for edge in state_space.exits(state))
            for state in states
        )
        fingerprint = cls._fingerprint(states, edges)
        return cls(
            states=states,
            edges=edges,
            edge_sources=edge_sources,
            edge_targets=edge_targets,
            outgoing_edges=outgoing,
            fingerprint=fingerprint,
        )

    @property
    def n_states(self) -> int:
        return len(self.states)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    @property
    def max_exits(self) -> int:
        return max((len(row) for row in self.outgoing_edges), default=0)

    @property
    def edge_source_array(self) -> Array:
        return jnp.asarray(self.edge_sources, dtype=jnp.int32)

    @property
    def edge_target_array(self) -> Array:
        return jnp.asarray(self.edge_targets, dtype=jnp.int32)

    @property
    def outgoing_edge_array(self) -> Array:
        """Padded ``(n_states, max_exits)`` outgoing-edge table."""
        width = max(self.max_exits, 1)
        return jnp.asarray(
            [row + (0,) * (width - len(row)) for row in self.outgoing_edges],
            dtype=jnp.int32,
        )

    @property
    def outgoing_mask(self) -> Array:
        """Validity mask corresponding to :attr:`outgoing_edge_array`."""
        width = max(self.max_exits, 1)
        return jnp.asarray(
            [
                (True,) * len(row) + (False,) * (width - len(row))
                for row in self.outgoing_edges
            ],
            dtype=jnp.bool_,
        )

    def edge_id(self, source: str, target: str) -> int:
        try:
            return self.edges.index((source, target))
        except ValueError as exc:
            raise ValueError(
                f"Transition '{source}' -> '{target}' is not in the topology."
            ) from exc

    def state_id(self, state: str) -> int:
        try:
            return self.states.index(state)
        except ValueError as exc:
            raise ValueError(f"Unknown state '{state}'.") from exc

    def validate_state_space(self, state_space: StateSpace) -> None:
        """Reject a state space with different labels, ordering, or edges."""
        supplied = type(self).from_state_space(state_space)
        if supplied.fingerprint != self.fingerprint:
            raise ValueError(
                "StateSpace does not match the fitted topology: expected "
                f"states={self.states!r}, edges={self.edges!r}; got "
                f"states={supplied.states!r}, edges={supplied.edges!r}."
            )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "states": list(self.states),
            "edges": [list(edge) for edge in self.edges],
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> TopologySpec:
        states = tuple(str(value) for value in metadata["states"])
        edges = tuple((str(edge[0]), str(edge[1])) for edge in metadata["edges"])
        state_index = {state: index for index, state in enumerate(states)}
        if len(state_index) != len(states):
            raise ValueError("Artifact topology contains duplicate states.")
        try:
            edge_sources = tuple(state_index[source] for source, _ in edges)
            edge_targets = tuple(state_index[target] for _, target in edges)
        except KeyError as exc:
            raise ValueError("Artifact edge references an unknown state.") from exc
        if len(set(edges)) != len(edges):
            raise ValueError("Artifact topology contains duplicate edges.")
        outgoing = tuple(
            tuple(
                i
                for i, source_id in enumerate(edge_sources)
                if source_id == state_id
            )
            for state_id in range(len(states))
        )
        actual = cls._fingerprint(states, edges)
        recorded = str(metadata.get("fingerprint", actual))
        if recorded != actual:
            raise ValueError("Artifact topology fingerprint is invalid.")
        return cls(states, edges, edge_sources, edge_targets, outgoing, actual)

    @staticmethod
    def _fingerprint(
        states: tuple[str, ...], edges: tuple[tuple[str, str], ...]
    ) -> str:
        payload = json.dumps(
            {"states": states, "edges": edges},
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
