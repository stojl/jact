"""Adapters from fitted all-edge models to JACT intensity callables."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np
from jax import Array

from ..model import Model
from ..state_space import StateSpace
from ..typing import GroupedIntensity
from ..wrappers import bind_exit_intensity, bind_grouped_intensity
from .structured import StructuredLogHazardModel

if TYPE_CHECKING:
    from .artifact import FittedArtifact

__all__ = ["JACTAdapter"]


@dataclass(frozen=True)
class JACTAdapter:
    """Validated reconstruction of JACT-native grouped intensities."""

    artifact: FittedArtifact

    def build_model(
        self,
        state_space: StateSpace,
        *,
        assignment: str = "exits",
    ) -> Model:
        """Build a model using per-source exits or one global edge group."""
        self.artifact.topology.validate_state_space(state_space)
        if assignment == "exits":
            exits = {
                source: self.exit_intensity(state_space, source)
                for source in state_space.transient
            }
            return state_space.build(exits=exits)
        if assignment == "group":
            edges = self.artifact.topology.edges
            grouped = self.grouped_intensity(state_space, edges)
            return state_space.build(groups={grouped: edges})
        raise ValueError("assignment must be 'exits' or 'group'.")

    def exit_intensity(
        self,
        state_space: StateSpace,
        source: str,
    ) -> GroupedIntensity:
        """Create an ``exits={source: ...}`` callable in target-state order."""
        self.artifact.topology.validate_state_space(state_space)
        edges = state_space.exits(source)
        if not edges:
            raise ValueError(f"State '{source}' has no outgoing transitions.")
        edge_ids = tuple(self.artifact.topology.edge_id(*edge) for edge in edges)
        source_id = self.artifact.topology.state_id(source)
        feature_fn = self._feature_fn()
        structured = StructuredLogHazardModel(
            self.artifact.topology, self.artifact.model_config
        )

        def apply(
            params: Mapping[str, Array], features: tuple[Array, Array, Array]
        ) -> Array:
            clock, duration, covariates = features
            source_array = jnp.full(clock.shape, source_id, dtype=jnp.int32)
            scores = structured(params, source_array, clock, duration, covariates)
            return self.artifact.link_scores(scores[..., jnp.asarray(edge_ids)])

        return bind_exit_intensity(
            apply,
            self.artifact.params,
            feature_fn,
            output_count=len(edge_ids),
            output_axis=-1,
        )

    def grouped_intensity(
        self,
        state_space: StateSpace,
        edges: Sequence[tuple[str, str]],
    ) -> GroupedIntensity:
        """Create a callable for an arbitrary ordered edge group."""
        self.artifact.topology.validate_state_space(state_space)
        ordered = tuple(edges)
        if not ordered:
            raise ValueError("edges must contain at least one transition.")
        edge_ids = tuple(self.artifact.topology.edge_id(*edge) for edge in ordered)
        source_ids = tuple(
            self.artifact.topology.state_id(source) for source, _ in ordered
        )
        feature_fn = self._feature_fn()
        structured = StructuredLogHazardModel(
            self.artifact.topology, self.artifact.model_config
        )

        def apply(
            params: Mapping[str, Array], features: tuple[Array, Array, Array]
        ) -> Array:
            clock, duration, covariates = features
            scores_by_source: dict[int, Array] = {}
            for source_id in dict.fromkeys(source_ids):
                source_array = jnp.full(clock.shape, source_id, dtype=jnp.int32)
                scores_by_source[source_id] = structured(
                    params, source_array, clock, duration, covariates
                )
            selected = [
                scores_by_source[source_id][..., edge_id]
                for source_id, edge_id in zip(source_ids, edge_ids)
            ]
            return self.artifact.link_scores(jnp.stack(selected, axis=-1))

        return bind_grouped_intensity(
            apply,
            self.artifact.params,
            feature_fn,
            output_count=len(ordered),
            output_axis=-1,
        )

    def validate_hazards(
        self,
        state_space: StateSpace,
        *,
        t: Any,
        d: Any,
        covariates: Mapping[str, Any],
    ) -> None:
        """Evaluate every exported source and reject invalid hazards."""
        self.artifact.topology.validate_state_space(state_space)
        features = self._feature_fn()(t, d, **covariates)
        clock, duration, encoded = features
        structured = StructuredLogHazardModel(
            self.artifact.topology, self.artifact.model_config
        )
        for source in state_space.transient:
            source_id = self.artifact.topology.state_id(source)
            source_array = jnp.full(clock.shape, source_id, dtype=jnp.int32)
            raw = structured(
                self.artifact.params,
                source_array,
                clock,
                duration,
                encoded,
            )
            edge_ids = tuple(
                self.artifact.topology.edge_id(*edge)
                for edge in state_space.exits(source)
            )
            raw_selected = np.asarray(raw[..., jnp.asarray(edge_ids)])
            if not np.all(np.isfinite(raw_selected)):
                raise ValueError(
                    f"Unwrapped scores for source '{source}' are non-finite."
                )
            intensity = self.exit_intensity(state_space, source)
            output = np.asarray(intensity(t, d, **covariates))
            if not np.all(np.isfinite(output)):
                raise ValueError(
                    f"Exported hazards for source '{source}' are non-finite."
                )
            if np.any(output < 0.0):
                raise ValueError(
                    f"Exported hazards for source '{source}' are negative."
                )

    def _feature_fn(self) -> Any:
        encoder = self.artifact.encoder

        def features(t: Array, d: Array, **kwargs: Any) -> tuple[Array, Array, Array]:
            covariates = encoder.grid(t, d, kwargs)
            leading = covariates.shape[:-1]
            clock = jnp.broadcast_to(jnp.asarray(t), leading)
            duration_input = jnp.asarray(d)
            if duration_input.ndim == 1:
                duration_input = duration_input[None, :]
            duration = jnp.broadcast_to(duration_input, leading)
            return clock, duration, covariates

        return features
