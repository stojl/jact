"""Canonical fixed-shape event-history batches."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .topology import TopologySpec

__all__ = ["CanonicalBatch"]


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CanonicalBatch:
    """A batch of interval-split competing-risk observations.

    ``event_edge`` uses ``-1`` for a censored interval. ``covariates`` is a
    dense, already encoded ``(rows, features)`` matrix. ``weight`` multiplies
    the row log-likelihood; callers must use one consistent population,
    sampling-correction, or frequency-weight interpretation. Padding is
    represented by ``valid=False`` and therefore contributes exactly zero.
    """

    source_state: Array
    event_edge: Array
    t0: Array
    t1: Array
    d0: Array
    d1: Array
    covariates: Array
    weight: Array
    valid: Array
    subject_id: Array | None = None

    @classmethod
    def from_arrays(
        cls,
        *,
        source_state: ArrayLike,
        event_edge: ArrayLike,
        t0: ArrayLike,
        t1: ArrayLike,
        d0: ArrayLike,
        d1: ArrayLike,
        covariates: ArrayLike,
        weight: ArrayLike | None = None,
        valid: ArrayLike | None = None,
        subject_id: ArrayLike | None = None,
        dtype: Any | None = None,
    ) -> CanonicalBatch:
        """Construct a batch while normalising dtypes and default fields."""
        t0_array = jnp.asarray(t0, dtype=dtype)
        n_rows = _row_count(t0_array)
        covariate_array = jnp.asarray(covariates, dtype=dtype)
        if covariate_array.ndim == 1:
            covariate_array = covariate_array[:, None]
        default_dtype = t0_array.dtype
        return cls(
            source_state=jnp.asarray(source_state, dtype=jnp.int32),
            event_edge=jnp.asarray(event_edge, dtype=jnp.int32),
            t0=t0_array,
            t1=jnp.asarray(t1, dtype=default_dtype),
            d0=jnp.asarray(d0, dtype=default_dtype),
            d1=jnp.asarray(d1, dtype=default_dtype),
            covariates=covariate_array,
            weight=jnp.ones((n_rows,), dtype=default_dtype)
            if weight is None
            else jnp.asarray(weight, dtype=default_dtype),
            valid=jnp.ones((n_rows,), dtype=jnp.bool_)
            if valid is None
            else jnp.asarray(valid, dtype=jnp.bool_),
            subject_id=None
            if subject_id is None
            else jnp.asarray(subject_id, dtype=jnp.int32),
        )

    @property
    def n_rows(self) -> int:
        return self.t0.shape[0]

    @property
    def n_features(self) -> int:
        return self.covariates.shape[1]

    @property
    def exposure(self) -> Array:
        return self.t1 - self.t0

    @property
    def midpoint_t(self) -> Array:
        return (self.t0 + self.t1) * 0.5

    @property
    def midpoint_d(self) -> Array:
        return (self.d0 + self.d1) * 0.5

    def replace(self, **changes: Any) -> CanonicalBatch:
        return replace(self, **changes)

    def validate(
        self,
        topology: TopologySpec,
        *,
        duration_tolerance: float = 1e-6,
    ) -> CanonicalBatch:
        """Validate shapes, intervals, identifiers, weights, and event origins."""
        fields = {
            "source_state": self.source_state,
            "event_edge": self.event_edge,
            "t0": self.t0,
            "t1": self.t1,
            "d0": self.d0,
            "d1": self.d1,
            "weight": self.weight,
            "valid": self.valid,
        }
        for name, value in fields.items():
            if value.ndim != 1 or value.shape[0] != self.n_rows:
                raise ValueError(f"{name} must have shape ({self.n_rows},).")
        if self.covariates.ndim != 2 or self.covariates.shape[0] != self.n_rows:
            raise ValueError("covariates must have shape (rows, features).")
        if self.subject_id is not None and (
            self.subject_id.ndim != 1 or self.subject_id.shape[0] != self.n_rows
        ):
            raise ValueError(f"subject_id must have shape ({self.n_rows},).")

        valid = np.asarray(self.valid, dtype=bool)
        source = np.asarray(self.source_state)[valid]
        event = np.asarray(self.event_edge)[valid]
        exposure = np.asarray(self.exposure)[valid]
        duration_exposure = np.asarray(self.d1 - self.d0)[valid]
        weight = np.asarray(self.weight)[valid]
        numeric = np.column_stack(
            [
                np.asarray(self.t0)[valid],
                np.asarray(self.t1)[valid],
                np.asarray(self.d0)[valid],
                np.asarray(self.d1)[valid],
                np.asarray(self.covariates)[valid],
            ]
        )
        if not np.all(np.isfinite(numeric)):
            raise ValueError("Valid rows must contain finite numeric values.")
        if np.any(exposure < 0.0):
            raise ValueError("t1 must be greater than or equal to t0.")
        if np.any(np.asarray(self.d0)[valid] < 0.0) or np.any(
            np.asarray(self.d1)[valid] < 0.0
        ):
            raise ValueError("Source-state durations must be non-negative.")
        if np.any(np.abs(exposure - duration_exposure) > duration_tolerance):
            raise ValueError("Clock-time and duration exposure must agree.")
        if np.any((source < 0) | (source >= topology.n_states)):
            raise ValueError("source_state contains an invalid state identifier.")
        if np.any((event < -1) | (event >= topology.n_edges)):
            raise ValueError("event_edge contains an invalid edge identifier.")
        if np.any(~np.isfinite(weight)) or np.any(weight < 0.0):
            raise ValueError("Weights on valid rows must be finite and non-negative.")
        event_rows = event >= 0
        if np.any(event_rows):
            edge_sources = np.asarray(topology.edge_sources)
            if np.any(edge_sources[event[event_rows]] != source[event_rows]):
                raise ValueError("An event edge does not exit its row's source state.")
        return self

    def tree_flatten(self) -> tuple[tuple[Any, ...], None]:
        return (
            (
                self.source_state,
                self.event_edge,
                self.t0,
                self.t1,
                self.d0,
                self.d1,
                self.covariates,
                self.weight,
                self.valid,
                self.subject_id,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(
        cls, aux_data: None, children: tuple[Any, ...]
    ) -> CanonicalBatch:
        del aux_data
        return cls(*children)


def _row_count(array: Array) -> int:
    if array.ndim != 1:
        raise ValueError("Interval columns must be one-dimensional.")
    return array.shape[0]
