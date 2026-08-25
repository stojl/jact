"""Continuous-time competing-risks likelihood backends."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Literal, NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import Array

from .data import CanonicalBatch
from .topology import TopologySpec

__all__ = [
    "LikelihoodResult",
    "fixed_quadrature_nll",
    "piecewise_exponential_nll",
]

LogHazardFn = Callable[[Mapping[str, Array], Array, Array, Array, Array], Array]
Reduction = Literal["mean", "sum", "none"]


class LikelihoodResult(NamedTuple):
    """Loss plus auditable row-level likelihood components."""

    loss: Array
    row_log_likelihood: Array
    event_log_hazard: Array
    cumulative_hazard: Array
    effective_weight: Array


def piecewise_exponential_nll(
    log_hazards: LogHazardFn,
    params: Mapping[str, Array],
    batch: CanonicalBatch,
    topology: TopologySpec,
    *,
    reduction: Reduction = "mean",
) -> LikelihoodResult:
    """Midpoint piecewise-exponential competing-risks negative log-likelihood."""
    source = jnp.where(batch.valid, batch.source_state, 0)
    midpoint_t = jnp.where(batch.valid, batch.midpoint_t, 0.0)
    midpoint_d = jnp.where(batch.valid, batch.midpoint_d, 0.0)
    covariates = jnp.where(batch.valid[:, None], batch.covariates, 0.0)
    scores = log_hazards(
        params,
        source,
        midpoint_t,
        midpoint_d,
        covariates,
    )
    _check_score_shape(scores, batch, topology, quadrature_nodes=None)
    total_hazard = _sum_source_hazards(scores, source, topology)
    cumulative_hazard = jnp.where(batch.valid, batch.exposure * total_hazard, 0.0)
    event_score = _event_scores(scores, batch.event_edge, topology.n_edges)
    return _finish(event_score, cumulative_hazard, batch, reduction)


def fixed_quadrature_nll(
    log_hazards: LogHazardFn,
    params: Mapping[str, Array],
    batch: CanonicalBatch,
    topology: TopologySpec,
    *,
    order: int = 4,
    reduction: Reduction = "mean",
) -> LikelihoodResult:
    """Gauss-Legendre integrated competing-risks negative log-likelihood.

    Covariates are constant within a canonical row. Event hazards are always
    evaluated separately at the interval's pre-transition endpoint.
    """
    if not isinstance(order, int) or isinstance(order, bool):
        raise TypeError("order must be an integer.")
    if order <= 0:
        raise ValueError("order must be positive.")
    node_values, node_weights = np.polynomial.legendre.leggauss(order)
    nodes = jnp.asarray(node_values, dtype=batch.t0.dtype)
    weights = jnp.asarray(node_weights, dtype=batch.t0.dtype)
    exposure = jnp.where(batch.valid, batch.exposure, 0.0)
    offsets = exposure[:, None] * (nodes[None, :] + 1.0) * 0.5
    safe_t0 = jnp.where(batch.valid, batch.t0, 0.0)
    safe_d0 = jnp.where(batch.valid, batch.d0, 0.0)
    clock = safe_t0[:, None] + offsets
    duration = safe_d0[:, None] + offsets
    row_source = jnp.where(batch.valid, batch.source_state, 0)
    source = jnp.broadcast_to(row_source[:, None], clock.shape)
    safe_covariates = jnp.where(batch.valid[:, None], batch.covariates, 0.0)
    covariates = jnp.broadcast_to(
        safe_covariates[:, None, :],
        (batch.n_rows, order, batch.n_features),
    )
    scores = log_hazards(params, source, clock, duration, covariates)
    _check_score_shape(scores, batch, topology, quadrature_nodes=order)
    total_hazard = _sum_source_hazards(scores, source, topology)
    cumulative_hazard = exposure * 0.5 * jnp.sum(total_hazard * weights, axis=1)

    endpoint_scores = log_hazards(
        params,
        row_source,
        jnp.where(batch.valid, batch.t1, 0.0),
        jnp.where(batch.valid, batch.d1, 0.0),
        safe_covariates,
    )
    _check_score_shape(endpoint_scores, batch, topology, quadrature_nodes=None)
    event_score = _event_scores(endpoint_scores, batch.event_edge, topology.n_edges)
    return _finish(event_score, cumulative_hazard, batch, reduction)


def _sum_source_hazards(
    scores: Array, source_state: Array, topology: TopologySpec
) -> Array:
    valid_edge = source_state[..., None] == topology.edge_source_array
    return jnp.sum(jnp.where(valid_edge, jnp.exp(scores), 0.0), axis=-1)


def _event_scores(scores: Array, event_edge: Array, n_edges: int) -> Array:
    if n_edges == 0:
        return jnp.zeros(event_edge.shape, dtype=scores.dtype)
    safe_edge = jnp.clip(event_edge, 0, n_edges - 1)
    selected = jnp.take_along_axis(scores, safe_edge[:, None], axis=-1)[:, 0]
    return jnp.where(event_edge >= 0, selected, 0.0)


def _finish(
    event_score: Array,
    cumulative_hazard: Array,
    batch: CanonicalBatch,
    reduction: Reduction,
) -> LikelihoodResult:
    if reduction not in ("mean", "sum", "none"):
        raise ValueError("reduction must be 'mean', 'sum', or 'none'.")
    event_score = jnp.where(batch.valid, event_score, 0.0)
    cumulative_hazard = jnp.where(batch.valid, cumulative_hazard, 0.0)
    row_log_likelihood = event_score - cumulative_hazard
    effective_weight = jnp.where(batch.valid, batch.weight, 0.0)
    weighted_nll = -effective_weight * row_log_likelihood
    if reduction == "none":
        loss = weighted_nll
    elif reduction == "sum":
        loss = jnp.sum(weighted_nll)
    else:
        denominator = jnp.sum(effective_weight)
        loss = jnp.where(denominator > 0.0, jnp.sum(weighted_nll) / denominator, 0.0)
    return LikelihoodResult(
        loss=loss,
        row_log_likelihood=row_log_likelihood,
        event_log_hazard=event_score,
        cumulative_hazard=cumulative_hazard,
        effective_weight=effective_weight,
    )


def _check_score_shape(
    scores: Array,
    batch: CanonicalBatch,
    topology: TopologySpec,
    quadrature_nodes: int | None,
) -> None:
    expected = (
        (batch.n_rows, topology.n_edges)
        if quadrature_nodes is None
        else (batch.n_rows, quadrature_nodes, topology.n_edges)
    )
    if scores.shape != expected:
        raise ValueError(
            f"log_hazards returned shape {scores.shape}; expected {expected}."
        )
