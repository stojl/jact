"""A compact structured log-hazard model implemented with pure JAX."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from .data import CanonicalBatch
from .topology import TopologySpec

__all__ = [
    "StructuredModelConfig",
    "StructuredLogHazardModel",
    "empirical_intercepts",
    "linear_spline_basis",
]


@dataclass(frozen=True)
class StructuredModelConfig:
    """Static architecture and scaling configuration."""

    n_features: int
    time_knots: tuple[float, ...] = ()
    duration_knots: tuple[float, ...] = ()
    embedding_size: int = 4
    hidden_size: int = 16
    rank: int = 4
    time_center: float = 0.0
    time_scale: float = 1.0
    duration_center: float = 0.0
    duration_scale: float = 1.0

    def __post_init__(self) -> None:
        integer_fields = (
            self.n_features,
            self.embedding_size,
            self.hidden_size,
            self.rank,
        )
        if any(value < 0 for value in integer_fields):
            raise ValueError("Model dimensions must be non-negative.")
        if self.hidden_size == 0 and self.rank != 0:
            raise ValueError("rank must be zero when hidden_size is zero.")
        if self.hidden_size > 0 and self.rank == 0:
            raise ValueError("rank must be positive when hidden_size is positive.")
        if self.time_scale <= 0.0 or self.duration_scale <= 0.0:
            raise ValueError("Time and duration scales must be positive.")
        _validate_knots(self.time_knots, "time_knots")
        _validate_knots(self.duration_knots, "duration_knots")

    def to_metadata(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> StructuredModelConfig:
        values = dict(metadata)
        values["time_knots"] = tuple(float(x) for x in values.get("time_knots", ()))
        values["duration_knots"] = tuple(
            float(x) for x in values.get("duration_knots", ())
        )
        return cls(**values)


@dataclass(frozen=True)
class StructuredLogHazardModel:
    """Joint all-edge model with explicit effects and a low-rank shared MLP."""

    topology: TopologySpec
    config: StructuredModelConfig

    def init(
        self,
        key: Array,
        *,
        intercept: ArrayLike | None = None,
        dtype: Any = jnp.float32,
    ) -> dict[str, Array]:
        cfg = self.config
        if intercept is None:
            edge_intercept = jnp.zeros((self.topology.n_edges,), dtype=dtype)
        else:
            edge_intercept = jnp.asarray(intercept, dtype=dtype)
            if edge_intercept.shape != (self.topology.n_edges,):
                raise ValueError(
                    f"intercept must have shape ({self.topology.n_edges},)."
                )
        keys = jax.random.split(key, 5)
        input_size = 2 + cfg.n_features + cfg.embedding_size
        params = {
            "intercept": edge_intercept,
            "linear": jnp.zeros((self.topology.n_edges, cfg.n_features), dtype=dtype),
            "time_spline": jnp.zeros(
                (self.topology.n_edges, len(cfg.time_knots)), dtype=dtype
            ),
            "duration_spline": jnp.zeros(
                (self.topology.n_edges, len(cfg.duration_knots)), dtype=dtype
            ),
            "source_embedding": _small_normal(
                keys[0], (self.topology.n_states, cfg.embedding_size), dtype
            ),
            "hidden_kernel": _glorot(keys[1], (input_size, cfg.hidden_size), dtype),
            "hidden_bias": jnp.zeros((cfg.hidden_size,), dtype=dtype),
            "projection": _small_normal(keys[2], (cfg.hidden_size, cfg.rank), dtype),
            # Near-zero residuals preserve empirical-rate initialization.
            "edge_loading": _small_normal(
                keys[3], (self.topology.n_edges, cfg.rank), dtype, scale=1e-3
            ),
        }
        return params

    def __call__(
        self,
        params: Mapping[str, Array],
        source_state: ArrayLike,
        t: ArrayLike,
        d: ArrayLike,
        covariates: ArrayLike,
    ) -> Array:
        """Return log-hazards with shape ``(..., n_edges)``."""
        cfg = self.config
        x = jnp.asarray(covariates)
        if x.shape[-1] != cfg.n_features:
            raise ValueError(
                f"covariates must end in {cfg.n_features} features; got {x.shape}."
            )
        leading = x.shape[:-1]
        source = jnp.broadcast_to(jnp.asarray(source_state, dtype=jnp.int32), leading)
        clock = jnp.broadcast_to(jnp.asarray(t), leading)
        duration = jnp.broadcast_to(jnp.asarray(d), leading)

        result = jnp.broadcast_to(
            params["intercept"], (*leading, self.topology.n_edges)
        )
        if cfg.n_features:
            result = result + jnp.einsum("...f,ef->...e", x, params["linear"])
        if cfg.time_knots:
            basis = linear_spline_basis(clock, cfg.time_knots)
            result = result + jnp.einsum("...k,ek->...e", basis, params["time_spline"])
        if cfg.duration_knots:
            basis = linear_spline_basis(duration, cfg.duration_knots)
            result = result + jnp.einsum(
                "...k,ek->...e", basis, params["duration_spline"]
            )
        if cfg.hidden_size:
            scaled_t = (clock - cfg.time_center) / cfg.time_scale
            scaled_d = (duration - cfg.duration_center) / cfg.duration_scale
            embedding = params["source_embedding"][source]
            mlp_input = jnp.concatenate(
                [scaled_t[..., None], scaled_d[..., None], x, embedding], axis=-1
            )
            hidden = jax.nn.silu(
                jnp.einsum("...f,fh->...h", mlp_input, params["hidden_kernel"])
                + params["hidden_bias"]
            )
            latent = jnp.einsum("...h,hr->...r", hidden, params["projection"])
            result = result + jnp.einsum(
                "...r,er->...e", latent, params["edge_loading"]
            )
        return result

    def penalty(
        self,
        params: Mapping[str, Array],
        *,
        smoothness: float = 0.0,
        weight_decay: float = 0.0,
        edge_specific: float = 0.0,
    ) -> Array:
        """Curvature and weight regularization for use in a training loss."""
        dtype = params["intercept"].dtype
        value = jnp.asarray(0.0, dtype=dtype)
        if smoothness:
            for name in ("time_spline", "duration_spline"):
                coefficients = params[name]
                if coefficients.shape[-1] >= 3:
                    second_difference = jnp.diff(coefficients, n=2, axis=-1)
                    value = value + smoothness * jnp.sum(second_difference**2)
        if weight_decay:
            for name in (
                "linear",
                "source_embedding",
                "hidden_kernel",
                "projection",
            ):
                value = value + weight_decay * jnp.sum(params[name] ** 2)
        if edge_specific:
            value = value + edge_specific * jnp.sum(params["edge_loading"] ** 2)
        return value


def empirical_intercepts(
    batch: CanonicalBatch,
    topology: TopologySpec,
    *,
    event_prior: float = 0.5,
    exposure_prior: float = 1.0,
) -> Array:
    """Weighted empirical edge-rate log intercepts."""
    if event_prior <= 0.0 or exposure_prior <= 0.0:
        raise ValueError("event_prior and exposure_prior must be positive.")
    if topology.n_edges == 0:
        return jnp.empty((0,), dtype=batch.t0.dtype)
    active_weight = jnp.where(batch.valid, batch.weight, 0.0)
    safe_source = jnp.where(batch.valid, batch.source_state, 0)
    safe_exposure = jnp.where(batch.valid, batch.exposure, 0.0)
    source_exposure = jnp.zeros((topology.n_states,), dtype=batch.t0.dtype).at[
        safe_source
    ].add(active_weight * safe_exposure)
    safe_event = jnp.clip(batch.event_edge, 0, max(topology.n_edges - 1, 0))
    event_weight = active_weight * (batch.event_edge >= 0)
    edge_events = jnp.zeros((topology.n_edges,), dtype=batch.t0.dtype).at[
        safe_event
    ].add(event_weight)
    edge_exposure = source_exposure[topology.edge_source_array]
    return jnp.log((edge_events + event_prior) / (edge_exposure + exposure_prior))


def linear_spline_basis(x: ArrayLike, knots: tuple[float, ...]) -> Array:
    """Piecewise-linear hat basis with stable, flat boundary behavior."""
    values = jnp.asarray(x)
    if not knots:
        return jnp.empty((*values.shape, 0), dtype=values.dtype)
    if len(knots) == 1:
        return jnp.ones((*values.shape, 1), dtype=values.dtype)
    knot_array = jnp.asarray(knots, dtype=values.dtype)
    clipped = jnp.clip(values, knot_array[0], knot_array[-1])
    right = jnp.searchsorted(knot_array, clipped, side="right")
    right = jnp.clip(right, 1, len(knots) - 1)
    left = right - 1
    span = knot_array[right] - knot_array[left]
    right_weight = (clipped - knot_array[left]) / span
    basis = jax.nn.one_hot(left, len(knots), dtype=values.dtype) * (
        1.0 - right_weight[..., None]
    )
    return basis + jax.nn.one_hot(right, len(knots), dtype=values.dtype) * right_weight[
        ..., None
    ]


def _validate_knots(knots: tuple[float, ...], name: str) -> None:
    if any(right <= left for left, right in zip(knots, knots[1:])):
        raise ValueError(f"{name} must be strictly increasing.")


def _small_normal(
    key: Array, shape: tuple[int, ...], dtype: Any, scale: float = 0.01
) -> Array:
    return jax.random.normal(key, shape, dtype=dtype) * scale


def _glorot(key: Array, shape: tuple[int, int], dtype: Any) -> Array:
    if 0 in shape:
        return jnp.empty(shape, dtype=dtype)
    limit = jnp.sqrt(jnp.asarray(6.0 / (shape[0] + shape[1]), dtype=dtype))
    return jax.random.uniform(key, shape, dtype=dtype, minval=-limit, maxval=limit)
