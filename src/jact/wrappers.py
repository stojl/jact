# pyright: strict, reportMissingImports=false, reportUnknownMemberType=false

"""Convenience wrappers for fitted model intensity functions."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import jax.numpy as jnp
from jax import Array

from .typing import GroupedIntensity, Intensity

__all__ = [
    "bind_intensity",
    "bind_grouped_intensity",
    "bind_exit_intensity",
]

_ModelApply = Callable[..., Any]
_FeatureFn = Callable[..., Any]


def bind_intensity(
    apply_fn: _ModelApply,
    params: Any,
    feature_fn: _FeatureFn,
    *,
    model_state: Mapping[str, Any] | None = None,
    apply_kwargs: Mapping[str, Any] | None = None,
) -> Intensity:
    """Bind a fitted model apply function as a single-transition intensity."""
    _validate_common(apply_fn, feature_fn, model_state, apply_kwargs)
    bound_apply_kwargs = {} if apply_kwargs is None else dict(apply_kwargs)

    def intensity(
        t: jnp.ndarray,
        d: jnp.ndarray,
        /,
        **kwargs: Any,
    ) -> Array:
        features = feature_fn(t, d, **kwargs)
        raw = _apply_model(apply_fn, params, model_state, features, bound_apply_kwargs)
        output = jnp.asarray(raw)
        _check_single_shape(output, d)
        return jnp.maximum(output, 0.0)

    return intensity


def bind_grouped_intensity(
    apply_fn: _ModelApply,
    params: Any,
    feature_fn: _FeatureFn,
    *,
    output_count: int,
    output_axis: int = -1,
    model_state: Mapping[str, Any] | None = None,
    apply_kwargs: Mapping[str, Any] | None = None,
) -> GroupedIntensity:
    """Bind a fitted model apply function as a grouped intensity callable."""
    _validate_grouped(
        apply_fn,
        feature_fn,
        output_count,
        output_axis,
        model_state,
        apply_kwargs,
    )
    bound_apply_kwargs = {} if apply_kwargs is None else dict(apply_kwargs)

    def intensity(
        t: jnp.ndarray,
        d: jnp.ndarray,
        /,
        **kwargs: Any,
    ) -> Array:
        features = feature_fn(t, d, **kwargs)
        raw = _apply_model(apply_fn, params, model_state, features, bound_apply_kwargs)
        output = jnp.asarray(raw)
        _check_grouped_rank(output)
        normalized = jnp.moveaxis(output, output_axis, 0)
        _check_grouped_shape(normalized, d, output_count)
        return jnp.maximum(normalized, 0.0)

    return intensity


def bind_exit_intensity(
    apply_fn: _ModelApply,
    params: Any,
    feature_fn: _FeatureFn,
    *,
    output_count: int,
    output_axis: int = -1,
    model_state: Mapping[str, Any] | None = None,
    apply_kwargs: Mapping[str, Any] | None = None,
) -> GroupedIntensity:
    """Bind a fitted model apply function for an ``exits={...}`` assignment."""
    return bind_grouped_intensity(
        apply_fn,
        params,
        feature_fn,
        output_count=output_count,
        output_axis=output_axis,
        model_state=model_state,
        apply_kwargs=apply_kwargs,
    )


def _validate_common(
    apply_fn: _ModelApply,
    feature_fn: _FeatureFn,
    model_state: Mapping[str, Any] | None,
    apply_kwargs: Mapping[str, Any] | None,
) -> None:
    if not callable(apply_fn):
        raise TypeError("apply_fn must be callable.")
    if not callable(feature_fn):
        raise TypeError("feature_fn must be callable.")
    if model_state is not None and not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
        model_state, Mapping
    ):
        raise TypeError("model_state must be a mapping or None.")
    if apply_kwargs is not None and not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
        apply_kwargs, Mapping
    ):
        raise TypeError("apply_kwargs must be a mapping or None.")


def _validate_grouped(
    apply_fn: _ModelApply,
    feature_fn: _FeatureFn,
    output_count: int,
    output_axis: int,
    model_state: Mapping[str, Any] | None,
    apply_kwargs: Mapping[str, Any] | None,
) -> None:
    _validate_common(apply_fn, feature_fn, model_state, apply_kwargs)
    if (
        not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
            output_count, int
        )
        or isinstance(output_count, bool)
    ):
        raise TypeError("output_count must be a positive integer.")
    if output_count <= 0:
        raise ValueError("output_count must be a positive integer.")
    if (
        not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
            output_axis, int
        )
        or isinstance(output_axis, bool)
    ):
        raise TypeError("output_axis must be an integer.")


def _apply_model(
    apply_fn: _ModelApply,
    params: Any,
    model_state: Mapping[str, Any] | None,
    features: Any,
    apply_kwargs: Mapping[str, Any],
) -> Any:
    if model_state is None:
        return apply_fn(params, features, **apply_kwargs)
    return apply_fn({"params": params, **model_state}, features, **apply_kwargs)


def _check_single_shape(output: Array, d: Array) -> None:
    expected = (1, jnp.shape(d)[-1])
    if not _can_broadcast_to(output.shape, expected):
        raise ValueError(
            "Wrapped intensity output must be broadcastable to (batch, D); "
            f"got {output.shape} for duration width {expected[-1]}."
        )


def _check_grouped_rank(output: Array) -> None:
    if output.ndim == 0:
        raise ValueError(
            "Grouped intensity output must include an output axis; "
            f"got shape {output.shape}."
        )


def _check_grouped_shape(output: Array, d: Array, output_count: int) -> None:
    expected_width = jnp.shape(d)[-1]
    if output.shape[0] != output_count:
        raise ValueError(
            "Grouped intensity output must include output_count="
            f"{output_count} on its normalized leading axis; "
            f"got {output.shape}."
        )
    selected_shape = output.shape[1:]
    if not _can_broadcast_to(selected_shape, (1, expected_width)):
        raise ValueError(
            "Each grouped intensity output must be broadcastable to "
            f"(batch, D) with D={expected_width}; got {output.shape}."
        )


def _can_broadcast_to(
    shape: tuple[int, ...],
    target_shape: tuple[int, ...],
) -> bool:
    try:
        jnp.broadcast_shapes(shape, target_shape)
    except ValueError:
        return False
    return True
