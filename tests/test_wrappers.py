from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import pytest

import jact


def _feature_fn(t, d, **kwargs):
    age = kwargs["age"][:, None]
    return jnp.broadcast_to(age + t + d, (age.shape[0], d.shape[-1]))


def _apply_single(params, features, **kwargs):
    offset = kwargs.get("offset", 0.0)
    return params["scale"] * features + offset


def _apply_grouped_last(params, features):
    values = [params["base"] + i * features for i in range(3)]
    return jnp.stack(values, axis=-1)


def _apply_grouped_first(params, features):
    values = [params["base"] + i * features for i in range(3)]
    return jnp.stack(values, axis=0)


def test_bind_intensity_returns_batch_by_duration_output():
    intensity = jact.wrappers.bind_intensity(
        _apply_single,
        {"scale": 2.0},
        _feature_fn,
        apply_kwargs={"offset": -100.0},
    )

    out = intensity(0.5, jnp.array([[0.0, 1.0]]), age=jnp.array([40.0, 50.0]))

    assert out.shape == (2, 2)
    assert jnp.all(out >= 0.0)
    expected = jnp.maximum(
        2.0 * jnp.array([[40.5, 41.5], [50.5, 51.5]]) - 100.0,
        0.0,
    )
    assert jnp.allclose(out, expected)


def test_bind_grouped_intensity_normalizes_batch_duration_output_axis_last():
    intensity = jact.wrappers.bind_grouped_intensity(
        _apply_grouped_last,
        {"base": -1.0},
        _feature_fn,
        output_count=3,
        output_axis=-1,
    )

    out = intensity(0.0, jnp.array([[1.0, 2.0]]), age=jnp.array([10.0, 20.0]))

    assert out.shape == (3, 2, 2)
    assert jnp.all(out >= 0.0)
    assert jnp.allclose(out[0], jnp.maximum(jnp.full((2, 2), -1.0), 0.0))
    assert jnp.allclose(out[2], jnp.array([[21.0, 23.0], [41.0, 43.0]]))


def test_bind_grouped_intensity_keeps_output_axis_zero():
    intensity = jact.wrappers.bind_grouped_intensity(
        _apply_grouped_first,
        {"base": 0.5},
        _feature_fn,
        output_count=3,
        output_axis=0,
    )

    out = intensity(0.0, jnp.array([[1.0, 2.0]]), age=jnp.array([10.0, 20.0]))

    assert out.shape == (3, 2, 2)
    assert jnp.allclose(out[1], jnp.array([[11.5, 12.5], [21.5, 22.5]]))


def test_bind_exit_intensity_rejects_mismatched_output_count():
    intensity = jact.wrappers.bind_exit_intensity(
        _apply_grouped_last,
        {"base": 0.0},
        _feature_fn,
        output_count=2,
    )

    with pytest.raises(ValueError, match="output_count"):
        intensity(0.0, jnp.array([[1.0]]), age=jnp.array([10.0]))


@pytest.mark.parametrize(
    ("apply_fn", "feature_fn", "match"),
    [
        (object(), _feature_fn, "apply_fn must be callable"),
        (_apply_single, object(), "feature_fn must be callable"),
    ],
)
def test_wrappers_reject_non_callable_functions(apply_fn, feature_fn, match):
    with pytest.raises(TypeError, match=match):
        jact.wrappers.bind_intensity(apply_fn, {"scale": 1.0}, feature_fn)


def test_bind_intensity_rejects_invalid_output_rank():
    def apply_rank_one(params, features):
        del params
        return features[:, 0]

    intensity = jact.wrappers.bind_intensity(apply_rank_one, {}, _feature_fn)

    with pytest.raises(ValueError, match=r"\(batch, D\)"):
        intensity(0.0, jnp.array([[1.0, 2.0]]), age=jnp.array([10.0]))


def test_wrapped_transition_intensity_solves_through_state_space_build():
    state_space = jact.StateSpace(
        states=["alive", "dead"],
        transitions=[("alive", "dead")],
    )
    intensity = jact.wrappers.bind_intensity(
        _apply_single,
        {"scale": 0.0},
        _feature_fn,
        apply_kwargs={"offset": 0.2},
    )
    model = state_space.build(transitions={("alive", "dead"): intensity})

    result = model.solve(
        initial="alive",
        horizon=1,
        steps_per_unit=40,
        age=jnp.array([30.0, 40.0]),
    )

    expected_alive = jnp.exp(-0.2 * jnp.linspace(0.0, 1.0, 41))
    assert result.probability.shape == (41, 2, 2)
    assert jnp.allclose(result.probability[:, 0, 0], expected_alive, atol=2e-3)


def test_gradients_flow_through_params():
    intensity = jact.wrappers.bind_intensity(_apply_single, {"scale": 0.1}, _feature_fn)

    def loss(scale):
        params: Mapping[str, Any] = {"scale": scale}
        bound = jact.wrappers.bind_intensity(_apply_single, params, _feature_fn)
        out = bound(0.0, jnp.array([[1.0, 2.0]]), age=jnp.array([10.0, 20.0]))
        return jnp.sum(out)

    assert jnp.allclose(jax.grad(loss)(0.1), 66.0)
    assert intensity(0.0, jnp.array([[1.0]]), age=jnp.array([10.0])).shape == (1, 1)


@pytest.mark.parametrize(
    "name",
    ["bind_intensity", "bind_grouped_intensity", "bind_exit_intensity"],
)
def test_intensity_wrappers_are_not_top_level_aliases(name):
    assert not hasattr(jact, name)
