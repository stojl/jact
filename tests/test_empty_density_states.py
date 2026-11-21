"""Point-mass-only states retain full-grid callable semantics and cashflows."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jact
from jact import cashflows as cf
from jact import probability as pr


@pytest.mark.parametrize("probability_limit", [None, 1.0, 8.0])
def test_no_incoming_state_matches_full_grid_with_cashflows(probability_limit):
    space = jact.StateSpace(
        ("active", "disabled", "dead"),
        (("active", "disabled"), ("active", "dead"), ("disabled", "dead")),
    )

    def hazard(t, d, **kwargs):
        return 0.13 + 0.02 * t + 0.03 * d + 0.09 * d.mean(axis=-1, keepdims=True)

    def payment(t, d, **kwargs):
        return 1 + t + d * d + 0.1 * d.mean(axis=-1, keepdims=True)

    model = space.build(
        transitions={
            ("active", "disabled"): hazard,
            ("active", "dead"): hazard,
            ("disabled", "dead"): hazard,
        }
    )
    cashflows = space.cashflows(
        {
            "rate": cf.StateRate({"active": payment, "disabled": payment}),
            "exit": cf.TransitionLump({("active", "disabled"): payment}),
            "scheduled": cf.ScheduledEvent(
                when=lambda **kwargs: 1.5, payments={"active": payment}
            ),
            "duration": cf.DurationEvent(
                at_durations={"active": 2.0}, payments={"active": payment}
            ),
        }
    )
    options: dict[str, Any] = dict(
        initial_duration=jnp.array([0.25, 1.25]),
        record_every=2,
        intensity_duration_limit=0.6,
        payment_duration_limit=0.75,
        probability_duration_limit=probability_limit,
        cashflows=cashflows,
    )

    actual = model.solve("active", 4, 8, **options)
    full = model.solve("active", 4, 8, probability=pr.Full(), **options)
    assert actual.states == full.states
    active = full.states.index("active")
    density = full.probability["density"]
    np.testing.assert_array_equal(density[:, :, active], 0)
    continuous = density.sum(axis=-1)
    if probability_limit is not None:
        assert "tail" in full.probability
        tail_mass = full.probability["tail"]["mass"]
        np.testing.assert_array_equal(tail_mass[:, :, active], 0)
        continuous = continuous + tail_mass
    expected = continuous.at[:, :, active].add(full.probability["point_mass"]["active"])
    np.testing.assert_allclose(actual.probability, expected, atol=2e-6, rtol=2e-6)

    assert actual.cashflows is not None
    assert full.cashflows is not None
    actual_leaves, actual_tree = jax.tree_util.tree_flatten(actual.cashflows)
    expected_leaves, expected_tree = jax.tree_util.tree_flatten(full.cashflows)
    assert actual_tree == expected_tree
    for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves, strict=True):
        np.testing.assert_allclose(actual_leaf, expected_leaf, atol=2e-6, rtol=2e-6)
        assert bool(jnp.all(expected_leaf.sum(axis=0) > 0))


def test_no_incoming_state_preserves_grid_output_validation():
    space = jact.StateSpace(("active", "dead"), (("active", "dead"),))

    def invalid_grid_shape(t, d, **kwargs):
        return jnp.ones((1, 1)) if d.shape[-1] == 1 else jnp.ones((1, 3))

    model = space.build(transitions={("active", "dead"): invalid_grid_shape})
    with pytest.raises(ValueError, match="Intensity output"):
        model.solve("active", 3, 4)
