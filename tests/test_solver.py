"""Analytical and API-level tests for the public solver entrypoint."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

import jact
from jact.callbacks import StateCarry


LAMBDA_HD = 0.3
MU_HM = 0.2
NU_DM = 0.8


def _constant_intensity(rate: float):
    def fn(t, d, **kwargs):
        batch = kwargs["age"].shape[0]
        return jnp.full((batch, d.shape[-1]), rate)

    return fn


def _illness_death_closed_form_from_healthy(times: jnp.ndarray) -> jnp.ndarray:
    healthy = jnp.exp(-(LAMBDA_HD + MU_HM) * times)
    disabled = (
        LAMBDA_HD
        * (
            jnp.exp(-(LAMBDA_HD + MU_HM) * times)
            - jnp.exp(-NU_DM * times)
        )
        / (NU_DM - LAMBDA_HD - MU_HM)
    )
    dead = 1.0 - healthy - disabled
    return jnp.stack([healthy, disabled, dead], axis=-1)


def _illness_death_closed_form_from_disabled(times: jnp.ndarray) -> jnp.ndarray:
    disabled = jnp.exp(-NU_DM * times)
    dead = 1.0 - disabled
    return jnp.stack([disabled, dead], axis=-1)


@pytest.fixture
def illness_death_model():
    state_space = jact.StateSpace(
        states=["healthy", "disabled", "dead"],
        transitions=[
            ("healthy", "disabled"),
            ("healthy", "dead"),
            ("disabled", "dead"),
        ],
    )
    return state_space.build(
        transitions={
            ("healthy", "disabled"): _constant_intensity(LAMBDA_HD),
            ("healthy", "dead"): _constant_intensity(MU_HM),
            ("disabled", "dead"): _constant_intensity(NU_DM),
        }
    )


class TestSolverAgainstClosedForm:
    def test_solve_matches_closed_form_illness_death_from_healthy(
        self, illness_death_model
    ):
        horizon = 3
        steps_per_unit = 200
        batch_size = 3
        times = jnp.linspace(
            0.0, horizon, horizon * steps_per_unit + 1, endpoint=True
        )
        expected = jnp.broadcast_to(
            _illness_death_closed_form_from_healthy(times)[:, None, :],
            (times.shape[0], batch_size, 3),
        )

        result = illness_death_model.solve(
            initial="healthy",
            horizon=horizon,
            steps_per_unit=steps_per_unit,
            callback="collapse_point_no_duration",
            age=jnp.arange(batch_size, dtype=jnp.float32),
        )

        probability = result["probability"]

        assert result["states"] == ("healthy", "disabled", "dead")
        assert probability.shape == (times.shape[0], batch_size, 3)
        assert jnp.allclose(
            jnp.sum(probability, axis=-1),
            jnp.ones((times.shape[0], batch_size)),
            atol=5e-4,
            rtol=0.0,
        )
        assert jnp.allclose(probability, expected, atol=5e-3, rtol=0.0)

    def test_solve_matches_closed_form_from_disabled_on_reduced_subgraph(
        self, illness_death_model
    ):
        horizon = 3
        steps_per_unit = 200
        batch_size = 2
        times = jnp.linspace(
            0.0, horizon, horizon * steps_per_unit + 1, endpoint=True
        )
        expected = jnp.broadcast_to(
            _illness_death_closed_form_from_disabled(times)[:, None, :],
            (times.shape[0], batch_size, 2),
        )

        result = illness_death_model.solve(
            initial="disabled",
            horizon=horizon,
            steps_per_unit=steps_per_unit,
            callback="collapse_point_no_duration",
            age=jnp.arange(batch_size, dtype=jnp.float32),
        )

        probability = result["probability"]

        assert result["states"] == ("disabled", "dead")
        assert probability.shape == (times.shape[0], batch_size, 2)
        assert jnp.allclose(
            jnp.sum(probability, axis=-1),
            jnp.ones((times.shape[0], batch_size)),
            atol=5e-4,
            rtol=0.0,
        )
        assert jnp.allclose(probability, expected, atol=5e-3, rtol=0.0)


class TestSolverEntry:
    def test_string_shortcut_accepts_scalar_initial_duration(
        self, illness_death_model
    ):
        result = illness_death_model.solve(
            initial="healthy",
            initial_duration=2.0,
            horizon=1,
            steps_per_unit=8,
            callback="point_only",
            age=jnp.arange(2, dtype=jnp.float32),
        )

        healthy_point, disabled_point, dead_point = result["probability"]
        assert result["states"] == ("healthy", "disabled", "dead")
        assert healthy_point.shape == (9, 2, 8)
        assert disabled_point is None
        assert dead_point is None
        assert jnp.allclose(jnp.sum(healthy_point[0], axis=-1), jnp.ones((2,)))

    def test_string_shortcut_accepts_batch_initial_duration(
        self, illness_death_model
    ):
        result = illness_death_model.solve(
            initial="healthy",
            initial_duration=jnp.array([0.0, 2.0]),
            horizon=1,
            steps_per_unit=8,
            callback="point_only",
            age=jnp.arange(2, dtype=jnp.float32),
        )

        healthy_point, _, _ = result["probability"]
        assert healthy_point.shape == (9, 2, 8)
        assert jnp.argmax(healthy_point[0], axis=-1).tolist() == [0, 7]

    def test_integer_shortcut_uses_full_model_state_list(
        self, illness_death_model
    ):
        initial = jnp.array([0, 1, 2], dtype=jnp.int32)

        result = illness_death_model.solve(
            initial=initial,
            horizon=1,
            steps_per_unit=8,
            callback="point_only",
            age=jnp.arange(3, dtype=jnp.float32),
        )

        initial_point = result["probability"]
        assert result["states"] == ("healthy", "disabled", "dead")
        assert all(point is not None for point in initial_point)
        assert jnp.argmax(initial_point[0][0], axis=-1).tolist() == [0, 0, 0]
        assert jnp.argmax(initial_point[1][0], axis=-1).tolist() == [0, 0, 0]
        assert jnp.argmax(initial_point[2][0], axis=-1).tolist() == [0, 0, 0]

    def test_per_individual_distribution_reduces_to_declared_subgraph(
        self, illness_death_model
    ):
        dist = jact.InitialDistribution.per_individual(
            initial_states=("healthy", "disabled"),
            states=jnp.array([0, 1], dtype=jnp.int32),
            duration=jnp.array([0.0, 0.0]),
        )

        result = illness_death_model.solve(
            initial=dist,
            horizon=1,
            steps_per_unit=8,
            callback="point_only",
            age=jnp.arange(2, dtype=jnp.float32),
        )

        assert result["states"] == ("healthy", "disabled", "dead")
        assert len(result["probability"]) == 3
        assert all(point is not None for point in result["probability"][:2])
        assert result["probability"][2] is None

    def test_mixture_initial_distribution_reduces_to_declared_subgraph(
        self, illness_death_model
    ):
        dist = jact.InitialDistribution(
            components={
                "disabled": {
                    "mass": jnp.array([1.0, 1.0]),
                    "duration": jnp.zeros((2,)),
                },
            }
        )

        result = illness_death_model.solve(
            initial=dist,
            horizon=1,
            steps_per_unit=8,
            callback="collapse_point_no_duration",
            age=jnp.arange(2, dtype=jnp.float32),
        )

        assert result["states"] == ("disabled", "dead")
        assert result["probability"].shape == (9, 2, 2)

    def test_initial_duration_is_rejected_for_initial_distribution(
        self, illness_death_model
    ):
        dist = jact.InitialDistribution.at("healthy")

        with pytest.raises(ValueError, match="initial_duration"):
            illness_death_model.solve(
                initial=dist,
                initial_duration=1.0,
                horizon=1,
                steps_per_unit=8,
                age=jnp.arange(2, dtype=jnp.float32),
            )

    def test_invalid_record_every_is_rejected(self, illness_death_model):
        with pytest.raises(ValueError, match="record_every"):
            illness_death_model.solve(
                initial="healthy",
                horizon=1,
                steps_per_unit=8,
                record_every=3,
                age=jnp.arange(2, dtype=jnp.float32),
            )

    def test_invalid_state_name_is_rejected(self, illness_death_model):
        with pytest.raises(ValueError, match="not a declared state"):
            illness_death_model.solve(
                initial=jact.InitialDistribution.at("unknown"),
                horizon=1,
                steps_per_unit=8,
                age=jnp.arange(2, dtype=jnp.float32),
            )

    def test_invalid_indices_are_rejected(self, illness_death_model):
        dist = jact.InitialDistribution.per_individual(
            initial_states=("healthy", "disabled"),
            states=jnp.array([0, 2], dtype=jnp.int32),
        )

        with pytest.raises(ValueError, match="index into the declared"):
            illness_death_model.solve(
                initial=dist,
                horizon=1,
                steps_per_unit=8,
                age=jnp.arange(2, dtype=jnp.float32),
            )

    def test_batch_mismatches_are_rejected(self, illness_death_model):
        dist = jact.InitialDistribution.at(
            "healthy", duration=jnp.zeros((3,))
        )

        with pytest.raises(ValueError, match="batch size"):
            illness_death_model.solve(
                initial=dist,
                horizon=1,
                steps_per_unit=8,
                age=jnp.arange(2, dtype=jnp.float32),
            )


class TestBuiltInCallbacks:
    def test_builtin_callbacks_follow_documented_shapes(
        self, illness_death_model
    ):
        kwargs = dict(
            initial="healthy",
            horizon=1,
            steps_per_unit=8,
            record_every=2,
            age=jnp.arange(2, dtype=jnp.float32),
        )

        default_result = illness_death_model.solve(
            callback="default", **kwargs
        )["probability"]
        assert len(default_result) == 3
        assert isinstance(default_result[0], StateCarry)
        assert default_result[0].density.shape == (5, 2, 8)
        assert default_result[0].point_mass.shape == (5, 2, 8)

        no_duration_result = illness_death_model.solve(
            callback="no_duration", **kwargs
        )["probability"]
        assert isinstance(no_duration_result[0], StateCarry)
        assert no_duration_result[0].density.shape == (5, 2)
        assert no_duration_result[0].point_mass.shape == (5, 2)

        collapse_result = illness_death_model.solve(
            callback="collapse_point", **kwargs
        )["probability"]
        assert len(collapse_result) == 3
        assert collapse_result[0].shape == (5, 2, 8)

        collapse_no_duration_result = illness_death_model.solve(
            callback="collapse_point_no_duration", **kwargs
        )["probability"]
        assert collapse_no_duration_result.shape == (5, 2, 3)

        point_only_result = illness_death_model.solve(
            callback="point_only", **kwargs
        )["probability"]
        assert point_only_result[0].shape == (5, 2, 8)
        assert point_only_result[1] is None

        point_only_no_duration_result = illness_death_model.solve(
            callback="point_only_no_duration", **kwargs
        )["probability"]
        assert point_only_no_duration_result[0].shape == (5, 2)
        assert point_only_no_duration_result[1] is None

        no_point_result = illness_death_model.solve(
            callback="no_point", **kwargs
        )["probability"]
        assert no_point_result[0].shape == (5, 2, 8)

        no_point_no_duration_result = illness_death_model.solve(
            callback="no_point_no_duration", **kwargs
        )["probability"]
        assert no_point_no_duration_result.shape == (5, 2, 3)

        none_result = illness_death_model.solve(
            callback=None, **kwargs
        )["probability"]
        assert none_result is None
