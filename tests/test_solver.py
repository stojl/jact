"""Analytical and API-level tests for the public solver entrypoint."""
# ruff: noqa: I001

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import pytest

import jact
from jact.callbacks import PointMass, StateCarry
from jact.solver import _KIND_STATE_RATE, _SOURCE_COMPONENT, _midpoint_solver

# JAX's JitWrapped exposes .clear_cache()/._cache_size() at runtime but they
# aren't in pyright's stubs — alias to Any for the cache-management tests.
_solver_cache: Any = _midpoint_solver

LAMBDA_HD = 0.3
MU_HM = 0.2
NU_DM = 0.8


def _constant_intensity(rate: float):
    def fn(t, d, **kwargs):
        batch = kwargs["age"].shape[0]
        return jnp.full((batch, d.shape[-1]), rate)

    return fn


def _duration_intensity(t, d, **kwargs):
    batch = kwargs["age"].shape[0]
    return jnp.broadcast_to(d, (batch, d.shape[-1]))


def _time_duration_intensity(
    base: float,
    time_coef: float,
    duration_coef: float,
):
    def fn(t, d, **kwargs):
        batch = kwargs["age"].shape[0]
        level = base + time_coef * t + duration_coef * d
        return jnp.broadcast_to(level, (batch, d.shape[-1]))

    return fn


def _time_duration_covariate_intensity(
    base: float,
    time_coef: float,
    duration_coef: float,
    age_coef: float,
):
    def fn(t, d, **kwargs):
        age = kwargs["age"][:, None]
        return base + time_coef * t + duration_coef * d + age_coef * age

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


def _survival_under_duration_hazard(
    times: jnp.ndarray,
    d_0: jnp.ndarray,
) -> jnp.ndarray:
    return jnp.exp(-(d_0[None, :] * times[:, None] + 0.5 * times[:, None] ** 2))


def _symmetric_two_state_cycle_closed_form(
    times: jnp.ndarray,
    rate: float,
) -> jnp.ndarray:
    oscillation = 0.5 * jnp.exp(-2.0 * rate * times)
    return jnp.stack([0.5 + oscillation, 0.5 - oscillation], axis=-1)


def _scalar_output_intensity(t, d, **kwargs):
    del t, d, kwargs
    return jnp.array(0.1)


def _rank_one_output_intensity(t, d, **kwargs):
    del t, d
    return jnp.full((kwargs["age"].shape[0],), 0.1)


def _wrong_width_output_intensity(t, d, **kwargs):
    batch = kwargs["age"].shape[0]
    return jnp.full((batch, d.shape[-1] + 1), 0.1)


def _rate_parameter_intensity(t, d, **kwargs):
    del t
    rate = kwargs["rate"][:, None]
    return jnp.broadcast_to(rate, (rate.shape[0], d.shape[-1]))


def _healthy_probability_callback(state):
    carry = state[0]
    total = jnp.sum(carry.density, axis=-1)
    if carry.point_mass is not None:
        total = total + carry.point_mass.value
    return total


class TestPointMassValidation:
    def test_rejects_incompatible_value_and_duration_shapes(self):
        with pytest.raises(ValueError, match=r"d_0 must have shape \(2, 3\)"):
            PointMass(
                value=jnp.ones((2, 3)),
                d_0=jnp.zeros((3, 2)),
            )

    def test_rejects_incompatible_explicit_log_value_shape(self):
        with pytest.raises(
            ValueError,
            match=r"log_value must have shape \(2, 3\)",
        ):
            PointMass(
                value=jnp.ones((2, 3)),
                d_0=jnp.zeros((2, 3)),
                log_value=jnp.zeros((2, 2)),
            )

    def test_rejects_negative_concrete_value(self):
        with pytest.raises(ValueError, match="value must be non-negative"):
            PointMass(
                value=jnp.array([[1.0, -0.1]]),
                d_0=jnp.zeros((1, 2)),
            )


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


@pytest.fixture
def duration_to_death_model():
    state_space = jact.StateSpace(
        states=["healthy", "disabled", "dead"],
        transitions=[
            ("healthy", "dead"),
            ("disabled", "dead"),
        ],
    )
    return state_space.build(
        transitions={
            ("healthy", "dead"): _duration_intensity,
            ("disabled", "dead"): _duration_intensity,
        }
    )


@pytest.fixture
def mixed_time_duration_model():
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
            ("healthy", "disabled"): _time_duration_intensity(0.03, 0.01, 0.02),
            ("healthy", "dead"): _time_duration_intensity(0.02, 0.005, 0.01),
            ("disabled", "dead"): _time_duration_intensity(0.08, 0.004, 0.015),
        }
    )


@pytest.fixture
def mixed_time_duration_covariate_model():
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
            ("healthy", "disabled"): _time_duration_covariate_intensity(
                0.01, 0.008, 0.01, 0.0004
            ),
            ("healthy", "dead"): _time_duration_covariate_intensity(
                0.005, 0.004, 0.006, 0.0002
            ),
            ("disabled", "dead"): _time_duration_covariate_intensity(
                0.03, 0.006, 0.012, 0.0003
            ),
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
            probability="state_probability",
            age=jnp.arange(batch_size, dtype=jnp.float32),
        )

        probability = result.probability

        assert result.states == ("healthy", "disabled", "dead")
        assert probability.shape == (times.shape[0], batch_size, 3)
        assert jnp.allclose(
            jnp.sum(probability, axis=-1),
            jnp.ones((times.shape[0], batch_size)),
            atol=5e-4,
            rtol=0.0,
        )
        assert jnp.allclose(probability, expected, atol=5e-3, rtol=0.0)


class TestSolverAutodiff:
    @pytest.fixture
    def autodiff_model(self):
        state_space = jact.StateSpace(
            states=["healthy", "dead"],
            transitions=[("healthy", "dead")],
        )
        return state_space.build(
            transitions={("healthy", "dead"): _rate_parameter_intensity}
        )

    def test_state_probability_supports_reverse_mode_with_record_every(
        self, autodiff_model
    ):
        horizon = 2
        rate = jnp.array(0.3, dtype=jnp.float32)

        def loss(rate_scalar):
            probability = autodiff_model.solve(
                initial="healthy",
                horizon=horizon,
                steps_per_unit=8,
                record_every=4,
                probability="state_probability",
                rate=jnp.array([rate_scalar], dtype=jnp.float32),
            ).probability
            return probability[-1, 0, 1]

        grad = jax.grad(loss)(rate)
        expected = horizon * jnp.exp(-rate * horizon)

        assert jnp.allclose(grad, expected, atol=1e-6, rtol=1e-6)

    def test_state_probability_supports_forward_mode(self, autodiff_model):
        horizon = 2
        rate = jnp.array(0.3, dtype=jnp.float32)

        def loss(rate_scalar):
            probability = autodiff_model.solve(
                initial="healthy",
                horizon=horizon,
                steps_per_unit=8,
                probability="state_probability",
                rate=jnp.array([rate_scalar], dtype=jnp.float32),
            ).probability
            return probability[-1, 0, 1]

        primal, tangent = jax.jvp(loss, (rate,), (jnp.array(1.0, dtype=rate.dtype),))
        expected_primal = 1.0 - jnp.exp(-rate * horizon)
        expected_tangent = horizon * jnp.exp(-rate * horizon)

        assert jnp.allclose(primal, expected_primal, atol=1e-6, rtol=1e-6)
        assert jnp.allclose(tangent, expected_tangent, atol=1e-6, rtol=1e-6)

    def test_custom_probability_callback_supports_reverse_mode(
        self, autodiff_model
    ):
        horizon = 2
        rate = jnp.array(0.3, dtype=jnp.float32)

        def loss(rate_scalar):
            probability = autodiff_model.solve(
                initial="healthy",
                horizon=horizon,
                steps_per_unit=8,
                record_every=4,
                probability=_healthy_probability_callback,
                rate=jnp.array([rate_scalar], dtype=jnp.float32),
            ).probability
            return probability[-1, 0]

        grad = jax.grad(loss)(rate)
        expected = -horizon * jnp.exp(-rate * horizon)

        assert jnp.allclose(grad, expected, atol=1e-6, rtol=1e-6)

    def test_solve_conserves_mass_for_two_state_cycle_with_backward_transition(self):
        rate = 0.7
        horizon = 1
        steps_per_unit = 1000
        batch_size = 2
        times = jnp.linspace(
            0.0, horizon, horizon * steps_per_unit + 1, endpoint=True
        )
        expected = jnp.broadcast_to(
            _symmetric_two_state_cycle_closed_form(times, rate)[:, None, :],
            (times.shape[0], batch_size, 2),
        )
        state_space = jact.StateSpace(
            states=["a", "b"],
            transitions=[("a", "b"), ("b", "a")],
        )
        model = state_space.build(
            transitions={
                ("a", "b"): _constant_intensity(rate),
                ("b", "a"): _constant_intensity(rate),
            }
        )

        result = model.solve(
            initial="a",
            horizon=horizon,
            steps_per_unit=steps_per_unit,
            probability="state_probability",
            age=jnp.arange(batch_size, dtype=jnp.float32),
        )
        probability = result.probability
        assert probability is not None

        assert result.states == ("a", "b")
        assert probability.shape == (times.shape[0], batch_size, 2)
        assert jnp.allclose(
            jnp.sum(probability, axis=-1),
            jnp.ones((times.shape[0], batch_size)),
            atol=3e-6,
            rtol=0.0,
        )
        assert jnp.allclose(probability, expected, atol=5e-4, rtol=0.0)


class TestSolverContinuityAndStability:
    def test_density_only_advection_avoids_float32_survival_spike(self):
        rate = 0.25
        horizon = 2
        steps_per_unit = 1024
        batch_size = 1
        solver_steps = horizon * steps_per_unit
        step_size = 1 / steps_per_unit

        def constant_intensity(t, d, **kwargs):
            del t, kwargs
            return jnp.full((batch_size, d.shape[-1]), rate, dtype=jnp.float32)

        def unit_payment(t, d, **kwargs):
            del t, kwargs
            return jnp.ones((batch_size, d.shape[-1]), dtype=jnp.float32)

        alive_density = jnp.zeros(
            (batch_size, solver_steps),
            dtype=jnp.float32,
        ).at[:, 0].set(1.0)
        dead_density = jnp.zeros_like(alive_density)
        state_0 = (
            StateCarry(density=alive_density, point_mass=None),
            StateCarry(density=dead_density, point_mass=None),
        )
        grid = jnp.linspace(
            0.0,
            horizon,
            solver_steps + 1,
            endpoint=True,
            dtype=jnp.float32,
        )[None, :]
        duration_left = grid[:, :-1]
        duration_mid = 0.5 * (duration_left + grid[:, 1:])

        result = _midpoint_solver(
            state_0,
            duration_mid,
            duration_left,
            step_size,
            ((None, constant_intensity), (None, None)),
            {},
            lambda _state: None,
            solver_steps,
            ((_KIND_STATE_RATE, ((0, unit_payment),)),),
            (
                (
                    "annuity",
                    True,
                    None,
                    ((_SOURCE_COMPONENT, 0),),
                    ("annuity",),
                    "single",
                ),
            ),
        )

        annuity = result["cashflow_terminal"][0][0]
        expected = (1.0 - jnp.exp(-rate * horizon)) / rate

        assert jnp.allclose(annuity, expected, atol=5e-6, rtol=0.0)

    def test_constant_intensity_matches_closed_form_at_benchmark_resolution(
        self, illness_death_model
    ):
        horizon = 3
        steps_per_unit = 4
        batch_size = 4
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
            probability="state_probability",
            age=jnp.arange(batch_size, dtype=jnp.float32),
        )
        probability = result.probability

        assert result.states == ("healthy", "disabled", "dead")
        assert probability.shape == expected.shape
        assert jnp.allclose(probability, expected, atol=2.5e-2, rtol=0.0)

    def test_record_every_preserves_closed_form_constant_intensity(
        self, illness_death_model
    ):
        horizon = 3
        steps_per_unit = 8
        record_every = 2
        batch_size = 3
        full_times = jnp.linspace(
            0.0, horizon, horizon * steps_per_unit + 1, endpoint=True
        )
        recorded_times = full_times[::record_every]
        expected = jnp.broadcast_to(
            _illness_death_closed_form_from_healthy(recorded_times)[:, None, :],
            (recorded_times.shape[0], batch_size, 3),
        )

        result = illness_death_model.solve(
            initial="healthy",
            horizon=horizon,
            steps_per_unit=steps_per_unit,
            record_every=record_every,
            probability="state_probability",
            age=jnp.arange(batch_size, dtype=jnp.float32),
        )

        probability = result.probability

        assert result.states == ("healthy", "disabled", "dead")
        assert probability.shape == expected.shape
        assert jnp.allclose(probability, expected, atol=9e-3, rtol=0.0)

    def test_grid_aligned_discontinuity_uses_stable_midpoint_path(self):
        state_space = jact.StateSpace(
            states=["healthy", "dead"],
            transitions=[("healthy", "dead")],
        )

        def aligned_jump(t, d, **kwargs):
            batch = kwargs["age"].shape[0]
            level = jnp.where(t < 0.5, 0.2, 0.8)
            return jnp.broadcast_to(level, (batch, d.shape[-1]))

        model = state_space.build(
            transitions={("healthy", "dead"): aligned_jump}
        )

        result = model.solve(
            initial="healthy",
            horizon=1,
            steps_per_unit=4,
            probability="state_probability",
            age=jnp.arange(2, dtype=jnp.float32),
        ).probability
        assert result is not None

        step_hazards = jnp.array([0.05, 0.05, 0.2, 0.2], dtype=jnp.float32)
        expected_survival = jnp.concatenate(
            [jnp.ones((1,), dtype=jnp.float32), jnp.exp(-jnp.cumsum(step_hazards))]
        )
        expected = jnp.stack(
            [expected_survival, 1.0 - expected_survival],
            axis=-1,
        )

        assert jnp.allclose(result[:, 0, :], expected, atol=1e-6, rtol=0.0)
        assert jnp.all(jnp.isfinite(result))

    def test_zero_hazard_branch_stays_finite_and_identity(self):
        state_space = jact.StateSpace(
            states=["healthy", "dead"],
            transitions=[("healthy", "dead")],
        )
        zero_model = state_space.build(
            transitions={("healthy", "dead"): _constant_intensity(0.0)}
        )

        probability = zero_model.solve(
            initial="healthy",
            horizon=2,
            steps_per_unit=6,
            probability="state_probability",
            age=jnp.arange(3, dtype=jnp.float32),
        ).probability
        assert probability is not None

        assert jnp.all(jnp.isfinite(probability))
        assert jnp.allclose(probability[..., 0], 1.0, atol=1e-7, rtol=0.0)
        assert jnp.allclose(probability[..., 1], 0.0, atol=1e-7, rtol=0.0)

    def test_freeze_initial_keyword_is_rejected(self, illness_death_model):
        with pytest.raises(
            TypeError,
            match="unexpected keyword argument 'freeze_initial'",
        ):
            illness_death_model.solve(
                initial="healthy",
                horizon=2,
                steps_per_unit=6,
                freeze_initial=False,
                probability="state_probability",
                age=jnp.arange(3, dtype=jnp.float32),
            )

    def test_large_hazard_branch_remains_finite_and_nonnegative(self):
        state_space = jact.StateSpace(
            states=["healthy", "dead"],
            transitions=[("healthy", "dead")],
        )
        stiff_model = state_space.build(
            transitions={("healthy", "dead"): _constant_intensity(1.0e6)}
        )

        probability = stiff_model.solve(
            initial="healthy",
            horizon=1,
            steps_per_unit=4,
            probability="state_probability",
            age=jnp.arange(2, dtype=jnp.float32),
        ).probability
        assert probability is not None

        assert jnp.all(jnp.isfinite(probability))
        assert jnp.all(probability >= 0.0)
        assert jnp.allclose(
            jnp.sum(probability, axis=-1),
            1.0,
            atol=1e-6,
            rtol=0.0,
        )
        assert jnp.all(probability[-1, :, 1] > 0.999)

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
            probability="state_probability",
            age=jnp.arange(batch_size, dtype=jnp.float32),
        )

        probability = result.probability

        assert result.states == ("disabled", "dead")
        assert probability.shape == (times.shape[0], batch_size, 2)
        assert jnp.allclose(
            jnp.sum(probability, axis=-1),
            jnp.ones((times.shape[0], batch_size)),
            atol=5e-4,
            rtol=0.0,
        )
        assert jnp.allclose(probability, expected, atol=5e-3, rtol=0.0)


class TestSolverEntry:
    def test_invalid_solver_dimensions_are_rejected(self, illness_death_model):
        for horizon in (0, -1, 1.5):
            with pytest.raises(ValueError, match="horizon must be a positive integer"):
                illness_death_model.solve(
                    initial="healthy",
                    horizon=horizon,
                    steps_per_unit=8,
                    age=jnp.arange(2, dtype=jnp.float32),
                )

        for steps_per_unit in (0, -2, 2.5):
            with pytest.raises(
                ValueError,
                match="steps_per_unit must be a positive integer",
            ):
                illness_death_model.solve(
                    initial="healthy",
                    horizon=1,
                    steps_per_unit=steps_per_unit,
                    age=jnp.arange(2, dtype=jnp.float32),
                )

    def test_string_shortcut_accepts_scalar_initial_duration(
        self, illness_death_model
    ):
        result = illness_death_model.solve(
            initial="healthy",
            initial_duration=2.0,
            horizon=1,
            steps_per_unit=8,
            probability="point_mass",
            age=jnp.arange(2, dtype=jnp.float32),
        )

        point_result = result.probability
        assert result.states == ("healthy", "disabled", "dead")
        assert "healthy" in point_result
        assert "disabled" not in point_result
        assert "dead" not in point_result
        healthy_point = point_result["healthy"]
        assert healthy_point.shape == (9, 2)
        assert jnp.allclose(healthy_point[0], jnp.ones((2,)))

    def test_string_shortcut_accepts_batch_initial_duration(
        self, illness_death_model
    ):
        result = illness_death_model.solve(
            initial="healthy",
            initial_duration=jnp.array([0.0, 0.37]),
            horizon=1,
            steps_per_unit=8,
            probability="point_mass",
            age=jnp.arange(2, dtype=jnp.float32),
        )

        healthy_point = result.probability["healthy"]
        assert healthy_point.shape == (9, 2)

    def test_integer_shortcut_uses_full_model_state_list(
        self, illness_death_model
    ):
        initial = jnp.array([0, 1, 2], dtype=jnp.int32)

        result = illness_death_model.solve(
            initial=initial,
            horizon=1,
            steps_per_unit=8,
            probability="point_mass",
            age=jnp.arange(3, dtype=jnp.float32),
        )

        initial_point = result.probability
        assert result.states == ("healthy", "disabled", "dead")
        assert set(initial_point.keys()) == {"healthy", "disabled", "dead"}
        healthy = initial_point["healthy"]
        disabled = initial_point["disabled"]
        dead = initial_point["dead"]
        assert jnp.allclose(healthy[0], jnp.array([1.0, 0.0, 0.0]))
        assert jnp.allclose(disabled[0], jnp.array([0.0, 1.0, 0.0]))
        assert jnp.allclose(dead[0], jnp.array([0.0, 0.0, 1.0]))

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
            probability="point_mass",
            age=jnp.arange(2, dtype=jnp.float32),
        )

        assert result.states == ("healthy", "disabled", "dead")
        assert set(result.probability.keys()) == {"healthy", "disabled"}

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
            probability="state_probability",
            age=jnp.arange(2, dtype=jnp.float32),
        )

        assert result.states == ("disabled", "dead")
        assert result.probability.shape == (9, 2, 2)

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

    @pytest.mark.parametrize(
        ("intensity_fn", "steps_per_unit"),
        [
            (_scalar_output_intensity, 4),
            (_rank_one_output_intensity, 4),
            (_wrong_width_output_intensity, 4),
        ],
    )
    def test_reference_callable_output_shape_is_validated(
        self,
        intensity_fn,
        steps_per_unit,
    ):
        state_space = jact.StateSpace(
            states=["healthy", "dead"],
            transitions=[("healthy", "dead")],
        )
        model = state_space.build(
            transitions={("healthy", "dead"): intensity_fn}
        )

        with pytest.raises(
            ValueError,
            match=r"Reference intensity output must have shape \(batch, 4\)\.",
        ):
            model.solve(
                initial="healthy",
                horizon=1,
                steps_per_unit=steps_per_unit,
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

    def test_off_grid_initial_duration_matches_closed_form(
        self, duration_to_death_model
    ):
        horizon = 1
        steps_per_unit = 400
        d_0 = jnp.array([0.37], dtype=jnp.float32)
        times = jnp.linspace(
            0.0, horizon, horizon * steps_per_unit + 1, endpoint=True
        )
        survival = _survival_under_duration_hazard(times, d_0)
        expected = jnp.stack([survival, 1.0 - survival], axis=-1)

        result = duration_to_death_model.solve(
            initial="healthy",
            initial_duration=d_0,
            horizon=horizon,
            steps_per_unit=steps_per_unit,
            probability="state_probability",
            age=jnp.arange(1, dtype=jnp.float32),
        )

        probability = result.probability

        assert result.states == ("healthy", "dead")
        assert probability.shape == (times.shape[0], 1, 2)
        assert jnp.allclose(
            probability[:, 0, :],
            expected[:, 0, :],
            atol=1e-3,
            rtol=0.0,
        )

    def test_off_grid_component_mixture_matches_closed_form_and_conserves_mass(
        self, duration_to_death_model
    ):
        horizon = 1
        steps_per_unit = 400
        times = jnp.linspace(
            0.0, horizon, horizon * steps_per_unit + 1, endpoint=True
        )
        healthy_mass = jnp.array([1.0, 0.25], dtype=jnp.float32)
        healthy_d0 = jnp.array([0.0, 0.37], dtype=jnp.float32)
        disabled_mass = jnp.array([0.0, 0.75], dtype=jnp.float32)
        disabled_d0 = jnp.array([0.63, 0.0], dtype=jnp.float32)
        healthy_survival = _survival_under_duration_hazard(times, healthy_d0)
        disabled_survival = _survival_under_duration_hazard(times, disabled_d0)
        expected_healthy = healthy_survival * healthy_mass[None, :]
        expected_disabled = disabled_survival * disabled_mass[None, :]
        expected_dead = 1.0 - expected_healthy - expected_disabled
        expected = jnp.stack(
            [expected_healthy, expected_disabled, expected_dead],
            axis=-1,
        )

        result = duration_to_death_model.solve(
            initial=jact.InitialDistribution(
                components={
                    "healthy": {
                        "mass": healthy_mass,
                        "duration": healthy_d0,
                    },
                    "disabled": {
                        "mass": disabled_mass,
                        "duration": disabled_d0,
                    },
                }
            ),
            horizon=horizon,
            steps_per_unit=steps_per_unit,
            probability="state_probability",
            age=jnp.arange(2, dtype=jnp.float32),
        )

        probability = result.probability

        assert result.states == ("healthy", "disabled", "dead")
        assert probability.shape == (times.shape[0], 2, 3)
        assert jnp.allclose(probability, expected, atol=1e-3, rtol=0.0)
        assert jnp.allclose(
            jnp.sum(probability, axis=-1),
            jnp.ones((times.shape[0], 2)),
            atol=1e-6,
            rtol=0.0,
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

        full_result = illness_death_model.solve(
            probability="full", **kwargs
        ).probability
        assert set(full_result.keys()) == {"density", "point_mass"}
        assert full_result["density"].shape == (5, 2, 3, 8)
        assert "healthy" in full_result["point_mass"]
        assert "disabled" not in full_result["point_mass"]
        assert "dead" not in full_result["point_mass"]
        assert full_result["point_mass"]["healthy"].shape == (5, 2)

        marginal_components_result = illness_death_model.solve(
            probability="marginal_components", **kwargs
        ).probability
        assert set(marginal_components_result.keys()) == {"density", "point_mass"}
        assert marginal_components_result["density"].shape == (5, 2, 3)
        marginal_pm = marginal_components_result["point_mass"]
        assert "healthy" in marginal_pm
        assert marginal_pm["healthy"].shape == (5, 2)

        state_probability_result = illness_death_model.solve(
            probability="state_probability", **kwargs
        ).probability
        assert state_probability_result.shape == (5, 2, 3)

        point_mass_result = illness_death_model.solve(
            probability="point_mass", **kwargs
        ).probability
        assert set(point_mass_result.keys()) == {"healthy"}
        assert point_mass_result["healthy"].shape == (5, 2)

        density_result = illness_death_model.solve(
            probability="density", **kwargs
        ).probability
        assert density_result.shape == (5, 2, 3, 8)

        density_probability_result = illness_death_model.solve(
            probability="density_probability", **kwargs
        ).probability
        assert density_probability_result.shape == (5, 2, 3)

        none_result = illness_death_model.solve(
            probability="none", **kwargs
        ).probability
        assert none_result is None

        disabled_result = illness_death_model.solve(probability=None, **kwargs)
        assert disabled_result.probability is None

    @pytest.mark.parametrize(
        "callback_name",
        [
            "none",
            "full",
            "marginal_components",
            "state_probability",
            "point_mass",
            "density",
            "density_probability",
        ],
    )
    def test_builtin_callbacks_do_not_recompile_on_repeat(
        self,
        illness_death_model,
        callback_name,
    ):
        _solver_cache.clear_cache()
        kwargs = dict(
            initial="healthy",
            horizon=1,
            steps_per_unit=8,
            record_every=2,
            age=jnp.arange(2, dtype=jnp.float32),
        )

        cache_before = _solver_cache._cache_size()
        illness_death_model.solve(probability=callback_name, **kwargs)
        cache_after_first = _solver_cache._cache_size()
        illness_death_model.solve(probability=callback_name, **kwargs)
        cache_after_second = _solver_cache._cache_size()

        assert cache_after_first == cache_before + 1
        assert cache_after_second == cache_after_first

    def test_reusing_same_custom_callback_does_not_recompile(
        self,
        illness_death_model,
    ):
        _solver_cache.clear_cache()
        kwargs = dict(
            initial="healthy",
            horizon=1,
            steps_per_unit=8,
            record_every=2,
            age=jnp.arange(2, dtype=jnp.float32),
        )

        cache_before = _solver_cache._cache_size()
        illness_death_model.solve(
            probability=_healthy_probability_callback, **kwargs
        )
        cache_after_first = _solver_cache._cache_size()
        illness_death_model.solve(
            probability=_healthy_probability_callback, **kwargs
        )
        cache_after_second = _solver_cache._cache_size()

        assert cache_after_first == cache_before + 1
        assert cache_after_second == cache_after_first

    def test_fresh_custom_callback_objects_do_recompile(
        self,
        illness_death_model,
    ):
        _solver_cache.clear_cache()
        kwargs = dict(
            initial="healthy",
            horizon=1,
            steps_per_unit=8,
            record_every=2,
            age=jnp.arange(2, dtype=jnp.float32),
        )

        def make_callback():
            def callback(state):
                return _healthy_probability_callback(state)

            return callback

        cache_before = _solver_cache._cache_size()
        illness_death_model.solve(probability=make_callback(), **kwargs)
        cache_after_first = _solver_cache._cache_size()
        illness_death_model.solve(probability=make_callback(), **kwargs)
        cache_after_second = _solver_cache._cache_size()

        assert cache_after_first == cache_before + 1
        assert cache_after_second == cache_after_first + 1

    def test_removed_builtin_callbacks_raise_unknown_callback(
        self, illness_death_model
    ):
        kwargs = dict(
            initial="healthy",
            horizon=1,
            steps_per_unit=8,
            age=jnp.arange(2, dtype=jnp.float32),
        )

        with pytest.raises(ValueError, match="collapse_point"):
            illness_death_model.solve(probability="collapse_point", **kwargs)

        with pytest.raises(ValueError, match="point_only_no_duration"):
            illness_death_model.solve(
                probability="point_only_no_duration", **kwargs
            )


class TestModelResultIsPyTree:
    """`ModelResult` must remain a registered PyTree so users can
    `jax.jit(model.solve)` and `jax.tree.map` over the result.
    """

    def _kwargs(self):
        return dict(
            initial="healthy",
            horizon=1,
            steps_per_unit=8,
            age=jnp.arange(2, dtype=jnp.float32),
        )

    def test_jit_round_trip_returns_model_result(self, illness_death_model):
        kwargs = self._kwargs()
        out = jax.jit(lambda: illness_death_model.solve(**kwargs))()
        assert isinstance(out, jact.ModelResult)
        assert out.states == ("healthy", "disabled", "dead")
        assert out.probability.shape == (9, 2, 3)
        assert out.cashflows is None

    def test_jit_round_trip_with_cashflows(self, illness_death_model):
        kwargs = self._kwargs()
        cashflows = illness_death_model.state_space.cashflows(
            {"annuity": jact.StateRate({"healthy": _constant_payment(1.0)})}
        )
        out = jax.jit(
            lambda: illness_death_model.solve(
                cashflows=cashflows,
                cashflow_views={"annuity": jact.Raw("annuity")},
                **kwargs,
            )
        )()
        assert isinstance(out, jact.ModelResult)
        assert out.cashflows["annuity"].shape == (8, 2)

    def test_jit_round_trip_with_disabled_probability(
        self, illness_death_model
    ):
        kwargs = self._kwargs()
        out = jax.jit(
            lambda: illness_death_model.solve(probability=None, **kwargs)
        )()
        assert isinstance(out, jact.ModelResult)
        assert out.probability is None

    def test_tree_map_reaches_array_leaves(self, illness_death_model):
        out = illness_death_model.solve(**self._kwargs())
        doubled = jax.tree.map(lambda x: x * 2, out)
        assert isinstance(doubled, jact.ModelResult)
        assert doubled.states == out.states
        assert jnp.allclose(doubled.probability, out.probability * 2)


def _constant_payment(amount: float):
    def fn(t, d, **kwargs):
        return jnp.full_like(d, amount)

    return fn
