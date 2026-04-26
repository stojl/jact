"""Analytical and API-level tests for the public solver entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "docs" / "original_prototype"),
)

import jax.numpy as jnp
import pytest
import prototype_8

import jact
from jact.callbacks import PointMass, StateCarry

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


def _prototype_collapse_point_no_duration(p, p_point):
    p = p.at[..., 0, :].add(p_point)
    return jnp.sum(p, axis=-1)

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


class TestSolverContinuityAndStability:
    def test_constant_intensity_stays_consistent_with_prototype(
        self, illness_death_model
    ):
        ages = jnp.arange(4, dtype=jnp.float32)

        current = illness_death_model.solve(
            initial="healthy",
            horizon=3,
            steps_per_unit=4,
            callback="collapse_point_no_duration",
            age=ages,
        )
        prototype = prototype_8.semimarkov_solver(
            units=3,
            discretization_unit=4,
            intensity=(
                (None, _constant_intensity(LAMBDA_HD), _constant_intensity(MU_HM)),
                (None, None, _constant_intensity(NU_DM)),
                (None, None, None),
            ),
            intensity_kwargs={"age": ages},
            prob_callback=_prototype_collapse_point_no_duration,
            transpose_result=True,
        )

        prototype_probability = jnp.swapaxes(prototype["probability"], 0, 1)
        max_abs_diff = jnp.max(
            jnp.abs(current["probability"][:-1] - prototype_probability[:-1])
        )

        assert current["states"] == ("healthy", "disabled", "dead")
        assert prototype_probability.shape == current["probability"].shape
        assert float(max_abs_diff) < 2.1e-2

    def test_smooth_intensity_stays_consistent_with_prototype(
        self, mixed_time_duration_model
    ):
        ages = jnp.array([40.0, 55.0], dtype=jnp.float32)

        current = mixed_time_duration_model.solve(
            initial="healthy",
            horizon=2,
            steps_per_unit=6,
            callback="collapse_point_no_duration",
            age=ages,
        )
        prototype = prototype_8.semimarkov_solver(
            units=2,
            discretization_unit=6,
            intensity=(
                (
                    None,
                    _time_duration_intensity(0.03, 0.01, 0.02),
                    _time_duration_intensity(0.02, 0.005, 0.01),
                ),
                (None, None, _time_duration_intensity(0.08, 0.004, 0.015)),
                (None, None, None),
            ),
            intensity_kwargs={"age": ages},
            prob_callback=_prototype_collapse_point_no_duration,
            transpose_result=True,
        )

        prototype_probability = jnp.swapaxes(prototype["probability"], 0, 1)
        max_abs_diff = jnp.max(
            jnp.abs(current["probability"][:-1] - prototype_probability[:-1])
        )

        assert current["states"] == ("healthy", "disabled", "dead")
        assert prototype_probability.shape == current["probability"].shape
        assert float(max_abs_diff) < 1e-3

    def test_covariate_intensity_stays_consistent_with_prototype(
        self, mixed_time_duration_covariate_model
    ):
        ages = jnp.array([45.0, 60.0, 75.0], dtype=jnp.float32)

        current = mixed_time_duration_covariate_model.solve(
            initial="healthy",
            horizon=2,
            steps_per_unit=8,
            record_every=2,
            callback="collapse_point_no_duration",
            age=ages,
        )
        prototype = prototype_8.semimarkov_solver(
            units=2,
            discretization_unit=8,
            intensity=(
                (
                    None,
                    _time_duration_covariate_intensity(
                        0.01, 0.008, 0.01, 0.0004
                    ),
                    _time_duration_covariate_intensity(
                        0.005, 0.004, 0.006, 0.0002
                    ),
                ),
                (
                    None,
                    None,
                    _time_duration_covariate_intensity(
                        0.03, 0.006, 0.012, 0.0003
                    ),
                ),
                (None, None, None),
            ),
            intensity_kwargs={"age": ages},
            prob_callback=_prototype_collapse_point_no_duration,
            transpose_result=True,
        )

        prototype_probability = jnp.swapaxes(prototype["probability"], 0, 1)[::2]
        max_abs_diff = jnp.max(
            jnp.abs(current["probability"][:-1] - prototype_probability[:-1])
        )

        assert current["states"] == ("healthy", "disabled", "dead")
        assert prototype_probability.shape == current["probability"].shape
        assert float(max_abs_diff) < 5e-4

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
            callback="collapse_point_no_duration",
            age=jnp.arange(2, dtype=jnp.float32),
        )["probability"]

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
            callback="collapse_point_no_duration",
            age=jnp.arange(3, dtype=jnp.float32),
        )["probability"]

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
                callback="collapse_point_no_duration",
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
            callback="collapse_point_no_duration",
            age=jnp.arange(2, dtype=jnp.float32),
        )["probability"]

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
        assert isinstance(healthy_point, PointMass)
        assert healthy_point.value.shape == (9, 2)
        assert healthy_point.d_0.shape == (9, 2)
        assert disabled_point is None
        assert dead_point is None
        assert jnp.allclose(healthy_point.value[0], jnp.ones((2,)))
        assert jnp.allclose(healthy_point.d_0[0], jnp.full((2,), 2.0))

    def test_string_shortcut_accepts_batch_initial_duration(
        self, illness_death_model
    ):
        result = illness_death_model.solve(
            initial="healthy",
            initial_duration=jnp.array([0.0, 0.37]),
            horizon=1,
            steps_per_unit=8,
            callback="point_only",
            age=jnp.arange(2, dtype=jnp.float32),
        )

        healthy_point, _, _ = result["probability"]
        assert isinstance(healthy_point, PointMass)
        assert healthy_point.value.shape == (9, 2)
        assert jnp.allclose(healthy_point.d_0[0], jnp.array([0.0, 0.37]))

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
        assert jnp.allclose(initial_point[0].value[0], jnp.array([1.0, 0.0, 0.0]))
        assert jnp.allclose(initial_point[1].value[0], jnp.array([0.0, 1.0, 0.0]))
        assert jnp.allclose(initial_point[2].value[0], jnp.array([0.0, 0.0, 1.0]))
        assert jnp.allclose(initial_point[0].d_0[0], jnp.zeros((3,)))
        assert jnp.allclose(initial_point[1].d_0[0], jnp.zeros((3,)))
        assert jnp.allclose(initial_point[2].d_0[0], jnp.zeros((3,)))

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
            callback="collapse_point_no_duration",
            age=jnp.arange(1, dtype=jnp.float32),
        )

        probability = result["probability"]

        assert result["states"] == ("healthy", "dead")
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
            callback="collapse_point_no_duration",
            age=jnp.arange(2, dtype=jnp.float32),
        )

        probability = result["probability"]

        assert result["states"] == ("healthy", "disabled", "dead")
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

        default_result = illness_death_model.solve(
            callback="default", **kwargs
        )["probability"]
        assert len(default_result) == 3
        assert isinstance(default_result[0], StateCarry)
        assert default_result[0].density.shape == (5, 2, 8)
        assert isinstance(default_result[0].point_mass, PointMass)
        assert default_result[0].point_mass.value.shape == (5, 2)
        assert default_result[0].point_mass.d_0.shape == (5, 2)

        no_duration_result = illness_death_model.solve(
            callback="no_duration", **kwargs
        )["probability"]
        assert isinstance(no_duration_result[0], StateCarry)
        assert no_duration_result[0].density.shape == (5, 2)
        assert isinstance(no_duration_result[0].point_mass, PointMass)
        assert no_duration_result[0].point_mass.value.shape == (5, 2)
        assert no_duration_result[0].point_mass.d_0.shape == (5, 2)

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
        assert isinstance(point_only_result[0], PointMass)
        assert point_only_result[0].value.shape == (5, 2)
        assert point_only_result[0].d_0.shape == (5, 2)
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
