#!/usr/bin/env python3
"""Benchmark the production end-to-end solver and cashflow valuation.

This benchmark measures warm JIT execution only:
- it excludes first-call compilation time
- it forces synchronization after every timed run
- it measures only production-relevant end-to-end solve timings

By default the script requires a GPU backend. Pass ``--allow-cpu`` only
for local smoke checks when a CUDA-visible device is unavailable.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jact


MAX_BATCH_SIZE = 1_000
DEFAULT_BATCH_SIZE = 1_000
DEFAULT_HORIZON = 30
DEFAULT_STEPS_PER_UNIT = 12
DEFAULT_WARMUP_RUNS = 1
DEFAULT_TIMED_RUNS = 20
DEFAULT_CORRECTNESS_BATCH_SIZE = 128
DEFAULT_CORRECTNESS_ATOL = 1e-5
DEFAULT_STATE_COUNT = 12
TOPOLOGY_CHOICES = ("sparse", "dense", "all")
INTENSITY_PROFILE_CHOICES = ("simple", "involved")
CASHFLOW_SCENARIO_CHOICES = (
    "none",
    "unit-state-terminal",
    "unit-state-terminal-with-probability",
    "unit-state-stream",
    "unit-state-stream-with-probability",
    "unit-transition-terminal",
    "mixed-streams",
    "involved-payments",
    "scheduled-terminal",
    "all",
)


@dataclass(frozen=True)
class BenchmarkConfig:
    batch_size: int
    horizon: int
    steps_per_unit: int
    warmup_runs: int
    timed_runs: int
    correctness_batch_size: int
    correctness_atol: float
    allow_cpu: bool
    topology: str
    state_count: int
    intensity_profile: str
    cashflow_scenarios: str

    @property
    def solver_steps(self) -> int:
        return self.horizon * self.steps_per_unit

    @property
    def step_size(self) -> float:
        return 1.0 / self.steps_per_unit


@dataclass(frozen=True)
class TimingStats:
    name: str
    median_ms: float
    min_ms: float
    p95_ms: float


@dataclass(frozen=True)
class Scenario:
    topology: str

    @property
    def label(self) -> str:
        return f"e2e:{self.topology}"


@dataclass(frozen=True)
class CashflowScenario:
    name: str
    cashflows: jact.CashflowDeclaration
    views: dict[str, Any]
    probability: Any = None
    record_every: int = 1


@dataclass(frozen=True)
class TopologySpec:
    name: str
    transitions: tuple[tuple[int, int], ...]

    @property
    def state_count(self) -> int:
        max_state = max(max(src, tgt) for src, tgt in self.transitions)
        return max_state + 1


def _parse_args() -> BenchmarkConfig:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument(
        "--steps-per-unit",
        type=int,
        default=DEFAULT_STEPS_PER_UNIT,
    )
    parser.add_argument("--warmup-runs", type=int, default=DEFAULT_WARMUP_RUNS)
    parser.add_argument("--timed-runs", type=int, default=DEFAULT_TIMED_RUNS)
    parser.add_argument(
        "--correctness-batch-size",
        type=int,
        default=DEFAULT_CORRECTNESS_BATCH_SIZE,
    )
    parser.add_argument(
        "--correctness-atol",
        type=float,
        default=DEFAULT_CORRECTNESS_ATOL,
    )
    parser.add_argument(
        "--topology",
        choices=TOPOLOGY_CHOICES,
        default="all",
    )
    parser.add_argument("--state-count", type=int, default=DEFAULT_STATE_COUNT)
    parser.add_argument(
        "--intensity-profile",
        choices=INTENSITY_PROFILE_CHOICES,
        default="simple",
    )
    parser.add_argument(
        "--cashflow-scenarios",
        choices=CASHFLOW_SCENARIO_CHOICES,
        default="all",
        help=(
            "Cashflow scenarios to benchmark after the probability baseline. "
            "Use 'none' to run only the probability benchmark."
        ),
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU execution for smoke checks when no GPU is visible.",
    )
    args = parser.parse_args()

    if args.batch_size <= 0 or args.batch_size > MAX_BATCH_SIZE:
        raise SystemExit(
            f"batch_size must be in 1..{MAX_BATCH_SIZE}, got {args.batch_size}."
        )
    if (
        args.correctness_batch_size <= 0
        or args.correctness_batch_size > MAX_BATCH_SIZE
    ):
        raise SystemExit(
            "correctness_batch_size must be in "
            f"1..{MAX_BATCH_SIZE}, got {args.correctness_batch_size}."
        )
    if args.state_count < 2:
        raise SystemExit("state_count must be at least 2.")
    if args.intensity_profile == "involved" and args.state_count != 3:
        raise SystemExit(
            "The involved intensity profile currently requires --state-count 3."
        )

    return BenchmarkConfig(
        batch_size=args.batch_size,
        horizon=args.horizon,
        steps_per_unit=args.steps_per_unit,
        warmup_runs=args.warmup_runs,
        timed_runs=args.timed_runs,
        correctness_batch_size=args.correctness_batch_size,
        correctness_atol=args.correctness_atol,
        allow_cpu=args.allow_cpu,
        topology=args.topology,
        state_count=args.state_count,
        intensity_profile=args.intensity_profile,
        cashflow_scenarios=args.cashflow_scenarios,
    )


def _constant_intensity(rate: float, dtype: jnp.dtype) -> Callable[..., jnp.ndarray]:
    rate_value = jnp.asarray(rate, dtype=dtype)

    def fn(t: jnp.ndarray, d: jnp.ndarray, **kwargs: Any) -> jnp.ndarray:
        del t
        batch = kwargs["age"].shape[0]
        return jnp.full((batch, d.shape[-1]), rate_value, dtype=dtype)

    return fn


def _involved_intensity(
    src: int,
    tgt: int,
    dtype: jnp.dtype,
) -> Callable[..., jnp.ndarray]:
    base = jnp.asarray(0.012 + 0.004 * src + 0.003 * tgt, dtype=dtype)
    src_scale = jnp.asarray(src + 1.0, dtype=dtype)
    tgt_scale = jnp.asarray(tgt + 1.0, dtype=dtype)

    def fn(t: jnp.ndarray, d: jnp.ndarray, **kwargs: Any) -> jnp.ndarray:
        age = jnp.asarray(kwargs["age"], dtype=dtype)[:, None]
        duration = jnp.asarray(d, dtype=dtype)
        time = jnp.asarray(t, dtype=dtype)

        normalized_age = age / jnp.asarray(1000.0, dtype=dtype)
        phase = (
            src_scale * duration
            + 0.35 * tgt_scale * time
            + 0.12 * normalized_age
        )
        oscillation = jnp.sin(phase) ** 2 + 0.5 * jnp.cos(0.7 * phase + tgt_scale)
        trend = jnp.log1p(duration + 0.15 * src_scale) / (1.0 + 0.2 * tgt_scale)
        interaction = jnp.sqrt(duration + 1.0 + normalized_age) / (2.0 + src_scale)
        gate = jax.nn.sigmoid(0.8 * duration - 0.25 * normalized_age + 0.3 * tgt_scale)

        return base + 0.008 * oscillation + 0.01 * trend + 0.006 * interaction * gate

    return fn


def _rate_for_transition(src: int, tgt: int) -> float:
    return 0.01 + 0.002 * (src + 1) + 0.001 * (tgt - src)


def _constant_payment(amount: float, dtype: jnp.dtype) -> Callable[..., jnp.ndarray]:
    amount_value = jnp.asarray(amount, dtype=dtype)

    def fn(t: jnp.ndarray, d: jnp.ndarray, **kwargs: Any) -> jnp.ndarray:
        del t
        batch = kwargs["age"].shape[0]
        return jnp.full((batch, d.shape[-1]), amount_value, dtype=dtype)

    return fn


def _involved_payment(scale: float, dtype: jnp.dtype) -> Callable[..., jnp.ndarray]:
    scale_value = jnp.asarray(scale, dtype=dtype)

    def fn(t: jnp.ndarray, d: jnp.ndarray, **kwargs: Any) -> jnp.ndarray:
        age = jnp.asarray(kwargs["age"], dtype=dtype)[:, None]
        salary = jnp.asarray(kwargs["salary"], dtype=dtype)[:, None]
        duration = jnp.asarray(d, dtype=dtype)
        time = jnp.asarray(t, dtype=dtype)
        age_factor = 1.0 + age / jnp.asarray(120.0, dtype=dtype)
        duration_factor = 1.0 + 0.15 * jnp.log1p(duration)
        seasonality = 1.0 + 0.05 * jnp.sin(0.6 * time + 0.2 * duration)
        salary_factor = salary / jnp.asarray(100_000.0, dtype=dtype)
        return scale_value * age_factor * duration_factor * seasonality * salary_factor

    return fn


def _involved_weight(t: jnp.ndarray, **kwargs: Any) -> jnp.ndarray:
    age = kwargs["age"]
    rate = 0.015 + 0.00005 * age
    return jnp.exp(-rate * t)


def _event_time(**kwargs: Any) -> jnp.ndarray:
    return kwargs["event_time"]


def _illness_death_closed_form_from_healthy(
    times: jnp.ndarray,
    lambda_hd: float,
    mu_hm: float,
    nu_dm: float,
) -> jnp.ndarray:
    healthy = jnp.exp(-(lambda_hd + mu_hm) * times)
    disabled = (
        lambda_hd
        * (healthy - jnp.exp(-nu_dm * times))
        / (nu_dm - lambda_hd - mu_hm)
    )
    dead = 1.0 - healthy - disabled
    return jnp.stack([healthy, disabled, dead], axis=-1)


def _survival_under_duration_hazard(
    times: jnp.ndarray,
    d_0: jnp.ndarray,
) -> jnp.ndarray:
    return jnp.exp(-(d_0[None, :] * times[:, None] + 0.5 * times[:, None] ** 2))


def _build_topology(name: str, state_count: int) -> TopologySpec:
    if name == "sparse":
        transitions = tuple((i, i + 1) for i in range(state_count - 1))
    elif name == "dense":
        transitions = tuple(
            (i, j)
            for i in range(state_count - 1)
            for j in range(i + 1, state_count)
        )
    else:
        raise ValueError(f"Unknown topology: {name}")
    return TopologySpec(name=name, transitions=transitions)


def _build_simple_intensity_matrix(
    topology: TopologySpec,
    dtype: jnp.dtype,
) -> tuple[tuple[Callable[..., jnp.ndarray] | None, ...], ...]:
    matrix: list[list[Callable[..., jnp.ndarray] | None]] = [
        [None for _ in range(topology.state_count)]
        for _ in range(topology.state_count)
    ]
    for src, tgt in topology.transitions:
        matrix[src][tgt] = _constant_intensity(
            _rate_for_transition(src, tgt),
            dtype,
        )
    return tuple(tuple(row) for row in matrix)


def _build_involved_intensity_matrix(
    topology: TopologySpec,
    dtype: jnp.dtype,
) -> tuple[tuple[Callable[..., jnp.ndarray] | None, ...], ...]:
    if topology.state_count != 3:
        raise ValueError("The involved intensity profile expects a 3-state topology.")

    matrix: list[list[Callable[..., jnp.ndarray] | None]] = [
        [None for _ in range(topology.state_count)]
        for _ in range(topology.state_count)
    ]
    for src, tgt in topology.transitions:
        matrix[src][tgt] = _involved_intensity(src, tgt, dtype)
    return tuple(tuple(row) for row in matrix)


def _build_intensity_matrix(
    config: BenchmarkConfig,
    topology: TopologySpec,
    dtype: jnp.dtype,
) -> tuple[tuple[Callable[..., jnp.ndarray] | None, ...], ...]:
    if config.intensity_profile == "simple":
        return _build_simple_intensity_matrix(topology, dtype)
    if config.intensity_profile == "involved":
        return _build_involved_intensity_matrix(topology, dtype)
    raise ValueError(f"Unknown intensity profile: {config.intensity_profile}")


def _state_names(state_count: int) -> list[str]:
    return [f"s{i}" for i in range(state_count)]


def _build_model(
    topology: TopologySpec,
    intensity: tuple[tuple[Callable[..., jnp.ndarray] | None, ...], ...],
):
    states = _state_names(topology.state_count)
    transitions = [(states[src], states[tgt]) for src, tgt in topology.transitions]
    transition_map = {
        (states[src], states[tgt]): intensity[src][tgt]
        for src, tgt in topology.transitions
    }
    state_space = jact.StateSpace(states=states, transitions=transitions)
    return state_space.build(transitions=transition_map)


def _build_cashflow_scenarios(
    config: BenchmarkConfig,
    topology: TopologySpec,
    model,
    dtype: jnp.dtype,
) -> list[CashflowScenario]:
    if config.cashflow_scenarios == "none":
        return []

    states = _state_names(topology.state_count)
    transient_states = tuple(sorted({states[src] for src, _ in topology.transitions}))
    transitions = tuple(
        (states[src], states[tgt]) for src, tgt in topology.transitions
    )
    step_count = config.solver_steps
    scenarios: list[CashflowScenario] = []

    def include(name: str) -> bool:
        return config.cashflow_scenarios in ("all", name)

    def unit_state_cashflows():
        return model.state_space.cashflows({
            "annuity": jact.StateRate({
                state: _constant_payment(1.0, dtype) for state in transient_states
            })
        })

    if include("unit-state-terminal"):
        scenarios.append(
            CashflowScenario(
                name="unit-state-terminal",
                cashflows=unit_state_cashflows(),
                views={"pv": jact.Total(terminal=True)},
                record_every=step_count,
            )
        )

    if include("unit-state-terminal-with-probability"):
        scenarios.append(
            CashflowScenario(
                name="unit-state-terminal-with-probability",
                cashflows=unit_state_cashflows(),
                views={"pv": jact.Total(terminal=True)},
                probability="collapse_point_no_duration",
                record_every=step_count,
            )
        )

    if include("unit-state-stream"):
        scenarios.append(
            CashflowScenario(
                name="unit-state-stream",
                cashflows=unit_state_cashflows(),
                views={"pv": jact.Total()},
            )
        )

    if include("unit-state-stream-with-probability"):
        scenarios.append(
            CashflowScenario(
                name="unit-state-stream-with-probability",
                cashflows=unit_state_cashflows(),
                views={"pv": jact.Total()},
                probability="collapse_point_no_duration",
            )
        )

    if include("unit-transition-terminal"):
        cashflows = model.state_space.cashflows({
            "benefit": jact.TransitionLump({
                transition: _constant_payment(1.0, dtype)
                for transition in transitions
            })
        })
        scenarios.append(
            CashflowScenario(
                name="unit-transition-terminal",
                cashflows=cashflows,
                views={"pv": jact.Total(terminal=True)},
                record_every=step_count,
            )
        )

    if include("mixed-streams"):
        event_states = transient_states[: max(1, min(2, len(transient_states)))]
        cashflows = model.state_space.cashflows({
            "premium": jact.StateRate({
                state: _constant_payment(-1.0, dtype) for state in transient_states
            }),
            "benefit": jact.TransitionLump({
                transition: _constant_payment(10.0, dtype)
                for transition in transitions
            }),
            "bonus": jact.ScheduledEvent(
                when=_event_time,
                payments={
                    state: _constant_payment(2.0, dtype) for state in event_states
                },
            ),
        })
        scenarios.append(
            CashflowScenario(
                name="mixed-streams",
                cashflows=cashflows,
                views={
                    "raw": jact.Raw(),
                    "total": jact.Total(),
                    "terminal": jact.Total(terminal=True),
                },
            )
        )

    if include("involved-payments"):
        state_payments = {
            state: _involved_payment(0.25 + 0.05 * index, dtype)
            for index, state in enumerate(transient_states)
        }
        transition_payments = {
            transition: _involved_payment(5.0 + index, dtype)
            for index, transition in enumerate(transitions)
        }
        cashflows = model.state_space.cashflows({
            "salary_rate": jact.StateRate(state_payments),
            "claim": jact.TransitionLump(transition_payments),
        })
        scenarios.append(
            CashflowScenario(
                name="involved-payments",
                cashflows=cashflows,
                views={
                    "net": jact.Total(weight=_involved_weight),
                    "state": jact.ByState(weight=_involved_weight, terminal=True),
                    "kind": jact.ByKind(weight=_involved_weight, terminal=True),
                },
            )
        )

    if include("scheduled-terminal"):
        event_states = transient_states[: max(1, min(3, len(transient_states)))]
        cashflows = model.state_space.cashflows({
            "scheduled": jact.ScheduledEvent(
                when=_event_time,
                payments={
                    state: _involved_payment(1.5 + index, dtype)
                    for index, state in enumerate(event_states)
                },
            )
        })
        scenarios.append(
            CashflowScenario(
                name="scheduled-terminal",
                cashflows=cashflows,
                views={
                    "pv": jact.Total(weight=_involved_weight, terminal=True),
                    "state": jact.ByState(weight=_involved_weight, terminal=True),
                },
                record_every=step_count,
            )
        )

    return scenarios


def _block_until_ready(tree: Any) -> Any:
    jax.tree_util.tree_map(
        lambda leaf: leaf.block_until_ready()
        if hasattr(leaf, "block_until_ready")
        else leaf,
        tree,
    )
    return tree


def _maybe_require_gpu(config: BenchmarkConfig) -> jax.Device:
    backend = jax.default_backend()
    devices = jax.devices()
    if backend != "gpu" and not config.allow_cpu:
        raise SystemExit(
            "GPU backend required for this benchmark.\n"
            f"Detected backend: {backend}\n"
            f"Visible devices: {devices}\n"
            "Fix CUDA/JAX device visibility or rerun with --allow-cpu "
            "for a non-representative smoke check."
        )
    return devices[0]


def _selected_scenarios(config: BenchmarkConfig) -> list[Scenario]:
    topologies = (
        ("sparse", "dense") if config.topology == "all" else (config.topology,)
    )
    return [Scenario(topology=topology) for topology in topologies]


def _benchmark_header(config: BenchmarkConfig, scenarios: list[Scenario]) -> None:
    device = jax.devices()[0]
    print("solver benchmark")
    print(f"python: {sys.version.split()[0]}")
    print(f"jax: {jax.__version__}")
    print(f"jaxlib: {jax.lib.__version__}")
    print(f"backend: {jax.default_backend()}")
    print(f"device: {device}")
    print(f"x64 enabled: {jax.config.jax_enable_x64}")
    print(f"batch_size: {config.batch_size}")
    print(f"correctness_batch_size: {config.correctness_batch_size}")
    print(f"horizon: {config.horizon}")
    print(f"steps_per_unit: {config.steps_per_unit}")
    print(f"solver_steps: {config.solver_steps}")
    print(f"state_count: {config.state_count}")
    print(f"intensity_profile: {config.intensity_profile}")
    print(f"cashflow_scenarios: {config.cashflow_scenarios}")
    print(f"warmup_runs: {config.warmup_runs}")
    print(f"timed_runs: {config.timed_runs}")
    print(f"scenarios: {', '.join(scenario.label for scenario in scenarios)}")
    print()


def _run_analytic_correctness_checks(
    config: BenchmarkConfig,
    dtype: jnp.dtype,
) -> None:
    batch_size = min(config.correctness_batch_size, 8)

    lambda_hd = 0.3
    mu_hm = 0.2
    nu_dm = 0.8
    horizon = 3
    steps_per_unit = max(8, config.steps_per_unit)
    times = jnp.linspace(0.0, horizon, horizon * steps_per_unit + 1, endpoint=True)
    expected = jnp.broadcast_to(
        _illness_death_closed_form_from_healthy(
            times,
            lambda_hd,
            mu_hm,
            nu_dm,
        )[:, None, :],
        (times.shape[0], batch_size, 3),
    )
    illness_death = jact.StateSpace(
        states=["healthy", "disabled", "dead"],
        transitions=[
            ("healthy", "disabled"),
            ("healthy", "dead"),
            ("disabled", "dead"),
        ],
    ).build(
        transitions={
            ("healthy", "disabled"): _constant_intensity(lambda_hd, dtype),
            ("healthy", "dead"): _constant_intensity(mu_hm, dtype),
            ("disabled", "dead"): _constant_intensity(nu_dm, dtype),
        }
    )
    result = illness_death.solve(
        initial="healthy",
        horizon=horizon,
        steps_per_unit=steps_per_unit,
        probability="collapse_point_no_duration",
        age=jnp.arange(batch_size, dtype=dtype),
    )
    _block_until_ready(result)
    max_abs_diff = float(jnp.max(jnp.abs(result["probability"] - expected)))
    if not jnp.allclose(result["probability"], expected, atol=1e-2, rtol=0.0):
        raise SystemExit(
            "Analytic correctness failed for illness-death benchmark.\n"
            f"max_abs_diff={max_abs_diff:.3e}"
        )
    print(
        "analytic correctness passed: "
        "scenario=illness-death "
        f"shape={result['probability'].shape} "
        f"max_abs_diff={max_abs_diff:.3e}"
    )

    duration_horizon = 1
    duration_steps = max(64, config.steps_per_unit)
    duration_times = jnp.linspace(
        0.0,
        duration_horizon,
        duration_horizon * duration_steps + 1,
        endpoint=True,
    )
    initial_duration = jnp.linspace(
        jnp.asarray(0.0, dtype=dtype),
        jnp.asarray(0.7, dtype=dtype),
        batch_size,
        dtype=dtype,
    )
    survival = _survival_under_duration_hazard(duration_times, initial_duration)
    expected = jnp.stack([survival, 1.0 - survival], axis=-1)
    duration_model = jact.StateSpace(
        states=["alive", "dead"],
        transitions=[("alive", "dead")],
    ).build(
        transitions={
            ("alive", "dead"): lambda t, d, **kwargs: jnp.broadcast_to(
                d,
                (kwargs["age"].shape[0], d.shape[-1]),
            )
        }
    )
    result = duration_model.solve(
        initial="alive",
        initial_duration=initial_duration,
        horizon=duration_horizon,
        steps_per_unit=duration_steps,
        probability="collapse_point_no_duration",
        record_every=2,
        age=jnp.arange(batch_size, dtype=dtype),
    )
    _block_until_ready(result)
    recorded_expected = expected[::2]
    max_abs_diff = float(jnp.max(jnp.abs(result["probability"] - recorded_expected)))
    if not jnp.allclose(
        result["probability"],
        recorded_expected,
        atol=2e-3,
        rtol=0.0,
    ):
        raise SystemExit(
            "Analytic correctness failed for duration-hazard benchmark.\n"
            f"max_abs_diff={max_abs_diff:.3e}"
        )
    print(
        "analytic correctness passed: "
        "scenario=duration-hazard-recorded "
        f"shape={result['probability'].shape} "
        f"max_abs_diff={max_abs_diff:.3e}"
    )
    print()


def _run_e2e_sanity_check(
    config: BenchmarkConfig,
    topology: TopologySpec,
    intensity,
    dtype: jnp.dtype,
) -> None:
    batch_size = config.correctness_batch_size
    ages = jnp.arange(batch_size, dtype=dtype)
    model = _build_model(topology, intensity)

    current_result = model.solve(
        initial="s0",
        horizon=config.horizon,
        steps_per_unit=config.steps_per_unit,
        probability="collapse_point_no_duration",
        age=ages,
    )

    _block_until_ready(current_result)

    current_probability = current_result["probability"]
    if not jnp.all(jnp.isfinite(current_probability)):
        raise SystemExit(
            f"Sanity check failed for topology={topology.name}.\n"
            "Encountered non-finite probabilities."
        )
    mass = jnp.sum(current_probability, axis=-1)
    if not jnp.allclose(
        mass,
        jnp.ones_like(mass),
        atol=config.correctness_atol,
        rtol=0.0,
    ):
        max_abs_diff = float(jnp.max(jnp.abs(mass - 1.0)))
        raise SystemExit(
            f"Sanity check failed for topology={topology.name}.\n"
            f"Probability mass deviated from one by {max_abs_diff:.3e}."
        )

    print(
        "sanity passed: "
        f"benchmark=e2e topology={topology.name} "
        f"shape={current_probability.shape} "
        f"atol={config.correctness_atol:g}"
    )


def _time_runs(name: str, fn: Callable[[], Any], timed_runs: int) -> TimingStats:
    timings_ms = []
    for _ in range(timed_runs):
        start_ns = time.perf_counter_ns()
        result = fn()
        _block_until_ready(result)
        elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
        timings_ms.append(elapsed_ms)

    p95_index = max(
        0,
        min(len(timings_ms) - 1, math.ceil(0.95 * len(timings_ms)) - 1),
    )
    sorted_timings = sorted(timings_ms)
    return TimingStats(
        name=name,
        median_ms=statistics.median(timings_ms),
        min_ms=min(timings_ms),
        p95_ms=sorted_timings[p95_index],
    )


def _warmup(fn: Callable[[], Any], warmup_runs: int) -> None:
    for _ in range(warmup_runs):
        result = fn()
        _block_until_ready(result)


def _print_timing_summary(
    current_stats: TimingStats,
    current_no_probability_stats: TimingStats,
    topology: str,
) -> None:
    print(f"timings: benchmark=e2e topology={topology}")
    for stats in (current_stats, current_no_probability_stats):
        print(
            f"{stats.name}: "
            f"median={stats.median_ms:.3f} ms, "
            f"min={stats.min_ms:.3f} ms, "
            f"p95={stats.p95_ms:.3f} ms"
        )
    print()


def _all_finite(tree: Any) -> bool:
    leaves = jax.tree_util.tree_leaves(tree)
    return all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)


def _cashflow_shapes(tree: Any) -> Any:
    return jax.tree_util.tree_map(lambda leaf: leaf.shape, tree)


def _print_cashflow_timing_summary(
    stats: TimingStats,
    topology: str,
    scenario: str,
) -> None:
    print(f"timings: benchmark=cashflow topology={topology} scenario={scenario}")
    print(
        f"current: median={stats.median_ms:.3f} ms, "
        f"min={stats.min_ms:.3f} ms, "
        f"p95={stats.p95_ms:.3f} ms"
    )
    print()


def _benchmark_e2e(
    config: BenchmarkConfig,
    topology: TopologySpec,
    intensity,
    dtype: jnp.dtype,
) -> None:
    ages = jnp.arange(config.batch_size, dtype=dtype)
    model = _build_model(topology, intensity)

    def run_current():
        return model.solve(
            initial="s0",
            horizon=config.horizon,
            steps_per_unit=config.steps_per_unit,
            probability="collapse_point_no_duration",
            age=ages,
        )

    def run_current_no_probability():
        return model.solve(
            initial="s0",
            horizon=config.horizon,
            steps_per_unit=config.steps_per_unit,
            probability=None,
            age=ages,
        )

    _warmup(run_current, config.warmup_runs)
    _warmup(run_current_no_probability, config.warmup_runs)

    current_stats = _time_runs("current", run_current, config.timed_runs)
    current_no_probability_stats = _time_runs(
        "current_probability_none",
        run_current_no_probability,
        config.timed_runs,
    )
    _print_timing_summary(
        current_stats,
        current_no_probability_stats,
        topology.name,
    )


def _benchmark_cashflows(
    config: BenchmarkConfig,
    topology: TopologySpec,
    model,
    scenario: CashflowScenario,
    dtype: jnp.dtype,
) -> None:
    ages = jnp.linspace(
        jnp.asarray(35.0, dtype=dtype),
        jnp.asarray(75.0, dtype=dtype),
        config.batch_size,
        dtype=dtype,
    )
    salary = jnp.linspace(
        jnp.asarray(50_000.0, dtype=dtype),
        jnp.asarray(150_000.0, dtype=dtype),
        config.batch_size,
        dtype=dtype,
    )
    event_time = jnp.linspace(
        jnp.asarray(0.0, dtype=dtype),
        jnp.asarray(config.horizon, dtype=dtype),
        config.batch_size,
        dtype=dtype,
    )

    def run_current():
        return model.solve(
            initial="s0",
            horizon=config.horizon,
            steps_per_unit=config.steps_per_unit,
            probability=scenario.probability,
            cashflows=scenario.cashflows,
            cashflow_views=scenario.views,
            record_every=scenario.record_every,
            age=ages,
            salary=salary,
            event_time=event_time,
        )

    _warmup(run_current, config.warmup_runs)
    result = run_current()
    _block_until_ready(result)
    cashflow_result = result["cashflows"]
    if not _all_finite(cashflow_result):
        raise SystemExit(
            f"Cashflow sanity check failed for topology={topology.name} "
            f"scenario={scenario.name}: encountered non-finite values."
        )
    print(
        "sanity passed: "
        f"benchmark=cashflow topology={topology.name} "
        f"scenario={scenario.name} shapes={_cashflow_shapes(cashflow_result)}"
    )

    stats = _time_runs(
        f"cashflow:{scenario.name}",
        run_current,
        config.timed_runs,
    )
    _print_cashflow_timing_summary(stats, topology.name, scenario.name)


def main() -> None:
    config = _parse_args()
    _maybe_require_gpu(config)
    scenarios = _selected_scenarios(config)
    dtype = jnp.float32

    _benchmark_header(config, scenarios)
    _run_analytic_correctness_checks(config, dtype)

    for scenario in scenarios:
        topology = _build_topology(scenario.topology, config.state_count)
        intensity = _build_intensity_matrix(config, topology, dtype)
        model = _build_model(topology, intensity)
        _run_e2e_sanity_check(config, topology, intensity, dtype)
        _benchmark_e2e(config, topology, intensity, dtype)
        for cashflow_scenario in _build_cashflow_scenarios(
            config,
            topology,
            model,
            dtype,
        ):
            _benchmark_cashflows(
                config,
                topology,
                model,
                cashflow_scenario,
                dtype,
            )


if __name__ == "__main__":
    main()
