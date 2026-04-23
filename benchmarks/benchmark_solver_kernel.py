#!/usr/bin/env python3
"""Benchmark the production end-to-end solver against the original prototype.

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
PROTO_DIR = REPO_ROOT / "docs" / "original_prototype"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PROTO_DIR) not in sys.path:
    sys.path.insert(0, str(PROTO_DIR))

import prototype_8

import jact


MAX_BATCH_SIZE = 1_000
DEFAULT_BATCH_SIZE = 1_000
DEFAULT_HORIZON = 30
DEFAULT_STEPS_PER_UNIT = 12
DEFAULT_WARMUP_RUNS = 1
DEFAULT_TIMED_RUNS = 20
DEFAULT_CORRECTNESS_BATCH_SIZE = 128
DEFAULT_CORRECTNESS_ATOL = 1e-6
DEFAULT_STATE_COUNT = 12
TOPOLOGY_CHOICES = ("sparse", "dense", "all")
INTENSITY_PROFILE_CHOICES = ("simple", "involved")


@dataclass(frozen=True)
class BenchmarkConfig:
    batch_size: int
    horizon: int
    steps_per_unit: int
    warmup_runs: int
    timed_runs: int
    correctness_batch_size: int
    correctness_atol: float
    perturbation: float
    allow_cpu: bool
    topology: str
    state_count: int
    intensity_profile: str

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
    parser.add_argument("--perturbation", type=float, default=1e-12)
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
        perturbation=args.perturbation,
        allow_cpu=args.allow_cpu,
        topology=args.topology,
        state_count=args.state_count,
        intensity_profile=args.intensity_profile,
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


def _run_prototype_e2e(
    *,
    horizon: int,
    steps_per_unit: int,
    intensity,
    ages: jnp.ndarray,
    perturbation: float,
):
    semimarkov_solver = getattr(
        prototype_8.semimarkov_solver,
        "__wrapped__",
        prototype_8.semimarkov_solver,
    )
    return semimarkov_solver(
        units=horizon,
        discretization_unit=steps_per_unit,
        intensity=intensity,
        intensity_kwargs={"age": ages},
        prob_callback=_prototype_collapse_point_no_duration,
        pertubation=perturbation,
        transpose_result=True,
    )


@jax.jit
def _prototype_collapse_point_no_duration(
    p: jnp.ndarray,
    p_point: jnp.ndarray,
) -> jnp.ndarray:
    p = p.at[..., 0, :].add(p_point)
    return jnp.sum(p, axis=-1)


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
    print(f"warmup_runs: {config.warmup_runs}")
    print(f"timed_runs: {config.timed_runs}")
    print(f"perturbation: {config.perturbation}")
    print(f"scenarios: {', '.join(scenario.label for scenario in scenarios)}")
    print()


def _run_e2e_correctness_check(
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
        callback="collapse_point_no_duration",
        perturbation=config.perturbation,
        age=ages,
    )
    prototype_result = _run_prototype_e2e(
        horizon=config.horizon,
        steps_per_unit=config.steps_per_unit,
        intensity=intensity,
        ages=ages,
        perturbation=config.perturbation,
    )

    _block_until_ready(current_result)
    _block_until_ready(prototype_result)

    current_probability = current_result["probability"]
    prototype_probability = jnp.swapaxes(prototype_result["probability"], 0, 1)
    if not jnp.allclose(
        current_probability[:-1],
        prototype_probability[:-1],
        atol=config.correctness_atol,
        rtol=0.0,
    ):
        max_abs_diff = float(
            jnp.max(
                jnp.abs(current_probability[:-1] - prototype_probability[:-1])
            )
        )
        raise SystemExit(
            f"End-to-end correctness check failed for topology={topology.name}.\n"
            f"Maximum absolute difference: {max_abs_diff:.3e}"
        )

    print(
        "correctness passed: "
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
    prototype_stats: TimingStats,
    topology: str,
) -> None:
    speedup = prototype_stats.median_ms / current_stats.median_ms
    print(f"timings: benchmark=e2e topology={topology}")
    for stats in (current_stats, prototype_stats):
        print(
            f"{stats.name}: "
            f"median={stats.median_ms:.3f} ms, "
            f"min={stats.min_ms:.3f} ms, "
            f"p95={stats.p95_ms:.3f} ms"
        )
    print(f"speedup_vs_prototype: {speedup:.3f}x")
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
            callback="collapse_point_no_duration",
            perturbation=config.perturbation,
            age=ages,
        )

    def run_prototype():
        return _run_prototype_e2e(
            horizon=config.horizon,
            steps_per_unit=config.steps_per_unit,
            intensity=intensity,
            ages=ages,
            perturbation=config.perturbation,
        )

    _warmup(run_current, config.warmup_runs)
    _warmup(run_prototype, config.warmup_runs)

    current_stats = _time_runs("current", run_current, config.timed_runs)
    prototype_stats = _time_runs("prototype", run_prototype, config.timed_runs)
    _print_timing_summary(current_stats, prototype_stats, topology.name)


def main() -> None:
    config = _parse_args()
    _maybe_require_gpu(config)
    scenarios = _selected_scenarios(config)
    dtype = jnp.float32

    _benchmark_header(config, scenarios)

    for scenario in scenarios:
        topology = _build_topology(scenario.topology, config.state_count)
        intensity = _build_intensity_matrix(config, topology, dtype)
        _run_e2e_correctness_check(config, topology, intensity, dtype)
        _benchmark_e2e(config, topology, intensity, dtype)


if __name__ == "__main__":
    main()
