"""Reproducible GPU benchmark matrix for the JACT event simulator.

Primary baseline:

    PYTHONPATH=src python benchmarks/benchmark_simulator.py

All scenarios and parameter sweeps:

    PYTHONPATH=src python benchmarks/benchmark_simulator.py \
        --scenario all --sweep all --json

GPU execution is required unless ``--allow-cpu`` is supplied.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import re
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from typing import Any


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _release_tuple(version: str) -> tuple[int, ...] | None:
    match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", version)
    if match is None:
        return None
    return tuple(int(value or 0) for value in match.groups())


def _package_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "jact": _distribution_version("jact"),
        "jax": _distribution_version("jax"),
        "jaxlib": _distribution_version("jaxlib"),
        "jax_cuda_plugin": _distribution_version("jax-cuda12-plugin"),
        "jax_cuda_pjrt": _distribution_version("jax-cuda12-pjrt"),
        "cudnn": _distribution_version("nvidia-cudnn-cu12"),
        "numpy": _distribution_version("numpy"),
    }


def _check_package_compatibility() -> None:
    """Stop before importing JAX when plugin versions are visibly incompatible."""
    versions = _package_versions()
    jax_release = _release_tuple(versions["jax"])
    jaxlib_release = _release_tuple(versions["jaxlib"])
    plugin_release = _release_tuple(versions["jax_cuda_plugin"])
    problems: list[str] = []
    if jax_release is None or jaxlib_release is None:
        problems.append("JAX and JAXLIB must both be installed.")
    elif jax_release[:3] != jaxlib_release[:3]:
        problems.append(
            f"JAX {versions['jax']} and JAXLIB {versions['jaxlib']} "
            "do not have matching release versions."
        )
    if (
        plugin_release is not None
        and jaxlib_release is not None
        and plugin_release[:3] != jaxlib_release[:3]
    ):
        problems.append(
            f"CUDA plugin {versions['jax_cuda_plugin']} and JAXLIB "
            f"{versions['jaxlib']} do not have matching release versions."
        )
    if problems:
        details = "\n".join(f"  - {problem}" for problem in problems)
        installed = json.dumps(versions, indent=2, sort_keys=True)
        raise SystemExit(
            "Incompatible JAX benchmark environment detected before backend "
            f"initialization:\n{details}\nInstalled versions:\n{installed}"
        )


if __name__ == "__main__":
    _check_package_compatibility()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import jact  # noqa: E402


@dataclass(frozen=True)
class Scenario:
    name: str
    model: jact.Model
    initial: Any
    covariates: dict[str, jax.Array]


@dataclass(frozen=True)
class BenchmarkResult:
    scenario: str
    individuals: int
    replicates: int
    horizon: int
    steps_per_unit: int
    max_jumps: int
    warmups: int
    timed_runs: int
    compile_first_seconds: float
    median_seconds: float
    minimum_seconds: float
    p95_seconds: float
    trajectory_throughput: float
    jump_throughput: float
    recorded_jumps: int
    overflow_count: int
    device_count: int
    local_batch_size: int
    padding_rows: int
    memory_before_bytes: int | None
    memory_after_compile_bytes: int | None
    peak_memory_bytes: int | None


def build_standard(individuals: int) -> Scenario:
    state_space = jact.StateSpace(
        states=("healthy", "disabled", "dead"),
        transitions=(
            ("healthy", "disabled"),
            ("healthy", "dead"),
            ("disabled", "healthy"),
            ("disabled", "dead"),
        ),
    )

    def incidence(t, d, *, age):
        del d
        return 0.01 * jnp.exp(0.04 * (age[:, None] + t - 50.0))

    def healthy_mortality(t, d, *, age):
        del d
        return 0.001 * jnp.exp(0.08 * (age[:, None] + t - 50.0))

    def recovery(t, d, *, age):
        del t, age
        return jnp.where(d < 1.0, 0.8, 0.2)

    def disabled_mortality(t, d, *, age):
        return (
            0.003
            * jnp.exp(0.08 * (age[:, None] + t - 50.0))
            * (1.0 + 0.5 * d)
        )

    model = state_space.build(
        transitions={
            ("healthy", "disabled"): incidence,
            ("healthy", "dead"): healthy_mortality,
            ("disabled", "healthy"): recovery,
            ("disabled", "dead"): disabled_mortality,
        }
    )
    return Scenario(
        "standard",
        model,
        "healthy",
        {"age": jnp.linspace(30.0, 80.0, individuals)},
    )


def build_no_transition(individuals: int) -> Scenario:
    state_space = jact.StateSpace(states=("alive",), transitions=())
    return Scenario(
        "no-transition",
        state_space.build(),
        "alive",
        {"row": jnp.arange(individuals, dtype=jnp.float32)},
    )


def build_high_transition(individuals: int) -> Scenario:
    state_space = jact.StateSpace(
        states=("a", "b"),
        transitions=(("a", "b"), ("b", "a")),
    )
    model = state_space.build(
        transitions={
            ("a", "b"): lambda t, d, **kwargs: 8.0 + 0.0 * kwargs["row"][:, None],
            ("b", "a"): lambda t, d, **kwargs: 8.0 + 0.0 * kwargs["row"][:, None],
        }
    )
    return Scenario(
        "high-transition",
        model,
        "a",
        {"row": jnp.arange(individuals, dtype=jnp.float32)},
    )


def build_grouped_involved(individuals: int) -> Scenario:
    state_space = jact.StateSpace(
        states=("active", "cause-a", "cause-b", "cause-c"),
        transitions=(
            ("active", "cause-a"),
            ("active", "cause-b"),
            ("active", "cause-c"),
        ),
    )

    def grouped(t, d, *, age, score):
        attained_age = age[:, None] + t
        base = (
            jnp.log1p(jnp.exp(0.03 * (attained_age - 45.0)))
            * (1.0 + jnp.square(jnp.sin(d + score[:, None])))
            / 100.0
        )
        return jnp.stack(
            (
                base,
                base * jnp.exp(0.1 * score[:, None]),
                base * (1.0 + jnp.sqrt(d + 0.25)),
            ),
            axis=0,
        )

    model = state_space.build(
        groups={
            grouped: (
                ("active", "cause-a"),
                ("active", "cause-b"),
                ("active", "cause-c"),
            )
        }
    )
    return Scenario(
        "grouped-involved",
        model,
        "active",
        {
            "age": jnp.linspace(25.0, 85.0, individuals),
            "score": jnp.linspace(-1.0, 1.0, individuals),
        },
    )


SCENARIO_BUILDERS = {
    "standard": build_standard,
    "no-transition": build_no_transition,
    "high-transition": build_high_transition,
    "grouped-involved": build_grouped_involved,
}


def _resize_scenario(scenario: Scenario, individuals: int) -> Scenario:
    if scenario.name == "standard":
        covariates = {"age": jnp.linspace(30.0, 80.0, individuals)}
    elif scenario.name in ("no-transition", "high-transition"):
        covariates = {"row": jnp.arange(individuals, dtype=jnp.float32)}
    else:
        covariates = {
            "age": jnp.linspace(25.0, 85.0, individuals),
            "score": jnp.linspace(-1.0, 1.0, individuals),
        }
    return Scenario(
        scenario.name,
        scenario.model,
        scenario.initial,
        covariates,
    )


def _driver_version() -> str:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return "unavailable"
    return completed.stdout.strip().splitlines()[0]


def _memory_stats(device: Any) -> dict[str, int]:
    try:
        stats = device.memory_stats()
    except Exception:
        return {}
    if stats is None:
        return {}
    return {str(name): int(value) for name, value in stats.items()}


def _memory_value(stats: dict[str, int], *names: str) -> int | None:
    for name in names:
        if name in stats:
            return stats[name]
    return None


def _block(result: jact.SimulationResult) -> None:
    result.jump_times.block_until_ready()


def benchmark_case(
    scenario: Scenario,
    *,
    individuals: int,
    replicates: int,
    horizon: int,
    steps_per_unit: int,
    max_jumps: int,
    devices: int | None,
    warmups: int,
    runs: int,
) -> BenchmarkResult:
    def run(key: jax.Array) -> jact.SimulationResult:
        return scenario.model.simulate(
            initial=scenario.initial,
            horizon=horizon,
            steps_per_unit=steps_per_unit,
            max_jumps=max_jumps,
            replicates=replicates,
            key=key,
            devices=devices,
            **scenario.covariates,
        )

    memory_before = _memory_stats(jax.devices()[0])
    started = time.perf_counter()
    result = run(jax.random.key(0))
    _block(result)
    compile_first = time.perf_counter() - started
    for warmup_index in range(1, warmups):
        result = run(jax.random.key(warmup_index))
        _block(result)
    memory_after = _memory_stats(jax.devices()[0])

    timings: list[float] = []
    for run_index in range(runs):
        started = time.perf_counter()
        result = run(jax.random.key(warmups + run_index))
        _block(result)
        timings.append(time.perf_counter() - started)

    median = statistics.median(timings)
    recorded_jumps = int(np.asarray(jnp.sum(result.jump_count)))
    overflow_count = int(np.asarray(jnp.sum(result.overflow)))
    trajectory_count = individuals * replicates
    device_count = 1 if devices is None else devices
    local_batch_size = (individuals + device_count - 1) // device_count
    return BenchmarkResult(
        scenario=scenario.name,
        individuals=individuals,
        replicates=replicates,
        horizon=horizon,
        steps_per_unit=steps_per_unit,
        max_jumps=max_jumps,
        warmups=warmups,
        timed_runs=runs,
        compile_first_seconds=compile_first,
        median_seconds=median,
        minimum_seconds=min(timings),
        p95_seconds=float(np.percentile(np.asarray(timings), 95)),
        trajectory_throughput=trajectory_count / median,
        jump_throughput=recorded_jumps / median,
        recorded_jumps=recorded_jumps,
        overflow_count=overflow_count,
        device_count=device_count,
        local_batch_size=local_batch_size,
        padding_rows=local_batch_size * device_count - individuals,
        memory_before_bytes=_memory_value(
            memory_before,
            "bytes_in_use",
            "bytes_reserved",
        ),
        memory_after_compile_bytes=_memory_value(
            memory_after,
            "bytes_in_use",
            "bytes_reserved",
        ),
        peak_memory_bytes=_memory_value(
            memory_after,
            "peak_bytes_in_use",
            "peak_bytes_reserved",
        ),
    )


def _print_result(result: BenchmarkResult) -> None:
    print(
        f"{result.scenario}: B={result.individuals:,}, "
        f"R={result.replicates}, steps={result.steps_per_unit}, "
        f"max_jumps={result.max_jumps}"
    )
    print(
        f"  compile-first={result.compile_first_seconds:.6f}s  "
        f"median={result.median_seconds:.6f}s  "
        f"min={result.minimum_seconds:.6f}s  "
        f"p95={result.p95_seconds:.6f}s"
    )
    print(
        f"  trajectories={result.trajectory_throughput:,.0f}/s  "
        f"jumps={result.jump_throughput:,.0f}/s  "
        f"recorded={result.recorded_jumps:,}  "
        f"overflow={result.overflow_count:,}"
    )
    print(
        f"  devices={result.device_count}  "
        f"local_batch={result.local_batch_size:,}  "
        f"padding={result.padding_rows:,}  "
        f"memory(before/after/peak)="
        f"{result.memory_before_bytes}/"
        f"{result.memory_after_compile_bytes}/"
        f"{result.peak_memory_bytes}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        choices=(*SCENARIO_BUILDERS, "all"),
        default="standard",
    )
    parser.add_argument(
        "--sweep",
        action="append",
        choices=("steps", "max-jumps", "replicates", "batch", "all"),
        default=[],
    )
    parser.add_argument("--individuals", type=int, default=1_000)
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--steps-per-unit", type=int, default=12)
    parser.add_argument("--max-jumps", type=int, default=64)
    parser.add_argument("--devices", type=int)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def _validate_environment(allow_cpu: bool) -> dict[str, Any]:
    versions = _package_versions()
    try:
        backend = jax.default_backend()
        devices = jax.devices()
        probe = jnp.arange(8.0)
        jnp.sum(probe * probe).block_until_ready()
    except Exception as exc:
        raise SystemExit(
            "JAX backend initialization failed. Check the CUDA plugin, NVIDIA "
            f"driver, and cuDNN installation. Versions: {versions}. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc
    if backend != "gpu" and not allow_cpu:
        raise SystemExit(
            "This benchmark requires a working JAX GPU backend by default; "
            f"detected backend={backend!r}. Use --allow-cpu only for functional "
            f"smoke tests. Versions: {versions}"
        )
    return {
        "backend": backend,
        "devices": [str(device) for device in devices],
        "device_kind": getattr(devices[0], "device_kind", "unknown"),
        "driver": _driver_version(),
        "versions": versions,
    }


def _case_matrix(
    args: argparse.Namespace,
) -> list[tuple[str, int, int, int, int]]:
    scenarios = (
        tuple(SCENARIO_BUILDERS)
        if args.scenario == "all"
        else (args.scenario,)
    )
    cases = [
        (
            name,
            args.individuals,
            args.replicates,
            args.steps_per_unit,
            args.max_jumps,
        )
        for name in scenarios
    ]
    sweeps = set(args.sweep)
    if "all" in sweeps:
        sweeps = {"steps", "max-jumps", "replicates", "batch"}
    if "steps" in sweeps:
        cases.extend(
            ("standard", args.individuals, args.replicates, value, args.max_jumps)
            for value in (1, 2, 4, 8, 12, 24)
        )
    if "max-jumps" in sweeps:
        cases.extend(
            (
                "standard",
                args.individuals,
                args.replicates,
                args.steps_per_unit,
                value,
            )
            for value in (8, 16, 32, 64, 128)
        )
    if "replicates" in sweeps:
        cases.extend(
            (
                "standard",
                args.individuals,
                value,
                args.steps_per_unit,
                args.max_jumps,
            )
            for value in (1, 2, 4, 8, 16)
        )
    if "batch" in sweeps:
        cases.extend(
            (
                "standard",
                value,
                args.replicates,
                args.steps_per_unit,
                args.max_jumps,
            )
            for value in (896, 928, 960, 992, 1_000, 1_008, 1_024, 1_056)
        )
    return list(dict.fromkeys(cases))


def main() -> None:
    args = parse_args()
    if args.warmups < 1:
        raise SystemExit("--warmups must be at least 1.")
    if args.runs < 1:
        raise SystemExit("--runs must be at least 1.")
    environment = _validate_environment(args.allow_cpu)
    results: list[BenchmarkResult] = []
    scenarios: dict[str, Scenario] = {}
    for name, individuals, replicates, steps_per_unit, max_jumps in _case_matrix(
        args
    ):
        base_scenario = scenarios.get(name)
        if base_scenario is None:
            base_scenario = SCENARIO_BUILDERS[name](args.individuals)
            scenarios[name] = base_scenario
        scenario = _resize_scenario(base_scenario, individuals)
        result = benchmark_case(
            scenario,
            individuals=individuals,
            replicates=replicates,
            horizon=args.horizon,
            steps_per_unit=steps_per_unit,
            max_jumps=max_jumps,
            devices=args.devices,
            warmups=args.warmups,
            runs=args.runs,
        )
        results.append(result)
        if not args.json:
            _print_result(result)

    if args.json:
        print(
            json.dumps(
                {
                    "environment": environment,
                    "results": [asdict(result) for result in results],
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(f"backend: {environment['backend']}")
        print(f"device: {environment['device_kind']}")
        print(f"driver: {environment['driver']}")
        print(
            "versions: "
            + ", ".join(
                f"{name}={version}"
                for name, version in environment["versions"].items()
            )
        )


if __name__ == "__main__":
    main()
