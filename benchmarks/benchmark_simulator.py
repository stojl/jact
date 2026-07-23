"""Small command-line benchmark for the JACT event simulator.

Run from a source checkout, for example:

    PYTHONPATH=src python benchmarks/benchmark_simulator.py \
        --individuals 100000 --replicates 10 --max-jumps 64
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp

import jact


def build_model() -> jact.Model:
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

    return state_space.build(
        transitions={
            ("healthy", "disabled"): incidence,
            ("healthy", "dead"): healthy_mortality,
            ("disabled", "healthy"): recovery,
            ("disabled", "dead"): disabled_mortality,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--individuals", type=int, default=10_000)
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--steps-per-unit", type=int, default=12)
    parser.add_argument("--max-jumps", type=int, default=64)
    parser.add_argument("--runs", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = build_model()
    ages = jnp.linspace(30.0, 80.0, args.individuals)

    def run(key):
        return model.simulate(
            initial="healthy",
            horizon=args.horizon,
            steps_per_unit=args.steps_per_unit,
            max_jumps=args.max_jumps,
            replicates=args.replicates,
            key=key,
            age=ages,
        )

    start = time.perf_counter()
    result = run(jax.random.key(0))
    result.jump_times.block_until_ready()
    compile_and_first = time.perf_counter() - start

    timings: list[float] = []
    for run_index in range(args.runs):
        start = time.perf_counter()
        result = run(jax.random.key(run_index + 1))
        result.jump_times.block_until_ready()
        timings.append(time.perf_counter() - start)

    trajectories = args.individuals * args.replicates
    best = min(timings)
    print(f"backend: {jax.default_backend()}")
    print(f"compile + first run: {compile_and_first:.3f} s")
    print(f"best execution: {best:.3f} s")
    print(f"throughput: {trajectories / best:,.0f} trajectories/s")
    print(f"recorded jumps: {int(jnp.sum(result.jump_count)):,}")
    print(f"overflowed trajectories: {int(jnp.sum(result.overflow)):,}")


if __name__ == "__main__":
    main()
