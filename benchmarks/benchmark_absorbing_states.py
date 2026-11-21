#!/usr/bin/env python3
"""Compare warmed GPU throughput across two JACT source trees.

Extract the baseline package first, then run the benchmark against it.

    mkdir -p /tmp/jact-before
    git archive 3a21b27 src/jact | tar -x -C /tmp/jact-before
    PYTHONPATH=src python benchmarks/benchmark_absorbing_states.py \
        --baseline-src /tmp/jact-before/src
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import statistics
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import jact


def load_baseline(source: Path):
    package = source / "jact"
    spec = importlib.util.spec_from_file_location(
        "jact_baseline",
        package / "__init__.py",
        submodule_search_locations=[str(package)],
    )
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load a JACT package from {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_solve(lib, scenario, horizon, steps_per_unit, limit):
    edges = (
        ("active", "disabled"),
        ("active", "dead"),
        ("disabled", "active"),
        ("disabled", "dead"),
    )
    space = lib.StateSpace(("active", "disabled", "dead"), edges)

    def hazard(scale):
        def evaluate(t, d, *, age, **kwargs):
            return scale * (
                0.035 + 0.003 * t + 0.0003 * age[:, None] + 0.06 * jnp.exp(-d)
            )

        return evaluate

    model = space.build(
        transitions={
            edge: hazard(scale) for edge, scale in zip(edges, (1.0, 0.4, 0.7, 0.5))
        }
    )
    initial = space.initial_distribution(
        {
            "active": {"mass": 0.9, "duration": 0.0},
            "disabled": {"mass": 0.1, "duration": 1.5},
        }
    )

    def benefit(t, d, *, age, **kwargs):
        return (0.8 + 0.15 * jnp.exp(-0.2 * d)) * (1 + 0.004 * age[:, None])

    def discount(t, **kwargs):
        return jnp.exp(-0.025 * t)

    cf = lib.cashflows
    cashflows = views = None
    if scenario != "probability":
        rates = {"disabled": benefit}
        if scenario == "observed":
            # A duration-dependent payment in the absorbing state keeps its grid.
            rates["dead"] = benefit
        cashflows = space.cashflows(
            {
                "benefit": cf.StateRate(rates),
                "death": cf.TransitionLump(
                    {("active", "dead"): benefit, ("disabled", "dead"): benefit}
                ),
            }
        )
        views = {"pv": cf.Total(terminal=True, weight=discount)}

    probability = None if scenario == "terminal" else lib.probability.StateProbability()
    record_every = 1 if scenario == "probability" else steps_per_unit

    def solve(age):
        return model.solve(
            initial,
            horizon,
            steps_per_unit,
            age=age,
            probability=probability,
            record_every=record_every,
            cashflows=cashflows,
            cashflow_views=views,
            probability_duration_limit=limit,
        )

    return jax.jit(solve)


def benchmark(libraries, inputs, scenario, limit, args):
    executables = {}
    reference = None
    max_abs_diff = 0.0
    for label, lib in libraries.items():
        solve = make_solve(lib, scenario, args.horizon, args.steps_per_unit, limit)
        start = time.perf_counter()
        executable = solve.lower(inputs).compile()
        compile_seconds = time.perf_counter() - start
        solved = jax.device_get(executable(inputs))
        result = (solved.probability, solved.cashflows)
        if reference is None:
            reference = result
            states = solved.states
        else:
            assert solved.states == states
            assert jax.tree.structure(reference) == jax.tree.structure(result)
            for before, after in zip(
                jax.tree.leaves(reference), jax.tree.leaves(result)
            ):
                np.testing.assert_allclose(before, after, rtol=3e-5, atol=1e-6)
                max_abs_diff = max(max_abs_diff, float(np.max(np.abs(before - after))))
        executables[label] = (executable, compile_seconds)

    timings = {label: [] for label in executables}
    for i in range(args.warmups + args.repeats):
        labels = list(executables)
        if i % 2:
            labels.reverse()
        for label in labels:
            start = time.perf_counter()
            jax.block_until_ready(executables[label][0](inputs))
            elapsed = time.perf_counter() - start
            if i >= args.warmups:
                timings[label].append(elapsed)

    row = {
        "scenario": scenario,
        "batch": inputs.shape[0],
        "probability_duration_limit": limit,
        "max_abs_diff": max_abs_diff,
    }
    for label, samples in timings.items():
        seconds = statistics.median(samples)
        row[label] = {
            "median_ms": 1e3 * seconds,
            "policies_per_second": inputs.shape[0] / seconds,
            "compile_seconds": executables[label][1],
        }
    row["speedup"] = row["baseline"]["median_ms"] / row["current"]["median_ms"]
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-src", type=Path, required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[64, 1024, 4096])
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--steps-per-unit", type=int, default=12)
    parser.add_argument("--compression-limit", type=float, default=2.0)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=["probability", "combined", "terminal", "observed"],
        default=["probability", "combined", "terminal", "observed"],
    )
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=15)
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        parser.error("A JAX GPU backend is required.")
    libs = {"baseline": load_baseline(args.baseline_src), "current": jact}
    metadata = {
        "jax": jax.__version__,
        "gpu": jax.devices()[0].device_kind,
        "horizon": args.horizon,
        "steps_per_unit": args.steps_per_unit,
        "warmups": args.warmups,
        "repeats": args.repeats,
    }
    print(json.dumps(metadata), flush=True)
    for batch, limit, scenario in itertools.product(
        args.batches, (None, args.compression_limit), args.scenarios
    ):
        inputs = jax.block_until_ready(jnp.linspace(35, 75, batch, dtype=jnp.float32))
        print(json.dumps(benchmark(libs, inputs, scenario, limit, args)), flush=True)


if __name__ == "__main__":
    main()
