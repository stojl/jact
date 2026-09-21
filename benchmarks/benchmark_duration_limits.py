#!/usr/bin/env python3
"""Warm, synchronized duration-limit comparisons; no timing assertions.

Run from the repository with ``python benchmarks/benchmark_duration_limits.py``.
Each scenario records only its initial and final diagnostics. Carry bytes count
all numeric state leaves (including point values/log-values/durations and tails),
not output buffers, compilation memory, or temporary arithmetic arrays.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import jact
from jact import cashflows as cf
from jact.probability import _state_probability_callback, _tail_callback


def diagnostic(state):
    return {
        "state": _state_probability_callback(state),
        "tail": _tail_callback(state)["mass"],
        "carry_bytes": jnp.asarray(
            sum(x.size * x.dtype.itemsize for x in jax.tree_util.tree_leaves(state))
        ),
    }


def build(profile):
    space = jact.StateSpace(states=("a", "b"), transitions=(("a", "b"), ("b", "a")))
    # Dense nonlinear layers make duration evaluation material in the expensive
    # profile. These functions remain smooth and vary beyond the cutoffs.
    weights = (
        jnp.asarray(np.random.default_rng(12).normal(size=(32, 32)), dtype=jnp.float32)
        / 6
    )
    frequencies = jnp.linspace(0.1, 1.0, 32)

    def base(t, d, **kwargs):
        x = d + kwargs["x"] + 0.01 * t
        if profile == "expensive":
            features = jnp.sin(x[..., None] * frequencies)
            for _ in range(3):
                features = jnp.tanh(features @ weights)
            return 0.1 + 0.02 * x + 0.05 * jax.nn.sigmoid(features.mean(axis=-1))
        return 0.1 + 0.02 * x

    def payment(t, d, **kwargs):
        return 1 + 5 * base(t, d, **kwargs)

    model = space.build(transitions={("a", "b"): base, ("b", "a"): base})
    declaration = space.cashflows(
        {
            "rate": cf.StateRate({"a": payment, "b": payment}),
            "jump": cf.TransitionLump({("a", "b"): payment, ("b", "a"): payment}),
        }
    )
    return model, declaration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizons", type=int, nargs="+", default=[10, 20, 40])
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--steps-per-unit", type=int, default=12)
    parser.add_argument("--limit", type=float, default=2.0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    settings = {
        "jax": jax.__version__,
        "devices": [str(d) for d in jax.devices()],
        "python": platform.python_version(),
        "machine": platform.machine(),
        "horizons": args.horizons,
        "batch": args.batch,
        "steps_per_unit": args.steps_per_unit,
        "cutoff": args.limit,
        "warmups": 1,
        "repeats": args.repeats,
        "dtype": "float32",
        "record_every": "horizon * steps_per_unit",
        "states": 2,
    }
    modes = {
        "exact": {},
        "intensity": {"intensity_duration_limit": args.limit},
        "payment": {"payment_duration_limit": args.limit},
        "probability": {"probability_duration_limit": args.limit},
        "all": {
            "intensity_duration_limit": args.limit,
            "payment_duration_limit": args.limit,
            "probability_duration_limit": args.limit,
        },
    }
    print(json.dumps(settings), flush=True)
    rows = []
    for profile in ("cheap", "expensive"):
        model, declaration = build(profile)
        for horizon in args.horizons:
            exact = None
            for mode, limits in modes.items():

                def run():
                    return model.solve(
                        "a",
                        horizon,
                        args.steps_per_unit,
                        x=jnp.linspace(0.0, 1.0, args.batch)[:, None],
                        probability=diagnostic,
                        cashflows=declaration,
                        cashflow_views={"total": cf.Total(terminal=True)},
                        record_every=horizon * args.steps_per_unit,
                        **limits,
                    )

                result = jax.block_until_ready(run())
                if exact is None:
                    exact = result
                timings = []
                for _ in range(args.repeats):
                    start = time.perf_counter()
                    result = jax.block_until_ready(run())
                    timings.append(1000 * (time.perf_counter() - start))
                cashflows: Any = result.cashflows
                exact_cashflows: Any = exact.cashflows
                row = {
                    "profile": profile,
                    "horizon": horizon,
                    "mode": mode,
                    "median_ms": statistics.median(timings),
                    "carry_bytes": int(result.probability["carry_bytes"][-1]),
                    "mean_final_tail_mass": float(
                        result.probability["tail"][-1].sum(axis=-1).mean()
                    ),
                    "max_probability_error": float(
                        jnp.max(
                            jnp.abs(
                                result.probability["state"] - exact.probability["state"]
                            )
                        )
                    ),
                    "max_cashflow_error": float(
                        jnp.max(jnp.abs(cashflows["total"] - exact_cashflows["total"]))
                    ),
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
            jax.clear_caches()
    if args.output:
        args.output.write_text(
            json.dumps({"settings": settings, "results": rows}, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
