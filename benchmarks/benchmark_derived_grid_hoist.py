#!/usr/bin/env python3
"""Warm GPU benchmark for a shared nonlinear duration-derived field.

The duration basis uses the nonlinear feature stack from
``benchmark_duration_limits.py``. Intensities and three payment kinds consume
the same field on midpoint and left-edge grids and on point characteristics.
Run this script with a checkout on ``PYTHONPATH`` to compare implementations.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import jact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--steps-per-unit", type=int, default=12)
    parser.add_argument("--duration-limit", type=float)
    parser.add_argument("--warmups", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    weights = jnp.asarray(
        np.random.default_rng(12).normal(size=(32, 32)) / 6,
        dtype=jnp.float32,
    )
    frequencies = jnp.linspace(0.1, 1.0, 32)

    def duration_basis(d, scale):
        features = jnp.sin((d * scale)[..., None] * frequencies)
        for _ in range(3):
            features = jnp.tanh(features @ weights)
        return features.mean(axis=-1)

    def intensity(factor):
        def fn(t, d, **kw):
            return 0.04 + factor * kw["basis"] + 0.0005 * t

        return fn

    def payment(factor):
        def fn(t, d, **kw):
            return 1.0 + factor * kw["basis"] + 0.002 * t

        return fn

    space = jact.StateSpace(
        ("active", "disabled", "dead"),
        (("active", "disabled"), ("active", "dead"), ("disabled", "dead")),
    )
    model = space.build(
        transitions={
            ("active", "disabled"): intensity(0.010),
            ("active", "dead"): intensity(0.008),
            ("disabled", "dead"): intensity(0.012),
        },
        derived={"basis": duration_basis},
    )
    cashflows = space.cashflows(
        {
            "income": jact.cashflows.StateRate(
                {"active": payment(0.5), "disabled": payment(0.8)}
            ),
            "claims": jact.cashflows.TransitionLump(
                {
                    ("active", "disabled"): payment(2.0),
                    ("active", "dead"): payment(3.0),
                    ("disabled", "dead"): payment(4.0),
                }
            ),
            "bonus": jact.cashflows.ScheduledEvent(
                when=lambda **kw: args.horizon / 2,
                payments={"active": payment(1.5)},
            ),
        }
    )
    scale = jnp.linspace(0.5, 1.5, args.batch, dtype=jnp.float32)[:, None]
    initial_duration = jnp.linspace(0.0, 1.0, args.batch, dtype=jnp.float32)

    def run():
        return model.solve(
            initial="active",
            initial_duration=initial_duration,
            horizon=args.horizon,
            steps_per_unit=args.steps_per_unit,
            probability=None,
            cashflows=cashflows,
            cashflow_views={"pv": jact.cashflows.Total(terminal=True)},
            record_every=args.horizon * args.steps_per_unit,
            intensity_duration_limit=args.duration_limit,
            payment_duration_limit=args.duration_limit,
            scale=scale,
        ).cashflows["pv"]

    for _ in range(args.warmups):
        jax.block_until_ready(run())
    timings = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        value = jax.block_until_ready(run())
        timings.append(1000 * (time.perf_counter() - start))

    if args.output is not None:
        np.save(args.output, np.asarray(value))

    print(
        json.dumps(
            {
                "backend": jax.default_backend(),
                "device": str(jax.devices()[0]),
                "jax": jax.__version__,
                "batch": args.batch,
                "horizon": args.horizon,
                "steps_per_unit": args.steps_per_unit,
                "duration_limit": args.duration_limit,
                "warmups": args.warmups,
                "repeats": args.repeats,
                "median_ms": statistics.median(timings),
                "min_ms": min(timings),
                "mean_pv": float(jnp.mean(value)),
            }
        )
    )


if __name__ == "__main__":
    main()
