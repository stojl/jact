#!/usr/bin/env python3
"""Compare ordinary payments with named shared cores on the active JAX backend.

Example:
    PYTHONPATH=src python benchmarks/benchmark_shared_payment_cores.py \
        --batches 64 256 --steps-per-unit 32 128 --consumers 1 2 8
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import jact
from jact import cashflows as cf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, nargs="+", default=[128])
    parser.add_argument("--steps-per-unit", type=int, nargs="+", default=[64])
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--consumers", type=int, nargs="+", default=[1, 2, 8, 16])
    parser.add_argument(
        "--kinds", nargs="+", choices=["cheap", "derived"], default=["cheap", "derived"]
    )
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    space = jact.StateSpace(["active", "disabled"], [("active", "disabled")])
    model = space.build(transitions={("active", "disabled"): lambda t, d, **kw: 0.15})
    frequencies = jnp.linspace(0.1, 1.0, 16)
    matrix = jnp.asarray(
        np.random.default_rng(12).normal(size=(16, 16)) / 4, dtype=jnp.float32
    )

    def basis(d, scale):
        values = jnp.sin((d * scale[:, None])[..., None] * frequencies)
        for _ in range(3):
            values = jnp.tanh(values @ matrix)
        return values.mean(axis=-1)

    def cheap(t, d, **kw):
        return jnp.exp(-0.2 * d) * (1 + 0.01 * t)

    def derived_core(t, d, **kw):
        return (1 + 0.1 * kw["basis"]) * (1 + 0.01 * t)

    rows = []
    for batch, steps, count, kind in itertools.product(
        args.batches,
        args.steps_per_unit,
        args.consumers,
        args.kinds,
    ):
        core = cheap if kind == "cheap" else derived_core
        fields = {} if kind == "cheap" else {"basis": basis}
        inputs = {
            "scale": jnp.linspace(0.5, 1.5, batch),
            "amounts": jnp.linspace(0.5, 2.0, batch * count).reshape(batch, count),
        }
        runs = {}
        row = dict(
            batch=batch, duration_cells=args.horizon * steps, consumers=count, core=kind
        )
        for shared in (False, True):
            components = {}
            for i in range(count):

                def weight(t, *, _i=i, **kw):
                    return kw["amounts"][:, _i] * (1 + 0.03 * t)

                if shared:
                    payment = cf.Scaled("base", weight=weight)
                else:

                    def payment(t, d, *, _weight=weight, **kw):
                        return core(t, d, **kw) * _weight(t, **kw)[:, None]

                components[str(i)] = cf.StateRate({"disabled": payment})
            declaration = space.cashflows(
                components, cores={"base": core}, derived=fields
            )

            def solve(data):
                return model.solve(
                    "active",
                    args.horizon,
                    steps,
                    cashflows=declaration,
                    probability=None,
                    cashflow_views={"raw": cf.Raw(terminal=True)},
                    **data,
                ).cashflows

            run = jax.jit(solve)
            start = time.perf_counter()
            compiled = run.lower(inputs).compile()
            compile_seconds = time.perf_counter() - start
            result = jax.block_until_ready(compiled(inputs))
            label = "shared" if shared else "ordinary"
            row[label + "_compile_seconds"] = compile_seconds
            runs[label] = (compiled, result)

        # Compare every component before measuring warmed execution.
        for left, right in zip(
            jax.tree.leaves(runs["ordinary"][1]), jax.tree.leaves(runs["shared"][1])
        ):
            np.testing.assert_allclose(left, right, rtol=3e-5, atol=1e-6)
        samples = {"ordinary": [], "shared": []}
        for iteration in range(args.warmups + args.repeats):
            # Alternate order to reduce systematic warmup/clock bias.
            order = (
                ("ordinary", "shared") if iteration % 2 == 0 else ("shared", "ordinary")
            )
            for label in order:
                start = time.perf_counter()
                jax.block_until_ready(runs[label][0](inputs))
                elapsed = time.perf_counter() - start
                if iteration >= args.warmups:
                    samples[label].append(elapsed)
        for label, times in samples.items():
            row[label + "_seconds"] = statistics.median(times)
        row["speedup"] = row["ordinary_seconds"] / row["shared_seconds"]
        rows.append(row)
        print(json.dumps(row), flush=True)
    report = {
        "jax": jax.__version__,
        "devices": [str(device) for device in jax.devices()],
        "backend": jax.default_backend(),
        "arguments": vars(args) | {"output": str(args.output)},
        "results": rows,
    }
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    else:
        print(
            json.dumps(
                {key: value for key, value in report.items() if key != "results"}
            )
        )


if __name__ == "__main__":
    main()
