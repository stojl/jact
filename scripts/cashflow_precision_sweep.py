"""Convergence sweep: terminal cashflow error vs steps_per_unit, float32 vs float64.

Runs in two passes (jax_enable_x64 is global, so each precision needs its own
process). Invoke with no args to run both and print a merged table; invoke with
--dtype f32|f64 to run a single precision and emit CSV on stdout.

Setup mirrors tests/test_cashflows.py::test_constant_intensity_time_duration_state_rate_matches_closed_form
- alive -> dead with constant hazard rate
- StateRate cashflow paying base + time_coef*t + duration_coef*d while alive
- terminal=True, so the accumulator runs the full horizon
"""

from __future__ import annotations

import argparse
import subprocess
import sys


RATE = 0.25
HORIZON = 2
BASE = 1.5
TIME_COEF = 0.4
DURATION_COEF = 0.2
STEPS_SWEEP = (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)


def run_sweep(use_x64: bool) -> list[tuple[int, float]]:
    import jax

    if use_x64:
        jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    import jact

    dtype = jnp.float64 if use_x64 else jnp.float32

    def constant_intensity(rate, batch):
        def fn(t, d, **kwargs):
            size = kwargs["age"].shape[0] if "age" in kwargs else batch
            return jnp.full((size, d.shape[-1]), rate, dtype=dtype)

        return fn

    def time_duration_payment(base, time_coef, duration_coef, batch):
        def fn(t, d, **kwargs):
            size = kwargs["age"].shape[0] if "age" in kwargs else batch
            level = base + time_coef * t + duration_coef * d
            return jnp.broadcast_to(jnp.asarray(level, dtype=dtype), (size, d.shape[-1]))

        return fn

    initial_duration = jnp.array([0.0, 0.75], dtype=dtype)

    survival = jnp.exp(-RATE * HORIZON)
    integral_0 = (1.0 - survival) / RATE
    integral_1 = (1.0 - survival * (1.0 + RATE * HORIZON)) / RATE**2
    expected = (
        (BASE + DURATION_COEF * initial_duration) * integral_0
        + (TIME_COEF + DURATION_COEF) * integral_1
    )

    ss = jact.StateSpace(["alive", "dead"], [("alive", "dead")])
    model = ss.build(transitions={("alive", "dead"): constant_intensity(RATE, 2)})
    cashflows = ss.cashflows({
        "annuity": jact.StateRate({
            "alive": time_duration_payment(BASE, TIME_COEF, DURATION_COEF, 2)
        })
    })

    rows: list[tuple[int, float]] = []
    for steps in STEPS_SWEEP:
        result = model.solve(
            initial="alive",
            initial_duration=initial_duration,
            horizon=HORIZON,
            steps_per_unit=steps,
            probability=None,
            cashflows=cashflows,
            cashflow_views={"annuity": jact.Raw("annuity", terminal=True)},
            age=jnp.arange(2.0, dtype=dtype),
        )
        err = float(jnp.max(jnp.abs(result["cashflows"]["annuity"] - expected)))
        rows.append((steps, err))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=("f32", "f64"))
    args = parser.parse_args()

    if args.dtype is not None:
        rows = run_sweep(use_x64=(args.dtype == "f64"))
        for steps, err in rows:
            print(f"{steps},{err:.6e}")
        return

    out_f32 = subprocess.run(
        [sys.executable, __file__, "--dtype", "f32"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    out_f64 = subprocess.run(
        [sys.executable, __file__, "--dtype", "f64"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    f32 = {int(s): float(e) for s, e in (l.split(",") for l in out_f32.strip().splitlines())}
    f64 = {int(s): float(e) for s, e in (l.split(",") for l in out_f64.strip().splitlines())}

    print(f"{'steps_per_unit':>14}  {'err float32':>14}  {'err float64':>14}")
    print(f"{'-'*14}  {'-'*14}  {'-'*14}")
    for steps in STEPS_SWEEP:
        print(f"{steps:>14}  {f32[steps]:>14.3e}  {f64[steps]:>14.3e}")


if __name__ == "__main__":
    main()
