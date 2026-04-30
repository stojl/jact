"""Convergence sweep for float32 density-only advection precision.

This bypasses the public initial-distribution API and seeds one unit of mass
directly into the duration-density grid, with no point mass. The setup isolates
the density advection recurrence under a constant hazard and values a unit
state-rate annuity.
"""

from __future__ import annotations

import argparse
import subprocess
import sys


RATE = 0.25
HORIZON = 2
STEPS_SWEEP = (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)


def run_sweep(use_x64: bool) -> list[tuple[int, float]]:
    import jax

    if use_x64:
        jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from jact.callbacks import StateCarry
    from jact.solver import _KIND_STATE_RATE, _SOURCE_COMPONENT, _midpoint_solver

    dtype = jnp.float64 if use_x64 else jnp.float32
    batch_size = 1

    def constant_intensity(t, d, **kwargs):
        del t, kwargs
        return jnp.full((batch_size, d.shape[-1]), RATE, dtype=dtype)

    def unit_payment(t, d, **kwargs):
        del t, kwargs
        return jnp.ones((batch_size, d.shape[-1]), dtype=dtype)

    expected = (1.0 - jnp.exp(jnp.asarray(-RATE * HORIZON, dtype=dtype))) / RATE
    rows: list[tuple[int, float]] = []
    for steps_per_unit in STEPS_SWEEP:
        solver_steps = HORIZON * steps_per_unit
        step_size = 1 / steps_per_unit
        alive_density = jnp.zeros((batch_size, solver_steps), dtype=dtype)
        alive_density = alive_density.at[:, 0].set(1.0)
        dead_density = jnp.zeros_like(alive_density)
        state_0 = (
            StateCarry(density=alive_density, point_mass=None),
            StateCarry(density=dead_density, point_mass=None),
        )
        grid = jnp.linspace(
            0.0,
            HORIZON,
            solver_steps + 1,
            endpoint=True,
            dtype=dtype,
        )[None, :]
        duration_left = grid[:, :-1]
        duration_mid = 0.5 * (duration_left + grid[:, 1:])

        result = _midpoint_solver(
            state_0,
            duration_mid,
            duration_left,
            step_size,
            ((None, constant_intensity), (None, None)),
            {},
            lambda _state: None,
            solver_steps,
            ((_KIND_STATE_RATE, ((0, unit_payment),)),),
            (
                (
                    "annuity",
                    True,
                    None,
                    ((_SOURCE_COMPONENT, 0),),
                    ("annuity",),
                    "single",
                ),
            ),
        )
        value = result["cashflow_terminal"][0][0]
        rows.append((steps_per_unit, float(jnp.max(jnp.abs(value - expected)))))
    return rows


def main() -> None:
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

    f32 = {
        int(steps): float(error)
        for steps, error in (
            line.split(",") for line in out_f32.strip().splitlines()
        )
    }
    f64 = {
        int(steps): float(error)
        for steps, error in (
            line.split(",") for line in out_f64.strip().splitlines()
        )
    }

    print(f"{'steps_per_unit':>14}  {'density err float32':>20}  {'float64':>14}")
    print(f"{'-' * 14}  {'-' * 20}  {'-' * 14}")
    for steps in STEPS_SWEEP:
        print(f"{steps:>14}  {f32[steps]:>20.3e}  {f64[steps]:>14.3e}")


if __name__ == "__main__":
    main()
