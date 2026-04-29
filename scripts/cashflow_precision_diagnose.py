"""Diagnose where the float32 error comes from.

Same setup as cashflow_precision_sweep.py but compares three accumulation paths:
  1. terminal=True (in-solver accumulator)
  2. streamed, summed outside the solver in float32 (host-side jnp.sum)
  3. streamed, summed outside the solver in float64 (host-side, after upcast)

If (1) ~ (2) and both >> (3), the error is in the per-step contributions
themselves (density advection / midpoint quadrature in float32), not in the
terminal accumulator. If (1) >> (2), the in-solver accumulator is at fault.
After the solver's survival-advection fixes, this script is also useful for
checking whether terminal scan summation is a meaningful remaining error source.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import jact


RATE = 0.25
HORIZON = 2
BASE = 1.5
TIME_COEF = 0.4
DURATION_COEF = 0.2
STEPS_SWEEP = (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)


def constant_intensity(rate, batch):
    def fn(t, d, **kwargs):
        size = kwargs["age"].shape[0] if "age" in kwargs else batch
        return jnp.full((size, d.shape[-1]), rate, dtype=jnp.float32)

    return fn


def time_duration_payment(base, time_coef, duration_coef, batch):
    def fn(t, d, **kwargs):
        size = kwargs["age"].shape[0] if "age" in kwargs else batch
        level = base + time_coef * t + duration_coef * d
        return jnp.broadcast_to(jnp.asarray(level, dtype=jnp.float32), (size, d.shape[-1]))

    return fn


def main():
    initial_duration = jnp.array([0.0, 0.75], dtype=jnp.float32)
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

    print(f"{'steps':>6}  {'terminal':>12}  {'stream_f32':>12}  {'stream_f64':>12}")
    print(f"{'-'*6}  {'-'*12}  {'-'*12}  {'-'*12}")
    for steps in STEPS_SWEEP:
        result = model.solve(
            initial="alive",
            initial_duration=initial_duration,
            horizon=HORIZON,
            steps_per_unit=steps,
            probability=None,
            cashflows=cashflows,
            cashflow_views={
                "annuity_term": jact.Raw("annuity", terminal=True),
                "annuity_stream": jact.Raw("annuity"),
            },
            age=jnp.arange(2.0, dtype=jnp.float32),
        )
        terminal_val = result["cashflows"]["annuity_term"]
        stream = result["cashflows"]["annuity_stream"]  # (T_out, batch)
        stream_f32 = jnp.sum(stream, axis=0)
        stream_f64 = jnp.sum(stream.astype(jnp.float64), axis=0).astype(jnp.float32)

        e_term = float(jnp.max(jnp.abs(terminal_val - expected)))
        e_s32 = float(jnp.max(jnp.abs(stream_f32 - expected)))
        e_s64 = float(jnp.max(jnp.abs(stream_f64 - expected)))
        print(f"{steps:>6}  {e_term:>12.3e}  {e_s32:>12.3e}  {e_s64:>12.3e}")


if __name__ == "__main__":
    main()
