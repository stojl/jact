"""Inspect per-step contribution magnitudes and compare summation strategies.

If pairwise summation (jnp.sum, which is tree-reduced) and naive sequential
summation give different results, the per-step values themselves have enough
spread that summation order matters and compensated summation upstream might
help. If they all agree, the per-step values are systematically biased and
the fix has to be in the per-step computation.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import jact


RATE = 0.25
HORIZON = 2
BASE = 1.5
TIME_COEF = 0.4
DURATION_COEF = 0.2


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


def neumaier_sum(arr_1d):
    s = np.float32(0.0)
    c = np.float32(0.0)
    for x in arr_1d:
        x = np.float32(x)
        t = s + x
        if abs(s) >= abs(x):
            c = c + ((s - t) + x)
        else:
            c = c + ((x - t) + s)
        s = t
    return s + c


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

    for steps in (256, 1024, 4096):
        result = model.solve(
            initial="alive",
            initial_duration=initial_duration,
            horizon=HORIZON,
            steps_per_unit=steps,
            probability=None,
            cashflows=cashflows,
            cashflow_views={"annuity": jact.Raw("annuity")},
            age=jnp.arange(2.0, dtype=jnp.float32),
        )
        stream = np.asarray(result["cashflows"]["annuity"])  # (T_out, batch)
        per_step = stream[:, 0]  # individual 0
        exp_0 = float(expected[0])

        sum_naive = np.float32(0.0)
        for v in per_step:
            sum_naive = np.float32(sum_naive + v)
        sum_pairwise = float(np.sum(per_step))
        sum_neumaier = float(neumaier_sum(per_step))
        sum_f64 = float(np.sum(per_step.astype(np.float64)))

        print(f"\nsteps={steps}, expected[0]={exp_0:.10f}")
        print(f"  naive seq f32 sum:  {float(sum_naive):.10f}  err {abs(float(sum_naive)-exp_0):.3e}")
        print(f"  pairwise   f32 sum: {sum_pairwise:.10f}  err {abs(sum_pairwise-exp_0):.3e}")
        print(f"  Neumaier   f32 sum: {sum_neumaier:.10f}  err {abs(sum_neumaier-exp_0):.3e}")
        print(f"  upcast     f64 sum: {sum_f64:.10f}  err {abs(sum_f64-exp_0):.3e}")
        print(f"  per_step min/max/mean: {per_step.min():.3e} / {per_step.max():.3e} / {per_step.mean():.3e}")
        print(f"  n_steps={len(per_step)}")


if __name__ == "__main__":
    main()
