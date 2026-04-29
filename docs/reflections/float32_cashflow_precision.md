# Float32 cashflow precision: where the error actually comes from

## 2026-04-29 investigation update

The original reproduction was point-mass dominated, not duration-density
dominated. `initial="alive"` with per-individual `initial_duration` seeds the
alive state as a `PointMass`; the alive duration density is zero throughout the
survival-model annuity, so the wide
`jnp.sum(density_midpoint * payment, axis=-1)` path is not the source of the
observed `steps_per_unit=1024` spike in that case.

The spike is reproduced by the scalar point-mass recurrence alone. The solver
was repeatedly multiplying the point value by a rounded float32 survival factor
near one. At `steps_per_unit=1024`, the rounded `exp(-rate * dt)` factor has
enough negative bias that thousands of updates overwhelm the midpoint
quadrature error.

Implemented fix: carry point-mass survival internally as `log_value`, update it
additively by subtracting the per-step integrated point hazard, and materialise
the public `.value` with `exp(log_value)`. This preserves callback-visible
point-mass values while avoiding repeated near-one multiplication in the carry.

Post-fix sweep for the original reproduction:

```
steps_per_unit   err float32   err float64
            32     1.144e-05     1.101e-05
            64     3.099e-06     2.752e-06
           128     0.000e+00     6.879e-07
           256     4.768e-07     1.720e-07
           512     1.907e-06     4.299e-08
          1024     2.146e-06     1.075e-08
          2048     3.338e-06     2.687e-09
          4096     1.860e-05     6.718e-10
          8192     2.694e-05     1.679e-10
```

The wide density reduction may still deserve separate diagnostics for models
that actually put material mass on the duration grid, but it is not the cause
of this specific reproduction.

### Density-advection follow-up

A private density-only probe seeded one unit of mass directly into the alive
duration grid, with no point mass, then valued a constant state-rate annuity
under the same constant hazard. This isolates `_advance_density`'s repeated
`density * survival` update:

```
steps_per_unit   density-path err float32
            32                 3.815e-06
            64                 4.768e-07
           128                 0.000e+00
           256                 1.192e-07
           512                 1.192e-07
          1024                 4.315e-05
          2048                 2.229e-05
          4096                 1.109e-05
          8192                 5.245e-06
```

So the same near-one survival-factor rounding issue does exist for material
density advection. A scalar recurrence matching the solver's
`1 - (-expm1(-h))` survival update reproduces the spike, while a log-survival
prototype (`log_density -= h`, materialise with `exp(log_density)`) reduces
the `steps_per_unit=1024` error to roughly `1e-7` in this isolated case.

Conclusion: the float32 convergence failure is not unique to point masses.
Point masses exposed it first because the original reproduction never put mass
on the duration grid, but density advection can hit the same coherent rounding
bias whenever material mass is repeatedly transported by near-one survival
factors.

The density fix is more invasive than the point-mass fix because density has
additive inflow and last-bin pooling. A robust implementation should either
carry internal `log_density` with `logaddexp` at merge points, or apply a
narrower log-survival correction along pure advection characteristics while
keeping transition inflow reductions in linear space.

## Symptom

On float32 inputs (the common GPU case, where float64 throughput is poor), the
terminal-cashflow analytical-comparison error gets *worse* past a certain
`steps_per_unit`, instead of continuing to converge as `O(1/N²)`.

Reproduction script: `scripts/cashflow_precision_sweep.py` — sweeps the
constant-intensity time/duration annuity from
`tests/test_cashflows.py::test_constant_intensity_time_duration_state_rate_matches_closed_form`
across `steps_per_unit ∈ {32, …, 8192}`, in both float32 and float64.

```
steps_per_unit   err float32   err float64
            32     1.264e-05     1.101e-05
            64     5.960e-06     2.752e-06
           128     1.431e-06     6.879e-07
           256     4.768e-07     1.720e-07   ← float32 floor
           512     2.623e-06     4.299e-08
          1024     1.018e-04     1.075e-08   ← float32 spike (~100× worse)
          2048     4.935e-05     2.687e-09
          4096     1.550e-05     6.722e-10
          8192     4.411e-05     1.669e-10
```

float64 stays on the textbook midpoint slope. float32 is clean up to ~256
steps, spikes hard at 1024, then drifts noisily. The non-monotone spike at
N=1024 is reproducible across runs — this is not random noise.

## What this is *not*

Initial hypothesis: terminal `+=` accumulation across all `n_steps` carries
recurrent rounding error, fixable with Kahan / Neumaier compensated summation
in the terminal carry inside `_midpoint_solver`.

This hypothesis is **wrong**. Two diagnostics in `scripts/` rule it out:

`scripts/cashflow_precision_diagnose.py` runs the same model with both a
`terminal=True` view (in-solver scan accumulator) and a streamed view, and
host-side-sums the streamed `(T_out, batch)` array independently. Result:
identical error to the in-solver terminal accumulator at every `steps_per_unit`.
The two paths have entirely different summation structures; only the per-step
values they share are the same.

`scripts/cashflow_precision_step.py` takes those per-step streamed values and
sums them four ways: naïve sequential f32, JAX pairwise (`jnp.sum`), full
Python Neumaier in f32, and upcast-to-f64-then-sum. At `steps=1024`:

```
naïve sequential f32:  err 9.68e-05
pairwise (jnp.sum):    err 9.80e-05
Neumaier f32:          err 9.80e-05
upcast to f64 + sum:   err 9.80e-05
```

All four agree. **Even summing in float64 cannot recover the precision.** The
information is gone before the values reach the accumulator. Compensated
summation in the terminal carry, or anywhere else *downstream* of the per-step
emit, can therefore never help — there is nothing left to compensate.

A Neumaier prototype was implemented in `_midpoint_solver` (compensated
`(value, comp)` carry, branchless `jnp.where` Neumaier add, materialised at
solve-end) and confirmed to produce float32 errors within rounding noise of
the unmodified main branch. The prototype was reverted.

## Where the error actually originates

The per-step contribution for a `StateRate` cashflow is, schematically:

```python
contribution = step_size * jnp.sum(density_midpoint * payment, axis=-1)
```

— summing over the duration grid of length `D = horizon * steps_per_unit`. At
`steps_per_unit=1024`, that is a 2048-wide reduction.

Two suspects, both upstream of the terminal accumulator:

1. **The inner `jnp.sum(... , axis=-1)`**: a wide reduction in float32 over a
   density vector that is mostly near-zero (only the first ~`step_index` slots
   are populated; the rest are exponentially tiny tails or zeros). Pairwise
   summation on a vector with that magnitude profile loses precision through
   cancellation noise.

2. **The density advection itself**: the `(batch, D)` density is updated each
   step by an exponential survival factor; over thousands of float32
   multiplications the density carries cumulative rounding bias.

The N=1024 spike in particular looks like a discretisation/representation
artefact of `D` rather than smooth accumulation drift — accumulation drift
would be monotone in N, but the error *recovers* somewhat at N=2048 and
beyond. This points at the inner reduction (whose nature changes with `D`)
rather than at advection (whose drift would only grow with N).

## What to do — float64 upcast is undesirable on GPU

Global `jax_enable_x64=True` collapses GPU throughput on consumer cards by
roughly 32×, and even targeted local upcasts inside the solver hot path
(`density.astype(jnp.float64)` for the inner sum, then back) trade away the
main reason for staying in float32. Worth keeping in the toolbox as a
documented fallback for accuracy-critical runs, but not the default.

Float32-friendly options, ordered by expected impact and surgicality:

### A. Tighten the inner reduction

Replace the implicit pairwise `jnp.sum(density * payment, axis=-1)` inside
`_compute_cashflow_step` with one of:

- **Pre-multiplication scaling**: factor out the largest payment magnitude
  before the sum, restore after. Reduces dynamic range seen by the reduction.
- **Sparse reduction over the populated prefix**: only the first `step_index`
  slots carry mass — the rest are zero or denormal-tail. Slicing or masking
  the reduction to the populated prefix avoids summing many noise-floor terms
  into the result. The slice length depends on the step counter and is
  per-step static if expressed as a `jnp.where(arange < k, density, 0.0)`,
  but the rounding profile is *the same* unless we also reorder. The actual
  win comes from **sorting by magnitude before summing** (smallest first) or
  using `jnp.cumsum` and reading out the running total — both reduce
  accumulation noise without changing dtype.
- **Block-Kahan reduction**: sum in groups of B (e.g. 32 or 64) with Neumaier
  per-block, then combine block totals. JAX-friendly via reshape-and-reduce.

### B. Stabilise the density advection

If diagnostic (2) above is the real culprit, no amount of inner-sum work will
help. Probe via a separate sweep that reads the density at `t = horizon`
directly (e.g. through `result["probability"]` with a no-op cashflow) and
checks whether `1 - exp(-rate * horizon)` is recovered to float32 ulp. If the
density itself has drifted, the fix has to live in `_advance_solver_step_from_dynamics`
(consider compensated multiplication of the survival factor along the
characteristic, or reformulating the advection as a log-space update where
multiplications become additions amenable to compensated summation).

### C. Document the precision floor

Even with (A) and (B), float32 has a hard ulp floor on GPU. Document, in
`docs/api_spec.md` under the existing "Memory budget" section or a new
"Precision" section, the empirical guidance:

- For float32 GPU runs, error stops improving past a model-specific
  `steps_per_unit` (~256 for the basic survival model; varies with hazard
  scale and `D`).
- Increasing `steps_per_unit` past that floor *costs* precision rather than
  buying it.
- Users who need both speed (float32) and convergence past that floor should
  reach for a future hardened solver path; users who can tolerate the
  throughput hit should run float64 for accuracy-critical valuations.

## Recommendation

Investigate (A) first — implement a magnitude-sorted or block-Kahan
reduction in `_compute_cashflow_step`'s inner sum, validated against the
sweep script. If the spike at N=1024 collapses, ship that. If it does not,
move to (B). Either way (C) is worth doing in parallel — even an ideal
float32 solver has a precision floor and users should know where it sits.
