# Same-step interval cashflow update scheme

This note records the solver update scheme used for smooth interval
cashflows after the same-step transfer settlement change. The public API
does not change. The change is entirely inside `jact/solver.py`.

The important distinction is:

- `StateRate` and `TransitionLump` are smooth interval cashflows and use
  the intra-step transfer model.
- `ScheduledEvent` and `DurationEvent` remain grid/snapshot events and do
  not use interval quadrature corrections.

## Per-step inputs

For one step `[t, t + dt]`, `_solver_step_dynamics` evaluates transition
intensities once, at the existing midpoint samples:

- density hazards at `(t + dt / 2, duration_mid)`,
- point-mass hazards at `(t + dt / 2, d_0 + t + dt / 2)`.

Those evaluations produce integrated hazards:

```text
H_ij[k] = dt * mu_ij(t + dt / 2, d_mid[k])
H_i[k]  = sum_j H_ij[k]
```

and the competing-risks transfer factor:

```text
F_i[k] = (1 - exp(-H_i[k])) / H_i[k]
```

with the usual `F_i[k] = 1` limit when `H_i[k] == 0`.

No additional transition-intensity evaluations are introduced by the
same-step transfer or cashflow correction.

## Pre-step mass transfer

For each source state `i`, mass present at the start of the step transfers
to each target `j` by:

```text
M_ij = sum_k p_i[k] * H_ij[k] * F_i[k]
```

Point masses use the analogous point hazard and point transfer factor.
The raw inflow to target `j` is:

```text
I_j = sum_i M_ij
```

This is the same raw target inflow used by probability advancement and by
the interval cashflow correction.

## Same-step settlement of target inflow

New inflow enters the target state during the step, not at the next grid
boundary. To avoid first-order lag, the solver settles that inflow over the
latter half-step using the target row's duration-zero integrated hazards.

For a target/intermediate state `j` that can itself exit during the step:

```text
half_total_j    = 0.5 * H_j[0]
half_survival_j = 1 / (1 + half_total_j)
S_j             = I_j * half_survival_j
```

`S_j` is the same-step inflow that survives in state `j` to the end of the
step under the mass-conserving settlement approximation.

For each downstream transition `j -> l`, the same-step chained exit mass is:

```text
C_jl = I_j * 0.5 * H_jl[0] * half_survival_j
```

If `j` has no outgoing density hazards, `S_j = I_j` and there is no chained
exit.

This scheme intentionally reuses `H_j[0]` and `H_jl[0]`. It does not perform
extra quarter-step or endpoint intensity evaluations.

## Density advancement

The existing density shift still handles pre-step mass survival:

```text
p_i_next[k + 1] += p_i[k] * exp(-H_i[k])
```

Same-step surviving target inflow is split between duration bins zero and
one:

```text
p_j_next[0] += 0.5 * S_j
p_j_next[1] += 0.5 * S_j
```

This midpoint split places the newly entered surviving mass at an average
duration of roughly `dt / 2` at the next grid boundary. If there is only one
duration bin, both halves remain in bin zero.

Chained exit mass `C_jl` enters the downstream target `l` at duration zero:

```text
p_l_next[0] += C_jl
```

## StateRate cashflows

The ordinary midpoint contribution from mass present at the start of the
step is unchanged:

```text
dt * sum_k p_i[k] * exp(-0.5 * H_i[k]) * payment_i(t + dt / 2, d_mid[k])
```

The same-step target-state correction adds a latter-half-step payment for
surviving inflow:

```text
0.5 * dt * S_i * payment_i(t + dt / 2, dt / 2)
```

This payment evaluation is only needed for states that can receive
same-step inflow. It is a payment-function evaluation, not a transition
intensity evaluation.

The contribution is routed to:

- the cashflow component,
- the target state `i`,
- kind `state_rate`.

## TransitionLump cashflows

The ordinary transition-lump contribution from mass present at the start of
the step is unchanged:

```text
sum_k p_i[k] * H_ij[k] * F_i[k] * payment_ij(t + dt / 2, d_mid[k])
```

The same-step chained-exit correction adds:

```text
C_ij * payment_ij(t + dt / 2, 0)
```

This payment evaluation is only needed when source state `i` can receive
same-step inflow, because only then can there be a same-step chained exit
from `i`. It is also a payment-function evaluation, not an extra transition
intensity evaluation.

The contribution is routed to:

- the cashflow component,
- the intermediate/source state `i`,
- kind `transition_lump`.

## Event cashflows

Events deliberately stay outside this interval correction.

`ScheduledEvent` is evaluated at the pre-step event time:

```text
payment(t, duration_left)
```

and uses the pre-step state snapshot. Newly transferred same-step mass does
not participate in an event at the beginning of that same step.

`DurationEvent` remains grid/snapshot based. It reads the density at the
snapped duration index and uses the effective snapped duration. It does not
receive a same-step interval correction.

Event contributions continue to flow through the separate event
accumulators so view weights apply at event time rather than midpoint time.

## Accuracy and cost summary

The reason for the scheme is to remove first-order lag in smooth interval
cashflows caused by ignoring mass that enters a state during the same step.
With the same-step terms, `StateRate` on a newly entered state and downstream
`TransitionLump` cashflows follow the same second-order behavior as the
probability update for smooth models.

The cost profile is:

- no public API changes,
- no extra transition-intensity evaluations,
- extra payment-function evaluations only for attached states or transitions
  that can receive same-step inflow by topology,
- unchanged snapshot semantics for `ScheduledEvent` and `DurationEvent`.
