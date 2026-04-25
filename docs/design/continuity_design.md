# Reflections on continuity design

The current perturbation scheme is trying to serve two incompatible
roles:

- resolve what an intensity means at a jump;
- provide second-order quadrature on smooth problems.

That is too fragile a foundation if the package is meant to support both
smooth models (for example neural nets) and genuinely discontinuous
models (for example regression trees). The cleaner design is to make
continuity an explicit property of each assigned intensity callable and
pick the quadrature rule from that information.

## The right granularity

Continuity is not a property of the whole model. It is a property of a
particular assigned callable, and it may differ in clock time `t` and in
duration `d`.

The solver should therefore reason per assigned callable, not globally,
and should distinguish:

- continuity in `t`;
- continuity in `d`.

The minimal metadata I would recommend is:

```python
@dataclass(frozen=True)
class IntensitySpec:
    fn: Callable
    continuity_t: Literal["unknown", "discontinuous", "continuous"] = "unknown"
    continuity_d: Literal["unknown", "discontinuous", "continuous"] = "unknown"
```

with conservative defaults:

- `unknown` is treated like `discontinuous` for solver selection;
- `discontinuous` means jumps may occur, but only on user-aligned grid
  lines;
- `continuous` means the solver may use endpoint-based second-order
  quadrature along the transported characteristic.

This keeps the callable interface itself simple:
`fn(t, d, **kwargs) -> array`.

## Why time and duration continuity are different

The solver does not integrate an arbitrary two-dimensional average of
`mu(t, d)`. It integrates the intensity seen by transported mass along a
characteristic.

For density mass in duration slot `k`, one step follows

\[
d(t) = d_k + (t - t_n), \qquad t \in [t_n, t_{n+1}].
\]

For a point mass it follows

\[
d(t) = d_0 + t.
\]

This is why continuity must be separated by axis:

- discontinuity in `t` can break time quadrature inside a step;
- discontinuity in `d` can break quadrature along the transported
  duration path even if the callable is continuous in `t`.

So a callable that is continuous in time but discontinuous in duration
is **not** eligible for the continuous-path rule. The same holds in the
opposite direction.

## Recommended quadrature policy

For each transition `i -> j`, define the per-step integrated hazard

\[
A_{ij}^{(n)}[k] \approx \int_{t_n}^{t_{n+1}} \mu_{ij}(t, d(t))\,dt.
\]

The recommended rule is:

- if the callable is declared continuous in both `t` and `d`, use the
  endpoint Heun/trapezoidal rule along the characteristic;
- otherwise use the midpoint rule along the characteristic.

### Continuous path: Heun / trapezoidal

If `continuity_t = continuity_d = "continuous"`, approximate

\[
A_{ij}^{(n)}[k]
=
\frac{dt}{2}
\left[
\mu_{ij}(t_n, d_k)
+
\mu_{ij}(t_{n+1}, d_k + dt)
\right].
\]

This is the right choice for smooth hazards:

- second-order on smooth problems;
- close to the current solver structure;
- natural path to adaptive stepping later.

It also admits the useful reuse optimisation: the right-end evaluation
of step `n` is the left-end evaluation of step `n+1`, so after startup a
continuous callable needs only one fresh endpoint evaluation per step.

### Conservative path: midpoint

If either axis is `discontinuous` or `unknown`, approximate

\[
A_{ij}^{(n)}[k]
=
dt \cdot \mu_{ij}(t_n + dt/2, d_k + dt/2).
\]

This is the right rule for grid-aligned jumps:

- it samples strictly inside the half-open cell;
- it never asks what the intensity "at the jump" means;
- it is exact for piecewise-constant aligned hazards;
- it stays second-order as long as the callable is smooth inside each
  traversed cell.

## One shared update rule

The solver should not become "one solver for continuous callables" and
"another solver for discontinuous callables". The quadrature can differ
per transition, but the state update should be shared.

For each source state and duration cell, first compute all transitionwise
integrated hazards `A_ij`, then aggregate

\[
A_i = \sum_j A_{ij}.
\]

Use the shared competing-risks update

\[
S_i = \exp(-A_i)
\]

and

\[
T_{ij} =
\begin{cases}
\dfrac{A_{ij}}{A_i}(1 - S_i), & A_i > 0 \\
0, & A_i = 0.
\end{cases}
\]

Then:

- surviving mass in source state `i` shifts one duration slot to the
  right with factor `S_i`;
- transferred mass to target `j` is injected into duration zero using
  `T_ij`.

This is the key point that makes mixed schemes work. Different exits out
of the same source state may use different quadrature rules, but they
still contribute to one consistent competing-risks update.

## Mixed models

Yes: different schemes can and should be used for different assigned
intensities.

For example, one source state may have:

- a smooth neural-net exit that is continuous in both `t` and `d`, so it
  uses Heun/trapezoidal quadrature;
- a tree-based exit with jumps in duration, so it uses midpoint;
- another exit with a single policy-change jump in time, also using
  midpoint.

That mix is numerically coherent as long as the solver does **not**
apply separate survival factors per transition. All exits from a source
state must first be converted to integrated hazards `A_ij`, then
combined into one source-state survival factor `S_i`.

So the recommended granularity is:

- continuity metadata: per assigned callable;
- quadrature choice: per assigned callable;
- mass update: per source state after aggregating all exits.

## Point masses

The same policy should apply to point masses. The only difference is the
characteristic:

\[
d(t) = d_0 + t.
\]

So for a point mass carried by source state `i`:

- if the callable is continuous in both axes, use endpoint
  Heun/trapezoidal quadrature along `(t, d_0 + t)`;
- otherwise use midpoint quadrature at
  `(t_n + dt/2, d_0 + t_n + dt/2)`.

This keeps density and point-mass handling aligned under the same
continuity policy.

## Convergence statements

The relevant order statements are:

- midpoint is second-order on a step if the callable is smooth on the
  interior of the characteristic segment for that step;
- midpoint remains globally second-order for a callable if every jump in
  either `t` or `d` lies on a grid line, so no traversed step crosses a
  jump;
- midpoint drops to first order for that callable if a jump in either
  variable lies inside a traversed cell;
- endpoint Heun/trapezoidal is second-order only when the callable is
  continuous in both `t` and `d` along the characteristic;
- mixed models remain second-order provided each callable gets the
  quadrature rule implied by its own continuity metadata and all
  discontinuities are grid-aligned.

There is no need to claim anything higher than second order here.

## Practical recommendation

My preferred design is:

1. remove `perturbation` from the core numerical story;
2. let continuity be optional metadata on each assigned callable;
3. distinguish continuity in `t` from continuity in `d`;
4. use Heun/trapezoidal only for callables continuous in both axes;
5. use midpoint for all other callables;
6. unify everything through per-transition integrated hazards and a
   shared competing-risks update.

This gives the package a coherent story for smooth ML models,
discontinuous tree-based models, and mixtures of the two in one reduced
model without overloading the solver with ambiguous boundary semantics.
