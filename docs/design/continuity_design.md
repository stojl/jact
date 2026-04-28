# ADR: midpoint-only quadrature

## Status

Accepted.

## Context

Earlier versions of `jact` supported two quadrature paths for each assigned intensity:

- midpoint along the transported characteristic,
- endpoint Heun/trapezoid for callables labeled continuous in both time and duration.

That design required continuity metadata on assigned callables and additional solver machinery so mixed models could use different quadrature rules in the same scan. In practice this created two problems:

1. fitted hazards from tree-based or otherwise irregular models rarely come with trustworthy continuity guarantees,
2. the solver and public API became more complex than the value of the extra branch justified.

## Decision

`jact` now uses midpoint quadrature for every transition intensity and every point-mass update.

The public callable interface stays minimal:

```python
fn(t, d, **kwargs) -> array
```

There is no solver-facing interface for declaring continuity classes or jump locations.

## Rationale

- Midpoint is already second-order for smooth hazards.
- Midpoint is robust for irregular fitted hazards where continuity metadata would be guesswork.
- Wrong continuity metadata was a silent failure mode that could make results worse rather than better.
- Removing mixed-rule support deletes cache reuse plumbing, per-transition quadrature branching, and related API surface.

## Consequences

- The solver is simpler and the public API is smaller.
- Users no longer need to wrap intensities in metadata objects.
- Hazards with jumps strictly inside traversed cells can still lose order. This is documented as part of the solver contract.
- For tree-based or other piecewise hazards, users should align split points in time and duration to the solver grid when possible.

## Historical evidence

The design-study notebook [convergence_notebook.ipynb](../convergence_notebook.ipynb) preserves the midpoint vs trapezoid comparison that motivated this decision. The trapezoid curves in that notebook are now local reference calculations, not active `jact` solver modes.
