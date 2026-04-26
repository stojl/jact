# Point-mass evolution under heterogeneous starts: implementation reflections

This note is the downstream companion to [docs/implementation/initial_distribution.md](/home/lucas/Documents/jact/docs/implementation/initial_distribution.md). Once a canonicalised `InitialDistribution` reaches the solver, the initial condition is represented as one optional point mass per declared initial state plus one duration density per reachable state.

The public contract lives in [docs/api_spec.md](/home/lucas/Documents/jact/docs/api_spec.md). This note records the implementation shape and the consequences for callbacks and frozen-source mode.

## Solver-facing representation

Per reachable state, the solver carries:

```python
StateCarry(
    density: (batch, D),
    point_mass: PointMass | None,
)

PointMass(
    value: (batch,),
    d_0: (batch,),
)
```

This representation is deliberate:

- `density` is the absolutely continuous part on the solver grid,
- `point_mass` is the per-individual Dirac part seeded from the initial condition,
- off-grid starting durations are preserved exactly in `PointMass.d_0`,
- target states receive inflow into `density`, not into a new target point mass.

The split avoids diffusing a Dirac through the finite-difference scheme and keeps off-grid `d_0` exact.

## Default point-mass evolution

With `freeze_initial=False`, point masses evolve along the characteristic `(t, d_0 + t)`:

- outgoing hazards are evaluated at the midpoint sample `(t + dt / 2, d_0 + t + dt / 2)`,
- the point-mass value decays by the same competing-risks rule used for density survival,
- outgoing mass is routed into the target state's density inflow for duration zero.

This is the ordinary “initial point mass that then evolves” interpretation.

## Frozen-source mode

With `freeze_initial=True`, the same `PointMass` object is reused but with different step semantics:

- hazards are evaluated at fixed duration `d_0`,
- `PointMass.value` is left unchanged for the full solve,
- outgoing mass is still transferred into target densities every step.

So frozen mode is a persistent source, not a stationary-distribution solver and not a display-only overlay.

Two consequences matter:

- callbacks that include point mass continue to show the pinned source automatically,
- total reported mass can exceed 1 because the frozen source is exogenous and keeps feeding the dynamic density.

## Callback implications

The built-in callbacks already align with this representation:

- `"default"` returns the full `StateCarry` pytree,
- `"point_only"` and `"point_only_no_duration"` expose the `PointMass` or its `value`,
- `"collapse_point"` and `"collapse_point_no_duration"` add `point_mass.value` to the reported state total,
- `"no_point"` and `"no_point_no_duration"` ignore point mass entirely.

Frozen-source mode does not require separate callback variants because the same `PointMass` surface is preserved.

## Implementation touchpoints

The solver behavior is concentrated in `jact/solver.py`:

- seeding stores `PointMass(value, d_0)` verbatim,
- density hazards are always evaluated on the transported duration grid,
- point-mass hazards switch between the dynamic characteristic and fixed-`d_0` mode based on `freeze_initial`,
- the point-mass survival update is either ordinary exponential decay or the identity map in frozen mode.

The callback layer in `jact/callbacks.py` stays representation-level and does not need special-case knowledge beyond reading `StateCarry.point_mass`.
