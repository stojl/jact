# Point-mass evolution under heterogeneous starts: implementation reflections

This note is the downstream companion to [docs/implementation/initial_distribution.md](/home/lucas/Documents/jact/docs/implementation/initial_distribution.md). Once a canonicalised `InitialDistribution` reaches the solver, how should the evolution machinery handle heterogeneous starting conditions? It assumes the reader has read [api_spec.md](/home/lucas/Documents/jact/docs/api_spec.md) §Solver state and §Interaction with the solver state, and [design/solver.md](/home/lucas/Documents/jact/docs/design/solver.md) §3. The *why* (the point mass is a scalar 1-D problem along its characteristic) is argued there. This note is the *how*.

## Scope

Three orthogonal axes of heterogeneity reach the solver through `_CanonicalDistribution`:

- **Multiple declared initial states** — `canonical.states` can have length > 1.
- **Per-individual `d_0` within a state** — `canonical.durations[i]` can be `(batch,)`, and off-grid values must be preserved exactly.
- **Per-individual mass within a state** — `canonical.masses[i]` can be `(batch,)`.

These axes are independent. The spec allows any combination. The solver must handle all of them without loss.

## What the current solver already gets right

Two of the three axes are handled correctly today by the existing grid-based representation:

- `solve()` in `jact/solver.py` allocates one `StateCarry.point_mass` per declared initial state, and leaves `point_mass = None` for every reachable non-declared state. Multi-state heterogeneity is already structural.
- `_compute_derivative` iterates `(i, j)` over reachable states. Its `if carry_i.point_mass is not None` branch adds per-source point-mass contributions to `next_inflow[j]`. Multiple simultaneous point-mass sources compose additively, which is mathematically correct.
- Point-mass outflow feeds the *target density* (slot 0), not any target point mass. This matches the physical picture: a Dirac is a property of where an individual started, not of whatever state they transition into. Targets that are themselves declared initial retain their own point-mass slot; arriving mass joins the density.
- `_update_point_mass` is per-state and per-slot. Independent per-state evolution is therefore automatic once the seeding is in place.

Per-individual mass and multi-state heterogeneity are mechanically fine. The sore spot is elsewhere.

## The bug: grid-snapping of `d_0`

`_seed_point_mass` in `jact/solver.py` rounds every per-individual duration to the nearest grid slot:

```python
indices = jnp.rint(duration_arr * steps_per_unit).astype(jnp.int32)
indices = jnp.clip(indices, 0, max(duration_slots - 1, 0))
return jax.nn.one_hot(indices, duration_slots, ...) * mass_arr[:, None]
```

This silently collapses `d_0` onto the duration grid. At `steps_per_unit = 12` the snap can move `d_0` by up to ~0.042 time units per individual.

**This violates the spec.** Both [api_spec.md](/home/lucas/Documents/jact/docs/api_spec.md) §Solver state and §Interaction with the solver state promise that per-individual `d_0` need not land on the grid, precisely because the point mass is meant to be a 1-D scalar problem along its characteristic — not a grid-bound object co-evolved with the density.

The downstream effects of the snap:

- Every intensity evaluation against that point mass happens at `d_0_snapped + k * step_size`, not `d_0 + k * step_size`. The seed error propagates for the life of the point mass.
- The direction of the bias is whatever direction the hazard is monotone over the ulp-scale neighbourhood of the snapped duration. For steep hazards near `d_0`, the error is not negligible.
- Memory is wasted: `(batch, D)` per declared initial state, of which at most one slot per individual is non-zero at `t = 0` — and, by construction, the non-zero support never spreads.

Fixing the bug is representational. Keeping the `(batch, D)` scatter and trying to "interpolate better at seed time" does not work: the subsequent grid-based evolution reads whatever slot was seeded, so any seed-time interpolation drifts the point mass off a single slot and immediately "diffuses" the Dirac into its neighbours — the exact outcome the point-mass/density split is there to prevent.

## The deliberate `μ⁺` / `μ_avg` asymmetry

Before switching the representation, it's worth naming an asymmetry in `_compute_derivative` that looks like a bug until you understand it:

- `mu_plus_slice = mu_plus[..., :-1]` — one-sided, used for the point-mass inflow to the target state.
- `mu_avg = 0.5 * (mu_plus[..., :-1] + mu_minus[..., 1:])` — the two-sided Heun duration corrector, used for the density's own decay and for density-to-density inflow.

The asymmetry is deliberate. Averaging across neighbouring duration slots smears a Dirac; the density is smooth and benefits from the corrector. Under the scalar point-mass representation described below, the whole question dissolves: there is a single duration `d_0 + t` per individual, and you evaluate the intensity there once.

## The scalar per-individual representation

Since snapping is off the table, the representation has to become:

```
StateCarry:
    density: (batch, D)
    point_mass: PointMass | None
        value: (batch,)      # current mass
        d_0:   (batch,)      # constant per individual, set at seeding
```

With `t` supplied by the scan, the characteristic `(t, d_0 + t)` is reconstructed arithmetically per step. No duration axis on the point mass, no scatter, no rounding.

Concretely:

- **Seeding** stores `mass_arr` and `duration_arr` verbatim. There is nothing to interpolate, round, or one-hot.
- **Point-mass decay** multiplies `value` by the per-step survival factor along the characteristic — the cumulative-hazard integral over `[t, t + step_size]` evaluated at per-individual duration `d_0 + s`. Any integrator suffices here. A Heun-consistent two-evaluation scheme keeps the order of the density intact; an analytic exponential of the integrated hazard is strictly better when the closed form is available.
- **Density feed-in from point mass** evaluates the outgoing hazard at per-individual `d = d_0 + t`, multiplies by `value`, and drops the resulting per-individual scalar into `next_inflow[j]`, just as today. The `next_inflow[j]` plumbing is unchanged — only its source term differs.
- **Absorbing-boundary behaviour** of the `D − 1` slot is a density concept only. The point mass never accumulates at a duration boundary because it has no duration axis to run off the end of. Mass leaves only via hazard, not via the grid.

The factorisation keeps the point mass and density mathematically aligned with their respective physics: scalar exponential decay along a characteristic for the point mass, advection-reaction for the density.

## Intensity evaluation at per-individual durations

The pinch point is that the current callable contract returns `(batch, D)` from a duration grid `d: (1, D)`. Evaluating at per-individual `d_0 + t` needs one of:

- **Diagonal extract.** Pass `d: (1, batch)` and keep the diagonal of the `(batch, batch)` return. Zero protocol change, `O(batch²)` compute per call per intensity. Only viable if the point-mass path is a small fraction of solver cost — at the target batch sizes (100k+) it is not.
- **`jax.vmap` over the batch.** Wrap the existing callable with `vmap(..., in_axes=(None, None, 0, ...))` and pass `d: (1, 1)` per individual. No user-facing protocol change; relies on XLA to fuse the per-individual evaluation back into the natural batched kernel. Lowest-risk prototype.
- **Scalar-mode protocol.** Add a second callable signature `fn(t, d, **kwargs) -> (batch,)` for the per-individual path, distinct from the grid-mode signature. Cleanest semantics, largest surface-area change — every user-written intensity grows a second code path.

The `vmap` route should land first on the basis of "no protocol change, measure, then decide". If fusion is poor in practice, fall back to the scalar-mode protocol.

## Callback contract changes

Callbacks currently receive `point_mass: (batch, D) | None`. They have to change:

- `"default"` surfaces the full pytree. After the switch it surfaces `PointMass | None` per state (or a flattened `{value, d_0}` leaf, whichever the pytree registration prefers).
- `"no_duration"` currently returns `point_mass[..., -1]` per state — mass still concentrated at the terminal slot, which under the point-mass-at-boundary interpretation was always close to zero. Its replacement is simpler and cleaner: return `value` (scalar per individual).
- `"collapse_point"` and `"collapse_point_no_duration"` fold the point mass into `density[..., 0]`. The fold is no longer a one-hot scatter at a rounded index. It becomes a per-individual `density[..., 0] + value`, with the duration information kept in `PointMass.d_0` and discarded by the collapse. A helper that reconstitutes a `(batch, D)` density with the point mass deposited at `round(d_0 * steps_per_unit)` is still useful for downstream plotting, but it is a user-facing convenience, not part of the solver-internal fold.
- `"point_only"` and `"point_only_no_duration"` return the `PointMass` object (or `value`) per state.
- `"no_point"` and `"no_point_no_duration"` are unaffected.

This is a breaking change to the callback contract, and it is necessary. [design/solver.md](/home/lucas/Documents/jact/docs/design/solver.md) §4 already flags a coordinated callback-contract break when the solver state moves toward a pytree-of-per-state-tensors; this is the same break. Landing both together avoids paying the break twice.

## JIT boundary

The static fields are unchanged: the declared initial-state set remains the topological input. The traced fields shift from `point_mass: (batch, D)` to a `PointMass` pytree with `value: (batch,)` and `d_0: (batch,)`. Presence/absence of a point mass per state stays static. Re-trace triggers are unchanged.

The JIT-boundary table in [api_spec.md](/home/lucas/Documents/jact/docs/api_spec.md) §Solver should be updated in the same change that ships the representation — the row naming `point_mass` shape is what moves.

## Recommended ordering

1. **This note first**, used as the contract for the refactor.
2. Register `PointMass` as a pytree node behind `StateCarry.point_mass`.
3. Rewrite `_seed_point_mass` to store `value`, `d_0` verbatim.
4. Rewrite `_update_point_mass` and the point-mass branch of `_compute_derivative` in terms of per-individual scalar evolution. Prototype the `vmap` intensity path; benchmark against the diagonal-extract fallback.
5. Update the built-in callbacks and callback tests to the new contract.
6. Update [api_spec.md](/home/lucas/Documents/jact/docs/api_spec.md) §Solver state, §Callbacks, §JIT boundary in lockstep.
7. Backfill tests for heterogeneous off-grid `d_0`:
   - single-state, off-grid `d_0`, analytic reference (one transition with closed-form survival);
   - multi-state declared components with mixed on- and off-grid `d_0`;
   - mass conservation with a mixture of zero-mass and positive-mass rows;
   - agreement with the pre-refactor implementation on grid-aligned `d_0`.

## Open spec questions

These are worth raising when the representation lands, not now:

- After the switch, what does `"default"` return for a state with no point mass — `None` (today's convention) or a sentinel `PointMass(value=0, d_0=0)`? The former keeps the current pytree shape; the latter is easier to consume uniformly at the cost of a sentinel that can be misread as "zero-mass point mass at duration zero".
- Does `"no_duration"` keep its name once the point-mass path has no duration axis to marginalise over? A rename would be clearer but costs an additional breaking point; probably not worth it.
- Does the spec want to name the `vmap`-vs-scalar-mode intensity decision, or leave it as an implementation detail? The intensity contract belongs in the spec; an internal evaluation strategy does not. Only the decision itself needs surfacing if and only if it leaks into user-written callables.

## Summary

The current implementation handles multi-state and per-individual-mass heterogeneity correctly. It does not handle off-grid `d_0`: `_seed_point_mass` snaps to the grid, contradicting the spec. The fix is representational — store the point mass as a per-individual scalar with its own `d_0`, evaluate the intensity at `d_0 + t` per individual, and route the inflow to target densities as today. The callback contract changes in consequence, and the change should be coordinated with the pytree move already flagged in the design notes.
