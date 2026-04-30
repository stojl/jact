# Solver design reflections

> Historical note: this document discusses an earlier continuity-aware / perturbation-based solver design. The current implementation uses midpoint quadrature only and the normative contract lives in `docs/api_spec.md`.

Notes toward a solver specification. These reflect on the concerns
raised in conversation plus a few cross-cutting observations. Not a
spec — a scratchpad for the design discussion.

**No active users yet.** `jact` has no downstream consumers at this
point, so breaking changes are on the table everywhere in this
document. Where a section weighs "breaking-change cost vs. cleaner
design", the breaking-change cost should be treated as low. This is
the moment to get the shape right, not after an ecosystem forms.

---

## 1. Discontinuities along the grid

The solver is intended to handle **càdlàg** intensities (right-continuous,
with left limits). This should be stated explicitly in the spec — it
sets the conventions for every other decision below (what "evaluate at
a jump" means, which limit counts, what the user is promising about
their callable).

### The current situation

`_heun_solver` evaluates every intensity at perturbed points per scan
step, nudging both clock time `t` and duration grid `d` by
`± ε` (`ε = 1e-12` by default). The intent is to read the intensity
from "inside" the continuous pieces on either side of a jump.

### Why I think this is fragile

**Absolute perturbation doesn't scale with the argument.** In IEEE
float64, the ulp at `x = 1.0` is ~`2.2e-16`; at `x = 30.0` it's
~`3.6e-15`; at `x = 100.0` it's ~`1.4e-14`. So `1e-12` is still
resolvable over a 30-year horizon — *just barely*, and any arithmetic
downstream (addition with `attained_age`, `exp`, `log`) erodes those
last few bits quickly. For larger timescales the perturbation is
already about an order of magnitude above the ulp, leaving almost no
margin. In float32 this collapses entirely.

**The perturbation is invisible to the user.** A user writes
`jnp.where(d < tau, a, b)` in their intensity. The solver evaluates
at `tau ± 1e-12`. Whether the two evaluations straddle `tau` depends
on how `tau` was computed, how `d` was accumulated inside the
callable, and float rounding. Nothing gives the user a knob to say
"my jump is at exactly `tau`, please evaluate from both sides
cleanly".

**The correction is a point-evaluation trick in a 2nd-order scheme.**
Heun is 2nd-order *for smooth right-hand sides*. If the intensity has
a finite jump, the local error near the jump is O(step_size) regardless
of how cleverly you sample — you lose an order of convergence through
the jump. Perturbing the evaluation point doesn't fix this; it just
picks a consistent side to be wrong on.

**The "plus" and "minus" evaluations are asymmetric.** `m_p_slice =
μ⁺[:-1]` and `m_avg = 0.5 (μ⁺[:-1] + μ⁻[1:])` use different grid
offsets — `μ⁺` reads "left" durations and `μ⁻` reads "right"
durations, one duration slot apart. This is actually *the Heun
corrector, applied at the intensity level*, and it's a clever reuse.
But `ε` is serving two jobs simultaneously: (a) move off the grid
point to avoid the jump, (b) give the symmetric average for the
2nd-order correction. Those jobs want different scales.

### Options to consider

**(a) Declared break points.** Extend the intensity protocol so a
callable can advertise a sorted array of break times/durations where
it is discontinuous. The solver aligns the grid (or sub-steps the
Heun scheme) to land on the left and right of each break exactly.
This localises the problem: no perturbation outside declared breaks,
clean left/right evaluations at them.

```python
@dataclass
class Intensity:
    fn: Callable
    breaks_t: Sequence[float] = ()   # clock-time jumps
    breaks_d: Sequence[float] = ()   # duration jumps
```

Downside: breaks out of the pure-callable contract. Users who don't
care about jumps still get the simple path.

**(b) Piecewise callables.** The intensity is a list of `(interval,
fn)` pairs. The solver picks the right piece per grid slot. Most
explicit, most intrusive.

**(c) Relative perturbation.** Replace `ε = 1e-12` with `ε = rtol *
(1 + |d|)`. Cheap, no protocol change, but doesn't address the
2nd-order-through-jumps problem and still leaves users without a
reliable way to place jumps on grid points.

**(d) Left/right-evaluation protocol.** Callables opt into a
two-argument interface `fn(t, d, side)` where `side ∈ {left, right}`
is a compile-time constant. Inside, users do their own branching.
The solver requests the side it needs, no perturbation required.
Clean semantics, burden on the user.

**(e) Change the discretisation near breaks.** Leave the protocol as
black-box callables, but compile a schedule of step-halving around
declared break points. Composes with (a).

### My take

I think **(a)** is the right primary direction. Discontinuities are a
first-class modelling concern (benefit entitlement boundaries, policy
changes, waiting periods, age cutoffs) and pretending they can be
handled by a black-box callable plus a magic constant is going to
bite. The API could stay backwards-compatible: callables without a
declared breaks list get treated the same way they are today.

For the spec, whatever we choose, we should document **precisely**
what "evaluate at a grid point where the intensity is discontinuous"
means under the càdlàg convention — it should be the right limit by
default, matching the mathematical convention. The prototype does
something approximately symmetric but it's not spelled out.

---

## 2. Subsampled recording

### The use case

`horizon = 30`, `steps_per_unit = 12` → 360 inner steps. If the user
only needs the probability state at, say, every 12th step (once per
time unit), recording every step wastes memory and I/O, especially
when the callback returns `default` (full `(batch, J, D)` tensor).

The savings are substantial: a `(batch=100_000, J=10, D=360)` tensor
at float32 is ~1.4 GB per recorded step. At 360 steps you can't even
allocate this; at 30 recorded steps it's ~40 MB — tractable.

### Double-scan pattern

The natural JAX idiom:

```python
def outer_step(carry, t_out):
    def inner_step(carry, t_in):
        # one Heun step
        ...
        return next_carry, None  # no per-inner-step recording

    carry, _ = lax.scan(inner_step, carry, inner_times)
    return carry, prob_callback(*carry)  # record once per outer step

_, history = lax.scan(outer_step, init, outer_times)
```

Single XLA program, no per-inner-step history allocation. Maps
directly onto the current solver with minimal restructuring.

### Design considerations

- **API surface.** A single optional `record_every: int` (default 1)
  keeps it simple. Spec'd to require `record_every` divide
  `steps_per_unit * horizon` evenly (otherwise raise).
- **Initial recording.** The current solver prepends the callback
  value at `t=0`. With `record_every`, the recorded times become
  `0, record_every * step_size, ..., horizon`. Length =
  `(solver_steps // record_every) + 1`.
- **Interaction with callbacks.** Transparent — the callback is
  simply called less often.
- **Naming.** `record_every` or `output_stride` both read well. I
  prefer `record_every`.
- **Alternative framing.** Decouple compute resolution
  (`steps_per_unit`) from output resolution (`output_steps_per_unit`).
  More user-friendly, slightly more implementation work.
- **Non-uniform output.** Future extension:
  `output_times: Sequence[float]` for arbitrary sampling points.
  Doesn't fit the double-scan directly.

### My take

Double-scan is the right mechanism. `record_every` is the right
minimal API. Bake it in early — users with large batch sizes will hit
the memory ceiling almost immediately without it. Specify the
divisibility rule: `(horizon * steps_per_unit) % record_every == 0`.

---

## 3. Point mass and heterogeneous starting durations

### Where we are today

The point mass `p_point` is a `(batch, D)` array non-zero only at
duration slot 0 at `t = 0`. It decays via the outflow from state 0,
is rigidly shifted along the duration axis with `p`, and seeds mass
into other states via the `i == 0` special case in `_compute_core`.

The point mass is treated specially precisely to avoid numerically
diffusing a Dirac through a finite-difference scheme.

### The observation

The point mass is *uncoupled* from the absolutely continuous density
`p`. Specifically, `dp_point / dt = -μ_total(t, d) * p_point`, and
since `p_point` lives only along a single moving duration ray
(`d_at_time_t = d_0 + t`), this is a **scalar ODE per individual**
with analytic solution:

```
p_point(t) = exp(-∫₀ᵗ μ_total(s, d_0 + s) ds)
```

So you could:

1. Precompute `p_point` along the characteristic `s → (s, d_0 + s)`
   for each individual (1-D quadrature, vectorised).
2. Use its derivative as a known source term when advancing `p`.

### Why this matters for heterogeneous `d_0`

The current design assumes `d_0 = 0` for every individual. If each
individual starts with their own duration offset `d_0_b`, the 2-D
grid approach breaks down: `p_point` no longer sits on the same
duration slot across the batch. You'd have to keep the point mass in
a staggered representation per individual, or pay the cost of
carrying a full `(batch, D)` representation just to encode "a point
mass at slot `k_b`".

The split approach handles this cleanly:

- **Phase 1 (per-individual 1-D solve for point mass):** each
  individual has its own `d_0_b`. Scalar ODE along the
  characteristic in `t`. Embarrassingly parallel over the batch, no
  duration grid, arbitrary starting durations with no special
  machinery.
- **Phase 2 (absolutely continuous density):** source term for `p`
  is now a known function of `t` and `b`, namely the decay rate of
  `p_point_b`. The 2-D solver runs for `p` only, driven by inflows
  from other states and by this precomputed source.

### Benefits

1. **Arbitrary `d_0` per individual** with no representational
   awkwardness.
2. **Accuracy.** `p_point` no longer sits in a finite-difference
   scheme; it's analytic or high-order 1-D quadrature. The "Dirac
   diffusion" problem disappears.
3. **Simpler solver core.** `_compute_core` loses the `i == 0`
   special case. The inflow term becomes uniform across sources.
4. **Composability.** Phase 1 is a natural place to hook things
   like "person has been in state X for `d_0` already and is
   observed at age `a_0`" — common in longitudinal data.

### Costs

1. **Coupling back to phase 2.** The phase-1 output feeds into
   phase 2 as a source term, which means phase 2 needs the
   time-history of the point mass (or its derivative). Cheap
   `(batch, solver_steps)` array, but new tensor in the carry.
2. **Two passes.** Two scans (or one scan that carries both
   phases). More code to type-check.
3. **Point masses in non-initial states.** A user might later want
   to start a fraction of the batch in state X. Conceptually the
   same problem on a different row — the uniformity "point mass is
   always tied to state 0" needs generalising.
4. **Quadrature choice for phase 1.** Involves a time integral of
   total exit intensity. If intensities are càdlàg (see §1),
   phase-1 quadrature needs to know about breaks too. So the
   discontinuity story becomes *more* important once we split.

### My take

The split is the right long-term shape. Even setting aside
heterogeneous `d_0`, it's a cleaner factorisation:

- The point mass's physics (scalar exponential decay along a
  characteristic) is fundamentally different from the density's
  physics (2-D advection-reaction with duration shift). Treating
  them the same is a historical artifact.
- Carrying the `i == 0` special case through the derivative
  computation is the tell that the current model is fusing two
  different objects.

Shape the spec so that the point mass is a first-class, separately
handled object, and the absolutely continuous density is a second,
coupled-only-via-source-term object. That keeps the door open for
heterogeneous starting durations, multiple initial states with their
own point masses, and analytic / high-order integration for the
point mass.

Open API question: how do we expose `d_0`? Options: optional
`initial_duration` kwarg on `solve` (scalar or `(batch,)`); a richer
`initial_distribution` object encoding state + duration + mass per
individual. I'd lean toward the richer object eventually, but start
with the kwarg.

Implementation-side reflections on how the solver should evolve point
masses once heterogeneous starts reach it (off-grid per-individual
`d_0`, multi-state declarations, per-individual mass) live in
[notes/implementation/point_mass.md](../implementation/point_mass.md).

---

## 4. Solver state structure: one big tensor vs. list-of-tensors

### Where we are today

Two concatenated tensors in the scan carry:

- `p`: `(batch, J, D)` — every state's density stacked along axis 1.
- `p_point`: `(batch, D)` — attached to state 0.

### Why this works well

- **One allocation.** XLA gets a single contiguous buffer per
  object.
- **Single vmap over batch.** `_compute_derivative` does one vmap
  that sees both the density and the intensity matrix uniformly.
- **Uniform broadcast ops.** `-p * outflow_avg` is one array op,
  not a per-state loop.
- **JAX-friendly.** Everything fuses into a tidy XLA program.

### Why it's questionable

- **Same duration depth `D` for every state.** A Markov
  (duration-independent) state carries a full `(batch, D)` slice
  that is completely redundant — every duration slot evolves
  identically after mass distributes. You pay the full cost of the
  semi-Markov machinery even for states that have no duration
  dependence.
- **Same duration depth for states with different natural scales.**
  If `disabled → dead` depends on duration since disability but
  `healthy → dead` doesn't depend on duration at all, we still
  allocate `D` slots for both. For large `J`, large `D`, and a few
  genuinely Markov states, this is pure waste.
- **Callback uniformity.** Every callback sees `(batch, J, D)` and
  has to know to treat state 0 specially for the point mass. There
  is no clean way for a callback to say "for Markov states, return
  a scalar; for semi-Markov states, return a density".
- **Unreachable states are already pruned**, so the obvious win
  ("don't carry unreachable states") is already captured. The
  remaining question is heterogeneity *within* reachable states.

### List-of-tensors (pytree) alternative

Carry per-state arrays as a tuple/dict:

```python
p: tuple[jnp.ndarray, ...]  # one entry per reachable state
# p[j].shape == (batch, D_j)
```

where `D_j` is the duration depth required by state `j` (Markov →
`D_j = 1` literally; semi-Markov → whatever resolution is needed).
`jax.lax.scan` accepts pytree carries natively, so this is
mechanically fine.

The `_compute_core` loops are *already* Python for-loops over `(i,
j)` that build lists of arrays and only `jnp.stack` them at the end.
Shifting to per-state arrays is mostly deleting the final stack —
the accumulation pattern is already there.

**Pros:**

- Each state sized to its own needs. Markov states are genuinely
  free (`D_j = 1`).
- Mixed Markov / semi-Markov models become natural and efficient.
- Memory scales with the union of per-state needs, not the
  worst-case `J * D`.
- If we go ahead with the point-mass split (§3), the point mass
  naturally becomes a per-state field in the struct rather than a
  special-cased parallel tensor.
- Opens the door to per-state duration grid *resolution*, not just
  depth — useful when different states have different natural
  timescales (e.g. very short acute phases vs. long chronic
  phases). Probably not a v1 feature, but free in this
  representation.

**Cons:**

- **More static structure in the trace.** Per-state `D_j` is
  static, so every change to the per-state depths re-traces. Same
  category as sparsity pattern re-traces today; not a regression,
  but the surface area grows.
- **Callback contract changes.** Callbacks now receive a pytree
  rather than a 3-D tensor. Built-in callbacks need rewriting;
  user-written callbacks need updating. This is a breaking change
  if we've already shipped.
- **No single stacked `(batch, J, D)` for callbacks that want
  one.** A callback that wants a uniform 3-D view has to pad itself
  — cheap but awkward.
- **Inflow bookkeeping.** Cross-state inflow (mass moving from
  state `i` to state `j`) deposits into `p[j][..., 0]`. Already how
  `_compute_core` works — each `next_inflow[j]` becomes the new
  slot-0 value for state `j`. So no real change, just different
  slicing syntax.
- **XLA fusion.** Per-state ops might fuse less effectively than
  one stacked op. In practice XLA fuses pytree operations pretty
  well, and the `_compute_core` inner pattern is already per-state
  — I don't expect a meaningful slowdown, but it's worth measuring
  before committing.

### Hybrid: tuple of uniformly-shaped tensors

Keep everything `(batch, D)` but make the top-level a tuple/list
over states rather than a leading axis. This gives:

- Same memory as today (no per-state depth savings).
- Easier path to per-state point masses (§3).
- Decouples "number of states" from "a tensor axis" — the solver
  stops caring about `J` as a shape, treating it purely as
  structure.
- Callback contract still changes (pytree instead of 3-D tensor),
  so the breaking-change cost is paid either way.

This is a useful stepping stone: it captures the *structural*
benefits of list-of-tensors without committing to heterogeneous
`D_j` until we have a use case.

### My take

List-of-tensors is probably the right long-term shape, for two
reasons independent of the Markov/semi-Markov size asymmetry:

1. **It aligns the data structure with the mathematics.** In the
   actual PDE, each state has its own duration density; they
   communicate via inflow scalars. Treating them as slices of a
   stacked tensor is a pun that's convenient until you want
   heterogeneity (per-state durations, per-state point masses,
   per-state resolutions) — then it becomes a straitjacket.
2. **It composes with every other improvement we're
   considering.** Point-mass split (§3): each state can have its
   own point mass. Discontinuity protocol (§1): break-point
   schedules are per-intensity, which means per-cell, which means
   they compose naturally with per-state data. Subsampled
   recording (§2): unaffected. The list-of-tensors change unlocks
   the others.

Pragmatically, the order of work probably wants to be:

1. Commit to the hybrid (tuple-of-`(batch, D)`-tensors) — same
   memory, cleaner structure, breaking-change for callbacks paid
   once.
2. Layer the point-mass split on top (each tuple entry gets a
   density and a point mass, or None for the point mass if the
   state never carries one).
3. Allow per-state `D_j` as an optimisation — only when someone
   needs it.

The cost is the callback-contract break. If we're going to break
it, better now (before the ecosystem around jact grows) than
later.

Counter-argument worth taking seriously: "single big tensor is
simple, fast, and the current solver demonstrates it's already
competitive. Everything else is YAGNI." That's fair. If we adopt
list-of-tensors, the spec should name the concrete wins it's
unlocking (heterogeneous `d_0`, per-state point masses, Markov
states at `D=1`) and be honest that without those wins the change
is pure churn.

---

## 5. Axis ordering of the probability state

### Where we are today

Inside the scan:

- `p`: `(batch, state, duration)` — the carry.
- `p_point`: `(batch, duration)` — the carry.

`jax.lax.scan` stacks each callback output along a new leading
axis, so after the scan the raw history has time as axis 0:

- `(time, batch, state, duration)` for the `default` callback's
  density leaf.
- `(time, batch, state)` for `collapse_point_no_duration`.
- `(time, batch)` for `point_only_no_duration`.

Then `_transpose_probability` does a rank-dependent shuffle:

```python
if N == 1: return x                                    # time only
if N == 2: return jnp.transpose(x, (1, 0))             # (time, batch) → (batch, time)
return jnp.moveaxis(x, 0, -2)                          # (T, B, J, D) → (B, J, T, D)
```

So the final user-visible shape for the `default` callback is
`(batch, state, time, duration)`.

### Is this ordering natural?

It's *a* reasonable choice but not an obviously natural one. The
right axis order depends on what you do with the result, and there
are three competing patterns:

1. **Per-individual trajectory plotting.** "Show me the history of
   state probabilities for individual `b`." You want `arr[b]` to
   give the full time × state × duration slab. The current ordering
   is good for this: batch first, time buried inside.
2. **Time-snapshot analysis.** "Give me the state distribution at
   time `t`, across the whole batch." You want `arr[..., t, ...]`
   to be cheap. Time-as-axis-0 is best; time-as-axis-2 is OK with
   ellipsis indexing.
3. **Time reductions.** "Integrate `discount(t) * P(state=j, t)`
   over time." You want the time axis to be contiguous for
   reductions. Last-axis reductions are cheapest in C-order;
   leading-axis reductions in F-order.

The current choice privileges pattern (1). Actuarial/statistical
workflows more often want (2) or (3). There's no obviously
correct answer, but there is a question worth surfacing in the
spec: *what are we optimising for?*

### Compute performance

Short answer: the transpose itself is almost certainly free, but
the rank-dependent logic is a liability.

**The transpose is (nearly) free at the XLA level.** `jnp.moveaxis`
and `jnp.transpose` are logical ops; XLA decides the physical
layout of the output buffer based on downstream consumers. Within a
single `jax.jit`'d program the transpose typically becomes a
layout-constraint hint and emits no kernel. At the program boundary
— when the array leaves the XLA world (e.g. converted to NumPy for
plotting) — a physical transpose may happen, but that cost is paid
once, outside the scan, and is linear in the array size. For a
`(batch, J, T, D)` result at `batch=100_000`, `J=3`, `T=360`,
`D=360`, a single float32 transpose is ~150 GB/s on modern DRAM →
~hundreds of ms. Non-zero but not in the hot path.

**The carry layout is what actually matters for performance, and
it's already right.** Inside the scan:

- `jax.vmap(_compute_core, in_axes=(0, 0, ...))` wants the batch
  axis at position 0.
- Duration-axis shifts (`_update_p`, `_update_p_point`) use `...,
  1:` slicing — they want duration as the last (innermost) axis.
- The per-state Python loops in `_compute_core` don't care about
  array layout; they index along the state axis by position.

So `(batch, state, duration)` for the carry is the right choice
and has nothing to do with the final user-visible ordering.

**The rank-dependent transpose is the real concern.** The
`_transpose_probability` function behaves differently for rank 1,
2, and ≥3, applied per-leaf of the callback output pytree. This
works by accident for the built-in callbacks but will silently
misbehave for user callbacks returning unusual shapes. For
example, a user callback returning a scalar per step `()` becomes
`(T,)` and is left alone (correct by luck); a user callback
returning `(T,)` per step would become `(T, T)` and get swapped
(probably wrong). Rank-dispatch is a fragile API surface.

### Alternatives

**(A) Time as axis 0, no transpose.**

Drop `_transpose_probability` entirely. The user-visible shape
matches the scan-native shape:

- `default`: `(time, batch, state, duration)` and `(time, batch,
  duration)`.
- `collapse_point_no_duration`: `(time, batch, state)`.
- `no_point_no_duration`: `(time, batch, state)`.

Pros:
- Simplest code; no rank-dependent logic.
- Matches JAX convention (scan, `diffrax`, most RNN outputs all
  put time first).
- Callbacks with arbitrary output shapes just work — time is
  always prepended.
- Zero-cost at compile time *and* avoids the end-of-program
  transpose on big tensors.

Cons:
- Per-individual plotting requires `arr[:, b, ...]` instead of
  `arr[b]`. Fine idiomatically, slightly less pleasant.

**(B) Time as the last axis.**

- `default`: `(batch, state, duration, time)`.
- Natural for time reductions (discounted sums).
- Requires a post-scan transpose (not free at the program
  boundary, but still a one-shot cost).
- Less conventional in JAX.

**(C) Time as axis 1 (after batch).**

- `default`: `(batch, time, state, duration)`.
- Common RNN/sequence convention.
- Still requires a post-scan transpose.
- Per-individual access: `arr[b]` gives `(time, state, duration)`,
  which is natural.

**(D) Current ordering, but make the transpose non-dispatching.**

Keep `(batch, state, time, duration)` but stop using rank
dispatch. Instead, either:

- Carry static metadata per leaf saying which axis is time
  (controlled by the callback), then moveaxis per leaf. Most
  flexible.
- Declare that time is always axis 0 for user callbacks and only
  the built-in callbacks opt into the "nice" transposed shape.
- Make `transpose_result` a per-leaf spec
  (e.g. `{"probability": 2}` to move time to axis 2 for that
  leaf only), defaulting to `0` (no transpose).

**(E) Always time-as-axis-0, let the user opt into a transpose.**

Return the scan-native shape; expose a utility like
`jact.move_time_axis(result, to=-2)` for users who want the old
behaviour. Keep the solver itself rank-agnostic.

### Interaction with §4 (list-of-tensors)

If the solver state becomes a pytree over states, the "state"
axis is *in the tree structure*, not a tensor axis. Per-state
tensors become `(batch, duration)` and the scan output per state
is `(time, batch, duration)`. That collapses the problem: there is
no "where does state go" question, just "where does time go",
which is much cleaner.

Under that representation, options (A), (B), (C) become:

- (A) `(time, batch, duration)` per state.
- (B) `(batch, duration, time)` per state.
- (C) `(batch, time, duration)` per state.

All three are cleaner than the current four-axis tangle. (A) is
still free; (B) and (C) need one transpose per state but the
transposes are per-state and small.

### My take

Three claims:

1. **The transpose is almost certainly not a compute bottleneck.**
   XLA handles logical transposes well. If the transpose is
   costing measurable time, we should benchmark first — intuition
   here is unreliable.

2. **The rank-dependent transpose is the real problem, and it's
   an API problem, not a performance problem.** It breaks the
   callback contract for user callbacks with unusual output shapes
   and it's genuinely hard to explain in a spec. "Time becomes
   axis 0 after the scan, then we moveaxis it based on rank" is
   not something I'd want to write in a docstring.

3. **Picking time-as-axis-0 and leaving it alone is the cleanest
   spec choice.** It matches JAX conventions, has no rank
   dispatch, survives the pytree transition in §4 unchanged, and
   gives users a single invariant ("time is always the leading
   axis of every callback output leaf"). Users who want a
   different layout get it with one explicit `moveaxis` call. The
   solver stays dumb; the convention is clear.

If we really want the current `(batch, state, time, duration)`
for the default callback, I'd express it as a property of the
*callback*, not of the solver. The built-in
`collapse_point_no_duration` could explicitly emit a
`(batch, state)` snapshot per step with declared semantics, and a
post-processing step stacks those into `(batch, state, time)`.
That pushes the axis-order decision into the place that knows
what the output means.

The counter-argument: "users are already used to `(batch, state,
time, duration)`, breaking it is churn". Same breaking-change
question as §4. If we're going to pay the cost there, paying it
here too is a small marginal increase. If we're not, the
rank-dependent transpose at least deserves documentation that
calls out its fragility.

---

## Cross-cutting observations

### The duration grid coincides with the clock-time grid

`D = horizon * steps_per_unit` and the duration grid is literally
the same `linspace(0, horizon, D+1)` that drives the scan. This is
mathematically consistent (duration advances at rate 1 with clock
time) but welds two concerns:

- Time resolution (how finely we integrate).
- Maximum duration trackable (bounded by `horizon`).

For long-horizon solves of models whose intensity flattens after
some duration threshold, you're carrying an enormous duration axis
for no benefit. Spec should call out that duration resolution and
time resolution are not independent today — and the
list-of-tensors option (§4) is the natural place to break the
coupling later.

### Static intensity-matrix structure

Sparsity pattern is part of the JIT trace. Every model rebuild
re-traces. Probably unavoidable given the sparsity-in-trace
benefit, but the spec should call it out so users aren't
surprised by compile-time dominance during experimentation.

### The `μ⁺ / μ⁻` split is doing work that isn't named

`m_p_slice = μ⁺[:-1]` and `m_avg = 0.5(μ⁺[:-1] + μ⁻[1:])` is
really "evaluate at duration `d_k` from the left interval endpoint"
and "average with duration `d_{k+1}` from the right interval
endpoint". Under a proper càdlàg protocol (§1) these would be
named by what they mean. Right now it looks like a numerical trick
when it's actually the heart of the duration-average that makes
the method 2nd-order.

### Callback output shape coupling to time axis

`_transpose_probability` is rank-dependent (1-D unchanged, 2-D
swapped, higher moved from axis 0 to `-2`). Works for current
callbacks but is fragile for user callbacks returning unusual
trees. Spec should either document transpose semantics per-leaf
explicitly or simplify (always leave time as the leading axis, let
users transpose downstream).

---

## Things I think should go in the spec (non-exhaustive)

1. **Càdlàg intensity convention.** State explicitly that
   intensities are assumed càdlàg (right-continuous with left
   limits). At a jump, evaluation returns the right limit by
   default. This frames every subsequent decision.

2. **Discontinuity handling.** Contract for what evaluating an
   intensity "at a jump" means. Either a declared-breakpoints
   extension to the protocol, or a documented guarantee about
   left/right limits, with a user-tunable perturbation scale as
   fallback.

3. **Output stride.** `record_every` (or equivalent).
   Divisibility rule. Shape of output leading axis. How it
   composes with `transpose_result`.

4. **Initial conditions.** Formal definition of the initial
   state (point mass at `d=0` by default), with a forward-looking
   hook for per-individual `d_0` and/or a richer initial
   distribution object. The initial state is always at reduced
   index 0.

5. **Point mass and density as separate objects.** Even if not
   implemented immediately, name them as conceptually separate
   objects that happen to be co-evolved. Puts heterogeneous-`d_0`
   and analytic-point-mass improvements inside the spec envelope.

6. **Perturbation semantics.** What `perturbation` means today,
   why it's there, and that it will be replaced/obsoleted by the
   discontinuity protocol. Default value. Precision requirements
   (float64 recommended; float32 caveats).

7. **Solver state structure.** Decide and document: single big
   tensor, tuple-of-per-state-tensors, or pytree with per-state
   `D_j`. Lock the callback contract accordingly. Breaking
   changes here are easier to do once, early.

8. **Callback shape contract.** Signature (what the callback
   receives), output pytree, transpose semantics. Depends on (7).

9. **Reduction to reachable subgraph.** Already implemented.
   Spec should make clear that reduced-index-0 is always the
   initial state — load-bearing for the point mass and for future
   heterogeneous-`d_0` work.

10. **JIT boundary.** What's static, what's traced, what
    triggers a re-compile. Especially: matrix sparsity structure
    (static), callback function (static), per-state duration
    depths if adopted (static), kwargs names and shapes (traced
    for values, static for tree structure).

11. **Numerical order through discontinuities.** Honest about
    Heun being 2nd-order away from jumps, at best 1st-order
    across them. If (§1) is resolved with declared breakpoints +
    sub-stepping, we recover 2nd-order everywhere.

12. **Memory budget.** Given `batch`, `J` (and per-state `D_j`
    if we go that way), `record_every`, and callback choice, an
    explicit formula for peak output memory. Users will hit OOM
    if this isn't spelled out.
