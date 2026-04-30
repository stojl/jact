# InitialDistribution design reflections

Notes toward a revised `InitialDistribution` specification. Follow-up
to `docs/api_spec.md` and in the same scratchpad style as `solver.md`.
Not a spec — a design discussion.

**No active users yet.** Breaking changes are on the table.

---

## 1. Where we are today

The current spec carves out four routes into the solver:

1. `initial="healthy"` — all individuals in one state at `d_0 = 0`.
2. `initial="healthy", initial_duration=d_0` — single state, per-individual `d_0`.
3. `initial=state_array` (shape `(batch,)`) — per-individual initial state.
4. `initial=InitialDistribution(components={...})` — full control.

Routes 1–3 are lifted into an `InitialDistribution` at `solve()`-entry.
Then `solve()` calls `Model.reduce(initial_states)`, where
`initial_states` is the set of keys of
`InitialDistribution.components`, and threads the reduced subgraph
through the midpoint scan.

The spec already commits to an invariant:

> The set of initial states is a **structural** property of the
> `InitialDistribution`, fixed at object-construction … and *not*
> inferred at solver-runtime from mass values.

and flags `per_individual` as the awkward case:

> `per_individual` requires `state_array` to be a **concrete
> (host-side) array at construction time**, not a traced JAX value.
> The constructor inspects the array in pure Python to extract the
> unique state set … Calling `per_individual` from inside the user's
> own `jax.jit` / `vmap` is therefore unsupported.

## 2. What's fragile, and what isn't

Let's separate the questions, because it's easy to over-correct.

### 2.1 Reduction is fine — it's an optimisation, not a contract

`Model.reduce` cuts the reachable subgraph from a given initial-state
set and hands the solver a smaller problem. Semantically equivalent to
solving over the full model; just cheaper. Done once, in pure Python,
before tracing. No JIT issue.

It is *tempting* to promote `ReducedModel` to a mandatory step in the
user's pipeline on the grounds that "it forces the structural decision
to be explicit". That's over-engineering. The reduction is a compute
saving; the structural decision that matters — *which states are
initial* — doesn't need a new object to live in. It can live on the
distribution, where it already does.

The primary flow stays:

```python
dist   = jact.InitialDistribution(components={...})
result = model.solve(dist, horizon=..., steps_per_unit=..., **kwargs)
```

`solve()` does the reduction internally, invisibly, as it already does.

### 2.2 The `(mass, duration) per component` primary constructor is fine

```python
dist = jact.InitialDistribution(
    components={
        "healthy":  {"mass": p_h,     "duration": d_h},
        "disabled": {"mass": 1 - p_h, "duration": d_d},
    },
)
```

The keys of `components` are *written by the user*. They are
structural, static, and declared. `mass` and `duration` are traced. No
JIT issue. The "Static-topology invariant" subsection in the current
spec is really defending *this* construction — and it doesn't need
defending, because the user wrote the keys.

### 2.3 The `per_individual` path is where the design breaks

```python
dist = jact.InitialDistribution.per_individual(
    states=state_array,       # (batch,) of names or indices
    duration=d_0_array,
)
```

Under the current spec:
- `state_array` must be host-side concrete (not a `Tracer`).
- The constructor runs `np.unique(state_array)` to discover the
  initial-state set.
- The set becomes the static key set of `components`, encoded as
  one-hot weights per individual.

Three failure modes:

**It's a JIT trap.** `model.solve(initial=state_array, ...)` *looks*
like "just data". A user who wraps `solve` in their own `jit`, or
calls it inside a `vmap` / `grad`, hits a `TracerArrayConversionError`
or similar as soon as the array becomes a `Tracer`. The spec documents
this, but the surface invites the mistake.

**The static topology is data-dependent.** Two batches with different
unique-state sets — one `{healthy}`, one `{healthy, disabled}` —
reduce to different subgraphs, produce different pytree carries, and
re-trace the whole scan. Silent re-compilation on data is the opposite
of the trace-stability guarantee the rest of the API gives.

**"Forgot to include a state" is invisible.** If the batch happens to
omit `disabled`, the distribution's key set is `{healthy}`, the
reduction excludes `disabled` entirely, and the next batch with one
disabled individual re-traces. The user never asked for this
variability.

### 2.4 Seeding the point mass is fine, once the set is pinned

Given a static set and per-component `(mass, duration)`:

```
for j in range(K):                              # K static
    m_j  = dist.mass[j]                         # (batch,) traced
    d0_j = dist.duration[j]                     # (batch,) traced
    # one-hot over (batch, D) at the slot closest to d0_j * steps_per_unit,
    # scaled by m_j — fully traced
    state[j].point_mass = seed(m_j, d0_j)
```

No host-side work. Static pieces are `K` and `D`, both known before
tracing. This also composes with `solver.md §3`'s future point-mass
split — if `point_mass` eventually becomes a per-individual scalar
evolving along the characteristic `(s, d_0 + s)`, seeding is literally
`point_mass = mass`.

Seeding is not the problem. **The problem is that the "known set"
under `per_individual` is derived from runtime data instead of
declared by the user.**

## 3. The minimal fix

Require the user to declare the set on `per_individual`, and drop the
corresponding `solve()` shortcut.

### 3.1 `per_individual` takes an optional static set; default is "all states"

```python
# Default — no reduction. Indices refer to the model's full state list.
dist = jact.InitialDistribution.per_individual(
    states=idx_array,           # (batch,) int32, values in [0, n_states) — TRACED
    duration=d_0_array,         # scalar or (batch,)
)

# Opt-in — declare a smaller set to unlock the reduction optimisation.
dist = jact.InitialDistribution.per_individual(
    initial_states=("healthy", "disabled"),  # static tuple
    states=idx_array,                        # (batch,) int32, values in [0, K)
    duration=d_0_array,
)
```

- `initial_states` is optional. When provided, it's a Python tuple of
  state names — static, user-declared — and `states` indexes into it.
  When omitted, the sentinel `None` means "the full state list of the
  model at solve-time"; `states` indexes into the model's native
  state ordering, and `solve()` skips the reduction step (every state
  is treated as potentially initial).
- `states` is always a traceable integer array. Can be a `Tracer`.
  The only thing that varies is *what it indexes into*.
- Internal seeding is `one_hot(states, K) * mass_scalar` times
  `one_hot(duration_slot, D)` — all traced, all inside `jit`. `K` is
  static either way: either the length of the user's tuple or the
  length of the model's state list.
- JIT cache keys on the set. A distribution with `initial_states=None`
  passed to two models with different state counts re-traces — same
  rule as today, and the user asked for it by opting out of
  declaration.

Rationale: reduction is a compute optimisation, not a structural
contract. Users who don't care about reduction shouldn't be made to
think about initial-state sets. Users who do care (large state spaces,
repeated solves from a small origin set) declare `initial_states` and
pay one line for the win.

Users with a `(batch,)` array of state *names* convert to indices
host-side before calling `per_individual`. The `StateSpace` already
exposes `state_index(name)` for this; a one-line list comprehension
plus `jnp.array` does the job. No dedicated `by_name` constructor
— the conversion is cheap, explicit, and avoids a second public entry
point whose only purpose is to do a dict lookup.

### 3.2 The `initial=state_array` shortcut on `solve()` stays

With the default of §3.1 — "omit `initial_states` ⇒ full model state
list, no reduction" — the `(batch,)`-array shortcut no longer has a
data-dependent topology problem. There's nothing to infer: the set
is just `model.states`, known statically from the `Model`.

```python
# Integer indices into model.states — fully traceable.
model.solve(initial=idx_array, initial_duration=d_0, ...)
#   ⟶ dist = InitialDistribution.per_individual(states=idx_array, duration=d_0)
#      model.solve(initial=dist, ...)
```

The integer-array form is jit-clean and is the only per-individual
shortcut on `solve()`. Users with a `(batch,)` array of state names
convert to indices themselves using `state_space.state_index(...)` —
one line, explicit, no hidden Python introspection inside `solve()`.
`initial="healthy"` stays as sugar for the single-state case because
a single name is a compile-time constant, not array data — no
introspection required.

Users who want the reduction optimisation — a small initial-state
set carved out of a large state space — fall back to constructing the
distribution explicitly:

```python
dist = jact.InitialDistribution.per_individual(
    initial_states=("healthy", "disabled"),
    states=idx_array,                         # values in [0, 2)
    duration=d_0_array,
)
model.solve(initial=dist, horizon=..., **covariates)
```

One extra line, in exchange for a smaller compiled graph. Good
tradeoff — and explicitly opt-in, which is the whole point.

### 3.3 Sugar that can stay

```python
model.solve(initial="healthy", ...)                     # sugar: InitialDistribution.at("healthy")
model.solve(initial="healthy", initial_duration=d_0, ...) # sugar: InitialDistribution.at("healthy", duration=d_0)
model.solve(initial=dist, ...)                          # explicit — covers all other cases
```

Both sugar paths are pure-Python prologues: the state name is a
compile-time constant, so the key set `{"healthy"}` is trivially
static. `initial_duration` is the only kwarg left on `solve()` and
survives in the sugar path only (single state, obvious).

## 4. Shape of the revised surface

| Current                                   | Revised                                                                 |
|-------------------------------------------|-------------------------------------------------------------------------|
| `initial="healthy"`                       | same                                                                    |
| `initial="healthy", initial_duration=d0`  | same                                                                    |
| `initial=idx_array` (int indices)         | **kept** — lifts to `per_individual(states=idx_array, duration=d0)`; no reduction, fully traceable |
| `initial=name_array` (host-side names)    | **removed** — users convert names to indices with `state_space.state_index(...)` and use the integer-array shortcut |
| `initial=InitialDistribution(...)`        | same (primary full-control entry; also the opt-in reduction path)       |
| `InitialDistribution(components={...})`   | same                                                                    |
| `InitialDistribution.at(state, duration)` | same                                                                    |
| `InitialDistribution.per_individual(states=host_array, duration=...)` | `per_individual(states=idx_array, duration=..., initial_states=...)` — indices traced (integer indices only), `initial_states` optional (default: full model state list, no reduction) |

Nothing else in the spec moves. `Model.reduce` stays as the power-user
API for people who want to hold onto a reduced model across many
solves; `solve()` still calls it internally.

## 5. StateSpace / Model / ReducedModel

Three-layer decomposition as-is:

- `StateSpace` — topology; pure data.
- `Model` — `StateSpace` + intensity callables.
- `ReducedModel` — `Model` scoped to a reachable subgraph from a
  given initial-state set.

The decomposition is correct. Each layer adds exactly one kind of
information, and each is stable on the JIT boundary in the ways we
already want:

| Layer          | Adds                                                | Static on JIT boundary?                      |
|----------------|-----------------------------------------------------|----------------------------------------------|
| `StateSpace`   | states, transitions                                  | pure pre-trace data                          |
| `Model`        | intensity callables per transition                   | sparsity pattern + callables are static      |
| `ReducedModel` | initial-state set, reachable subgraph, solver matrix | subgraph + initial-set are static            |

`ReducedModel` is a valid user-visible construct for power users who
want to:
- inspect the reduced subgraph (`reduced.reachable_states`);
- solve repeatedly with different distributions over the same initial
  set (cache the reduction work);
- hand the reduction to downstream code that needs the structure.

But it should not be mandatory. The ergonomic default — "I have a
`Model` and a distribution; give me probabilities" — is the common
path and it should stay a single call.

Nothing about `InitialDistribution` needs to be *bound* to either
`Model` or `ReducedModel`. Keep the distribution state-space-agnostic
(matches the current spec's open design note). Validation against the
model's state space happens at `solve()`-entry; it's cheap and
localised.

## 6. What the spec should say

Concrete edits to `docs/api_spec.md`:

1. **`per_individual` signature** — add `initial_states` as an
   **optional** static tuple (default `None` ≡ "use the model's full
   state list at solve-time, skip reduction"); change `states` to
   accept a traced integer array indexing into either the tuple or
   the full model state list depending on whether `initial_states` is
   supplied. Drop the host-side name-array path. Users converting
   names to indices do so with `state_space.state_index(...)` before
   calling, at either the distribution or `solve()` layer.
2. **`solve()` shortcut table** — keep `initial=idx_array` (integer
   indices only). Clarify that it's traceable and lifts to
   `per_individual(...)` with no declared `initial_states` (full state
   list, no reduction). Drop the `initial=name_array` row; users
   convert names to indices themselves.
3. **"Static-topology invariant" subsection** — shrinks to one
   paragraph: "the initial-state set is a structural field of the
   distribution, declared by the user — either the keys of
   `components`, the single state passed to `at`, or the
   `initial_states` tuple passed to `per_individual`. When
   `per_individual` omits `initial_states`, the set defaults to the
   model's full state list at solve-time (no reduction). In every
   case the set is static on the JIT boundary; mass and duration
   values are traced."
4. **"Open design note"** — replace with a note that `InitialDistribution`
   stays state-space-agnostic deliberately, and that its key set is
   always user-declared and never data-derived. Remove the
   "state_space.initial_distribution(...)" binding-constructor
   musings.
5. **JIT-boundary table** — no change; it already lists the set as
   static and mass/duration as traced. Clarify under the table that
   "set" means "user-declared on the distribution".
6. **Examples** — update the per-individual-state example to show
   two flavours: the jit-clean integer-array shortcut
   (`initial=idx_array`) and the explicit opt-in-to-reduction form
   (`InitialDistribution.per_individual(initial_states=..., ...)`).

No changes to `StateSpace`, `Model.build`, `Model.reduce`, `solve()`'s
numerics, callbacks, intensity protocol, or solver state. The fix is
entirely inside `InitialDistribution` and one row of the `solve()`
shortcut table.

## 7. My take

Three claims:

1. **Reduction is an optimisation; the initial-state set is a
   declaration.** Conflating them — promoting `ReducedModel` to a
   mandatory ceremony just to anchor the set — pays ergonomic cost for
   structural clarity that the distribution can already carry. Keep
   reduction invisible; make the set explicit.

2. **The current design is 95% right.** The primary constructor
   (`components={...}`) is already user-declared and JIT-clean. The
   `at(...)` shortcut is fine. The only actual bug is `per_individual`
   inferring the set from data. Fix that specifically; leave the rest
   alone.

3. **Keep the public surface small.** `per_individual` takes integer
   indices only; no name-array convenience on either the constructor
   or `solve()`. Users with name arrays convert via
   `state_space.state_index(...)` — one line, explicit, and it makes
   the name→index step visible at the call site where it belongs.

### Counter-argument worth taking seriously

"Won't dropping reduction-by-default regress performance for users
who were relying on the current implicit reduction under
`initial=state_array`?"

Yes, slightly — with the default of §3.1, `initial=idx_array`
compiles to a graph over the full state space, not the unique-set
subgraph the current spec would produce. Two responses:

- Reduction is only a worthwhile optimisation when a large fraction
  of states are excluded. For moderate-size state spaces the
  overhead is small and the jit-cleanliness gain is large.
- Users who need the optimisation get it with one declared line:
  `per_individual(initial_states=(...), states=idx, duration=d0)`.
  Opt-in, obvious, re-trace-stable.

The current implicit behaviour optimises silently at the cost of
data-dependent topology. The revised behaviour doesn't optimise
silently but remains easy to opt into. That's the right default.

### Order of work (for a follow-up spec revision)

1. Update `InitialDistribution.per_individual` signature in the spec:
   add optional `initial_states` (default `None` ≡ full model state
   list, no reduction); retype `states` as traced integer indices.
   Drop the name-array flow entirely.
2. Update the `solve()` shortcut table: `initial=idx_array` stays as
   traceable sugar; `initial=name_array` is removed.
3. Rewrite the "Static-topology invariant" and "Open design note"
   subsections as described in §6.
4. Update the per-individual example block.
5. Cross-reference from `notes/design/solver.md §3` so the
   point-mass-split design note matches the new seeding story.
