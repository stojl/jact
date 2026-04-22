# jact — API spec (short reference)

Condensed mirror of `docs/api_spec.md`. Same normative content — signatures, shapes, validation rules, JIT-boundary tables, invariants — with prose, rationale, and worked examples stripped. For worked examples and design rationale, see the full spec.

## Overview

`jact` is a JAX framework for computing transition probabilities in multi-state models with duration-dependent transition intensities (semi-Markov). Three layers:

- **StateSpace** — topology only (states, allowed transitions); stable, serialisable.
- **Model** — StateSpace + intensity callables; immutable, swappable.
- **Solver** — Heun-scheme numerical engine on the reachable subgraph.

All intensities must be JIT-compatible. The full pipeline covariates → probabilities compiles to one XLA program.

## Module layout

```
jact/
├── __init__.py              # Public API: StateSpace, Model, InitialDistribution, solve, callbacks
├── state_space.py           # StateSpace class + InitialDistribution helpers
├── model.py                 # Model, ReducedModel, TransitionInfo
├── initial_distribution.py  # InitialDistribution class
├── solver.py                # Semi-Markov solver (Heun scheme)
├── intensity/
│   ├── __init__.py
│   ├── parametric.py        # Built-in parametric hazards (future)
│   └── wrappers.py          # Adapters for common model types (future)
└── callbacks.py             # Probability output callbacks
```

---

## StateSpace

```python
state_space = jact.StateSpace(
    states=["healthy", "disabled", "dead"],
    transitions=[("healthy", "disabled"), ("healthy", "dead"), ("disabled", "dead")],
)
```

**Construction-time validation:** no duplicate state names; all referenced states exist; no self-transitions; no duplicate transitions.

**Surface:**

| Name | Description |
|---|---|
| `states` | Tuple of state names |
| `n_states` | `len(states)` |
| `transitions` | Frozenset of `(src, tgt)` |
| `absorbing` | States with no outgoing transitions |
| `transient` | States with outgoing transitions |
| `exits(s)` | Tuple of `(s, tgt)` transitions |
| `targets(s)` | Tuple of target states of `s` |
| `sources(s)` | Tuple of source states to `s` |
| `has_transition(s, t)` | bool |
| `state_index(s)` | int |
| `reachable_from(s)` | BFS; starting state first, then reachable states in original order |

**Serialisation:** `state_space.to_json(path)` / `StateSpace.from_json(path)`.

---

## Model

Built via `state_space.build(transitions=..., exits=..., groups=...)`. Every declared transition must be assigned **exactly once** across the three kwargs; `build()` validates no gaps / no overlaps.

| Kwarg | Coverage | Callable return shape |
|---|---|---|
| `transitions={(src, tgt): fn}` | One transition | `(batch, D)` |
| `exits={src: fn}` | **All** exits from `src`, ordered by `state_space.targets(src)` | `(n_targets, batch, D)` |
| `groups={fn: [(src, tgt), ...]}` | Arbitrary set, in listed order | `(n_transitions, batch, D)` |

`exits` always covers *all* exits — for partial coverage use `groups`. All three can be combined in one `build()` call.

### `Model.reduce(initial_states)`

Accepts a single state name **or** an iterable of state names. Extracts the reachable subgraph (union of reachability from each initial state). Initial states occupy the first `K` reduced indices in state-space ordering; non-initial reachable states follow.

```python
reduced.initial_states     # tuple of declared initial states
reduced.reachable_states   # full reduced state tuple
reduced.n_states
reduced.solver_matrix      # matrix of callables
```

### `model.info(src, tgt) → TransitionInfo`

Returns `TransitionInfo(source, target, assignment, callable, index)`.

---

## InitialDistribution

Encodes the joint `(state, duration)` distribution at `t = 0`, per individual. Three usage patterns: all in one state at `d_0=0`; all in one state with per-individual `d_0`; mixture across states with per-individual `(mass, duration)` per state.

### Construction

```python
# Primary
jact.InitialDistribution(
    components={"healthy": {"mass": mass_h, "duration": d_h},
                "disabled": {"mass": mass_d, "duration": d_d}},
    normalise=True,
)

# All individuals: single state, scalar or (batch,) duration
jact.InitialDistribution.at(state, duration=0.0)

# Per-individual initial state
jact.InitialDistribution.per_individual(
    states=idx_array,          # (batch,) int32, TRACED
    duration=d_0_array,        # scalar or (batch,), optional
    initial_states=None,       # optional static tuple of state names
)
```

`per_individual.states` is a traced `(batch,)` int32 index array:
- `initial_states=<tuple>` → indices into that tuple; solver reduces to reachable subgraph from those states.
- `initial_states=None` → indices into model's full state list; **no reduction** (every state potentially initial).

`per_individual` is index-only; users with a name array convert host-side via `state_space.state_index(...)` or use the `StateSpace.initial_per_individual` helper.

### `solve()` shortcuts

```python
model.solve(initial="healthy", ...)                         # lifted to at()
model.solve(initial="healthy", initial_duration=d_0, ...)
model.solve(initial=idx_array, ...)                         # (batch,) int32 indices into model.states; no reduction
model.solve(initial=idx_array, initial_duration=d_0, ...)
model.solve(initial=jact.InitialDistribution(...), ...)
```

`initial_duration` is valid only with `str` / `(batch,)` forms. Passing it with an `InitialDistribution` raises `ValueError` (duration is already encoded).

### `StateSpace` eager-validation helpers

State-space-agnostic by design; these are ergonomic wrappers that validate names eagerly against `self.states` and return a plain `InitialDistribution`:

- `state_space.initial_at(state, duration=0.0)`
- `state_space.initial_per_individual(state_names=... | state_indices=..., duration=..., initial_states=None)` — exactly one of `state_names` / `state_indices` required; `state_names` lookup happens against `self.states` (or against `initial_states` if given).
- `state_space.initial_distribution(components=..., normalise=True)`

### Validation

At **construction**:
- `mass` and `duration` shape-consistent across components (all scalar, or all `(batch,)` with matching batch).
- `mass >= 0`, `duration >= 0` pointwise.
- If `normalise=True` (default): per-individual `sum(mass) == 1` within `1e-6`; else `ValueError`.
- If `normalise=False`: sum check skipped; output linear in input mass.

At **`solve()`-entry**:
- Every declared state name exists in the model's state space.
- For `per_individual` with explicit `initial_states`: `states ∈ [0, len(initial_states))`; with `initial_states=None`: `states ∈ [0, n_states)`.
- Batch dimension matches covariate batch.

### Static-topology invariant

The initial-state set is a **structural, user-declared** field of the distribution — keys of `components`, the name passed to `at`, or the `initial_states` tuple on `per_individual` (defaulting to the model's full state list when omitted). Static on the JIT boundary; mass/duration arrays are traced.

Declaring a state with all-zero mass still allocates a point-mass slot — the set lives in the declaration, not in the mass values.

### Interaction with solver state

Every state declared in the distribution is seeded with `StateCarry.point_mass` at `t=0` encoding its per-individual `(mass, duration)`. Reachable states *not* declared keep `point_mass = None`. Point mass evolves along characteristic `(s, d_0 + s)` as a per-individual scalar problem, so per-individual `d_0` need **not** land on the duration grid.

---

## Intensity protocol

```python
def intensity(t, d, **kwargs) -> jnp.ndarray: ...
```

| Arg | Type | Meaning |
|---|---|---|
| `t` | scalar float | Clock time, `0 → horizon` |
| `d` | `(1, D)` | Duration grid; entry `k` = `k / steps_per_unit`; leading `1` broadcasts over batch |
| `**kwargs` | `(batch, ...)` | Covariate arrays from `solve()`; unused kwargs ignored |

`t` and `d` play distinct roles: `t` = clock time (attained age = `baseline_age + t`); `d` = duration in current state. Markov uses `t` only; pure duration-dependent uses `d` only; semi-Markov uses both.

**Càdlàg convention:** right-continuous with left limits; at a discontinuity, default is the right limit.

**JAX requirements:** pure (no side effects, no mutation); JIT-compatible (no data-dependent Python control flow, no non-JAX ops); closes over static values only (fitted params become compile-time constants).

### Return shapes by assignment

| Assignment | Shape |
|---|---|
| `transitions` | `(batch, D)` |
| `exits` | `(n_targets, batch, D)` ordered by `state_space.targets(source)` |
| `groups` | `(n_transitions, batch, D)` in listed order |

Solver itself only ever sees `(batch, D)`: `exits` and `groups` are pre-sliced at build time.

### Discontinuity handling (WIP)

Current scheme: evaluate each intensity at `d ± perturbation` and nudge `t ± perturbation` each step (default `1e-12`). Known limitations: absolute ε doesn't scale with argument (collapses in float32); perturbation invisible to user; Heun drops to **1st-order across finite jumps** regardless of `perturbation`.

Long-term options under consideration: (a) declared break points, (b) piecewise callables, (c) relative perturbation, (d) left/right-evaluation protocol, (e) adaptive sub-stepping around break points. Until resolved: treat intensities as smooth, or place jumps well inside `(perturbation, horizon − perturbation)` on cleanly representable values.

---

## Solver

Heun 2nd-order predictor-corrector inside `jax.lax.scan`, vmapped over the batch axis. One XLA program.

### Solver state

Per reachable state:

```python
class StateCarry(NamedTuple):
    density: jnp.ndarray              # (batch, D)
    point_mass: jnp.ndarray | None    # (batch, D) or None
```

- `density[b, k]`: density at batch `b`, duration slot `k` (slot 0 = "entered just now", slot `k` = "entered `k` solver steps ago").
- `point_mass`: `None` for states not declared in `InitialDistribution`; `(batch, D)` for every declared state.

Full solver state is `tuple[StateCarry, ...]` in `reachable_states` order.

**Physics separation:** `density` evolves by advection-reaction with rigid duration shift (slot `k → k+1`); `point_mass` evolves by scalar exponential decay along characteristic `(s, d_0 + s)` — a 1-D problem per individual. Kept separate to avoid diffusing a Dirac through the finite-difference scheme and to allow off-grid per-individual `d_0`.

### `solve()` parameters

| Parameter | Type | Description |
|---|---|---|
| `initial` | `str`, `(batch,)` int32 array, or `InitialDistribution` | Initial condition. `str` = all in this state at `d_0=0`. `(batch,)` = per-individual indices into `model.states` (no reduction). `InitialDistribution` = full control + opt-in reduction. |
| `initial_duration` | `float` or `(batch,)` | Per-individual `d_0` for `str` / `(batch,)` forms. Default `0`. ValueError if passed with `InitialDistribution`. |
| `horizon` | `int` | Time units to solve over. |
| `steps_per_unit` | `int` | Resolution; `D = horizon * steps_per_unit`. |
| `callback` | `str`, callable, or `None` | Default `"collapse_point_no_duration"`. |
| `record_every` | `int` | Record every `N`-th step. Must divide `horizon * steps_per_unit`; else `ValueError`. Default `1`. |
| `perturbation` | `float` | Grid perturbation, default `1e-12`. |
| `**kwargs` | arrays `(batch, ...)` | Covariates. `initial` and `initial_duration` are reserved names. |

### Result

```python
result["probability"]   # callback output; time is the leading axis of every leaf
result["states"]        # tuple of reachable state names, initial states first
```

Recorded time axis length: `T_out = (horizon * steps_per_unit) // record_every + 1`, covering `t = 0, record_every * step_size, ..., horizon`.

### Reduction to reachable subgraph

`solve()` auto-reduces via `Model.reduce(initial_states)`, where `initial_states` = set declared on the `InitialDistribution`. Unreachable states are excluded entirely.

**Initial-state set is structural** — always user-declared on the distribution; never inferred from runtime mass or index-array contents. Declaring a state with all-zero mass still allocates and traces through its point-mass slot.

**Reduced-index ordering:** initial states first (in state-space order), non-initial reachable states follow. `K=1` case reduces to "initial state at reduced index 0" (backward compatible).

### Initial conditions at `t=0`

- Declared reachable state `j`: `state[j].point_mass` encodes per-individual `(mass, duration)`; `state[j].density = 0`.
- Non-declared reachable state `j`: `state[j].density = 0`, `state[j].point_mass = None`.
- Per-individual `d_0` does not need to land on the duration grid (evolves along characteristic).

v1 seeds only the point-mass component; an absolutely continuous starting density is forward-looking.

### Heun scheme (per scan step, `step_size = 1/steps_per_unit`)

1. **Predictor** — evaluate intensities at `t` (nudged by `±perturbation`); compute per-state derivatives (outflows from `density` / `point_mass`, inflows to `density`); Euler step.
2. **Corrector** — evaluate at `t + step_size`; recompute derivatives; average with predictor.
3. **Duration shift** — slot `k → k+1`; slot 0 of `density` receives fresh inflow; slot 0 of `point_mass` zeroed; mass at final slot truncated.

### Numerical order

2nd-order on smooth intensities; 1st-order across finite jumps regardless of `perturbation`.

### JIT boundary

| Static (trace-time constants) | Traced (runtime values) |
|---|---|
| Matrix sparsity pattern (positions of `None` cells) | Covariate arrays (`**kwargs`) |
| Callback function | Fitted parameters captured in closures |
| Presence/absence of `point_mass` per state | `InitialDistribution` mass and duration arrays |
| Set of initial states (declared on the distribution) | |
| `step_size`, `record_every`, `perturbation` | |

Changing any static field re-traces. Changing `mass` / `duration` / `states`-index values inside an existing initial-state set does not.

### Open design questions

1. Discontinuity handling protocol.
2. Per-state duration depth `D_j` (pytree state structure already allows it; Markov states would collapse to `D_j = 1`).
3. Absolutely continuous initial component (extend `InitialDistribution` with optional per-state `density: (batch, D)`).

Tracked in `docs/design/solver.md`.

---

## Callbacks

```python
def callback(state: tuple[StateCarry, ...]) -> PyTree: ...
```

`lax.scan` stacks the returned PyTree along a new leading axis. **Time is always the leading axis of every output leaf** — no rank-dependent transpose; downstream axis moves are the user's responsibility.

### Built-ins

| Name | Description | Per-step output |
|---|---|---|
| `"default"` | Full pytree, no reduction | `tuple[StateCarry, ...]` — each leaf `(batch, D)` (or `None`) |
| `"no_duration"` | Marginalise over duration, keep pytree | Per state: `(density[..., -1], point_mass[..., -1] or None)` — each `(batch,)` |
| `"collapse_point"` | Fold `point_mass` into `density[..., 0]`; drop `point_mass` | Tuple of per-state `density` — each `(batch, D)` |
| `"collapse_point_no_duration"` | Collapse then marginalise; re-stack | Single array `(batch, J)` |
| `"point_only"` | Per-state `point_mass` (or `None`) | Tuple per state — each `(batch, D)` or `None` |
| `"point_only_no_duration"` | Per-state `point_mass[..., -1]` (or `None`) | Tuple per state — each `(batch,)` or `None` |
| `"no_point"` | Per-state `density` | Tuple per state — each `(batch, D)` |
| `"no_point_no_duration"` | Marginalise density, re-stack | Single array `(batch, J)` |
| `"none"` | Record nothing | `None` |

Convention: `_no_duration` names that imply a state-indexed vector re-stack to `(batch, J)`; duration-preserving callbacks keep per-state pytree structure. `collapse_point_no_duration` is the canonical actuarial output — recorded shape `(T_out, batch, J)`.

Custom callbacks may return any PyTree; solver stacks each leaf along a leading time axis. This is the extension point for future cashflow / integral-transform features.

---

## Full example

See `api_spec.md` §Full example for a worked StateSpace → intensities → `build()` → `solve()` run, plus the competing-risks example with a joint cause-specific neural net.

---

## Design principles

1. Separation of structure and models (`StateSpace` stable; `Model` swappable).
2. Uniform callable interface: solver only sees `(t, d, **kwargs) → array`.
3. Compute only what's needed: reduce to reachable subgraph from the declared initial-state set.
4. Fail early, fail clearly: validation at `StateSpace` construction and `Model.build()` time; error messages reference names, not indices.
5. JIT everything: one XLA program end-to-end; no Python callbacks inside the solver loop.
6. Batch-first: 100K+ individuals in a single pass; covariates are arrays.
7. Point mass and continuous density are separate objects: distinct physics (advection-reaction vs scalar decay along characteristic); enables first-class `InitialDistribution` with off-grid per-individual `d_0`.

---

## Future work

- Discontinuity handling protocol (restore 2nd-order across jumps).
- Per-state duration depth `D_j`.
- Absolutely continuous initial distribution (per-state `density: (batch, D)` field).
- Pre-computation protocol (two-phase `prepare`/`evaluate`).
- Built-in parametric hazards (Gompertz, Weibull, piecewise constant, …).
- Cashflow computation via the callback system (integral transforms for actuarial present values).
