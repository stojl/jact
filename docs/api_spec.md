# jact — API specification

## Overview

`jact` is a JAX framework for computing transition probabilities in multi-state models with duration-dependent transition intensities (semi-Markov models). It provides a pipeline from fitted intensity models to transition probabilities, supporting 100K+ individuals in a single vectorized pass.

The framework separates three concerns:

- **StateSpace**: the structural definition of states and allowed transitions (stable, reusable, serializable).
- **Model**: a StateSpace bound to intensity callables and optional `TransitionSpec` metadata (swappable, experimentable).
- **Solver**: the numerical engine that computes transition probabilities from a given initial state, operating only on the reachable subgraph.

All intensity models must be JIT-compatible. The entire pipeline from covariates to transition probabilities compiles into a single XLA program.

---

## Module layout

```
jact/
├── __init__.py              # Public API: StateSpace, Model, TransitionSpec, InitialDistribution, solve, callbacks
├── state_space.py           # StateSpace class + InitialDistribution helpers
├── model.py                 # Model, ReducedModel, TransitionInfo
├── initial_distribution.py  # InitialDistribution class
├── solver.py                # Semi-Markov solver (per-transition quadrature + shared update)
├── intensity/
│   ├── __init__.py
│   ├── parametric.py        # Built-in parametric hazards (future)
│   └── wrappers.py          # Adapters for common model types (future)
└── callbacks.py             # Probability output callbacks
```

---

## StateSpace

The `StateSpace` defines the topology of the multi-state model: which states exist and which transitions between them are possible. It carries no intensity models and no data.

### Construction

```python
import jact

state_space = jact.StateSpace(
    states=["healthy", "disabled", "dead"],
    transitions=[
        ("healthy", "disabled"),
        ("healthy", "dead"),
        ("disabled", "dead"),
    ],
)
```

### Validation at construction

The `StateSpace` validates:
- No duplicate state names.
- All states referenced in transitions exist in the state list.
- No self-transitions (i → i).
- No duplicate transitions.

### Properties

```python
state_space.states          # ("healthy", "disabled", "dead")
state_space.n_states        # 3
state_space.transitions     # frozenset of transition tuples
state_space.absorbing       # ("dead",) — states with no outgoing transitions
state_space.transient       # ("healthy", "disabled") — states with outgoing transitions
```

### Queries

```python
state_space.exits("healthy")              # (("healthy", "disabled"), ("healthy", "dead"))
state_space.targets("healthy")            # ("disabled", "dead")
state_space.sources("dead")               # ("healthy", "disabled")
state_space.has_transition("healthy", "dead")  # True
state_space.state_index("disabled")       # 1
```

### Reachability

Given a starting state, the `StateSpace` computes which states are reachable via a breadth-first traversal of the transition graph. The starting state is always first in the result, followed by other reachable states in their original ordering.

```python
state_space.reachable_from("healthy")     # ("healthy", "disabled", "dead")
state_space.reachable_from("disabled")    # ("disabled", "dead")
state_space.reachable_from("dead")        # ("dead",)
```

This is used by the solver to reduce the computation to only the relevant states.

### Serialization

The `StateSpace` is a plain data object that can be serialized and reused across projects.

```python
state_space.to_json("disability_model.json")
state_space = jact.StateSpace.from_json("disability_model.json")
```

---

## Model

A `Model` is a `StateSpace` paired with intensity callables. It is the object passed to the solver. Models are immutable — to experiment with different intensity models, build a new `Model` from the same `StateSpace`.

### Building a model

Models are created via `StateSpace.build()`. This method accepts three optional keyword arguments for assigning intensity callables to transitions. Every transition declared in the `StateSpace` must be covered exactly once across all arguments.

Each assigned callable may be passed either as a bare callable or wrapped in a `TransitionSpec` carrying solver-relevant metadata:

```python
@dataclass(frozen=True)
class TransitionSpec:
    fn: Callable
    continuity_t: Literal["unknown", "discontinuous", "continuous"] = "unknown"
    continuity_d: Literal["unknown", "discontinuous", "continuous"] = "unknown"
```

Bare callables remain valid and are treated as if wrapped with `continuity_t="unknown"` and `continuity_d="unknown"`.

#### `transitions` — one callable per transition

The simplest case. Each transition gets its own intensity function.

```python
model = state_space.build(
    transitions={
        ("healthy", "disabled"): onset_fn,
        ("healthy", "dead"): jact.TransitionSpec(
            fn=mortality_fn,
            continuity_t="continuous",
            continuity_d="continuous",
        ),
        ("disabled", "dead"): disabled_mortality_fn,
    }
)
```

#### `exits` — one callable for all exits from a state

For competing risks or joint cause-specific models where a single model produces intensities for all transitions out of a given state in one forward pass.

```python
model = state_space.build(
    exits={
        "healthy": joint_cause_model,
    },
    transitions={
        ("disabled", "dead"): disabled_mortality_fn,
    },
)
```

The assigned callable or `TransitionSpec` must return an array whose first axis indexes over the target states in the order given by `state_space.targets(source)`.

`exits` always means *all* exits from that state. For partial coverage, use `groups`.

#### `groups` — one callable for an arbitrary set of transitions

For models that span an arbitrary subset of the transition matrix, such as shared frailty models, full-system neural networks, or any model where a single forward pass produces intensities for transitions that don't share a source state.

```python
model = state_space.build(
    groups={
        jact.TransitionSpec(
            fn=shared_frailty_model,
            continuity_t="discontinuous",
            continuity_d="continuous",
        ): [
            ("healthy", "dead"),
            ("disabled", "dead"),
        ],
    },
    transitions={
        ("healthy", "disabled"): onset_fn,
    },
)
```

The assigned callable or `TransitionSpec` must return an array whose first axis indexes over the listed transitions in the order provided.

### Continuity metadata on assigned callables

Continuity is a property of the assigned callable, not of the model as a whole. The solver reasons separately about continuity in clock time `t` and continuity in duration `d`:

- `continuity_t="continuous"` means the callable is continuous in clock time along the steps the solver traverses.
- `continuity_d="continuous"` means the callable is continuous in duration along the transported characteristic.
- `discontinuous` means jumps may occur, but only on user-aligned grid lines.
- `unknown` is treated conservatively like `discontinuous` for solver selection.

Both axes matter. A callable that is continuous in time but discontinuous in duration still uses the conservative midpoint rule, because the characteristic moves through duration over the step. The same holds in the opposite direction.

#### Combining all three

All three arguments can be used in a single `build()` call. `build()` validates that every declared transition is assigned exactly once — no gaps, no overlaps.

Consider a state space with a recovery transition:

```python
state_space = jact.StateSpace(
    states=["healthy", "disabled", "recovered", "dead"],
    transitions=[
        ("healthy", "disabled"),
        ("healthy", "dead"),
        ("disabled", "recovered"),
        ("disabled", "dead"),
        ("recovered", "dead"),
    ],
)

model = state_space.build(
    exits={
        "disabled": joint_recovery_and_mortality_model,
    },
    groups={
        shared_mortality_frailty: [
            ("healthy", "dead"),
            ("recovered", "dead"),
        ],
    },
    transitions={
        ("healthy", "disabled"): onset_fn,
    },
)
```

Here `exits` covers both transitions out of `disabled`, `groups` pairs two mortality transitions that don't share a source, and `transitions` handles the remaining single transition.

### Experimenting with models

The separation of `StateSpace` and `Model` makes it easy to swap intensity models while keeping the same structure:

```python
model_glm = state_space.build(
    transitions={
        ("healthy", "disabled"): onset_glm,
        ("healthy", "dead"): mortality_gompertz,
        ("disabled", "dead"): disabled_mort_glm,
    }
)

model_nn = state_space.build(
    transitions={
        ("healthy", "disabled"): onset_neural_net,
        ("healthy", "dead"): mortality_gompertz,
        ("disabled", "dead"): disabled_mort_glm,
    }
)
```

### Model reduction

When solving from a given `InitialDistribution` (or the equivalent shortcuts), the `Model` reduces itself to the reachable subgraph. This is handled automatically by `solve()`, but can also be called directly:

```python
reduced = model.reduce("disabled")                       # single initial state
reduced.initial_states     # ("disabled",)
reduced.reachable_states   # ("disabled", "dead")
reduced.n_states           # 2
reduced.solver_matrix      # 2×2 matrix of callables

reduced = model.reduce({"healthy", "disabled"})           # mixture of initial states
reduced.initial_states     # ("healthy", "disabled")     — state-space ordering
reduced.reachable_states   # ("healthy", "disabled", "dead")
```

`Model.reduce` accepts a single state name or an iterable of state names (the initial-state set). Initial states occupy the first `K` reduced indices in state-space ordering; non-initial reachable states follow. States not reachable from any initial state are excluded entirely, saving computation.

### Inspecting a model

```python
model.info("healthy", "disabled")
# → TransitionInfo(source="healthy", target="disabled",
#                   assignment="exits", callable=joint_cause_model, index=0)
```

---

## InitialDistribution

The `InitialDistribution` encodes the joint distribution over `(state, duration)` at `t = 0`, per individual. It is the input that tells `solve()` where probability mass starts.

### What it covers

Three usage patterns, increasing in generality:

1. **All individuals start in the same state at duration 0** — the original default.
2. **All individuals start in the same state, with per-individual duration `d_0`** — common when joining longitudinal data where each individual's time-in-state at observation start is known.
3. **Mass spread across multiple initial states, with per-individual mass and duration per state** — for heterogeneous populations or epistemic uncertainty about the starting state.

An absolutely continuous starting *density* — a distribution over duration rather than a Dirac at a single point — is forward-looking; the object is designed to grow into it without breaking the v1 API. See **Future work**.

### Construction

Primary constructor:

```python
dist = jact.InitialDistribution(
    components={
        "healthy":  {"mass": mass_h,  "duration": d_h},
        "disabled": {"mass": mass_d,  "duration": d_d},
    },
    normalise=True,   # default
)
```

`components` is a mapping from state name to a `(mass, duration)` pair. Each entry contributes a per-individual point mass concentrated at `duration` with weight `mass`. `mass` and `duration` are scalars or `(batch,)` arrays; scalars broadcast over the batch.

If `normalise=True` (the default), component masses are rescaled per individual so the total mass across declared states is 1 before solving. Rows whose total mass is already 1 are unchanged; rows whose total mass is positive but not 1 are rescaled proportionally; rows whose total mass is 0 remain all zero. If `normalise=False`, masses are used exactly as supplied.

### Convenience constructors

```python
# All individuals: state "healthy", d_0 = 0
dist = jact.InitialDistribution.at("healthy", duration=0.0)

# Per-individual initial state (and optionally per-individual d_0)
dist = jact.InitialDistribution.per_individual(
    states=idx_array,           # (batch,) int32 indices — TRACED
    duration=d_0_array,         # scalar or (batch,), optional
    initial_states=None,        # optional static tuple of state names
)
```

`states` is a traced `(batch,)` integer array. Its values are indices into either the optional `initial_states` tuple or, when `initial_states=None`, the model's full state list at `solve()`-entry. Being fully traceable, `per_individual` may be called from inside the user's own `jax.jit` / `vmap`.

`initial_states` is a Python tuple of state names — static, user-declared. When provided, `states` indexes into that tuple and the solver reduces the model to the reachable subgraph from those states. When omitted (`None`, the default), the full model state list is used and **no reduction is performed**: every state is treated as potentially initial.

Users with a `(batch,)` array of state *names* can either convert to indices host-side before calling — `jnp.array([state_space.state_index(s) for s in names])` — or use the `StateSpace.initial_per_individual` helper below, which absorbs the conversion and validates names eagerly. The underlying `InitialDistribution.per_individual` constructor remains index-only to keep the name→index step visible when the user wants it that way.

### Eager-validation helpers on `StateSpace`

The constructors above are **state-space-agnostic**: they accept state names as opaque strings and defer name validation to `solve()`-entry. That is deliberate (see the **Design note** below), but it means a typo like `components={"helthy": ...}` does not surface until the solver runs.

For users who prefer **fail-early** validation, and for users with a `(batch,)` array of state *names* rather than indices, `StateSpace` exposes three thin helpers that wrap the constructors above, validate every state name against `self.states` immediately, and return a plain `InitialDistribution`:

```python
dist = state_space.initial_at("healthy", duration=0.0)

dist = state_space.initial_per_individual(
    state_names=name_array,                      # (batch,) host-side array of state names
    duration=d_0_array,
    initial_states=("healthy", "disabled"),      # optional; validated eagerly
)

dist = state_space.initial_per_individual(
    state_indices=idx_array,                     # (batch,) int32 array of indices (traced)
    duration=d_0_array,
    initial_states=None,
)

dist = state_space.initial_distribution(
    components={
        "healthy":  {"mass": mass_h, "duration": d_h},
        "disabled": {"mass": mass_d, "duration": d_d},
    },
    normalise=True,                         # rescale per-individual masses to sum to 1
)
```

Exactly one of `state_names` / `state_indices` is required on `initial_per_individual`. When `state_names` is used, the helper does the name→index lookup against `self.states` (or against `initial_states`, if provided) and the resulting `InitialDistribution` is indistinguishable from one built via the index path.

These helpers are **purely ergonomic**. They return plain `InitialDistribution` objects; nothing downstream — `solve()` semantics, the JIT boundary, the reduction rules, the static-topology invariant — is affected. Users who want portability across multiple `StateSpace`s (one distribution reused across models that share state names) stay on the constructors in the previous subsection.

### Shortcuts on `solve()`

The `solve()` `initial` parameter accepts any of three forms; the first two are constructed into an `InitialDistribution` internally:

```python
model.solve(initial="healthy", ...)                            # str shorthand
model.solve(initial="healthy", initial_duration=d_0, ...)      # str + scalar or (batch,) d_0
model.solve(initial=idx_array, ...)                            # (batch,) int32 indices into model.states
model.solve(initial=idx_array, initial_duration=d_0, ...)      # same + per-individual d_0
model.solve(initial=jact.InitialDistribution(...), ...)        # full control
```

The `(batch,)` shortcut takes **integer indices only** (into `model.states`) and is fully jit-clean: it lifts to `InitialDistribution.per_individual(states=idx_array, duration=d_0, initial_states=None)`, i.e. the full model state list with no reduction. Users with a `(batch,)` array of state names convert to indices themselves via `state_space.state_index(...)`; there is no name-array shortcut. Users who want the reduction optimisation construct the distribution explicitly with a declared `initial_states` tuple — see the opt-in reduction example below.

`initial_duration` is valid only on the `str` and `(batch,)` paths. Passing it together with an `InitialDistribution` raises `ValueError` — duration is encoded in the object.

### Validation

At `InitialDistribution` construction:

- `mass` and `duration` arrays per component are mutually shape-consistent (all scalar, or all `(batch,)` with matching batch dimension across components).
- `mass >= 0` pointwise.
- `duration >= 0` pointwise.
- If `normalise=True` (default), per-individual component masses are normalised before use so their sum is 1.
- If `normalise=True` and a row already sums to 1, it is unchanged.
- If `normalise=True` and a row sums to a positive value different from 1, it is rescaled proportionally.
- If `normalise=True` and a row sums to 0, it remains all zero.
- If `normalise=False`, no normalisation is applied. Output is then linear in the input mass scale — the user is responsible for downstream interpretation.

At `solve()`:

- Every declared state name — whether from the keys of `components`, the single name passed to `at`, or the `initial_states` tuple on `per_individual` — exists in the model's state space.
- For `per_individual` with an explicit `initial_states` tuple, `states` values lie in `[0, len(initial_states))`; with `initial_states=None`, in `[0, n_states)`.
- The batch dimension of the distribution matches the batch dimension of the covariates.

### Interaction with the solver state

Each state declared in the `InitialDistribution` (i.e. a key in `components`, the single name passed to `at`, or a member of the `initial_states` tuple on `per_individual` — *not* conditional on its mass values being non-zero) is seeded with a `StateCarry.point_mass` at `t = 0` representing its per-individual mass concentrated at its per-individual duration. Reachable states *not* declared in the distribution keep `point_mass = None`.

The point mass evolves along its characteristic `(s, d_0 + s)` as a per-individual scalar problem (see **Solver → Solver state** and **Design principles** §7), so per-individual `d_0` need **not** land on the duration grid.

### Static-topology invariant

The initial-state set is a **structural field of the distribution, declared by the user** — either the keys of `components`, the single state passed to `at`, or the `initial_states` tuple passed to `per_individual`. When `per_individual` omits `initial_states`, the set defaults to the model's full state list at `solve()`-entry (no reduction). In every case the set is static on the JIT boundary; mass and duration values are traced.

A user who declares `{"healthy": ..., "disabled": ...}` in `components` and passes all-zero mass for `disabled` still pays the cost of allocating a point-mass slot for `disabled`. This is deliberate — the set lives in the declaration, not in the mass values.

### Examples

All individuals start healthy at `d_0 = 0` (equivalent to `initial="healthy"`):

```python
dist = jact.InitialDistribution.at("healthy")
result = model.solve(initial=dist, horizon=30, steps_per_unit=12, baseline_age=ages)
```

All individuals start healthy, but with a per-individual duration:

```python
result = model.solve(
    initial="healthy",
    initial_duration=time_in_state_at_observation,   # (batch,)
    horizon=30, steps_per_unit=12, baseline_age=ages,
)
```

Per-individual initial state, derived from data (jit-clean integer-index shortcut):

```python
names = ...  # (batch,) host-side numpy array of state names
idx = jnp.array([state_space.state_index(s) for s in names])
result = model.solve(
    initial=idx,
    initial_duration=time_in_state_at_observation,
    horizon=30, steps_per_unit=12, baseline_age=ages,
)
```

Opt-in reduction — declare a small initial-state set to carve out the reachable subgraph:

```python
dist = jact.InitialDistribution.per_individual(
    initial_states=("healthy", "disabled"),
    states=idx_array,                            # (batch,) int32, values in [0, 2)
    duration=d_0_array,
)
result = model.solve(initial=dist, horizon=30, steps_per_unit=12, baseline_age=ages)
```

Mixture across states with per-individual mass and duration:

```python
dist = jact.InitialDistribution(
    components={
        "healthy":  {"mass": p_h,     "duration": jnp.zeros_like(p_h)},
        "disabled": {"mass": 1 - p_h, "duration": d_disabled},
    },
)
result = model.solve(initial=dist, horizon=30, steps_per_unit=12, baseline_age=ages)
```

With `normalise=True`, the example above is unchanged if `p_h + (1 - p_h) == 1` per individual, because those rows already sum to 1.

If instead the same individual-level mixture were supplied on another mass scale, for example `{"healthy": 2 * p_h, "disabled": 2 * (1 - p_h)}`, `normalise=True` would rescale it back to the same proportions before solving, while `normalise=False` would preserve the factor of 2 and therefore preserve linearity in the supplied mass scale.

### Design note — state-space-agnostic construction

`InitialDistribution` is state-space-agnostic **deliberately**: one distribution can be reused across models that share state names, and construction stays free of model dependencies. State-name validation happens at `solve()`-entry, not at construction. The key set of the distribution is always **user-declared** — by the keys of `components`, the single name for `at`, or the `initial_states` tuple on `per_individual` — and is never inferred from runtime data.

`StateSpace` additionally exposes convenience constructors (`initial_at`, `initial_per_individual`, `initial_distribution`; see **Eager-validation helpers on `StateSpace`** above) that validate state names immediately against a specific `StateSpace` and return the same `InitialDistribution` object type. They are purely ergonomic — the state-space-agnostic constructors remain the canonical low-level API, the `solve()`-entry check remains the authoritative validation, and the initial-state set is still user-declared in every case.

---

## Intensity protocol

Intensity callables are **pure functions** bound to transitions at `Model.build()` time. The callable interface stays simple; continuity is carried separately on the surrounding `TransitionSpec`.

### Call signature

```python
def intensity(t, d, **kwargs) -> jnp.ndarray:
    ...
```

| Argument | Type | Description |
|---|---|---|
| `t` | `float` scalar | Current clock time. Advances from `0` to `horizon` as the solver marches forward. |
| `d` | `jnp.ndarray`, shape `(1, D)` | Duration grid for the source state. Entry `k` corresponds to a duration of `k / steps_per_unit`. The leading `1` axis broadcasts over the batch dimension. |
| `**kwargs` | `jnp.ndarray`, shape `(batch, ...)` | Covariate arrays passed through from `solve()`. Each callable consumes the subset it needs; unused kwargs are ignored. |

The solver maintains a probability density over duration `d` for each state and advances it in clock time `t`. Per solver step the callable receives the current `t` and the full duration array `d` simultaneously and returns the intensity surface `μ(t, d)` over all `D` duration points in one call.

`t` and `d` play distinct roles. For a population observed at `t=0` with known baseline ages, attained age at clock time `t` is `baseline_age + t` — age advances with clock time, not with duration in state. Duration `d` enters separately when the intensity depends on how long the individual has been in the current state. A Markov intensity uses only `t`; a pure duration-dependent intensity uses only `d`; a semi-Markov intensity uses both.

### `TransitionSpec`

```python
@dataclass(frozen=True)
class TransitionSpec:
    fn: Callable
    continuity_t: Literal["unknown", "discontinuous", "continuous"] = "unknown"
    continuity_d: Literal["unknown", "discontinuous", "continuous"] = "unknown"
```

The solver uses the continuity metadata on each assigned callable to choose the per-step quadrature rule:

- `continuity_t="continuous"` and `continuity_d="continuous"`: endpoint Heun/trapezoidal along the transported characteristic.
- Any other combination: midpoint along the transported characteristic.

This requires both axes. A callable continuous in time but discontinuous in duration still uses midpoint, because the characteristic moves through duration over the step. The same holds in the opposite direction.

### JAX requirements

The callable is traced by JAX and compiled into the solver's `lax.scan` body. It must be:

- **Pure**: no Python side effects, no mutation.
- **JIT-compatible**: no data-dependent Python control flow, no non-JAX operations.
- **Closed over static values only**: fitted parameters (arrays, scalars) may be captured in the closure; they become compile-time constants.

### Return shapes by assignment type

| Assignment | Return shape | Description |
|---|---|---|
| `transitions` (single) | `(batch, D)` | One intensity surface |
| `exits` (all exits) | `(n_targets, batch, D)` | One per target state, ordered by `state_space.targets(source)` |
| `groups` (arbitrary) | `(n_transitions, batch, D)` | One per listed transition, in the order provided to `build()` |

The solver itself only ever sees `(batch, D)`: `exits` and `groups` callables are pre-sliced by `Model._build_full_solver_matrix` at build time, while their `TransitionSpec` metadata stays attached to the assigned callable.

### Examples

**Markov — Gompertz mortality** (depends only on attained age, i.e. clock time):

```python
alpha, beta = fit_gompertz(data)

def gompertz_mortality(t, d, baseline_age, **kwargs):
    attained_age = baseline_age + t   # (batch,)
    return jnp.exp(alpha + beta * attained_age)[:, None] * jnp.ones_like(d)  # (batch, D)
```

**Semi-Markov — duration-dependent onset** (depends on both attained age and duration in state):

```python
coef = fit_glm(data).coef_

def onset_semi_markov(t, d, baseline_age, **kwargs):
    attained_age = baseline_age + t          # (batch,)
    lp = coef[0] + coef[1] * attained_age   # (batch,)
    return jnp.exp(lp[:, None]) * baseline(d)  # (batch, D)
```

**Competing risks — neural network with multiple output heads**:

```python
def joint_hazard(t, d, baseline_age, bmi, smoking, **kwargs):
    features = jnp.stack([baseline_age + t, bmi, smoking], axis=-1)  # (batch, p)
    log_hazards = net(features)                                        # (batch, n_targets)
    return jnp.exp(log_hazards).T[:, :, None] * baseline(d)          # (n_targets, batch, D)
```

### Continuity policy

The solver does not interpret a global "continuous model" flag. Continuity is determined per assigned callable:

- `unknown` is treated conservatively like `discontinuous`.
- `discontinuous` means jumps may occur, but only on user-aligned grid lines.
- The quadrature decision is local to the assigned callable, while the mass update remains shared across all exits from the same source state.

---

## Solver

### What the solver computes

For a reduced model rooted at an initial state, the solver advances a probability state over `[0, horizon]` inside a single `jax.lax.scan`, vectorised over a batch axis via `jax.vmap`. Each scan step computes per-transition integrated hazards using the quadrature rule implied by that transition's `TransitionSpec`, then applies one shared competing-risks update per source state. The full pipeline from covariates to transition probabilities compiles into one XLA program.

### Solver state

Per reachable state, the solver tracks two conceptually separate objects:

- **`density`** — the absolutely continuous duration density for that state, shape `(batch, D)`. Entry `density[b, k]` is the density at batch element `b` at duration slot `k`. Slot 0 is "entered the current state just now"; slot `k` is "entered `k` solver steps ago".
- **`point_mass`** — a per-individual point mass at the state's `d_0`, stored as `PointMass(value, d_0)` or `None`. `None` for states that never carry one. For states that do — every state declared in the active `InitialDistribution` — `value` has shape `(batch,)` and tracks the current mass, while `d_0` has shape `(batch,)` and stores the per-individual initial duration exactly.

The full solver state is a pytree, one entry per reachable state in `reachable_states` order:

```python
state: tuple[StateCarry, ...]            # length = J = reduced.n_states

@jax.tree_util.register_pytree_node_class
class PointMass:
    value: jnp.ndarray                   # (batch,)
    d_0: jnp.ndarray                     # (batch,)

class StateCarry(NamedTuple):
    density: jnp.ndarray                 # (batch, D)
    point_mass: PointMass | None
```

`density` evolves by advection-reaction with a rigid duration shift (mass at duration `k` becomes mass at duration `k+1` each step). `point_mass` evolves by a scalar exponential decay along the characteristic `(s, d_0 + s)` — a 1-D problem per individual, not a 2-D one. They are co-evolved but mathematically distinct; keeping them separate avoids diffusing a Dirac through the finite-difference scheme. This factorisation is what makes per-individual `d_0` (off-grid) and per-state initial point masses first-class via the `InitialDistribution`; analytic or high-order point-mass integration remains a future extension.

### Calling the solver

```python
result = model.solve(
    initial="healthy",
    horizon=10,
    steps_per_unit=12,
    callback="collapse_point_no_duration",
    record_every=1,
    baseline_age=age_array,
    bmi=bmi_array,
)
```

Or equivalently via the functional interface:

```python
result = jact.solve(model, initial=..., horizon=..., steps_per_unit=..., **covariates)
```

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `initial` | `str`, `(batch,)` int32 array, or `InitialDistribution` | Initial condition. `str` = all individuals start in this state at `d_0 = 0`. `(batch,)` integer array = per-individual initial state, with values interpreted as indices into `model.states` (jit-clean; users with a name array convert via `state_space.state_index(...)`). `InitialDistribution` = full control (mixtures, per-individual mass and duration; also the opt-in-to-reduction entry point). See the **InitialDistribution** section. |
| `initial_duration` | `float` or `(batch,)` array | Per-individual `d_0` for the `str` and `(batch,)` array forms of `initial`. Default: `0`. Passing this together with an `InitialDistribution` raises `ValueError` — duration is encoded in the object. |
| `horizon` | `int` | Number of time units to solve over. |
| `steps_per_unit` | `int` | Discretisation resolution per time unit. `D = horizon * steps_per_unit`. |
| `callback` | `str`, `callable`, or `None` | Probability callback (default: `"collapse_point_no_duration"`). |
| `record_every` | `int` | Record the callback output every `record_every`-th step. Must divide `horizon * steps_per_unit` evenly; otherwise `ValueError`. Default: `1`. |
| `**kwargs` | `jnp.ndarray` | Covariate arrays, each of shape `(batch, ...)`. The names `initial` and `initial_duration` are reserved — don't use them as covariate names. |

### Result

```python
result["probability"]   # callback output, time as the leading axis of every leaf
result["states"]        # tuple of state names in reachable order, initial states first
```

The recorded time axis has length `T_out = (horizon * steps_per_unit) // record_every + 1`, covering `t = 0, record_every * step_size, 2 * record_every * step_size, ..., horizon`.

```python
result = model.solve(initial="disabled", horizon=30, steps_per_unit=12, baseline_age=ages)
result["states"]        # ("disabled", "dead")
result["probability"]   # only 2 states computed, not 3 — per callback shape
```

### Reduction to reachable subgraph

Solving with a given `InitialDistribution` automatically reduces the model to the reachable subgraph via `Model.reduce(initial_states)`, where `initial_states` is the set of state names declared in the distribution. Unreachable states are excluded entirely.

**Initial-state set is structural.** The set of initial states is **user-declared** on the distribution — by the keys of `InitialDistribution.components`, the single name passed to `at`, or the `initial_states` tuple passed to `per_individual`. When `per_individual` omits `initial_states`, the set defaults to the model's full state list and no reduction is performed. In every case the set is part of the static trace shape — never inferred from runtime mass values or from the contents of an index array. Declaring a state with all-zero mass still allocates and traces through that state's point-mass slot (see the **Static-topology invariant** subsection of `InitialDistribution`).

**Reduced-index ordering.** Initial states occupy the first `K` reduced indices in state-space ordering, where `K` is the number of distinct states in the initial-state set; non-initial reachable states follow in their original ordering. For the common case `K = 1` (e.g. `initial="healthy"`), the initial state is at reduced index 0, matching the previous spec exactly. `result["states"]` records the mapping from reduced index back to state name.

Solver internals that were previously keyed on "state 0 is the initial state" — the point-mass initial condition, per-individual `d_0`, future absolutely continuous initial densities — instead iterate over the initial-state set recorded by the `InitialDistribution`.

### Initial conditions

Initial conditions are given by an `InitialDistribution` (see the dedicated section), passed via `solve(initial=...)`. The shorthand `initial="healthy"` is defined as `InitialDistribution.at("healthy", duration=0.0)`; the `(batch,)` integer-array shorthand is defined as `InitialDistribution.per_individual(states=idx_array, duration=..., initial_states=None)` — i.e. indices into the model's full state list, no reduction.

At `t = 0`:

- For every reachable state `j` declared in the `InitialDistribution`: `state[j].point_mass` is seeded to encode that state's per-individual mass at its per-individual duration. `state[j].density = 0`.
- For every reachable state `j` *not* declared in the `InitialDistribution`: `state[j].density = 0`, `state[j].point_mass = None`.

Because `point_mass` evolves along its characteristic `(s, d_0 + s)` as a scalar problem per individual, per-individual `d_0` does **not** need to land on the duration grid. v1 seeds only the point-mass component of each declared state; seeding a non-zero `density` at `t = 0` (an absolutely continuous starting distribution over duration) is forward-looking — see **Future work**.

### Per-step update

Each scan step advances the state by `step_size = 1 / steps_per_unit`:

1. **Per-transition integrated hazards** — for each transition `i -> j`, compute `A_ij` along the transported characteristic:

   ```text
   A_ij^(n)[k] ≈ ∫[t_n, t_(n+1)] μ_ij(t, d(t)) dt
   ```

   If `continuity_t == continuity_d == "continuous"`, use endpoint Heun/trapezoidal:

   ```text
   A_ij^(n)[k] = (dt / 2) · [μ_ij(t_n, d_k) + μ_ij(t_(n+1), d_k + dt)]
   ```

   Otherwise use midpoint:

   ```text
   A_ij^(n)[k] = dt · μ_ij(t_n + dt / 2, d_k + dt / 2)
   ```

2. **Shared competing-risks aggregation** — for each source state, aggregate

   ```text
   A_i = Σ_j A_ij
   S_i = exp(-A_i)
   ```

   and transition fractions

   ```text
   T_ij = (A_ij / A_i) · (1 - S_i)   if A_i > 0
   T_ij = 0                          if A_i = 0
   ```

   Mixed models remain coherent because different exits may use different quadrature rules, but all exits from the same source state still feed one shared survival factor.

3. **State transport** — surviving density mass shifts one duration slot to the right with factor `S_i`; transferred mass is injected into duration zero using `T_ij`. Point mass follows the same policy along the characteristic `(t, d_0 + t)`: continuous/continuous uses endpoint Heun/trapezoidal, otherwise midpoint.

### Numerical order

The relevant convergence statements are:

- Midpoint is second-order on a step if the callable is smooth on the interior of the traversed characteristic segment.
- Midpoint remains globally second-order for a callable if every jump in `t` or `d` lies on a grid line, so no traversed step crosses a jump.
- Midpoint drops to first order for that callable if a jump in either variable lies inside a traversed cell.
- Endpoint Heun/trapezoidal is second-order only when the callable is continuous in both `t` and `d` along the characteristic.
- Mixed models remain second-order provided each callable gets the quadrature rule implied by its own continuity metadata and all discontinuities are grid-aligned.

### JIT boundary

| Static (trace-time constants) | Traced (runtime values) |
|---|---|
| Matrix sparsity pattern (positions of `None` cells) | Covariate arrays (`**kwargs`) |
| Callback function | Fitted parameters captured in closures |
| Presence/absence of `point_mass` per state | `PointMass.value` and `PointMass.d_0` arrays from `InitialDistribution` |
| Set of initial states (declared on the distribution) | |
| `step_size`, `record_every` | |
| Assigned `TransitionSpec` continuity metadata | |

The set of initial states is always **user-declared** on the distribution — by the keys of `InitialDistribution.components`, the single name for `at`, or the `initial_states` tuple on `per_individual` (defaulting to the model's full state list when omitted). It is never inferred from runtime mass or from the contents of an index array.

Changing any static field triggers a re-trace. Rebuilding a `Model` with a different sparsity pattern re-traces; changing only parameter values inside existing callables does not. Changing `TransitionSpec` continuity metadata also re-traces, because quadrature choice is structural. Changing the *set* of initial states (e.g. adding `"disabled"` as a possible initial state) re-traces; changing only the per-individual `mass` / `duration` / `states`-index values inside an existing initial-state set does not. This is the reason the spec decides initial-state membership structurally rather than by inspecting runtime data — the latter would be a data-dependent topology change, incompatible with the trace contract.

### Memory budget

Peak output memory in bytes, at float32:

```
bytes ≈ 4 * T_out * product(callback_output_non_time_dims)
```

where `T_out = (horizon * steps_per_unit) // record_every + 1`. Worked examples at `batch = 100_000`, `J = 10`, `D = 360`, `T_out = 361`:

- `default` (density leaves `(time, batch, D)` plus point-mass leaves `(time, batch)` for declared states): still dominated by density, ≈ **520 GB** at this resolution. Infeasible.
- Same, but `record_every = 12` (`T_out = 31`): ≈ **44 GB**. Still large; tractable on a big GPU.
- `collapse_point_no_duration` (`(T_out, batch, J)`): ≈ **1.4 GB**. Comfortable.

Pick `callback` and `record_every` before scaling `batch`.

### Open design questions

The following items are intentionally left open by this spec and tracked in `docs/design/solver.md`:

1. **Per-state duration depth `D_j`** — currently uniform `D = horizon * steps_per_unit`. The pytree state structure allows per-state `D_j` as a future optimisation; Markov states would collapse to `D_j = 1`.
2. **Absolutely continuous initial component** — `InitialDistribution` v1 carries per-state point masses only. Extending each component with an optional `density: (batch, D)` field (an absolutely continuous starting distribution over duration) is forward-looking; the object is shaped so this can be added without breaking existing constructions.

---

## Callbacks

Callbacks control what is extracted from the solver state at each recorded step. They determine the shape and content of `result["probability"]`.

### Signature

```python
def callback(state: tuple[StateCarry, ...]) -> PyTree:
    ...
```

A callback receives the full pytree solver state (one `StateCarry` per reachable state, in `reachable_states` order) and returns an arbitrary PyTree. `lax.scan` stacks the returned PyTree along a new leading axis across time, and **time is always the leading axis of every output leaf**. No rank-dependent transpose is applied; downstream axis moves are the user's responsibility.

### Built-in callbacks

Under uniform `D`, the built-in callbacks produce the following per-step output shapes (the recorded result prepends a time axis of length `T_out` to each leaf):

| Name | Description | Per-step output |
|---|---|---|
| `"default"` | Full pytree state, no reduction | `tuple[StateCarry, ...]` — `density: (batch, D)`, `point_mass: PointMass(value=(batch,), d_0=(batch,))` or `None` |
| `"no_duration"` | Marginalise over duration, preserve pytree | Per state: `density = sum over duration`, `point_mass = PointMass(value=(batch,), d_0=(batch,))` or `None` |
| `"collapse_point"` | Fold `point_mass` into `density[..., 0]` per state; drop `point_mass` | Tuple of per-state `density` — each leaf `(batch, D)` |
| `"collapse_point_no_duration"` | Collapse then marginalise; re-stack across states | Single array `(batch, J)` |
| `"point_only"` | Per-state `point_mass` (or `None`) | Tuple per state — each leaf `PointMass(value=(batch,), d_0=(batch,))` or `None` |
| `"point_only_no_duration"` | Per-state point-mass value (or `None`) | Tuple per state — each leaf `(batch,)` or `None` |
| `"no_point"` | Per-state `density` | Tuple per state — each leaf `(batch, D)` |
| `"no_point_no_duration"` | Marginalise density, re-stack across states | Single array `(batch, J)` |
| `"none"` | Record nothing | `None` |

Convention: callbacks whose names end in `_no_duration` and whose semantics imply a state-indexed vector re-stack into a single `(batch, J)` array for convenience; callbacks that preserve the duration axis keep the per-state pytree structure. `collapse_point_no_duration` is the canonical callback for actuarial transition-probability output; its recorded shape is `(T_out, batch, J)`.

### Custom callbacks

A callback can return any PyTree; the solver stacks each leaf along a new leading time axis:

```python
def total_mass_per_state(state):
    """Sum of density + point mass per reachable state, per individual."""
    totals = []
    for carry in state:
        total = jnp.sum(carry.density, axis=-1)                   # (batch,)
        if carry.point_mass is not None:
            total = total + carry.point_mass.value
        totals.append(total)
    return jnp.stack(totals, axis=-1)                              # (batch, J)

# recorded shape: (T_out, batch, J)
```

The callback system is also the extension point for future features like **cashflow computation** — integral transforms over the duration density for actuarial present values.

---

## Full example

> **Note on output layout.** The `StateSpace`, `Model.build()`, and `solve()` call surfaces below are stable. The shape of `result["probability"]` reflects the target layout described in the Solver and Callbacks sections: **time is the leading axis of every leaf**. The current `solver.py` still emits `(batch, J, T_out, ...)` under a rank-dependent transpose; that implementation detail is scheduled to be removed along with the `transpose_result` kwarg.

```python
import jax.numpy as jnp
import jact

# 1. Define the state space (once, reuse across experiments)
state_space = jact.StateSpace(
    states=["healthy", "disabled", "dead"],
    transitions=[
        ("healthy", "disabled"),
        ("healthy", "dead"),
        ("disabled", "dead"),
    ],
)

# 2. Define intensity models
def onset_intensity(t, d, baseline_age, **kwargs):
    attained_age = baseline_age + t   # (batch,)
    return jnp.exp(-5.0 + 0.04 * attained_age)[:, None] * jnp.ones_like(d)

def mortality_healthy(t, d, baseline_age, **kwargs):
    attained_age = baseline_age + t
    return jnp.exp(-10.0 + 0.08 * attained_age)[:, None] * jnp.ones_like(d)

def mortality_disabled(t, d, baseline_age, **kwargs):
    attained_age = baseline_age + t
    return jnp.exp(-8.0 + 0.08 * attained_age)[:, None] * jnp.ones_like(d)

# 3. Build the model
model = state_space.build(
    transitions={
        ("healthy", "disabled"): onset_intensity,
        ("healthy", "dead"): mortality_healthy,
        ("disabled", "dead"): mortality_disabled,
    }
)

# 4. Compute transition probabilities from different starting states
ages = jnp.linspace(30, 80, 100_000)

# From healthy: computes all 3 reachable states
result_h = model.solve(
    initial="healthy",
    horizon=30,
    steps_per_unit=12,
    baseline_age=ages,
)
result_h["states"]  # ("healthy", "disabled", "dead")

# From disabled: computes only 2 reachable states
result_d = model.solve(
    initial="disabled",
    horizon=30,
    steps_per_unit=12,
    baseline_age=ages,
)
result_d["states"]  # ("disabled", "dead")
```

### Experimenting with models

```python
model_nn = state_space.build(
    transitions={
        ("healthy", "disabled"): trained_neural_net,
        ("healthy", "dead"): mortality_healthy,
        ("disabled", "dead"): mortality_disabled,
    }
)

result_nn = model_nn.solve(
    initial="healthy", horizon=30, steps_per_unit=12, baseline_age=ages
)

diff = jax.tree.map(
    lambda a, b: jnp.abs(a - b),
    result_h["probability"],
    result_nn["probability"],
)
```

### Competing risks example

```python
state_space_cr = jact.StateSpace(
    states=["healthy", "cancer", "heart_disease", "stroke", "disabled", "dead"],
    transitions=[
        ("healthy", "cancer"),
        ("healthy", "heart_disease"),
        ("healthy", "stroke"),
        ("healthy", "disabled"),
        ("healthy", "dead"),
        ("cancer", "dead"),
        ("heart_disease", "dead"),
        ("stroke", "dead"),
        ("disabled", "dead"),
    ],
)

# One neural net produces all 5 exit intensities from "healthy"
model_cr = state_space_cr.build(
    exits={
        "healthy": joint_cause_specific_neural_net,
    },
    transitions={
        ("cancer", "dead"): cancer_mortality,
        ("heart_disease", "dead"): cardiac_mortality,
        ("stroke", "dead"): stroke_mortality,
        ("disabled", "dead"): disabled_mortality,
    },
)

result = model_cr.solve(
    initial="healthy", horizon=30, steps_per_unit=12, **covariates
)
result["states"]
# ("healthy", "cancer", "heart_disease", "stroke", "disabled", "dead")
```

---

## Design principles

1. **Separation of structure and models.** The `StateSpace` is the stable backbone. Models are swappable experiments bound to the same structure.

2. **Uniform callable interface.** The solver doesn't know or care whether an intensity comes from a Gompertz function, a GLM, or a neural network. All it sees is `TransitionSpec(fn=..., ...)` bound to a transition, with `fn(t, d, **kwargs) → array`.

3. **Compute only what's needed.** Given an initial-state set (one or many states declared via the `InitialDistribution`), the solver reduces to the reachable subgraph — the union of reachability from each initial state. Unreachable states are excluded entirely.

4. **Fail early, fail clearly.** Validation happens at `StateSpace` construction and `Model.build()` time, not deep inside the solver. Error messages reference state names and transitions, not matrix indices.

5. **JIT everything.** The entire pipeline from covariates to transition probabilities compiles into a single XLA program. No Python callbacks inside the solver loop.

6. **Batch-first.** The framework is designed for 100K+ individuals in a single pass. Covariates are arrays, not scalars. The solver vectorizes over the batch dimension.

7. **Point mass and continuous density are separate objects.** Per state, the absolutely continuous duration density and the point mass at duration zero are tracked independently. They have different physics (advection-reaction with duration shift vs. scalar exponential decay along a characteristic); co-evolving them cleanly is what makes the `InitialDistribution` first-class — point masses can be carried by any state declared in the distribution, with per-individual mass and per-individual `d_0` (which need not land on the duration grid). The same factorisation keeps the door open for analytic or high-order integration of the point mass and for an absolutely continuous initial-density extension.

---

## Future work

- **Per-state duration depth `D_j`**: let each reachable state pick its own duration depth, collapsing Markov states to `D_j = 1`. Enabled by the pytree solver state.
- **Absolutely continuous initial distribution**: extend `InitialDistribution` so each per-state component can carry an optional `density: (batch, D)` field alongside the point-mass `(mass, duration)` pair. Lets the solver be seeded with a starting distribution over duration, not just a Dirac per individual. Designed so v1 constructions keep working.
- **Pre-computation protocol**: two-phase `prepare`/`evaluate` for intensity models with static covariate contributions.
- **Built-in parametric hazards**: Gompertz, Weibull, piecewise constant, and other standard forms in `jact.intensity`.
- **Broader transition metadata**: evolve `TransitionSpec` beyond continuity metadata as cashflow-related transition callables are designed.
- **Cashflow computation**: integral transforms over the duration density for actuarial present values, extending the callback system.
