# jact — API specification

## Overview

`jact` is a JAX framework for computing transition probabilities in multi-state models with duration-dependent transition intensities (semi-Markov models). It provides a pipeline from fitted intensity models to transition probabilities, supporting 100K+ individuals in a single vectorized pass.

The framework separates three concerns:

- **StateSpace**: the structural definition of states and allowed transitions (stable, reusable, serializable).
- **Model**: a StateSpace bound to intensity callables (swappable, experimentable).
- **Solver**: the numerical engine that computes transition probabilities from a given initial state, operating only on the reachable subgraph.

All intensity models must be JIT-compatible. The entire pipeline from covariates to transition probabilities compiles into a single XLA program.

---

## Module layout

```
jact/
├── __init__.py              # Public API: StateSpace, Model, solve, callbacks
├── state_space.py           # StateSpace class
├── model.py                 # Model, ReducedModel, TransitionInfo
├── solver.py                # Semi-Markov solver (Heun scheme)
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

#### `transitions` — one callable per transition

The simplest case. Each transition gets its own intensity function.

```python
model = state_space.build(
    transitions={
        ("healthy", "disabled"): onset_fn,
        ("healthy", "dead"): mortality_fn,
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

The callable assigned via `exits` must return an array whose first axis indexes over the target states in the order given by `state_space.targets(source)`.

`exits` always means *all* exits from that state. For partial coverage, use `groups`.

#### `groups` — one callable for an arbitrary set of transitions

For models that span an arbitrary subset of the transition matrix, such as shared frailty models, full-system neural networks, or any model where a single forward pass produces intensities for transitions that don't share a source state.

```python
model = state_space.build(
    groups={
        shared_frailty_model: [
            ("healthy", "dead"),
            ("disabled", "dead"),
        ],
    },
    transitions={
        ("healthy", "disabled"): onset_fn,
    },
)
```

The callable must return an array whose first axis indexes over the listed transitions in the order provided.

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

When solving from a specific initial state, the `Model` reduces itself to the reachable subgraph. This is handled automatically by `solve()`, but can also be called directly:

```python
reduced = model.reduce("disabled")
reduced.initial            # "disabled"
reduced.reachable_states   # ("disabled", "dead")
reduced.n_states           # 2
reduced.solver_matrix      # 2×2 matrix of callables
```

The initial state is always at index 0 in the reduced system. States not reachable from the initial state are excluded entirely, saving computation.

### Inspecting a model

```python
model.info("healthy", "disabled")
# → TransitionInfo(source="healthy", target="disabled",
#                   assignment="exits", callable=joint_cause_model, index=0)
```

---

## Intensity protocol

Intensity callables are **pure functions** bound to transitions at `Model.build()` time. The solver treats every cell of the (reduced) intensity matrix uniformly; fitted model parameters are captured via closures and become compile-time constants.

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

#### On `t` and `d`

The solver maintains a probability density over duration `d` for each state and advances it in clock time `t`. Per solver step the callable receives the current `t` and the full duration array `d` simultaneously and must return the intensity surface `μ(t, d)` over all `D` duration points in a single call.

`t` and `d` play distinct roles. For a population observed at `t=0` with known baseline ages, attained age at clock time `t` is `baseline_age + t` — age advances with clock time, not with duration in state. Duration `d` enters separately when the intensity depends on how long the individual has been in the current state. A Markov intensity uses only `t`; a pure duration-dependent intensity uses only `d`; a semi-Markov intensity uses both.

#### Càdlàg convention

Intensities are assumed **càdlàg** (right-continuous with left limits). At a point of discontinuity, the default evaluation is the right limit. This is the mathematical convention the solver assumes throughout; everything in the discontinuity-handling discussion below inherits it.

#### JAX requirements

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

The solver itself only ever sees `(batch, D)`: `exits` and `groups` callables are pre-sliced by `Model._build_full_solver_matrix` at build time.

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

### Discontinuity handling (open question, WIP)

Discontinuities along `t` or `d` — benefit entitlement boundaries, policy changes, waiting periods, age cutoffs — are a first-class modelling concern. The current solver handles them by evaluating every intensity at `d ± perturbation` and nudging clock time by `t ± perturbation` on each step, where `perturbation` defaults to `1e-12`. This scheme has three known limitations:

1. **Absolute `ε` does not scale with the argument.** In IEEE float64, the ulp at `|d| = 30` is ~`3.6e-15`; `1e-12` leaves very little margin once downstream arithmetic (`exp`, `log`, additions with `baseline_age`) erodes the last few bits. Float32 collapses this entirely.
2. **The perturbation is invisible to the user.** Whether two evaluations at `tau ± 1e-12` straddle a user-placed jump at `tau` depends on how `tau` was computed inside the callable. There is no knob for the user to declare "my jump is at exactly `tau`".
3. **Heun is 2nd-order only for smooth right-hand sides.** Through a finite jump, local error is `O(step_size)` regardless of the sampling scheme. Perturbation picks a consistent side to be wrong on; it does not restore the second order.

Options under consideration for the long-term protocol (see `docs/design/solver.md` §1 for the full treatment):

- **(a) Declared break points** — callables advertise a sorted array of jump times/durations; the solver aligns the grid or sub-steps around them.
- **(b) Piecewise callables** — the intensity is a list of `(interval, fn)` pairs.
- **(c) Relative perturbation** — replace `1e-12` with `rtol * (1 + |d|)`.
- **(d) Left/right-evaluation protocol** — callables opt into `fn(t, d, side)` where `side ∈ {left, right}` is a compile-time constant; the solver requests the side it needs, no perturbation required.
- **(e) Adaptive sub-stepping around declared break points** — composes with (a).

Until this is resolved: treat intensities as smooth, or place jumps well inside `(perturbation, horizon − perturbation)` on cleanly representable values.

---

## Solver

### What the solver computes

For a reduced model rooted at an initial state, the solver advances a probability state over `[0, horizon]` using a **Heun (2nd-order predictor-corrector) scheme** inside a single `jax.lax.scan`, vectorised over a batch axis via `jax.vmap`. The full pipeline from covariates to transition probabilities compiles into one XLA program.

### Solver state

Per reachable state, the solver tracks two conceptually separate objects:

- **`density`** — the absolutely continuous duration density for that state, shape `(batch, D)`. Entry `density[b, k]` is the density at batch element `b` at duration slot `k`. Slot 0 is "entered the current state just now"; slot `k` is "entered `k` solver steps ago".
- **`point_mass`** — a point mass at duration zero, shape `(batch, D)` or `None`. `None` for states that never carry one. For states that do (today: the initial state only), the `(batch, D)` tensor tracks the shifted-along-duration Dirac that seeded at `t=0`.

The full solver state is a pytree, one entry per reachable state in `reachable_states` order:

```python
state: tuple[StateCarry, ...]            # length = J = reduced.n_states

class StateCarry(NamedTuple):
    density: jnp.ndarray                 # (batch, D)
    point_mass: jnp.ndarray | None       # (batch, D) or None
```

`density` evolves by advection-reaction with a rigid duration shift (mass at duration `k` becomes mass at duration `k+1` each step). `point_mass` evolves by a scalar exponential decay along the characteristic `(s, d_0 + s)` — a 1-D problem, not a 2-D one. They are co-evolved but mathematically distinct; keeping them separate avoids diffusing a Dirac through the finite-difference scheme and makes heterogeneous `d_0` and analytic point-mass integration available as future extensions.

### Calling the solver

```python
result = model.solve(
    initial="healthy",
    horizon=10,
    steps_per_unit=12,
    callback="collapse_point_no_duration",
    record_every=1,
    perturbation=1e-12,
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
| `initial` | `str` | Starting state. The reachable subgraph from `initial` is computed; `initial` is always at reduced index 0. |
| `horizon` | `int` | Number of time units to solve over. |
| `steps_per_unit` | `int` | Discretisation resolution per time unit. `D = horizon * steps_per_unit`. |
| `callback` | `str`, `callable`, or `None` | Probability callback (default: `"collapse_point_no_duration"`). |
| `record_every` | `int` | Record the callback output every `record_every`-th step. Must divide `horizon * steps_per_unit` evenly; otherwise `ValueError`. Default: `1`. |
| `perturbation` | `float` | Grid perturbation for the current discontinuity scheme (default: `1e-12`; see Intensity protocol §Discontinuity handling). |
| `**kwargs` | `jnp.ndarray` | Covariate arrays, each of shape `(batch, ...)`. |

The name `initial_duration` is **reserved** for a future heterogeneous-`d_0` kwarg; don't use it as a covariate name.

### Result

```python
result["probability"]   # callback output, time as the leading axis of every leaf
result["states"]        # tuple of state names in reachable order, initial first
```

The recorded time axis has length `T_out = (horizon * steps_per_unit) // record_every + 1`, covering `t = 0, record_every * step_size, 2 * record_every * step_size, ..., horizon`.

```python
result = model.solve(initial="disabled", horizon=30, steps_per_unit=12, baseline_age=ages)
result["states"]        # ("disabled", "dead")
result["probability"]   # only 2 states computed, not 3 — per callback shape
```

### Reduction to reachable subgraph

Solving from `initial` automatically reduces the model to the reachable subgraph via `Model.reduce(initial)`. Unreachable states are excluded entirely.

**Load-bearing invariant.** The initial state is **always** at reduced index 0. Several solver internals depend on this — the point-mass initial condition, the heterogeneous-`d_0` hook, future per-state point masses. `result["states"]` records the mapping from reduced index back to state name.

### Initial conditions

At `t = 0`, all probability mass lives on the initial state's `point_mass` at duration zero:

- `state[0].density = 0`
- `state[0].point_mass[..., 0] = 1`, zero elsewhere
- for all `j > 0`: `state[j].density = 0`, `state[j].point_mass = None`

The point-mass formulation makes per-individual `d_0` a drop-in extension: instead of initialising at slot 0 for everyone, individual `b` initialises at their own duration slot (or, equivalently, phase 1 of a split solve treats `point_mass` as a per-individual scalar ODE along its characteristic). Not yet exposed.

### Heun scheme

Each scan step advances the state by `step_size = 1 / steps_per_unit`:

1. **Predictor** — evaluate intensities at clock time `t` (nudged by `±perturbation`; see Intensity protocol §Discontinuity handling), compute per-state derivatives (outflows from `density` and `point_mass`, inflows to `density`), take an Euler step.
2. **Corrector** — evaluate intensities at `t + step_size`, recompute derivatives, average with the predictor's derivatives.
3. **Duration shift** — mass at duration `k` becomes mass at duration `k+1`; slot 0 of `density` receives fresh inflow from other states; slot 0 of `point_mass` is zeroed. Mass reaching the final slot is truncated (the grid is sized so this corresponds to "in-state since `t=0`").

### Numerical order

**Second-order on smooth intensities**; **first-order across finite jumps** regardless of `perturbation`. Resolving the discontinuity protocol to sub-step around declared break points would restore second order everywhere (see Intensity protocol §Discontinuity handling).

### JIT boundary

| Static (trace-time constants) | Traced (runtime values) |
|---|---|
| Matrix sparsity pattern (positions of `None` cells) | Covariate arrays (`**kwargs`) |
| Callback function | Fitted parameters captured in closures |
| Presence/absence of `point_mass` per state | |
| `step_size`, `record_every`, `perturbation` | |

Changing any static field triggers a re-trace. Rebuilding a `Model` with a different sparsity pattern re-traces; changing only parameter values inside existing callables does not.

### Memory budget

Peak output memory in bytes, at float32:

```
bytes ≈ 4 * T_out * product(callback_output_non_time_dims)
```

where `T_out = (horizon * steps_per_unit) // record_every + 1`. Worked examples at `batch = 100_000`, `J = 10`, `D = 360`, `T_out = 361`:

- `default` (leaves `(time, batch, D)` per state's density + point mass): dominated by `4 * T_out * batch * J * D` ≈ **520 GB**. Infeasible at this resolution.
- Same, but `record_every = 12` (`T_out = 31`): ≈ **44 GB**. Still large; tractable on a big GPU.
- `collapse_point_no_duration` (`(T_out, batch, J)`): ≈ **1.4 GB**. Comfortable.

Pick `callback` and `record_every` before scaling `batch`.

### Open design questions

The following items are intentionally left open by this spec and tracked in `docs/design/solver.md`:

1. **Discontinuity handling protocol** — see Intensity protocol §Discontinuity handling.
2. **Per-state duration depth `D_j`** — currently uniform `D = horizon * steps_per_unit`. The pytree state structure allows per-state `D_j` as a future optimisation; Markov states would collapse to `D_j = 1`.
3. **Heterogeneous `d_0` and `initial_distribution`** — `initial_duration` is a reserved kwarg; a richer `initial_distribution` object encoding state + duration + mass per individual is forward-looking.
4. **Per-state point mass for non-initial states** — the pytree allows `point_mass` on any state, but initial-condition construction currently seeds only state 0.

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
| `"default"` | Full pytree state, no reduction | `tuple[StateCarry, ...]` — each leaf `(batch, D)` (or `None` for absent point masses) |
| `"no_duration"` | Marginalise over duration, preserve pytree | Per state: `(density[..., -1], point_mass[..., -1] or None)` — each leaf `(batch,)` |
| `"collapse_point"` | Fold `point_mass` into `density[..., 0]` per state; drop `point_mass` | Tuple of per-state `density` — each leaf `(batch, D)` |
| `"collapse_point_no_duration"` | Collapse then marginalise; re-stack across states | Single array `(batch, J)` |
| `"point_only"` | Per-state `point_mass` (or `None`) | Tuple per state — each leaf `(batch, D)` or `None` |
| `"point_only_no_duration"` | Per-state `point_mass[..., -1]` (or `None`) | Tuple per state — each leaf `(batch,)` or `None` |
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
            total = total + jnp.sum(carry.point_mass, axis=-1)
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

2. **Uniform callable interface.** The solver doesn't know or care whether an intensity comes from a Gompertz function, a GLM, or a neural network. All it sees is `(t, d, **kwargs) → array`.

3. **Compute only what's needed.** Given an initial state, the solver reduces to the reachable subgraph. Unreachable states are excluded entirely.

4. **Fail early, fail clearly.** Validation happens at `StateSpace` construction and `Model.build()` time, not deep inside the solver. Error messages reference state names and transitions, not matrix indices.

5. **JIT everything.** The entire pipeline from covariates to transition probabilities compiles into a single XLA program. No Python callbacks inside the solver loop.

6. **Batch-first.** The framework is designed for 100K+ individuals in a single pass. Covariates are arrays, not scalars. The solver vectorizes over the batch dimension.

7. **Point mass and continuous density are separate objects.** Per state, the absolutely continuous duration density and the point mass at duration zero are tracked independently. They have different physics (advection-reaction with duration shift vs. scalar exponential decay along a characteristic); co-evolving them cleanly keeps the door open for heterogeneous initial durations, per-state point masses, and analytic or high-order integration for the point mass.

---

## Future work

- **Discontinuity handling protocol**: declared break points, piecewise callables, or a left/right-evaluation protocol, resolving the open question in the Intensity protocol section and restoring 2nd-order convergence across jumps.
- **Per-state duration depth `D_j`**: let each reachable state pick its own duration depth, collapsing Markov states to `D_j = 1`. Enabled by the pytree solver state.
- **Heterogeneous initial duration `d_0`**: per-individual starting durations via an `initial_duration` kwarg on `solve()`; later, a richer `initial_distribution` object encoding state + duration + mass per individual.
- **Per-state point masses**: generalise the initial-condition construction so states other than the initial one can carry their own point mass (already supported in the solver state shape, just not in the initialiser).
- **Pre-computation protocol**: two-phase `prepare`/`evaluate` for intensity models with static covariate contributions.
- **Built-in parametric hazards**: Gompertz, Weibull, piecewise constant, and other standard forms in `jact.intensity`.
- **Cashflow computation**: integral transforms over the duration density for actuarial present values, extending the callback system.
