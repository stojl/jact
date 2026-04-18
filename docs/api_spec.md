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

All three arguments can be used in a single `build()` call. The `StateSpace` validates that every declared transition is assigned exactly once — no gaps, no overlaps.

```python
model = state_space.build(
    exits={
        "healthy": joint_onset_and_mortality_model,
    },
    groups={
        shared_model: [("disabled", "healthy"), ("disabled", "dead")],
    },
)
```

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

Intensity callables are **pure functions**. No class protocol or two-phase interface is supported — a callable is the only contract. Fitted model parameters are captured via closures.

### Call signature

```python
def intensity(t, d, **kwargs) -> jnp.ndarray:
    ...
```

| Argument | Type | Description |
|---|---|---|
| `t` | `float` scalar | Current clock time. Ranges from `0` to `horizon` as the solver marches forward. |
| `d` | `jnp.ndarray`, shape `(1, D)` | Duration grid points. Each entry is the time spent in the current state at that duration slot. The `1` axis broadcasts over the batch dimension. |
| `**kwargs` | `jnp.ndarray`, shape `(batch, ...)` | Covariate arrays passed through from `solve()`. Each callable consumes the subset it needs; unused kwargs are ignored. |

#### On `t` and `d`

The solver maintains a probability density over duration `d` for each state and advances it in clock time `t`. At each solver step the callable receives the current `t` and the full duration array `d` simultaneously, and must return the intensity surface `μ(t, d)` over all `D` duration points in a single call.

`t` and `d` play distinct roles. For a population observed at `t=0` with known baseline ages, attained age at clock time `t` is `baseline_age + t` — age advances with clock time, not with duration in state. Duration `d` enters separately when the intensity depends on how long the individual has been in the current state, which is the semi-Markov component. A Markov intensity uses only `t`; a pure duration-dependent intensity uses only `d`; a semi-Markov intensity uses both.

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

---

## Solver

### Calling the solver

```python
result = model.solve(
    initial="healthy",
    horizon=10,
    steps_per_unit=12,
    age=age_array,
    bmi=bmi_array,
)
```

Or equivalently via the functional interface:

```python
result = jact.solve(
    model,
    initial="healthy",
    horizon=10,
    steps_per_unit=12,
    age=age_array,
    bmi=bmi_array,
)
```

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `initial` | `str` | Starting state. Only reachable states are computed. |
| `horizon` | `int` | Number of time units to solve over. |
| `steps_per_unit` | `int` | Discretization resolution per time unit. |
| `callback` | `str`, `callable`, or `None` | Probability callback (default: `"collapse_point_no_duration"`). |
| `perturbation` | `float` | Grid perturbation for finite differences (default: `1e-12`). |
| `transpose_result` | `bool` | Transpose time axis in result (default: `True`). |
| `**kwargs` | `jnp.ndarray` | Covariate arrays, each of shape `(batch, ...)`. |

### Result

```python
result["probability"]   # Transition probabilities (shape depends on callback)
result["states"]        # Tuple of state names in result ordering
```

The `"states"` key tells you which state corresponds to which index in the probability array. The initial state is always first.

```python
result = model.solve(initial="disabled", horizon=30, steps_per_unit=12, age=ages)
result["states"]        # ("disabled", "dead")
result["probability"]   # Only 2 states computed, not 3
```

### Solver internals

The solver implements a Heun scheme (second-order predictor-corrector) using `jax.lax.scan`. At each time step:

1. Evaluate intensity functions at the current `(t, d)`.
2. Compute outflows (probability leaving each state) and inflows (probability entering each state).
3. Predict the next state using an Euler step.
4. Evaluate intensities at the predicted state.
5. Correct using the average of the two derivatives.
6. Shift the duration axis (probability at duration d moves to d+1).

The point mass at duration zero (initial state probability) is tracked separately for numerical accuracy and handled internally by the solver. Users do not interact with it directly; the callback system abstracts over it.

---

## Callbacks

Callbacks control what is extracted from the solver's internal state at each time step. They determine the shape and content of `result["probability"]`.

### Built-in callbacks

| Name | Description | Output shape |
|---|---|---|
| `"default"` | Full density and point mass, no reduction | `(p, p_point)` |
| `"no_duration"` | Marginalize over duration | `(batch, J)`, `(batch,)` |
| `"collapse_point"` | Collapse point mass into first state's density | `(batch, J, D)` |
| `"collapse_point_no_duration"` | Collapse + marginalize (most common) | `(batch, J)` |
| `"point_only"` | Point mass only | `(batch, D)` |
| `"point_only_no_duration"` | Point mass, marginalized | `(batch,)` |
| `"no_point"` | Density only | `(batch, J, D)` |
| `"no_point_no_duration"` | Density, marginalized | `(batch, J)` |
| `"none"` | Record nothing | `None` |

### Custom callbacks

A callback receives the raw solver state and returns an arbitrary PyTree:

```python
def my_callback(p, p_point):
    """
    p:       shape (batch, n_states, D)  — duration density per state
    p_point: shape (batch, D)            — point mass for initial state
    """
    return jnp.sum(p[:, 0, :], axis=-1) + jnp.sum(p_point, axis=-1)
```

The callback system is also the extension point for future features like cashflow computation, which involves integral transforms over the duration density.

---

## Full example

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

---

## Future work

- **Pre-computation protocol**: two-phase `prepare`/`evaluate` for intensity models with static covariate contributions.
- **Built-in parametric hazards**: Gompertz, Weibull, piecewise constant, and other standard forms in `jact.intensity`.
- **Cashflow computation**: integral transforms over the duration density for actuarial present values, extending the callback system.
