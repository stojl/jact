# jact — API specification

## Overview

`jact` is a JAX framework for computing transition probabilities in multi-state models with duration-dependent transition intensities. It is built for the pipeline from fitted intensity models to probabilities for large cohorts.

The current implementation separates three concerns:

- **StateSpace**: the structural definition of states and allowed transitions.
- **Model**: a `StateSpace` bound to intensity callables.
- **Solver**: a midpoint-quadrature kernel over the reachable subgraph.

This specification also describes a planned cashflow declaration layer:

- **cashflows**: a reusable declaration of named cashflow components built from a `StateSpace`.

All intensity models must be JIT-compatible. The full pipeline from covariates to transition probabilities compiles into a single XLA program.

## Module layout

```text
jact/
├── __init__.py              # Public API: StateSpace, Model, InitialDistribution, solve, callbacks
├── state_space.py           # StateSpace class + InitialDistribution helpers
├── model.py                 # Model, ReducedModel, TransitionInfo
├── initial_distribution.py  # InitialDistribution class
├── solver.py                # Semi-Markov solver
└── callbacks.py             # Probability output callbacks
```

The cashflow API described below is planned but not yet implemented, so no dedicated cashflow module is part of the current package layout.

## StateSpace

`StateSpace` defines topology only. It carries no data and no intensity models.

```python
state_space = jact.StateSpace(
    states=["healthy", "disabled", "dead"],
    transitions=[
        ("healthy", "disabled"),
        ("healthy", "dead"),
        ("disabled", "dead"),
    ],
)
```

Construction validates:

- no duplicate state names,
- no transitions to unknown states,
- no self-transitions,
- no duplicate transitions.

Surface:

```python
state_space.states
state_space.n_states
state_space.transitions
state_space.absorbing
state_space.transient
state_space.exits("healthy")
state_space.targets("healthy")
state_space.sources("dead")
state_space.has_transition("healthy", "dead")
state_space.state_index("disabled")
state_space.reachable_from("healthy")
```

Reachability is used by the solver to reduce work to the relevant subgraph.

Serialization:

```python
state_space.to_json("model.json")
loaded = jact.StateSpace.from_json("model.json")
```

## Model

`Model` binds intensity callables to a `StateSpace`. It is created through `StateSpace.build()` and is immutable.

### Building a model

Every declared transition must be assigned exactly once across these three kwargs:

```python
model = state_space.build(
    transitions={...},
    exits={...},
    groups={...},
)
```

Assignment modes:

| Kwarg | Coverage | Callable return shape |
|---|---|---|
| `transitions={(src, tgt): fn}` | One transition | `(batch, D)` |
| `exits={src: fn}` | All exits from `src`, ordered by `state_space.targets(src)` | `(n_targets, batch, D)` |
| `groups={fn: [(src, tgt), ...]}` | Arbitrary set, in listed order | `(n_transitions, batch, D)` |

Examples:

```python
model = state_space.build(
    transitions={
        ("healthy", "disabled"): onset_fn,
        ("disabled", "dead"): disabled_mortality_fn,
    },
    groups={
        joint_mortality_fn: [
            ("healthy", "dead"),
            ("recovered", "dead"),
        ],
    },
)
```

`exits` always covers every exit from the given source state. For partial coverage, use `groups`.

### Reduction

`Model.reduce(initial_states)` extracts the reachable subgraph from one or more declared initial states:

```python
reduced = model.reduce("disabled")

reduced.initial_states
reduced.reachable_states
reduced.n_states
reduced.solver_matrix
```

Initial states occupy the first reduced indices in state-space order; remaining reachable states follow in their original order.

### Transition metadata

`model.info(src, tgt)` returns:

```python
TransitionInfo(source, target, assignment, callable, index)
```

`index` is `None` for single-transition assignments and the slice index for `exits` / `groups`.

## InitialDistribution

`InitialDistribution` encodes the joint `(state, duration)` distribution at `t = 0`.

Construction patterns:

```python
jact.InitialDistribution(
    components={
        "healthy": {"mass": mass_h, "duration": d_h},
        "disabled": {"mass": mass_d, "duration": d_d},
    },
    normalise=True,
)

jact.InitialDistribution.at("healthy", duration=0.0)

jact.InitialDistribution.per_individual(
    states=idx_array,
    duration=d_0_array,
    initial_states=None,
)
```

Rules:

- `per_individual.states` is a traced `(batch,)` int32 index array.
- `initial_states=<tuple>` means indices are into that tuple and the solver reduces to the union of states reachable from it.
- `initial_states=None` means indices are into the full model state list and no reduction is performed.
- `initial_duration` on `solve()` is valid only with the `str` / `(batch,)` `initial` shortcuts.
- Masses are normalised per individual by default.

State-space helpers perform eager name validation:

```python
state_space.initial_at(...)
state_space.initial_per_individual(...)
state_space.initial_distribution(...)
```

## Intensity protocol

Every intensity callable has the same interface:

```python
def intensity(t, d, **kwargs) -> jnp.ndarray: ...
```

Arguments:

| Arg | Type | Meaning |
|---|---|---|
| `t` | scalar float | Clock time |
| `d` | `(1, D)` | Duration grid broadcast over batch |
| `**kwargs` | `(batch, ...)` arrays | Covariates supplied at `solve()` |

`t` and `d` play distinct roles:

- `t` is clock time,
- `d` is duration in the current state,
- attained-age effects are expressed as covariates like `baseline_age + t`.

Return shapes:

| Assignment | Shape |
|---|---|
| `transitions` | `(batch, D)` |
| `exits` | `(n_targets, batch, D)` |
| `groups` | `(n_transitions, batch, D)` |

JAX requirements:

- pure: no side effects or mutation,
- JIT-compatible: no data-dependent Python control flow or non-JAX ops,
- closed over static values only.

## Cashflows (planned)

This section specifies the intended public cashflow API. It is a documentation-only plan at present and is not implemented in the current package.

### Cashflow declaration layer

Cashflows are declared from a `StateSpace`, not from a `Model`:

```python
cashflows = state_space.cashflows(
    components={...},
)
```

The returned object is a reusable cashflow declaration built against the state-space topology. The exact concrete class name is not yet fixed.

The cashflow declaration answers:

- which named cashflow components exist,
- what kind of component each is,
- where each component attaches,
- what callable defines the payment amount.

Aggregation, valuation, cumulative totals, and terminal totals are not declared here. Those remain solve-time or post-processing concerns.

Validation is structural and uses the `StateSpace` only:

- state-rate attachments must reference declared states,
- transition-lump attachments must reference declared transitions,
- scheduled-event state attachments must reference declared states,
- component names must be unique.

### Component-first grammar

The top-level public grammar is component-first:

```python
cashflows = state_space.cashflows(
    components={
        "premium": {
            "kind": "state_rate",
            "states": {
                "healthy": premium_fn,
            },
        },
        "death_benefit": {
            "kind": "transition_lump",
            "transitions": {
                ("healthy", "dead"): death_fn,
            },
        },
        "retirement_bonus": {
            "kind": "scheduled_event",
            "when": event_time_fn,
            "states": {
                "healthy": bonus_fn,
            },
        },
    },
)
```

Each component name identifies one raw cashflow stream. Components are the basis for later aggregation and reporting.

### Component kinds

#### State-rate

`"kind": "state_rate"` attaches payment-rate callables to states.

Recommended v1 attachment shape:

```python
{
    "kind": "state_rate",
    "states": {
        "healthy": premium_fn,
        "disabled": waiver_fn,
    },
}
```

Interpretation: expected cashflow is generated continuously while the policy occupies the attached state.

#### Transition-lump

`"kind": "transition_lump"` attaches lump-sum payment callables to declared transitions.

Recommended v1 attachment shape:

```python
{
    "kind": "transition_lump",
    "transitions": {
        ("healthy", "dead"): death_fn,
        ("healthy", "disabled"): onset_fn,
    },
}
```

Interpretation: expected cashflow is generated when the attached transition occurs.

#### Scheduled-event

`"kind": "scheduled_event"` declares a deterministic event-time rule plus state-conditioned payment callables.

Recommended v1 attachment shape:

```python
{
    "kind": "scheduled_event",
    "when": event_time_fn,
    "states": {
        "healthy": bonus_fn,
        "disabled": reduced_bonus_fn,
    },
}
```

Interpretation: expected cashflow is generated at the component's deterministic event time, conditional on the occupied state at that time.

### Callable protocols

Payment functions for `state_rate`, `transition_lump`, and `scheduled_event` share one protocol:

```python
def payment(t, d, **kwargs) -> jnp.ndarray: ...
```

Arguments:

| Arg | Type | Meaning |
|---|---|---|
| `t` | scalar float | Clock time at which the payment is evaluated |
| `d` | duration representation | Duration in the current state |
| `**kwargs` | `(batch, ...)` arrays | Solve-time covariates |

Interpretation depends on component kind:

- for `state_rate`, `payment(t, d, **kwargs)` is a rate while occupying the state,
- for `transition_lump`, it is a lump amount if the transition occurs at `(t, d)`,
- for `scheduled_event`, it is a payment amount evaluated only at the declared event time.

Scheduled-event time rules use a separate protocol:

```python
def when(**kwargs) -> jnp.ndarray: ...
```

Return value:

- shape `(batch,)`,
- one deterministic event time per individual in v1.

The event-time rule is user-defined and may depend on the same solve-time covariates passed to intensities and payment functions. The package does not define business-specific helpers such as `AtAge(70)`.

### Scheduled-event policy in v1

The planned first version keeps scheduled deterministic events intentionally narrow:

- event times may be data-dependent per individual,
- `when(**kwargs)` returns exactly one event time per individual,
- event times must be aligned to the solver grid,
- off-grid event times are rejected,
- multiple event times per component are future work.

If an event lands exactly on a solver grid point, the left-versus-right timing convention remains an open design point.

### Aggregation and valuation

The declared cashflow object preserves raw named components. It does not itself decide:

- how components are grouped,
- whether outputs are raw, cumulative, or terminal,
- whether valuation or discounting is applied.

Those concerns remain outside the declaration object. The planned v1 solve surface focuses on streamed cashflow output only.

## Solver

For a reduced model rooted at the declared initial-state set, the solver advances a probability state over `[0, horizon]` inside one `jax.lax.scan`, vectorized over the batch axis.

### Solver state

Per reachable state, the solver tracks:

- `density`: the duration density on the solver grid, shape `(batch, D)`,
- `point_mass`: either `None` or `PointMass(value, d_0)` with `value` and `d_0` shape `(batch,)`.

```python
class StateCarry(NamedTuple):
    density: jnp.ndarray
    point_mass: PointMass | None
```

The full solver state is a tuple of `StateCarry`, one per reachable state.

### Calling solve

Current implemented call surface:

```python
result = model.solve(
    initial="healthy",
    horizon=10,
    steps_per_unit=12,
    callback="collapse_point_no_duration",
    record_every=1,
    age=age_array,
)
```

Or:

```python
result = jact.solve(model, initial=..., horizon=..., steps_per_unit=..., **covariates)
```

Parameters:

| Parameter | Type | Description |
|---|---|---|
| `initial` | `str`, `(batch,)` int array, or `InitialDistribution` | Initial condition |
| `initial_duration` | float or `(batch,)` array | Per-individual `d_0` for `str` / `(batch,)` initial forms |
| `horizon` | int | Number of time units |
| `steps_per_unit` | int | Discretisation resolution per time unit |
| `callback` | `str`, callable, or `None` | Probability callback |
| `record_every` | int | Must divide `horizon * steps_per_unit` |
| `**kwargs` | arrays | Covariates with a shared leading batch dimension |

Result:

```python
result["probability"]
result["states"]
```

### Planned cashflow solve extension

The intended cashflow API extends the solve surface rather than introducing a separate cashflow solver:

```python
result = model.solve(
    initial="healthy",
    horizon=10,
    steps_per_unit=12,
    probability="collapse_point_no_duration",
    cashflows=cashflows,
    cashflow_groups={"benefits": ["death_benefit", "retirement_bonus"]},
    record_every=1,
    age=age_array,
)
```

Planned semantics:

| Parameter | Type | Meaning |
|---|---|---|
| `probability` | `str`, callable, or `None` | Probability reporting control |
| `cashflows` | cashflow declaration or `None` | Named cashflow components to evaluate |
| `cashflow_groups` | mapping or `None` | Solve-time aggregation of named components |

`probability=None` disables probability output.
`cashflows=None` disables cashflow output.

Planned result keys:

```python
result["probability"]
result["cashflows"]
result["states"]
```

The exact disabled-output convention, such as omitted key versus `None`, is not yet fixed.

### Per-step update

Each solver step uses midpoint quadrature along the transported characteristic:

```text
A_ij^(n)[k] = dt · μ_ij(t_n + dt / 2, d_k + dt / 2)
```

for density mass.

Point masses follow the same transported characteristic as the density:

- hazards are evaluated at the midpoint sample `(t + dt / 2, d_0 + t + dt / 2)`,
- the point-mass value decays by the same competing-risks update used for density survival,
- outgoing point-mass mass is routed into target densities at duration zero.

The step then:

1. Computes all per-transition integrated hazards `A_ij`.
2. Aggregates exits from each source state into one competing-risks update.
3. Forms survival `S_i = exp(-Σ_j A_ij)` and stable transfer fractions.
4. Shifts surviving density one duration slot to the right.
5. Injects transferred mass into duration zero.
6. Decays point masses along `(t, d_0 + t)` and transfers their outgoing mass into target densities.

Built-in callbacks keep their existing names and expose point masses directly from the solver state:

- `collapse_point` and `collapse_point_no_duration` add point-mass value to the reported state total,
- `point_only` and `point_only_no_duration` show the current point mass directly,
- `no_point` and `no_point_no_duration` continue to exclude point mass entirely.

## Numerical contract

- Midpoint is second-order when the hazard is smooth on the traversed characteristic segment.
- Midpoint remains globally second-order for a callable if all jumps in `t` or `d` lie on solver grid lines.
- If a jump lies strictly inside a traversed cell, convergence for that callable can drop to first order.
- This tradeoff is explicit: the solver does not expose split or jump locations through the callable API.
- For piecewise-constant, tree-based, or other discontinuous hazards, align split points in `t` and `d` to the solver grid when possible.

## JIT boundary

Static:

- sparsity pattern of the reduced transition matrix,
- callback function,
- presence or absence of `point_mass` per state,
- declared set of initial states,
- `step_size`,
- `record_every`.

Traced:

- covariate arrays,
- parameters captured in closures,
- per-individual masses and durations,
- `PointMass.value`,
- `PointMass.d_0`,
- per-individual initial-state index arrays.

Changing any static field retraces. Changing only parameter values inside an existing callable does not.
