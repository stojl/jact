# jact — API specification

## Overview

`jact` is a JAX framework for transition probabilities in multi-state models
with duration-dependent intensities. It is designed for the pipeline from
fitted hazard callables to probabilities and expected cashflow streams for
large cohorts.

The public API has four layers:

- `StateSpace`: states and allowed transitions only.
- `Model`: a `StateSpace` bound to intensity callables.
- `CashflowDeclaration`: reusable named cashflow components declared from a
  `StateSpace`.
- `solve()`: a midpoint-quadrature solver over the reachable subgraph that can
  emit both probabilities and cashflows in one fused JAX program.

Use `import jact` for the main surface: `jact.StateSpace`,
`jact.InitialDistribution`, `jact.solve`, `jact.StateRate`,
`jact.TransitionLump`, `jact.ScheduledEvent`, `jact.Raw`, `jact.Group`,
`jact.Total`, `jact.ByState`, and `jact.ByKind`.
Advanced inspection and callback objects stay in submodules such as
`jact.callbacks.StateCarry`, `jact.callbacks.PointMass`,
`jact.model.ReducedModel`, and `jact.model.TransitionInfo`.

Files under `archive/original_prototype/` and `notes/` are background material
only. They are not part of the public contract.

## StateSpace

`StateSpace` defines topology only. It carries no intensities and no data.

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

| Name | Meaning |
|---|---|
| `states` | Tuple of state names in declared order |
| `n_states` | Number of states |
| `transitions` | `frozenset[(src, tgt)]` of declared transitions |
| `absorbing` | States with no outgoing transitions |
| `transient` | States with outgoing transitions |
| `exits(state)` | Outgoing transitions from `state`, ordered by target-state order |
| `targets(state)` | Outgoing target states from `state`, ordered by state order |
| `sources(state)` | Source states with a transition into `state`, ordered by state order |
| `has_transition(src, tgt)` | Whether `(src, tgt)` is declared |
| `state_index(state)` | Zero-based index in `states` |
| `reachable_from(state)` | Starting state first, then reachable states in original state order |
| `build(transitions=..., exits=..., groups=...)` | Create a `Model` |
| `cashflows({...})` | Create a `CashflowDeclaration` |
| `initial_at(state, duration=0.0)` | Create an `InitialDistribution` |
| `initial_distribution(components=..., normalise=True)` | Create an `InitialDistribution` |
| `initial_per_individual(...)` | Create an `InitialDistribution` from per-individual initial states |

Serialization:

```python
state_space.to_json("model.json")
loaded = jact.StateSpace.from_json("model.json")
```

## Model

`Model` binds intensity callables to a `StateSpace`. Create it via
`state_space.build(...)`.

### Building a model

Every declared transition must be assigned exactly once across these three
kwargs:

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
| `transitions={(src, tgt): fn}` | Exactly one transition | `(batch, D)` |
| `exits={src: fn}` | Every exit from `src`, ordered by `state_space.targets(src)` | `(n_targets, batch, D)` |
| `groups={fn: [(src, tgt), ...]}` | Arbitrary listed transitions | `(n_transitions, batch, D)` |

Notes:

- `exits` always means all exits from that source state. For partial coverage,
  use `groups`.
- `build()` rejects gaps, overlaps, unknown transitions, empty groups, and
  non-callable assignments.
- `exits` and `groups` are sliced at model-build time so the solver always sees
  per-transition callables returning `(batch, D)`.

### Reduction

`Model.reduce(initial_states)` extracts the reachable subgraph from a declared
initial-state set.

```python
reduced = model.reduce(("healthy", "disabled"))

reduced.initial_states
reduced.reachable_states
reduced.n_states
reduced.solver_matrix
```

Rules:

- `initial_states` may be a single state name or an iterable of state names.
- duplicates are rejected.
- reduced indices place declared initial states first in state-space order; the
  remaining reachable states follow in their original state-space order.

### Transition metadata

`model.info(src, tgt)` returns:

```python
TransitionInfo(source, target, assignment, callable, index)
```

`index` is `None` for `transitions` assignments and the output slot index for
`exits` and `groups`.

## InitialDistribution

`InitialDistribution` encodes the joint `(state, duration)` distribution at
`t = 0`.

Construction:

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

Key rules:

- `mass` and `duration` values may be scalars or `(batch,)` arrays.
- masses are normalised per individual by default.
- `per_individual.states` is a traced `(batch,)` integer array.
- `per_individual.initial_states=<tuple>` means the indices are into that tuple
  and the solver reduces to the union of states reachable from it.
- `per_individual.initial_states=None` means the indices are into the model's
  full state list and no reduction is performed.
- the declared initial-state set is structural. It is the keys of
  `components`, the state passed to `at()`, or the `initial_states` tuple on
  `per_individual()`. It is never inferred from runtime masses or runtime index
  values.
- declaring a state with zero mass still keeps that state in the structural
  initial-state set.

State-space helpers perform eager name validation:

```python
state_space.initial_at(state, duration=0.0)
state_space.initial_distribution(components, normalise=True)
state_space.initial_per_individual(
    state_names=...,        # OR state_indices=...; exactly one is required
    state_indices=...,
    duration=0.0,
    initial_states=None,
)
```

`initial_per_individual()` is keyword-only. `state_names` are validated against
the state space and converted host-side. `state_indices` are a `(batch,)`
integer array indexing into `initial_states` if that tuple is supplied,
otherwise into the model's full state list.

## Intensity protocol

Every intensity callable has this interface:

```python
def intensity(t, d, **kwargs) -> jnp.ndarray: ...
```

Arguments:

| Arg | Type | Meaning |
|---|---|---|
| `t` | scalar float | Clock time |
| `d` | `(1, D)` | Duration grid broadcast over batch |
| `**kwargs` | `(batch, ...)` arrays | Solve-time covariates |

Interpretation:

- `t` is clock time,
- `d` is duration in the current state,
- attained-age or calendar-time effects are expressed through covariates such
  as `baseline_age + t`.

Return shapes:

| Assignment mode | Shape |
|---|---|
| `transitions` | `(batch, D)` |
| `exits` | `(n_targets, batch, D)` |
| `groups` | `(n_transitions, batch, D)` |

All intensity callables must be pure and JIT-compatible.

## Cashflows

### Declaration

Cashflows are declared from a `StateSpace`, not from a `Model`:

```python
cashflows = state_space.cashflows({
    "premium": jact.StateRate({"healthy": premium_fn}),
    "death": jact.TransitionLump({("healthy", "dead"): death_fn}),
    "bonus": jact.ScheduledEvent(
        when=event_time_fn,
        payments={"healthy": bonus_fn},
    ),
})
```

The returned object is a `CashflowDeclaration` with this surface:

```python
cashflows.state_space
cashflows.names
cashflows.component("premium")
```

Validation is structural:

- `cashflows()` requires a non-empty component mapping,
- component names must be unique non-empty strings,
- `StateRate` keys must be declared states,
- `TransitionLump` keys must be declared transitions,
- `ScheduledEvent.payments` keys must be declared states,
- every payment callable and every `ScheduledEvent.when` callable must be
  callable.

### Component kinds

- `StateRate(payments)` records expected payment rate while occupying an
  attached state.
- `TransitionLump(payments)` records expected payment when an attached
  transition occurs.
- `ScheduledEvent(when=..., payments=...)` records expected payment at one
  deterministic event time per individual, conditional on the occupied state at
  that time.

`StateRate` and `TransitionLump` take their `payments` mapping positionally.

### Callable protocols

Payment callables for all component kinds share one interface:

```python
def payment(t, d, **kwargs) -> jnp.ndarray: ...
```

The return shape is always `(batch, D)`.

Scheduled-event timing uses a separate rule:

```python
def when(**kwargs) -> jnp.ndarray: ...
```

The return shape is `(batch,)`: one event time per individual.

### Scheduled-event policy

Scheduled events are intentionally narrow:

- one event time per individual,
- event times may depend on solve-time covariates,
- times near a solver grid point are snapped to that grid point to absorb
  floating-point noise,
- other off-grid times are floored to the step that starts before the event,
- `event_time < 0` or an effective step outside the solver grid produces no
  payment,
- an event at the right boundary `t = horizon` produces no payment because no
  solver step starts there,
- payment evaluation uses the left endpoint `t_n` and the left duration grid,
  not the midpoint sample,
- state occupancy uses the pre-step convention: the event sees the state before
  transitions for that step are applied.

Multiple event times per component are out of scope.

### Recording semantics

Cashflow streams use interval accumulation:

- each streamed entry is the sum of all inner-step contributions in that record
  interval,
- scheduled events contribute to the unique interval containing their effective
  event time,
- probability output uses snapshot semantics instead and includes time zero.

Host-side post-processing:

- cumulative streamed output: `jnp.cumsum(stream, axis=0)`,
- terminal-from-stream output: `stream.sum(axis=0)`.

### Views

Aggregation and time-only weighting are declared per solve through
`cashflow_views`:

```python
cashflow_views = {
    "raw": jact.Raw(),
    "benefits": jact.Group(["death", "bonus"]),
    "total": jact.Total(),
    "by_state": jact.ByState(),
    "by_kind": jact.ByKind(),
}
```

View types:

| View | Constructor | Output leaves |
|---|---|---|
| `Raw` | `Raw(name=None, *, weight=None, terminal=False)` | All declared components when `name is None`; one named component otherwise |
| `Group` | `Group(members, *, weight=None, terminal=False)` | One leaf summing the named components |
| `Total` | `Total(*, weight=None, terminal=False)` | One leaf summing all components |
| `ByState` | `ByState(*, weight=None, terminal=False)` | One leaf per reachable state in the reduced solve |
| `ByKind` | `ByKind(*, weight=None, terminal=False)` | One leaf per component kind |

Shared view fields:

- `weight`: `None`, a Python scalar, a rank-0 array, or a callable
  `(t, **kwargs) -> (batch,)` or broadcast-compatible array. It is evaluated
  once per inner solver step at that step's midpoint and multiplies the
  contribution attributed to that step.
- discounting is one example of user-supplied weighting logic, for example
  `weight=lambda t, **kwargs: jnp.exp(-0.03 * t)`.
- more complex term-structure or valuation logic belongs in user code passed
  through `weight=`; `jact` does not provide a built-in discounting helper.
- `terminal`: `False` streams one entry per `record_every` interval.
  `True` collapses the time axis and returns one `(batch,)` accumulator per
  leaf.

Default and validation rules:

- if `cashflows` is supplied and `cashflow_views` is omitted or `None`, the
  solver uses `{"raw": Raw()}`,
- if `cashflows is None`, any non-`None` `cashflow_views` is rejected,
- `cashflow_views={}` is allowed and returns an empty `result["cashflows"]`
  mapping,
- view names must be unique non-empty strings,
- `Raw(name=...)` and `Group(members)` must reference declared component names,
- `Group.members` is frozen during validation,
- `terminal` must be `bool`,
- non-scalar array weights are rejected.

Semantics:

- `Raw()` returns every declared component.
- `Raw(name)` returns one named component.
- `Group(members)` returns the sum of the named components.
- `Total()` returns the sum of all components.
- `ByState()` returns one key per reachable state in reduced order, including
  zero-valued leaves for reachable states with no contributions. `StateRate`
  contributes to the occupied state, `TransitionLump` contributes to the source
  state, and `ScheduledEvent` contributes to the pre-step occupied state.
- `ByKind()` returns keys `"state_rate"`, `"transition_lump"`, and
  `"scheduled_event"`, including zero-valued leaves when a kind is absent.

## Solver

Call through `model.solve(...)` or `jact.solve(model, ...)`.

```python
def flat_discount(t, **kwargs):
    return jnp.exp(-0.03 * t)


result = model.solve(
    initial="healthy",
    horizon=10,
    steps_per_unit=12,
    probability="state_probability",
    cashflows=cashflows,
    cashflow_views={"pv": jact.Total(weight=flat_discount, terminal=True)},
    record_every=12,
    age=age_array,
)
```

Parameters:

| Parameter | Type | Meaning |
|---|---|---|
| `initial` | `str`, `(batch,)` int array, or `InitialDistribution` | Initial condition |
| `initial_duration` | float or `(batch,)` array | Only valid with `str` and `(batch,)` `initial` forms |
| `horizon` | positive int | Number of time units |
| `steps_per_unit` | positive int | Solver steps per time unit |
| `probability` | `str`, callable, or `None` | Probability reducer; default is `"state_probability"` |
| `cashflows` | `CashflowDeclaration` or `None` | Cashflow components to evaluate |
| `cashflow_views` | mapping or `None` | Solve-time views; requires `cashflows` |
| `record_every` | positive int | Must divide `horizon * steps_per_unit` |
| `**kwargs` | arrays | Covariates with a shared leading batch dimension |

Validation and defaults:

- `initial_duration` with an `InitialDistribution` is rejected,
- reserved covariate names `initial` and `initial_duration` are rejected,
- legacy kwargs `callback` and `freeze_initial` are rejected,
- `cashflows` must be declared from `model.state_space`,
- covariate values must have shape `(batch, ...)`; scalars are rejected,
- `record_every` must divide `horizon * steps_per_unit`.

Initial forms:

- `initial="healthy"` means all individuals start in that state at duration
  `initial_duration`,
- `initial=idx_array` means a traced `(batch,)` integer array indexing into the
  model's full state list and performs no reduction,
- `initial=InitialDistribution(...)` gives full control over masses, durations,
  and the declared initial-state set used for reduction.

Result keys:

```python
result["states"]        # always present
result["probability"]   # omitted when probability=None
result["cashflows"]     # omitted when cashflows is None
```

`result["states"]` is the tuple of reachable states in reduced order.
Disabled outputs are omitted entirely rather than filled with `None`.
`probability="none"` is different from `probability=None`: the string selects a
callback that returns `None`, while `None` disables probability output and
omits the key.

### Output shapes

Let `T_out = horizon * steps_per_unit / record_every`.

- probability output uses snapshot semantics and has a leading time axis of
  length `T_out + 1` because it includes the initial state at `t = 0`,
- streamed cashflow leaves have leading shape `(T_out, batch)`,
- terminal cashflow leaves have shape `(batch,)`,
- time is always the leading axis of streamed outputs.

Per view:

| View | `terminal=False` | `terminal=True` |
|---|---|---|
| `Raw(name)`, `Group`, `Total` | `(T_out, batch)` | `(batch,)` |
| `Raw()` | `{component_name: (T_out, batch)}` | `{component_name: (batch,)}` |
| `ByState` | `{state_name: (T_out, batch)}` | `{state_name: (batch,)}` |
| `ByKind` | `{kind_name: (T_out, batch)}` | `{kind_name: (batch,)}` |

### Solver state and update

The reduced solver state is one `StateCarry` per reachable state:

```python
class StateCarry(NamedTuple):
    density: jnp.ndarray
    point_mass: PointMass | None
```

- `density` is the duration density on the solver grid, shape `(batch, D)`,
- `point_mass` is a per-individual Dirac `PointMass(value, d_0)` or `None`.

Each solver step:

1. evaluates transition hazards with midpoint quadrature along the transported
   characteristic,
2. aggregates exits from the same source state into one competing-risks update,
3. shifts surviving density one duration slot to the right,
4. injects transferred mass into duration zero,
5. evolves point masses along `(t, d_0 + t)` with the same competing-risks
   logic and routes their outgoing mass into duration-zero density.

Built-in probability callbacks. Output shapes use `T` for the recorded
time axis (length `horizon * steps_per_unit / record_every + 1`), `B` for
batch, `S` for the number of reachable states (in `result["states"]`
order), and `D` for the duration grid:

| Callback | Output |
|---|---|
| `none` | `None` |
| `state_probability` | `(T, B, S)` tensor — duration-marginal density plus point-mass `value` per state. |
| `density_probability` | `(T, B, S)` tensor — duration-marginal density only; excludes point masses. |
| `density` | `(T, B, S, D)` tensor — continuous duration density per state; excludes point masses. |
| `point_mass` | `{state_name: (T, B)}` — only states that carry a point mass appear. |
| `marginal_components` | `{"density": (T, B, S), "point_mass": {state_name: (T, B)}}` |
| `full` | `{"density": (T, B, S, D), "point_mass": {state_name: (T, B)}}` |

Built-in reducers do not expose point-mass duration. If a downstream consumer
needs `PointMass.d_0`, use a custom probability callback and read it directly
from the internal `StateCarry` objects.
After marginalizing over duration, `state_probability` combines continuous
and point-mass probability into total state occupancy.

Custom probability callbacks have signature
`(state: tuple[StateCarry, ...]) -> PyTree`. Their returned PyTree is
stacked by `jax.lax.scan` along a new leading time axis. `StateCarry` and
`PointMass` are internal solver types and live under `jact.callbacks`;
they are not part of the documented public API surface but remain
importable for advanced use.

## Numerical and JIT contract

Midpoint quadrature is second-order when hazards are smooth on the traversed
characteristic segment. For discontinuous hazards, keep jumps aligned with the
solver grid in `t` and `d` when possible; jumps strictly inside a traversed
cell can reduce convergence order.

Static at trace time:

- reduced transition-matrix sparsity pattern,
- probability reducer callable identity,
- presence or absence of point mass per reachable state,
- declared initial-state set,
- `step_size` and `record_every`,
- cashflow component names, kinds, and attachment points,
- payment, `when`, and weight callable identities,
- cashflow view names, kinds, and `terminal` flags.

Traced at runtime:

- covariate arrays,
- parameters captured inside existing callables,
- per-individual initial-state index arrays,
- per-individual masses and durations,
- `PointMass.value` and `PointMass.d_0`.

Changing a structural item retraces. Changing only runtime values inside an
existing structure does not.
