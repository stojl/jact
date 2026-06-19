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

Use `import jact` for the main surface. The top-level names are
`jact.StateSpace`, `jact.Model`, `jact.InitialDistribution`,
`jact.ModelResult`, and `jact.solve`. Domain-specific types and fitted-model
helpers live under public submodules:

- `jact.cashflows` — declarations (`StateRate`, `TransitionLump`,
  `ScheduledEvent`, `DurationEvent`, `CashflowDeclaration`) and views (`Raw`,
  `Group`, `Total`, `ByState`, `ByKind`).
- `jact.probability` — output reducers (`StateProbability`,
  `DensityProbability`, `Density`, `PointMass`, `MarginalComponents`,
  `Full`) and the `ProbabilityOutput` union.
- `jact.wrappers` — fitted-model intensity helpers (`bind_intensity`,
  `bind_grouped_intensity`, `bind_exit_intensity`).

Advanced/internal inspection symbols are importable for users who need deeper
debugging hooks, but they are not part of the main top-level `jact` surface:
`StateCarry` is available as `jact.probability.StateCarry`; `ReducedModel` and
`TransitionInfo` live under `jact.model`.

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
| `transitions={(src, tgt): fn}` | Exactly one transition | broadcastable to `(batch, D)` |
| `exits={src: fn}` | Every exit from `src`, ordered by `state_space.targets(src)` | leading output axis, each selected output broadcastable to `(batch, D)` |
| `groups={fn: [(src, tgt), ...]}` | Arbitrary listed transitions | leading output axis, each selected output broadcastable to `(batch, D)` |

Notes:

- `exits` always means all exits from that source state. For partial coverage,
  use `groups`.
- `build()` rejects gaps, overlaps, unknown transitions, empty groups, and
  non-callable assignments.
- `exits` and `groups` are sliced at model-build time so the solver always sees
  one selected transition output, then broadcasts that output to `(batch, D)`.

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

`InitialDistribution` has two jobs at `t = 0`:

1. declare the structural initial-state set,
2. provide the runtime `(state, duration)` distribution within that set.

That structural set drives model reduction. Reduction follows the states
declared by construction:

- the keys of `components`,
- the state passed to `at()`,
- the `initial_states` tuple passed to `per_individual()`,
- or the model's full state list when `per_individual(initial_states=None)`.

Runtime masses and runtime index values never shrink that structure. A declared
state with zero mass still remains part of the structural initial-state set.

Teach the constructors as a ladder from simple to powerful:

### 1. Single-state shorthand at solve entry

```python
model.solve(initial="healthy", horizon=30, steps_per_unit=12, ...)
```

This is the simplest entry path:

- one declared structural initial state,
- all mass starts there,
- duration is zero unless `initial_duration` is supplied.

It is convenience syntax for the explicit single-state form below.

### 2. Single-state explicit form

```python
jact.InitialDistribution.at("healthy", duration=2.0)
```

This exposes the object model directly:

- one declared structural initial state,
- one runtime duration assignment,
- reduction follows the declared state `"healthy"`.

### 3. Mixture over declared initial states

```python
jact.InitialDistribution(
    components={
        "healthy": {"mass": mass_h, "duration": d_h},
        "disabled": {"mass": mass_d, "duration": d_d},
    },
    normalise=True,
)
```

Here the state names are the declared structural initial-state set. `mass` and
`duration` are runtime numeric values attached to those states.

### 4. Per-individual indices into a declared initial-state tuple

```python
jact.InitialDistribution.per_individual(
    states=idx_array,
    duration=d_0_array,
    initial_states=("healthy", "disabled"),
)
```

`states` does not name model states directly in this mode. It indexes into the
declared `initial_states` tuple, and reduction follows that tuple.

### 5. Per-individual indices into the full model state list

```python
jact.InitialDistribution.per_individual(
    states=idx_array,
    duration=d_0_array,
    initial_states=None,
)
```

`initial_states=None` changes what the indices mean: they now refer to the
model's full state list, so no reduction is performed.

Core rules:

- `components` must be a non-empty mapping.
- each component payload must contain both `mass` and `duration`.
- `mass` and `duration` values may be scalars or `(batch,)` arrays.
- concrete `mass` and `duration` values must be non-negative.
- scalar and `(batch,)` component shapes must have compatible batch dimensions.
- masses are normalised per individual by default.
- `declared_initial_states` returns the structural initial-state tuple, or
  `None` for `per_individual(initial_states=None)`.
- `per_individual.states` is a traced `(batch,)` integer array.
- `per_individual.initial_states=<tuple>` means the indices are into that tuple
  and the solver reduces to the union of states reachable from it.
- `per_individual.initial_states=None` means the indices are into the model's
  full state list and no reduction is performed.

Reduction example:

```python
initial = jact.InitialDistribution(
    components={
        "healthy": {"mass": 1.0, "duration": 0.0},
        "disabled": {"mass": 0.0, "duration": 2.0},
    }
)
```

`"disabled"` is still part of the declared structural initial-state set even
though its runtime mass is zero, so reduction still follows the declared keys
`("healthy", "disabled")`.

Common confusions:

- Zero mass does not remove a declared state.
- Declared states are not inferred from runtime mass support.
- `per_individual(initial_states=(...))` and
  `per_individual(initial_states=None)` use different index spaces.
- `initial="healthy"` and `InitialDistribution.at("healthy", ...)` are the same
  model of the world; one is shorthand and the other is explicit.

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
| `**kwargs` | scalar or `(batch, ...)` arrays | Solve-time covariates |

Interpretation:

- `t` is clock time,
- `d` is duration in the current state,
- `**kwargs` are solve-time covariates. Scalars with shape `()` are replicated
  constants and do not define batch size. Non-scalar covariates use axis 0 as
  the batch axis. A rank-1 value such as `jnp.arange(batch_size)` is batched
  with one scalar per individual; a value with shape `(batch_size, 1)` is also
  batched, with one length-1 vector per individual.
- Batch size is inferred only from solver inputs: `InitialDistribution`
  masses/durations, per-individual `initial`, `initial_duration`, and
  non-scalar covariates. If none of those has a batch axis, the solve represents
  one individual. Callable outputs never define batch size.
- attained-age or calendar-time effects are expressed through covariates such
  as `baseline_age + t`.

Return shapes:

| Assignment mode | Shape |
|---|---|
| `transitions` | broadcastable to `(batch, D)` |
| `exits` | leading target axis; each selected output broadcastable to `(batch, D)` |
| `groups` | leading transition axis; each selected output broadcastable to `(batch, D)` |

Useful single-transition return shapes include scalar `()`, `(D,)`, `(1, D)`,
`(batch, 1)`, and `(batch, D)`.

All intensity callables must be pure and JIT-compatible.

## Fitted-model wrappers

The wrapper helpers adapt fitted model inference functions to the intensity
protocol above. They are framework-agnostic closures; no `StateSpace` or
solver changes are involved.

### Single intensity

```python
jact.wrappers.bind_intensity(
    apply_fn,
    params,
    feature_fn,
    *,
    model_state=None,
    apply_kwargs=None,
)
```

The returned callable has the ordinary intensity signature:

```python
intensity(t, d, **kwargs) -> jnp.ndarray
```

Call flow:

```python
features = feature_fn(t, d, **kwargs)
raw = apply_fn(params, features, **apply_kwargs)
```

If `model_state is not None`, the apply call receives a variable mapping:

```python
raw = apply_fn({"params": params, **model_state}, features, **apply_kwargs)
```

The raw output must be broadcastable to `(batch, D)`. The returned hazard is
`jnp.maximum(raw, 0.0)`.

### Grouped and exit intensities

```python
jact.wrappers.bind_grouped_intensity(
    apply_fn,
    params,
    feature_fn,
    *,
    output_count,
    output_axis=-1,
    model_state=None,
    apply_kwargs=None,
)

jact.wrappers.bind_exit_intensity(
    apply_fn,
    params,
    feature_fn,
    *,
    output_count,
    output_axis=-1,
    model_state=None,
    apply_kwargs=None,
)
```

`bind_exit_intensity()` is an alias-shaped helper for readability when the
callable is passed through `exits={...}`. Both helpers move `output_axis` to
the front and require `output_count` on that normalized leading axis. Each
selected output must be broadcastable to `(batch, D)`. For example, model
output `(batch, D, K)` uses `output_axis=-1`, while output `(K, batch, D)` uses
`output_axis=0`.

The returned grouped hazard is `jnp.maximum(normalized, 0.0)`.

Construction validation:

- `apply_fn` and `feature_fn` must be callable.
- `model_state` must be a mapping or `None`.
- `apply_kwargs` must be a mapping or `None`.
- `output_count` must be a positive integer.
- `output_axis` must be an integer.

## Cashflows

### Declaration

Cashflows are declared from a `StateSpace`, not from a `Model`:

```python
cashflows = state_space.cashflows({
    "premium": jact.cashflows.StateRate({"healthy": premium_fn}),
    "death": jact.cashflows.TransitionLump({("healthy", "dead"): death_fn}),
    "bonus": jact.cashflows.ScheduledEvent(
        when=event_time_fn,
        payments={"healthy": bonus_fn},
    ),
    "waiting_period": jact.cashflows.DurationEvent(
        at_durations={"disabled": 0.25},
        payments={"disabled": disability_bonus_fn},
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
- component names must be non-empty strings,
- `StateRate` keys must be declared states,
- `TransitionLump` keys must be declared transitions,
- `ScheduledEvent.payments` keys must be declared states,
- `DurationEvent.at_durations` and `DurationEvent.payments` keys must be the
  same declared states,
- every payment callable and every `ScheduledEvent.when` callable must be
  callable,
- every `DurationEvent.at_durations` value must be a scalar or callable.

### Component kinds

- `StateRate(payments)` records expected payment rate while occupying an
  attached state.
- `TransitionLump(payments)` records expected payment when an attached
  transition occurs.
- `ScheduledEvent(when=..., payments=...)` records expected payment at one
  deterministic event time per individual, conditional on the occupied state at
  that time.
- `DurationEvent(at_durations=..., payments=...)` records a one-time expected
  payment when duration in an attached occupied state reaches that state's
  target duration.

`StateRate` and `TransitionLump` take their `payments` mapping positionally.

### Callable protocols

Payment callables for all component kinds share one interface:

```python
def payment(t, d, **kwargs) -> jnp.ndarray: ...
```

The `**kwargs` argument follows the same batch-axis rule as intensity
callables. The return value must be broadcastable to `(batch, D)`.

Scheduled-event timing uses a separate rule:

```python
def when(**kwargs) -> jnp.ndarray: ...
```

The `**kwargs` argument follows the same batch-axis rule as intensity
callables. The return value must be scalar or broadcastable to `(batch,)`: one
event time per individual.

Duration-event target durations use:

```python
def at_duration(**kwargs) -> jnp.ndarray: ...
```

`at_durations` values are target state durations. They may be Python scalars,
rank-0 arrays, or callables returning a scalar or value broadcastable to
`(batch,)`.

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

### Duration-event policy

Duration events are keyed by target duration already spent in the occupied
state, not by calendar time or elapsed time since the solve started:

- each attached state has one target duration per individual,
- a state receives a duration-event payment only when it has both an
  `at_durations` target and a payment callable in the component declaration,
- target durations may depend on solve-time covariates,
- effective target durations lie on the solver duration grid,
- target durations near a solver grid point are snapped to that grid point to
  absorb floating-point noise,
- other off-grid target durations are floored to the duration-grid cell that
  starts before the supplied target, matching scheduled-event indexing,
- a negative target duration or an effective target duration outside the solver
  grid produces no payment,
- an event at duration `horizon` produces no payment because no solver step
  starts at the right boundary,
- payment evaluation uses the step left endpoint `t_n` and the effective target
  duration, not the midpoint sample,
- state occupancy uses the pre-step convention: the event sees mass occupying
  the state before transitions for that step are applied,
- point masses keep their exact starting duration, which need not lie on the
  solver grid,
- a point mass pays when its exact current duration reaches the effective target
  duration during the solve; starts already past the effective target do not
  pay, while starts exactly at the effective target pay at the first solver
  step,
- density mass pays from the matching effective duration-grid cell.

Duration events are one-time boundary events. They are not recurring rates for
all durations greater than or equal to the target duration.

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
    "raw": jact.cashflows.Raw(),
    "benefits": jact.cashflows.Group(["death", "bonus"]),
    "total": jact.cashflows.Total(),
    "by_state": jact.cashflows.ByState(),
    "by_kind": jact.cashflows.ByKind(),
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
  `(t, **kwargs)` returning a scalar or value broadcastable to `(batch,)`. It is evaluated
  once per inner solver step at that step's midpoint and multiplies the
  contribution attributed to that step. Callable weights receive the same
  batch-major `**kwargs` as intensity and payment callables.
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
- `cashflow_views={}` is allowed and returns an empty `result.cashflows`
  mapping,
- view names must be non-empty strings,
- `Raw(name=...)` and `Group(members)` must reference declared component names,
- `Group.members` is frozen during validation,
- `terminal` must be `bool`,
- non-scalar array weights must be broadcastable to `(batch,)`.

Semantics:

- `Raw()` returns every declared component.
- `Raw(name)` returns one named component.
- `Group(members)` returns the sum of the named components.
- `Total()` returns the sum of all components.
- `ByState()` returns one key per reachable state in reduced order, including
  zero-valued leaves for reachable states with no contributions. `StateRate`
  contributes to the occupied state, `TransitionLump` contributes to the source
  state, and `ScheduledEvent` and `DurationEvent` contribute to the pre-step
  occupied state.
- `ByKind()` returns keys `"state_rate"`, `"transition_lump"`,
  `"scheduled_event"`, and `"duration_event"`, including zero-valued leaves
  when a kind is absent.

## Solver

Call through `model.solve(...)` or `jact.solve(model, ...)`.

```python
def flat_discount(t, **kwargs):
    return jnp.exp(-0.03 * t)


result = model.solve(
    initial="healthy",
    horizon=10,
    steps_per_unit=12,
    probability=jact.probability.StateProbability(),
    cashflows=cashflows,
    cashflow_views={"pv": jact.cashflows.Total(weight=flat_discount, terminal=True)},
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
| `probability` | `ProbabilityOutput`, callable, or `None` | Probability reducer; default is `jact.probability.StateProbability()` |
| `cashflows` | `CashflowDeclaration` or `None` | Cashflow components to evaluate |
| `cashflow_views` | mapping or `None` | Solve-time views; requires `cashflows` |
| `record_every` | positive int | Must divide `horizon * steps_per_unit` |
| `devices` | int, sequence of `jax.Device`, or `None` | Optional local devices for batch-sharded execution |
| `**kwargs` | arrays | Scalar constants or covariates with a shared leading batch dimension |

Validation and defaults:

- `initial_duration` with an `InitialDistribution` is rejected,
- reserved covariate names `initial` and `initial_duration` are rejected,
- legacy kwargs `callback` and `freeze_initial` are rejected,
- `cashflows` must be declared from `model.state_space`,
- non-scalar covariate values must have shape `(batch, ...)` with a shared
  leading batch dimension; scalar covariates are replicated constants,
- `record_every` must divide `horizon * steps_per_unit`,
- `devices` must not be `bool`,
- integer `devices` counts must be positive,
- requested integer `devices` counts cannot exceed the number of available
  local devices,
- `devices=None` uses the single-device JIT path; `devices=1` also stays on
  that path; selecting two or more local devices splits the batch axis across
  devices and restores the documented output shapes.

### Device sharding

`devices` controls optional local multi-device execution. The public result
shapes are unchanged for every setting:

- `devices=None`: use the ordinary single-device `jax.jit` solver path,
- `devices=1`: explicitly select one local device and still use the
  single-device path,
- `devices=N` with `N >= 2`: select the first `N` local devices and run the
  solver with `jax.pmap`,
- `devices=(device0, device1, ...)`: run on the supplied local device
  sequence.

Multi-device execution shards only the leading batch axis. Every non-scalar
solver covariate in `**kwargs`, every initial mass/duration array, and every
solver carry leaf is padded if needed so the batch divides evenly by the
selected device count, then reshaped from `(batch, ...)` to
`(devices, per_device_batch, ...)`. Scalar covariates are passed to every device
unchanged.

Inside intensity, payment, scheduled-event, and callable weight functions on
the multi-device path, each callable sees only its local shard:

```python
age = jnp.arange(5)          # public shape: (5,)
x = jnp.ones((5, 4))         # public shape: (5, 4)

result = model.solve(..., devices=2, age=age, x=x)
```

The batch is padded from 5 to 6, split across 2 devices, and each callable sees
local arrays shaped like:

```python
kwargs["age"].shape == (3,)
kwargs["x"].shape == (3, 4)
```

Scalar covariates with shape `()` keep that scalar shape inside callables on the
multi-device path.

The padded rows are internal only. Solver outputs are merged back and sliced to
the original public batch size, so probability leaves still use `(T, 5, ...)`,
streamed cashflow leaves still use `(T_out, 5)`, and terminal cashflow leaves
still use `(5,)`.

Only axis 0 is treated as the batch axis. For example:

| Public kwarg shape | Meaning |
|---|---|
| `(B,)` | one scalar per individual |
| `(B, 1)` | one length-1 vector per individual |
| `(B, K)` | one length-`K` vector per individual |
| `(B, T, K)` | one `(T, K)` array per individual |

All non-batch dimensions are preserved on each shard. Scalar covariates with
shape `()` are replicated constants and do not define the batch size.

Initial forms:

- `initial="healthy"` means all individuals start in that state at duration
  `initial_duration`,
- `initial=idx_array` means a traced `(batch,)` integer array indexing into the
  model's full state list and performs no reduction,
- `initial=InitialDistribution(...)` gives full control over masses, durations,
  and the declared initial-state set used for reduction.

Result:

`solve()` returns a `ModelResult` dataclass with attribute-only access:

```python
result.states         # tuple[str, ...] — always set
result.probability    # None when probability=None was passed
result.cashflows      # None when cashflows=None was passed
```

`result.states` is the tuple of reachable states in reduced order. Disabled
outputs are `None` rather than missing attributes. `probability=None`
disables probability output entirely.

`ModelResult` is registered as a JAX PyTree, so `jax.jit(model.solve)` and
`jax.tree.map(...)` over the result both work. `states` is treated as static
aux data; `probability` and `cashflows` are children.

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
    point_mass: _PointMass | None
```

- `density` is the duration density on the solver grid, shape `(batch, D)`,
- `point_mass` is a per-individual Dirac `_PointMass(value, d_0)` or `None`.

Each solver step:

1. evaluates transition hazards with midpoint quadrature along the transported
   characteristic,
2. aggregates exits from the same source state into one competing-risks update,
3. shifts surviving density one duration slot to the right,
4. injects transferred mass into duration zero,
5. evolves point masses along `(t, d_0 + t)` with the same competing-risks
   logic and routes their outgoing mass into duration-zero density.

### Probability outputs

Built-in probability output reducers. Output shapes use `T` for the
recorded time axis (length `horizon * steps_per_unit / record_every + 1`),
`B` for batch, `S` for the number of reachable states (in `result.states`
order), and `D` for the duration grid:

| Reducer | Output |
|---|---|
| `StateProbability()` | `(T, B, S)` tensor — duration-marginal density plus point-mass `value` per state. |
| `DensityProbability()` | `(T, B, S)` tensor — duration-marginal density only; excludes point masses. |
| `Density()` | `(T, B, S, D)` tensor — continuous duration density per state; excludes point masses. |
| `PointMass()` | `{state_name: (T, B)}` — only states that carry a point mass appear. |
| `MarginalComponents()` | `{"density": (T, B, S), "point_mass": {state_name: (T, B)}}` |
| `Full()` | `{"density": (T, B, S, D), "point_mass": {state_name: (T, B)}}` |

These types live under `jact.probability` and form the
`ProbabilityOutput` union. Built-in reducers do not expose point-mass
duration. If a downstream consumer needs `_PointMass.d_0`, supply a custom
probability callable and read it directly from the internal `StateCarry`
objects.

After marginalizing over duration, `StateProbability` combines continuous
and point-mass probability into total state occupancy.

Custom probability callables have signature
`(state: tuple[StateCarry, ...]) -> PyTree`. Their returned PyTree is
stacked by `jax.lax.scan` along a new leading time axis. `StateCarry` and
`_PointMass` are advanced/internal solver inspection symbols and live under
`jact.probability`; they are not part of the main top-level `jact` surface but
remain importable for advanced use.

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
