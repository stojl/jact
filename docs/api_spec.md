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
cashflows = state_space.cashflows({
    name: component,
    ...
})
```

The argument is a single flat mapping from component name to a typed component object (`StateRate`, `TransitionLump`, or `ScheduledEvent`). The returned object is a reusable cashflow declaration built against the state-space topology. The exact concrete class name of the declaration is not yet fixed.

The cashflow declaration answers:

- which named cashflow components exist,
- what kind of component each is (carried by the component's Python type),
- where each component attaches,
- what callable defines the payment amount.

Aggregation, valuation, cumulative totals, and terminal totals are not declared on this object. Aggregation and time-only valuation are declared per solve via `cashflow_views` (see §"Cashflow views" and §"Cashflow valuation" below); cumulative and terminal-from-stream totals are recovered host-side from the recorded interval streams.

Validation is structural and uses the `StateSpace` only:

- `StateRate` attachment keys must reference declared states,
- `TransitionLump` attachment keys must reference declared transitions,
- `ScheduledEvent.payments` keys must reference declared states,
- component names must be unique.

### Component-first grammar

The top-level public grammar is component-first. Each entry maps a component name to a typed component object; the object's Python type discriminates the kind:

```python
cashflows = state_space.cashflows({
    "premium":          StateRate({"healthy": premium_fn}),
    "death_benefit":    TransitionLump({("healthy", "dead"): death_fn}),
    "retirement_bonus": ScheduledEvent(
        when=event_time_fn,
        payments={"healthy": bonus_fn},
    ),
})
```

Each component name identifies one raw cashflow stream. Components are the basis for later aggregation and reporting.

`StateRate`, `TransitionLump`, and `ScheduledEvent` are small frozen dataclasses parallel to other typed declaration objects in `jact`. The Python type carries the kind, so no `kind` discriminator string is needed and pyright sees the per-kind required fields.

### Component kinds

#### State-rate

`StateRate` attaches payment-rate callables to states. The attachment dict is the sole positional argument:

```python
StateRate({
    "healthy":  premium_fn,
    "disabled": waiver_fn,
})
```

Interpretation: expected cashflow is generated continuously while the policy occupies the attached state.

#### Transition-lump

`TransitionLump` attaches lump-sum payment callables to declared transitions. The attachment dict is the sole positional argument:

```python
TransitionLump({
    ("healthy", "dead"):     death_fn,
    ("healthy", "disabled"): onset_fn,
})
```

Interpretation: expected cashflow is generated when the attached transition occurs.

#### Scheduled-event

`ScheduledEvent` declares a deterministic event-time rule (`when`) plus state-conditioned payment callables (`payments`):

```python
ScheduledEvent(
    when=event_time_fn,
    payments={
        "healthy":  bonus_fn,
        "disabled": reduced_bonus_fn,
    },
)
```

Interpretation: expected cashflow is generated at the component's deterministic event time, conditional on the occupied state at that time.

The `payments` field name is consistent with the positional payment dicts on `StateRate` and `TransitionLump`: in every kind the attachment dict is keyed by an attachment point (state or transition) and valued by a payment callable.

### Callable protocols

Payment functions for `StateRate`, `TransitionLump`, and `ScheduledEvent` share one protocol:

```python
def payment(t, d, **kwargs) -> jnp.ndarray: ...
```

Arguments:

| Arg | Type | Meaning |
|---|---|---|
| `t` | scalar float | Clock time at which the payment is evaluated |
| `d` | `(1, D)` | Duration grid broadcast over batch |
| `**kwargs` | `(batch, ...)` arrays | Solve-time covariates |

Return shape:

- `(batch, D)` for every component kind.

Interpretation depends on component kind:

- for `StateRate`, `payment(t, d, **kwargs)` is a rate while occupying the state,
- for `TransitionLump`, it is a lump amount if the transition occurs at `(t, d)`; duration dependence is allowed,
- for `ScheduledEvent`, it is a payment amount evaluated only at the effective event time on the same duration grid.

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
- event times are snapped to the solver grid by flooring to the greatest grid time less than or equal to the returned value,
- snapping is silent and defines the effective event time used by the solver,
- event times exactly on the solver grid use that grid time unchanged,
- state occupancy at the effective event time uses the pre-step convention,
- multiple event times per component are future work.

### Recording semantics

Cashflow streams are recorded with **interval accumulation** semantics. Each entry of a streamed cashflow leaf is the sum of inner-step contributions generated over the record period that the entry indexes:

- state-rate and transition-lump contributions are summed across every inner step spanning the period,
- a scheduled event lands in the unique record period containing its effective event time and contributes to that period only.

This differs from the snapshot semantics used for probability output. Cumulative output is `jnp.cumsum(stream, axis=0)` host-side; terminal-from-stream output is `stream.sum(axis=0)`. A separate carry-only terminal mode is available per view via `terminal=True` (see §"Cashflow views"). Sample semantics are not offered for state-rate or transition-lump cashflows: the instantaneous value is a rate, not a payment, and its meaning is unstable under refinement of `record_every`.

### Cashflow views

Aggregation is declared per solve through a flat mapping from view name to a typed view object:

```python
cashflow_views = {
    "premium":         Raw("premium"),
    "benefits":        Group(["death_benefit", "retirement_bonus"]),
    "total":           Total(),
    "by_state":        ByState(),
    "by_kind":         ByKind(),
}
```

Each entry maps a user-facing view name to one self-describing object; the Python type carries the kind of view. Each declared view name becomes one entry in `result["cashflows"]`.

The v1 view types are small frozen dataclasses parallel to the typed component objects:

| View | Constructor | Output leaves |
|---|---|---|
| `Raw` | `Raw(name: str | None = None, *, weight=None, terminal=False)` | `{component_name: stream}` for every declared component when `name is None`; `{name: stream}` for the single named component otherwise |
| `Group` | `Group(members: Sequence[str], *, weight=None, terminal=False)` | one stream summing the named components |
| `Total` | `Total(*, weight=None, terminal=False)` | one stream summing every declared component |
| `ByState` | `ByState(*, weight=None, terminal=False)` | `{state_name: stream}` keyed by every reachable state in the reduced solve |
| `ByKind` | `ByKind(*, weight=None, terminal=False)` | `{kind_name: stream}` keyed by component kind |

Every view shares two optional fields:

| Field | Type | Default | Meaning |
|---|---|---|---|
| `weight` | `Callable[[float, ...], jnp.ndarray] | float | None` | `None` | Per-step multiplicative factor applied before recording or terminal accumulation. `None` is unweighted. A scalar is sugar for `lambda t, **kw: scalar`. |
| `terminal` | `bool` | `False` | `False` records one entry per `record_every` (interval semantics, see above); `True` collapses the time axis to a single carry-only accumulator per leaf. This is terminal-only output, not cumulative streaming output. |

The `weight` callable returns the per-step factor directly: no implicit exponentiation, no implicit cumulative product. For continuously discounted weights, use the `discount_factor` helper described in §"Cashflow valuation".

Within a single `cashflow_views` mapping, streamed and terminal views can coexist freely; each leaf's shape is decided by that view's `terminal` setting (see §"Output shape" within §"Planned cashflow solve extension").

View semantics:

- `Raw(name)` returns the single named component.
- `Raw()` returns every declared component.
- `Group(members)` returns the sum of the named components.
- `Total()` returns the sum of all declared components.
- `ByState()` returns one key per reachable state in the reduced solve, including zero-valued leaves for reachable states with no contributions. `StateRate` contributes to its attached state, `TransitionLump` contributes to its source state, and `ScheduledEvent` contributes to the state occupied at the effective event time.
- `ByKind()` returns keys `"state_rate"`, `"transition_lump"`, and `"scheduled_event"`.

Validation is structural and uses the cashflow declaration only:

- view names are unique across the dict,
- `Group.members` and `Raw(name=...)` reference declared component names,
- `terminal` is a `bool`,
- `weight` is `None`, a Python scalar, or a callable.

Within-component splitting (a `PerAttachment` view exposing one stream per attachment point of a single component) is sketched in `docs/design/cashflow_aggregation.md` §6 and deferred to a future version.

### Cashflow valuation

Time-only valuation — discounting, indexation, scenario reweighting, deterministic unit-of-account changes — is expressed via the `weight=` field on a view. There is no separate `valuations` dict and no parallel result key: every weighted output lands under `result["cashflows"][view_name]`.

`terminal=True` is the carry-only mode: the solver maintains a single `(batch,)` accumulator per terminal-mode view and emits no per-step entry, so no `(T_out, batch)` stream is materialised. This is the configuration in which solver-side weighting buys something post-processing cannot.

```python
from jact import discount_factor

cashflow_views = {
    "pv_total":       Total(weight=discount_factor(rate=r), terminal=True),
    "pv_total_stream": Total(weight=discount_factor(rate=r)),
    "real":           Group(["death_benefit", "retirement_bonus"], weight=index_curve),
    "pv_by_state":    ByState(weight=discount_factor(rate=r), terminal=True),
}
```

`jact.discount_factor(rate=...)` is the canonical numerics helper for the continuously discounted weight `D(t) ≈ exp(-int_0^t r(s) ds)`:

- `rate` is a callable `(t, **kwargs) -> (batch,)` or a scalar; `rate=0.03` is sugar for `lambda t, **kw: 0.03`.
- The returned callable evaluates the running discount factor against the solver step grid using the same midpoint approximation as the rest of the solver. The within-interval weight applied to the contribution attributed to interval `[t_n, t_{n+1}]` is `exp(-r(t_n + dt/2) · dt) · D(t_n)`.
- Because the recording default is interval accumulation, the discount weight applied is the within-interval weight for the interval the cashflow is attributed to, not a point-time weight at the recording boundary.

Out of scope for v1 (deferred to a future functor protocol; see `docs/design/cashflow_valuation.md` §4.6):

- non-linear-in-cashflow transforms (capping, flooring, utility),
- path-dependent transforms (running maxima, threshold accumulators, look-back guarantees),
- user-defined accumulator carry beyond `(batch,)` running sums.

Anything outside the time-local linear-weight envelope continues to be expressible by baking the weight into the payment callable or post-processing a streamed view host-side.

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
    cashflow_views={
        "benefits": Group(["death_benefit", "retirement_bonus"]),
        "pv_total": Total(weight=discount_factor(rate=0.03), terminal=True),
    },
    record_every=1,
    age=age_array,
)
```

Planned semantics:

| Parameter | Type | Meaning |
|---|---|---|
| `probability` | `str`, callable, or `None` | Probability reporting control |
| `cashflows` | cashflow declaration or `None` | Named cashflow components to evaluate |
| `cashflow_views` | `dict[str, View]` or `None` | Solve-time aggregation and time-only valuation declared per view |

`probability=None` disables probability output.
`cashflows=None` disables cashflow output.

When `cashflows` is supplied and `cashflow_views` is `None` (or omitted), the solver behaves exactly as if `cashflow_views={"raw": Raw()}` had been passed. When `cashflow_views` is supplied, the solver returns exactly the requested views; raw components are not added implicitly.

Planned result keys:

```python
result["probability"]
result["cashflows"]
result["states"]
```

`result["cashflows"]` is a flat mapping from view name to that view's output; streamed and terminal views can coexist within the same result. The exact disabled-output convention, such as omitted key versus `None`, is not yet fixed.

#### Output shape

Per leaf, shape depends on the view kind and on `terminal`:

| View | `terminal=False` (streamed) | `terminal=True` (terminal) |
|---|---|---|
| `Raw(name)`, `Group`, `Total` | `(T_out, batch)` | `(batch,)` |
| `Raw()` | `{component_name: (T_out, batch)}` for every declared component | `{component_name: (batch,)}` |
| `ByState` | `{state_name: (T_out, batch)}` for every reachable state in the reduced solve | `{state_name: (batch,)}` for every reachable state in the reduced solve |
| `ByKind` | `{kind_name: (T_out, batch)}` | `{kind_name: (batch,)}` |

`T_out = horizon * steps_per_unit / record_every`, matching the probability output. Time is the leading axis of every streamed leaf; terminal leaves drop the time axis. Batch is always preserved.

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
3. Forms survival `S_i = exp(-Σ_j A_ij)` and transfer fractions `A_ij / (Σ_j A_ij) * (1 - exp(-Σ_j A_ij))` using `expm1` for numerical accuracy.
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
