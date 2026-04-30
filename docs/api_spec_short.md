# jact — API spec (short reference)

Condensed mirror of `docs/api_spec.md`. Same normative content with fewer examples.

## Overview

`jact` has four implemented layers:

- **StateSpace**: topology only.
- **Model**: topology plus intensity callables.
- **Cashflows**: reusable named cashflow declarations built from a `StateSpace`.
- **Solver**: midpoint quadrature on the reachable subgraph; emits probabilities and cashflow streams.

All intensity, payment, and weight callables must be JIT-compatible. The full pipeline from covariates to probabilities and cashflow streams compiles into one XLA program.

## Module layout

```text
jact/
├── __init__.py              # Public API: StateSpace, Model, InitialDistribution, solve, callbacks,
│                            #   StateRate, TransitionLump, ScheduledEvent,
│                            #   Raw, Group, Total, ByState, ByKind,
│                            #   CashflowDeclaration, discount_factor
├── state_space.py           # StateSpace + InitialDistribution and cashflow helpers
├── model.py                 # Model, ReducedModel, TransitionInfo
├── initial_distribution.py  # InitialDistribution
├── cashflows.py             # Cashflow components, views, CashflowDeclaration, discount_factor
├── solver.py                # Semi-Markov solver with cashflow accumulation
└── callbacks.py             # Probability output callbacks
```

Use `import jact` for the main user API: `jact.StateSpace`,
`jact.InitialDistribution`, `jact.solve`, and cashflow declarations like
`jact.StateRate` and `jact.Total`. Advanced inspection and callback state
objects remain in submodules, such as `jact.callbacks.PointMass` and
`jact.model.ReducedModel`.

`archive/original_prototype/` is retained as historical documentation only, not
as public package API and not as a runtime reference for tests or benchmarks.

## StateSpace

```python
state_space = jact.StateSpace(
    states=["healthy", "disabled", "dead"],
    transitions=[("healthy", "disabled"), ("healthy", "dead"), ("disabled", "dead")],
)
```

Construction validates no duplicate states, no unknown transition endpoints, no self-transitions, and no duplicate transitions.

Surface:

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
| `build(transitions=..., exits=..., groups=...)` | → `Model` |
| `cashflows({...})` | → `CashflowDeclaration` |
| `initial_at(state, duration=0.0)` | → `InitialDistribution` |
| `initial_distribution(components, normalise=True)` | → `InitialDistribution` |
| `initial_per_individual(state_names=... \| state_indices=..., duration=..., initial_states=None)` | → `InitialDistribution` |

## Model

Built via `state_space.build(transitions=..., exits=..., groups=...)`. Every declared transition must be assigned exactly once across the three kwargs.

| Kwarg | Coverage | Callable return shape |
|---|---|---|
| `transitions={(src, tgt): fn}` | One transition | `(batch, D)` |
| `exits={src: fn}` | All exits from `src`, ordered by `state_space.targets(src)` | `(n_targets, batch, D)` |
| `groups={fn: [(src, tgt), ...]}` | Arbitrary set, in listed order | `(n_transitions, batch, D)` |

`Model.reduce(initial_states)` accepts a single state name or an iterable of state names and returns:

```python
reduced.initial_states
reduced.reachable_states
reduced.n_states
reduced.solver_matrix
```

`model.info(src, tgt)` returns `TransitionInfo(source, target, assignment, callable, index)`.

## InitialDistribution

Three entry points:

```python
jact.InitialDistribution(components=..., normalise=True)
jact.InitialDistribution.at(state, duration=0.0)
jact.InitialDistribution.per_individual(states=idx_array, duration=d_0, initial_states=None)
```

Key rules:

- `per_individual.states` is a traced `(batch,)` int32 array.
- `initial_states=None` means indices are into the full model state list and no reduction is done.
- `initial_duration` is valid only for the `str` / `(batch,)` `initial` shortcuts to `solve()`.
- Component masses are normalised per individual by default.

State-space helpers (eager name validation):

```python
state_space.initial_at(state, duration=0.0)
state_space.initial_distribution(components, normalise=True)
state_space.initial_per_individual(
    state_names=...,        # OR state_indices=... — exactly one, keyword-only
    state_indices=...,
    duration=0.0,
    initial_states=None,
)
```

## Intensity protocol

```python
def intensity(t, d, **kwargs) -> jnp.ndarray: ...
```

| Arg | Type | Meaning |
|---|---|---|
| `t` | scalar float | Clock time |
| `d` | `(1, D)` | Duration grid, broadcast over batch |
| `**kwargs` | `(batch, ...)` arrays | Covariates passed from `solve()` |

Return shapes:

| Assignment | Shape |
|---|---|
| `transitions` | `(batch, D)` |
| `exits` | `(n_targets, batch, D)` |
| `groups` | `(n_transitions, batch, D)` |

JAX requirements: pure, JIT-compatible, and closed over static values only.

## Cashflows

Construction takes a flat `name -> component` mapping; the component's Python type is the kind discriminator:

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

The returned object is a `CashflowDeclaration` (frozen dataclass) bound to the state-space topology and reusable across solves. Surface: `declaration.state_space`, `declaration.names`, `declaration.component(name)`.

Validation is structural, against the `StateSpace` only: attachment keys must reference declared states or transitions, and component names must be unique.

Component types:

- `StateRate(payments)` — payment rate while occupying an attached state; `payments` keyed by state name.
- `TransitionLump(payments)` — lump amount if an attached transition occurs; `payments` keyed by `(src, tgt)`.
- `ScheduledEvent(when=..., payments=...)` — deterministic event time plus state-conditioned payment; `payments` keyed by state name.

`StateRate` and `TransitionLump` take their `payments` dict positionally.

Payment callable protocol for all three kinds:

```python
def payment(t, d, **kwargs) -> jnp.ndarray: ...   # returns (batch, D)
```

Scheduled-event time rule:

```python
def when(**kwargs) -> jnp.ndarray: ...            # returns (batch,)
```

Scheduled-event policy:

- one event time per individual
- event times may depend on solve-time covariates
- event times within numerical tolerance of a solver grid point snap to that grid point; other off-grid times are floored silently
- the tolerance is only for floating-point representation noise, not a business grace period
- state occupancy at the effective event time uses the pre-step convention

Out of scope: multiple event times per component (future work).

Aggregation and time-only valuation are not declared on the cashflow object; they are passed per solve via `cashflow_views` (see "Cashflow views" below). Cumulative and terminal-from-stream totals are recovered host-side from recorded interval streams.

### Recording semantics

Cashflow streams are recorded with **interval accumulation**: each entry of a streamed leaf is the sum of inner-step contributions over the record period it indexes. Scheduled events contribute to the unique period containing their effective event time. Sample semantics are not offered for state-rate or transition-lump cashflows. Cumulative output is `jnp.cumsum(stream, axis=0)` host-side; terminal-from-stream is `stream.sum(axis=0)`.

### Cashflow views

Aggregation is declared per solve as a flat mapping from view name to a typed view object:

```python
cashflow_views = {
    "benefits":    Group(["death_benefit", "retirement_bonus"]),
    "total":       Total(),
    "by_state":    ByState(),
    "by_kind":     ByKind(),
    "premium_raw": Raw("premium"),
}
```

View types and their leaf shapes:

| View | Constructor | Leaves |
|---|---|---|
| `Raw` | `Raw(name=None, *, weight=None, terminal=False)` | `{component_name: stream}` for every component when `name is None`; one `{name: stream}` otherwise |
| `Group` | `Group(members, *, weight=None, terminal=False)` | one stream summing the named components |
| `Total` | `Total(*, weight=None, terminal=False)` | one stream summing every declared component |
| `ByState` | `ByState(*, weight=None, terminal=False)` | `{state_name: stream}` keyed by every reachable state in the reduced solve |
| `ByKind` | `ByKind(*, weight=None, terminal=False)` | `{kind_name: stream}` keyed by component kind (`"state_rate"`, `"transition_lump"`, `"scheduled_event"`) |

Every view shares two optional fields:

- `weight: Callable[[float, ...], jnp.ndarray] | float | None = None` — per-step multiplicative factor applied before recording or terminal accumulation. Returns the per-step factor directly (no implicit exponentiation, no implicit cumulative product). A scalar is sugar for a constant callable.
- `terminal: bool = False` — `False` records one entry per `record_every` (interval); `True` collapses the time axis to a single carry-only `(batch,)` accumulator per leaf.

`ByState` includes zero-valued leaves for reachable states with no contributions. `StateRate` contributes to its attached state, `TransitionLump` to its source state, `ScheduledEvent` to the state occupied at the effective event time.

Validation is structural: view names unique, `Group.members` and `Raw(name=...)` references must be declared component names, `terminal` is a `bool`, `weight` is `None` / scalar / callable.

`PerAttachment` (one stream per attachment point of a single component) is sketched in `notes/design/cashflow_aggregation.md` §6 and deferred.

### Cashflow valuation

Time-only valuation is expressed via `weight=` on a view; there is no separate valuation dict. Everything lands under `result["cashflows"][view_name]`. `terminal=True` is the carry-only mode where solver-side weighting buys what host-side post-processing cannot — no `(T_out, batch)` stream is materialised.

`jact.discount_factor(rate=...)` is the canonical numerics helper for `D(t) ≈ exp(-int_0^t r(s) ds)`. `rate` is a callable `(t, **kwargs) -> (batch,)` or scalar; the returned weight uses the same midpoint approximation as the solver, applied to the within-interval contribution.

Out of scope: non-linear-in-cashflow transforms (capping, flooring), path-dependent transforms, and user-defined accumulator carry. Use a payment-callable bake or host-side post-processing for those.

## Solver

The solver advances the reduced state inside one `jax.lax.scan`. Each step:

1. Evaluates every transition hazard with midpoint quadrature along the transported characteristic.
2. Aggregates exits from the same source state into one competing-risks update.
3. Shifts surviving density one duration slot to the right and injects transferred mass into duration zero.
4. Evolves point masses along `(t, d_0 + t)` with the same competing-risks update.

Solver state is one `StateCarry` per reachable state:

```python
class StateCarry(NamedTuple):
    density: jnp.ndarray
    point_mass: PointMass | None
```

`density` has shape `(batch, D)`. `PointMass.value` and `PointMass.d_0` have shape `(batch,)`.

## Calling solve

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

Or via the module-level entry point:

```python
result = jact.solve(model, initial=..., horizon=..., steps_per_unit=..., **covariates)
```

Parameters:

| Parameter | Type | Description |
|---|---|---|
| `initial` | `str`, `(batch,)` int array, or `InitialDistribution` | Initial condition |
| `initial_duration` | float or `(batch,)` array | Per-individual `d_0` for `str` / `(batch,)` initial forms |
| `horizon` | int | Number of time units |
| `steps_per_unit` | int | Time discretisation resolution |
| `probability` | `str`, callable, or `None` | Probability output reducer. Defaults to `"collapse_point_no_duration"`. `None` disables probability output |
| `cashflows` | `CashflowDeclaration` or `None` | Cashflow components to evaluate. `None` disables cashflow output |
| `cashflow_views` | `dict[str, View]` or `None` | Solve-time aggregation and time-only valuation declared per view. Requires `cashflows`. When omitted with `cashflows` supplied, defaults to `{"raw": Raw()}` |
| `record_every` | int | Must divide `horizon * steps_per_unit` |
| `**kwargs` | arrays | Covariates with leading batch dimension |

Result keys:

```python
result["probability"]   # omitted when probability=None
result["cashflows"]     # omitted when cashflows is None
result["states"]        # always present: tuple of reachable-state names in reduced order
```

Disabled outputs are dropped from the result dict — no `None` placeholder is emitted. `result["cashflows"]` is a flat mapping from view name to that view's output. Streamed leaves carry `(T_out, batch)`; terminal (`terminal=True`) leaves carry `(batch,)`. Dict-valued views (`Raw()`, `ByState`, `ByKind`) drop the time axis on each leaf when terminal. Streamed and terminal views can coexist within one result.

Built-in callbacks: `default`, `no_duration`, `collapse_point`, `collapse_point_no_duration`, `point_only`, `point_only_no_duration`, `no_point`, `no_point_no_duration`, `none`. `probability=None` disables probability output entirely and omits the result key.

## Numerical contract

- Midpoint is second-order when the hazard is smooth on the traversed step.
- Midpoint remains globally second-order for a callable if all jumps in `t` or `d` are aligned to solver grid lines.
- If a jump lies strictly inside a traversed cell, convergence for that callable can drop to first order.
- For tree-based or other piecewise hazards, align split points in `t` and `d` to the solver grid when possible.

## JIT boundary

Static:

- Matrix sparsity pattern
- Probability reducer callable identity
- Presence or absence of `point_mass` per state
- Declared set of initial states
- `step_size` and `record_every`
- Declared cashflow component names, kinds, and attachment points
- Payment and `when` callable identities
- Declared cashflow view names, kinds, and `terminal` flag
- `weight` callable identity (or scalar value bound at trace time)

Traced:

- Covariate arrays
- Parameters captured in closures of intensity, payment, `when`, and `weight` callables
- `PointMass.value`
- `PointMass.d_0`
- Per-individual masses, durations, and state-index arrays from `InitialDistribution`
