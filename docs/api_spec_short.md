# jact — API spec (short reference)

Condensed mirror of `docs/api_spec.md`. Same normative content with fewer examples.

## Overview

`jact` currently has three implemented layers:

- **StateSpace**: topology only.
- **Model**: topology plus intensity callables.
- **Solver**: midpoint quadrature on the reachable subgraph.

This short spec also describes a planned cashflow declaration layer:

- **cashflows**: reusable named cashflow declarations built from a `StateSpace`.

All intensities must be JIT-compatible. The full pipeline from covariates to probabilities compiles into one XLA program.

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

The planned cashflow API is documentation-only for now and is not yet reflected in the package layout.

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

## Cashflows (planned)

This is the intended public API, not current implemented behavior.

Construction takes a flat `name -> component` mapping; the component's Python type is the kind discriminator:

```python
cashflows = state_space.cashflows({
    name: component,
    ...
})
```

Validation is against the `StateSpace`: attachment keys must reference declared states or transitions, and component names must be unique.

Example:

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

Component types:

- `StateRate(payments)` — payment rate while occupying an attached state; `payments` keyed by state name.
- `TransitionLump(payments)` — lump amount if an attached transition occurs; `payments` keyed by `(src, tgt)`.
- `ScheduledEvent(when=..., payments=...)` — deterministic event time plus state-conditioned payment; `payments` keyed by state name.

`StateRate` and `TransitionLump` take their `payments` dict positionally.

Payment callable protocol for all three kinds:

```python
def payment(t, d, **kwargs) -> jnp.ndarray: ...
```

Scheduled-event time rule:

```python
def when(**kwargs) -> jnp.ndarray: ...
```

V1 scheduled-event policy:

- `when(**kwargs)` returns shape `(batch,)`
- one event time per individual
- event times may depend on solve-time covariates
- event times must be grid-aligned
- off-grid times are rejected

Aggregation and time-only valuation are not declared on the cashflow object; they are passed per solve via `cashflow_views` (see "Cashflow views" below). Cumulative and terminal-from-stream totals are recovered host-side from recorded interval streams.

### Recording semantics

Cashflow streams are recorded with **interval accumulation**: each entry of a streamed leaf is the sum of inner-step contributions over the record period it indexes. Scheduled events contribute to the unique period containing their event time. Sample semantics are not offered for state-rate or transition-lump cashflows. Cumulative output is `jnp.cumsum(stream, axis=0)` host-side.

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

V1 view types and their leaf shapes:

| View | Constructor | Leaves |
|---|---|---|
| `Raw` | `Raw(name=None, *, weight=None, accumulate=False)` | `{component_name: stream}` for every component when `name is None`; one `{name: stream}` otherwise |
| `Group` | `Group(members, *, weight=None, accumulate=False)` | one stream summing the named components |
| `Total` | `Total(*, weight=None, accumulate=False)` | one stream summing every declared component |
| `ByState` | `ByState(*, weight=None, accumulate=False)` | `{state_name: stream}` keyed by attachment state |
| `ByKind` | `ByKind(*, weight=None, accumulate=False)` | `{kind_name: stream}` keyed by component kind |

Every view shares two optional fields:

- `weight: Callable[[float, ...], jnp.ndarray] | float | None = None` — per-step multiplicative factor applied before recording or accumulation. Returns the per-step factor directly (no implicit exponentiation, no implicit cumulative product). A scalar is sugar for a constant callable.
- `accumulate: bool = False` — `False` records one entry per `record_every` (interval); `True` collapses the time axis to a single carry-only `(batch,)` accumulator per leaf.

Validation is structural: view names unique, `Group.members` and `Raw(name=...)` references must be declared component names, `accumulate` is a `bool`, `weight` is `None` / scalar / callable.

`PerAttachment` (one stream per attachment point of a single component) is sketched in `docs/design/cashflow_aggregation.md` §6 and deferred.

### Cashflow valuation

Time-only valuation is expressed via `weight=` on a view; there is no separate valuation dict. Everything lands under `result["cashflows"][view_name]`. `accumulate=True` is the carry-only mode where solver-side weighting buys what host-side post-processing cannot — no `(T_out, batch)` stream is materialised.

`jact.discount_factor(rate=...)` is the canonical numerics helper for `D(t) ≈ exp(-int_0^t r(s) ds)`. `rate` is a callable `(t, **kwargs) -> (batch,)` or scalar; the returned weight uses the same midpoint approximation as the solver, applied to the within-interval contribution.

Out of scope for v1: non-linear-in-cashflow transforms (capping, flooring), path-dependent transforms, and user-defined accumulator carry. Use a payment-callable bake or host-side post-processing for those.

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
| `steps_per_unit` | int | Time discretisation resolution |
| `callback` | `str`, callable, or `None` | Probability callback |
| `record_every` | int | Must divide `horizon * steps_per_unit` |
| `**kwargs` | arrays | Covariates with leading batch dimension |

Result:

```python
result["probability"]
result["states"]
```

Planned solve extension for cashflows:

```python
result = model.solve(
    initial="healthy",
    horizon=10,
    steps_per_unit=12,
    probability="collapse_point_no_duration",
    cashflows=cashflows,
    cashflow_views={
        "benefits": Group(["death_benefit", "retirement_bonus"]),
        "pv_total": Total(weight=discount_factor(rate=0.03), accumulate=True),
    },
    record_every=1,
    age=age_array,
)
```

Planned additional solve arguments:

| Parameter | Type | Description |
|---|---|---|
| `probability` | `str`, callable, or `None` | Probability reporting control |
| `cashflows` | cashflow declaration or `None` | Cashflow components to evaluate |
| `cashflow_views` | `dict[str, View]` or `None` | Solve-time aggregation and time-only valuation declared per view |

Omitting `cashflow_views` (or `None`) returns one streamed leaf per declared component — equivalent to a single implicit `Raw()`. A non-empty `cashflow_views` returns exactly the requested views; raw components are not added implicitly.

Planned result extension:

```python
result["probability"]
result["cashflows"]
result["states"]
```

`result["cashflows"]` is a flat mapping from view name to that view's output. Streamed leaves carry `(T_out, batch)`; terminal (`accumulate=True`) leaves carry `(batch,)`. Dict-valued views (`Raw()`, `ByState`, `ByKind`) drop the time axis on each leaf when accumulated. Streamed and terminal views can coexist within one result.

## Numerical contract

- Midpoint is second-order when the hazard is smooth on the traversed step.
- Midpoint remains globally second-order for a callable if all jumps in `t` or `d` are aligned to solver grid lines.
- If a jump lies strictly inside a traversed cell, convergence for that callable can drop to first order.
- For tree-based or other piecewise hazards, align split points in `t` and `d` to the solver grid when possible.

## JIT boundary

Static:

- Matrix sparsity pattern
- Callback function
- Presence or absence of `point_mass` per state
- Declared set of initial states
- `step_size` and `record_every`

Traced:

- Covariate arrays
- Fitted parameters captured in closures
- `PointMass.value`
- `PointMass.d_0`
- Per-individual masses, durations, and state-index arrays from `InitialDistribution`
