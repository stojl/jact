# jact — API spec (short reference)

Condensed mirror of `docs/api_spec.md`. Same normative content with fewer examples.

## Overview

`jact` has three layers:

- **StateSpace**: topology only.
- **Model**: topology plus intensity callables.
- **Solver**: midpoint quadrature on the reachable subgraph.

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

## Solver

The solver advances the reduced state inside one `jax.lax.scan`. Each step:

1. Evaluates every transition hazard with midpoint quadrature along the transported characteristic.
2. Aggregates exits from the same source state into one competing-risks update.
3. Shifts surviving density one duration slot to the right and injects transferred mass into duration zero.
4. Evolves point masses along `(t, d_0 + t)` by default, or keeps them pinned at fixed `d_0` and uses them as persistent sources when `freeze_initial=True`.

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
    freeze_initial=True,
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
| `freeze_initial` | bool | Keep seeded point masses pinned and use them as persistent sources |
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

## Numerical contract

- Midpoint is second-order when the hazard is smooth on the traversed step.
- Midpoint remains globally second-order for a callable if all jumps in `t` or `d` are aligned to solver grid lines.
- If a jump lies strictly inside a traversed cell, convergence for that callable can drop to first order.
- For tree-based or other piecewise hazards, align split points in `t` and `d` to the solver grid when possible.
- `freeze_initial=True` adds a persistent source, so outputs that include point mass are no longer constrained to sum to 1.

## JIT boundary

Static:

- Matrix sparsity pattern
- Callback function
- Presence or absence of `point_mass` per state
- Declared set of initial states
- `freeze_initial`
- `step_size` and `record_every`

Traced:

- Covariate arrays
- Fitted parameters captured in closures
- `PointMass.value`
- `PointMass.d_0`
- Per-individual masses, durations, and state-index arrays from `InitialDistribution`
