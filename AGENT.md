# AGENT.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
pip install -e ".[dev]"           # install with dev deps (pyright, ruff, pytest)
pyright jact                      # type check public API
ruff check jact                   # lint (imports, style, unused code)
pytest                            # run all tests
pytest tests/test_state_space.py  # run one file
pytest -k test_reachable_from     # run tests matching a name
pytest -x                         # stop on first failure
```

**Quality checks:**
- **pyright** (basic mode) catches type mismatches and undefined names
- **ruff** enforces import order, detects unused code, and flags common errors
- Run both before committing: `pyright jact && ruff check jact`

## Architecture

`jact` computes transition probabilities for semi-Markov multi-state models with duration-dependent intensities, vectorized over 100K+ individuals via JAX. The full pipeline — covariates → probabilities — compiles into a single XLA program via `jax.lax.scan`.

The design separates structure, bindings, initial conditions, and numerics:

- **`state_space.py` — `StateSpace`**: topology only (states, allowed transitions). No intensities, no data. All structural validation at construction (no duplicate states, no self-transitions, no duplicate transitions, every referenced state exists). BFS reachability feeds the solver's pruning. Key surface: `states`, `n_states`, `transitions`, `absorbing`, `transient`, `exits()`, `targets()`, `sources()`, `has_transition()`, `state_index()`, `reachable_from()`. Serialisable via `state_space.to_json(path)` / `StateSpace.from_json(path)`.

- **`model.py` — `Model` / `ReducedModel` / `TransitionInfo`**: binds intensity callables to a `StateSpace` via `StateSpace.build(transitions=..., exits=..., groups=...)`. `build()` enforces that every declared transition is covered **exactly once** across the three kwargs. `Model.reduce(initial_states)` accepts a single state name *or* an iterable of state names and extracts the reachable subgraph; initial states occupy the first `K` reduced indices in state-space ordering, with non-initial reachable states following.

- **`initial_distribution.py` — `InitialDistribution`**: encodes the joint `(state, duration)` distribution at `t = 0` per individual. Construction paths:
  - `InitialDistribution(components={state: {"mass": ..., "duration": ...}}, normalise=True)` — primary; `mass` and `duration` are scalar or `(batch,)`.
  - `InitialDistribution.at(state, duration=0.0)` — all individuals in a single state.
  - `InitialDistribution.per_individual(states=..., duration=..., initial_states=None)` — `states` is a **traced** `(batch,)` int32 index array; the static initial-state set is carried by the optional `initial_states` tuple of state names. With `initial_states=<tuple>`, `states` indexes into that tuple and the solver reduces to the reachable subgraph from those states. With `initial_states=None`, `states` indexes into the model's full state list and **no reduction** happens (every model state is potentially initial). Users with a name array convert host-side via `state_space.state_index(...)` or use the `StateSpace.initial_per_individual` helper.

  The initial-state *set* is **structural** and static on the JIT boundary — it's the keys of `components`, the name passed to `at`, or the `initial_states` tuple on `per_individual` (defaulting to the model's full state list when omitted); never inferred from runtime mass or index-array contents. Declaring a state with all-zero mass still allocates its point-mass slot. This is what lets `Model.reduce` run in pure Python before tracing.

  `StateSpace` exposes ergonomic wrappers that validate state names eagerly against `self.states` and return a plain `InitialDistribution`:
  - `state_space.initial_at(state, duration=0.0)`
  - `state_space.initial_per_individual(state_names=... | state_indices=..., duration=..., initial_states=None)` — exactly one of `state_names` / `state_indices` required.
  - `state_space.initial_distribution(components=..., normalise=True)`

- **`solver.py` — `solve()` / Heun scheme**: 2nd-order predictor-corrector inside `jax.lax.scan`, vmapped over the batch axis. Per reachable state, the carry is a `StateCarry` tracking two conceptually separate objects:
  - `density: (batch, D)` — absolutely continuous duration density.
  - `point_mass: (batch, D) | None` — per-individual Dirac evolving along the characteristic `(s, d_0 + s)`; `None` for states that never carry one; `(batch, D)` for every state declared in the active `InitialDistribution`.

  Full solver state is `tuple[StateCarry, ...]` in reachable-state order. `density` evolves by advection-reaction with rigid duration shift (slot `k` → slot `k+1` each step); `point_mass` evolves by scalar exponential decay along its characteristic — a 1-D problem per individual. They're kept separate to avoid diffusing a Dirac through the finite-difference scheme and to let per-individual `d_0` sit off the duration grid. `solve()` parameters: `initial`, `initial_duration`, `horizon`, `steps_per_unit` (so `D = horizon * steps_per_unit`), `callback`, `record_every` (must divide `horizon * steps_per_unit`; else `ValueError`), `perturbation`, plus `**kwargs` covariates. `initial_duration` is valid only on the `str` / `(batch,)` forms of `initial`; passing it with an `InitialDistribution` raises `ValueError`.

- **`callbacks.py`**: functions `(state: tuple[StateCarry, ...]) → PyTree` that reduce solver state each step. `lax.scan` stacks the returned PyTree along a new leading **time** axis — time is always the leading axis of every output leaf, no rank-dependent transpose. Built-ins: `"default"`, `"no_duration"`, `"collapse_point"`, `"collapse_point_no_duration"` (canonical actuarial output, `(T_out, batch, J)`), `"point_only"`, `"point_only_no_duration"`, `"no_point"`, `"no_point_no_duration"`, `"none"`. This is the extension point for future cashflow / integral-transform features.

- **`intensity/`** (stub subpackage): `parametric.py` (built-in parametric hazards, future) and `wrappers.py` (adapters for common model types, future). Currently placeholders.

### The three ways to assign intensities to transitions

When calling `state_space.build(...)`, each transition must be assigned **exactly once** via one of:

- `transitions={(src, tgt): fn}` — one callable per transition, returning `(batch, D)`.
- `exits={src: fn}` — one callable covering *all* exits from `src`, returning `(n_targets, batch, D)` in the order of `state_space.targets(src)`. Always means *all* exits; for partial coverage use `groups`.
- `groups={fn: [(src, tgt), ...]}` — one callable covering an arbitrary set of transitions, returning `(n_transitions, batch, D)` in the listed order.

All three can be used together; `build()` validates no gaps, no overlaps. `exits` and `groups` callables are sliced at matrix-build time so the solver itself only ever sees uniform `(batch, D)` outputs.

### Intensity callable contract

```python
def intensity(t, d, **kwargs) -> jnp.ndarray:
    # t: scalar clock time (advances 0 → horizon)
    # d: (1, D) duration grid; entry k is duration k / steps_per_unit
    # kwargs: (batch, ...) covariate arrays passed through from solve()
    # returns: (batch, D) — or (n_targets, batch, D) for exits
    #                    — or (n_transitions, batch, D) for groups
```

Callables must be **pure** and **JIT-compatible** (no data-dependent Python control flow, no non-JAX ops). Fitted parameters are captured via closures and become compile-time constants. `t` is clock time (use `baseline_age + t` for attained age); `d` is duration-in-current-state; a Markov intensity uses only `t`, a pure duration-dependent one only `d`, semi-Markov uses both.

Intensities are assumed **càdlàg** (right-continuous with left limits); default evaluation at a discontinuity is the right limit.

**Discontinuity handling is WIP.** The current scheme perturbs `t` and `d` by `±perturbation` (default `1e-12`) each step. Known limitations: absolute `ε` doesn't scale with the argument (and collapses in float32); the perturbation is invisible to the user; Heun drops from 2nd- to 1st-order across finite jumps regardless of `perturbation`. Until this is resolved, treat intensities as smooth or place jumps well inside `(perturbation, horizon − perturbation)` on cleanly representable values. See `docs/api_spec.md` §Discontinuity handling and `docs/design/solver.md` §1 for the long-term protocol options.

### Initial conditions and reachability

`solve(initial=...)` accepts three forms:
- `str` — all individuals start in this state at `d_0 = 0` (lifted to `InitialDistribution.at(...)`).
- `(batch,)` int32 array — traced per-individual indices into `model.states` (lifted to `InitialDistribution.per_individual(states=..., initial_states=None)`); **no reduction** — every model state is potentially initial.
- `InitialDistribution` — full control including mixtures, per-individual `(mass, duration)` per state, and opt-in reduction via the declared initial-state set.

`solve()` calls `Model.reduce(initial_states)` where `initial_states` is the **declared** initial-state set on the distribution (keys of `components`, the name passed to `at`, or `per_individual.initial_states`); it is never inferred from runtime mass or index-array contents. The reduced subgraph is the union of reachability from each initial state. Initial states occupy the first `K` reduced indices in state-space ordering; non-initial reachable states follow. For the common `K = 1` case this matches the previous "initial state at index 0" behaviour. `result["states"]` records the reduced-index → state-name mapping.

At `t = 0`: every state declared in the `InitialDistribution` is seeded with `point_mass` encoding its per-individual mass at its per-individual duration, and `density = 0`. Every reachable state *not* declared has `density = 0` and `point_mass = None`. Because `point_mass` evolves along the characteristic `(s, d_0 + s)` as a per-individual scalar problem, per-individual `d_0` need **not** land on the duration grid.

### JIT boundary

| Static (trace-time) | Traced (runtime) |
|---|---|
| Matrix sparsity pattern (positions of `None` cells) | Covariate arrays (`**kwargs`) |
| Callback function | Fitted parameters captured in closures |
| Presence/absence of `point_mass` per state | `InitialDistribution` mass and duration arrays |
| **Set of initial states** (declared on the distribution) | |
| `step_size`, `record_every`, `perturbation` | |

Changing the declared *set* of initial states re-traces; changing `mass` / `duration` values or `states`-index values within an existing set does not. Rebuilding a `Model` with a different sparsity pattern re-traces; changing parameter values inside existing callables does not. This is why initial-state membership is decided structurally rather than by inspecting mass at runtime — the latter would be a data-dependent topology change incompatible with the trace contract.

## Conventions

- `docs/api_spec.md` is the authoritative API contract — tests reference it explicitly ("per docs/api_spec.md"). Keep it in sync when changing public surface.
- `docs/api_spec_short.md` is a condensed mirror of the spec — same normative content, stripped of examples and rationale. Prefer it for fast review in a fresh session; fall back to `api_spec.md` for worked examples and design rationale.
- `docs/original_prototype/prototype_8.py` is the reference numerics the solver was ported from; consult it when debugging solver behavior.
- Python >= 3.10 is required (uses `X | Y` union syntax and PEP 604 features in places).
- **Pick `callback` and `record_every` before scaling `batch`.** The `default` callback retains `(time, batch, D)` per state and blows up memory fast; `collapse_point_no_duration` at `(T_out, batch, J)` is the canonical actuarial output and stays compact. See `docs/api_spec.md` §Memory budget for worked examples.
- **Output axis convention**: time is the leading axis of every callback leaf. The spec flags that the current `solver.py` may still emit `(batch, J, T_out, ...)` under a rank-dependent transpose scheduled for removal along with the `transpose_result` kwarg — keep this in mind when touching solver output code and verify against the spec before relying on either layout.
