---
name: jact
description: Use when helping users model transition probabilities, semi-Markov state systems, duration-dependent hazards, fitted intensity models, expected cashflows, or actuarial state occupancy with jact.
---

# jact Modeling Skill

Use this skill when helping users model transition probabilities, semi-Markov
state systems, duration-dependent hazards, fitted intensity models, or expected
cashflows with `jact`.

This is application guidance for using the installed package. Do not rely on
repository-only files such as `notes/` or `archive/`, and do not include
development commands, release steps, or contributor conventions in user-facing
answers.

## Modeling Workflow

When helping a user, first identify:

- states and allowed transitions,
- which transitions have known formulas versus fitted models,
- covariates and their batch shape,
- the initial condition and any starting durations,
- whether the desired output is state occupancy, duration diagnostics,
  component cashflows, grouped cashflows, or a terminal value.

1. Define the state topology with `jact.StateSpace`.
2. Bind transition intensities with `state_space.build(...)`.
3. Choose an initial state, per-individual initial states, or an
   `InitialDistribution`.
4. Call `model.solve(...)` with horizon, grid resolution, output reducers, and
   user covariates.
5. Select probability reducers and cashflow views that match the user's
   question.

Core pattern:

```python
import jax.numpy as jnp
import jact

state_space = jact.StateSpace(
    states=["healthy", "disabled", "dead"],
    transitions=[
        ("healthy", "disabled"),
        ("healthy", "dead"),
        ("disabled", "dead"),
    ],
)


def disability_onset(t, d, *, age):
    return jnp.full((age.shape[0], d.shape[-1]), 0.03)


def healthy_mortality(t, d, *, age):
    attained_age = jnp.broadcast_to(
        age[:, None] + t,
        (age.shape[0], d.shape[-1]),
    )
    return 0.002 * jnp.exp(0.08 * (attained_age - 50.0))


def disabled_mortality(t, d, *, age):
    return 2.0 * healthy_mortality(t, d, age=age)


model = state_space.build(
    transitions={
        ("healthy", "disabled"): disability_onset,
        ("healthy", "dead"): healthy_mortality,
        ("disabled", "dead"): disabled_mortality,
    }
)

ages = jnp.array([40.0, 55.0, 70.0])
result = model.solve(
    initial="healthy",
    horizon=20,
    steps_per_unit=12,
    record_every=12,
    probability=jact.probability.StateProbability(),
    age=ages,
)

annual_state_probabilities = result.probability
reachable_states = result.states
```

## Application Contracts

- Single-transition intensity callables use `fn(t, d, **kwargs)` and return
  non-negative arrays shaped `(batch, D)`.
- Grouped intensity callables and exit callables return non-negative arrays
  shaped `(K, batch, D)`, where `K` is the number of covered transitions.
- `t` is scalar-like for the current solver time. `d` is the duration grid with
  the duration axis last; use `d.shape[-1]` for `D`.
- User covariates are passed as keyword arguments to `solve`, then forwarded to
  intensity and cashflow callables. Batched covariates should use leading shape
  `(batch, ...)`.
- Recorded probability outputs and streamed cashflow outputs use time as the
  leading axis. Probability snapshots include `t=0`; streamed cashflows do not.
- `result.states` gives the reachable-state order used by probability tensors.
- Treat a built `Model` as immutable. Build a new model when topology or
  transition assignments change.

## Initial Conditions

Use the simplest initial condition that represents the problem:

- `initial="healthy"` for a shared point-mass start in one state.
- `initial=state_space.initial_at("disabled", duration=durations)` for explicit
  starting duration in one state.
- `initial=state_space.initial_per_individual(...)` when each individual starts
  in a possibly different state.
- `initial=state_space.initial_distribution(...)` for mixed starting masses
  across multiple states.

Pass `initial_duration=...` only for simple shared-state starts. For more
structured starts, prefer the `StateSpace` initial-distribution helpers.

Example mixed initial distribution:

```python
initial = state_space.initial_distribution(
    {
        "healthy": {"mass": jnp.array([0.9, 0.7]), "duration": 0.0},
        "disabled": {"mass": jnp.array([0.1, 0.3]), "duration": jnp.array([2.0, 5.0])},
    }
)

result = model.solve(
    initial=initial,
    horizon=15,
    steps_per_unit=12,
    age=jnp.array([45.0, 60.0]),
)
```

## Probability Output Guidance

Choose the smallest reducer that answers the user's question:

- Use `jact.probability.StateProbability()` for compact actuarial state
  occupancy. This is usually the right default.
- Use `jact.probability.DensityProbability()` when continuous duration density
  should be marginalized over duration but point masses should be excluded.
- Use `jact.probability.Density()`, `jact.probability.PointMass()`, or
  `jact.probability.Full()` only when the user needs duration-level diagnostics
  or point-mass inspection.
- Use `probability=None` for cashflow-only valuation.

Do not use string probability modes such as `"state"` or `"full"`. Pass reducer
instances from `jact.probability` or a custom callable.

## Fitted Intensity Models

When feature construction and model application are separate, wrap fitted models
with `jact.wrappers` instead of mixing feature logic into the transition map.
The wrappers validate shapes, clamp hazards to non-negative values, and normalize
grouped outputs.

```python
import jax.numpy as jnp
import jact


def features(t, d, *, age, smoker):
    attained_age = jnp.broadcast_to(age[:, None] + t, (age.shape[0], d.shape[-1]))
    duration = jnp.broadcast_to(d, attained_age.shape)
    smoker_x = jnp.broadcast_to(smoker[:, None], attained_age.shape)
    return jnp.stack([attained_age, duration, smoker_x], axis=-1)


def apply(params, x):
    linear = params["intercept"] + jnp.sum(x * params["coef"], axis=-1)
    return jnp.exp(linear)


mortality = jact.wrappers.bind_intensity(apply, fitted_params, features)

model = state_space.build(
    transitions={
        ("healthy", "disabled"): disability_onset,
        ("healthy", "dead"): mortality,
        ("disabled", "dead"): disabled_mortality,
    }
)
```

For one fitted model that emits several hazards at once, use
`jact.wrappers.bind_grouped_intensity(..., output_count=K)` with
`groups={wrapped_fn: [(source, target), ...]}` or
`jact.wrappers.bind_exit_intensity(..., output_count=K)` with
`exits={"state": wrapped_fn}`.

Complete exit-wrapper pattern:

```python
def exit_features(t, d, *, age):
    attained_age = jnp.broadcast_to(age[:, None] + t, (age.shape[0], d.shape[-1]))
    duration = jnp.broadcast_to(d, attained_age.shape)
    return jnp.stack([attained_age, duration], axis=-1)


def exit_apply(params, x):
    # Returns hazards for healthy->disabled and healthy->dead on the last axis.
    linear = params["intercept"] + x @ params["coef"]
    return jnp.exp(linear)


healthy_exits = jact.wrappers.bind_exit_intensity(
    exit_apply,
    fitted_exit_params,
    exit_features,
    output_count=2,
)

model = state_space.build(
    exits={"healthy": healthy_exits},
    transitions={("disabled", "dead"): disabled_mortality},
)
```

The order of a multi-output exit callable follows `state_space.exits(source)`,
which is ordered by target-state order.

## Debugging Shape Errors

Most modeling mistakes are shape mistakes. Check these before changing the
state-space topology:

- Return `(batch, D)` from each single-transition intensity, not `(D,)`,
  `(batch,)`, or `(batch, 1)` unless `D == 1`.
- Use `jnp.broadcast_to(value[:, None], (value.shape[0], d.shape[-1]))` when a
  covariate is per individual but the solver needs one value per duration-grid
  cell.
- Use `jnp.broadcast_to(d, (batch, d.shape[-1]))` when an intensity depends on
  duration.
- For grouped, exit, or fitted multi-output models, normalize to
  `(K, batch, D)`. If the fitted model emits `(batch, D, K)`, use
  `jact.wrappers.bind_grouped_intensity(..., output_axis=-1)`.
- All covariates passed to `solve` must agree on the leading batch dimension.

## Cashflow Guidance

Declare cashflow components from the same `StateSpace`, then select solve-time
views:

- `jact.cashflows.Raw()` returns component-level streams for inspection.
- `jact.cashflows.Group([...])` aggregates selected components.
- `jact.cashflows.Total(...)` aggregates all components and is the usual choice
  for present value-style outputs.
- Set `terminal=True` on a view when the user wants one accumulated value per
  individual instead of a time stream.
- Use `weight=` for discounting, accumulation, or other time-dependent weights.

```python
import jax.numpy as jnp
import jact


def annual_premium(t, d, *, age):
    return jnp.full((age.shape[0], d.shape[-1]), -1_200.0)


def death_benefit(t, d, *, age):
    return jnp.full((age.shape[0], d.shape[-1]), 100_000.0)


cashflows = state_space.cashflows(
    {
        "premium": jact.cashflows.StateRate({"healthy": annual_premium}),
        "death_benefit": jact.cashflows.TransitionLump(
            {
                ("healthy", "dead"): death_benefit,
                ("disabled", "dead"): death_benefit,
            }
        ),
    }
)

result = model.solve(
    initial="healthy",
    horizon=30,
    steps_per_unit=12,
    record_every=12,
    probability=None,
    cashflows=cashflows,
    cashflow_views={
        "raw": jact.cashflows.Raw(),
        "benefits": jact.cashflows.Group(["death_benefit"]),
        "pv": jact.cashflows.Total(
            weight=lambda t, **kwargs: jnp.exp(-0.03 * t),
            terminal=True,
        ),
    },
    age=ages,
)

component_streams = result.cashflows["raw"]
benefit_stream = result.cashflows["benefits"]
present_value = result.cashflows["pv"]
```

## Avoid

- Do not import domain types such as `StateRate`, `Raw`, or `StateProbability`
  from the top-level `jact` namespace. Use `jact.cashflows`,
  `jact.probability`, and `jact.wrappers`.
- Do not mutate model topology or transition assignments after construction.
  Create a new `StateSpace` or `Model`.
- Do not treat `notes/`, `archive/`, benchmark scripts, or development docs as
  public API.
- Do not answer modeling questions with repository development instructions.
  Point users to `docs/api_spec.md` for the full public contract when they need
  details beyond this skill.
