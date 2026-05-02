# Design note: model-object wrappers for fitted intensities

## Status

Proposed.

## Context

`jact` currently expects transition intensities as solver-shaped callables:

```python
fn(t, d, **kwargs) -> array
```

That interface is small and composable, but it is lower-level than many
users will expect when coming from `flax`, `optax`, `equinox`, or similar
JAX-native modelling stacks.

A typical ML workflow looks like this:

1. define a module object,
2. train it with some external training loop,
3. keep the trained parameters and any model state,
4. expect to hand the trained object to downstream code directly.

Today, a `jact` user must write an adapter manually:

- convert solver inputs `(t, d, **kwargs)` into feature tensors,
- call `module.apply(...)` or equivalent,
- reshape outputs into `(batch, D)` or `(n_transitions, batch, D)`,
- then pass the adapter into `state_space.build(...)`.

This is workable, but verbose enough that it weakens the "fit a flexible
model, then solve" story in the README. A wrapper layer would improve
adoptability by making the handoff from training code to `jact` feel
native rather than improvised.

---

## Goal

Add a convenient, explicit wrapper API so users can pass fitted model
objects into `jact` without writing a custom solver adapter each time.

The wrapper layer should:

- preserve the current callable-based core,
- stay fully JAX-native and JIT-compatible,
- work with separate-per-transition models and joint multi-output models,
- support common module styles such as `flax.linen.Module.apply(...)`,
- make the feature construction step explicit, since `jact` cannot infer
  model features from `(t, d, **kwargs)` on its own.

## Non-goals

- `jact` will not take over training, optimisation, batching, or dataset
  management.
- `jact` will not become framework-agnostic across non-JAX ecosystems.
  The target is JAX-native model objects.
- The wrapper layer will not replace the existing callable protocol.
  Direct callables remain the lowest-level contract and the internal
  solver interface.

---

## Proposed public shape

### 1. Add wrapper constructors in the top-level namespace

Recommended API:

```python
jact.bind_intensity(...)
jact.bind_grouped_intensity(...)
jact.bind_exit_intensity(...)
```

These constructors return ordinary callables with the existing solver
signature, so they fit naturally into:

```python
state_space.build(
    transitions=...,
    groups=...,
    exits=...,
)
```

This keeps `StateSpace.build(...)` unchanged and avoids introducing a
second, parallel model-building API prematurely.

### 2. Transition-level wrapper

For a model object that emits one hazard surface:

```python
healthy_dead = jact.bind_intensity(
    apply_fn=hazard_net.apply,
    params=params,
    feature_fn=feature_fn,
    model_state=batch_stats,   # optional
    apply_kwargs={"train": False},  # optional
)

model = state_space.build(
    transitions={
        ("healthy", "dead"): healthy_dead,
    }
)
```

### 3. Group wrapper

For a model object that jointly emits several hazards:

```python
joint_hazards = jact.bind_grouped_intensity(
    apply_fn=hazard_net.apply,
    params=params,
    feature_fn=feature_fn,
    outputs=[
        ("healthy", "disabled"),
        ("healthy", "dead"),
        ("disabled", "dead"),
    ],
)

model = state_space.build(
    groups={
        joint_hazards: [
            ("healthy", "disabled"),
            ("healthy", "dead"),
            ("disabled", "dead"),
        ]
    }
)
```

### 4. Exit wrapper

For one model object that emits all exits from a source state:

```python
healthy_exits = jact.bind_exit_intensity(
    apply_fn=hazard_net.apply,
    params=params,
    source="healthy",
    feature_fn=feature_fn,
)

model = state_space.build(
    exits={
        "healthy": healthy_exits,
    }
)
```

---

## Required wrapper inputs

Each wrapper should accept the following data explicitly.

### `apply_fn`

The function used for inference, typically:

- `module.apply` for Flax,
- a bound callable for Equinox or custom JAX models,
- any other JAX-native forward function.

`jact` should not depend on a specific framework type. The contract is
"callable inference function", not "Flax module instance".

### `params`

Framework-specific trained parameters, passed through untouched to
`apply_fn`.

### `feature_fn`

A required user-supplied function:

```python
feature_fn(t, d, **kwargs) -> features
```

This is the central abstraction. It converts solver coordinates into
model inputs. `jact` should not try to guess whether a model wants:

- attained age,
- current-state duration,
- calendar time,
- static covariates,
- interaction terms,
- embeddings,
- or some nested pytree input.

The feature function should be allowed to return any pytree accepted by
`apply_fn`, not just a single dense array.

### Optional `model_state`

Needed for frameworks such as Flax when inference depends on mutable
collections such as batch statistics.

The wrapper should support:

- no model state,
- immutable state passed into inference,
- optional extraction of the `"params"` collection when the user passes a
  full variables dict directly.

### Optional `apply_kwargs`

Extra static kwargs forwarded to `apply_fn`, for example `train=False`.

These should be treated as static wrapper configuration, not dynamic
solve-time inputs.

---

## Output-shape contract

The wrappers should normalise framework outputs into the shapes already
expected by `StateSpace.build(...)`.

### `bind_intensity`

- expected model output: `(batch, D)` or broadcast-compatible equivalent,
- wrapper output: `(batch, D)`.

### `bind_grouped_intensity`

- expected model output: `(batch, D, K)` or `(K, batch, D)`,
- wrapper normalises to: `(K, batch, D)`,
- `K` must equal `len(outputs)`.

### `bind_exit_intensity`

- expected model output: `(batch, D, K)` or `(K, batch, D)`,
- wrapper normalises to: `(K, batch, D)`,
- `K` must equal the number of exits from the given source state.

The wrappers should accept one explicit `output_axis` option for grouped
and exit models instead of trying to infer arbitrary layouts.

Recommended default:

```python
output_axis=-1
```

so `(batch, D, K)` is the default expectation for modern NN code.

---

## Validation rules

Validation should happen at wrapper-construction time where possible.

### Structural checks

- `apply_fn` must be callable.
- `feature_fn` must be callable.
- grouped `outputs` must be non-empty.
- `apply_kwargs` must be a mapping if provided.

### Shape checks

The wrapper should run one eager reference call during construction
using small dummy arrays, analogous to other `jact` validation paths.

It should validate:

- transition wrappers produce rank-2 `(batch, D)` output or a supported
  broadcastable equivalent,
- grouped/exit wrappers produce one supported rank-3 layout,
- output count matches declared transitions or exits.

Error messages should mention:

- expected shape family,
- actual shape received,
- which wrapper constructor failed.

### No hidden transition mapping

Grouped wrappers should require the user to declare output transition
order explicitly. `jact` should not guess transition identity from model
output names or dataclass fields.

---

## Recommended implementation strategy

### Phase 1: generic wrappers only

Implement framework-agnostic wrappers that operate on plain callables:

- `bind_intensity`
- `bind_grouped_intensity`
- `bind_exit_intensity`

Internally these produce ordinary solver callables and do not modify the
solver.

This phase delivers nearly all usability value while keeping risk low.

### Phase 2: optional convenience aliases

If Phase 1 lands well, consider adding thin aliases specialised for
common patterns, for example:

```python
jact.bind_flax_intensity(...)
```

These should remain optional sugar over the generic wrappers, not a
separate capability layer.

### Phase 3: higher-level build helpers only if needed

Only after wrapper constructors exist should we consider a higher-level
API such as:

```python
state_space.build_from_model(...)
```

This should be deferred. It adds another entry point and risks API
duplication before we know whether the wrappers alone are sufficient.

---

## Example target usage

### Separate per-transition models

```python
def feature_fn(t, d, *, age):
    attained_age = age[:, None] + t
    return jnp.stack([attained_age, d], axis=-1)

model = state_space.build(
    transitions={
        ("healthy", "disabled"): jact.bind_intensity(
            apply_fn=onset_net.apply,
            params=onset_params,
            feature_fn=feature_fn,
        ),
        ("healthy", "dead"): jact.bind_intensity(
            apply_fn=healthy_dead_net.apply,
            params=healthy_dead_params,
            feature_fn=feature_fn,
        ),
        ("disabled", "dead"): jact.bind_intensity(
            apply_fn=disabled_dead_net.apply,
            params=disabled_dead_params,
            feature_fn=feature_fn,
        ),
    }
)
```

### One joint multi-output model

```python
joint_intensity = jact.bind_grouped_intensity(
    apply_fn=joint_net.apply,
    params=joint_params,
    feature_fn=feature_fn,
    outputs=[
        ("healthy", "disabled"),
        ("healthy", "dead"),
        ("disabled", "dead"),
    ],
)

model = state_space.build(
    groups={
        joint_intensity: [
            ("healthy", "disabled"),
            ("healthy", "dead"),
            ("disabled", "dead"),
        ]
    }
)
```

These examples are intentionally thin. The user hands over model-object
inference plus a feature function and stays inside the existing
`build(...)` grammar.

---

## Documentation plan

When the wrapper API is implemented, documentation should change in four
places:

1. `README.md`
   Add a short example showing a wrapped fitted model object rather than
   only hand-written callables.
2. `docs/example_notebook.ipynb`
   Keep the current manual callable example as the minimal baseline.
3. Add a dedicated notebook for fitted model objects
   Prefer a short Flax/Optax-style example that uses
   `bind_grouped_intensity(...)`.
4. `docs/api_spec.md`
   Document wrapper constructors as convenience layers over the core
   callable protocol.

The docs should make the layering explicit:

- solver contract: still `fn(t, d, **kwargs)`,
- wrapper layer: helper for turning trained model objects into that
  contract.

---

## Testing plan

Add tests at three levels.

### Unit tests

- transition wrapper returns `(batch, D)` with a simple fake `apply_fn`,
- grouped wrapper returns `(K, batch, D)` with both supported output
  axis conventions,
- exit wrapper validates output count against source exits,
- wrapper rejects incompatible output rank or mismatched output count.

### Integration tests

- wrapped transition model solves through `model.solve(...)`,
- wrapped grouped model solves through `model.solve(...)`,
- wrapped model supports `jax.jit` around solve,
- wrapped model supports autodiff through solve where the wrapped model
  itself is differentiable.

### Documentation tests

- notebook example executes end to end,
- README example remains synchronized with the implemented constructor
  names and argument order.

---

## Recommendation

Implement `bind_intensity`, `bind_grouped_intensity`, and
`bind_exit_intensity` as framework-agnostic wrapper constructors that
return ordinary solver callables.

This improves the fitted-model user experience substantially without
changing the solver core or replacing the existing `StateSpace.build(...)`
grammar. It also keeps the design honest: users still define the crucial
`feature_fn`, but they no longer have to hand-roll output reshaping and
`apply(...)` plumbing every time.
