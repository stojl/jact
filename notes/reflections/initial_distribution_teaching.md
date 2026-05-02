# Teaching `InitialDistribution`

This note is about how to explain `InitialDistribution` clearly. It is not a
design note and not an implementation note. The semantics are already sound.
The problem is that the object carries several important ideas at once, and the
current documentation mostly states those ideas rather than teaching them in a
good order.

The recommendation in this note is simple:

- teach structure first,
- teach runtime numeric values second,
- teach constructors as a ladder from simple to powerful,
- repeat the reduction rule in more than one place.

---

## The main teaching rule

`InitialDistribution` has two jobs:

1. declare the structural initial-state set,
2. describe the runtime `(state, duration)` distribution within that set.

That is the core framing. Everything else follows from it.

This should be the first thing a user learns, because it explains the most
important distinction in the API:

- declared initial states are structural,
- masses and durations are numeric values inside that structure.

That in turn explains the reduction behavior:

- reduction follows the declared structural initial-state set,
- reduction does not inspect runtime mass support,
- a declared state with zero mass still remains part of the structural initial
  set.

This is the rule that should be repeated in docstrings, in the API spec, and in
at least one worked example.

---

## Why users get confused

The difficulty is not that `InitialDistribution` is poorly named. The object is
harder to learn because it brings together four distinct concerns:

- which states are structurally declared as possible initial states,
- how much mass each individual places in those states,
- what duration value is attached to each initial state,
- whether per-individual indices refer to a restricted initial-state tuple or
  to the model's full state list.

Those are all real concerns. The documentation should not pretend otherwise.
What it can do is explain them in a stable order so the user does not have to
infer the model from scattered rules.

The most common confusion is thinking that runtime masses determine the
structural initial-state set. They do not. The structural set is declared by
construction:

- the key set of `components`,
- the state passed to `at(...)`,
- the `initial_states` tuple passed to `per_individual(...)`,
- or, when `per_individual(initial_states=None)`, the model's full state list.

Runtime mass never shrinks that structure.

---

## Teach the constructors as a progression

The constructors should not be presented as flat peers. They are easier to
understand when taught in increasing order of power.

### 1. Single-state shorthand

```python
model.solve(initial="healthy", ...)
```

Teach this first. It is the simplest entry path:

- one declared initial state,
- all mass starts there,
- duration is zero unless `initial_duration` is supplied.

The user should be told that this is sugar for the explicit single-state form.

### 2. Single-state explicit form

```python
initial = jact.InitialDistribution.at("healthy", duration=2.0)
```

This should be introduced as "the same structure, but explicit." Its value is
not complexity for its own sake. Its value is that it exposes the object model:

- one declared state,
- one duration assignment,
- structural declaration made explicit.

### 3. Mixture over declared initial states

```python
initial = jact.InitialDistribution(
    components={
        "healthy": {"mass": 0.7, "duration": 0.0},
        "disabled": {"mass": 0.3, "duration": 4.0},
    }
)
```

This is the first place where the structure-versus-values distinction should be
spelled out directly:

- `"healthy"` and `"disabled"` are the declared structural initial states,
- `mass` and `duration` are runtime numeric data attached to them.

This is also the right place to teach the zero-mass rule:

```python
initial = jact.InitialDistribution(
    components={
        "healthy": {"mass": 1.0, "duration": 0.0},
        "disabled": {"mass": 0.0, "duration": 4.0},
    }
)
```

Even here, `"disabled"` is still part of the declared structural initial-state
set. That should be shown, not only stated.

### 4. Per-individual indices into a declared initial-state set

```python
initial = jact.InitialDistribution.per_individual(
    states=idx,
    duration=d0,
    initial_states=("healthy", "disabled"),
)
```

This is where the constructor becomes more abstract. The key teaching point is
that `states` does not name model states directly. It indexes into the declared
`initial_states` tuple.

So the semantics are:

- the declared structural initial-state set is `("healthy", "disabled")`,
- each individual picks one of those states by index,
- reduction follows that declared tuple.

This should be taught as a restricted index space.

### 5. Per-individual indices into the full model state list

```python
initial = jact.InitialDistribution.per_individual(
    states=full_state_idx,
    duration=d0,
    initial_states=None,
)
```

This is the mode users are most likely to misread unless the docs are very
explicit. The key teaching sentence should be:

`initial_states=None` means the indices refer to the model's full state list.

That is the whole semantic pivot. It is not a small optional flag. It changes
what the indices mean.

This should be taught as the full-model index-space mode, in direct contrast to
the restricted-tuple mode above.

---

## Common confusions to teach explicitly

The public docs should contain a short section that answers the same confusions
every time.

### 1. Zero mass does not remove a declared state

If a state appears in `components`, it is structurally declared even when its
mass is zero.

### 2. Declared states are not inferred from runtime mass

The solver never looks at realized mass values to decide which states count as
initial states for reduction.

### 3. `per_individual` has two index modes

- `initial_states=(...)` means indices refer to that tuple.
- `initial_states=None` means indices refer to the model's full state list.

This should be shown side by side in the docs.

### 4. The shorthand forms are convenience, not different semantics

`initial="healthy"` and `InitialDistribution.at("healthy", ...)` are not two
different models of the world. One is sugar over the other.

---

## Documentation recommendations

The semantics do not need to change. The teaching order does.

### 1. Start with the structural rule in the API spec

The `InitialDistribution` section in `docs/api_spec.md` should lead with the
two-job framing:

- structural declaration of initial states,
- runtime numeric distribution within that declaration.

The reduction rule should appear immediately after that, not only later in a
bullet list.

### 2. Reorder examples to match the constructor ladder

Examples should move in this order:

- single-state shorthand,
- single-state explicit form,
- multi-state mixture with `components`,
- `per_individual` over a declared tuple,
- `per_individual` over the full model state list.

That gives the user one mental model that grows, instead of several unrelated
entry points.

### 3. Add one explicit reduction example

There should be one example whose only job is to show that declared structure,
not runtime mass support, drives reduction.

For example:

```python
initial = jact.InitialDistribution(
    components={
        "healthy": {"mass": 1.0, "duration": 0.0},
        "disabled": {"mass": 0.0, "duration": 2.0},
    }
)
```

The explanatory text should say plainly that `"disabled"` is still part of the
declared initial-state set even though its runtime mass is zero.

### 4. Add a short “common confusions” subsection

This should appear in the API spec or an example notebook, not only in private
notes. The confusion points are stable enough that users will keep running into
them unless they are answered directly.

### 5. Keep the implementation language out of the main teaching path

Users do not need to learn canonicalization details, point-mass seeding, or
internal solver representation before they understand the object. Those belong
in implementation notes, not in the first teaching surface.

---

## Bottom line

`InitialDistribution` does not need a simpler semantic model. It needs a better
taught one.

The right teaching strategy is:

- structure first,
- values second,
- constructors as a progression,
- repeated emphasis that reduction follows declared structure, not runtime mass.

If the docs adopt that framing consistently, the existing object should become
substantially easier to learn without changing the API at all.
