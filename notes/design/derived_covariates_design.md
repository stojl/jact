# Derived Covariates and the Shared Exogenous Field Graph

## Status

Proposed.

## Summary

`jact` currently exposes three conceptually separate kinds of inputs to solver callables:

- solver time `t`,
- state duration `d`,
- solve-time covariates passed through `**kwargs`.

This proposal unifies them.

The solver environment is an **acyclic graph of exogenous fields**. Its roots are solve inputs and solver primitives. Solver-provided coordinate fields such as `t` and `d`, together with user-defined derived covariates, are fields in that same graph.

A field may depend on:

- supplied solve-time covariates,
- solver-provided fields such as `t` and `d`,
- fixed model parameters,
- other derived fields.

It may never depend on the evolving solver state.

Model definitions, cashflow declarations, and individual solves may each contribute derived fields. These declarations compose into one shared graph whose resolved values form the common environment consumed by intensities, payments, cashflow weights, and other solver callables.

The original motivation is performance: shared feature construction should not be repeated independently inside every callable. The larger result is architectural. Feature construction, solver coordinates, and solve-time inputs become parts of one explicit exogenous environment.

---

## 1. Motivation

The current callable contract is intentionally small:

```python
fn(t, d, **kwargs) -> array
```

Here:

- `t` is solver clock time,
- `d` is duration in the current state,
- `**kwargs` contains solve-time covariates.

In practice, many callables reconstruct the same quantities:

```python
attained_age = baseline_age + t
duration_basis = spline_basis(d)
calendar_factor = f(t, scenario)
features = make_features(attained_age, duration_basis, sex)
```

If several intensities, payments, or valuation functions use the same derived quantity, the same transformation may be repeated many times.

That creates three problems:

1. **Repeated computation.** Shared features are reconstructed independently.
2. **Repeated modelling logic.** Important transformations are duplicated across callables.
3. **Weak ownership.** Features intrinsic to a model or cashflow declaration are hidden inside low-level functions.

The proposed field graph addresses all three.

---

## 2. The generalization

The fundamental object is not a fixed covariate array.

It is an **exogenous field available to solver callables**.

Examples include:

```text
baseline_age
sex
salary
t
d
attained_age
duration_basis
risk_features
discount_factor
```

Some fields are supplied by the user. Some are supplied by the solver. Some are derived from others.

For example:

```text
attained_age = baseline_age + t
duration_basis = B(d)
risk_features = f(attained_age, duration_basis, sex)
```

The important point is that `t` and `d` do not need to be treated as belonging to a separate conceptual category.

They are canonical fields supplied by the solver.

---

## 3. Solver-provided fields

The solver contributes canonical fields to the shared environment.

The two fundamental examples are:

- `t`: global solver clock time,
- `d`: duration in the current state.

Conceptually, these are no different from other derived fields except for their provenance.

A useful dependency picture is:

```text
solver primitives ──> t ───────────────┐
                                       ├── attained_age
baseline_age ───────────────────────────┘

solver geometry ────> d ──> duration_basis

attained_age ───────┐
duration_basis ─────┼──> risk_features
sex ────────────────┘
```

This means the field algebra does not need special categories such as:

```text
time covariate
duration covariate
time-duration covariate
```

Those are consequences of dependency structure, not separate types.

---

## 4. Evaluation context

Although `t` and `d` participate in the same field graph as user-defined quantities, the solver still determines their meaning.

The useful conceptual object is an **evaluation context**.

An evaluation context supplies the canonical solver fields appropriate to a particular callable evaluation:

```text
evaluation context
       │
       ├── t
       └── d
```

Derived fields are then evaluated relative to that context.

This is especially important for `d`. Duration is not merely a global coordinate array. It means duration in the current state, and its concrete representation may differ across legitimate solver evaluation contexts.

The abstraction should therefore be:

> The solver provides canonical fields such as `t` and `d`; user-defined fields derive from them.

rather than:

> The user-facing field graph must expose all low-level numerical geometry.

The numerical scheme remains concrete. The field layer begins at the canonical solver fields.

---

## 5. The shared exogenous field graph

The solver should be understood as operating in two conceptual stages:

```text
solve inputs
    +
solver-provided fields
    ↓
resolve exogenous field graph
    ↓
shared environment
    ↓
state evolution
```

The resulting environment is shared by all downstream consumers:

```text
                  ┌─ intensities
                  ├─ cashflow payments
shared environment├─ cashflow weights
                  └─ other solver callables
```

A callable should not need to know how a value was produced.

For example:

```python
def mortality(t, d, **kwargs):
    features = kwargs["features"]
    ...
```

`features` may have been:

- supplied directly,
- derived only from solve inputs,
- derived from `t`,
- derived from `d`,
- derived jointly from `t` and `d`,
- composed from several other fields.

From the consumer's perspective, it is simply part of the environment.

---

## 6. Derived fields form a dependency graph

Derived fields may depend on other derived fields.

For example:

```text
baseline_age ─┐
              ├── attained_age ─────┐
t ────────────┘                     │
                                    ├── risk_features
d ────────> duration_basis ─────────┤
sex ────────────────────────────────┘
```

A declaration may look like:

```python
derived={
    "attained_age":
        lambda t, baseline_age: baseline_age + t,

    "duration_basis":
        lambda d: spline_basis(d),

    "risk_features":
        lambda attained_age, duration_basis, sex:
            make_features(attained_age, duration_basis, sex),
}
```

Function parameters declare dependencies.

The graph must be acyclic.

The dimensionality and solver-domain dependence of each field follow from its ancestry. Users should not manually declare execution schedules or axis types unless future experience demonstrates a genuine need.

---

## 7. Hard boundary: no solver-state dependencies

The field graph is strictly exogenous to state evolution.

Allowed dependencies include:

- solve inputs,
- fixed model parameters,
- solver-provided fields,
- other derived fields.

Forbidden dependencies include:

- state probabilities,
- state densities,
- point-mass values,
- accumulated cashflows,
- any other evolving solver state.

The dependency direction is one-way:

```text
solve inputs + solver fields
            ↓
      derived fields
            ↓
        callables
            ↓
      state evolution
```

There is no arrow back upward.

This is a numerical-method boundary, not merely an API preference.

If a derived quantity depended on the current numerical solution, its meaning would depend on choices such as:

- left-endpoint state,
- midpoint state,
- predictor state,
- corrected state,
- explicit versus implicit evaluation.

Those choices define numerical schemes and must not be introduced implicitly through the field mechanism.

A useful rule is:

> If a quantity is completely determined by solve inputs, fixed parameters, solver-provided fields, and other exogenous fields, it may belong to the shared field graph.

---

## 8. Pre-state-evolution semantics

The complete exogenous graph is conceptually determined before state evolution begins.

This applies even when fields vary over time or duration.

For example:

```text
attained_age(t) = baseline_age + t
duration_basis(d) = B(d)
risk_surface(t, d) = f(attained_age(t), duration_basis(d))
```

All three are exogenous.

Once the solve inputs and relevant solver geometry are known, their values are determined independently of the evolving solution.

"Resolved before the scan" is therefore a semantic statement rather than a requirement about a specific tensor representation.

The state evolution consumes a prescribed field environment. It does not define or mutate that environment.

---

## 9. Ownership and declaration scopes

Derived fields may naturally belong to different reusable objects.

The proposal has three declaration scopes:

1. model-level,
2. cashflow-declaration-level,
3. solve-level.

All three contribute nodes to one combined graph.

### 9.1 Model-level fields

Features intrinsic to the transition model should be declared with the model.

```python
model = state_space.build(
    transitions=...,
    derived={
        "age":
            lambda t, baseline_age: baseline_age + t,

        "duration_basis":
            lambda d: spline_basis(d),

        "risk_features":
            lambda age, duration_basis, sex:
                make_risk_features(age, duration_basis, sex),
    },
)
```

Then solving supplies only concrete inputs:

```python
result = model.solve(
    initial="healthy",
    horizon=30,
    steps_per_unit=12,
    baseline_age=baseline_age,
    sex=sex,
)
```

This couples reusable feature construction to the intensities that conceptually depend on it.

### 9.2 Cashflow-level fields

A reusable `CashflowDeclaration` should also be able to contribute derived fields.

```python
cashflows = state_space.cashflows(
    {
        "salary": jact.StateRate(...),
        "death": jact.TransitionLump(...),
    },
    derived={
        "salary_index":
            lambda t, inflation:
                index_salary(t, inflation),

        "benefit_features":
            lambda age, salary_index, salary:
                make_benefit_features(age, salary_index, salary),
    },
)
```

Cashflow-derived nodes may depend on model-derived nodes.

This makes reusable payment logic self-contained without creating a separate cashflow-only environment.

### 9.3 Solve-level fields

An individual solve may extend the graph with run-specific quantities.

```python
result = model.solve(
    initial="healthy",
    horizon=30,
    steps_per_unit=12,

    cashflows=cashflows,
    cashflow_views=views,

    derived={
        "discount":
            lambda t, interest_rate:
                jnp.exp(-interest_rate * t),
    },

    baseline_age=baseline_age,
    sex=sex,
    salary=salary,
    inflation=inflation,
    interest_rate=interest_rate,
)
```

Solve-level fields are appropriate for quantities belonging to a particular valuation or run rather than to the reusable model or cashflow declaration.

---

## 10. One graph, not separate namespaces

Although definitions may originate from the model, cashflow declaration, or solve call, they compose into one graph:

```text
model-derived
      +
cashflow-derived
      +
solve-derived
      ↓
combined exogenous field DAG
      ↓
shared environment
```

Declaration scope represents **ownership and reuse**, not visibility.

A cashflow may consume a model-derived feature.

A solve-level field may consume model- or cashflow-derived fields.

All downstream callables consume the same shared environment.

This is essential to the original resource-sharing motivation. Separate environments for intensities, cashflows, and valuation would recreate duplication at a different layer.

---

## 11. Views consume but do not initially own fields

Cashflow views may depend on derived fields, especially through weighting functions.

However, individual views should not initially introduce their own local `derived` mappings.

Instead, view-specific derived quantities should normally be declared at solve level.

For example:

```python
views = {
    "pv": jact.cashflows.Total(
        weight=lambda t, **kwargs: kwargs["discount"],
        terminal=True,
    ),
}

result = model.solve(
    ...,
    cashflows=cashflows,
    cashflow_views=views,
    derived={
        "discount":
            lambda t, interest_rate:
                jnp.exp(-interest_rate * t),
    },
    interest_rate=interest_rate,
)
```

This avoids ambiguous questions about local view namespaces and keeps sharing explicit.

If views later become richer reusable declarations in their own right, a view-collection-level derived scope can be considered separately.

---

## 12. Name collisions

Derived definitions from different ownership layers should compose additively rather than silently override one another.

For example, if the model defines:

```python
derived={
    "age": lambda t, baseline_age: baseline_age + t,
}
```

then a solve should not silently replace it with another definition of `"age"`.

Overriding a model-level field could unintentionally alter transition intensities and therefore change the stochastic model itself.

Preferred rule:

> Duplicate field names across composed declaration scopes are rejected.

If a different definition is required, it should receive a distinct name or the owning object should be defined differently.

---

## 13. Proposed public API

A minimal API uses a `derived` mapping at each supported ownership level.

### Model

```python
model = state_space.build(
    transitions=...,
    exits=...,
    groups=...,
    derived={
        "age":
            lambda t, baseline_age: baseline_age + t,

        "duration_basis":
            lambda d: spline_basis(d),

        "features":
            lambda age, duration_basis, sex:
                make_features(age, duration_basis, sex),
    },
)
```

### Cashflow declaration

```python
cashflows = state_space.cashflows(
    {
        "premium": jact.StateRate(...),
        "death": jact.TransitionLump(...),
    },
    derived={
        "indexed_salary":
            lambda t, salary, inflation:
                salary * salary_index(t, inflation),
    },
)
```

### Solve

```python
result = model.solve(
    initial="healthy",
    horizon=30,
    steps_per_unit=12,

    cashflows=cashflows,
    cashflow_views=views,

    derived={
        "discount":
            lambda t, interest_rate:
                jnp.exp(-interest_rate * t),
    },

    baseline_age=baseline_age,
    sex=sex,
    salary=salary,
    inflation=inflation,
    interest_rate=interest_rate,
)
```

The dependency graph is inferred from function parameter names.

---

## 14. Why `t` and `d` can remain explicit callable arguments

The generalized field model does not require changing the current public callable signature.

The callable interface may remain:

```python
fn(t, d, **kwargs)
```

even though conceptually `t`, `d`, and the values in `kwargs` belong to the same exogenous environment.

This is an ergonomic projection of the richer field graph:

```text
shared exogenous field graph
           ↓
     evaluation context
           ↓
(t, d, resolved kwargs)
           ↓
        callable
```

Keeping `t` and `d` explicit preserves readability and backwards compatibility without making them conceptually privileged inside the field system.

---

## 15. Why bare functions are sufficient

An explicit wrapper such as:

```python
jact.derived(lambda ...)
```

could be introduced, but it may not be necessary initially.

Inside an explicitly named `derived={...}` mapping, a callable is already unambiguously a field declaration:

```python
derived={
    "age": lambda t, baseline_age: baseline_age + t,
}
```

This keeps the public surface small.

A wrapper could still become useful later if derived declarations need metadata or configuration.

---

## 16. Relationship to fitted-model wrappers

The field layer complements fitted-model wrappers.

Today, a fitted intensity wrapper may construct features internally:

```python
features = feature_fn(t, d, **kwargs)
raw = apply_fn(params, features, ...)
```

If several fitted models use the same feature transformation, this duplicates expensive work.

With shared derived fields:

```python
derived={
    "features":
        lambda t, d, baseline_age, sex:
            make_features(t, d, baseline_age, sex),
}
```

several fitted models may consume the same `features`.

This makes wrappers responsible primarily for inference rather than repeated feature engineering.

---

## 17. Relationship to grouped callables

Derived fields solve the problem of **shared exogenous inputs and shared intermediate features**.

They do not necessarily solve every form of repeated callable evaluation.

A grouped model that jointly computes several transition hazards is a separate sharing concern.

The two ideas should remain conceptually distinct:

```text
derived fields
    share exogenous feature computation

grouped callable evaluation
    share model forward computation
```

Both improve reuse, but they operate at different layers.

---

## 18. Design principles

### 18.1 One field algebra

User inputs, solver-provided coordinates, and derived quantities participate in one exogenous dependency system.

### 18.2 Solver coordinates are canonical fields

`t` and `d` are supplied by the solver but are not fundamentally different from other fields in the dependency graph.

### 18.3 Exogenous only

No field may depend on evolving solver state.

### 18.4 One shared environment

Model, cashflow, and solve declarations compose into one environment rather than isolated namespaces.

### 18.5 Dependency determines domain

Users declare dependencies, not execution schedules or tensor-axis categories.

Avoid APIs such as:

```python
schedule="time"
axes=("time", "duration")
```

unless future experience demonstrates that explicit metadata is necessary.

### 18.6 Consumers remain simple

The existing callable contract may remain:

```python
fn(t, d, **kwargs)
```

Callables consume resolved values without needing to know their provenance.

### 18.7 Ownership follows reuse

- model-level definitions belong to the transition model,
- cashflow-level definitions belong to reusable cashflow logic,
- solve-level definitions belong to a particular run or valuation.

### 18.8 No silent overrides

Duplicate names across composed field declarations are rejected.

### 18.9 Pre-state-evolution semantics

The complete exogenous graph is determined independently of state evolution.

### 18.10 Numerical geometry remains explicit

The field abstraction should not attempt to turn the entire numerical solver into a generic dependency graph. Low-level solver geometry remains an implementation concern. The field system begins at canonical solver-provided fields such as `t` and `d`.

---

## 19. Examples

### 19.1 Attained age shared by several intensities

```python
model = state_space.build(
    transitions={
        ("healthy", "disabled"): disability,
        ("healthy", "dead"): mortality,
        ("disabled", "dead"): disabled_mortality,
    },
    derived={
        "age":
            lambda t, baseline_age:
                baseline_age + t,
    },
)
```

All three intensity callables may consume:

```python
kwargs["age"]
```

without independently computing attained age.

### 19.2 Shared duration basis

```python
model = state_space.build(
    transitions=...,
    derived={
        "duration_basis":
            lambda d: spline_basis(d),
    },
)
```

Any callable using the same duration basis may consume:

```python
kwargs["duration_basis"]
```

rather than rebuilding it internally.

### 19.3 Joint time-duration features

```python
model = state_space.build(
    transitions=...,
    derived={
        "age":
            lambda t, baseline_age:
                baseline_age + t,

        "duration_basis":
            lambda d:
                spline_basis(d),

        "risk_features":
            lambda age, duration_basis, sex:
                make_risk_features(
                    age,
                    duration_basis,
                    sex,
                ),
    },
)
```

`risk_features` varies jointly over solver geometry but remains completely exogenous to state evolution.

### 19.4 Cashflow reuse of model fields

```python
model = state_space.build(
    transitions=...,
    derived={
        "age":
            lambda t, baseline_age:
                baseline_age + t,
    },
)

cashflows = state_space.cashflows(
    {
        "death": jact.TransitionLump(...),
    },
    derived={
        "death_benefit":
            lambda age, salary:
                benefit_formula(age, salary),
    },
)
```

The cashflow declaration reuses the model-level `age`.

### 19.5 Solve-specific valuation

```python
result = model.solve(
    ...,
    cashflows=cashflows,
    cashflow_views={
        "pv": jact.cashflows.Total(
            weight=lambda t, **kwargs:
                kwargs["discount"],
            terminal=True,
        ),
    },
    derived={
        "discount":
            lambda t, interest_rate:
                jnp.exp(-interest_rate * t),
    },
    interest_rate=interest_rate,
)
```

The discount factor belongs to this valuation rather than to the underlying transition model or reusable cashflow declaration.

---

## 20. Non-goals

This proposal does not attempt to:

- allow model dynamics to depend on the evolving numerical solution through derived fields,
- define implicit or feedback-driven transition systems,
- replace the existing callable protocol,
- require users to specify tensor domains or solver schedules manually,
- make views independent feature namespaces,
- solve grouped-callable forward-pass sharing,
- expose low-level solver geometry as part of the public field graph,
- prescribe the internal materialization, caching, or hoisting strategy.

The proposal defines semantics and ownership first. Execution strategy can be optimized independently while preserving those semantics.

---

## 21. Conceptual definition

A concise definition for public documentation could be:

> **The solver environment is an acyclic graph of exogenous fields. Its roots are solve inputs and solver-provided primitives. Canonical solver fields such as clock time `t` and state duration `d`, together with user-defined derived fields, participate in the same dependency graph. Model, cashflow, and solve declarations may contribute derived nodes. The graph is independent of the evolving solver state and forms the shared environment consumed by solver callables.**

A shorter design statement is:

> **`jact` callables consume a shared exogenous field graph.**

The original performance motivation then follows naturally: shared dependencies correspond to shared computation rather than repeated feature construction inside individual callables.
