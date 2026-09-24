# Derived-Field Runtime Remediation

## Status

Proposed.

## Scope

This note describes how to fix the performance limitations in Jact's derived-field runtime while preserving the existing derived-field semantics.

The intended behavior remains:

> Jact callables consume a shared exogenous field graph.

The implementation must additionally ensure that:

1. only fields required by an active consumer are evaluated; and
2. fields are evaluated at the longest valid lifetime induced by the current evaluation context.

The design below addresses both.

---

## 1. Problem statement

Jact's derived-field graph correctly models dependencies between exogenous fields, but the current runtime is too eager.

### 1.1 Eager graph evaluation

`FieldRuntime.resolve()` currently walks the complete topologically ordered graph and evaluates every node whose `t` and `d` requirements are available.

This means that resolving an intensity context may also evaluate unrelated cashflow or valuation fields.

The effect is particularly costly when unrelated nodes produce large arrays, for example:

- annuity-rate tensors,
- retirement tapers,
- grouped payment matrices,
- fixed-cost lookup arrays,
- benefit feature tensors.

The current runtime therefore shares computation only within a context after it has already decided to materialize too much of the graph.

### 1.2 Duration lifetime is too coarse

Each node currently carries transitive flags roughly equivalent to:

```text
needs_time
needs_duration
```

Any field depending on `d` is therefore treated as runtime-context dependent.

This loses an important distinction.

For example:

```python
duration_basis = lambda d: spline_basis(d)
```

has very different execution properties depending on what `d` means.

On a canonical density grid, `d` may be invariant across the entire solve and the basis can be resolved once before the state scan.

On a point-mass characteristic,

```text
d = d0 + t
```

the same field varies with time and individual and should normally be resolved in the point context.

The lifetime belongs to the field **under a particular context binding**, not to the field definition alone.

---

## 2. Design goal

Replace eager graph interpretation with statically prepared **resolution plans**.

A resolution plan combines:

1. the fields required by a consumer;
2. the transitive dependency closure of those fields;
3. the evaluation context in which canonical fields such as `t` and `d` are bound;
4. the effective lifetime of every node under that context.

The runtime then executes only the required portions of the graph and hoists each portion to the longest valid lifetime.

Conceptually:

```text
FieldGraph
    +
Consumer requirements
    +
Evaluation context
    ↓
ResolutionPlan
    ↓
lifetime-partitioned field evaluation
    ↓
callable
```

---

## 3. Principle A: demand-driven dependency closure

Field evaluation must be demand-driven.

For a consumer requiring fields

```text
R = {r1, r2, ...}
```

define its required closure as:

```text
C = R ∪ ancestors(R)
```

Only nodes in `C` are eligible for evaluation.

No other derived field may be materialized merely because its dependencies are available.

For example:

```text
age ───────────────┐
duration_basis ────┼──> risk_features ──> mortality intensity
sex ───────────────┘

salary_index ──────┐
annuity_rate ──────┼──> benefit_features ──> payment
salary ────────────┘
```

Resolving the mortality intensity must not evaluate:

```text
salary_index
annuity_rate
benefit_features
```

even if they are valid under the same `(t, d)` context.

---

## 4. Consumer requirements must be static

The set of fields required by a callable must be known before tracing the solver step.

Runtime inspection of Python code or dictionary access is not acceptable.

The preferred rule is:

> Consumer-field dependencies are resolved statically when a callable is bound.

There are three possible mechanisms.

### 4.1 Named callable parameters

Preferred where practical:

```python
def mortality(t, d, *, risk_features):
    ...
```

The required field set is directly inspectable:

```text
{"risk_features"}
```

### 4.2 Explicit metadata

For callables that retain the current `**kwargs` style:

```python
jact.bind(..., uses=("risk_features",))
```

or an equivalent internal annotation may declare the dependency set.

### 4.3 Conservative fallback

A callable using unconstrained `**kwargs` without dependency metadata may conservatively mean:

```text
requires all visible fields
```

This preserves compatibility but forfeits demand-driven pruning for that callable.

Jact should not inspect callable source code to infer `kwargs["name"]` accesses.

---

## 5. Consumer specifications

Each solver consumer should have a static specification.

Conceptually:

```text
ConsumerSpec
    callable
    required_fields
    evaluation_kind
```

Relevant consumer classes include:

- single-transition intensity,
- grouped intensity,
- state-rate payment,
- transition-lump payment,
- scheduled-event payment,
- duration-event payment,
- event-time callable,
- duration-target callable,
- cashflow-view weight.

The `required_fields` set identifies terminal nodes of the field graph needed by that consumer.

The field graph can precompute the dependency closure for each unique requirement set.

---

## 6. Principle B: lifetime is context-dependent

A field does not have one universal runtime lifetime.

The lifetime is induced by the bindings supplied by an evaluation context.

For example:

```python
duration_basis = lambda d: B(d)
```

has one definition but multiple possible lifetimes:

```text
density midpoint grid:
    d = fixed canonical grid
    -> layout-static

payment left grid:
    d = another fixed canonical grid
    -> layout-static

point mass:
    d = d0 + t
    -> context-local / step-local
```

Therefore `needs_duration=True` is insufficient as an execution classification.

---

## 7. Evaluation contexts

The solver should define a finite set of evaluation-context kinds.

A context binds canonical fields such as `t` and `d` to concrete solver geometry.

Typical contexts include:

```text
intensity density midpoint
intensity point midpoint
payment density midpoint
payment density left edge
payment point midpoint
payment point left edge
time-only weight midpoint
time-only weight left edge
scheduled-event phase
duration-event phase
```

A context need not correspond one-to-one with every call site. Contexts should be shared when their canonical bindings and geometry are identical.

Conceptually:

```text
EvaluationContext
    t_binding
    d_binding
    geometry_identity
    phase
```

The important part is the semantic binding, not the exact class shape.

---

## 8. Lifetime lattice

Jact should use a small execution-lifetime lattice.

A useful ordering is:

```text
solve-static
    <
layout-static
    <
phase-local
    <
context-local
```

where "less than" means longer-lived and therefore more hoistable.

Suggested meanings:

### 8.1 Solve-static

Depends only on solve inputs and other solve-static fields.

Examples:

```text
scaled_salary = salary / 1000
sex_embedding = table[sex]
```

Resolved before the scan.

### 8.2 Layout-static

Depends on a canonical geometry layout that is fixed for the solve.

Examples:

```text
B(duration_mid_grid)
B(payment_left_grid)
lookup(compact_intensity_grid)
```

Resolved before the scan once per required layout.

### 8.3 Phase-local / time-local

Depends on `t` but not on a dynamic `d`.

Examples:

```text
attained_age = baseline_age + t
discount = exp(-r * t)
```

Resolved once for the relevant step or evaluation phase.

### 8.4 Context-local

Depends on context-specific geometry whose value changes at runtime.

Examples:

```text
B(d0 + t)
F(t, d0 + t, x)
```

for point masses.

Also includes joint `t,d` fields when `d` is dynamic.

---

## 9. Lifetime propagation

Lifetime should be derived by dependency propagation.

Let `L_ctx(x)` denote the lifetime of field `x` under evaluation context `ctx`.

For a derived field

```text
z = f(x1, ..., xn)
```

define:

```text
L_ctx(z) = join(
    L_ctx(x1),
    ...,
    L_ctx(xn)
)
```

where `join` selects the shortest-lived dependency.

Example:

```text
baseline_age      solve-static
t                 phase-local
d_grid            layout-static

age = baseline_age + t
                  -> phase-local

basis = B(d_grid)
                  -> layout-static

features = F(age, basis)
                  -> phase-local
```

For a point context:

```text
d_point = d0 + t  context-local

basis = B(d_point)
                  -> context-local

features = F(age, basis)
                  -> context-local
```

This removes the need for fixed node-level flags such as:

```text
needs_time
needs_duration
```

Those may still be useful as descriptive metadata, but they must not determine execution lifetime.

---

## 10. Resolution plans

A `ResolutionPlan` should be prepared statically for each relevant combination of:

```text
consumer requirement set
evaluation context kind
```

The plan contains:

```text
required terminal fields
transitive dependency closure
topological evaluation order
lifetime of each node under the context
partition of nodes by lifetime
```

Example for a density intensity:

```text
required:
    risk_features

closure:
    baseline_age
    sex
    age
    duration_basis
    risk_features

solve-static:
    baseline_age
    sex

layout-static:
    duration_basis

phase-local:
    age
    risk_features

context-local:
    none
```

For the same intensity evaluated on a point mass:

```text
required:
    risk_features

closure:
    same nodes

solve-static:
    baseline_age
    sex

layout-static:
    none

phase-local:
    age

context-local:
    duration_basis
    risk_features
```

The producer graph is the same.

Only the context changes.

---

## 11. Runtime structure

The current step-local `FieldRuntime` should no longer be the sole cache.

Instead, field values should live at the lifetime where they are valid.

Conceptually:

```text
SolveFieldValues
    solve-static values

LayoutFieldValues
    layout-static values keyed by layout identity

StepFieldRuntime
    phase-local values

ContextFieldRuntime
    context-local values
```

A callable environment is assembled from the relevant layers:

```text
solve-static
    +
layout-static
    +
phase-local
    +
context-local
```

Only fields in the active consumer's resolution plan are exposed or materialized.

---

## 12. Canonical grid hoisting

Duration-only fields on canonical grids should be hoisted before the scan.

For example:

```python
duration_basis = lambda d: spline_basis(d)
```

may need separate resolved values for:

```text
intensity midpoint grid
payment midpoint grid
payment left-edge grid
compact intensity grid under a duration limit
compact payment grid under a duration limit
```

Each distinct layout should be resolved at most once per solve if some active consumer requires the field on that layout.

This is one of the primary performance wins expected from the derived-field system.

---

## 13. Layout identity

The runtime needs a stable notion of layout identity.

A layout identity should represent semantic solver geometry, for example:

```text
("intensity", "mid", intensity_duration_limit)
("payment", "mid", payment_duration_limit)
("payment", "left", payment_duration_limit)
```

or an equivalent prepared-layout object.

The identity should describe the layout structurally.

It must not use traced array values as Python dictionary keys.

Resolved layout-static arrays should be dynamic JAX values associated with statically known layout slots.

---

## 14. Point-mass fields

Point-mass duration is:

```text
d_point = d0 + t
```

or its duration-limited equivalent.

A duration-derived field therefore follows the point characteristic:

```text
B_point(t, b) = B(d0[b] + t)
```

This is still fully exogenous.

However, Jact should distinguish:

```text
semantic resolvability
```

from:

```text
execution hoisting
```

The entire point trajectory can in principle be known before state evolution, but materializing large `(time, batch, feature...)` tensors may be more expensive than recomputing them stepwise.

Therefore the default execution lifetime for point-duration fields should remain context-local unless a future cost model or explicit strategy justifies precomputation.

Canonical grid fields are the safe first target for pre-scan hoisting.

---

## 15. Time-only sharing

Time-dependent fields should also be shared across consumers using the same phase.

For example:

```python
age = lambda t, baseline_age: baseline_age + t
discount = lambda t, rate: jnp.exp(-rate * t)
```

If several consumers evaluate at the same midpoint time, each required time-only field should be resolved once for that phase.

A payment left-edge evaluation and an intensity midpoint evaluation are different contexts because their `t` bindings differ.

The runtime should therefore share by semantic phase, not merely by solver-step index.

---

## 16. Avoiding accidental cross-scope work

Model, cashflow, and solve-derived fields may continue to compose into one graph.

Separate graph namespaces are not required for performance.

Demand-driven closure is sufficient to prevent unrelated work:

```text
model field not required by this payment
    -> not evaluated

cashflow field not required by this intensity
    -> not evaluated
```

Scope remains an ownership concept.

Demand determines execution.

This preserves the elegant unified environment without reintroducing model/cashflow/view silos.

---

## 17. Static versus dynamic data

The remediation must preserve JAX compilation reuse.

The following should be static solver structure:

```text
FieldGraph topology
ConsumerSpecs
ResolutionPlans
evaluation-context kinds
layout structure
```

Resolved numeric field arrays should remain ordinary dynamic PyTree leaves.

Do not create new static callables or closure-captured numeric constants for each solve.

In particular, layout-static precomputed arrays should be passed through the compiled solver as dynamic values where appropriate.

This allows repeated fixed-shape batches to reuse the same compiled program.

---

## 18. Proposed internal API shape

Exact naming is open, but the architecture could resemble:

```python
graph = FieldGraph.build(...)

consumer = ConsumerSpec(
    required_fields=("risk_features",),
    ...
)

plan = graph.plan(
    required=consumer.required_fields,
    context=DENSITY_INTENSITY_MID,
)

prepared = prepare_fields(
    graph,
    plans=all_plans,
    solve_inputs=solve_inputs,
    layouts=layouts,
)
```

Then inside the step:

```python
env = prepared.resolve(
    plan,
    t=current_t,
    d=current_d,
)
```

`resolve()` executes only the plan's unresolved phase-local or context-local nodes.

The graph itself is no longer interpreted eagerly on every call.

---

## 19. Migration path

A staged implementation is preferable.

### Phase 1: demand-driven closure

Add static consumer dependency sets and evaluate only their graph closure.

This should immediately eliminate accidental model/cashflow cross-evaluation.

Keep current lifetime behavior initially if necessary.

### Phase 2: layout-static duration hoisting

Introduce canonical layout identities.

Resolve required duration-only subgraphs once per layout before the scan.

This targets the largest obvious missed optimization.

### Phase 3: generalized lifetime propagation

Replace `needs_time` / `needs_duration` execution decisions with context-induced lifetime analysis.

Unify solve-static, layout-static, time-local, and context-local planning.

### Phase 4: optional point-characteristic optimization

Only after profiling, consider selectively precomputing point-characteristic fields.

This should be driven by measured compute-versus-memory tradeoffs rather than semantic purity.

---

## 20. Validation requirements

The implementation should reject or conservatively handle:

- cycles in derived fields,
- missing dependencies,
- duplicate field names,
- ambiguous consumer dependency declarations,
- a consumer declaring a field that does not exist,
- a resolution plan requiring `d` in a context without a duration binding,
- a resolution plan requiring `t` in a context without a time binding.

The field graph remains purely exogenous.

No state-dependent fields are introduced.

---

## 21. Correctness tests

The remediation must preserve existing numerical results.

Required tests include:

### 21.1 Dependency pruning

Construct:

```text
A -> B -> consumer 1
C -> D -> consumer 2
```

Verify that consumer 1 never evaluates `C` or `D`.

### 21.2 Shared dependency

Construct:

```text
A -> B -> consumer 1
     B -> consumer 2
```

Verify that `B` is evaluated once per shared context.

### 21.3 Grid-static duration field

Define:

```python
basis = lambda d: ...
```

Verify that density-grid resolution occurs once per distinct canonical layout, not once per solver step.

### 21.4 Point duration field

Verify that the same `basis(d)` definition is evaluated on:

```text
d = d0 + t
```

for point contexts and produces the same result as direct callable evaluation.

### 21.5 Time phase separation

Verify that midpoint and left-edge consumers receive different values for fields depending on `t`.

### 21.6 Scope composition

Verify that:

- model-derived fields can feed cashflows,
- cashflow-derived fields can feed solve-derived fields,
- unused fields from any scope are not evaluated.

### 21.7 JIT and PMAP

Verify correctness under:

- single-device JIT,
- multiple devices,
- gradients,
- repeated solves with identical shapes.

---

## 22. Performance tests

The feature should have dedicated performance regressions.

### 22.1 Unused expensive field

Add an expensive payment-derived field to a model with no active payment consumer.

Expected:

```text
no measurable intensity runtime regression
```

### 22.2 Shared expensive field

Use several consumers of one expensive derived field.

Expected:

```text
one field evaluation per shared context
```

### 22.3 Duration-basis benchmark

Compare:

```text
basis computed inside each callable
```

against:

```text
layout-static derived basis
```

for increasing:

- number of transitions,
- duration-grid width,
- batch size.

Expected:

```text
derived-field version improves as reuse increases
```

### 22.4 Payment-heavy profitability model

Reproduce the profitability workload that exposed the issue.

Measure separately:

- derived-field preparation,
- state scan,
- cashflow evaluation,
- full solve,
- peak memory / HBM traffic if available.

The remediation should demonstrate that unrelated payment features are absent from intensity traces and that fixed duration transforms are hoisted.

---

## 23. Success criteria

The remediation is successful when all of the following hold:

1. resolving a callable never evaluates fields outside its static dependency closure;
2. duration-only fields on fixed canonical grids are not recomputed every solver step;
3. the same field definition may have different lifetimes under density and point contexts;
4. model, cashflow, and solve-derived fields continue to compose into one graph;
5. no solver-state dependency is introduced;
6. existing numerical behavior is preserved;
7. JIT and PMAP compilation reuse is not degraded;
8. representative shared-feature workloads show measurable runtime or memory improvement.

---

## 24. Final design statement

The corrected runtime should implement the following principle:

> **A derived field is evaluated only when demanded by an active consumer, and only at the shortest scope required by the dependencies of that field under the current solver evaluation context.**

Equivalently:

```text
FieldGraph
    defines what depends on what

ConsumerSpec
    defines what is needed

EvaluationContext
    defines what t and d mean

ResolutionPlan
    determines what to compute and how long it may live
```

This preserves the original elegance of the shared exogenous field graph while making the performance model match the semantic model.
