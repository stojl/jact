# Package API review

This note reviews the current public API of `jact` as it exists in the
repository today. It is not a replacement for the normative contract in
`docs/api_spec.md`, and it is not a redesign proposal. The goal is to assess
the API honestly in the frame that matters for this package:

- `jact` is a flexible actuarial computation kernel,
- likely users are actuaries or actuarial engineers,
- some domain concepts are canonical and should be expected to exist as
  first-class public abstractions.

That frame matters. An API review that treats `jact` as if it were trying to be
a minimal, beginner-oriented DSL would criticize the wrong things. The right
question is not "why is there so much structure?" but "is the structure aligned
with the domain, and is the user-facing complexity concentrated in the right
places?"

---

## What the API gets right

### 1. The layer split is strong

The main object model is well judged:

- `StateSpace` describes topology only.
- `Model` binds intensities to that topology.
- `InitialDistribution` describes the `(state, duration)` condition at
  `t = 0`.
- `solve()` runs the numerical kernel and emits probability and cashflow
  outputs.

This is a good separation of concerns for the problem domain. It keeps graph
structure, fitted hazard functions, starting conditions, and output choice from
collapsing into one object. That matters for actuarial work, where the same
state space is often reused with different fitted models, different cohorts,
and different valuation views.

### 2. The cashflow vocabulary is domain-correct

`StateRate`, `TransitionLump`, and `ScheduledEvent` are not arbitrary API
ornaments. They are canonical ways to describe insurance and pension cashflows:

- occupancy-conditioned rate payments,
- transition-triggered lump sums,
- state-conditioned payments at deterministic event times.

Making these explicit public types is the right decision. They express the
attachment semantics directly, they are easy to validate against a
`StateSpace`, and they give the cashflow declaration layer a recognizable
actuarial grammar.

The same applies to solve-time views such as `Raw`, `Group`, `Total`,
`ByState`, and `ByKind`. They make it possible to keep declaration and
aggregation separate, which is exactly what a reusable kernel should do.

### 3. The API is correctly biased toward validation

The package consistently validates structural mistakes early:

- unknown states and transitions,
- missing or overlapping model coverage,
- invalid initial-state declarations,
- inconsistent batch shapes,
- invalid cashflow attachments and views.

That is a strength. In this domain, silent acceptance of a structurally wrong
model is much worse than a somewhat strict entry surface.

### 4. The package keeps the computation kernel flexible

The callable-based intensity protocol is a good core contract. It is small,
JAX-native, and leaves room for parametric hazards, GLMs, neural models, or
hand-written functions without forcing any one modeling stack.

That flexibility is a real asset. The API does not assume that users all fit
their models the same way, and it does not prematurely hard-code one feature
construction story.

---

## Where the API is under pressure

The main weaknesses are not in the conceptual decomposition. They are in the
ergonomics of the public surface that sits on top of that decomposition.

### 1. `solve()` is carrying too many concerns at once

`Model.solve(...)` is powerful, but it is doing a great deal of API work in one
place. One call currently handles:

- initial-condition shorthand versus full initial distributions,
- reduction to the reachable subgraph,
- solver resolution and recording stride,
- probability output selection,
- optional cashflow declaration,
- optional cashflow aggregation views,
- arbitrary solve-time covariates.

This is coherent from the implementation's point of view, but it makes the main
entrypoint heavily doc-driven. A user will usually need the spec open to know
which combinations are available and what structure comes back.

For a kernel API this is acceptable up to a point, but it is the clearest place
where flexibility is starting to dominate discoverability.

### 2. `InitialDistribution` is conceptually correct but semantically dense

`InitialDistribution` solves a real problem and does it with more precision than
the common shorthand forms can express. That is good.

The difficulty is that several distinct ideas are encoded in one abstraction:

- the declared structural initial-state set,
- runtime mass allocation across those states,
- per-individual duration offsets,
- whether solver reduction should happen on a restricted state set or on the
  full model state list.

Those are valid concerns, but the semantics take time to learn. In particular,
the distinction between "declared initial states" and "states with positive
runtime mass" is subtle but important: reduction follows the declared
structural initial-state set, not the runtime mass support. The same is true of
the
`per_individual(..., initial_states=None)` mode, where indices refer to the full
model state list and no structural reduction is implied.

This is not a conceptual mistake. It is an ergonomics cost that should be
recognized as such.

### 3. The model-building grammar has one awkward corner

`StateSpace.build(transitions=..., exits=..., groups=...)` is mostly a strong
surface. The distinction between single-transition assignment, all-exits
assignment, and grouped assignment is meaningful and matches common modeling
shapes.

The awkward part is `groups={callable: [(src, tgt), ...]}`. Using callables as
mapping keys works, but it is less natural than the rest of the package. It
reads more like an implementation convenience than a user-shaped modeling
grammar.

This is not severe enough to justify a redesign by itself, but it is the least
elegant corner of the model-construction surface.

### 4. Probability output selection is flexible but weakly typed

The built-in probability reducers are useful, but the public surface is driven
by strings such as:

- `"state_probability"`,
- `"density_probability"`,
- `"density"`,
- `"point_mass"`,
- `"marginal_components"`,
- `"full"`.

This works, and it keeps the solver callback story open-ended, but it weakens
the typed feel of the rest of the object model. The user must know not only the
string names, but also the shape contract associated with each one.

Again, this is a reasonable kernel tradeoff. It is still a real usability cost.

### 5. The public API expects users to understand the system model

This package is not hard because the naming is bad. Most of the naming is good.
It is hard because the public API exposes real solver concepts:

- duration grids,
- structural versus runtime initial conditions,
- reduced reachable state sets,
- attachment semantics for cashflows,
- streamed versus terminal views.

That may be unavoidable for a kernel. But it means examples and documentation
are not a nice-to-have around the API. They are part of the API.

---

## Recommendations that preserve the kernel design

The right response is not to flatten the package into a smaller but weaker API.
The right response is to keep the current decomposition and reduce the amount of
implicit knowledge a user must carry.

### 1. Keep the current core abstractions

Do not remove or collapse:

- `StateSpace`,
- `Model`,
- `InitialDistribution`,
- `StateRate`,
- `TransitionLump`,
- `ScheduledEvent`.

These are doing correct conceptual work. They are part of what makes `jact`
look like an actuarial computation library instead of a generic tensor program.

### 2. Add a clearer "happy path" around `solve()`

The expert surface can stay as it is. But the package would benefit from making
the common path easier to see:

- start from one initial state,
- solve for a cohort with covariates,
- request ordinary state probabilities,
- optionally attach a cashflow declaration with one or two obvious views.

This does not require a second solver architecture. It mainly requires examples,
documentation emphasis, and possibly a small amount of helper surface so the
default workflow is visible without reading the full callback and reducer story.

### 3. Make the `InitialDistribution` semantics easier to teach

The object itself is justified. The main improvement needed is clearer framing:

- structural initial states versus runtime mass support,
- restricted initial-state index spaces versus full model index spaces,
- why zero-mass declared states still matter for reduction,
- when the shorthand forms are sufficient and when they are not.

This is mostly a documentation problem, not an abstraction problem.

### 4. Revisit the grouped-intensity surface if future changes are made

If the model-building API is touched again, `groups` is the first area worth
re-examining. The current behavior is useful, but the public grammar could
likely be made more natural without giving up the grouped multi-output concept.

This is a refinement target, not an urgent flaw.

### 5. Treat examples as part of the public surface

For a package like this, examples are not secondary. The README, notebook
walkthroughs, and API-spec examples do real interface work. They are what turn
a flexible kernel into an understandable tool.

The examples should continue to show:

- simple single-state-start solves,
- mixed assignment modes in `build(...)`,
- explicit `InitialDistribution` use when duration matters,
- cashflow declarations using canonical actuarial concepts,
- solve-time views as separate from declaration.

---

## Bottom line

The current API is good. It has a clear conceptual core, it respects the
problem domain, and it exposes the right actuarial objects as first-class public
types.

Its weaknesses are mostly the weaknesses of a serious kernel API:

- the main solve surface is dense,
- some important semantics are learned rather than obvious,
- one or two corners of the grammar are less natural than the rest,
- documentation has to carry a lot of meaning.

That is not a reason to simplify away the domain model. It is a reason to keep
the current architecture and improve the user-facing affordances around it.

In short: the package API is structurally sound, domain-appropriate, and worth
keeping. The next improvements should focus on ergonomics and teaching, not on
replacing the core abstractions.
