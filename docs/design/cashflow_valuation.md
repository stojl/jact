# Cashflow valuation: folding time weights into the view layer

Notes on the grammar for declaring cashflow valuation. This is a
follow-up to [cashflow.md](cashflow.md), which fixes the conceptual
decomposition; [cashflow_api.md](cashflow_api.md), which settles the
typed-component grammar; and [cashflow_aggregation.md](cashflow_aggregation.md),
which settles the typed-view grammar for declaring what to
materialise. None of those docs commits to a grammar for *valuation*
— `cashflow.md` §7 argues for a separate layer but does not pin down
a shape, and `cashflow_aggregation.md` §7 Q4 explicitly leaves open
whether valuation should slot into the view grammar or stay separate.

This document answers `cashflow_aggregation.md` §7 Q4 in the
affirmative for v1. The argument below also **updates the §7
recommendation in `cashflow.md`**: the layering between raw cashflow
and its valuation is conceptually real, but the v1 grammar collapses
it into the view layer because the only transforms v1 has to handle
are time-only multiplicative weights and their terminal sums. The
genuine "separate layer" only earns its keep at v2, when transforms
escape the linear-time-only envelope.

The current `docs/api_spec.md` does not mention valuation at all;
discounting and other post-weighting are implicitly host-side. This
doc is about the surface syntax for declarable valuation, in the
same scratchpad style as `cashflow_aggregation.md`.

---

## 1. What "valuation" means here

A valuation is a transform from a raw expected-cashflow stream to a
re-weighted or re-aggregated stream. The motivating cases are:

- **discounting**, the dominant case: weight each contribution by
  `exp(-int_0^t r(s) ds)` and optionally sum to a terminal scalar,
- **inflation or indexation adjustments**, weighting by an external
  curve,
- **scenario-specific time weights**, e.g. weight horizons by a
  scenario probability,
- **unit-of-account changes**, e.g. multiplying by a deterministic
  exchange-rate path,
- **terminal accumulation**, with or without a non-trivial weight.

The invariant: the input is a cashflow stream (a single component or
a declared aggregated view), and the output is either a stream of
the same outer batch shape or a terminal scalar of shape `(batch,)`.

It is useful to distinguish valuation from three nearby concepts
that already have grammar:

- **aggregation** (`cashflow_aggregation.md`) collapses tag axes —
  component, kind, attachment. Valuation does not collapse tags; it
  re-weights the contributions and optionally collapses the time
  axis.
- **recording semantics** (`cashflow.md` §4) governs the layout of
  the time axis. Valuation acts on whatever time axis recording
  produces.
- **in-transform weighting** is dependence on state, duration, or
  transition identity *before* aggregation. Per `cashflow.md` §7,
  that belongs inside the payment callable, not here. Valuation is
  the place for weights that depend only on time (and optionally on
  solve-time covariates or the cashflow value itself).

---

## 2. Dimensions a valuation transform may depend on

Every grammar decision below turns on what the valuation weight is
allowed to depend on. The taxonomy:

- **time-only** — `v(t) = exp(-r t)`, indexation against a curve.
  The weight is a function of clock time alone. This is the dominant
  case for actuarial discounting.
- **time + solve-time covariates** — `v(t, **kwargs)` where `kwargs`
  carries e.g. a stochastic short rate or scenario id, broadcast to
  `(batch, ...)`. Same shape as the intensity protocol, so the
  callable contract carries over.
- **time + cashflow value** — capping, flooring, non-linear utility
  applied to the per-step cashflow. Pulls the cashflow value into
  the weight; still local in time but no longer linear in the
  cashflow contribution.
- **path-dependent** — running maxima, threshold accumulators,
  look-back guarantees. Requires the solver to keep extra state in
  the carry. Out of scope for v1.

Time-only and time-plus-covariate cover virtually every practical
discounting and indexation use. They are also the cases where
solver-side fusion buys the most: the weight evaluates once per
solver step and multiplies into a cheap accumulator. Critically,
they are also exactly the cases that stay **linear in the cashflow
contribution**, so the same per-step weight applies uniformly across
every component contributing to the same view.

The third and fourth bullets are not weighting in the same sense.
Capping/flooring is a non-linear function of the per-step cashflow;
path-dependence requires extra carry state. Both are deferred to v2.
What is left for v1 is a single primitive: a multiplicative,
time-local weight on the per-step cashflow contribution.

---

## 3. Why this transform should be declarable, not just post-processed

`cashflow.md` §7 frames valuation as a post-processing layer applied
after raw cashflow generation. That framing is correct as a
*semantic* default — the algebra works either way. But there are
two reasons to make this transform declarable rather than purely
host-side, both following the same shape as the analogous argument
in `cashflow_aggregation.md` §3:

- **Memory.** A user who only wants the present value of a total
  cashflow over a long horizon × large batch should not have to
  materialise a `(T_out, batch)` view stream just to discount-and-sum
  on the host. The discounted terminal scalar is one number per
  individual; the solver should be free to keep only that. With
  many components and many views, the saving compounds.
- **Composition with terminal-only mode.** `cashflow.md` §6.2
  already contemplates a "running totals in the carry, return only
  terminal values" mode. A time-local weight fuses naturally into
  that accumulator — the per-step weight slots into the same scan.
  Without a declarable hook, terminal mode either bakes one fixed
  convention (undiscounted? discounted at what rate?) or duplicates
  the full stream just to weight it host-side.

Equivalently, the declarable thing is **what to materialise *with
what weight***. That is a refinement of the view's role
(`cashflow_aggregation.md` §3 — "aggregation declarations are a
partial specification of the output PyTree shape"), not a new axis
parallel to it. The solver allocates one accumulator per declared
view, applying that view's weight per step and either keeping the
running stream or collapsing to a terminal scalar.

Post-processing remains the right answer for one-off, exploratory,
or unusual valuations. The grammar does not displace it; it gives
the common cases a fast path.

---

## 4. Survey of declaration shapes

Following the structure of `cashflow_aggregation.md` §4 and ordered
from least to most invasive.

### 4.1 Pure post-processing (status quo)

```python
result = model.solve(cashflows=cashflows, cashflow_views={"total": Total()}, ...)
pv = (result["cashflows"]["total"] * discount_factors).sum(axis=0)
```

Pros:

- Zero new API surface.
- Maximally composable; any host-side numerics applies.

Cons:

- Forfeits the memory and terminal-mode wins in §3.
- Every downstream tool reimplements "discounted total".
- Forces materialisation of streams the user only wants in scalar
  form.

### 4.2 Bake the weight into each component callable

```python
def premium_fn(t, d, **kwargs):
    return raw_premium(t, d, **kwargs) * discount_factor(t, **kwargs)
```

Mathematically equivalent to any external valuation: the same
expected cashflow comes out. This is the "no new API" alternative
the user might reach for first.

Pros:

- Zero grammar change.
- Maximally explicit at the call site.

Cons:

- Computationally wasteful when many components share the same
  `w(t)`. The shared envelope is evaluated once per component per
  step; with N components that is N evaluations of the same
  function at the same `t`, plus N copies of the data dependency.
- Couples each component callable to a particular valuation
  convention. A user who wants both a raw and a discounted view of
  the same model has to declare two parallel sets of components.
- Loses the terminal-mode win — the cashflow stream is still
  generated component-wise.

The combination of "shared across components" and "fuses into a
single accumulator step" is exactly what motivates pulling the
weight out of the components and into the view layer.

### 4.3 Weight + accumulate flag on the view

```python
cashflow_views = {
    "total":    Total(),
    "pv_total": Total(weight=discount_factor(rate=r), accumulate=True),
    "real":     Group(["death_benefit", "retirement_bonus"], weight=index_curve),
}
```

Each view in `cashflow_aggregation.md` §5 (`Raw`, `Group`, `Total`,
`ByState`, `ByKind`) gains two optional fields:

- `weight: Callable[[float, ...], jnp.ndarray] | float | None` — the
  per-step multiplicative factor.
- `accumulate: bool = False` — when `True`, collapse the time axis and
  return a single `(batch,)` scalar accumulated across the full
  horizon instead of the recorded stream.

`discount_factor(rate=...)` is a small numerics helper (§7) that
returns the standard `(t, **kwargs) → (batch,)` callable; the
existing midpoint quadrature plumbing lives behind it.

Pros:

- One axis (the view) carries both "what to sum" and "with what
  weight". The two are fused inside one accumulator step on the
  solver side, which is also how the user thinks about "discounted
  total".
- Pyright sees `weight`/`accumulate` per view; no string indirection
  between two parallel dicts.
- Collapses the five-class taxonomy of the original draft into two
  optional fields. `Identity` disappears (it is "no weight"),
  `Discount`/`Weighted` become `Total(weight=...)`,
  `PresentValue`/`WeightedSum` become `Total(weight=..., accumulate=True)`.
- No new top-level result key — every output is a view, lands under
  `result["cashflows"][view_name]`.

Cons:

- Views grow two fields beyond the bare `cashflow_aggregation.md`
  §5 set.
- A user who wants the same view in both streamed and terminal form
  has to declare two view names.

### 4.4 Separate `valuations` dict (the original recommendation)

```python
result = model.solve(
    cashflow_views={
        "total":    Total(),
        "benefits": Group(["death_benefit", "retirement_bonus"]),
    },
    valuations={
        "pv_total":    PresentValue("total",    rate=r),
        "pv_benefits": PresentValue("benefits", rate=r),
    },
)
```

Each valuation references a declared view by name and produces an
entry in `result["valuations"]`.

Pros:

- View grammar stays untouched.
- Self-describing per-leaf — `PresentValue("total", rate=r)` says
  what it is at a glance.

Cons:

- Two parallel dicts on `solve()`, with `(view, valuation)` identity
  split across them.
- View↔valuation reference is a string name; renaming a view
  silently breaks every valuation that referenced it. A whole
  validation pass exists to catch that.
- Materialisation gets weird: a user who wants only `PresentValue`
  has declared both a view (which they may not want streamed) and
  a valuation (which they do want). The "skip the view if only
  terminal valuations consume it" optimisation is a tacit feature
  of every implementation, not a property of the grammar.
- Five class names (`Identity`, `Discount`, `PresentValue`,
  `Weighted`, `WeightedSum`) for what is really one primitive plus
  a stream/terminal toggle — the doc's own §5.5 admits this.
- Most of the open-questions list in the original draft was about
  the view↔valuation boundary itself, which is a sign the boundary
  is the wrong cut.

### 4.5 Wrapper view types

```python
cashflow_views = {
    "pv_total":    Discounted(Total(), rate=r),
    "pv_benefits": PresentValue(Group(["death_benefit", "retirement_bonus"]), rate=r),
}
```

Pros:

- Composable; any view can be wrapped.
- Mirrors typed-view grammar; pyright sees the wrapper.

Cons:

- Introduces a nesting-of-views vocabulary that
  `cashflow_aggregation.md` §7 Q3 deliberately avoided (views stay
  flat over component names only).
- A deeper stack — `Discounted(Indexed(Total(), curve), rate)` —
  starts to look like a tiny expression DSL, exactly the shape
  `cashflow_api.md` §4.4 pushed back on for the linear-combination
  proposal.
- Doubles the namespace burden: the user has to invent both a
  meaningful view name *and* a meaningful wrapper composition.

### 4.6 First-class valuation functor protocol

A `Valuation` is a small object exposing a per-step weight
contribution and an `accumulate(carry, weighted_value) → carry`.
Built-ins implement it; users can write their own.

Pros:

- Most general; users get custom valuations without forking.
- Genuine home for the cases v1 cannot express: capping/flooring
  on cashflow value, path-dependent transforms, user-defined
  accumulator carry.

Cons:

- Largest API surface; exposes solver-internal accumulator
  semantics in the public protocol — once published, hard to
  evolve.
- Premature for v1, where everything in scope is a multiplicative
  time-local weight that does not need a custom accumulator.

This is the natural v2 home for transforms that escape §2's
linear-time-only envelope. v1's on-view weight is a strict subset
of what such a protocol would express.

---

## 5. Preferred direction: weighted aggregation on the view

Adopt §4.3.

The argument tracks the typed-view recommendation in
`cashflow_aggregation.md` §5 and the typed-component recommendation
in `cashflow_api.md` §5:

- the view layer already settled "what to materialise"; weighting
  is a refinement of that question, not a parallel axis,
- pyright-checkable per-view `weight` and `accumulate`,
- one flat dict from view name to a typed view object, with the
  Python type carrying the discrimination,
- composes cleanly with future extensions,
- post-processing (§4.1) remains available for one-off use.

The v1 view set from `cashflow_aggregation.md` §5 is unchanged in
*identity* — `Raw`, `Group`, `Total`, `ByState`, `ByKind`. Each
gains two optional fields:

| Field    | Type                                              | Default     | Meaning |
|----------|---------------------------------------------------|-------------|---------|
| `weight` | `Callable[[float, ...], jnp.ndarray] \| float \| None` | `None`      | Per-step multiplicative factor. `None` is unweighted. A scalar is sugar for `lambda t, **kw: scalar`. |
| `accumulate` | `bool`                                          | `False`     | `False` records per `record_every`; `True` collapses the time axis to a single `(batch,)` per leaf. |

Worked examples (cumulative across one solve, no parallel dicts):

```python
cashflow_views = {
    "raw":          Raw(),
    "total":        Total(),
    "pv_total":     Total(weight=discount_factor(rate=r), accumulate=True),
    "pv_total_stream": Total(weight=discount_factor(rate=r)),
    "real":         Group(["death_benefit", "retirement_bonus"], weight=index_curve),
    "pv_by_state":  ByState(weight=discount_factor(rate=r), accumulate=True),
}
```

`weight` is a callable `(t, **kwargs) → (batch,)` returning the
per-step factor directly (no implicit exponentiation, no implicit
cumulative product). `discount_factor(rate=...)` is the canonical
factory for the `exp(-int_0^t r(s) ds)` weight; it lives in §7.

### 5.1 Default behaviour

If `cashflow_views` is omitted, the solver returns raw component
streams as before. If `weight` and `accumulate` are both at their
defaults, the view behaves exactly as in `cashflow_aggregation.md`
§5 — the change is purely additive.

If `accumulate=True` is set, the solver allocates a `(batch,)`
accumulator for that view and never materialises its time-resolved
stream; the corresponding `result["cashflows"][view_name]` is
`(batch,)` rather than `(T_out, batch)`. A user who wants both
streamed and terminal forms of the same view declares two view
names with different `accumulate` settings.

### 5.2 Validation

Validation stays a `StateSpace`-/declaration-only concern, mirroring
`cashflow_aggregation.md` §5.2:

- view names are unique across the dict,
- `weight` is `None`, a Python scalar, or callable,
- `accumulate` is a `bool`,
- the view's existing per-type validation (`Group` member names,
  etc.) is unchanged.

### 5.3 Why on-view rather than a parallel `valuations` dict

Contrast with §4.4:

- The view↔valuation string reference disappears. A view is the
  one and only thing carrying the weight; nothing else refers to
  it by name.
- The terminal-vs-streamed split lives once, on the view, where it
  belongs as a materialisation property — not duplicated across
  classes (`Discount`/`PresentValue`, `Weighted`/`WeightedSum`).
- The five-class taxonomy collapses to two optional fields.
- The "skip materialising the view if only terminal valuations
  consume it" optimisation is no longer a hidden feature — it is
  literal `accumulate=True`.
- The `result["valuations"]` vs `result["cashflows"]` split
  disappears. Everything is a view; everything lands under
  `result["cashflows"]`.

The cost is a richer view dataclass. Two optional fields against
five classes plus a parallel dict is a clear win.

### 5.4 Why not wrap views

Wrapping (§4.5) reintroduces the nesting vocabulary
`cashflow_aggregation.md` §7 Q3 deliberately avoided. The
compositional appeal is real, but the cases where it matters
(chained valuations, `Discounted(Indexed(Total(), curve), rate)`)
are also the cases where weights compose by simple product —
host-side or via a small `multiply(w1, w2)` helper that returns
a `(t, **kwargs) → (batch,)` callable. No grammar change required.

### 5.5 What v1 still does not cover

These are deferred to a future v2 functor protocol (§4.6):

- capping, flooring, or other non-linear functions of the per-step
  cashflow,
- path-dependent transforms (running maxima, threshold
  accumulators, look-back guarantees),
- user-defined accumulator carry beyond `(batch,)` running sums.

For v1, anything outside the time-local linear-weight envelope
falls back to host-side post-processing of a streamed view (§4.1).

---

## 6. Output shape

Everything is a view, so the only output slot is
`result["cashflows"][view_name]`. There is no `result["valuations"]`.

- Streamed views (`accumulate=False`) keep the outer shape of their
  underlying view: typically `(T_out, batch)` for `Total`/`Group`,
  `{component_name: (T_out, batch)}` for `Raw`,
  `{state_name: (T_out, batch)}` for `ByState`, etc.
- Terminal views (`accumulate=True`) collapse the leading time
  axis: `(batch,)` for `Total`/`Group`, `{component_name: (batch,)}`
  for `Raw`, `{state_name: (batch,)}` for `ByState`, etc.

Both forms can coexist in one `cashflow_views` dict under different
view names.

---

## 7. Discounting specifics

Discounting is the motivating case and deserves a dedicated note.

`discount_factor(rate=...)` is the canonical factory for the weight
that appears in `Total(weight=discount_factor(rate=r))` and similar
forms. It is a small numerics helper, not a view type:

- `rate` is a callable `(t, **kwargs) → (batch,)` or scalar. A
  constant shortcut `rate=0.03` is accepted as syntactic sugar for
  `lambda t, **kw: 0.03`.
- The returned callable produces the running discount factor
  `D(t) ≈ exp(-int_0^t r(s) ds)` evaluated against the solver
  step grid. Per-step, the within-interval weight applied to the
  contribution attributed to interval `[t_n, t_{n+1}]` is
  `exp(-r(t_n + dt/2) · dt) · D(t_n)`, using the same midpoint
  approximation as the rest of the solver and consistent with the
  underlying probability scheme.
- Because the recording-semantics default is interval accumulation
  (`cashflow.md` §4), the discount weight applied is the
  within-interval weight for the interval the cashflow is
  attributed to, not a point-time weight at the recording boundary.

Folding `discount_factor` into a `Total(..., accumulate=True)`
view fuses the per-step discount weight, the per-step cashflow
contribution, and the terminal accumulator into one scan step. No
intermediate stream materialises; the solver keeps a single
`(batch,)` accumulator per terminal-mode view.

`discount_factor` is the single home for the discount-factor
quadrature. Future refinements (e.g. exact integration when `r` is
piecewise constant on the grid) live there and apply uniformly to
every view that uses the helper.

---

## 8. Open questions

1. Should `weight` accept a precomputed array of shape `(T_out,)`
   or `(T_out, batch)` as an alternative to a callable? The array
   form sidesteps quadrature entirely but loses solve-time
   covariate dependence and forces alignment with `record_every`.
   Leaning: yes, accept arrays as a convenience; document the
   alignment requirement.
2. Should `weight` and `accumulate` apply uniformly to *every* view
   type, including `Raw()`? `Raw(weight=...)` would mean "raw
   per-component streams, each multiplied by the same envelope",
   which is useful for scenario weighting but mildly blurs the
   role of `Raw`. Leaning: yes, uniform; `Raw(weight=...)` is just
   N parallel weighted streams.
3. Should `ByState` allow per-state weights, e.g. `weight={state:
   callable}`? That is per-attachment splitting territory and
   overlaps with the `PerAttachment` v2 sketch in
   `cashflow_aggregation.md` §6. Leaning: no for v1; defer with
   `PerAttachment`.
4. Should declaring the same view with `accumulate=False` and
   `accumulate=True` under two view names share a single
   per-step weight evaluation under the hood, or is allocating two
   accumulators (with one extra weight evaluation per step) fine?
   Leaning: don't optimise yet; accumulators are cheap.
5. Path-dependent valuations (running maxima, threshold
   accumulators, look-back guarantees) and non-linear-in-cashflow
   transforms (capping, flooring, utility): explicitly out of
   scope for v1. They are the v2 functor-protocol territory
   (§4.6) — extra carry state in the solver, custom accumulator
   semantics. Restated here so the v1/v2 boundary stays visible.
6. Should `cashflow.md` §7 be updated to match this doc's framing,
   or is the §1 pointer at the top of this doc enough? Leaning:
   patch §7 in a follow-up; this doc is the source of truth on
   valuation grammar.

---

## 9. Recommendation

Adopt the on-view weighted-aggregation grammar from §5:

- extend each view in `cashflow_aggregation.md` §5 (`Raw`, `Group`,
  `Total`, `ByState`, `ByKind`) with two optional fields, `weight=`
  and `accumulate=`,
- provide `discount_factor(rate=...)` as the canonical numerics
  helper for the discount-factor weight,
- no `valuations` dict, no `Identity`/`Discount`/`PresentValue`/
  `Weighted`/`WeightedSum` classes,
- streamed and terminal outputs both land under
  `result["cashflows"][view_name]`,
- post-processing (§4.1) remains supported for one-offs the user
  does not want to declare upfront,
- non-linear, path-dependent, and custom-accumulator transforms
  are deferred to a v2 functor protocol (§4.6),
- everything in `cashflow_api.md` and `cashflow_aggregation.md`
  carries over unchanged; `cashflow.md` §7's "valuation as a
  separate layer" framing is updated by this doc and a follow-up
  edit there is desirable but not blocking.

The wrapper-view alternative (§4.5) and the parallel-dict form
(§4.4) remain expressible on top of this grammar via post-
processing or a thin v2 layer if practice argues for them later.
The bake-into-component path (§4.2) and post-processing-only path
(§4.1) should remain documented as explicit fallbacks for cases
the v1 set does not cover, but should not be the only path for the
common discounting case before any user-facing release.
