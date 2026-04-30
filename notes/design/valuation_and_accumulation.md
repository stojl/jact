# Valuation and accumulation: deferred questions

Notes on the valuation layer and the accumulation modes for cashflow
output. Follow-up to [cashflow.md](cashflow.md), which fixed the
conceptual decomposition and committed to component-wise named
streams, and to [cashflow_aggregation.md](cashflow_aggregation.md),
which settled a typed-view grammar for declaring what to materialise.

The threads picked up here are explicitly deferred in those docs:

- `cashflow.md` §4 — interval accumulation flagged as the leading
  default for cashflow recording, but not committed against the
  alternatives.
- `cashflow.md` §6 — terminal totals; both "post-process recorded
  streams" (§6.1) and "direct accumulated-output mode" (§6.2) are
  named without choosing.
- `cashflow.md` §7 — valuation as a separate post-processing layer,
  with in-callable weighting kept as an option for path-dependent
  cases.
- `cashflow_aggregation.md` §7 Q4 — whether the typed-view grammar
  should extend to valuation (e.g. `Discounted(Total(), rate=...)`),
  with a current bias against.

These are one design problem, not three. The terminal-only memory
win in `cashflow.md` §6.2 only pays off when valuation can fold into
the carry; the view-grammar question in `cashflow_aggregation.md`
§7 Q4 cannot be closed without a position on where valuation lives.
This doc takes that joint problem head-on.

---

## 1. The accumulation lattice

Four modes recur in the literature and in the existing design notes:

1. **Sample** — record `c(t_n)` at the endpoint of each record
   period.
2. **Interval** — record `∫_{t_{n-1}}^{t_n} c(s) ds` (or its
   discrete analogue: the sum of inner-step contributions spanning
   the period).
3. **Cumulative** — record `∫_0^{t_n} c(s) ds` for each `n`.
4. **Terminal** — return only `∫_0^{T} c(s) ds`, no time series.

These are not all on the same footing.

### 1.1 Sample is a category error for two of three component kinds

For `StateRate` and `TransitionLump`, the underlying object is a
*rate* or a *triggered lump density*. The instantaneous value at
`t_n` is not a payment; it is a rate whose physical units differ
from the integrated quantity. Recording it produces a number whose
meaning is unstable under refinement of `record_every`.

For `ScheduledEvent`, sample makes sense because the event itself is
an instantaneous payment at a known clock time. But it is a special
case, not a general default.

The probability solver gets away with sample semantics because
probability is a state quantity, not a flow. Cashflow is different.

### 1.2 Interval is the natural base

Interval accumulation matches the way state-rate and
transition-lump cashflows are *generated*: continuously across
inner steps. The sum of inner-step contributions over a record
period is a payment over the period, with units consistent with
what the user expects.

A scheduled event lands inside exactly one record period (assuming
the v1 grid-aligned restriction from `cashflow.md` §3 and
`api_spec.md` §"Scheduled-event policy in v1") and contributes its
amount to that period's interval value.

This is the conclusion `cashflow.md` §4 already leans toward.
Nothing in the rest of this doc weakens it.

### 1.3 Cumulative is recoverable from interval

A cumulative stream is `cumsum(interval_stream)` along the time
axis. There is no reason to give it a separate mode in the solver.
A user who wants cumulative output runs `jnp.cumsum` on the recorded
interval stream and is done.

### 1.4 Terminal is the only mode that needs solver-side support

`Terminal = sum(interval_stream)` *is* recoverable from the stream,
but only if the stream was materialised. The whole point of a
terminal mode is the case where materialising the stream is
wasteful — long horizon, large batch, many components, no
time-resolved need. The solver should be able to maintain a single
scalar per component-or-view per individual instead of stacking a
`(T_out, batch, ...)` array along the scan axis.

So among the four modes, two collapse into post-processing
(cumulative, terminal-from-stream) and one is a category error for
two-thirds of components (sample). What is left is:

- **interval** as the recorded base,
- **terminal-only** as an opt-in scalar mode for memory.

---

## 2. Direct terminal-only mode

The terminal-only mode in `cashflow.md` §6.2 is the only one with a
real implementation question.

### 2.1 When it pays off

- horizon is long (large `T_out`),
- batch is large,
- many named components or views are requested,
- the user has no need for time-resolved cashflow output.

In all four conditions stacked, the streamed output dominates
memory. Terminal-only replaces a `(T_out, batch, ...)` leaf with a
`(batch, ...)` scalar accumulator per component-or-view. For
`T_out` in the hundreds and many components, that is one to two
orders of magnitude.

### 2.2 What changes in the solver

Streamed mode keeps the inner contribution at each step in the
output stream emitted by `lax.scan`. Terminal mode keeps a running
sum in the *carry* and emits nothing per step:

```text
streamed:   carry = solver_state          ;  emit = interval_y_n
terminal:   carry = (solver_state, acc_n) ;  emit = nothing
            acc_n = acc_{n-1} + interval_y_n
```

The interval-`y_n` computation is identical between the two; only
the storage strategy differs. That keeps terminal mode an isolated
extension, not a fork of the cashflow transform itself.

### 2.3 Public surface

The minimal disturbance is a `solve()`-level flag — `cashflow_terminal=True`
or an analogous spelling. Output shape changes from a stream PyTree
to a scalar PyTree mirroring the requested views; everything else
stays. Whether the flag is solve-wide or per-view is left as an
open question (§5).

---

## 3. Valuation — placement options

Valuation means weighting expected cashflow contributions by some
factor before reporting. The canonical case is discounting; more
general functionals (inflation indexation, scenario reweighting,
deflators) fit the same template. `cashflow.md` §7 already pushes
back on calling this a "discounting layer" and prefers the broader
framing.

There are three places valuation could live.

### 3.1 (a) Inside the payment callable

The user multiplies the discount factor into the returned amount:

```python
def death_fn_pv(t, d, **kwargs):
    return discount(t) * death_fn(t, d, **kwargs)
```

Pros:

- Most general. Path-dependent weights (state, duration, transition
  identity) are trivially expressible.
- Zero new public surface.

Cons:

- Time-only factors are duplicated across every callable.
- The output is no longer a raw expected cashflow stream; it is a
  pre-valued stream. Re-using the same `cashflow` declaration with a
  different valuation forces a new declaration with rebound
  callables.
- `cashflow.md` §7 already argues the duplication kills clean
  composition with aggregation.

### 3.2 (b) `solve()`-level argument applied before recording

The user passes a valuation callable, and the solver multiplies it
into each per-step contribution before accumulating or recording:

```python
solve(..., valuation=discount_fn)
```

Pros:

- One declaration of the discount factor; every component sees it.
- Composes with terminal-only mode (§2): the per-step contribution
  is already weighted, so the carry accumulates a present value
  directly.
- Reusing the same `cashflow` declaration with a different
  valuation is a one-keyword change at solve-time.

Cons:

- Adds a new `solve()` argument and a small protocol commitment for
  the valuation callable.
- Time-only is the natural fit; path-dependent valuation does not
  belong here (§4).

### 3.3 (c) Pure post-processing

The solver returns raw expected cashflow; the user multiplies by
`discount(t)` host-side and sums.

Pros:

- Simplest possible surface. No new solve-time argument.
- Fully general for time-only valuation (see §4).
- Lets the same recorded stream be re-valued under multiple
  scenarios without re-solving.

Cons:

- Loses the terminal-only memory win: the user has to materialise
  the stream first, then sum it down. For users who only ever want
  a present value, that is the wrong default.

---

## 4. Time-only valuation commutes with linear aggregation

The structural property that simplifies most of this design is that
time-only valuation commutes with sums over components (and over
attachment points, and over batch).

For any time-only weight `w(t)`:

```text
PV(Group([A, B]))(t) = w(t) · (A(t) + B(t))
                     = w(t) · A(t) + w(t) · B(t)
                     = PV(A)(t) + PV(B)(t)
```

Same identity for `Total()`, `ByState()`, `ByKind()`, and any
linear view from `cashflow_aggregation.md` §5. Aggregation is a
linear operator on streams; time-only valuation is a pointwise
linear operator on streams; they commute.

The implication is strong. For *any* view granularity returned by
the solver, the user can apply post-processing time-only valuation
afterwards and recover the same PV they would have got from a
solver-side valuation. Solver-side valuation is therefore not about
expressivity; it is only about avoiding the intermediate stream.

That is what reduces the design space: solver-side valuation is
useful exactly in combination with terminal-only mode.

---

## 5. Path-dependent valuation does not commute

Weights that depend on state, duration, or transition identity do
*not* commute with aggregation. A weight `w(state, t)` applied to
`Total()` is ill-defined — the `Total()` stream has already
collapsed the state dimension.

Two clean choices for that case:

- bake the weight into the payment callable (the §3.1 path);
  `cashflow.md` §7 already recommends this for genuinely
  path-dependent weights;
- keep per-attachment streams live, e.g. via the `PerAttachment`
  view sketched in `cashflow_aggregation.md` §6, and apply
  per-attachment weights afterward.

Neither needs new top-level surface. Path-dependent valuation is
left out of the `solve()`-level valuation argument deliberately.
That argument is for time-only weights; everything else is the
user's job, with `PerAttachment` available as an escape hatch when
v2 lands it.

---

## 6. Should valuation enter the view grammar?

`cashflow_aggregation.md` §7 Q4 flags `Discounted(Total(), rate=...)`
and similar nested-view spellings as a possibility, with a current
bias against. This doc commits to the bias.

Views answer the question "*what should the solver materialise?*".
Valuation answers the question "*how should it weight what is
materialised?*". The two are orthogonal:

- across views, the same valuation applies pointwise;
- across valuations, the same view shape applies.

Folding valuation into the view grammar — a `Discounted(view, ...)`
wrapper or a `Group([...], discount=...)` field — produces a
combinatorial class explosion (`Discounted(Total())`,
`Discounted(Group([...]))`, `Discounted(ByState())`, ...). Worse,
it hides that the cross-product is structurally rectangular: every
view × every valuation is a valid pair, so a wrapper-per-view
duplicates the same field everywhere.

Keeping valuation a separate top-level argument keeps the axes
visibly orthogonal. The user picks a view; the user picks a
valuation; the solver applies them composably.

This also matches the framing on the probability side: the
`probability` argument controls *what to record*, and is independent
of any post-processing the user might apply. Cashflow gets the
same split: views control what to materialise, valuation controls
the weight, and the two compose.

---

## 7. Where solver-side valuation actually pays off

Cross-cutting §1.4 (terminal-only) and §4 (time-only valuation
commutes), there is exactly one combination where solver-side
valuation buys something post-processing cannot:

```text
terminal_PV[v] = Σ_n  w(t_n) · interval_contribution_n[v]
```

accumulated into one scalar per view `v` per individual. The carry
is a single number per `v`; no intermediate stream is allocated.

For streamed output, post-processing produces an identical answer:

```text
streamed_PV[v][n] = w(t_n) · interval_stream[v][n]
```

is one host-side multiply on the recorded stream. The solver-side
multiply at trace time is not faster, not more accurate, and not
more general. It is only useful when the alternative is paying for
the stream's storage, i.e. terminal-only mode.

That single observation is what makes it safe to keep the
valuation argument narrow and time-only: its presence inside the
solver carry is justified by terminal-only, and terminal-only does
not need the path-dependent power that would force valuation into
the callables.

---

## 8. Interaction summary

The pieces fit together cleanly:

| Feature | Where it lives | Recovers from |
|---|---|---|
| Sample recording | not exposed (category error for rate kinds) | n/a |
| Interval recording | solver, default | n/a |
| Cumulative recording | post-process | `cumsum(interval)` |
| Terminal recording (with stream) | post-process | `sum(interval)` |
| Terminal recording (no stream) | solver, opt-in flag | not recoverable |
| Time-only valuation, streamed | post-process or solver-side | identical results |
| Time-only valuation, terminal-only | solver-side, before accumulation | required for memory win |
| Path-dependent valuation | inside payment callable | n/a |
| Per-attachment valuation | `PerAttachment` view (v2) + post-process | n/a |

Everything in the bottom three rows is deferred or punted to the
user. Everything in the top six is settled by the recommendation
below.

---

## 9. Recommendation

Adopt the following directions:

- **Interval accumulation** is the default for cashflow recording.
  This confirms `cashflow.md` §4. State-rate and transition-lump
  contributions are summed across the inner steps spanning each
  record period; scheduled events contribute to the period
  containing their event time.

- **Cumulative recording is not a solver mode.** Users who want it
  apply `cumsum` to the recorded stream.

- **Terminal-only recording is an opt-in `solve()`-level mode.**
  When enabled, the solver returns a scalar per view per individual
  and emits nothing per step. Output shape mirrors the requested
  views with the time axis dropped. Exact spelling of the flag is
  open.

- **Time-only valuation is a `solve()`-level kwarg** applied to each
  per-step contribution before the contribution enters the
  recorded stream or the terminal accumulator. Streamed output
  with valuation is equivalent to host-side post-processing;
  terminal-only output with valuation is the case that requires
  solver-side support.

- **Path-dependent valuation stays inside the payment callable.**
  The `solve()`-level valuation argument does not accept weights
  that depend on state, duration, or transition identity. Users
  with such weights bake them into their payment callables
  (`cashflow.md` §7) or wait for `PerAttachment` (v2).

- **Views and valuation are orthogonal grammar axes.** Valuation
  does *not* nest inside the view grammar. This closes
  `cashflow_aggregation.md` §7 Q4 in the negative.

The combined effect is:

- one base output (component-wise interval streams),
- one optional opt-in (terminal-only, returning scalars),
- one optional weight (time-only valuation, applied before
  recording),
- one orthogonal grammar (views, defined in
  `cashflow_aggregation.md`),
- one escape hatch (path-dependent weights inside callables).

Each piece has a single reason to exist and composes cleanly with
the others.

---

## 10. Open questions

These remain for implementation planning, not blockers for the
design:

1. Exact spelling of the terminal-only flag. Candidates:
   `cashflow_terminal=True`, `cashflow_mode="terminal"`, a per-view
   option on each typed view object. Per-view is more flexible but
   complicates the carry; solve-wide is simpler.
2. Whether multiple named valuations should be declarable in one
   solve, e.g. `valuations={"pv_3pct": fn1, "pv_5pct": fn2}`,
   producing one accumulator per (view, valuation) pair. Useful
   for sensitivities; redundant for single-rate use.
3. The left/right state convention at on-grid scheduled events,
   still open from `cashflow.md` §9 Q6. Independent of valuation
   choice but interacts with the interval the event is assigned
   to.
4. Whether terminal-only is solve-wide (one flag) or per-view (one
   field per view object). Solve-wide is the simpler v1; per-view
   is a clean extension if mixed needs appear.
5. Shape of the valuation callable's contract. Two candidates:
   `(t,) -> scalar` (purely time-only, simplest) or
   `(t, **kwargs) -> (batch,)` (time-only-but-cohort-varying, e.g.
   per-individual interest rates from covariates). The second is a
   superset and matches the existing intensity callable signature
   in spirit.
6. Whether `valuation=None` is the disabled-output convention or
   `valuation=lambda t: 1.0` is the explicit identity. Same
   ergonomics question as `probability=None` vs `cashflows=None`
   in `api_spec.md` §"Planned cashflow solve extension".
