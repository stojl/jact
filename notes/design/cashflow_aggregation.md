# Cashflow aggregation: declaring what to materialise

Notes on the grammar for declaring cashflow aggregation. This is a
follow-up to [cashflow.md](cashflow.md), which fixes the conceptual
decomposition (state-rate, transition-lump, scheduled-event) and
commits to component-wise named cashflows as the base output, and to
[cashflow_api.md](cashflow_api.md), which settles the typed-component
grammar for declaring components. Neither doc commits to a grammar
for *aggregating* the resulting streams.

The current `docs/api_spec.md` sketch (§"Planned cashflow solve
extension") only offers a flat `cashflow_groups: dict[str, list[str]]`
for name-keyed unions of component streams. That covers one axis of
aggregation. Real users plausibly want more: a grand total, per-state
attribution, per-kind grouping, or per-attachment splitting within a
single component. This document is about the surface syntax for those
choices, in the same scratchpad style as `cashflow_api.md`.

---

## 1. What "aggregation" actually means here

Each contribution to expected cashflow carries several natural tags:

- **component name** — user-declared, e.g. `"premium"`,
  `"death_benefit"`.
- **component kind** — `StateRate` / `TransitionLump` /
  `ScheduledEvent`.
- **attachment point** — state name (state-rate, scheduled-event) or
  transition pair (transition-lump).
- **time** — already covered by `cashflow.md` §4 (interval
  accumulation as the leading interpretation).
- **batch** — always preserved per individual.

Aggregation is choosing which of these tag axes to *collapse* and
which to *keep*. Time and batch are out of scope for this doc — they
are governed by recording semantics and per-individual evaluation
respectively. The interesting axes are component, kind, and
attachment point.

---

## 2. Two orthogonal axes

The design space splits cleanly into two axes that the grammar should
keep separate:

### 2.1 Across-component grouping

Multiple named components are unioned into one stream. Examples:

- `"benefits" = death_benefit + retirement_bonus`
- `"total"    = every component`
- `"by_kind"` keeps one stream per component kind, summing the
  components in each kind.
- `"by_state"` collapses the component dimension and keeps one
  stream per attachment state.

This is the axis that the existing `cashflow_groups` sketch in
`api_spec.md` already addresses, plus structural variants.

### 2.2 Within-component splitting

The opposite direction. A `StateRate({"healthy": ..., "disabled":
...})` could expose either:

- one stream summed over its attachment states (the default in
  `cashflow.md`), or
- one stream per attachment state (a finer output, not a separate
  component).

The same applies to a `TransitionLump` covering several transitions
or a `ScheduledEvent` keyed by several occupied states. A per-
attachment view is a *finer* output, not a coarser one.

### 2.3 Why keep these axes separate

Across-component grouping always reduces information. Within-component
splitting always preserves more of it. They are independent, and
collapsing them into one dict (e.g. forcing within-component splits to
masquerade as multiple "components") would break the
"component-name = one declared cashflow stream" framing from
`cashflow_api.md` §2.

For v1, across-component grouping plus structural views (kind,
attachment state) is the priority. Within-component splitting is
genuinely new — its grammar is sketched in §6 below but not committed
to v1.

---

## 3. Why aggregation should be declarable, not just post-processed

`cashflow.md` §5 already argues that the base object is component-wise
expected cashflow, with aggregation a secondary layer. That framing
makes post-processing the obvious path: solve, get one stream per
component, then sum on the host side.

But there is a memory tension that the same document acknowledges in
§6:

- with N components, large `batch`, and a long horizon, materialising
  every per-component stream just to sum them afterwards is wasteful;
- if the user only wants `"total"`, the solver should be free to
  allocate one accumulator, not N.

A declared aggregation is a hint to the solver about *what to
materialise*. It is the cashflow analogue of the "direct accumulated-
output mode" in `cashflow.md` §6.2: a way to say "skip the per-
component stream, give me the sum directly". With a flexible grammar,
the same mechanism covers grand totals, named groups, structural
views, and any combination of them.

Equivalently: aggregation declarations are a partial specification of
the output PyTree shape. The solver allocates exactly the leaves the
user asks for.

---

## 4. Survey of declaration shapes

Following the structure of `cashflow_api.md` §§2–4.

### 4.1 Flat name-keyed groups (status quo)

```python
cashflow_groups = {
    "benefits": ["death_benefit", "retirement_bonus"],
}
```

Pros:

- Simplest possible shape.
- Already in the spec sketch.

Cons:

- Covers only across-component grouping by *named union*.
- No vocabulary for grand totals (other than listing every component
  name), per-kind grouping, or per-attachment-state grouping.
- No way to drop raw components — the user always pays for them.

### 4.2 Hierarchical groups (nested dicts)

```python
cashflow_groups = {
    "benefits": {
        "death": ["death_benefit"],
        "bonus": ["retirement_bonus"],
    },
    "expenses": ["admin_cost"],
}
```

Output is a nested PyTree mirroring the input.

Pros:

- Lets the user shape the output dict directly.

Cons:

- Nesting is purely cosmetic — the user can wrap a flat output dict
  in their own nesting host-side at zero cost.
- Mixes "this is a leaf list of components" and "this is a sub-tree"
  in the same dict shape, which is the kind of structural overload
  `cashflow_api.md` §1.4 already pushed back on for components.
- Still has no vocabulary for structural views or for opting out of
  raw components.

### 4.3 Typed view objects

A dict from view name to a typed view object that says what to
compute, parallel to the typed-component grammar in
`cashflow_api.md` §2:

```python
cashflow_views = {
    "premium":     Raw("premium"),
    "benefits":    Group(["death_benefit", "retirement_bonus"]),
    "total":       Total(),
    "by_state":    ByState(),
    "by_kind":     ByKind(),
}
```

Each entry maps a user-facing view name to one self-describing object.
The Python type carries the kind of view; required fields are
type-checked.

Pros:

- Consistent with `StateRate` / `TransitionLump` / `ScheduledEvent`
  on the declaration side. Same idiom for both halves of the
  cashflow API.
- Pyright sees per-view required fields. `Group` needs members;
  `Total` does not.
- The view name maps directly to one entry in `result["cashflows"]`.
  The "names are the base unit" framing from `cashflow_api.md` §2.1
  carries over from declaration to output.
- Mixes named groups, grand totals, and structural views in one flat
  dict, distinguishable by view type rather than by nesting depth.
- Each view is a hint about *what to materialise*. The solver can
  allocate exactly what the user asked for and skip everything
  else.

Cons:

- Adds a small set of new public class names beyond
  `StateRate`/`TransitionLump`/`ScheduledEvent`.
- Slightly more verbose than a plain dict-of-lists for the simplest
  named-union case.

### 4.4 Linear-combination DSL

Frame every aggregation as a linear combination of component
contributions:

```python
cashflow_views = {
    "benefits": LinearCombination({
        "death_benefit":    1.0,
        "retirement_bonus": 1.0,
    }),
    "net": LinearCombination({
        "premium":       -1.0,
        "death_benefit": +1.0,
    }),
}
```

Pros:

- Most general. Subsumes all of §4.1–§4.3 mechanically.
- "Net cashflow = benefits − premiums" becomes a first-class
  expression rather than a post-processing step.

Cons:

- Most abstraction for a v1 the typed-view grammar can already
  cover.
- The interesting use case (signed sums) is also expressible via
  user-side post-processing of named outputs.
- Pushes a numerics concept (linear combination) into the
  declaration surface, where every other declaration object in
  `jact` is purely structural.

### 4.5 Just retain raw, recommend post-processing

```python
solve(cashflows=cashflows, ...)
# always returns one stream per component;
# user does any aggregation themselves.
```

Pros:

- No new API surface at all.
- Maximally composable.

Cons:

- Forfeits the memory win in §3 for users who only want totals or
  groups.
- Forces every downstream tool to re-derive the same boilerplate
  for "give me the total".
- Makes structural views (per-state, per-kind) something every user
  reimplements.

---

## 5. Preferred direction: typed view objects

Adopt §4.3.

The argument is the same as the typed-component recommendation in
`cashflow_api.md` §5:

- consistent with the rest of `jact`,
- pyright-checkable per-view required fields,
- one flat dict from view name to view, with the Python type carrying
  the discrimination,
- composes cleanly with future extensions.

The minimal v1 view set:

| View         | Output shape                            | Meaning |
|--------------|-----------------------------------------|---------|
| `Raw()`      | `{component_name: stream}` for *every* declared component | Per-component streams |
| `Raw(name)`  | `{name: stream}` for one named component | Single per-component stream |
| `Group([...])` | single stream                         | Sum of named components |
| `Total()`    | single stream                           | Sum over all declared components |
| `ByState()`  | `{state_name: stream}`                  | Sum collapsed by attachment state |
| `ByKind()`   | `{kind_name: stream}`                   | Sum collapsed by component kind |

Each declared view name becomes one entry in `result["cashflows"]`.
Time follows the rules already fixed in `cashflow.md` §4 (interval
accumulation as the leading interpretation); aggregation happens
before recording, so views and `record_every` do not interact.

### 5.1 Default behaviour

If `cashflow_views` is omitted, the solver returns raw component
streams — equivalent to a single implicit `{"raw": Raw()}`. This
preserves the current `cashflow.md` §5 framing: components are the
base output.

If the user passes `cashflow_views`, the solver returns exactly the
named views. Raw components are *not* added implicitly — that is the
whole point of the declaration: skipping the raw stream when only a
total is wanted is the memory win in §3. A user who wants both can
include `Raw()` explicitly:

```python
cashflow_views = {
    "raw":   Raw(),
    "total": Total(),
}
```

### 5.2 Validation

Validation stays a `StateSpace`-only concern, like `cashflow_api.md`
§2:

- `Group`/`Raw(name)` arguments must reference declared component
  names.
- View names must be unique across the dict.
- `ByState()` and `ByKind()` need no extra inputs and validate
  trivially.

### 5.3 Why not make views part of `state_space.cashflows(...)`

The declared cashflow object describes *which streams exist*. Views
describe *what to materialise on a particular solve*. Different
solves of the same cashflow declaration may legitimately want
different views (raw for diagnostics, totals for production).

Keeping views a `solve()`-level argument matches the analogous split
on the probability side: the callback (a per-solve concern) is not
baked into the model.

A future ergonomic helper like `cashflows.with_views(...)` that
bundles a declaration with default views is a possible extension but
not a v1 need.

---

## 6. Within-component splitting (v2 sketch)

The within-component axis from §2.2 is not committed in v1, but the
typed-view grammar makes it a natural fit:

```python
cashflow_views = {
    "premium_by_state": PerAttachment("premium"),
}
```

Output:

- `PerAttachment(name)` → `{attachment_key: stream}` for that
  component, where `attachment_key` is a state name (state-rate,
  scheduled-event) or `(src, tgt)` (transition-lump).

Adding `PerAttachment` later is purely additive: it does not change
any v1 view's shape, and the v1 set in §5 stays the same.

The reason to flag it now and not implement it now is that
within-component splitting requires the solver to keep per-attachment
contributions live inside the carry, while every v1 view collapses to
shapes the solver can produce by direct accumulation. Postponing
keeps the v1 implementation surface small.

---

## 7. Open questions

1. Is `Raw()` (no arg) the right way to spell "all declared
   components", or should the default be implicit and `Raw()` only
   accept a name? The former is symmetric with `Total()`; the latter
   forces the user to spell out which raw streams they want.
2. How should `ScheduledEvent` slot into `ByState()` and `ByKind()`?
   Most naturally: `ByState()` keys it under the occupied state at
   event time (consistent with state-rate and with the
   `ScheduledEvent.payments` framing in `cashflow_api.md` §2.1);
   `ByKind()` keys it under `"scheduled_event"`.
3. Should views be nestable — i.e. `Group(["benefits", "premiums"])`
   referring to other declared views, not just raw components? Most
   permissive, but cycles and ordering become a small concern. Flat
   over component names only is simpler and is enough for v1.
4. Does the same grammar extend to valuation? `Discounted(Total(),
   rate=...)` is tempting but conflicts with `cashflow.md` §7's
   recommendation to keep valuation a separate post-processing
   layer. The doc's current bias is to leave valuation outside the
   view grammar.
5. If the solver's accumulator structure ends up identical for
   `Total()` and a `Group([...])` covering every component, should
   the two be deduplicated under the hood, or kept as two
   user-visible names sharing one accumulator?
6. Should `ByState()` only key by states that *have* an attached
   component, or by every reachable state with zero streams for
   the rest? The first is more compact; the second composes
   better with downstream code that expects a fixed key set.

---

## 8. Recommendation

Adopt the typed-view grammar from §5:

- `cashflow_views: dict[str, View]` as a `solve()`-level kwarg,
- `Raw`, `Group`, `Total`, `ByState`, `ByKind` as the v1 view types,
- views materialise exactly the requested streams; raw components
  are not implicit when `cashflow_views` is non-empty,
- within-component splitting (`PerAttachment`) sketched but deferred
  to v2,
- everything in `cashflow.md` about recording semantics, output
  shape, terminal totals, and valuation kept as-is.

The hierarchical-dict alternative (§4.2) and the linear-combination
DSL (§4.4) are both expressible on top of the typed-view grammar via
post-processing or v2 extensions, so neither is foreclosed by this
choice. The plain-dict-of-lists status quo (§4.1) should be retired
before any user-facing release.
