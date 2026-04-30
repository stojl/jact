# Cashflow API: shape of the declaration grammar

Notes on the public grammar for declaring cashflow components. This is a
follow-up to [cashflow.md](cashflow.md), which fixes the conceptual model
(state-rate, transition-lump, scheduled-event; named components as the
base output). This document is about the surface syntax those concepts
should be expressed in, not about the underlying transforms.

The current draft in `docs/api_spec.md` uses a stringly-typed `kind`
discriminator with dict-of-dict nesting:

```python
cashflows = state_space.cashflows(
    components={
        "premium": {
            "kind": "state_rate",
            "states": {"healthy": premium_fn},
        },
        "death_benefit": {
            "kind": "transition_lump",
            "transitions": {("healthy", "dead"): death_fn},
        },
        "retirement_bonus": {
            "kind": "scheduled_event",
            "when": event_time_fn,
            "states": {"healthy": bonus_fn},
        },
    },
)
```

That shape works, but it is the least elegant part of the cashflow
plan. The rest of `jact` does not look like this.

---

## 1. What is unelegant about the current draft

### 1.1 Inconsistent with the rest of `jact`

`jact` uses dataclasses and structural kwargs to express variants:

- `TransitionSpec(fn=..., continuity_t=..., continuity_d=...)` carries
  per-callable metadata.
- `StateSpace.build(transitions=..., exits=..., groups=...)` dispatches
  between assignment modes via named kwargs, not a discriminator string.

The cashflow grammar uses neither pattern. It introduces a new
convention where component variants are encoded as a `"kind"` string
inside a nested dict. That is the only place in the public API that
discriminates this way.

### 1.2 Stringly-typed and not type-checkable

`"kind": "state_rate"` is a free string. A typo silently produces a
different validation path or a generic "unknown kind" error. The shape
of the surrounding dict (`"states"` vs `"transitions"` vs `"when" +
"states"`) depends on the discriminator value, but pyright cannot see
that link.

### 1.3 Nesting depth

The current grammar is four levels deep:

```
cashflows -> components -> name -> {kind, attachment-key, attachment-dict, fn}
```

The middle level (`name -> {kind, ...}`) is pure ceremony. The
component name and the component definition are split across a key and
a dict-with-discriminator instead of being a single object.

### 1.4 Field naming inside `scheduled_event`

For state-rate, the inner `"states"` dict means "state currently
occupied -> payment rate while occupied". For scheduled-event,
`"states"` means "state currently occupied at event time -> payment
amount evaluated at event time". Both are payments keyed by state, but
that is not what the field name says. The reader has to re-derive it
each time.

---

## 2. Preferred direction: typed component classes

The cleanest fit with the rest of `jact` is to make each component a
small dataclass parallel to `TransitionSpec`, and keep the top-level
grammar a single flat dict from name to component:

```python
cashflows = state_space.cashflows({
    "premium":          StateRate({"healthy": premium_fn}),
    "death_benefit":    TransitionLump({("healthy", "dead"): death_fn}),
    "retirement_bonus": ScheduledEvent(
        when=event_time_fn,
        payments={"healthy": bonus_fn},
    ),
})
```

What this changes:

- The `kind` discriminator is gone. The Python type *is* the
  discriminator.
- One level of nesting is removed. The component name maps directly to
  one self-describing object.
- Pyright sees the per-kind required fields. Mistyping `payments` as
  `payement` is caught at static-check time, not at runtime validation.
- The grammar mirrors `TransitionSpec`: a component is a small frozen
  object that pairs callables with a small amount of structural
  metadata.

The validation surface is unchanged from the cashflow design:

- `StateRate` keys must be declared states,
- `TransitionLump` keys must be declared transitions,
- `ScheduledEvent.payments` keys must be declared states,
- component names must be unique.

These remain `StateSpace`-only checks; no `Model` is involved.

### 2.1 Field naming

Each component type uses the same shape: an attachment dict whose keys
are state names or transitions, and whose values are payment callables.
A consistent name across kinds makes that parallel visible:

| Component        | Attachment-dict keys           | Payment dict field name |
|------------------|--------------------------------|-------------------------|
| `StateRate`      | state name                     | positional / `payments` |
| `TransitionLump` | `(src, tgt)` transition        | positional / `payments` |
| `ScheduledEvent` | state name (occupied at event) | `payments`              |

The current draft uses `"states"` for both `state_rate` and
`scheduled_event`; renaming to `payments` removes that overload.
`StateRate` and `TransitionLump` can take the dict positionally
since there is only one such argument.

### 2.2 What the declaration object stays

Everything else in `cashflow.md` carries over unchanged:

- the declaration is reusable across models,
- aggregation, valuation, cumulative totals, and terminal totals stay
  outside the declaration,
- the declared object's only job is to answer the four questions in
  `api_spec.md` §"Cashflow declaration layer": which components exist,
  what kind, where attached, what callable.

The grammar change is cosmetic in the sense that no transform behaviour
moves; it is structural in the sense that the public types change.

---

## 3. Alternative considered: kind-keyed kwargs on `cashflows()`

A second option mirrors `StateSpace.build(...)`:

```python
cashflows = state_space.cashflows(
    state_rate={
        "premium": {"healthy": premium_fn},
        "waiver":  {"disabled": waiver_fn},
    },
    transition_lump={
        "death_benefit": {("healthy", "dead"): death_fn},
    },
    scheduled_event={
        "retirement_bonus": ScheduledEvent(
            when=event_time_fn,
            payments={"healthy": bonus_fn},
        ),
    },
)
```

Pros:

- Direct structural parallel with `build(transitions=, exits=,
  groups=)`. Users learn one dispatch idiom in `jact` and it shows up
  in both intensity and cashflow declaration.
- No new class names for the two simple kinds (`StateRate`,
  `TransitionLump` collapse back into plain dicts).

Cons:

- Component identity is now split across kinds. To enumerate every
  declared component the user (or downstream code) has to walk three
  separate kwargs and merge them. The "names are the base unit"
  framing weakens.
- Scheduled-event still needs a class because it carries a `when`
  callable plus a payments dict; the grammar is no longer fully
  uniform across kinds.
- Name-uniqueness validation crosses three kwargs instead of being a
  single dict-key check.

This is a reasonable second choice. I prefer the typed-component
version (§2) because the component dict is the natural unit for both
the user and downstream aggregation/valuation layers.

---

## 4. Alternative considered: keep the `kind` string

Pros:

- Fully JSON-shaped; in principle serialisable to a config file.
- Zero new public class names.

Cons:

- Payment callables are not JSON-serialisable, so the JSON-shape
  argument does not translate into a real persistence story.
- All the issues in §1 remain.

I do not recommend this path.

---

## 5. Recommendation

Adopt the typed-component grammar from §2:

- `state_space.cashflows({name: Component, ...})` as the entry point,
- `StateRate`, `TransitionLump`, `ScheduledEvent` as the component
  types,
- `payments` as the consistent name for the per-attachment payment
  dict where it is not the sole positional argument,
- everything in `cashflow.md` about transforms, recording semantics,
  aggregation, and valuation kept as-is.

The kind-keyed kwargs alternative (§3) is acceptable if a stronger
parallel with `build()` turns out to matter more in practice than
keeping component identity in one flat dict. The stringly-typed draft
(§4) should be retired before any user-facing release.
