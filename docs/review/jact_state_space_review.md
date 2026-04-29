# jact/state_space.py Review

Reviewed file: `jact/state_space.py`

## Findings

1. Input validation assumes concrete sequence semantics.

   `_validate_inputs()` calls `len()`, `set()`, and `count()` on `states` and
   `transitions`. The docstrings say `Sequence`, but tests also describe
   accepting iterables. Generators or one-shot iterables will fail with low-level
   errors. Suggested change: coerce `states` and `transitions` to tuples at the
   start of `__init__`, then validate the tuples.

2. State names are not type-checked.

   Duplicate and membership checks work for hashable non-string state labels,
   but the public API and docs consistently describe state names as strings.
   Suggested change: reject non-string states up front, matching the validation
   already used by `InitialDistribution`.

3. JSON transition order is not state-space order.

   `to_json()` serializes `sorted(self._transitions)`, which is lexical tuple
   order rather than source/target state order. This does not break current
   behavior because transitions are stored as a set, but it makes the serialized
   topology less aligned with the declared state order. Suggested change: emit
   transitions ordered by source and target state index.

## Tests To Add

- Non-string state labels are rejected.
- Generator input either works after tuple coercion or is explicitly rejected.
- JSON output transition order follows state order.

