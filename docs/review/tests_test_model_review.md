# tests/test_model.py Review

Reviewed file: `tests/test_model.py`

## Findings

1. Non-callable assignment coverage is missing.

   The tests cover missing, overlapping, and unknown transition assignments, but
   do not cover non-callable objects passed as intensities. This matches a
   production validation gap in `jact/model.py`.

2. Multi-output callable shape errors are not tested.

   `exits` and `groups` tests verify happy-path slicing, but not too-few output
   slices or wrong rank. Suggested change: add failure tests that lock in a
   clear error.

3. Empty group transition lists are not tested.

   Empty groups are currently ignored and lead to indirect coverage errors.
   Suggested change: decide and test direct rejection or the current indirect
   behavior.

## Tests To Add

- Non-callable `transitions`, `exits`, and `groups` entries.
- Too-short multi-output callable for an exit or group.
- Empty group assignment.

