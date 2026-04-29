# tests/test_solver.py Review

Reviewed file: `tests/test_solver.py`

## Findings

1. Missing validation tests for positive solver dimensions.

   The solver currently lacks clean validation for `horizon <= 0` and
   `steps_per_unit <= 0`. This test file should lock in clear `ValueError`
   behavior once implemented.

2. Reference callable output shape validation is not covered.

   There are no tests for intensities returning scalar, rank-1, or wrong-width
   arrays. Suggested change: add malformed callable tests matching the proposed
   solver validation.

3. Prototype imports are path-mutating.

   The file inserts `docs/original_prototype` into `sys.path` to import
   `prototype_8`. This is pragmatic, but it couples test import behavior to
   mutable global path state. Suggested change: isolate prototype imports in a
   helper module or use `importlib` from an explicit file path.

## Tests To Add

- `horizon=0`, negative horizon, and `steps_per_unit=0` fail cleanly.
- Malformed intensity output shape fails cleanly.
- Prototype comparison import path remains isolated.

