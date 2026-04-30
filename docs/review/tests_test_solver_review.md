# tests/test_solver.py Review

Reviewed file: `tests/test_solver.py`

## Findings

1. Prototype imports are path-mutating.

   The file inserts `docs/original_prototype` into `sys.path` to import
   `prototype_8`. This is pragmatic, but it couples test import behavior to
   mutable global path state. Suggested change: isolate prototype imports in a
   helper module or use `importlib` from an explicit file path.

## Tests To Add

- Prototype comparison import path remains isolated.
