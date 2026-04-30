# tests/test_initial_distribution_jit.py Review

Reviewed file: `tests/test_initial_distribution_jit.py`

## Status

Two review items are now resolved; one related behavior remains covered outside
this file.

## Resolved

1. Pytree registration is required, not skipped.

   The file now asserts that `InitialDistribution` is a registered JAX pytree
   and no longer treats loss of pytree registration as a skip-only condition.

2. Invalid runtime indices are covered at the JIT boundary.

   A jitted `per_individual(...)` path now verifies that out-of-range integer
   indices fail instead of silently producing zero mass.

## Covered Elsewhere

1. Concrete integer-dtype validation already exists in
   `tests/test_initial_distribution_integration.py`.

   That integration file rejects float state indices directly. No duplicate
   dtype test is needed here unless bool-specific behavior becomes part of the
   contract.
