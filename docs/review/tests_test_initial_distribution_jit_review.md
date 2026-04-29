# tests/test_initial_distribution_jit.py Review

Reviewed file: `tests/test_initial_distribution_jit.py`

## Findings

1. The pytree registration guard skips rather than fails.

   `_pytree_registered` skips Group B if `InitialDistribution` is not a pytree.
   Since pytree behavior is part of the documented contract, a regression here
   should probably fail. Suggested change: turn the skip guard into a direct
   assertion.

2. The tests do not cover invalid traced index values at solve time.

   Constructors are trace-clean, but invalid index values are only partly
   covered outside JIT. Suggested change: add a jitted solve or canonicalization
   boundary test for invalid per-individual indices if feasible.

3. Integer dtype is not asserted.

   The spec says `states` is an int32 index array. This file does not cover
   float or bool index arrays. Suggested change: add concrete dtype tests once
   production validation is tightened.

## Tests To Add

- Pytree registration is required, not skipped.
- Concrete non-integer state index arrays are rejected.
- JIT-boundary behavior for invalid runtime indices is documented.

