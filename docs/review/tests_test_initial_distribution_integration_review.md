# tests/test_initial_distribution_integration.py Review

Reviewed file: `tests/test_initial_distribution_integration.py`

## Findings

1. All-zero mixture behavior is not covered.

   Integration tests cover mixtures and batch mismatches, but not a normalized
   distribution whose total mass is zero. Suggested change: add a test after the
   intended behavior is decided in `InitialDistribution`.

2. Name helper coverage does not include invalid names with restricted
   `initial_states`.

   `initial_per_individual()` has a useful host-side name conversion path. The
   tests cover matching name/index paths, but not a name outside the restricted
   declared state set. Suggested change: add that failure case.

3. Solver integration coverage is mostly point-mass oriented.

   That matches the current implementation, but if density seeding is added
   later this file should grow explicit integration tests for mixed density and
   point-mass starts.

## Tests To Add

- Zero-total `InitialDistribution(..., normalise=True)` behavior.
- Invalid `state_names` when `initial_states` restricts the allowed set.
- Future density-seeded initial distributions, if introduced.

