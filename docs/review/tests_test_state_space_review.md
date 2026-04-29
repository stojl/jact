# tests/test_state_space.py Review

Reviewed file: `tests/test_state_space.py`

## Findings

1. Several exception assertions are too broad.

   Tests such as duplicate states and unknown transitions use
   `pytest.raises((ValueError, Exception))`, which would pass for almost any
   failure. Suggested change: assert the exact expected exception type and a
   useful message fragment.

2. Missing validation coverage for state label types.

   The production API treats states as names, but this test file does not cover
   non-string state values. Suggested change: add a test once production
   behavior is decided.

3. JSON order is not asserted.

   Roundtrip tests compare transition sets and queries, but not the serialized
   order. Suggested change: assert deterministic JSON order if stable serialized
   diffs matter.

## Tests To Add

- Exact exception type/message assertions for invalid construction.
- Non-string state input behavior.
- Serialized transition order behavior.

