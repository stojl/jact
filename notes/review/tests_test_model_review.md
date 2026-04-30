# tests/test_model.py Review

Reviewed file: `tests/test_model.py`

## Status

All previously noted gaps are now covered by the current test file.

## Resolved

1. Non-callable assignment coverage exists.

   The file now rejects non-callable `transitions`, `exits`, and `groups`
   assignments directly.

2. Multi-output callable shape failure is tested.

   `test_exit_assignment_rejects_too_few_outputs` locks in the reduced-model
   failure for underspecified multi-output exit callables.

3. Empty group assignment is tested.

   `test_build_rejects_empty_group_assignment` covers direct rejection.
