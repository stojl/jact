# tests/test_cashflows.py Review

Reviewed file: `tests/test_cashflows.py`

## Status

All previously noted gaps are now covered by the current test file.

## Resolved

1. Empty cashflow view mapping is tested.

   `test_empty_cashflow_view_mapping_returns_empty_outputs` locks in
   `cashflow_views={}` returning `result["cashflows"] == {}`.

2. Mutation-after-validation is covered.

   `test_cashflow_declaration_copies_payment_mappings` verifies that mutating
   the original payments mapping after declaration does not alter the validated
   cashflow component.

3. Rank-0 JAX scalar weights are covered.

   `test_cashflow_views_accept_rank_zero_array_weight` accepts rank-0 arrays,
   and `test_cashflow_views_reject_non_scalar_array_weight` rejects non-scalars.
