# tests/test_cashflows.py Review

Reviewed file: `tests/test_cashflows.py`

## Findings

1. Empty cashflow view mapping is not tested.

   The docs say `cashflow_views={}` should produce no cashflow outputs. The
   current solver crashes in this case. Add a regression test once the solver is
   fixed.

2. Mutation-after-validation is not covered.

   Cashflow declarations retain user-supplied payment mappings. A test should
   cover whether mutating the original dict after `state_space.cashflows()`
   changes the declaration.

3. Weight scalar forms are under-specified.

   Tests cover Python scalar weights and callable weights, but not rank-0 JAX
   arrays. Suggested change: add an explicit accept-or-reject test after the API
   decision.

## Tests To Add

- `cashflow_views={}` returns `result["cashflows"] == {}`.
- Mutating original payment dicts after validation cannot bypass validation.
- Rank-0 JAX scalar weight behavior.

