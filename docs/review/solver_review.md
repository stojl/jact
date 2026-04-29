# Solver Review

Reviewed file: `jact/solver.py`

This review covers `jact/solver.py` after the final prototype pass. The
recommendations below focus on API behavior, validation, and maintainability;
they intentionally avoid changes to the solver numerics.

## Findings

1. `cashflow_views={}` is documented as valid but currently crashes.

   `docs/api_spec.md` states that passing an empty cashflow view mapping should
   produce no cashflow views rather than defaulting to `Raw()`. The current
   solver enters the cashflow scan branch when cashflow components exist, even
   when no stream or terminal views exist, which can fail with:

   ```text
   AttributeError: 'tuple' object has no attribute 'density'
   ```

   Suggested change: when prepared cashflow views are empty, skip cashflow view
   evaluation and return `result["cashflows"] = {}`.

2. `horizon` and `steps_per_unit` need explicit positive-integer validation.

   Invalid values currently fail through lower-level implementation errors:

   - `horizon=0` can produce an internal `IndexError`.
   - `steps_per_unit=0` produces `ZeroDivisionError`.

   Suggested change: reject non-positive or non-integral values at `solve()`
   entry with clear `ValueError` messages before computing `solver_steps`.

3. The reference callable output shape is assumed, not validated.

   `solve()` uses the first available intensity callable, or a cashflow payment
   callable for transition-free cashflow-only models, to infer dtype and batch
   size. The code assumes that this callable returns a rank-2 array shaped
   `(batch, solver_steps)`. A malformed scalar or rank-1 output will fail later
   with less actionable errors.

   Suggested change: validate the reference output rank and duration-grid width
   immediately after evaluation, and raise a clear `ValueError` if it is not
   shaped `(batch, solver_steps)`.

## Lower-Risk Cleanup

`_solver_step` and `_solver_step_dynamics` duplicate the hazard traversal logic.
Because this area is numerically sensitive, avoid broad refactors now. If this
is cleaned up later, prefer a narrow helper that only packages per-row hazards
and survival factors, and keep the existing solver parity tests as the safety
net.

## Verification

Focused tests passed during review:

```text
pytest -q tests/test_solver.py tests/test_cashflows.py
37 passed
```
