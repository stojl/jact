# Package Cleanup Design

## Goal

Make `src/jact` easier to read and maintain without changing public APIs,
numerical behavior, JIT compatibility, or warm GPU performance.

## Scope

This cleanup covers all modules under `src/jact`. The main readability pressure
is `solver.py`, but the pass is package-wide: validation, helper naming, repeated
control flow, imports, and small local abstractions are all in scope. Large
module splits are out of scope for this pass unless an extraction is clearly
low-risk and preserves static JAX structures.

## Design

The refactor keeps current module boundaries and public exports stable. Changes
should be local and behavior-preserving:

- Use clearer helper names and smaller helpers where code is currently repeated.
- Consolidate repeated validation paths in `initial_distribution.py`,
  `cashflows.py`, `model.py`, and `state_space.py`.
- Keep solver changes focused on local simplification: cashflow contribution
  helpers, dense state conversion naming, and repeated source/view reduction
  code.
- Avoid changing solver algorithms, array shapes, pytree structures, JIT
  `static_argnames`, or benchmark configuration.

## Verification

Correctness is verified with the existing test suite and static checks when
available. Performance scope is warm JIT GPU execution only, using
`benchmarks/benchmark_solver_kernel.py` without `--allow-cpu`.

The benchmark should be run with a small but representative command before and
after the cleanup, then compared for obvious regressions:

```bash
python3.12 benchmarks/benchmark_solver_kernel.py --topology all --cashflow-scenarios none --warmup-runs 1 --timed-runs 20
```

If the sandbox cannot access GPU devices, the command must be run elevated.
