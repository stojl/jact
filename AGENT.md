# Agent Guidance

This file provides guidance to coding agents when working with code in this repository.

## Commands

```bash
pip install -e ".[dev]"           # install with dev deps (pyright, ruff, pytest)
pyright jact                      # type check public API
ruff check jact                   # lint (imports, style, unused code)
pytest                            # run all tests
pytest tests/test_state_space.py  # run one file
pytest -k test_reachable_from     # run tests matching a name
pytest -x                         # stop on first failure
```

**Quality checks:**
- **pyright** (basic mode) catches type mismatches and undefined names
- **ruff** enforces import order, detects unused code, and flags common errors
- Run both before committing: `pyright jact && ruff check jact`

## Architecture

`jact` computes transition probabilities for semi-Markov multi-state models with
duration-dependent intensities and can evaluate cashflows in the same fused JAX
solve.

Read `docs/api_spec.md` as the public contract. It is the only normative API
specification in the repo.

- `StateSpace` is topology only. It validates state names and transitions and
  provides `build()`, `cashflows()`, and `initial_*()` helpers.
- `Model` binds intensities to transitions via `transitions`, `exits`, and
  `groups`. Every declared transition must be covered exactly once.
- `InitialDistribution` encodes the joint `(state, duration)` distribution at
  `t = 0`. The declared initial-state set is structural and part of the JIT
  boundary.
- `solve()` reduces to the reachable subgraph, advances one `StateCarry` per
  reachable state with midpoint quadrature, and can emit probability output and
  cashflows together.
- Cashflows are declared from a `StateSpace` and aggregated per solve through
  `cashflow_views`. Streamed cashflows use interval accumulation; probability
  output uses snapshot semantics.

## Conventions

- Keep `docs/api_spec.md` aligned with the implementation and tests whenever
  the public surface changes.
- `archive/original_prototype/` and `notes/` are background material, not API.
- Time is the leading axis of every probability leaf and every streamed
  cashflow leaf. Terminal cashflow leaves drop the time axis entirely.
- `probability=None` omits the `"probability"` key. `cashflows=None` omits the
  `"cashflows"` key.
- If `cashflows` is supplied and `cashflow_views` is omitted or `None`, the
  solver defaults to `{"raw": Raw()}`. `cashflow_views={}` is allowed and
  returns an empty mapping.
- Cashflow view weights are plain user-supplied scalars or callables evaluated
  at each inner-step midpoint. Discount construction belongs in caller code.
- Reserved covariate names `initial` and `initial_duration` are rejected, as
  are legacy kwargs `callback` and `freeze_initial`.
- `collapse_point_no_duration` is the canonical compact probability output for
  actuarial work. Pick `probability` and `record_every` before scaling batch
  size.
- `archive/original_prototype/prototype_8.py` is historical reference code for
  solver behavior, not public API.
- Python >= 3.10 is required.
