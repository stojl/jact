# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

`jact` computes transition probabilities for semi-Markov multi-state models with duration-dependent intensities, vectorized over 100K+ individuals via JAX. The full pipeline — covariates → probabilities — compiles into a single XLA program via `jax.lax.scan`.

The design splits three concerns, each in its own module:

- **`state_space.py` — `StateSpace`**: topology only (states, allowed transitions). No intensities, no data, serializable to JSON. Performs all structural validation at construction. Computes reachability via BFS; the solver uses this to prune unreachable states.
- **`model.py` — `Model` / `ReducedModel`**: binds intensity callables to a `StateSpace`. Built via `StateSpace.build()`, which takes three optional kwargs (`transitions`, `exits`, `groups`) and validates that every declared transition is covered **exactly once** across all three. `Model.reduce(initial)` extracts the reachable subgraph as a J×J matrix of callables with `initial` at index 0.
- **`solver.py` — `solve()` / `_heun_solver`**: Heun (2nd-order predictor-corrector) scheme inside `jax.lax.scan`. Tracks two objects: `p` (absolutely continuous duration density, `(batch, J, D)`) and `p_point` (point mass at duration zero for the initial state, `(batch, D)`) — kept separate for numerical accuracy. The core derivative is computed per-individual in `_compute_core` and `vmap`ped over the batch axis. The `mu_*_matrix` is a nested Python list; the solver is `@jit`ed with the *structure* static, so rebuilding a model reconciles a new trace.
- **`callbacks.py`**: small functions `(p, p_point) → PyTree` that reduce the solver state each step (e.g. marginalize over duration, fold the point mass into the first state). Resolved by name or passed as a callable. This is also the intended extension point for cashflow/integral-transform features.

### The three ways to assign intensities to transitions

When calling `state_space.build(...)`, each transition must be assigned exactly once via one of:

- `transitions={(src, tgt): fn}` — one callable per transition, returning `(batch, D)`.
- `exits={src: fn}` — one callable covering *all* exits from `src`, returning `(n_targets, batch, D)` in the order of `state_space.targets(src)`.
- `groups={fn: [(src, tgt), ...]}` — one callable covering an arbitrary set of transitions, returning `(n_transitions, batch, D)` in the listed order.

`exits` and `groups` callables are sliced at matrix-build time via `_make_slice_wrapper` so the solver sees uniform `(batch, D)` outputs.

### Intensity callable contract

```python
def intensity(t, d, **kwargs) -> jnp.ndarray:
    # t: scalar clock time (ranges 0 → horizon)
    # d: (1, D) duration grid
    # kwargs: (batch, ...) covariate arrays
    # returns: (batch, D) — or (n, batch, D) for exits/groups
```

Callables must be **pure** and **JIT-compatible** (no data-dependent Python control flow, no non-JAX ops). Fitted parameters are captured via closures and become compile-time constants. `t` is clock time (use `baseline_age + t` for attained age); `d` is duration-in-current-state — using both is what makes an intensity semi-Markov.

### Reachability and reduction

Solving from an initial state automatically calls `Model.reduce(initial)`, which runs BFS from `initial` and extracts only the reachable rows/columns of the full J×J intensity matrix. The initial state is always at index 0 in the reduced system. `result["states"]` in the output tells you which reduced index maps to which state name.

## Conventions

- `docs/api_spec.md` is the authoritative API contract — tests reference it explicitly ("per docs/api_spec.md"). Keep it in sync when changing public surface.
- `docs/original_prototype/prototype_8.py` is the reference numerics the solver was ported from; consult it when debugging solver behavior.
- Python >= 3.10 is required (uses `X | Y` union syntax and PEP 604 features in places).
