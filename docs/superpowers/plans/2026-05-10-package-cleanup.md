# Package Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `src/jact` for readability while preserving behavior and warm JIT GPU performance.

**Architecture:** Keep public modules and exports stable. Prefer small local helper extraction and validation simplification over large structural moves, especially in JAX-jitted solver code.

**Tech Stack:** Python 3.12, JAX, pytest, ruff, pyright, GPU benchmark script in `benchmarks/benchmark_solver_kernel.py`.

---

### Task 1: Baseline Verification

**Files:**
- Read: `src/jact/*.py`
- Read: `tests/test_*.py`
- Read: `benchmarks/benchmark_solver_kernel.py`

- [x] Run `python3.12 -m pytest` and record whether the current tree passes.
- [x] Run `python3.12 benchmarks/benchmark_solver_kernel.py --topology all --cashflow-scenarios none --warmup-runs 1 --timed-runs 20` using elevated permissions if GPU is not visible in the sandbox.
- [x] Do not edit code until the baseline state is understood.

### Task 2: Package-Wide Local Cleanup

**Files:**
- Modify: `src/jact/state_space.py`
- Modify: `src/jact/model.py`
- Modify: `src/jact/initial_distribution.py`
- Modify: `src/jact/cashflows.py`
- Modify: `src/jact/probability.py`
- Modify: `src/jact/result.py`
- Modify: `src/jact/__init__.py`

- [x] Remove unused imports and simplify type imports.
- [x] Extract or rename local helpers only where it reduces repeated logic.
- [x] Keep all public class names, function names, and exports unchanged.
- [x] Run focused tests for changed modules after each meaningful edit.

### Task 3: Solver Readability Cleanup

**Files:**
- Modify: `src/jact/solver.py`
- Test: `tests/test_solver.py`
- Test: `tests/test_cashflows.py`

- [x] Review dense state and point-mass helpers; leave names unchanged where renaming does not improve clarity.
- [x] Extract repeated cashflow contribution code only when the helper remains JAX-friendly.
- [x] Preserve `_solve_impl` JIT signature, static argument names, tuple structures, and array shapes.
- [x] Run `python3.12 -m pytest tests/test_solver.py tests/test_cashflows.py`.

### Task 4: Final Verification

**Files:**
- Read: full git diff

- [x] Run `ruff check src/jact tests` if available.
- [x] Run `pyright` if available.
- [x] Run `python3.12 -m pytest`.
- [x] Run the warm JIT GPU benchmark command from Task 1 again.
- [x] Compare benchmark output against baseline and report any material timing changes.
