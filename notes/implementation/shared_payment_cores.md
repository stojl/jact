# Plan: shared integration for scaled payments

Status: implemented and validated.

The steps below record the implementation design. Prepared attachment records
carry shared task IDs, resolved core definitions, and consumer weights. A local
dictionary in the traced cashflow step reuses each complete base contribution;
it adds no cross-step cache or scan carry. Registry names determine sharing,
while callable identity is retained only for correct static JIT cache keys.

The [CPU benchmark results](../reflections/shared_payment_core_benchmark.md)
show 2.77–4.72x speedups for eight consumers. Single-consumer cases have no
consistent benefit, including one measured slowdown.

Validation on 2026-09-24:

- `.venv/bin/pyright`: no errors or warnings.
- `.venv/bin/ruff check src tests`: passed.
- `.venv/bin/pytest -q`: 334 passed in 384.18 seconds.
- All 16 benchmark cases passed numerical output comparisons; traced-program
  tests verify computation sharing for all four component kinds.

## Objective

Allow named payments to share a duration-dependent core while applying distinct
duration-independent weights. Evaluate and integrate each compatible core once
per solver step, then scale its complete contribution for each consumer. Preserve
component names, attribution, views, recording, and terminal accumulation.

For a state-rate payment `b_i(t, d, x) = a_i(t, x) * g(t, d, x)`, compute the
expected contribution of `g` once and multiply it by each `a_i`. The shared
contribution includes the density reduction, point masses, and same-step inflow
corrections. Transition payments similarly include the transition-specific
hazard and transfer corrections.

Derived fields continue to share exogenous function evaluations. Shared payment
contributions depend on the evolving probability state and belong in the
cashflow execution plan, rather than the derived-field graph.

## Public contract

Add `jact.cashflows.Scaled(core: str, *, weight=1.0)` as an immutable declaration
and a keyword-only `cores: Mapping[str, Payment] | None = None` argument to
`StateSpace.cashflows`. Each registry name has one authoritative definition;
payments created independently can refer to that same name.

```python
def eligible(t, d, **kwargs):
    return jnp.where(d >= 0.5, 1.0, 0.0)

cashflows = state_space.cashflows({
    "disability_income": cf.StateRate({
        "disabled": cf.Scaled(
            "eligible", weight=lambda t, **kw: 0.60 * kw["salary"],
        ),
    }),
    "pension_contribution": cf.StateRate({
        "disabled": cf.Scaled(
            "eligible", weight=lambda t, **kw: 0.10 * kw["salary"],
        ),
    }),
}, cores={"eligible": eligible})
```

- Accept `Payment | Scaled` in every payment mapping: `StateRate`,
  `TransitionLump`, `ScheduledEvent`, and `DurationEvent`.
- `core` is a non-empty registry name, resolved within this cashflow declaration.
  Registry values are ordinary `jact.typing.Payment` callables. Do not make
  `Scaled` itself a payment callable: the solver explicitly resolves and lowers
  the declaration. Unknown names are errors at declaration validation, even on
  attachments that might later be pruned as unreachable.
- The registry is local to `CashflowDeclaration`, copied at declaration time,
  and exposed as an immutable mapping. No global registry or override hierarchy
  is introduced. Core names have a separate namespace from component and derived
  field names. A core is not itself an output component or a derived field.
- Unused core definitions are allowed and are not evaluated. A missing or empty
  registry is valid when no payment references a named core.
- `weight` follows existing view-weight value conventions: a scalar, rank-zero
  array, `None` for unity, or a `Weight` callable `(t, **kwargs)` returning a
  scalar or `(batch,)`-broadcastable value. Per-individual arrays are supplied
  through solve inputs and returned by a callable.
- Weight callables receive inputs and time-only derived fields, without `d` or
  duration-dependent derived fields, including fields indirectly depending on
  `d`. Reuse the existing time-context field resolver. Report invalid weight
  shapes or unavailable fields with component/attachment context; do not attempt
  to prove arbitrary Python callables duration-independent through inspection.
- Preserve existing scalar normalization and broadcasting rules. Do not add
  non-negativity restrictions to weights.
- A core may depend on time, duration, and covariates, and its definition may be
  factory-created. Sharing is expressed by the same registry name, not callable
  identity or inferred equivalence. Distinct names remain distinct even if their
  definitions use the same callable. A reference never supplies a second
  definition that could conflict with the registry.
- Reject callables and nested `Scaled` declarations as the `core` argument;
  registry values must be ordinary payment callables, not references or wrappers.
  General expression trees, sums of cores, and automatic algebraic factorization
  are outside scope.
- Keep the existing `Payment` protocol unchanged and update mapping annotations
  and exports to describe the new declaration explicitly.

## 1. Add declarations and validation

Files: `src/jact/cashflows.py`, `src/jact/state_space.py`,
`tests/test_cashflows.py`, `tests/test_typing.py`, and `tests/typing_consumer.py`.

Add `Scaled`, its export, widened payment mapping annotations, the `cores`
argument, and the declaration registry. Freeze a copy of the registry while
retaining its original callable values. Preserve existing positional constructor
compatibility when extending `CashflowDeclaration`. Validate names, definitions,
and references before solve preparation.
Factor weight validation/normalization for reuse with existing views while
preserving their behavior and errors. Include useful error context for malformed
registries, unknown references, nested wrappers, invalid weights, and invalid
runtime weight shapes. Mutating the caller's original registry must not change
an existing declaration.

Deliverable: all four component constructors accept the new declaration, and
plain payment callables retain their current API.

## 2. Prepare unique core tasks and consumer references

Files: `src/jact/_cashflow_ir.py` and `_prepare_cashflow_components` in
`src/jact/solver.py`.

Represent computation separately from attribution:

1. Unique core tasks contain the resolved registry name, its callable definition,
   and its integration context.
2. Consumers reference a task ID and retain the component index, attachment,
   kind, weight, and midpoint/event classification.

Assign task IDs during preparation after reachable-state reduction. Group named
references by registry name and compatible context, resolving each name to its
single definition. Do not compare core callables to discover sharing.

Separately, JIT cache correctness still requires the static plan to retain and
distinguish actual callable definitions and configuration. Two declarations that
use the same name with different definitions must not reuse stale compiled code.
Use identity-aware callable records, following `_derived._Node`, for this
internal cache concern, including callables with custom equality or no hash.
Neither a registry name nor an allocation-specific task ID is sufficient as the
entire JIT cache key. This is independent of the public sharing rule.

Sharing keys are conservative:

| Kind | Required matching context in addition to registry name |
| --- | --- |
| State rate | Source/occupied state |
| Transition lump | Source state and transition hazard slot |
| Scheduled event | Occupied state and `when` callable identity |
| Duration event | Occupied state and duration target: normalized scalar value or callable identity |

Kinds never share tasks with each other. Phase and payment duration layout are
fixed by kind and solve configuration; keep those assumptions explicit. Different
event callables that happen to return the same time or target remain separate.
Weights and component names do not participate in the core sharing key.

Lower ordinary payments as private, per-attachment unit-weight tasks. Do not
implicitly merge these with named cores or with other ordinary payments. To
request an unscaled use of a named core, use `Scaled("eligible")`. Each consumer
still contributes separately to output totals. Preserve declaration and
attachment order when aggregating consumers.

Deliverable: two scaled references to `"eligible"` on the same state reference
one task, even when their payment declarations are created by separate factory
calls. Different names, states, transitions, or event contexts remain distinct.

## 3. Execute each complete base contribution once per step

Files: `_compute_cashflow_step`, event resolution helpers, grid-context setup,
and scan setup in `src/jact/solver.py`.

Extract the existing per-attachment contribution calculations into task
evaluation without changing the quadrature formulas. Evaluate each unique task
once per inner step and retain its `(batch,)` contribution for consumers.

- State rates retain density midpoint factors, point survival, and same-step
  survived-inflow corrections.
- Transition lumps retain the selected hazard, transfer factors, point exits,
  and chained-exit corrections.
- Scheduled events retain their current activation masks, grid rounding,
  horizon rules, left-grid payment evaluation, and point masses.
- Duration events retain target resolution, target density extraction, point
  crossing behavior, and payment evaluation at effective duration.
- Apply existing payment duration limits to the core, including tail handling
  and threshold reuse. Preserve intensity duration limits independently.

Resolve schedules and duration targets against task identities. Update derived
evaluation context keys, particularly duration-event keys currently based on
component indices, so sharing does not mix distinct duration contexts. Preserve
existing duration-only field hoisting and input-only event resolution.

This is a static execution plan traced into JAX, with fresh numerical results
each inner step. Do not cache integrated values across time steps or solves, and
do not add the base contributions to the scan carry or public result.

## 4. Scale and attribute contributions

Reuse or generalize `_evaluate_weight` with caller-specific error labels.
Continuous state-rate and transition-lump weights use the step midpoint.
Scheduled-event and duration-event weights use the existing left-endpoint event
sampling convention, including existing grid rounding behavior.

Multiply each complete base contribution by its consumer weight, then populate
the existing component, state, kind, and event aggregations. Apply view weights
afterwards, as today. This preserves `Raw`, `Group`, `Total`, `ByState`, `ByKind`,
streamed output, and terminal output semantics.

Apply component weights at each inner step before `record_every` accumulation.
Avoid any numerical multiplication for unity weights where practical. Two
consumers with weights `w1` and `w2` contribute `w1 * base + w2 * base` to totals;
the internal base is not an additional named component or an extra contribution.

## 5. Verify semantics and actual sharing

Add focused numerical tests against manually multiplied ordinary payment
functions, using floating-point tolerances rather than bitwise equality:

- All four component kinds; scalar and per-person time-varying weights; weights
  using input-only and time-derived fields; negative and zero weights.
- Mixed density and initial point mass, nonzero initial duration, same-step
  inflow, chained exits, duration limits, and event boundary/rounding cases.
- All view types, a second view-level discount weight, terminal outputs, and
  `record_every > 1` with time-varying weights.
- Ordinary and scaled consumers together, repeated use in different components,
  reachable-state pruning, and unchanged probability results.
- JIT execution and gradients through core parameters, weight inputs, and
  model inputs, compared with the unfactored formulation.
- Invalid declarations and weights, including attempted use of a
  duration-dependent derived field from a weight.

Add structural tests proving shared task references and isolation between
incompatible contexts. Test independently factory-created payment declarations
referencing the same name, different names pointing to the same callable,
unscaled named references, unused registry entries, unknown names, and registry
mutation isolation. Test separate declarations using the same name with different
factory-created definitions, including equal-comparing/unhashable callable
objects, to guard against stale JIT cache reuse.
Supplement numerical tests with a focused tracing/IR check that one shared task
emits one base reduction path; Python call counts alone do not prove runtime
savings after JIT compilation.

Run `pyright`, `ruff check src tests`, and the full `pytest` suite once the
implementation and focused checks pass.

## 6. Benchmark and document

Add `benchmarks/benchmark_shared_payment_cores.py`. Compare ordinary multiplied
payments against shared scaled payments for increasing consumer counts, batch
sizes, and duration widths. Include a cheap duration-dependent core to expose
reduction cost and a more expensive core backed by shared derived fields to
separate this improvement from existing field hoisting.

Check numerical agreement before timing; warm compilation, synchronize device
results, and report compilation separately from steady-state solve time.
Include a one-consumer case to detect added overhead. Report backend and problem
dimensions. Do not use fragile wall-clock thresholds in CI or assume speedup is
proportional to the number of payments: probability propagation is unchanged.

Update `docs/api_spec.md` as the normative contract, `README.md` with the
two-weight example, and `CHANGELOG.md`. Document declaration-local registry
definitions, name-based sharing, compatible attachment contexts, time sampling,
duration-independent weights, and preservation of named components. Include
factory-created consumers referencing a single named definition. Explain that
different names or ordinary callables do not implicitly share computations.

## Completion criteria

- The two-weight example works with one prepared core task and two separately
  addressable component outputs.
- Independently constructed consumers share by name without requiring callable
  object reuse, and conflicting definitions cannot be supplied at reference sites.
- All cashflow kinds preserve unfactored payment semantics and existing views.
- Tests demonstrate both numerical equivalence and integration sharing.
- Benchmarks quantify the benefit and any one-consumer overhead.
- Public documentation and typing match the implementation, and required checks
  pass.
