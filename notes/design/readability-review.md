# Python readability review

The API contract is in `docs/api_spec.md`. For code that runs while JAX
traces, keep static branches, callback evaluation order, shapes, dtypes,
reduction order, and scan carries visible. Compare lowered and optimized
IR for representative paths, then measure warmed, synchronized throughput
on each available backend. Keep compilation and warm-up outside the timing.

## Solver

- [x] Name the source, target, hazard row, and transferred fraction in
  `_compute_same_step_transfers`. Handle states with no incoming edges
  up front while preserving one chained cashflow slot per outgoing edge.
- [x] In `_advance_solver_step_from_dynamics`, name the stored mass before
  distinguishing empty, marginal, and full density storage. Keep the
  zero-tail reinsertion beside the slicing rule.
- [x] In `_evaluation_layout`, distinguish sample indices from callable
  output slots and explain where the limit slot comes from.
- [x] Give state-rate and transition-lump formulas named functions. Keep
  named-core reuse, scaling, and accumulation at the attachment boundary.
- [x] In `_compute_cashflow_step`, clarify scheduled and duration event
  formulas and their two accumulation destinations. Preserve the order of
  additions for overlapping components and events.
- [x] Name the midpoint and left-edge view weights and show the order in
  which the two subtotals are combined.
- [x] Trace a recording interval through `_midpoint_solver` and name
  the separate interval and terminal accumulators around the nested scans.
- [x] In public `solve`, put duration-grid construction and initial-carry
  assembly behind named stages.
- [x] Name batch padding and document the axes restored when reassembling
  streamed and terminal results from device shards.
- [x] Document early view validation and later binding to reachable-state
  indices in public `solve`.

## Core declarations and output

- [x] Follow a grouped intensity from `StateSpace.build` through
  `Model.reduce`. Name source and column indices for author order and
  reduced order wherever both are live.
- [x] Follow scalar shorthand, mixture, and indexed inputs through
  `InitialDistribution`. Make concrete checks and traced checks explicit
  at their respective boundaries.
- [x] In cashflow validation, name event selectors separately from payment
  mappings and keep validation and normalized construction adjacent.
- [x] Name the dependency order in the derived field graph and document how
  duration-only grids and step-local fields are reused.
- [x] Use the field-value vocabulary consistently when normalizing PyTree leaves.
- [x] Name probability reducer stages and the transition-major axis of
  grouped fitted-model outputs.
- [x] Document the public result axes and static PyTree metadata. Explain
  how internal view source slots and event subtotals feed those outputs.
  The callable protocols already describe their input and output contracts.
- [x] Make the bundled skill read visible at the install write. The
  overwrite guard runs before creating the target directory.

## Supporting code

- [x] Document that the solver benchmark times public dispatch and waits
  for readiness. Name the percentile rank separately from its list index.
- [x] Review focused benchmark setup for absorbing and uniform states,
  duration limits, derived grids, and shared payment cores. Keep lowering,
  correctness comparisons, and result formatting outside each timed sample.
  Label the duration-limit benchmark's covariate construction as part of
  its end-to-end sample.
- [x] Bring the cashflow precision diagnostics back to the current public
  API. Use named internal cashflow specs in the density diagnostic, which
  seeds the density carry directly.
- [x] Make the installed-wheel import check readable as Python source in
  the isolated environment where it executes.
- [x] Review the test entry points and fixtures for topology and validation,
  solver and probability outputs, distribution and gradients, cashflows and
  derived fields, wrappers, installer and installed typing. Spell out the
  setup and output axes in the padded two-device regression.
- [x] In the prototype, explain the Heun carries, the source and target
  orientation, and the historical final-cell duration reducers. Express the
  reference callable search in row order.

## Verification

- Against baseline `a8df2d8`, eight probability cases and eight cashflow
  cases on an RTX A4000 matched in StableHLO, optimized HLO with source
  metadata stripped, and output bytes. Cases cover two topologies, compact
  and full probability output, and both limited and unlimited durations.
- A synchronized end-to-end GPU sweep of 24 cases used 1,000 individuals,
  48 steps, eight warmups and 40 samples per case. Relative changes in
  median latency ranged from −14.0% to +15.2%, with a median change of
  +2.2%.
- A dense-model repeat used baseline, refactor, refactor, baseline order
  with 20 warmups and 100 samples. Comparing the mean of each version's two
  medians, latency changed by +1.0% with probability, +3.5% with probability
  disabled, −2.8% for state rates, +0.6% for scheduled events and +0.1% for
  duration events. The initial slowdowns of the state-rate and scheduled
  event cases did not reproduce.
- An interleaved compiled GPU benchmark covered 16 combinations of batch
  size, probability compression and output selection. With 20 warmups and
  100 samples, the median throughput change across cases was −0.06%, with
  a range of −0.55% to +0.26%. All compared outputs had zero absolute error.
- A broader interleaved run on the same GPU covered 24 sparse and dense
  benchmark cases at 1,000 individuals and 48 steps, with 20 warmups and
  100 timed samples per version. Public solve throughput changed by
  −1.42% to +0.93% with a median change of −0.06%. The 22 compiled
  workloads that return results changed by −0.67% to +1.39%, with a
  median change of +0.08%. The two compiled output-disabled controls
  returned no numerical results and ran in about 30 microseconds.
- The final tree matched both sets of GPU compilation probes. A source
  comparison also matched 103 public function, method, and class signatures,
  including overloads, to the baseline.
- On CPU, the same 16 compilation probes matched in StableHLO, optimized
  HLO and output bytes. Eight interleaved compiled CPU workloads had zero
  output difference. Median throughput changes ranged from −0.83% to
  +0.84%, with a median change across cases of +0.56%.
- The final CPU suite passed 358 tests. In the targeted CUDA run, 159
  tests passed and two CPU-specific compiler tests needed the CPU backend
  enabled. Both passed on rerun with CUDA as the default backend and CPU
  also enabled. Ruff, Pyright, and the isolated installed-wheel typing
  check passed.
