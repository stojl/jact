# Release notes

## Unreleased

## 0.1.16

- Accumulate continuous probability directly in absorbing states when requested
  outputs and payments only need the duration marginal.
- Skip empty continuous duration grids for states with no incoming transitions
  while continuing to evaluate their initial point masses and payments.
- Accumulate the duration marginal for states whose exit intensities and
  payments broadcast across duration. Detect eligible states from callback
  output shapes during JIT tracing and preserve full grids for duration events
  and duration-resolved outputs.

## 0.1.15

- Accept JAX pytrees as named solve inputs, preserving their structure in
  callbacks and derived fields. Validate numeric leaves and report input paths
  when batch sizes disagree.
- Preserve structured derived-field values through dependent fields and solver
  evaluations, including JIT and multi-device solves.

## 0.1.14

- Add declaration-local named payment `cores` and `cashflows.Scaled` references.
  Compatible payments share their complete integrated core contribution before
  applying individual duration-independent weights, across all cashflow kinds.
- Preserve probability mass when initial masses or per-individual durations
  are supplied as integers.
- Keep probability and cashflow gradients finite and correct at zero initial
  mass by tracking initial mass separately from accumulated survival.

## 0.1.13

- Reuse duration-only derived fields across solver steps on fixed density grids,
  including intensity, state-rate, transition-lump, and scheduled-event
  evaluations. Point-mass fields continue to use their changing durations.
- Document how to structure shared derived fields for compiled performance.

## 0.1.12

- Add shared derived fields to models, cashflow declarations, and solves. Fields
  can depend on solve inputs, time, duration, and other fields, with dependency
  validation and reuse across intensity and payment callables. Model displays
  list their derived field names.
- Add solve-time duration limits for intensity and payment evaluation, plus
  optional compression of older continuous probability into a fixed-duration
  tail. `jact.probability.Tail()` exposes the compressed mass and duration.
- Document the duration approximation and add a derived-fields notebook.
