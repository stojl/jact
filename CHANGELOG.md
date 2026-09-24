# Release notes

## Unreleased

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
