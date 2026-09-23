# Release notes

## 0.1.12

- Add shared derived fields to models, cashflow declarations, and solves. Fields
  can depend on solve inputs, time, duration, and other fields, with dependency
  validation and reuse across intensity and payment callables. Model displays
  list their derived field names.
- Add solve-time duration limits for intensity and payment evaluation, plus
  optional compression of older continuous probability into a fixed-duration
  tail. `jact.probability.Tail()` exposes the compressed mass and duration.
- Document the duration approximation and add a derived-fields notebook.
