# jact/callbacks.py Review

Reviewed file: `jact/callbacks.py`

## Findings

1. `PointMass` has no public-shape validation.

   `PointMass` is exported and can be constructed directly, but it does not
   validate that `value`, `d_0`, and `log_value` have compatible shapes. Solver
   construction currently supplies consistent arrays, so this is mostly a
   public API hardening issue. Suggested change: validate shape compatibility
   when arrays are concrete.

2. Negative point-mass values are not rejected.

   `PointMass.__init__()` computes `log_value` with `jnp.where(value > 0, ...)`.
   Negative values produce `-inf` logs while the raw `value` remains negative.
   Suggested change: reject concrete negative values or keep `PointMass` private
   if direct construction is not intended.

3. Callback assumptions depend on non-empty duration axes.

   `collapse_point()` writes to duration slot zero. This is fine for valid solver
   inputs, but invalid `horizon` or `steps_per_unit` values can surface here as
   low-level indexing failures. This is covered by the `jact/solver.py` review's
   recommendation to validate solver dimensions at entry.

## Tests To Add

- Direct `PointMass` construction with incompatible shapes fails clearly.
- Direct `PointMass` construction with negative values has documented behavior.

