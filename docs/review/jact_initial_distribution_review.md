# jact/initial_distribution.py Review

Reviewed file: `jact/initial_distribution.py`

## Findings

1. Component payload type errors are indirect.

   `_component_payload()` assumes each component payload is mapping-like. Passing
   a non-mapping payload fails through membership checks such as `"mass" not in
   payload`, which can produce confusing Python errors. Suggested change:
   validate `isinstance(payload, Mapping)` and raise a clear `TypeError` or
   `ValueError`.

2. `per_individual()` does not enforce integer state indices.

   The public contract describes `states` as a rank-1 integer index array.
   Current validation checks only rank and then passes the values to
   `jax.nn.one_hot()`. Suggested change: reject concrete non-integer dtypes and
   document the behavior for traced values.

3. All-zero normalized mixtures remain all zero.

   `_normalise_masses()` avoids division by zero by leaving zero totals
   unchanged. That is numerically safe, but a distribution with zero total mass
   may be surprising as a public initial distribution. Suggested change: decide
   whether all-zero totals are valid; if invalid, reject concrete all-zero totals
   with a clear error.

## Tests To Add

- Non-mapping component payloads produce a clear error.
- Float `per_individual` state indices are rejected.
- All-zero `normalise=True` mixtures have documented behavior.

