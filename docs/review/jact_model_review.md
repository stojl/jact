# jact/model.py Review

Reviewed file: `jact/model.py`

## Findings

1. Model build does not validate that assigned intensities are callable.

   `_register_transition()` accepts any object as `fn`. A non-callable
   assignment is accepted during model construction and fails later during
   solving. Suggested change: reject non-callables in `_register_transition()`.

2. Multi-output assignment shape errors surface late.

   `exits` and `groups` are wrapped by `_make_slice_wrapper()` without checking
   that the callable returns enough leading transition outputs. If an exit/group
   callable returns too few slices, the error appears inside solve-time JAX code.
   Suggested change: add a focused validation test or a small optional shape
   check at solve/reference evaluation time.

3. Empty group assignments are not rejected directly.

   A group entry with an empty transition list is silently ignored and any
   intended coverage fails later as "not covered". Suggested change: reject
   empty group transition lists with a clearer build-time error.

## Tests To Add

- Non-callable transition, exit, and group assignments are rejected at build.
- Group/exits callable returning too few outputs fails with a clear error.
- Empty group transition lists are rejected directly.

